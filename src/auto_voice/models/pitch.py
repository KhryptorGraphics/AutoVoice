"""RMVPE pitch extractor for singing voice conversion.

Implements a simplified RMVPE (Robust Model for Vocal Pitch Estimation)
architecture based on the Interspeech 2023 paper. Uses a deep residual
CNN operating on mel spectrograms to produce cent-based pitch estimates
with voicing probabilities.

Key design choices:
- 20ms hop size (320 samples at 16kHz) matching ContentVec frame rate
- 360 bins per octave (10-cent resolution)
- 6 octaves coverage (C1=32.7Hz to C7=2093Hz, clipped to f0_min/f0_max)
- Weighted average of cent bins for sub-cent precision
- No fallback: raises RuntimeError on failure
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual block for RMVPE feature extraction."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class RMVPEBackbone(nn.Module):
    """Deep residual CNN backbone for RMVPE.

    Processes mel spectrograms through a series of residual blocks
    with progressive channel expansion, producing frame-level features
    for pitch classification.
    """

    def __init__(self, n_mels: int = 128, n_blocks: int = 6,
                 base_channels: int = 64):
        super().__init__()
        self.n_mels = n_mels

        # Initial convolution from mel spectrogram
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )

        # Residual blocks with channel expansion
        layers = []
        in_ch = base_channels
        for i in range(n_blocks):
            out_ch = min(base_channels * (2 ** (i // 2)), 512)
            if out_ch != in_ch:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                ))
            layers.append(ResBlock(out_ch, kernel_size=3, dilation=1))
            layers.append(ResBlock(out_ch, kernel_size=3, dilation=2))
            in_ch = out_ch

        self.blocks = nn.Sequential(*layers)
        self.final_channels = in_ch

        # Collapse frequency dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Process mel spectrogram.

        Args:
            mel: [B, 1, n_mels, T] mel spectrogram

        Returns:
            [B, channels, T] frame-level features
        """
        x = self.input_conv(mel)  # [B, C, n_mels, T]
        x = self.blocks(x)  # [B, C', n_mels', T]
        x = self.freq_pool(x)  # [B, C', 1, T]
        x = x.squeeze(2)  # [B, C', T]
        return x


class RMVPEPitchExtractor(nn.Module):
    """RMVPE-based pitch extractor for singing voice.

    Extracts F0 contour directly from audio waveform using a deep
    residual network operating on mel spectrograms. Outputs F0 in Hz
    with voiced/unvoiced decisions.

    Architecture (Interspeech 2023):
    - Input: 128-band mel spectrogram, 20ms hop
    - Backbone: 6-block deep residual CNN (7M params)
    - Output: 360 bins per octave × 6 octaves + voicing probability
    - Decoding: Weighted average of activated bins for sub-cent precision
    """

    def __init__(self, pretrained: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 hop_size: int = 320,
                 f0_min: float = 50.0,
                 f0_max: float = 1100.0,
                 n_mels: int = 128,
                 sample_rate: int = 16000):
        super().__init__()
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of octaves and bins
        self.n_octaves = 6  # C1 to C7 (covers 50-1100 Hz range)
        self.bins_per_octave = 360  # 10-cent resolution
        self.n_bins = self.n_octaves * self.bins_per_octave  # 2160 total

        # Cents-to-frequency mapping
        # Reference: C1 = 32.7 Hz
        cents = torch.arange(self.n_bins, dtype=torch.float32)
        # Each bin = 1200/360 = 3.33 cents
        # Frequency = ref * 2^(cents_offset / 1200)
        ref_freq = 32.7  # C1
        cents_offset = cents * (1200.0 / self.bins_per_octave)
        self.register_buffer(
            'cents_mapping',
            ref_freq * torch.pow(2.0, cents_offset / 1200.0)
        )

        # Backbone
        self.model = RMVPEBackbone(n_mels=n_mels, n_blocks=6, base_channels=64)

        # Pitch output head (frame features → bin logits)
        backbone_channels = self.model.final_channels
        self.pitch_head = nn.Sequential(
            nn.Linear(backbone_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.n_bins),
        )

        # Voicing head (binary: voiced/unvoiced)
        self.voicing_head = nn.Sequential(
            nn.Linear(backbone_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Mel spectrogram transform parameters
        self._mel_basis = None
        self.n_fft = 1024
        self.win_length = 1024

        # Load pretrained weights if provided
        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str):
        """Load pretrained RMVPE weights."""
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"RMVPE pretrained weights not found: {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        logger.info(f"RMVPE weights loaded from {path}")

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from audio.

        Args:
            audio: [B, T] waveform at 16kHz

        Returns:
            [B, 1, n_mels, N_frames] mel spectrogram
        """
        # STFT
        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_size,
            win_length=self.win_length, window=window,
            return_complex=True,
        )
        magnitude = stft.abs()  # [B, n_fft//2+1, N_frames]

        # Mel filterbank
        if self._mel_basis is None or self._mel_basis.device != audio.device:
            self._mel_basis = self._create_mel_filterbank(audio.device)

        mel = torch.matmul(self._mel_basis, magnitude)  # [B, n_mels, N_frames]
        mel = torch.log(mel.clamp(min=1e-5))

        return mel.unsqueeze(1)  # [B, 1, n_mels, N_frames]

    def _create_mel_filterbank(self, device: torch.device) -> torch.Tensor:
        """Create mel filterbank matrix."""
        n_freqs = self.n_fft // 2 + 1
        # Mel scale boundaries
        mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + (self.sample_rate / 2) / 700.0)
        mels = torch.linspace(mel_min, mel_max, self.n_mels + 2, device=device)
        freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Frequency bin indices
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freqs, device=device)

        # Triangular filters
        filterbank = torch.zeros(self.n_mels, n_freqs, device=device)
        for i in range(self.n_mels):
            lower = freqs[i]
            center = freqs[i + 1]
            upper = freqs[i + 2]

            # Rising slope
            up_slope = (fft_freqs - lower) / (center - lower + 1e-8)
            # Falling slope
            down_slope = (upper - fft_freqs) / (upper - center + 1e-8)

            filterbank[i] = torch.maximum(
                torch.zeros_like(fft_freqs),
                torch.minimum(up_slope, down_slope)
            )

        return filterbank

    def _decode_pitch(self, logits: torch.Tensor,
                      voicing: torch.Tensor) -> torch.Tensor:
        """Decode pitch bin logits to F0 in Hz.

        Uses weighted average of activated bins for sub-cent precision.

        Args:
            logits: [B, T, n_bins] pitch bin logits
            voicing: [B, T, 1] voicing probability

        Returns:
            [B, T] F0 in Hz (0 for unvoiced frames)
        """
        # Softmax over bins for probability distribution
        probs = F.softmax(logits, dim=-1)  # [B, T, n_bins]

        # Weighted average frequency
        cents_map = self.cents_mapping.to(logits.device)  # [n_bins]
        f0 = torch.sum(probs * cents_map.unsqueeze(0).unsqueeze(0), dim=-1)  # [B, T]

        # Apply voicing mask
        voicing_prob = voicing.squeeze(-1)  # [B, T]
        f0 = f0 * (voicing_prob > 0.5).float()

        # Clamp to valid range
        f0 = f0.clamp(0.0, self.f0_max)
        # Zero out values below f0_min (treat as unvoiced)
        f0 = f0 * (f0 >= self.f0_min).float()

        return f0

    def extract(self, audio: torch.Tensor,
                return_voicing: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract F0 from audio waveform.

        Args:
            audio: Waveform at 16kHz, shape [B, T] or [T].
            return_voicing: If True, also return voicing probabilities.

        Returns:
            f0: [B, N_frames] F0 in Hz (0 for unvoiced)
            voicing: [B, N_frames] voicing probability (if return_voicing=True)

        Raises:
            RuntimeError: If audio is too short for analysis.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Minimum length check: need at least 1 full window
        min_samples = self.n_fft + self.hop_size
        if audio.shape[1] < min_samples:
            raise RuntimeError(
                f"Audio too short for pitch analysis: {audio.shape[1]} samples, "
                f"need at least {min_samples}"
            )

        # Compute mel spectrogram
        mel = self._compute_mel(audio)  # [B, 1, n_mels, T]

        # Extract features through backbone
        features = self.model(mel)  # [B, C, T]
        features = features.transpose(1, 2)  # [B, T, C]

        # Pitch classification
        pitch_logits = self.pitch_head(features)  # [B, T, n_bins]

        # Voicing detection
        voicing_prob = self.voicing_head(features)  # [B, T, 1]

        # Decode to F0 in Hz
        f0 = self._decode_pitch(pitch_logits, voicing_prob)  # [B, T]

        if return_voicing:
            return f0, voicing_prob.squeeze(-1)
        return f0

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass - extract F0."""
        return self.extract(audio)
