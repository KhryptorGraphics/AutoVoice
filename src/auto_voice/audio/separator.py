"""Mel-Band RoFormer vocal separator for source separation.

Implements a simplified Mel-Band RoFormer architecture for vocal/
instrumental separation, based on the SDX'23 winning approach.

Key design choices:
- Mel-scale frequency band splitting (non-uniform bandwidth)
- RoPE (Rotary Position Embeddings) in transformer layers
- Complex-valued mask estimation for phase-aware separation
- 44.1kHz processing with 2048 n_fft, 512 hop
- No fallback: raises RuntimeError on failure
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_mel_band_splits(n_fft: int, sample_rate: int,
                            n_bands: int = 32) -> list:
    """Compute mel-scale frequency band boundaries.

    Args:
        n_fft: FFT size
        sample_rate: Audio sample rate
        n_bands: Number of mel-scale bands

    Returns:
        List of (start_bin, end_bin) tuples for each band
    """
    n_freqs = n_fft // 2 + 1
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)

    mel_points = np.linspace(mel_min, mel_max, n_bands + 1)
    freq_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # Convert frequencies to FFT bin indices
    bin_indices = np.round(freq_points * n_fft / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_freqs - 1)

    bands = []
    for i in range(n_bands):
        start = int(bin_indices[i])
        end = int(bin_indices[i + 1])
        if end <= start:
            end = start + 1  # Minimum 1 bin per band
        bands.append((start, min(end, n_freqs)))

    return bands


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for temporal modeling."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: [B, T, D] input tensor

        Returns:
            [B, T, D] with rotary embeddings applied
        """
        T = x.shape[1]
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]

        cos_emb = emb.cos().unsqueeze(0)  # [1, T, D]
        sin_emb = emb.sin().unsqueeze(0)

        # Rotate pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

        return x * cos_emb + rotated * sin_emb


class BandTransformerBlock(nn.Module):
    """Transformer block for processing a frequency band."""

    def __init__(self, dim: int, n_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.rope = RotaryPositionEmbedding(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process band features.

        Args:
            x: [B, T, D] band features over time

        Returns:
            [B, T, D] processed features
        """
        # Self-attention with RoPE
        normed = self.norm1(x)
        rope_x = self.rope(normed)
        attn_out, _ = self.attn(rope_x, rope_x, rope_x)
        x = x + attn_out

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class MelBandRoFormer(nn.Module):
    """Mel-Band RoFormer for vocal/instrumental separation.

    Architecture (SDX'23):
    - Input: Complex STFT at 44.1kHz
    - Band splitting: 32 mel-scale frequency bands
    - Per-band processing: Transformer with RoPE
    - Cross-band attention: Information sharing between bands
    - Output: Complex-valued soft masks for vocal/instrumental
    - Reconstruction: Masked STFT → iSTFT

    The mel-scale banding gives more resolution to low frequencies
    (where fundamental frequencies of voice are concentrated) and
    less to high frequencies (where harmonics have less perceptual weight).
    """

    def __init__(self, pretrained: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_bands: int = 32,
                 hidden_dim: int = 128,
                 n_layers: int = 6,
                 n_heads: int = 4):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute mel-scale band splits
        self.band_splits = compute_mel_band_splits(n_fft, sample_rate, n_bands)

        # Per-band projection (complex STFT bins → hidden_dim)
        n_freqs = n_fft // 2 + 1
        self.band_projections = nn.ModuleList()
        self.band_output_projections = nn.ModuleList()
        for start, end in self.band_splits:
            band_width = end - start
            # Input: real + imag = 2 * band_width (for mono)
            self.band_projections.append(
                nn.Linear(band_width * 2, hidden_dim)
            )
            # Output: 2 sources × 2 (real+imag) × band_width
            self.band_output_projections.append(
                nn.Linear(hidden_dim, band_width * 2 * 2)
            )

        # Transformer layers (shared across bands)
        self.transformer = nn.ModuleList([
            BandTransformerBlock(hidden_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Cross-band attention (band interaction)
        self.cross_band_norm = nn.LayerNorm(hidden_dim)
        self.cross_band_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True
        )

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Separator weights not found: {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        logger.info(f"MelBandRoFormer weights loaded from {path}")

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute STFT.

        Args:
            audio: [B, T] or [B, C, T] waveform

        Returns:
            [B, n_freqs, N_frames] complex STFT
        """
        is_stereo = audio.dim() == 3
        if is_stereo:
            # Process channels independently, take first channel for now
            B, C, T = audio.shape
            audio_mono = audio.mean(dim=1)  # Mix to mono for processing
        else:
            audio_mono = audio
            B = audio.shape[0]

        window = torch.hann_window(self.n_fft, device=audio.device)
        stft = torch.stft(
            audio_mono, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window,
            return_complex=True,
        )  # [B, n_freqs, N_frames]
        return stft

    def _istft(self, stft: torch.Tensor, length: int) -> torch.Tensor:
        """Compute inverse STFT.

        Args:
            stft: [B, n_freqs, N_frames] complex STFT
            length: Target output length

        Returns:
            [B, T] waveform
        """
        window = torch.hann_window(self.n_fft, device=stft.device)
        audio = torch.istft(
            stft, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window,
            length=length,
        )
        return audio

    def _process_bands(self, stft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process STFT through band transformer.

        Args:
            stft: [B, n_freqs, N_frames] complex STFT

        Returns:
            vocal_mask, instrumental_mask: [B, n_freqs, N_frames] complex masks
        """
        B, F, T = stft.shape

        # Split into real/imag for processing
        stft_ri = torch.stack([stft.real, stft.imag], dim=-1)  # [B, F, T, 2]

        # Project each band to hidden_dim
        band_features = []
        for i, (start, end) in enumerate(self.band_splits):
            band = stft_ri[:, start:end, :, :]  # [B, band_width, T, 2]
            band_flat = band.permute(0, 2, 1, 3).reshape(B, T, -1)  # [B, T, band_width*2]
            proj = self.band_projections[i](band_flat)  # [B, T, hidden_dim]
            band_features.append(proj)

        # Stack bands: [B, n_bands, T, hidden_dim]
        band_stack = torch.stack(band_features, dim=1)

        # Process each band through transformer
        processed_bands = []
        for b in range(self.n_bands):
            x = band_stack[:, b, :, :]  # [B, T, hidden_dim]
            for layer in self.transformer:
                x = layer(x)
            processed_bands.append(x)

        # Cross-band attention
        processed_stack = torch.stack(processed_bands, dim=1)  # [B, n_bands, T, D]
        # Reshape for cross-band: treat T as batch, bands as sequence
        B2, N, T2, D = processed_stack.shape
        cross_input = processed_stack.permute(0, 2, 1, 3).reshape(B2 * T2, N, D)
        normed = self.cross_band_norm(cross_input)
        cross_out, _ = self.cross_band_attn(normed, normed, normed)
        cross_out = cross_out.reshape(B2, T2, N, D).permute(0, 2, 1, 3)
        processed_stack = processed_stack + cross_out

        # Reconstruct masks for each band
        vocal_mask = torch.zeros(B, F, T, dtype=stft.dtype, device=stft.device)
        instrumental_mask = torch.zeros_like(vocal_mask)

        for i, (start, end) in enumerate(self.band_splits):
            band_width = end - start
            band_out = self.band_output_projections[i](
                processed_stack[:, i, :, :]  # [B, T, hidden_dim]
            )  # [B, T, band_width * 2 * 2]

            # Reshape to [B, T, 2_sources, band_width, 2_ri]
            band_out = band_out.reshape(B, T, 2, band_width, 2)

            # Convert to complex masks via sigmoid
            vocal_ri = torch.sigmoid(band_out[:, :, 0, :, :])  # [B, T, bw, 2]
            inst_ri = torch.sigmoid(band_out[:, :, 1, :, :])

            # Complex mask: (real + j*imag)
            vocal_complex = torch.complex(vocal_ri[..., 0], vocal_ri[..., 1])
            inst_complex = torch.complex(inst_ri[..., 0], inst_ri[..., 1])

            # Apply to frequency bins [B, T, band_width] → [B, band_width, T]
            vocal_mask[:, start:end, :] = vocal_complex.permute(0, 2, 1)
            instrumental_mask[:, start:end, :] = inst_complex.permute(0, 2, 1)

        return vocal_mask, instrumental_mask

    def separate(self, audio: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate audio into vocals and instrumental.

        Args:
            audio: Waveform [B, T] (mono) or [B, C, T] (stereo) at 44.1kHz

        Returns:
            vocals: Same shape as input
            instrumental: Same shape as input

        Raises:
            RuntimeError: If audio is too short for separation.
        """
        is_stereo = audio.dim() == 3

        # Get the time dimension length
        if is_stereo:
            T = audio.shape[2]
        else:
            T = audio.shape[1]

        # Minimum length check
        min_samples = self.n_fft + self.hop_length
        if T < min_samples:
            raise RuntimeError(
                f"Audio too short for separation: {T} samples, "
                f"need at least {min_samples}"
            )

        # Compute STFT (handles stereo internally)
        stft = self._stft(audio)  # [B, n_freqs, N_frames]

        # Process through band transformer
        with torch.no_grad():
            vocal_mask, inst_mask = self._process_bands(stft)

        # Apply masks
        vocal_stft = stft * vocal_mask
        inst_stft = stft * inst_mask

        # Inverse STFT
        if is_stereo:
            mono_T = audio.shape[2]
        else:
            mono_T = audio.shape[1]

        vocals = self._istft(vocal_stft, mono_T)
        instrumental = self._istft(inst_stft, mono_T)

        # Restore stereo shape if needed
        if is_stereo:
            vocals = vocals.unsqueeze(1).expand_as(audio)
            instrumental = instrumental.unsqueeze(1).expand_as(audio)

        return vocals, instrumental

    def extract_vocals(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract only vocals from audio.

        Convenience method that discards the instrumental stem.

        Args:
            audio: Waveform [B, T] or [B, C, T]

        Returns:
            vocals: Same shape as input
        """
        vocals, _ = self.separate(audio)
        return vocals

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - separate vocals and instrumental."""
        return self.separate(audio)
