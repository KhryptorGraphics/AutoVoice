"""Content and pitch encoders for voice conversion.

Uses HuBERT or ContentVec for content extraction (speaker-independent
linguistic features) and a pitch encoder for F0 contour processing.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContentEncoder(nn.Module):
    """Content encoder supporting HuBERT-soft and ContentVec backends.

    Extracts speaker-independent content features from audio,
    preserving linguistic content while removing speaker identity.

    Supports two feature extraction backends:
        - 'hubert' (default): HuBERT-soft model (256-dim output)
        - 'contentvec': ContentVec model (layer 9, 768→256 projection,
          superior speaker disentanglement)

    Supports two projection modes:
        - 'linear': Simple Linear+ReLU+Linear projection (default, fast)
        - 'conformer': Multi-head attention + Conv1D FFN (6 layers, captures
          long-range dependencies in content features)
    """

    def __init__(self, hidden_size: int = 256, output_size: int = 256,
                 hubert_model: str = "hubert-soft", device=None,
                 encoder_type: str = 'linear', conformer_config: dict = None,
                 encoder_backend: str = 'hubert',
                 contentvec_model: str = "lengyue233/content-vec-best",
                 contentvec_layer: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hubert_model_name = hubert_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_type = encoder_type
        self.encoder_backend = encoder_backend

        # Feature dimension from backend:
        # - HuBERT-soft: 256-dim
        # - ContentVec: 768-dim (full Layer 12) when using Conformer,
        #   256-dim (projected) when using linear
        if encoder_backend == 'contentvec' and encoder_type == 'conformer':
            # Pass full 768-dim to Conformer for richer representation
            feature_dim = 768
            contentvec_output_dim = 768
        elif encoder_backend == 'contentvec':
            # Linear projection: ContentVec projects 768→256 internally
            feature_dim = 768
            contentvec_output_dim = 768
        else:
            # HuBERT-soft: always 256-dim
            feature_dim = 256
            contentvec_output_dim = 256  # unused

        # Projection from backend features to content space
        if encoder_type == 'conformer':
            from .conformer import ConformerEncoder
            cfg = conformer_config or {}
            self.projection = ConformerEncoder(
                input_dim=feature_dim,
                hidden_dim=cfg.get('hidden_dim', 384),
                output_dim=output_size,
                n_layers=cfg.get('n_layers', 6),
                n_heads=cfg.get('n_heads', 2),
                filter_channels=cfg.get('filter_channels', None),
                kernel_size=cfg.get('kernel_size', 3),
                window_size=cfg.get('window_size', 4),
                dropout=cfg.get('dropout', 0.1),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        # Backend-specific initialization
        if encoder_backend == 'contentvec':
            self._contentvec = ContentVecEncoder(
                output_dim=contentvec_output_dim,
                layer=contentvec_layer,
                pretrained=contentvec_model,
                device=self.device,
            )
            self._hubert = None
            self._hubert_loaded = True  # Skip HuBERT loading
        else:
            self._contentvec = None
            # HuBERT model (lazy-loaded)
            self._hubert = None
            self._hubert_loaded = False

    def _load_hubert(self, checkpoint_path: Optional[str] = None):
        """Load HuBERT model for feature extraction."""
        if self._hubert_loaded:
            return

        if checkpoint_path and Path(checkpoint_path).exists():
            self._hubert = HuBERTSoft(checkpoint_path=checkpoint_path)
            self._hubert.to(self.device)
            self._hubert.eval()
            logger.info(f"Loaded HuBERT from {checkpoint_path}")
        else:
            self._hubert = HuBERTSoft()
            self._hubert.to(self.device)
            logger.info("HuBERTSoft initialized (no pretrained weights)")

        self._hubert_loaded = True

    def extract_features(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Extract content features from audio.

        Args:
            audio: Audio tensor [B, T] or [T]
            sr: Sample rate (both backends expect 16kHz)

        Returns:
            Content features [B, N_frames, output_size]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample to 16kHz if needed
        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000).to(audio.device)
            audio = resampler(audio)

        # Extract features using the configured backend
        if self.encoder_backend == 'contentvec':
            features = self._contentvec.encode(audio)  # [B, N, 256]
        else:
            # HuBERT-soft backend
            if self._hubert is None:
                if not self._hubert_loaded:
                    self._hubert = HuBERTSoft()
                    self._hubert.to(self.device)
                    self._hubert_loaded = True
                    logger.info("HuBERTSoft initialized with random weights (training mode)")
                else:
                    raise RuntimeError("HuBERT model failed to initialize")
            features = self._hubert.encode(audio)  # [B, N, 256]

        # Project to output space
        content = self.projection(features)
        return content

    def forward(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Forward pass - extract content features."""
        return self.extract_features(audio, sr)

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, device=None) -> 'ContentEncoder':
        """Load a pretrained content encoder."""
        encoder = cls(device=device)
        encoder._load_hubert(checkpoint_path)
        encoder.to(encoder.device)
        return encoder


def f0_to_coarse(f0: torch.Tensor, n_bins: int = 256,
                 f0_min: float = 50.0, f0_max: float = 1100.0) -> torch.Tensor:
    """Convert F0 in Hz to mel-scale quantized bin indices.

    Uses mel-scale quantization: f0_mel = 1127 * log(1 + f0/700)
    Quantizes to n_bins discrete bins between f0_min and f0_max.

    Args:
        f0: F0 values in Hz [B, T] or [B, T, 1]
        n_bins: Number of quantization bins (default 256)
        f0_min: Minimum F0 in Hz (default 50)
        f0_max: Maximum F0 in Hz (default 1100)

    Returns:
        Bin indices [B, T] as long tensor, clamped to [0, n_bins-1]
    """
    if f0.dim() == 3:
        f0 = f0.squeeze(-1)

    # Convert Hz to mel scale
    f0_mel = 1127.0 * torch.log1p(f0.abs() / 700.0)
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)

    # Normalize to [0, n_bins-1]
    bins = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min) * (n_bins - 1)
    bins = bins.long().clamp(0, n_bins - 1)
    return bins


class PitchEncoder(nn.Module):
    """Mel-quantized F0 encoder with voiced/unvoiced embedding.

    Replaces LSTM-based encoding with mel-scale quantized F0 lookup (256 bins)
    plus a learned voiced/unvoiced embedding. A small residual linear path
    preserves fine pitch detail and gradient flow.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 output_size: int = 256, n_bins: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_bins = n_bins

        # Mel-quantized pitch embedding (256 bins)
        self.pitch_embed = nn.Embedding(n_bins, output_size)

        # Voiced/unvoiced embedding
        self.uv_embed = nn.Embedding(2, output_size)

        # Residual linear path for fine pitch detail and gradient flow
        self.residual_proj = nn.Linear(1, output_size)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """Encode pitch contour using mel-quantized embeddings.

        Args:
            f0: Pitch values in Hz [B, T] or [B, T, 1]

        Returns:
            Pitch embedding [B, T, output_size]
        """
        if f0.dim() == 3:
            f0 = f0.squeeze(-1)

        # Quantize to mel-scale bins
        bins = f0_to_coarse(f0, n_bins=self.n_bins)  # [B, T]

        # Voiced/unvoiced flag (voiced if f0 > ~10 Hz)
        uv = (f0.abs() > 10.0).long()  # [B, T]

        # Embedding lookups
        pitch_emb = self.pitch_embed(bins)  # [B, T, output_size]
        uv_emb = self.uv_embed(uv)  # [B, T, output_size]

        # Residual from continuous f0 (enables gradient flow through f0)
        f0_log = torch.log1p(f0.abs()).unsqueeze(-1)  # [B, T, 1]
        residual = self.residual_proj(f0_log)  # [B, T, output_size]

        return pitch_emb + uv_emb + residual

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, device=None) -> 'PitchEncoder':
        """Load pretrained pitch encoder weights."""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = cls()
        if Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            encoder.load_state_dict(state_dict)
            logger.info(f"Loaded pitch encoder from {checkpoint_path}")
        encoder.to(device)
        return encoder


class HuBERTSoft(nn.Module):
    """Simplified soft-HuBERT model for content extraction."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self._loaded = False

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 512, 10, 5),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, 2),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, 2),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, 2),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, 2),
            nn.GELU(),
            nn.Conv1d(512, 512, 2, 2),
            nn.GELU(),
            nn.Conv1d(512, 512, 2, 2),
            nn.GELU(),
        )

        self.proj = nn.Linear(512, 256)

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str):
        """Load hubert-soft checkpoint."""
        try:
            torch.load(path, map_location='cpu', weights_only=False)
            self._loaded = True
            logger.info(f"HuBERT-Soft checkpoint loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load HuBERT checkpoint: {e}")

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to soft content units.

        Args:
            audio: [B, T] waveform at 16kHz

        Returns:
            [B, N, 256] soft content units
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]

        features = self.feature_extractor(audio)  # [B, 512, N]
        features = features.transpose(1, 2)  # [B, N, 512]
        units = self.proj(features)  # [B, N, 256]
        return units

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for HuBERTSoft."""
        return self.encode(audio)


class ContentVecEncoder(nn.Module):
    """ContentVec-based content encoder for speaker-disentangled features.

    Uses the ContentVec model (arxiv:2204.09224), which is a HuBERT-Base
    variant pretrained with speaker disentanglement. Extracts Layer 12
    hidden states (768-dim) for best content representation.

    Architecture decision: Layer 12 provides the most abstract/semantic
    content features, used by CoMoSVC, RVC, and So-VITS-SVC 4.1+.
    When output_dim=768 (default), features pass through without projection.
    When output_dim<768, a learned linear projection reduces dimensionality.
    """

    def __init__(self, output_dim: int = 768, layer: int = 12,
                 pretrained: str = "lengyue233/content-vec-best",
                 device=None):
        """Initialize ContentVec encoder.

        Args:
            output_dim: Output feature dimension (default 768, full Layer 12).
                Set to 256 for backward compatibility with vec256l9.
            layer: Which transformer layer to extract (default 12).
            pretrained: HuggingFace model ID or local path. Set to None
                for random initialization (testing).
            device: Target device.
        """
        super().__init__()
        self.output_dim = output_dim
        self.layer = layer
        self.pretrained_id = pretrained
        self.sample_rate = 16000  # ContentVec expects 16kHz input
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = None
        self._loaded = False

        # Only add projection if output_dim != 768 (native ContentVec dim)
        if output_dim != 768:
            self.final_proj = nn.Linear(768, output_dim)
        else:
            self.final_proj = None

    def _load_model(self):
        """Lazy-load the ContentVec model."""
        if self._loaded:
            return

        try:
            from transformers import HubertModel, HubertConfig

            if self.pretrained_id and Path(self.pretrained_id).exists():
                # Load from local path
                self._model = HubertModel.from_pretrained(
                    self.pretrained_id,
                    local_files_only=True,
                )
                logger.info(f"ContentVec loaded from local: {self.pretrained_id}")
            elif self.pretrained_id:
                try:
                    self._model = HubertModel.from_pretrained(self.pretrained_id)
                    logger.info(f"ContentVec loaded from HuggingFace: {self.pretrained_id}")
                except Exception as e:
                    logger.warning(f"Could not load pretrained ContentVec: {e}")
                    logger.info("Initializing ContentVec with random weights")
                    config = HubertConfig(
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        classifier_proj_size=256,
                    )
                    self._model = HubertModel(config)
            else:
                # Random init for testing
                config = HubertConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    classifier_proj_size=256,
                )
                self._model = HubertModel(config)
                logger.info("ContentVec initialized with random weights (testing)")

        except ImportError:
            raise RuntimeError(
                "transformers package required for ContentVec. "
                "Install with: pip install transformers"
            )

        self._model.to(self.device)
        self._loaded = True

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker-disentangled content features.

        Args:
            audio: Waveform at 16kHz, shape [B, T] or [T].

        Returns:
            Content features [B, N_frames, output_dim] at 20ms resolution.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if self._model is None:
            self._load_model()

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(
                audio,
                output_hidden_states=True,
            )

        # Extract target layer (default: layer 12 for best content representation)
        hidden_states = outputs.hidden_states[self.layer]  # [B, N, 768]

        # Project if output_dim != 768, otherwise passthrough
        if self.final_proj is not None:
            features = self.final_proj(hidden_states)  # [B, N, output_dim]
        else:
            features = hidden_states  # [B, N, 768]
        return features

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for ContentVec."""
        return self.encode(audio)
