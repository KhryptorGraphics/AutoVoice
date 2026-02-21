"""Model manager for voice conversion inference.

Orchestrates content encoding, pitch encoding, SoVitsSvc, and HiFiGAN vocoder
with frame alignment. No fallback behavior - raises RuntimeError if models
are not loaded.
"""
import logging
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages voice models and runs frame-aligned inference.

    Raises RuntimeError if any required model is not loaded or if invalid
    configuration values are provided. No fallback behavior.

    Supported config keys:
        sample_rate (int): Audio sample rate. Default: 22050.
        vocoder_type (str): Vocoder backend - 'hifigan' or 'bigvgan'. Default: 'hifigan'.
        encoder_backend (str): Feature extractor - 'hubert' or 'contentvec'. Default: 'hubert'.
        encoder_type (str): Encoder architecture - 'linear' or 'conformer'. Default: 'linear'.
        conformer_config (dict): Conformer hyperparams (n_layers, n_heads, etc.).
        hubert_path (str): Path to HuBERT checkpoint.
        vocoder_path (str): Path to vocoder checkpoint.
    """

    VALID_VOCODER_TYPES = ('hifigan', 'bigvgan')
    VALID_ENCODER_BACKENDS = ('hubert', 'contentvec')
    VALID_ENCODER_TYPES = ('linear', 'conformer')

    def __init__(self, device=None, config: Optional[Dict] = None):
        """Initialize ModelManager with device and configuration.

        Args:
            device: PyTorch device (cuda/cpu). Auto-detects if None.
            config: Optional configuration dict. Supported keys:
                - sample_rate (int): Audio sample rate. Default: 22050.
                - vocoder_type (str): 'hifigan' or 'bigvgan'. Default: 'hifigan'.
                - encoder_backend (str): 'hubert' or 'contentvec'. Default: 'hubert'.
                - encoder_type (str): 'linear' or 'conformer'. Default: 'linear'.
                - conformer_config (dict): Conformer hyperparams if encoder_type='conformer'.

        Raises:
            RuntimeError: If config contains invalid vocoder_type, encoder_backend,
                or encoder_type values.

        Models are not loaded until load() is called.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}

        # Validate config values if present
        self._validate_config(self.config)

        self.sample_rate = self.config.get('sample_rate', 22050)

        # Shared models (initialized in load())
        self._content_encoder = None
        self._pitch_encoder = None
        self._vocoder = None

        # Per-speaker trained models
        self._sovits_models: Dict[str, object] = {}
        self._speaker_embeddings: Dict[str, np.ndarray] = {}

    def _validate_config(self, config: Dict) -> None:
        """Validate configuration values against allowed options.

        Args:
            config: Configuration dict to validate. Checks keys:
                - vocoder_type: Must be in VALID_VOCODER_TYPES
                - encoder_backend: Must be in VALID_ENCODER_BACKENDS
                - encoder_type: Must be in VALID_ENCODER_TYPES

        Raises:
            RuntimeError: If any config value is invalid. Error message
                includes the invalid value and list of valid options.

        Unknown config keys are ignored (future compatibility).
        """
        if 'vocoder_type' in config:
            if config['vocoder_type'] not in self.VALID_VOCODER_TYPES:
                raise RuntimeError(
                    f"Invalid vocoder_type: '{config['vocoder_type']}'. "
                    f"Valid options: {list(self.VALID_VOCODER_TYPES)}"
                )
        if 'encoder_backend' in config:
            if config['encoder_backend'] not in self.VALID_ENCODER_BACKENDS:
                raise RuntimeError(
                    f"Invalid encoder_backend: '{config['encoder_backend']}'. "
                    f"Valid options: {list(self.VALID_ENCODER_BACKENDS)}"
                )
        if 'encoder_type' in config:
            if config['encoder_type'] not in self.VALID_ENCODER_TYPES:
                raise RuntimeError(
                    f"Invalid encoder_type: '{config['encoder_type']}'. "
                    f"Valid options: {list(self.VALID_ENCODER_TYPES)}"
                )

    def load(self, hubert_path: Optional[str] = None,
             vocoder_path: Optional[str] = None,
             vocoder_type: str = 'hifigan',
             encoder_backend: str = 'hubert',
             encoder_type: str = 'linear',
             conformer_config: Optional[Dict] = None):
        """Load shared models. Must be called before infer().

        Args:
            hubert_path: Path to HuBERT checkpoint (None for random weights).
            vocoder_path: Path to vocoder checkpoint (None for random weights).
            vocoder_type: Vocoder backend to use. Valid: 'hifigan', 'bigvgan'.
            encoder_backend: Feature extractor. Valid: 'hubert', 'contentvec'.
            encoder_type: Encoder architecture. Valid: 'linear', 'conformer'.
            conformer_config: Dict of conformer hyperparams when encoder_type='conformer'.
                Keys: n_layers (int), n_heads (int), d_model (int), ff_dim (int),
                kernel_size (int), dropout (float).

        Raises:
            RuntimeError: If vocoder_type, encoder_backend, or encoder_type is invalid.

        If paths are None, models initialize with random weights
        (suitable for training, not inference).
        """
        from ..models.encoder import ContentEncoder, PitchEncoder
        from ..models.vocoder import HiFiGANVocoder, BigVGANVocoder

        self._content_encoder = ContentEncoder(
            output_size=768,  # 768-dim for best quality (ContentVec native)
            device=self.device,
            encoder_backend=encoder_backend,
            encoder_type=encoder_type,
            conformer_config=conformer_config,
        )
        if hubert_path and encoder_backend == 'hubert':
            self._content_encoder._load_hubert(hubert_path)
        self._content_encoder.to(self.device)

        self._pitch_encoder = PitchEncoder(output_size=768).to(self.device)  # Match 768-dim

        if vocoder_type == 'bigvgan':
            self._vocoder = BigVGANVocoder(device=self.device)
        elif vocoder_type == 'hifigan':
            self._vocoder = HiFiGANVocoder(device=self.device)
        else:
            raise RuntimeError(f"Unknown vocoder_type: {vocoder_type}. Use 'hifigan' or 'bigvgan'.")

        if vocoder_path:
            self._vocoder.load_checkpoint(vocoder_path)

    def load_voice_model(self, model_path: str, speaker_id: str,
                         speaker_embedding: Optional[np.ndarray] = None):
        """Load a trained per-speaker SoVitsSvc model for voice conversion.

        Args:
            model_path: Path to trained SoVitsSvc checkpoint file.
            speaker_id: Unique identifier for this speaker (used in infer()).
            speaker_embedding: Optional 256-dim speaker embedding vector.
                If provided, stored for later retrieval.

        The loaded model is stored in _sovits_models dict keyed by speaker_id.
        Multiple models can be loaded for different speakers.

        Raises:
            FileNotFoundError: If model_path does not exist.
            RuntimeError: If checkpoint cannot be loaded.
        """
        from ..models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc.load_pretrained(model_path, device=self.device)
        self._sovits_models[speaker_id] = model
        if speaker_embedding is not None:
            self._speaker_embeddings[speaker_id] = speaker_embedding

    def infer(self, audio: np.ndarray, speaker_id: str,
              speaker_embedding: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Convert audio to target speaker's voice. No fallbacks.

        Args:
            audio: Input audio waveform (float32, mono)
            speaker_id: Target speaker identifier
            speaker_embedding: Target speaker embedding [256]
            sr: Sample rate of input audio

        Returns:
            Converted audio waveform (float32, same length as input)

        Raises:
            RuntimeError: If any model is not loaded
        """
        if self._content_encoder is None:
            raise RuntimeError("ContentEncoder not loaded. Call load() first.")
        if self._pitch_encoder is None:
            raise RuntimeError("PitchEncoder not loaded. Call load() first.")
        if self._vocoder is None:
            raise RuntimeError("Vocoder not loaded. Call load() first.")
        if speaker_id not in self._sovits_models:
            raise RuntimeError(
                f"No trained model for speaker '{speaker_id}'. "
                f"Train a model first or call load_voice_model()."
            )

        import librosa
        import scipy.signal

        # 1. Extract content features (WHAT is being sung)
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        with torch.no_grad():
            content = self._content_encoder.extract_features(
                audio_tensor, sr=sr
            )  # [1, N, 256]

        # 2. Extract F0 (HOW it's being sung - original artist's melody)
        f0, voiced, _ = librosa.pyin(
            audio, fmin=50, fmax=1100, sr=sr, hop_length=512
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)  # [1, T]
        with torch.no_grad():
            pitch = self._pitch_encoder(f0_tensor)  # [1, T, 256]

        # 3. Frame alignment - align content and pitch to same resolution
        target_frames = min(content.shape[1], pitch.shape[1])
        if target_frames == 0:
            return np.zeros_like(audio)

        content = F.interpolate(
            content.transpose(1, 2), size=target_frames,
            mode='linear', align_corners=False
        ).transpose(1, 2)  # [1, target_frames, 256]
        pitch = F.interpolate(
            pitch.transpose(1, 2), size=target_frames,
            mode='linear', align_corners=False
        ).transpose(1, 2)  # [1, target_frames, 256]

        # 4. Speaker embedding (WHO should sing - target person)
        speaker = torch.from_numpy(speaker_embedding).float().unsqueeze(0).to(self.device)

        # 5. SoVitsSvc inference -> mel spectrogram
        sovits = self._sovits_models[speaker_id]
        with torch.no_grad():
            mel_pred = sovits.infer(content, pitch, speaker)  # [1, 80, target_frames]

        # 6. HiFiGAN vocoder -> audio waveform
        with torch.no_grad():
            output_audio = self._vocoder.synthesize(mel_pred)  # [1, T_audio]

        # 7. Resample to match input length
        output_np = output_audio.squeeze(0).cpu().numpy()
        if len(output_np) != len(audio):
            output_np = scipy.signal.resample(output_np, len(audio)).astype(np.float32)

        # 8. Normalize
        peak = np.abs(output_np).max()
        if peak > 0.95:
            output_np = output_np * (0.95 / peak)
        elif peak > 0:
            output_np = output_np * (0.9 / peak)

        return output_np
