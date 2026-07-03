"""Model manager for voice conversion inference.

Orchestrates content encoding, pitch encoding, SoVitsSvc, and HiFiGAN vocoder
with frame alignment. No fallback behavior - raises RuntimeError if models
are not loaded.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build_voice_model_from_checkpoint(checkpoint: Any, model_path: str, device) -> torch.nn.Module:
    """Construct the model class that produced a trained checkpoint.

    Detects the artifact family from the state dict (CoMoSVCDecoder training
    artifacts vs SoVitsSvc states), derives dimensions from tensor shapes
    (including LoRA-wrapped ``<name>.original.*`` layouts), loads strictly,
    and rejects deltas-only LoRA payloads (they cannot be served without the
    exact training base).

    Shared by ModelManager and the realtime/SOTA serving paths so every
    consumer loads trained artifacts identically.

    Raises:
        RuntimeError: Unrecognized format, deltas-only artifact, or a state
            dict that does not fit its detected architecture.
    """
    # Explicit architecture tag (written by newer artifacts) wins over the
    # key-sniffing heuristic below, which stays as the legacy fallback.
    if isinstance(checkpoint, dict) and checkpoint.get('architecture') == 'diffusion_mel':
        from ..models.diffusion_decoder import DiffusionMelDecoder
        d_state = checkpoint.get('model_state_dict')
        if not isinstance(d_state, dict):
            raise RuntimeError(
                f"diffusion_mel checkpoint {model_path} is missing model_state_dict"
            )
        model = DiffusionMelDecoder(device=device, **(checkpoint.get('config') or {}))
        model.load_state_dict(d_state, strict=True)
        model.to(device)
        model.eval()
        logger.info(
            "Loaded DiffusionMelDecoder from %s (n_mels=%d, hidden=%d, blocks=%d)",
            model_path, model.n_mels, model.hidden_dim, model.n_blocks,
        )
        return model

    lora_config: Dict[str, Any] = {}
    state = None
    if isinstance(checkpoint, dict):
        for wrapper_key in ('model_state_dict', 'model', 'state_dict'):
            if isinstance(checkpoint.get(wrapper_key), dict):
                state = checkpoint[wrapper_key]
                lora_config = checkpoint.get('lora_config') or {}
                break
        else:
            if isinstance(checkpoint.get('lora_state'), dict):
                raise RuntimeError(
                    f"Checkpoint {model_path} contains only LoRA deltas and no base "
                    "weights; it cannot be served without the exact training base. "
                    "Retrain the profile to produce a self-contained artifact."
                )
            if checkpoint and all(hasattr(v, 'shape') for v in checkpoint.values()):
                state = checkpoint
    if state is None:
        raise RuntimeError(f"Unrecognized voice model checkpoint format: {model_path}")

    keys = set(state.keys())
    base_keys = {k for k in keys if '.adapter.lora_' not in k}

    if not base_keys:
        raise RuntimeError(
            f"Checkpoint {model_path} contains only LoRA deltas and no base "
            "weights; it cannot be served without the exact training base. "
            "Retrain the profile to produce a self-contained artifact."
        )

    def _param(*names):
        # LoRA injection wraps a Linear as <name>.original.* + <name>.adapter.*
        for name in names:
            if name in state:
                return state[name]
        return None

    input_proj = _param('input_proj.weight', 'input_proj.original.weight')
    gamma_proj = _param(
        'speaker_film.gamma_proj.weight',
        'speaker_film.gamma_proj.original.weight',
    )
    if input_proj is not None and gamma_proj is not None:
        from ..models.svc_decoder import CoMoSVCDecoder

        hidden_dim, content_plus_pitch = input_proj.shape
        speaker_dim = gamma_proj.shape[1]
        n_mels = state['output_proj.2.weight'].shape[0]
        layer_indices = [
            int(k.split('.')[2]) for k in keys
            if k.startswith('backbone.layers.') and k.split('.')[2].isdigit()
        ]
        n_layers = (max(layer_indices) + 1) if layer_indices else 8
        model = CoMoSVCDecoder(
            content_dim=content_plus_pitch // 2,
            pitch_dim=content_plus_pitch // 2,
            speaker_dim=speaker_dim,
            n_mels=n_mels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            device=device,
        )
        lora_keys = keys - base_keys
        if lora_keys:
            rank_source = state.get('input_proj.adapter.lora_A')
            rank = int(lora_config.get('rank') or (rank_source.shape[0] if rank_source is not None else 8))
            alpha = int(lora_config.get('alpha') or 16)
            dropout = float(lora_config.get('dropout') or 0.0)
            model.inject_lora(rank=rank, alpha=alpha, dropout=dropout)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        logger.info(
            "Loaded CoMoSVC voice model from %s (n_mels=%d, hidden=%d, layers=%d, lora=%s)",
            model_path, n_mels, hidden_dim, n_layers, bool(lora_keys),
        )
        return model

    if 'content_proj.weight' in keys:
        from ..models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc.load_pretrained(model_path, device=device)
        return model

    raise RuntimeError(
        f"Checkpoint {model_path} does not match any known voice model "
        "architecture (expected CoMoSVCDecoder or SoVitsSvc keys)."
    )


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
        self._pitch_encoder_dim = None
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

        self._pitch_encoder = PitchEncoder(output_size=768).to(self.device)  # Default SoVitsSvc contract.
        self._pitch_encoder_dim = 768

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
        """Load a trained per-speaker voice model for conversion.

        Supports both artifact families produced in this repo:
        - CoMoSVCDecoder states (what the training job manager saves for
          full-model and self-contained adapter artifacts)
        - SoVitsSvc states (legacy/e2e-test artifacts)

        The artifact family and model dimensions are derived from the state
        dict itself, so a checkpoint is always loaded into the class that
        produced it (loading across classes silently yields a random model).

        Args:
            model_path: Path to trained model checkpoint file.
            speaker_id: Unique identifier for this speaker (used in infer()).
            speaker_embedding: Optional 256-dim speaker embedding vector.
                If provided, stored for later retrieval.

        Raises:
            FileNotFoundError: If model_path does not exist.
            RuntimeError: If the checkpoint format is unrecognized, is a
                deltas-only LoRA artifact (not servable without its training
                base), or does not fit its detected architecture.
        """
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = self._build_voice_model(checkpoint, model_path)
        self._sovits_models[speaker_id] = model
        if speaker_embedding is not None:
            self._speaker_embeddings[speaker_id] = speaker_embedding

    def _build_voice_model(self, checkpoint: Any, model_path: str):
        """Construct the correct model class for a trained checkpoint."""
        return build_voice_model_from_checkpoint(checkpoint, model_path, self.device)

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
            )  # [1, N, 768]

        # 2. Extract F0 (HOW it's being sung - original artist's melody)
        f0, voiced, _ = librosa.pyin(
            audio, fmin=50, fmax=1100, sr=sr, hop_length=512
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)  # [1, T]
        # 5. SoVitsSvc inference -> mel spectrogram
        sovits = self._sovits_models[speaker_id]
        expected_pitch_dim = int(getattr(sovits, "pitch_dim", 768))
        if self._pitch_encoder is None or self._pitch_encoder_dim != expected_pitch_dim:
            from ..models.encoder import PitchEncoder
            self._pitch_encoder = PitchEncoder(output_size=expected_pitch_dim).to(self.device)
            self._pitch_encoder_dim = expected_pitch_dim

        with torch.no_grad():
            pitch = self._pitch_encoder(f0_tensor)

        # 3. Frame alignment — align content and pitch to the VOCODER GRID so
        # the vocoder renders exactly len(audio) samples. Aligning to
        # min(content, pitch) instead produced a mel at the pyin frame rate
        # (~43 fps); the hop-256 vocoder then emitted ~half-length audio,
        # which step 7 stretched back to length by resampling — dropping the
        # pitch a full octave (measured 240->114 Hz) and scrambling the
        # melody. len(audio)//hop is the frame count whose synthesis is
        # already the right length, so no pitch-corrupting resample is needed.
        vocoder_hop = int(getattr(self._vocoder, 'hop_size', 256) or 256)
        if min(content.shape[1], pitch.shape[1]) == 0:
            return np.zeros_like(audio)
        target_frames = max(1, len(audio) // vocoder_hop)

        content = F.interpolate(
            content.transpose(1, 2), size=target_frames,
            mode='linear', align_corners=False
        ).transpose(1, 2)  # [1, target_frames, 768]
        pitch = F.interpolate(
            pitch.transpose(1, 2), size=target_frames,
            mode='linear', align_corners=False
        ).transpose(1, 2)  # [1, target_frames, 768]

        # 4. Speaker embedding (WHO should sing - target person)
        speaker = torch.from_numpy(speaker_embedding).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            mel_pred = sovits.infer(content, pitch, speaker)  # [1, 80, target_frames]

        # Decoder predicts a [0,1]-normalized mel; denormalize back to the
        # log-mel the vocoder was trained on.
        from ..models.vocoder import denormalize_log_mel
        mel_pred = denormalize_log_mel(mel_pred)

        # 6. HiFiGAN vocoder -> audio waveform
        with torch.no_grad():
            output_audio = self._vocoder.synthesize(mel_pred)  # [1, T_audio]

        # 7. Enforce exact length by trim/pad — NOT resampling, which would
        # change the pitch. The vocoder-grid alignment above already makes the
        # output essentially len(audio) samples; this only fixes off-by-a-frame.
        output_np = output_audio.squeeze(0).cpu().numpy().astype(np.float32)
        if len(output_np) > len(audio):
            output_np = output_np[:len(audio)]
        elif len(output_np) < len(audio):
            output_np = np.pad(output_np, (0, len(audio) - len(output_np)))

        # 8. Normalize
        peak = np.abs(output_np).max()
        if peak > 0.95:
            output_np = output_np * (0.95 / peak)
        elif peak > 0:
            output_np = output_np * (0.9 / peak)

        return output_np
