"""Vocal/instrumental separation using Demucs HTDemucs model.

No fallback behavior - raises RuntimeError if Demucs is unavailable.
"""
import logging
import gc
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Test suites patch these module-level hooks directly. They remain lazy-populated
# from demucs during ``VocalSeparator`` initialization.
get_model = None
apply_model = None


class VocalSeparator:
    """Separates vocals from instrumental using Demucs HTDemucs.

    Uses the pretrained HTDemucs model for high-quality source separation.
    Raises RuntimeError if Demucs cannot be loaded - no silent fallback.
    """

    def __init__(self, device=None, model_name: str = 'htdemucs',
                 segment: Optional[float] = None):
        """Initialize VocalSeparator.

        Args:
            device: Torch device for inference.
            model_name: Demucs model name (e.g. 'htdemucs', 'htdemucs_ft').
            segment: Segment length in seconds for chunked processing.
                     Lower values use less GPU memory. None uses model default.

        Raises:
            RuntimeError: If demucs package is not installed.
        """
        global get_model, apply_model

        explicit_backend_override = (
            callable(get_model)
            and callable(apply_model)
            and (
                hasattr(get_model, 'assert_called')
                or hasattr(apply_model, 'assert_called')
            )
        )

        if explicit_backend_override:
            demucs_get_model = get_model
            demucs_apply_model = apply_model
        else:
            try:
                from demucs.pretrained import get_model as demucs_get_model
                from demucs.apply import apply_model as demucs_apply_model
            except ImportError as e:
                raise RuntimeError(
                    f"Demucs is required for vocal separation but is not installed: {e}. "
                    f"Install with: pip install demucs"
                )

            get_model = demucs_get_model
            apply_model = demucs_apply_model

        if device is None:
            resolved_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, torch.device):
            resolved_device = device
        else:
            resolved_device = torch.device(device)

        self.device = resolved_device
        self.model_name = model_name
        self.segment = segment
        self._model = None
        self._apply_model = apply_model
        self._get_model = get_model

    @staticmethod
    def _normalize_sample_rate(rate) -> int:
        """Convert sample-rate metadata to a stable integer for resampling APIs."""
        normalized = int(round(float(rate)))
        if normalized <= 0:
            raise RuntimeError(f"Invalid sample rate for separation: {rate!r}")
        return normalized

    def _load_model(self):
        """Lazy-load Demucs model.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        if self._model is not None:
            return

        try:
            self._model = self._get_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info(
                f"Demucs model '{self.model_name}' loaded on {self.device} "
                f"(sources: {self._model.sources}, samplerate: {self._model.samplerate})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Demucs model '{self.model_name}': {e}"
            )

    def unload(self) -> None:
        """Release the Demucs model and cached accelerator memory."""
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        logger.info("Demucs model unloaded")

    @property
    def model_sample_rate(self) -> int:
        """Return the model's expected sample rate."""
        self._load_model()
        return self._normalize_sample_rate(self._model.samplerate)

    @property
    def sources(self):
        """Return list of source names the model separates."""
        self._load_model()
        return list(self._model.sources)

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate audio into vocals and instrumental.

        Args:
            audio: Mono or stereo audio as numpy array.
                   Shape: (samples,) for mono or (channels, samples) for stereo.
            sr: Sample rate of input audio.

        Returns:
            Dict with 'vocals' and 'instrumental' numpy arrays (mono, float32).

        Raises:
            RuntimeError: If model cannot be loaded or separation fails.
            ValueError: If audio is empty or has invalid shape.
        """
        if audio.size == 0:
            raise ValueError("Cannot separate empty audio")
        if audio.ndim > 2:
            raise ValueError(f"Audio must be 1D (mono) or 2D (stereo), got {audio.ndim}D")

        self._load_model()

        # Prepare input: demucs expects (batch, channels, samples)
        if audio.ndim == 1:
            # Mono -> stereo (duplicate channel)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.expand(-1, 2, -1).contiguous()  # [1, 2, samples]
        else:
            # Already multichannel
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, C, samples]

        audio_tensor = audio_tensor.to(self.device)

        # Resample to model's expected rate if needed
        model_sr = self._normalize_sample_rate(self._model.samplerate)
        input_sr = self._normalize_sample_rate(sr)
        if input_sr != model_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(input_sr, model_sr).to(self.device)
            audio_tensor = resampler(audio_tensor)

        # Clear GPU cache before processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Apply model with optional segment size for memory-efficient processing
        apply_kwargs = {}
        if self.segment is not None:
            apply_kwargs['segment'] = self.segment
            # Let demucs use its default overlap (0.25) - don't override

        with torch.no_grad():
            sources = self._apply_model(
                self._model, audio_tensor, **apply_kwargs
            )
        # sources shape: (batch, n_sources, channels, samples)

        # Clear GPU cache after processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract vocals
        source_names = list(self._model.sources)
        if 'vocals' not in source_names:
            raise RuntimeError(
                f"Model '{self.model_name}' does not have a 'vocals' source. "
                f"Available sources: {source_names}"
            )
        vocals_idx = source_names.index('vocals')

        # Vocals: mean across channels -> mono
        vocals = sources[0, vocals_idx].mean(dim=0).cpu().numpy()

        # Instrumental: sum all non-vocal sources, mean across channels -> mono
        non_vocal_indices = [i for i in range(len(source_names)) if i != vocals_idx]
        instrumental = sources[0, non_vocal_indices].sum(dim=0).mean(dim=0).cpu().numpy()

        # Resample back to input sample rate if needed
        if input_sr != model_sr:
            import librosa
            vocals = librosa.resample(vocals, orig_sr=model_sr, target_sr=input_sr)
            instrumental = librosa.resample(instrumental, orig_sr=model_sr, target_sr=input_sr)

        # Match original length
        orig_len = len(audio) if audio.ndim == 1 else audio.shape[-1]
        if len(vocals) > orig_len:
            vocals = vocals[:orig_len]
            instrumental = instrumental[:orig_len]
        elif len(vocals) < orig_len:
            # Pad with zeros if resampling caused shortening
            vocals = np.pad(vocals, (0, orig_len - len(vocals)))
            instrumental = np.pad(instrumental, (0, orig_len - len(instrumental)))

        result = {
            'vocals': vocals.astype(np.float32),
            'instrumental': instrumental.astype(np.float32),
        }
        del audio_tensor, sources, vocals, instrumental
        if input_sr != model_sr and 'resampler' in locals():
            del resampler
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return result
