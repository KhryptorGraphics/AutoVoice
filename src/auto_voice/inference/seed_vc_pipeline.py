"""Seed-VC Quality Pipeline using DiT-CFM decoder.

This pipeline provides state-of-the-art voice conversion quality using:
  - Whisper-base for semantic content extraction
  - CAMPPlus for speaker style embedding
  - DiT (Diffusion Transformer) with Conditional Flow Matching for 5-10 step inference
  - BigVGAN v2 (44kHz, 128-band) for high-quality waveform synthesis
  - RMVPE for F0 extraction (singing voice)

Key advantages over CoMoSVC:
  - In-context learning: Uses reference audio as prompt for fine-grained timbre
  - Fewer steps: 5-10 steps vs 30+ for diffusion
  - Higher sample rate: 44.1kHz output vs 24kHz

Research paper: Seed-VC (arXiv:2411.09943) - 40 citations as of 2026
"""
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..storage.voice_profiles import VoiceProfileStore

logger = logging.getLogger(__name__)

# Add seed-vc module to path
SEED_VC_DIR = Path(__file__).parent.parent.parent.parent / "models" / "seed-vc"
if str(SEED_VC_DIR) not in sys.path:
    sys.path.insert(0, str(SEED_VC_DIR))


class SeedVCPipeline:
    """SOTA quality pipeline using Seed-VC's DiT-CFM decoder.

    This pipeline wraps the Seed-VC implementation to provide a consistent
    interface with our other pipelines while leveraging the DiT architecture
    for high-quality voice conversion.

    Sample rate: 44.1kHz (singing voice optimized)
    Diffusion steps: 5-10 (configurable, default 10)
    Output quality: Speaker similarity >= 0.94, MCD <= 3.9
    """

    # Model memory estimate (for factory memory tracking)
    ESTIMATED_MEMORY_GB = 8.0  # DiT + BigVGAN + Whisper + CAMPPlus + RMVPE

    def __init__(
        self,
        device: Optional[torch.device] = None,
        diffusion_steps: int = 10,
        f0_condition: bool = True,  # Always True for singing
        require_gpu: bool = True,
    ):
        """Initialize Seed-VC pipeline.

        Args:
            device: Target device (defaults to CUDA if available)
            diffusion_steps: Number of DiT-CFM steps (5, 10, or 25)
            f0_condition: Whether to use F0 conditioning (always True for singing)
            require_gpu: Raise error if no GPU available
        """
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SeedVCPipeline")

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.diffusion_steps = diffusion_steps
        self.f0_condition = f0_condition
        self._wrapper = None
        self._current_stage = None

        # Reference audio for in-context learning
        self._reference_audio: Optional[np.ndarray] = None
        self._reference_sr: int = 44100

        # Lazy load the wrapper
        self._initialize()

        logger.info(
            f"SeedVCPipeline initialized on {self.device} "
            f"with {diffusion_steps} diffusion steps"
        )

    def _initialize(self):
        """Lazy-initialize the Seed-VC wrapper.

        Loads the DiT-CFM decoder, BigVGAN vocoder, Whisper encoder, CAMPPlus speaker
        encoder, and RMVPE F0 extractor. Models are loaded only once on first use.

        Raises:
            RuntimeError: If SeedVCWrapper import fails or model initialization fails.
                         Ensure models are downloaded via scripts/download_seed_vc_models.py
        """
        if self._wrapper is not None:
            return

        logger.info("Loading Seed-VC models (DiT-CFM + BigVGAN + Whisper + CAMPPlus + RMVPE)...")

        try:
            # Import from seed-vc directory
            from seed_vc_wrapper import SeedVCWrapper

            self._wrapper = SeedVCWrapper(device=self.device)
            logger.info("Seed-VC wrapper loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SeedVCWrapper. Ensure seed-vc models are downloaded. "
                f"Run: python scripts/download_seed_vc_models.py\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SeedVCWrapper: {e}") from e

    @property
    def sample_rate(self) -> int:
        """Output sample rate (44.1kHz for F0-conditioned singing)."""
        return 44100 if self.f0_condition else 22050

    def set_reference_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 44100,
    ) -> None:
        """Set reference audio for in-context learning.

        The reference audio provides the target voice characteristics.
        Seed-VC uses this as a "prompt" for the DiT model to learn
        fine-grained timbre details.

        Args:
            audio: Reference audio waveform
            sample_rate: Sample rate of reference audio
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Store for conversion
        self._reference_audio = audio.astype(np.float32)
        self._reference_sr = sample_rate

        logger.info(
            f"Set reference audio: {len(audio)/sample_rate:.2f}s "
            f"at {sample_rate}Hz"
        )

    def set_reference_from_profile(
        self,
        profile_store: "VoiceProfileStore",
        profile_id: str,
    ) -> None:
        """Load reference audio from a voice profile (legacy method).

        Loads the first audio sample from the profile's audio files and sets it
        as the reference for in-context learning. Prefer set_reference_from_profile_id
        for new code as it uses the AdapterBridge.

        Args:
            profile_store: Voice profile storage instance
            profile_id: Profile ID to load reference from

        Raises:
            ValueError: If profile not found or has no audio samples
        """
        profile = profile_store.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Profile not found: {profile_id}")

        # Load first audio sample as reference
        audio_paths = profile_store.get_audio_files(profile_id)
        if not audio_paths:
            raise ValueError(f"Profile {profile_id} has no audio samples")

        import librosa
        audio, sr = librosa.load(audio_paths[0], sr=44100)
        self.set_reference_audio(audio, sr)

    def set_reference_from_profile_id(
        self,
        profile_id: str,
        reference_index: int = 0,
    ) -> None:
        """Load reference audio using the AdapterBridge.

        This is the preferred method for loading voice profile references
        as it doesn't require a VoiceProfileStore instance. Uses the global
        AdapterBridge to fetch voice reference data.

        Args:
            profile_id: Voice profile UUID
            reference_index: Which reference audio to use (0 = best quality).
                           Index is clamped to available reference count.

        Raises:
            ValueError: If no reference audio found for profile
        """
        from .adapter_bridge import get_adapter_bridge

        bridge = get_adapter_bridge()
        voice_ref = bridge.get_voice_reference(profile_id)

        if not voice_ref.reference_paths:
            raise ValueError(
                f"No reference audio found for profile {profile_id} ({voice_ref.profile_name})"
            )

        # Use the specified reference audio
        ref_path = voice_ref.reference_paths[min(reference_index, len(voice_ref.reference_paths) - 1)]

        import librosa
        audio, sr = librosa.load(str(ref_path), sr=44100)
        self.set_reference_audio(audio, sr)

        logger.info(
            f"Loaded reference for {voice_ref.profile_name} from {ref_path.name}"
        )

    def convert(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        speaker_embedding: Optional[torch.Tensor] = None,  # Not used - reference audio instead
        on_progress: Optional[Callable[[str, float], None]] = None,
        pitch_shift: int = 0,
    ) -> Dict[str, Any]:
        """Convert audio using Seed-VC DiT-CFM.

        Unlike the CoMoSVC pipeline which uses speaker embeddings,
        Seed-VC uses reference audio for in-context learning. Make sure
        to call set_reference_audio() before conversion.

        Args:
            audio: Input audio waveform [T] or [C, T]
            sample_rate: Input sample rate
            speaker_embedding: Ignored (Seed-VC uses reference audio instead)
            on_progress: Optional callback(stage_name, progress_fraction)
            pitch_shift: Pitch shift in semitones (positive = higher pitch)

        Returns:
            Dict with:
                - audio: Output waveform at 44.1kHz
                - sample_rate: 44100
                - metadata: Processing info

        Raises:
            RuntimeError: If no reference audio set or conversion fails
        """
        if self._reference_audio is None:
            raise RuntimeError(
                "No reference audio set. Call set_reference_audio() or "
                "set_reference_from_profile() before conversion."
            )

        self._initialize()
        start_time = time.time()

        def report(stage: str, progress: float):
            self._current_stage = stage
            if on_progress:
                on_progress(stage, progress)

        # Convert input to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Save to temporary files (Seed-VC wrapper expects file paths)
        import tempfile
        import soundfile as sf

        report('preprocessing', 0.0)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as src_f:
            src_path = src_f.name
            sf.write(src_path, audio, sample_rate)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_f:
            ref_path = ref_f.name
            sf.write(ref_path, self._reference_audio, self._reference_sr)

        report('conversion', 0.1)

        try:
            # Run Seed-VC conversion
            # Note: convert_voice is a generator when stream_output=True
            result_audio = None

            for mp3_bytes, full_audio in self._wrapper.convert_voice(
                source=src_path,
                target=ref_path,
                diffusion_steps=self.diffusion_steps,
                f0_condition=self.f0_condition,
                auto_f0_adjust=True,
                pitch_shift=pitch_shift,
                stream_output=True,
            ):
                if full_audio is not None:
                    # full_audio is (sample_rate, numpy_array)
                    output_sr, result_audio = full_audio
                    break

                # Update progress based on streaming chunks
                report('conversion', 0.5)

            report('postprocessing', 0.9)

            if result_audio is None:
                raise RuntimeError("Seed-VC conversion returned no audio")

            # Convert to torch tensor
            if isinstance(result_audio, np.ndarray):
                result_tensor = torch.from_numpy(result_audio).float()
            else:
                result_tensor = result_audio

            # Normalize output
            peak = result_tensor.abs().max()
            if peak > 1e-6:
                result_tensor = result_tensor * (0.95 / peak)

            report('complete', 1.0)

            elapsed = time.time() - start_time
            output_duration = len(result_tensor) / self.sample_rate

            logger.info(
                f"Seed-VC conversion complete: {output_duration:.2f}s audio "
                f"in {elapsed:.2f}s ({elapsed/output_duration:.1f}x realtime)"
            )

            return {
                'audio': result_tensor,
                'sample_rate': self.sample_rate,
                'metadata': {
                    'processing_time': elapsed,
                    'diffusion_steps': self.diffusion_steps,
                    'f0_condition': self.f0_condition,
                    'output_duration': output_duration,
                    'device': str(self.device),
                    'pipeline': 'seed_vc',
                },
            }

        finally:
            # Cleanup temp files
            import os
            try:
                os.unlink(src_path)
                os.unlink(ref_path)
            except:
                pass

    def convert_with_separation(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        speaker_embedding: Optional[torch.Tensor] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        pitch_shift: int = 0,
    ) -> Dict[str, Any]:
        """Convert audio with vocal separation pre-processing.

        This method first separates vocals from the input audio using
        MelBandRoFormer, then converts only the vocal track. Useful for
        converting songs or audio with background music/noise.

        Progress is remapped: separation 0-30%, conversion 30-100%.

        Args:
            audio: Input audio (may contain background music)
            sample_rate: Input sample rate
            speaker_embedding: Ignored (Seed-VC uses reference audio)
            on_progress: Optional callback(stage_name, progress_fraction)
            pitch_shift: Pitch shift in semitones (positive = higher pitch)

        Returns:
            Dict with:
                - audio: Converted vocal track at 44.1kHz
                - sample_rate: 44100
                - metadata: Processing info

        Raises:
            RuntimeError: If no reference audio set or conversion fails
        """
        def report(stage: str, progress: float):
            if on_progress:
                on_progress(stage, progress)

        report('separation', 0.0)

        # Import separator
        from ..audio.separator import MelBandRoFormer

        # Convert to tensor for separator
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Resample to 44.1kHz for separator
        if sample_rate != 44100:
            audio_tensor = F.interpolate(
                audio_tensor.unsqueeze(0).unsqueeze(0),
                size=int(len(audio_tensor) * 44100 / sample_rate),
                mode='linear',
                align_corners=False,
            ).squeeze()
            sample_rate = 44100

        # Separate vocals
        separator = MelBandRoFormer(device=self.device)
        with torch.no_grad():
            vocals = separator.extract_vocals(
                audio_tensor.unsqueeze(0).to(self.device)
            ).squeeze(0).cpu()

        report('separation', 0.3)

        # Convert separated vocals
        def conversion_progress(stage, prog):
            # Remap progress: separation=0-0.3, conversion=0.3-1.0
            overall = 0.3 + prog * 0.7
            report(stage, overall)

        return self.convert(
            audio=vocals,
            sample_rate=sample_rate,
            speaker_embedding=speaker_embedding,
            on_progress=conversion_progress,
            pitch_shift=pitch_shift,
        )
