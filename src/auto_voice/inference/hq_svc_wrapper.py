"""HQ-SVC Wrapper for cutting-edge voice conversion and super-resolution.

HQ-SVC (AAAI 2026) provides:
  - Zero-shot singing voice conversion
  - Super-resolution: 16kHz → 44.1kHz upsampling

Architecture:
  FACodec (content extraction @ 16kHz) → DDSP (initial synthesis)
  → Diffusion (mel refinement) → NSF-HiFiGAN (vocoder @ 44.1kHz)

No fallback behavior: raises RuntimeError on failure.
"""
import logging
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# HQ-SVC repository path
HQ_SVC_ROOT = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'models', 'hq-svc'
)
HQ_SVC_ROOT = os.path.abspath(HQ_SVC_ROOT)

# Minimum input duration (500ms at 16kHz = 8000 samples)
# RMVPE F0 extraction needs ~500ms minimum for reliable pitch tracking
MIN_DURATION_SAMPLES_16K = 8000


class HQSVCWrapper:
    """Wrapper for HQ-SVC cutting-edge voice conversion.

    Provides two modes:
      1. Super-resolution: Upsample 16kHz audio to 44.1kHz
      2. Voice conversion: Convert source voice to target speaker

    All operations produce 44.1kHz output audio.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
        require_gpu: bool = True,
    ):
        """Initialize HQ-SVC wrapper with all components.

        Args:
            device: Target device (defaults to CUDA if available)
            config_path: Path to HQ-SVC config YAML
            require_gpu: If True, raise RuntimeError if CUDA unavailable
        """
        # Check GPU requirement: either CUDA unavailable or explicit CPU device
        if require_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for HQSVCWrapper")
            if device is not None and device.type == 'cpu':
                raise RuntimeError("CUDA is required for HQSVCWrapper but CPU device specified")

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.output_sample_rate = 44100
        self.encoder_sample_rate = 16000

        # Add HQ-SVC paths to sys.path for imports
        self._setup_paths()

        # Load configuration
        config_path = config_path or os.path.join(
            HQ_SVC_ROOT, 'configs', 'hq_svc_infer.yaml'
        )
        self.args = self._load_config(config_path)

        # Initialize all components
        try:
            self._init_models()
        finally:
            # Restore original working directory
            if hasattr(self, '_original_cwd'):
                os.chdir(self._original_cwd)

        logger.info(f"HQSVCWrapper initialized on {self.device}")

    def _setup_paths(self):
        """Add HQ-SVC paths to sys.path for imports.

        We need to carefully manage sys.path to avoid conflicts with
        other 'utils' packages. HQ-SVC must be at the front.
        """
        # Remove any existing utils-related paths that might conflict
        # and add HQ-SVC root at the very beginning
        if HQ_SVC_ROOT not in sys.path:
            sys.path.insert(0, HQ_SVC_ROOT)

        # Store original cwd and change to HQ-SVC root for imports
        self._original_cwd = os.getcwd()
        os.chdir(HQ_SVC_ROOT)

    def _load_config(self, config_path: str):
        """Load HQ-SVC configuration from YAML file."""
        from logger.utils import load_config

        if not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found: {config_path}")

        args = load_config(config_path)
        args.config = config_path
        args.device = str(self.device)

        # Set defaults if not in config
        if not hasattr(args, 'sample_rate'):
            args.sample_rate = 44100
        if not hasattr(args, 'encoder_sr'):
            args.encoder_sr = 16000
        if not hasattr(args, 'infer_speedup'):
            args.infer_speedup = 10
        if not hasattr(args, 'infer_method'):
            args.infer_method = 'dpm-solver'
        if not hasattr(args, 'vocoder'):
            args.vocoder = 'nsf-hifigan'

        return args

    def _init_models(self):
        """Initialize all HQ-SVC model components."""
        from utils.models.models_v2_beta import load_hq_svc
        from utils.vocoder import Vocoder
        from utils.data_preprocessing import (
            load_facodec, load_f0_extractor, load_volume_extractor
        )

        # Vocoder (NSF-HiFiGAN @ 44.1kHz)
        vocoder_path = os.path.join(
            HQ_SVC_ROOT, 'utils', 'pretrain', 'nsf_hifigan', 'model'
        )
        self.vocoder = Vocoder(
            vocoder_type='nsf-hifigan',
            vocoder_ckpt=vocoder_path,
            device=self.device
        )

        # HQ-SVC generator (DDSP + Diffusion)
        model_path = os.path.join(
            HQ_SVC_ROOT, self.args.model_path
        )
        if not os.path.exists(model_path):
            # Try absolute path from config
            model_path = self.args.model_path
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model weights not found: {model_path}")

        self.net_g = load_hq_svc(
            mode='infer',
            device=str(self.device),
            model_path=model_path,
            args=self.args
        )
        self.net_g.eval()

        # FACodec encoder/decoder (content extraction)
        self.fa_encoder, self.fa_decoder = load_facodec(self.device)

        # F0 extractor (RMVPE)
        self.f0_extractor = load_f0_extractor(self.args)

        # Volume extractor
        self.volume_extractor = load_volume_extractor(self.args)

        logger.info("All HQ-SVC components loaded successfully")

    def _resample(
        self, audio: torch.Tensor, from_sr: int, to_sr: int
    ) -> torch.Tensor:
        """Resample audio between sample rates using linear interpolation."""
        if from_sr == to_sr:
            return audio

        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
            target_len = int(audio.shape[2] * to_sr / from_sr)
            resampled = F.interpolate(
                audio, size=target_len, mode='linear', align_corners=False
            )
            return resampled.squeeze(0).squeeze(0)
        else:
            audio = audio.unsqueeze(0)
            target_len = int(audio.shape[2] * to_sr / from_sr)
            resampled = F.interpolate(
                audio, size=target_len, mode='linear', align_corners=False
            )
            return resampled.squeeze(0)

    def _to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert to mono by averaging channels."""
        if audio.dim() == 1:
            return audio
        if audio.dim() == 2:
            return audio.mean(dim=0)
        raise RuntimeError(f"Unexpected audio shape: {audio.shape}")

    def _wav_pad(self, wav: np.ndarray, multiple: int = 200) -> np.ndarray:
        """Pad waveform to multiple of given value."""
        seq_len = wav.shape[0]
        padded_len = ((seq_len + (multiple - 1)) // multiple) * multiple
        if padded_len > seq_len:
            wav = np.pad(wav, (0, padded_len - seq_len), mode='reflect')
        return wav

    def _process_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> Dict:
        """Process audio to extract all features for HQ-SVC.

        Uses the original HQ-SVC get_processed_file function for correct
        feature extraction and alignment.

        Returns dict with: vq_post, spk, f0, f0_origin, vol, mel
        """
        import tempfile
        import soundfile as sf
        from utils.data_preprocessing import get_processed_file

        # Convert to mono and numpy
        audio_mono = self._to_mono(audio)

        # Resample to 44.1kHz (expected input rate for HQ-SVC preprocessing)
        audio_44k = self._resample(audio_mono, sample_rate, 44100)
        audio_44k_np = audio_44k.cpu().numpy()

        # Save to temp file for get_processed_file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_44k_np, 44100)

        try:
            # Use original HQ-SVC preprocessing
            data = get_processed_file(
                temp_path,
                sr=44100,
                encoder_sr=16000,
                mel_extractor=self.vocoder,
                volume_extractor=self.volume_extractor,
                f0_extractor=self.f0_extractor,
                fa_encoder=self.fa_encoder,
                fa_decoder=self.fa_decoder,
                content_encoder=None,
                spk_encoder=None,
                device=self.device,
            )
        finally:
            # Clean up temp file
            os.unlink(temp_path)

        if data is None:
            raise RuntimeError("Failed to process audio - check F0 extraction")

        return data

    def extract_speaker_embedding(
        self,
        audio: Union[torch.Tensor, List[torch.Tensor]],
        sample_rate: int,
    ) -> torch.Tensor:
        """Extract speaker embedding from reference audio.

        Args:
            audio: Reference audio tensor(s), [T] or list of [T]
            sample_rate: Audio sample rate

        Returns:
            [256] L2-normalized speaker embedding
        """
        if isinstance(audio, list):
            # Process multiple references and average
            embeddings = []
            for a in audio:
                data = self._process_audio(a, sample_rate)
                embeddings.append(data['spk'].squeeze())
            embedding = torch.stack(embeddings).mean(dim=0)
        else:
            data = self._process_audio(audio, sample_rate)
            embedding = data['spk'].squeeze()

        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=0)
        return embedding

    def super_resolve(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> Dict:
        """Upsample audio to 44.1kHz using HQ-SVC super-resolution.

        This mode uses the source audio's own speaker embedding,
        effectively reconstructing the audio at higher quality/sample rate.

        Args:
            audio: Input audio tensor [T]
            sample_rate: Input sample rate
            on_progress: Optional callback(stage, progress)

        Returns:
            Dict with 'audio', 'sample_rate', 'metadata'
        """
        start_time = time.time()

        # Validate input
        audio_mono = self._to_mono(audio)
        duration_16k = int(len(audio_mono) * 16000 / sample_rate)
        if duration_16k < MIN_DURATION_SAMPLES_16K:
            raise RuntimeError(
                f"Audio too short: {duration_16k} samples at 16kHz "
                f"(minimum {MIN_DURATION_SAMPLES_16K})"
            )

        def report(stage: str, progress: float):
            if on_progress:
                on_progress(stage, progress)

        report('preprocessing', 0.0)

        # Process source audio (extracts its own speaker embedding)
        src_data = self._process_audio(audio, sample_rate)

        report('encoding', 0.3)

        # Use source's own speaker embedding for reconstruction
        spk_emb = src_data['spk'].squeeze().to(self.device)

        # Prepare inputs
        vq_post = src_data['vq_post'].unsqueeze(0).to(self.device)
        f0 = src_data['f0'].unsqueeze(0).to(self.device)
        vol = src_data['vol'].unsqueeze(0).to(self.device)

        report('diffusion', 0.5)

        # Run HQ-SVC inference
        with torch.no_grad():
            mel_g = self.net_g(
                vq_post, f0, vol, spk_emb,
                gt_spec=None,
                infer=True,
                infer_speedup=self.args.infer_speedup,
                method=self.args.infer_method,
                vocoder=self.vocoder
            )

            report('vocoder', 0.8)

            # Vocoder synthesis
            if self.args.vocoder == 'nsf-hifigan':
                wav_out = self.vocoder.infer(mel_g, f0)
            else:
                wav_out = self.vocoder.infer(mel_g)

        # Post-process output
        wav_out = wav_out.squeeze().cpu()

        # Normalize to [-1, 1]
        peak = wav_out.abs().max()
        if peak > 1e-6:
            wav_out = wav_out / peak * 0.95

        report('complete', 1.0)

        elapsed = time.time() - start_time
        output_duration = len(wav_out) / 44100

        logger.info(
            f"HQ-SVC super-resolution: {output_duration:.2f}s audio "
            f"in {elapsed:.2f}s ({elapsed/output_duration:.1f}x realtime)"
        )

        return {
            'audio': wav_out,
            'sample_rate': 44100,
            'metadata': {
                'processing_time': elapsed,
                'output_duration': output_duration,
                'mode': 'super_resolution',
                'device': str(self.device),
            },
        }

    def convert(
        self,
        source_audio: torch.Tensor,
        source_sample_rate: int,
        target_audio: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        target_sample_rate: Optional[int] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        pitch_shift: int = 0,
        auto_pitch: bool = False,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> Dict:
        """Convert source audio to target speaker voice.

        Either target_audio or speaker_embedding must be provided.

        Args:
            source_audio: Source audio tensor [T]
            source_sample_rate: Source sample rate
            target_audio: Target reference audio(s) for speaker embedding
            target_sample_rate: Target audio sample rate
            speaker_embedding: Pre-computed [256] speaker embedding
            pitch_shift: Semitone shift (positive = higher pitch)
            auto_pitch: Auto-compute pitch shift to match target
            on_progress: Optional callback(stage, progress)

        Returns:
            Dict with 'audio', 'sample_rate', 'metadata'
        """
        start_time = time.time()

        # Validate inputs
        if target_audio is None and speaker_embedding is None:
            raise RuntimeError(
                "Either target_audio or speaker_embedding must be provided"
            )

        source_mono = self._to_mono(source_audio)
        duration_16k = int(len(source_mono) * 16000 / source_sample_rate)
        if duration_16k < MIN_DURATION_SAMPLES_16K:
            raise RuntimeError(
                f"Audio too short: {duration_16k} samples at 16kHz "
                f"(minimum {MIN_DURATION_SAMPLES_16K})"
            )

        def report(stage: str, progress: float):
            if on_progress:
                on_progress(stage, progress)

        report('preprocessing', 0.0)

        # Get target speaker embedding
        if speaker_embedding is not None:
            spk_emb = speaker_embedding.to(self.device)
            all_tar_f0 = None
        else:
            # Extract from target audio(s)
            if isinstance(target_audio, list):
                spk_list = []
                f0_list = []
                for t_audio in target_audio:
                    t_data = self._process_audio(t_audio, target_sample_rate)
                    spk_list.append(t_data['spk'])
                    f0_list.append(t_data['f0_origin'])
                spk_emb = torch.stack(spk_list).mean(dim=0).squeeze().to(self.device)
                all_tar_f0 = np.concatenate(f0_list)
            else:
                t_data = self._process_audio(target_audio, target_sample_rate)
                spk_emb = t_data['spk'].squeeze().to(self.device)
                all_tar_f0 = t_data['f0_origin']

        report('encoding', 0.3)

        # Process source audio
        src_data = self._process_audio(source_audio, source_sample_rate)

        # Prepare inputs
        vq_post = src_data['vq_post'].unsqueeze(0).to(self.device)
        f0 = src_data['f0'].unsqueeze(0).to(self.device)
        vol = src_data['vol'].unsqueeze(0).to(self.device)

        # Auto pitch adjustment
        computed_shift = pitch_shift
        if auto_pitch and all_tar_f0 is not None:
            src_f0_valid = src_data['f0_origin'][src_data['f0_origin'] > 0]
            tar_f0_valid = all_tar_f0[all_tar_f0 > 0]
            if len(src_f0_valid) > 0 and len(tar_f0_valid) > 0:
                computed_shift = round(
                    12 * np.log2(tar_f0_valid.mean() / src_f0_valid.mean())
                )

        # Apply pitch shift
        if computed_shift != 0:
            f0 = f0 * (2 ** (computed_shift / 12))

        report('diffusion', 0.5)

        # Run HQ-SVC inference
        with torch.no_grad():
            mel_g = self.net_g(
                vq_post, f0, vol, spk_emb,
                gt_spec=None,
                infer=True,
                infer_speedup=self.args.infer_speedup,
                method=self.args.infer_method,
                vocoder=self.vocoder
            )

            report('vocoder', 0.8)

            # Vocoder synthesis
            if self.args.vocoder == 'nsf-hifigan':
                wav_out = self.vocoder.infer(mel_g, f0)
            else:
                wav_out = self.vocoder.infer(mel_g)

        # Post-process
        wav_out = wav_out.squeeze().cpu()

        # Normalize
        peak = wav_out.abs().max()
        if peak > 1e-6:
            wav_out = wav_out / peak * 0.95

        report('complete', 1.0)

        elapsed = time.time() - start_time
        output_duration = len(wav_out) / 44100

        logger.info(
            f"HQ-SVC conversion: {output_duration:.2f}s audio "
            f"in {elapsed:.2f}s ({elapsed/output_duration:.1f}x realtime)"
        )

        return {
            'audio': wav_out,
            'sample_rate': 44100,
            'metadata': {
                'processing_time': elapsed,
                'output_duration': output_duration,
                'mode': 'voice_conversion',
                'pitch_shift': computed_shift,
                'device': str(self.device),
            },
        }
