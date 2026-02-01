"""MeanVC Streaming Pipeline for Real-Time Voice Conversion.

This pipeline provides single-step voice conversion with streaming support using:
  - FastU2++ ASR model for content feature extraction (bottleneck features)
  - WavLM + ECAPA-TDNN for speaker embeddings
  - DiT with Mean Flows for single-step inference
  - Vocos vocoder for 16kHz output

Key advantages:
  - Single-step inference (1-2 NFE) via mean flows
  - Chunk-wise autoregressive processing with KV-cache
  - <100ms chunk latency for real-time streaming
  - 14M parameters (lightweight compared to full diffusion models)

Research paper: MeanVC (arXiv:2510.08392)
GitHub: https://github.com/ASLP-lab/MeanVC
"""
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)

# Add MeanVC to path
MEANVC_DIR = Path(__file__).parent.parent.parent.parent / "models" / "meanvc"
MEANVC_SRC = MEANVC_DIR / "src"
if str(MEANVC_DIR) not in sys.path:
    sys.path.insert(0, str(MEANVC_DIR))
if str(MEANVC_SRC) not in sys.path:
    sys.path.insert(0, str(MEANVC_SRC))


def _amp_to_db(x: torch.Tensor, min_level_db: float) -> torch.Tensor:
    """Convert amplitude to decibels."""
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S: torch.Tensor, max_abs_value: float, min_db: float) -> torch.Tensor:
    """Normalize spectrogram."""
    return torch.clamp(
        (2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value,
        -max_abs_value, max_abs_value
    )


class MelSpectrogramFeatures(nn.Module):
    """Mel spectrogram extractor for MeanVC."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_size: int = 640,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: int = 0,
        fmax: int = 8000,
        center: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        dtype_device = str(y.dtype) + '_' + str(y.device)
        fmax_dtype_device = str(self.fmax) + '_' + dtype_device
        wnsize_dtype_device = str(self.win_size) + '_' + dtype_device

        if fmax_dtype_device not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate, n_fft=self.n_fft,
                n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
            )
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
                dtype=y.dtype, device=y.device
            )
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(
                self.win_size
            ).to(dtype=y.dtype, device=y.device)

        spec = torch.stft(
            y, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.hann_window[wnsize_dtype_device],
            center=self.center,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=False
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)
        spec = _amp_to_db(spec, -115) - 20
        spec = _normalize(spec, 1, -115)
        return spec


def extract_fbanks(
    wav: np.ndarray,
    sample_rate: int = 16000,
    mel_bins: int = 80,
    frame_length: float = 25,
    frame_shift: float = 10,
) -> torch.Tensor:
    """Extract filter bank features for ASR model."""
    wav = wav * (1 << 15)
    wav = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav,
        frame_length=frame_length,
        frame_shift=frame_shift,
        snip_edges=True,
        num_mel_bins=mel_bins,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=sample_rate,
    )
    return fbanks.unsqueeze(0)


class MeanVCPipeline:
    """Streaming voice conversion pipeline using MeanVC.

    This pipeline provides real-time voice conversion with:
    - Single-step inference via mean flows (1-2 NFE)
    - Chunk-wise autoregressive processing with KV-cache
    - <100ms latency per chunk

    Sample rate: 16kHz (both input and output)
    Chunk size: 200ms (3200 samples at 16kHz)

    Usage:
        pipeline = MeanVCPipeline()
        pipeline.set_reference_audio(ref_audio, sr=16000)

        for chunk in audio_stream:
            output = pipeline.process_chunk(chunk)
            play(output)
    """

    # Model memory estimate (for factory memory tracking)
    ESTIMATED_MEMORY_GB = 4.0  # ASR + VC + Vocoder + WavLM

    def __init__(
        self,
        device: Optional[torch.device] = None,
        steps: int = 2,
        require_gpu: bool = False,  # MeanVC can run on CPU!
    ):
        """Initialize MeanVC pipeline.

        Args:
            device: Target device (defaults to CPU for real-time)
            steps: Number of inference steps (1 or 2, default 2)
            require_gpu: Whether to require GPU
        """
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        # MeanVC is optimized for CPU streaming
        self.device = device or torch.device('cpu')
        self.steps = steps

        # Sample rates
        self.sample_rate = 16000
        self.output_sample_rate = 16000

        # Streaming state
        self._reference_audio: Optional[torch.Tensor] = None
        self._spk_emb: Optional[torch.Tensor] = None
        self._prompt_mel: Optional[torch.Tensor] = None
        self._initialized = False

        # Timesteps for inference
        if self.steps == 1:
            self._timesteps = torch.tensor([1.0, 0.0], device=self.device)
        elif self.steps == 2:
            self._timesteps = torch.tensor([1.0, 0.8, 0.0], device=self.device)
        else:
            self._timesteps = torch.linspace(1.0, 0.0, self.steps + 1, device=self.device)

        # Latency tracking
        self._latency_history: Dict[str, deque] = {
            'asr': deque(maxlen=100),
            'vc': deque(maxlen=100),
            'vocoder': deque(maxlen=100),
            'total': deque(maxlen=100),
        }

        # Lazy load models
        self._asr_model = None
        self._vc_model = None
        self._vocoder = None
        self._sv_model = None
        self._mel_extractor = None

        # Streaming parameters (from MeanVC) - MUST be before _reset_streaming_state()
        self._decoding_chunk_size = 5
        self._num_decoding_left_chunks = 2
        self._subsampling = 4
        self._context = 7
        self._stride = self._subsampling * self._decoding_chunk_size  # 20
        self._required_cache_size = self._decoding_chunk_size * self._num_decoding_left_chunks
        self._chunk_samples = 160 * self._stride  # 3200 = 200ms at 16kHz
        self._vc_chunk_frames = self._decoding_chunk_size * 4  # 20 frames

        # Vocoder overlap for smooth transitions
        self._vocoder_overlap = 3
        self._upsample_factor = 160
        self._vocoder_wav_overlap = (self._vocoder_overlap - 1) * self._upsample_factor

        # Initialize streaming state (requires params above)
        self._reset_streaming_state()

        logger.info(
            f"MeanVCPipeline initialized on {self.device} "
            f"with {steps} steps, {self._chunk_samples} samples/chunk"
        )

    def _reset_streaming_state(self) -> None:
        """Reset streaming state for new session."""
        self._samples_cache_len = 720
        self._samples_cache = None
        self._att_cache = torch.zeros((0, 0, 0, 0), device=self.device)
        self._cnn_cache = torch.zeros((0, 0, 0, 0), device=self.device)
        self._asr_offset = 0
        self._encoder_output_cache = None
        self._vc_offset = 0
        self._vc_cache = None
        self._vc_kv_cache = None
        self._vocoder_cache = None
        self._last_wav = None
        self._need_extra_data = True

        # Crossfade buffers
        self._down_linspace = torch.linspace(
            1, 0, steps=self._vocoder_wav_overlap
        ).numpy()
        self._up_linspace = torch.linspace(
            0, 1, steps=self._vocoder_wav_overlap
        ).numpy()

    def _load_models(self) -> None:
        """Lazy-load all models."""
        if self._initialized:
            return

        logger.info("Loading MeanVC models...")

        ckpt_dir = MEANVC_DIR / "src" / "ckpt"
        sv_ckpt = MEANVC_DIR / "src" / "runtime" / "speaker_verification" / "ckpt" / "wavlm_large_finetune.pth"

        # Check model files exist
        required_files = [
            ckpt_dir / "fastu2++.pt",
            ckpt_dir / "meanvc_200ms.pt",
            ckpt_dir / "vocos.pt",
            sv_ckpt,
        ]
        for f in required_files:
            if not f.exists():
                raise FileNotFoundError(
                    f"Missing MeanVC model: {f}\n"
                    f"Run: cd {MEANVC_DIR} && python download_ckpt.py"
                )

        # Load JIT models (fast loading)
        logger.info("Loading ASR model...")
        self._asr_model = torch.jit.load(
            str(ckpt_dir / "fastu2++.pt")
        ).to(self.device)

        logger.info("Loading VC model...")
        self._vc_model = torch.jit.load(
            str(ckpt_dir / "meanvc_200ms.pt")
        ).to(self.device)

        logger.info("Loading vocoder...")
        self._vocoder_path = str(ckpt_dir / "vocos.pt")
        self._vocoder = torch.jit.load(self._vocoder_path).to(self.device)

        logger.info("Loading speaker verification model...")
        from runtime.speaker_verification.verification import init_model as init_sv_model
        self._sv_model = init_sv_model('wavlm_large', str(sv_ckpt))
        self._sv_model = self._sv_model.to(self.device)
        self._sv_model.eval()  # Set to evaluation mode

        logger.info("Loading mel extractor...")
        self._mel_extractor = MelSpectrogramFeatures(
            sample_rate=16000, n_fft=1024, win_size=640,
            hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True
        ).to(self.device)

        self._initialized = True
        logger.info("MeanVC models loaded successfully")

    def set_reference_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> None:
        """Set reference audio for voice cloning.

        Args:
            audio: Reference audio waveform
            sample_rate: Sample rate (will be resampled to 16kHz)
        """
        self._load_models()

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        audio = audio.float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample if needed
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(self.device)
            audio = resampler(audio)

        self._reference_audio = audio

        # Extract speaker embedding and prompt mel
        with torch.no_grad():
            self._spk_emb = self._sv_model(audio)  # [1, 256]
            self._prompt_mel = self._mel_extractor(audio)  # [1, 80, T]
            self._prompt_mel = self._prompt_mel.transpose(1, 2)  # [1, T, 80]

        # Reset streaming state for new reference
        self._reset_streaming_state()

        logger.info(
            f"Set reference audio: {audio.shape[1]/16000:.2f}s, "
            f"speaker embedding shape: {self._spk_emb.shape}"
        )

    def set_reference_from_profile_id(
        self,
        profile_id: str,
        reference_index: int = 0,
    ) -> None:
        """Load reference audio using the AdapterBridge.

        Args:
            profile_id: Voice profile UUID
            reference_index: Which reference audio to use (0 = best quality)
        """
        from .adapter_bridge import get_adapter_bridge
        import librosa

        bridge = get_adapter_bridge()
        voice_ref = bridge.get_voice_reference(profile_id)

        if not voice_ref.reference_paths:
            raise ValueError(
                f"No reference audio for profile {profile_id} ({voice_ref.profile_name})"
            )

        ref_path = voice_ref.reference_paths[
            min(reference_index, len(voice_ref.reference_paths) - 1)
        ]

        audio, sr = librosa.load(str(ref_path), sr=16000)
        self.set_reference_audio(audio, sr)

        logger.info(
            f"Loaded reference for {voice_ref.profile_name} from {ref_path.name}"
        )

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Process a single audio chunk through voice conversion.

        Args:
            audio: Input audio chunk at 16kHz, float32, mono

        Returns:
            Converted audio chunk at 16kHz, float32
        """
        if self._spk_emb is None:
            raise RuntimeError("No reference audio set. Call set_reference_audio() first.")

        self._load_models()
        total_start = time.perf_counter()

        audio = np.asarray(audio, dtype=np.float32)

        with torch.no_grad():
            # Cache concatenation
            if self._samples_cache is not None:
                audio = np.concatenate((self._samples_cache, audio))
            self._samples_cache = audio[-self._samples_cache_len:]

            # 1. ASR feature extraction
            t0 = time.perf_counter()
            fbanks = extract_fbanks(audio, frame_shift=10).float().to(self.device)

            # Reset ASR state if offset approaches position encoding limit
            # WeNet's default max_len is typically 5000 frames; reset at 4000 to be safe
            ASR_OFFSET_RESET_THRESHOLD = 4000
            if self._asr_offset > ASR_OFFSET_RESET_THRESHOLD:
                logger.debug(
                    f"Resetting ASR state: offset {self._asr_offset} > {ASR_OFFSET_RESET_THRESHOLD}"
                )
                self._asr_offset = 0
                self._att_cache = torch.zeros((0, 0, 0, 0), device=self.device)
                self._cnn_cache = torch.zeros((0, 0, 0, 0), device=self.device)

            encoder_output, self._att_cache, self._cnn_cache = \
                self._asr_model.forward_encoder_chunk(
                    fbanks, self._asr_offset, self._required_cache_size,
                    self._att_cache, self._cnn_cache
                )

            self._asr_offset += encoder_output.size(1)

            # Handle encoder output caching
            if self._encoder_output_cache is None:
                encoder_output = torch.cat(
                    [encoder_output[:, 0:1, :], encoder_output], dim=1
                )
            else:
                encoder_output = torch.cat(
                    [self._encoder_output_cache, encoder_output], dim=1
                )
            self._encoder_output_cache = encoder_output[:, -1:, :]

            # Upsample to VC frame rate
            encoder_output_up = encoder_output.transpose(1, 2)
            encoder_output_up = F.interpolate(
                encoder_output_up,
                size=self._vc_chunk_frames + 1,
                mode='linear',
                align_corners=True
            )
            encoder_output_up = encoder_output_up.transpose(1, 2)
            encoder_output_up = encoder_output_up[:, 1:, :]

            self._latency_history['asr'].append(time.perf_counter() - t0)

            # 2. VC inference (mean flow)
            t0 = time.perf_counter()
            x = torch.randn(
                1, encoder_output_up.shape[1], 80,
                device=self.device, dtype=encoder_output_up.dtype
            )

            for i in range(self.steps):
                t = self._timesteps[i]
                r = self._timesteps[i + 1]
                t_tensor = torch.full((1,), t, device=self.device)
                r_tensor = torch.full((1,), r, device=self.device)

                u, tmp_kv_cache = self._vc_model(
                    x, t_tensor, r_tensor,
                    cache=self._vc_cache,
                    cond=encoder_output_up,
                    spks=self._spk_emb,
                    prompts=self._prompt_mel,
                    offset=self._vc_offset,
                    kv_cache=self._vc_kv_cache
                )
                x = x - (t - r) * u

            self._vc_kv_cache = tmp_kv_cache
            self._vc_offset += x.shape[1]
            self._vc_cache = x

            # KV cache truncation to prevent unbounded growth
            VC_KV_CACHE_MAX_LEN = 100
            if (self._vc_offset > 40 and self._vc_kv_cache is not None and
                    self._vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN):
                for i in range(len(self._vc_kv_cache)):
                    new_k = self._vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    new_v = self._vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    self._vc_kv_cache[i] = (new_k, new_v)

            self._latency_history['vc'].append(time.perf_counter() - t0)

            # 3. Vocoder synthesis
            t0 = time.perf_counter()
            mel = x.transpose(1, 2)

            if self._vocoder_cache is not None:
                mel = torch.cat([self._vocoder_cache, mel], dim=-1)
            self._vocoder_cache = mel[:, :, -self._vocoder_overlap:]

            mel = (mel + 1) / 2
            # WORKAROUND: Run vocoder on CPU to avoid TorchScript CUDA complex dtype issues
            # The Vocos ISTFT uses complex numbers which have problems with TorchScript on CUDA
            if self.device.type == 'cuda':
                mel_cpu = mel.cpu()
                if not hasattr(self, '_vocoder_cpu'):
                    self._vocoder_cpu = torch.jit.load(self._vocoder_path).cpu()
                wav = self._vocoder_cpu.decode(mel_cpu).squeeze()
            else:
                wav = self._vocoder.decode(mel).squeeze()
            wav = wav.detach().cpu().numpy()

            self._latency_history['vocoder'].append(time.perf_counter() - t0)

            # 4. Overlap-add crossfade
            if self._last_wav is not None:
                front_wav = wav[:self._vocoder_wav_overlap]
                smooth_front = (
                    self._last_wav * self._down_linspace +
                    front_wav * self._up_linspace
                )
                new_wav = np.concatenate([
                    smooth_front,
                    wav[self._vocoder_wav_overlap:-self._vocoder_wav_overlap]
                ], axis=0)
            else:
                new_wav = wav[:-self._vocoder_wav_overlap]

            self._last_wav = wav[-self._vocoder_wav_overlap:]

        self._latency_history['total'].append(time.perf_counter() - total_start)

        return new_wav.astype(np.float32)

    def convert(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        speaker_embedding: Optional[torch.Tensor] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        pitch_shift: int = 0,
    ) -> Dict[str, Any]:
        """Convert full audio (non-streaming).

        Args:
            audio: Input audio waveform
            sample_rate: Input sample rate
            speaker_embedding: Ignored (uses reference audio instead)
            on_progress: Progress callback
            pitch_shift: Ignored (MeanVC doesn't support pitch shift)

        Returns:
            Dict with converted audio and metadata
        """
        if self._spk_emb is None:
            raise RuntimeError("No reference audio set.")

        self._load_models()
        start_time = time.time()

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Convert to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Resample to 16kHz
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Reset streaming state
        self._reset_streaming_state()

        # Reset vocoder call counter for fresh conversion
        self._vocoder_call_count = 0

        # Process in chunks
        output_chunks = []
        total_samples = len(audio)
        processed = 0

        while processed < total_samples:
            end = min(processed + self._chunk_samples, total_samples)
            chunk = audio[processed:end]

            # Pad last chunk if needed
            if len(chunk) < self._chunk_samples:
                chunk = np.pad(chunk, (0, self._chunk_samples - len(chunk)))

            output_chunk = self.process_chunk(chunk)
            output_chunks.append(output_chunk)

            processed = end
            if on_progress:
                on_progress('conversion', processed / total_samples)

        # Concatenate output
        output = np.concatenate(output_chunks)

        # Trim to match input duration
        target_len = int(len(audio) * self.output_sample_rate / 16000)
        if len(output) > target_len:
            output = output[:target_len]

        elapsed = time.time() - start_time
        output_duration = len(output) / self.output_sample_rate

        logger.info(
            f"MeanVC conversion complete: {output_duration:.2f}s audio "
            f"in {elapsed:.2f}s ({elapsed/output_duration:.2f}x realtime)"
        )

        return {
            'audio': torch.from_numpy(output),
            'sample_rate': self.output_sample_rate,
            'metadata': {
                'processing_time': elapsed,
                'steps': self.steps,
                'output_duration': output_duration,
                'device': str(self.device),
                'pipeline': 'meanvc',
            },
        }

    def get_latency_metrics(self) -> Dict[str, float]:
        """Get average latency for each component."""
        metrics = {}
        for name, history in self._latency_history.items():
            if history:
                avg_seconds = np.mean(list(history))
                metrics[f'{name}_ms'] = avg_seconds * 1000
            else:
                metrics[f'{name}_ms'] = 0.0
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        latency = self.get_latency_metrics()
        return {
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'output_sample_rate': self.output_sample_rate,
            'steps': self.steps,
            'has_reference': self._spk_emb is not None,
            'chunk_size_ms': self._chunk_samples / self.sample_rate * 1000,
            **latency,
        }

    def reset_session(self) -> None:
        """Reset streaming session (keeps reference audio)."""
        self._reset_streaming_state()
        logger.info("MeanVC streaming session reset")

    @property
    def chunk_size(self) -> int:
        """Expected chunk size in samples."""
        return self._chunk_samples
