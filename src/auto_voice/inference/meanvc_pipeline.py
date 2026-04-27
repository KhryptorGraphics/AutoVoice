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
import types
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


def _install_torchaudio_sox_effects_compat() -> None:
    """Provide the removed torchaudio.sox_effects API expected by upstream MeanVC."""
    try:
        import torchaudio.sox_effects  # type: ignore[attr-defined]  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass

    import torchaudio

    module = types.ModuleType("torchaudio.sox_effects")

    def apply_effects_tensor(waveform, sample_rate, effects=None, channels_first=True):
        return waveform, sample_rate

    def apply_effects_file(path, effects=None, normalize=True, channels_first=True, format=None):
        return torchaudio.load(
            path,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
        )

    module.apply_effects_tensor = apply_effects_tensor
    module.apply_effects_file = apply_effects_file
    sys.modules["torchaudio.sox_effects"] = module
    setattr(torchaudio, "sox_effects", module)


def _amp_to_db(x: torch.Tensor, min_level_db: float) -> torch.Tensor:
    """Convert amplitude spectrogram to decibels with floor clipping.

    Args:
        x: Amplitude spectrogram tensor
        min_level_db: Minimum dB level for floor clipping (e.g., -115)

    Returns:
        Spectrogram in dB scale with values >= min_level_db
    """
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S: torch.Tensor, max_abs_value: float, min_db: float) -> torch.Tensor:
    """Normalize spectrogram to [-max_abs_value, max_abs_value] range.

    Args:
        S: Spectrogram tensor in dB scale
        max_abs_value: Maximum absolute value for output range (typically 1.0)
        min_db: Minimum dB value used for normalization (e.g., -115)

    Returns:
        Normalized spectrogram clamped to [-max_abs_value, max_abs_value]
    """
    return torch.clamp(
        (2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value,
        -max_abs_value, max_abs_value
    )


class MelSpectrogramFeatures(nn.Module):
    """Mel spectrogram extractor for MeanVC vocoder input.

    Extracts 80-dim mel spectrograms at 16kHz with normalized dB scale.
    Uses cached mel filterbanks and Hann windows for efficiency across devices.
    """

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
        """Initialize mel spectrogram extractor.

        Args:
            sample_rate: Audio sample rate in Hz (default 16000)
            n_fft: FFT size (default 1024)
            win_size: Window size in samples (default 640 = 40ms at 16kHz)
            hop_length: Hop size in samples (default 160 = 10ms at 16kHz)
            n_mels: Number of mel bins (default 80)
            fmin: Minimum frequency in Hz (default 0)
            fmax: Maximum frequency in Hz (default 8000 = Nyquist at 16kHz)
            center: Whether to center STFT frames (default True)
        """
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
        """Extract mel spectrogram from audio waveform.

        Args:
            y: Audio waveform tensor of shape [B, T] or [T]

        Returns:
            Mel spectrogram of shape [B, n_mels, T'] where T' = T // hop_length.
            Values are normalized to [-1, 1] range after dB conversion.
        """
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
    """Extract Kaldi-style filter bank features for ASR content encoder.

    Applies 16-bit quantization before FBANK extraction to match training.

    Args:
        wav: Audio waveform as float32 numpy array, range [-1, 1]
        sample_rate: Sample rate in Hz (default 16000)
        mel_bins: Number of mel frequency bins (default 80)
        frame_length: Frame length in milliseconds (default 25)
        frame_shift: Frame shift in milliseconds (default 10)

    Returns:
        Filter bank features of shape [1, T, mel_bins] where T = num_frames
    """
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

        Configures streaming parameters for 200ms chunks (3200 samples at 16kHz)
        with autoregressive KV-cache for low-latency real-time processing.

        Args:
            device: Target device (defaults to CPU for real-time)
            steps: Number of inference steps (1 or 2, default 2)
            require_gpu: Whether to require GPU

        Raises:
            RuntimeError: If require_gpu=True but CUDA is not available
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
        """Reset streaming state for new conversion session.

        Clears all autoregressive caches and state variables used for chunk-wise
        processing. Must be called before starting a new audio stream.

        State variables:
            - _samples_cache: 720 sample audio cache (45ms at 16kHz) for context
            - _att_cache/_cnn_cache: ASR encoder attention and CNN caches
            - _asr_offset: Frame offset for ASR positional encoding
            - _encoder_output_cache: Last ASR frame for overlap
            - _vc_offset: Frame offset for VC model KV-cache
            - _vc_cache/_vc_kv_cache: VC model output and key-value caches
            - _vocoder_cache: 3-frame mel overlap cache for smooth transitions
            - _last_wav: Previous chunk's tail for crossfade (320 samples)
        """
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
        """Lazy-load all MeanVC models on first use.

        Loads JIT-compiled models for fast initialization:
        - FastU2++ ASR encoder for content features
        - MeanVC DiT model for voice conversion
        - Vocos vocoder for waveform synthesis
        - WavLM+ECAPA-TDNN for speaker embeddings

        Raises:
            FileNotFoundError: If required model checkpoint files are missing.
                Run `cd models/meanvc && python download_ckpt.py` to download.
        """
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
        _install_torchaudio_sox_effects_compat()
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
        """Set reference audio for target voice and reset streaming state.

        Extracts 256-dim speaker embedding and mel spectrogram prompt from
        the reference audio. These are used as conditioning for all subsequent
        voice conversions.

        Args:
            audio: Reference audio waveform (mono or will be converted)
            sample_rate: Input sample rate (will be resampled to 16kHz if needed)
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
        """Load reference audio from voice profile via AdapterBridge.

        Retrieves voice profile metadata and loads the specified reference
        audio file, then calls set_reference_audio() to extract embeddings.

        Args:
            profile_id: Voice profile UUID
            reference_index: Which reference audio to use (0 = best quality)

        Raises:
            ValueError: If profile has no reference audio files
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
        """Process a single audio chunk through streaming voice conversion.

        Chunk size: 3200 samples (200ms at 16kHz)
        Expected input: 16kHz mono audio, float32 in range [-1, 1]
        Output: 16kHz converted audio with <100ms latency

        Streaming state management:
        - Maintains 720-sample audio cache for ASR context
        - Stores ASR attention/CNN caches for autoregressive processing
        - Tracks frame offsets for positional encoding
        - Keeps VC model KV-cache (truncated at 100 frames)
        - Uses 3-frame mel overlap for vocoder continuity
        - Applies crossfade on 320-sample overlap between chunks

        Args:
            audio: Input audio chunk at 16kHz, float32, mono.
                   Typically 3200 samples but flexible.

        Returns:
            Converted audio chunk at 16kHz, float32.
            Length varies due to overlap-add but approximately matches input.

        Raises:
            RuntimeError: If set_reference_audio() not called before processing
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
        """Convert full audio file using chunk-wise streaming pipeline.

        Processes audio in 3200-sample chunks (200ms at 16kHz) with streaming
        state. Automatically handles resampling and mono conversion.

        Args:
            audio: Input audio waveform (mono or stereo)
            sample_rate: Input sample rate (will be resampled to 16kHz)
            speaker_embedding: Ignored (uses reference audio set via set_reference_audio)
            on_progress: Optional callback(stage: str, progress: float) for progress updates
            pitch_shift: Ignored (MeanVC doesn't support pitch shifting)

        Returns:
            Dict containing:
                - audio: Converted waveform as torch.Tensor
                - sample_rate: Output sample rate (16000)
                - metadata: Dict with processing_time, steps, output_duration, device, pipeline

        Raises:
            RuntimeError: If set_reference_audio() not called before conversion
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
        """Get average latency for each pipeline component.

        Tracks moving average over last 100 chunks for:
        - asr_ms: ASR feature extraction time
        - vc_ms: Voice conversion inference time
        - vocoder_ms: Waveform synthesis time
        - total_ms: End-to-end chunk processing time

        Returns:
            Dict mapping component names to average latency in milliseconds
        """
        metrics = {}
        for name, history in self._latency_history.items():
            if history:
                avg_seconds = np.mean(list(history))
                metrics[f'{name}_ms'] = avg_seconds * 1000
            else:
                metrics[f'{name}_ms'] = 0.0
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics and configuration.

        Returns:
            Dict containing:
                - device: Device string (cpu or cuda:N)
                - sample_rate: Input sample rate (16000)
                - output_sample_rate: Output sample rate (16000)
                - steps: Number of mean flow inference steps (1 or 2)
                - has_reference: Whether reference audio is loaded
                - chunk_size_ms: Chunk duration in milliseconds (200.0)
                - asr_ms, vc_ms, vocoder_ms, total_ms: Average latencies
        """
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
        """Reset streaming session state while preserving reference audio.

        Clears all autoregressive caches and offsets to start a new audio stream
        with the same target voice. Use when starting a new input audio file
        without changing the voice profile.

        Preserves:
            - Speaker embedding (_spk_emb)
            - Prompt mel spectrogram (_prompt_mel)
            - Reference audio (_reference_audio)

        Resets:
            - All ASR and VC caches
            - Frame offsets
            - Vocoder overlap buffers
        """
        self._reset_streaming_state()
        logger.info("MeanVC streaming session reset")

    @property
    def chunk_size(self) -> int:
        """Expected chunk size for optimal streaming performance.

        Returns:
            Chunk size in samples (3200 = 200ms at 16kHz). This is the
            optimal size for the ASR and VC models' chunk configuration.
        """
        return self._chunk_samples
