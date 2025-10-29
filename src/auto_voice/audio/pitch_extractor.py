"""Singing pitch extraction using torchcrepe with GPU acceleration"""

from __future__ import annotations
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..utils.gpu_manager import GPUManager
    import yaml as _yaml
    import torchcrepe as _torchcrepe
    import librosa as _librosa
else:
    # Runtime imports with fallbacks
    try:
        import yaml
    except ImportError:
        yaml = None  # type: ignore

    try:
        import torchcrepe
    except ImportError:
        torchcrepe = None  # type: ignore

    try:
        import librosa
    except ImportError:
        librosa = None  # type: ignore

# NumPy and PyTorch availability checks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

# Feature availability flags
TORCHCREPE_AVAILABLE = torchcrepe is not None
LIBROSA_AVAILABLE = librosa is not None

from .processor import AudioProcessor

logger = logging.getLogger(__name__)


class PitchExtractionError(Exception):
    """Base exception for pitch extraction errors"""
    pass


class ModelLoadError(PitchExtractionError):
    """Exception raised when model loading fails"""
    pass


class SingingPitchExtractor:
    """High-accuracy pitch extraction optimized for singing voice with GPU acceleration

    This class uses torchcrepe (PyTorch port of CREPE) for state-of-the-art pitch detection
    with comprehensive vibrato detection and analysis capabilities.

    Features:
        - GPU-accelerated pitch extraction using torchcrepe
        - Vibrato detection with rate and depth measurement
        - Post-processing with median/mean filtering
        - Real-time mode with CUDA kernel fallback
        - Batch processing for multiple audio files
        - Comprehensive statistics and analysis

    Example:
        >>> extractor = SingingPitchExtractor(device='cuda', gpu_manager=gpu_manager)
        >>> f0_data = extractor.extract_f0_contour('singing.wav')
        >>> print(f"Mean F0: {f0_data['f0'].mean():.1f} Hz")
        >>> print(f"Vibrato rate: {f0_data['vibrato']['rate_hz']:.1f} Hz")

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        device (str): Device for processing ('cuda', 'cpu', etc.)
        gpu_manager: Optional GPUManager for GPU acceleration
        audio_processor (AudioProcessor): Audio I/O handler
        model (str): CREPE model name ('tiny' or 'full')
        lock (threading.RLock): Thread safety lock
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        gpu_manager: Optional["GPUManager"] = None
    ):
        """Initialize SingingPitchExtractor

        Args:
            config: Optional configuration dictionary
            device: Optional device string ('cuda', 'cpu', 'cuda:0', etc.)
            gpu_manager: Optional GPUManager instance for GPU acceleration

        Raises:
            ModelLoadError: If torchcrepe is not available
        """
        if not TORCHCREPE_AVAILABLE:
            raise ModelLoadError("torchcrepe is not available. Install with: pip install torchcrepe")

        if not TORCH_AVAILABLE:
            raise ModelLoadError("PyTorch is not available. Please install torch.")

        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config)

        # Set device
        if device is not None:
            self.device = device
        elif gpu_manager is not None and hasattr(gpu_manager, 'device'):
            self.device = str(gpu_manager.device) if hasattr(gpu_manager.device, '__str__') else 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.gpu_manager = gpu_manager

        # Initialize AudioProcessor for audio I/O
        self.audio_processor = AudioProcessor()

        # Configuration parameters
        self.model = self.config.get('model', 'full')
        self.fmin = self.config.get('fmin', 80.0)
        self.fmax = self.config.get('fmax', 1000.0)
        self.hop_length_ms = self.config.get('hop_length_ms', 10.0)
        self.batch_size = self.config.get('batch_size', 2048)
        self.decoder = self.config.get('decoder', 'viterbi')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.21)
        # COMMENT 3 FIX: Add dedicated cuda_cmnd_threshold instead of reusing confidence_threshold
        self.cuda_cmnd_threshold = self.config.get('cuda_cmnd_threshold', 0.15)
        self.median_filter_width = self.config.get('median_filter_width', 3)
        self.mean_filter_width = self.config.get('mean_filter_width', 3)

        # Vibrato detection parameters
        self.vibrato_rate_range = self.config.get('vibrato_rate_range', [4.0, 8.0])
        self.vibrato_min_depth_cents = self.config.get('vibrato_min_depth_cents', 20.0)
        self.vibrato_min_duration_ms = self.config.get('vibrato_min_duration_ms', 250.0)
        self.vibrato_regularity_threshold = self.config.get('vibrato_regularity_threshold', 0.5)

        # Pitch correction parameters
        self.pitch_correction_tolerance_cents = self.config.get('pitch_correction_tolerance_cents', 50.0)
        self.pitch_correction_reference_scale = self.config.get('pitch_correction_reference_scale', 'C')

        # Real-time streaming parameters
        self.realtime_overlap_frames = self.config.get('realtime_overlap_frames', 5)
        self.realtime_buffer_size = self.config.get('realtime_buffer_size', 4096)
        self.realtime_smoothing_window = self.config.get('realtime_smoothing_window', 5)

        # GPU optimization
        self.gpu_acceleration = self.config.get('gpu_acceleration', True)
        self.mixed_precision = self.config.get('mixed_precision', True)
        self.use_cuda_kernel_fallback = self.config.get('use_cuda_kernel_fallback', True)
        # COMMENT 5 FIX: Add use_gpu_vibrato_analysis flag
        self.use_gpu_vibrato_analysis = self.config.get('use_gpu_vibrato_analysis', False)

        self.logger.info(
            f"SingingPitchExtractor initialized: model={self.model}, device={self.device}, "
            f"fmin={self.fmin}Hz, fmax={self.fmax}Hz"
        )

    def _call_torchcrepe_predict(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        hop_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call torchcrepe.predict with decoder compatibility handling

        Args:
            audio: Audio tensor
            sample_rate: Sample rate
            hop_length: Hop length in samples

        Returns:
            Tuple of (pitch, periodicity) tensors
        """
        # Map decoder option to torchcrepe decoders
        decoder_fn = None
        if self.decoder == 'viterbi':
            decoder_fn = torchcrepe.decode.viterbi
        elif self.decoder == 'argmax':
            if hasattr(torchcrepe.decode, 'argmax'):
                decoder_fn = torchcrepe.decode.argmax
            else:
                self.logger.warning("torchcrepe.decode.argmax not available, using viterbi")
                decoder_fn = torchcrepe.decode.viterbi
        elif self.decoder == 'weighted_argmax':
            if hasattr(torchcrepe.decode, 'weighted_argmax'):
                decoder_fn = torchcrepe.decode.weighted_argmax
            else:
                self.logger.warning("torchcrepe.decode.weighted_argmax not available, using viterbi")
                decoder_fn = torchcrepe.decode.viterbi

        # Try calling with decoder parameter, fall back if not supported
        try:
            pitch, periodicity = torchcrepe.predict(
                audio,
                sample_rate,
                hop_length=hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                model=self.model,
                batch_size=self.batch_size,
                device=self.device,
                return_periodicity=True,
                decoder=decoder_fn
            )
        except TypeError as e:
            # Decoder parameter not supported in this torchcrepe version
            if 'decoder' in str(e):
                self.logger.warning("torchcrepe.predict does not support 'decoder' parameter, calling without it")
                pitch, periodicity = torchcrepe.predict(
                    audio,
                    sample_rate,
                    hop_length=hop_length,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    model=self.model,
                    batch_size=self.batch_size,
                    device=self.device,
                    return_periodicity=True
                )
            else:
                raise

        return pitch, periodicity

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from multiple sources

        Priority: constructor config > YAML file > environment variables > defaults

        Args:
            config: Configuration dictionary from constructor

        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        final_config = {
            'model': 'full',
            'fmin': 80.0,
            'fmax': 1000.0,
            'hop_length_ms': 10.0,
            'batch_size': 2048,
            'decoder': 'viterbi',
            'confidence_threshold': 0.21,
            # COMMENT 3 FIX: Add cuda_cmnd_threshold default
            'cuda_cmnd_threshold': 0.15,
            'median_filter_width': 3,
            'mean_filter_width': 3,
            'vibrato_rate_range': [4.0, 8.0],
            'vibrato_min_depth_cents': 20.0,
            'vibrato_min_duration_ms': 250.0,
            'vibrato_regularity_threshold': 0.5,
            'pitch_correction_tolerance_cents': 50.0,
            'pitch_correction_reference_scale': 'C',
            'realtime_overlap_frames': 5,
            'realtime_buffer_size': 4096,
            'realtime_smoothing_window': 5,
            'gpu_acceleration': True,
            'mixed_precision': True,
            'use_cuda_kernel_fallback': True,
            # COMMENT 5 FIX: Add use_gpu_vibrato_analysis default
            'use_gpu_vibrato_analysis': False
        }

        # Load from YAML if available
        config_path = Path('config/audio_config.yaml')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'singing_pitch' in yaml_config:
                        final_config.update(yaml_config['singing_pitch'])
            except Exception as e:
                self.logger.warning(f"Failed to load YAML config: {e}")

        # Override with environment variables
        env_mapping = {
            'AUTOVOICE_PITCH_MODEL': ('model', str),
            'AUTOVOICE_PITCH_FMIN': ('fmin', float),
            'AUTOVOICE_PITCH_FMAX': ('fmax', float),
            'AUTOVOICE_PITCH_HOP_LENGTH': ('hop_length_ms', float),
            'AUTOVOICE_PITCH_BATCH_SIZE': ('batch_size', int),
            'AUTOVOICE_PITCH_DECODER': ('decoder', str)
        }
        for env_var, (config_key, value_type) in env_mapping.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    if value_type == int:
                        value = int(value)
                    elif value_type == float:
                        value = float(value)
                    # str type needs no conversion
                    final_config[config_key] = value
                except ValueError:
                    self.logger.warning(f"Invalid value for {env_var}: {os.environ[env_var]}")

        # Override with constructor config (highest priority)
        if config:
            final_config.update(config)

        return final_config

    def extract_f0_contour(
        self,
        audio: Union["torch.Tensor", "np.ndarray", str],
        sample_rate: Optional[int] = None,
        return_confidence: bool = True,
        return_times: bool = True
    ) -> Dict[str, Any]:
        """Extract F0 contour from audio using torchcrepe

        Args:
            audio: Audio tensor, numpy array, or file path
            sample_rate: Sample rate (required if audio is tensor/array)
            return_confidence: Include confidence scores in output
            return_times: Include time stamps in output

        Returns:
            Dictionary containing:
                - 'f0': Pitch contour in Hz
                - 'voiced': Boolean mask for voiced frames
                - 'confidence': Periodicity/confidence scores (if return_confidence=True)
                - 'times': Time stamps in seconds (if return_times=True)
                - 'vibrato': Vibrato parameters dict
                - 'sample_rate': Sample rate used
                - 'hop_length': Hop length used

        Raises:
            ValueError: If audio is empty or too short
            PitchExtractionError: If extraction fails
        """
        with self.lock:
            # Load audio if file path (before validation)
            if isinstance(audio, str):
                audio, original_sr = self.audio_processor.load_audio(audio, return_sr=True)
                # Use the processor's target sample rate (resampled SR), not original
                sample_rate = self.audio_processor.sample_rate
                # Keep as tensor, don't convert to numpy here

            # Validate sample_rate before try block so ValueError surfaces unchanged
            if sample_rate is None:
                raise ValueError("sample_rate must be provided for tensor/array input")

            # Validate audio is not empty or too short before try block
            if isinstance(audio, np.ndarray):
                if audio.size == 0:
                    raise ValueError("Audio array is empty")
                if len(audio) < 100:
                    raise ValueError(f"Audio is too short ({len(audio)} samples), need at least 100 samples")
            elif isinstance(audio, torch.Tensor):
                if audio.numel() == 0:
                    raise ValueError("Audio tensor is empty")
                if audio.numel() < 100:
                    raise ValueError(f"Audio is too short ({audio.numel()} samples), need at least 100 samples")

            try:
                # Convert to torch tensor if numpy
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()

                # Ensure mono
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)

                # Normalize to [-1, 1]
                if audio.abs().max() > 1.0:
                    audio = audio / audio.abs().max()

                # Move to device
                original_device = audio.device
                if self.gpu_acceleration and self.device != 'cpu':
                    audio = audio.to(self.device)

                # Compute hop length in samples
                hop_length = int(self.hop_length_ms * sample_rate / 1000.0)

                # Use GPU manager context if available
                context = self.gpu_manager.device_context() if self.gpu_manager else self._null_context()

                with context:
                    # Ensure audio is float32 for torchcrepe
                    if audio.dtype != torch.float32:
                        audio = audio.float()

                    # Call torchcrepe without autocast to maintain pitch accuracy
                    with torch.no_grad():
                        pitch, periodicity = self._call_torchcrepe_predict(audio, sample_rate, hop_length)

                # Squeeze batch dimension from torchcrepe output (batch, time) -> (time,)
                if pitch.dim() > 1:
                    pitch = pitch.squeeze(0)
                if periodicity.dim() > 1:
                    periodicity = periodicity.squeeze(0)

                # Post-processing
                pitch, periodicity = self._post_process(pitch, periodicity, hop_length, sample_rate)

                # Compute voiced/unvoiced mask (ensure 1D)
                voiced = periodicity > self.confidence_threshold
                if voiced.dim() > 1:
                    voiced = voiced.squeeze(0)

                # COMMENT 5 FIX: Detect vibrato with GPU kernel if enabled
                vibrato_data = self._detect_vibrato_with_gpu_fallback(
                    pitch, voiced, sample_rate, hop_length
                )

                # Ensure all tensors are 1D and convert to numpy
                pitch_np = pitch.squeeze().cpu().numpy() if isinstance(pitch, torch.Tensor) else pitch
                voiced_np = voiced.squeeze().cpu().numpy() if isinstance(voiced, torch.Tensor) else voiced
                confidence_np = periodicity.squeeze().cpu().numpy() if isinstance(periodicity, torch.Tensor) else periodicity

                # Build result dictionary
                result = {
                    'f0': pitch_np,
                    'voiced': voiced_np,
                    'vibrato': vibrato_data,
                    'sample_rate': sample_rate,
                    'hop_length': hop_length
                }

                if return_confidence:
                    result['confidence'] = confidence_np

                if return_times:
                    n_frames = len(pitch_np)
                    result['times'] = np.arange(n_frames) * hop_length / sample_rate

                return result

            except ValueError:
                # Let ValueError pass through unchanged for input validation errors
                raise
            except Exception as e:
                self.logger.error(f"F0 extraction failed: {e}")
                raise PitchExtractionError(f"Failed to extract F0 contour: {e}") from e

    def _post_process(
        self,
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        hop_length: int,
        sample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply post-processing to pitch and periodicity

        Args:
            pitch: Raw pitch contour
            periodicity: Raw periodicity scores
            hop_length: Hop length in samples
            sample_rate: Sample rate

        Returns:
            Tuple of (processed_pitch, processed_periodicity)
        """
        # Median filter on periodicity to reduce spurious voiced frames
        if self.median_filter_width > 1:
            periodicity = self._median_filter_1d(periodicity, self.median_filter_width)

        # Threshold pitch using periodicity
        pitch = torch.where(periodicity > self.confidence_threshold, pitch, torch.zeros_like(pitch))

        # Mean filter on pitch for smoothing
        if self.mean_filter_width > 1:
            pitch = self._mean_filter_1d(pitch, self.mean_filter_width)

        return pitch, periodicity

    def _median_filter_1d(self, x: torch.Tensor, width: int) -> torch.Tensor:
        """Apply 1D median filter"""
        if width < 2:
            return x

        # Pad
        pad_width = width // 2
        x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (pad_width, pad_width), mode='replicate')

        # Use unfold to create sliding windows
        windows = x_padded.unfold(2, width, 1).squeeze(0).squeeze(0)

        # Compute median
        result = windows.median(dim=1)[0]

        return result

    def _mean_filter_1d(self, x: torch.Tensor, width: int) -> torch.Tensor:
        """Apply 1D mean filter"""
        if width < 2:
            return x

        # Pad
        pad_width = width // 2
        x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (pad_width, pad_width), mode='replicate')

        # Use unfold to create sliding windows
        windows = x_padded.unfold(2, width, 1).squeeze(0).squeeze(0)

        # Compute mean
        result = windows.mean(dim=1)

        return result

    def _detect_vibrato_with_gpu_fallback(
        self,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        sample_rate: int,
        hop_length: int
    ) -> Dict[str, Any]:
        """Detect vibrato with GPU kernel fallback to CPU implementation

        COMMENT 5 FIX: Wrapper that tries GPU vibrato analysis kernel first,
        falls back to CPU _detect_vibrato on failure or if disabled.

        Args:
            f0: Pitch contour in Hz
            voiced: Voiced/unvoiced mask
            sample_rate: Sample rate
            hop_length: Hop length in samples

        Returns:
            Dictionary with vibrato parameters
        """
        # COMMENT 5 FIX: Try GPU vibrato analysis if enabled
        if self.use_gpu_vibrato_analysis and self.device != 'cpu':
            try:
                _ck = self._load_cuda_extension()
                if _ck is not None and hasattr(_ck, 'launch_vibrato_analysis'):
                    # Ensure tensors are on GPU
                    if isinstance(f0, torch.Tensor):
                        f0_gpu = f0.to(self.device) if f0.device.type != 'cuda' else f0
                    else:
                        f0_gpu = torch.from_numpy(f0).to(self.device)

                    # Prepare output tensors
                    n_frames = len(f0_gpu)
                    vibrato_rate = torch.zeros(n_frames, device=self.device)
                    vibrato_depth = torch.zeros(n_frames, device=self.device)

                    # Call GPU kernel
                    _ck.launch_vibrato_analysis(
                        f0_gpu, vibrato_rate, vibrato_depth,
                        hop_length, float(sample_rate)
                    )

                    # Process results
                    rate_vals = vibrato_rate.cpu().numpy()
                    depth_vals = vibrato_depth.cpu().numpy()

                    # Filter valid vibrato detections
                    valid_mask = (rate_vals > 0) & (depth_vals >= self.vibrato_min_depth_cents)
                    if np.any(valid_mask):
                        mean_rate = float(np.mean(rate_vals[valid_mask]))
                        mean_depth = float(np.mean(depth_vals[valid_mask]))

                        return {
                            'has_vibrato': True,
                            'rate_hz': mean_rate,
                            'depth_cents': mean_depth,
                            'segments': []  # GPU kernel doesn't provide segments
                        }
                    else:
                        return {
                            'has_vibrato': False,
                            'rate_hz': 0.0,
                            'depth_cents': 0.0,
                            'segments': []
                        }

            except Exception as e:
                self.logger.debug(f"GPU vibrato analysis failed, falling back to CPU: {e}")

        # COMMENT 5 FIX: Fallback to CPU implementation
        return self._detect_vibrato(f0, voiced, sample_rate, hop_length)

    def _detect_vibrato(
        self,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        sample_rate: int,
        hop_length: int
    ) -> Dict[str, Any]:
        """Detect vibrato from F0 contour

        Args:
            f0: Pitch contour in Hz
            voiced: Voiced/unvoiced mask
            sample_rate: Sample rate
            hop_length: Hop length in samples

        Returns:
            Dictionary with vibrato parameters
        """
        try:
            # Convert to numpy for easier processing
            if isinstance(f0, torch.Tensor):
                f0 = f0.cpu().numpy()
            if isinstance(voiced, torch.Tensor):
                voiced = voiced.cpu().numpy()

            # Convert to cents (reference = 440 Hz)
            f0_cents = np.zeros_like(f0)
            mask = (f0 > 0) & voiced
            f0_cents[mask] = 1200.0 * np.log2(f0[mask] / 440.0)

            # Set unvoiced to NaN
            f0_cents[~mask] = np.nan

            # Detrend using moving average
            window_size = int(0.3 * sample_rate / hop_length)  # 300ms window
            if window_size < 3:
                window_size = 3

            trend = self._moving_average(f0_cents, window_size)
            detrended = f0_cents - trend

            # Apply bandpass filter in vibrato range (4-8 Hz) via FFT
            frame_rate = sample_rate / hop_length
            detrended_filtered = self._bandpass_filter_fft(detrended, frame_rate,
                                                          self.vibrato_rate_range[0],
                                                          self.vibrato_rate_range[1])

            # Find voiced segments longer than minimum duration
            min_frames = int(self.vibrato_min_duration_ms * sample_rate / (1000.0 * hop_length))
            # Merge adjacent segments separated by short gaps (up to 3 frames)
            raw_segments = self._find_voiced_segments(voiced, min_frames // 2)  # Use lower threshold initially
            segments = self._merge_close_segments(raw_segments, max_gap=3)

            vibrato_segments = []
            total_rate = 0.0
            total_depth = 0.0
            vibrato_count = 0

            for start, end in segments:
                # Reduce minimum frames requirement to 70% of original for flexibility
                if end - start < int(min_frames * 0.7):
                    continue

                # Use filtered signal for better depth estimation
                seg_detrended = detrended_filtered[start:end]
                seg_valid = ~np.isnan(seg_detrended)

                # Reduce valid points requirement to fraction of segment length
                min_valid_points = max(3, int((end - start) * 0.7))
                if seg_valid.sum() < min_valid_points:
                    continue

                # Compute autocorrelation
                valid_data = seg_detrended[seg_valid]
                if len(valid_data) < 10:
                    continue

                acf = np.correlate(valid_data, valid_data, mode='full')
                acf = acf[len(acf)//2:]
                acf = acf / acf[0]  # Normalize

                # Find peak in 4-8 Hz range (vibrato range)
                frame_rate = sample_rate / hop_length
                lag_min = int(frame_rate / self.vibrato_rate_range[1])
                lag_max = int(frame_rate / self.vibrato_rate_range[0])

                if lag_max >= len(acf):
                    continue

                peak_lag = np.argmax(acf[lag_min:lag_max]) + lag_min
                peak_val = acf[peak_lag]

                if peak_val < 0.3:  # Weak correlation
                    continue

                # Compute vibrato rate
                vibrato_rate = frame_rate / peak_lag

                # Compute vibrato depth using Hilbert transform envelope
                try:
                    # Use Hilbert transform for better envelope estimation
                    from scipy.signal import hilbert
                    try:
                        analytic_signal = hilbert(valid_data)
                        envelope = np.abs(analytic_signal)
                        depth_cents = 2.0 * np.median(envelope)
                    except Exception:
                        # Fallback to simple envelope
                        envelope = np.abs(valid_data)
                        depth_cents = 2.0 * np.median(envelope)

                    if depth_cents >= self.vibrato_min_depth_cents:
                        # Vibrato detected
                        time_start = start * hop_length / sample_rate
                        time_end = end * hop_length / sample_rate
                        vibrato_segments.append((time_start, time_end, vibrato_rate, depth_cents))

                        total_rate += vibrato_rate
                        total_depth += depth_cents
                        vibrato_count += 1
                except Exception:
                    continue

            # Build result
            has_vibrato = vibrato_count > 0
            mean_rate = total_rate / vibrato_count if vibrato_count > 0 else 0.0
            mean_depth = total_depth / vibrato_count if vibrato_count > 0 else 0.0

            return {
                'has_vibrato': has_vibrato,
                'rate_hz': float(mean_rate),
                'depth_cents': float(mean_depth),
                'segments': vibrato_segments
            }

        except Exception as e:
            self.logger.warning(f"Vibrato detection failed: {e}")
            return {
                'has_vibrato': False,
                'rate_hz': 0.0,
                'depth_cents': 0.0,
                'segments': []
            }

    def _moving_average(self, x: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average with NaN handling using efficient convolution

        COMMENT 1 FIX: Use NumPy-only implementation without SciPy dependency
        """
        if window < 2:
            return x.copy()

        # Create mask for valid (non-NaN) values
        valid_mask = ~np.isnan(x)

        # Replace NaNs with zeros for convolution
        x_filled = np.where(valid_mask, x, 0.0)

        # Create uniform kernel
        kernel = np.ones(window) / window

        # COMMENT 1 FIX: Use np.convolve instead of scipy.signal.convolve
        # Convolve both the data and the mask
        sum_values = np.convolve(x_filled, kernel, mode='same')
        sum_weights = np.convolve(valid_mask.astype(float), kernel, mode='same')

        # Avoid division by zero
        result = np.where(sum_weights > 0, sum_values / sum_weights, np.nan)

        return result

    def _bandpass_filter_fft(self, x: np.ndarray, fs: float, lowcut: float, highcut: float) -> np.ndarray:
        """Apply bandpass filter using FFT (handles NaN)"""
        # Handle NaN by interpolation
        valid_mask = ~np.isnan(x)
        if valid_mask.sum() < 3:
            return x

        # Simple linear interpolation for NaN values
        x_interp = x.copy()
        if not valid_mask.all():
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(~valid_mask)[0]
            x_interp[invalid_indices] = np.interp(invalid_indices, valid_indices, x[valid_indices])

        # FFT-based bandpass
        n = len(x_interp)
        fft_vals = np.fft.fft(x_interp)
        freqs = np.fft.fftfreq(n, 1.0 / fs)

        # Create bandpass mask
        mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
        fft_vals[~mask] = 0

        # Inverse FFT
        filtered = np.fft.ifft(fft_vals).real

        # Restore NaN where original had NaN
        filtered[~valid_mask] = np.nan

        return filtered

    def _find_voiced_segments(self, voiced: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
        """Find continuous voiced segments"""
        segments = []
        in_segment = False
        start = 0

        for i in range(len(voiced)):
            if voiced[i] and not in_segment:
                start = i
                in_segment = True
            elif not voiced[i] and in_segment:
                if i - start >= min_frames:
                    segments.append((start, i))
                in_segment = False

        if in_segment and len(voiced) - start >= min_frames:
            segments.append((start, len(voiced)))

        return segments

    def _merge_close_segments(self, segments: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
        """Merge segments separated by small gaps"""
        if not segments:
            return []

        merged = []
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            if start - current_end <= max_gap:
                # Merge with current segment
                current_end = end
            else:
                # Save current segment and start new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Add final segment
        merged.append((current_start, current_end))
        return merged

    def classify_vibrato(self, f0_data: Dict[str, Any]) -> Dict[str, Union[bool, float]]:
        """Classify vibrato characteristics from F0 data using frequency modulation analysis

        This method performs detailed vibrato classification by analyzing the frequency
        modulation patterns in the F0 contour. It detects vibrato presence, measures
        its rate (Hz), extent (cents), and regularity.

        Args:
            f0_data: F0 data dictionary from extract_f0_contour() containing:
                - 'f0': Pitch contour (Hz)
                - 'voiced': Voiced/unvoiced mask
                - 'sample_rate': Sample rate
                - 'hop_length': Hop length in samples

        Returns:
            Dictionary containing:
                - 'vibrato_detected': bool, whether vibrato is present
                - 'rate_hz': float, average vibrato rate in Hz (0.0 if not detected)
                - 'extent_cents': float, vibrato depth/extent in cents (0.0 if not detected)
                - 'regularity_score': float, vibrato regularity 0.0-1.0 (0.0 if not detected)
                - 'segments': list of (start_time, end_time, rate, depth) tuples

        Example:
            >>> f0_data = extractor.extract_f0_contour('singing.wav')
            >>> vibrato = extractor.classify_vibrato(f0_data)
            >>> if vibrato['vibrato_detected']:
            ...     print(f"Vibrato: {vibrato['rate_hz']:.2f} Hz, {vibrato['extent_cents']:.1f} cents")
            ...     print(f"Regularity: {vibrato['regularity_score']:.2f}")

        Note:
            - Handles silence and noise by analyzing only voiced segments
            - Rapid pitch changes are filtered using the configured vibrato rate range
            - Returns zeros for all metrics if no vibrato is detected
            - Thread-safe for concurrent calls
        """
        with self.lock:
            try:
                # Extract required data
                f0 = f0_data['f0']
                voiced = f0_data['voiced']
                sample_rate = f0_data['sample_rate']
                hop_length = f0_data['hop_length']

                # Handle empty or all-unvoiced cases
                if len(f0) == 0 or not np.any(voiced):
                    return {
                        'vibrato_detected': False,
                        'rate_hz': 0.0,
                        'extent_cents': 0.0,
                        'regularity_score': 0.0,
                        'segments': []
                    }

                # Convert to cents (reference = 440 Hz)
                f0_cents = np.zeros_like(f0)
                mask = (f0 > 0) & voiced
                if not np.any(mask):
                    return {
                        'vibrato_detected': False,
                        'rate_hz': 0.0,
                        'extent_cents': 0.0,
                        'regularity_score': 0.0,
                        'segments': []
                    }

                f0_cents[mask] = 1200.0 * np.log2(f0[mask] / 440.0)
                f0_cents[~mask] = np.nan

                # Detrend using moving average
                window_size = max(3, int(0.3 * sample_rate / hop_length))  # 300ms window
                trend = self._moving_average(f0_cents, window_size)
                detrended = f0_cents - trend

                # Apply bandpass filter in vibrato range
                frame_rate = sample_rate / hop_length
                detrended_filtered = self._bandpass_filter_fft(
                    detrended, frame_rate,
                    self.vibrato_rate_range[0],
                    self.vibrato_rate_range[1]
                )

                # Find voiced segments
                min_frames = int(self.vibrato_min_duration_ms * sample_rate / (1000.0 * hop_length))
                raw_segments = self._find_voiced_segments(voiced, min_frames // 2)
                segments = self._merge_close_segments(raw_segments, max_gap=3)

                vibrato_segments = []
                total_rate = 0.0
                total_depth = 0.0
                total_regularity = 0.0
                vibrato_count = 0

                for start, end in segments:
                    if end - start < int(min_frames * 0.7):
                        continue

                    seg_detrended = detrended_filtered[start:end]
                    seg_valid = ~np.isnan(seg_detrended)

                    min_valid_points = max(3, int((end - start) * 0.7))
                    if seg_valid.sum() < min_valid_points:
                        continue

                    # Compute autocorrelation for rate detection
                    valid_data = seg_detrended[seg_valid]
                    if len(valid_data) < 10:
                        continue

                    acf = np.correlate(valid_data, valid_data, mode='full')
                    acf = acf[len(acf)//2:]
                    if acf[0] <= 0:
                        continue
                    acf = acf / acf[0]  # Normalize

                    # Find peak in vibrato range
                    lag_min = max(1, int(frame_rate / self.vibrato_rate_range[1]))
                    lag_max = int(frame_rate / self.vibrato_rate_range[0])

                    if lag_max >= len(acf) or lag_min >= lag_max:
                        continue

                    peak_lag = np.argmax(acf[lag_min:lag_max]) + lag_min
                    peak_val = acf[peak_lag]

                    if peak_val < 0.3:  # Weak correlation
                        continue

                    # Compute vibrato rate
                    vibrato_rate = frame_rate / peak_lag

                    # Compute vibrato extent using Hilbert transform
                    try:
                        from scipy.signal import hilbert
                        analytic_signal = hilbert(valid_data)
                        envelope = np.abs(analytic_signal)
                        depth_cents = 2.0 * np.median(envelope)
                    except Exception:
                        # Fallback to simple envelope
                        envelope = np.abs(valid_data)
                        depth_cents = 2.0 * np.median(envelope)

                    # Compute regularity score using autocorrelation decay
                    # More regular vibrato maintains higher correlation at multiples of the period
                    regularity_score = peak_val  # Use peak correlation as regularity measure
                    if lag_max + peak_lag < len(acf):
                        # Check second peak (regularity indicator)
                        second_peak_val = acf[peak_lag * 2] if peak_lag * 2 < len(acf) else 0.0
                        regularity_score = (peak_val + second_peak_val) / 2.0

                    regularity_score = np.clip(regularity_score, 0.0, 1.0)

                    if depth_cents >= self.vibrato_min_depth_cents:
                        # Vibrato detected
                        time_start = start * hop_length / sample_rate
                        time_end = end * hop_length / sample_rate
                        vibrato_segments.append((time_start, time_end, vibrato_rate, depth_cents))

                        total_rate += vibrato_rate
                        total_depth += depth_cents
                        total_regularity += regularity_score
                        vibrato_count += 1

                # Build result
                vibrato_detected = vibrato_count > 0
                mean_rate = total_rate / vibrato_count if vibrato_count > 0 else 0.0
                mean_depth = total_depth / vibrato_count if vibrato_count > 0 else 0.0
                mean_regularity = total_regularity / vibrato_count if vibrato_count > 0 else 0.0

                return {
                    'vibrato_detected': vibrato_detected,
                    'rate_hz': float(mean_rate),
                    'extent_cents': float(mean_depth),
                    'regularity_score': float(mean_regularity),
                    'segments': vibrato_segments
                }

            except Exception as e:
                self.logger.warning(f"Vibrato classification failed: {e}")
                return {
                    'vibrato_detected': False,
                    'rate_hz': 0.0,
                    'extent_cents': 0.0,
                    'regularity_score': 0.0,
                    'segments': []
                }

    def suggest_pitch_corrections(
        self,
        f0_data: Dict[str, Any],
        reference_scale: str = 'C',
        tolerance_cents: float = 50.0
    ) -> List[Dict[str, Union[float, str]]]:
        """Suggest pitch corrections to align singing with reference musical scale

        This method analyzes the F0 contour and identifies notes that deviate from
        the reference scale by more than the tolerance. It returns correction
        suggestions with timestamps, detected notes, and target notes.

        Args:
            f0_data: F0 data dictionary from extract_f0_contour() containing:
                - 'f0': Pitch contour (Hz)
                - 'voiced': Voiced/unvoiced mask
                - 'times': Time stamps in seconds
            reference_scale: Reference scale name ('C', 'D', 'E', etc.) for correction
            tolerance_cents: Tolerance in cents (default: 50.0). Notes within
                tolerance are not corrected.

        Returns:
            List of correction dictionaries, each containing:
                - 'timestamp': float, time in seconds
                - 'detected_f0_hz': float, detected frequency
                - 'detected_note': str, detected note name (e.g., 'C4', 'F#5')
                - 'target_note': str, target note name in reference scale
                - 'target_f0_hz': float, target frequency
                - 'correction_cents': float, correction amount in cents (negative = flatten)

        Example:
            >>> f0_data = extractor.extract_f0_contour('singing.wav')
            >>> corrections = extractor.suggest_pitch_corrections(f0_data, 'C', 50.0)
            >>> for corr in corrections[:5]:
            ...     print(f"{corr['timestamp']:.2f}s: {corr['detected_note']} -> {corr['target_note']} "
            ...           f"({corr['correction_cents']:+.1f} cents)")

        Note:
            - Only analyzes voiced frames to avoid spurious corrections
            - Handles silence by skipping unvoiced regions
            - Handles noise by filtering based on confidence threshold
            - Rapid pitch changes are processed individually for each frame
            - Thread-safe for concurrent calls
        """
        with self.lock:
            try:
                # Extract required data
                f0 = f0_data['f0']
                voiced = f0_data['voiced']
                times = f0_data.get('times', np.arange(len(f0)) * f0_data['hop_length'] / f0_data['sample_rate'])

                # Validate inputs
                if len(f0) == 0 or not np.any(voiced):
                    return []

                # Use provided tolerance or default from config
                if tolerance_cents is None:
                    tolerance_cents = self.pitch_correction_tolerance_cents
                if reference_scale is None:
                    reference_scale = self.pitch_correction_reference_scale

                # Get scale notes
                scale_notes = self._get_scale_notes(reference_scale)

                corrections = []
                for i in range(len(f0)):
                    # Skip unvoiced frames
                    if not voiced[i] or f0[i] <= 0:
                        continue

                    # Get detected note name and cents offset
                    detected_note, cents_offset = self._f0_to_note_name(f0[i])

                    # Check if note is in scale
                    if detected_note not in scale_notes:
                        # Find nearest scale note
                        target_note = self._find_nearest_scale_note(f0[i], scale_notes)
                        target_f0 = self._note_name_to_f0(target_note)
                        correction_cents = 1200.0 * np.log2(target_f0 / f0[i])

                        # Only suggest correction if deviation exceeds tolerance
                        if abs(correction_cents) > tolerance_cents:
                            corrections.append({
                                'timestamp': float(times[i]),
                                'detected_f0_hz': float(f0[i]),
                                'detected_note': detected_note,
                                'target_note': target_note,
                                'target_f0_hz': float(target_f0),
                                'correction_cents': float(correction_cents)
                            })
                    else:
                        # Note is in scale, but check if it needs fine-tuning
                        target_f0 = self._note_name_to_f0(detected_note)
                        correction_cents = 1200.0 * np.log2(target_f0 / f0[i])

                        if abs(correction_cents) > tolerance_cents:
                            corrections.append({
                                'timestamp': float(times[i]),
                                'detected_f0_hz': float(f0[i]),
                                'detected_note': detected_note,
                                'target_note': detected_note,
                                'target_f0_hz': float(target_f0),
                                'correction_cents': float(correction_cents)
                            })

                return corrections

            except Exception as e:
                self.logger.error(f"Pitch correction suggestion failed: {e}")
                return []

    def _f0_to_note_name(self, f0_hz: float) -> Tuple[str, float]:
        """Convert F0 frequency to note name with cents offset

        Args:
            f0_hz: Frequency in Hz

        Returns:
            Tuple of (note_name, cents_offset) where note_name is like 'C4' or 'F#5'
            and cents_offset is the deviation in cents from the exact note
        """
        if f0_hz <= 0:
            return 'N/A', 0.0

        # A4 = 440 Hz reference
        a4_freq = 440.0
        semitones_from_a4 = 12.0 * np.log2(f0_hz / a4_freq)

        # Round to nearest semitone
        nearest_semitone = round(semitones_from_a4)
        cents_offset = (semitones_from_a4 - nearest_semitone) * 100.0

        # Note names (A4 = 0)
        note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

        # Calculate octave and note
        octave = 4 + (nearest_semitone + 12) // 12
        note_index = (nearest_semitone + 12) % 12
        note_name = f"{note_names[note_index]}{octave}"

        return note_name, cents_offset

    def _get_scale_notes(self, reference_scale: str) -> List[str]:
        """Get note names in the major scale for the reference key

        Args:
            reference_scale: Root note of scale ('C', 'D', 'E', 'F', 'G', 'A', 'B')

        Returns:
            List of note names in the major scale (all octaves)
        """
        # Major scale intervals (semitones from root)
        major_scale_intervals = [0, 2, 4, 5, 7, 9, 11]

        # All note names
        all_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Find root index
        reference_scale = reference_scale.upper().strip()
        if reference_scale not in all_notes:
            self.logger.warning(f"Unknown scale {reference_scale}, defaulting to C major")
            reference_scale = 'C'

        root_index = all_notes.index(reference_scale)

        # Generate scale notes (including all octaves)
        scale_notes = []
        for interval in major_scale_intervals:
            note_index = (root_index + interval) % 12
            note_name = all_notes[note_index]
            # Add all octaves (0-8)
            for octave in range(0, 9):
                scale_notes.append(f"{note_name}{octave}")

        return scale_notes

    def _find_nearest_scale_note(self, f0_hz: float, scale_notes: List[str]) -> str:
        """Find the nearest note in the scale to the given F0

        Args:
            f0_hz: Frequency in Hz
            scale_notes: List of note names in the scale

        Returns:
            Note name of the nearest scale note
        """
        if f0_hz <= 0:
            return 'C4'

        # Convert all scale notes to frequencies and find nearest
        min_diff = float('inf')
        nearest_note = 'C4'

        for note in scale_notes:
            note_f0 = self._note_name_to_f0(note)
            diff = abs(note_f0 - f0_hz)
            if diff < min_diff:
                min_diff = diff
                nearest_note = note

        return nearest_note

    def _note_name_to_f0(self, note_name: str) -> float:
        """Convert note name to F0 frequency

        Args:
            note_name: Note name like 'C4', 'F#5', etc.

        Returns:
            Frequency in Hz
        """
        try:
            # Parse note name
            if len(note_name) < 2:
                return 440.0  # Default to A4

            # Extract note and octave
            if note_name[1] == '#':
                note = note_name[:2]
                octave = int(note_name[2:])
            else:
                note = note_name[0]
                octave = int(note_name[1:])

            # Note index relative to A
            note_map = {
                'A': 0, 'A#': 1, 'B': 2, 'C': 3, 'C#': 4, 'D': 5,
                'D#': 6, 'E': 7, 'F': 8, 'F#': 9, 'G': 10, 'G#': 11
            }

            if note not in note_map:
                return 440.0

            # Calculate semitones from A4
            semitones = note_map[note] + (octave - 4) * 12

            # Calculate frequency
            f0 = 440.0 * (2.0 ** (semitones / 12.0))
            return f0

        except Exception as e:
            self.logger.warning(f"Failed to parse note name {note_name}: {e}")
            return 440.0

    def create_realtime_state(self, sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """Create initial state dictionary for real-time streaming

        This method initializes the state required for stateful real-time pitch
        extraction with overlap buffering and smoothing.

        Args:
            sample_rate: Optional sample rate to store in state for consistency validation

        Returns:
            Dictionary containing:
                - 'overlap_buffer': Empty audio buffer for frame overlap
                - 'smoothing_history': Empty pitch history for temporal smoothing
                - 'frame_count': Frame counter for tracking
                - 'last_pitch': Last valid pitch value for continuity
                - 'sample_rate': Stored sample rate for consistency (COMMENT 6 FIX)

        Example:
            >>> extractor = SingingPitchExtractor()
            >>> state = extractor.create_realtime_state(sample_rate=22050)
            >>> for chunk in audio_stream:
            ...     f0 = extractor.extract_f0_realtime(chunk, state=state)
            ...     # Process f0...

        Note:
            - Creates CPU tensors by default; will be moved to device during processing
            - Thread-safe for concurrent state creation
            - COMMENT 6 FIX: sample_rate stored for persistence and validation
        """
        with self.lock:
            return {
                'overlap_buffer': torch.zeros(self.realtime_buffer_size, dtype=torch.float32),
                'smoothing_history': [],
                'frame_count': 0,
                'last_pitch': 0.0,
                # COMMENT 6 FIX: Store sample_rate in state
                'sample_rate': sample_rate
            }

    def reset_realtime_state(self, state: Dict[str, Any]) -> None:
        """Reset real-time state to initial values

        This method clears the overlap buffer and smoothing history while preserving
        the state dictionary structure.

        Args:
            state: State dictionary from create_realtime_state() or previous processing

        Example:
            >>> state = extractor.create_realtime_state()
            >>> # ... process some audio ...
            >>> extractor.reset_realtime_state(state)  # Clear for new stream

        Note:
            - Preserves tensor device placement and sample_rate (COMMENT 6 FIX)
            - Thread-safe for concurrent reset operations
        """
        with self.lock:
            if 'overlap_buffer' in state:
                state['overlap_buffer'].zero_()
            state['smoothing_history'] = []
            state['frame_count'] = 0
            state['last_pitch'] = 0.0
            # COMMENT 6 FIX: Preserve sample_rate across resets

    def extract_f0_realtime(
        self,
        audio_chunk: "torch.Tensor",
        sample_rate: Optional[int] = None,
        state: Optional[Dict[str, Any]] = None,
        use_cuda_kernel: bool = True,
        return_device: str = 'cpu'  # COMMENT 4 FIX: Add return_device flag
    ) -> "torch.Tensor":
        """Extract F0 for real-time applications with stateful overlap buffering and smoothing

        This method provides optimized real-time pitch extraction using CUDA kernels or
        fast torchcrepe. It supports stateful processing with overlap buffering for
        seamless frame-to-frame transitions and temporal smoothing for stable pitch output.

        Args:
            audio_chunk: Audio tensor chunk for processing (1D tensor)
            sample_rate: Sample rate in Hz (required if not using state with sample_rate)
            state: Optional state dictionary from create_realtime_state() for stateful
                processing with overlap buffering and smoothing. If None, processes
                without state (less smooth).
            use_cuda_kernel: Use CUDA kernel if available (default: True)
            return_device: Device for return tensor ('cpu' or 'cuda'). Default: 'cpu' (COMMENT 4 FIX)

        Returns:
            F0 tensor with pitch values in Hz on specified device. Shape depends on chunk
            size and hop length. Returns single-frame or multi-frame tensor depending on input size.
            Returns empty tensor (or single zero) when chunk produces zero frames (too short).
            COMMENT 4 FIX: Returns tensor normalized to CPU by default (.detach().cpu())

        Example:
            >>> # Stateless processing (simple)
            >>> extractor = SingingPitchExtractor()
            >>> f0 = extractor.extract_f0_realtime(audio_chunk, sample_rate=22050)

            >>> # Stateful processing (recommended for streaming)
            >>> state = extractor.create_realtime_state()
            >>> for chunk in audio_stream:
            ...     f0 = extractor.extract_f0_realtime(chunk, sample_rate=22050, state=state)
            ...     # Process f0 with smooth transitions...

        Note:
            - Handles rapid pitch changes with temporal smoothing
            - Edge case: silence returns zeros
            - Edge case: noise is filtered using confidence threshold
            - Edge case: short chunks producing zero frames return empty/zero tensor (COMMENT 2)
            - State management is thread-safe
            - Falls back to torchcrepe if CUDA kernel unavailable
            - Uses 'tiny' model for real-time speed
        """
        with self.lock:
            # COMMENT 6 FIX: Use sample_rate from state if available, validate consistency
            if state is not None and 'sample_rate' in state and state['sample_rate'] is not None:
                if sample_rate is not None and sample_rate != state['sample_rate']:
                    self.logger.warning(
                        f"Sample rate mismatch: provided {sample_rate} Hz, state has {state['sample_rate']} Hz. "
                        f"Using state sample rate for consistency."
                    )
                sample_rate = state['sample_rate']
            elif sample_rate is None:
                sample_rate = 22050  # Default

            # COMMENT 6 FIX: Store sample_rate in state if provided
            if state is not None and 'sample_rate' not in state:
                state['sample_rate'] = sample_rate

            # Process with overlap buffer if state provided
            if state is not None:
                # Get overlap buffer
                overlap_buffer = state.get('overlap_buffer')

                # Ensure audio chunk is on same device as overlap buffer
                if overlap_buffer is not None:
                    audio_chunk = audio_chunk.to(overlap_buffer.device)

                # Concatenate overlap buffer with new chunk
                if overlap_buffer is not None and overlap_buffer.numel() > 0 and torch.any(overlap_buffer != 0):
                    combined_audio = torch.cat([overlap_buffer, audio_chunk])
                else:
                    combined_audio = audio_chunk

                # COMMENT 2 FIX: Guard overlap buffer update with positive check
                # Update overlap buffer with last portion of current chunk
                overlap_size = min(len(audio_chunk) // 4, self.realtime_buffer_size)  # 25% overlap
                if overlap_size > 0 and len(audio_chunk) >= overlap_size:
                    state['overlap_buffer'] = audio_chunk[-overlap_size:].clone()

                audio_to_process = combined_audio
            else:
                audio_to_process = audio_chunk

        # Try CUDA kernel if requested and available
        if use_cuda_kernel and self.use_cuda_kernel_fallback:
            try:
                # COMMENT 7 FIX: Use _load_cuda_extension with fallback paths
                _ck = self._load_cuda_extension()
                if _ck is None:
                    raise ImportError("cuda_kernels not available via any fallback path")

                # COMMENT 1 FIX: Use audio_to_process instead of undefined 'audio'
                # Prepare tensors
                if audio_to_process.device.type != 'cuda':
                    audio_to_process = audio_to_process.cuda()

                # Compute frame parameters consistently with CUDA kernel
                frame_length = 2048  # Must match CUDA kernel default
                hop_length = int(self.hop_length_ms * sample_rate / 1000.0)

                # COMMENT 1 FIX: Use audio_to_process for frame count
                # Compute n_frames identically to CUDA kernel
                n_samples = len(audio_to_process)
                n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

                # COMMENT 2 FIX: Early return for short chunks producing zero frames
                if n_frames <= 0:
                    # Return empty tensor or single zero on requested device
                    if return_device == 'cpu':
                        return torch.zeros(1, dtype=torch.float32)
                    else:
                        return torch.zeros(1, dtype=torch.float32, device=self.device)

                output_pitch = torch.zeros(n_frames, device=audio_to_process.device)
                output_confidence = torch.zeros(n_frames, device=audio_to_process.device)
                output_vibrato = torch.zeros(n_frames, device=audio_to_process.device)

                # COMMENT 1 FIX: Use audio_to_process in kernel call
                # COMMENT 3 FIX: Use cuda_cmnd_threshold instead of confidence_threshold
                _ck.launch_pitch_detection(audio_to_process, output_pitch, output_confidence,
                                          output_vibrato, float(sample_rate),
                                          frame_length, hop_length,
                                          float(self.fmin), float(self.fmax),
                                          float(self.cuda_cmnd_threshold))

                # COMMENT 1 FIX: Temporal smoothing operates on result from audio_to_process
                # Apply temporal smoothing if state provided
                if state is not None:
                    output_pitch = self._apply_temporal_smoothing(output_pitch, state)

                # COMMENT 1 FIX: Return-device normalization operates on result from audio_to_process
                # COMMENT 4 FIX: Normalize to requested device
                if return_device == 'cpu' and output_pitch.device.type != 'cpu':
                    output_pitch = output_pitch.detach().cpu()
                elif return_device == 'cuda' and output_pitch.device.type == 'cpu':
                    output_pitch = output_pitch.to(self.device)

                return output_pitch
            except Exception as e:
                self.logger.warning(f"CUDA kernel fallback failed, using torchcrepe: {e}")

        # Fallback to torchcrepe with tiny model for speed
        with torch.no_grad():
            hop_length = int(self.hop_length_ms * sample_rate / 1000.0)
            # COMMENT 1 FIX: Use audio_to_process instead of undefined 'audio'
            pred = torchcrepe.predict(
                audio_to_process,
                sample_rate,
                hop_length=hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                model='tiny',  # Faster model for real-time
                batch_size=self.batch_size,
                device=self.device,
                return_periodicity=False
            )
            # Handle both tuple (pitch, periodicity) and single pitch return
            if isinstance(pred, tuple):
                pitch = pred[0]
            else:
                pitch = pred
            # Squeeze batch dimension if present
            if pitch.dim() > 1:
                pitch = pitch.squeeze(0)

            # Apply temporal smoothing if state provided
            if state is not None:
                pitch = self._apply_temporal_smoothing(pitch, state)

            # COMMENT 4 FIX: Normalize to requested device
            if return_device == 'cpu' and pitch.device.type != 'cpu':
                pitch = pitch.detach().cpu()
            elif return_device == 'cuda' and pitch.device.type == 'cpu':
                pitch = pitch.to(self.device)

            return pitch

    def _apply_temporal_smoothing(
        self,
        pitch: torch.Tensor,
        state: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply temporal smoothing to pitch output using state history

        Args:
            pitch: Raw pitch tensor
            state: State dictionary with smoothing_history

        Returns:
            Smoothed pitch tensor
        """
        smoothing_history = state.get('smoothing_history', [])

        # Convert to list if tensor
        if pitch.dim() == 0:
            pitch_values = [float(pitch.item())]
        else:
            pitch_values = pitch.cpu().tolist() if isinstance(pitch, torch.Tensor) else [pitch]

        # Smooth each value
        smoothed = []
        for val in pitch_values:
            # Add to history
            smoothing_history.append(val)

            # Keep only recent history
            if len(smoothing_history) > self.realtime_smoothing_window:
                smoothing_history.pop(0)

            # Compute smoothed value (median filter for robustness)
            if len(smoothing_history) > 0:
                sorted_history = sorted(smoothing_history)
                median_idx = len(sorted_history) // 2
                smoothed_val = sorted_history[median_idx]
            else:
                smoothed_val = val

            smoothed.append(smoothed_val)

        # Update state
        state['smoothing_history'] = smoothing_history

        # Convert back to tensor
        if len(smoothed) == 1:
            return torch.tensor(smoothed[0], dtype=pitch.dtype, device=pitch.device)
        else:
            return torch.tensor(smoothed, dtype=pitch.dtype, device=pitch.device)

    def batch_extract(
        self,
        audio_list: List[Union["torch.Tensor", str]],
        sample_rate: Optional[int] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """Extract F0 from multiple audio files/tensors with true batching

        Args:
            audio_list: List of audio tensors or file paths
            sample_rate: Sample rate (required for tensors)

        Returns:
            List of F0 data dictionaries

        Note:
            This implementation uses true batching for items with the same sample rate
            and similar lengths. Items are grouped by sample rate and processed together.
        """
        if not audio_list:
            return []

        # Group items by sample rate for true batching
        sr_groups = {}  # sample_rate -> [(index, audio, sr)]

        for idx, audio in enumerate(audio_list):
            try:
                # Load and determine sample rate
                if isinstance(audio, str):
                    audio_data, original_sr = self.audio_processor.load_audio(audio, return_sr=True)
                    # Use the processor's target sample rate (resampled SR), not original
                    sr = self.audio_processor.sample_rate
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.cpu().numpy()
                else:
                    audio_data = audio
                    sr = sample_rate
                    if sr is None:
                        self.logger.error(f"Sample rate required for tensor at index {idx}")
                        sr_groups.setdefault('error', []).append((idx, None, None))
                        continue

                # Convert to numpy if needed
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()

                # Ensure 1D
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=0)

                # Group by sample rate and track original length
                if sr not in sr_groups:
                    sr_groups[sr] = []
                sr_groups[sr].append((idx, audio_data, sr, len(audio_data)))

            except Exception as e:
                self.logger.error(f"Failed to load item {idx}: {e}")
                sr_groups.setdefault('error', []).append((idx, None, None))

        # Process each sample rate group with true batching
        results = [None] * len(audio_list)

        for sr, items in sr_groups.items():
            if sr == 'error':
                # Handle errors
                for idx, _, _, _ in items:
                    results[idx] = None
                continue

            # Find max length in group for padding
            max_len = max(orig_len for _, _, _, orig_len in items)

            # Pad and stack into batch tensor
            batch_audio = []
            for _, audio_data, _, orig_len in items:
                # Pad to max length
                if len(audio_data) < max_len:
                    padded = np.pad(audio_data, (0, max_len - len(audio_data)), mode='constant')
                else:
                    padded = audio_data
                batch_audio.append(padded)

            batch_audio = np.stack(batch_audio, axis=0)  # (B, T)
            batch_tensor = torch.from_numpy(batch_audio).float()

            # Move to device
            if self.gpu_acceleration and self.device != 'cpu':
                batch_tensor = batch_tensor.to(self.device)

            # Extract F0 for entire batch
            try:
                hop_length = int(self.hop_length_ms * sr / 1000.0)

                context = self.gpu_manager.device_context() if self.gpu_manager else self._null_context()
                with context:
                    with torch.no_grad():
                        # Process entire batch at once
                        pitch_batch, periodicity_batch = self._call_torchcrepe_predict(batch_tensor, sr, hop_length)

                # Split results per item
                for batch_idx, (orig_idx, audio_data, _, orig_len) in enumerate(items):
                    try:
                        # Extract this item's results
                        pitch = pitch_batch[batch_idx] if pitch_batch.dim() > 1 else pitch_batch
                        periodicity = periodicity_batch[batch_idx] if periodicity_batch.dim() > 1 else periodicity_batch

                        # COMMENT 3 FIX: Trim to original length using torchcrepe-aligned framing formula
                        # Only trim if this item was padded
                        if orig_len < max_len:
                            # Compute expected number of frames from original unpadded length
                            # torchcrepe uses: max(0, (n_samples - win_length) // hop_length + 1)
                            # with win_length=1024 by default (matches torchcrepe's CREPE model)
                            win_length = 1024  # Torchcrepe default window length
                            expected_frames = max(0, (orig_len - win_length) // hop_length + 1)

                            if expected_frames > 0 and len(pitch) > expected_frames:
                                pitch = pitch[:expected_frames]
                                periodicity = periodicity[:expected_frames]
                            elif expected_frames == 0:
                                # Audio too short, return single zero frame
                                pitch = pitch[:1] * 0
                                periodicity = periodicity[:1] * 0

                        # Post-process
                        pitch, periodicity = self._post_process(pitch, periodicity, hop_length, sr)

                        # Compute voiced mask
                        voiced = periodicity > self.confidence_threshold

                        # COMMENT 5 FIX: Detect vibrato with GPU kernel if enabled
                        vibrato_data = self._detect_vibrato_with_gpu_fallback(
                            pitch, voiced, sr, hop_length
                        )

                        # Convert to numpy
                        pitch_np = pitch.squeeze().cpu().numpy() if isinstance(pitch, torch.Tensor) else pitch
                        voiced_np = voiced.squeeze().cpu().numpy() if isinstance(voiced, torch.Tensor) else voiced
                        confidence_np = periodicity.squeeze().cpu().numpy() if isinstance(periodicity, torch.Tensor) else periodicity

                        # Build result
                        result = {
                            'f0': pitch_np,
                            'voiced': voiced_np,
                            'confidence': confidence_np,
                            'vibrato': vibrato_data,
                            'sample_rate': sr,
                            'hop_length': hop_length,
                            'times': np.arange(len(pitch_np)) * hop_length / sr
                        }

                        results[orig_idx] = result

                    except Exception as e:
                        self.logger.error(f"Failed to process batch item {orig_idx}: {e}")
                        results[orig_idx] = None

            except Exception as e:
                self.logger.error(f"Batch extraction failed for sample rate {sr}: {e}")
                for idx, _, _ in items:
                    if results[idx] is None:
                        results[idx] = None

        return results

    def get_pitch_statistics(self, f0_data: Dict) -> Dict[str, float]:
        """Compute pitch statistics from F0 data

        Args:
            f0_data: F0 data dictionary from extract_f0_contour

        Returns:
            Dictionary with statistics
        """
        f0 = f0_data['f0']
        voiced = f0_data['voiced']

        # Extract voiced F0 values
        f0_voiced = f0[voiced]

        if len(f0_voiced) == 0:
            return {
                'mean_f0': 0.0,
                'std_f0': 0.0,
                'min_f0': 0.0,
                'max_f0': 0.0,
                'range_semitones': 0.0,
                'voiced_fraction': 0.0
            }

        mean_f0 = float(np.mean(f0_voiced))
        std_f0 = float(np.std(f0_voiced))
        min_f0 = float(np.min(f0_voiced))
        max_f0 = float(np.max(f0_voiced))

        # Compute range in semitones
        if min_f0 > 0:
            range_semitones = 12.0 * np.log2(max_f0 / min_f0)
        else:
            range_semitones = 0.0

        voiced_fraction = float(np.sum(voiced)) / len(voiced) if len(voiced) > 0 else 0.0

        return {
            'mean_f0': mean_f0,
            'std_f0': std_f0,
            'min_f0': min_f0,
            'max_f0': max_f0,
            'range_semitones': float(range_semitones),
            'voiced_fraction': voiced_fraction
        }

    def _load_cuda_extension(self) -> Optional[Any]:
        """Load CUDA extension with fallback import paths

        COMMENT 7 FIX: Augment with fallback paths and AUTOVOICE_CUDA_MODULE env var

        Tries multiple import strategies in order:
        1. Environment variable AUTOVOICE_CUDA_MODULE
        2. Direct import: cuda_kernels
        3. Namespaced imports: auto_voice.cuda_kernels, src.cuda_kernels
        4. Relative import from package
        5. sys.modules scan for already-imported modules ending with .cuda_kernels

        The sys.modules scan handles cases where extensions are preloaded or use
        non-standard packaging. It prefers longer module names (more specific) to
        avoid ambiguous picks.

        Returns:
            CUDA kernels module if available, None otherwise
        """
        # COMMENT 7 FIX: Check environment variable first
        env_module = os.environ.get('AUTOVOICE_CUDA_MODULE')
        if env_module:
            try:
                import importlib
                module = importlib.import_module(env_module)
                self.logger.info(f"Loaded CUDA extension from env var: {env_module}")
                return module
            except ImportError as e:
                self.logger.warning(f"Failed to import CUDA module from env var {env_module}: {e}")

        # COMMENT 7 FIX: Try standard import paths
        import_paths = [
            'cuda_kernels',
            'auto_voice.cuda_kernels',
            'src.cuda_kernels'
        ]

        for import_path in import_paths:
            try:
                import importlib
                module = importlib.import_module(import_path)
                self.logger.info(f"Loaded CUDA extension from: {import_path}")
                return module
            except ImportError:
                continue

        # COMMENT 7 FIX: Try relative import as last resort
        try:
            from .. import cuda_kernels
            self.logger.info("Loaded CUDA extension via relative import")
            return cuda_kernels
        except ImportError:
            pass

        # NEW: Scan sys.modules for already-imported CUDA modules
        # This handles cases where extensions are preloaded or use non-standard packaging
        import sys
        candidates = []

        for module_name, module_obj in sys.modules.items():
            if module_obj is None:
                continue
            # Match modules ending with '.cuda_kernels' or exactly 'cuda_kernels'
            if module_name == 'cuda_kernels' or module_name.endswith('.cuda_kernels'):
                candidates.append((module_name, module_obj))
                self.logger.debug(f"Found candidate CUDA module in sys.modules: {module_name}")

        if candidates:
            # Prefer more specific names (longer module names) to avoid ambiguous picks
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            selected_name, selected_module = candidates[0]
            self.logger.info(f"Loaded CUDA extension from sys.modules: {selected_name}")
            return selected_module

        self.logger.warning("CUDA extension not available via any import path or sys.modules scan")
        return None

    @staticmethod
    def _null_context():
        """Null context manager for when GPU manager is not available"""
        from contextlib import nullcontext
        return nullcontext()
