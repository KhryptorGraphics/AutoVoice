"""Singing pitch extraction using torchcrepe with GPU acceleration"""

from __future__ import annotations
import logging
import os
import threading
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchcrepe
    TORCHCREPE_AVAILABLE = True
except ImportError:
    TORCHCREPE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

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
        gpu_manager: Optional[Any] = None
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
        self.median_filter_width = self.config.get('median_filter_width', 3)
        self.mean_filter_width = self.config.get('mean_filter_width', 3)

        # Vibrato detection parameters
        self.vibrato_rate_range = self.config.get('vibrato_rate_range', [4.0, 8.0])
        self.vibrato_min_depth_cents = self.config.get('vibrato_min_depth_cents', 20.0)
        self.vibrato_min_duration_ms = self.config.get('vibrato_min_duration_ms', 250.0)

        # GPU optimization
        self.gpu_acceleration = self.config.get('gpu_acceleration', True)
        self.mixed_precision = self.config.get('mixed_precision', True)
        self.use_cuda_kernel_fallback = self.config.get('use_cuda_kernel_fallback', True)

        self.logger.info(
            f"SingingPitchExtractor initialized: model={self.model}, device={self.device}, "
            f"fmin={self.fmin}Hz, fmax={self.fmax}Hz"
        )

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
            'median_filter_width': 3,
            'mean_filter_width': 3,
            'vibrato_rate_range': [4.0, 8.0],
            'vibrato_min_depth_cents': 20.0,
            'vibrato_min_duration_ms': 250.0,
            'gpu_acceleration': True,
            'mixed_precision': True,
            'use_cuda_kernel_fallback': True
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
            'AUTOVOICE_PITCH_MODEL': 'model',
            'AUTOVOICE_PITCH_FMIN': 'fmin',
            'AUTOVOICE_PITCH_FMAX': 'fmax',
            'AUTOVOICE_PITCH_HOP_LENGTH': 'hop_length_ms'
        }
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    if config_key in ['fmin', 'fmax', 'hop_length_ms']:
                        value = float(value)
                    final_config[config_key] = value
                except ValueError:
                    self.logger.warning(f"Invalid value for {env_var}: {os.environ[env_var]}")

        # Override with constructor config (highest priority)
        if config:
            final_config.update(config)

        return final_config

    def extract_f0_contour(
        self,
        audio: Union[torch.Tensor, np.ndarray, str],
        sample_rate: Optional[int] = None,
        return_confidence: bool = True,
        return_times: bool = True
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
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
            PitchExtractionError: If extraction fails
        """
        with self.lock:
            try:
                # Load audio if file path
                if isinstance(audio, str):
                    audio, sample_rate = self.audio_processor.load_audio(audio, return_sr=True)
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()

                if sample_rate is None:
                    raise ValueError("sample_rate must be provided for tensor/array input")

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
                    with torch.no_grad():
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

                        # Call torchcrepe
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

                # Post-processing
                pitch, periodicity = self._post_process(pitch, periodicity, hop_length, sample_rate)

                # Compute voiced/unvoiced mask
                voiced = periodicity > self.confidence_threshold

                # Detect vibrato
                vibrato_data = self._detect_vibrato(pitch, voiced, sample_rate, hop_length)

                # Build result dictionary
                result = {
                    'f0': pitch.cpu().numpy() if isinstance(pitch, torch.Tensor) else pitch,
                    'voiced': voiced.cpu().numpy() if isinstance(voiced, torch.Tensor) else voiced,
                    'vibrato': vibrato_data,
                    'sample_rate': sample_rate,
                    'hop_length': hop_length
                }

                if return_confidence:
                    result['confidence'] = periodicity.cpu().numpy() if isinstance(periodicity, torch.Tensor) else periodicity

                if return_times:
                    n_frames = len(pitch)
                    result['times'] = np.arange(n_frames) * hop_length / sample_rate

                return result

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
            segments = self._find_voiced_segments(voiced, min_frames)

            vibrato_segments = []
            total_rate = 0.0
            total_depth = 0.0
            vibrato_count = 0

            for start, end in segments:
                if end - start < min_frames:
                    continue

                # Use filtered signal for better depth estimation
                seg_detrended = detrended_filtered[start:end]
                seg_valid = ~np.isnan(seg_detrended)

                if seg_valid.sum() < min_frames:
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
        """Compute moving average with NaN handling"""
        result = np.full_like(x, np.nan)
        for i in range(len(x)):
            start = max(0, i - window//2)
            end = min(len(x), i + window//2 + 1)
            window_data = x[start:end]
            valid = ~np.isnan(window_data)
            if valid.sum() > 0:
                result[i] = np.mean(window_data[valid])
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

    def extract_f0_realtime(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        use_cuda_kernel: bool = True
    ) -> torch.Tensor:
        """Extract F0 for real-time applications using CUDA kernel or fast torchcrepe

        Args:
            audio: Audio tensor
            sample_rate: Sample rate
            use_cuda_kernel: Use CUDA kernel if available

        Returns:
            F0 tensor
        """
        # Try CUDA kernel if requested and available
        if use_cuda_kernel and self.use_cuda_kernel_fallback:
            try:
                import cuda_kernels
                # Prepare tensors
                if audio.device.type != 'cuda':
                    audio = audio.cuda()

                # Compute frame parameters consistently with CUDA kernel
                frame_length = 2048  # Must match CUDA kernel default
                hop_length = int(self.hop_length_ms * sample_rate / 1000.0)

                # Compute n_frames identically to CUDA kernel
                n_samples = len(audio)
                n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

                output_pitch = torch.zeros(n_frames, device=audio.device)
                output_confidence = torch.zeros(n_frames, device=audio.device)
                output_vibrato = torch.zeros(n_frames, device=audio.device)

                cuda_kernels.launch_pitch_detection(audio, output_pitch, output_confidence,
                                                   output_vibrato, float(sample_rate),
                                                   frame_length, hop_length)
                return output_pitch
            except Exception as e:
                self.logger.warning(f"CUDA kernel fallback failed, using torchcrepe: {e}")

        # Fallback to torchcrepe with tiny model for speed
        with torch.no_grad():
            hop_length = int(self.hop_length_ms * sample_rate / 1000.0)
            pitch, _ = torchcrepe.predict(
                audio,
                sample_rate,
                hop_length=hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                model='tiny',  # Faster model for real-time
                batch_size=self.batch_size,
                device=self.device,
                return_periodicity=False
            )
            return pitch

    def batch_extract(
        self,
        audio_list: List[Union[torch.Tensor, str]],
        sample_rate: Optional[int] = None
    ) -> List[Dict]:
        """Extract F0 from multiple audio files/tensors

        Args:
            audio_list: List of audio tensors or file paths
            sample_rate: Sample rate (required for tensors)

        Returns:
            List of F0 data dictionaries
        """
        results = []
        for audio in audio_list:
            try:
                result = self.extract_f0_contour(audio, sample_rate)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch extraction failed for item: {e}")
                results.append(None)
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

    @staticmethod
    def _null_context():
        """Null context manager for when GPU manager is not available"""
        from contextlib import nullcontext
        return nullcontext()
