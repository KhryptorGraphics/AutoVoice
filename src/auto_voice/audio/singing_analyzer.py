"""Singing voice analysis including breathiness, dynamics, and vocal quality"""

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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import parselmouth as pm
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .processor import AudioProcessor
from .pitch_extractor import SingingPitchExtractor

logger = logging.getLogger(__name__)


class SingingAnalysisError(Exception):
    """Base exception for singing analysis errors"""
    pass


class SingingAnalyzer:
    """Comprehensive singing voice analysis

    This class provides detailed analysis of singing voice including:
    - Breathiness detection (CPP, HNR, spectral tilt)
    - Dynamics analysis (RMS envelope, crescendos, diminuendos)
    - Vocal quality metrics (jitter, shimmer, spectral features)
    - Singing technique detection

    Example:
        >>> analyzer = SingingAnalyzer(device='cuda', gpu_manager=gpu_manager)
        >>> features = analyzer.analyze_singing_features('singing.wav')
        >>> print(f"Breathiness: {features['breathiness']['breathiness_score']:.2f}")
        >>> print(f"Dynamic range: {features['dynamics']['dynamic_range_db']:.1f} dB")

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        device (str): Device for processing
        gpu_manager: Optional GPUManager
        audio_processor (AudioProcessor): Audio I/O handler
        pitch_extractor (SingingPitchExtractor): Pitch extraction
        lock (threading.RLock): Thread safety lock
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        gpu_manager: Optional[Any] = None
    ):
        """Initialize SingingAnalyzer

        Args:
            config: Optional configuration dictionary
            device: Optional device string
            gpu_manager: Optional GPUManager instance
        """
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
            self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'

        self.gpu_manager = gpu_manager

        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.pitch_extractor = SingingPitchExtractor(device=self.device, gpu_manager=gpu_manager)

        # Configuration parameters
        self.hop_length_ms = self.config.get('hop_length_ms', 10.0)
        self.frame_length_ms = self.config.get('frame_length_ms', 25.0)

        # Breathiness
        self.breathiness_method = self.config.get('breathiness_method', 'cpp')
        self.use_parselmouth = self.config.get('use_parselmouth', True) and PARSELMOUTH_AVAILABLE
        self.cpp_fmin = self.config.get('cpp_fmin', 60.0)
        self.cpp_fmax = self.config.get('cpp_fmax', 300.0)
        self.hnr_min_pitch = self.config.get('hnr_min_pitch', 75.0)

        breathiness_weights = self.config.get('breathiness_weights', {})
        self.cpp_weight = breathiness_weights.get('cpp', 0.5)
        self.hnr_weight = breathiness_weights.get('hnr', 0.3)
        self.spectral_weight = breathiness_weights.get('spectral', 0.2)

        # Dynamics
        self.dynamics_smoothing_ms = self.config.get('dynamics_smoothing_ms', 50.0)
        self.dynamic_range_threshold_db = self.config.get('dynamic_range_threshold_db', 3.0)
        self.accent_threshold_db = self.config.get('accent_threshold_db', 6.0)

        # Vocal quality
        self.compute_jitter = self.config.get('compute_jitter', True)
        self.compute_shimmer = self.config.get('compute_shimmer', True)
        self.compute_spectral = self.config.get('compute_spectral', True)

        # Technique detection
        technique_thresholds = self.config.get('technique_thresholds', {})
        self.breathy_threshold = technique_thresholds.get('breathy_score', 0.6)
        self.belting_energy_db = technique_thresholds.get('belting_energy_db', -10.0)
        self.falsetto_f0_hz = technique_thresholds.get('falsetto_f0_hz', 400.0)
        self.vocal_fry_f0_hz = technique_thresholds.get('vocal_fry_f0_hz', 80.0)

        # GPU optimization
        self.gpu_acceleration = self.config.get('gpu_acceleration', True)

        if not self.use_parselmouth:
            self.logger.warning("praat-parselmouth not available, using fallback methods for breathiness")

        self.logger.info(
            f"SingingAnalyzer initialized: device={self.device}, "
            f"use_parselmouth={self.use_parselmouth}, method={self.breathiness_method}"
        )

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from multiple sources"""
        final_config = {
            'hop_length_ms': 10.0,
            'frame_length_ms': 25.0,
            'breathiness_method': 'cpp',
            'use_parselmouth': True,
            'cpp_fmin': 60.0,
            'cpp_fmax': 300.0,
            'hnr_min_pitch': 75.0,
            'breathiness_weights': {'cpp': 0.5, 'hnr': 0.3, 'spectral': 0.2},
            'dynamics_smoothing_ms': 50.0,
            'dynamic_range_threshold_db': 3.0,
            'accent_threshold_db': 6.0,
            'compute_jitter': True,
            'compute_shimmer': True,
            'compute_spectral': True,
            'technique_thresholds': {
                'breathy_score': 0.6,
                'belting_energy_db': -10.0,
                'falsetto_f0_hz': 400.0,
                'vocal_fry_f0_hz': 80.0
            },
            'gpu_acceleration': True
        }

        # Load from YAML
        config_path = Path('config/audio_config.yaml')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'singing_analysis' in yaml_config:
                        final_config.update(yaml_config['singing_analysis'])
            except Exception as e:
                self.logger.warning(f"Failed to load YAML config: {e}")

        # Override with constructor config
        if config:
            final_config.update(config)

        return final_config

    def analyze_singing_features(
        self,
        audio: Union[torch.Tensor, np.ndarray, str],
        sample_rate: Optional[int] = None,
        f0_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Comprehensive singing feature analysis

        Args:
            audio: Audio tensor, numpy array, or file path
            sample_rate: Sample rate (required for tensor/array)
            f0_data: Optional pre-computed F0 data

        Returns:
            Dictionary containing:
                - 'breathiness': Breathiness metrics
                - 'dynamics': Dynamics analysis
                - 'vibrato': Vibrato parameters
                - 'vocal_quality': Vocal quality metrics
                - 'f0_data': F0 contour data
                - 'times': Time stamps
                - 'sample_rate': Sample rate
        """
        with self.lock:
            try:
                # Load audio if file path
                if isinstance(audio, str):
                    audio_data, sample_rate = self.audio_processor.load_audio(audio, return_sr=True)
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.numpy()
                else:
                    audio_data = audio

                if sample_rate is None:
                    raise ValueError("sample_rate must be provided")

                # Convert to numpy if torch tensor
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()

                # Ensure 1D
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=0)

                # Extract F0 if not provided
                if f0_data is None:
                    f0_data = self.pitch_extractor.extract_f0_contour(audio_data, sample_rate)

                # Compute features
                breathiness_data = self.compute_breathiness(audio_data, sample_rate, f0_data)
                dynamics_data = self.compute_dynamics(audio_data, sample_rate)
                vocal_quality_data = self.compute_vocal_quality(audio_data, sample_rate, f0_data)

                # Build result
                result = {
                    'breathiness': breathiness_data,
                    'dynamics': dynamics_data,
                    'vibrato': f0_data.get('vibrato', {}),
                    'vocal_quality': vocal_quality_data,
                    'f0_data': f0_data,
                    'times': f0_data.get('times', None),
                    'sample_rate': sample_rate
                }

                return result

            except Exception as e:
                self.logger.error(f"Singing analysis failed: {e}")
                raise SingingAnalysisError(f"Failed to analyze singing features: {e}") from e

    def compute_breathiness(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        f0_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Compute breathiness metrics

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            f0_data: Optional F0 data

        Returns:
            Dictionary with breathiness metrics
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure float32
        audio = audio.astype(np.float32)

        if self.use_parselmouth and PARSELMOUTH_AVAILABLE:
            try:
                # Use praat-parselmouth
                snd = pm.Sound(audio, sampling_frequency=sample_rate)
                duration = snd.get_total_duration()

                # Compute CPP (Cepstral Peak Prominence)
                pcg = call(snd, "To PowerCepstrogram", 0.01, self.cpp_fmin, 5000)
                cpp_mean = call(pcg, "Get peak prominence (hillenbrand)", 0, 0, self.cpp_fmin, self.cpp_fmax)

                # Frame-wise CPP
                times = np.arange(0.01, duration, 0.01)
                cpp_track = []
                for t in times:
                    try:
                        cpp_val = call(pcg, "Get peak prominence (hillenbrand)", t, t, self.cpp_fmin, self.cpp_fmax)
                        cpp_track.append(cpp_val if cpp_val is not None else 0.0)
                    except Exception:
                        cpp_track.append(0.0)
                cpp_track = np.array(cpp_track)

                # Compute HNR (Harmonic-to-Noise Ratio)
                harm = snd.to_harmonicity_ac(time_step=0.01, minimum_pitch=self.hnr_min_pitch)
                hnr_mean = call(harm, "Get mean", 0, 0)

                # Frame-wise HNR
                hnr_track = []
                for t in times:
                    try:
                        hnr_val = call(harm, "Get value in time", t)
                        hnr_track.append(hnr_val if hnr_val is not None else 0.0)
                    except Exception:
                        hnr_track.append(0.0)
                hnr_track = np.array(hnr_track)

                # Compute spectral tilt (H1-H2) - simplified
                h1_h2 = self._compute_spectral_tilt(audio, sample_rate, f0_data)

                # Normalize metrics
                cpp_norm = np.clip((20.0 - cpp_mean) / 15.0, 0.0, 1.0)  # Higher CPP = less breathy
                hnr_norm = np.clip((20.0 - hnr_mean) / 15.0, 0.0, 1.0)  # Higher HNR = less breathy
                h1h2_norm = np.clip(h1_h2 / 20.0, 0.0, 1.0)  # Higher H1-H2 = more breathy

                # Combined breathiness score
                breathiness_score = (
                    self.cpp_weight * cpp_norm +
                    self.hnr_weight * hnr_norm +
                    self.spectral_weight * h1h2_norm
                )

                return {
                    'cpp': float(cpp_mean),
                    'cpp_track': cpp_track,
                    'hnr': float(hnr_mean),
                    'hnr_track': hnr_track,
                    'h1_h2': float(h1_h2),
                    'breathiness_score': float(breathiness_score),
                    'method': 'parselmouth'
                }

            except Exception as e:
                self.logger.warning(f"Parselmouth breathiness failed: {e}, using fallback")

        # Fallback method using librosa/basic DSP
        return self._compute_breathiness_fallback(audio, sample_rate, f0_data)

    def _compute_breathiness_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Fallback breathiness computation using basic DSP"""
        # Compute spectrogram
        hop_length = int(self.hop_length_ms * sample_rate / 1000.0)
        n_fft = int(self.frame_length_ms * sample_rate / 1000.0)

        if LIBROSA_AVAILABLE:
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        else:
            # Very basic fallback
            S = np.abs(np.fft.rfft(audio))

        # Estimate HNR from spectral content
        # Lower frequencies = harmonics, higher = noise
        harmonic_band = S[:len(S)//4].mean()
        noise_band = S[len(S)//2:].mean()

        hnr_estimate = 10.0 * np.log10((harmonic_band + 1e-10) / (noise_band + 1e-10))
        hnr_estimate = np.clip(hnr_estimate, -10, 30)

        # Simple breathiness score
        breathiness_score = np.clip((20.0 - hnr_estimate) / 20.0, 0.0, 1.0)

        return {
            'cpp': 0.0,  # Not available in fallback
            'cpp_track': np.array([]),
            'hnr': float(hnr_estimate),
            'hnr_track': np.array([]),
            'h1_h2': 0.0,
            'breathiness_score': float(breathiness_score),
            'method': 'fallback'
        }

    def _compute_spectral_tilt(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0_data: Optional[Dict]
    ) -> float:
        """Compute spectral tilt (H1-H2)"""
        try:
            if not LIBROSA_AVAILABLE:
                return 0.0

            # Compute FFT size from frame_length_ms
            n_fft = int(self.frame_length_ms * sample_rate / 1000.0)
            if n_fft < 512:
                n_fft = 512

            # Compute spectrum with explicit n_fft
            S = np.abs(librosa.stft(audio, n_fft=n_fft))
            S_db = librosa.amplitude_to_db(S + 1e-10)

            # Compute frequencies with matching n_fft
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

            # Ensure frequency array matches spectrum size
            if len(freqs) != S_db.shape[0]:
                self.logger.warning(f"Frequency array size mismatch: {len(freqs)} vs {S_db.shape[0]}")
                return 0.0

            # Find first two harmonic peaks (simplified)
            h1_range = (freqs > 80) & (freqs < 300)
            h2_range = (freqs > 300) & (freqs < 600)

            h1_mag = np.mean(S_db[h1_range]) if h1_range.any() else 0.0
            h2_mag = np.mean(S_db[h2_range]) if h2_range.any() else 0.0

            return float(h1_mag - h2_mag)
        except Exception as e:
            self.logger.warning(f"Spectral tilt computation failed: {e}")
            return 0.0

    def compute_dynamics(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        smoothing_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compute dynamics (energy envelope)

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            smoothing_ms: RMS smoothing window in ms

        Returns:
            Dictionary with dynamics features
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if smoothing_ms is None:
            smoothing_ms = self.dynamics_smoothing_ms

        # Frame audio
        hop_length = int(self.hop_length_ms * sample_rate / 1000.0)
        frame_length = int(self.frame_length_ms * sample_rate / 1000.0)

        # Compute RMS energy
        rms = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms_val = np.sqrt(np.mean(frame**2))
            rms.append(rms_val)

        rms = np.array(rms)

        # Guard against empty/very short audio
        if len(rms) == 0:
            return {
                'rms_envelope': np.array([]),
                'db_envelope': np.array([]),
                'dynamic_range_db': 0.0,
                'mean_db': 0.0,
                'std_db': 0.0,
                'crescendos': [],
                'diminuendos': [],
                'accents': [],
                'times': np.array([])
            }

        # Smooth RMS
        smooth_window = int(smoothing_ms / self.hop_length_ms)
        if smooth_window > 1:
            rms = np.convolve(rms, np.ones(smooth_window)/smooth_window, mode='same')

        # Convert to dB
        db = 20.0 * np.log10(rms + 1e-10)

        # Compute statistics
        dynamic_range = float(np.max(db) - np.min(db))
        mean_db = float(np.mean(db))
        std_db = float(np.std(db))

        # Detect crescendos and diminuendos
        crescendos, diminuendos = self._detect_dynamic_contours(db, sample_rate, hop_length)

        # Detect accents (sudden peaks)
        accents = self._detect_accents(db, sample_rate, hop_length)

        # Time stamps
        times = np.arange(len(rms)) * hop_length / sample_rate

        return {
            'rms_envelope': rms,
            'db_envelope': db,
            'dynamic_range_db': dynamic_range,
            'mean_db': mean_db,
            'std_db': std_db,
            'crescendos': crescendos,
            'diminuendos': diminuendos,
            'accents': accents,
            'times': times
        }

    def _detect_dynamic_contours(
        self,
        db: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Detect crescendos and diminuendos"""
        # Compute derivative
        diff = np.diff(db)

        crescendos = []
        diminuendos = []

        # Find sustained increases/decreases
        min_duration_frames = int(0.5 * sample_rate / hop_length)  # 0.5 second minimum

        in_crescendo = False
        in_diminuendo = False
        start_idx = 0

        for i in range(len(diff)):
            if diff[i] > self.dynamic_range_threshold_db / 10.0:  # Threshold per frame
                if not in_crescendo:
                    start_idx = i
                    in_crescendo = True
                in_diminuendo = False
            elif diff[i] < -self.dynamic_range_threshold_db / 10.0:
                if not in_diminuendo:
                    start_idx = i
                    in_diminuendo = True
                in_crescendo = False
            else:
                # End of contour
                if in_crescendo and i - start_idx >= min_duration_frames:
                    start_time = start_idx * hop_length / sample_rate
                    end_time = i * hop_length / sample_rate
                    slope = (db[i] - db[start_idx]) / (end_time - start_time)
                    crescendos.append((start_time, end_time, slope))

                if in_diminuendo and i - start_idx >= min_duration_frames:
                    start_time = start_idx * hop_length / sample_rate
                    end_time = i * hop_length / sample_rate
                    slope = (db[i] - db[start_idx]) / (end_time - start_time)
                    diminuendos.append((start_time, end_time, slope))

                in_crescendo = False
                in_diminuendo = False

        return crescendos, diminuendos

    def _detect_accents(self, db: np.ndarray, sample_rate: int, hop_length: int) -> List[float]:
        """Detect sudden energy peaks (accents)"""
        # Find local maxima
        accents = []

        for i in range(1, len(db) - 1):
            if db[i] > db[i-1] and db[i] > db[i+1]:
                # Check if peak is significant
                local_mean = np.mean(db[max(0, i-5):min(len(db), i+5)])
                if db[i] - local_mean > self.accent_threshold_db:
                    time = i * hop_length / sample_rate
                    accents.append(time)

        return accents

    def compute_vocal_quality(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        f0_data: Dict
    ) -> Dict[str, Any]:
        """Compute vocal quality metrics

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            f0_data: F0 contour data

        Returns:
            Dictionary with vocal quality metrics
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        result = {}

        # Jitter (pitch perturbation)
        if self.compute_jitter:
            jitter = self._compute_jitter(f0_data)
            result['jitter_percent'] = jitter

        # Shimmer (amplitude perturbation)
        if self.compute_shimmer:
            shimmer = self._compute_shimmer(audio, sample_rate, f0_data)
            result['shimmer_percent'] = shimmer

        # Spectral features
        if self.compute_spectral:
            spectral_features = self._compute_spectral_features(audio, sample_rate)
            result.update(spectral_features)

        # Combined quality score (0-1, higher = better)
        quality_score = 1.0 - (result.get('jitter_percent', 0) / 5.0 + result.get('shimmer_percent', 0) / 10.0) / 2.0
        quality_score = np.clip(quality_score, 0.0, 1.0)
        result['quality_score'] = float(quality_score)

        return result

    def _compute_jitter(self, f0_data: Dict) -> float:
        """Compute jitter (pitch perturbation)"""
        f0 = f0_data['f0']
        voiced = f0_data['voiced']

        # Extract voiced F0
        f0_voiced = f0[voiced]

        if len(f0_voiced) < 3:
            return 0.0

        # Compute periods
        periods = 1.0 / (f0_voiced + 1e-10)

        # Period-to-period variation
        diff_periods = np.abs(np.diff(periods))
        mean_period = np.mean(periods)

        if mean_period == 0:
            return 0.0

        jitter = np.mean(diff_periods) / mean_period * 100.0  # Percentage

        return float(np.clip(jitter, 0, 100))

    def _compute_shimmer(self, audio: np.ndarray, sample_rate: int, f0_data: Dict) -> float:
        """Compute shimmer (amplitude perturbation)"""
        f0 = f0_data['f0']
        voiced = f0_data['voiced']

        if voiced.sum() < 3:
            return 0.0

        # Extract peak amplitudes per pitch period
        hop_length = f0_data.get('hop_length', int(0.01 * sample_rate))

        amplitudes = []
        for i, is_voiced in enumerate(voiced):
            if is_voiced and f0[i] > 0:
                # Get corresponding audio segment
                start = i * hop_length
                period_samples = int(sample_rate / f0[i])
                end = start + period_samples

                if end < len(audio):
                    segment = audio[start:end]
                    amp = np.max(np.abs(segment))
                    amplitudes.append(amp)

        if len(amplitudes) < 3:
            return 0.0

        amplitudes = np.array(amplitudes)

        # Amplitude variation
        diff_amp = np.abs(np.diff(amplitudes))
        mean_amp = np.mean(amplitudes)

        if mean_amp == 0:
            return 0.0

        shimmer = np.mean(diff_amp) / mean_amp * 100.0  # Percentage

        return float(np.clip(shimmer, 0, 100))

    def _compute_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Compute spectral features"""
        if not LIBROSA_AVAILABLE:
            return {
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_flux': 0.0
            }

        # Compute spectral features using librosa
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)

        # Spectral flux (change in spectrum over time)
        S = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

        return {
            'spectral_centroid': float(np.mean(centroid)),
            'spectral_rolloff': float(np.mean(rolloff)),
            'spectral_flux': float(np.mean(flux))
        }

    def detect_vocal_techniques(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        f0_data: Dict,
        breathiness_data: Dict
    ) -> Dict[str, Dict[str, Union[bool, float]]]:
        """Detect singing techniques

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            f0_data: F0 contour data
            breathiness_data: Breathiness metrics

        Returns:
            Dictionary with detected techniques and confidence scores
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        techniques = {}

        # Vibrato
        vibrato = f0_data.get('vibrato', {})
        techniques['vibrato'] = {
            'detected': vibrato.get('has_vibrato', False),
            'confidence': 1.0 if vibrato.get('has_vibrato', False) else 0.0
        }

        # Breathy
        breathiness_score = breathiness_data.get('breathiness_score', 0.0)
        techniques['breathy'] = {
            'detected': breathiness_score > self.breathy_threshold,
            'confidence': float(breathiness_score)
        }

        # Belting (high energy + high F0 + low breathiness)
        f0_voiced = f0_data['f0'][f0_data['voiced']]
        if len(f0_voiced) > 0:
            mean_f0 = np.mean(f0_voiced)
            rms = np.sqrt(np.mean(audio**2))
            rms_db = 20 * np.log10(rms + 1e-10)

            belting_detected = (
                rms_db > self.belting_energy_db and
                mean_f0 > 250 and
                breathiness_score < 0.4
            )
            techniques['belting'] = {
                'detected': bool(belting_detected),
                'confidence': float(np.clip((rms_db - self.belting_energy_db) / 20.0, 0, 1))
            }

            # Falsetto (high F0 + moderate breathiness)
            falsetto_detected = mean_f0 > self.falsetto_f0_hz and breathiness_score > 0.3
            techniques['falsetto'] = {
                'detected': bool(falsetto_detected),
                'confidence': float(np.clip(mean_f0 / self.falsetto_f0_hz, 0, 1))
            }

            # Vocal fry (very low F0)
            vocal_fry_detected = mean_f0 < self.vocal_fry_f0_hz
            techniques['vocal_fry'] = {
                'detected': bool(vocal_fry_detected),
                'confidence': float(np.clip((self.vocal_fry_f0_hz - mean_f0) / self.vocal_fry_f0_hz, 0, 1))
            }
        else:
            techniques['belting'] = {'detected': False, 'confidence': 0.0}
            techniques['falsetto'] = {'detected': False, 'confidence': 0.0}
            techniques['vocal_fry'] = {'detected': False, 'confidence': 0.0}

        return techniques

    def get_summary_statistics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from full feature analysis

        Args:
            features: Output from analyze_singing_features

        Returns:
            Simplified summary dictionary
        """
        summary = {}

        # Breathiness
        if 'breathiness' in features:
            summary['breathiness_score'] = features['breathiness'].get('breathiness_score', 0.0)
            summary['cpp'] = features['breathiness'].get('cpp', 0.0)
            summary['hnr'] = features['breathiness'].get('hnr', 0.0)

        # Dynamics
        if 'dynamics' in features:
            summary['dynamic_range_db'] = features['dynamics'].get('dynamic_range_db', 0.0)
            summary['mean_loudness_db'] = features['dynamics'].get('mean_db', 0.0)

        # Vibrato
        if 'vibrato' in features:
            summary['has_vibrato'] = features['vibrato'].get('has_vibrato', False)
            summary['vibrato_rate'] = features['vibrato'].get('rate_hz', 0.0)
            summary['vibrato_depth'] = features['vibrato'].get('depth_cents', 0.0)

        # Vocal quality
        if 'vocal_quality' in features:
            summary['vocal_quality_score'] = features['vocal_quality'].get('quality_score', 0.0)
            summary['jitter'] = features['vocal_quality'].get('jitter_percent', 0.0)
            summary['shimmer'] = features['vocal_quality'].get('shimmer_percent', 0.0)

        # F0 statistics
        if 'f0_data' in features:
            f0 = features['f0_data']['f0']
            voiced = features['f0_data']['voiced']
            f0_voiced = f0[voiced]
            if len(f0_voiced) > 0:
                summary['mean_f0'] = float(np.mean(f0_voiced))
                summary['f0_range_st'] = float(12 * np.log2(np.max(f0_voiced) / (np.min(f0_voiced) + 1e-10)))
            else:
                summary['mean_f0'] = 0.0
                summary['f0_range_st'] = 0.0

        return summary
