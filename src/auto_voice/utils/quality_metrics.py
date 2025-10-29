"""
Comprehensive quality evaluation metrics for singing voice conversion.

This module provides objective and subjective quality assessment metrics for voice
conversion systems, including pitch accuracy, speaker similarity, naturalness, and
intelligibility evaluation.
"""

import torch
import numpy as np
import librosa
import scipy.stats
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Import dependencies
try:
    import pystoi
    pystoi_available = True
except ImportError:
    pystoi_available = False
    logger.warning("PySTOI not available. STOI metrics will be disabled.")

try:
    import pesq
    pesq_available = True
except ImportError:
    pesq_available = False
    logger.warning("PESQ not available. PESQ metrics will be disabled.")

try:
    from nisqa.NISQA_model import nisqaModel
    nisqa_available = True
except ImportError:
    nisqa_available = False
    logger.warning("NISQA not available. NISQA-based MOS metrics will be disabled.")

# Import internal dependencies
from ..audio.pitch_extractor import SingingPitchExtractor
from ..audio.processor import AudioProcessor
from ..models.speaker_encoder import SpeakerEncoder


@dataclass
class AudioAlignmentResult:
    """Result of audio alignment operation."""
    source_audio: torch.Tensor
    target_audio: torch.Tensor
    aligned_target: torch.Tensor
    alignment_score: float
    delay_samples: int
    optimal_length: int


class AudioAligner:
    """Advanced audio alignment for comparing source and converted audio."""

    def __init__(self, sample_rate: int = 44100, max_delay_sec: float = 0.2):
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_sec * sample_rate)

    def align_audio(self, source_audio: torch.Tensor, target_audio: torch.Tensor) -> AudioAlignmentResult:
        """
        Align target audio to source audio using cross-correlation.

        Args:
            source_audio: Source audio waveform (channels, samples) or (samples,)
            target_audio: Target/converted audio waveform (channels, samples) or (samples,)

        Returns:
            AudioAlignmentResult: Alignment information and aligned audio
        """
        # Normalize input shapes to (channels, samples) early
        if source_audio.dim() == 3:  # (batch, channels, samples)
            source_audio = source_audio.squeeze(0)
        if source_audio.dim() == 1:  # (samples,) -> (1, samples)
            source_audio = source_audio.unsqueeze(0)

        if target_audio.dim() == 3:  # (batch, channels, samples)
            target_audio = target_audio.squeeze(0)
        if target_audio.dim() == 1:  # (samples,) -> (1, samples)
            target_audio = target_audio.unsqueeze(0)

        # Now both tensors are guaranteed to be (channels, samples)

        # Ensure mono audio for correlation
        source_mono = source_audio.mean(dim=0) if source_audio.shape[0] > 1 else source_audio[0]
        target_mono = target_audio.mean(dim=0) if target_audio.shape[0] > 1 else target_audio[0]

        # Convert to numpy for cross-correlation
        source_np = source_mono.detach().cpu().numpy()
        target_np = target_mono.detach().cpu().numpy()

        # Perform FFT-based cross-correlation (faster for long audio)
        try:
            from scipy.signal import correlate
            correlation = correlate(source_np, target_np, mode='full', method='fft')
        except ImportError:
            # Fallback to numpy if scipy not available
            correlation = np.correlate(source_np, target_np, mode='full')

        # Limit search to max_delay_samples range around zero lag
        center_idx = len(source_np) - 1
        start_idx = max(0, center_idx - self.max_delay_samples)
        end_idx = min(len(correlation), center_idx + self.max_delay_samples + 1)

        # Find maximum correlation within limited range
        search_range = correlation[start_idx:end_idx]
        max_index_in_range = np.argmax(np.abs(search_range))
        max_index = start_idx + max_index_in_range
        delay_samples = max_index - center_idx

        # Clamp delay to reasonable range (should already be within range, but ensure)
        delay_samples = np.clip(delay_samples, -self.max_delay_samples, self.max_delay_samples)

        # Align the target audio
        if delay_samples > 0:
            aligned_target = torch.nn.functional.pad(target_audio, (delay_samples, 0))
            aligned_target = aligned_target[:, :source_audio.shape[-1]]
        else:
            aligned_target = target_audio[:, abs(delay_samples):source_audio.shape[-1] + abs(delay_samples)]
            aligned_target = torch.nn.functional.pad(aligned_target, (abs(delay_samples), 0))

        # Ensure aligned target has the same length as source
        if aligned_target.shape[-1] < source_audio.shape[-1]:
            aligned_target = torch.nn.functional.pad(aligned_target, (0, source_audio.shape[-1] - aligned_target.shape[-1]))
        elif aligned_target.shape[-1] > source_audio.shape[-1]:
            aligned_target = aligned_target[:, :source_audio.shape[-1]]

        # Calculate alignment quality score
        alignment_score = float(np.abs(correlation[max_index]) / np.sqrt(
            np.sum(source_np**2) * np.sum(target_np**2)
        ))

        optimal_length = min(len(source_np), len(target_np))

        return AudioAlignmentResult(
            source_audio=source_audio,
            target_audio=target_audio,
            aligned_target=aligned_target,
            alignment_score=alignment_score,
            delay_samples=int(delay_samples),
            optimal_length=optimal_length
        )


class AudioNormalizer:
    """Audio normalization utilities for consistent evaluation."""

    @staticmethod
    def normalize_audio(audio: torch.Tensor, target_level: float = -12.0) -> torch.Tensor:
        """
        Normalize audio to target RMS level in dBFS.

        Args:
            audio: Input audio waveform
            target_level: Target RMS level in dBFS

        Returns:
            Normalized audio
        """
        # Calculate RMS level
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms == 0:
            return audio

        # Convert target dBFS to linear
        target_linear = 10 ** (target_level / 20.0)

        # Calculate gain
        gain = target_linear / (rms + 1e-8)

        # Apply gain
        normalized = audio * gain

        return normalized

    @staticmethod
    def standardize_audio_length(audio: torch.Tensor, target_samples: Optional[int] = None) -> torch.Tensor:
        """
        Pad or trim audio to standard length.

        Args:
            audio: Input audio
            target_samples: Target sample count

        Returns:
            Standardized length audio
        """
        current_samples = audio.shape[-1]

        if target_samples is None:
            return audio

        if current_samples < target_samples:
            # Pad with zeros
            pad_amount = target_samples - current_samples
            return torch.nn.functional.pad(audio, (0, pad_amount))
        elif current_samples > target_samples:
            # Trim
            return audio[:, :target_samples] if audio.dim() > 1 else audio[:target_samples]
        else:
            return audio


@dataclass
class PitchAccuracyResult:
    """Results from pitch accuracy evaluation."""
    rmse_hz: float  # RMSE in Hz domain
    rmse_log2: float  # RMSE in log2 domain (semitones)
    correlation: float
    voiced_accuracy: float
    octave_errors: int
    pitch_range_error: float
    confidence_score: float
    f0_source: Optional[np.ndarray] = None
    f0_target: Optional[np.ndarray] = None
    # Deprecated field for backwards compatibility
    rmse: Optional[float] = None


class PitchAccuracyMetrics:
    """Evaluates pitch accuracy between source and converted audio."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.pitch_extractor = SingingPitchExtractor(sample_rate=sample_rate)

    def evaluate_pitch_accuracy(self, source_audio: torch.Tensor,
                              target_audio: torch.Tensor) -> PitchAccuracyResult:
        """
        Evaluate pitch accuracy metrics between source and target audio.

        Args:
            source_audio: Source singing voice audio
            target_audio: Converted singing voice audio

        Returns:
            PitchAccuracyResult: Comprehensive pitch accuracy metrics
        """
        # Extract fundamental frequencies using correct method
        f0_source_dict = self.pitch_extractor.extract_f0_contour(
            source_audio, sample_rate=self.sample_rate, return_confidence=True, return_times=True
        )
        f0_target_dict = self.pitch_extractor.extract_f0_contour(
            target_audio, sample_rate=self.sample_rate, return_confidence=True, return_times=True
        )

        # Extract F0 arrays and voiced masks
        f0_source = f0_source_dict['f0']
        f0_target = f0_target_dict['f0']
        voiced_source = f0_source_dict['voiced']
        voiced_target = f0_target_dict['voiced']

        # Ensure arrays are compatible for comparison
        if len(f0_source) != len(f0_target):
            min_len = min(len(f0_source), len(f0_target))
            f0_source = f0_source[:min_len]
            f0_target = f0_target[:min_len]
            voiced_source = voiced_source[:min_len]
            voiced_target = voiced_target[:min_len]

        # Filter voiced regions (use voiced masks from extractor)
        voiced_mask = voiced_source & voiced_target & (f0_source > 0) & (f0_target > 0)
        f0_source_voiced = f0_source[voiced_mask]
        f0_target_voiced = f0_target[voiced_mask]

        if len(f0_source_voiced) == 0:
            return PitchAccuracyResult(
                rmse_hz=0.0, rmse_log2=0.0, rmse=0.0, correlation=0.0, voiced_accuracy=0.0,
                octave_errors=0, pitch_range_error=0.0, confidence_score=0.0
            )

        # Calculate RMSE in Hz domain (as per requirement)
        rmse_hz = np.sqrt(np.mean((f0_target_voiced - f0_source_voiced) ** 2))

        # Calculate RMSE in log F0 domain (melodic accuracy, for reference)
        log_f0_source = np.log2(f0_source_voiced)
        log_f0_target = np.log2(f0_target_voiced)
        rmse_log2 = np.sqrt(np.mean((log_f0_target - log_f0_source) ** 2))

        # Calculate correlation
        correlation = float(scipy.stats.pearsonr(log_f0_source, log_f0_target)[0])
        correlation = correlation if not np.isnan(correlation) else 0.0

        # Voiced accuracy (percentage of frames where F0 deviations are less than quarter-tone)
        f0_ratios = f0_target_voiced / (f0_source_voiced + 1e-8)
        quarter_tone_deviation = np.abs(np.log2(f0_ratios)) <= 0.25  # quarter tone = 2^(0.25)
        voiced_accuracy = np.mean(quarter_tone_deviation)

        # Count octave errors (coarse pitch errors)
        octave_errors = np.sum(np.abs(np.log2(f0_ratios)) >= 1.0)

        # Pitch range error (difference in F0 range)
        range_source = np.ptp(f0_source_voiced)  # peak-to-peak
        range_target = np.ptp(f0_target_voiced)
        pitch_range_error = abs(range_target - range_source)

        # Confidence score based on multiple metrics
        # Normalized Hz RMSE: lower is better, typical range 0-50 Hz
        normalized_hz_rmse = max(0.0, 1.0 - rmse_hz / 50.0)
        confidence_score = (
            (voiced_accuracy * 0.5) +
            (normalized_hz_rmse * 0.3) +
            ((correlation + 1.0) / 2.0 * 0.2)
        )

        return PitchAccuracyResult(
            rmse_hz=float(rmse_hz),
            rmse_log2=float(rmse_log2),
            rmse=float(rmse_log2),  # Backwards compatibility
            correlation=correlation,
            voiced_accuracy=float(voiced_accuracy),
            octave_errors=int(octave_errors),
            pitch_range_error=float(pitch_range_error),
            confidence_score=float(confidence_score),
            f0_source=f0_source,
            f0_target=f0_target
        )

    def calculate_pitch_accuracy(self, f0_source: np.ndarray, f0_target: np.ndarray,
                                sample_rate: int = 44100) -> PitchAccuracyResult:
        """
        Backward compatibility wrapper for evaluate_pitch_accuracy.
        Accepts pre-extracted F0 arrays instead of audio tensors.

        Args:
            f0_source: Source F0 contour (Hz)
            f0_target: Target F0 contour (Hz)
            sample_rate: Audio sample rate

        Returns:
            PitchAccuracyResult: Pitch accuracy metrics
        """
        # Filter voiced regions (f0 > 0)
        voiced_mask = (f0_source > 0) & (f0_target > 0)
        f0_source_voiced = f0_source[voiced_mask]
        f0_target_voiced = f0_target[voiced_mask]

        if len(f0_source_voiced) == 0:
            return PitchAccuracyResult(
                rmse_hz=0.0, rmse_log2=0.0, rmse=0.0, correlation=0.0,
                voiced_accuracy=0.0, octave_errors=0, pitch_range_error=0.0,
                confidence_score=0.0, f0_source=f0_source, f0_target=f0_target
            )

        # Calculate RMSE in Hz
        rmse_hz = np.sqrt(np.mean((f0_source_voiced - f0_target_voiced) ** 2))

        # Calculate correlation
        if len(f0_source_voiced) > 1:
            correlation = np.corrcoef(f0_source_voiced, f0_target_voiced)[0, 1]
        else:
            correlation = 1.0

        return PitchAccuracyResult(
            rmse_hz=rmse_hz,
            rmse_log2=0.0,  # Not computed for raw arrays
            rmse=rmse_hz,
            correlation=correlation,
            voiced_accuracy=1.0,  # Not computed for raw arrays
            octave_errors=0,  # Not computed for raw arrays
            pitch_range_error=0.0,  # Not computed for raw arrays
            confidence_score=correlation,
            f0_source=f0_source,
            f0_target=f0_target
        )


@dataclass
class SpeakerSimilarityResult:
    """Results from speaker similarity evaluation."""
    cosine_similarity: float
    embedding_distance: float
    confidence_score: float
    source_embedding: Optional[np.ndarray] = None
    target_embedding: Optional[np.ndarray] = None


class SpeakerSimilarityMetrics:
    """Evaluates speaker similarity between converted audio and target speaker profile."""

    def __init__(self):
        self.speaker_encoder = SpeakerEncoder()
        self.audio_processor = AudioProcessor()

    def _extract_embedding(self, audio: torch.Tensor) -> np.ndarray:
        """
        Extract speaker embedding from audio, ensuring mono input.

        Args:
            audio: Audio tensor (mono or stereo)

        Returns:
            Speaker embedding as numpy array
        """
        # Ensure audio is mono for embedding extraction
        mono_audio = audio.mean(dim=0) if audio.dim() > 1 else audio
        return self.speaker_encoder.extract_embedding(mono_audio)

    def evaluate_speaker_similarity(self, converted_audio: torch.Tensor,
                                  target_speaker_embedding: Optional[np.ndarray] = None,
                                  target_audio: Optional[torch.Tensor] = None) -> SpeakerSimilarityResult:
        """
        Evaluate speaker similarity using embeddings.

        Compares converted audio against target speaker profile embedding (if provided)
        or target audio embedding (as fallback).

        Args:
            converted_audio: Converted speaker audio to evaluate
            target_speaker_embedding: Target speaker profile embedding (preferred)
            target_audio: Target speaker audio (fallback if embedding not provided)

        Returns:
            SpeakerSimilarityResult: Speaker similarity metrics
        """
        try:
            # Extract converted audio embedding
            converted_embedding = self._extract_embedding(converted_audio)

            # Get target embedding
            if target_speaker_embedding is not None:
                # Use provided target speaker profile embedding
                target_embedding = target_speaker_embedding
            elif target_audio is not None:
                # Fallback: extract embedding from target audio
                target_embedding = self._extract_embedding(target_audio)
            else:
                raise ValueError("Either target_speaker_embedding or target_audio must be provided")

            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                torch.FloatTensor(converted_embedding),
                torch.FloatTensor(target_embedding),
                dim=0
            ).item()

            # Calculate Euclidean distance
            embedding_dist = float(np.linalg.norm(converted_embedding - target_embedding))

            # Confidence score based on similarity
            confidence_score = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

            return SpeakerSimilarityResult(
                cosine_similarity=cos_sim,
                embedding_distance=embedding_dist,
                confidence_score=confidence_score,
                source_embedding=converted_embedding,
                target_embedding=target_embedding
            )

        except Exception as e:
            logger.warning(f"Speaker similarity evaluation failed: {e}")
            return SpeakerSimilarityResult(
                cosine_similarity=0.0, embedding_distance=1e6, confidence_score=0.0
            )

    def compute_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding

        Returns:
            Cosine similarity score
        """
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.FloatTensor(embedding1),
            torch.FloatTensor(embedding2),
            dim=0
        ).item()
        return cos_sim

    def calculate_similarity(self, source_audio: torch.Tensor, target_audio: torch.Tensor,
                           sample_rate: int = 44100) -> SpeakerSimilarityResult:
        """
        Backward compatibility wrapper for evaluate_speaker_similarity.

        Args:
            source_audio: Source/converted audio tensor
            target_audio: Target audio tensor to compare against
            sample_rate: Audio sample rate (unused, kept for compatibility)

        Returns:
            SpeakerSimilarityResult: Speaker similarity metrics
        """
        return self.evaluate_speaker_similarity(
            converted_audio=source_audio,
            target_audio=target_audio
        )


@dataclass
class NaturalnessResult:
    """Results from naturalness evaluation."""
    spectral_distortion: float
    harmonic_to_noise: float
    mos_estimation: float
    confidence_score: float
    spectrogram_source: Optional[np.ndarray] = None
    spectrogram_target: Optional[np.ndarray] = None
    mos_method: str = 'heuristic'  # Method used: 'heuristic', 'nisqa', 'both'
    mos_nisqa: Optional[float] = None  # NISQA MOS score if available
    mos_heuristic: Optional[float] = None  # Heuristic MOS score if available
    mcd: Optional[float] = None  # Mel-Cepstral Distortion if computed


class NaturalnessMetrics:
    """Evaluates naturalness and audio quality."""

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512,
                 mos_method: str = 'heuristic', compute_mcd: bool = True):
        """
        Initialize naturalness metrics evaluator.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size for spectrogram analysis
            hop_length: Hop length for STFT
            mos_method: Method for MOS calculation. Options:
                - 'heuristic': Use spectral distortion-based heuristic (default)
                - 'nisqa': Use NISQA model for MOS prediction
                - 'both': Calculate both heuristic and NISQA scores
            compute_mcd: Whether to compute Mel-Cepstral Distortion (default: True)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mos_method = mos_method
        self.compute_mcd = compute_mcd
        self.nisqa_model = None

        # Load NISQA model if requested and available
        if mos_method in ['nisqa', 'both']:
            if nisqa_available:
                try:
                    logger.info("Loading NISQA model for MOS prediction...")
                    self.nisqa_model = nisqaModel()
                    logger.info(f"NISQA model loaded successfully. Using method: {mos_method}")
                except Exception as e:
                    logger.warning(f"Failed to load NISQA model: {e}. Falling back to heuristic method.")
                    self.mos_method = 'heuristic'
            else:
                logger.warning(f"NISQA not available. Falling back to heuristic method from: {mos_method}")
                self.mos_method = 'heuristic'

    def compute_mel_cepstral_distortion(self, source_audio: np.ndarray, target_audio: np.ndarray) -> float:
        """
        Compute Mel-Cepstral Distortion (MCD) between source and target audio.

        MCD measures the spectral distance between two audio signals using mel-frequency
        cepstral coefficients (MFCCs). Lower values indicate better similarity.

        Args:
            source_audio: Source audio waveform
            target_audio: Target/converted audio waveform

        Returns:
            MCD value in dB (typical range: 4-10 dB, lower is better)
        """
        try:
            # Extract MFCCs using librosa
            n_mfcc = 13  # Standard number of MFCC coefficients

            # Compute MFCCs (excluding 0th coefficient which represents energy)
            mfcc_source = librosa.feature.mfcc(
                y=source_audio, sr=self.sample_rate, n_mfcc=n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )[1:, :]  # Exclude 0th coefficient

            mfcc_target = librosa.feature.mfcc(
                y=target_audio, sr=self.sample_rate, n_mfcc=n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )[1:, :]  # Exclude 0th coefficient

            # Align frame counts
            min_frames = min(mfcc_source.shape[1], mfcc_target.shape[1])
            mfcc_source = mfcc_source[:, :min_frames]
            mfcc_target = mfcc_target[:, :min_frames]

            # Compute MCD using the standard formula:
            # MCD = (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))
            # where c1, c2 are MFCC vectors

            diff = mfcc_source - mfcc_target
            squared_diff = np.sum(diff ** 2, axis=0)  # Sum over coefficients
            mcd_frames = np.sqrt(2.0 * squared_diff)  # Per-frame MCD

            # Convert to dB scale
            mcd_db = (10.0 / np.log(10.0)) * np.mean(mcd_frames)

            return float(mcd_db)

        except Exception as e:
            logger.warning(f"MCD computation failed: {e}")
            return 0.0

    def evaluate_naturalness(self, source_audio: torch.Tensor,
                           target_audio: torch.Tensor) -> NaturalnessResult:
        """
        Evaluate naturalness using spectral analysis and MOS estimation.

        Args:
            source_audio: Source audio
            target_audio: Converted audio

        Returns:
            NaturalnessResult: Naturalness evaluation results
        """
        # Convert to numpy for librosa operations
        source_np = source_audio.detach().cpu().numpy()
        target_np = target_audio.detach().cpu().numpy()

        if source_np.ndim > 1:
            source_np = source_np.squeeze()
        if target_np.ndim > 1:
            target_np = target_np.squeeze()

        # Ensure same length
        min_len = min(len(source_np), len(target_np))
        source_np = source_np[:min_len]
        target_np = target_np[:min_len]

        try:
            # Spectral analysis
            D_source = librosa.stft(source_np, n_fft=self.n_fft, hop_length=self.hop_length)
            D_target = librosa.stft(target_np, n_fft=self.n_fft, hop_length=self.hop_length)

            # Magnitude spectrograms
            S_source = np.abs(D_source)
            S_target = np.abs(D_target)

            # Spectral distortion (log-magnitude difference) - use amplitude_to_db for magnitude
            spec_distortion = np.mean(np.abs(librosa.amplitude_to_db(S_source) - librosa.amplitude_to_db(S_target)))

            # Harmonic-to-noise ratio (simplified approximation)
            harmonic_noise = 10.0 - spec_distortion / 10.0  # Rough approximation

            # MOS estimation - method depends on configuration
            mos_heuristic = None
            mos_nisqa = None
            mos_estimate = 1.0

            # Calculate heuristic MOS if needed
            if self.mos_method in ['heuristic', 'both']:
                mos_heuristic = max(1.0, min(5.0, 5.0 - spec_distortion / 20.0))
                mos_estimate = mos_heuristic
                logger.debug(f"Heuristic MOS: {mos_heuristic:.2f}")

            # Calculate NISQA MOS if model available
            if self.mos_method in ['nisqa', 'both'] and self.nisqa_model is not None:
                try:
                    # NISQA expects 48kHz audio
                    if self.sample_rate != 48000:
                        target_nisqa = librosa.resample(target_np, orig_sr=self.sample_rate, target_sr=48000)
                    else:
                        target_nisqa = target_np

                    # NISQA prediction (expects dict with audio and sample rate)
                    nisqa_input = {
                        'audio': target_nisqa,
                        'sr': 48000
                    }
                    mos_nisqa = self.nisqa_model.predict(nisqa_input)['mos']
                    mos_estimate = mos_nisqa
                    logger.debug(f"NISQA MOS: {mos_nisqa:.2f}")
                except Exception as e:
                    logger.warning(f"NISQA MOS prediction failed: {e}. Using heuristic fallback.")
                    if mos_heuristic is not None:
                        mos_estimate = mos_heuristic
                    else:
                        mos_heuristic = max(1.0, min(5.0, 5.0 - spec_distortion / 20.0))
                        mos_estimate = mos_heuristic

            # If 'both' method, use average or prefer NISQA
            if self.mos_method == 'both' and mos_heuristic is not None and mos_nisqa is not None:
                mos_estimate = mos_nisqa  # Prefer NISQA when available
                logger.debug(f"Using NISQA MOS: {mos_nisqa:.2f} (heuristic: {mos_heuristic:.2f})")

            # Compute MCD if requested
            mcd_value = None
            if self.compute_mcd:
                mcd_value = self.compute_mel_cepstral_distortion(source_np, target_np)
                logger.debug(f"MCD: {mcd_value:.2f} dB")

            # Confidence based on signal characteristics and method
            base_confidence = 1.0 if spec_distortion < 10.0 else max(0.1, 1.0 - spec_distortion / 100.0)
            # Boost confidence if NISQA is used (more reliable)
            if self.mos_method == 'nisqa' and mos_nisqa is not None:
                confidence_score = min(1.0, base_confidence * 1.2)
            else:
                confidence_score = base_confidence

            return NaturalnessResult(
                spectral_distortion=float(spec_distortion),
                harmonic_to_noise=float(harmonic_noise),
                mos_estimation=float(mos_estimate),
                confidence_score=float(confidence_score),
                spectrogram_source=S_source,
                spectrogram_target=S_target,
                mos_method=self.mos_method,
                mos_nisqa=float(mos_nisqa) if mos_nisqa is not None else None,
                mos_heuristic=float(mos_heuristic) if mos_heuristic is not None else None,
                mcd=mcd_value
            )

        except Exception as e:
            logger.warning(f"Naturalness evaluation failed: {e}")
            return NaturalnessResult(
                spectral_distortion=1e6, harmonic_to_noise=0.0,
                mos_estimation=1.0, confidence_score=0.0,
                mos_method=self.mos_method, mcd=None
            )


@dataclass
class IntelligibilityResult:
    """Results from intelligibility evaluation."""
    stoi_score: float
    estoi_score: float
    pesq_score: float
    confidence_score: float
    stoi_available: bool = pystoi_available
    pesq_available: bool = pesq_available


class IntelligibilityMetrics:
    """Evaluates speech intelligibility using STOI and PESQ metrics."""

    def __init__(self, sample_rate: int = 16000, pesq_mode: str = 'wb', stoi_sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.pesq_mode = pesq_mode
        self.stoi_sample_rate = stoi_sample_rate  # STOI requires specific sample rates

    def evaluate_intelligibility(self, source_audio: torch.Tensor,
                               target_audio: torch.Tensor) -> IntelligibilityResult:
        """
        Evaluate speech intelligibility using STOI and PESQ.

        Args:
            source_audio: Clean source audio
            target_audio: Processed/noisy audio

        Returns:
            IntelligibilityResult: Intelligibility evaluation results
        """
        stoi_score = 0.0
        estoi_score = 0.0
        pesq_score = 0.0

        # Convert to numpy and ensure proper format
        source_np = source_audio.detach().cpu().numpy().astype(np.float32)
        target_np = target_audio.detach().cpu().numpy().astype(np.float32)

        if source_np.ndim > 1:
            source_np = source_np.squeeze()
        if target_np.ndim > 1:
            target_np = target_np.squeeze()

        # Ensure same length
        min_len = min(len(source_np), len(target_np))
        source_np = source_np[:min_len]
        target_np = target_np[:min_len]

        # Evaluate STOI - resample to supported rate
        if pystoi_available:
            try:
                # Resample to STOI-supported sample rate if needed
                if self.sample_rate != self.stoi_sample_rate:
                    source_stoi = librosa.resample(source_np, orig_sr=self.sample_rate, target_sr=self.stoi_sample_rate)
                    target_stoi = librosa.resample(target_np, orig_sr=self.sample_rate, target_sr=self.stoi_sample_rate)
                else:
                    source_stoi = source_np
                    target_stoi = target_np

                stoi_score = pystoi.stoi(source_stoi, target_stoi, fs_sig=self.stoi_sample_rate, extended=False)
                estoi_score = pystoi.stoi(source_stoi, target_stoi, fs_sig=self.stoi_sample_rate, extended=True)
            except Exception as e:
                logger.warning(f"STOI evaluation failed: {e}")

        # Evaluate PESQ
        if pesq_available:
            try:
                # PESQ requires 16kHz, mono audio
                if self.sample_rate != 16000:
                    source_resampled = librosa.resample(source_np, orig_sr=self.sample_rate, target_sr=16000)
                    target_resampled = librosa.resample(target_np, orig_sr=self.sample_rate, target_sr=16000)
                else:
                    source_resampled = source_np
                    target_resampled = target_np

                pesq_score = pesq.pesq(16000, source_resampled, target_resampled, self.pesq_mode)
            except Exception as e:
                logger.warning(f"PESQ evaluation failed: {e}")

        # Calculate confidence based on metric availability and validity
        metrics_available = sum([pystoi_available, pesq_available])
        confidence_score = min(1.0, metrics_available * 0.5)

        return IntelligibilityResult(
            stoi_score=stoi_score,
            estoi_score=estoi_score,
            pesq_score=pesq_score,
            confidence_score=confidence_score
        )


@dataclass
class QualityMetricsResult:
    """Comprehensive quality evaluation result."""
    pitch_accuracy: PitchAccuracyResult
    speaker_similarity: SpeakerSimilarityResult
    naturalness: NaturalnessResult
    intelligibility: IntelligibilityResult
    overall_quality_score: float
    evaluation_timestamp: float
    processing_time_seconds: float


class QualityMetricsAggregator:
    """Aggregates multiple quality metrics into a comprehensive evaluation."""

    def __init__(self, sample_rate: int = 44100, mos_method: str = 'heuristic', compute_mcd: bool = True):
        """
        Initialize quality metrics aggregator.

        Args:
            sample_rate: Audio sample rate
            mos_method: MOS calculation method for naturalness metrics.
                       Options: 'heuristic', 'nisqa', 'both'
            compute_mcd: Whether to compute Mel-Cepstral Distortion
        """
        self.sample_rate = sample_rate
        self.aligner = AudioAligner(sample_rate=sample_rate)
        self.normalizer = AudioNormalizer()

        # Initialize metric evaluators
        self.pitch_metrics = PitchAccuracyMetrics(sample_rate=sample_rate)
        self.speaker_metrics = SpeakerSimilarityMetrics()
        self.naturalness_metrics = NaturalnessMetrics(sample_rate=sample_rate, mos_method=mos_method, compute_mcd=compute_mcd)
        self.intelligibility_metrics = IntelligibilityMetrics(sample_rate=sample_rate)

    def evaluate(self, source_audio: torch.Tensor, target_audio: torch.Tensor,
                align_audio: bool = True, target_speaker_embedding: Optional[np.ndarray] = None) -> QualityMetricsResult:
        """
        Perform comprehensive quality evaluation.

        Args:
            source_audio: Source audio waveform
            target_audio: Converted/target audio waveform
            align_audio: Whether to align audio for comparison
            target_speaker_embedding: Optional target speaker profile embedding

        Returns:
            QualityMetricsResult: Comprehensive evaluation results
        """
        import time
        start_time = time.time()

        # Prepare audio for evaluation
        if align_audio:
            alignment_result = self.aligner.align_audio(source_audio, target_audio)
            source_eval = alignment_result.source_audio
            target_eval = alignment_result.aligned_target
        else:
            source_eval = source_audio
            target_eval = target_audio

        # Normalize audio levels
        source_eval = self.normalizer.normalize_audio(source_eval)
        target_eval = self.normalizer.normalize_audio(target_eval)

        # Evaluate each metric category
        pitch_result = self.pitch_metrics.evaluate_pitch_accuracy(source_eval, target_eval)
        speaker_result = self.speaker_metrics.evaluate_speaker_similarity(
            target_eval, target_speaker_embedding=target_speaker_embedding
        )
        naturalness_result = self.naturalness_metrics.evaluate_naturalness(source_eval, target_eval)
        intelligibility_result = self.intelligibility_metrics.evaluate_intelligibility(source_eval, target_eval)

        # Calculate overall quality score using normalized metric values
        # Normalize each metric to [0, 1] range
        normalized_pitch = max(0.0, 1.0 - pitch_result.rmse_hz / 50.0)  # Typical range 0-50 Hz
        normalized_speaker = (speaker_result.cosine_similarity + 1.0) / 2.0  # From [-1, 1] to [0, 1]
        normalized_naturalness = max(0.0, 1.0 - naturalness_result.spectral_distortion / 100.0)  # Typical range 0-100 dB
        normalized_intelligibility = (intelligibility_result.stoi_score + intelligibility_result.estoi_score) / 2.0  # Already [0, 1]

        # Weighted combination
        overall_score = (
            normalized_pitch * 0.3 +           # 30% weight on pitch accuracy
            normalized_speaker * 0.3 +         # 30% weight on speaker similarity
            normalized_naturalness * 0.25 +    # 25% weight on naturalness
            normalized_intelligibility * 0.15  # 15% weight on intelligibility
        )

        processing_time = time.time() - start_time

        return QualityMetricsResult(
            pitch_accuracy=pitch_result,
            speaker_similarity=speaker_result,
            naturalness=naturalness_result,
            intelligibility=intelligibility_result,
            overall_quality_score=float(overall_score),
            evaluation_timestamp=time.time(),
            processing_time_seconds=processing_time
        )

    def evaluate_batch(self, source_audios: List[torch.Tensor],
                      target_audios: List[torch.Tensor],
                      align_audio: bool = True) -> List[QualityMetricsResult]:
        """
        Evaluate quality for a batch of audio pairs.

        Args:
            source_audios: List of source audio waveforms
            target_audios: List of target audio waveforms
            align_audio: Whether to align audio for comparison

        Returns:
            List of comprehensive evaluation results
        """
        results = []
        for source, target in zip(source_audios, target_audios):
            result = self.evaluate(source, target, align_audio)
            results.append(result)

        return results

    def get_summary_statistics(self, results: List[QualityMetricsResult]) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across multiple evaluation results.

        Args:
            results: List of quality evaluation results

        Returns:
            Dictionary of summary statistics by metric category
        """
        if not results:
            return {}

        # Extract metric values
        pitch_rmses_hz = [r.pitch_accuracy.rmse_hz for r in results]
        pitch_rmses_log2 = [r.pitch_accuracy.rmse_log2 for r in results]
        pitch_correlations = [r.pitch_accuracy.correlation for r in results]
        speaker_similarities = [r.speaker_similarity.cosine_similarity for r in results]
        spectral_distortions = [r.naturalness.spectral_distortion for r in results]
        stoi_scores = [r.intelligibility.stoi_score for r in results]
        pesq_scores = [r.intelligibility.pesq_score for r in results]
        overall_scores = [r.overall_quality_score for r in results]

        def compute_stats(values: List[float]) -> Dict[str, float]:
            values_np = np.array(values)
            return {
                'mean': float(np.mean(values_np)),
                'std': float(np.std(values_np)),
                'min': float(np.min(values_np)),
                'max': float(np.max(values_np)),
                'median': float(np.median(values_np))
            }

        return {
            'pitch_accuracy': {
                'rmse_hz': compute_stats(pitch_rmses_hz),
                'rmse_log2': compute_stats(pitch_rmses_log2),
                'correlation': compute_stats(pitch_correlations)
            },
            'speaker_similarity': {
                'cosine_similarity': compute_stats(speaker_similarities)
            },
            'naturalness': {
                'spectral_distortion': compute_stats(spectral_distortions)
            },
            'intelligibility': {
                'stoi': compute_stats(stoi_scores),
                'pesq': compute_stats(pesq_scores)
            },
            'overall': {
                'quality_score': compute_stats(overall_scores)
            }
        }
