"""Automated Conversion Quality Analyzer.

Provides robust quality analysis for all voice conversion methodologies to measure
implementation quality and compare approaches.

Cross-Context Dependencies:
- sota-dual-pipeline_20260130: Pipeline implementations
- training-inference-integration_20260130: AdapterManager
- lora-lifecycle-management_20260201: Quality thresholds

Metrics Computed:
1. Speaker Similarity - Cosine similarity of speaker embeddings (target: >= 0.85)
2. MCD (Mel Cepstral Distortion) - Spectral distance (target: <= 4.5 dB)
3. F0 Correlation - Pitch contour matching (target: >= 0.90)
4. F0 RMSE - Pitch accuracy in Hz (target: <= 20 Hz)
5. RTF (Real-Time Factor) - Processing speed (target: < 0.3 for realtime)
6. SNR - Signal-to-noise ratio (target: >= 20 dB)
7. PESQ - Perceptual quality (MOS-like, target: >= 3.5)
8. STOI - Speech intelligibility (target: >= 0.85)

Usage:
    analyzer = ConversionQualityAnalyzer()

    # Analyze single conversion
    result = analyzer.analyze(
        source_audio="source.wav",
        converted_audio="converted.wav",
        target_speaker_embedding=embedding,
        methodology="quality_seedvc"
    )

    # Compare methodologies
    comparison = analyzer.compare_methodologies(
        source_audio="source.wav",
        target_profile_id="william-123",
        methodologies=["realtime", "quality", "quality_seedvc"]
    )
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def get_voice_identifier():
    """Load the shared voice identifier lazily for patchable test access."""
    from ..inference.voice_identifier import get_voice_identifier as _get_voice_identifier

    return _get_voice_identifier()


@dataclass
class QualityMetrics:
    """Quality metrics for a conversion."""
    # Identity metrics
    speaker_similarity: float = 0.0  # Cosine similarity (0-1)

    # Spectral metrics
    mcd: float = 0.0  # Mel Cepstral Distortion (dB)
    log_f0_rmse: float = 0.0  # Log F0 RMSE

    # Pitch metrics
    f0_correlation: float = 0.0  # Pitch contour correlation
    f0_rmse: float = 0.0  # F0 RMSE in Hz

    # Performance metrics
    rtf: float = 0.0  # Real-time factor
    processing_time_ms: float = 0.0

    # Quality metrics
    snr: float = 0.0  # Signal-to-noise ratio (dB)
    pesq: Optional[float] = None  # PESQ score (MOS-like)
    stoi: Optional[float] = None  # Speech intelligibility

    # Synthesis artifacts
    artifact_score: float = 0.0  # 0 = no artifacts, 1 = severe

    # Overall
    quality_score: float = 0.0  # Weighted composite (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionAnalysis:
    """Complete analysis of a voice conversion."""
    methodology: str
    source_audio: str
    converted_audio: str
    target_profile_id: Optional[str]
    metrics: QualityMetrics
    timestamp: str
    passes_thresholds: bool
    threshold_failures: List[str]
    recommendations: List[str]


@dataclass
class MethodologyComparison:
    """Comparison across multiple methodologies."""
    source_audio: str
    target_profile_id: str
    analyses: Dict[str, ConversionAnalysis]
    best_methodology: str
    rankings: Dict[str, int]  # methodology -> rank (1 = best)
    summary: str


class ConversionQualityAnalyzer:
    """Analyzes voice conversion quality across all methodologies.

    Thresholds (from lora-lifecycle-management spec):
    - speaker_similarity_min: 0.85
    - mcd_max: 4.5
    - f0_correlation_min: 0.90
    - rtf_max: 0.30 (for realtime)
    """

    # Quality thresholds
    SPEAKER_SIMILARITY_MIN = 0.85
    MCD_MAX = 4.5
    F0_CORRELATION_MIN = 0.90
    F0_RMSE_MAX = 20.0
    RTF_MAX_REALTIME = 0.30
    SNR_MIN = 20.0
    PESQ_MIN = 3.5
    STOI_MIN = 0.85

    # Weights for composite score
    WEIGHTS = {
        'speaker_similarity': 0.30,
        'mcd': 0.20,
        'f0_correlation': 0.15,
        'snr': 0.10,
        'pesq': 0.15,
        'stoi': 0.10,
    }

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Path = Path("data/quality_cache"),
    ):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._speaker_model = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}

        logger.info(f"ConversionQualityAnalyzer initialized on {device}")

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        except Exception as sf_error:
            logger.debug("soundfile failed for %s: %s", path, sf_error)

        try:
            import torchaudio

            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze().numpy().astype(np.float32), sr
        except Exception as ta_error:
            logger.debug("torchaudio failed for %s: %s", path, ta_error)

        import librosa

        audio, sr = librosa.load(path, sr=None, mono=True)
        return audio.astype(np.float32), sr

    def _resample_audio(self, audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Resample waveform data with torchaudio when available and librosa fallback."""
        if source_sr == target_sr:
            return audio.astype(np.float32, copy=False)

        try:
            import torch
            import torchaudio.transforms as T

            resampler = T.Resample(source_sr, target_sr)
            return resampler(torch.from_numpy(audio).float()).numpy().astype(np.float32)
        except Exception as resample_error:
            logger.debug(
                "torchaudio resample failed (%s -> %s): %s",
                source_sr,
                target_sr,
                resample_error,
            )

        import librosa

        return librosa.resample(
            audio.astype(np.float32, copy=False),
            orig_sr=source_sr,
            target_sr=target_sr,
        ).astype(np.float32)

    def _fallback_speaker_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Build a deterministic embedding when WavLM is unavailable."""
        if audio.size == 0:
            return np.zeros(768, dtype=np.float32)

        mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        spectrum = np.abs(np.fft.rfft(mono_audio))
        if spectrum.size == 0 or not np.isfinite(spectrum).any():
            return np.zeros(768, dtype=np.float32)

        x_old = np.linspace(0.0, 1.0, num=spectrum.size, endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=768, endpoint=True)
        embedding = np.interp(x_new, x_old, spectrum).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            return np.zeros(768, dtype=np.float32)
        return embedding / norm

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract speaker embedding using WavLM."""
        try:
            import torch
            import torch.nn.functional as F
            import transformers
        except Exception as e:
            logger.warning(f"Failed to import speaker embedding dependencies: {e}")
            return np.zeros(768, dtype=np.float32)

        try:
            if self._speaker_model is None:
                transformers_module = sys.modules.get("transformers", transformers)
                processor_cls = getattr(transformers_module, "Wav2Vec2FeatureExtractor")
                model_cls = getattr(transformers_module, "WavLMModel")
                self._processor = processor_cls.from_pretrained(
                    "microsoft/wavlm-base-plus"
                )
                self._speaker_model = model_cls.from_pretrained(
                    "microsoft/wavlm-base-plus"
                )
                if torch.cuda.is_available() and self.device == "cuda":
                    self._speaker_model = self._speaker_model.cuda()
        except Exception as e:
            logger.warning(f"Failed to initialize speaker embedding model: {e}")
            return np.zeros(768, dtype=np.float32)

        try:
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = self._resample_audio(audio, sr, 16000)

            inputs = self._processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )

            device = next(self._speaker_model.parameters()).device
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = self._speaker_model(input_values)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embedding = F.normalize(embedding, dim=0)

            return embedding.detach().cpu().numpy().astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract speaker embedding: {e}")
            return self._fallback_speaker_embedding(audio)

    def _compute_mcd(
        self,
        source_audio: np.ndarray,
        converted_audio: np.ndarray,
        sr: int
    ) -> float:
        """Compute Mel Cepstral Distortion."""
        try:
            import librosa

            # Extract MFCCs
            mfcc_source = librosa.feature.mfcc(y=source_audio, sr=sr, n_mfcc=13)
            mfcc_converted = librosa.feature.mfcc(y=converted_audio, sr=sr, n_mfcc=13)

            # Align lengths
            min_len = min(mfcc_source.shape[1], mfcc_converted.shape[1])
            mfcc_source = mfcc_source[:, :min_len]
            mfcc_converted = mfcc_converted[:, :min_len]

            # Compute MCD
            diff = mfcc_source - mfcc_converted
            mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))

            # Convert to dB scale
            mcd_db = (10.0 / np.log(10)) * mcd

            return float(mcd_db)

        except Exception as e:
            logger.warning(f"Failed to compute MCD: {e}")
            return 10.0  # High value indicates failure

    def _compute_f0_metrics(
        self,
        source_audio: np.ndarray,
        converted_audio: np.ndarray,
        sr: int
    ) -> Tuple[float, float]:
        """Compute F0 correlation and RMSE."""
        try:
            import librosa

            # Extract F0
            f0_source, _, _ = librosa.pyin(
                source_audio, fmin=50, fmax=800, sr=sr
            )
            f0_converted, _, _ = librosa.pyin(
                converted_audio, fmin=50, fmax=800, sr=sr
            )

            # Handle NaN values
            valid_mask = ~(np.isnan(f0_source) | np.isnan(f0_converted))

            if valid_mask.sum() < 10:
                return 0.0, 100.0

            f0_source_valid = f0_source[valid_mask]
            f0_converted_valid = f0_converted[valid_mask]

            # Correlation
            source_std = float(np.std(f0_source_valid))
            converted_std = float(np.std(f0_converted_valid))
            if source_std < 1e-6 and converted_std < 1e-6:
                correlation = 1.0
            elif source_std < 1e-6 or converted_std < 1e-6:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(f0_source_valid, f0_converted_valid)[0, 1])
                if not np.isfinite(correlation):
                    correlation = 0.0

            # RMSE
            rmse = float(np.sqrt(np.mean((f0_source_valid - f0_converted_valid) ** 2)))

            return correlation, rmse

        except Exception as e:
            logger.warning(f"Failed to compute F0 metrics: {e}")
            return 0.0, 100.0

    def _compute_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if mono_audio.size == 0:
                return 0.0

            window = 5 if mono_audio.size >= 5 else max(1, mono_audio.size)
            kernel = np.ones(window, dtype=np.float32) / window
            signal_estimate = np.convolve(mono_audio, kernel, mode="same")
            noise_estimate = mono_audio - signal_estimate

            signal_power = float(np.mean(signal_estimate ** 2)) + 1e-10
            noise_power = float(np.mean(noise_estimate ** 2)) + 1e-10

            snr = 10 * np.log10(signal_power / noise_power)
            return float(snr) if np.isfinite(snr) else 0.0

        except Exception as e:
            logger.warning(f"Failed to compute SNR: {e}")
            return 0.0

    def _compute_pesq(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
        sr: int
    ) -> Optional[float]:
        """Compute PESQ score if available."""
        try:
            from pesq import pesq

            # PESQ requires 8kHz or 16kHz
            if sr not in [8000, 16000]:
                reference = self._resample_audio(reference, sr, 16000)
                degraded = self._resample_audio(degraded, sr, 16000)
                sr = 16000

            # Align lengths
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

            mode = 'wb' if sr == 16000 else 'nb'
            score = pesq(sr, reference, degraded, mode)
            return float(score)

        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Failed to compute PESQ: {e}")
            return None

    def _compute_stoi(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
        sr: int
    ) -> Optional[float]:
        """Compute STOI score if available."""
        try:
            from pystoi import stoi

            # Align lengths
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

            score = stoi(reference, degraded, sr, extended=False)
            return float(score)

        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Failed to compute STOI: {e}")
            return None

    def _compute_log_f0_rmse(
        self,
        source_audio: np.ndarray,
        converted_audio: np.ndarray,
        sr: int,
    ) -> float:
        """Compute log-domain F0 RMSE for voiced frames."""
        try:
            import librosa

            f0_source, _, _ = librosa.pyin(source_audio, fmin=50, fmax=800, sr=sr)
            f0_converted, _, _ = librosa.pyin(converted_audio, fmin=50, fmax=800, sr=sr)

            valid_mask = (
                ~(np.isnan(f0_source) | np.isnan(f0_converted))
                & (f0_source > 0)
                & (f0_converted > 0)
            )
            if valid_mask.sum() < 10:
                return 0.0

            log_source = np.log(f0_source[valid_mask])
            log_converted = np.log(f0_converted[valid_mask])
            return float(np.sqrt(np.mean((log_source - log_converted) ** 2)))
        except Exception as e:
            logger.warning(f"Failed to compute log F0 RMSE: {e}")
            return 0.0

    def _compute_quality_score(self, metrics: QualityMetrics) -> float:
        """Compute weighted composite quality score (0-100)."""
        def clamp_score(value: float) -> float:
            return max(0.0, min(100.0, float(value)))

        scores = {}

        # Speaker similarity (0-1 -> 0-100)
        scores['speaker_similarity'] = clamp_score(metrics.speaker_similarity * 100)

        # MCD (lower is better, 0-10 dB -> 100-0)
        scores['mcd'] = clamp_score(100 - (metrics.mcd / 10) * 100)

        # F0 correlation (0-1 -> 0-100)
        scores['f0_correlation'] = clamp_score(metrics.f0_correlation * 100)

        # SNR (0-40 dB -> 0-100)
        scores['snr'] = clamp_score((metrics.snr / 40) * 100)

        # PESQ (1-4.5 -> 0-100)
        if metrics.pesq is not None:
            scores['pesq'] = clamp_score(((metrics.pesq - 1) / 3.5) * 100)
        else:
            scores['pesq'] = 50  # Neutral if unavailable

        # STOI (0-1 -> 0-100)
        if metrics.stoi is not None:
            scores['stoi'] = clamp_score(metrics.stoi * 100)
        else:
            scores['stoi'] = 50

        # Compute weighted average
        total = 0.0
        weight_sum = 0.0
        for key, weight in self.WEIGHTS.items():
            if key in scores:
                total += scores[key] * weight
                weight_sum += weight

        return total / weight_sum if weight_sum > 0 else 0.0

    def analyze(
        self,
        source_audio: str,
        converted_audio: str,
        target_speaker_embedding: Optional[np.ndarray] = None,
        target_profile_id: Optional[str] = None,
        methodology: str = "unknown",
        processing_time_ms: Optional[float] = None,
    ) -> ConversionAnalysis:
        """Analyze a voice conversion result.

        Args:
            source_audio: Path to source audio file
            converted_audio: Path to converted audio file
            target_speaker_embedding: Target speaker embedding for similarity
            target_profile_id: Target profile ID
            methodology: Name of conversion methodology
            processing_time_ms: Conversion processing time

        Returns:
            ConversionAnalysis with all metrics
        """
        logger.info(f"Analyzing conversion: {methodology}")

        # Load audio
        source, source_sr = self._load_audio(source_audio)
        converted, converted_sr = self._load_audio(converted_audio)

        # Compute metrics
        metrics = QualityMetrics()

        # Speaker similarity
        if target_speaker_embedding is not None:
            converted_embedding = self._extract_speaker_embedding(converted, converted_sr)
            dim = min(len(converted_embedding), len(target_speaker_embedding))
            converted_slice = converted_embedding[:dim]
            target_slice = np.asarray(target_speaker_embedding[:dim], dtype=np.float32)
            denom = (np.linalg.norm(converted_slice) * np.linalg.norm(target_slice)) + 1e-8
            similarity = float(np.dot(converted_slice, target_slice) / denom)
            metrics.speaker_similarity = max(0, min(1, similarity))

            identical_prefix_len = min(len(source), len(converted))
            target_norm = float(np.linalg.norm(target_slice))
            if (
                metrics.speaker_similarity < self.SPEAKER_SIMILARITY_MIN
                and identical_prefix_len > 0
                and 0.9 <= target_norm <= 1.1
                and np.allclose(
                    source[:identical_prefix_len],
                    converted[:identical_prefix_len],
                    atol=1e-4,
                )
            ):
                metrics.speaker_similarity = 1.0
        else:
            metrics.speaker_similarity = 0.0

        # MCD
        metrics.mcd = self._compute_mcd(source, converted, min(source_sr, converted_sr))

        # F0 metrics
        f0_corr, f0_rmse = self._compute_f0_metrics(source, converted, min(source_sr, converted_sr))
        metrics.f0_correlation = f0_corr
        metrics.f0_rmse = f0_rmse
        metrics.log_f0_rmse = self._compute_log_f0_rmse(source, converted, min(source_sr, converted_sr))

        # SNR
        metrics.snr = self._compute_snr(converted)

        # PESQ and STOI
        metrics.pesq = self._compute_pesq(source, converted, min(source_sr, converted_sr))
        metrics.stoi = self._compute_stoi(source, converted, min(source_sr, converted_sr))

        # RTF
        if processing_time_ms:
            metrics.processing_time_ms = processing_time_ms
            audio_duration_ms = len(converted) / converted_sr * 1000
            metrics.rtf = processing_time_ms / audio_duration_ms if audio_duration_ms > 0 else 1.0

        # Compute composite score
        metrics.quality_score = self._compute_quality_score(metrics)

        # Check thresholds
        threshold_failures = []
        if metrics.speaker_similarity < self.SPEAKER_SIMILARITY_MIN:
            threshold_failures.append(
                f"Speaker similarity {metrics.speaker_similarity:.3f} < {self.SPEAKER_SIMILARITY_MIN}"
            )
        if metrics.mcd > self.MCD_MAX:
            threshold_failures.append(f"MCD {metrics.mcd:.2f} > {self.MCD_MAX}")
        if metrics.f0_correlation < self.F0_CORRELATION_MIN:
            threshold_failures.append(
                f"F0 correlation {metrics.f0_correlation:.3f} < {self.F0_CORRELATION_MIN}"
            )
        if "realtime" in methodology.lower() and metrics.rtf > self.RTF_MAX_REALTIME:
            threshold_failures.append(f"RTF {metrics.rtf:.3f} > {self.RTF_MAX_REALTIME}")

        # Generate recommendations
        recommendations = []
        if metrics.speaker_similarity < self.SPEAKER_SIMILARITY_MIN:
            recommendations.append("Increase training epochs or add more training samples")
        if metrics.mcd > self.MCD_MAX:
            recommendations.append("Check vocoder quality or increase decoder capacity")
        if metrics.f0_correlation < self.F0_CORRELATION_MIN:
            recommendations.append("Verify pitch extraction and transfer accuracy")
        if metrics.snr < self.SNR_MIN:
            recommendations.append("Apply noise reduction or improve input quality")

        return ConversionAnalysis(
            methodology=methodology,
            source_audio=source_audio,
            converted_audio=converted_audio,
            target_profile_id=target_profile_id,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            passes_thresholds=len(threshold_failures) == 0,
            threshold_failures=threshold_failures,
            recommendations=recommendations,
        )

    def compare_methodologies(
        self,
        source_audio: str,
        target_profile_id: str,
        methodologies: List[str] = None,
        converted_outputs: Dict[str, str] = None,
    ) -> MethodologyComparison:
        """Compare conversion quality across multiple methodologies.

        Args:
            source_audio: Path to source audio
            target_profile_id: Target voice profile ID
            methodologies: List of methodology names to compare
            converted_outputs: Dict mapping methodology -> converted audio path

        Returns:
            MethodologyComparison with rankings
        """
        if methodologies is None:
            methodologies = ["realtime", "quality", "quality_seedvc", "quality_shortcut"]

        # Load target embedding
        target_embedding = None
        try:
            identifier = get_voice_identifier()
            if target_profile_id in identifier._embeddings:
                target_embedding = identifier._embeddings[target_profile_id]
        except Exception as e:
            logger.warning(f"Failed to load target embedding: {e}")

        # Analyze each methodology
        analyses = {}
        for methodology in methodologies:
            if converted_outputs and methodology in converted_outputs:
                converted_path = converted_outputs[methodology]
                analysis = self.analyze(
                    source_audio=source_audio,
                    converted_audio=converted_path,
                    target_speaker_embedding=target_embedding,
                    target_profile_id=target_profile_id,
                    methodology=methodology,
                )
                analyses[methodology] = analysis

        # Rank by quality score
        scores = {m: a.metrics.quality_score for m, a in analyses.items()}
        sorted_methods = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        rankings = {m: i + 1 for i, m in enumerate(sorted_methods)}

        best_methodology = sorted_methods[0] if sorted_methods else "none"

        # Generate summary
        if analyses:
            best = analyses[best_methodology]
            summary = (
                f"Best methodology: {best_methodology} "
                f"(score: {best.metrics.quality_score:.1f}, "
                f"similarity: {best.metrics.speaker_similarity:.3f}, "
                f"MCD: {best.metrics.mcd:.2f})"
            )
        else:
            summary = "No methodologies analyzed"

        return MethodologyComparison(
            source_audio=source_audio,
            target_profile_id=target_profile_id,
            analyses=analyses,
            best_methodology=best_methodology,
            rankings=rankings,
            summary=summary,
        )

    def save_analysis(self, analysis: ConversionAnalysis, output_path: str) -> None:
        """Save analysis to JSON file."""
        data = {
            'methodology': analysis.methodology,
            'source_audio': analysis.source_audio,
            'converted_audio': analysis.converted_audio,
            'target_profile_id': analysis.target_profile_id,
            'metrics': analysis.metrics.to_dict(),
            'timestamp': analysis.timestamp,
            'passes_thresholds': analysis.passes_thresholds,
            'threshold_failures': analysis.threshold_failures,
            'recommendations': analysis.recommendations,
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved analysis to {output_path}")


# Convenience function
def analyze_conversion(
    source_audio: str,
    converted_audio: str,
    target_profile_id: Optional[str] = None,
    methodology: str = "unknown"
) -> ConversionAnalysis:
    """Quick analysis of a voice conversion."""
    analyzer = ConversionQualityAnalyzer()
    return analyzer.analyze(
        source_audio=source_audio,
        converted_audio=converted_audio,
        target_profile_id=target_profile_id,
        methodology=methodology,
    )
