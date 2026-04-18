"""Comprehensive quality metrics for voice conversion evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .metrics import (
    pesq as pesq_metric,
    pitch_rmse,
    signal_to_noise_ratio,
    stoi as stoi_metric,
)


class QualityMetrics:
    """Unified quality metrics for voice conversion evaluation."""

    def __init__(
        self,
        n_mfcc: int = 13,
        hop_length: int = 256,
    ):
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self._mos_predictor = None
        self._speaker_similarity = None
        self._pitch_accuracy = None

    @staticmethod
    def _to_numpy(audio: Any) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy().squeeze().astype(np.float64)
        return np.asarray(audio, dtype=np.float64).squeeze()

    def _get_mos_predictor(self):
        if self._mos_predictor is None:
            self._mos_predictor = MOSPredictor()
        return self._mos_predictor

    def _get_speaker_metric(self):
        if self._speaker_similarity is None:
            self._speaker_similarity = SpeakerSimilarity()
        return self._speaker_similarity

    def _get_pitch_metric(self):
        if self._pitch_accuracy is None:
            self._pitch_accuracy = PitchAccuracy()
        return self._pitch_accuracy

    def compute_mcd(
        self,
        reference: torch.Tensor,
        converted: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        """Compute Mel Cepstral Distortion (MCD) in dB.

        MCD measures spectral distortion between reference and converted audio.
        Lower is better. SOTA target: < 5.0 dB.

        Args:
            reference: Reference audio tensor
            converted: Converted audio tensor
            sample_rate: Audio sample rate

        Returns:
            MCD value in dB
        """
        ref_np = self._to_numpy(reference).astype(np.float32)
        conv_np = self._to_numpy(converted).astype(np.float32)

        min_len = min(len(ref_np), len(conv_np))
        ref_np = ref_np[:min_len]
        conv_np = conv_np[:min_len]

        ref_mfcc = librosa.feature.mfcc(
            y=ref_np,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
        )
        conv_mfcc = librosa.feature.mfcc(
            y=conv_np,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
        )

        min_frames = min(ref_mfcc.shape[1], conv_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_frames]
        conv_mfcc = conv_mfcc[:, :min_frames]

        diff = ref_mfcc[1:, :] - conv_mfcc[1:, :]
        mcd = np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
        return float(mcd)

    def compute_f0_rmse(
        self,
        reference: torch.Tensor,
        converted: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        """Compute F0 RMSE (pitch accuracy) in cents.

        Measures pitch deviation between reference and converted audio.
        Lower is better. SOTA target: < 20 cents.

        Args:
            reference: Reference audio tensor
            converted: Converted audio tensor
            sample_rate: Audio sample rate

        Returns:
            F0 RMSE in cents (1200 cents = 1 octave)
        """
        ref_np = self._to_numpy(reference)
        conv_np = self._to_numpy(converted)

        f0_ref, voiced_ref, _ = librosa.pyin(
            ref_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            hop_length=self.hop_length,
        )
        f0_conv, voiced_conv, _ = librosa.pyin(
            conv_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            hop_length=self.hop_length,
        )

        f0_ref = np.nan_to_num(f0_ref, nan=0.0)
        f0_conv = np.nan_to_num(f0_conv, nan=0.0)

        return pitch_rmse(f0_conv, f0_ref, voiced_only=True)

    def compute_pitch_correlation(
        self,
        reference: torch.Tensor,
        converted: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        return self._get_pitch_metric().compute_correlation(
            self._to_numpy(reference),
            self._to_numpy(converted),
            sample_rate,
        )

    def compute_speaker_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between speaker embeddings.

        Higher is better. SOTA target: > 0.85.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding

        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Ensure tensors
        if not isinstance(embedding1, torch.Tensor):
            embedding1 = torch.tensor(embedding1)
        if not isinstance(embedding2, torch.Tensor):
            embedding2 = torch.tensor(embedding2)

        # Flatten if needed
        embedding1 = embedding1.flatten().float()
        embedding2 = embedding2.flatten().float()

        # Compute cosine similarity
        similarity = F.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0),
        )

        return float(similarity.item())

    def compute_audio_speaker_similarity(
        self,
        reference_audio: torch.Tensor,
        converted_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        return self._get_speaker_metric().compute(
            self._to_numpy(reference_audio),
            self._to_numpy(converted_audio),
            sample_rate,
        )

    def predict_mos(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        return self._get_mos_predictor().predict(self._to_numpy(audio), sample_rate)

    def compute_pesq(
        self,
        reference_audio: torch.Tensor,
        converted_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        reference_np = self._to_numpy(reference_audio).astype(np.float32)
        converted_np = self._to_numpy(converted_audio).astype(np.float32)

        if sample_rate not in {8000, 16000}:
            reference_np = librosa.resample(reference_np, orig_sr=sample_rate, target_sr=16000)
            converted_np = librosa.resample(converted_np, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        try:
            return pesq_metric(reference_np, converted_np, sr=sample_rate)
        except ValueError:
            return pesq_metric(reference_np, converted_np, sr=16000)

    def compute_stoi(
        self,
        reference_audio: torch.Tensor,
        converted_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        return stoi_metric(
            self._to_numpy(reference_audio),
            self._to_numpy(converted_audio),
            sr=sample_rate,
        )

    def compute_snr(
        self,
        reference_audio: torch.Tensor,
        converted_audio: torch.Tensor,
    ) -> float:
        return signal_to_noise_ratio(
            self._to_numpy(reference_audio),
            self._to_numpy(converted_audio),
        )

    def compute_all(
        self,
        reference_audio: torch.Tensor,
        converted_audio: torch.Tensor,
        target_speaker: Optional[torch.Tensor] = None,
        sample_rate: int = 24000,
        converted_speaker: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute the full benchmark metric set."""
        results: Dict[str, float] = {
            "mcd": self.compute_mcd(reference_audio, converted_audio, sample_rate),
            "f0_rmse": self.compute_f0_rmse(reference_audio, converted_audio, sample_rate),
            "pitch_corr": self.compute_pitch_correlation(
                reference_audio,
                converted_audio,
                sample_rate,
            ),
            "mos_pred": self.predict_mos(converted_audio, sample_rate),
            "pesq": self.compute_pesq(reference_audio, converted_audio, sample_rate),
            "stoi": self.compute_stoi(reference_audio, converted_audio, sample_rate),
            "snr": self.compute_snr(reference_audio, converted_audio),
        }

        if target_speaker is not None and converted_speaker is not None:
            try:
                results["speaker_similarity"] = self.compute_speaker_similarity(
                    target_speaker,
                    converted_speaker,
                )
            except RuntimeError:
                results["speaker_similarity"] = self.compute_audio_speaker_similarity(
                    reference_audio,
                    converted_audio,
                    sample_rate,
                )
        else:
            results["speaker_similarity"] = self.compute_audio_speaker_similarity(
                reference_audio,
                converted_audio,
                sample_rate,
            )

        return results


# ============================================================================
# MOS Predictor
# ============================================================================


class MOSPredictor:
    """Predicts Mean Opinion Score (MOS) for audio quality.

    Uses a simple heuristic based on spectral features.
    In production, this would use a trained neural MOS predictor.
    """

    def __init__(self, device: str = 'cuda:0'):
        """Initialize MOS predictor.

        Args:
            device: Device for computation
        """
        self.device = device
        self._hop_length = 256

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        """Predict MOS for a single audio sample.

        Args:
            audio: Audio waveform (1D numpy array)
            sample_rate: Sample rate in Hz

        Returns:
            Predicted MOS score (1.0-5.0)
        """
        audio = np.asarray(audio).squeeze()

        # Simple quality heuristic based on:
        # 1. SNR estimation (higher = better)
        # 2. Spectral flatness (speech-like)
        # 3. Dynamic range

        # Estimate SNR via spectral analysis
        spec = np.abs(librosa.stft(audio.astype(np.float32), hop_length=self._hop_length))
        energy = np.mean(spec ** 2)
        noise_floor = np.percentile(spec ** 2, 10)
        snr_db = 10 * np.log10(energy / (noise_floor + 1e-10))

        # Spectral flatness (Wiener entropy)
        flatness = librosa.feature.spectral_flatness(y=audio.astype(np.float32), hop_length=self._hop_length)
        mean_flatness = np.mean(flatness)

        # Dynamic range
        rms = librosa.feature.rms(y=audio.astype(np.float32), hop_length=self._hop_length)
        dynamic_range = np.max(rms) / (np.mean(rms) + 1e-10)

        # Combine into MOS estimate
        # Higher SNR = better, optimal flatness around 0.1 for speech, good dynamic range
        snr_score = np.clip(snr_db / 40, 0, 1) * 2.0  # 0-2 points
        flatness_score = 1.0 - np.abs(mean_flatness - 0.1) * 5  # peak at 0.1
        flatness_score = np.clip(flatness_score, 0, 1) * 1.5  # 0-1.5 points
        dynamic_score = np.clip(dynamic_range / 5, 0, 1) * 1.5  # 0-1.5 points

        mos = 1.0 + snr_score + flatness_score + dynamic_score
        return float(np.clip(mos, 1.0, 5.0))

    def predict_batch(self, audios: list, sample_rate: int) -> list:
        """Predict MOS for a batch of audio samples.

        Args:
            audios: List of audio waveforms
            sample_rate: Sample rate in Hz

        Returns:
            List of predicted MOS scores
        """
        return [self.predict(audio, sample_rate) for audio in audios]


# ============================================================================
# Speaker Similarity
# ============================================================================


class SpeakerSimilarity:
    """Compute speaker similarity between audio samples."""

    def __init__(self, device: str = 'cuda:0'):
        """Initialize speaker similarity metric.

        Args:
            device: Device for computation
        """
        self.device = device
        self._n_mfcc = 20
        self._hop_length = 256

    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding from audio.

        Uses MFCC statistics as a simple embedding.

        Args:
            audio: Audio waveform
            sample_rate: Sample rate in Hz

        Returns:
            Speaker embedding vector
        """
        audio = np.asarray(audio).squeeze().astype(np.float32)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=self._n_mfcc,
            hop_length=self._hop_length,
        )

        # Use mean and std as embedding
        embedding = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
        ])

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def compute(self, audio1: np.ndarray, audio2: np.ndarray, sample_rate: int) -> float:
        """Compute speaker similarity between two audio samples.

        Args:
            audio1: First audio waveform
            audio2: Second audio waveform
            sample_rate: Sample rate in Hz

        Returns:
            Cosine similarity (0-1, higher = more similar)
        """
        emb1 = self.extract_embedding(audio1, sample_rate)
        emb2 = self.extract_embedding(audio2, sample_rate)

        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(np.clip(similarity, 0, 1))


# ============================================================================
# Pitch Accuracy
# ============================================================================


class PitchAccuracy:
    """Compute pitch accuracy metrics."""

    def __init__(self, device: str = 'cuda:0'):
        """Initialize pitch accuracy metric.

        Args:
            device: Device for computation
        """
        self.device = device
        self._hop_length = 256

    def _extract_f0(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract F0 contour from audio."""
        audio = np.asarray(audio).squeeze().astype(np.float64)

        f0, voiced, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            hop_length=self._hop_length,
        )

        return np.nan_to_num(f0, nan=0.0)

    def compute_rmse(self, audio1: np.ndarray, audio2: np.ndarray, sample_rate: int) -> float:
        """Compute F0 RMSE between two audio samples.

        Args:
            audio1: First audio waveform
            audio2: Second audio waveform
            sample_rate: Sample rate in Hz

        Returns:
            F0 RMSE in Hz
        """
        f0_1 = self._extract_f0(audio1, sample_rate)
        f0_2 = self._extract_f0(audio2, sample_rate)

        # Align lengths
        min_len = min(len(f0_1), len(f0_2))
        f0_1 = f0_1[:min_len]
        f0_2 = f0_2[:min_len]

        # Only compare voiced regions
        voiced_mask = (f0_1 > 0) & (f0_2 > 0)
        if not np.any(voiced_mask):
            return 0.0

        diff = f0_1[voiced_mask] - f0_2[voiced_mask]
        rmse = np.sqrt(np.mean(diff ** 2))

        return float(rmse)

    def compute_correlation(self, audio1: np.ndarray, audio2: np.ndarray, sample_rate: int) -> float:
        """Compute pitch correlation between two audio samples.

        Args:
            audio1: First audio waveform
            audio2: Second audio waveform
            sample_rate: Sample rate in Hz

        Returns:
            Pearson correlation coefficient
        """
        f0_1 = self._extract_f0(audio1, sample_rate)
        f0_2 = self._extract_f0(audio2, sample_rate)

        # Align lengths
        min_len = min(len(f0_1), len(f0_2))
        f0_1 = f0_1[:min_len]
        f0_2 = f0_2[:min_len]

        # Only compare voiced regions
        voiced_mask = (f0_1 > 0) & (f0_2 > 0)
        if not np.any(voiced_mask):
            return 0.0

        # Check if signals are identical (zero variance causes NaN correlation)
        f0_1_voiced = f0_1[voiced_mask]
        f0_2_voiced = f0_2[voiced_mask]

        if np.allclose(f0_1_voiced, f0_2_voiced):
            return 1.0  # Identical signals have perfect correlation

        correlation = np.corrcoef(f0_1_voiced, f0_2_voiced)[0, 1]
        return float(np.nan_to_num(correlation, nan=1.0))


# ============================================================================
# Quality Benchmark Runner
# ============================================================================


class QualityBenchmarkRunner:
    """Run comprehensive quality benchmarks."""

    def __init__(self, device: str = 'cuda:0', output_dir: str = './benchmark_results'):
        """Initialize benchmark runner.

        Args:
            device: Device for computation
            output_dir: Directory for output files
        """
        self.device = device
        self.output_dir = output_dir
        self._mos_predictor = MOSPredictor(device)
        self._speaker_similarity = SpeakerSimilarity(device)
        self._pitch_accuracy = PitchAccuracy(device)
        self._results: list = []

    def benchmark_pair(
        self,
        reference: np.ndarray,
        converted: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Benchmark a single reference/converted pair.

        Args:
            reference: Reference audio waveform
            converted: Converted audio waveform
            sample_rate: Sample rate in Hz

        Returns:
            Dict with quality metrics
        """
        results = {
            'mos': self._mos_predictor.predict(converted, sample_rate),
            'speaker_similarity': self._speaker_similarity.compute(reference, converted, sample_rate),
            'pitch_rmse': self._pitch_accuracy.compute_rmse(reference, converted, sample_rate),
        }

        self._results.append(results)
        return results

    def benchmark_batch(
        self,
        pairs: list,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Benchmark a batch of audio pairs.

        Args:
            pairs: List of (reference, converted) tuples
            sample_rate: Sample rate in Hz

        Returns:
            Dict with aggregate metrics
        """
        all_results = []
        for reference, converted in pairs:
            result = self.benchmark_pair(reference, converted, sample_rate)
            all_results.append(result)

        # Compute aggregates
        mos_scores = [r['mos'] for r in all_results]
        similarities = [r['speaker_similarity'] for r in all_results]
        pitch_rmses = [r['pitch_rmse'] for r in all_results]

        return {
            'results': all_results,
            'mean_mos': float(np.mean(mos_scores)),
            'std_mos': float(np.std(mos_scores)),
            'mean_similarity': float(np.mean(similarities)),
            'mean_pitch_rmse': float(np.mean(pitch_rmses)),
            'count': len(pairs),
        }

    def generate_report(self, output_path: str) -> None:
        """Generate benchmark report.

        Args:
            output_path: Path to output JSON file
        """
        import json

        report = {
            'device': self.device,
            'num_samples': len(self._results),
            'results': self._results,
            'summary': aggregate_quality_stats(self._results) if self._results else {},
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


# ============================================================================
# Comparison and Aggregation Functions
# ============================================================================


def compare_model_quality(
    reference: np.ndarray,
    base_output: np.ndarray,
    trained_output: np.ndarray,
    sample_rate: int,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """Compare quality between base and trained model outputs.

    Args:
        reference: Reference audio
        base_output: Output from base model
        trained_output: Output from trained model
        sample_rate: Sample rate in Hz
        device: Device for computation

    Returns:
        Comparison results with metrics and improvement
    """
    runner = QualityBenchmarkRunner(device=device)

    base_metrics = runner.benchmark_pair(reference, base_output, sample_rate)
    runner._results.clear()  # Reset for trained
    trained_metrics = runner.benchmark_pair(reference, trained_output, sample_rate)

    improvement = {
        'mos_delta': trained_metrics['mos'] - base_metrics['mos'],
        'similarity_delta': trained_metrics['speaker_similarity'] - base_metrics['speaker_similarity'],
        'pitch_rmse_delta': base_metrics['pitch_rmse'] - trained_metrics['pitch_rmse'],  # Lower is better
    }

    return {
        'base_metrics': base_metrics,
        'trained_metrics': trained_metrics,
        'improvement': improvement,
    }


def aggregate_quality_stats(metrics_list: list) -> Dict[str, float]:
    """Aggregate quality statistics from multiple evaluations.

    Args:
        metrics_list: List of metric dicts

    Returns:
        Aggregated statistics
    """
    if not metrics_list:
        return {}

    mos_scores = [m.get('mos', 0) for m in metrics_list]
    similarities = [m.get('speaker_similarity', 0) for m in metrics_list]
    pitch_rmses = [m.get('pitch_rmse', 0) for m in metrics_list]

    return {
        'mean_mos': float(np.mean(mos_scores)),
        'std_mos': float(np.std(mos_scores)),
        'min_mos': float(np.min(mos_scores)),
        'max_mos': float(np.max(mos_scores)),
        'p50_mos': float(np.percentile(mos_scores, 50)),
        'p95_mos': float(np.percentile(mos_scores, 95)),
        'mean_speaker_similarity': float(np.mean(similarities)),
        'std_speaker_similarity': float(np.std(similarities)),
        'mean_pitch_rmse': float(np.mean(pitch_rmses)),
        'std_pitch_rmse': float(np.std(pitch_rmses)),
        'count': len(metrics_list),
    }
