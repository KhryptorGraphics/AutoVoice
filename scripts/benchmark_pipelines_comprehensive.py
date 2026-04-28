#!/usr/bin/env python3
"""Comprehensive Performance Benchmark Suite for AutoVoice Pipelines.

Benchmarks all voice conversion pipeline types with detailed metrics:
- realtime: Low-latency ContentVec + SimpleDecoder + HiFiGAN (22kHz)
- quality: SOTA CoMoSVC consistency model (24kHz)
- quality_seedvc: Seed-VC DiT-CFM with BigVGAN (44.1kHz)
- realtime_meanvc: MeanVC streaming with mean flows (16kHz)

Metrics measured:
- RTF (Real-Time Factor) - must be <1.0 for realtime pipelines
- Latency (ms) - time to first output chunk
- GPU Memory (MB) - peak memory allocation
- MCD (Mel Cepstral Distortion) - synthesis quality metric
- Speaker Similarity - cosine similarity of embeddings vs reference

Usage:
    python scripts/benchmark_pipelines_comprehensive.py
    python scripts/benchmark_pipelines_comprehensive.py --pipelines realtime quality
    python scripts/benchmark_pipelines_comprehensive.py --iterations 20 --output reports/benchmark.json
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'meanvc'))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Comprehensive metrics for a single pipeline benchmark."""
    pipeline_name: str
    pipeline_type: str

    # Timing metrics
    total_time_s: float = 0.0
    rtf: float = 0.0  # Real-Time Factor (processing_time / audio_duration)
    latency_ms: float = 0.0  # Time to first output
    warmup_time_s: float = 0.0

    # Memory metrics
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    cpu_memory_mb: float = 0.0

    # Quality metrics
    mcd: float = 0.0  # Mel Cepstral Distortion (lower is better)
    speaker_similarity: float = 0.0  # Cosine similarity (higher is better)
    pitch_corr: float = 0.0  # Pitch correlation (higher is better)
    metric_basis: Dict[str, str] = field(default_factory=dict)
    metric_applicability: Dict[str, bool] = field(default_factory=dict)

    # Audio metadata
    input_duration_s: float = 0.0
    output_duration_s: float = 0.0
    input_sample_rate: int = 0
    output_sample_rate: int = 0

    # Iteration stats
    iterations: int = 0
    times: List[float] = field(default_factory=list)
    rtf_mean: float = 0.0
    rtf_std: float = 0.0
    rtf_min: float = 0.0
    rtf_max: float = 0.0

    # Status
    success: bool = True
    error: Optional[str] = None


def assign_conversion_quality_metrics(
    metrics: PipelineMetrics,
    *,
    output_audio: np.ndarray,
    output_sr: int,
    source_audio: np.ndarray,
    source_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
) -> None:
    """Populate quality metrics with explicit conversion metric provenance.

    Pitch preservation is computed against the source performance, but remains
    informational for this smoke benchmark until an aligned pitch-preservation
    fixture suite is selected. MCD is only a production gate for aligned
    same-content targets, which this zero-shot benchmark does not provide.
    """
    metrics.speaker_similarity = compute_speaker_similarity(
        output_audio, output_sr,
        reference_audio, reference_sr,
    )
    metrics.pitch_corr = compute_pitch_correlation(
        output_audio, output_sr,
        source_audio, source_sr,
    )
    metrics.mcd = 0.0
    metrics.metric_basis.update({
        "speaker_similarity": "converted_output_vs_target_reference",
        "pitch_corr": "informational_converted_output_vs_source_performance",
        "mcd": "not_applicable_without_aligned_same_content_target",
    })
    metrics.metric_applicability.update({
        "speaker_similarity": True,
        "pitch_corr": False,
        "mcd": False,
    })


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all pipelines."""
    timestamp: str
    system_info: Dict[str, Any]
    test_audio: Dict[str, Any]
    pipelines: Dict[str, PipelineMetrics]
    summary: Dict[str, Any]


def get_system_info() -> Dict[str, Any]:
    """Gather system information for reproducibility."""
    info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'platform': sys.platform,
    }

    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'gpu_memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        })

    return info


def reset_gpu_memory():
    """Reset GPU memory for accurate measurement."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> Tuple[float, float]:
    """Get current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return allocated, peak


def compute_speaker_similarity(
    audio1: np.ndarray, sr1: int,
    audio2: np.ndarray, sr2: int,
    device: torch.device = torch.device('cpu'),
) -> float:
    """Compute speaker similarity using mel-statistic embeddings.

    Uses the same approach as the training pipeline: mean+std of mel bands.
    """
    try:
        import librosa

        # Resample to common rate
        target_sr = 22050
        if sr1 != target_sr:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sr)
        if sr2 != target_sr:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sr)

        # Compute mel spectrograms
        def get_mel_embedding(audio: np.ndarray) -> np.ndarray:
            """Extract mel-statistic speaker embedding."""
            mel = librosa.feature.melspectrogram(
                y=audio, sr=target_sr, n_mels=128, fmax=8000, hop_length=256
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Mean + std across time = 256-dim embedding
            mean = np.mean(mel_db, axis=1)
            std = np.std(mel_db, axis=1)
            embedding = np.concatenate([mean, std])

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

            return embedding

        emb1 = get_mel_embedding(audio1)
        emb2 = get_mel_embedding(audio2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(np.clip(similarity, -1.0, 1.0))

    except Exception as e:
        logger.warning(f"Speaker similarity computation failed: {e}")
        return 0.0


def compute_mcd(
    audio1: np.ndarray, sr1: int,
    audio2: np.ndarray, sr2: int,
    n_mfcc: int = 13,
) -> float:
    """Compute Mel Cepstral Distortion between two audio signals."""
    try:
        import librosa

        # Resample to common rate
        target_sr = min(sr1, sr2)
        if sr1 != target_sr:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sr)
        if sr2 != target_sr:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sr)

        # Match lengths
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        # Extract MFCCs
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=target_sr, n_mfcc=n_mfcc)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=target_sr, n_mfcc=n_mfcc)

        # Match frame counts
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_frames]
        mfcc2 = mfcc2[:, :min_frames]

        # MCD (skip 0th coefficient)
        diff = mfcc1[1:] - mfcc2[1:]
        mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))

        return float(mcd)

    except Exception as e:
        logger.warning(f"MCD computation failed: {e}")
        return 0.0


def compute_pitch_correlation(
    audio1: np.ndarray,
    sr1: int,
    audio2: np.ndarray,
    sr2: int,
) -> float:
    """Compute a simple F0 correlation signal between two audio clips."""
    try:
        import librosa

        target_sr = 22050
        if sr1 != target_sr:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sr)
        if sr2 != target_sr:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sr)

        f0_1, voiced_1, _ = librosa.pyin(audio1, fmin=50, fmax=1100, sr=target_sr)
        f0_2, voiced_2, _ = librosa.pyin(audio2, fmin=50, fmax=1100, sr=target_sr)
        f0_1 = np.nan_to_num(f0_1, nan=0.0)
        f0_2 = np.nan_to_num(f0_2, nan=0.0)
        frame_count = min(len(f0_1), len(f0_2))
        if frame_count < 2:
            return 0.0

        voiced = np.asarray(voiced_1[:frame_count], dtype=bool) & np.asarray(voiced_2[:frame_count], dtype=bool)
        if int(np.sum(voiced)) < 2:
            return 0.0

        corr = np.corrcoef(f0_1[:frame_count][voiced], f0_2[:frame_count][voiced])[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(np.clip(corr, -1.0, 1.0))
    except Exception as e:
        logger.warning(f"Pitch correlation computation failed: {e}")
        return 0.0


def load_test_audio(
    audio_path: Path,
    duration_limit: Optional[float] = None,
) -> Tuple[np.ndarray, int, float]:
    """Load test audio file.

    Returns:
        (audio_array, sample_rate, duration_seconds)
    """
    import librosa

    audio, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=duration_limit)
    duration = len(audio) / sr

    return audio, sr, duration


def benchmark_realtime_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    iterations: int = 10,
    device: torch.device = torch.device('cuda'),
) -> PipelineMetrics:
    """Benchmark the realtime (karaoke) pipeline."""
    from auto_voice.inference.realtime_pipeline import RealtimePipeline

    metrics = PipelineMetrics(
        pipeline_name="Realtime Karaoke",
        pipeline_type="realtime",
        input_sample_rate=sample_rate,
        iterations=iterations,
    )

    try:
        reset_gpu_memory()

        # Initialize pipeline
        warmup_start = time.perf_counter()
        pipeline = RealtimePipeline(device=device)

        # Create speaker embedding from reference
        import librosa
        ref_16k = librosa.resample(reference_audio, orig_sr=reference_sr, target_sr=16000)
        ref_mel = librosa.feature.melspectrogram(y=ref_16k, sr=16000, n_mels=128, hop_length=256)
        ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)
        mean = np.mean(ref_mel_db, axis=1)
        std = np.std(ref_mel_db, axis=1)
        speaker_embedding = np.concatenate([mean, std])
        speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

        pipeline.set_speaker_embedding(speaker_embedding)

        metrics.warmup_time_s = time.perf_counter() - warmup_start

        # Resample input to pipeline sample rate
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        metrics.input_duration_s = len(audio) / sample_rate

        _, mem_after_init = get_gpu_memory_mb()

        # Warmup run
        _ = pipeline.process_chunk(audio_16k[:16000])  # 1 second warmup

        # Benchmark iterations
        times = []
        outputs = []

        for i in range(iterations):
            reset_gpu_memory()

            start_time = time.perf_counter()
            output = pipeline.process_chunk(audio_16k)
            elapsed = time.perf_counter() - start_time

            times.append(elapsed)
            if i == 0:
                outputs.append(output)

        # Collect metrics
        metrics.times = times
        metrics.total_time_s = np.mean(times)
        metrics.rtf = metrics.total_time_s / metrics.input_duration_s
        metrics.rtf_mean = np.mean([t / metrics.input_duration_s for t in times])
        metrics.rtf_std = np.std([t / metrics.input_duration_s for t in times])
        metrics.rtf_min = np.min([t / metrics.input_duration_s for t in times])
        metrics.rtf_max = np.max([t / metrics.input_duration_s for t in times])

        # Latency metrics
        latency_metrics = pipeline.get_latency_metrics()
        metrics.latency_ms = latency_metrics.get('total_ms', 0.0)

        # Memory metrics
        _, metrics.gpu_memory_peak_mb = get_gpu_memory_mb()
        metrics.gpu_memory_allocated_mb = mem_after_init

        # Output metadata
        metrics.output_sample_rate = pipeline.output_sample_rate
        metrics.output_duration_s = len(outputs[0]) / metrics.output_sample_rate

        # Quality metrics
        if len(outputs) > 0:
            assign_conversion_quality_metrics(
                metrics,
                output_audio=outputs[0],
                output_sr=metrics.output_sample_rate,
                source_audio=audio,
                source_sr=sample_rate,
                reference_audio=reference_audio,
                reference_sr=reference_sr,
            )

    except Exception as e:
        metrics.success = False
        metrics.error = str(e)
        logger.error(f"Realtime pipeline benchmark failed: {e}")

    return metrics


def benchmark_quality_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    iterations: int = 5,
    device: torch.device = torch.device('cuda'),
) -> PipelineMetrics:
    """Benchmark the SOTA CoMoSVC quality pipeline."""
    from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

    metrics = PipelineMetrics(
        pipeline_name="Quality (CoMoSVC)",
        pipeline_type="quality",
        input_sample_rate=sample_rate,
        iterations=iterations,
    )

    try:
        reset_gpu_memory()

        # Initialize pipeline
        warmup_start = time.perf_counter()
        pipeline = SOTAConversionPipeline(
            device=device,
            n_steps=1,  # Consistency model for speed
            require_gpu=True,
        )

        # Create speaker embedding from reference
        import librosa
        ref_22k = librosa.resample(reference_audio, orig_sr=reference_sr, target_sr=22050)
        ref_mel = librosa.feature.melspectrogram(y=ref_22k, sr=22050, n_mels=128, hop_length=256)
        ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)
        mean = np.mean(ref_mel_db, axis=1)
        std = np.std(ref_mel_db, axis=1)
        speaker_embedding = np.concatenate([mean, std])
        speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)
        speaker_tensor = torch.from_numpy(speaker_embedding).float().to(device)

        metrics.warmup_time_s = time.perf_counter() - warmup_start
        metrics.input_duration_s = len(audio) / sample_rate

        _, mem_after_init = get_gpu_memory_mb()

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Warmup run (short segment)
        warmup_len = min(len(audio), sample_rate * 2)  # 2 seconds
        _ = pipeline.convert(audio_tensor[:warmup_len], sample_rate, speaker_tensor)

        # Benchmark iterations
        times = []
        outputs = []

        for i in range(iterations):
            reset_gpu_memory()

            start_time = time.perf_counter()
            result = pipeline.convert(audio_tensor, sample_rate, speaker_tensor)
            elapsed = time.perf_counter() - start_time

            times.append(elapsed)
            if i == 0:
                outputs.append(result)

        # Collect metrics
        metrics.times = times
        metrics.total_time_s = np.mean(times)
        metrics.rtf = metrics.total_time_s / metrics.input_duration_s
        metrics.rtf_mean = np.mean([t / metrics.input_duration_s for t in times])
        metrics.rtf_std = np.std([t / metrics.input_duration_s for t in times])
        metrics.rtf_min = np.min([t / metrics.input_duration_s for t in times])
        metrics.rtf_max = np.max([t / metrics.input_duration_s for t in times])

        # Memory metrics
        _, metrics.gpu_memory_peak_mb = get_gpu_memory_mb()
        metrics.gpu_memory_allocated_mb = mem_after_init

        # Output metadata
        metrics.output_sample_rate = outputs[0]['sample_rate']
        output_audio = outputs[0]['audio'].cpu().numpy()
        metrics.output_duration_s = len(output_audio) / metrics.output_sample_rate

        # Quality metrics
        assign_conversion_quality_metrics(
            metrics,
            output_audio=output_audio,
            output_sr=metrics.output_sample_rate,
            source_audio=audio,
            source_sr=sample_rate,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
        )

    except Exception as e:
        metrics.success = False
        metrics.error = str(e)
        logger.error(f"Quality pipeline benchmark failed: {e}")

    return metrics


def benchmark_seedvc_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    iterations: int = 5,
    device: torch.device = torch.device('cuda'),
) -> PipelineMetrics:
    """Benchmark the Seed-VC DiT-CFM pipeline."""
    from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

    metrics = PipelineMetrics(
        pipeline_name="Quality (Seed-VC DiT-CFM)",
        pipeline_type="quality_seedvc",
        input_sample_rate=sample_rate,
        iterations=iterations,
    )

    try:
        reset_gpu_memory()

        # Initialize pipeline
        warmup_start = time.perf_counter()
        pipeline = SeedVCPipeline(
            device=device,
            diffusion_steps=10,
            f0_condition=True,
            require_gpu=True,
        )

        # Set reference audio for in-context learning
        pipeline.set_reference_audio(reference_audio, reference_sr)

        metrics.warmup_time_s = time.perf_counter() - warmup_start
        metrics.input_duration_s = len(audio) / sample_rate

        _, mem_after_init = get_gpu_memory_mb()

        # Warmup run (short segment)
        warmup_len = min(len(audio), sample_rate * 2)
        _ = pipeline.convert(audio[:warmup_len], sample_rate)

        # Benchmark iterations
        times = []
        outputs = []

        for i in range(iterations):
            reset_gpu_memory()

            start_time = time.perf_counter()
            result = pipeline.convert(audio, sample_rate)
            elapsed = time.perf_counter() - start_time

            times.append(elapsed)
            if i == 0:
                outputs.append(result)

        # Collect metrics
        metrics.times = times
        metrics.total_time_s = np.mean(times)
        metrics.rtf = metrics.total_time_s / metrics.input_duration_s
        metrics.rtf_mean = np.mean([t / metrics.input_duration_s for t in times])
        metrics.rtf_std = np.std([t / metrics.input_duration_s for t in times])
        metrics.rtf_min = np.min([t / metrics.input_duration_s for t in times])
        metrics.rtf_max = np.max([t / metrics.input_duration_s for t in times])

        # Memory metrics
        _, metrics.gpu_memory_peak_mb = get_gpu_memory_mb()
        metrics.gpu_memory_allocated_mb = mem_after_init

        # Output metadata
        metrics.output_sample_rate = outputs[0]['sample_rate']
        output_audio = outputs[0]['audio'].cpu().numpy() if torch.is_tensor(outputs[0]['audio']) else outputs[0]['audio']
        metrics.output_duration_s = len(output_audio) / metrics.output_sample_rate

        # Quality metrics
        assign_conversion_quality_metrics(
            metrics,
            output_audio=output_audio,
            output_sr=metrics.output_sample_rate,
            source_audio=audio,
            source_sr=sample_rate,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
        )

    except Exception as e:
        metrics.success = False
        metrics.error = str(e)
        logger.error(f"Seed-VC pipeline benchmark failed: {e}")

    return metrics


def benchmark_meanvc_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    iterations: int = 10,
    device: torch.device = torch.device('cpu'),  # MeanVC optimized for CPU
) -> PipelineMetrics:
    """Benchmark the MeanVC streaming pipeline."""
    from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

    metrics = PipelineMetrics(
        pipeline_name="Realtime (MeanVC Streaming)",
        pipeline_type="realtime_meanvc",
        input_sample_rate=sample_rate,
        iterations=iterations,
    )

    try:
        reset_gpu_memory()

        # Initialize pipeline
        warmup_start = time.perf_counter()
        pipeline = MeanVCPipeline(
            device=device,
            steps=2,  # 2-step for quality
            require_gpu=False,
        )

        # Set reference audio
        pipeline.set_reference_audio(reference_audio, reference_sr)

        metrics.warmup_time_s = time.perf_counter() - warmup_start
        metrics.input_duration_s = len(audio) / sample_rate

        if torch.cuda.is_available():
            _, mem_after_init = get_gpu_memory_mb()
        else:
            mem_after_init = 0.0

        # Warmup run
        pipeline.reset_session()
        warmup_chunk = audio[:pipeline.chunk_size] if len(audio) > pipeline.chunk_size else audio
        import librosa
        warmup_16k = librosa.resample(warmup_chunk, orig_sr=sample_rate, target_sr=16000)
        if len(warmup_16k) >= pipeline.chunk_size:
            _ = pipeline.process_chunk(warmup_16k[:pipeline.chunk_size])

        # Benchmark iterations (full conversion)
        times = []
        outputs = []

        for i in range(iterations):
            pipeline.reset_session()
            reset_gpu_memory()

            start_time = time.perf_counter()
            result = pipeline.convert(audio, sample_rate)
            elapsed = time.perf_counter() - start_time

            times.append(elapsed)
            if i == 0:
                outputs.append(result)

        # Collect metrics
        metrics.times = times
        metrics.total_time_s = np.mean(times)
        metrics.rtf = metrics.total_time_s / metrics.input_duration_s
        metrics.rtf_mean = np.mean([t / metrics.input_duration_s for t in times])
        metrics.rtf_std = np.std([t / metrics.input_duration_s for t in times])
        metrics.rtf_min = np.min([t / metrics.input_duration_s for t in times])
        metrics.rtf_max = np.max([t / metrics.input_duration_s for t in times])

        # Latency metrics
        latency_metrics = pipeline.get_latency_metrics()
        metrics.latency_ms = latency_metrics.get('total_ms', 0.0)

        # Memory metrics
        if torch.cuda.is_available():
            _, metrics.gpu_memory_peak_mb = get_gpu_memory_mb()
        metrics.gpu_memory_allocated_mb = mem_after_init

        # Output metadata
        metrics.output_sample_rate = outputs[0]['sample_rate']
        output_audio = outputs[0]['audio'].cpu().numpy() if torch.is_tensor(outputs[0]['audio']) else outputs[0]['audio']
        metrics.output_duration_s = len(output_audio) / metrics.output_sample_rate

        # Quality metrics
        assign_conversion_quality_metrics(
            metrics,
            output_audio=output_audio,
            output_sr=metrics.output_sample_rate,
            source_audio=audio,
            source_sr=sample_rate,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
        )

    except Exception as e:
        metrics.success = False
        metrics.error = str(e)
        logger.error(f"MeanVC pipeline benchmark failed: {e}")

    return metrics


def generate_summary(pipelines: Dict[str, PipelineMetrics]) -> Dict[str, Any]:
    """Generate summary statistics across all pipelines."""
    successful = {k: v for k, v in pipelines.items() if v.success}

    if not successful:
        return {'error': 'No successful benchmarks'}

    # Find best performers
    fastest = min(successful.values(), key=lambda x: x.rtf)
    best_quality = max(successful.values(), key=lambda x: x.speaker_similarity)
    lowest_mcd = min(successful.values(), key=lambda x: x.mcd if x.mcd > 0 else float('inf'))
    lowest_memory = min(successful.values(), key=lambda x: x.gpu_memory_peak_mb if x.gpu_memory_peak_mb > 0 else float('inf'))

    # Realtime compliance check
    realtime_compliant = [k for k, v in successful.items() if v.rtf < 1.0]
    streaming_compliant = [k for k, v in successful.items() if v.latency_ms < 100 and v.latency_ms > 0]

    return {
        'total_pipelines': len(pipelines),
        'successful_pipelines': len(successful),
        'fastest_pipeline': {
            'name': fastest.pipeline_name,
            'type': fastest.pipeline_type,
            'rtf': fastest.rtf,
        },
        'best_quality_pipeline': {
            'name': best_quality.pipeline_name,
            'type': best_quality.pipeline_type,
            'speaker_similarity': best_quality.speaker_similarity,
        },
        'lowest_mcd_pipeline': {
            'name': lowest_mcd.pipeline_name,
            'type': lowest_mcd.pipeline_type,
            'mcd': lowest_mcd.mcd,
        },
        'lowest_memory_pipeline': {
            'name': lowest_memory.pipeline_name,
            'type': lowest_memory.pipeline_type,
            'gpu_memory_mb': lowest_memory.gpu_memory_peak_mb,
        },
        'realtime_compliant': realtime_compliant,
        'streaming_compliant': streaming_compliant,
        'recommendations': {
            'karaoke': 'realtime' if 'realtime' in realtime_compliant else fastest.pipeline_type,
            'studio': best_quality.pipeline_type,
            'streaming': streaming_compliant[0] if streaming_compliant else fastest.pipeline_type,
        },
    }


def print_report(report: BenchmarkReport):
    """Print formatted benchmark report to console."""
    print("\n" + "=" * 80)
    print("  AUTOVOICE COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)

    print(f"\nTimestamp: {report.timestamp}")
    print(f"Platform: {report.system_info.get('platform', 'unknown')}")
    if report.system_info.get('cuda_available'):
        print(f"GPU: {report.system_info.get('gpu_name', 'unknown')}")
        print(f"CUDA: {report.system_info.get('cuda_version', 'unknown')}")

    print(f"\nTest Audio: {report.test_audio.get('path', 'unknown')}")
    print(f"Duration: {report.test_audio.get('duration_s', 0):.2f}s")
    print(f"Sample Rate: {report.test_audio.get('sample_rate', 0)}Hz")

    print("\n" + "-" * 80)
    print("PIPELINE RESULTS")
    print("-" * 80)

    header = f"{'Pipeline':<30} {'RTF':<8} {'Latency':<10} {'Memory':<12} {'Similarity':<12} {'Pitch':<8} {'MCD':<8}"
    print(header)
    print("-" * 80)

    for name, metrics in report.pipelines.items():
        if metrics.success:
            latency_str = f"{metrics.latency_ms:.1f}ms" if metrics.latency_ms > 0 else "N/A"
            memory_str = f"{metrics.gpu_memory_peak_mb:.0f}MB" if metrics.gpu_memory_peak_mb > 0 else "CPU"
            print(f"{metrics.pipeline_name:<30} {metrics.rtf:<8.3f} {latency_str:<10} {memory_str:<12} "
                  f"{metrics.speaker_similarity:<12.3f} {metrics.pitch_corr:<8.3f} {metrics.mcd:<8.2f}")
        else:
            print(f"{metrics.pipeline_name:<30} FAILED: {metrics.error}")

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    summary = report.summary
    if 'error' not in summary:
        print(f"\nFastest Pipeline: {summary['fastest_pipeline']['name']} (RTF: {summary['fastest_pipeline']['rtf']:.3f})")
        print(f"Best Quality: {summary['best_quality_pipeline']['name']} (Similarity: {summary['best_quality_pipeline']['speaker_similarity']:.3f})")
        print(f"Lowest MCD: {summary['lowest_mcd_pipeline']['name']} (MCD: {summary['lowest_mcd_pipeline']['mcd']:.2f})")

        print(f"\nRealtime Compliant (RTF < 1.0): {', '.join(summary['realtime_compliant']) or 'None'}")
        print(f"Streaming Compliant (<100ms): {', '.join(summary['streaming_compliant']) or 'None'}")

        print("\nRecommendations:")
        print(f"  - Karaoke/Live: {summary['recommendations']['karaoke']}")
        print(f"  - Studio/Offline: {summary['recommendations']['studio']}")
        print(f"  - Streaming: {summary['recommendations']['streaming']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive pipeline benchmarking')
    parser.add_argument('--audio', type=str,
                        default='tests/quality_samples/william_singe_pillowtalk.wav',
                        help='Input audio file for benchmarking')
    parser.add_argument('--reference', type=str,
                        default='tests/quality_samples/conor_maynard_pillowtalk.wav',
                        help='Reference audio for speaker similarity')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Maximum audio duration to benchmark (seconds)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of benchmark iterations per pipeline')
    parser.add_argument('--pipelines', nargs='+',
                        default=['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc'],
                        help='Pipelines to benchmark')
    parser.add_argument('--output', type=str, default='reports/performance_report.json',
                        help='Output JSON report path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for GPU pipelines (cuda or cpu)')

    args = parser.parse_args()

    # Setup paths
    os.chdir(Path(__file__).parent.parent)

    # Check audio files exist
    audio_path = Path(args.audio)
    reference_path = Path(args.reference)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    if not reference_path.exists():
        logger.error(f"Reference file not found: {reference_path}")
        return 1

    # Load audio
    logger.info(f"Loading test audio: {audio_path}")
    audio, audio_sr, audio_duration = load_test_audio(audio_path, args.duration)

    logger.info(f"Loading reference audio: {reference_path}")
    reference, ref_sr, ref_duration = load_test_audio(reference_path, args.duration)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize report
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        system_info=get_system_info(),
        test_audio={
            'path': str(audio_path),
            'duration_s': audio_duration,
            'sample_rate': audio_sr,
        },
        pipelines={},
        summary={},
    )

    # Benchmark functions mapping
    benchmark_funcs = {
        'realtime': benchmark_realtime_pipeline,
        'quality': benchmark_quality_pipeline,
        'quality_seedvc': benchmark_seedvc_pipeline,
        'realtime_meanvc': benchmark_meanvc_pipeline,
    }

    # Run benchmarks
    for pipeline_type in args.pipelines:
        if pipeline_type not in benchmark_funcs:
            logger.warning(f"Unknown pipeline type: {pipeline_type}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {pipeline_type}")
        logger.info(f"{'='*60}")

        func = benchmark_funcs[pipeline_type]

        # MeanVC uses CPU by default
        bench_device = torch.device('cpu') if pipeline_type == 'realtime_meanvc' else device

        metrics = func(
            audio=audio,
            sample_rate=audio_sr,
            reference_audio=reference,
            reference_sr=ref_sr,
            iterations=args.iterations,
            device=bench_device,
        )

        report.pipelines[pipeline_type] = metrics

        if metrics.success:
            logger.info(
                f"RTF: {metrics.rtf:.3f}, Similarity: {metrics.speaker_similarity:.3f}, "
                f"Pitch: {metrics.pitch_corr:.3f}, MCD: {metrics.mcd:.2f}"
            )
        else:
            logger.error(f"Failed: {metrics.error}")

        # Clean up between pipelines
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate summary
    report.summary = generate_summary(report.pipelines)

    # Print report
    print_report(report)

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict for JSON serialization
    report_dict = {
        'timestamp': report.timestamp,
        'system_info': report.system_info,
        'test_audio': report.test_audio,
        'pipelines': {k: asdict(v) for k, v in report.pipelines.items()},
        'summary': report.summary,
    }

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
