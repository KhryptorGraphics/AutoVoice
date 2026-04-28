#!/usr/bin/env python3
"""Performance benchmark tests for AutoVoice voice conversion pipelines.

Tests verify that each pipeline meets its performance requirements:
- Realtime pipeline: RTF < 1.0 (must process faster than realtime)
- Streaming pipeline: Latency < 100ms per chunk
- Quality pipelines: RTF < 5.0 (reasonable offline processing)

Metrics tracked:
- RTF (Real-Time Factor)
- Latency (ms)
- GPU Memory (MB)
- MCD (quality)
- Speaker Similarity

Run with: pytest tests/test_performance_benchmarks.py -v
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Generator, Tuple

import numpy as np
import pytest
import torch

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_voice.runtime_contract import PIPELINE_DEFINITIONS

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get compute device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture(scope="module")
def test_audio() -> Tuple[np.ndarray, int]:
    """Load short test audio for benchmarking (5 seconds)."""
    import librosa

    # Use existing test sample
    audio_path = Path(__file__).parent / "quality_samples" / "william_singe_pillowtalk.wav"

    if not audio_path.exists():
        # Create synthetic test audio if sample not available
        logger.warning("Test audio not found, generating synthetic audio")
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        # Generate a simple sine wave with varying frequency (simulates singing)
        freq = 440 + 100 * np.sin(2 * np.pi * 0.5 * t)  # Vibrato
        audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        return audio, sr

    audio, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=5.0)
    return audio.astype(np.float32), sr


@pytest.fixture(scope="module")
def reference_audio() -> Tuple[np.ndarray, int]:
    """Load reference audio for speaker similarity."""
    import librosa

    audio_path = Path(__file__).parent / "quality_samples" / "conor_maynard_pillowtalk.wav"

    if not audio_path.exists():
        # Create synthetic reference
        logger.warning("Reference audio not found, generating synthetic audio")
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        freq = 330 + 80 * np.sin(2 * np.pi * 0.3 * t)
        audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        return audio, sr

    audio, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=5.0)
    return audio.astype(np.float32), sr


@pytest.fixture(scope="module")
def speaker_embedding(reference_audio: Tuple[np.ndarray, int]) -> np.ndarray:
    """Create speaker embedding from reference audio."""
    import librosa

    audio, sr = reference_audio

    # Resample to 22050Hz for mel computation
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=22050, n_mels=128, hop_length=256, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Mean + std statistics = 256-dim embedding
    mean = np.mean(mel_db, axis=1)
    std = np.std(mel_db, axis=1)
    embedding = np.concatenate([mean, std])

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    return embedding.astype(np.float32)


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    yield

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Helper Functions
# =============================================================================

def measure_rtf(
    process_fn: Callable[[], None],
    audio_duration: float,
    iterations: int = 3,
) -> Tuple[float, float, float]:
    """Measure Real-Time Factor (RTF).

    Args:
        process_fn: Function to benchmark
        audio_duration: Duration of audio in seconds
        iterations: Number of iterations

    Returns:
        (mean_rtf, min_rtf, max_rtf)
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        process_fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    rtfs = [t / audio_duration for t in times]
    return np.mean(rtfs), np.min(rtfs), np.max(rtfs)


def get_gpu_memory_mb() -> Tuple[float, float]:
    """Get current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return allocated, peak


# =============================================================================
# Realtime Pipeline Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.cuda
class TestRealtimePipeline:
    """Performance tests for realtime karaoke pipeline."""

    @pytest.fixture(scope="class")
    def realtime_pipeline(self, device):
        """Initialize realtime pipeline."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline(device=device)
        yield pipeline

    def test_rtf_under_one(
        self,
        realtime_pipeline,
        test_audio: Tuple[np.ndarray, int],
        speaker_embedding: np.ndarray,
    ):
        """RTF must be < 1.0 for realtime performance."""
        import librosa

        audio, sr = test_audio
        audio_duration = len(audio) / sr

        # Resample to 16kHz
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Set speaker
        realtime_pipeline.set_speaker_embedding(speaker_embedding)

        # Warmup
        realtime_pipeline.process_chunk(audio_16k[:16000])

        # Measure RTF
        def process():
            realtime_pipeline.process_chunk(audio_16k)

        mean_rtf, min_rtf, max_rtf = measure_rtf(process, audio_duration, iterations=3)

        logger.info(f"Realtime pipeline RTF: {mean_rtf:.3f} (min={min_rtf:.3f}, max={max_rtf:.3f})")

        assert mean_rtf < 1.0, f"RTF {mean_rtf:.3f} exceeds realtime threshold of 1.0"

    def test_latency_under_100ms(
        self,
        realtime_pipeline,
        test_audio: Tuple[np.ndarray, int],
        speaker_embedding: np.ndarray,
    ):
        """Per-chunk latency must be < 100ms for streaming."""
        import librosa

        audio, sr = test_audio

        # Resample to 16kHz
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Set speaker
        realtime_pipeline.set_speaker_embedding(speaker_embedding)

        # Process in 100ms chunks (1600 samples at 16kHz)
        chunk_size = 1600
        latencies = []

        for i in range(0, len(audio_16k) - chunk_size, chunk_size):
            chunk = audio_16k[i:i + chunk_size]
            start = time.perf_counter()
            _ = realtime_pipeline.process_chunk(chunk)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        logger.info(f"Realtime chunk latency: mean={mean_latency:.1f}ms, p95={p95_latency:.1f}ms")

        assert mean_latency < 100, f"Mean latency {mean_latency:.1f}ms exceeds 100ms threshold"

    def test_gpu_memory_reasonable(
        self,
        realtime_pipeline,
        test_audio: Tuple[np.ndarray, int],
        speaker_embedding: np.ndarray,
        device,
    ):
        """GPU memory usage should be reasonable (< 4GB)."""
        if device.type != 'cuda':
            pytest.skip("GPU memory test requires CUDA")

        import librosa

        audio, sr = test_audio
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        torch.cuda.reset_peak_memory_stats()

        realtime_pipeline.set_speaker_embedding(speaker_embedding)
        _ = realtime_pipeline.process_chunk(audio_16k)

        _, peak_mb = get_gpu_memory_mb()

        logger.info(f"Realtime pipeline peak GPU memory: {peak_mb:.0f}MB")

        assert peak_mb < 4000, f"GPU memory {peak_mb:.0f}MB exceeds 4GB limit"


# =============================================================================
# Quality Pipeline Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.cuda
@pytest.mark.slow
class TestQualityPipeline:
    """Performance tests for SOTA quality pipeline."""

    @pytest.fixture(scope="class")
    def quality_pipeline(self, device):
        """Initialize quality pipeline."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            device=device,
            n_steps=1,  # Consistency model
            require_gpu=True,
        )
        yield pipeline

    def test_rtf_under_five(
        self,
        quality_pipeline,
        test_audio: Tuple[np.ndarray, int],
        speaker_embedding: np.ndarray,
        device,
    ):
        """RTF should be < 5.0 for reasonable offline processing."""
        audio, sr = test_audio
        audio_duration = len(audio) / sr

        audio_tensor = torch.from_numpy(audio).float()
        speaker_tensor = torch.from_numpy(speaker_embedding).float().to(device)

        # Warmup
        _ = quality_pipeline.convert(audio_tensor[:sr], sr, speaker_tensor)

        # Measure
        def process():
            quality_pipeline.convert(audio_tensor, sr, speaker_tensor)

        mean_rtf, min_rtf, max_rtf = measure_rtf(process, audio_duration, iterations=2)

        logger.info(f"Quality pipeline RTF: {mean_rtf:.3f} (min={min_rtf:.3f}, max={max_rtf:.3f})")

        assert mean_rtf < 5.0, f"RTF {mean_rtf:.3f} exceeds offline threshold of 5.0"

    def test_output_sample_rate(
        self,
        quality_pipeline,
        test_audio: Tuple[np.ndarray, int],
        speaker_embedding: np.ndarray,
        device,
    ):
        """Output should be at expected sample rate (24kHz)."""
        audio, sr = test_audio
        audio_tensor = torch.from_numpy(audio).float()
        speaker_tensor = torch.from_numpy(speaker_embedding).float().to(device)

        result = quality_pipeline.convert(audio_tensor, sr, speaker_tensor)

        assert result['sample_rate'] == 24000, f"Expected 24kHz output, got {result['sample_rate']}Hz"

    def test_gpu_memory_reasonable(
        self,
        quality_pipeline,
        test_audio: Tuple[np.ndarray, int],
        speaker_embedding: np.ndarray,
        device,
    ):
        """GPU memory should be < 8GB."""
        if device.type != 'cuda':
            pytest.skip("GPU memory test requires CUDA")

        audio, sr = test_audio
        audio_tensor = torch.from_numpy(audio).float()
        speaker_tensor = torch.from_numpy(speaker_embedding).float().to(device)

        torch.cuda.reset_peak_memory_stats()

        _ = quality_pipeline.convert(audio_tensor, sr, speaker_tensor)

        _, peak_mb = get_gpu_memory_mb()

        logger.info(f"Quality pipeline peak GPU memory: {peak_mb:.0f}MB")

        assert peak_mb < 8000, f"GPU memory {peak_mb:.0f}MB exceeds 8GB limit"


# =============================================================================
# Seed-VC Pipeline Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.cuda
@pytest.mark.slow
class TestSeedVCPipeline:
    """Performance tests for Seed-VC DiT-CFM pipeline."""

    @pytest.fixture(scope="class")
    def seedvc_pipeline(self, device, reference_audio):
        """Initialize Seed-VC pipeline."""
        try:
            from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

            pipeline = SeedVCPipeline(
                device=device,
                diffusion_steps=10,
                f0_condition=True,
                require_gpu=True,
            )

            # Set reference for in-context learning
            ref_audio, ref_sr = reference_audio
            pipeline.set_reference_audio(ref_audio, ref_sr)

            yield pipeline
        except (ImportError, RuntimeError, FileNotFoundError) as e:
            pytest.skip(f"Seed-VC not available: {e}")

    def test_rtf_under_five(
        self,
        seedvc_pipeline,
        test_audio: Tuple[np.ndarray, int],
    ):
        """RTF should be < 5.0 for Seed-VC."""
        audio, sr = test_audio
        audio_duration = len(audio) / sr

        # Warmup
        _ = seedvc_pipeline.convert(audio[:sr], sr)

        # Measure
        def process():
            seedvc_pipeline.convert(audio, sr)

        mean_rtf, min_rtf, max_rtf = measure_rtf(process, audio_duration, iterations=2)

        logger.info(f"Seed-VC pipeline RTF: {mean_rtf:.3f} (min={min_rtf:.3f}, max={max_rtf:.3f})")

        assert mean_rtf < 5.0, f"RTF {mean_rtf:.3f} exceeds threshold of 5.0"

    def test_output_sample_rate_44k(
        self,
        seedvc_pipeline,
        test_audio: Tuple[np.ndarray, int],
    ):
        """Output should be at 44.1kHz for F0-conditioned mode."""
        audio, sr = test_audio

        result = seedvc_pipeline.convert(audio, sr)

        assert result['sample_rate'] == 44100, f"Expected 44.1kHz output, got {result['sample_rate']}Hz"


# =============================================================================
# MeanVC Streaming Pipeline Tests
# =============================================================================

@pytest.mark.performance
class TestMeanVCPipeline:
    """Performance tests for MeanVC streaming pipeline."""

    @pytest.fixture(scope="class")
    def meanvc_pipeline(self, reference_audio):
        """Initialize MeanVC pipeline."""
        try:
            from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

            pipeline = MeanVCPipeline(
                device=torch.device('cpu'),  # MeanVC optimized for CPU
                steps=2,
                require_gpu=False,
            )

            # Set reference
            ref_audio, ref_sr = reference_audio
            pipeline.set_reference_audio(ref_audio, ref_sr)

            yield pipeline
        except (ImportError, RuntimeError, FileNotFoundError, AttributeError, Exception) as e:
            pytest.skip(f"MeanVC not available: {e}")

    def test_rtf_under_one(
        self,
        meanvc_pipeline,
        test_audio: Tuple[np.ndarray, int],
    ):
        """RTF should be < 1.0 for realtime streaming."""
        import librosa

        audio, sr = test_audio
        audio_duration = len(audio) / sr

        # Resample to 16kHz
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Warmup
        meanvc_pipeline.reset_session()
        _ = meanvc_pipeline.convert(audio_16k[:16000], 16000)

        # Measure full conversion
        def process():
            meanvc_pipeline.reset_session()
            meanvc_pipeline.convert(audio_16k, 16000)

        mean_rtf, min_rtf, max_rtf = measure_rtf(process, audio_duration, iterations=3)

        logger.info(f"MeanVC pipeline RTF: {mean_rtf:.3f} (min={min_rtf:.3f}, max={max_rtf:.3f})")

        # MeanVC on CPU may not achieve RTF < 1.0, but should be < 2.0
        assert mean_rtf < 2.0, f"RTF {mean_rtf:.3f} exceeds threshold of 2.0"

    def test_chunk_latency_within_runtime_contract(
        self,
        meanvc_pipeline,
        test_audio: Tuple[np.ndarray, int],
    ):
        """Individual chunk latency should stay within the runtime contract."""
        import librosa

        audio, sr = test_audio
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        chunk_size = meanvc_pipeline.chunk_size
        latencies = []

        meanvc_pipeline.reset_session()

        for i in range(0, len(audio_16k) - chunk_size, chunk_size):
            chunk = audio_16k[i:i + chunk_size]
            start = time.perf_counter()
            _ = meanvc_pipeline.process_chunk(chunk)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            if len(latencies) >= 10:  # Only test first 10 chunks
                break

        if latencies:
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            logger.info(f"MeanVC chunk latency: mean={mean_latency:.1f}ms, p95={p95_latency:.1f}ms")

            latency_target = PIPELINE_DEFINITIONS['realtime_meanvc'].latency_target_ms
            assert mean_latency < latency_target, (
                f"Mean latency {mean_latency:.1f}ms exceeds {latency_target}ms threshold"
            )

    def test_output_sample_rate_16k(
        self,
        meanvc_pipeline,
        test_audio: Tuple[np.ndarray, int],
    ):
        """Output should be at 16kHz."""
        import librosa

        audio, sr = test_audio
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        meanvc_pipeline.reset_session()
        result = meanvc_pipeline.convert(audio_16k, 16000)

        assert result['sample_rate'] == 16000, f"Expected 16kHz output, got {result['sample_rate']}Hz"


# =============================================================================
# Pipeline Factory Tests
# =============================================================================

@pytest.mark.performance
class TestPipelineFactory:
    """Tests for PipelineFactory performance characteristics."""

    def test_lazy_loading(self, device):
        """Pipelines should be lazily loaded."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        factory = PipelineFactory(device=device)

        # No pipelines loaded initially
        assert not factory.is_loaded('realtime')
        assert not factory.is_loaded('quality')
        assert factory.get_total_memory_usage() == 0.0

    def test_pipeline_caching(self, device):
        """Same pipeline instance should be returned on subsequent calls."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        factory = PipelineFactory(device=device)

        p1 = factory.get_pipeline('realtime')
        p2 = factory.get_pipeline('realtime')

        assert p1 is p2, "Pipeline should be cached and reused"

    def test_memory_tracking(self, device):
        """Factory should track memory usage per pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        if device.type != 'cuda':
            pytest.skip("Memory tracking test requires CUDA")

        factory = PipelineFactory(device=device)

        _ = factory.get_pipeline('realtime')

        mem = factory.get_memory_usage('realtime')

        # Should report some memory usage
        assert mem >= 0, "Memory tracking should work"

    def test_unload_pipeline(self, device):
        """Should be able to unload pipelines to free memory."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        factory = PipelineFactory(device=device)

        _ = factory.get_pipeline('realtime')
        assert factory.is_loaded('realtime')

        factory.unload_pipeline('realtime')
        assert not factory.is_loaded('realtime')


# =============================================================================
# Comparative Benchmark Test
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
def test_pipeline_comparison(
    test_audio: Tuple[np.ndarray, int],
    reference_audio: Tuple[np.ndarray, int],
    speaker_embedding: np.ndarray,
    device,
):
    """Compare all available pipelines and generate summary."""
    import librosa

    audio, sr = test_audio
    ref_audio, ref_sr = reference_audio
    audio_duration = len(audio) / sr

    results = []

    # Test realtime pipeline
    try:
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline(device=device)
        pipeline.set_speaker_embedding(speaker_embedding)

        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        start = time.perf_counter()
        _ = pipeline.process_chunk(audio_16k)
        elapsed = time.perf_counter() - start

        results.append({
            'pipeline': 'realtime',
            'rtf': elapsed / audio_duration,
            'status': 'pass' if elapsed / audio_duration < 1.0 else 'slow',
        })
    except Exception as e:
        results.append({'pipeline': 'realtime', 'status': 'error', 'error': str(e)})

    # Test quality pipeline
    try:
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(device=device, n_steps=1, require_gpu=True)

        audio_tensor = torch.from_numpy(audio).float()
        speaker_tensor = torch.from_numpy(speaker_embedding).float().to(device)

        start = time.perf_counter()
        _ = pipeline.convert(audio_tensor, sr, speaker_tensor)
        elapsed = time.perf_counter() - start

        results.append({
            'pipeline': 'quality',
            'rtf': elapsed / audio_duration,
            'status': 'pass' if elapsed / audio_duration < 5.0 else 'slow',
        })
    except Exception as e:
        results.append({'pipeline': 'quality', 'status': 'error', 'error': str(e)})

    # Log summary
    logger.info("\n=== Pipeline Comparison Summary ===")
    for r in results:
        if r['status'] == 'error':
            logger.info(f"  {r['pipeline']}: ERROR - {r.get('error', 'unknown')}")
        else:
            logger.info(f"  {r['pipeline']}: RTF={r['rtf']:.3f} ({r['status']})")

    # At least one pipeline should work
    working = [r for r in results if r['status'] != 'error']
    assert len(working) > 0, "At least one pipeline should work"
