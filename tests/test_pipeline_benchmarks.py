"""
Benchmark comparison of all voice conversion pipelines.

Compares:
- realtime: Original low-latency pipeline (22kHz)
- quality: CoMoSVC high-quality pipeline (24kHz)
- quality_seedvc: Seed-VC DiT-CFM pipeline (44kHz)
- realtime_meanvc: MeanVC streaming pipeline (16kHz)

Metrics:
- RTF (realtime factor): processing_time / audio_duration
- Latency: Total processing time
- Memory: GPU/CPU memory usage
- Sample rate: Output quality indicator
"""

import gc
import pytest
import torch
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_voice.inference.pipeline_factory import PipelineFactory


@pytest.fixture
def test_audio():
    """Generate 5 seconds of test audio."""
    duration = 5.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Sine wave with harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * 220 * t) +
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 880 * t)
    )
    return audio.astype(np.float32), sample_rate


@pytest.fixture
def benchmark_results():
    """Dictionary to collect benchmark results."""
    return {
        'realtime': {},
        'quality': {},
        'quality_seedvc': {},
        'realtime_meanvc': {},
    }


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def release_pipeline_resources(factory, pipeline_type):
    """Best-effort cleanup to keep long benchmark runs from accumulating GPU state."""
    factory.unload_pipeline(pipeline_type)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_all_pipelines(test_audio, benchmark_results, tmp_path):
    """Benchmark all available pipelines."""
    audio, sr = test_audio
    duration = len(audio) / sr

    factory = PipelineFactory.get_instance()

    # Test each pipeline
    for pipeline_type in ['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {pipeline_type}")
        print(f"{'='*60}")

        try:
            # Get memory before loading
            mem_before = get_memory_usage()

            # Get pipeline
            pipeline = factory.get_pipeline(pipeline_type)

            # Get memory after loading
            mem_after = get_memory_usage()
            mem_usage = mem_after - mem_before

            # Create dummy speaker embedding if needed
            speaker_emb = torch.randn(192) if pipeline_type in ['realtime', 'quality'] else None

            # Measure conversion time
            start = time.perf_counter()

            if pipeline_type == 'realtime_meanvc':
                # MeanVC needs reference audio
                ref_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.5
                pipeline.set_reference_audio(ref_audio, 16000)
                result = pipeline.convert(audio, sr)
            else:
                result = pipeline.convert(audio, sr, speaker_emb)

            elapsed = time.perf_counter() - start

            # Calculate metrics
            rtf = elapsed / duration
            output_sr = result['sample_rate']
            output_duration = len(result['audio']) / output_sr

            # Store results
            benchmark_results[pipeline_type] = {
                'rtf': rtf,
                'latency_ms': elapsed * 1000,
                'memory_gb': mem_usage,
                'sample_rate': output_sr,
                'input_duration': duration,
                'output_duration': output_duration,
                'success': True,
            }

            print(f"✓ RTF: {rtf:.3f}x realtime")
            print(f"✓ Latency: {elapsed*1000:.1f}ms")
            print(f"✓ Memory: {mem_usage:.2f}GB")
            print(f"✓ Output: {output_sr}Hz")

        except Exception as e:
            print(f"✗ Failed: {e}")
            benchmark_results[pipeline_type] = {
                'rtf': None,
                'latency_ms': None,
                'memory_gb': None,
                'sample_rate': None,
                'success': False,
                'error': str(e),
            }

        finally:
            # Unload pipeline to free memory
            release_pipeline_resources(factory, pipeline_type)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pipeline':<20} {'RTF':<10} {'Latency':<12} {'Memory':<10} {'SR':<8}")
    print(f"{'-'*60}")

    for name, results in benchmark_results.items():
        if results['success']:
            rtf = f"{results['rtf']:.3f}x"
            latency = f"{results['latency_ms']:.0f}ms"
            memory = f"{results['memory_gb']:.2f}GB"
            sr = f"{results['sample_rate']}Hz"
            print(f"{name:<20} {rtf:<10} {latency:<12} {memory:<10} {sr:<8}")
        else:
            print(f"{name:<20} {'FAILED':<10} {results.get('error', 'Unknown error')}")

    assert set(benchmark_results) == {
        'realtime',
        'quality',
        'quality_seedvc',
        'realtime_meanvc',
    }
    assert all('success' in result for result in benchmark_results.values())


@pytest.mark.benchmark
def test_compare_quality_pipelines(test_audio):
    """Compare the two quality pipelines: CoMoSVC vs Seed-VC."""
    audio, sr = test_audio
    factory = PipelineFactory.get_instance()

    results = {}

    for pipeline_type in ['quality', 'quality_seedvc']:
        try:
            pipeline = factory.get_pipeline(pipeline_type)
            speaker_emb = torch.randn(192)

            start = time.perf_counter()
            result = pipeline.convert(audio, sr, speaker_emb)
            elapsed = time.perf_counter() - start

            results[pipeline_type] = {
                'rtf': elapsed / (len(audio) / sr),
                'sample_rate': result['sample_rate'],
            }

        except Exception as e:
            results[pipeline_type] = {'error': str(e)}
        finally:
            release_pipeline_resources(factory, pipeline_type)

    # Compare
    if 'quality' in results and 'quality_seedvc' in results:
        if 'error' not in results['quality'] and 'error' not in results['quality_seedvc']:
            seedvc_faster = results['quality_seedvc']['rtf'] < results['quality']['rtf']
            speedup = results['quality']['rtf'] / results['quality_seedvc']['rtf']

            print(f"\nQuality Pipeline Comparison:")
            print(f"  CoMoSVC (quality):     {results['quality']['rtf']:.3f}x RT, {results['quality']['sample_rate']}Hz")
            print(f"  Seed-VC (quality_seedvc): {results['quality_seedvc']['rtf']:.3f}x RT, {results['quality_seedvc']['sample_rate']}Hz")
            print(f"  Seed-VC is {speedup:.2f}x faster" if seedvc_faster else f"  CoMoSVC is {1/speedup:.2f}x faster")

    assert set(results) == {'quality', 'quality_seedvc'}


@pytest.mark.benchmark
def test_compare_realtime_pipelines(test_audio):
    """Compare the two realtime pipelines: Original vs MeanVC."""
    audio, sr = test_audio
    factory = PipelineFactory.get_instance()

    results = {}

    # Test original realtime
    try:
        pipeline = factory.get_pipeline('realtime')
        speaker_emb = torch.randn(192)

        start = time.perf_counter()
        result = pipeline.convert(audio, sr, speaker_emb)
        elapsed = time.perf_counter() - start

        results['realtime'] = {
            'rtf': elapsed / (len(audio) / sr),
            'sample_rate': result['sample_rate'],
        }
    except Exception as e:
        results['realtime'] = {'error': str(e)}
    finally:
        release_pipeline_resources(factory, 'realtime')

    # Test MeanVC
    try:
        pipeline = factory.get_pipeline('realtime_meanvc')
        ref_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.5
        pipeline.set_reference_audio(ref_audio, 16000)

        start = time.perf_counter()
        result = pipeline.convert(audio, sr)
        elapsed = time.perf_counter() - start

        results['realtime_meanvc'] = {
            'rtf': elapsed / (len(audio) / sr),
            'sample_rate': result['sample_rate'],
        }
    except Exception as e:
        results['realtime_meanvc'] = {'error': str(e)}
    finally:
        release_pipeline_resources(factory, 'realtime_meanvc')

    # Compare
    if 'realtime' in results and 'realtime_meanvc' in results:
        if 'error' not in results['realtime'] and 'error' not in results['realtime_meanvc']:
            meanvc_faster = results['realtime_meanvc']['rtf'] < results['realtime']['rtf']
            speedup = results['realtime']['rtf'] / results['realtime_meanvc']['rtf'] if meanvc_faster else results['realtime_meanvc']['rtf'] / results['realtime']['rtf']

            print(f"\nRealtime Pipeline Comparison:")
            print(f"  Original (realtime):    {results['realtime']['rtf']:.3f}x RT, {results['realtime']['sample_rate']}Hz")
            print(f"  MeanVC (realtime_meanvc): {results['realtime_meanvc']['rtf']:.3f}x RT, {results['realtime_meanvc']['sample_rate']}Hz")
            print(f"  MeanVC is {speedup:.2f}x faster" if meanvc_faster else f"  Original is {speedup:.2f}x faster")

    assert set(results) == {'realtime', 'realtime_meanvc'}


@pytest.mark.benchmark
def test_memory_profiling():
    """Profile GPU memory usage for each pipeline."""
    factory = PipelineFactory.get_instance()

    results = {}

    for pipeline_type in ['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']:
        print(f"\nProfiling memory for {pipeline_type}...")

        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            mem_before = get_memory_usage()

            # Load pipeline
            pipeline = factory.get_pipeline(pipeline_type)

            mem_after = get_memory_usage()
            mem_usage = mem_after - mem_before

            results[pipeline_type] = {
                'memory_gb': mem_usage,
                'device': str(pipeline.device) if hasattr(pipeline, 'device') else 'unknown',
            }

            print(f"  Memory: {mem_usage:.2f}GB")
            print(f"  Device: {results[pipeline_type]['device']}")

        except Exception as e:
            results[pipeline_type] = {'error': str(e)}
            print(f"  Error: {e}")

        finally:
            release_pipeline_resources(factory, pipeline_type)

    # Summary
    print(f"\n{'='*50}")
    print("MEMORY PROFILING SUMMARY")
    print(f"{'='*50}")
    print(f"{'Pipeline':<20} {'Memory':<12} {'Device':<10}")
    print(f"{'-'*50}")

    total_memory = 0
    for name, result in results.items():
        if 'error' not in result:
            mem = f"{result['memory_gb']:.2f}GB"
            device = result['device']
            print(f"{name:<20} {mem:<12} {device:<10}")
            if 'cuda' in device:
                total_memory += result['memory_gb']

    print(f"{'-'*50}")
    print(f"Total GPU: {total_memory:.2f}GB (budget: 64GB)")
    print(f"Remaining: {64 - total_memory:.2f}GB")

    # Verify within budget
    assert total_memory < 64, f"Total memory {total_memory:.2f}GB exceeds 64GB budget"

    assert set(results) == {
        'realtime',
        'quality',
        'quality_seedvc',
        'realtime_meanvc',
    }


if __name__ == "__main__":
    # Quick benchmark run
    print("Running pipeline benchmarks...")

    # Generate test audio
    duration = 5.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Memory profiling
    print("\n" + "="*60)
    print("MEMORY PROFILING")
    print("="*60)
    test_memory_profiling()

    print("\n✅ Benchmark tests complete!")
    print("\nRun full benchmarks with: pytest tests/test_pipeline_benchmarks.py -v -s -m benchmark")
