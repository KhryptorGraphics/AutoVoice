#!/usr/bin/env python3
"""Task 6.3: Benchmark GPU memory usage for both pipelines.

Measures peak GPU memory consumption to verify both pipelines fit in 64GB budget.
"""

import os
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import numpy as np
import librosa

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig
from quality_pipeline import QualityVoiceConverter, QualityConfig


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 / 1024


def benchmark_realtime_memory():
    """Benchmark realtime pipeline memory usage."""
    print("\n" + "=" * 70)
    print("REALTIME PIPELINE MEMORY BENCHMARK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory benchmark")
        return None

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    baseline = get_gpu_memory_mb()
    print(f"\nBaseline GPU memory: {baseline:.1f} MB")

    # Initialize converter
    config = RealtimeConfig(
        sample_rate=22050,
        chunk_size_ms=100,
        fp16=True,
        device="cuda"
    )
    converter = RealtimeVoiceConverter(config)
    init_memory = get_gpu_memory_mb()
    print(f"After initialization: {init_memory:.1f} MB (+{init_memory - baseline:.1f} MB)")

    # Load test audio
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=10.0)
    speaker_embedding = np.random.randn(256).astype(np.float32)

    # Convert (this will load models)
    print("\nConverting...")
    converted, _ = converter.convert_full(audio, sr, speaker_embedding)

    peak_memory = get_gpu_memory_mb()
    print(f"Peak memory during conversion: {peak_memory:.1f} MB (+{peak_memory - baseline:.1f} MB)")

    # Unload
    converter.unload()
    torch.cuda.empty_cache()
    gc.collect()

    after_unload = get_gpu_memory_mb()
    print(f"After unload: {after_unload:.1f} MB (recovered {peak_memory - after_unload:.1f} MB)")

    return {
        "baseline": baseline,
        "peak": peak_memory,
        "after_unload": after_unload,
        "net_usage": peak_memory - baseline
    }


def benchmark_quality_memory():
    """Benchmark quality pipeline memory usage."""
    print("\n" + "=" * 70)
    print("QUALITY PIPELINE MEMORY BENCHMARK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory benchmark")
        return None

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    baseline = get_gpu_memory_mb()
    print(f"\nBaseline GPU memory: {baseline:.1f} MB")

    # Initialize converter
    config = QualityConfig(
        sample_rate=44100,
        diffusion_steps=30,
        fp16=True,
        device="cuda"
    )
    converter = QualityVoiceConverter(config)
    init_memory = get_gpu_memory_mb()
    print(f"After initialization: {init_memory:.1f} MB (+{init_memory - baseline:.1f} MB)")

    # Load test audio
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=10.0)
    reference = librosa.load("data/separated_youtube/conor_maynard/08NWh97_DME_vocals.wav",
                            sr=None, mono=True, duration=10.0)[0]

    # Convert (this will load models)
    print("\nConverting...")
    converted, _ = converter.convert(audio, sr, reference, sr)

    peak_memory = get_gpu_memory_mb()
    print(f"Peak memory during conversion: {peak_memory:.1f} MB (+{peak_memory - baseline:.1f} MB)")

    # Unload
    converter.unload()
    torch.cuda.empty_cache()
    gc.collect()

    after_unload = get_gpu_memory_mb()
    print(f"After unload: {after_unload:.1f} MB (recovered {peak_memory - after_unload:.1f} MB)")

    return {
        "baseline": baseline,
        "peak": peak_memory,
        "after_unload": after_unload,
        "net_usage": peak_memory - baseline
    }


def main():
    os.chdir(Path(__file__).parent.parent)

    print("\n" + "=" * 70)
    print("  TASK 6.3: GPU MEMORY BENCHMARK")
    print("  Testing memory usage for Realtime vs Quality pipelines")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available - cannot benchmark GPU memory")
        print("Skipping memory benchmark")
        return 0

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    print(f"\nTotal GPU memory: {total_memory:.1f} GB")
    print(f"Target budget: 64 GB")

    # Benchmark realtime pipeline
    realtime_results = benchmark_realtime_memory()

    # Clear before next benchmark
    torch.cuda.empty_cache()
    gc.collect()

    # Benchmark quality pipeline
    quality_results = benchmark_quality_memory()

    # Summary
    print("\n" + "=" * 70)
    print("MEMORY USAGE SUMMARY")
    print("=" * 70)

    if realtime_results and quality_results:
        print(f"\n{'Pipeline':<20} {'Peak (MB)':<15} {'Peak (GB)':<15} {'% of 64GB':<15}")
        print("-" * 70)

        realtime_peak_gb = realtime_results['peak'] / 1024
        realtime_pct = (realtime_peak_gb / 64) * 100
        print(f"{'Realtime':<20} {realtime_results['peak']:<15.1f} "
              f"{realtime_peak_gb:<15.2f} {realtime_pct:<15.1f}%")

        quality_peak_gb = quality_results['peak'] / 1024
        quality_pct = (quality_peak_gb / 64) * 100
        print(f"{'Quality':<20} {quality_results['peak']:<15.1f} "
              f"{quality_peak_gb:<15.2f} {quality_pct:<15.1f}%")

        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)

        if realtime_peak_gb < 64 and quality_peak_gb < 64:
            print(f"\n✓ Both pipelines fit within 64GB budget")
            print(f"  Realtime: {realtime_peak_gb:.2f} GB / 64 GB ({realtime_pct:.1f}%)")
            print(f"  Quality:  {quality_peak_gb:.2f} GB / 64 GB ({quality_pct:.1f}%)")
        else:
            print(f"\n✗ One or more pipelines exceed 64GB budget")

        print("\nMemory Recovery:")
        print(f"  Realtime: {realtime_results['peak'] - realtime_results['after_unload']:.1f} MB recovered")
        print(f"  Quality:  {quality_results['peak'] - quality_results['after_unload']:.1f} MB recovered")

    print("\n" + "=" * 70)
    print("✓ TASK 6.3 COMPLETE")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
