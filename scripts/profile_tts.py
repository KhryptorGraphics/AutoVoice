#!/usr/bin/env python3
"""
TTS Performance Profiling Script

Benchmarks TTS synthesis latency, throughput, and memory usage.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def profile_tts_synthesis(
    text: str = "Hello, this is a test.",
    num_warmups: int = 3,
    num_iterations: int = 10,
    gpu_id: int = 0,
    output_dir: Path = Path(".")
) -> Dict[str, Any]:
    """
    Profile TTS synthesis performance.

    Args:
        text: Text to synthesize (should produce ~1 second of audio)
        num_warmups: Number of warmup iterations
        num_iterations: Number of measured iterations
        gpu_id: GPU device index
        output_dir: Output directory for results

    Returns:
        Dictionary with profiling results
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not available")
        return {'error': 'PyTorch not available'}

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        print(f"Using GPU device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")

    try:
        # Import TTS pipeline (adjust based on actual implementation)
        from src.auto_voice.inference.tts_pipeline import TTSPipeline
    except ImportError:
        print("Warning: TTSPipeline not available, using mock implementation")

        # Mock TTS pipeline for testing
        class TTSPipeline:
            def __init__(self, device):
                self.device = device

            def synthesize(self, text):
                """Mock synthesis - generates 1 second of audio"""
                sample_rate = 22050
                duration = 1.0
                audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
                time.sleep(0.01)  # Simulate processing time
                return audio, sample_rate

    # Initialize pipeline
    print("Initializing TTS pipeline...")
    pipeline = TTSPipeline(device=device)

    # Reset GPU memory stats
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup iterations
    print(f"Running {num_warmups} warmup iterations...")
    for i in range(num_warmups):
        audio, sr = pipeline.synthesize(text)
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    # Measured iterations
    print(f"Running {num_iterations} measured iterations...")
    latencies = []

    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

    for i in range(num_iterations):
        if device.startswith('cuda'):
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        audio, sr = pipeline.synthesize(text)

        if device.startswith('cuda'):
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed * 1000)  # Convert to ms

    # Calculate metrics
    avg_latency_ms = np.mean(latencies)
    throughput_req_s = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    # Get GPU memory stats
    if device.startswith('cuda'):
        peak_memory = torch.cuda.max_memory_allocated()
        memory_peak_mb = peak_memory / 1024 / 1024

        gpu_info = {
            'index': gpu_id,
            'name': torch.cuda.get_device_name(gpu_id),
            'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(gpu_id))),
            'total_memory_gb': torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        }
    else:
        memory_peak_mb = 0.0
        gpu_info = None

    # Build results
    results = {
        'tts_latency_ms': avg_latency_ms,
        'tts_throughput': throughput_req_s,
        'tts_memory_peak_mb': memory_peak_mb,
        'device': device,
        'gpu_info': gpu_info,
        'text_input': text,
        'num_iterations': num_iterations,
        'latencies_ms': {
            'mean': avg_latency_ms,
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies))
        }
    }

    print(f"\nTTS Profiling Results:")
    print(f"  Average Latency: {avg_latency_ms:.2f} ms")
    print(f"  Throughput: {throughput_req_s:.2f} req/s")
    if device.startswith('cuda'):
        print(f"  Peak GPU Memory: {memory_peak_mb:.2f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="TTS Performance Profiling")
    parser.add_argument(
        '--text',
        type=str,
        default="Hello, this is a test of the text to speech system.",
        help='Text to synthesize (default: produces ~1s audio)'
    )
    parser.add_argument(
        '--warmups',
        type=int,
        default=3,
        help='Number of warmup iterations (default: 3)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of measured iterations (default: 10)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device index (default: 0)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory for results (default: current directory)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer iterations (2 warmups, 5 iterations)'
    )

    args = parser.parse_args()

    # Adjust iterations for quick mode
    if args.quick:
        num_warmups = 2
        num_iterations = 5
        print("Quick mode: 2 warmups, 5 iterations")
    else:
        num_warmups = args.warmups
        num_iterations = args.iterations

    print("="*60)
    print("TTS Performance Profiling")
    print("="*60)

    # Run profiling
    results = profile_tts_synthesis(
        text=args.text,
        num_warmups=num_warmups,
        num_iterations=num_iterations,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir
    )

    if 'error' in results:
        print(f"\nError: {results['error']}")
        return 1

    # Save results
    output_file = args.output_dir / 'tts_profile.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
