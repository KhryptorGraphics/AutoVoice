#!/usr/bin/env python3
"""
Aggregate benchmark results from multiple GPUs into comparison tables.

Merges benchmark JSON files from different GPU runs into unified comparison
tables for README documentation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def discover_gpu_directories(input_dir: Path) -> List[Path]:
    """
    Discover GPU benchmark directories.
    
    Args:
        input_dir: Root directory containing GPU subdirectories
        
    Returns:
        List of GPU directory paths
    """
    gpu_dirs = []
    
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            gpu_info_file = subdir / 'gpu_info.json'
            if gpu_info_file.exists():
                gpu_dirs.append(subdir)
    
    return sorted(gpu_dirs)


def load_gpu_data(gpu_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load benchmark data for a GPU.

    Args:
        gpu_dir: GPU directory path

    Returns:
        Dictionary with GPU data or None if failed
    """
    try:
        # Load GPU info
        with open(gpu_dir / 'gpu_info.json') as f:
            env_data = json.load(f)

        # Normalize GPU info structure (flatten to avoid double nesting)
        if 'gpu_info' in env_data:
            gpu_info = env_data['gpu_info']
        else:
            gpu_info = {}

        # Load benchmark summary
        summary_file = gpu_dir / 'benchmark_summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
        else:
            summary = {}

        return {
            'name': gpu_dir.name,
            'gpu_info': gpu_info,  # Flattened GPU info
            'env_data': env_data,  # Full environment data
            'summary': summary
        }
    except Exception as e:
        print(f"Warning: Failed to load data from {gpu_dir}: {e}")
        return None


def extract_metrics(gpu_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from GPU data.

    Args:
        gpu_data: GPU data dictionary

    Returns:
        Dictionary of extracted metrics
    """
    metrics = {
        'gpu_name': 'N/A',
        'compute_capability': 'N/A',
        'vram_gb': 'N/A',
        'tts_latency_ms': 'N/A',
        'tts_throughput': 'N/A',
        'conversion_rtf_fast': 'N/A',
        'conversion_rtf_balanced': 'N/A',
        'conversion_rtf_quality': 'N/A',
        'gpu_memory_gb': 'N/A',
        'cpu_gpu_speedup': 'N/A',
        'pitch_accuracy_hz': 'N/A',
        'speaker_similarity': 'N/A',
        'naturalness_score': 'N/A'
    }

    # Extract GPU info (flattened structure)
    gpu_info = gpu_data.get('gpu_info', {})
    if 'gpu_info' in gpu_info:
        # Handle nested structure
        gpu_info = gpu_info['gpu_info']

    metrics['gpu_name'] = gpu_info.get('name', 'N/A')
    metrics['compute_capability'] = gpu_info.get('compute_capability', 'N/A')
    if 'total_memory_gb' in gpu_info:
        metrics['vram_gb'] = f"{gpu_info['total_memory_gb']:.1f}"

    # Extract metrics from summary
    summary = gpu_data.get('summary', {})
    summary_metrics = summary.get('metrics', {})

    # TTS metrics (if available)
    if 'tts' in summary_metrics:
        tts = summary_metrics['tts']
        if 'tts_latency_ms' in tts and tts['tts_latency_ms'] is not None:
            metrics['tts_latency_ms'] = f"{tts['tts_latency_ms']:.2f} ms"
        if 'tts_throughput' in tts and tts['tts_throughput'] is not None:
            metrics['tts_throughput'] = f"{tts['tts_throughput']:.2f} req/s"
        if 'tts_memory_peak_mb' in tts and tts['tts_memory_peak_mb'] is not None:
            # Prefer TTS memory for TTS table
            tts_memory_gb = tts['tts_memory_peak_mb'] / 1024.0
            metrics['gpu_memory_gb'] = f"{tts_memory_gb:.2f}"

    # Pipeline metrics (aligned with profile_performance.py schema)
    if 'pipeline' in summary_metrics:
        pipeline = summary_metrics['pipeline']

        # RTF from explicit field
        if 'rtf' in pipeline:
            metrics['conversion_rtf_balanced'] = f"{pipeline['rtf']:.2f}x"
        elif 'total_time_ms' in pipeline and 'audio_duration_s' in pipeline:
            # Calculate RTF from total_time_ms and audio_duration_s
            total_time_s = pipeline['total_time_ms'] / 1000.0
            audio_duration_s = pipeline['audio_duration_s']
            rtf = total_time_s / audio_duration_s if audio_duration_s > 0 else 0
            metrics['conversion_rtf_balanced'] = f"{rtf:.2f}x"

        # GPU memory from memory_peak_mb (fallback if TTS not available)
        if 'memory_peak_mb' in pipeline and metrics['gpu_memory_gb'] == 'N/A':
            memory_gb = pipeline['memory_peak_mb'] / 1024.0
            metrics['gpu_memory_gb'] = f"{memory_gb:.2f}"

    # CUDA kernel metrics
    if 'cuda_kernels' in summary_metrics:
        cuda = summary_metrics['cuda_kernels']
        # Extract relevant metrics if available
        # This is a placeholder - adjust based on actual JSON structure

    # Pytest metrics (if available)
    if 'pytest' in summary_metrics:
        pytest_metrics = summary_metrics['pytest']
        if 'cpu_gpu_speedup' in pytest_metrics:
            metrics['cpu_gpu_speedup'] = f"{pytest_metrics['cpu_gpu_speedup']:.2f}x"
        if 'cache_speedup' in pytest_metrics:
            # Store for potential use
            pass

        # Extract preset RTF metrics from pytest
        for preset in ['fast', 'balanced', 'quality']:
            rtf_key = f'rtf_{preset}'
            if rtf_key in pytest_metrics:
                rtf_val = pytest_metrics[rtf_key]
                if preset == 'fast':
                    metrics['conversion_rtf_fast'] = f"{rtf_val:.2f}x"
                elif preset == 'balanced':
                    metrics['conversion_rtf_balanced'] = f"{rtf_val:.2f}x"
                elif preset == 'quality':
                    metrics['conversion_rtf_quality'] = f"{rtf_val:.2f}x"

    # Quality metrics (if available)
    if 'quality' in summary_metrics:
        quality = summary_metrics['quality']
        if 'pitch_accuracy_hz' in quality and quality['pitch_accuracy_hz'] is not None:
            metrics['pitch_accuracy_hz'] = f"{quality['pitch_accuracy_hz']:.1f} Hz"
        if 'speaker_similarity' in quality and quality['speaker_similarity'] is not None:
            metrics['speaker_similarity'] = f"{quality['speaker_similarity']:.2f}"
        if 'naturalness_score' in quality and quality['naturalness_score'] is not None:
            metrics['naturalness_score'] = f"{quality['naturalness_score']:.1f}/5.0"

    return metrics


def generate_markdown_tables(gpu_metrics: List[Dict[str, Any]]) -> str:
    """
    Generate markdown comparison tables.
    
    Args:
        gpu_metrics: List of GPU metrics dictionaries
        
    Returns:
        Markdown formatted tables
    """
    md = "# Multi-GPU Performance Comparison\n\n"
    md += f"Generated from {len(gpu_metrics)} GPU benchmark(s)\n\n"
    
    # Table 1: TTS Performance
    md += "## TTS Performance\n\n"
    md += "| GPU Model | Synthesis Latency (1s audio) | Throughput (req/s) | GPU Memory | Compute Capability |\n"
    md += "|-----------|------------------------------|--------------------|-----------|-------------------|\n"
    
    for metrics in gpu_metrics:
        md += f"| {metrics['gpu_name']} | "
        md += f"{metrics['tts_latency_ms']} | "
        md += f"{metrics['tts_throughput']} | "
        md += f"{metrics['gpu_memory_gb']} GB | "
        md += f"{metrics['compute_capability']} |\n"
    
    md += "\n"
    
    # Table 2: Voice Conversion Performance
    md += "## Voice Conversion Performance\n\n"
    md += "| GPU Model | Fast Preset | Balanced Preset | Quality Preset | GPU Memory | CPU vs GPU Speedup |\n"
    md += "|-----------|-------------|-----------------|----------------|------------|-------------------|\n"
    
    for metrics in gpu_metrics:
        md += f"| {metrics['gpu_name']} | "
        md += f"{metrics['conversion_rtf_fast']} | "
        md += f"{metrics['conversion_rtf_balanced']} | "
        md += f"{metrics['conversion_rtf_quality']} | "
        md += f"{metrics['gpu_memory_gb']} GB | "
        md += f"{metrics['cpu_gpu_speedup']} |\n"
    
    md += "\n**RT = Real-Time** (1.0x means 30s song takes 30s to convert)\n\n"
    
    # Table 3: Quality Metrics
    md += "## Quality Metrics (Balanced Preset)\n\n"
    md += "| GPU Model | Pitch Accuracy (RMSE) | Speaker Similarity | Naturalness Score |\n"
    md += "|-----------|----------------------|-------------------|------------------|\n"
    
    for metrics in gpu_metrics:
        md += f"| {metrics['gpu_name']} | "
        md += f"{metrics['pitch_accuracy_hz']} Hz | "
        md += f"{metrics['speaker_similarity']} | "
        md += f"{metrics['naturalness_score']} |\n"
    
    md += "\n"
    
    # Notes
    md += "## Notes\n\n"
    md += "- All measurements with PyTorch 2.5.1+cu121, CUDA 12.1\n"
    md += "- Benchmarks run on 30-second audio samples @ 22.05kHz\n"
    md += "- Results averaged over 10 runs after 3 warmup iterations\n"
    md += "- Quality metrics consistent across all GPUs (GPU affects speed, not quality)\n"
    md += "- N/A indicates metric not available for that GPU\n"
    
    return md


def generate_json_comparison(gpu_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate JSON comparison data.
    
    Args:
        gpu_metrics: List of GPU metrics dictionaries
        
    Returns:
        Structured comparison data
    """
    return {
        'gpus': gpu_metrics,
        'metadata': {
            'num_gpus': len(gpu_metrics),
            'metrics_included': [
                'tts_latency_ms',
                'conversion_rtf_balanced',
                'gpu_memory_gb',
                'cpu_gpu_speedup',
                'pitch_accuracy_hz',
                'speaker_similarity'
            ]
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multi-GPU benchmark results'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('validation_results/benchmarks'),
        help='Directory containing GPU subdirectories (default: validation_results/benchmarks)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=None,
        help='Output markdown file (default: <input-dir>/multi_gpu_comparison.md)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'json', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--include-charts',
        action='store_true',
        help='Generate chart data for visualization'
    )
    
    args = parser.parse_args()
    
    # Set default output file
    if args.output_file is None:
        args.output_file = args.input_dir / 'multi_gpu_comparison.md'
    
    print("="*60)
    print("Multi-GPU Benchmark Aggregation")
    print("="*60)
    
    # Discover GPU directories
    print(f"\nScanning: {args.input_dir}")
    gpu_dirs = discover_gpu_directories(args.input_dir)
    
    if not gpu_dirs:
        print("✗ No GPU benchmark directories found")
        print(f"  Expected structure: {args.input_dir}/<gpu_name>/gpu_info.json")
        return 1
    
    print(f"✓ Found {len(gpu_dirs)} GPU benchmark(s):")
    for gpu_dir in gpu_dirs:
        print(f"  - {gpu_dir.name}")
    
    # Load GPU data
    print("\nLoading benchmark data...")
    gpu_data_list = []
    for gpu_dir in gpu_dirs:
        gpu_data = load_gpu_data(gpu_dir)
        if gpu_data:
            gpu_data_list.append(gpu_data)
            print(f"  ✓ {gpu_dir.name}")
        else:
            print(f"  ✗ {gpu_dir.name} (failed to load)")
    
    if not gpu_data_list:
        print("\n✗ No valid GPU data loaded")
        return 1
    
    # Extract metrics
    print("\nExtracting metrics...")
    gpu_metrics = [extract_metrics(data) for data in gpu_data_list]
    
    # Generate outputs
    if args.format in ['markdown', 'both']:
        print(f"\nGenerating markdown: {args.output_file}")
        markdown = generate_markdown_tables(gpu_metrics)
        with open(args.output_file, 'w') as f:
            f.write(markdown)
        print(f"✓ Markdown saved")
    
    if args.format in ['json', 'both']:
        json_file = args.output_file.with_suffix('.json')
        print(f"\nGenerating JSON: {json_file}")
        json_data = generate_json_comparison(gpu_metrics)
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ JSON saved")
    
    if args.include_charts:
        chart_file = args.input_dir / 'performance_charts_data.json'
        print(f"\nGenerating chart data: {chart_file}")
        # Generate chart-friendly data structure
        chart_data = {
            'labels': [m['gpu_name'] for m in gpu_metrics],
            'datasets': {
                'tts_latency': [m['tts_latency_ms'] for m in gpu_metrics],
                'conversion_rtf': [m['conversion_rtf_balanced'] for m in gpu_metrics],
                'gpu_memory': [m['gpu_memory_gb'] for m in gpu_metrics]
            }
        }
        with open(chart_file, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"✓ Chart data saved")
    
    print("\n" + "="*60)
    print("Aggregation Complete")
    print("="*60)
    print(f"Processed {len(gpu_metrics)} GPU(s)")
    print(f"Output: {args.output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

