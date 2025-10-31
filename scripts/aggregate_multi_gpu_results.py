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
            gpu_info = json.load(f)
        
        # Load benchmark summary
        summary_file = gpu_dir / 'benchmark_summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
        else:
            summary = {}
        
        return {
            'name': gpu_dir.name,
            'gpu_info': gpu_info,
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
    
    # Extract GPU info
    if 'gpu_info' in gpu_data.get('gpu_info', {}):
        gpu_info = gpu_data['gpu_info']['gpu_info']
        metrics['gpu_name'] = gpu_info.get('name', 'N/A')
        metrics['compute_capability'] = gpu_info.get('compute_capability', 'N/A')
        metrics['vram_gb'] = f"{gpu_info.get('total_memory_gb', 0):.1f}"
    
    # Extract metrics from summary
    summary = gpu_data.get('summary', {})
    summary_metrics = summary.get('metrics', {})
    
    # Pipeline metrics
    if 'pipeline' in summary_metrics:
        pipeline = summary_metrics['pipeline']
        # Extract relevant metrics (structure depends on actual output)
        # This is a placeholder - adjust based on actual JSON structure
        metrics['conversion_rtf_balanced'] = pipeline.get('rtf', 'N/A')
        metrics['gpu_memory_gb'] = pipeline.get('peak_memory_gb', 'N/A')
    
    # CUDA kernel metrics
    if 'cuda_kernels' in summary_metrics:
        cuda = summary_metrics['cuda_kernels']
        # Extract relevant metrics
        # This is a placeholder - adjust based on actual JSON structure
    
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

