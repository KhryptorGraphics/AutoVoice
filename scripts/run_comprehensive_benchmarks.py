#!/usr/bin/env python3
"""
Comprehensive benchmark orchestration script.

Runs all performance benchmarks (pytest tests, pipeline profiling, CUDA kernel profiling)
and aggregates results into structured JSON files for multi-GPU comparison.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None


def check_environment() -> Dict[str, Any]:
    """
    Validate environment and collect GPU information.

    Returns:
        Dictionary with environment information
    """
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_available': torch is not None,
        'cuda_available': False,
        'gpu_info': None
    }

    if torch is not None:
        env_info['pytorch_version'] = torch.__version__
        env_info['cuda_available'] = torch.cuda.is_available()

        if torch.cuda.is_available():
            env_info['gpu_info'] = {
                'name': torch.cuda.get_device_name(0),
                'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(0))),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'cuda_version': torch.version.cuda,
            }

            # Try to get driver version from nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    env_info['gpu_info']['driver_version'] = result.stdout.strip()
            except Exception:
                pass

    return env_info


def check_test_data(test_data_dir: Path) -> bool:
    """
    Check if test data exists, generate if missing.

    Args:
        test_data_dir: Test data directory

    Returns:
        True if test data is available
    """
    metadata_file = test_data_dir / 'metadata.json'

    if metadata_file.exists():
        print("✓ Test data found")
        return True

    print("⚠ Test data not found, generating...")

    # Run generate_benchmark_test_data.py
    script_path = Path(__file__).parent / 'generate_benchmark_test_data.py'

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), '--output-dir', str(test_data_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✓ Test data generated successfully")
            return True
        else:
            print(f"✗ Test data generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Test data generation error: {e}")
        return False


def run_pytest_benchmarks(output_dir: Path, quick: bool = False) -> Optional[Path]:
    """
    Run pytest performance tests.

    Args:
        output_dir: Output directory
        quick: Run quick benchmarks only

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running pytest performance tests...")
    print("="*60)

    output_file = output_dir / 'pytest_results.json'

    # Build pytest command
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_performance.py',
        '-v',
        '--json-report',
        f'--json-report-file={output_file}'
    ]

    if quick:
        # Run only quick tests
        cmd.extend(['-k', 'not slow'])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )

        print(result.stdout)

        if result.returncode in [0, 1]:  # 0 = all passed, 1 = some failed
            print(f"✓ Pytest results saved to: {output_file}")
            return output_file
        else:
            print(f"✗ Pytest execution failed with code {result.returncode}")
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print("✗ Pytest execution timed out")
        return None
    except Exception as e:
        print(f"✗ Pytest execution error: {e}")
        return None


def run_pipeline_profiling(output_dir: Path) -> Optional[Path]:
    """
    Run pipeline profiling.

    Args:
        output_dir: Output directory

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running pipeline profiling...")
    print("="*60)

    output_file = output_dir / 'pipeline_profile.json'
    script_path = Path(__file__).parent / 'profile_performance.py'

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), '--output-dir', str(output_dir)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        print(result.stdout)

        if result.returncode == 0:
            # Move performance_breakdown.json to pipeline_profile.json
            source = output_dir / 'performance_breakdown.json'
            if source.exists():
                source.rename(output_file)
            print(f"✓ Pipeline profiling results saved to: {output_file}")
            return output_file
        else:
            print(f"✗ Pipeline profiling failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ Pipeline profiling error: {e}")
        return None


def run_cuda_kernel_profiling(output_dir: Path) -> Optional[Path]:
    """
    Run CUDA kernel profiling.

    Args:
        output_dir: Output directory

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running CUDA kernel profiling...")
    print("="*60)

    output_file = output_dir / 'cuda_kernels_profile.json'
    script_path = Path(__file__).parent / 'profile_cuda_kernels.py'

    try:
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                '--kernel', 'all',
                '--iterations', '100',
                '--output', str(output_file)
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        print(result.stdout)

        if result.returncode == 0:
            print(f"✓ CUDA kernel profiling results saved to: {output_file}")
            return output_file
        else:
            print(f"✗ CUDA kernel profiling failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ CUDA kernel profiling error: {e}")
        return None


def aggregate_results(output_dir: Path, env_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate all benchmark results.

    Args:
        output_dir: Output directory
        env_info: Environment information

    Returns:
        Aggregated benchmark summary
    """
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)

    summary = {
        'environment': env_info,
        'metrics': {},
        'files': {}
    }

    # Load pytest results
    pytest_file = output_dir / 'pytest_results.json'
    if pytest_file.exists():
        with open(pytest_file) as f:
            summary['files']['pytest'] = str(pytest_file)
            # Extract key metrics from pytest results
            # (actual extraction depends on pytest-json-report format)

    # Load pipeline profiling results
    pipeline_file = output_dir / 'pipeline_profile.json'
    if pipeline_file.exists():
        with open(pipeline_file) as f:
            pipeline_data = json.load(f)
            summary['files']['pipeline'] = str(pipeline_file)
            summary['metrics']['pipeline'] = pipeline_data

    # Load CUDA kernel profiling results
    cuda_file = output_dir / 'cuda_kernels_profile.json'
    if cuda_file.exists():
        with open(cuda_file) as f:
            cuda_data = json.load(f)
            summary['files']['cuda_kernels'] = str(cuda_file)
            summary['metrics']['cuda_kernels'] = cuda_data

    # Save summary
    summary_file = output_dir / 'benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Benchmark summary saved to: {summary_file}")

    return summary





def generate_report(summary: Dict[str, Any], output_dir: Path):
    """
    Generate human-readable benchmark report.

    Args:
        summary: Benchmark summary
        output_dir: Output directory
    """
    report_file = output_dir / 'benchmark_report.md'

    with open(report_file, 'w') as f:
        f.write("# Performance Benchmark Report\n\n")

        # Environment info
        f.write("## Environment\n\n")
        env = summary['environment']
        f.write(f"- **Timestamp**: {env['timestamp']}\n")
        f.write(f"- **Python**: {env['python_version'].split()[0]}\n")

        if env.get('pytorch_available'):
            f.write(f"- **PyTorch**: {env.get('pytorch_version', 'N/A')}\n")

        if env.get('gpu_info'):
            gpu = env['gpu_info']
            f.write(f"- **GPU**: {gpu['name']}\n")
            f.write(f"- **Compute Capability**: {gpu['compute_capability']}\n")
            f.write(f"- **VRAM**: {gpu['total_memory_gb']:.1f} GB\n")
            f.write(f"- **CUDA**: {gpu.get('cuda_version', 'N/A')}\n")
            f.write(f"- **Driver**: {gpu.get('driver_version', 'N/A')}\n")

        f.write("\n## Results\n\n")

        # Add metrics sections
        if 'pipeline' in summary.get('metrics', {}):
            f.write("### Pipeline Performance\n\n")
            f.write("See `pipeline_profile.json` for detailed results.\n\n")

        if 'cuda_kernels' in summary.get('metrics', {}):
            f.write("### CUDA Kernel Performance\n\n")
            f.write("See `cuda_kernels_profile.json` for detailed results.\n\n")

        f.write("## Files\n\n")
        for name, path in summary.get('files', {}).items():
            f.write(f"- **{name}**: `{path}`\n")

    print(f"✓ Benchmark report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive performance benchmarks'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('validation_results/benchmarks'),
        help='Output directory (default: validation_results/benchmarks)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )
    parser.add_argument(
        '--skip-pytest',
        action='store_true',
        help='Skip pytest performance tests'
    )
    parser.add_argument(
        '--skip-profiling',
        action='store_true',
        help='Skip pipeline profiling'
    )
    parser.add_argument(
        '--skip-cuda-kernels',
        action='store_true',
        help='Skip CUDA kernel profiling'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmarks only (1s, 5s audio)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full benchmarks including 60s audio'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("AutoVoice Comprehensive Benchmark Suite")
    print("="*60)

    # Check environment
    print("\nChecking environment...")
    env_info = check_environment()

    if not env_info['pytorch_available']:
        print("✗ PyTorch not available")
        return 1

    if not env_info['cuda_available']:
        print("⚠ CUDA not available, some benchmarks will be skipped")
    else:
        print(f"✓ GPU: {env_info['gpu_info']['name']}")

    # Save GPU info
    gpu_info_file = args.output_dir / 'gpu_info.json'
    with open(gpu_info_file, 'w') as f:
        json.dump(env_info, f, indent=2)

    # Check test data
    test_data_dir = Path('tests/data/benchmark')
    if not check_test_data(test_data_dir):
        print("✗ Test data preparation failed")
        return 1

    # Run benchmarks
    results = {}

    if not args.skip_pytest:
        pytest_result = run_pytest_benchmarks(args.output_dir, args.quick)
        results['pytest'] = pytest_result is not None

    if not args.skip_profiling:
        pipeline_result = run_pipeline_profiling(args.output_dir)
        results['pipeline'] = pipeline_result is not None

    if not args.skip_cuda_kernels and env_info['cuda_available']:
        cuda_result = run_cuda_kernel_profiling(args.output_dir)
        results['cuda_kernels'] = cuda_result is not None

    # Aggregate results
    summary = aggregate_results(args.output_dir, env_info)

    # Generate report
    generate_report(summary, args.output_dir)

    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Results:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print("\nFor detailed results, see:")
    print(f"  {args.output_dir / 'benchmark_report.md'}")
    print(f"  {args.output_dir / 'benchmark_summary.json'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
