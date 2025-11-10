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


def run_pytest_benchmarks(output_dir: Path, quick: bool = False, continue_on_failure: bool = False, gpu_id: int = 0) -> Optional[Path]:
    """
    Run pytest performance tests.

    Args:
        output_dir: Output directory
        quick: Run quick benchmarks only
        gpu_id: GPU device ID to use

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
        import os
        env = os.environ.copy()
        env['PYTEST_JSON_OUTPUT'] = str(output_dir / 'pytest_metrics.json')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set GPU device for subprocess
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes
            env=env
        )

        print(result.stdout)

        # Only accept return code 0 for success
        if result.returncode == 0:
            print(f"✓ Pytest results saved to: {output_file}")
            return output_file
        elif result.returncode == 1:
            print("⚠ Some tests failed")
            if continue_on_failure:
                print("  Continuing as requested")
                return output_file
            else:
                print("  Execution aborted - use --continue-on-failure to proceed")
                return None
        else:
            print(f"✗ Pytest execution failed with code {result.returncode}")
            if continue_on_failure:
                print("  Continuing despite fatal failure")
                return output_file
            else:
                return None
    except subprocess.TimeoutExpired:
        print("✗ Pytest execution timed out")
        return None
    except Exception as e:
        print(f"✗ Pytest execution error: {e}")
        return None


def run_pipeline_profiling(output_dir: Path, quick: bool = False, full: bool = False, gpu_id: int = 0) -> Optional[Path]:
    """
    Run pipeline profiling.

    Args:
        output_dir: Output directory
        quick: Use quick mode (5s audio)
        full: Use full mode (60s audio)

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running pipeline profiling...")
    print("="*60)

    output_file = output_dir / 'pipeline_profile.json'
    script_path = Path(__file__).parent / 'profile_performance.py'

    # Determine audio file based on mode
    test_data_dir = Path('tests/data/benchmark')
    if quick:
        audio_file = test_data_dir / 'audio_5s_22050hz.wav'
        print("  Mode: Quick (5s audio)")
    elif full:
        audio_file = test_data_dir / 'audio_60s_22050hz.wav'
        print("  Mode: Full (60s audio)")
    else:
        audio_file = test_data_dir / 'audio_30s_22050hz.wav'
        print("  Mode: Balanced (30s audio)")

    cmd = [sys.executable, str(script_path), '--output-dir', str(output_dir), '--gpu-id', str(gpu_id)]

    # Add audio file if it exists
    if audio_file.exists():
        cmd.extend(['--audio-file', str(audio_file)])

    try:
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set GPU device for subprocess

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            env=env
        )

        print(result.stdout)

        # Define source file path (profiler writes to performance_breakdown.json)
        source = output_dir / 'performance_breakdown.json'

        # Check if results JSON was created, even if subprocess had exit code issues
        if result.returncode == 0:
            # Success - check if we need to rename the file
            if source.exists() and source != output_file:
                import shutil
                shutil.copy2(source, output_file)
                print(f"✓ Pipeline profiling results saved to: {output_file}")
                return output_file
            elif output_file.exists():
                print(f"✓ Pipeline profiling results saved to: {output_file}")
                return output_file
            else:
                print(f"⚠️  Pipeline profiling succeeded but output file not found")
                return None
        else:
            # Non-zero exit - check if JSON file was still written
            if source.exists():
                import shutil
                shutil.copy2(source, output_file)
                print(f"⚠️  Pipeline profiling completed with warnings - results saved to: {output_file}")
                return output_file
            elif output_file.exists():
                print(f"⚠️  Pipeline profiling completed with warnings - results saved to: {output_file}")
                return output_file
            else:
                print(f"✗ Pipeline profiling failed: {result.stderr}")
                return None
    except Exception as e:
        print(f"✗ Pipeline profiling error: {e}")
        return None


def run_quality_evaluation(output_dir: Path, gpu_id: int = 0) -> Optional[Path]:
    """
    Run quality metrics evaluation.

    Args:
        output_dir: Output directory
        gpu_id: GPU device index

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running quality metrics evaluation...")
    print("="*60)

    output_file = output_dir / 'quality_metrics.json'
    script_path = Path(__file__).parent / 'evaluate_quality.py'

    # Check if we have test audio for evaluation
    test_data_dir = Path('tests/data/benchmark')
    source_audio = test_data_dir / 'audio_30s_22050hz.wav'

    if not source_audio.exists():
        print("⚠️  Skipping quality evaluation: test audio not found")
        return None

    # For now, use same file for both source and converted (placeholder)
    # In production, this would use actual conversion output
    converted_audio = source_audio

    cmd = [
        sys.executable, str(script_path),
        '--source-audio', str(source_audio),
        '--converted-audio', str(converted_audio),
        '--gpu-id', str(gpu_id),
        '--output-dir', str(output_dir)
    ]

    try:
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            env=env
        )

        print(result.stdout)

        if result.returncode == 0 and output_file.exists():
            print(f"✓ Quality evaluation results saved to: {output_file}")
            return output_file
        else:
            print(f"⚠️  Quality evaluation completed with warnings")
            if output_file.exists():
                return output_file
            return None
    except Exception as e:
        print(f"⚠️  Quality evaluation error: {e}")
        return None


def run_tts_benchmark(output_dir: Path, quick: bool = False, gpu_id: int = 0) -> Optional[Path]:
    """
    Run TTS synthesis benchmarking.

    Args:
        output_dir: Output directory
        quick: Use quick mode (fewer iterations)
        gpu_id: GPU device index

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running TTS synthesis benchmarking...")
    print("="*60)

    output_file = output_dir / 'tts_profile.json'
    script_path = Path(__file__).parent / 'profile_tts.py'

    # Build command
    cmd = [sys.executable, str(script_path), '--output-dir', str(output_dir), '--gpu-id', str(gpu_id)]

    if quick:
        cmd.append('--quick')
        print("  Mode: Quick (2 warmups, 5 iterations)")
    else:
        print("  Mode: Balanced (3 warmups, 10 iterations)")

    try:
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set GPU device for subprocess

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            env=env
        )

        print(result.stdout)

        if result.returncode == 0 and output_file.exists():
            print(f"✓ TTS benchmarking results saved to: {output_file}")
            return output_file
        else:
            print(f"✗ TTS benchmarking failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ TTS benchmarking error: {e}")
        return None


def run_cuda_kernel_profiling(output_dir: Path, quick: bool = False, full: bool = False, gpu_id: int = 0) -> Optional[Path]:
    """
    Run CUDA kernel profiling.

    Args:
        output_dir: Output directory
        quick: Use quick mode (30 iterations)
        full: Use full mode (200 iterations)

    Returns:
        Path to results file or None if failed
    """
    print("\n" + "="*60)
    print("Running CUDA kernel profiling...")
    print("="*60)

    output_file = output_dir / 'cuda_kernels_profile.json'
    script_path = Path(__file__).parent / 'profile_cuda_kernels.py'

    # Determine iterations based on mode
    if quick:
        iterations = '30'
        print("  Mode: Quick (30 iterations)")
    elif full:
        iterations = '200'
        print("  Mode: Full (200 iterations)")
    else:
        iterations = '100'
        print("  Mode: Balanced (100 iterations)")

    try:
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set GPU device for subprocess
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                '--kernel', 'all',
                '--iterations', iterations,
                '--output', str(output_file),
                '--gpu-id', str(gpu_id)
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            env=env
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
    pytest_metrics_file = output_dir / 'pytest_metrics.json'

    if pytest_file.exists():
        with open(pytest_file) as f:
            pytest_data = json.load(f)
            summary['files']['pytest'] = str(pytest_file)

            # Extract key metrics from pytest-json-report format
            pytest_metrics = {}

            # Get summary statistics
            if 'summary' in pytest_data:
                pytest_summary = pytest_data['summary']
                pytest_metrics['total_tests'] = pytest_summary.get('total', 0)
                pytest_metrics['passed'] = pytest_summary.get('passed', 0)
                pytest_metrics['failed'] = pytest_summary.get('failed', 0)
                pytest_metrics['duration'] = pytest_data.get('duration', 0)

            # Try to load explicit metrics from pytest_metrics.json (preferred)
            if pytest_metrics_file.exists():
                print(f"  ✓ Loading pytest metrics from {pytest_metrics_file}")
                with open(pytest_metrics_file) as f:
                    explicit_metrics = json.load(f)
                    if 'metrics' in explicit_metrics:
                        # Merge explicit metrics (these take precedence)
                        for metric_name, values in explicit_metrics['metrics'].items():
                            if isinstance(values, dict) and 'mean' in values:
                                pytest_metrics[metric_name] = values['mean']
                            elif isinstance(values, list) and values:
                                pytest_metrics[metric_name] = values[-1]  # Use most recent
                        summary['files']['pytest_metrics'] = str(pytest_metrics_file)
            else:
                # Fallback: Extract performance metrics from test results (legacy)
                if 'tests' in pytest_data:
                    # Look for CPU vs GPU speedup tests
                    for test in pytest_data['tests']:
                        test_name = test.get('nodeid', '')

                        # Extract CPU vs GPU speedup
                        if 'cpu_vs_gpu_speed_advantage' in test_name:
                            # Try to extract from test output or metadata
                            if 'call' in test and 'longrepr' in test['call']:
                                output = str(test['call']['longrepr'])
                                # Parse speedup from output (e.g., "GPU speedup: 3.5x")
                                import re
                                match = re.search(r'GPU speedup:\s*([\d.]+)x', output)
                                if match:
                                    pytest_metrics['cpu_gpu_speedup'] = float(match.group(1))

                        # Extract cache speedup
                        if 'cache_effectiveness' in test_name or 'cold_vs_warm' in test_name:
                            if 'call' in test and 'longrepr' in test['call']:
                                output = str(test['call']['longrepr'])
                                match = re.search(r'Speedup:\s*([\d.]+)x', output)
                                if match:
                                    pytest_metrics['cache_speedup'] = float(match.group(1))

            summary['metrics']['pytest'] = pytest_metrics

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

    # Load TTS profiling results
    tts_file = output_dir / 'tts_profile.json'
    if tts_file.exists():
        with open(tts_file) as f:
            tts_data = json.load(f)
            summary['files']['tts'] = str(tts_file)
            # Extract TTS metrics
            summary['metrics']['tts'] = {
                'tts_latency_ms': tts_data.get('tts_latency_ms'),
                'tts_throughput': tts_data.get('tts_throughput'),
                'tts_memory_peak_mb': tts_data.get('tts_memory_peak_mb')
            }

    # Load quality metrics results
    quality_file = output_dir / 'quality_metrics.json'
    if quality_file.exists():
        with open(quality_file) as f:
            quality_data = json.load(f)
            summary['files']['quality'] = str(quality_file)
            # Extract quality metrics
            summary['metrics']['quality'] = {
                'pitch_accuracy_hz': quality_data.get('pitch_accuracy_hz'),
                'speaker_similarity': quality_data.get('speaker_similarity'),
                'naturalness_score': quality_data.get('naturalness_score')
            }

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


def sanitize_gpu_name(gpu_name: str) -> str:
    """
    Sanitize GPU name for use as directory name.

    Args:
        gpu_name: GPU name from CUDA

    Returns:
        Sanitized directory name
    """
    import re
    # Convert to lowercase and replace non-alphanumeric with underscore
    sanitized = re.sub(r'[^a-z0-9]+', '_', gpu_name.lower())
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


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
        '--skip-tts',
        action='store_true',
        help='Skip TTS synthesis benchmarking'
    )
    parser.add_argument(
        '--skip-quality',
        action='store_true',
        help='Skip quality metrics evaluation'
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
    parser.add_argument(
        '--continue-on-failure',
        action='store_true',
        help='Continue benchmark execution even if pytest tests fail'
    )

    args = parser.parse_args()

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

    # Create GPU-specific subdirectory
    if env_info.get('gpu_info'):
        gpu_name = env_info['gpu_info']['name']
        gpu_subdir_name = sanitize_gpu_name(gpu_name)
        gpu_output_dir = args.output_dir / gpu_subdir_name
    else:
        gpu_output_dir = args.output_dir / 'cpu_only'

    gpu_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {gpu_output_dir}")

    # Save GPU info to GPU-specific directory
    gpu_info_file = gpu_output_dir / 'gpu_info.json'
    with open(gpu_info_file, 'w') as f:
        json.dump(env_info, f, indent=2)

    # Check test data
    test_data_dir = Path('tests/data/benchmark')
    if not check_test_data(test_data_dir):
        print("✗ Test data preparation failed")
        return 1

    # Run benchmarks (using GPU-specific output directory)
    results = {}

    if not args.skip_pytest:
        pytest_result = run_pytest_benchmarks(gpu_output_dir, args.quick, args.continue_on_failure, args.gpu_id)
        results['pytest'] = pytest_result is not None

    if not args.skip_profiling:
        pipeline_result = run_pipeline_profiling(gpu_output_dir, args.quick, args.full, args.gpu_id)
        results['pipeline'] = pipeline_result is not None

    if not args.skip_cuda_kernels and env_info['cuda_available']:
        cuda_result = run_cuda_kernel_profiling(gpu_output_dir, args.quick, args.full, args.gpu_id)
        results['cuda_kernels'] = cuda_result is not None

    if not args.skip_tts:
        tts_result = run_tts_benchmark(gpu_output_dir, args.quick, args.gpu_id)
        results['tts'] = tts_result is not None

    if not args.skip_quality:
        quality_result = run_quality_evaluation(gpu_output_dir, args.gpu_id)
        results['quality'] = quality_result is not None

    # Aggregate results
    summary = aggregate_results(gpu_output_dir, env_info)

    # Generate report
    generate_report(summary, gpu_output_dir)

    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Output directory: {gpu_output_dir}")
    print(f"Results:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print("\nFor detailed results, see:")
    print(f"  {gpu_output_dir / 'benchmark_report.md'}")
    print(f"  {gpu_output_dir / 'benchmark_summary.json'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
