"""Inference latency profiling for voice conversion pipeline.

Task 7.3: Profile inference latency with continuous training models

Provides:
- Stage-by-stage latency breakdown
- CUDA synchronization for accurate GPU timing
- Real-time factor (RTF) calculation
- Latency regression detection
- Benchmark suite for model comparison
"""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Standard pipeline stages for voice conversion
PIPELINE_STAGES = [
    'load_audio',
    'vocal_separation',
    'pitch_extraction',
    'technique_detection',
    'voice_conversion',
    'mixing',
]


class InferenceLatencyProfiler:
    """Profile inference latency with GPU synchronization.

    Tracks execution time for different pipeline stages and provides
    statistics and RTF (real-time factor) calculations.
    """

    def __init__(
        self,
        device: str = 'cuda:0',
        sync_cuda: bool = True,
        warmup_runs: int = 0,
    ):
        """Initialize latency profiler.

        Args:
            device: CUDA device to profile
            sync_cuda: Whether to synchronize CUDA before/after timing
            warmup_runs: Number of initial runs to skip in measurements
        """
        self.device = device
        self.sync_cuda = sync_cuda
        self.warmup_runs = warmup_runs
        self.measurements: Dict[str, List[float]] = {}
        self._run_counts: Dict[str, int] = {}

        # Parse device index for CUDA sync
        self._device_idx = int(device.split(':')[1]) if ':' in device else 0

    @contextmanager
    def measure_stage(self, stage_name: str):
        """Context manager to measure execution time of a stage.

        Args:
            stage_name: Name of the pipeline stage

        Yields:
            Control to the measured block

        Example:
            with profiler.measure_stage('voice_conversion'):
                output = model(input)
        """
        # Track run count for warmup
        if stage_name not in self._run_counts:
            self._run_counts[stage_name] = 0
        self._run_counts[stage_name] += 1

        # Skip warmup runs
        if self._run_counts[stage_name] <= self.warmup_runs:
            yield
            return

        # Sync CUDA before timing if requested
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(self._device_idx)

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Sync CUDA after operation completes
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize(self._device_idx)

            elapsed = time.perf_counter() - start_time

            if stage_name not in self.measurements:
                self.measurements[stage_name] = []
            self.measurements[stage_name].append(elapsed)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all measured stages.

        Returns:
            Dict mapping stage names to their statistics:
            - mean_ms: Mean latency in milliseconds
            - std_ms: Standard deviation in milliseconds
            - min_ms: Minimum latency in milliseconds
            - max_ms: Maximum latency in milliseconds
            - count: Number of measurements
        """
        import numpy as np

        stats = {}
        for stage_name, times in self.measurements.items():
            if not times:
                continue

            times_ms = [t * 1000 for t in times]
            stats[stage_name] = {
                'mean_ms': float(np.mean(times_ms)),
                'std_ms': float(np.std(times_ms)),
                'min_ms': float(np.min(times_ms)),
                'max_ms': float(np.max(times_ms)),
                'count': len(times),
            }

        return stats

    def calculate_rtf(self, stage_name: str, audio_duration_s: float) -> float:
        """Calculate real-time factor for a stage.

        RTF < 1.0 means faster than real-time.
        RTF > 1.0 means slower than real-time.

        Args:
            stage_name: Name of the stage to calculate RTF for
            audio_duration_s: Duration of processed audio in seconds

        Returns:
            Real-time factor (processing_time / audio_duration)
        """
        if stage_name not in self.measurements or not self.measurements[stage_name]:
            return 0.0

        import numpy as np
        mean_time = float(np.mean(self.measurements[stage_name]))
        return mean_time / audio_duration_s if audio_duration_s > 0 else 0.0

    def clear(self):
        """Clear all measurements."""
        self.measurements = {}
        self._run_counts = {}

    def export_json(self, output_path: str):
        """Export profile results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = {
            'device': self.device,
            'sync_cuda': self.sync_cuda,
            'warmup_runs': self.warmup_runs,
            'stats': self.get_stats(),
            'raw_measurements': {k: v for k, v in self.measurements.items()},
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported profile results to {output_path}")


def compare_model_latency(
    base_model: nn.Module,
    trained_model: nn.Module,
    input_fn: Callable[[], Dict[str, torch.Tensor]],
    device: torch.device,
    num_runs: int = 10,
) -> Dict[str, float]:
    """Compare latency between base and trained models.

    Args:
        base_model: Base/pre-trained model
        trained_model: Fine-tuned model
        input_fn: Function that returns model inputs
        device: Device to run on
        num_runs: Number of runs for averaging

    Returns:
        Dict with comparison metrics:
        - base_mean_ms: Base model mean latency
        - trained_mean_ms: Trained model mean latency
        - latency_diff_ms: Difference (trained - base)
        - latency_diff_pct: Percentage difference
    """
    import numpy as np

    base_profiler = InferenceLatencyProfiler(device=str(device), sync_cuda=True)
    trained_profiler = InferenceLatencyProfiler(device=str(device), sync_cuda=True)

    base_model.train(False)
    trained_model.train(False)

    with torch.no_grad():
        # Profile base model
        for _ in range(num_runs):
            inputs = input_fn()
            with base_profiler.measure_stage('forward'):
                base_model(**inputs)

        # Profile trained model
        for _ in range(num_runs):
            inputs = input_fn()
            with trained_profiler.measure_stage('forward'):
                trained_model(**inputs)

    base_stats = base_profiler.get_stats()
    trained_stats = trained_profiler.get_stats()

    base_mean = base_stats['forward']['mean_ms']
    trained_mean = trained_stats['forward']['mean_ms']

    return {
        'base_mean_ms': base_mean,
        'trained_mean_ms': trained_mean,
        'latency_diff_ms': trained_mean - base_mean,
        'latency_diff_pct': (trained_mean - base_mean) / base_mean * 100 if base_mean > 0 else 0,
    }


def detect_latency_regression(
    baseline_stats: Dict[str, Dict[str, float]],
    current_stats: Dict[str, Dict[str, float]],
    threshold_pct: float = 20.0,
) -> Dict[str, Any]:
    """Detect latency regression between baseline and current stats.

    Args:
        baseline_stats: Baseline latency statistics
        current_stats: Current latency statistics
        threshold_pct: Percentage threshold for regression detection

    Returns:
        Dict with:
        - has_regression: True if any stage regressed
        - stages_with_regression: List of regressed stages
        - details: Per-stage regression details
    """
    stages_with_regression = []
    details = {}

    for stage_name, baseline in baseline_stats.items():
        if stage_name not in current_stats:
            continue

        current = current_stats[stage_name]
        baseline_mean = baseline.get('mean_ms', 0)
        current_mean = current.get('mean_ms', 0)

        if baseline_mean > 0:
            diff_pct = (current_mean - baseline_mean) / baseline_mean * 100

            details[stage_name] = {
                'baseline_ms': baseline_mean,
                'current_ms': current_mean,
                'diff_pct': diff_pct,
                'regressed': diff_pct > threshold_pct,
            }

            if diff_pct > threshold_pct:
                stages_with_regression.append(stage_name)

    return {
        'has_regression': len(stages_with_regression) > 0,
        'stages_with_regression': stages_with_regression,
        'threshold_pct': threshold_pct,
        'details': details,
    }


class LatencyBenchmarkRunner:
    """Run comprehensive latency benchmarks for models.

    Executes standardized benchmarks and generates reports.
    """

    def __init__(
        self,
        device: str = 'cuda:0',
        output_dir: str = './benchmark_results',
    ):
        """Initialize benchmark runner.

        Args:
            device: CUDA device to benchmark on
            output_dir: Directory for output files
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: Dict[str, Any] = {}

    def run_benchmark(
        self,
        model_name: str,
        input_sizes: List[Tuple[int, int, int]],  # (batch, dim, seq)
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """Run benchmark for a model with various input sizes.

        Args:
            model_name: Name for the benchmark
            input_sizes: List of (batch_size, dim, seq_len) tuples
            num_runs: Number of runs per input size
            warmup_runs: Number of warmup runs to skip

        Returns:
            Benchmark results dict
        """
        profiler = InferenceLatencyProfiler(
            device=self.device,
            sync_cuda=True,
            warmup_runs=warmup_runs,
        )

        latencies = {}

        for batch, dim, seq in input_sizes:
            size_key = f"{batch}x{dim}x{seq}"

            # Create dummy tensor and run timed operations
            for _ in range(warmup_runs + num_runs):
                with profiler.measure_stage(size_key):
                    x = torch.randn(batch, dim, seq, device=self.device)
                    # Simulate some GPU work
                    y = x @ x.transpose(-1, -2)
                    torch.cuda.synchronize()
                    del x, y

            stats = profiler.get_stats()
            if size_key in stats:
                latencies[size_key] = stats[size_key]

        self._results = {
            'model_name': model_name,
            'device': self.device,
            'input_sizes': input_sizes,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
            'latencies': latencies,
        }

        return self._results

    def generate_report(self, output_path: str):
        """Generate benchmark report.

        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self._results, f, indent=2)

        logger.info(f"Benchmark report saved to {output_path}")
