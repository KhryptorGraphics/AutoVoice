"""Performance testing and benchmarking fixtures.

Provides comprehensive performance measurement, profiling, and
resource monitoring fixtures for optimization and regression testing.
"""

import pytest
import time
import psutil
import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json


# ============================================================================
# Performance Benchmarking
# ============================================================================

@pytest.fixture
def performance_benchmarker():
    """Comprehensive performance benchmarking fixture.

    Provides timing, throughput, and latency measurements with
    statistical analysis.

    Examples:
        bench = performance_benchmarker
        with bench.measure('model_inference'):
            output = model(input)
        stats = bench.get_statistics()
    """
    class PerformanceBenchmarker:
        def __init__(self):
            self.measurements = {}
            self.current_measurement = None
            self.start_time = None

        def measure(self, name: str):
            """Context manager for timing measurements.

            Args:
                name: Measurement name

            Returns:
                Context manager
            """
            from contextlib import contextmanager

            @contextmanager
            def timer():
                start = time.perf_counter()
                yield
                elapsed = time.perf_counter() - start

                if name not in self.measurements:
                    self.measurements[name] = []

                self.measurements[name].append(elapsed)

            return timer()

        def benchmark(
            self,
            func: Callable,
            *args,
            iterations: int = 100,
            warmup: int = 10,
            **kwargs
        ) -> Dict[str, float]:
            """Benchmark function with multiple iterations.

            Args:
                func: Function to benchmark
                *args: Function arguments
                iterations: Number of iterations
                warmup: Warmup iterations
                **kwargs: Function keyword arguments

            Returns:
                Dict with timing statistics
            """
            timings = []

            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)

            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                func(*args, **kwargs)
                timings.append(time.perf_counter() - start)

            return {
                'mean': np.mean(timings),
                'std': np.std(timings),
                'min': np.min(timings),
                'max': np.max(timings),
                'median': np.median(timings),
                'p95': np.percentile(timings, 95),
                'p99': np.percentile(timings, 99),
                'iterations': iterations,
            }

        def get_statistics(self, name: Optional[str] = None) -> Dict[str, Any]:
            """Get timing statistics.

            Args:
                name: Specific measurement name (None = all)

            Returns:
                Statistics dict
            """
            if name is not None:
                if name not in self.measurements:
                    return {}

                timings = self.measurements[name]
                return {
                    'count': len(timings),
                    'mean': np.mean(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'median': np.median(timings),
                    'total': np.sum(timings),
                }

            # All measurements
            stats = {}
            for measurement_name in self.measurements:
                stats[measurement_name] = self.get_statistics(measurement_name)

            return stats

        def compare(self, name1: str, name2: str) -> Dict[str, Any]:
            """Compare two measurements.

            Args:
                name1: First measurement
                name2: Second measurement

            Returns:
                Comparison results
            """
            stats1 = self.get_statistics(name1)
            stats2 = self.get_statistics(name2)

            if not stats1 or not stats2:
                return {}

            speedup = stats1['mean'] / stats2['mean']

            return {
                'measurement1': name1,
                'measurement2': name2,
                'speedup': speedup,
                'improvement_percent': (speedup - 1) * 100,
                'faster': name2 if speedup > 1 else name1,
            }

        def save_results(self, filepath: Path):
            """Save benchmark results to JSON.

            Args:
                filepath: Output file path
            """
            stats = self.get_statistics()

            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)

    return PerformanceBenchmarker()


@pytest.fixture
def resource_profiler():
    """Resource usage profiler for CPU, memory, and GPU monitoring.

    Tracks resource consumption during test execution.

    Examples:
        profiler = resource_profiler
        with profiler.profile():
            # ... operations ...
        print(profiler.get_summary())
    """
    class ResourceProfiler:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.snapshots = []
            self.is_profiling = False
            self.cuda_available = torch.cuda.is_available()

        def profile(self):
            """Context manager for resource profiling.

            Returns:
                Context manager
            """
            from contextlib import contextmanager

            @contextmanager
            def profiler():
                self.start()
                try:
                    yield self
                finally:
                    self.stop()

            return profiler()

        def start(self):
            """Start profiling."""
            self.is_profiling = True
            self.snapshots = []
            self._take_snapshot('start')

        def stop(self):
            """Stop profiling."""
            if not self.is_profiling:
                return

            self._take_snapshot('stop')
            self.is_profiling = False

        def checkpoint(self, name: str = 'checkpoint'):
            """Take resource snapshot.

            Args:
                name: Checkpoint name
            """
            if self.is_profiling:
                self._take_snapshot(name)

        def _take_snapshot(self, event: str):
            """Take resource usage snapshot.

            Args:
                event: Event name
            """
            # CPU and memory
            cpu_percent = self.process.cpu_percent()
            mem_info = self.process.memory_info()

            snapshot = {
                'event': event,
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_rss_mb': mem_info.rss / 1024 / 1024,
                'memory_vms_mb': mem_info.vms / 1024 / 1024,
            }

            # GPU if available
            if self.cuda_available:
                snapshot['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                snapshot['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

            self.snapshots.append(snapshot)

        def get_summary(self) -> Dict[str, Any]:
            """Get profiling summary.

            Returns:
                Summary dict
            """
            if len(self.snapshots) < 2:
                return {}

            start = self.snapshots[0]
            end = self.snapshots[-1]

            summary = {
                'duration': end['timestamp'] - start['timestamp'],
                'cpu': {
                    'max_percent': max(s['cpu_percent'] for s in self.snapshots),
                    'avg_percent': np.mean([s['cpu_percent'] for s in self.snapshots]),
                },
                'memory': {
                    'start_mb': start['memory_rss_mb'],
                    'end_mb': end['memory_rss_mb'],
                    'peak_mb': max(s['memory_rss_mb'] for s in self.snapshots),
                    'delta_mb': end['memory_rss_mb'] - start['memory_rss_mb'],
                }
            }

            if self.cuda_available:
                summary['gpu'] = {
                    'start_allocated_mb': start.get('gpu_allocated_mb', 0),
                    'end_allocated_mb': end.get('gpu_allocated_mb', 0),
                    'peak_allocated_mb': max(
                        s.get('gpu_allocated_mb', 0) for s in self.snapshots
                    ),
                    'delta_allocated_mb': (
                        end.get('gpu_allocated_mb', 0) - start.get('gpu_allocated_mb', 0)
                    ),
                }

            return summary

        def get_snapshots(self) -> List[Dict[str, Any]]:
            """Get all snapshots.

            Returns:
                List of snapshot dicts
            """
            return self.snapshots.copy()

    return ResourceProfiler()


@pytest.fixture
def throughput_tester():
    """Test throughput and real-time factor for audio processing.

    Measures how many seconds of audio can be processed per second.

    Examples:
        tester = throughput_tester
        rtf = tester.measure_rtf(process_func, audio, sample_rate)
        assert rtf < 0.5  # Real-time processing
    """
    class ThroughputTester:
        def measure_rtf(
            self,
            func: Callable,
            audio: np.ndarray,
            sample_rate: int,
            iterations: int = 10
        ) -> Dict[str, float]:
            """Measure Real-Time Factor (RTF).

            RTF = processing_time / audio_duration
            RTF < 1.0 means faster than real-time

            Args:
                func: Processing function
                audio: Input audio
                sample_rate: Sample rate in Hz
                iterations: Number of iterations

            Returns:
                Dict with RTF statistics
            """
            audio_duration = len(audio) / sample_rate
            rtf_values = []

            for _ in range(iterations):
                start = time.perf_counter()
                func(audio)
                processing_time = time.perf_counter() - start

                rtf = processing_time / audio_duration
                rtf_values.append(rtf)

            return {
                'mean_rtf': np.mean(rtf_values),
                'min_rtf': np.min(rtf_values),
                'max_rtf': np.max(rtf_values),
                'std_rtf': np.std(rtf_values),
                'audio_duration': audio_duration,
                'is_realtime': np.mean(rtf_values) < 1.0,
            }

        def measure_throughput(
            self,
            func: Callable,
            batch_size: int,
            input_generator: Callable,
            duration: float = 10.0
        ) -> Dict[str, float]:
            """Measure processing throughput.

            Args:
                func: Processing function
                batch_size: Batch size
                input_generator: Function to generate input
                duration: Test duration in seconds

            Returns:
                Throughput statistics
            """
            start_time = time.time()
            num_batches = 0
            total_samples = 0

            while time.time() - start_time < duration:
                batch = input_generator(batch_size)
                func(batch)

                num_batches += 1
                total_samples += batch_size

            elapsed = time.time() - start_time

            return {
                'throughput_samples_per_sec': total_samples / elapsed,
                'throughput_batches_per_sec': num_batches / elapsed,
                'total_samples': total_samples,
                'total_batches': num_batches,
                'duration': elapsed,
            }

    return ThroughputTester()


@pytest.fixture
def regression_tester(tmp_path: Path):
    """Performance regression testing fixture.

    Compares current performance against baseline metrics.

    Examples:
        tester = regression_tester
        tester.set_baseline('inference_time', 0.05)
        current = tester.measure('inference_time', lambda: model(input))
        assert tester.check_regression('inference_time')
    """
    class RegressionTester:
        def __init__(self, baseline_path: Path):
            self.baseline_path = baseline_path / 'performance_baseline.json'
            self.baselines = self._load_baselines()
            self.current_metrics = {}

        def _load_baselines(self) -> Dict[str, float]:
            """Load baseline metrics from file."""
            if self.baseline_path.exists():
                with open(self.baseline_path) as f:
                    return json.load(f)
            return {}

        def _save_baselines(self):
            """Save baseline metrics to file."""
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_path, 'w') as f:
                json.dump(self.baselines, f, indent=2)

        def set_baseline(self, name: str, value: float):
            """Set baseline metric.

            Args:
                name: Metric name
                value: Baseline value
            """
            self.baselines[name] = value
            self._save_baselines()

        def measure(
            self,
            name: str,
            func: Callable,
            iterations: int = 10
        ) -> float:
            """Measure current metric.

            Args:
                name: Metric name
                func: Function to measure
                iterations: Number of iterations

            Returns:
                Mean measurement
            """
            timings = []

            for _ in range(iterations):
                start = time.perf_counter()
                func()
                timings.append(time.perf_counter() - start)

            mean_time = np.mean(timings)
            self.current_metrics[name] = mean_time

            return mean_time

        def check_regression(
            self,
            name: str,
            tolerance: float = 0.1
        ) -> bool:
            """Check for performance regression.

            Args:
                name: Metric name
                tolerance: Acceptable degradation (0.1 = 10%)

            Returns:
                True if no regression detected
            """
            if name not in self.baselines:
                return True  # No baseline to compare

            if name not in self.current_metrics:
                return False  # No current measurement

            baseline = self.baselines[name]
            current = self.current_metrics[name]

            # Check if current is within tolerance of baseline
            max_allowed = baseline * (1 + tolerance)

            return current <= max_allowed

        def get_report(self) -> Dict[str, Any]:
            """Get regression test report.

            Returns:
                Report dict
            """
            report = {
                'metrics': [],
                'regressions': [],
                'improvements': [],
            }

            for name in set(list(self.baselines.keys()) + list(self.current_metrics.keys())):
                baseline = self.baselines.get(name)
                current = self.current_metrics.get(name)

                if baseline is None or current is None:
                    continue

                change_percent = ((current - baseline) / baseline) * 100

                metric_info = {
                    'name': name,
                    'baseline': baseline,
                    'current': current,
                    'change_percent': change_percent,
                    'is_regression': change_percent > 10,
                    'is_improvement': change_percent < -10,
                }

                report['metrics'].append(metric_info)

                if metric_info['is_regression']:
                    report['regressions'].append(metric_info)
                elif metric_info['is_improvement']:
                    report['improvements'].append(metric_info)

            return report

    return RegressionTester(tmp_path)


__all__ = [
    'performance_benchmarker',
    'resource_profiler',
    'throughput_tester',
    'regression_tester',
]
