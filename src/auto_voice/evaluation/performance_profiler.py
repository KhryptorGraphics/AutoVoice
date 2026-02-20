"""Performance profiler for voice conversion pipeline.

Measures GPU memory usage, inference latency, and generates
performance reports for optimization guidance.
"""
import time
from typing import Dict, List, Optional, Any
import torch
import numpy as np


class PerformanceProfiler:
    """Profiles performance of voice conversion pipeline.

    Measures GPU memory, inference time, and provides reporting.

    Args:
        device: Torch device for profiling
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self._measurements: Dict[str, List[float]] = {}

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with allocated_mb and reserved_mb
        """
        if not torch.cuda.is_available():
            return {'allocated_mb': 0.0, 'reserved_mb': 0.0}

        allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
        }

    def profile_inference(
        self,
        pipeline,
        audio: torch.Tensor,
        speaker: torch.Tensor,
        sample_rate: int = 24000,
        warmup_runs: int = 1,
        timed_runs: int = 3,
    ) -> Dict[str, Any]:
        """Profile inference timing for pipeline.

        Args:
            pipeline: Voice conversion pipeline with convert() method
            audio: Input audio tensor
            speaker: Speaker embedding tensor
            sample_rate: Audio sample rate
            warmup_runs: Number of warmup runs before timing
            timed_runs: Number of runs to time

        Returns:
            Dictionary with total_ms and per_component timing
        """
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                pipeline.convert(audio, sample_rate, speaker)

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(timed_runs):
            start = time.time()
            with torch.no_grad():
                pipeline.convert(audio, sample_rate, speaker)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        return {
            'total_ms': np.mean(times),
            'total_std_ms': np.std(times),
            'per_component': {},  # Would need instrumented pipeline
            'memory_mb': self.get_gpu_memory_usage()['allocated_mb'],
        }

    def record_measurement(self, name: str, value: float) -> None:
        """Record a measurement for later aggregation.

        Args:
            name: Measurement name
            value: Measurement value
        """
        if name not in self._measurements:
            self._measurements[name] = []
        self._measurements[name].append(value)

    def clear_measurements(self) -> None:
        """Clear all recorded measurements."""
        self._measurements.clear()

    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """Generate summary report from recorded measurements.

        Returns:
            Dictionary with statistics for each measurement
        """
        report = {}

        for name, values in self._measurements.items():
            arr = np.array(values)
            report[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'count': len(arr),
            }

        return report

    def profile_memory_by_component(
        self,
        pipeline,
    ) -> Dict[str, float]:
        """Profile memory usage per component.

        Args:
            pipeline: Voice conversion pipeline

        Returns:
            Dictionary with memory usage per component
        """
        if not torch.cuda.is_available():
            return {}

        memory_by_component = {}

        # Get baseline
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = torch.cuda.memory_allocated(self.device)

        # Profile each component if available
        components = [
            ('separator', getattr(pipeline, 'separator', None)),
            ('content_extractor', getattr(pipeline, 'content_extractor', None)),
            ('pitch_extractor', getattr(pipeline, 'pitch_extractor', None)),
            ('decoder', getattr(pipeline, 'decoder', None)),
            ('vocoder', getattr(pipeline, 'vocoder', None)),
        ]

        for name, component in components:
            if component is not None:
                # Move to device and measure
                component.to(self.device)
                torch.cuda.synchronize()
                current = torch.cuda.memory_allocated(self.device)
                memory_by_component[name] = (current - baseline) / (1024 * 1024)

        return memory_by_component
