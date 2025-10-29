"""GPU utilization monitoring for performance validation.

This module provides GPU monitoring capabilities for performance tests,
including real-time utilization tracking and reporting.
"""

import time
import threading
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUUtilizationMonitor:
    """Context manager for monitoring GPU utilization during operations.

    Samples GPU utilization at regular intervals and computes statistics.
    Compatible with nvidia-smi for monitoring NVIDIA GPUs.

    Example:
        >>> monitor = GPUUtilizationMonitor(device_id=0, sampling_interval=0.1)
        >>> with monitor:
        ...     # Run GPU operations
        ...     result = pipeline.convert_song(...)
        >>> stats = monitor.get_average_utilization()
        >>> print(f"Average GPU utilization: {stats:.1f}%")
    """

    def __init__(
        self,
        device_id: int = 0,
        sampling_interval: float = 0.1,
        use_pynvml: bool = True
    ):
        """Initialize GPU utilization monitor.

        Args:
            device_id: GPU device ID to monitor (default: 0)
            sampling_interval: Sampling interval in seconds (default: 0.1)
            use_pynvml: Use pynvml for monitoring if available, fallback to nvidia-smi
        """
        self.device_id = device_id
        self.sampling_interval = sampling_interval
        self.use_pynvml = use_pynvml

        self.samples: List[float] = []
        self.timestamps: List[float] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Check if CUDA is available
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
        except ImportError:
            self.cuda_available = False
            logger.warning("PyTorch not available, GPU monitoring disabled")

        # Try to initialize pynvml
        self.pynvml_available = False
        if self.use_pynvml and self.cuda_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.pynvml_available = True
                logger.debug(f"pynvml initialized for GPU {device_id}")
            except Exception as e:
                logger.debug(f"pynvml not available: {e}, will use nvidia-smi")
                self.pynvml_available = False

    def __enter__(self):
        """Start monitoring when entering context."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring when exiting context."""
        self.stop_monitoring()
        return False

    def start_monitoring(self):
        """Start GPU utilization monitoring in background thread."""
        if not self.cuda_available:
            logger.warning("CUDA not available, GPU monitoring disabled")
            return

        self.monitoring = True
        self.samples = []
        self.timestamps = []

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.debug(f"Started GPU monitoring (interval={self.sampling_interval}s)")

    def stop_monitoring(self):
        """Stop GPU utilization monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        logger.debug(f"Stopped GPU monitoring ({len(self.samples)} samples collected)")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                utilization = self._get_gpu_utilization()
                if utilization is not None:
                    self.samples.append(utilization)
                    self.timestamps.append(time.time())
            except Exception as e:
                logger.debug(f"Error sampling GPU utilization: {e}")

            time.sleep(self.sampling_interval)

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization percentage (0-100) or None if unavailable
        """
        if not self.cuda_available:
            return None

        # Try pynvml first
        if self.pynvml_available:
            try:
                import pynvml
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                return float(utilization.gpu)
            except Exception as e:
                logger.debug(f"pynvml query failed: {e}")
                # Don't retry pynvml if it fails
                self.pynvml_available = False

        # Fallback to nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu',
                    '--format=csv,noheader,nounits',
                    f'--id={self.device_id}'
                ],
                capture_output=True,
                text=True,
                timeout=1.0
            )

            if result.returncode == 0:
                utilization_str = result.stdout.strip()
                return float(utilization_str)
            else:
                logger.debug(f"nvidia-smi failed: {result.stderr}")
                return None

        except Exception as e:
            logger.debug(f"nvidia-smi not available: {e}")
            return None

    def get_average_utilization(self) -> float:
        """Get average GPU utilization across all samples.

        Returns:
            Average GPU utilization percentage (0-100), or 0.0 if no samples
        """
        if not self.samples:
            return 0.0

        import numpy as np
        return float(np.mean(self.samples))

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU utilization statistics.

        Returns:
            Dictionary with utilization statistics:
            {
                'average': float,
                'min': float,
                'max': float,
                'std': float,
                'samples': int,
                'duration_seconds': float
            }
        """
        if not self.samples:
            return {
                'average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'samples': 0,
                'duration_seconds': 0.0
            }

        import numpy as np

        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0

        return {
            'average': float(np.mean(self.samples)),
            'min': float(np.min(self.samples)),
            'max': float(np.max(self.samples)),
            'std': float(np.std(self.samples)),
            'samples': len(self.samples),
            'duration_seconds': duration
        }

    def __del__(self):
        """Cleanup on deletion."""
        if self.pynvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
