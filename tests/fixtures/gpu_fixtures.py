"""GPU and CUDA-related pytest fixtures for AutoVoice testing.

Provides GPU memory tracking, CUDA context management, and multi-GPU
testing utilities.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


# ============================================================================
# GPU Context Management
# ============================================================================

@pytest.fixture
def gpu_context_manager(cuda_available: bool):
    """Context manager for GPU operations with automatic cleanup.

    Provides automatic CUDA cache clearing, memory tracking, and
    error handling for GPU operations.

    Examples:
        with gpu_context_manager() as ctx:
            tensor = torch.randn(1000, 1000, device='cuda')
            result = model(tensor)
        print(ctx.peak_memory_mb)  # Peak memory used
    """
    @contextmanager
    def manager(
        device: Optional[str] = None,
        memory_fraction: float = 0.9,
        clear_cache: bool = True
    ):
        """GPU context manager.

        Args:
            device: Device string ('cuda:0', etc.), None for auto
            memory_fraction: Fraction of GPU memory to use
            clear_cache: Clear CUDA cache before and after

        Yields:
            Context object with memory stats
        """
        if not cuda_available:
            yield type('Context', (), {'peak_memory_mb': 0, 'device': 'cpu'})()
            return

        # Setup
        if clear_cache:
            torch.cuda.empty_cache()

        device_obj = torch.device(device if device else 'cuda')

        torch.cuda.reset_peak_memory_stats(device_obj)
        initial_memory = torch.cuda.memory_allocated(device_obj)

        context = type('Context', (), {
            'device': device_obj,
            'initial_memory_mb': initial_memory / 1024 / 1024,
            'peak_memory_mb': 0,
            'final_memory_mb': 0,
            'memory_delta_mb': 0,
        })()

        try:
            yield context
        finally:
            # Cleanup and stats
            torch.cuda.synchronize(device_obj)

            peak_memory = torch.cuda.max_memory_allocated(device_obj)
            final_memory = torch.cuda.memory_allocated(device_obj)

            context.peak_memory_mb = peak_memory / 1024 / 1024
            context.final_memory_mb = final_memory / 1024 / 1024
            context.memory_delta_mb = (final_memory - initial_memory) / 1024 / 1024

            if clear_cache:
                torch.cuda.empty_cache()

    return manager


@pytest.fixture
def cuda_memory_tracker(cuda_available: bool):
    """Advanced CUDA memory tracking fixture.

    Tracks memory allocations, deallocations, and provides leak detection
    with detailed reporting.

    Examples:
        tracker = cuda_memory_tracker
        tracker.start()
        # ... GPU operations ...
        stats = tracker.stop()
        assert stats['leaked_mb'] < 10  # Less than 10MB leak
    """
    if not cuda_available:
        pytest.skip("CUDA not available for memory tracking")

    class CUDAMemoryTracker:
        def __init__(self):
            self.device = torch.device('cuda')
            self.snapshots = []
            self.is_tracking = False
            self.initial_memory = 0
            self.peak_memory = 0

        def start(self):
            """Start memory tracking."""
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

            self.initial_memory = torch.cuda.memory_allocated(self.device)
            self.is_tracking = True

            self.snapshots.append({
                'event': 'start',
                'memory_mb': self.initial_memory / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024 / 1024,
            })

        def checkpoint(self, name: str = 'checkpoint'):
            """Take a memory snapshot.

            Args:
                name: Name for this checkpoint
            """
            if not self.is_tracking:
                return

            current_memory = torch.cuda.memory_allocated(self.device)
            reserved_memory = torch.cuda.memory_reserved(self.device)

            self.snapshots.append({
                'event': name,
                'memory_mb': current_memory / 1024 / 1024,
                'reserved_mb': reserved_memory / 1024 / 1024,
                'delta_from_start_mb': (current_memory - self.initial_memory) / 1024 / 1024,
            })

        def stop(self) -> Dict[str, Any]:
            """Stop tracking and return statistics.

            Returns:
                Dict with memory statistics
            """
            if not self.is_tracking:
                return {}

            torch.cuda.synchronize(self.device)

            final_memory = torch.cuda.memory_allocated(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            reserved_memory = torch.cuda.memory_reserved(self.device)

            self.snapshots.append({
                'event': 'stop',
                'memory_mb': final_memory / 1024 / 1024,
                'reserved_mb': reserved_memory / 1024 / 1024,
            })

            self.is_tracking = False

            # Calculate statistics
            leaked_mb = (final_memory - self.initial_memory) / 1024 / 1024
            peak_mb = peak_memory / 1024 / 1024

            return {
                'initial_mb': self.initial_memory / 1024 / 1024,
                'final_mb': final_memory / 1024 / 1024,
                'peak_mb': peak_mb,
                'leaked_mb': leaked_mb,
                'reserved_mb': reserved_memory / 1024 / 1024,
                'snapshots': self.snapshots,
                'has_leak': leaked_mb > 10.0,  # >10MB considered leak
            }

        def get_current_stats(self) -> Dict[str, float]:
            """Get current memory statistics.

            Returns:
                Dict with current memory usage
            """
            return {
                'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024 / 1024,
                'peak_mb': torch.cuda.max_memory_allocated(self.device) / 1024 / 1024,
            }

    return CUDAMemoryTracker()


# ============================================================================
# Multi-GPU Fixtures
# ============================================================================

@pytest.fixture
def multi_gpu_config(cuda_available: bool):
    """Configuration and utilities for multi-GPU testing.

    Provides device lists, data parallel wrappers, and distributed
    testing utilities.

    Examples:
        config = multi_gpu_config
        if config.num_gpus > 1:
            model = config.data_parallel(model)
    """
    if not cuda_available:
        pytest.skip("CUDA not available for multi-GPU testing")

    num_gpus = torch.cuda.device_count()

    class MultiGPUConfig:
        def __init__(self):
            self.num_gpus = num_gpus
            self.device_ids = list(range(num_gpus))
            self.primary_device = torch.device('cuda:0')

        def get_devices(self) -> List[torch.device]:
            """Get list of available CUDA devices."""
            return [torch.device(f'cuda:{i}') for i in self.device_ids]

        def data_parallel(self, model, device_ids: Optional[List[int]] = None):
            """Wrap model in DataParallel.

            Args:
                model: Model to parallelize
                device_ids: Device IDs to use (None = all)

            Returns:
                DataParallel wrapped model
            """
            if self.num_gpus < 2:
                pytest.skip("Multiple GPUs required")

            devices = device_ids if device_ids else self.device_ids
            return torch.nn.DataParallel(model, device_ids=devices)

        def split_batch(self, batch, device_ids: Optional[List[int]] = None):
            """Split batch across devices.

            Args:
                batch: Batch to split (tensor or list of tensors)
                device_ids: Device IDs to split across

            Returns:
                List of batch chunks
            """
            devices = device_ids if device_ids else self.device_ids
            n_devices = len(devices)

            if isinstance(batch, torch.Tensor):
                chunk_size = batch.size(0) // n_devices
                chunks = batch.split(chunk_size)
                return [chunk.to(f'cuda:{i}') for i, chunk in zip(devices, chunks)]
            else:
                # Handle tuple/list of tensors
                chunk_size = batch[0].size(0) // n_devices
                chunks = []
                for i, device_id in enumerate(devices):
                    chunk = tuple(
                        tensor.split(chunk_size)[i].to(f'cuda:{device_id}')
                        for tensor in batch
                    )
                    chunks.append(chunk)
                return chunks

    config = MultiGPUConfig()

    if config.num_gpus == 0:
        pytest.skip("No GPUs available for multi-GPU testing")

    return config


@pytest.fixture
def gpu_stress_tester(cuda_available: bool):
    """Stress testing fixture for GPU operations.

    Provides methods to test GPU stability, memory limits, and
    performance under load.

    Examples:
        tester = gpu_stress_tester
        max_batch = tester.find_max_batch_size(model, input_shape)
        tester.stress_test(model, duration=60)
    """
    if not cuda_available:
        pytest.skip("CUDA not available for stress testing")

    class GPUStressTester:
        def __init__(self):
            self.device = torch.device('cuda')

        def find_max_batch_size(
            self,
            model,
            input_shape: tuple,
            max_batch: int = 256,
            safety_factor: float = 0.8
        ) -> int:
            """Find maximum batch size that fits in GPU memory.

            Args:
                model: Model to test
                input_shape: Input shape (without batch dimension)
                max_batch: Maximum batch size to try
                safety_factor: Safety factor (0.8 = 80% of max)

            Returns:
                Maximum safe batch size
            """
            model.eval()

            # Binary search for max batch size
            low, high = 1, max_batch
            last_successful = 1

            while low <= high:
                mid = (low + high) // 2

                try:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        input_tensor = torch.randn(mid, *input_shape, device=self.device)
                        _ = model(input_tensor)

                    last_successful = mid
                    low = mid + 1

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        high = mid - 1
                    else:
                        raise

            return int(last_successful * safety_factor)

        def stress_test(
            self,
            operation: callable,
            duration: float = 10.0,
            check_interval: float = 1.0
        ) -> Dict[str, Any]:
            """Run stress test on GPU operation.

            Args:
                operation: Callable operation to stress test
                duration: Duration in seconds
                check_interval: Memory check interval in seconds

            Returns:
                Dict with stress test results
            """
            import time

            start_time = time.time()
            iterations = 0
            errors = []
            memory_samples = []

            while time.time() - start_time < duration:
                try:
                    operation()
                    iterations += 1

                    # Periodic memory check
                    if iterations % int(check_interval * 10) == 0:
                        memory_samples.append(
                            torch.cuda.memory_allocated(self.device) / 1024 / 1024
                        )

                except Exception as e:
                    errors.append(str(e))

            return {
                'duration': time.time() - start_time,
                'iterations': iterations,
                'iterations_per_sec': iterations / (time.time() - start_time),
                'errors': errors,
                'num_errors': len(errors),
                'memory_samples_mb': memory_samples,
                'avg_memory_mb': np.mean(memory_samples) if memory_samples else 0,
                'peak_memory_mb': np.max(memory_samples) if memory_samples else 0,
            }

    return GPUStressTester()


__all__ = [
    'gpu_context_manager',
    'cuda_memory_tracker',
    'multi_gpu_config',
    'gpu_stress_tester',
]
