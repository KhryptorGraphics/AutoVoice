"""GPU resource management and optimization."""

import torch
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and device selection."""

    def __init__(self, device_id: Optional[int] = None):
        """Initialize GPU manager.

        Args:
            device_id: Specific GPU device ID to use, or None for auto-select
        """
        self.device_id = device_id
        self.device = None
        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            if device_id is not None:
                self.device = torch.device(f'cuda:{device_id}')
            else:
                self.device = torch.device('cuda')

            # Set current device
            torch.cuda.set_device(self.device)

            # Log device info
            device_name = torch.cuda.get_device_name(self.device)
            logger.info(f"Using GPU: {device_name}")
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")

    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device

    def get_memory_info(self) -> dict:
        """Get GPU memory information."""
        if not self.cuda_available:
            return {'available': False}

        return {
            'available': True,
            'total': torch.cuda.get_device_properties(self.device).total_memory,
            'allocated': torch.cuda.memory_allocated(self.device),
            'cached': torch.cuda.memory_reserved(self.device),
            'free': torch.cuda.get_device_properties(self.device).total_memory -
                    torch.cuda.memory_allocated(self.device)
        }

    def optimize_batch_size(self, model_size: int, sample_size: int) -> int:
        """Calculate optimal batch size based on available memory.

        Args:
            model_size: Size of model parameters in bytes
            sample_size: Size of single sample in bytes

        Returns:
            Optimal batch size
        """
        if not self.cuda_available:
            return 1

        memory_info = self.get_memory_info()
        available_memory = memory_info['free'] * 0.8  # Use 80% of free memory

        # Account for gradients and intermediate activations
        overhead_factor = 3.0
        memory_per_sample = sample_size * overhead_factor

        # Calculate batch size
        batch_size = int((available_memory - model_size) / memory_per_sample)
        batch_size = max(1, min(batch_size, 256))  # Clamp between 1 and 256

        return batch_size

    def clear_cache(self):
        """Clear GPU cache."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def synchronize(self):
        """Synchronize GPU operations."""
        if self.cuda_available:
            torch.cuda.synchronize(self.device)

    def enable_mixed_precision(self):
        """Enable mixed precision training."""
        if self.cuda_available:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Mixed precision enabled")

    def get_compute_capability(self) -> Tuple[int, int]:
        """Get GPU compute capability."""
        if not self.cuda_available:
            return (0, 0)

        major, minor = torch.cuda.get_device_capability(self.device)
        return (major, minor)

    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self.cuda_available