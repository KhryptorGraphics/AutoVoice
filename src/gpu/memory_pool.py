"""Memory pool for efficient GPU memory management."""

import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MemoryPool:
    """Manages a pool of pre-allocated GPU memory buffers."""

    def __init__(self, device: torch.device, initial_size: int = 1024 * 1024 * 100):
        """Initialize memory pool.

        Args:
            device: Target device for memory allocation
            initial_size: Initial pool size in bytes
        """
        self.device = device
        self.pool_size = initial_size
        self.buffers: Dict[int, List[torch.Tensor]] = {}
        self.allocated: Dict[int, List[torch.Tensor]] = {}

    def allocate(self, size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate a buffer from the pool.

        Args:
            size: Size of buffer in elements
            dtype: Data type of buffer

        Returns:
            Allocated tensor
        """
        key = (size, dtype)

        # Check if we have a free buffer
        if key in self.buffers and self.buffers[key]:
            buffer = self.buffers[key].pop()
            if key not in self.allocated:
                self.allocated[key] = []
            self.allocated[key].append(buffer)
            return buffer

        # Allocate new buffer
        buffer = torch.empty(size, dtype=dtype, device=self.device)
        if key not in self.allocated:
            self.allocated[key] = []
        self.allocated[key].append(buffer)

        logger.debug(f"Allocated new buffer: size={size}, dtype={dtype}")
        return buffer

    def release(self, buffer: torch.Tensor):
        """Release a buffer back to the pool.

        Args:
            buffer: Buffer to release
        """
        size = buffer.numel()
        dtype = buffer.dtype
        key = (size, dtype)

        # Move from allocated to free pool
        if key in self.allocated and buffer in self.allocated[key]:
            self.allocated[key].remove(buffer)
            if key not in self.buffers:
                self.buffers[key] = []
            self.buffers[key].append(buffer)
            logger.debug(f"Released buffer: size={size}, dtype={dtype}")

    def clear(self):
        """Clear all buffers in the pool."""
        self.buffers.clear()
        self.allocated.clear()
        torch.cuda.empty_cache()
        logger.info("Memory pool cleared")

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        total_free = sum(len(buffers) for buffers in self.buffers.values())
        total_allocated = sum(len(buffers) for buffers in self.allocated.values())

        free_memory = sum(
            buf.numel() * buf.element_size()
            for buffers in self.buffers.values()
            for buf in buffers
        )

        allocated_memory = sum(
            buf.numel() * buf.element_size()
            for buffers in self.allocated.values()
            for buf in buffers
        )

        return {
            'free_buffers': total_free,
            'allocated_buffers': total_allocated,
            'free_memory_mb': free_memory / (1024 * 1024),
            'allocated_memory_mb': allocated_memory / (1024 * 1024)
        }

    def optimize(self):
        """Optimize pool by removing unused buffers."""
        # Remove excess free buffers
        for key in list(self.buffers.keys()):
            if len(self.buffers[key]) > 10:
                # Keep only 10 buffers of each type
                self.buffers[key] = self.buffers[key][:10]

        torch.cuda.empty_cache()
        logger.info("Memory pool optimized")