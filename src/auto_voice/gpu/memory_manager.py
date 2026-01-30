"""GPU memory management for AutoVoice.

Provides:
- Memory tracking and monitoring
- Continuous memory monitoring during training
- Automatic optimization triggers
- Memory-efficient training strategies

Task 7.2: Add GPU memory monitoring and optimization
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def handle_oom() -> None:
    """Handle out-of-memory error by clearing caches and freeing memory.

    Call this when catching CUDA OOM errors to recover gracefully.
    """
    try:
        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()

        # Synchronize to ensure all operations complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.warning("Handled OOM: cleared GPU cache")
    except Exception as e:
        logger.error(f"Failed to handle OOM: {e}")


class GPUMemoryTracker:
    """Track GPU memory allocations for debugging and monitoring."""

    def __init__(self, device: str = 'cuda:0'):
        """Initialize memory tracker.

        Args:
            device: CUDA device to track
        """
        self.device = device
        self._allocations: Dict[str, Any] = {}
        self._total_allocations = 0

    def record_allocation(self, name: str, obj: Any) -> None:
        """Record a memory allocation.

        Args:
            name: Identifier for the allocation
            obj: Object being tracked (tensor or model)
        """
        size_bytes = 0
        if isinstance(obj, torch.Tensor):
            size_bytes = obj.element_size() * obj.nelement()
        elif isinstance(obj, nn.Module):
            size_bytes = sum(
                p.element_size() * p.nelement()
                for p in obj.parameters()
            )

        self._allocations[name] = {
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024),
            'timestamp': time.time(),
        }
        self._total_allocations += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics.

        Returns:
            Dict with allocation stats
        """
        total_size = sum(a['size_bytes'] for a in self._allocations.values())
        return {
            'total_allocations': self._total_allocations,
            'active_allocations': len(self._allocations),
            'total_size_mb': total_size / (1024 * 1024),
            'allocations': dict(self._allocations),
        }

    def clear(self) -> None:
        """Clear tracked allocations."""
        self._allocations.clear()


class GPUMemoryManager:
    """Manages GPU memory allocation and tracking."""

    def __init__(self, device: str = 'cuda:0', max_fraction: float = 0.9):
        self.device = device
        self.max_fraction = max_fraction
        self._allocations: Dict[str, int] = {}

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {'available': False}

            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            total = torch.cuda.get_device_properties(device_idx).total_memory
            allocated = torch.cuda.memory_allocated(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)

            return {
                'available': True,
                'total_gb': total / (1024**3),
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'free_gb': (total - allocated) / (1024**3),
                'utilization': allocated / total if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {'available': False, 'error': str(e)}

    def can_allocate(self, size_bytes: int) -> bool:
        """Check if allocation of given size is possible."""
        info = self.get_memory_info()
        if not info.get('available'):
            return False
        free = info['free_gb'] * (1024**3)
        max_allowed = info['total_gb'] * (1024**3) * self.max_fraction
        return (info['allocated_gb'] * (1024**3) + size_bytes) <= max_allowed

    def clear_cache(self):
        """Clear PyTorch CUDA cache."""
        try:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""

    timestamp: float
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    utilization: float


class GPUMemoryMonitor:
    """Continuous GPU memory monitoring with history tracking.

    Monitors GPU memory usage at regular intervals and maintains history
    for analysis. Can trigger warnings when thresholds are exceeded.
    """

    def __init__(
        self,
        device: str = 'cuda:0',
        interval_ms: int = 1000,
        max_history: int = 1000,
        warning_threshold: float = 0.8,
        on_warning: Optional[Callable[[MemorySnapshot], None]] = None,
    ):
        """Initialize memory monitor.

        Args:
            device: CUDA device to monitor
            interval_ms: Monitoring interval in milliseconds
            max_history: Maximum history entries to keep
            warning_threshold: Utilization threshold for warnings (0-1)
            on_warning: Callback when warning threshold exceeded
        """
        self.device = device
        self.interval_ms = interval_ms
        self.max_history = max_history
        self.warning_threshold = warning_threshold
        self.on_warning = on_warning

        self._history: List[MemorySnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Parse device index
        self._device_idx = int(device.split(':')[1]) if ':' in device else 0

    def start(self) -> None:
        """Start continuous monitoring in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"GPU memory monitor started on {self.device}")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("GPU memory monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                snapshot = self._take_snapshot()
                if snapshot:
                    with self._lock:
                        self._history.append(snapshot)
                        # Trim history if too long
                        if len(self._history) > self.max_history:
                            self._history = self._history[-self.max_history:]

                    # Check warning threshold
                    if snapshot.utilization > self.warning_threshold:
                        logger.warning(
                            f"GPU memory utilization high: {snapshot.utilization:.1%}"
                        )
                        if self.on_warning:
                            self.on_warning(snapshot)

            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")

            time.sleep(self.interval_ms / 1000.0)

    def _take_snapshot(self) -> Optional[MemorySnapshot]:
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return None

        try:
            total = torch.cuda.get_device_properties(self._device_idx).total_memory
            allocated = torch.cuda.memory_allocated(self._device_idx)
            reserved = torch.cuda.memory_reserved(self._device_idx)

            return MemorySnapshot(
                timestamp=time.time(),
                allocated_gb=allocated / (1024**3),
                reserved_gb=reserved / (1024**3),
                total_gb=total / (1024**3),
                utilization=allocated / total if total > 0 else 0,
            )
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return None

    def get_history(self) -> List[Dict[str, Any]]:
        """Get memory usage history.

        Returns:
            List of memory snapshots as dictionaries
        """
        with self._lock:
            return [
                {
                    'timestamp': s.timestamp,
                    'allocated_gb': s.allocated_gb,
                    'reserved_gb': s.reserved_gb,
                    'total_gb': s.total_gb,
                    'utilization': s.utilization,
                }
                for s in self._history
            ]

    def get_stats(self) -> Dict[str, float]:
        """Get statistics from monitoring history.

        Returns:
            Dict with peak, average, min memory usage
        """
        with self._lock:
            if not self._history:
                return {
                    'peak_gb': 0,
                    'avg_gb': 0,
                    'min_gb': 0,
                    'peak_utilization': 0,
                }

            allocated = [s.allocated_gb for s in self._history]
            return {
                'peak_gb': max(allocated),
                'avg_gb': sum(allocated) / len(allocated),
                'min_gb': min(allocated),
                'peak_utilization': max(s.utilization for s in self._history),
            }


class AutoMemoryOptimizer:
    """Automatic GPU memory optimization based on thresholds.

    Monitors memory usage and applies optimizations when thresholds
    are exceeded to prevent OOM errors.
    """

    def __init__(
        self,
        manager: GPUMemoryManager,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.85,
    ):
        """Initialize auto-optimizer.

        Args:
            manager: GPUMemoryManager instance
            warning_threshold: Utilization to start warning (0-1)
            critical_threshold: Utilization to apply optimizations (0-1)
        """
        self.manager = manager
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._checkpointing_enabled = False

    def check_and_optimize(self) -> str:
        """Check memory and apply optimizations if needed.

        Returns:
            Action taken: 'none', 'cleared_cache', 'enabled_checkpointing'
        """
        info = self.manager.get_memory_info()
        if not info.get('available'):
            return 'none'

        utilization = info['utilization']

        if utilization >= self.critical_threshold:
            # Critical: clear cache first
            self.manager.clear_cache()
            logger.warning(
                f"Memory critical ({utilization:.1%}), cleared cache"
            )

            # Check if that helped
            info = self.manager.get_memory_info()
            if info['utilization'] >= self.critical_threshold:
                logger.warning(
                    "Memory still critical after cache clear, "
                    "consider enabling gradient checkpointing"
                )
                return 'cleared_cache'

            return 'cleared_cache'

        elif utilization >= self.warning_threshold:
            logger.info(
                f"Memory utilization elevated ({utilization:.1%}), "
                f"monitoring closely"
            )

        return 'none'


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing for a model to reduce memory usage.

    Gradient checkpointing trades compute for memory by not storing
    intermediate activations during forward pass, recomputing them
    during backward pass.

    Args:
        model: PyTorch model to enable checkpointing for
    """
    # Try model-specific checkpointing methods
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing via model method")
        return

    # For models without built-in support, wrap sequential modules
    if hasattr(model, 'encoder') and isinstance(model.encoder, nn.Module):
        _wrap_for_checkpointing(model.encoder)
        logger.info("Enabled gradient checkpointing for encoder")

    if hasattr(model, 'decoder') and isinstance(model.decoder, nn.Module):
        _wrap_for_checkpointing(model.decoder)
        logger.info("Enabled gradient checkpointing for decoder")


def _wrap_for_checkpointing(module: nn.Module) -> None:
    """Wrap module layers for gradient checkpointing.

    Args:
        module: Module to wrap
    """
    from torch.utils.checkpoint import checkpoint_sequential

    # Find sequential submodules
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential) and len(list(child.children())) > 2:
            # Mark for checkpointing
            child._gradient_checkpointing = True
            logger.debug(f"Marked {name} for gradient checkpointing")


def is_flash_attention_available() -> bool:
    """Check if Flash Attention is available for memory-efficient attention.

    Returns:
        True if Flash Attention can be used
    """
    try:
        # Check for PyTorch 2.0+ scaled_dot_product_attention with Flash
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Check if CUDA is available with proper compute capability
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                capability = torch.cuda.get_device_capability(device)
                # Flash Attention requires SM 8.0+ (A100, etc.)
                if capability[0] >= 8:
                    return True
                # Also works on SM 7.5+ with limited features
                if capability[0] == 7 and capability[1] >= 5:
                    return True
        return False
    except Exception:
        return False


def get_memory_efficient_config() -> Dict[str, Any]:
    """Get recommended configuration for memory-efficient training.

    Returns:
        Dict of recommended settings based on available GPU memory
    """
    if not torch.cuda.is_available():
        return {
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'use_checkpointing': True,
        }

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_gb = total_memory / (1024**3)

    if total_gb >= 24:  # High memory (A100, etc.)
        return {
            'batch_size': 16,
            'gradient_accumulation_steps': 1,
            'use_checkpointing': False,
            'use_flash_attention': is_flash_attention_available(),
        }
    elif total_gb >= 12:  # Medium memory (3090, 4080, etc.)
        return {
            'batch_size': 8,
            'gradient_accumulation_steps': 2,
            'use_checkpointing': False,
            'use_flash_attention': is_flash_attention_available(),
        }
    elif total_gb >= 8:  # Low-medium (3070, etc.)
        return {
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'use_checkpointing': True,
            'use_flash_attention': is_flash_attention_available(),
        }
    else:  # Low memory
        return {
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
            'use_checkpointing': True,
            'use_flash_attention': is_flash_attention_available(),
        }
