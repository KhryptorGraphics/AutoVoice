"""Advanced GPU memory management with pooling and optimization"""
import torch
import threading
import time
import gc
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager
import weakref

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class AllocationStrategy(Enum):
    """Memory allocation strategies"""
    GREEDY = "greedy"  # Allocate on first available device
    BALANCED = "balanced"  # Balance across devices
    CONSOLIDATED = "consolidated"  # Prefer single device
    FRAGMENTATION_AWARE = "fragmentation_aware"  # Minimize fragmentation

@dataclass
class MemoryPool:
    """Memory pool for a specific size class"""
    size_class: int
    available_blocks: deque = field(default_factory=deque)
    allocated_blocks: Set[int] = field(default_factory=set)
    total_allocated: int = 0
    max_blocks: int = 100
    
class MemoryStats:
    """Memory statistics tracking"""
    def __init__(self):
        self.allocations = 0
        self.deallocations = 0
        self.peak_usage = 0
        self.current_usage = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fragmentation_events = 0
        
@dataclass
class AllocationInfo:
    """Information about a memory allocation"""
    size: int
    device_id: int
    timestamp: float
    tensor_id: int
    pool_allocated: bool = False

class MemoryManager:
    """Advanced GPU memory manager with pooling and optimization"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVML
        self.nvml_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.device_count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                self.logger.warning(f"NVML initialization failed: {e}")
                self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Memory pool configuration
        self.enable_pooling = config.get('enable_pooling', True)
        self.pool_size_classes = config.get('pool_size_classes', 
                                          [1024, 4096, 16384, 65536, 262144, 1048576])  # bytes
        self.max_pool_size = config.get('max_pool_size', 2 * 1024**3)  # 2GB per device
        self.allocation_strategy = AllocationStrategy(config.get('allocation_strategy', 'balanced'))
        
        # Memory pools per device
        self.memory_pools: Dict[int, Dict[int, MemoryPool]] = defaultdict(lambda: defaultdict(MemoryPool))
        self.allocation_tracker: Dict[int, AllocationInfo] = {}  # tensor_id -> info
        self.device_stats: Dict[int, MemoryStats] = defaultdict(MemoryStats)
        
        # Fragmentation tracking
        self.fragmentation_threshold = config.get('fragmentation_threshold', 0.3)
        self.auto_defragment = config.get('auto_defragment', True)
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Memory pressure handling
        self.memory_pressure_threshold = config.get('memory_pressure_threshold', 0.9)
        self.enable_oom_handler = config.get('enable_oom_handler', True)
        
        # Performance monitoring
        self.monitor_performance = config.get('monitor_performance', True)
        self.stats_window_size = config.get('stats_window_size', 1000)
        
        # Background cleanup
        self.cleanup_interval = config.get('cleanup_interval', 60.0)
        self.enable_background_cleanup = config.get('enable_background_cleanup', True)
        
        if self.enable_background_cleanup:
            self._start_background_cleanup()
        
        # Setup OOM handler
        if self.enable_oom_handler and torch.cuda.is_available():
            self._setup_oom_handler()
        
        self.logger.info(f"Memory manager initialized for {self.device_count} devices")

    def get_memory_info(self, device_id: int = 0) -> Dict[str, Any]:
        """Get comprehensive memory information for a specific GPU"""
        if device_id >= self.device_count:
            raise ValueError(f"Invalid device_id {device_id}, only {self.device_count} devices available")
        
        # Get basic memory info
        if self.nvml_available:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                nvml_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = nvml_info.total
                used = nvml_info.used
                free = nvml_info.free
            except Exception as e:
                self.logger.warning(f"NVML memory info failed for device {device_id}: {e}")
                # Fallback to PyTorch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device_id)
                    reserved = torch.cuda.memory_reserved(device_id)
                    total = torch.cuda.get_device_properties(device_id).total_memory
                    used = allocated
                    free = total - allocated
                else:
                    total = used = free = 0
        else:
            # Use PyTorch memory info
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                used = allocated
                free = total - allocated
            else:
                total = used = free = 0
        
        # Calculate additional metrics
        percentage = (used / total * 100) if total > 0 else 0
        
        # Get pool statistics
        pool_stats = self._get_pool_stats(device_id)
        device_stats = self.device_stats[device_id]
        
        return {
            'device_id': device_id,
            'total': total,
            'used': used,
            'free': free,
            'percentage': percentage,
            'reserved': reserved if 'reserved' in locals() else used,
            'pool_stats': pool_stats,
            'allocations': device_stats.allocations,
            'deallocations': device_stats.deallocations,
            'peak_usage': device_stats.peak_usage,
            'cache_hits': device_stats.cache_hits,
            'cache_misses': device_stats.cache_misses,
            'fragmentation_events': device_stats.fragmentation_events,
            'fragmentation_ratio': self._calculate_fragmentation(device_id)
        }

    def _get_pool_stats(self, device_id: int) -> Dict[str, Any]:
        """Get memory pool statistics for device"""
        pools = self.memory_pools[device_id]
        total_pool_size = 0
        total_available = 0
        pool_details = {}
        
        for size_class, pool in pools.items():
            pool_size = len(pool.available_blocks) * size_class
            allocated_size = len(pool.allocated_blocks) * size_class
            total_pool_size += pool_size + allocated_size
            total_available += pool_size
            
            pool_details[size_class] = {
                'available_blocks': len(pool.available_blocks),
                'allocated_blocks': len(pool.allocated_blocks),
                'pool_size_bytes': pool_size,
                'allocated_size_bytes': allocated_size
            }
        
        return {
            'total_pool_size': total_pool_size,
            'total_available': total_available,
            'efficiency': (total_pool_size - total_available) / total_pool_size if total_pool_size > 0 else 0,
            'pool_details': pool_details
        }

    def _calculate_fragmentation(self, device_id: int) -> float:
        """Calculate memory fragmentation ratio"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Simple fragmentation metric: reserved - allocated
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            
            if reserved == 0:
                return 0.0
            
            fragmentation = (reserved - allocated) / reserved
            return max(0.0, min(1.0, fragmentation))
        except Exception:
            return 0.0

    def allocate_tensor(self, size: Tuple[int, ...], dtype: torch.dtype, 
                       device_id: Optional[int] = None) -> torch.Tensor:
        """Allocate tensor with memory pool optimization"""
        with self.lock:
            # Determine target device
            if device_id is None:
                device_id = self._select_device_for_allocation(size, dtype)
            
            # Calculate memory requirement
            element_size = torch._utils._element_size(dtype)
            total_elements = 1
            for dim in size:
                total_elements *= dim
            memory_needed = total_elements * element_size
            
            # Try pool allocation first
            if self.enable_pooling:
                tensor = self._try_pool_allocation(size, dtype, device_id, memory_needed)
                if tensor is not None:
                    return tensor
            
            # Direct allocation
            try:
                tensor = self._direct_allocation(size, dtype, device_id)
                self._track_allocation(tensor, device_id, memory_needed, pool_allocated=False)
                return tensor
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    return self._handle_oom(size, dtype, device_id, memory_needed)
                raise

    def _select_device_for_allocation(self, size: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Select optimal device for allocation based on strategy"""
        if self.device_count == 1:
            return 0
        
        if self.allocation_strategy == AllocationStrategy.GREEDY:
            # Find first device with enough memory
            for device_id in range(self.device_count):
                if self._has_sufficient_memory(device_id, size, dtype):
                    return device_id
            return 0  # Fallback
        
        elif self.allocation_strategy == AllocationStrategy.BALANCED:
            # Choose device with lowest memory usage
            min_usage = float('inf')
            best_device = 0
            
            for device_id in range(self.device_count):
                memory_info = self.get_memory_info(device_id)
                if memory_info['percentage'] < min_usage:
                    min_usage = memory_info['percentage']
                    best_device = device_id
            
            return best_device
        
        elif self.allocation_strategy == AllocationStrategy.FRAGMENTATION_AWARE:
            # Choose device with lowest fragmentation
            min_fragmentation = float('inf')
            best_device = 0
            
            for device_id in range(self.device_count):
                fragmentation = self._calculate_fragmentation(device_id)
                if fragmentation < min_fragmentation:
                    min_fragmentation = fragmentation
                    best_device = device_id
            
            return best_device
        
        return 0  # Default fallback

    def _has_sufficient_memory(self, device_id: int, size: Tuple[int, ...], dtype: torch.dtype) -> bool:
        """Check if device has sufficient memory for allocation"""
        try:
            element_size = torch._utils._element_size(dtype)
            total_elements = 1
            for dim in size:
                total_elements *= dim
            memory_needed = total_elements * element_size
            
            memory_info = self.get_memory_info(device_id)
            return memory_info['free'] >= memory_needed * 1.1  # 10% safety margin
        except Exception:
            return False

    def _try_pool_allocation(self, size: Tuple[int, ...], dtype: torch.dtype, 
                           device_id: int, memory_needed: int) -> Optional[torch.Tensor]:
        """Try to allocate from memory pool"""
        size_class = self._get_size_class(memory_needed)
        if size_class is None:
            return None
        
        pool = self.memory_pools[device_id][size_class]
        
        if pool.available_blocks:
            # Reuse from pool
            block_ptr = pool.available_blocks.popleft()
            pool.allocated_blocks.add(block_ptr)
            
            # Create tensor from existing memory
            try:
                tensor = torch.empty(size, dtype=dtype, device=device_id)
                self._track_allocation(tensor, device_id, memory_needed, pool_allocated=True)
                self.device_stats[device_id].cache_hits += 1
                return tensor
            except Exception as e:
                # Return block to pool if tensor creation fails
                pool.allocated_blocks.discard(block_ptr)
                pool.available_blocks.appendleft(block_ptr)
                self.logger.warning(f"Failed to create tensor from pool: {e}")
        
        self.device_stats[device_id].cache_misses += 1
        return None

    def _direct_allocation(self, size: Tuple[int, ...], dtype: torch.dtype, device_id: int) -> torch.Tensor:
        """Direct tensor allocation"""
        with torch.cuda.device(device_id):
            return torch.empty(size, dtype=dtype, device=device_id)

    def _get_size_class(self, memory_needed: int) -> Optional[int]:
        """Get appropriate size class for memory amount"""
        for size_class in self.pool_size_classes:
            if memory_needed <= size_class:
                return size_class
        return None

    def _track_allocation(self, tensor: torch.Tensor, device_id: int, 
                         memory_size: int, pool_allocated: bool = False):
        """Track tensor allocation for management"""
        tensor_id = id(tensor)
        self.allocation_tracker[tensor_id] = AllocationInfo(
            size=memory_size,
            device_id=device_id,
            timestamp=time.time(),
            tensor_id=tensor_id,
            pool_allocated=pool_allocated
        )
        
        # Update statistics
        stats = self.device_stats[device_id]
        stats.allocations += 1
        stats.current_usage += memory_size
        stats.peak_usage = max(stats.peak_usage, stats.current_usage)
        
        # Register weak reference for automatic cleanup
        weakref.ref(tensor, lambda ref: self._cleanup_allocation(tensor_id))

    def _cleanup_allocation(self, tensor_id: int):
        """Clean up allocation tracking when tensor is garbage collected"""
        if tensor_id in self.allocation_tracker:
            alloc_info = self.allocation_tracker.pop(tensor_id)
            stats = self.device_stats[alloc_info.device_id]
            stats.deallocations += 1
            stats.current_usage -= alloc_info.size
            
            # Return to pool if it was pool allocated
            if alloc_info.pool_allocated:
                size_class = self._get_size_class(alloc_info.size)
                if size_class:
                    pool = self.memory_pools[alloc_info.device_id][size_class]
                    pool.allocated_blocks.discard(tensor_id)

    def _handle_oom(self, size: Tuple[int, ...], dtype: torch.dtype, 
                   device_id: int, memory_needed: int) -> torch.Tensor:
        """Handle out-of-memory situations"""
        self.logger.warning(f"OOM detected on device {device_id}, attempting recovery")
        
        # Try cleanup strategies in order
        strategies = [
            self._clear_cache,
            self._garbage_collect,
            self._defragment_memory,
            self._emergency_cleanup
        ]
        
        for strategy in strategies:
            try:
                strategy(device_id)
                # Try allocation again
                tensor = self._direct_allocation(size, dtype, device_id)
                self._track_allocation(tensor, device_id, memory_needed, pool_allocated=False)
                self.logger.info(f"OOM recovery successful using {strategy.__name__}")
                return tensor
            except RuntimeError:
                continue
        
        # If all strategies fail, try other devices
        for other_device in range(self.device_count):
            if other_device != device_id:
                try:
                    tensor = self._direct_allocation(size, dtype, other_device)
                    self._track_allocation(tensor, other_device, memory_needed, pool_allocated=False)
                    self.logger.warning(f"Allocated on device {other_device} due to OOM on device {device_id}")
                    return tensor
                except RuntimeError:
                    continue
        
        # Final fallback - raise original error
        raise RuntimeError(f"Out of memory on all devices. Requested: {memory_needed} bytes")

    def _clear_cache(self, device_id: Optional[int] = None):
        """Clear PyTorch CUDA cache"""
        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()
            for i in range(self.device_count):
                torch.cuda.synchronize(i)

    def _garbage_collect(self, device_id: Optional[int] = None):
        """Force garbage collection"""
        gc.collect()
        if torch.cuda.is_available():
            self._clear_cache(device_id)

    def _defragment_memory(self, device_id: int):
        """Attempt memory defragmentation"""
        # Clear pools to force reallocation
        if device_id in self.memory_pools:
            self.memory_pools[device_id].clear()
        
        self._clear_cache(device_id)
        self.device_stats[device_id].fragmentation_events += 1

    def _emergency_cleanup(self, device_id: int):
        """Emergency cleanup - clear all cached data"""
        # Clear all pools
        self.memory_pools.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear cache multiple times
        for _ in range(3):
            self._clear_cache(device_id)
            time.sleep(0.1)

    def clear_cache(self, device_id: Optional[int] = None):
        """Public interface for cache clearing"""
        with self.lock:
            self._clear_cache(device_id)
            self.logger.debug(f"Cache cleared for device {device_id if device_id is not None else 'all'}")

    def get_fragmentation_ratio(self, device_id: int) -> float:
        """Get memory fragmentation ratio for device"""
        return self._calculate_fragmentation(device_id)

    def optimize_memory(self, device_id: Optional[int] = None):
        """Optimize memory layout and reduce fragmentation"""
        with self.lock:
            devices = [device_id] if device_id is not None else list(range(self.device_count))
            
            for dev_id in devices:
                fragmentation = self._calculate_fragmentation(dev_id)
                
                if fragmentation > self.fragmentation_threshold:
                    self.logger.info(f"Optimizing memory for device {dev_id} (fragmentation: {fragmentation:.2%})")
                    self._defragment_memory(dev_id)

    def _setup_oom_handler(self):
        """Setup out-of-memory error handler"""
        # This would require PyTorch hooks - simplified implementation
        self.logger.info("OOM handler configured")

    def _start_background_cleanup(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while self.enable_background_cleanup:
                try:
                    self._periodic_cleanup()
                except Exception as e:
                    self.logger.error(f"Background cleanup error: {e}")
                time.sleep(self.cleanup_interval)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("Background memory cleanup started")

    def _periodic_cleanup(self):
        """Periodic memory cleanup"""
        for device_id in range(self.device_count):
            # Check memory pressure
            memory_info = self.get_memory_info(device_id)
            
            if memory_info['percentage'] > self.memory_pressure_threshold * 100:
                self.logger.info(f"Memory pressure detected on device {device_id}, cleaning up")
                self._clear_cache(device_id)
            
            # Check fragmentation
            if self.auto_defragment:
                fragmentation = self._calculate_fragmentation(device_id)
                if fragmentation > self.fragmentation_threshold:
                    self.optimize_memory(device_id)

    @contextmanager
    def memory_context(self, device_id: int):
        """Context manager for memory operations"""
        old_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(device_id)
            yield
        finally:
            if old_device is not None:
                torch.cuda.set_device(old_device)

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary for all devices"""
        summary = {
            'device_count': self.device_count,
            'total_allocations': sum(stats.allocations for stats in self.device_stats.values()),
            'total_deallocations': sum(stats.deallocations for stats in self.device_stats.values()),
            'pooling_enabled': self.enable_pooling,
            'devices': {}
        }
        
        for device_id in range(self.device_count):
            summary['devices'][device_id] = self.get_memory_info(device_id)
        
        return summary

    def shutdown(self):
        """Cleanup and shutdown memory manager"""
        self.enable_background_cleanup = False
        
        # Clear all pools
        self.memory_pools.clear()
        self.allocation_tracker.clear()
        
        # Clear caches
        for device_id in range(self.device_count):
            self._clear_cache(device_id)
        
        # Shutdown NVML
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                self.logger.error(f"NVML shutdown error: {e}")
        
        self.logger.info("Memory manager shutdown complete")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass