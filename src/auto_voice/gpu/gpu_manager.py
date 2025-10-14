"""High-level GPU management and coordination for AutoVoice"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .cuda_manager import CUDAManager, DeviceInfo, DeviceState
from .memory_manager import MemoryManager, AllocationStrategy
from .performance_monitor import PerformanceMonitor, AlertLevel, MetricType

class OptimizationLevel(Enum):
    """GPU optimization levels"""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"

class ModelPrecision(Enum):
    """Model precision types"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    MIXED = "mixed"

@dataclass
class GPUConfig:
    """GPU configuration"""
    device_ids: Optional[List[int]] = None
    memory_fraction: Optional[float] = None
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    precision: ModelPrecision = ModelPrecision.MIXED
    enable_amp: bool = True
    enable_compile: bool = True
    enable_channels_last: bool = True
    enable_memory_pooling: bool = True
    enable_monitoring: bool = True
    allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED
    max_memory_cache_ratio: float = 0.8
    enable_profiling: bool = False
    distributed_backend: str = "nccl"
    enable_distributed: bool = False

class GPUManager:
    """High-level GPU management and coordination system"""
    
    def __init__(self, config: Union[Dict[str, Any], GPUConfig]):
        """Initialize GPU manager with comprehensive configuration"""
        # Convert dict config to GPUConfig if needed
        if isinstance(config, dict):
            self.config = self._convert_dict_config(config)
        else:
            self.config = config
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        
        # Initialize subsystems
        self.cuda_manager: Optional[CUDAManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # State tracking
        self.initialized = False
        self.current_device = None
        self.optimized_models = {}
        self.model_cache = {}
        
        # Performance tracking
        self.optimization_stats = {
            'models_optimized': 0,
            'memory_savings_mb': 0,
            'speedup_ratios': [],
            'compilation_times': []
        }
        
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available - GPU manager disabled")
            return
        
        # Initialize GPU subsystems
        self.initialize()
    
    def _convert_dict_config(self, config_dict: Dict[str, Any]) -> GPUConfig:
        """Convert dictionary configuration to GPUConfig"""
        return GPUConfig(
            device_ids=config_dict.get('device_ids'),
            memory_fraction=config_dict.get('memory_fraction'),
            optimization_level=OptimizationLevel(config_dict.get('optimization_level', 'basic')),
            precision=ModelPrecision(config_dict.get('precision', 'mixed')),
            enable_amp=config_dict.get('enable_amp', True),
            enable_compile=config_dict.get('enable_compile', True),
            enable_channels_last=config_dict.get('enable_channels_last', True),
            enable_memory_pooling=config_dict.get('enable_memory_pooling', True),
            enable_monitoring=config_dict.get('enable_monitoring', True),
            allocation_strategy=AllocationStrategy(config_dict.get('allocation_strategy', 'balanced')),
            max_memory_cache_ratio=config_dict.get('max_memory_cache_ratio', 0.8),
            enable_profiling=config_dict.get('enable_profiling', False),
            distributed_backend=config_dict.get('distributed_backend', 'nccl'),
            enable_distributed=config_dict.get('enable_distributed', False)
        )
    
    def initialize(self) -> bool:
        """Initialize all GPU subsystems"""
        if not TORCH_AVAILABLE:
            self.logger.error("Cannot initialize GPU manager - PyTorch not available")
            return False
        
        if self.initialized:
            return True
        
        with self.lock:
            try:
                # Initialize CUDA manager
                cuda_config = {
                    'max_retries': 3,
                    'retry_delay': 0.1,
                    'enable_health_check': True,
                    'health_check_interval': 30.0
                }
                self.cuda_manager = CUDAManager(cuda_config)
                
                if not self.cuda_manager.initialize():
                    self.logger.error("Failed to initialize CUDA manager")
                    return False
                
                # Initialize memory manager
                memory_config = {
                    'enable_pooling': self.config.enable_memory_pooling,
                    'allocation_strategy': self.config.allocation_strategy.value,
                    'max_pool_size': 2 * 1024**3,  # 2GB
                    'enable_oom_handler': True,
                    'monitor_performance': True
                }
                self.memory_manager = MemoryManager(memory_config)
                
                # Initialize performance monitor
                if self.config.enable_monitoring:
                    monitor_config = {
                        'sampling_interval': 1.0,
                        'enable_alerting': True,
                        'enable_profiling': self.config.enable_profiling,
                        'enable_continuous_monitoring': True
                    }
                    self.performance_monitor = PerformanceMonitor(monitor_config)
                    
                    # Setup alert callbacks
                    self.performance_monitor.add_alert_callback(self._handle_performance_alert)
                
                # Set memory fraction if specified
                if self.config.memory_fraction:
                    for device_id in range(self.cuda_manager.get_device_count()):
                        with torch.cuda.device(device_id):
                            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
                
                # Set default device
                best_device = self.cuda_manager.get_best_device()
                if best_device is not None:
                    self.current_device = best_device
                    self.cuda_manager.set_device(best_device)
                
                # Initialize distributed training if enabled
                if self.config.enable_distributed:
                    self._initialize_distributed()
                
                self.initialized = True
                self.logger.info(f"GPU manager initialized with {self.cuda_manager.get_device_count()} devices")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize GPU manager: {e}")
                return False
    
    def _initialize_distributed(self):
        """Initialize distributed training setup"""
        try:
            if not dist.is_available():
                self.logger.warning("Distributed training not available")
                return
            
            if not dist.is_initialized():
                # Initialize distributed process group
                dist.init_process_group(
                    backend=self.config.distributed_backend,
                    init_method='env://'
                )
                self.logger.info(f"Distributed training initialized with backend: {self.config.distributed_backend}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
    
    def _handle_performance_alert(self, alert):
        """Handle performance alerts"""
        if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self.logger.error(f"Performance alert: {alert.message}")
            
            # Take automatic actions for critical alerts
            if alert.level == AlertLevel.CRITICAL:
                if alert.metric_type == MetricType.MEMORY_UTILIZATION:
                    self.emergency_memory_cleanup()
                elif alert.metric_type == MetricType.TEMPERATURE:
                    self.reduce_workload(alert.device_id)
    
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup"""
        try:
            # Clear model cache
            self.model_cache.clear()
            
            # Force memory cleanup
            if self.memory_manager:
                self.memory_manager.clear_cache()
            
            # Garbage collect
            import gc
            gc.collect()
            
            self.logger.info("Emergency memory cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Emergency memory cleanup failed: {e}")
    
    def reduce_workload(self, device_id: int):
        """Reduce workload on overheating device"""
        try:
            device_info = self.cuda_manager.get_device_info(device_id)
            if device_info:
                device_info.state = DeviceState.DISABLED
                self.logger.warning(f"Device {device_id} disabled due to overheating")
        
        except Exception as e:
            self.logger.error(f"Failed to reduce workload on device {device_id}: {e}")
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        return TORCH_AVAILABLE and torch.cuda.is_available() and self.initialized
    
    def get_device(self, device_id: Optional[int] = None) -> torch.device:
        """Get PyTorch device object"""
        if not self.is_cuda_available():
            return torch.device('cpu')
        
        if device_id is None:
            device_id = self.current_device or 0
        
        return torch.device(f'cuda:{device_id}')
    
    def get_best_device(self, memory_required: int = 0) -> Optional[int]:
        """Get the best available device for allocation"""
        if not self.cuda_manager:
            return None
        
        return self.cuda_manager.get_best_device(memory_required)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU status"""
        if not TORCH_AVAILABLE:
            return {
                'torch_available': False,
                'cuda_available': False,
                'status': 'PyTorch not installed'
            }
        
        if not self.is_cuda_available():
            return {
                'torch_available': True,
                'cuda_available': False,
                'device': 'cpu',
                'status': 'CUDA not available or not initialized'
            }
        
        # Get comprehensive status from all subsystems
        status = {
            'torch_available': True,
            'cuda_available': True,
            'initialized': self.initialized,
            'current_device': self.current_device,
            'device_count': self.cuda_manager.get_device_count(),
            'optimization_stats': self.optimization_stats.copy()
        }
        
        # Add device information
        if self.cuda_manager:
            status['devices'] = {}
            for device_id in range(self.cuda_manager.get_device_count()):
                device_info = self.cuda_manager.get_device_info(device_id)
                if device_info:
                    status['devices'][device_id] = {
                        'name': device_info.name,
                        'compute_capability': device_info.compute_capability,
                        'total_memory_gb': device_info.total_memory / (1024**3),
                        'state': device_info.state.value,
                        'temperature': device_info.temperature,
                        'utilization': device_info.utilization
                    }
        
        # Add memory information
        if self.memory_manager:
            status['memory_summary'] = self.memory_manager.get_memory_summary()
        
        # Add performance information
        if self.performance_monitor:
            status['performance_summary'] = self.performance_monitor.get_performance_summary()
        
        return status
    
    def optimize_model(self, model: torch.nn.Module, model_name: str = "model") -> torch.nn.Module:
        """Optimize model for GPU execution with advanced techniques"""
        if not self.is_cuda_available():
            self.logger.warning("CUDA not available, returning model without optimization")
            return model
        
        model_id = f"{model_name}_{id(model)}"
        
        # Check cache first
        if model_id in self.model_cache:
            self.logger.debug(f"Returning cached optimized model: {model_name}")
            return self.model_cache[model_id]
        
        start_time = time.time()
        
        with self.lock:
            try:
                # Move to device
                device = self.get_device()
                model = model.to(device)
                
                # Apply optimizations based on level
                if self.config.optimization_level != OptimizationLevel.NONE:
                    model = self._apply_optimizations(model, model_name)
                
                # Setup precision
                model = self._setup_precision(model)
                
                # Enable channels last memory format if supported
                if self.config.enable_channels_last:
                    try:
                        model = model.to(memory_format=torch.channels_last)
                    except Exception as e:
                        self.logger.debug(f"Channels last not supported for {model_name}: {e}")
                
                # Compile model if enabled
                if self.config.enable_compile and hasattr(torch, 'compile'):
                    try:
                        model = torch.compile(model)
                        self.logger.info(f"Model {model_name} compiled successfully")
                    except Exception as e:
                        self.logger.warning(f"Model compilation failed for {model_name}: {e}")
                
                # Cache optimized model
                self.model_cache[model_id] = model
                
                # Update statistics
                compilation_time = time.time() - start_time
                self.optimization_stats['models_optimized'] += 1
                self.optimization_stats['compilation_times'].append(compilation_time)
                
                self.logger.info(f"Model {model_name} optimized in {compilation_time:.2f}s")
                return model
                
            except Exception as e:
                self.logger.error(f"Model optimization failed for {model_name}: {e}")
                return model
    
    def _apply_optimizations(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Apply model optimizations based on optimization level"""
        if self.config.optimization_level == OptimizationLevel.BASIC:
            # Basic optimizations
            model.eval()  # Set to eval mode for inference optimizations
            
        elif self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
            # Aggressive optimizations
            model.eval()
            
            # Try to fuse operations
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
                self.logger.debug(f"JIT optimization applied to {model_name}")
            except Exception as e:
                self.logger.debug(f"JIT optimization failed for {model_name}: {e}")
        
        elif self.config.optimization_level == OptimizationLevel.EXPERIMENTAL:
            # Experimental optimizations
            model.eval()
            
            # Try advanced fusion techniques
            try:
                # This would include experimental PyTorch optimizations
                pass
            except Exception as e:
                self.logger.debug(f"Experimental optimization failed for {model_name}: {e}")
        
        return model
    
    def _setup_precision(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup model precision based on configuration"""
        if self.config.precision == ModelPrecision.FP16:
            model = model.half()
        elif self.config.precision == ModelPrecision.BF16:
            if torch.cuda.is_bf16_supported():
                model = model.to(torch.bfloat16)
            else:
                self.logger.warning("BF16 not supported, falling back to FP16")
                model = model.half()
        elif self.config.precision == ModelPrecision.INT8:
            # This would require quantization libraries
            self.logger.warning("INT8 quantization not implemented, using FP32")
        # FP32 and MIXED precision handled automatically by AMP
        
        return model
    
    @contextmanager
    def device_context(self, device_id: Optional[int] = None):
        """Context manager for device operations"""
        if not self.is_cuda_available():
            yield torch.device('cpu')
            return
        
        target_device = device_id if device_id is not None else self.current_device
        
        if target_device is None:
            yield torch.device('cpu')
            return
        
        with self.cuda_manager.device_context(target_device):
            yield self.get_device(target_device)
    
    @contextmanager
    def profile_context(self, operation_name: str, device_id: Optional[int] = None):
        """Context manager for operation profiling"""
        if not self.performance_monitor or not self.is_cuda_available():
            yield
            return
        
        target_device = device_id if device_id is not None else self.current_device or 0
        
        with self.performance_monitor.profile_operation(operation_name, target_device):
            yield
    
    def clear_cache(self, device_id: Optional[int] = None):
        """Clear GPU memory cache"""
        if not self.is_cuda_available():
            return
        
        if self.memory_manager:
            self.memory_manager.clear_cache(device_id)
        else:
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        
        self.logger.info(f"GPU cache cleared for device {device_id or 'all'}")
    
    def optimize_memory(self, device_id: Optional[int] = None):
        """Optimize memory usage"""
        if self.memory_manager:
            self.memory_manager.optimize_memory(device_id)
        else:
            self.clear_cache(device_id)
    
    def get_memory_info(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get memory information"""
        if not self.is_cuda_available():
            return {'error': 'CUDA not available'}
        
        target_device = device_id if device_id is not None else self.current_device or 0
        
        if self.memory_manager:
            return self.memory_manager.get_memory_info(target_device)
        else:
            # Fallback to basic PyTorch memory info
            allocated = torch.cuda.memory_allocated(target_device)
            total = torch.cuda.get_device_properties(target_device).total_memory
            return {
                'device_id': target_device,
                'total': total,
                'used': allocated,
                'free': total - allocated,
                'percentage': (allocated / total) * 100
            }
    
    def benchmark_operation(self, operation_name: str, operation_func, 
                          num_runs: int = 10, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Benchmark GPU operation"""
        if not self.performance_monitor:
            raise RuntimeError("Performance monitor not initialized")
        
        target_device = device_id if device_id is not None else self.current_device or 0
        
        return self.performance_monitor.benchmark_operation(
            operation_name, operation_func, num_runs, device_id=target_device
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if self.performance_monitor:
            return self.performance_monitor.get_performance_summary()
        else:
            return {'error': 'Performance monitor not available'}
    
    def export_performance_data(self, filepath: str):
        """Export performance data to file"""
        if self.performance_monitor:
            self.performance_monitor.export_metrics(filepath)
        else:
            self.logger.warning("Performance monitor not available for export")
    
    def shutdown(self):
        """Shutdown GPU manager and cleanup resources"""
        try:
            # Clear model cache
            self.model_cache.clear()
            
            # Shutdown subsystems
            if self.performance_monitor:
                self.performance_monitor.shutdown()
            
            if self.memory_manager:
                self.memory_manager.shutdown()
            
            if self.cuda_manager:
                self.cuda_manager.shutdown()
            
            # Cleanup distributed if initialized
            if self.config.enable_distributed and dist.is_initialized():
                dist.destroy_process_group()
            
            self.initialized = False
            self.logger.info("GPU manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during GPU manager shutdown: {e}")
    
    def __del__(self):
        """Destructor cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
