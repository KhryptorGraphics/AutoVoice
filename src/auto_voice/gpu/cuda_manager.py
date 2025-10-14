"""CUDA device management and initialization with comprehensive error handling"""
import torch
import time
import threading
from typing import List, Dict, Any, Optional, Callable
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class CUDAError(Exception):
    """Custom CUDA error exception"""
    pass

class DeviceState(Enum):
    """Device state enumeration"""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class DeviceInfo:
    """Device information container"""
    device_id: int
    name: str
    compute_capability: tuple
    total_memory: int
    multiprocessor_count: int
    state: DeviceState
    temperature: Optional[float] = None
    utilization: Optional[float] = None
    power_usage: Optional[float] = None

class CUDAManager:
    """Comprehensive CUDA device manager with error handling and monitoring"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.devices: List[int] = []
        self.device_info: Dict[int, DeviceInfo] = {}
        self.current_device: Optional[int] = None
        self.nvml_initialized = False
        self.cuda_initialized = False
        self.error_callbacks: List[Callable] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Error handling configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 0.1)
        self.enable_health_check = config.get('enable_health_check', True)
        self.health_check_interval = config.get('health_check_interval', 30.0)
        
        # Start health monitoring if enabled
        if self.enable_health_check:
            self._start_health_monitor()

    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add error callback for custom error handling"""
        self.error_callbacks.append(callback)

    def _handle_error(self, operation: str, error: Exception) -> None:
        """Handle errors with callbacks and logging"""
        error_msg = f"CUDA error in {operation}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        for callback in self.error_callbacks:
            try:
                callback(operation, error)
            except Exception as cb_error:
                self.logger.error(f"Error in error callback: {cb_error}")

    def _retry_operation(self, operation: Callable, operation_name: str, *args, **kwargs):
        """Retry operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {operation_name} after {delay}s")
                    time.sleep(delay)
                else:
                    self._handle_error(operation_name, e)
                    raise CUDAError(f"Failed {operation_name} after {self.max_retries} retries: {e}")
        
        if last_exception:
            raise last_exception

    def initialize(self) -> bool:
        """Initialize CUDA with comprehensive error handling"""
        with self.lock:
            try:
                # Check CUDA availability
                if not torch.cuda.is_available():
                    self.logger.error("CUDA not available on this system")
                    return False

                # Initialize NVML if available
                if PYNVML_AVAILABLE and not self.nvml_initialized:
                    try:
                        pynvml.nvmlInit()
                        self.nvml_initialized = True
                        self.logger.info("NVML initialized successfully")
                    except pynvml.NVMLError as e:
                        self.logger.warning(f"NVML initialization failed: {e}")
                        self.nvml_initialized = False
                else:
                    self.logger.warning("pynvml not available, limited GPU monitoring")

                # Detect and validate devices
                device_count = torch.cuda.device_count()
                self.logger.info(f"Detected {device_count} CUDA devices")

                if device_count == 0:
                    self.logger.error("No CUDA devices found")
                    return False

                # Initialize each device
                for device_id in range(device_count):
                    if self._initialize_device(device_id):
                        self.devices.append(device_id)
                    else:
                        self.logger.warning(f"Failed to initialize device {device_id}")

                if not self.devices:
                    self.logger.error("No CUDA devices could be initialized")
                    return False

                # Set default device
                self.current_device = self.devices[0]
                torch.cuda.set_device(self.current_device)
                
                self.cuda_initialized = True
                self.logger.info(f"CUDA initialized with {len(self.devices)} devices")
                return True

            except Exception as e:
                self._handle_error("CUDA initialization", e)
                return False

    def _initialize_device(self, device_id: int) -> bool:
        """Initialize a specific CUDA device"""
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(device_id)
            
            # Create device info
            device_info = DeviceInfo(
                device_id=device_id,
                name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory=props.total_memory,
                multiprocessor_count=props.multi_processor_count,
                state=DeviceState.AVAILABLE
            )
            
            # Test device functionality
            if self._test_device(device_id):
                # Get additional metrics if NVML available
                if self.nvml_initialized:
                    self._update_device_metrics(device_info)
                
                self.device_info[device_id] = device_info
                self.logger.info(f"Device {device_id} initialized: {props.name} ")
                return True
            else:
                self.logger.error(f"Device {device_id} failed functionality test")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize device {device_id}: {e}")
            return False

    def _test_device(self, device_id: int) -> bool:
        """Test device functionality with a simple operation"""
        try:
            with torch.cuda.device(device_id):
                # Test memory allocation and basic operations
                test_tensor = torch.randn(100, 100, device=device_id)
                result = torch.matmul(test_tensor, test_tensor.t())
                torch.cuda.synchronize(device_id)
                del test_tensor, result
                torch.cuda.empty_cache()
                return True
        except Exception as e:
            self.logger.error(f"Device {device_id} functionality test failed: {e}")
            return False

    def _update_device_metrics(self, device_info: DeviceInfo) -> None:
        """Update device metrics using NVML"""
        if not self.nvml_initialized:
            return
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_info.device_id)
            
            # Get temperature
            try:
                device_info.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                pass
            
            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                device_info.utilization = util.gpu
            except pynvml.NVMLError:
                pass
            
            # Get power usage
            try:
                device_info.power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                pass
                
        except Exception as e:
            self.logger.warning(f"Failed to update metrics for device {device_info.device_id}: {e}")

    def get_device_count(self) -> int:
        """Get number of available CUDA devices"""
        return len(self.devices)

    def get_device_info(self, device_id: int) -> Optional[DeviceInfo]:
        """Get comprehensive device information"""
        device_info = self.device_info.get(device_id)
        if device_info and self.nvml_initialized:
            # Update real-time metrics
            self._update_device_metrics(device_info)
        return device_info

    def get_all_devices_info(self) -> Dict[int, DeviceInfo]:
        """Get information for all devices"""
        info = {}
        for device_id in self.devices:
            info[device_id] = self.get_device_info(device_id)
        return info

    def set_device(self, device_id: int) -> bool:
        """Set current CUDA device with validation"""
        if device_id not in self.devices:
            self.logger.error(f"Invalid device ID: {device_id}")
            return False
            
        device_info = self.device_info.get(device_id)
        if device_info and device_info.state != DeviceState.AVAILABLE:
            self.logger.error(f"Device {device_id} is not available (state: {device_info.state})")
            return False

        try:
            return self._retry_operation(
                lambda: self._set_device_impl(device_id),
                f"set_device_{device_id}"
            )
        except CUDAError:
            return False

    def _set_device_impl(self, device_id: int) -> bool:
        """Implementation of device setting"""
        torch.cuda.set_device(device_id)
        self.current_device = device_id
        self.logger.debug(f"Set current device to {device_id}")
        return True

    def get_current_device(self) -> Optional[int]:
        """Get currently selected device"""
        return self.current_device

    def is_device_available(self, device_id: int) -> bool:
        """Check if device is available for use"""
        device_info = self.device_info.get(device_id)
        return device_info is not None and device_info.state == DeviceState.AVAILABLE

    def get_best_device(self, memory_required: int = 0) -> Optional[int]:
        """Get the best available device based on criteria"""
        best_device = None
        best_score = -1
        
        for device_id in self.devices:
            device_info = self.get_device_info(device_id)
            if not device_info or device_info.state != DeviceState.AVAILABLE:
                continue
                
            # Check memory requirement
            if memory_required > 0:
                available_memory = self._get_available_memory(device_id)
                if available_memory < memory_required:
                    continue
            
            # Calculate device score (higher is better)
            score = self._calculate_device_score(device_info)
            if score > best_score:
                best_score = score
                best_device = device_id
        
        return best_device

    def _get_available_memory(self, device_id: int) -> int:
        """Get available memory for device"""
        try:
            with torch.cuda.device(device_id):
                return torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated()
        except Exception:
            return 0

    def _calculate_device_score(self, device_info: DeviceInfo) -> float:
        """Calculate device performance score"""
        score = 0.0
        
        # Compute capability score
        major, minor = device_info.compute_capability
        score += major * 10 + minor
        
        # Memory score (GB)
        score += (device_info.total_memory / (1024**3)) * 0.1
        
        # Multiprocessor count
        score += device_info.multiprocessor_count * 0.01
        
        # Utilization penalty (prefer less utilized devices)
        if device_info.utilization is not None:
            score -= device_info.utilization * 0.01
        
        # Temperature penalty
        if device_info.temperature is not None:
            if device_info.temperature > 80:
                score -= 1.0
            elif device_info.temperature > 70:
                score -= 0.5
        
        return score

    @contextmanager
    def device_context(self, device_id: int):
        """Context manager for temporary device switching"""
        old_device = self.current_device
        try:
            if self.set_device(device_id):
                yield device_id
            else:
                raise CUDAError(f"Failed to set device {device_id}")
        finally:
            if old_device is not None:
                self.set_device(old_device)

    def _start_health_monitor(self):
        """Start background health monitoring"""
        def health_check():
            while self.enable_health_check:
                try:
                    self._perform_health_check()
                except Exception as e:
                    self.logger.error(f"Health check failed: {e}")
                time.sleep(self.health_check_interval)
        
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
        self.logger.info("GPU health monitoring started")

    def _perform_health_check(self):
        """Perform health check on all devices"""
        for device_id in self.devices:
            device_info = self.device_info.get(device_id)
            if not device_info:
                continue
                
            try:
                # Update metrics
                if self.nvml_initialized:
                    self._update_device_metrics(device_info)
                
                # Check temperature
                if (device_info.temperature is not None and 
                    device_info.temperature > 90):
                    self.logger.warning(f"Device {device_id} temperature high: {device_info.temperature}°C")
                    device_info.state = DeviceState.ERROR
                
                # Test basic functionality
                if not self._test_device(device_id):
                    self.logger.error(f"Device {device_id} failed health check")
                    device_info.state = DeviceState.ERROR
                elif device_info.state == DeviceState.ERROR:
                    # Recovery
                    device_info.state = DeviceState.AVAILABLE
                    self.logger.info(f"Device {device_id} recovered")
                    
            except Exception as e:
                self.logger.error(f"Health check failed for device {device_id}: {e}")
                if device_info:
                    device_info.state = DeviceState.ERROR

    def shutdown(self):
        """Cleanup resources"""
        self.enable_health_check = False
        
        try:
            if self.nvml_initialized:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
        except Exception as e:
            self.logger.error(f"Error during NVML shutdown: {e}")
        
        self.cuda_initialized = False
        self.logger.info("CUDA manager shutdown complete")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass