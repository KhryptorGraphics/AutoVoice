"""GPU performance monitoring utilities"""
import torch
import pynvml
import time
from typing import Dict, Any, Optional


class PerformanceMonitor:
    """Monitors GPU performance metrics"""

    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = torch.cuda.device_count()

    def get_gpu_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """Get comprehensive GPU statistics"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Get various metrics
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

        return {
            'gpu_utilization': utilization.gpu,
            'memory_utilization': utilization.memory,
            'temperature': temperature,
            'power_watts': power,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3)
        }

    def profile_operation(self, func, *args, **kwargs) -> tuple:
        """Profile a GPU operation"""
        torch.cuda.synchronize()
        start_time = time.time()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end_time = time.time()

        return result, end_time - start_time

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass