"""GPU memory management utilities"""
import torch
import pynvml
from typing import Optional, Dict, Any


class MemoryManager:
    """Manages GPU memory allocation and monitoring"""

    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

    def get_memory_info(self, device_id: int = 0) -> Dict[str, Any]:
        """Get memory information for a specific GPU"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'total': info.total,
            'used': info.used,
            'free': info.free,
            'percentage': (info.used / info.total) * 100
        }

    def clear_cache(self):
        """Clear PyTorch CUDA cache"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass