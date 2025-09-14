"""
CUDA device management and initialization
"""
import torch
import pynvml
from typing import List, Dict, Any
import logging

class CUDAManager:
    """Manages CUDA devices and initialization"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.devices: List[int] = []
        self.device_properties: Dict[int, Any] = {}
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """Initialize CUDA and detect available devices"""
        try:
            if not torch.cuda.is_available():
                self.logger.error("CUDA not available")
                return False

            pynvml.nvmlInit()

            device_count = torch.cuda.device_count()
            self.logger.info(f"Found {device_count} CUDA devices")

            for i in range(device_count):
                self.devices.append(i)
                props = torch.cuda.get_device_properties(i)
                self.device_properties[i] = {
                    'name': props.name,
                    'major': props.major,
                    'minor': props.minor,
                    'memory': props.total_memory,
                    'multi_processor_count': props.multi_processor_count
                }
                self.logger.info(f"Device {i}: {props.name}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize CUDA: {e}")
            return False

    def get_device_count(self) -> int:
        """Get number of available CUDA devices"""
        return len(self.devices)

    def get_device_info(self, device_id: int) -> Dict[str, Any]:
        """Get device properties"""
        return self.device_properties.get(device_id, {})

    def set_device(self, device_id: int) -> bool:
        """Set current CUDA device"""
        try:
            if device_id in self.devices:
                torch.cuda.set_device(device_id)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to set device {device_id}: {e}")
            return False