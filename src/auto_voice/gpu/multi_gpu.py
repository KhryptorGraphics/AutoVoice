"""Multi-GPU coordination and management"""
import torch
import torch.distributed as dist
from typing import Optional, List


class MultiGPUManager:
    """Manages multi-GPU operations and coordination"""

    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.is_distributed = False

    def setup_distributed(self, rank: int, world_size: int, backend: str = 'nccl'):
        """Setup distributed training environment"""
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            dist.init_process_group(backend, rank=rank, world_size=world_size)
            self.is_distributed = True

    def get_device(self, device_id: Optional[int] = None) -> torch.device:
        """Get a specific device or the current device"""
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def distribute_model(self, model: torch.nn.Module, device_ids: Optional[List[int]] = None):
        """Distribute model across multiple GPUs"""
        if self.device_count > 1:
            device_ids = device_ids or list(range(self.device_count))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model.cuda()

    def cleanup(self):
        """Cleanup distributed processes"""
        if self.is_distributed:
            dist.destroy_process_group()