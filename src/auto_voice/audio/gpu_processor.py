"""
GPU-optimized audio processor with proper device handling and CUDA kernel validation
"""

import torch
from .processor import AudioProcessor

# Try to import CUDA kernels for validation
try:
    from auto_voice import cuda_kernels as _cuda_kernels
except ImportError:
    _cuda_kernels = None


class GPUAudioProcessor(AudioProcessor):
    """GPU-optimized audio processor.

    Inherits from AudioProcessor and provides GPU-specific optimizations for audio processing.
    Falls back to CPU when GPU is unavailable unless force_gpu is True.

    Args:
        device: Device specification ('cuda', 'cuda:0', 'cpu', etc.)
        force_gpu: If True, raises RuntimeError when GPU or CUDA kernels unavailable
    """

    def __init__(self, device: str = 'cuda', force_gpu: bool = False, config: dict = None):
        """Initialize GPU audio processor.

        Args:
            device: Device to use ('cuda', 'cuda:0', 'cpu', etc.)
            force_gpu: Force GPU usage - raises error if GPU/CUDA kernels unavailable
            config: Optional configuration dictionary for audio parameters
        """
        # Parse the device argument to normalize it
        dev = torch.device(device)

        # Determine if we can actually use CUDA
        cuda_available = dev.type == 'cuda' and torch.cuda.is_available()

        # Check force_gpu requirements
        if force_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU processing requested but CUDA is not available")
            if _cuda_kernels is None:
                raise RuntimeError("GPU processing requested but CUDA kernels are not available")

        # Set the actual device based on availability
        if cuda_available:
            actual_device = 'cuda'
        else:
            actual_device = 'cpu'

        # Initialize base class with config and device
        if config is None:
            config = {}
        super().__init__(config=config, device=actual_device)

        # Store device for use in GPU operations
        self.device = actual_device

        # Move tensors to appropriate device if they exist
        if hasattr(self, 'mel_basis') and self.mel_basis is not None:
            self.mel_basis = self.mel_basis.to(self.device)
        if hasattr(self, 'stft_window') and self.stft_window is not None:
            self.stft_window = self.stft_window.to(self.device)