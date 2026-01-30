"""GPU enforcement utilities for training operations.

Provides centralized GPU requirement checking to ensure all training
operations run on GPU without silent CPU fallback.

Task 4.9: Ensure all training operations run on GPU
"""

import functools
import logging
from typing import Callable, Optional, TypeVar

import torch

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def require_cuda(operation_name: str = "Training") -> None:
    """Verify CUDA is available, raise if not.

    Args:
        operation_name: Name of operation for error message

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for {operation_name}. "
            f"No CUDA-capable GPU detected. "
            f"Training operations must run on GPU to meet performance requirements."
        )


def get_training_device(
    device_id: Optional[int] = None,
    allow_cpu: bool = False,
) -> torch.device:
    """Get the device for training operations.

    Args:
        device_id: Optional specific CUDA device ID
        allow_cpu: If True, allow CPU fallback (for testing only)

    Returns:
        torch.device for training

    Raises:
        RuntimeError: If CUDA unavailable and allow_cpu=False
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")

    if allow_cpu:
        logger.warning(
            "CUDA not available, falling back to CPU. "
            "This should only be used for testing."
        )
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA is required for training. "
        "No CUDA-capable GPU detected. "
        "Set allow_cpu=True only for testing purposes."
    )


def enforce_gpu(func: F) -> F:
    """Decorator to enforce GPU availability before function execution.

    Usage:
        @enforce_gpu
        def train_model(samples):
            ...

    Raises:
        RuntimeError: If CUDA is not available when function is called
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        require_cuda(func.__name__)
        return func(*args, **kwargs)

    return wrapper  # type: ignore


class GPUTrainingContext:
    """Context manager that ensures GPU availability for training blocks.

    Usage:
        with GPUTrainingContext("fine-tuning") as ctx:
            model.to(ctx.device)
            # ... training code ...

    Raises:
        RuntimeError: On context entry if CUDA is not available
    """

    def __init__(
        self,
        operation_name: str = "training",
        device_id: Optional[int] = None,
    ):
        """Initialize GPU training context.

        Args:
            operation_name: Name for error messages
            device_id: Optional specific CUDA device ID
        """
        self.operation_name = operation_name
        self.device_id = device_id
        self._device: Optional[torch.device] = None

    @property
    def device(self) -> torch.device:
        """Get the CUDA device for this context."""
        if self._device is None:
            raise RuntimeError("Device not available outside context")
        return self._device

    def __enter__(self) -> "GPUTrainingContext":
        """Enter context, verifying CUDA availability."""
        require_cuda(self.operation_name)
        self._device = get_training_device(device_id=self.device_id)
        logger.debug(f"Entered GPU training context on {self._device}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context."""
        self._device = None
        return False  # Don't suppress exceptions


def verify_tensor_on_gpu(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Verify a tensor is on GPU, raise if not.

    Args:
        tensor: Tensor to verify
        name: Name for error message

    Raises:
        RuntimeError: If tensor is not on CUDA device
    """
    if not tensor.is_cuda:
        raise RuntimeError(
            f"{name} must be on GPU, but found on {tensor.device}. "
            f"Use tensor.to('cuda') to move to GPU."
        )


def verify_model_on_gpu(model: torch.nn.Module, name: str = "model") -> None:
    """Verify a model's parameters are on GPU.

    Args:
        model: Model to verify
        name: Name for error message

    Raises:
        RuntimeError: If model parameters are not on CUDA device
    """
    try:
        param = next(model.parameters())
        if not param.is_cuda:
            raise RuntimeError(
                f"{name} must be on GPU, but found on {param.device}. "
                f"Use model.to('cuda') to move to GPU."
            )
    except StopIteration:
        # Model has no parameters, can't verify
        pass
