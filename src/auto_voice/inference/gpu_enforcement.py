"""GPU enforcement utilities for inference operations.

Task 7.5: Add strict GPU-only checks (RuntimeError on any CPU fallback attempt)

Provides:
- GPU requirement verification for inference
- Tensor/model device verification
- Context manager for GPU-only inference blocks
- Strict mode to catch all CPU operations
"""

import functools
import logging
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, TypeVar, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def require_gpu_for_inference(operation_name: str = "Inference") -> None:
    """Verify CUDA is available for inference, raise if not.

    Args:
        operation_name: Name of operation for error message

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for {operation_name} inference. "
            f"No CUDA-capable GPU detected. "
            f"All inference operations must run on GPU for real-time performance."
        )


def get_inference_device(
    device_id: Optional[int] = None,
) -> torch.device:
    """Get the device for inference operations.

    Args:
        device_id: Optional specific CUDA device ID

    Returns:
        torch.device for inference

    Raises:
        RuntimeError: If CUDA unavailable
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for inference. "
            "No CUDA-capable GPU detected. "
            "CPU inference is not supported due to latency requirements."
        )

    if device_id is not None:
        return torch.device(f"cuda:{device_id}")
    return torch.device("cuda")


def enforce_inference_gpu(func: F) -> F:
    """Decorator to enforce GPU availability before inference function execution.

    Usage:
        @enforce_inference_gpu
        def convert_voice(audio, speaker):
            ...

    Raises:
        RuntimeError: If CUDA is not available when function is called
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        require_gpu_for_inference(func.__name__)
        return func(*args, **kwargs)

    return wrapper  # type: ignore


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
            f"Use tensor.to('cuda') to move to GPU before inference."
        )


def verify_all_tensors_on_gpu(tensors: Dict[str, torch.Tensor]) -> None:
    """Verify all tensors in a dict are on GPU.

    Args:
        tensors: Dict of name -> tensor mappings

    Raises:
        RuntimeError: If any tensor is not on CUDA device
    """
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            verify_tensor_on_gpu(tensor, name)


def verify_model_on_gpu(model: nn.Module, name: str = "model") -> None:
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
                f"Use model.to('cuda') to move to GPU before inference."
            )
    except StopIteration:
        # Model has no parameters, can't verify
        pass


class GPUInferenceContext:
    """Context manager that ensures GPU availability for inference blocks.

    Usage:
        with GPUInferenceContext("voice conversion") as ctx:
            model.to(ctx.device)
            output = model(input)

    Raises:
        RuntimeError: On context entry if CUDA is not available
    """

    def __init__(
        self,
        operation_name: str = "inference",
        device_id: Optional[int] = None,
    ):
        """Initialize GPU inference context.

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

    def __enter__(self) -> "GPUInferenceContext":
        """Enter context, verifying CUDA availability."""
        require_gpu_for_inference(self.operation_name)
        self._device = get_inference_device(device_id=self.device_id)
        logger.debug(f"Entered GPU inference context on {self._device}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context."""
        self._device = None
        return False  # Don't suppress exceptions


def check_pipeline_inputs(inputs: Dict[str, Any]) -> None:
    """Check that all tensor inputs are on GPU.

    Args:
        inputs: Dict of input names to values (may include non-tensors)

    Raises:
        RuntimeError: If any tensor input is on CPU
    """
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            verify_tensor_on_gpu(value, name)


def ensure_cuda_output(output: torch.Tensor, name: str = "output") -> None:
    """Ensure model output is on CUDA.

    Args:
        output: Model output tensor
        name: Name for error message

    Raises:
        RuntimeError: If output is on CPU
    """
    if not output.is_cuda:
        raise RuntimeError(
            f"{name} output must be on GPU, but found on {output.device}. "
            f"This indicates a CPU fallback occurred during inference."
        )


class StrictGPUMode:
    """Context manager that tracks CPU tensor allocations.

    Useful for debugging to find unexpected CPU operations.

    Usage:
        with StrictGPUMode() as strict:
            # ... inference code ...
            if strict.violation_count > 0:
                logger.warning(f"CPU violations: {strict.get_violations()}")
    """

    def __init__(self):
        """Initialize strict GPU mode."""
        self._violations: List[str] = []
        self._original_tensor_new = None
        self._original_zeros = None
        self._original_ones = None
        self._original_randn = None
        self._original_rand = None
        self._active = False

    def _record_violation(self, op_name: str, *args, **kwargs):
        """Record a CPU allocation violation."""
        device = kwargs.get('device', None)
        if device is None and len(args) > 0:
            # Check if any positional arg specifies device
            pass  # Can't easily detect device from positional args

        # If device not specified or is CPU, record violation
        if device is None or (isinstance(device, str) and 'cpu' in device.lower()):
            if device is None:
                self._violations.append(f"{op_name}: allocated on CPU (no device specified)")

    def _make_wrapper(self, original_fn, op_name: str):
        """Create a wrapper that records violations."""
        def wrapper(*args, **kwargs):
            device = kwargs.get('device', None)
            # Check if device is specified
            if device is None:
                # Default is CPU, record violation
                self._violations.append(f"{op_name}: allocated on CPU (default)")
            elif isinstance(device, str) and 'cpu' in device.lower():
                self._violations.append(f"{op_name}: explicitly allocated on CPU")
            elif isinstance(device, torch.device) and device.type == 'cpu':
                self._violations.append(f"{op_name}: explicitly allocated on CPU")
            return original_fn(*args, **kwargs)
        return wrapper

    @property
    def violation_count(self) -> int:
        """Get count of CPU allocation violations."""
        return len(self._violations)

    def get_violations(self) -> List[str]:
        """Get list of violation descriptions."""
        return list(self._violations)

    def __enter__(self) -> "StrictGPUMode":
        """Enter strict GPU mode, hooking tensor allocation functions."""
        require_gpu_for_inference("strict GPU mode")
        self._active = True

        # Store originals and install wrappers
        self._original_randn = torch.randn
        self._original_zeros = torch.zeros
        self._original_ones = torch.ones
        self._original_rand = torch.rand

        torch.randn = self._make_wrapper(self._original_randn, "torch.randn")
        torch.zeros = self._make_wrapper(self._original_zeros, "torch.zeros")
        torch.ones = self._make_wrapper(self._original_ones, "torch.ones")
        torch.rand = self._make_wrapper(self._original_rand, "torch.rand")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit strict mode, restoring original functions."""
        self._active = False

        # Restore originals
        if self._original_randn is not None:
            torch.randn = self._original_randn
        if self._original_zeros is not None:
            torch.zeros = self._original_zeros
        if self._original_ones is not None:
            torch.ones = self._original_ones
        if self._original_rand is not None:
            torch.rand = self._original_rand

        return False  # Don't suppress exceptions
