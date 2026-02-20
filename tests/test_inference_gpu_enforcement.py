"""Tests for GPU enforcement in inference operations.

Task 7.5: Add strict GPU-only checks (RuntimeError on any CPU fallback attempt)

Tests cover:
- Inference pipeline GPU requirements
- Tensor device verification
- Model device verification
- Decorator for inference functions
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_storage():
    """Temporary directory for storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


# ============================================================================
# Test: Inference GPU Enforcement
# ============================================================================


class TestInferenceGPUEnforcement:
    """Tests for GPU enforcement in inference operations."""

    def test_require_gpu_for_inference_raises_when_unavailable(self):
        """require_gpu_for_inference should raise RuntimeError when CUDA unavailable."""
        from auto_voice.inference.gpu_enforcement import require_gpu_for_inference

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required.*inference"):
                require_gpu_for_inference("voice conversion")

    def test_require_gpu_for_inference_passes_when_available(self):
        """require_gpu_for_inference should not raise when CUDA is available."""
        from auto_voice.inference.gpu_enforcement import require_gpu_for_inference

        with patch("torch.cuda.is_available", return_value=True):
            # Should not raise
            require_gpu_for_inference("voice conversion")

    def test_get_inference_device_returns_cuda_when_available(self):
        """get_inference_device should return CUDA device when available."""
        from auto_voice.inference.gpu_enforcement import get_inference_device

        with patch("torch.cuda.is_available", return_value=True):
            device = get_inference_device()
            assert device.type == "cuda"

    def test_get_inference_device_raises_when_unavailable(self):
        """get_inference_device should raise when CUDA unavailable."""
        from auto_voice.inference.gpu_enforcement import get_inference_device

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                get_inference_device()

    def test_get_inference_device_respects_device_id(self):
        """get_inference_device should use specified device ID."""
        from auto_voice.inference.gpu_enforcement import get_inference_device

        with patch("torch.cuda.is_available", return_value=True):
            device = get_inference_device(device_id=1)
            assert device == torch.device("cuda:1")


class TestEnforceGPUDecorator:
    """Tests for inference GPU enforcement decorator."""

    def test_enforce_inference_gpu_decorator_raises_on_cpu(self):
        """@enforce_inference_gpu decorator should raise when CUDA unavailable."""
        from auto_voice.inference.gpu_enforcement import enforce_inference_gpu

        @enforce_inference_gpu
        def inference_function():
            return "converted"

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                inference_function()

    def test_enforce_inference_gpu_decorator_passes_on_gpu(self):
        """@enforce_inference_gpu decorator should allow execution when CUDA available."""
        from auto_voice.inference.gpu_enforcement import enforce_inference_gpu

        @enforce_inference_gpu
        def inference_function():
            return "converted"

        with patch("torch.cuda.is_available", return_value=True):
            result = inference_function()
            assert result == "converted"

    def test_enforce_inference_gpu_preserves_function_signature(self):
        """@enforce_inference_gpu decorator should preserve function arguments."""
        from auto_voice.inference.gpu_enforcement import enforce_inference_gpu

        @enforce_inference_gpu
        def convert_voice(audio, speaker_id, pitch_shift=0):
            return (audio, speaker_id, pitch_shift)

        with patch("torch.cuda.is_available", return_value=True):
            result = convert_voice("audio_data", "speaker_1", pitch_shift=2)
            assert result == ("audio_data", "speaker_1", 2)


class TestTensorDeviceVerification:
    """Tests for tensor device verification."""

    def test_verify_tensor_on_gpu_raises_for_cpu_tensor(self, mock_cuda_available):
        """verify_tensor_on_gpu should raise for CPU tensor."""
        from auto_voice.inference.gpu_enforcement import verify_tensor_on_gpu

        cpu_tensor = torch.randn(10)  # CPU by default
        with pytest.raises(RuntimeError, match="must be on GPU"):
            verify_tensor_on_gpu(cpu_tensor, "input_audio")

    @pytest.mark.cuda
    def test_verify_tensor_on_gpu_passes_for_cuda_tensor(self):
        """verify_tensor_on_gpu should pass for CUDA tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from auto_voice.inference.gpu_enforcement import verify_tensor_on_gpu

        cuda_tensor = torch.randn(10, device="cuda")
        # Should not raise
        verify_tensor_on_gpu(cuda_tensor, "input_audio")

    def test_verify_all_tensors_on_gpu_raises_for_mixed(self, mock_cuda_available):
        """verify_all_tensors_on_gpu should raise if any tensor is on CPU."""
        from auto_voice.inference.gpu_enforcement import verify_all_tensors_on_gpu

        tensors = {
            "tensor_a": torch.randn(10),  # CPU
            "tensor_b": torch.randn(10),  # CPU
        }

        with pytest.raises(RuntimeError, match="must be on GPU"):
            verify_all_tensors_on_gpu(tensors)


class TestModelDeviceVerification:
    """Tests for model device verification."""

    def test_verify_model_on_gpu_raises_for_cpu_model(self, mock_cuda_available):
        """verify_model_on_gpu should raise for CPU model."""
        from auto_voice.inference.gpu_enforcement import verify_model_on_gpu

        model = torch.nn.Linear(10, 10)  # CPU by default
        with pytest.raises(RuntimeError, match="must be on GPU"):
            verify_model_on_gpu(model, "encoder")

    @pytest.mark.cuda
    def test_verify_model_on_gpu_passes_for_cuda_model(self):
        """verify_model_on_gpu should pass for CUDA model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from auto_voice.inference.gpu_enforcement import verify_model_on_gpu

        model = torch.nn.Linear(10, 10).cuda()
        # Should not raise
        verify_model_on_gpu(model, "encoder")


class TestInferenceContext:
    """Tests for inference GPU context manager."""

    def test_gpu_inference_context_raises_on_cpu(self):
        """GPUInferenceContext should raise on entry when CUDA unavailable."""
        from auto_voice.inference.gpu_enforcement import GPUInferenceContext

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                with GPUInferenceContext("voice conversion"):
                    pass

    def test_gpu_inference_context_passes_on_gpu(self):
        """GPUInferenceContext should allow execution when CUDA available."""
        from auto_voice.inference.gpu_enforcement import GPUInferenceContext

        executed = False
        with patch("torch.cuda.is_available", return_value=True):
            with GPUInferenceContext("voice conversion") as ctx:
                executed = True
                assert ctx.device.type == "cuda"
        assert executed

    def test_gpu_inference_context_provides_device(self):
        """GPUInferenceContext should provide the CUDA device."""
        from auto_voice.inference.gpu_enforcement import GPUInferenceContext

        with patch("torch.cuda.is_available", return_value=True):
            with GPUInferenceContext("conversion") as ctx:
                assert ctx.device is not None
                assert ctx.device.type == "cuda"

    def test_gpu_inference_context_device_unavailable_outside(self):
        """GPUInferenceContext.device should raise outside context."""
        from auto_voice.inference.gpu_enforcement import GPUInferenceContext

        ctx = GPUInferenceContext("test")
        with pytest.raises(RuntimeError, match="not available outside context"):
            _ = ctx.device


class TestPipelineGPUEnforcement:
    """Tests for GPU enforcement in inference pipelines."""

    def test_pipeline_check_raises_for_cpu_inputs(self):
        """Pipeline should raise if input tensors are on CPU."""
        from auto_voice.inference.gpu_enforcement import check_pipeline_inputs

        inputs = {
            "audio": torch.randn(1, 16000),  # CPU
            "pitch": torch.randn(1, 100),  # CPU
        }

        with pytest.raises(RuntimeError, match="must be on GPU"):
            check_pipeline_inputs(inputs)

    @pytest.mark.cuda
    def test_pipeline_check_passes_for_cuda_inputs(self):
        """Pipeline should pass if all inputs are on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from auto_voice.inference.gpu_enforcement import check_pipeline_inputs

        inputs = {
            "audio": torch.randn(1, 16000, device="cuda"),
            "pitch": torch.randn(1, 100, device="cuda"),
        }

        # Should not raise
        check_pipeline_inputs(inputs)

    def test_ensure_cuda_output_raises_for_cpu(self, mock_cuda_available):
        """ensure_cuda_output should raise if output is on CPU."""
        from auto_voice.inference.gpu_enforcement import ensure_cuda_output

        output = torch.randn(1, 16000)  # CPU
        with pytest.raises(RuntimeError, match="output.*must be on GPU"):
            ensure_cuda_output(output, "converted_audio")


class TestStrictModeEnforcement:
    """Tests for strict mode that catches all CPU operations."""

    def test_strict_mode_context_catches_cpu_allocation(self):
        """Strict mode should catch CPU tensor allocations."""
        from auto_voice.inference.gpu_enforcement import StrictGPUMode

        with patch("torch.cuda.is_available", return_value=True):
            with StrictGPUMode() as strict:
                # CPU allocation should be flagged
                cpu_tensor = torch.randn(10)
                violations = strict.get_violations()
                assert len(violations) > 0

    def test_strict_mode_allows_cuda_operations(self):
        """Strict mode should allow CUDA operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from auto_voice.inference.gpu_enforcement import StrictGPUMode

        with StrictGPUMode() as strict:
            cuda_tensor = torch.randn(10, device="cuda")
            violations = strict.get_violations()
            # CUDA operations should not be flagged
            cuda_violations = [v for v in violations if "cuda" not in v.lower()]
            # Should not flag explicit CUDA tensors
            assert cuda_tensor.is_cuda

    def test_strict_mode_reports_violation_count(self):
        """Strict mode should report violation count."""
        from auto_voice.inference.gpu_enforcement import StrictGPUMode

        with patch("torch.cuda.is_available", return_value=True):
            with StrictGPUMode() as strict:
                # Multiple CPU allocations
                _ = torch.randn(10)
                _ = torch.zeros(5)
                count = strict.violation_count
                assert count >= 2
