"""TDD tests for GPU enforcement in training operations (Task 4.9).

Ensures all training operations require GPU and raise RuntimeError on CPU fallback.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# === Fixtures ===


@pytest.fixture
def temp_storage():
    """Temporary directory for storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# === Test Classes ===


class TestGPUEnforcementUtility:
    """Tests for the GPU enforcement utility."""

    def test_require_cuda_raises_when_unavailable(self):
        """require_cuda should raise RuntimeError when CUDA unavailable."""
        from auto_voice.training.gpu_enforcement import require_cuda

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                require_cuda("test operation")

    def test_require_cuda_passes_when_available(self):
        """require_cuda should not raise when CUDA is available."""
        from auto_voice.training.gpu_enforcement import require_cuda

        with patch("torch.cuda.is_available", return_value=True):
            # Should not raise
            require_cuda("test operation")

    def test_get_training_device_returns_cuda_when_available(self):
        """get_training_device should return CUDA device when available."""
        from auto_voice.training.gpu_enforcement import get_training_device

        with patch("torch.cuda.is_available", return_value=True):
            device = get_training_device()
            assert device.type == "cuda"

    def test_get_training_device_raises_when_unavailable(self):
        """get_training_device should raise when CUDA unavailable."""
        from auto_voice.training.gpu_enforcement import get_training_device

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                get_training_device()

    def test_get_training_device_allows_cpu_fallback_when_specified(self):
        """get_training_device can allow CPU when explicitly requested."""
        from auto_voice.training.gpu_enforcement import get_training_device

        with patch("torch.cuda.is_available", return_value=False):
            device = get_training_device(allow_cpu=True)
            assert device.type == "cpu"


class TestTrainingJobManagerGPUEnforcement:
    """Tests for GPU enforcement in TrainingJobManager."""

    def test_job_manager_requires_gpu_by_default(self, temp_storage):
        """TrainingJobManager should require GPU by default."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                TrainingJobManager(storage_path=temp_storage)

    def test_job_manager_can_skip_gpu_check_for_testing(self, temp_storage):
        """TrainingJobManager should allow skipping GPU check for tests."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=False):
            # Should not raise when require_gpu=False
            manager = TrainingJobManager(storage_path=temp_storage, require_gpu=False)
            assert manager is not None

    def test_execute_job_requires_cuda(self, temp_storage):
        """execute_job should verify CUDA before execution."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        job = manager.create_job(profile_id="test", sample_ids=["s1"])

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                manager.execute_job(job.job_id)


class TestFineTuningPipelineGPUEnforcement:
    """Tests for GPU enforcement in FineTuningPipeline."""

    def test_pipeline_can_require_gpu(self, temp_storage):
        """FineTuningPipeline should raise when GPU required but unavailable."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(1))])

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                FineTuningPipeline(
                    base_model=mock_model,
                    output_dir=temp_storage,
                    require_gpu=True,
                )

    def test_pipeline_allows_cpu_when_not_required(self, temp_storage):
        """FineTuningPipeline should work on CPU when GPU not required."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(1))])
        mock_model.to = MagicMock(return_value=mock_model)

        with patch("torch.cuda.is_available", return_value=False):
            # Should not raise when require_gpu=False (default)
            pipeline = FineTuningPipeline(
                base_model=mock_model,
                output_dir=temp_storage,
                require_gpu=False,
            )
            assert pipeline.device.type == "cpu"


class TestGPUEnforcementDecorator:
    """Tests for GPU enforcement decorator."""

    def test_enforce_gpu_decorator_raises_on_cpu(self):
        """@enforce_gpu decorator should raise when CUDA unavailable."""
        from auto_voice.training.gpu_enforcement import enforce_gpu

        @enforce_gpu
        def training_function():
            return "trained"

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                training_function()

    def test_enforce_gpu_decorator_passes_on_gpu(self):
        """@enforce_gpu decorator should allow execution when CUDA available."""
        from auto_voice.training.gpu_enforcement import enforce_gpu

        @enforce_gpu
        def training_function():
            return "trained"

        with patch("torch.cuda.is_available", return_value=True):
            result = training_function()
            assert result == "trained"

    def test_enforce_gpu_decorator_preserves_function_args(self):
        """@enforce_gpu decorator should pass through arguments."""
        from auto_voice.training.gpu_enforcement import enforce_gpu

        @enforce_gpu
        def training_function(a, b, c=None):
            return (a, b, c)

        with patch("torch.cuda.is_available", return_value=True):
            result = training_function(1, 2, c=3)
            assert result == (1, 2, 3)


class TestGPUContextManager:
    """Tests for GPU context manager."""

    def test_gpu_training_context_raises_on_cpu(self):
        """GPUTrainingContext should raise on entry when CUDA unavailable."""
        from auto_voice.training.gpu_enforcement import GPUTrainingContext

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                with GPUTrainingContext("test training"):
                    pass

    def test_gpu_training_context_passes_on_gpu(self):
        """GPUTrainingContext should allow execution when CUDA available."""
        from auto_voice.training.gpu_enforcement import GPUTrainingContext

        executed = False
        with patch("torch.cuda.is_available", return_value=True):
            with GPUTrainingContext("test training") as ctx:
                executed = True
                assert ctx.device.type == "cuda"
        assert executed

    def test_gpu_training_context_provides_device(self):
        """GPUTrainingContext should provide the CUDA device."""
        from auto_voice.training.gpu_enforcement import GPUTrainingContext

        with patch("torch.cuda.is_available", return_value=True):
            with GPUTrainingContext("test") as ctx:
                assert ctx.device is not None
                assert ctx.device.type == "cuda"
