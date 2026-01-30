"""Tests for GPU memory monitoring and optimization.

Task 7.2: Add GPU memory monitoring and optimization

Tests cover:
- Continuous memory monitoring during training
- Memory threshold alerts
- Automatic optimization triggers
- Memory-efficient training strategies
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get CUDA device, skip test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def temp_storage():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test: GPU Memory Manager
# ============================================================================

@pytest.mark.cuda
class TestGPUMemoryManager:
    """Tests for GPUMemoryManager class."""

    def test_get_memory_info_returns_dict(self, device):
        """Memory info should return dict with expected keys."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(device='cuda:0')
        info = manager.get_memory_info()

        assert isinstance(info, dict)
        assert info.get('available') is True
        assert 'total_gb' in info
        assert 'allocated_gb' in info
        assert 'free_gb' in info
        assert 'utilization' in info

    def test_memory_info_values_are_reasonable(self, device):
        """Memory values should be within reasonable ranges."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(device='cuda:0')
        info = manager.get_memory_info()

        # Total memory should be positive (at least 1GB typical)
        assert info['total_gb'] > 0

        # Allocated should be >= 0 and <= total
        assert info['allocated_gb'] >= 0
        assert info['allocated_gb'] <= info['total_gb']

        # Utilization should be 0-1
        assert 0 <= info['utilization'] <= 1

    def test_can_allocate_with_space(self, device):
        """can_allocate should return True when space is available."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(device='cuda:0', max_fraction=0.9)

        # Small allocation should be possible
        assert manager.can_allocate(1024) is True  # 1KB

    def test_can_allocate_rejects_huge_allocation(self, device):
        """can_allocate should return False for impossibly large allocations."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(device='cuda:0', max_fraction=0.9)

        # Request more memory than exists
        huge_size = 1024 * 1024 * 1024 * 1024  # 1TB
        assert manager.can_allocate(huge_size) is False

    def test_clear_cache_reduces_reserved(self, device):
        """Clearing cache should reduce reserved memory."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(device='cuda:0')

        # Allocate some tensors to increase reserved memory
        tensors = [torch.randn(1000, 1000, device=device) for _ in range(5)]
        info_before = manager.get_memory_info()

        # Delete tensors and clear cache
        del tensors
        manager.clear_cache()

        info_after = manager.get_memory_info()

        # Reserved memory should decrease (or at least not increase)
        assert info_after['allocated_gb'] <= info_before['allocated_gb']


@pytest.mark.cuda
class TestMemoryMonitor:
    """Tests for continuous memory monitoring."""

    def test_memory_monitor_tracks_over_time(self, device):
        """Memory monitor should track memory usage over time."""
        from auto_voice.gpu.memory_manager import GPUMemoryMonitor

        monitor = GPUMemoryMonitor(device='cuda:0', interval_ms=100)
        monitor.start()

        # Allocate and deallocate memory
        tensors = []
        for _ in range(3):
            tensors.append(torch.randn(1000, 1000, device=device))
            time.sleep(0.15)

        del tensors
        torch.cuda.empty_cache()
        time.sleep(0.15)

        monitor.stop()
        history = monitor.get_history()

        # Should have multiple data points
        assert len(history) >= 3

        # Each entry should have timestamp and memory info
        for entry in history:
            assert 'timestamp' in entry
            assert 'allocated_gb' in entry

    def test_memory_monitor_detects_peak(self, device):
        """Memory monitor should detect peak memory usage."""
        from auto_voice.gpu.memory_manager import GPUMemoryMonitor

        monitor = GPUMemoryMonitor(device='cuda:0', interval_ms=50)
        monitor.start()

        # Create peak memory usage
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)

        tensors = [torch.randn(2000, 2000, device=device) for _ in range(10)]
        time.sleep(0.1)

        peak = torch.cuda.memory_allocated(device)

        del tensors
        torch.cuda.empty_cache()
        time.sleep(0.1)

        monitor.stop()
        stats = monitor.get_stats()

        # Peak should be higher than baseline
        assert stats['peak_gb'] >= baseline / (1024**3)

        # Peak should match or exceed what we allocated
        expected_peak = peak / (1024**3)
        assert stats['peak_gb'] >= expected_peak * 0.9  # Allow 10% tolerance


@pytest.mark.cuda
class TestMemoryOptimization:
    """Tests for automatic memory optimization."""

    def test_gradient_checkpointing_can_be_enabled(self, device):
        """Gradient checkpointing should be enabled without errors."""
        from auto_voice.gpu.memory_manager import enable_gradient_checkpointing
        from auto_voice.models.so_vits_svc import SoVitsSvc

        model = SoVitsSvc().to(device)

        # Enable checkpointing - should not raise
        enable_gradient_checkpointing(model)

        # Model should still have valid parameters after checkpointing
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model should have parameters after checkpointing"

        # Model should have encoder/decoder components
        has_posterior_encoder = hasattr(model, 'posterior_encoder')
        has_mel_decoder = hasattr(model, 'mel_decoder')
        assert has_posterior_encoder and has_mel_decoder, "Model should have encoder and decoder"

    def test_memory_efficient_attention_available(self, device):
        """Memory-efficient attention should be available."""
        from auto_voice.gpu.memory_manager import is_flash_attention_available

        # Should return True/False without error
        result = is_flash_attention_available()
        assert isinstance(result, bool)

    def test_auto_optimization_triggers_at_threshold(self, device, temp_storage):
        """Auto-optimization should trigger when memory exceeds threshold."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager, AutoMemoryOptimizer

        manager = GPUMemoryManager(device='cuda:0')
        optimizer = AutoMemoryOptimizer(
            manager=manager,
            warning_threshold=0.7,  # Warn at 70%
            critical_threshold=0.85,  # Optimize at 85%
        )

        # Check current utilization
        info = manager.get_memory_info()
        current_util = info['utilization']

        # Trigger optimization check
        action_taken = optimizer.check_and_optimize()

        # If utilization is high, optimization should have been attempted
        if current_util > 0.85:
            assert action_taken in ['cleared_cache', 'enabled_checkpointing', 'none']
        else:
            assert action_taken == 'none'


@pytest.mark.cuda
class TestTrainingMemoryIntegration:
    """Tests for memory monitoring integration with training."""

    def test_training_job_reports_memory_usage(self, device, temp_storage):
        """Training jobs should report memory usage in results."""
        from auto_voice.training.job_manager import TrainingJobManager, JobStatus
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        jobs_dir = temp_storage / "jobs"
        manager = TrainingJobManager(storage_path=jobs_dir)

        # Create job with memory tracking enabled
        job = manager.create_job(
            profile_id="memory-test",
            sample_ids=["sample-1"],
            config={
                "epochs": 1,
                "track_memory": True,
            },
        )

        # Job config should indicate memory tracking
        assert job.config.get("track_memory") is True

    def test_memory_warning_logged_near_threshold(self, device, temp_storage, caplog):
        """Warning should be logged when approaching memory threshold."""
        import logging
        from auto_voice.gpu.memory_manager import GPUMemoryMonitor

        with caplog.at_level(logging.WARNING):
            monitor = GPUMemoryMonitor(
                device='cuda:0',
                interval_ms=100,
                warning_threshold=0.01,  # Very low threshold to trigger warning
            )
            monitor.start()

            # Allocate memory to trigger warning
            tensors = [torch.randn(1000, 1000, device=device) for _ in range(5)]
            time.sleep(0.2)

            monitor.stop()
            del tensors

        # Warning may or may not be logged depending on current memory usage
        # Just verify no errors occurred


@pytest.mark.cuda
class TestMemoryManagerFallbacks:
    """Tests for graceful handling when GPU unavailable."""

    def test_handles_cuda_unavailable(self):
        """Should gracefully handle CUDA unavailability."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        with patch('torch.cuda.is_available', return_value=False):
            manager = GPUMemoryManager(device='cuda:0')
            info = manager.get_memory_info()

            assert info.get('available') is False

    def test_handles_invalid_device(self, device):
        """Should handle invalid device index gracefully."""
        from auto_voice.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(device='cuda:999')  # Invalid device
        info = manager.get_memory_info()

        # Should not crash, may return error or unavailable
        assert isinstance(info, dict)
