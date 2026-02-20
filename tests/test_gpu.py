"""Tests for GPU module."""
import pytest
import numpy as np


class TestCUDAKernels:
    """CUDA kernel interface tests."""

    @pytest.mark.smoke
    def test_import(self):
        from auto_voice.gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE, get_kernel_metrics
        assert isinstance(CUDA_KERNELS_AVAILABLE, bool)

    def test_get_kernel_metrics_returns_list(self):
        from auto_voice.gpu.cuda_kernels import get_kernel_metrics
        metrics = get_kernel_metrics()
        assert isinstance(metrics, list)

    def test_reset_kernel_metrics(self):
        from auto_voice.gpu.cuda_kernels import reset_kernel_metrics, get_kernel_metrics
        reset_kernel_metrics()
        assert get_kernel_metrics() == []

    @pytest.mark.cuda
    def test_pitch_detect_gpu_fallback(self):
        import torch
        from auto_voice.gpu.cuda_kernels import pitch_detect_gpu

        audio = torch.randn(22050, device='cuda')
        f0 = pitch_detect_gpu(audio, sample_rate=22050, hop_length=512)
        assert isinstance(f0, torch.Tensor)
        assert f0.shape[0] == 22050 // 512

    @pytest.mark.cuda
    def test_pitch_detect_cpu(self):
        import torch
        from auto_voice.gpu.cuda_kernels import pitch_detect_gpu

        audio = torch.randn(22050)
        f0 = pitch_detect_gpu(audio, sample_rate=22050, hop_length=512)
        assert isinstance(f0, torch.Tensor)


class TestGPUMemoryManager:
    """GPU memory manager tests."""

    @pytest.mark.smoke
    def test_import(self):
        from auto_voice.gpu.memory_manager import GPUMemoryManager
        mgr = GPUMemoryManager()
        assert mgr.device == 'cuda:0'

    @pytest.mark.cuda
    def test_get_memory_info(self):
        from auto_voice.gpu.memory_manager import GPUMemoryManager
        mgr = GPUMemoryManager()
        info = mgr.get_memory_info()
        assert info['available'] is True
        assert info['total_gb'] > 0
        assert info['allocated_gb'] >= 0

    @pytest.mark.cuda
    def test_can_allocate(self):
        from auto_voice.gpu.memory_manager import GPUMemoryManager
        mgr = GPUMemoryManager()
        # Should be able to allocate 1MB
        assert mgr.can_allocate(1024 * 1024) is True

    @pytest.mark.cuda
    def test_clear_cache(self):
        import torch
        from auto_voice.gpu.memory_manager import GPUMemoryManager
        mgr = GPUMemoryManager()
        # Allocate some GPU memory
        x = torch.randn(1000, 1000, device='cuda')
        del x
        mgr.clear_cache()  # Should not raise
