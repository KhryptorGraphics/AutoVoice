"""Additional non-CUDA coverage for GPU fallback and monitoring paths."""

from __future__ import annotations

import importlib
import logging
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import auto_voice.gpu.cuda_kernels as cuda_kernels
import auto_voice.gpu.memory_manager as memory_manager


def test_cuda_kernels_reload_with_extension_available(monkeypatch):
    fake_cuda = SimpleNamespace(pitch_detect=lambda *args: "pitch", synthesis=lambda *args: "wave")

    monkeypatch.setitem(__import__("sys").modules, "auto_voice._cuda_kernels", fake_cuda)
    reloaded = importlib.reload(cuda_kernels)
    try:
        assert reloaded.CUDA_KERNELS_AVAILABLE is True
        assert reloaded._cuda_module is fake_cuda
    finally:
        monkeypatch.delitem(__import__("sys").modules, "auto_voice._cuda_kernels", raising=False)
        importlib.reload(cuda_kernels)


def test_pitch_detect_gpu_uses_custom_kernel_and_records_metric(monkeypatch):
    sentinel = object()
    fake_audio = SimpleNamespace(is_cuda=True, device="cuda:0")
    fake_cuda = SimpleNamespace(pitch_detect=lambda *args: sentinel)

    monkeypatch.setattr(cuda_kernels, "CUDA_KERNELS_AVAILABLE", True)
    monkeypatch.setattr(cuda_kernels, "_cuda_module", fake_cuda)
    cuda_kernels.reset_kernel_metrics()

    result = cuda_kernels.pitch_detect_gpu(fake_audio, sample_rate=22050)

    assert result is sentinel
    assert cuda_kernels.get_kernel_metrics() == [
        {"name": "pitch_detect", "calls": 1, "device": "cuda:0"}
    ]


def test_pitch_detect_gpu_falls_back_after_kernel_error(monkeypatch, caplog):
    audio = torch.randn(2048)
    fake_cuda = SimpleNamespace(pitch_detect=lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))

    monkeypatch.setattr(cuda_kernels, "CUDA_KERNELS_AVAILABLE", True)
    monkeypatch.setattr(cuda_kernels, "_cuda_module", fake_cuda)

    with caplog.at_level(logging.WARNING), patch_tensor_is_cuda(True):
        result = cuda_kernels.pitch_detect_gpu(audio, sample_rate=22050, hop_length=512)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 4
    assert "falling back" in caplog.text


def test_synthesis_gpu_uses_custom_kernel_and_records_metric(monkeypatch):
    sentinel = object()
    fake_features = SimpleNamespace(is_cuda=True, device="cuda:0")
    fake_cuda = SimpleNamespace(synthesis=lambda *args: sentinel)

    monkeypatch.setattr(cuda_kernels, "CUDA_KERNELS_AVAILABLE", True)
    monkeypatch.setattr(cuda_kernels, "_cuda_module", fake_cuda)
    cuda_kernels.reset_kernel_metrics()

    result = cuda_kernels.synthesis_gpu(fake_features, speaker_embedding=None, sample_rate=44100)

    assert result is sentinel
    assert cuda_kernels.get_kernel_metrics() == [
        {"name": "synthesis", "calls": 1, "device": "cuda:0"}
    ]


def test_synthesis_gpu_falls_back_to_istft(monkeypatch, caplog):
    features = torch.abs(torch.randn(1025, 4))
    fake_cuda = SimpleNamespace(synthesis=lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))

    monkeypatch.setattr(cuda_kernels, "CUDA_KERNELS_AVAILABLE", True)
    monkeypatch.setattr(cuda_kernels, "_cuda_module", fake_cuda)
    monkeypatch.setattr(torch, "istft", lambda *args, **kwargs: torch.ones(32))

    with caplog.at_level(logging.WARNING), patch_tensor_is_cuda(True):
        result = cuda_kernels.synthesis_gpu(features, speaker_embedding=None, sample_rate=44100)

    assert torch.equal(result, torch.ones(32))
    assert "falling back" in caplog.text


def test_synthesis_gpu_returns_features_for_non_spectrogram_input():
    features = torch.randn(1, 2, 3)

    result = cuda_kernels.synthesis_gpu(features, speaker_embedding=None, sample_rate=44100)

    assert result is features


def test_handle_oom_success(monkeypatch):
    empty_cache = MagicMock()
    synchronize = MagicMock()

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    memory_manager.handle_oom()

    empty_cache.assert_called_once()
    synchronize.assert_called_once()


def test_handle_oom_logs_cleanup_failure(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock(side_effect=RuntimeError("bad cache")))

    with caplog.at_level(logging.ERROR):
        memory_manager.handle_oom()

    assert "Failed to handle OOM" in caplog.text


def test_gpu_memory_tracker_records_tensor_module_and_unknown_object():
    tracker = memory_manager.GPUMemoryTracker()
    tensor = torch.ones(4)
    module = nn.Linear(2, 3)

    tracker.record_allocation("tensor", tensor)
    tracker.record_allocation("module", module)
    tracker.record_allocation("other", object())

    stats = tracker.get_stats()

    assert stats["total_allocations"] == 3
    assert stats["active_allocations"] == 3
    assert stats["allocations"]["tensor"]["size_bytes"] == tensor.element_size() * tensor.nelement()
    assert stats["allocations"]["module"]["size_bytes"] > 0
    assert stats["allocations"]["other"]["size_bytes"] == 0

    tracker.clear()
    assert tracker.get_stats()["active_allocations"] == 0


def test_gpu_memory_manager_handles_info_error(monkeypatch):
    manager = memory_manager.GPUMemoryManager()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", MagicMock(side_effect=RuntimeError("bad device")))

    info = manager.get_memory_info()

    assert info["available"] is False
    assert "bad device" in info["error"]


def test_gpu_memory_manager_can_allocate_false_when_unavailable(monkeypatch):
    manager = memory_manager.GPUMemoryManager()
    monkeypatch.setattr(manager, "get_memory_info", lambda: {"available": False})
    assert manager.can_allocate(1024) is False


def test_gpu_memory_manager_clear_cache_warns_on_exception(monkeypatch, caplog):
    manager = memory_manager.GPUMemoryManager()
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock(side_effect=RuntimeError("bad cache")))

    with caplog.at_level(logging.WARNING):
        manager.clear_cache()

    assert "Failed to clear GPU cache" in caplog.text


def test_memory_monitor_start_is_idempotent_and_empty_stats():
    monitor = memory_manager.GPUMemoryMonitor()
    monitor._running = True
    monitor._thread = None

    monitor.start()
    assert monitor._thread is None
    assert monitor.get_stats() == {
        "peak_gb": 0,
        "avg_gb": 0,
        "min_gb": 0,
        "peak_utilization": 0,
    }


def test_memory_monitor_take_snapshot_handles_unavailable(monkeypatch):
    monitor = memory_manager.GPUMemoryMonitor()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert monitor._take_snapshot() is None


def test_memory_monitor_take_snapshot_handles_exception(monkeypatch, caplog):
    monitor = memory_manager.GPUMemoryMonitor()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", MagicMock(side_effect=RuntimeError("snapshot failed")))

    with caplog.at_level(logging.ERROR):
        snapshot = monitor._take_snapshot()

    assert snapshot is None
    assert "Failed to take memory snapshot" in caplog.text


def test_memory_monitor_loop_trims_history_and_calls_warning_callback(monkeypatch):
    callback = MagicMock()
    monitor = memory_manager.GPUMemoryMonitor(max_history=1, warning_threshold=0.5, on_warning=callback)
    snapshot = memory_manager.MemorySnapshot(
        timestamp=1.0,
        allocated_gb=1.0,
        reserved_gb=1.2,
        total_gb=2.0,
        utilization=0.75,
    )
    calls = {"count": 0}

    def fake_take_snapshot():
        calls["count"] += 1
        if calls["count"] >= 2:
            monitor._running = False
        return snapshot

    monkeypatch.setattr(monitor, "_take_snapshot", fake_take_snapshot)
    monkeypatch.setattr(memory_manager.time, "sleep", lambda *_: None)

    monitor._running = True
    monitor._monitor_loop()

    assert len(monitor._history) == 1
    assert callback.call_count == 2


def test_memory_monitor_loop_logs_errors(monkeypatch, caplog):
    monitor = memory_manager.GPUMemoryMonitor()

    def fake_take_snapshot():
        monitor._running = False
        raise RuntimeError("monitor failed")

    monkeypatch.setattr(monitor, "_take_snapshot", fake_take_snapshot)
    monkeypatch.setattr(memory_manager.time, "sleep", lambda *_: None)

    with caplog.at_level(logging.ERROR):
        monitor._running = True
        monitor._monitor_loop()

    assert "Error in memory monitor" in caplog.text


def test_auto_memory_optimizer_branches(caplog):
    unavailable_manager = MagicMock()
    unavailable_manager.get_memory_info.return_value = {"available": False}
    assert memory_manager.AutoMemoryOptimizer(unavailable_manager).check_and_optimize() == "none"

    warning_manager = MagicMock()
    warning_manager.get_memory_info.return_value = {"available": True, "utilization": 0.75}
    with caplog.at_level(logging.INFO):
        assert memory_manager.AutoMemoryOptimizer(
            warning_manager, warning_threshold=0.7, critical_threshold=0.9
        ).check_and_optimize() == "none"
    assert "Memory utilization elevated" in caplog.text

    critical_manager = MagicMock()
    critical_manager.get_memory_info.side_effect = [
        {"available": True, "utilization": 0.95},
        {"available": True, "utilization": 0.95},
    ]
    optimizer = memory_manager.AutoMemoryOptimizer(
        critical_manager, warning_threshold=0.7, critical_threshold=0.9
    )
    assert optimizer.check_and_optimize() == "cleared_cache"
    critical_manager.clear_cache.assert_called_once()


def test_enable_gradient_checkpointing_prefers_model_method():
    class Checkpointable(nn.Module):
        def __init__(self):
            super().__init__()
            self.enabled = False

        def gradient_checkpointing_enable(self):
            self.enabled = True

    model = Checkpointable()
    memory_manager.enable_gradient_checkpointing(model)
    assert model.enabled is True


def test_enable_gradient_checkpointing_wraps_encoder_and_decoder():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Wrapper()
            self.decoder = Wrapper()

    model = Model()
    memory_manager.enable_gradient_checkpointing(model)

    assert model.encoder.seq._gradient_checkpointing is True
    assert model.decoder.seq._gradient_checkpointing is True


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 0), True), ((7, 5), True), ((7, 0), False)],
)
def test_is_flash_attention_available_capability_matrix(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)

    assert memory_manager.is_flash_attention_available() is expected


def test_is_flash_attention_available_handles_exception(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", MagicMock(side_effect=RuntimeError("cap fail")))

    assert memory_manager.is_flash_attention_available() is False


@pytest.mark.parametrize(
    ("cuda_available", "total_gb", "flash_available", "expected"),
    [
        (False, None, False, {"batch_size": 1, "gradient_accumulation_steps": 8, "use_checkpointing": True}),
        (True, 24, True, {"batch_size": 16, "gradient_accumulation_steps": 1, "use_checkpointing": False, "use_flash_attention": True}),
        (True, 12, False, {"batch_size": 8, "gradient_accumulation_steps": 2, "use_checkpointing": False, "use_flash_attention": False}),
        (True, 8, True, {"batch_size": 4, "gradient_accumulation_steps": 4, "use_checkpointing": True, "use_flash_attention": True}),
        (True, 4, False, {"batch_size": 2, "gradient_accumulation_steps": 8, "use_checkpointing": True, "use_flash_attention": False}),
    ],
)
def test_get_memory_efficient_config_by_tier(monkeypatch, cuda_available, total_gb, flash_available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(memory_manager, "is_flash_attention_available", lambda: flash_available)

    if cuda_available:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda *_: types.SimpleNamespace(total_memory=int(total_gb * (1024**3))),
        )

    assert memory_manager.get_memory_efficient_config() == expected


class patch_tensor_is_cuda:
    """Temporarily force tensors down CUDA-only branches without real CUDA."""

    def __init__(self, value: bool):
        self.value = value
        self._patcher = None

    def __enter__(self):
        from unittest.mock import patch

        self._patcher = patch.object(torch.Tensor, "is_cuda", new=property(lambda _self: self.value))
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._patcher is not None
        self._patcher.stop()
        return False
