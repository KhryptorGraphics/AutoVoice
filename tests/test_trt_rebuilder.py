"""Unit coverage for inference.trt_rebuilder."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from auto_voice.inference.trt_rebuilder import TRTEngineManager


class TinyModel(nn.Module):
    """Small deterministic model for engine-manager tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


@pytest.fixture
def model():
    return TinyModel()


@pytest.fixture
def manager(tmp_path):
    return TRTEngineManager(cache_dir=str(tmp_path / "engines"), precision="fp16")


def test_compute_checksum_is_deterministic_and_changes_on_weight_update(manager, model):
    """Checksums should be stable for identical weights and change after tuning."""
    original = manager.compute_model_checksum(model)
    repeated = manager.compute_model_checksum(model)

    with torch.no_grad():
        model.linear.weight.add_(1.0)

    updated = manager.compute_model_checksum(model)

    assert original == repeated
    assert updated != original


def test_get_engine_path_and_registration_round_trip(manager, model):
    """Engine filenames and registration metadata should track the model checksum."""
    engine_path = manager.get_engine_path("encoder", model)
    manager.register_model("encoder", model)

    assert engine_path.name.startswith("encoder_")
    assert engine_path.name.endswith("_fp16.engine")
    assert manager._registered_models["encoder"]["engine_path"] is None


def test_needs_rebuild_handles_unregistered_changed_and_missing_engine(manager, model, tmp_path):
    """Rebuild checks should trigger for first use, weight changes, and deleted engines."""
    assert manager.needs_rebuild("encoder", model) is True

    manager.register_model("encoder", model)
    engine_path = manager.get_engine_path("encoder", model)
    engine_path.touch()
    manager._mark_engine_built("encoder", model)

    assert manager.needs_rebuild("encoder", model) is False

    engine_path.unlink()
    assert manager.needs_rebuild("encoder", model) is True

    engine_path.touch()
    manager._mark_engine_built("encoder", model)
    with torch.no_grad():
        model.linear.weight.mul_(2.0)
    assert manager.needs_rebuild("encoder", model) is True


def test_metadata_helpers_save_load_and_missing(manager, model):
    """Metadata helpers should persist JSON and return None when absent."""
    payload = {"model_name": "encoder", "checksum": manager.compute_model_checksum(model)}
    manager._store_engine_metadata("encoder", payload)

    assert manager._get_engine_metadata("missing") is None
    assert manager._get_engine_metadata("encoder") == payload


def test_cleanup_old_engines_keeps_current_and_warns_on_failure(manager, caplog):
    """Cleanup should remove stale engines, keep current ones, and warn on unlink failures."""
    model_name = "encoder"
    first = manager.cache_dir / f"{model_name}_00000001_fp16.engine"
    second = manager.cache_dir / f"{model_name}_00000002_fp16.engine"
    third = manager.cache_dir / f"{model_name}_00000003_fp16.engine"
    for path in (first, second, third):
        path.write_text("engine")
    first.touch()
    second.touch()
    third.touch()
    os_times = {
        first: (1, 1),
        second: (3, 3),
        third: (2, 2),
    }
    for path, times in os_times.items():
        os.utime(path, times)
    manager._current_engines[model_name] = str(second)

    original_unlink = Path.unlink

    def flaky_unlink(path_obj):
        if path_obj == first:
            raise OSError("cannot remove")
        return original_unlink(path_obj)

    with patch("pathlib.Path.unlink", autospec=True, side_effect=flaky_unlink):
        removed = manager.cleanup_old_engines(keep_count=1)

    assert removed == [str(third)]
    assert first.exists() is True
    assert second.exists() is True
    assert not third.exists()
    assert "Failed to remove" in caplog.text

    removed = manager.cleanup_old_engines(keep_count=2)
    assert removed == []


def test_save_and_load_state_success(manager, model):
    """Manager state should round-trip through disk persistence."""
    manager.register_model("encoder", model)
    engine_path = manager.get_engine_path("encoder", model)
    engine_path.touch()
    manager._mark_engine_built("encoder", model)
    manager.save_state()

    restored = TRTEngineManager(cache_dir=str(manager.cache_dir), precision="fp32")
    assert restored.load_state() is True
    assert restored._registered_models == manager._registered_models
    assert restored._current_engines == manager._current_engines


def test_load_state_handles_missing_and_corrupt_files(manager, caplog):
    """Missing or invalid state files should fail gracefully."""
    assert manager.load_state() is False

    manager._state_file.write_text("{not-json")
    assert manager.load_state() is False
    assert "Failed to load state" in caplog.text


def test_rebuild_engine_success_updates_state_and_metadata(manager, model):
    """Successful rebuild should export, build, persist metadata, and remove temp ONNX."""
    built = {}

    def export_fn(export_model, onnx_path):
        assert export_model is model
        Path(onnx_path).write_text("onnx")

    class FakeBuilder:
        def __init__(self, precision):
            built["precision"] = precision

        def build_engine(self, onnx_path, engine_path, dynamic_shapes):
            built["onnx_path"] = onnx_path
            built["engine_path"] = engine_path
            built["dynamic_shapes"] = dynamic_shapes
            Path(engine_path).write_text("engine")

    with patch("auto_voice.inference.trt_pipeline.TRTEngineBuilder", FakeBuilder):
        engine_path = manager.rebuild_engine(
            "encoder",
            model,
            export_fn,
            dynamic_shapes={"audio": {"min": [1, 64]}},
        )

    metadata = manager._get_engine_metadata("encoder")
    assert Path(engine_path).exists() is True
    assert built["precision"] == "fp16"
    assert built["dynamic_shapes"] == {"audio": {"min": [1, 64]}}
    assert not Path(built["onnx_path"]).exists()
    assert metadata["engine_path"] == engine_path
    assert json.loads(manager._state_file.read_text())["current_engines"]["encoder"] == engine_path


def test_rebuild_engine_cleans_temp_onnx_on_builder_failure(manager, model):
    """Temporary ONNX exports should be removed even when engine building fails."""

    def export_fn(_model, onnx_path):
        Path(onnx_path).write_text("onnx")

    class FailingBuilder:
        def __init__(self, precision):
            self.precision = precision

        def build_engine(self, onnx_path, engine_path, dynamic_shapes):
            raise RuntimeError("build failed")

    with patch("auto_voice.inference.trt_pipeline.TRTEngineBuilder", FailingBuilder):
        with pytest.raises(RuntimeError, match="build failed"):
            manager.rebuild_engine("encoder", model, export_fn)

    assert not (manager.cache_dir / "encoder_temp.onnx").exists()
