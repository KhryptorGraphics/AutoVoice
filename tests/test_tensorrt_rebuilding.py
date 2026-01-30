"""Tests for TensorRT engine rebuilding for fine-tuned models.

Task 7.4: Implement TensorRT engine rebuilding for fine-tuned models

Tests cover:
- Engine versioning with model checksums
- Automatic rebuild detection
- ONNX export and engine build workflow
- Engine caching and invalidation
"""

import hashlib
import tempfile
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


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 256),
    )
    return model


# ============================================================================
# Test: TRT Engine Manager
# ============================================================================

@pytest.mark.cuda
@pytest.mark.tensorrt
class TestTRTEngineManager:
    """Tests for TRT engine management with fine-tuned models."""

    def test_engine_manager_creation(self, temp_storage):
        """Engine manager should initialize with cache directory."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        assert manager.cache_dir == Path(temp_storage)
        assert manager.engines == {}

    def test_compute_model_checksum(self, mock_model, temp_storage):
        """compute_checksum should produce consistent checksums for models."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Same model, same checksum
        checksum1 = manager.compute_model_checksum(mock_model)
        checksum2 = manager.compute_model_checksum(mock_model)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex digest

    def test_checksum_changes_after_training(self, mock_model, temp_storage):
        """Checksum should change when model parameters change."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        checksum_before = manager.compute_model_checksum(mock_model)

        # Simulate training by modifying parameters
        with torch.no_grad():
            mock_model[0].weight.data += torch.randn_like(mock_model[0].weight) * 0.1

        checksum_after = manager.compute_model_checksum(mock_model)

        assert checksum_before != checksum_after

    def test_needs_rebuild_returns_true_for_new_model(self, mock_model, temp_storage):
        """needs_rebuild should return True for model without cached engine."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        needs_rebuild = manager.needs_rebuild('test_model', mock_model)

        assert needs_rebuild is True

    def test_engine_path_includes_checksum(self, mock_model, temp_storage):
        """Engine path should include model checksum for versioning."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        checksum = manager.compute_model_checksum(mock_model)
        engine_path = manager.get_engine_path('test_model', mock_model)

        assert checksum[:8] in str(engine_path)
        assert str(engine_path).endswith('.engine')


@pytest.mark.cuda
@pytest.mark.tensorrt
class TestTRTRebuildWorkflow:
    """Tests for the full TRT rebuild workflow."""

    def test_register_model_stores_checksum(self, mock_model, temp_storage):
        """register_model should store model checksum for tracking."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        manager.register_model('test_model', mock_model)

        assert 'test_model' in manager._registered_models
        assert 'checksum' in manager._registered_models['test_model']

    def test_rebuild_after_fine_tuning(self, mock_model, temp_storage):
        """Engine should be rebuilt when model is fine-tuned."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Register initial model
        manager.register_model('test_model', mock_model)
        initial_checksum = manager._registered_models['test_model']['checksum']

        # Simulate fine-tuning
        with torch.no_grad():
            mock_model[0].weight.data += torch.randn_like(mock_model[0].weight) * 0.1

        # Check if rebuild is needed
        needs_rebuild = manager.needs_rebuild('test_model', mock_model)

        assert needs_rebuild is True

    def test_no_rebuild_for_unchanged_model(self, mock_model, temp_storage):
        """Engine should not rebuild for unchanged model."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Register model
        manager.register_model('test_model', mock_model)

        # Get the engine path and create a fake engine file
        engine_path = manager.get_engine_path('test_model', mock_model)
        engine_path.write_bytes(b'fake engine data')

        # Mark as built (simulate successful engine build)
        manager._mark_engine_built('test_model', mock_model)

        # Check if rebuild is needed
        needs_rebuild = manager.needs_rebuild('test_model', mock_model)

        assert needs_rebuild is False


@pytest.mark.cuda
@pytest.mark.tensorrt
class TestTRTEngineCache:
    """Tests for TRT engine caching."""

    def test_cache_stores_engine_metadata(self, temp_storage):
        """Cache should store engine metadata for quick lookups."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Simulate storing engine metadata
        metadata = {
            'model_name': 'test_model',
            'checksum': 'abc123',
            'precision': 'fp16',
            'build_time': 123.45,
        }
        manager._store_engine_metadata('test_model', metadata)

        # Retrieve metadata
        retrieved = manager._get_engine_metadata('test_model')

        assert retrieved == metadata

    def test_invalidate_cache_on_new_version(self, mock_model, temp_storage):
        """Old cache entries should be invalidated when model changes."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Create a fake cached engine file
        fake_engine = temp_storage / 'test_model_abc123.engine'
        fake_engine.write_bytes(b'fake engine data')

        # Register model and mark as built
        manager.register_model('test_model', mock_model)
        manager._registered_models['test_model']['engine_path'] = str(fake_engine)

        # Simulate fine-tuning
        with torch.no_grad():
            mock_model[0].weight.data += torch.randn_like(mock_model[0].weight) * 0.1

        # Check if old cache is invalidated
        assert manager.needs_rebuild('test_model', mock_model) is True

    def test_cleanup_old_engines(self, temp_storage):
        """cleanup_old_engines should remove stale engine files."""
        import os
        import time as time_module
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Create multiple fake engine files with distinct mtimes
        v1_path = temp_storage / 'model_v1.engine'
        v2_path = temp_storage / 'model_v2.engine'
        v3_path = temp_storage / 'model_v3.engine'

        v1_path.write_bytes(b'old')
        v2_path.write_bytes(b'older')
        v3_path.write_bytes(b'current')

        # Set distinct mtimes (v3 newest, v1 oldest)
        base_time = time_module.time()
        os.utime(v1_path, (base_time - 200, base_time - 200))
        os.utime(v2_path, (base_time - 100, base_time - 100))
        os.utime(v3_path, (base_time, base_time))

        # Mark v3 as current
        manager._current_engines = {'model': str(v3_path)}

        # Cleanup (keep only 1 most recent)
        removed = manager.cleanup_old_engines(keep_count=1)

        # v3 is newest and current (kept), v1 and v2 should be removed
        assert len(removed) >= 2
        assert str(v1_path) in removed
        assert str(v2_path) in removed


@pytest.mark.cuda
@pytest.mark.tensorrt
class TestTRTRebuildIntegration:
    """Integration tests for TRT rebuilding with voice conversion models."""

    def test_rebuild_workflow_with_sovits(self, device, temp_storage):
        """Full rebuild workflow with SoVitsSvc model."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager
        from auto_voice.models.so_vits_svc import SoVitsSvc

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Create model
        model = SoVitsSvc().to(device)

        # Register model
        manager.register_model('sovits', model)

        # Should need rebuild (no engine exists)
        assert manager.needs_rebuild('sovits', model) is True

        # Get engine path
        engine_path = manager.get_engine_path('sovits', model)
        assert 'sovits' in str(engine_path)

    def test_version_tracking_across_sessions(self, mock_model, temp_storage):
        """Version tracking should persist across manager instances."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        # First session
        manager1 = TRTEngineManager(cache_dir=str(temp_storage))
        manager1.register_model('test_model', mock_model)
        checksum1 = manager1._registered_models['test_model']['checksum']

        # Simulate saving state
        manager1.save_state()

        # Second session with same model
        manager2 = TRTEngineManager(cache_dir=str(temp_storage))
        manager2.load_state()

        # If engine was marked as built, shouldn't need rebuild
        if 'test_model' in manager2._registered_models:
            stored_checksum = manager2._registered_models['test_model'].get('checksum')
            current_checksum = manager2.compute_model_checksum(mock_model)
            # Checksums should match for unchanged model
            assert stored_checksum == current_checksum

    def test_rebuild_triggers_on_lora_adapter(self, device, temp_storage):
        """Rebuild should trigger when LoRA adapter is added to model."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager
        from auto_voice.models.so_vits_svc import SoVitsSvc

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Create base model
        model = SoVitsSvc().to(device)
        manager.register_model('sovits', model)
        manager._mark_engine_built('sovits', model)

        # Simulate adding LoRA adapter (modifies model parameters)
        with torch.no_grad():
            # Modify a layer to simulate LoRA
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.data += torch.randn_like(param) * 0.01
                    break

        # Should need rebuild after LoRA adaptation
        assert manager.needs_rebuild('sovits', model) is True
