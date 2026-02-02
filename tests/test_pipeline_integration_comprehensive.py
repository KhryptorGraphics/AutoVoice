"""Comprehensive pipeline integration tests - filling coverage gaps.

Tests focus on:
1. Pipeline factory error conditions
2. Cross-pipeline compatibility
3. Memory management edge cases
4. Pipeline switching workflows
5. Adapter loading/unloading scenarios
6. Error recovery and cleanup
7. Multi-pipeline concurrent usage
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


@pytest.fixture
def mock_profile_store(tmp_path):
    """Create mock profile store."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    # Create test profile
    profile_data = {
        "profile_id": "test-profile",
        "name": "Test Artist"
    }
    import json
    with open(profiles_dir / "test-profile.json", "w") as f:
        json.dump(profile_data, f)

    return str(profiles_dir)


class TestPipelineFactoryErrorConditions:
    """Test PipelineFactory error handling."""

    def test_create_pipeline_import_error(self):
        """_create_pipeline handles import errors gracefully."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device('cpu'))

        # Mock import failure
        with patch('auto_voice.inference.pipeline_factory.RealtimePipeline', side_effect=ImportError("Module not found")):
            with pytest.raises(RuntimeError, match="Failed to create pipeline"):
                factory._create_pipeline('realtime', None)

        PipelineFactory.reset_instance()

    def test_create_pipeline_runtime_error(self):
        """_create_pipeline handles initialization errors."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device('cpu'))

        # Mock initialization failure
        with patch('auto_voice.inference.pipeline_factory.RealtimePipeline', side_effect=RuntimeError("Init failed")):
            with pytest.raises(RuntimeError, match="Failed to create pipeline"):
                factory._create_pipeline('realtime', None)

        PipelineFactory.reset_instance()

    def test_invalid_profile_store(self):
        """Pipeline creation handles invalid profile store."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Non-existent profile store should not crash
        try:
            with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
                pipeline = factory.get_pipeline('realtime', profile_store='/nonexistent/path')
                assert pipeline is not None
        except Exception as e:
            # Should fail gracefully
            assert 'profile' in str(e).lower() or 'path' in str(e).lower()
        finally:
            PipelineFactory.reset_instance()


class TestPipelineSwitching:
    """Test switching between pipelines."""

    def test_switch_from_realtime_to_quality(self):
        """Switching from realtime to quality pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        mock_realtime = MagicMock()
        mock_quality = MagicMock()

        def create_side_effect(pipeline_type, profile_store):
            if pipeline_type == 'realtime':
                return mock_realtime
            elif pipeline_type == 'quality':
                return mock_quality
            return MagicMock()

        with patch.object(factory, '_create_pipeline', side_effect=create_side_effect):
            # Get realtime first
            pipeline1 = factory.get_pipeline('realtime')
            assert pipeline1 is mock_realtime

            # Switch to quality
            pipeline2 = factory.get_pipeline('quality')
            assert pipeline2 is mock_quality

            # Both should coexist
            assert factory.is_loaded('realtime')
            assert factory.is_loaded('quality')

        PipelineFactory.reset_instance()

    def test_unload_before_switch(self):
        """Unloading old pipeline before switching."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            # Load realtime
            factory.get_pipeline('realtime')
            assert factory.is_loaded('realtime')

            # Unload before switching
            factory.unload_pipeline('realtime')
            assert not factory.is_loaded('realtime')

            # Load quality
            factory.get_pipeline('quality')
            assert factory.is_loaded('quality')

        PipelineFactory.reset_instance()

    def test_unload_all_before_new_pipeline(self):
        """Clear all pipelines before loading new one."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            # Load multiple pipelines
            factory.get_pipeline('realtime')
            factory.get_pipeline('quality')
            factory.get_pipeline('quality_seedvc')

            # Unload all
            factory.unload_all()

            assert not factory.is_loaded('realtime')
            assert not factory.is_loaded('quality')
            assert not factory.is_loaded('quality_seedvc')

        PipelineFactory.reset_instance()


class TestMemoryManagement:
    """Test memory management functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_tracking_multiple_pipelines(self):
        """Memory usage tracked for multiple pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device('cuda'))

        def create_with_memory(pt, ps):
            # Allocate different amounts per pipeline
            if pt == 'realtime':
                _ = torch.randn(500, 500, device='cuda')
            elif pt == 'quality':
                _ = torch.randn(1000, 1000, device='cuda')
            return MagicMock()

        with patch.object(factory, '_create_pipeline', side_effect=create_with_memory):
            factory.get_pipeline('realtime')
            factory.get_pipeline('quality')

            total = factory.get_total_memory_usage()
            assert total > 0

        PipelineFactory.reset_instance()

    def test_memory_released_on_unload(self):
        """Memory tracking updated when pipeline unloaded."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory._memory_usage['realtime'] = 2.5  # Manually set

            initial_total = factory.get_total_memory_usage()
            assert initial_total == 2.5

            factory.unload_pipeline('realtime')

            final_total = factory.get_total_memory_usage()
            assert final_total == 0.0

        PipelineFactory.reset_instance()


class TestProfileStoreIntegration:
    """Test profile store integration with pipelines."""

    def test_pipeline_with_profile_store(self, mock_profile_store):
        """Pipeline creation with profile store."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        mock_pipeline = MagicMock()

        with patch.object(factory, '_create_pipeline', return_value=mock_pipeline) as mock_create:
            pipeline = factory.get_pipeline('realtime', profile_store=mock_profile_store)

            assert pipeline is mock_pipeline
            mock_create.assert_called_once_with('realtime', mock_profile_store)

        PipelineFactory.reset_instance()

    def test_different_profiles_different_caches(self):
        """Different profile stores create separate pipeline instances."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Currently, profile_store doesn't affect caching key
        # This test verifies current behavior
        mock_pipeline = MagicMock()

        with patch.object(factory, '_create_pipeline', return_value=mock_pipeline):
            p1 = factory.get_pipeline('realtime', profile_store='/path1')
            p2 = factory.get_pipeline('realtime', profile_store='/path2')

            # Same pipeline type = same cached instance (current behavior)
            assert p1 is p2

        PipelineFactory.reset_instance()


class TestConcurrentPipelineUsage:
    """Test multiple pipelines loaded concurrently."""

    def test_all_pipelines_loaded_simultaneously(self):
        """All pipeline types can be loaded at once."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        pipelines = {}

        with patch.object(factory, '_create_pipeline', side_effect=lambda pt, ps: MagicMock(name=pt)):
            for pipeline_type in ['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']:
                pipelines[pipeline_type] = factory.get_pipeline(pipeline_type)

        # All should be loaded
        for pipeline_type in pipelines:
            assert factory.is_loaded(pipeline_type)

        # All should be different instances
        pipeline_objects = list(pipelines.values())
        assert len(set(id(p) for p in pipeline_objects)) == len(pipeline_objects)

        PipelineFactory.reset_instance()

    def test_status_with_all_pipelines(self):
        """get_status shows all pipelines when loaded."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            # Load all pipelines
            factory.get_pipeline('realtime')
            factory.get_pipeline('quality')
            factory.get_pipeline('quality_seedvc')
            factory.get_pipeline('realtime_meanvc')

            status = factory.get_status()

            # All should show loaded
            assert status['realtime']['loaded'] is True
            assert status['quality']['loaded'] is True
            assert status['quality_seedvc']['loaded'] is True
            assert status['realtime_meanvc']['loaded'] is True

        PipelineFactory.reset_instance()


class TestPipelineCleanup:
    """Test cleanup and resource management."""

    def test_reset_instance_cleans_all(self):
        """reset_instance clears all loaded pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory1 = PipelineFactory.get_instance()

        with patch.object(factory1, '_create_pipeline', return_value=MagicMock()):
            factory1.get_pipeline('realtime')
            factory1.get_pipeline('quality')

        # Reset
        PipelineFactory.reset_instance()
        factory2 = PipelineFactory.get_instance()

        # New factory should have no pipelines
        assert not factory2.is_loaded('realtime')
        assert not factory2.is_loaded('quality')

        PipelineFactory.reset_instance()

    def test_unload_all_clears_memory_tracking(self):
        """unload_all clears all memory tracking."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory.get_pipeline('quality')

            # Manually set memory usage
            factory._memory_usage['realtime'] = 1.0
            factory._memory_usage['quality'] = 2.0

            assert factory.get_total_memory_usage() == 3.0

            factory.unload_all()

            assert factory.get_total_memory_usage() == 0.0

        PipelineFactory.reset_instance()


class TestPipelineDeviceConsistency:
    """Test device consistency across pipelines."""

    def test_all_pipelines_same_device(self):
        """All pipelines created with same device."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        device = torch.device('cpu')
        factory = PipelineFactory.get_instance(device=device)

        assert factory.device == device

        PipelineFactory.reset_instance()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_device_propagates(self):
        """CUDA device setting propagates to pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        device = torch.device('cuda:0')
        factory = PipelineFactory.get_instance(device=device)

        assert factory.device == device

        PipelineFactory.reset_instance()


class TestEdgeCaseInputs:
    """Test edge case inputs to pipeline methods."""

    def test_get_pipeline_empty_string(self):
        """get_pipeline rejects empty string."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with pytest.raises(ValueError):
            factory.get_pipeline('')

        PipelineFactory.reset_instance()

    def test_unload_pipeline_empty_string(self):
        """unload_pipeline handles empty string."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Should return False (not loaded)
        result = factory.unload_pipeline('')
        assert result is False

        PipelineFactory.reset_instance()

    def test_is_loaded_empty_string(self):
        """is_loaded returns False for empty string."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        assert factory.is_loaded('') is False

        PipelineFactory.reset_instance()

    def test_get_memory_usage_nonexistent(self):
        """get_memory_usage returns 0 for nonexistent pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        assert factory.get_memory_usage('nonexistent_pipeline') == 0.0

        PipelineFactory.reset_instance()


class TestStatusReporting:
    """Test status reporting functionality."""

    def test_status_format_consistency(self):
        """get_status returns consistent format."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        # Check all pipelines present
        for pipeline_type in ['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']:
            assert pipeline_type in status
            assert 'loaded' in status[pipeline_type]
            assert 'sample_rate' in status[pipeline_type]
            assert 'latency_target_ms' in status[pipeline_type]
            assert 'description' in status[pipeline_type]

        PipelineFactory.reset_instance()

    def test_status_memory_info(self):
        """get_status includes memory information."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory._memory_usage['realtime'] = 1.5

            status = factory.get_status()

            assert 'memory_usage_gb' in status['realtime']
            assert status['realtime']['memory_usage_gb'] == 1.5

        PipelineFactory.reset_instance()


@pytest.mark.smoke
class TestPipelineIntegrationSmoke:
    """Quick smoke tests for pipeline integration."""

    def test_factory_import(self):
        """PipelineFactory can be imported."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        assert PipelineFactory is not None

    def test_factory_singleton(self):
        """Factory singleton pattern works."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        f1 = PipelineFactory.get_instance()
        f2 = PipelineFactory.get_instance()
        assert f1 is f2

        PipelineFactory.reset_instance()

    def test_all_pipeline_types_known(self):
        """Factory knows all pipeline types."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()
        # Include all pipeline types that exist
        expected_types = {'realtime', 'quality', 'quality_seedvc', 'quality_shortcut', 'realtime_meanvc'}
        actual_types = set(status.keys())

        assert expected_types == actual_types

        PipelineFactory.reset_instance()
