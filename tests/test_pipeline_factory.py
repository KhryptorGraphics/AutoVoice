"""Tests for PipelineFactory - unified pipeline management.

The PipelineFactory provides:
- Lazy loading: Pipelines only initialized when first requested
- Caching: Re-uses existing pipeline instances
- Memory management: Can unload pipelines to free GPU memory
- Unified interface: All pipelines accessible via same API

Tests cover:
- Singleton behavior
- Lazy loading verification
- Pipeline type routing
- Memory tracking
- Pipeline unloading
"""
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


class TestPipelineFactoryInit:
    """Test PipelineFactory initialization."""

    def test_import_succeeds(self):
        """PipelineFactory can be imported."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        assert PipelineFactory is not None

    def test_singleton_pattern(self):
        """get_instance returns singleton."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        # Reset singleton
        PipelineFactory.reset_instance()

        factory1 = PipelineFactory.get_instance()
        factory2 = PipelineFactory.get_instance()

        assert factory1 is factory2

        # Clean up
        PipelineFactory.reset_instance()

    def test_reset_instance_clears_singleton(self):
        """reset_instance clears the singleton."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        factory1 = PipelineFactory.get_instance()
        PipelineFactory.reset_instance()
        factory2 = PipelineFactory.get_instance()

        assert factory1 is not factory2

        # Clean up
        PipelineFactory.reset_instance()

    def test_init_with_device(self):
        """Factory accepts device parameter."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device("cpu"))

        assert factory.device == torch.device("cpu")

        PipelineFactory.reset_instance()

    def test_init_default_device_cuda_if_available(self):
        """Factory defaults to CUDA if available."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        if torch.cuda.is_available():
            assert factory.device.type == "cuda"
        else:
            assert factory.device.type == "cpu"

        PipelineFactory.reset_instance()

    def test_init_creates_empty_caches(self):
        """Factory starts with empty pipeline caches."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        assert len(factory._pipelines) == 0
        assert len(factory._memory_usage) == 0

        PipelineFactory.reset_instance()


class TestPipelineTypeRouting:
    """Test pipeline type routing in get_pipeline."""

    def test_invalid_pipeline_type_raises(self):
        """get_pipeline raises ValueError for invalid type."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with pytest.raises(ValueError, match="Unknown pipeline type"):
            factory.get_pipeline("invalid_type")

        PipelineFactory.reset_instance()

    def test_valid_pipeline_types(self):
        """get_pipeline accepts all valid pipeline types."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        valid_types = ['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Just check validation doesn't raise - actual creation mocked
        for pt in valid_types:
            # Mock the creation to avoid loading real models
            with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
                pipeline = factory.get_pipeline(pt)
                assert pipeline is not None

        PipelineFactory.reset_instance()


class TestLazyLoading:
    """Test lazy loading behavior."""

    def test_pipeline_not_loaded_until_requested(self):
        """Pipelines are not loaded on factory init."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # No pipelines loaded yet
        assert not factory.is_loaded('realtime')
        assert not factory.is_loaded('quality')
        assert not factory.is_loaded('quality_seedvc')
        assert not factory.is_loaded('realtime_meanvc')

        PipelineFactory.reset_instance()

    def test_pipeline_loaded_on_first_request(self):
        """Pipeline loads on first get_pipeline call."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        mock_pipeline = MagicMock()
        with patch.object(factory, '_create_pipeline', return_value=mock_pipeline) as mock_create:
            pipeline = factory.get_pipeline('realtime')

            assert factory.is_loaded('realtime')
            mock_create.assert_called_once_with('realtime', None)

        PipelineFactory.reset_instance()

    def test_pipeline_reused_on_second_request(self):
        """Subsequent requests return cached pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        mock_pipeline = MagicMock()
        with patch.object(factory, '_create_pipeline', return_value=mock_pipeline) as mock_create:
            pipeline1 = factory.get_pipeline('realtime')
            pipeline2 = factory.get_pipeline('realtime')

            assert pipeline1 is pipeline2
            # Only created once
            assert mock_create.call_count == 1

        PipelineFactory.reset_instance()


class TestPipelineCaching:
    """Test pipeline caching behavior."""

    def test_different_types_have_different_instances(self):
        """Different pipeline types create different instances."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        realtime_mock = MagicMock()
        quality_mock = MagicMock()

        def create_side_effect(pipeline_type, profile_store):
            if pipeline_type == 'realtime':
                return realtime_mock
            elif pipeline_type == 'quality':
                return quality_mock
            return MagicMock()

        with patch.object(factory, '_create_pipeline', side_effect=create_side_effect):
            realtime = factory.get_pipeline('realtime')
            quality = factory.get_pipeline('quality')

            assert realtime is not quality
            assert realtime is realtime_mock
            assert quality is quality_mock

        PipelineFactory.reset_instance()

    def test_is_loaded_returns_correct_status(self):
        """is_loaded correctly reports pipeline status."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            assert not factory.is_loaded('realtime')

            factory.get_pipeline('realtime')
            assert factory.is_loaded('realtime')
            assert not factory.is_loaded('quality')

        PipelineFactory.reset_instance()


class TestMemoryTracking:
    """Test GPU memory tracking."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage_tracked(self):
        """Memory usage is tracked for loaded pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device("cuda"))

        # Create a pipeline that allocates some GPU memory
        def create_with_memory(pt, ps):
            mock = MagicMock()
            # Allocate some memory
            torch.cuda.empty_cache()
            _ = torch.randn(1000, 1000, device="cuda")
            return mock

        with patch.object(factory, '_create_pipeline', side_effect=create_with_memory):
            factory.get_pipeline('realtime')

            # Should have some memory recorded
            assert factory.get_memory_usage('realtime') >= 0

        PipelineFactory.reset_instance()

    def test_get_memory_usage_returns_zero_for_unloaded(self):
        """get_memory_usage returns 0 for unloaded pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        assert factory.get_memory_usage('realtime') == 0.0
        assert factory.get_memory_usage('quality') == 0.0

        PipelineFactory.reset_instance()

    def test_get_total_memory_usage(self):
        """get_total_memory_usage sums all pipeline memory."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Manually set memory usage for testing
        factory._memory_usage['realtime'] = 1.5
        factory._memory_usage['quality'] = 2.5

        assert factory.get_total_memory_usage() == 4.0

        PipelineFactory.reset_instance()


class TestPipelineUnloading:
    """Test pipeline unloading functionality."""

    def test_unload_pipeline_removes_from_cache(self):
        """unload_pipeline removes pipeline from cache."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            assert factory.is_loaded('realtime')

            result = factory.unload_pipeline('realtime')

            assert result is True
            assert not factory.is_loaded('realtime')

        PipelineFactory.reset_instance()

    def test_unload_pipeline_clears_memory_tracking(self):
        """unload_pipeline clears memory tracking."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory._memory_usage['realtime'] = 1.5

            factory.unload_pipeline('realtime')

            assert factory.get_memory_usage('realtime') == 0.0

        PipelineFactory.reset_instance()

    def test_unload_pipeline_returns_false_if_not_loaded(self):
        """unload_pipeline returns False for unloaded pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        result = factory.unload_pipeline('realtime')
        assert result is False

        PipelineFactory.reset_instance()

    def test_unload_all(self):
        """unload_all removes all cached pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory.get_pipeline('quality')

            assert factory.is_loaded('realtime')
            assert factory.is_loaded('quality')

            factory.unload_all()

            assert not factory.is_loaded('realtime')
            assert not factory.is_loaded('quality')

        PipelineFactory.reset_instance()


class TestGetStatus:
    """Test get_status for API responses."""

    def test_get_status_returns_all_pipeline_info(self):
        """get_status returns info for all pipeline types."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert 'realtime' in status
        assert 'quality' in status
        assert 'quality_seedvc' in status
        assert 'realtime_meanvc' in status

        PipelineFactory.reset_instance()

    def test_get_status_shows_loaded_state(self):
        """get_status shows whether pipelines are loaded."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        # All should be unloaded initially
        assert status['realtime']['loaded'] is False
        assert status['quality']['loaded'] is False

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')

            status = factory.get_status()
            assert status['realtime']['loaded'] is True
            assert status['quality']['loaded'] is False

        PipelineFactory.reset_instance()

    def test_get_status_includes_sample_rates(self):
        """get_status includes sample rate info."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert status['realtime']['sample_rate'] == 22050
        assert status['quality']['sample_rate'] == 24000
        assert status['quality_seedvc']['sample_rate'] == 44100
        assert status['realtime_meanvc']['sample_rate'] == 16000

        PipelineFactory.reset_instance()

    def test_get_status_includes_latency_targets(self):
        """get_status includes latency targets."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert status['realtime']['latency_target_ms'] == 100
        assert status['quality']['latency_target_ms'] == 3000
        assert status['quality_seedvc']['latency_target_ms'] == 2000
        assert status['realtime_meanvc']['latency_target_ms'] == 80

        PipelineFactory.reset_instance()

    def test_get_status_includes_descriptions(self):
        """get_status includes pipeline descriptions."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert 'description' in status['realtime']
        assert 'description' in status['quality']
        assert 'karaoke' in status['realtime']['description'].lower()
        assert 'features' in status['quality_seedvc']

        PipelineFactory.reset_instance()


class TestPipelineCreation:
    """Test actual pipeline creation logic."""

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_create_realtime_pipeline(self):
        """_create_pipeline creates RealtimePipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # This loads real models - slow
        pipeline = factory._create_pipeline('realtime', None)

        assert isinstance(pipeline, RealtimePipeline)

        PipelineFactory.reset_instance()

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_create_quality_pipeline(self):
        """_create_pipeline creates SOTAConversionPipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        pipeline = factory._create_pipeline('quality', None)

        assert isinstance(pipeline, SOTAConversionPipeline)

        PipelineFactory.reset_instance()

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_create_quality_seedvc_pipeline(self):
        """_create_pipeline creates SeedVCPipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        pipeline = factory._create_pipeline('quality_seedvc', None)

        assert isinstance(pipeline, SeedVCPipeline)

        PipelineFactory.reset_instance()

    def test_create_realtime_meanvc_uses_cpu(self):
        """MeanVC pipeline uses CPU by default."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Mock to avoid loading real models - patch at the source module
        with patch('auto_voice.inference.meanvc_pipeline.MeanVCPipeline') as MockMeanVC:
            mock_instance = MagicMock()
            MockMeanVC.return_value = mock_instance

            pipeline = factory._create_pipeline('realtime_meanvc', None)

            # Should be called with CPU device
            MockMeanVC.assert_called_once()
            call_kwargs = MockMeanVC.call_args[1]
            assert call_kwargs['device'] == torch.device('cpu')
            assert call_kwargs['require_gpu'] is False

        PipelineFactory.reset_instance()
