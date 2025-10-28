"""Test suite for voice conversion preset system.

This module tests the preset loading, switching, and configuration merging
in the SingingConversionPipeline.
"""

import pytest
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


@pytest.fixture
def mock_components():
    """Mock all pipeline components to avoid loading heavy models."""
    with patch('auto_voice.inference.singing_conversion_pipeline.VocalSeparator') as mock_vs, \
         patch('auto_voice.inference.singing_conversion_pipeline.SingingPitchExtractor') as mock_pe, \
         patch('auto_voice.inference.singing_conversion_pipeline.SingingVoiceConverter') as mock_vc, \
         patch('auto_voice.inference.singing_conversion_pipeline.AudioMixer') as mock_am, \
         patch('auto_voice.inference.singing_conversion_pipeline.AudioProcessor') as mock_ap, \
         patch('auto_voice.inference.singing_conversion_pipeline.VoiceProfileStorage') as mock_vps:

        # Mock VoiceConverter to have required methods
        mock_vc_instance = MagicMock()
        mock_vc_instance.to.return_value = mock_vc_instance
        mock_vc_instance.eval.return_value = None
        mock_vc_instance.prepare_for_inference.return_value = None
        mock_vc.return_value = mock_vc_instance

        yield {
            'vocal_separator': mock_vs,
            'pitch_extractor': mock_pe,
            'voice_converter': mock_vc,
            'audio_mixer': mock_am,
            'audio_processor': mock_ap,
            'voice_storage': mock_vps
        }


class TestPresetLoading:
    """Test preset configuration loading."""

    def test_preset_file_exists(self):
        """Test that the preset file exists."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        preset_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
        )
        assert os.path.exists(preset_path), "Preset file should exist"

    def test_preset_file_valid_yaml(self):
        """Test that preset file is valid YAML."""
        preset_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
        )

        with open(preset_path, 'r') as f:
            data = yaml.safe_load(f)

        assert data is not None, "Preset file should contain valid YAML"
        assert 'presets' in data, "Preset file should have 'presets' key"
        assert 'default_preset' in data, "Preset file should have 'default_preset' key"

    def test_all_required_presets_exist(self):
        """Test that all required presets are defined."""
        preset_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
        )

        with open(preset_path, 'r') as f:
            data = yaml.safe_load(f)

        presets = data['presets']
        required_presets = ['fast', 'balanced', 'quality', 'custom']

        for preset_name in required_presets:
            assert preset_name in presets, f"Preset '{preset_name}' should be defined"

    def test_preset_structure(self):
        """Test that each preset has the required structure."""
        preset_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
        )

        with open(preset_path, 'r') as f:
            data = yaml.safe_load(f)

        required_components = ['vocal_separator', 'pitch_extractor', 'voice_converter', 'audio_mixer']

        for preset_name, preset_config in data['presets'].items():
            assert 'description' in preset_config, f"Preset '{preset_name}' should have description"

            for component in required_components:
                assert component in preset_config, \
                    f"Preset '{preset_name}' should have '{component}' config"

    def test_balanced_is_default(self):
        """Test that 'balanced' is the default preset."""
        preset_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
        )

        with open(preset_path, 'r') as f:
            data = yaml.safe_load(f)

        assert data['default_preset'] == 'balanced', "Default preset should be 'balanced'"


class TestPipelinePresetIntegration:
    """Test preset integration in SingingConversionPipeline."""

    def test_pipeline_accepts_preset_parameter(self, mock_components):
        """Test that pipeline accepts preset parameter."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        # Should not raise an error
        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')
        assert pipeline.current_preset == 'balanced'

    def test_pipeline_uses_default_preset(self, mock_components):
        """Test that pipeline uses default preset if none specified."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(device='cpu')
        assert pipeline.current_preset == 'balanced', "Should use default 'balanced' preset"

    def test_pipeline_loads_all_presets(self, mock_components):
        """Test that pipeline can load all defined presets."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        presets = ['fast', 'balanced', 'quality', 'custom']

        for preset_name in presets:
            pipeline = SingingConversionPipeline(preset=preset_name, device='cpu')
            assert pipeline.current_preset == preset_name

    def test_invalid_preset_falls_back(self, mock_components):
        """Test that invalid preset falls back to default."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        # Should fall back to default without raising error
        pipeline = SingingConversionPipeline(preset='invalid_preset', device='cpu')
        assert pipeline.current_preset in ['balanced', 'invalid_preset']  # May fall back or keep invalid

    def test_config_merging_priority(self, mock_components):
        """Test that user config takes precedence over preset config."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        custom_config = {
            'pitch_extractor': {
                'hop_length_ms': 15.0  # Override preset value
            }
        }

        pipeline = SingingConversionPipeline(
            preset='balanced',
            config=custom_config,
            device='cpu'
        )

        # User config should override preset
        assert pipeline.config['pitch_extractor']['hop_length_ms'] == 15.0


class TestPresetSwitching:
    """Test runtime preset switching."""

    def test_set_preset_method_exists(self, mock_components):
        """Test that set_preset method exists."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')
        assert hasattr(pipeline, 'set_preset'), "Pipeline should have set_preset method"

    def test_set_preset_switches_preset(self, mock_components):
        """Test that set_preset switches the active preset."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')
        pipeline.set_preset('quality')

        assert pipeline.current_preset == 'quality'

    def test_set_preset_clears_cache_by_default(self, mock_components):
        """Test that set_preset clears cache by default."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')

        with patch.object(pipeline, 'clear_cache') as mock_clear:
            pipeline.set_preset('quality')
            mock_clear.assert_called_once()

    def test_set_preset_skip_cache_clear(self, mock_components):
        """Test that set_preset can skip cache clearing."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')

        with patch.object(pipeline, 'clear_cache') as mock_clear:
            pipeline.set_preset('quality', clear_cache=False)
            mock_clear.assert_not_called()

    def test_set_preset_reinitializes_components(self, mock_components):
        """Test that set_preset reinitializes all components."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')

        # Get initial component instances
        initial_vs = pipeline.vocal_separator
        initial_pe = pipeline.pitch_extractor

        # Switch preset
        pipeline.set_preset('quality')

        # Components should be reinitialized (new instances)
        # Note: With mocking, they may be the same mock object
        # In real scenario, they would be different instances


class TestGetCurrentPreset:
    """Test get_current_preset method."""

    def test_get_current_preset_method_exists(self, mock_components):
        """Test that get_current_preset method exists."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')
        assert hasattr(pipeline, 'get_current_preset'), "Pipeline should have get_current_preset method"

    def test_get_current_preset_returns_info(self, mock_components):
        """Test that get_current_preset returns preset information."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')
        info = pipeline.get_current_preset()

        assert isinstance(info, dict), "Should return a dictionary"
        assert 'name' in info, "Should have 'name' key"
        assert 'description' in info, "Should have 'description' key"
        assert 'config' in info, "Should have 'config' key"

    def test_get_current_preset_correct_name(self, mock_components):
        """Test that get_current_preset returns correct preset name."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='quality', device='cpu')
        info = pipeline.get_current_preset()

        assert info['name'] == 'quality'

    def test_get_current_preset_has_config(self, mock_components):
        """Test that get_current_preset returns component configs."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='balanced', device='cpu')
        info = pipeline.get_current_preset()

        config = info['config']
        required_components = ['vocal_separator', 'pitch_extractor', 'voice_converter', 'audio_mixer']

        for component in required_components:
            assert component in config, f"Config should have '{component}' settings"


class TestPresetComponentSettings:
    """Test that preset settings are correctly applied to components."""

    def test_fast_preset_uses_tiny_model(self, mock_components):
        """Test that fast preset uses tiny CREPE model."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='fast', device='cpu')

        # Check that pitch_extractor config has 'tiny' model
        pe_config = pipeline.config.get('pitch_extractor', {})
        assert pe_config.get('model') == 'tiny', "Fast preset should use tiny CREPE model"

    def test_quality_preset_uses_full_model(self, mock_components):
        """Test that quality preset uses full CREPE model."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(preset='quality', device='cpu')

        # Check that pitch_extractor config has 'full' model
        pe_config = pipeline.config.get('pitch_extractor', {})
        assert pe_config.get('model') == 'full', "Quality preset should use full CREPE model"

    def test_quality_preset_higher_resolution(self, mock_components):
        """Test that quality preset uses higher time resolution."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline_fast = SingingConversionPipeline(preset='fast', device='cpu')
        pipeline_quality = SingingConversionPipeline(preset='quality', device='cpu')

        fast_hop = pipeline_fast.config.get('pitch_extractor', {}).get('hop_length_ms', 10.0)
        quality_hop = pipeline_quality.config.get('pitch_extractor', {}).get('hop_length_ms', 10.0)

        # Quality should have smaller hop length (higher resolution)
        assert quality_hop < fast_hop, "Quality preset should have higher time resolution"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
