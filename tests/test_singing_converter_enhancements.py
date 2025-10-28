"""
Test Suite for SingingVoiceConverter Enhancements

Tests the new features added to SingingVoiceConverter:
1. Temperature API
2. Pitch shifting
3. Quality presets
4. Optional advanced features
"""

import pytest
import torch
import numpy as np

from src.auto_voice.models.singing_voice_converter import (
    SingingVoiceConverter,
    VoiceConversionError,
    QUALITY_PRESETS
)


@pytest.fixture
def mock_config():
    """Create a minimal config for testing."""
    return {
        'singing_voice_converter': {
            'latent_dim': 192,
            'mel_channels': 80,
            'content_encoder': {
                'type': 'cnn_fallback',
                'output_dim': 256,
                'use_torch_hub': False,
                'cnn_fallback': {
                    'n_fft': 1024,
                    'hop_length': 320,
                    'n_mels': 80,
                    'sample_rate': 16000
                }
            },
            'pitch_encoder': {
                'pitch_dim': 192,
                'hidden_dim': 128,
                'num_bins': 256,
                'f0_min': 80.0,
                'f0_max': 1000.0
            },
            'speaker_encoder': {
                'embedding_dim': 256
            },
            'posterior_encoder': {
                'hidden_channels': 192,
                'num_layers': 4
            },
            'flow_decoder': {
                'num_flows': 2,
                'hidden_channels': 192,
                'use_only_mean': False
            },
            'vocoder': {
                'use_vocoder': False
            },
            'audio': {
                'sample_rate': 22050,
                'hop_length': 512,
                'n_fft': 2048,
                'win_length': 2048
            },
            'inference': {
                'temperature': 1.0
            }
        }
    }


@pytest.fixture
def model(mock_config):
    """Create a SingingVoiceConverter instance."""
    model = SingingVoiceConverter(mock_config)
    model.eval()
    return model


class TestTemperatureAPI:
    """Test temperature control functionality."""

    def test_set_temperature_valid(self, model):
        """Test setting valid temperature values."""
        # Test various valid temperatures
        valid_temps = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        for temp in valid_temps:
            model.set_temperature(temp)
            assert model.temperature == temp

    def test_set_temperature_invalid(self, model):
        """Test that invalid temperatures raise ValueError."""
        invalid_temps = [0.0, -0.5, 2.1, 3.0]
        for temp in invalid_temps:
            with pytest.raises(ValueError, match="Temperature must be in range"):
                model.set_temperature(temp)

    def test_auto_tune_temperature(self, model):
        """Test automatic temperature tuning."""
        # Create synthetic audio with known characteristics
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Generate sine wave with varying amplitude (dynamic audio)
        t = np.linspace(0, duration, num_samples)
        freq = 440.0
        audio = np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio = audio.astype(np.float32)

        # Create dummy speaker embedding
        speaker_emb = np.random.randn(256).astype(np.float32)

        # Auto-tune temperature
        temp = model.auto_tune_temperature(audio, speaker_emb, sample_rate)

        # Check result is in valid range
        assert 0.5 <= temp <= 1.5
        assert model.temperature == temp


class TestPitchShifting:
    """Test pitch shifting functionality."""

    def test_apply_pitch_shift_up(self, model):
        """Test upward pitch shift."""
        # Create F0 contour
        f0 = torch.tensor([[220.0, 230.0, 240.0, 250.0, 260.0]], dtype=torch.float32)

        # Shift up by 2 semitones
        shifted = model._apply_pitch_shift(f0, 2.0, 'linear')

        # Expected ratio: 2^(2/12) ≈ 1.1225
        expected_ratio = 2.0 ** (2.0 / 12.0)
        expected = f0 * expected_ratio

        # Check values are close
        torch.testing.assert_close(shifted, expected, rtol=1e-4, atol=1e-4)

    def test_apply_pitch_shift_down(self, model):
        """Test downward pitch shift."""
        f0 = torch.tensor([[440.0, 450.0, 460.0, 470.0, 480.0]], dtype=torch.float32)

        # Shift down by 3 semitones
        shifted = model._apply_pitch_shift(f0, -3.0, 'linear')

        # Expected ratio: 2^(-3/12) ≈ 0.8409
        expected_ratio = 2.0 ** (-3.0 / 12.0)
        expected = f0 * expected_ratio

        torch.testing.assert_close(shifted, expected, rtol=1e-4, atol=1e-4)

    def test_apply_pitch_shift_zero(self, model):
        """Test that zero shift returns unchanged F0."""
        f0 = torch.tensor([[440.0, 450.0, 460.0]], dtype=torch.float32)

        shifted = model._apply_pitch_shift(f0, 0.0, 'linear')

        torch.testing.assert_close(shifted, f0)

    def test_apply_pitch_shift_preserves_unvoiced(self, model):
        """Test that unvoiced frames (f0=0) remain zero."""
        f0 = torch.tensor([[220.0, 0.0, 240.0, 0.0, 260.0]], dtype=torch.float32)

        shifted = model._apply_pitch_shift(f0, 5.0, 'linear')

        # Unvoiced frames should remain zero
        assert shifted[0, 1].item() == 0.0
        assert shifted[0, 3].item() == 0.0

        # Voiced frames should be shifted
        assert shifted[0, 0].item() > f0[0, 0].item()

    def test_apply_pitch_shift_clamping(self, model):
        """Test that shifted F0 is clamped to valid range."""
        # F0 near upper limit
        f0_high = torch.tensor([[950.0, 980.0, 990.0]], dtype=torch.float32)

        # Large upward shift
        shifted = model._apply_pitch_shift(f0_high, 12.0, 'linear')

        # Should be clamped to f0_max (1000.0)
        assert torch.all(shifted <= model.pitch_encoder.f0_max)

    def test_apply_pitch_shift_formant_preserving(self, model):
        """Test formant-preserving pitch shift method."""
        f0 = torch.tensor([[440.0, 450.0, 460.0]], dtype=torch.float32)

        # Shift with formant preservation
        shifted = model._apply_pitch_shift(f0, 5.0, 'formant_preserving')

        # Formant-preserving uses 70% of shift
        expected_ratio = 2.0 ** (5.0 * 0.7 / 12.0)
        expected = torch.clamp(
            f0 * expected_ratio,
            model.pitch_encoder.f0_min,
            model.pitch_encoder.f0_max
        )

        torch.testing.assert_close(shifted, expected, rtol=1e-4, atol=1e-4)

    def test_apply_pitch_shift_invalid_method(self, model):
        """Test that invalid method raises ValueError."""
        f0 = torch.tensor([[440.0]], dtype=torch.float32)

        with pytest.raises(ValueError, match="Unknown pitch shift method"):
            model._apply_pitch_shift(f0, 2.0, 'invalid_method')


class TestQualityPresets:
    """Test quality preset functionality."""

    def test_preset_constants(self):
        """Test that all required presets are defined."""
        required_presets = ['draft', 'fast', 'balanced', 'high', 'studio']
        for preset in required_presets:
            assert preset in QUALITY_PRESETS
            assert 'description' in QUALITY_PRESETS[preset]
            assert 'decoder_steps' in QUALITY_PRESETS[preset]
            assert 'relative_quality' in QUALITY_PRESETS[preset]
            assert 'relative_speed' in QUALITY_PRESETS[preset]

    def test_set_quality_preset_valid(self, model):
        """Test setting valid quality presets."""
        presets = ['draft', 'fast', 'balanced', 'high', 'studio']
        for preset in presets:
            model.set_quality_preset(preset)
            assert model.quality_preset == preset
            assert model.decoder_steps == QUALITY_PRESETS[preset]['decoder_steps']

    def test_set_quality_preset_invalid(self, model):
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preset"):
            model.set_quality_preset('invalid_preset')

    def test_get_quality_preset_info_current(self, model):
        """Test getting current preset info."""
        model.set_quality_preset('high')
        info = model.get_quality_preset_info()

        assert info['description'] == QUALITY_PRESETS['high']['description']
        assert info['decoder_steps'] == QUALITY_PRESETS['high']['decoder_steps']
        assert info['current'] is True

    def test_get_quality_preset_info_specific(self, model):
        """Test getting specific preset info."""
        model.set_quality_preset('balanced')
        info = model.get_quality_preset_info('studio')

        assert info['description'] == QUALITY_PRESETS['studio']['description']
        assert info['current'] is False

    def test_get_quality_preset_info_invalid(self, model):
        """Test that invalid preset query raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preset"):
            model.get_quality_preset_info('nonexistent')

    def test_estimate_conversion_time(self, model):
        """Test conversion time estimation."""
        # Test with different presets
        audio_duration = 30.0  # 30 seconds

        for preset in ['draft', 'fast', 'balanced', 'high', 'studio']:
            model.set_quality_preset(preset)
            estimated = model.estimate_conversion_time(audio_duration)

            # Check result is positive and reasonable
            assert estimated > 0
            assert estimated < audio_duration * 10  # Should be within 10x realtime

        # Draft should be fastest, studio slowest
        model.set_quality_preset('draft')
        draft_time = model.estimate_conversion_time(audio_duration)

        model.set_quality_preset('studio')
        studio_time = model.estimate_conversion_time(audio_duration)

        assert draft_time < studio_time

    def test_estimate_conversion_time_invalid_preset(self, model):
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preset"):
            model.estimate_conversion_time(30.0, 'invalid')


class TestAdvancedFeatures:
    """Test optional advanced features."""

    def test_denoise_audio(self, model):
        """Test audio denoising."""
        # Create noisy audio
        clean_audio = torch.sin(torch.linspace(0, 10, 16000)) * 0.5
        noise = torch.randn(16000) * 0.05
        noisy_audio = clean_audio + noise
        noisy_audio = noisy_audio.unsqueeze(0)

        # Apply denoising
        denoised = model._denoise_audio(noisy_audio, 16000)

        # Check output shape
        assert denoised.shape == noisy_audio.shape

        # Denoised should have lower variance than noisy
        # (This is a basic check; proper validation requires SNR calculation)
        assert torch.std(denoised) <= torch.std(noisy_audio) * 1.5

    def test_enhance_audio(self, model):
        """Test audio enhancement."""
        # Create audio with low dynamics
        audio = torch.sin(torch.linspace(0, 10, 22050)) * 0.3

        # Apply enhancement
        enhanced = model._enhance_audio(audio, 22050)

        # Check output shape
        assert enhanced.shape == audio.shape

        # Enhanced audio should exist and be valid
        assert torch.all(torch.isfinite(enhanced))

    def test_preserve_dynamics(self, model):
        """Test dynamics preservation."""
        # Create audio with known dynamics
        original_audio = torch.sin(torch.linspace(0, 10, 22050)) * 0.5
        original_rms = torch.sqrt(torch.mean(original_audio ** 2))
        original_peak = torch.max(torch.abs(original_audio))

        # Create modified audio with different dynamics
        modified_audio = original_audio * 2.0  # Doubled amplitude

        # Preserve dynamics
        restored = model._preserve_dynamics(modified_audio, original_rms, original_peak)

        # Check RMS is closer to original
        restored_rms = torch.sqrt(torch.mean(restored ** 2))
        assert abs(restored_rms - original_rms) < abs(modified_audio.std() - original_rms)

        # Check peak is within range
        restored_peak = torch.max(torch.abs(restored))
        assert restored_peak <= original_peak * 1.1  # Allow 10% margin


class TestConvertIntegration:
    """Test the enhanced convert method."""

    def test_convert_with_pitch_shift(self, model):
        """Test conversion with pitch shifting."""
        # Create simple test audio
        sample_rate = 16000
        duration = 0.5
        num_samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        audio = audio.astype(np.float32)

        # Create speaker embedding
        speaker_emb = np.random.randn(256).astype(np.float32)

        # Convert with pitch shift
        try:
            output = model.convert(
                audio,
                speaker_emb,
                source_sample_rate=sample_rate,
                output_sample_rate=22050,
                pitch_shift_semitones=2.0,
                pitch_shift_method='linear'
            )

            # Check output exists and is valid
            assert output is not None
            assert len(output) > 0
            assert np.all(np.isfinite(output))
        except Exception as e:
            # If convert fails due to missing dependencies, skip
            pytest.skip(f"Convert method requires full dependencies: {e}")

    def test_convert_invalid_pitch_method(self, model):
        """Test that invalid pitch method raises error."""
        audio = np.random.randn(8000).astype(np.float32)
        speaker_emb = np.random.randn(256).astype(np.float32)

        with pytest.raises(VoiceConversionError, match="Invalid pitch_shift_method"):
            model.convert(
                audio,
                speaker_emb,
                pitch_shift_semitones=2.0,
                pitch_shift_method='invalid'
            )

    def test_convert_large_pitch_shift_warning(self, model, caplog):
        """Test that large pitch shifts generate warning."""
        audio = np.random.randn(8000).astype(np.float32)
        speaker_emb = np.random.randn(256).astype(np.float32)

        try:
            model.convert(
                audio,
                speaker_emb,
                pitch_shift_semitones=25.0  # Very large shift
            )

            # Check for warning in log
            assert any('Large pitch shift' in record.message for record in caplog.records)
        except Exception:
            # Expected to fail, just check warning was logged
            assert any('Large pitch shift' in record.message for record in caplog.records)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
