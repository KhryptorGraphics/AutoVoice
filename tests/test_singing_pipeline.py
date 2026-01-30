"""Tests for SingingConversionPipeline."""
import numpy as np
import pytest

from auto_voice.inference.singing_conversion_pipeline import (
    SingingConversionPipeline, SeparationError, ConversionError, PRESETS
)


class TestExceptions:
    """Pipeline exception tests."""

    @pytest.mark.smoke
    def test_separation_error(self):
        with pytest.raises(SeparationError):
            raise SeparationError("separation failed")

    def test_conversion_error(self):
        with pytest.raises(ConversionError):
            raise ConversionError("conversion failed")


class TestPresets:
    """Preset configuration tests."""

    @pytest.mark.smoke
    def test_all_presets_exist(self):
        for name in ['draft', 'fast', 'balanced', 'high', 'studio']:
            assert name in PRESETS

    def test_preset_has_required_keys(self):
        for name, config in PRESETS.items():
            assert 'n_steps' in config
            assert 'denoise' in config

    def test_preset_ordering(self):
        assert PRESETS['draft']['n_steps'] < PRESETS['balanced']['n_steps']
        assert PRESETS['balanced']['n_steps'] < PRESETS['studio']['n_steps']


class TestSingingConversionPipeline:
    """Pipeline integration tests."""

    @pytest.mark.smoke
    def test_init(self, singing_pipeline):
        assert singing_pipeline is not None
        assert singing_pipeline._sample_rate == 22050

    def test_convert_nonexistent_file(self, singing_pipeline):
        with pytest.raises(ConversionError, match="not found"):
            singing_pipeline.convert_song(
                song_path='/nonexistent.wav',
                target_profile_id='test-profile'
            )

    @pytest.mark.integration
    def test_convert_song_full(self, singing_pipeline, voice_cloner, sample_audio_file):
        # Create a profile first
        profile = voice_cloner.create_voice_profile(audio=sample_audio_file)

        result = singing_pipeline.convert_song(
            song_path=sample_audio_file,
            target_profile_id=profile['profile_id'],
            vocal_volume=1.0,
            instrumental_volume=0.9,
            preset='draft'
        )

        assert 'mixed_audio' in result
        assert isinstance(result['mixed_audio'], np.ndarray)
        assert result['mixed_audio'].size > 0
        assert 'sample_rate' in result
        assert result['sample_rate'] == 22050
        assert 'duration' in result
        assert result['duration'] > 0
        assert 'metadata' in result
        assert result['metadata']['preset'] == 'draft'
        assert 'f0_contour' in result

    @pytest.mark.integration
    def test_convert_with_stems(self, singing_pipeline, voice_cloner, sample_audio_file):
        profile = voice_cloner.create_voice_profile(audio=sample_audio_file)

        result = singing_pipeline.convert_song(
            song_path=sample_audio_file,
            target_profile_id=profile['profile_id'],
            return_stems=True,
            preset='draft'
        )

        assert 'stems' in result
        assert 'vocals' in result['stems']
        assert 'instrumental' in result['stems']
        assert result['stems']['vocals'].size > 0

    @pytest.mark.integration
    def test_convert_with_pitch_shift(self, singing_pipeline, voice_cloner, sample_audio_file):
        profile = voice_cloner.create_voice_profile(audio=sample_audio_file)

        result = singing_pipeline.convert_song(
            song_path=sample_audio_file,
            target_profile_id=profile['profile_id'],
            pitch_shift=2.0,
            preset='draft'
        )

        assert result['metadata']['pitch_shift'] == 2.0

    def test_convert_invalid_profile(self, singing_pipeline, sample_audio_file):
        from auto_voice.storage.voice_profiles import ProfileNotFoundError
        with pytest.raises(ProfileNotFoundError):
            singing_pipeline.convert_song(
                song_path=sample_audio_file,
                target_profile_id='nonexistent-profile'
            )

    def test_extract_pitch(self, singing_pipeline, sample_audio):
        audio, sr = sample_audio
        f0 = singing_pipeline._extract_pitch(audio, sr)
        assert isinstance(f0, np.ndarray)
        assert len(f0) > 0

    @pytest.mark.integration
    def test_convert_song_bigvgan_vocoder_config(self, voice_cloner, sample_audio_file):
        """convert_song() with vocoder_type='bigvgan' in config."""
        import torch
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        from auto_voice.inference.model_manager import ModelManager
        from auto_voice.models.so_vits_svc import SoVitsSvc

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pipeline = SingingConversionPipeline(
            device=device, voice_cloner=voice_cloner,
            config={
                'speaker_id': 'default',
                'vocoder_type': 'bigvgan',
            }
        )

        # Pre-load ModelManager with BigVGAN vocoder
        mm = ModelManager(device=device, config={'speaker_id': 'default'})
        mm.load(vocoder_type='bigvgan')
        model = SoVitsSvc(config={'n_mels': 100})
        model.to(device)
        mm._sovits_models['default'] = model
        pipeline._model_manager = mm

        profile = voice_cloner.create_voice_profile(audio=sample_audio_file)
        result = pipeline.convert_song(
            song_path=sample_audio_file,
            target_profile_id=profile['profile_id'],
            preset='draft'
        )

        assert 'mixed_audio' in result
        assert isinstance(result['mixed_audio'], np.ndarray)
        assert result['mixed_audio'].size > 0
        assert not np.any(np.isnan(result['mixed_audio']))
        assert result['sample_rate'] == 22050
