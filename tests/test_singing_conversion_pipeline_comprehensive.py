"""Comprehensive tests for singing_conversion_pipeline.py.

Target: Increase coverage from 57% to 95%+

Test Categories:
1. Initialization Tests
2. Lazy Loading Tests (_get_separator, _get_model_manager)
3. Audio Separation Tests
4. Voice Conversion Tests
5. Pitch Extraction Tests
6. Technique Detection Tests
7. convert_song E2E Tests
8. Error Handling Tests
9. Edge Case Tests
10. Configuration Tests
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

import numpy as np
import pytest
import torch

from auto_voice.inference.singing_conversion_pipeline import (
    SingingConversionPipeline,
    SeparationError,
    ConversionError,
    PRESETS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create temporary audio file."""
    audio_path = tmp_path / "test_song.wav"

    # Generate 2 seconds of 440Hz sine wave
    import soundfile as sf
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    sf.write(str(audio_path), audio, sr)
    return str(audio_path)


@pytest.fixture
def mock_voice_cloner():
    """Create mock voice cloner."""
    cloner = MagicMock()
    profile = {
        'profile_id': 'test-profile',
        'embedding': np.random.randn(256).astype(np.float32),
        'user_id': 'test-user',
    }
    cloner.load_voice_profile.return_value = profile
    return cloner


@pytest.fixture
def mock_separator():
    """Create mock vocal separator."""
    separator = MagicMock()

    def mock_separate(audio, sr):
        # Return half amplitude for vocals and instrumental
        vocals = audio * 0.5
        instrumental = audio * 0.5
        return {'vocals': vocals, 'instrumental': instrumental}

    separator.separate.side_effect = mock_separate
    return separator


@pytest.fixture
def mock_model_manager():
    """Create mock model manager."""
    manager = MagicMock()

    def mock_infer(vocals, speaker_id, target_embedding, sr):
        # Return slightly modified vocals
        return vocals * 0.8

    manager.infer.side_effect = mock_infer
    return manager


@pytest.fixture
def sample_audio():
    """Generate sample audio array."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32), sr


# ============================================================================
# Initialization Tests
# ============================================================================


def test_init_default():
    """Test pipeline initialization with defaults."""
    pipeline = SingingConversionPipeline()

    assert pipeline.device is not None
    assert pipeline.config == {}
    assert pipeline._separator is None
    assert pipeline._voice_cloner is None
    assert pipeline._sample_rate == 22050
    assert pipeline._model_manager is None


def test_init_with_device():
    """Test pipeline initialization with custom device."""
    device = torch.device('cpu')
    pipeline = SingingConversionPipeline(device=device)

    assert pipeline.device == device


def test_init_with_config():
    """Test pipeline initialization with config."""
    config = {
        'hubert_path': '/path/to/hubert.pt',
        'vocoder_path': '/path/to/vocoder.pt',
        'vocoder_type': 'bigvgan',
    }
    pipeline = SingingConversionPipeline(config=config)

    assert pipeline.config == config


def test_init_with_voice_cloner():
    """Test pipeline initialization with pre-configured voice cloner."""
    cloner = MagicMock()
    pipeline = SingingConversionPipeline(voice_cloner=cloner)

    assert pipeline._voice_cloner is cloner


@pytest.mark.cuda
def test_init_selects_cuda_when_available():
    """Test pipeline selects CUDA device when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pipeline = SingingConversionPipeline()
    assert pipeline.device.type == 'cuda'


# ============================================================================
# Lazy Loading Tests
# ============================================================================


@patch('auto_voice.audio.separation.VocalSeparator')
def test_get_separator_lazy_loads(mock_separator_class):
    """Test _get_separator lazy-loads VocalSeparator."""
    mock_separator = MagicMock()
    mock_separator_class.return_value = mock_separator

    pipeline = SingingConversionPipeline()
    assert pipeline._separator is None

    # First call loads
    result1 = pipeline._get_separator()
    assert result1 is mock_separator
    assert pipeline._separator is mock_separator
    mock_separator_class.assert_called_once()

    # Second call returns cached
    result2 = pipeline._get_separator()
    assert result2 is mock_separator
    assert mock_separator_class.call_count == 1  # Not called again


@patch('auto_voice.inference.model_manager.ModelManager')
def test_get_model_manager_lazy_loads(mock_manager_class):
    """Test _get_model_manager lazy-loads ModelManager."""
    mock_manager = MagicMock()
    mock_manager_class.return_value = mock_manager

    config = {
        'hubert_path': '/path/to/hubert.pt',
        'vocoder_path': '/path/to/vocoder.pt',
    }
    pipeline = SingingConversionPipeline(config=config)
    assert pipeline._model_manager is None

    # First call loads
    result = pipeline._get_model_manager()
    assert result is mock_manager
    assert pipeline._model_manager is mock_manager

    # Verify load was called with config
    mock_manager.load.assert_called_once()
    call_kwargs = mock_manager.load.call_args[1]
    assert call_kwargs['hubert_path'] == '/path/to/hubert.pt'
    assert call_kwargs['vocoder_path'] == '/path/to/vocoder.pt'


@patch('auto_voice.inference.model_manager.ModelManager')
def test_get_model_manager_loads_voice_model_when_configured(mock_manager_class):
    """Test _get_model_manager loads voice model if voice_model_path in config."""
    mock_manager = MagicMock()
    mock_manager_class.return_value = mock_manager

    config = {
        'voice_model_path': '/path/to/voice.pth',
        'speaker_id': 'alice',
    }
    pipeline = SingingConversionPipeline(config=config)

    pipeline._get_model_manager()

    mock_manager.load_voice_model.assert_called_once_with(
        '/path/to/voice.pth', 'alice'
    )


@patch('auto_voice.inference.model_manager.ModelManager')
def test_get_model_manager_uses_default_speaker_id(mock_manager_class):
    """Test _get_model_manager uses 'default' speaker_id when not specified."""
    mock_manager = MagicMock()
    mock_manager_class.return_value = mock_manager

    config = {'voice_model_path': '/path/to/voice.pth'}
    pipeline = SingingConversionPipeline(config=config)

    pipeline._get_model_manager()

    mock_manager.load_voice_model.assert_called_once_with(
        '/path/to/voice.pth', 'default'
    )


@patch('auto_voice.inference.model_manager.ModelManager')
def test_get_model_manager_respects_all_config_params(mock_manager_class):
    """Test _get_model_manager uses all config parameters."""
    mock_manager = MagicMock()
    mock_manager_class.return_value = mock_manager

    config = {
        'hubert_path': '/path/to/hubert.pt',
        'vocoder_path': '/path/to/vocoder.pt',
        'vocoder_type': 'bigvgan',
        'encoder_backend': 'contentvec',
        'encoder_type': 'conformer',
        'conformer_config': {'num_layers': 12},
    }
    pipeline = SingingConversionPipeline(config=config)

    pipeline._get_model_manager()

    call_kwargs = mock_manager.load.call_args[1]
    assert call_kwargs['vocoder_type'] == 'bigvgan'
    assert call_kwargs['encoder_backend'] == 'contentvec'
    assert call_kwargs['encoder_type'] == 'conformer'
    assert call_kwargs['conformer_config'] == {'num_layers': 12}


# ============================================================================
# Audio Separation Tests
# ============================================================================


@patch('auto_voice.audio.separation.VocalSeparator')
def test_separate_vocals_success(mock_separator_class, sample_audio):
    """Test _separate_vocals successfully separates audio."""
    audio, sr = sample_audio

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    pipeline = SingingConversionPipeline()
    result = pipeline._separate_vocals(audio, sr)

    assert 'vocals' in result
    assert 'instrumental' in result
    assert len(result['vocals']) == len(audio)
    mock_separator.separate.assert_called_once_with(audio, sr)


@patch('auto_voice.audio.separation.VocalSeparator')
def test_separate_vocals_raises_on_error(mock_separator_class, sample_audio):
    """Test _separate_vocals raises SeparationError on failure."""
    audio, sr = sample_audio

    mock_separator = MagicMock()
    mock_separator.separate.side_effect = RuntimeError("Demucs failed")
    mock_separator_class.return_value = mock_separator

    pipeline = SingingConversionPipeline()

    with pytest.raises(SeparationError, match="Vocal separation failed.*Demucs failed"):
        pipeline._separate_vocals(audio, sr)


# ============================================================================
# Voice Conversion Tests
# ============================================================================


@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_voice_success(mock_manager_class, sample_audio):
    """Test _convert_voice successfully converts vocals."""
    audio, sr = sample_audio
    target_embedding = np.random.randn(256).astype(np.float32)

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.8
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(config={})
    result = pipeline._convert_voice(audio, target_embedding, sr, speaker_id='default')

    assert len(result) == len(audio)
    mock_manager.infer.assert_called_once_with(audio, 'default', target_embedding, sr)


@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_voice_uses_speaker_id_from_config(mock_manager_class, sample_audio):
    """Test _convert_voice forwards the explicit speaker_id argument."""
    audio, sr = sample_audio
    target_embedding = np.random.randn(256).astype(np.float32)

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.8
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(config={'speaker_id': 'ignored'})
    pipeline._convert_voice(audio, target_embedding, sr, speaker_id='alice')

    call_args = mock_manager.infer.call_args[0]
    assert call_args[1] == 'alice'  # speaker_id


@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_voice_raises_on_runtime_error(mock_manager_class, sample_audio):
    """Test _convert_voice raises ConversionError on RuntimeError."""
    audio, sr = sample_audio
    target_embedding = np.random.randn(256).astype(np.float32)

    mock_manager = MagicMock()
    mock_manager.infer.side_effect = RuntimeError("CUDA OOM")
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(config={})

    with pytest.raises(ConversionError, match="Voice conversion failed.*CUDA OOM"):
        pipeline._convert_voice(audio, target_embedding, sr, speaker_id='default')


# ============================================================================
# Pitch Extraction Tests
# ============================================================================


def test_extract_pitch_success(sample_audio):
    """Test _extract_pitch extracts pitch contour."""
    audio, sr = sample_audio

    pipeline = SingingConversionPipeline()
    f0 = pipeline._extract_pitch(audio, sr)

    assert isinstance(f0, np.ndarray)
    assert len(f0) > 0
    assert not np.any(np.isnan(f0))


def test_extract_pitch_fallback_on_exception():
    """Test _extract_pitch returns zeros on exception."""
    audio = np.array([0.0])  # Too short for librosa.pyin
    sr = 22050

    pipeline = SingingConversionPipeline()
    f0 = pipeline._extract_pitch(audio, sr)

    # Should fallback to zeros
    assert isinstance(f0, np.ndarray)
    assert np.all(f0 == 0)


# ============================================================================
# Technique Detection Tests
# ============================================================================


@patch('auto_voice.audio.technique_detector.TechniqueAwarePitchExtractor')
def test_detect_techniques_success(mock_extractor_class, sample_audio):
    """Test _detect_techniques successfully detects vocal techniques."""
    audio, sr = sample_audio

    mock_extractor = MagicMock()
    mock_flags = MagicMock()
    mock_flags.has_vibrato = True
    mock_flags.has_melisma = False
    mock_flags.vibrato_rate = 5.5
    mock_flags.vibrato_depth_cents = 50.0

    mock_extractor.extract_with_flags.return_value = (np.zeros(100), mock_flags)
    mock_extractor_class.return_value = mock_extractor

    pipeline = SingingConversionPipeline()
    result = pipeline._detect_techniques(audio, sr)

    assert result is not None
    assert result['has_vibrato'] is True
    assert result['has_melisma'] is False
    assert result['vibrato_rate'] == 5.5
    assert result['vibrato_depth_cents'] == 50.0
    assert 'f0' in result
    assert 'technique_flags' in result


@patch('auto_voice.audio.technique_detector.TechniqueAwarePitchExtractor')
def test_detect_techniques_returns_none_on_exception(mock_extractor_class, sample_audio):
    """Test _detect_techniques returns None on exception."""
    audio, sr = sample_audio

    mock_extractor_class.side_effect = ImportError("TechniqueAwarePitchExtractor not found")

    pipeline = SingingConversionPipeline()
    result = pipeline._detect_techniques(audio, sr)

    assert result is None


# ============================================================================
# convert_song E2E Tests
# ============================================================================


@patch('librosa.pyin')
@patch('librosa.effects.pitch_shift')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_success(
    mock_manager_class, mock_separator_class, mock_load, mock_pitch_shift, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song successfully converts a song."""
    audio, sr = sample_audio

    # Mock librosa functions
    mock_load.return_value = (audio, sr)
    mock_pitch_shift.return_value = audio
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    # Mock separator
    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    # Mock model manager
    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(temp_audio_file, 'test-profile')

    assert 'mixed_audio' in result
    assert 'sample_rate' in result
    assert 'duration' in result
    assert 'metadata' in result
    assert 'f0_contour' in result
    assert 'f0_original' in result
    assert result['sample_rate'] == sr
    assert len(result['mixed_audio']) > 0


@patch('librosa.load')
def test_convert_song_file_not_found(mock_load):
    """Test convert_song raises error for non-existent file."""
    pipeline = SingingConversionPipeline()

    with pytest.raises(ConversionError, match="Song file not found"):
        pipeline.convert_song('/nonexistent/path.wav', 'test-profile')


@patch('librosa.load')
def test_convert_song_audio_load_failure(mock_load, temp_audio_file):
    """Test convert_song raises error when audio loading fails."""
    mock_load.side_effect = RuntimeError("Unsupported format")

    pipeline = SingingConversionPipeline()

    with pytest.raises(ConversionError, match="Failed to load audio"):
        pipeline.convert_song(temp_audio_file, 'test-profile')


@patch('librosa.load')
def test_convert_song_empty_audio(mock_load, temp_audio_file):
    """Test convert_song raises error for empty audio."""
    mock_load.return_value = (np.array([]), 22050)

    pipeline = SingingConversionPipeline()

    with pytest.raises(ConversionError, match="Empty audio file"):
        pipeline.convert_song(temp_audio_file, 'test-profile')


@patch('librosa.load')
@patch('auto_voice.inference.voice_cloner.VoiceCloner')
def test_convert_song_profile_not_found(mock_cloner_class, mock_load, temp_audio_file, sample_audio):
    """Test convert_song raises error when profile not found."""
    from auto_voice.storage.voice_profiles import ProfileNotFoundError

    audio, sr = sample_audio
    mock_load.return_value = (audio, sr)

    mock_cloner = MagicMock()
    mock_cloner.load_voice_profile.side_effect = RuntimeError("Profile not found")
    mock_cloner_class.return_value = mock_cloner

    pipeline = SingingConversionPipeline()

    with pytest.raises(ProfileNotFoundError):
        pipeline.convert_song(temp_audio_file, 'nonexistent-profile')


@patch('librosa.load')
def test_convert_song_profile_missing_embedding(mock_load, temp_audio_file, mock_voice_cloner, sample_audio):
    """Test convert_song raises error when profile missing embedding."""
    audio, sr = sample_audio
    mock_load.return_value = (audio, sr)

    # Profile without embedding
    mock_voice_cloner.load_voice_profile.return_value = {'profile_id': 'test'}

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)

    with pytest.raises(ConversionError, match="Profile missing embedding data"):
        pipeline.convert_song(temp_audio_file, 'test-profile')


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_with_stems(
    mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song returns stems when requested."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(temp_audio_file, 'test-profile', return_stems=True)

    assert 'stems' in result
    assert 'vocals' in result['stems']
    assert 'instrumental' in result['stems']


@patch('librosa.pyin')
@patch('librosa.effects.pitch_shift')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_with_pitch_shift(
    mock_manager_class, mock_separator_class, mock_load, mock_pitch_shift, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song applies pitch shift."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pitch_shift.return_value = audio * 1.1
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(temp_audio_file, 'test-profile', pitch_shift=2.0)

    mock_pitch_shift.assert_called_once()
    assert result['metadata']['pitch_shift'] == 2.0


@patch('librosa.pyin')
@patch('librosa.effects.pitch_shift')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_pitch_shift_failure_continues(
    mock_manager_class, mock_separator_class, mock_load, mock_pitch_shift, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song continues when pitch shift fails."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pitch_shift.side_effect = RuntimeError("Pitch shift failed")
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(temp_audio_file, 'test-profile', pitch_shift=2.0)

    # Should complete despite pitch shift failure
    assert 'mixed_audio' in result


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
@patch('auto_voice.audio.technique_detector.TechniqueAwarePitchExtractor')
def test_convert_song_with_technique_detection(
    mock_extractor_class, mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song includes technique detection results."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    mock_extractor = MagicMock()
    mock_flags = MagicMock()
    mock_flags.has_vibrato = True
    mock_flags.has_melisma = False
    mock_flags.vibrato_rate = 5.5
    mock_flags.vibrato_depth_cents = 50.0
    mock_extractor.extract_with_flags.return_value = (np.zeros(100), mock_flags)
    mock_extractor_class.return_value = mock_extractor

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(
        temp_audio_file, 'test-profile', preserve_techniques=True
    )

    assert 'techniques' in result
    assert result['techniques']['has_vibrato'] is True
    assert result['techniques']['vibrato_rate'] == 5.5


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_without_technique_detection(
    mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song skips technique detection when preserve_techniques=False."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(
        temp_audio_file, 'test-profile', preserve_techniques=False
    )

    assert 'techniques' not in result


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_volume_adjustments(
    mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song applies volume adjustments."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': np.ones_like(audio),
        'instrumental': np.ones_like(audio),
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = np.ones_like(audio)
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(
        temp_audio_file, 'test-profile',
        vocal_volume=0.5, instrumental_volume=0.3
    )

    assert result['metadata']['vocal_volume'] == 0.5
    assert result['metadata']['instrumental_volume'] == 0.3


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_normalizes_clipping(
    mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song normalizes audio to prevent clipping."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    # Create stems that would clip when mixed
    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': np.ones_like(audio) * 0.7,
        'instrumental': np.ones_like(audio) * 0.7,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = np.ones_like(audio) * 0.7
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    result = pipeline.convert_song(temp_audio_file, 'test-profile')

    # Check no clipping
    assert np.abs(result['mixed_audio']).max() <= 0.95


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_all_presets(
    mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song works with all preset options."""
    audio, sr = sample_audio

    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)

    for preset in PRESETS.keys():
        result = pipeline.convert_song(temp_audio_file, 'test-profile', preset=preset)
        assert result['metadata']['preset'] == preset


@patch('librosa.pyin')
@patch('librosa.load')
@patch('auto_voice.audio.separation.VocalSeparator')
@patch('auto_voice.inference.model_manager.ModelManager')
def test_convert_song_embedding_list_to_array(
    mock_manager_class, mock_separator_class, mock_load, mock_pyin,
    temp_audio_file, mock_voice_cloner, sample_audio
):
    """Test convert_song converts embedding from list to numpy array."""
    audio, sr = sample_audio
    mock_load.return_value = (audio, sr)
    mock_pyin.return_value = (np.zeros(100), np.ones(100), None)

    # Mock separator
    mock_separator = MagicMock()
    mock_separator.separate.return_value = {
        'vocals': audio * 0.6,
        'instrumental': audio * 0.4,
    }
    mock_separator_class.return_value = mock_separator

    # Mock model manager
    mock_manager = MagicMock()
    mock_manager.infer.return_value = audio * 0.5
    mock_manager_class.return_value = mock_manager

    # Return embedding as list instead of numpy array
    profile = {
        'profile_id': 'test-profile',
        'embedding': [0.1] * 256,  # List instead of np.array
        'user_id': 'test-user',
    }
    mock_voice_cloner.load_voice_profile.return_value = profile

    pipeline = SingingConversionPipeline(voice_cloner=mock_voice_cloner)
    # Should not raise, converts list to array internally
    result = pipeline.convert_song(temp_audio_file, 'test-profile')
    assert result is not None


# ============================================================================
# Preset Tests
# ============================================================================


def test_presets_exist():
    """Test all expected presets exist."""
    expected_presets = ['draft', 'fast', 'balanced', 'high', 'studio']
    for preset in expected_presets:
        assert preset in PRESETS


def test_presets_have_required_keys():
    """Test all presets have required configuration keys."""
    for preset, config in PRESETS.items():
        assert 'n_steps' in config
        assert 'denoise' in config
        assert isinstance(config['n_steps'], int)
        assert isinstance(config['denoise'], float)


def test_presets_ordered_by_quality():
    """Test presets are ordered by increasing quality."""
    preset_order = ['draft', 'fast', 'balanced', 'high', 'studio']
    steps = [PRESETS[p]['n_steps'] for p in preset_order]

    # n_steps should increase with quality
    for i in range(len(steps) - 1):
        assert steps[i] < steps[i + 1]
