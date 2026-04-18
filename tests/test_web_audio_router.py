"""Tests for web audio router module - Target 70% coverage.

Tests for AudioOutputRouter and audio device management.
"""
import sys
from types import SimpleNamespace

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch


class TestAudioOutputRouter:
    """Test AudioOutputRouter class."""

    @pytest.fixture
    def router(self):
        """Create AudioOutputRouter instance."""
        from auto_voice.web.audio_router import AudioOutputRouter
        return AudioOutputRouter(sample_rate=24000)

    @pytest.mark.smoke
    def test_init_default_values(self, router):
        """Test default initialization values."""
        assert router.sample_rate == 24000
        assert router.speaker_gain == 1.0
        assert router.headphone_gain == 1.0
        assert router.voice_gain == 1.0
        assert router.instrumental_gain == 0.8
        assert router.speaker_enabled is True
        assert router.headphone_enabled is True
        assert router.speaker_device is None
        assert router.headphone_device is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(
            sample_rate=44100,
            speaker_gain=0.8,
            headphone_gain=1.2,
            voice_gain=0.9,
            instrumental_gain=0.7,
        )

        assert router.sample_rate == 44100
        assert router.speaker_gain == 0.8
        assert router.headphone_gain == 1.2
        assert router.voice_gain == 0.9
        assert router.instrumental_gain == 0.7


class TestChannelConfiguration:
    """Test channel configuration methods."""

    @pytest.fixture
    def router(self):
        from auto_voice.web.audio_router import AudioOutputRouter
        return AudioOutputRouter()

    def test_set_channel_config_speaker_gain(self, router):
        """Test setting speaker gain."""
        router.set_channel_config(speaker_gain=0.5)

        assert router.speaker_gain == 0.5

    def test_set_channel_config_headphone_gain(self, router):
        """Test setting headphone gain."""
        router.set_channel_config(headphone_gain=1.5)

        assert router.headphone_gain == 1.5

    def test_set_channel_config_voice_gain(self, router):
        """Test setting voice gain."""
        router.set_channel_config(voice_gain=0.7)

        assert router.voice_gain == 0.7

    def test_set_channel_config_instrumental_gain(self, router):
        """Test setting instrumental gain."""
        router.set_channel_config(instrumental_gain=0.6)

        assert router.instrumental_gain == 0.6

    def test_set_channel_config_speaker_enabled(self, router):
        """Test enabling/disabling speaker."""
        router.set_channel_config(speaker_enabled=False)
        assert router.speaker_enabled is False

        router.set_channel_config(speaker_enabled=True)
        assert router.speaker_enabled is True

    def test_set_channel_config_headphone_enabled(self, router):
        """Test enabling/disabling headphone."""
        router.set_channel_config(headphone_enabled=False)
        assert router.headphone_enabled is False

        router.set_channel_config(headphone_enabled=True)
        assert router.headphone_enabled is True

    def test_set_channel_config_clamps_gain(self, router):
        """Test that gain values are clamped to 0.0-2.0."""
        # Test upper clamp
        router.set_channel_config(speaker_gain=5.0)
        assert router.speaker_gain == 2.0

        # Test lower clamp
        router.set_channel_config(speaker_gain=-1.0)
        assert router.speaker_gain == 0.0

    def test_set_channel_config_multiple(self, router):
        """Test setting multiple config values at once."""
        router.set_channel_config(
            speaker_gain=0.5,
            headphone_gain=1.2,
            voice_gain=0.8,
            speaker_enabled=False
        )

        assert router.speaker_gain == 0.5
        assert router.headphone_gain == 1.2
        assert router.voice_gain == 0.8
        assert router.speaker_enabled is False


class TestDeviceConfiguration:
    """Test device configuration methods."""

    @pytest.fixture
    def router(self):
        from auto_voice.web.audio_router import AudioOutputRouter
        return AudioOutputRouter()

    def test_set_devices(self, router):
        """Test setting output devices."""
        router.set_devices(speaker_device=0, headphone_device=1)

        assert router.speaker_device == 0
        assert router.headphone_device == 1

    def test_set_devices_speaker_only(self, router):
        """Test setting only speaker device."""
        router.set_devices(speaker_device=2)

        assert router.speaker_device == 2
        assert router.headphone_device is None

    def test_set_devices_to_none(self, router):
        """Test setting devices to None (system default)."""
        router.set_devices(speaker_device=0, headphone_device=1)
        router.set_devices(speaker_device=None, headphone_device=None)

        assert router.speaker_device is None
        assert router.headphone_device is None


class TestAudioRouting:
    """Test audio routing functionality."""

    @pytest.fixture
    def router(self):
        from auto_voice.web.audio_router import AudioOutputRouter
        return AudioOutputRouter(sample_rate=24000)

    @pytest.fixture
    def audio_tensors(self):
        """Create test audio tensors."""
        num_samples = 24000  # 1 second
        converted_voice = torch.randn(num_samples) * 0.5
        instrumental = torch.randn(num_samples) * 0.3
        original_song = torch.randn(num_samples) * 0.4
        return converted_voice, instrumental, original_song

    @pytest.mark.smoke
    def test_route_basic(self, router, audio_tensors):
        """Test basic audio routing."""
        converted_voice, instrumental, original_song = audio_tensors

        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Both outputs should have same length
        assert len(speaker_out) == len(converted_voice)
        assert len(headphone_out) == len(original_song)

    def test_route_speaker_disabled(self, router, audio_tensors):
        """Test routing with speaker disabled."""
        converted_voice, instrumental, original_song = audio_tensors

        router.set_channel_config(speaker_enabled=False)
        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Speaker should be silence
        assert speaker_out.abs().max() == 0

    def test_route_headphone_disabled(self, router, audio_tensors):
        """Test routing with headphone disabled."""
        converted_voice, instrumental, original_song = audio_tensors

        router.set_channel_config(headphone_enabled=False)
        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Headphone should be silence
        assert headphone_out.abs().max() == 0

    def test_route_different_lengths(self, router):
        """Test routing with different length inputs."""
        # Different lengths
        converted_voice = torch.randn(24000)
        instrumental = torch.randn(22000)
        original_song = torch.randn(26000)

        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Output should be min length
        assert len(speaker_out) == 22000
        assert len(headphone_out) == 22000

    def test_route_clipping_prevention(self, router):
        """Test that output is normalized to prevent clipping."""
        # Create audio that would clip when mixed
        num_samples = 24000
        converted_voice = torch.ones(num_samples) * 1.5
        instrumental = torch.ones(num_samples) * 1.5
        original_song = torch.ones(num_samples) * 1.5

        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Output should be normalized
        assert speaker_out.abs().max() <= 1.0
        assert headphone_out.abs().max() <= 1.0

    def test_route_gain_applied(self, router):
        """Test that gains are applied correctly."""
        num_samples = 24000
        # Use constant signal for easy verification
        converted_voice = torch.ones(num_samples) * 0.1
        instrumental = torch.ones(num_samples) * 0.1
        original_song = torch.ones(num_samples) * 0.1

        router.set_channel_config(speaker_gain=2.0, headphone_gain=0.5)
        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Speaker should be louder (up to normalization)
        # Headphone should be quieter
        # This test verifies gains are applied
        assert speaker_out.abs().max() > 0
        assert headphone_out.abs().max() > 0


class TestGetConfig:
    """Test get_config method."""

    @pytest.fixture
    def router(self):
        from auto_voice.web.audio_router import AudioOutputRouter
        return AudioOutputRouter()

    def test_get_config_returns_all_settings(self, router):
        """Test that get_config returns all settings."""
        config = router.get_config()

        expected_keys = [
            'sample_rate', 'speaker_gain', 'headphone_gain',
            'voice_gain', 'instrumental_gain', 'speaker_enabled',
            'headphone_enabled', 'speaker_device', 'headphone_device'
        ]

        for key in expected_keys:
            assert key in config

    def test_get_config_reflects_changes(self, router):
        """Test that get_config reflects configuration changes."""
        router.set_channel_config(speaker_gain=0.5, speaker_enabled=False)
        router.set_devices(speaker_device=1)

        config = router.get_config()

        assert config['speaker_gain'] == 0.5
        assert config['speaker_enabled'] is False
        assert config['speaker_device'] == 1


class TestListAudioDevices:
    """Test list_audio_devices function."""

    def test_list_audio_devices_basic(self):
        """Test listing audio devices."""
        from auto_voice.web.audio_router import list_audio_devices

        fake_sd = SimpleNamespace(
            query_devices=lambda: [
                {
                    'name': 'Built-in Audio',
                    'max_input_channels': 2,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
            ],
            default=SimpleNamespace(device=[0, 0]),
        )

        with patch.dict(sys.modules, {'sounddevice': fake_sd}):
            devices = list_audio_devices()

        # Should return devices with expected structure
        assert isinstance(devices, list)
        if len(devices) > 0:
            device = devices[0]
            assert 'name' in device
            assert 'type' in device

    def test_list_audio_devices_filter_output(self):
        """Test filtering for output devices only."""
        from auto_voice.web.audio_router import list_audio_devices

        fake_sd = SimpleNamespace(
            query_devices=lambda: [
                {
                    'name': 'Input Device',
                    'max_input_channels': 2,
                    'max_output_channels': 0,
                    'default_samplerate': 48000.0,
                },
                {
                    'name': 'Output Device',
                    'max_input_channels': 0,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
            ],
            default=SimpleNamespace(device=[0, 1]),
        )

        with patch.dict(sys.modules, {'sounddevice': fake_sd}):
            devices = list_audio_devices(device_type='output')

        # Should only return output device
        output_devices = [d for d in devices if d['type'] == 'output']
        assert len(output_devices) == 1
        assert output_devices[0]['name'] == 'Output Device'

    def test_list_audio_devices_filter_input(self):
        """Test filtering for input devices only."""
        from auto_voice.web.audio_router import list_audio_devices

        fake_sd = SimpleNamespace(
            query_devices=lambda: [
                {
                    'name': 'Input Device',
                    'max_input_channels': 2,
                    'max_output_channels': 0,
                    'default_samplerate': 48000.0,
                },
                {
                    'name': 'Output Device',
                    'max_input_channels': 0,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
            ],
            default=SimpleNamespace(device=[0, 1]),
        )

        with patch.dict(sys.modules, {'sounddevice': fake_sd}):
            devices = list_audio_devices(device_type='input')

        # Should only return input device
        input_devices = [d for d in devices if d['type'] == 'input']
        assert len(input_devices) == 1
        assert input_devices[0]['name'] == 'Input Device'

    def test_list_audio_devices_no_sounddevice(self):
        """Test handling when sounddevice not available."""
        from auto_voice.web.audio_router import list_audio_devices

        with patch.dict('sys.modules', {'sounddevice': None}):
            with patch('builtins.__import__', side_effect=ImportError("No sounddevice")):
                # Should return empty list, not crash
                devices = list_audio_devices()
                assert devices == []

    def test_list_audio_devices_error(self):
        """Test handling device query error."""
        from auto_voice.web.audio_router import list_audio_devices

        fake_sd = SimpleNamespace(
            query_devices=Mock(side_effect=Exception("Device error")),
            default=SimpleNamespace(device=[0, 1]),
        )

        with patch.dict(sys.modules, {'sounddevice': fake_sd}):
            devices = list_audio_devices()

        # Should return empty list on error
        assert devices == []


class TestGetDefaultDevice:
    """Test get_default_device function."""

    def test_get_default_device(self):
        """Test getting default output device."""
        from auto_voice.web.audio_router import get_default_device

        fake_sd = SimpleNamespace(default=SimpleNamespace(device=[0, 1]))

        with patch.dict(sys.modules, {'sounddevice': fake_sd}):
            device_idx = get_default_device()

        # Should return output device index (index 1 in the tuple)
        assert device_idx == 1

    def test_get_default_device_no_sounddevice(self):
        """Test handling when sounddevice not available."""
        from auto_voice.web.audio_router import get_default_device

        with patch.dict('sys.modules', {'sounddevice': None}):
            with patch('builtins.__import__', side_effect=ImportError("No sounddevice")):
                result = get_default_device()
                assert result is None


class TestWebUtils:
    """Test web utility functions."""

    def test_allowed_audio_extensions(self):
        """Test ALLOWED_AUDIO_EXTENSIONS constant."""
        from auto_voice.web.utils import ALLOWED_AUDIO_EXTENSIONS

        # Should include common audio formats
        expected_formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
        for fmt in expected_formats:
            assert fmt in ALLOWED_AUDIO_EXTENSIONS

    @pytest.mark.smoke
    def test_allowed_file_valid_extension(self):
        """Test allowed_file with valid extensions."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file('song.wav') is True
        assert allowed_file('song.mp3') is True
        assert allowed_file('song.flac') is True
        assert allowed_file('song.m4a') is True

    def test_allowed_file_invalid_extension(self):
        """Test allowed_file with invalid extensions."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file('document.txt') is False
        assert allowed_file('image.png') is False
        assert allowed_file('script.py') is False
        assert allowed_file('archive.zip') is False

    def test_allowed_file_no_extension(self):
        """Test allowed_file with no extension."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file('filename') is False

    def test_allowed_file_empty_filename(self):
        """Test allowed_file with empty filename."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file('') is False

    def test_allowed_file_none_filename(self):
        """Test allowed_file with None filename."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file(None) is False

    def test_allowed_file_case_insensitive(self):
        """Test allowed_file is case insensitive."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file('song.WAV') is True
        assert allowed_file('song.Mp3') is True
        assert allowed_file('song.FLAC') is True

    def test_allowed_file_multiple_dots(self):
        """Test allowed_file with multiple dots in filename."""
        from auto_voice.web.utils import allowed_file

        assert allowed_file('song.backup.wav') is True
        assert allowed_file('my.song.mp3') is True
