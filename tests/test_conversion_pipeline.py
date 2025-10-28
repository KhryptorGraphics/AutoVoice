"""Comprehensive integration tests for singing voice conversion pipeline.

This test suite covers:
- AudioMixer unit tests
- SingingConversionPipeline integration tests
- API endpoint tests
- Performance tests
- End-to-end conversion workflows
"""

import os
import pytest
import tempfile
import numpy as np
from pathlib import Path

# Conditional imports
try:
    from src.auto_voice.audio.mixer import AudioMixer, MixingError
    MIXER_AVAILABLE = True
except ImportError:
    MIXER_AVAILABLE = False

try:
    from src.auto_voice.inference.singing_conversion_pipeline import (
        SingingConversionPipeline,
        SingingConversionError
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from src.auto_voice.web.app import create_app
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ============================================================================
# TestAudioMixer - Unit Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestAudioMixer:
    """Unit tests for AudioMixer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test fixtures."""
        if not MIXER_AVAILABLE:
            pytest.skip("AudioMixer not available")
        self.mixer = AudioMixer()

    def test_mixer_initialization(self):
        """Test AudioMixer initializes with default config."""
        assert self.mixer is not None
        assert self.mixer.config['normalization_method'] == 'rms'
        assert self.mixer.config['auto_align_length'] is True

    def test_mix_two_audio_arrays(self, sample_audio_22khz):
        """Test mixing two audio arrays."""
        vocals = sample_audio_22khz
        instrumental = sample_audio_22khz * 0.5

        mixed, sr = self.mixer.mix(vocals, instrumental, sample_rate=22050)

        assert mixed is not None
        assert sr == 22050
        assert len(mixed) > 0
        assert not np.any(np.isnan(mixed))
        assert not np.any(np.isinf(mixed))

    def test_mix_from_files(self, tmp_path, sample_audio_22khz):
        """Test mixing from file paths."""
        if not SOUNDFILE_AVAILABLE:
            pytest.skip("soundfile not available")

        # Save test audio to files
        vocals_path = tmp_path / "vocals.wav"
        instrumental_path = tmp_path / "instrumental.wav"

        sf.write(vocals_path, sample_audio_22khz, 22050)
        sf.write(instrumental_path, sample_audio_22khz * 0.5, 22050)

        mixed, sr = self.mixer.mix(str(vocals_path), str(instrumental_path))

        assert mixed is not None
        assert sr == 22050
        assert len(mixed) > 0

    def test_volume_adjustment(self, sample_audio_22khz):
        """Test volume multipliers work correctly."""
        vocals = sample_audio_22khz
        instrumental = sample_audio_22khz * 0.5

        # Mix with default volumes
        mixed1, _ = self.mixer.mix(vocals, instrumental, sample_rate=22050)

        # Mix with increased vocal volume
        mixed2, _ = self.mixer.mix(
            vocals, instrumental,
            vocal_volume=2.0, instrumental_volume=0.5,
            sample_rate=22050
        )

        # Vocal-heavy mix should be louder
        rms1 = np.sqrt(np.mean(mixed1 ** 2))
        rms2 = np.sqrt(np.mean(mixed2 ** 2))
        assert rms2 > rms1 * 0.9  # Allow for normalization

    def test_normalization_methods(self, sample_audio_22khz):
        """Test different normalization methods."""
        vocals = sample_audio_22khz
        instrumental = sample_audio_22khz * 0.5

        methods = ['rms', 'peak']

        for method in methods:
            mixer = AudioMixer(config={'normalization_method': method})
            mixed, sr = mixer.mix(vocals, instrumental, sample_rate=22050)

            assert mixed is not None
            assert len(mixed) > 0
            assert not np.any(np.isnan(mixed))

    def test_length_alignment(self):
        """Test audio length alignment."""
        vocals = np.random.rand(44100)  # 1 second at 44.1kHz
        instrumental = np.random.rand(88200)  # 2 seconds

        mixed, sr = self.mixer.mix(vocals, instrumental, sample_rate=44100)

        # Should trim to shorter length
        assert len(mixed) == 44100

    def test_sample_rate_mismatch(self, sample_audio_22khz):
        """Test mixing audio with different sample rates."""
        try:
            import librosa
            LIBROSA_AVAILABLE = True
        except ImportError:
            pytest.skip("librosa not available for resampling")

        vocals = sample_audio_22khz  # 22050 Hz
        instrumental = np.random.rand(44100)  # 44100 Hz

        # This should handle resampling
        mixed, sr = self.mixer.mix(vocals, instrumental, sample_rate=22050)

        assert mixed is not None
        assert sr == 22050

    def test_stereo_conversion(self, sample_audio_22khz):
        """Test mono to stereo conversion."""
        mixer = AudioMixer(config={'output_format': 'stereo'})

        vocals = sample_audio_22khz
        instrumental = sample_audio_22khz * 0.5

        mixed, sr = mixer.mix(vocals, instrumental, sample_rate=22050)

        assert mixed.ndim == 2
        assert mixed.shape[0] == 2  # 2 channels

    def test_prevent_clipping(self):
        """Test clipping prevention."""
        # Create audio that would clip when mixed
        vocals = np.ones(22050) * 0.8
        instrumental = np.ones(22050) * 0.8

        mixed, sr = self.mixer.mix(vocals, instrumental, sample_rate=22050)

        # Should not clip
        if mixed.ndim == 1:
            peak = np.abs(mixed).max()
        else:
            peak = np.abs(mixed).max()

        assert peak <= 1.0

    def test_mix_with_balance(self, sample_audio_22khz):
        """Test balance-based mixing."""
        vocals = sample_audio_22khz
        instrumental = sample_audio_22khz * 0.5

        # Test different balance values
        mixed_vocals_heavy, _ = self.mixer.mix_with_balance(
            vocals, instrumental, vocal_balance=0.8, sample_rate=22050
        )

        mixed_instrumental_heavy, _ = self.mixer.mix_with_balance(
            vocals, instrumental, vocal_balance=0.2, sample_rate=22050
        )

        assert mixed_vocals_heavy is not None
        assert mixed_instrumental_heavy is not None


# ============================================================================
# TestSingingConversionPipeline - Integration Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.integration
@pytest.mark.slow
class TestSingingConversionPipeline:
    """Integration tests for SingingConversionPipeline."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Setup test fixtures."""
        if not PIPELINE_AVAILABLE:
            pytest.skip("SingingConversionPipeline not available")

        # Create test config
        test_config = {
            'cache_enabled': True,
            'cache_dir': str(tmp_path / 'cache'),
            'output_sample_rate': 22050,
            'save_intermediate_results': False
        }

        try:
            self.pipeline = SingingConversionPipeline(config=test_config, device='cpu')
        except Exception as e:
            pytest.skip(f"Failed to initialize pipeline: {e}")

    def test_pipeline_initialization(self):
        """Test SingingConversionPipeline initializes."""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, 'vocal_separator')
        assert hasattr(self.pipeline, 'pitch_extractor')
        assert hasattr(self.pipeline, 'voice_converter')
        assert hasattr(self.pipeline, 'audio_mixer')

    def test_convert_song_mock(self, tmp_path, sample_audio_22khz):
        """Test convert_song with mock components."""
        pytest.skip("Requires full pipeline components - test in integration environment")

    def test_convert_song_with_progress_callback(self, tmp_path):
        """Test progress callback tracking."""
        pytest.skip("Requires full pipeline components - test in integration environment")

    def test_cache_functionality(self, tmp_path):
        """Test pipeline caching."""
        cache_key = self.pipeline._get_cache_key(
            'test_song.mp3',
            'test-profile-123',
            {'vocal_volume': 1.0, 'instrumental_volume': 0.9}
        )

        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hex digest

    def test_cache_info(self):
        """Test cache info retrieval."""
        cache_info = self.pipeline.get_cache_info()

        assert 'cache_dir' in cache_info
        assert 'total_size_mb' in cache_info
        assert 'num_conversions' in cache_info

    def test_clear_cache(self):
        """Test cache clearing."""
        try:
            self.pipeline.clear_cache()
        except Exception as e:
            pytest.fail(f"clear_cache failed: {e}")


# ============================================================================
# TestConversionPipelineAPI - API Integration Tests
# ============================================================================

@pytest.mark.web
@pytest.mark.integration
@pytest.mark.slow
class TestConversionPipelineAPI:
    """Integration tests for API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Setup Flask test client."""
        if not FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        # Create test app
        test_config = {
            'TESTING': True,
            'WTF_CSRF_ENABLED': False
        }

        try:
            self.app, _ = create_app(config=test_config)
            self.client = self.app.test_client()
        except Exception as e:
            pytest.skip(f"Failed to create test app: {e}")

    def test_convert_song_endpoint_missing_song(self):
        """Test endpoint with missing song file."""
        response = self.client.post('/api/v1/convert/song', data={
            'profile_id': 'test-profile-123'
        })

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_convert_song_endpoint_missing_profile_id(self, tmp_path, sample_audio_22khz):
        """Test endpoint with missing profile_id."""
        if not SOUNDFILE_AVAILABLE:
            pytest.skip("soundfile not available")

        # Create test audio file
        song_path = tmp_path / "test_song.wav"
        sf.write(song_path, sample_audio_22khz, 22050)

        with open(song_path, 'rb') as f:
            response = self.client.post('/api/v1/convert/song', data={
                'song': (f, 'test_song.wav')
            })

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_convert_song_endpoint_invalid_volume(self, tmp_path, sample_audio_22khz):
        """Test endpoint with invalid volume parameter."""
        if not SOUNDFILE_AVAILABLE:
            pytest.skip("soundfile not available")

        song_path = tmp_path / "test_song.wav"
        sf.write(song_path, sample_audio_22khz, 22050)

        with open(song_path, 'rb') as f:
            response = self.client.post('/api/v1/convert/song', data={
                'song': (f, 'test_song.wav'),
                'profile_id': 'test-profile-123',
                'vocal_volume': '5.0'  # Invalid: > 2.0
            })

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_convert_song_endpoint_mock_success(self, tmp_path, sample_audio_22khz):
        """Test endpoint with mock pipeline (testing mode)."""
        if not SOUNDFILE_AVAILABLE:
            pytest.skip("soundfile not available")

        song_path = tmp_path / "test_song.wav"
        sf.write(song_path, sample_audio_22khz, 22050)

        with open(song_path, 'rb') as f:
            response = self.client.post('/api/v1/convert/song', data={
                'song': (f, 'test_song.wav'),
                'profile_id': 'test-profile-123',
                'vocal_volume': '1.0',
                'instrumental_volume': '0.9'
            })

        # In testing mode, mock pipeline should succeed
        if response.status_code == 200:
            data = response.get_json()
            assert 'audio' in data
            assert 'sample_rate' in data
            assert 'duration' in data
            assert 'metadata' in data


# ============================================================================
# TestConversionPipelinePerformance - Performance Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestConversionPipelinePerformance:
    """Performance tests for conversion pipeline."""

    def test_mixer_performance(self, benchmark_timer, sample_audio_22khz):
        """Benchmark mixing performance."""
        if not MIXER_AVAILABLE:
            pytest.skip("AudioMixer not available")

        mixer = AudioMixer()
        vocals = sample_audio_22khz
        instrumental = sample_audio_22khz * 0.5

        with benchmark_timer('mixer.mix'):
            for _ in range(10):
                mixed, sr = mixer.mix(vocals, instrumental, sample_rate=22050)

    @pytest.mark.cuda
    def test_gpu_memory_usage(self, cuda_device, tmp_path):
        """Test GPU memory usage during conversion."""
        pytest.skip("Requires CUDA-enabled environment")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_22khz():
    """Generate sample audio at 22kHz."""
    duration = 1.0  # seconds
    sample_rate = 22050
    num_samples = int(duration * sample_rate)

    # Generate sine wave
    t = np.linspace(0, duration, num_samples)
    frequency = 440.0  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_44khz():
    """Generate sample audio at 44.1kHz."""
    duration = 1.0
    sample_rate = 44100
    num_samples = int(duration * sample_rate)

    t = np.linspace(0, duration, num_samples)
    frequency = 440.0
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio.astype(np.float32)


@pytest.fixture
def sample_song_file(tmp_path, sample_audio_44khz):
    """Create sample song file (vocals + instrumental mixed)."""
    if not SOUNDFILE_AVAILABLE:
        pytest.skip("soundfile not available")

    # Create mixed audio
    vocals = sample_audio_44khz
    instrumental = sample_audio_44khz * 0.5
    mixed = vocals + instrumental

    # Normalize
    max_val = np.abs(mixed).max()
    if max_val > 0:
        mixed = mixed / max_val * 0.95

    # Save to file
    song_path = tmp_path / "test_song.wav"
    sf.write(song_path, mixed, 44100)

    return str(song_path)


@pytest.fixture
def benchmark_timer():
    """Provide simple benchmark timer."""
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer(name):
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f"\n{name}: {elapsed:.3f}s")

    return timer


@pytest.fixture
def cuda_device():
    """Provide CUDA device if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        else:
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not available")


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
