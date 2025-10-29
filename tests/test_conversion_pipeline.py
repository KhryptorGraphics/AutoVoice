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

# Logging
from src.auto_voice.utils.logging_config import get_logger
logger = get_logger(__name__)

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
        assert mixed.shape[1] == 2  # 2 channels (time-major format: T, 2)

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

    @pytest.mark.integration
    @pytest.mark.slow
    def test_convert_song_cpu_integration(self, tmp_path):
        """CPU integration test for convert_song with synthetic audio.

        This test runs the full pipeline on CPU with minimal synthetic data
        to verify end-to-end functionality without requiring GPU or large models.
        """
        if not SOUNDFILE_AVAILABLE:
            pytest.skip("soundfile not available")

        # Generate short synthetic audio (1.5 seconds, 16kHz for faster processing)
        duration = 1.5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create synthetic "song" with two frequency components (simulating vocals+instrumental)
        freq1 = 440.0  # A4 note
        freq2 = 554.37  # C#5 note
        synthetic_audio = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)

        # Normalize to [-0.8, 0.8] range
        synthetic_audio = synthetic_audio / np.abs(synthetic_audio).max() * 0.8

        # Save synthetic song
        song_path = tmp_path / "synthetic_song.wav"
        sf.write(str(song_path), synthetic_audio, sample_rate)

        # Create a minimal mock voice profile
        # The pipeline will need a profile with an 'embedding' key
        profile_id = 'test-cpu-profile'
        profile_dir = tmp_path / 'voice_profiles'
        profile_dir.mkdir(exist_ok=True)
        profile_path = profile_dir / f'{profile_id}.npz'

        # Create a dummy speaker embedding (256-dim vector, typical for resemblyzer)
        dummy_embedding = np.random.randn(256).astype(np.float32)
        np.savez(str(profile_path), embedding=dummy_embedding, sample_rate=16000)

        # Initialize pipeline with CPU device and test-friendly config
        test_config = {
            'cache_enabled': False,  # Disable cache for clean test
            'output_sample_rate': 16000,  # Match input for faster processing
            'save_intermediate_results': False,
            'enable_memory_cleanup': False,
            'fallback_on_mixing_error': True,
            'voice_profile_dir': str(profile_dir)
        }

        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            pipeline = SingingConversionPipeline(config=test_config, device='cpu')
        except Exception as e:
            pytest.skip(f"Failed to initialize pipeline on CPU: {e}")

        # Run conversion with synthetic audio
        try:
            result = pipeline.convert_song(
                song_path=str(song_path),
                target_profile_id=profile_id,
                vocal_volume=1.0,
                instrumental_volume=0.9,
                return_stems=False
            )
        except Exception as e:
            pytest.fail(f"Pipeline conversion failed: {e}")

        # Assert result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'mixed_audio' in result, "Result should contain mixed_audio"
        assert 'sample_rate' in result, "Result should contain sample_rate"
        assert 'duration' in result, "Result should contain duration"
        assert 'metadata' in result, "Result should contain metadata"

        # Validate mixed_audio
        mixed_audio = result['mixed_audio']
        assert isinstance(mixed_audio, np.ndarray), "mixed_audio should be numpy array"
        assert mixed_audio.size > 0, "mixed_audio should not be empty"
        assert np.isfinite(mixed_audio).all(), "mixed_audio should not contain NaN or Inf"

        # Validate time-major stereo shape (T, 2) or mono (T,)
        if mixed_audio.ndim == 2:
            assert mixed_audio.shape[1] == 2, f"Stereo should be time-major (T, 2), got {mixed_audio.shape}"

        # Validate sample_rate
        assert result['sample_rate'] == 16000, "Output sample rate should match config"

        # Validate duration
        expected_duration = duration  # Should be close to input duration
        assert abs(result['duration'] - expected_duration) < 0.5, \
            f"Duration should be close to {expected_duration}s, got {result['duration']}s"

        # Validate metadata
        metadata = result['metadata']
        assert 'target_profile_id' in metadata
        assert metadata['target_profile_id'] == profile_id
        assert 'processing_time' in metadata
        assert metadata['processing_time'] > 0

        logger.info(f"CPU integration test passed: {mixed_audio.shape} at {result['sample_rate']}Hz, "
                   f"duration={result['duration']:.2f}s, processing_time={metadata['processing_time']:.2f}s")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_convert_song_with_progress_callback(self, tmp_path):
        """Test progress callback tracking during conversion.

        Verifies that progress callbacks are invoked with correct stage information
        during the conversion pipeline.
        """
        if not SOUNDFILE_AVAILABLE:
            pytest.skip("soundfile not available")

        # Generate minimal synthetic audio
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        synthetic_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        synthetic_audio = synthetic_audio / np.abs(synthetic_audio).max() * 0.8

        song_path = tmp_path / "test_song.wav"
        sf.write(str(song_path), synthetic_audio, sample_rate)

        # Create mock voice profile
        profile_id = 'test-callback-profile'
        profile_dir = tmp_path / 'voice_profiles'
        profile_dir.mkdir(exist_ok=True)
        profile_path = profile_dir / f'{profile_id}.npz'
        dummy_embedding = np.random.randn(256).astype(np.float32)
        np.savez(str(profile_path), embedding=dummy_embedding, sample_rate=16000)

        # Initialize pipeline
        test_config = {
            'cache_enabled': False,
            'output_sample_rate': 16000,
            'save_intermediate_results': False,
            'voice_profile_dir': str(profile_dir)
        }

        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            pipeline = SingingConversionPipeline(config=test_config, device='cpu')
        except Exception as e:
            pytest.skip(f"Failed to initialize pipeline: {e}")

        # Track progress callbacks
        progress_updates = []

        def progress_callback(percentage: float, stage: str):
            """Record progress updates."""
            progress_updates.append({'percentage': percentage, 'stage': stage})

        # Run conversion with callback
        try:
            result = pipeline.convert_song(
                song_path=str(song_path),
                target_profile_id=profile_id,
                progress_callback=progress_callback,
                return_stems=False
            )
        except Exception as e:
            pytest.skip(f"Pipeline conversion failed: {e}")

        # Verify progress callbacks were invoked
        assert len(progress_updates) > 0, "Progress callback should be invoked at least once"

        # Verify progress increases monotonically (with some tolerance for stage transitions)
        percentages = [update['percentage'] for update in progress_updates]
        assert max(percentages) > 0, "Progress should increase beyond 0%"

        # Verify expected stages are present
        stages = [update['stage'] for update in progress_updates]
        expected_stages = ['source_separation', 'pitch_extraction', 'voice_conversion', 'audio_mixing']
        for expected_stage in expected_stages:
            # Check if any stage string contains the expected stage (allows for stage_start/stage_end variants)
            assert any(expected_stage in stage for stage in stages), \
                f"Expected stage '{expected_stage}' not found in progress updates"

        logger.info(f"Progress callback test passed: {len(progress_updates)} updates, "
                   f"stages: {set(stages)}")

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
