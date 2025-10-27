"""
Comprehensive vocal separation tests for AutoVoice.

Tests VocalSeparator with Demucs and Spleeter backends across various scenarios.
"""
import hashlib
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch


@pytest.mark.audio
@pytest.mark.unit
class TestVocalSeparator:
    """Test VocalSeparator from src/auto_voice/audio/source_separator.py"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            self.VocalSeparator = VocalSeparator

            # Initialize separator with test cache directory
            self.cache_dir = tmp_path / "test_cache"
            self.config = {
                'device': 'cpu',
                'cache_enabled': True,
                'cache_dir': str(self.cache_dir),
                'cache_size_limit_gb': 0.1,  # 100MB for tests
                'sample_rate': 44100
            }
            self.separator = VocalSeparator(config=self.config, device='cpu')

        except ImportError as e:
            pytest.skip(f"VocalSeparator not available: {e}")

    @pytest.fixture(autouse=True)
    def mock_separation_methods(self, monkeypatch):
        """Mock heavyweight separation methods to avoid model downloads in unit tests.

        This fixture ensures all unit tests avoid executing Demucs/Spleeter models,
        which are compute-heavy and may download models on first use.
        Uses deterministic outputs with fixed seed for caching tests.
        """
        # Set fixed seed for deterministic outputs
        np.random.seed(42)

        def mock_separate_demucs(audio, progress_callback=None):
            """Mock Demucs separation with deterministic output."""
            # Determine output length based on input
            if isinstance(audio, torch.Tensor):
                length = audio.shape[-1]
            else:
                length = audio.shape[-1] if audio.ndim > 1 else len(audio)

            # Return deterministic arrays with fixed seed
            rng = np.random.RandomState(42)
            vocals = rng.randn(2, length).astype(np.float32) * 0.3
            instrumental = rng.randn(2, length).astype(np.float32) * 0.3
            return vocals, instrumental

        def mock_separate_spleeter(audio, progress_callback=None):
            """Mock Spleeter separation with deterministic output."""
            # Determine output length based on input
            if isinstance(audio, torch.Tensor):
                length = audio.shape[-1]
            else:
                length = audio.shape[-1] if audio.ndim > 1 else len(audio)

            # Return deterministic arrays with fixed seed (different from demucs)
            rng = np.random.RandomState(43)
            vocals = rng.randn(2, length).astype(np.float32) * 0.3
            instrumental = rng.randn(2, length).astype(np.float32) * 0.3
            return vocals, instrumental

        # Apply mocks to separator instance if it exists
        if hasattr(self, 'separator'):
            monkeypatch.setattr(self.separator, '_separate_with_demucs', mock_separate_demucs)
            monkeypatch.setattr(self.separator, '_separate_with_spleeter', mock_separate_spleeter)

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    def test_separator_initialization(self):
        """Verify VocalSeparator initializes with default config."""
        assert self.separator is not None
        assert self.separator.backend in ['demucs', 'spleeter']
        assert self.separator.device == 'cpu'
        assert self.separator.cache_dir.exists()

    def test_separator_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'model': 'htdemucs',
            'cache_enabled': False,
            'sample_rate': 44100,
            'shifts': 2
        }
        separator = self.VocalSeparator(config=custom_config)
        assert separator.config['shifts'] == 2
        assert separator.config['cache_enabled'] is False

    def test_separator_backend_detection(self):
        """Test automatic backend detection and loading."""
        assert self.separator.backend is not None
        assert self.separator.model is not None
        assert hasattr(self.separator, 'audio_processor')

    # ========================================================================
    # Audio Format Tests
    # ========================================================================

    @pytest.fixture
    def synthetic_audio_file(self, tmp_path) -> Path:
        """Create synthetic audio file for testing."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Generate 2 seconds of stereo audio (vocals-like and instrumental-like)
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simulate vocals (higher frequency)
        vocals_freq = 440  # A4 note
        vocals = 0.3 * np.sin(2 * np.pi * vocals_freq * t)

        # Simulate instrumental (lower frequency)
        inst_freq = 220  # A3 note
        instrumental = 0.5 * np.sin(2 * np.pi * inst_freq * t)

        # Mix to stereo
        mixed = np.stack([vocals + instrumental, vocals + instrumental])

        # Save as WAV
        audio_path = tmp_path / "test_song.wav"
        sf.write(audio_path, mixed.T, sample_rate)

        return audio_path

    @pytest.mark.skipif(
        not pytest.importorskip("soundfile", reason="soundfile not available"),
        reason="Requires soundfile"
    )
    @pytest.mark.slow
    @pytest.mark.integration
    def test_separate_vocals_wav(self, synthetic_audio_file):
        """Test vocal separation on WAV file.

        Integration test: Uses mocked separation from autouse fixture (no real model execution).
        """
        vocals, instrumental = self.separator.separate_vocals(str(synthetic_audio_file))

        # Verify output
        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)
        assert vocals.shape == instrumental.shape
        assert vocals.ndim == 2  # Stereo
        assert not np.isnan(vocals).any()
        assert not np.isnan(instrumental).any()
        assert not np.isinf(vocals).any()
        assert not np.isinf(instrumental).any()

    @pytest.mark.parametrize("format_ext,skip_condition", [
        ("wav", False),
        ("flac", False),
        ("mp3", "MP3 encoder may not be available")
    ])
    @pytest.mark.slow
    @pytest.mark.integration
    def test_supported_audio_formats(self, tmp_path, format_ext, skip_condition):
        """Test separation with different audio formats.

        Integration test: Uses mocked separation from autouse fixture (no real model execution).
        """
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Generate test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.stack([audio, audio])

        # Create file in specified format
        audio_path = tmp_path / f"test_audio.{format_ext}"

        try:
            if format_ext == "wav":
                sf.write(audio_path, stereo.T, sample_rate, format='WAV')
            elif format_ext == "flac":
                sf.write(audio_path, stereo.T, sample_rate, format='FLAC')
            elif format_ext == "mp3":
                # MP3 requires special handling, skip if encoder not available
                try:
                    # Try to write MP3 - may fail if encoder not available
                    import subprocess
                    # Use ffmpeg or skip
                    pytest.skip(skip_condition)
                except Exception:
                    pytest.skip(skip_condition)
        except Exception as e:
            if skip_condition:
                pytest.skip(f"{skip_condition}: {e}")
            raise

        # Test separation - mocked separation will be called
        vocals, instrumental = self.separator.separate_vocals(str(audio_path))

        # Validate output
        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)
        assert vocals.ndim == 2  # Stereo
        assert vocals.shape == instrumental.shape
        assert vocals.dtype in [np.float32, np.float64]
        assert not np.isnan(vocals).any()
        assert not np.isinf(vocals).any()

    # ========================================================================
    # Stereo/Mono Handling Tests
    # ========================================================================

    @pytest.mark.integration
    def test_mono_to_stereo_conversion(self, tmp_path):
        """Test mono audio is converted to stereo for Demucs.

        Integration test: Uses mocked separation from autouse fixture (no real model execution).
        """
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create mono audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        mono_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        mono_path = tmp_path / "mono.wav"
        sf.write(mono_path, mono_audio, sample_rate)

        # Separate - mocked separation will be called
        vocals, instrumental = self.separator.separate_vocals(str(mono_path))

        # Should still return valid stereo output
        assert vocals.shape[0] == 2 or vocals.ndim == 1
        assert instrumental.shape[0] == 2 or instrumental.ndim == 1

    @pytest.mark.integration
    def test_stereo_audio_handling(self, synthetic_audio_file):
        """Test stereo audio is processed correctly.

        Integration test: Uses mocked separation from autouse fixture (no real model execution).
        """
        if not synthetic_audio_file.exists():
            pytest.skip("Test audio file not created")

        vocals, instrumental = self.separator.separate_vocals(str(synthetic_audio_file))

        # Verify stereo output
        assert vocals.shape[0] == 2  # 2 channels
        assert instrumental.shape[0] == 2

    # ========================================================================
    # Caching Tests
    # ========================================================================

    def test_cache_hit(self, synthetic_audio_file):
        """Test cache hit returns identical results faster.

        Unit test: Uses mocked separation (no real model execution).
        """
        if not synthetic_audio_file.exists():
            pytest.skip("Test audio file not created")

        # First call (cache miss) - mocked separation will be called
        start1 = time.time()
        vocals1, inst1 = self.separator.separate_vocals(str(synthetic_audio_file), use_cache=True)
        time1 = time.time() - start1

        # Second call (cache hit) - should load from cache
        start2 = time.time()
        vocals2, inst2 = self.separator.separate_vocals(str(synthetic_audio_file), use_cache=True)
        time2 = time.time() - start2

        # Cache hit should be much faster
        assert time2 < time1 * 0.5  # At least 2x faster

        # Results should be identical (deterministic due to fixed seed in mock)
        np.testing.assert_array_equal(vocals1, vocals2)
        np.testing.assert_array_equal(inst1, inst2)

    def test_cache_disabled(self, synthetic_audio_file):
        """Test separation with caching disabled.

        Unit test: Uses mocked separation (no real model execution).
        """
        if not synthetic_audio_file.exists():
            pytest.skip("Test audio file not created")

        # Separate with cache disabled - mocked separation will be called
        vocals, inst = self.separator.separate_vocals(
            str(synthetic_audio_file),
            use_cache=False
        )

        # Verify no cache files created
        cache_files = list(self.cache_dir.glob("*.npy"))
        assert len(cache_files) == 0

    def test_clear_cache(self, synthetic_audio_file):
        """Test cache clearing functionality.

        Unit test: Uses mocked separation (no real model execution).
        """
        if not synthetic_audio_file.exists():
            pytest.skip("Test audio file not created")

        # Populate cache - mocked separation will be called
        self.separator.separate_vocals(str(synthetic_audio_file))

        # Verify cache has files
        cache_files_before = list(self.cache_dir.glob("*.npy"))
        assert len(cache_files_before) > 0

        # Clear cache
        self.separator.clear_cache()

        # Verify cache is empty
        cache_files_after = list(self.cache_dir.glob("*.npy"))
        assert len(cache_files_after) == 0

    def test_cache_size_limit(self, tmp_path):
        """Test cache size limit enforcement with LRU eviction."""
        pytest.skip("Requires large test files for size limit testing")

    def test_get_cache_info(self, synthetic_audio_file):
        """Test cache info retrieval.

        Unit test: Uses mocked separation (no real model execution).
        """
        if not synthetic_audio_file.exists():
            pytest.skip("Test audio file not created")

        # Populate cache - mocked separation will be called
        self.separator.separate_vocals(str(synthetic_audio_file))

        # Get cache info
        info = self.separator.get_cache_info()

        assert 'total_size_mb' in info
        assert 'num_files' in info
        assert 'cache_dir' in info
        assert info['num_files'] > 0
        assert info['total_size_mb'] > 0

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_missing_file_error(self):
        """Test appropriate error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.separator.separate_vocals("/nonexistent/file.wav")

    def test_empty_audio_file(self, tmp_path):
        """Test handling of empty audio file."""
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()

        with pytest.raises(Exception):  # Should raise some error
            self.separator.separate_vocals(str(empty_file))

    def test_corrupted_audio_file(self, tmp_path):
        """Test handling of corrupted audio file."""
        corrupted = tmp_path / "corrupted.wav"
        corrupted.write_bytes(b"not a valid audio file")

        with pytest.raises(Exception):
            self.separator.separate_vocals(str(corrupted))

    # ========================================================================
    # Model Switching Tests
    # ========================================================================

    def test_set_model(self):
        """Test switching between models."""
        pytest.skip("Requires model availability testing")

    # ========================================================================
    # Utility Method Tests
    # ========================================================================

    def test_get_cache_key_consistency(self, tmp_path):
        """Test cache key generation is consistent."""
        # Create a temporary test file
        test_file = tmp_path / "test_audio.mp3"
        test_file.write_text("dummy audio content")

        audio_path = str(test_file)

        key1 = self.separator._get_cache_key(audio_path)
        key2 = self.separator._get_cache_key(audio_path)

        # Same path should produce same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create audio with values exceeding [-1, 1]
        audio = np.array([[-2.0, 1.5], [3.0, -0.5]])

        normalized = self.separator._normalize_audio(audio)

        # Should be scaled to fit within [-1, 1] with headroom
        assert normalized.max() <= 1.0
        assert normalized.min() >= -1.0
        assert np.abs(normalized).max() <= 0.95  # Headroom


@pytest.mark.audio
@pytest.mark.integration
class TestVocalSeparatorIntegration:
    """Integration tests for VocalSeparator with other components"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up integration test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            from src.auto_voice.audio.processor import AudioProcessor

            self.VocalSeparator = VocalSeparator
            self.AudioProcessor = AudioProcessor
            self.tmp_path = tmp_path

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_integration_with_audio_processor(self):
        """Test VocalSeparator integrates with AudioProcessor."""
        separator = self.VocalSeparator(device='cpu')

        # VocalSeparator should use AudioProcessor internally
        assert hasattr(separator, 'audio_processor')
        assert isinstance(separator.audio_processor, self.AudioProcessor)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA"
    )
    @pytest.mark.cuda
    def test_integration_with_gpu_manager(self):
        """Test VocalSeparator with GPUManager."""
        pytest.skip("Requires GPUManager integration")

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow: load → separate → save → verify."""
        pytest.skip("Requires complete workflow implementation")


@pytest.mark.audio
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestVocalSeparatorGPU:
    """GPU-specific tests for VocalSeparator"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up GPU test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator

            self.separator = VocalSeparator(
                config={'cache_enabled': False},
                device='cuda'
            )
        except ImportError:
            pytest.skip("VocalSeparator not available")

    def test_gpu_device_usage(self):
        """Test separator uses GPU when available."""
        assert self.separator.device == 'cuda' or 'cuda' in self.separator.device

    def test_gpu_memory_management(self):
        """Test GPU memory is properly managed."""
        pytest.skip("Requires GPU memory tracking")

    def test_mixed_precision_inference(self):
        """Test mixed precision (FP16) inference works."""
        pytest.skip("Requires audio file and precision testing")


@pytest.mark.audio
@pytest.mark.performance
class TestVocalSeparatorPerformance:
    """Performance tests for VocalSeparator"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up performance test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator

            self.separator = VocalSeparator(device='cpu')
        except ImportError:
            pytest.skip("VocalSeparator not available")

    @pytest.mark.slow
    def test_separation_speed_benchmark(self):
        """Benchmark separation speed."""
        pytest.skip("Requires performance benchmarking setup")

    @pytest.mark.slow
    def test_cache_speedup_measurement(self):
        """Measure cache hit speedup."""
        pytest.skip("Requires cache performance testing")

    def test_memory_usage_tracking(self):
        """Track memory usage during separation."""
        pytest.skip("Requires memory profiling")


# ============================================================================
# Additional Fixtures
# ============================================================================

@pytest.fixture
def vocal_separator(tmp_path):
    """Create VocalSeparator instance with test cache directory."""
    try:
        from src.auto_voice.audio.source_separator import VocalSeparator

        config = {
            'cache_enabled': True,
            'cache_dir': str(tmp_path / "cache"),
            'device': 'cpu'
        }
        return VocalSeparator(config=config)
    except ImportError:
        pytest.skip("VocalSeparator not available")


@pytest.fixture
def sample_song_file(tmp_path) -> Path:
    """Create synthetic 'song' with vocals + instrumental."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not available")

    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Simulate vocals (melody)
    vocals = 0.4 * np.sin(2 * np.pi * 440 * t)  # A4
    vocals += 0.2 * np.sin(2 * np.pi * 554.37 * t)  # C#5

    # Simulate instrumental (bass + drums approximation)
    bass = 0.6 * np.sin(2 * np.pi * 110 * t)  # A2
    drums = 0.3 * np.random.randn(len(t)) * np.sin(2 * np.pi * 2 * t)  # Noisy rhythm

    # Mix
    left = vocals + bass + drums
    right = vocals * 0.8 + bass + drums * 0.7
    stereo = np.stack([left, right])

    # Save
    song_path = tmp_path / "test_song.wav"
    sf.write(song_path, stereo.T, sample_rate)

    return song_path


# ============================================================================
# Fallback Behavior Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestFallbackBehavior:
    """Test Demucs→Spleeter fallback behavior"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator, SeparationError
            self.VocalSeparator = VocalSeparator
            self.SeparationError = SeparationError
            self.tmp_path = tmp_path
        except ImportError as e:
            pytest.skip(f"VocalSeparator not available: {e}")

    def test_demucs_to_spleeter_fallback(self, monkeypatch, tmp_path):
        """Test that Spleeter is invoked when Demucs fails."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.stack([audio, audio])
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, stereo.T, sample_rate)

        # Create separator with fallback enabled
        config = {
            'fallback_enabled': True,
            'cache_enabled': False,
            'backend_priority': ['demucs', 'spleeter']
        }
        separator = self.VocalSeparator(config=config, device='cpu')

        # Mock _separate_with_demucs to raise SeparationError
        original_demucs = separator._separate_with_demucs

        def mock_demucs_fail(*args, **kwargs):
            raise self.SeparationError("Simulated Demucs failure")

        monkeypatch.setattr(separator, '_separate_with_demucs', mock_demucs_fail)

        # Track if Spleeter was called
        spleeter_called = [False]
        original_spleeter = separator._separate_with_spleeter

        def mock_spleeter_track(*args, **kwargs):
            spleeter_called[0] = True
            # Return dummy results
            dummy_audio = np.random.randn(2, 44100)
            return dummy_audio, dummy_audio

        monkeypatch.setattr(separator, '_separate_with_spleeter', mock_spleeter_track)

        # Execute separation - should fall back to Spleeter
        try:
            vocals, instrumental = separator.separate_vocals(str(audio_path))
            # Verify Spleeter was called
            assert spleeter_called[0], "Spleeter fallback was not invoked"
        except Exception as e:
            # If both backends unavailable, that's acceptable
            if "not available" in str(e).lower():
                pytest.skip("Separation backends not available")
            raise

    def test_both_backends_fail(self, monkeypatch, tmp_path):
        """Test that SeparationError is raised when both backends fail."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.stack([audio, audio])
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, stereo.T, sample_rate)

        # Create separator
        config = {'fallback_enabled': True, 'cache_enabled': False}
        separator = self.VocalSeparator(config=config, device='cpu')

        # Mock both backends to fail
        def mock_fail(*args, **kwargs):
            raise self.SeparationError("Simulated failure")

        monkeypatch.setattr(separator, '_separate_with_demucs', mock_fail)
        monkeypatch.setattr(separator, '_separate_with_spleeter', mock_fail)

        # Should raise SeparationError
        with pytest.raises(self.SeparationError):
            separator.separate_vocals(str(audio_path))


# ============================================================================
# GPU and AMP Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestGPUandAMP:
    """Test GPU behavior and AMP usage"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up GPU test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            self.VocalSeparator = VocalSeparator
            self.tmp_path = tmp_path
        except ImportError as e:
            pytest.skip(f"VocalSeparator not available: {e}")

    def test_gpu_device_selection(self):
        """Test that GPU device is correctly selected."""
        separator = self.VocalSeparator(device='cuda')
        assert 'cuda' in separator.device
        assert separator.model is not None

    def test_amp_enabled_on_cuda(self, tmp_path):
        """Test that AMP is enabled only on CUDA devices."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create tiny test audio
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        stereo = np.stack([audio, audio])
        audio_path = tmp_path / "test_tiny.wav"
        sf.write(audio_path, stereo.T, sample_rate)

        # Create separator with mixed_precision enabled
        config = {'mixed_precision': True, 'cache_enabled': False}
        separator = self.VocalSeparator(config=config, device='cuda')

        # Run separation
        vocals, instrumental = separator.separate_vocals(str(audio_path))

        # Verify outputs are valid
        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)
        assert vocals.shape == instrumental.shape
        assert not np.isnan(vocals).any()
        assert not np.isinf(vocals).any()

    def test_amp_disabled_on_cpu(self, tmp_path):
        """Test that AMP is not used on CPU."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create test audio
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        stereo = np.stack([audio, audio])
        audio_path = tmp_path / "test_cpu.wav"
        sf.write(audio_path, stereo.T, sample_rate)

        # Create CPU separator
        config = {'mixed_precision': True, 'cache_enabled': False}
        separator = self.VocalSeparator(config=config, device='cpu')

        # AMP should not crash on CPU (autocast with enabled=False)
        vocals, instrumental = separator.separate_vocals(str(audio_path))

        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)


# ============================================================================
# Multi-Channel Handling Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestMultiChannelHandling:
    """Test multi-channel audio handling"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            self.VocalSeparator = VocalSeparator
            self.tmp_path = tmp_path
        except ImportError as e:
            pytest.skip(f"VocalSeparator not available: {e}")

    def test_multi_channel_downmix(self, tmp_path):
        """Test that multi-channel audio is properly downmixed.

        Note: AudioProcessor downmixes to mono, then VocalSeparator
        converts to stereo for processing.
        """
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create 4-channel test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # 4 channels with different content
        ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
        ch2 = 0.4 * np.sin(2 * np.pi * 554 * t)
        ch3 = 0.3 * np.sin(2 * np.pi * 659 * t)
        ch4 = 0.2 * np.sin(2 * np.pi * 880 * t)

        multi_channel = np.stack([ch1, ch2, ch3, ch4])

        # Try to save as multi-channel (may not be supported by all formats)
        audio_path = tmp_path / "multi_channel.wav"
        try:
            sf.write(audio_path, multi_channel.T, sample_rate)
        except Exception:
            pytest.skip("Multi-channel WAV writing not supported")

        # Create separator
        config = {'cache_enabled': False}
        separator = self.VocalSeparator(config=config, device='cpu')

        # Separate - should handle multi-channel input
        vocals, instrumental = separator.separate_vocals(str(audio_path))

        # Should return stereo output
        assert vocals.shape[0] == 2
        assert instrumental.shape[0] == 2
        assert vocals.ndim == 2
        assert not np.isnan(vocals).any()


# ============================================================================
# Edge-Case Tests for Silent and Noise-Only Inputs
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestEdgeCaseInputs:
    """Test edge cases like silent and noise-only audio"""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures"""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            self.VocalSeparator = VocalSeparator
            self.tmp_path = tmp_path
        except ImportError as e:
            pytest.skip(f"VocalSeparator not available: {e}")

    def test_silent_audio(self, sample_audio_silence, tmp_path):
        """Test separation with completely silent audio.

        Unit test: Uses mocked separation from autouse fixture (no real model execution).
        """
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create silent audio file
        sample_rate = 44100
        silent_stereo = np.stack([sample_audio_silence, sample_audio_silence])

        # Write to file
        audio_path = tmp_path / "silent.wav"
        sf.write(audio_path, silent_stereo.T, sample_rate)

        # Separate - mocked separation will be called
        vocals, instrumental = self.separator.separate_vocals(str(audio_path))

        # Validate outputs
        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)
        assert vocals.shape == instrumental.shape
        assert vocals.ndim == 2
        assert vocals.shape[0] == 2  # Stereo
        assert not np.isnan(vocals).any()
        assert not np.isinf(vocals).any()
        assert not np.isnan(instrumental).any()
        assert not np.isinf(instrumental).any()

    def test_noise_only_audio(self, sample_audio_noise, tmp_path):
        """Test separation with white noise audio.

        Unit test: Uses mocked separation from autouse fixture (no real model execution).
        """
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create noise audio file
        sample_rate = 44100
        # Resample noise to 44.1kHz if needed
        if len(sample_audio_noise) != sample_rate:
            # Simple resampling by repeating or truncating
            noise_44k = np.tile(sample_audio_noise, int(np.ceil(sample_rate / len(sample_audio_noise))))[:sample_rate]
        else:
            noise_44k = sample_audio_noise

        noise_stereo = np.stack([noise_44k, noise_44k])

        # Write to file
        audio_path = tmp_path / "noise.wav"
        sf.write(audio_path, noise_stereo.T, sample_rate)

        # Separate - mocked separation will be called
        vocals, instrumental = self.separator.separate_vocals(str(audio_path))

        # Validate outputs
        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)
        assert vocals.shape == instrumental.shape
        assert vocals.ndim == 2
        assert vocals.shape[0] == 2  # Stereo
        assert not np.isnan(vocals).any()
        assert not np.isinf(vocals).any()
        assert not np.isnan(instrumental).any()
        assert not np.isinf(instrumental).any()
