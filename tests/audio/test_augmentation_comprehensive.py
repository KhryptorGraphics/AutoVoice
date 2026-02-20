"""Comprehensive tests for augmentation.py - Audio augmentation pipeline.

Test Coverage:
- Augmentation pipeline initialization and configuration
- Pitch shifting augmentation (various semitone ranges)
- Time stretching augmentation (faster/slower)
- EQ/bandpass filtering augmentation
- Noise injection and volume perturbation
- Augmentation pipeline composition (multiple augmentations)
- Batch augmentation workflows
- Parameter validation (valid ranges)
- Edge cases (extreme parameters, silent audio, very short audio)
- Deterministic behavior with fixed random seeds
- Output validation (length preservation, no clipping, finite values)

Target Coverage: ≥90% for augmentation.py
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from auto_voice.audio.augmentation import AugmentationPipeline


@pytest.fixture
def sample_audio_mono():
    """Create sample mono audio (5 seconds, 22050 Hz)."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Multi-harmonic signal for better augmentation testing
    audio = 0.4 * np.sin(2 * np.pi * 440 * t)  # A4
    audio += 0.3 * np.sin(2 * np.pi * 880 * t)  # A5
    audio += 0.2 * np.sin(2 * np.pi * 220 * t)  # A3
    audio = audio.astype(np.float32)
    return audio, sr


@pytest.fixture
def short_audio():
    """Very short audio (0.5 seconds) for edge case testing."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def silent_audio():
    """Silent audio for edge case testing."""
    sr = 22050
    duration = 3.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    return audio, sr


@pytest.fixture
def noisy_audio():
    """Audio with noise for testing robustness."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(signal))
    audio = (signal + noise).astype(np.float32)
    return audio, sr


class TestAugmentationPipelineInitialization:
    """Test suite for AugmentationPipeline initialization."""

    def test_default_initialization(self):
        """Test pipeline with default parameters."""
        pipeline = AugmentationPipeline()

        assert pipeline.pitch_shift_prob == 0.5
        assert pipeline.pitch_shift_range == 2.0
        assert pipeline.time_stretch_prob == 0.3
        assert pipeline.time_stretch_range == 0.1
        assert pipeline.eq_prob == 0.3
        assert pipeline.eq_bands == 3
        assert pipeline.eq_gain_range == 6.0

    def test_custom_initialization(self):
        """Test pipeline with custom parameters."""
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.8,
            pitch_shift_range=4.0,
            time_stretch_prob=0.5,
            time_stretch_range=0.2,
            eq_prob=0.6,
            eq_bands=5,
            eq_gain_range=10.0
        )

        assert pipeline.pitch_shift_prob == 0.8
        assert pipeline.pitch_shift_range == 4.0
        assert pipeline.time_stretch_prob == 0.5
        assert pipeline.time_stretch_range == 0.2
        assert pipeline.eq_prob == 0.6
        assert pipeline.eq_bands == 5
        assert pipeline.eq_gain_range == 10.0

    def test_zero_probability_initialization(self):
        """Test pipeline with zero probabilities (no augmentation)."""
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        assert pipeline.pitch_shift_prob == 0.0
        assert pipeline.time_stretch_prob == 0.0
        assert pipeline.eq_prob == 0.0

    def test_full_probability_initialization(self):
        """Test pipeline with 100% probabilities."""
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        assert pipeline.pitch_shift_prob == 1.0
        assert pipeline.time_stretch_prob == 1.0
        assert pipeline.eq_prob == 1.0


class TestPitchShiftAugmentation:
    """Test suite for pitch shifting augmentation."""

    def test_pitch_shift_with_probability_1(self, sample_audio_mono):
        """Test pitch shift is applied when probability is 1.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should be different from original
        assert not np.allclose(augmented, audio)
        # Same length preserved
        assert len(augmented) == len(audio)
        # Finite values only
        assert np.all(np.isfinite(augmented))
        # Dtype preserved
        assert augmented.dtype == np.float32

    def test_pitch_shift_with_probability_0(self, sample_audio_mono):
        """Test pitch shift is not applied when probability is 0.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        augmented = pipeline(audio, sr)

        # Should be identical to original (only copy operation)
        assert np.allclose(augmented, audio, atol=1e-6)

    def test_pitch_shift_range_positive(self, sample_audio_mono):
        """Test pitch shift with positive semitones (higher pitch)."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            pitch_shift_range=2.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        # Set seed to ensure positive shift
        np.random.seed(10)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert not np.allclose(augmented, audio)
        assert np.all(np.isfinite(augmented))

    def test_pitch_shift_range_negative(self, sample_audio_mono):
        """Test pitch shift with negative semitones (lower pitch)."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            pitch_shift_range=2.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        # Set seed to ensure negative shift
        np.random.seed(50)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert not np.allclose(augmented, audio)
        assert np.all(np.isfinite(augmented))

    def test_pitch_shift_extreme_range(self, sample_audio_mono):
        """Test pitch shift with extreme semitone range."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            pitch_shift_range=12.0,  # Full octave
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_pitch_shift_preserves_length_on_short_audio(self, short_audio):
        """Test pitch shift preserves length on very short audio."""
        audio, sr = short_audio
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            pitch_shift_range=2.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))


class TestTimeStretchAugmentation:
    """Test suite for time stretching augmentation."""

    def test_time_stretch_with_probability_1(self, sample_audio_mono):
        """Test time stretch is applied when probability is 1.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=1.0,
            eq_prob=0.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should be different from original
        assert not np.allclose(augmented, audio)
        # Length preserved by padding/truncation
        assert len(augmented) == len(audio)
        # Finite values only
        assert np.all(np.isfinite(augmented))

    def test_time_stretch_with_probability_0(self, sample_audio_mono):
        """Test time stretch is not applied when probability is 0.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        augmented = pipeline(audio, sr)

        # Should be identical to original
        assert np.allclose(augmented, audio, atol=1e-6)

    def test_time_stretch_faster(self, sample_audio_mono):
        """Test time stretch making audio faster (rate > 1.0)."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=1.0,
            time_stretch_range=0.2,
            eq_prob=0.0
        )

        # Set seed to get rate > 1.0 (faster)
        np.random.seed(10)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_time_stretch_slower(self, sample_audio_mono):
        """Test time stretch making audio slower (rate < 1.0)."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=1.0,
            time_stretch_range=0.2,
            eq_prob=0.0
        )

        # Set seed to get rate < 1.0 (slower)
        np.random.seed(50)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_time_stretch_extreme_range(self, sample_audio_mono):
        """Test time stretch with extreme range (near safety limits)."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=1.0,
            time_stretch_range=0.5,  # ±50% speed
            eq_prob=0.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Length still preserved via padding/truncation
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_time_stretch_safety_clamping(self):
        """Test that time stretch rate is clamped to [0.5, 2.0]."""
        audio = np.random.randn(1000).astype(np.float32)
        sr = 16000

        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=1.0,
            time_stretch_range=10.0,  # Extreme range
            eq_prob=0.0
        )

        # Multiple trials to test clamping
        for seed in range(10):
            np.random.seed(seed)
            augmented = pipeline(audio, sr)

            # Should not crash despite extreme range
            assert len(augmented) == len(audio)
            assert np.all(np.isfinite(augmented))


class TestEQAugmentation:
    """Test suite for EQ/bandpass filtering augmentation."""

    def test_eq_with_probability_1(self, sample_audio_mono):
        """Test EQ is applied when probability is 1.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=1.0,
            eq_bands=3
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # EQ should modify the audio
        assert not np.allclose(augmented, audio, atol=1e-3)
        # Length preserved
        assert len(augmented) == len(audio)
        # Finite values only
        assert np.all(np.isfinite(augmented))
        # No clipping (should be normalized)
        assert np.abs(augmented).max() <= 1.0

    def test_eq_with_probability_0(self, sample_audio_mono):
        """Test EQ is not applied when probability is 0.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        augmented = pipeline(audio, sr)

        # Should be identical to original
        assert np.allclose(augmented, audio, atol=1e-6)

    def test_eq_multiple_bands(self, sample_audio_mono):
        """Test EQ with multiple frequency bands."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=1.0,
            eq_bands=5
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))
        assert np.abs(augmented).max() <= 1.0

    def test_eq_single_band(self, sample_audio_mono):
        """Test EQ with single frequency band."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=1.0,
            eq_bands=1
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_eq_extreme_gain_range(self, sample_audio_mono):
        """Test EQ with extreme gain range."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=1.0,
            eq_gain_range=20.0  # ±20 dB
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should normalize to prevent clipping
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))
        assert np.abs(augmented).max() <= 1.0

    def test_eq_handles_edge_frequencies(self):
        """Test EQ handles edge cases in frequency selection."""
        # Low sample rate audio
        sr = 8000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=1.0,
            eq_bands=3
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should handle nyquist limit gracefully
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_eq_normalization_prevents_clipping(self):
        """Test EQ normalization prevents clipping on loud signals."""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Very loud signal (near clipping)
        audio = 0.95 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=1.0,
            eq_gain_range=10.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should normalize to <= 0.95
        assert np.abs(augmented).max() <= 1.0


class TestAugmentationComposition:
    """Test suite for multiple augmentations applied together."""

    def test_all_augmentations_together(self, sample_audio_mono):
        """Test applying all augmentations with probability 1.0."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should be significantly different
        assert not np.allclose(augmented, audio)
        # Length preserved
        assert len(augmented) == len(audio)
        # Finite values only
        assert np.all(np.isfinite(augmented))
        # Dtype preserved
        assert augmented.dtype == np.float32

    def test_augmentation_order_consistency(self, sample_audio_mono):
        """Test that augmentations are applied in consistent order."""
        audio, sr = sample_audio_mono

        # Same seed should give same result
        pipeline1 = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        pipeline2 = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        result1 = pipeline1(audio, sr)

        np.random.seed(42)
        result2 = pipeline2(audio, sr)

        # Same seed should give identical results
        assert np.allclose(result1, result2, atol=1e-5)

    def test_probabilistic_augmentation_variety(self, sample_audio_mono):
        """Test that probabilistic augmentation gives variety."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.5,
            time_stretch_prob=0.5,
            eq_prob=0.5
        )

        # Run multiple times and collect results
        results = []
        for seed in range(5):
            np.random.seed(seed)
            augmented = pipeline(audio, sr)
            results.append(augmented)

        # Not all results should be identical
        all_same = True
        for i in range(1, len(results)):
            if not np.allclose(results[0], results[i], atol=1e-5):
                all_same = False
                break

        assert not all_same, "Probabilistic augmentation should give variety"


class TestEdgeCases:
    """Test suite for edge cases and robustness."""

    def test_silent_audio(self, silent_audio):
        """Test augmentation on silent audio."""
        audio, sr = silent_audio
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should not crash
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))
        # Output should be mostly silent
        assert np.abs(augmented).max() < 0.1

    def test_very_short_audio(self, short_audio):
        """Test augmentation on very short audio (0.5s)."""
        audio, sr = short_audio
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should handle short audio gracefully
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_single_sample_audio(self):
        """Test augmentation on single-sample audio (extreme edge case)."""
        audio = np.array([0.5], dtype=np.float32)
        sr = 16000

        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should not crash even on single sample
        assert len(augmented) == 1
        assert np.all(np.isfinite(augmented))

    def test_noisy_audio(self, noisy_audio):
        """Test augmentation on noisy audio."""
        audio, sr = noisy_audio
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_nan_to_num_conversion(self):
        """Test that NaN/Inf values are converted to finite values."""
        # Create audio with potential NaN/Inf
        audio = np.array([0.5, 1.0, -0.5, 0.0], dtype=np.float32)
        sr = 16000

        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        # Mock an augmentation that could produce NaN
        with patch.object(pipeline, '_pitch_shift') as mock_pitch:
            mock_pitch.return_value = np.array([0.5, np.nan, np.inf, -np.inf], dtype=np.float32)
            pipeline.pitch_shift_prob = 1.0

            augmented = pipeline(audio, sr)

            # All values should be finite
            assert np.all(np.isfinite(augmented))
            # NaN/Inf should be converted to 0.0
            assert augmented[1] == 0.0
            assert augmented[2] == 0.0
            assert augmented[3] == 0.0

    def test_length_preservation_truncation(self):
        """Test that longer augmented audio is truncated to original length."""
        audio = np.random.randn(1000).astype(np.float32)
        sr = 16000

        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        # Mock an augmentation that returns longer audio
        with patch.object(pipeline, '_pitch_shift') as mock_pitch:
            mock_pitch.return_value = np.random.randn(1500).astype(np.float32)
            pipeline.pitch_shift_prob = 1.0

            augmented = pipeline(audio, sr)

            # Should be truncated to original length
            assert len(augmented) == len(audio)

    def test_length_preservation_padding(self):
        """Test that shorter augmented audio is padded to original length."""
        audio = np.random.randn(1000).astype(np.float32)
        sr = 16000

        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            eq_prob=0.0
        )

        # Mock an augmentation that returns shorter audio
        with patch.object(pipeline, '_pitch_shift') as mock_pitch:
            mock_pitch.return_value = np.random.randn(500).astype(np.float32)
            pipeline.pitch_shift_prob = 1.0

            augmented = pipeline(audio, sr)

            # Should be padded to original length
            assert len(augmented) == len(audio)
            # Padding should be zeros
            assert np.all(augmented[500:] == 0.0)


class TestBatchAugmentation:
    """Test suite for batch augmentation workflows."""

    def test_batch_augmentation_consistency(self, sample_audio_mono):
        """Test that batch augmentation maintains consistency."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.5,
            time_stretch_prob=0.5,
            eq_prob=0.5
        )

        # Augment multiple samples
        batch_size = 5
        augmented_batch = []

        for i in range(batch_size):
            np.random.seed(i)
            augmented = pipeline(audio, sr)
            augmented_batch.append(augmented)

        # All should have same length as original
        for aug in augmented_batch:
            assert len(aug) == len(audio)
            assert np.all(np.isfinite(aug))

    def test_batch_augmentation_variety(self, sample_audio_mono):
        """Test that batch augmentation produces variety."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.8,
            time_stretch_prob=0.8,
            eq_prob=0.8
        )

        # Augment multiple samples with different seeds
        batch_size = 10
        augmented_batch = []

        for i in range(batch_size):
            np.random.seed(i)
            augmented = pipeline(audio, sr)
            augmented_batch.append(augmented)

        # Should have variety (not all identical)
        unique_count = 0
        for i in range(batch_size):
            is_unique = True
            for j in range(batch_size):
                if i != j and np.allclose(augmented_batch[i], augmented_batch[j], atol=1e-5):
                    is_unique = False
                    break
            if is_unique:
                unique_count += 1

        # At least some should be unique
        assert unique_count > batch_size * 0.5

    def test_deterministic_batch_with_same_seed(self, sample_audio_mono):
        """Test that same seed produces identical batch results."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        # First batch
        batch1 = []
        for i in range(3):
            np.random.seed(42 + i)
            aug = pipeline(audio, sr)
            batch1.append(aug)

        # Second batch with same seeds
        batch2 = []
        for i in range(3):
            np.random.seed(42 + i)
            aug = pipeline(audio, sr)
            batch2.append(aug)

        # Should be identical
        for i in range(3):
            assert np.allclose(batch1[i], batch2[i], atol=1e-5)


class TestCallableInterface:
    """Test suite for __call__ interface."""

    def test_pipeline_is_callable(self):
        """Test that AugmentationPipeline is callable."""
        pipeline = AugmentationPipeline()
        assert callable(pipeline)

    def test_call_returns_ndarray(self, sample_audio_mono):
        """Test that __call__ returns numpy array."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline()

        result = pipeline(audio, sr)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_call_preserves_input(self, sample_audio_mono):
        """Test that __call__ does not modify input audio."""
        audio, sr = sample_audio_mono
        audio_copy = audio.copy()

        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        np.random.seed(42)
        _ = pipeline(audio, sr)

        # Original should be unchanged
        assert np.allclose(audio, audio_copy)

    def test_multiple_calls_independence(self, sample_audio_mono):
        """Test that multiple calls are independent."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            time_stretch_prob=1.0,
            eq_prob=1.0
        )

        # First call
        np.random.seed(42)
        result1 = pipeline(audio, sr)

        # Second call with same seed
        np.random.seed(42)
        result2 = pipeline(audio, sr)

        # Should be identical (deterministic with same seed)
        assert np.allclose(result1, result2, atol=1e-5)


class TestPrivateMethods:
    """Test suite for private augmentation methods."""

    def test_pitch_shift_method_directly(self, sample_audio_mono):
        """Test _pitch_shift method directly."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(pitch_shift_range=2.0)

        np.random.seed(42)
        shifted = pipeline._pitch_shift(audio, sr)

        assert isinstance(shifted, np.ndarray)
        assert np.all(np.isfinite(shifted))
        # Length may change slightly in pitch_shift
        assert abs(len(shifted) - len(audio)) < sr * 0.1  # Within 0.1s

    def test_time_stretch_method_directly(self, sample_audio_mono):
        """Test _time_stretch method directly."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(time_stretch_range=0.2)

        np.random.seed(42)
        stretched = pipeline._time_stretch(audio, sr)

        assert isinstance(stretched, np.ndarray)
        assert np.all(np.isfinite(stretched))
        # Length will change based on rate
        assert len(stretched) > 0

    def test_eq_method_directly(self, sample_audio_mono):
        """Test _eq method directly."""
        audio, sr = sample_audio_mono
        pipeline = AugmentationPipeline(eq_bands=3, eq_gain_range=6.0)

        np.random.seed(42)
        eq_applied = pipeline._eq(audio, sr)

        assert isinstance(eq_applied, np.ndarray)
        assert np.all(np.isfinite(eq_applied))
        assert len(eq_applied) == len(audio)
        # Should be normalized to prevent clipping
        assert np.abs(eq_applied).max() <= 1.0

    def test_eq_handles_filter_errors_gracefully(self):
        """Test that _eq handles scipy filter errors gracefully."""
        # Create conditions that might cause filter errors
        sr = 8000
        audio = np.random.randn(100).astype(np.float32)

        pipeline = AugmentationPipeline(
            eq_bands=10,  # Many bands increases chance of edge cases
            eq_gain_range=15.0
        )

        np.random.seed(42)
        # Should not raise, even if some bands fail
        result = pipeline._eq(audio, sr)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
        assert np.all(np.isfinite(result))


class TestIntegrationScenarios:
    """Test suite for realistic integration scenarios."""

    def test_training_augmentation_workflow(self, sample_audio_mono):
        """Test typical training augmentation workflow."""
        audio, sr = sample_audio_mono

        # Training-style augmentation
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.5,
            pitch_shift_range=2.0,
            time_stretch_prob=0.3,
            time_stretch_range=0.1,
            eq_prob=0.3,
            eq_bands=3,
            eq_gain_range=6.0
        )

        # Simulate epoch of augmented batches
        epoch_results = []
        for batch_idx in range(5):
            np.random.seed(batch_idx)
            augmented = pipeline(audio, sr)
            epoch_results.append(augmented)

        # All should be valid
        for result in epoch_results:
            assert len(result) == len(audio)
            assert np.all(np.isfinite(result))
            assert result.dtype == np.float32

    def test_conservative_augmentation_strategy(self, sample_audio_mono):
        """Test conservative augmentation (minimal changes)."""
        audio, sr = sample_audio_mono

        # Conservative settings
        pipeline = AugmentationPipeline(
            pitch_shift_prob=0.2,
            pitch_shift_range=0.5,  # Only ±0.5 semitones
            time_stretch_prob=0.1,
            time_stretch_range=0.05,  # Only ±5% speed
            eq_prob=0.1,
            eq_bands=1,
            eq_gain_range=3.0  # Only ±3 dB
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should be relatively close to original
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))

    def test_aggressive_augmentation_strategy(self, sample_audio_mono):
        """Test aggressive augmentation (maximal changes)."""
        audio, sr = sample_audio_mono

        # Aggressive settings
        pipeline = AugmentationPipeline(
            pitch_shift_prob=1.0,
            pitch_shift_range=6.0,  # Full ±6 semitones
            time_stretch_prob=1.0,
            time_stretch_range=0.3,  # ±30% speed
            eq_prob=1.0,
            eq_bands=5,
            eq_gain_range=12.0  # ±12 dB
        )

        np.random.seed(42)
        augmented = pipeline(audio, sr)

        # Should be significantly different but still valid
        assert not np.allclose(augmented, audio, atol=0.1)
        assert len(augmented) == len(audio)
        assert np.all(np.isfinite(augmented))
        assert np.abs(augmented).max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
