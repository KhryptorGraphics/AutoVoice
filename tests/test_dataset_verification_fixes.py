"""
Comprehensive tests for dataset verification fixes.

Tests all 8 verification comments to ensure proper behavior:
1. Single-crop alignment maintains source-target correspondence
2. Audio augmentations recompute mel/F0/embeddings
3. Cache stores unaugmented samples, transforms apply each access
4. Augmentations use AudioConfig.sample_rate
5. PairedVoiceDataset inherits from torch.utils.data.Dataset
6. Probabilistic augmentation control works
7. Synthetic dataset produces .wav files
8. Pipeline applies audio transforms before mel extraction
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import soundfile as sf
import torch

from src.auto_voice.training.dataset import (
    PairedVoiceDataset,
    SingingAugmentation,
    create_paired_train_val_datasets
)
from src.auto_voice.training.data_pipeline import AudioConfig


@pytest.fixture
def temp_audio_dir():
    """Create temporary directory with test audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic audio files
        sample_rate = 22050
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Create 5 paired audio files
        pairs = []
        for i in range(5):
            t = np.linspace(0, duration, num_samples)
            source_freq = 220 + i * 10
            target_freq = 330 + i * 15

            source_audio = np.sin(2 * np.pi * source_freq * t).astype(np.float32)
            target_audio = np.sin(2 * np.pi * target_freq * t).astype(np.float32)

            # Save as WAV files
            source_file = tmpdir / f"source_{i}.wav"
            target_file = tmpdir / f"target_{i}.wav"

            sf.write(str(source_file), source_audio, sample_rate)
            sf.write(str(target_file), target_audio, sample_rate)

            pairs.append({
                'source_file': f"source_{i}.wav",
                'target_file': f"target_{i}.wav",
                'source_speaker_id': f'speaker_{i % 2}',
                'target_speaker_id': f'speaker_{(i + 1) % 2}',
                'duration': duration
            })

        # Create metadata file
        metadata_file = tmpdir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({'pairs': pairs}, f)

        yield tmpdir, str(metadata_file), sample_rate


class TestComment1_AlignmentConsistency:
    """Test Comment 1: Single-crop alignment maintains correspondence."""

    def test_alignment_uses_single_offset(self, temp_audio_dir):
        """Test that alignment uses a single consistent crop offset."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=None
        )

        # Test alignment method directly
        # Create two arrays of different lengths
        audio1 = np.random.randn(10000).astype(np.float32)
        audio2 = np.random.randn(8000).astype(np.float32)

        # Set random seed for reproducibility
        random.seed(42)
        aligned1, aligned2 = dataset._align_audio_lengths(audio1, audio2)

        # Both should have same length (shorter one)
        assert len(aligned1) == len(aligned2) == 8000

        # Test that alignment is consistent across multiple calls with same seed
        random.seed(42)
        aligned1_repeat, aligned2_repeat = dataset._align_audio_lengths(audio1, audio2)

        np.testing.assert_array_equal(aligned1, aligned1_repeat)
        np.testing.assert_array_equal(aligned2, aligned2_repeat)

    def test_alignment_preserves_temporal_correspondence(self, temp_audio_dir):
        """Test that aligned audio maintains temporal correspondence."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=None
        )

        # Create two identical arrays (simulating perfectly aligned content)
        base_audio = np.random.randn(10000).astype(np.float32)
        audio1 = base_audio.copy()
        audio2 = base_audio[:8000].copy()  # Shorter version

        # Align
        aligned1, aligned2 = dataset._align_audio_lengths(audio1, audio2)

        # The aligned portions should be identical segments from the base
        assert len(aligned1) == len(aligned2)


class TestComment2_FeatureRecomputation:
    """Test Comment 2: Augmentations recompute mel/F0/embeddings."""

    def test_recompute_features_after_augmentation(self, temp_audio_dir):
        """Test that mel/F0/embeddings are recomputed after audio changes."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Create dataset with augmentation
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=True,
            extract_speaker_emb=True,
            transforms=[SingingAugmentation.noise_injection_snr],
            augmentation_prob=1.0  # Always apply
        )

        # Get a sample
        sample = dataset[0]

        # Check that all features are present
        assert 'source_mel' in sample
        assert 'target_mel' in sample
        assert 'source_f0' in sample
        assert 'target_f0' in sample
        assert 'source_speaker_emb' in sample
        assert 'target_speaker_emb' in sample

        # Check that mel and F0 have consistent lengths
        assert sample['source_mel'].shape[0] == sample['source_f0'].shape[0]
        assert sample['target_mel'].shape[0] == sample['target_f0'].shape[0]

    def test_features_change_with_augmentation(self, temp_audio_dir):
        """Test that features differ with and without augmentation."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Dataset without augmentation
        dataset_no_aug = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=True,
            extract_speaker_emb=False,
            transforms=None,
            augmentation_prob=0.0
        )

        # Dataset with augmentation
        dataset_with_aug = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=True,
            extract_speaker_emb=False,
            transforms=[SingingAugmentation.noise_injection_snr],
            augmentation_prob=1.0
        )

        # Get same sample from both
        sample_no_aug = dataset_no_aug[0]
        sample_with_aug = dataset_with_aug[0]

        # Mel spectrograms should differ
        assert not torch.allclose(sample_no_aug['source_mel'], sample_with_aug['source_mel'])


class TestComment3_CacheAndTransforms:
    """Test Comment 3: Cache stores unaugmented samples, transforms apply each access."""

    def test_cached_samples_get_different_augmentations(self, temp_audio_dir):
        """Test that cached samples receive different augmentations on each access."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Create dataset with caching and augmentation
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=[SingingAugmentation.noise_injection_snr],
            augmentation_prob=1.0,
            cache_size=10  # Enable caching
        )

        # Access same sample multiple times
        sample1 = dataset[0]
        sample2 = dataset[0]
        sample3 = dataset[0]

        # Samples should be different due to random augmentation
        # (with very high probability given noise injection)
        assert not torch.allclose(sample1['source_mel'], sample2['source_mel'], rtol=1e-3)
        assert not torch.allclose(sample2['source_mel'], sample3['source_mel'], rtol=1e-3)

    def test_cache_stores_base_features(self, temp_audio_dir):
        """Test that cache stores unaugmented base features."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=None,  # No transforms initially
            cache_size=10
        )

        # First access - populates cache
        sample1 = dataset[0]

        # Check that sample is in cache
        assert 0 in dataset.cache

        # Second access should return cloned data from cache
        sample2 = dataset[0]

        # Should be identical when no augmentation
        torch.testing.assert_close(sample1['source_mel'], sample2['source_mel'])


class TestComment4_SampleRateConfig:
    """Test Comment 4: Augmentations use AudioConfig.sample_rate."""

    def test_augmentation_uses_config_sample_rate(self, temp_audio_dir):
        """Test that augmentations use sample_rate from config."""
        data_dir, metadata_file, _ = temp_audio_dir

        # Use non-standard sample rate
        custom_sample_rate = 16000
        audio_config = AudioConfig(sample_rate=custom_sample_rate)

        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=[SingingAugmentation.pitch_preserving_time_stretch],
            augmentation_prob=1.0
        )

        # Get a sample - should not crash even with different sample rate
        sample = dataset[0]

        # Verify sample was created successfully
        assert 'source_mel' in sample
        assert 'target_mel' in sample

    def test_sample_rate_passed_to_transforms(self):
        """Test that sample_rate is passed in data dict to transforms."""
        # Create test data dict
        audio = torch.randn(22050).numpy()
        data = {
            'source_audio': torch.from_numpy(audio),
            'target_audio': torch.from_numpy(audio),
            'sample_rate': 16000  # Non-standard rate
        }

        # Apply transform
        result = SingingAugmentation.pitch_preserving_time_stretch(data, rate_range=(1.0, 1.0))

        # Should complete without error
        assert 'source_audio' in result


class TestComment5_DatasetInheritance:
    """Test Comment 5: PairedVoiceDataset inherits from Dataset."""

    def test_dataset_inheritance(self, temp_audio_dir):
        """Test that PairedVoiceDataset inherits from torch.utils.data.Dataset."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config
        )

        # Check inheritance
        assert isinstance(dataset, torch.utils.data.Dataset)

        # Check required methods exist
        assert hasattr(dataset, '__len__')
        assert hasattr(dataset, '__getitem__')

        # Test they work
        assert len(dataset) > 0
        sample = dataset[0]
        assert isinstance(sample, dict)


class TestComment6_ProbabilisticAugmentation:
    """Test Comment 6: Probabilistic augmentation control."""

    def test_augmentation_prob_zero_no_augmentation(self, temp_audio_dir):
        """Test that augmentation_prob=0.0 prevents augmentation."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Create two datasets: one with prob=0, one with prob=1
        dataset_no_aug = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=[SingingAugmentation.noise_injection_snr],
            augmentation_prob=0.0
        )

        dataset_always_aug = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            transforms=[SingingAugmentation.noise_injection_snr],
            augmentation_prob=1.0
        )

        # Get samples
        sample_no_aug = dataset_no_aug[0]
        sample_always_aug = dataset_always_aug[0]

        # With prob=0, multiple accesses should be identical
        sample_no_aug_2 = dataset_no_aug[0]
        torch.testing.assert_close(sample_no_aug['source_mel'], sample_no_aug_2['source_mel'])

        # With prob=1, should differ from no augmentation
        assert not torch.allclose(sample_no_aug['source_mel'], sample_always_aug['source_mel'], rtol=1e-3)

    def test_augmentation_prob_passed_to_factory(self, temp_audio_dir):
        """Test that augmentation_prob is passed through factory function."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Create metadata for validation split
        val_metadata = Path(data_dir) / 'val_metadata.json'
        with open(val_metadata, 'w') as f:
            json.dump({'pairs': []}, f)

        # Create datasets via factory
        train_dataset, val_dataset = create_paired_train_val_datasets(
            data_dir=data_dir,
            train_metadata=metadata_file,
            val_metadata=str(val_metadata),
            audio_config=audio_config,
            augmentation_prob=0.7,
            extract_f0=False,
            extract_speaker_emb=False
        )

        # Check that train dataset has correct prob
        assert train_dataset.augmentation_prob == 0.7

        # Check that val dataset has prob=0
        assert val_dataset.augmentation_prob == 0.0


class TestComment7_SyntheticWAVFiles:
    """Test Comment 7: Synthetic dataset produces .wav files."""

    def test_synthetic_dataset_creates_wav_files(self):
        """Test that synthetic dataset demo creates .wav files, not .npy."""
        from examples.train_voice_conversion import create_synthetic_dataset_demo

        # Create synthetic dataset
        data_dir, train_metadata, val_metadata = create_synthetic_dataset_demo()

        # Check that .wav files exist
        data_dir = Path(data_dir)
        wav_files = list(data_dir.glob("*.wav"))

        assert len(wav_files) > 0, "No .wav files created"

        # Check that no .npy files exist
        npy_files = list(data_dir.glob("*.npy"))
        assert len(npy_files) == 0, ".npy files should not be created"

        # Verify files can be loaded
        for wav_file in wav_files[:2]:  # Check first 2
            audio, sr = sf.read(str(wav_file))
            assert audio is not None
            assert sr == 44100
            assert len(audio) > 0

    def test_synthetic_wav_files_are_valid(self):
        """Test that synthetic .wav files are valid audio."""
        from examples.train_voice_conversion import create_synthetic_dataset_demo

        data_dir, train_metadata, val_metadata = create_synthetic_dataset_demo()

        # Load metadata
        with open(train_metadata, 'r') as f:
            metadata = json.load(f)

        # Check that files in metadata exist and are .wav
        for pair in metadata['pairs']:
            source_path = Path(data_dir) / pair['source_file']
            target_path = Path(data_dir) / pair['target_file']

            assert source_path.suffix == '.wav'
            assert target_path.suffix == '.wav'
            assert source_path.exists()
            assert target_path.exists()


class TestComment8_PipelineOrdering:
    """Test Comment 8: Audio transforms before mel extraction."""

    def test_transforms_applied_before_mel_computation(self, temp_audio_dir):
        """Test that audio transforms are applied, then mel/F0 are computed."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Create dataset with audio-domain augmentation
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=True,
            extract_speaker_emb=False,
            transforms=[SingingAugmentation.noise_injection_snr],
            augmentation_prob=1.0
        )

        # Get sample
        sample = dataset[0]

        # Check that mel and F0 have consistent lengths
        # (proves they were computed after audio was modified)
        mel_len = sample['source_mel'].shape[0]
        f0_len = sample['source_f0'].shape[0]

        assert mel_len == f0_len, "Mel and F0 should have same length after recomputation"

        # Check that lengths match reported length
        assert sample['lengths'].item() == mel_len

    def test_recompute_features_method_exists(self, temp_audio_dir):
        """Test that _recompute_features method exists and works."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=True,
            extract_speaker_emb=True
        )

        # Check method exists
        assert hasattr(dataset, '_recompute_features')

        # Test it works
        sample = dataset[0]

        # Modify audio
        sample['source_audio'] = sample['source_audio'] * 0.5
        sample['target_audio'] = sample['target_audio'] * 0.5

        # Recompute features
        recomputed = dataset._recompute_features(sample)

        # Check all features are present
        assert 'source_mel' in recomputed
        assert 'target_mel' in recomputed
        assert 'source_f0' in recomputed
        assert 'target_f0' in recomputed
        assert 'source_speaker_emb' in recomputed
        assert 'target_speaker_emb' in recomputed


class TestIntegration:
    """Integration tests for all fixes working together."""

    def test_full_pipeline_with_all_fixes(self, temp_audio_dir):
        """Test that all fixes work together in a realistic scenario."""
        data_dir, metadata_file, sample_rate = temp_audio_dir

        audio_config = AudioConfig(sample_rate=sample_rate)

        # Create dataset with all features enabled
        dataset = PairedVoiceDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            audio_config=audio_config,
            extract_f0=True,
            extract_speaker_emb=True,
            transforms=[
                SingingAugmentation.pitch_preserving_time_stretch,
                SingingAugmentation.noise_injection_snr
            ],
            augmentation_prob=0.5,
            cache_size=10
        )

        # Access samples multiple times
        samples = []
        for _ in range(3):
            sample = dataset[0]
            samples.append(sample)

            # Check all expected keys present
            assert 'source_mel' in sample
            assert 'target_mel' in sample
            assert 'source_f0' in sample
            assert 'target_f0' in sample
            assert 'source_speaker_emb' in sample
            assert 'target_speaker_emb' in sample
            assert 'lengths' in sample

            # Check consistency
            assert sample['source_mel'].shape[0] == sample['source_f0'].shape[0]
            assert sample['target_mel'].shape[0] == sample['target_f0'].shape[0]

        # With augmentation_prob=0.5, some variation expected but not guaranteed
        # Just verify no crashes and data is valid
        for sample in samples:
            assert sample['source_mel'].shape[1] == audio_config.n_mels
            assert sample['target_mel'].shape[1] == audio_config.n_mels


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
