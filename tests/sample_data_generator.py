"""Sample training data generator for AutoVoice testing.

Generates synthetic mel-spectrograms and metadata for training pipeline tests.
Supports VoiceDataset, PairedVoiceDataset, and AugmentedVoiceDataset formats.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate synthetic training data for AutoVoice testing."""

    def __init__(
        self,
        output_dir: str = "data/sample_audio",
        sample_rate: int = 22050,
        n_mels: int = 80,
        mel_length: int = 100,
        num_speakers: int = 5
    ):
        """Initialize sample data generator.

        Args:
            output_dir: Output directory for generated data
            sample_rate: Audio sample rate (Hz)
            n_mels: Number of mel-spectrogram frequency bins
            mel_length: Time dimension of mel-spectrograms
            num_speakers: Number of synthetic speakers
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_length = mel_length
        self.num_speakers = num_speakers

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized generator: output_dir={output_dir}")

    def generate_mel_spectrogram(
        self,
        speaker_id: int,
        variation: float = 0.1
    ) -> np.ndarray:
        """Generate synthetic mel-spectrogram with speaker characteristics.

        Args:
            speaker_id: Speaker identifier (0 to num_speakers-1)
            variation: Amount of random variation (0-1)

        Returns:
            Mel-spectrogram array of shape (n_mels, mel_length)
        """
        # Base pattern varies by speaker
        base_freq = 100 + (speaker_id * 50)  # Speaker-specific fundamental
        t = np.linspace(0, 1, self.mel_length)

        # Create speaker-specific pattern
        mel = np.zeros((self.n_mels, self.mel_length))

        for i in range(self.n_mels):
            # Frequency-dependent amplitude
            freq_component = np.sin(2 * np.pi * base_freq * t * (i + 1) / self.n_mels)
            # Speaker-specific envelope
            envelope = np.exp(-t * (1 + speaker_id * 0.1))
            mel[i] = freq_component * envelope

        # Add random variation
        noise = np.random.randn(self.n_mels, self.mel_length) * variation
        mel += noise

        # Normalize to typical mel-spectrogram range
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        mel = mel * 80 - 40  # Scale to dB-like range [-40, 40]

        return mel.astype(np.float32)

    def generate_dataset(
        self,
        num_samples: int,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        save_format: str = 'npy'
    ) -> Dict[str, List[Dict]]:
        """Generate complete dataset with train/val/test splits.

        Args:
            num_samples: Total number of samples to generate
            split_ratios: (train, val, test) split ratios (must sum to 1.0)
            save_format: File format ('npy' or 'pt')

        Returns:
            Dictionary with 'train', 'val', 'test' metadata lists
        """
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

        # Calculate split sizes
        train_size = int(num_samples * split_ratios[0])
        val_size = int(num_samples * split_ratios[1])
        test_size = num_samples - train_size - val_size

        splits = {
            'train': train_size,
            'val': val_size,
            'test': test_size
        }

        metadata = {'train': [], 'val': [], 'test': []}
        sample_id = 0

        for split_name, split_size in splits.items():
            logger.info(f"Generating {split_size} samples for {split_name} split...")

            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)

            for i in range(split_size):
                # Assign speaker (round-robin)
                speaker_id = i % self.num_speakers

                # Generate mel-spectrogram
                mel = self.generate_mel_spectrogram(speaker_id)

                # Save file
                filename = f"sample_{sample_id:05d}_speaker_{speaker_id}.{save_format}"
                filepath = split_dir / filename

                if save_format == 'npy':
                    np.save(filepath, mel)
                elif save_format == 'pt':
                    import torch
                    torch.save(torch.from_numpy(mel), filepath)

                # Create metadata entry
                metadata[split_name].append({
                    'id': sample_id,
                    'file': str(filepath.relative_to(self.output_dir)),
                    'speaker_id': speaker_id,
                    'duration': mel.shape[1] / self.sample_rate,
                    'n_mels': mel.shape[0],
                    'mel_length': mel.shape[1],
                    'text': f"Sample {sample_id} from speaker {speaker_id}"
                })

                sample_id += 1

        logger.info(f"Generated {num_samples} total samples")
        return metadata

    def generate_paired_dataset(
        self,
        num_samples: int,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, List[Dict]]:
        """Generate paired dataset for voice conversion tasks.

        Args:
            num_samples: Total number of sample pairs to generate
            split_ratios: (train, val, test) split ratios

        Returns:
            Dictionary with 'train', 'val', 'test' paired metadata
        """
        # Generate base dataset
        metadata = self.generate_dataset(num_samples, split_ratios)

        # Create pairs for voice conversion
        paired_metadata = {'train': [], 'val': [], 'test': []}

        for split_name, samples in metadata.items():
            pairs_dir = self.output_dir / f"{split_name}_pairs"
            pairs_dir.mkdir(exist_ok=True)

            for i in range(0, len(samples) - 1, 2):
                source = samples[i]
                target_speaker = (source['speaker_id'] + 1) % self.num_speakers

                # Generate target mel-spectrogram
                target_mel = self.generate_mel_spectrogram(target_speaker)

                # Save target file
                target_filename = f"target_{i:05d}_speaker_{target_speaker}.npy"
                target_filepath = pairs_dir / target_filename
                np.save(target_filepath, target_mel)

                # Create paired entry
                paired_metadata[split_name].append({
                    'id': i,
                    'source_file': source['file'],
                    'target_file': str(target_filepath.relative_to(self.output_dir)),
                    'source_speaker': source['speaker_id'],
                    'target_speaker': target_speaker,
                    'text': source['text']
                })

        logger.info(f"Generated {num_samples} paired samples")
        return paired_metadata

    def save_metadata(
        self,
        metadata: Dict[str, List[Dict]],
        prefix: str = "metadata"
    ):
        """Save metadata JSON files for each split.

        Args:
            metadata: Dictionary with 'train', 'val', 'test' metadata
            prefix: Filename prefix
        """
        for split_name, samples in metadata.items():
            filename = f"{prefix}_{split_name}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump({
                    'samples': samples,
                    'num_samples': len(samples),
                    'num_speakers': self.num_speakers,
                    'sample_rate': self.sample_rate,
                    'n_mels': self.n_mels
                }, f, indent=2)

            logger.info(f"Saved metadata: {filepath}")

    def generate_complete_dataset(
        self,
        num_samples: int = 100,
        include_pairs: bool = False
    ):
        """Generate complete dataset with all metadata files.

        Args:
            num_samples: Total number of samples
            include_pairs: Whether to generate paired dataset
        """
        logger.info("=" * 60)
        logger.info("GENERATING COMPLETE SAMPLE DATASET")
        logger.info("=" * 60)

        # Generate standard dataset
        metadata = self.generate_dataset(num_samples)
        self.save_metadata(metadata, prefix="metadata")

        # Generate paired dataset if requested
        if include_pairs:
            paired_metadata = self.generate_paired_dataset(num_samples)
            self.save_metadata(paired_metadata, prefix="paired_metadata")

        # Generate dataset statistics
        stats = {
            'total_samples': num_samples,
            'num_speakers': self.num_speakers,
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'mel_length': self.mel_length,
            'splits': {
                split: len(samples)
                for split, samples in metadata.items()
            }
        }

        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("=" * 60)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total samples: {num_samples}")
        logger.info(f"Speakers: {self.num_speakers}")
        logger.info(f"Mel dimensions: ({self.n_mels}, {self.mel_length})")
        logger.info("=" * 60)


def main():
    """Generate sample datasets for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/sample_audio',
        help='Output directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Total number of samples'
    )
    parser.add_argument(
        '--num-speakers',
        type=int,
        default=5,
        help='Number of speakers'
    )
    parser.add_argument(
        '--n-mels',
        type=int,
        default=80,
        help='Mel-spectrogram frequency bins'
    )
    parser.add_argument(
        '--mel-length',
        type=int,
        default=100,
        help='Mel-spectrogram time dimension'
    )
    parser.add_argument(
        '--include-pairs',
        action='store_true',
        help='Generate paired dataset'
    )

    args = parser.parse_args()

    # Create generator
    generator = SampleDataGenerator(
        output_dir=args.output_dir,
        n_mels=args.n_mels,
        mel_length=args.mel_length,
        num_speakers=args.num_speakers
    )

    # Generate dataset
    generator.generate_complete_dataset(
        num_samples=args.num_samples,
        include_pairs=args.include_pairs
    )


if __name__ == '__main__':
    main()
