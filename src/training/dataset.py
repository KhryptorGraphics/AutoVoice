"""Dataset classes for voice training."""

import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
import logging

logger = logging.getLogger(__name__)


class VoiceDataset(Dataset):
    """Dataset for voice training."""

    def __init__(self, data_dir: str, metadata_file: str,
                sample_rate: int = 44100, segment_length: int = 16000):
        """Initialize voice dataset.

        Args:
            data_dir: Directory containing audio files
            metadata_file: JSON file with metadata
            sample_rate: Target sample rate
            segment_length: Length of audio segments
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.samples = self.metadata['samples']
        logger.info(f"Loaded dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data
        """
        sample = self.samples[idx]

        # Load audio
        audio_path = os.path.join(self.data_dir, sample['audio_file'])
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Extract segment
        waveform = self._extract_segment(waveform)

        # Extract features
        features = self._extract_features(waveform)

        return {
            'waveform': waveform,
            'features': features,
            'speaker_id': sample.get('speaker_id', 0),
            'text': sample.get('text', ''),
            'emotion': sample.get('emotion', 'neutral')
        }

    def _extract_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract fixed-length segment from waveform.

        Args:
            waveform: Input waveform

        Returns:
            Fixed-length segment
        """
        waveform = waveform.squeeze(0)

        if waveform.shape[0] >= self.segment_length:
            # Random crop
            start = torch.randint(0, waveform.shape[0] - self.segment_length + 1, (1,))
            segment = waveform[start:start + self.segment_length]
        else:
            # Pad if too short
            padding = self.segment_length - waveform.shape[0]
            segment = torch.nn.functional.pad(waveform, (0, padding))

        return segment

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract acoustic features.

        Args:
            waveform: Input waveform

        Returns:
            Acoustic features
        """
        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )

        mel = mel_transform(waveform.unsqueeze(0))
        mel = torch.log(mel + 1e-9)

        return mel.squeeze(0).transpose(0, 1)


class PairedVoiceDataset(VoiceDataset):
    """Dataset for paired voice training (voice conversion)."""

    def __init__(self, *args, paired_metadata_file: str, **kwargs):
        """Initialize paired dataset.

        Args:
            paired_metadata_file: JSON file with source-target pairs
        """
        super().__init__(*args, **kwargs)

        # Load paired metadata
        with open(paired_metadata_file, 'r') as f:
            self.pairs = json.load(f)['pairs']

        logger.info(f"Loaded {len(self.pairs)} voice pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """Get a paired sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with source and target samples
        """
        pair = self.pairs[idx]

        # Load source audio
        source_path = os.path.join(self.data_dir, pair['source_file'])
        source_waveform, source_sr = torchaudio.load(source_path)
        source_waveform = self._preprocess_audio(source_waveform, source_sr)

        # Load target audio
        target_path = os.path.join(self.data_dir, pair['target_file'])
        target_waveform, target_sr = torchaudio.load(target_path)
        target_waveform = self._preprocess_audio(target_waveform, target_sr)

        # Extract features
        source_features = self._extract_features(source_waveform)
        target_features = self._extract_features(target_waveform)

        return {
            'source_waveform': source_waveform,
            'target_waveform': target_waveform,
            'source_features': source_features,
            'target_features': target_features,
            'source_speaker': pair.get('source_speaker', 0),
            'target_speaker': pair.get('target_speaker', 1)
        }

    def _preprocess_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Preprocess audio waveform.

        Args:
            waveform: Input waveform
            sr: Sample rate

        Returns:
            Preprocessed waveform
        """
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Extract segment
        waveform = self._extract_segment(waveform)

        return waveform


class AugmentedVoiceDataset(VoiceDataset):
    """Dataset with data augmentation."""

    def __init__(self, *args, augmentation_prob: float = 0.5, **kwargs):
        """Initialize augmented dataset.

        Args:
            augmentation_prob: Probability of applying augmentation
        """
        super().__init__(*args, **kwargs)
        self.augmentation_prob = augmentation_prob

    def __getitem__(self, idx: int) -> Dict:
        """Get augmented sample.

        Args:
            idx: Sample index

        Returns:
            Augmented sample
        """
        sample = super().__getitem__(idx)

        # Apply augmentation with probability
        if torch.rand(1).item() < self.augmentation_prob:
            sample = self._augment_sample(sample)

        return sample

    def _augment_sample(self, sample: Dict) -> Dict:
        """Apply augmentation to sample.

        Args:
            sample: Original sample

        Returns:
            Augmented sample
        """
        waveform = sample['waveform']

        # Random augmentation selection
        augmentation = torch.randint(0, 4, (1,)).item()

        if augmentation == 0:
            # Add noise
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        elif augmentation == 1:
            # Time stretching
            stretch_rate = 0.8 + torch.rand(1).item() * 0.4
            waveform = self._time_stretch(waveform, stretch_rate)

        elif augmentation == 2:
            # Pitch shift
            semitones = torch.randint(-2, 3, (1,)).item()
            waveform = self._pitch_shift(waveform, semitones)

        elif augmentation == 3:
            # Volume change
            volume_factor = 0.5 + torch.rand(1).item()
            waveform = waveform * volume_factor

        # Update sample
        sample['waveform'] = waveform
        sample['features'] = self._extract_features(waveform)

        return sample

    def _time_stretch(self, waveform: torch.Tensor, rate: float) -> torch.Tensor:
        """Apply time stretching.

        Args:
            waveform: Input waveform
            rate: Stretch rate

        Returns:
            Stretched waveform
        """
        # Simple resampling-based stretching
        original_length = waveform.shape[0]
        stretched_length = int(original_length * rate)

        indices = torch.linspace(0, original_length - 1, stretched_length)
        indices = indices.long().clamp(0, original_length - 1)

        stretched = waveform[indices]

        # Resample back to original length
        if stretched.shape[0] != self.segment_length:
            indices = torch.linspace(0, stretched.shape[0] - 1, self.segment_length)
            indices = indices.long().clamp(0, stretched.shape[0] - 1)
            stretched = stretched[indices]

        return stretched

    def _pitch_shift(self, waveform: torch.Tensor, semitones: int) -> torch.Tensor:
        """Apply pitch shifting.

        Args:
            waveform: Input waveform
            semitones: Number of semitones to shift

        Returns:
            Pitch-shifted waveform
        """
        if semitones == 0:
            return waveform

        # Simple pitch shift using resampling
        factor = 2.0 ** (semitones / 12.0)
        shifted_length = int(waveform.shape[0] / factor)

        indices = torch.linspace(0, waveform.shape[0] - 1, shifted_length)
        indices = indices.long().clamp(0, waveform.shape[0] - 1)

        shifted = waveform[indices]

        # Pad or crop to original length
        if shifted.shape[0] < self.segment_length:
            padding = self.segment_length - shifted.shape[0]
            shifted = torch.nn.functional.pad(shifted, (0, padding))
        else:
            shifted = shifted[:self.segment_length]

        return shifted