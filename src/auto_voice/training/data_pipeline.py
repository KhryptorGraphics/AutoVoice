"""
Data pipeline for voice synthesis training
Includes VoiceDataset, DataLoader, and preprocessing utilities
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import json
import random
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration for audio preprocessing"""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    max_audio_length: int = 8192  # Max frames for consistent batching
    min_audio_length: int = 1024  # Min frames to prevent degenerate cases
    preemphasis: float = 0.97
    ref_level_db: float = 20.0
    min_level_db: float = -100.0

class AudioProcessor:
    """Audio preprocessing utilities for voice synthesis"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
        # Initialize mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax
        )
        
    def load_audio(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio with librosa for consistent preprocessing
            audio, sr = librosa.load(str(filepath), sr=self.config.sample_rate)
            
            # Apply preemphasis filter
            audio = self._preemphasis(audio)
            
            # Normalize audio
            audio = self._normalize_audio(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio {filepath}: {e}")
            return np.zeros(self.config.min_audio_length)
    
    def _preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply preemphasis filter"""
        return np.append(audio[0], audio[1:] - self.config.preemphasis * audio[:-1])
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        return np.clip(audio / np.max(np.abs(audio) + 1e-6), -1.0, 1.0)
    
    def audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel-spectrogram"""
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        # Convert to magnitude
        magnitude = np.abs(stft)
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, magnitude)
        
        # Convert to log scale
        mel_spec = self._amp_to_db(mel_spec)
        
        # Normalize
        mel_spec = self._normalize_mel(mel_spec)
        
        return mel_spec.T  # (time, mel_bins)
    
    def _amp_to_db(self, spec: np.ndarray) -> np.ndarray:
        """Convert amplitude to decibels"""
        return self.config.ref_level_db * np.log10(np.maximum(1e-5, spec))
    
    def _normalize_mel(self, mel_spec: np.ndarray) -> np.ndarray:
        """Normalize mel-spectrogram to [0, 1] range"""
        return np.clip(
            (mel_spec - self.config.min_level_db) / (-self.config.min_level_db),
            0.0, 1.0
        )
    
    def pad_or_trim_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pad or trim audio to consistent length"""
        if len(audio) > self.config.max_audio_length:
            # Random crop for data augmentation during training
            start = random.randint(0, len(audio) - self.config.max_audio_length)
            return audio[start:start + self.config.max_audio_length]
        else:
            # Pad with zeros
            return np.pad(audio, (0, self.config.max_audio_length - len(audio)))

class VoiceDataset(Dataset):
    """Dataset for voice synthesis training"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Optional[str] = None,
        audio_config: Optional[AudioConfig] = None,
        transforms: Optional[List[Callable]] = None,
        cache_size: int = 1000,
        num_workers: int = 4,
        load_in_memory: bool = False
    ):
        """
        Initialize voice dataset
        
        Args:
            data_dir: Directory containing audio files
            metadata_file: JSON file with audio metadata (filepath, speaker_id, text, etc.)
            audio_config: Audio preprocessing configuration
            transforms: List of data augmentation transforms
            cache_size: Number of processed samples to cache
            num_workers: Number of worker threads for parallel processing
            load_in_memory: Whether to preload all data in memory
        """
        self.data_dir = Path(data_dir)
        self.audio_config = audio_config or AudioConfig()
        self.transforms = transforms or []
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.load_in_memory = load_in_memory
        
        # Initialize audio processor
        self.processor = AudioProcessor(self.audio_config)
        
        # Load metadata
        self.samples = self._load_metadata(metadata_file)
        
        # Initialize cache for processed samples
        self._cache = {}
        
        # Preload data if requested
        if self.load_in_memory:
            self._preload_data()
            
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_metadata(self, metadata_file: Optional[str]) -> List[Dict[str, Any]]:
        """Load sample metadata from file or scan directory"""
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata.get('samples', metadata) if isinstance(metadata, dict) else metadata
        else:
            # Scan directory for audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            samples = []
            
            for audio_file in self.data_dir.rglob('*'):
                if audio_file.suffix.lower() in audio_extensions:
                    samples.append({
                        'audio_path': str(audio_file),
                        'speaker_id': audio_file.parent.name,  # Use parent dir as speaker ID
                        'duration': None  # Will be computed on demand
                    })
            
            return samples
    
    def _preload_data(self):
        """Preload all data into memory for faster training"""
        logger.info("Preloading dataset into memory...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_sample, i) for i in range(len(self.samples))]
            
            for i, future in enumerate(futures):
                try:
                    self._cache[i] = future.result()
                    if i % 100 == 0:
                        logger.info(f"Preloaded {i+1}/{len(self.samples)} samples")
                except Exception as e:
                    logger.error(f"Failed to preload sample {i}: {e}")
        
        logger.info("Dataset preloading completed")
    
    def _process_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Process a single sample and return tensors"""
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        
        # Load and process audio
        audio = self.processor.load_audio(audio_path)
        audio = self.processor.pad_or_trim_audio(audio)
        mel_spec = self.processor.audio_to_mel(audio)
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio)
        mel_tensor = torch.FloatTensor(mel_spec)
        
        # Process speaker ID
        speaker_id = sample.get('speaker_id', 'unknown')
        if isinstance(speaker_id, str):
            speaker_tensor = torch.LongTensor([hash(speaker_id) % 1000])  # Simple hash-based ID
        else:
            speaker_tensor = torch.LongTensor([int(speaker_id)])
        
        # Apply transforms
        data = {
            'audio': audio_tensor,
            'mel_spec': mel_tensor,
            'speaker_id': speaker_tensor,
            'original_length': torch.LongTensor([len(audio)])
        }
        
        for transform in self.transforms:
            data = transform(data)
        
        return data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        # Process sample
        data = self._process_sample(idx)
        
        # Cache if space available
        if len(self._cache) < self.cache_size:
            self._cache[idx] = data
        
        return data

class VoiceCollator:
    """Collate function for batching variable-length sequences"""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic padding"""
        # Extract sequences
        audio_seqs = [item['audio'] for item in batch]
        mel_seqs = [item['mel_spec'] for item in batch]
        speaker_ids = [item['speaker_id'] for item in batch]
        lengths = [item['original_length'] for item in batch]
        
        # Pad sequences to max length in batch
        audio_batch = torch.nn.utils.rnn.pad_sequence(
            audio_seqs, batch_first=True, padding_value=self.pad_value
        )
        
        mel_batch = torch.nn.utils.rnn.pad_sequence(
            mel_seqs, batch_first=True, padding_value=self.pad_value
        )
        
        # Stack other tensors
        speaker_batch = torch.stack(speaker_ids)
        length_batch = torch.stack(lengths)
        
        return {
            'audio': audio_batch,
            'mel_spec': mel_batch,
            'speaker_id': speaker_batch,
            'lengths': length_batch,
            'features': mel_batch,  # Alias for compatibility with trainer
            'target_features': audio_batch,  # Target for reconstruction
            'waveform': audio_batch  # Alias for compatibility
        }

class DataAugmentation:
    """Data augmentation transforms for voice synthesis"""
    
    @staticmethod
    def time_stretch(data: Dict[str, torch.Tensor], rate_range: Tuple[float, float] = (0.8, 1.2)) -> Dict[str, torch.Tensor]:
        """Apply random time stretching"""
        rate = random.uniform(*rate_range)
        
        # Stretch audio
        audio = data['audio'].numpy()
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
        data['audio'] = torch.FloatTensor(stretched_audio)
        
        return data
    
    @staticmethod
    def pitch_shift(data: Dict[str, torch.Tensor], semitone_range: Tuple[float, float] = (-2, 2)) -> Dict[str, torch.Tensor]:
        """Apply random pitch shifting"""
        n_steps = random.uniform(*semitone_range)
        
        # Shift audio pitch
        audio = data['audio'].numpy()
        shifted_audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=n_steps)
        data['audio'] = torch.FloatTensor(shifted_audio)
        
        return data
    
    @staticmethod
    def add_noise(data: Dict[str, torch.Tensor], noise_factor: float = 0.02) -> Dict[str, torch.Tensor]:
        """Add random Gaussian noise"""
        noise = torch.randn_like(data['audio']) * noise_factor
        data['audio'] = data['audio'] + noise
        return data
    
    @staticmethod
    def spec_augment(data: Dict[str, torch.Tensor], freq_mask: int = 8, time_mask: int = 10) -> Dict[str, torch.Tensor]:
        """Apply SpecAugment to mel-spectrogram"""
        mel_spec = data['mel_spec']
        
        # Frequency masking
        freq_start = random.randint(0, max(0, mel_spec.size(1) - freq_mask))
        mel_spec[:, freq_start:freq_start + freq_mask] = 0
        
        # Time masking
        time_start = random.randint(0, max(0, mel_spec.size(0) - time_mask))
        mel_spec[time_start:time_start + time_mask, :] = 0
        
        data['mel_spec'] = mel_spec
        return data

def create_voice_dataloader(
    dataset: VoiceDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    drop_last: bool = True
) -> DataLoader:
    """Create DataLoader for voice synthesis training"""
    
    # Use DistributedSampler for multi-GPU training
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling
    
    collate_fn = VoiceCollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )

def create_train_val_datasets(
    data_dir: Union[str, Path],
    val_split: float = 0.1,
    audio_config: Optional[AudioConfig] = None,
    train_transforms: Optional[List[Callable]] = None,
    **dataset_kwargs
) -> Tuple[VoiceDataset, VoiceDataset]:
    """Create training and validation datasets with proper splitting"""
    
    # Default augmentations for training
    if train_transforms is None:
        train_transforms = [
            DataAugmentation.add_noise,
            DataAugmentation.spec_augment
        ]
    
    # Load all samples first
    full_dataset = VoiceDataset(
        data_dir,
        audio_config=audio_config,
        transforms=None,  # No transforms for initial loading
        **dataset_kwargs
    )
    
    # Split indices
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create subset datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    
    # Create training dataset with augmentations
    train_dataset = VoiceDataset(
        data_dir,
        audio_config=audio_config,
        transforms=train_transforms,
        **dataset_kwargs
    )
    train_dataset.samples = train_samples
    
    # Create validation dataset without augmentations
    val_dataset = VoiceDataset(
        data_dir,
        audio_config=audio_config,
        transforms=None,
        **dataset_kwargs
    )
    val_dataset.samples = val_samples
    
    logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_dataset, val_dataset

# Legacy compatibility functions
def create_dataloaders(config: Dict[str, Any], distributed: bool = False) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders from configuration (legacy compatibility)"""
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    
    data_dir = dataset_config.get('data_dir', 'data/audio')
    batch_size = training_config.get('batch_size', 32)
    num_workers = training_config.get('num_workers', 4)
    
    # Create audio config
    audio_config = AudioConfig(
        sample_rate=dataset_config.get('sample_rate', 22050),
        n_mels=dataset_config.get('n_mels', 80),
        max_audio_length=dataset_config.get('max_audio_length', 8192)
    )
    
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        data_dir=data_dir,
        audio_config=audio_config,
        val_split=dataset_config.get('val_split', 0.1)
    )
    
    # Create dataloaders
    dataloaders = {}
    
    if len(train_dataset) > 0:
        dataloaders['train'] = create_voice_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            distributed=distributed
        )
    
    if len(val_dataset) > 0:
        dataloaders['val'] = create_voice_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            distributed=False
        )
    
    return dataloaders

def preprocess_batch(batch: Dict[str, torch.Tensor], device: torch.device, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """Preprocess batch with normalization and device transfer"""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    if normalize and 'features' in batch:
        features = batch['features']
        mean = features.mean(dim=[1, 2], keepdim=True)
        std = features.std(dim=[1, 2], keepdim=True)
        batch['features'] = (features - mean) / (std + 1e-8)
    
    if normalize and 'target_features' in batch:
        target = batch['target_features']
        mean = target.mean(dim=[1, 2], keepdim=True)
        std = target.std(dim=[1, 2], keepdim=True)
        batch['target_features'] = (target - mean) / (std + 1e-8)
    
    return batch

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = AudioConfig(
        sample_rate=22050,
        n_mels=80,
        max_audio_length=8192
    )
    
    # Test audio processing
    processor = AudioProcessor(config)
    
    # Test dataset creation (assuming data directory exists)
    # dataset = VoiceDataset("path/to/data", audio_config=config)
    # dataloader = create_voice_dataloader(dataset, batch_size=8)