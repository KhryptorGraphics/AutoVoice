"""Data utilities for AutoVoice."""

import torch
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_dataset(data_dir: str, metadata_file: str,
                split: str = 'train') -> Dict:
    """Load dataset metadata.

    Args:
        data_dir: Data directory path
        metadata_file: Metadata file path
        split: Dataset split ('train', 'val', 'test')

    Returns:
        Dataset dictionary
    """
    metadata_path = os.path.join(data_dir, metadata_file)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if split not in metadata:
        logger.warning(f"Split '{split}' not found, using all data")
        return metadata

    return metadata[split]


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, path: str,
                   additional_info: Optional[Dict] = None):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Save path
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: torch.device = torch.device('cpu')) -> Dict:
    """Load training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load into
        optimizer: Optional optimizer to load
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f"Checkpoint loaded from {path}")

    return checkpoint


def create_data_splits(data_list: List, train_ratio: float = 0.8,
                      val_ratio: float = 0.1, test_ratio: float = 0.1,
                      seed: int = 42) -> Tuple[List, List, List]:
    """Create train/val/test splits.

    Args:
        data_list: List of data samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Train, validation, and test lists
    """
    np.random.seed(seed)

    # Shuffle data
    indices = np.random.permutation(len(data_list))

    # Calculate split points
    train_end = int(len(data_list) * train_ratio)
    val_end = train_end + int(len(data_list) * val_ratio)

    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create splits
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    logger.info(f"Data splits created: train={len(train_data)}, "
               f"val={len(val_data)}, test={len(test_data)}")

    return train_data, val_data, test_data


def prepare_batch(batch: List[Dict], device: torch.device) -> Dict:
    """Prepare batch for training.

    Args:
        batch: List of samples
        device: Target device

    Returns:
        Batched dictionary
    """
    # Collect all keys
    keys = batch[0].keys()

    # Create batched dictionary
    batched = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # Handle different data types
        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            batched[key] = torch.stack(values).to(device)
        elif isinstance(values[0], (int, float)):
            # Convert to tensor
            batched[key] = torch.tensor(values, device=device)
        elif isinstance(values[0], str):
            # Keep as list
            batched[key] = values
        else:
            # Keep as is
            batched[key] = values

    return batched


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-length sequences.

    Args:
        batch: List of samples

    Returns:
        Collated batch
    """
    # Sort by sequence length (descending)
    batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)

    # Get max length
    max_len = batch[0]['features'].shape[0]

    # Pad sequences
    padded_features = []
    lengths = []

    for sample in batch:
        features = sample['features']
        length = features.shape[0]
        lengths.append(length)

        # Pad if needed
        if length < max_len:
            padding = torch.zeros(max_len - length, features.shape[1])
            features = torch.cat([features, padding], dim=0)

        padded_features.append(features)

    # Stack
    batched = {
        'features': torch.stack(padded_features),
        'lengths': torch.tensor(lengths)
    }

    # Add other fields
    for key in batch[0].keys():
        if key not in ['features', 'lengths']:
            if isinstance(batch[0][key], torch.Tensor):
                batched[key] = torch.stack([s[key] for s in batch])
            else:
                batched[key] = [s[key] for s in batch]

    return batched


def save_audio_samples(waveforms: List[torch.Tensor], sample_rate: int,
                      output_dir: str, prefix: str = 'sample'):
    """Save audio samples to files.

    Args:
        waveforms: List of audio waveforms
        sample_rate: Sample rate
        output_dir: Output directory
        prefix: File name prefix
    """
    import torchaudio

    os.makedirs(output_dir, exist_ok=True)

    for i, waveform in enumerate(waveforms):
        # Ensure proper shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Save
        output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
        torchaudio.save(output_path, waveform.cpu(), sample_rate)

    logger.info(f"Saved {len(waveforms)} audio samples to {output_dir}")


def compute_dataset_statistics(dataset) -> Dict:
    """Compute dataset statistics.

    Args:
        dataset: Dataset to analyze

    Returns:
        Statistics dictionary
    """
    stats = {
        'num_samples': len(dataset),
        'total_duration': 0.0,
        'mean_duration': 0.0,
        'std_duration': 0.0,
        'speakers': set(),
        'emotions': set()
    }

    durations = []

    for i in range(len(dataset)):
        sample = dataset[i]

        # Duration
        if 'waveform' in sample:
            duration = sample['waveform'].shape[-1] / dataset.sample_rate
            durations.append(duration)
            stats['total_duration'] += duration

        # Speaker
        if 'speaker_id' in sample:
            stats['speakers'].add(sample['speaker_id'])

        # Emotion
        if 'emotion' in sample:
            stats['emotions'].add(sample['emotion'])

    # Compute statistics
    if durations:
        stats['mean_duration'] = np.mean(durations)
        stats['std_duration'] = np.std(durations)

    stats['num_speakers'] = len(stats['speakers'])
    stats['num_emotions'] = len(stats['emotions'])

    # Convert sets to lists for JSON serialization
    stats['speakers'] = list(stats['speakers'])
    stats['emotions'] = list(stats['emotions'])

    return stats