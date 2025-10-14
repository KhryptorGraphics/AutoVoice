"""Data utilities for batching, collation, and preprocessing."""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from collections import defaultdict
import random
import logging

logger = logging.getLogger(__name__)


class DataCollator:
    """Base class for data collation functions."""
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of samples."""
        raise NotImplementedError


class AudioCollator(DataCollator):
    """Collator for audio data with padding and batching."""
    
    def __init__(
        self,
        padding: str = "max_length",
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
        pad_value: float = 0.0
    ):
        """
        Initialize audio collator.
        
        Args:
            padding: Padding strategy ("max_length", "longest", "none")
            max_length: Maximum sequence length for padding
            return_tensors: Format of returned tensors ("pt", "np")
            pad_value: Value to use for padding
        """
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate audio batch with padding."""
        if not batch:
            raise ValueError("Empty batch provided")
        
        # Determine batch size and feature dimensions
        batch_size = len(batch)
        
        # Extract all keys from the first sample
        keys = set(batch[0].keys())
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch if key in item]
            
            if not values:
                continue
                
            # Handle different data types
            if key in ['audio', 'mel_spectrogram', 'features']:
                collated[key] = self._collate_sequences(values)
            elif key in ['labels', 'targets']:
                collated[key] = self._collate_labels(values)
            elif key in ['length', 'duration']:
                collated[key] = self._collate_lengths(values)
            else:
                # Default: stack if numeric, otherwise keep as list
                try:
                    if isinstance(values[0], (int, float)):
                        collated[key] = self._to_tensor([values])
                    elif isinstance(values[0], (list, np.ndarray, torch.Tensor)):
                        collated[key] = self._collate_sequences(values)
                    else:
                        collated[key] = values
                except Exception as e:
                    logger.warning(f"Could not collate key '{key}': {e}")
                    collated[key] = values
        
        return collated
    
    def _collate_sequences(self, sequences: List[Union[np.ndarray, torch.Tensor, List]]) -> Union[torch.Tensor, np.ndarray]:
        """Collate sequence data with padding."""
        if not sequences:
            return self._to_tensor([])
        
        # Convert to numpy arrays if needed
        arrays = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.detach().cpu().numpy()
            elif isinstance(seq, list):
                seq = np.array(seq)
            arrays.append(seq)
        
        # Determine target length
        if self.padding == "max_length" and self.max_length:
            target_length = self.max_length
        elif self.padding == "longest":
            target_length = max(len(arr) for arr in arrays)
        else:
            # No padding - return as is (may cause shape issues)
            return self._to_tensor(arrays)
        
        # Pad sequences
        padded_arrays = []
        for arr in arrays:
            if len(arr) > target_length:
                # Truncate if too long
                arr = arr[:target_length]
            elif len(arr) < target_length:
                # Pad if too short
                if arr.ndim == 1:
                    pad_width = target_length - len(arr)
                    arr = np.pad(arr, (0, pad_width), constant_values=self.pad_value)
                else:
                    pad_width = ((0, target_length - len(arr)), (0, 0))
                    arr = np.pad(arr, pad_width, constant_values=self.pad_value)
            
            padded_arrays.append(arr)
        
        # Stack into batch
        batch_array = np.stack(padded_arrays, axis=0)
        return self._to_tensor(batch_array)
    
    def _collate_labels(self, labels: List[Any]) -> Union[torch.Tensor, np.ndarray]:
        """Collate label data."""
        if not labels:
            return self._to_tensor([])
        
        # Handle different label types
        if isinstance(labels[0], (int, float)):
            return self._to_tensor(labels)
        elif isinstance(labels[0], (list, np.ndarray, torch.Tensor)):
            return self._collate_sequences(labels)
        else:
            # Keep as list for string labels or complex objects
            return labels
    
    def _collate_lengths(self, lengths: List[int]) -> Union[torch.Tensor, np.ndarray]:
        """Collate sequence length information."""
        return self._to_tensor(lengths)
    
    def _to_tensor(self, data: Any) -> Union[torch.Tensor, np.ndarray]:
        """Convert data to appropriate tensor format."""
        if self.return_tensors == "pt":
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            elif isinstance(data, list):
                return torch.tensor(data)
            elif isinstance(data, torch.Tensor):
                return data
            else:
                return torch.tensor(data)
        elif self.return_tensors == "np":
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, list):
                return np.array(data)
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)
        else:
            return data


class DataBatcher:
    """Utility for creating batches from datasets."""
    
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize data batcher.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data before batching
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Function to collate batch samples
            seed: Random seed for shuffling
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or (lambda x: x)
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def batch_data(self, data: List[Any]) -> List[Any]:
        """Create batches from data."""
        if not data:
            return []
        
        # Shuffle if requested
        if self.shuffle:
            data = data.copy()
            random.shuffle(data)
        
        # Create batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last=True
            if self.drop_last and len(batch) < self.batch_size:
                continue
            
            # Apply collation function
            try:
                collated_batch = self.collate_fn(batch)
                batches.append(collated_batch)
            except Exception as e:
                logger.error(f"Error collating batch: {e}")
                # Return uncollated batch as fallback
                batches.append(batch)
        
        return batches
    
    def __call__(self, data: List[Any]) -> List[Any]:
        """Batch data (callable interface)."""
        return self.batch_data(data)


class DataSampler:
    """Utility for sampling data with various strategies."""
    
    @staticmethod
    def random_sample(data: List[Any], n: int, seed: Optional[int] = None) -> List[Any]:
        """Randomly sample n items from data."""
        if seed is not None:
            random.seed(seed)
        
        if n >= len(data):
            return data.copy()
        
        return random.sample(data, n)
    
    @staticmethod
    def stratified_sample(
        data: List[Any],
        labels: List[Any],
        n: int,
        seed: Optional[int] = None
    ) -> Tuple[List[Any], List[Any]]:
        """Stratified sampling to maintain class distribution."""
        if seed is not None:
            random.seed(seed)
        
        # Group data by labels
        label_groups = defaultdict(list)
        for item, label in zip(data, labels):
            label_groups[label].append((item, label))
        
        # Calculate samples per class
        n_classes = len(label_groups)
        samples_per_class = n // n_classes
        remaining_samples = n % n_classes
        
        sampled_data = []
        sampled_labels = []
        
        for i, (label, items) in enumerate(label_groups.items()):
            # Add extra sample to first classes if needed
            class_n = samples_per_class + (1 if i < remaining_samples else 0)
            class_n = min(class_n, len(items))
            
            sampled_items = random.sample(items, class_n)
            
            for item, label in sampled_items:
                sampled_data.append(item)
                sampled_labels.append(label)
        
        return sampled_data, sampled_labels
    
    @staticmethod
    def weighted_sample(
        data: List[Any],
        weights: List[float],
        n: int,
        replacement: bool = True,
        seed: Optional[int] = None
    ) -> List[Any]:
        """Sample data according to weights."""
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        indices = np.random.choice(
            len(data),
            size=n,
            replace=replacement,
            p=weights
        )
        
        return [data[i] for i in indices]


class DataPreprocessor:
    """Utility for common data preprocessing operations."""
    
    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        method: str = "peak",
        target_level: float = 0.95
    ) -> np.ndarray:
        """Normalize audio data."""
        if audio.size == 0:
            return audio
        
        if method == "peak":
            # Peak normalization
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio * (target_level / peak)
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio = audio * (target_level / rms)
        elif method == "lufs":
            # LUFS normalization (simplified)
            # This is a basic implementation - for production use a proper LUFS library
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                current_lufs = 20 * np.log10(rms)
                target_lufs = -23.0  # EBU R128 standard
                gain_db = target_lufs - current_lufs
                gain_linear = 10 ** (gain_db / 20)
                audio = audio * gain_linear
        
        return audio
    
    @staticmethod
    def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply gain in decibels to audio."""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """Trim leading and trailing silence from audio."""
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Find non-silent frames
        non_silent = audio_db > threshold_db
        
        # Find start and end of non-silent regions
        if not np.any(non_silent):
            return audio  # All silent
        
        start_idx = np.argmax(non_silent)
        end_idx = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        return audio[start_idx:end_idx + 1]
    
    @staticmethod
    def split_into_chunks(
        data: np.ndarray,
        chunk_size: int,
        hop_size: Optional[int] = None,
        pad_last: bool = True
    ) -> List[np.ndarray]:
        """Split data into overlapping or non-overlapping chunks."""
        if hop_size is None:
            hop_size = chunk_size  # Non-overlapping
        
        chunks = []
        for i in range(0, len(data), hop_size):
            chunk = data[i:i + chunk_size]
            
            if len(chunk) < chunk_size:
                if pad_last:
                    # Pad the last chunk
                    pad_width = chunk_size - len(chunk)
                    if chunk.ndim == 1:
                        chunk = np.pad(chunk, (0, pad_width), mode='constant')
                    else:
                        chunk = np.pad(chunk, ((0, pad_width), (0, 0)), mode='constant')
                else:
                    # Skip incomplete chunks
                    if len(chunk) == 0:
                        break
            
            chunks.append(chunk)
        
        return chunks


def create_data_loader(
    data: List[Any],
    batch_size: int = 32,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    **kwargs
) -> DataBatcher:
    """Create a data loader with specified parameters."""
    if collate_fn is None:
        collate_fn = AudioCollator(**kwargs)
    
    return DataBatcher(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **kwargs
    )


# Export all classes and functions
__all__ = [
    'DataCollator',
    'AudioCollator',
    'DataBatcher',
    'DataSampler',
    'DataPreprocessor',
    'create_data_loader'
]