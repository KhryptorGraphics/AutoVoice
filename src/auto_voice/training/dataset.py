"""
Voice Conversion Dataset

Provides PairedVoiceDataset for source-target audio pairs with singing-specific augmentation.
"""

import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torchaudio

# Optional pyrubberband for pitch-preserving time stretch
try:
    import pyrubberband as pyrb
    PYRUBBERBAND_AVAILABLE = True
except ImportError:
    PYRUBBERBAND_AVAILABLE = False

from .data_pipeline import AudioConfig, AudioProcessor, VoiceCollator
from ..audio.pitch_extractor import SingingPitchExtractor
from ..models.speaker_encoder import SpeakerEncoder

logger = logging.getLogger(__name__)


class PairedVoiceDataset(torch.utils.data.Dataset):
    """Dataset for voice conversion training with source-target audio pairs.

    Loads paired audio files with metadata, extracts features (mel-spectrograms,
    F0 contours, speaker embeddings), and applies optional augmentation transforms.

    Metadata JSON format:
        {
          "pairs": [
            {
              "source_file": "speaker1/song1.wav",
              "target_file": "speaker2/song1.wav",
              "source_speaker_id": "speaker1",
              "target_speaker_id": "speaker2",
              "duration": 5.2,
              "song_id": "song1"
            }
          ]
        }
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: str,
        audio_config: Optional[AudioConfig] = None,
        extract_f0: bool = True,
        extract_speaker_emb: bool = True,
        transforms: Optional[List[Callable]] = None,
        augmentation_prob: float = 0.5,
        cache_size: int = 500,
        device: Optional[str] = None,
        gpu_manager: Optional[Any] = None
    ):
        """Initialize PairedVoiceDataset.

        Args:
            data_dir: Directory containing paired audio files
            metadata_file: Path to JSON file with paired audio metadata
            audio_config: Audio processing configuration (default: AudioConfig())
            extract_f0: Whether to extract F0 contours (default: True)
            extract_speaker_emb: Whether to extract speaker embeddings (default: True)
            transforms: List of augmentation transforms to apply
            augmentation_prob: Probability of applying each transform (default: 0.5)
            cache_size: Number of processed samples to cache in memory
            device: Device for GPU acceleration ('cuda', 'cpu', etc.)
            gpu_manager: GPU manager instance for resource coordination
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.metadata_file = metadata_file
        self.audio_config = audio_config or AudioConfig()
        self.extract_f0 = extract_f0
        self.extract_speaker_emb = extract_speaker_emb
        self.transforms = transforms or []
        self.augmentation_prob = augmentation_prob
        self.cache_size = cache_size
        self.device = device
        self.gpu_manager = gpu_manager

        # Initialize audio processor
        self.audio_processor = AudioProcessor(self.audio_config)

        # Initialize pitch extractor if needed
        self.pitch_extractor = None
        if self.extract_f0:
            self.pitch_extractor = SingingPitchExtractor(
                device=device,
                gpu_manager=gpu_manager
            )
            logger.info("Initialized SingingPitchExtractor for F0 extraction")

        # Initialize speaker encoder if needed
        self.speaker_encoder = None
        if self.extract_speaker_emb:
            self.speaker_encoder = SpeakerEncoder(
                device=device,
                gpu_manager=gpu_manager
            )
            logger.info("Initialized SpeakerEncoder for speaker embedding extraction")

        # Load metadata
        self._load_metadata()

        # Initialize cache
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.cache_lock = threading.RLock()

        logger.info(
            f"Initialized PairedVoiceDataset with {len(self.pairs)} pairs, "
            f"cache_size={cache_size}, extract_f0={extract_f0}, "
            f"extract_speaker_emb={extract_speaker_emb}"
        )

    def _load_metadata(self):
        """Load paired audio metadata from JSON file."""
        metadata_path = Path(self.metadata_file)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.pairs = metadata.get('pairs', [])

        if not self.pairs:
            raise ValueError(f"No pairs found in metadata file: {metadata_path}")

        logger.info(f"Loaded {len(self.pairs)} pairs from {metadata_path}")

    def __len__(self) -> int:
        """Return number of paired samples."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a paired sample by index.

        Args:
            idx: Sample index

        Returns:
            Dict containing:
                - source_audio: Source audio waveform [T_audio]
                - target_audio: Target audio waveform [T_audio]
                - source_mel: Source mel-spectrogram [T_mel, 80]
                - target_mel: Target mel-spectrogram [T_mel, 80]
                - source_f0: Source F0 contour [T_f0] (if extract_f0=True)
                - target_f0: Target F0 contour [T_f0] (if extract_f0=True)
                - source_speaker_emb: Source speaker embedding [256] (if extract_speaker_emb=True)
                - target_speaker_emb: Target speaker embedding [256] (if extract_speaker_emb=True)
                - source_speaker_id: Source speaker ID (str)
                - target_speaker_id: Target speaker ID (str)
                - lengths: Mel-spectrogram length [1]
        """
        # Check cache first - get base features
        with self.cache_lock:
            if idx in self.cache:
                # Clone cached data to avoid modifying cache
                data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                       for k, v in self.cache[idx].items()}
            else:
                # Process sample from scratch
                data = self._process_sample(idx)

                # Cache base features if space available
                if len(self.cache) < self.cache_size:
                    self.cache[idx] = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                       for k, v in data.items()}

        # Apply transforms (augmentation) - always apply even on cached data
        data = self._apply_transforms(data)

        return data

    def _process_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Process a single paired sample.

        Args:
            idx: Sample index

        Returns:
            Processed sample dict
        """
        pair = self.pairs[idx]

        # Load audio files
        source_path = self.data_dir / pair['source_file']
        target_path = self.data_dir / pair['target_file']

        try:
            source_audio = self.audio_processor.load_audio(str(source_path))
            target_audio = self.audio_processor.load_audio(str(target_path))
        except Exception as e:
            logger.error(f"Failed to load audio pair {idx}: {e}")
            # Return zeros as fallback
            return self._get_zero_sample(pair)

        # Align audio lengths
        source_audio, target_audio = self._align_audio_lengths(source_audio, target_audio)

        # Extract mel-spectrograms
        source_mel = self.audio_processor.audio_to_mel(source_audio)
        target_mel = self.audio_processor.audio_to_mel(target_audio)

        # Ensure mel-spectrograms have same length
        min_mel_len = min(source_mel.shape[0], target_mel.shape[0])
        source_mel = source_mel[:min_mel_len]
        target_mel = target_mel[:min_mel_len]

        # Build result dict
        data = {
            'source_audio': torch.from_numpy(source_audio).float(),
            'target_audio': torch.from_numpy(target_audio).float(),
            'source_mel': torch.from_numpy(source_mel).float(),
            'target_mel': torch.from_numpy(target_mel).float(),
            'source_speaker_id': pair['source_speaker_id'],
            'target_speaker_id': pair['target_speaker_id'],
            'lengths': torch.LongTensor([min_mel_len])
        }

        # Extract F0 if enabled
        if self.extract_f0 and self.pitch_extractor is not None:
            try:
                # Extract source F0
                source_f0_result = self.pitch_extractor.extract_f0_contour(
                    source_audio,
                    self.audio_config.sample_rate
                )
                source_f0 = source_f0_result['f0']

                # Extract target F0
                target_f0_result = self.pitch_extractor.extract_f0_contour(
                    target_audio,
                    self.audio_config.sample_rate
                )
                target_f0 = target_f0_result['f0']

                # Interpolate F0 to match mel length if needed
                if len(source_f0) != min_mel_len:
                    source_f0 = self._interpolate_f0(source_f0, min_mel_len)
                if len(target_f0) != min_mel_len:
                    target_f0 = self._interpolate_f0(target_f0, min_mel_len)

                data['source_f0'] = torch.from_numpy(source_f0).float()
                data['target_f0'] = torch.from_numpy(target_f0).float()

            except Exception as e:
                logger.warning(f"F0 extraction failed for pair {idx}: {e}")
                # Use zeros as fallback
                data['source_f0'] = torch.zeros(min_mel_len)
                data['target_f0'] = torch.zeros(min_mel_len)

        # Extract speaker embeddings if enabled
        if self.extract_speaker_emb and self.speaker_encoder is not None:
            try:
                # Extract source speaker embedding
                source_emb = self.speaker_encoder.extract_embedding(
                    source_audio,
                    self.audio_config.sample_rate
                )

                # Extract target speaker embedding
                target_emb = self.speaker_encoder.extract_embedding(
                    target_audio,
                    self.audio_config.sample_rate
                )

                data['source_speaker_emb'] = torch.from_numpy(source_emb).float()
                data['target_speaker_emb'] = torch.from_numpy(target_emb).float()

            except Exception as e:
                logger.warning(f"Speaker embedding extraction failed for pair {idx}: {e}")
                # Use random embeddings as fallback
                data['source_speaker_emb'] = torch.randn(256)
                data['target_speaker_emb'] = torch.randn(256)

        return data

    def _align_audio_lengths(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two audio arrays to the same length using a single consistent crop offset.

        This ensures source and target remain temporally aligned by using the same
        random start position for cropping the longer audio.

        Args:
            audio1: First audio array
            audio2: Second audio array

        Returns:
            Tuple of aligned audio arrays with consistent temporal alignment
        """
        len1, len2 = len(audio1), len(audio2)

        if len1 == len2:
            return audio1, audio2

        # Use shorter length as target
        target_len = min(len1, len2)

        # Compute single random start offset
        # Apply to whichever audio is longer
        if len1 > len2:
            # audio1 is longer - crop it with random start
            start = random.randint(0, len1 - target_len)
            audio1 = audio1[start:start + target_len]
            # audio2 already matches target_len (it's the shorter one)
        elif len2 > len1:
            # audio2 is longer - crop it with random start
            start = random.randint(0, len2 - target_len)
            audio2 = audio2[start:start + target_len]
            # audio1 already matches target_len (it's the shorter one)

        return audio1, audio2

    def _apply_transforms(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation transforms with probabilistic application.

        Args:
            data: Sample dict with audio and features

        Returns:
            Augmented sample dict with recomputed features
        """
        if not self.transforms:
            return data

        # Apply each transform with configured probability
        audio_modified = False

        for transform in self.transforms:
            # Apply transform probabilistically using augmentation_prob
            if random.random() < self.augmentation_prob:
                # Apply transform - pass audio_config and sample_rate for transforms
                data_with_config = {
                    **data,
                    'audio_config': self.audio_config,
                    'sample_rate': self.audio_config.sample_rate
                }
                data = transform(data_with_config)
                audio_modified = True

        # If audio was modified, recompute mel-spectrograms and features
        if audio_modified:
            data = self._recompute_features(data)

        return data

    def _recompute_features(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recompute mel-spectrograms, F0, and embeddings after audio augmentation.

        Args:
            data: Sample dict with potentially modified audio

        Returns:
            Sample dict with recomputed features
        """
        # Convert audio back to numpy for processing
        source_audio = data['source_audio'].numpy()
        target_audio = data['target_audio'].numpy()

        # Recompute mel-spectrograms
        source_mel = self.audio_processor.audio_to_mel(source_audio)
        target_mel = self.audio_processor.audio_to_mel(target_audio)

        # Ensure mel-spectrograms have same length
        min_mel_len = min(source_mel.shape[0], target_mel.shape[0])
        source_mel = source_mel[:min_mel_len]
        target_mel = target_mel[:min_mel_len]

        # Update mel-spectrograms
        data['source_mel'] = torch.from_numpy(source_mel).float()
        data['target_mel'] = torch.from_numpy(target_mel).float()
        data['lengths'] = torch.LongTensor([min_mel_len])

        # Recompute F0 if enabled
        if self.extract_f0 and self.pitch_extractor is not None:
            try:
                # Extract source F0
                source_f0_result = self.pitch_extractor.extract_f0_contour(
                    source_audio,
                    self.audio_config.sample_rate
                )
                source_f0 = source_f0_result['f0']

                # Extract target F0
                target_f0_result = self.pitch_extractor.extract_f0_contour(
                    target_audio,
                    self.audio_config.sample_rate
                )
                target_f0 = target_f0_result['f0']

                # Interpolate F0 to match mel length if needed
                if len(source_f0) != min_mel_len:
                    source_f0 = self._interpolate_f0(source_f0, min_mel_len)
                if len(target_f0) != min_mel_len:
                    target_f0 = self._interpolate_f0(target_f0, min_mel_len)

                data['source_f0'] = torch.from_numpy(source_f0).float()
                data['target_f0'] = torch.from_numpy(target_f0).float()

            except Exception as e:
                logger.warning(f"F0 recomputation failed: {e}")
                # Keep existing F0 or use zeros
                if 'source_f0' not in data:
                    data['source_f0'] = torch.zeros(min_mel_len)
                    data['target_f0'] = torch.zeros(min_mel_len)

        # Recompute speaker embeddings if enabled
        if self.extract_speaker_emb and self.speaker_encoder is not None:
            try:
                # Extract source speaker embedding
                source_emb = self.speaker_encoder.extract_embedding(
                    source_audio,
                    self.audio_config.sample_rate
                )

                # Extract target speaker embedding
                target_emb = self.speaker_encoder.extract_embedding(
                    target_audio,
                    self.audio_config.sample_rate
                )

                data['source_speaker_emb'] = torch.from_numpy(source_emb).float()
                data['target_speaker_emb'] = torch.from_numpy(target_emb).float()

            except Exception as e:
                logger.warning(f"Speaker embedding recomputation failed: {e}")
                # Keep existing embeddings or use random
                if 'source_speaker_emb' not in data:
                    data['source_speaker_emb'] = torch.randn(256)
                    data['target_speaker_emb'] = torch.randn(256)

        return data

    def _interpolate_f0(self, f0: np.ndarray, target_len: int) -> np.ndarray:
        """Interpolate F0 contour to target length.

        Args:
            f0: F0 array
            target_len: Target length

        Returns:
            Interpolated F0 array
        """
        if len(f0) == target_len:
            return f0

        # Linear interpolation
        x_old = np.linspace(0, 1, len(f0))
        x_new = np.linspace(0, 1, target_len)
        f0_interp = np.interp(x_new, x_old, f0)

        return f0_interp

    def _get_zero_sample(self, pair: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get a zero-filled sample as fallback for loading errors.

        Args:
            pair: Metadata pair dict

        Returns:
            Zero-filled sample dict
        """
        # Create minimal length sample
        audio_len = self.audio_config.sample_rate  # 1 second
        mel_len = audio_len // self.audio_config.hop_length

        data = {
            'source_audio': torch.zeros(audio_len),
            'target_audio': torch.zeros(audio_len),
            'source_mel': torch.zeros(mel_len, self.audio_config.n_mels),
            'target_mel': torch.zeros(mel_len, self.audio_config.n_mels),
            'source_speaker_id': pair['source_speaker_id'],
            'target_speaker_id': pair['target_speaker_id'],
            'lengths': torch.LongTensor([mel_len])
        }

        if self.extract_f0:
            data['source_f0'] = torch.zeros(mel_len)
            data['target_f0'] = torch.zeros(mel_len)

        if self.extract_speaker_emb:
            data['source_speaker_emb'] = torch.randn(256)
            data['target_speaker_emb'] = torch.randn(256)

        return data

    def preload_data(self, num_workers: int = 4):
        """Preload all data into memory for faster training.

        Args:
            num_workers: Number of worker threads for parallel loading
        """
        logger.info(f"Preloading {len(self)} samples with {num_workers} workers...")

        def load_sample(idx):
            return idx, self._process_sample(idx)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(load_sample, i) for i in range(len(self))]

            for future in futures:
                try:
                    idx, data = future.result()
                    with self.cache_lock:
                        self.cache[idx] = data
                except Exception as e:
                    logger.error(f"Failed to preload sample: {e}")

        logger.info(f"Preloaded {len(self.cache)} samples")


class SingingAugmentation:
    """Singing-specific data augmentation transforms."""

    @staticmethod
    def pitch_preserving_time_stretch(
        data: Dict[str, torch.Tensor],
        rate_range: Tuple[float, float] = (0.9, 1.1)
    ) -> Dict[str, torch.Tensor]:
        """Apply time stretch while preserving pitch.

        Essential for singing voice augmentation where pitch must be preserved.
        Uses pyrubberband if available, otherwise falls back to librosa phase vocoder.

        Args:
            data: Sample dict with audio and features (must include 'sample_rate')
            rate_range: Time stretch rate range (e.g., (0.9, 1.1) for ±10%)

        Returns:
            Augmented sample dict
        """
        rate = random.uniform(*rate_range)

        if abs(rate - 1.0) < 0.01:  # Skip if rate very close to 1.0
            return data

        # Get sample rate from data dict (passed by dataset)
        sample_rate = data.get('sample_rate', 44100)

        # Apply to source audio
        source_audio_np = data['source_audio'].numpy()
        if PYRUBBERBAND_AVAILABLE:
            source_audio_stretched = pyrb.time_stretch(source_audio_np, sample_rate, rate)
        else:
            # Fallback to librosa phase vocoder
            logger.warning("pyrubberband not available, using librosa phase vocoder (may change pitch)")
            source_audio_stretched = librosa.effects.time_stretch(source_audio_np, rate=rate)

        # Apply to target audio
        target_audio_np = data['target_audio'].numpy()
        if PYRUBBERBAND_AVAILABLE:
            target_audio_stretched = pyrb.time_stretch(target_audio_np, sample_rate, rate)
        else:
            target_audio_stretched = librosa.effects.time_stretch(target_audio_np, rate=rate)

        # Update data
        data['source_audio'] = torch.from_numpy(source_audio_stretched).float()
        data['target_audio'] = torch.from_numpy(target_audio_stretched).float()

        # Note: mel-spectrograms and F0 will be recomputed if needed
        # For now, we keep original mel/f0 as they should be recomputed by trainer

        return data

    @staticmethod
    def formant_shift(
        data: Dict[str, torch.Tensor],
        semitone_range: Tuple[float, float] = (-2, 2)
    ) -> Dict[str, torch.Tensor]:
        """Shift formants (vocal tract resonances) without changing F0.

        This changes timbre while preserving pitch by warping the mel-spectrogram.

        Args:
            data: Sample dict with mel-spectrograms (must include 'audio_config')
            semitone_range: Formant shift range in semitones

        Returns:
            Augmented sample dict
        """
        n_steps = random.uniform(*semitone_range)

        if abs(n_steps) < 0.1:  # Skip if shift very small
            return data

        # Get sample rate from data dict (passed by dataset)
        sample_rate = data.get('sample_rate', 44100)

        # Apply formant shift to source mel (shift frequency axis)
        source_mel = data['source_mel'].numpy()
        # Simple frequency axis warping (approximate formant shift)
        shift_factor = 2 ** (n_steps / 12)
        n_mels, mel_len = source_mel.shape[1], source_mel.shape[0]

        # Create warped frequency indices
        freq_indices = np.arange(n_mels) * shift_factor
        freq_indices = np.clip(freq_indices, 0, n_mels - 1)

        # Interpolate mel values
        source_mel_shifted = np.zeros_like(source_mel)
        for t in range(mel_len):
            source_mel_shifted[t] = np.interp(
                np.arange(n_mels),
                freq_indices,
                source_mel[t]
            )

        # Apply to target mel
        target_mel = data['target_mel'].numpy()
        target_mel_shifted = np.zeros_like(target_mel)
        for t in range(len(target_mel)):
            target_mel_shifted[t] = np.interp(
                np.arange(n_mels),
                freq_indices,
                target_mel[t]
            )

        # Update data
        data['source_mel'] = torch.from_numpy(source_mel_shifted).float()
        data['target_mel'] = torch.from_numpy(target_mel_shifted).float()

        # F0 and audio remain unchanged (only timbre changes)

        return data

    @staticmethod
    def noise_injection_snr(
        data: Dict[str, torch.Tensor],
        snr_db_range: Tuple[float, float] = (20, 40)
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise with specified SNR range.

        Args:
            data: Sample dict with audio
            snr_db_range: SNR range in dB (higher = less noise)

        Returns:
            Augmented sample dict
        """
        snr_db = random.uniform(*snr_db_range)

        # Add noise to source audio
        source_audio = data['source_audio'].numpy()
        signal_power = np.mean(source_audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), source_audio.shape)
        source_audio_noisy = source_audio + noise

        # Add noise to target audio
        target_audio = data['target_audio'].numpy()
        signal_power = np.mean(target_audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), target_audio.shape)
        target_audio_noisy = target_audio + noise

        # Update data
        data['source_audio'] = torch.from_numpy(source_audio_noisy).float()
        data['target_audio'] = torch.from_numpy(target_audio_noisy).float()

        # Note: mel-spectrograms should be recomputed

        return data

    @staticmethod
    def vocal_tract_length_perturbation(
        data: Dict[str, torch.Tensor],
        alpha_range: Tuple[float, float] = (0.9, 1.1)
    ) -> Dict[str, torch.Tensor]:
        """Warp mel-spectrogram frequency axis to simulate vocal tract length changes.

        Args:
            data: Sample dict with mel-spectrograms
            alpha_range: Warping factor range

        Returns:
            Augmented sample dict
        """
        alpha = random.uniform(*alpha_range)

        if abs(alpha - 1.0) < 0.01:  # Skip if alpha very close to 1.0
            return data

        # Apply to source mel
        source_mel = data['source_mel'].numpy()
        n_mels = source_mel.shape[1]

        # Create warped frequency bins
        original_bins = np.arange(n_mels)
        warped_bins = original_bins * alpha
        warped_bins = np.clip(warped_bins, 0, n_mels - 1)

        # Interpolate
        source_mel_warped = np.zeros_like(source_mel)
        for t in range(len(source_mel)):
            source_mel_warped[t] = np.interp(original_bins, warped_bins, source_mel[t])

        # Apply to target mel
        target_mel = data['target_mel'].numpy()
        target_mel_warped = np.zeros_like(target_mel)
        for t in range(len(target_mel)):
            target_mel_warped[t] = np.interp(original_bins, warped_bins, target_mel[t])

        # Update data
        data['source_mel'] = torch.from_numpy(source_mel_warped).float()
        data['target_mel'] = torch.from_numpy(target_mel_warped).float()

        return data


class PairedVoiceCollator:
    """Collate function for batching paired voice conversion samples."""

    def __init__(self, pad_value: float = 0.0):
        """Initialize collator.

        Args:
            pad_value: Value to use for padding sequences
        """
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples with padding.

        Args:
            batch: List of sample dicts

        Returns:
            Batched dict with padded sequences
        """
        # Get max lengths
        max_audio_len = max(item['source_audio'].size(0) for item in batch)
        max_mel_len = max(item['source_mel'].size(0) for item in batch)

        # Initialize batched tensors
        batch_size = len(batch)
        n_mels = batch[0]['source_mel'].size(1)

        # Audio tensors
        source_audio = torch.zeros(batch_size, max_audio_len)
        target_audio = torch.zeros(batch_size, max_audio_len)

        # Mel tensors
        source_mel = torch.zeros(batch_size, max_mel_len, n_mels)
        target_mel = torch.zeros(batch_size, max_mel_len, n_mels)

        # F0 tensors (if present)
        has_f0 = 'source_f0' in batch[0]
        if has_f0:
            source_f0 = torch.zeros(batch_size, max_mel_len)
            target_f0 = torch.zeros(batch_size, max_mel_len)

        # Speaker embeddings (if present)
        has_speaker_emb = 'source_speaker_emb' in batch[0]
        if has_speaker_emb:
            emb_dim = batch[0]['source_speaker_emb'].size(0)
            source_speaker_emb = torch.zeros(batch_size, emb_dim)
            target_speaker_emb = torch.zeros(batch_size, emb_dim)

        # Lengths and speaker IDs
        lengths = torch.zeros(batch_size, dtype=torch.long)
        source_speaker_ids = []
        target_speaker_ids = []

        # Fill tensors
        for i, item in enumerate(batch):
            # Audio
            audio_len = item['source_audio'].size(0)
            source_audio[i, :audio_len] = item['source_audio']
            target_audio[i, :audio_len] = item['target_audio']

            # Mel
            mel_len = item['source_mel'].size(0)
            source_mel[i, :mel_len] = item['source_mel']
            target_mel[i, :mel_len] = item['target_mel']

            # F0
            if has_f0:
                f0_len = item['source_f0'].size(0)
                source_f0[i, :f0_len] = item['source_f0']
                target_f0[i, :f0_len] = item['target_f0']

            # Speaker embeddings
            if has_speaker_emb:
                source_speaker_emb[i] = item['source_speaker_emb']
                target_speaker_emb[i] = item['target_speaker_emb']

            # Lengths and IDs
            lengths[i] = mel_len
            source_speaker_ids.append(item['source_speaker_id'])
            target_speaker_ids.append(item['target_speaker_id'])

        # Create mask for variable lengths
        mel_mask = torch.arange(max_mel_len).unsqueeze(0) < lengths.unsqueeze(1)
        mel_mask = mel_mask.unsqueeze(1).float()  # [B, 1, T_mel]

        # Build result dict
        result = {
            'source_audio': source_audio,
            'target_audio': target_audio,
            'source_mel': source_mel,
            'target_mel': target_mel,
            'source_speaker_id': source_speaker_ids,
            'target_speaker_id': target_speaker_ids,
            'lengths': lengths,
            'mel_mask': mel_mask
        }

        if has_f0:
            result['source_f0'] = source_f0
            result['target_f0'] = target_f0

        if has_speaker_emb:
            result['source_speaker_emb'] = source_speaker_emb
            result['target_speaker_emb'] = target_speaker_emb

        return result


def create_paired_voice_dataloader(
    dataset: PairedVoiceDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    drop_last: bool = True
) -> torch.utils.data.DataLoader:
    """Create DataLoader for paired voice conversion training.

    Args:
        dataset: PairedVoiceDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        distributed: Whether to use distributed sampler
        drop_last: Whether to drop incomplete batches

    Returns:
        Configured DataLoader
    """
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False  # Sampler handles shuffling

    collator = PairedVoiceCollator()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


def create_paired_train_val_datasets(
    data_dir: Union[str, Path],
    train_metadata: str,
    val_metadata: str,
    audio_config: Optional[AudioConfig] = None,
    train_transforms: Optional[List[Callable]] = None,
    augmentation_prob: float = 0.5,
    **dataset_kwargs
) -> Tuple[PairedVoiceDataset, PairedVoiceDataset]:
    """Create training and validation datasets from separate metadata files.

    Args:
        data_dir: Directory containing audio files
        train_metadata: Path to training metadata JSON
        val_metadata: Path to validation metadata JSON
        audio_config: Audio processing configuration
        train_transforms: Augmentation transforms for training (default: standard augmentations)
        augmentation_prob: Probability of applying each transform (default: 0.5)
        **dataset_kwargs: Additional arguments for PairedVoiceDataset

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Default augmentation transforms for training
    if train_transforms is None:
        train_transforms = [
            SingingAugmentation.pitch_preserving_time_stretch,
            SingingAugmentation.formant_shift,
            SingingAugmentation.noise_injection_snr
        ]

    # Create training dataset with augmentation
    train_dataset = PairedVoiceDataset(
        data_dir=data_dir,
        metadata_file=train_metadata,
        audio_config=audio_config,
        transforms=train_transforms,
        augmentation_prob=augmentation_prob,
        **dataset_kwargs
    )

    # Create validation dataset without augmentation
    val_dataset = PairedVoiceDataset(
        data_dir=data_dir,
        metadata_file=val_metadata,
        audio_config=audio_config,
        transforms=None,  # No augmentation for validation
        augmentation_prob=0.0,  # Ensure no augmentation for validation
        **dataset_kwargs
    )

    return train_dataset, val_dataset
