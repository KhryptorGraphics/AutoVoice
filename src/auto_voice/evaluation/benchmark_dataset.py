"""Benchmark dataset for voice conversion quality evaluation.

Provides structured access to benchmark audio samples including
source audio, target speaker embeddings, and reference conversions.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torchaudio


class BenchmarkDataset:
    """Dataset for voice conversion benchmarking.

    Loads and provides access to benchmark samples with source audio,
    target speakers, and optional reference conversions.

    Args:
        data_dir: Path to benchmark data directory
        sample_rate: Target sample rate for audio (default: 24000)
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 24000,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.samples: List[Dict[str, Any]] = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Load samples from data directory."""
        if not self.data_dir.exists():
            raise RuntimeError(f"Benchmark data directory not found: {self.data_dir}")

        # Look for source audio files
        audio_files = list(self.data_dir.glob("**/*.wav")) + list(self.data_dir.glob("**/*.mp3"))

        for audio_path in audio_files:
            sample = self._load_sample(audio_path)
            if sample is not None:
                self.samples.append(sample)

    def _load_sample(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single benchmark sample.

        Expected directory structure:
        sample_dir/
          source.wav       - Source audio to convert
          speaker.pt       - Target speaker embedding
          reference.wav    - Optional reference conversion
          metadata.json    - Optional metadata

        Args:
            audio_path: Path to source audio file

        Returns:
            Sample dictionary or None if invalid
        """
        sample_dir = audio_path.parent

        try:
            # Load source audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            sample = {
                'source_audio': waveform.squeeze(),
                'source_path': str(audio_path),
                'metadata': {'sample_rate': self.sample_rate},
            }

            # Load speaker embedding if available
            speaker_path = sample_dir / "speaker.pt"
            if speaker_path.exists():
                sample['target_speaker'] = torch.load(speaker_path, weights_only=True)
            else:
                # Generate random speaker as placeholder
                sample['target_speaker'] = torch.randn(256)

            # Load reference if available
            ref_path = sample_dir / "reference.wav"
            if ref_path.exists():
                ref_waveform, ref_sr = torchaudio.load(ref_path)
                if ref_sr != self.sample_rate:
                    ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, self.sample_rate)
                if ref_waveform.shape[0] > 1:
                    ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
                sample['reference_audio'] = ref_waveform.squeeze()

            return sample

        except Exception as e:
            # Skip invalid samples
            return None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        return self.samples[idx]

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)
