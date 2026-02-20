"""Audio processing utilities."""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_EXTENSIONS = {
    'wav', 'mp3', 'flac', 'ogg', 'opus', 'aac', 'm4a', 'wma', 'aiff', 'webm'
}


class AudioProcessor:
    """Core audio processing operations."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load(self, path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        import librosa
        target_sr = sr or self.sample_rate
        audio, sr_out = librosa.load(path, sr=target_sr, mono=mono)
        return audio, sr_out

    def save(self, path: str, audio: np.ndarray, sr: Optional[int] = None):
        """Save audio to file."""
        import soundfile as sf
        sf.write(path, audio, sr or self.sample_rate)

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def normalize(self, audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
        """Normalize audio to peak amplitude."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio * (peak / max_val)
        return audio

    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Trim silence from beginning and end."""
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
        return trimmed

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono."""
        if audio.ndim == 1:
            return audio
        return np.mean(audio, axis=0)

    def validate_format(self, file_path: str) -> bool:
        """Validate if file has an allowed audio extension.

        Args:
            file_path: Path to the audio file

        Returns:
            True if file has valid audio extension

        Raises:
            ValueError: If file path is invalid or has unsupported extension
        """
        if not file_path or '.' not in file_path:
            raise ValueError("Invalid file path: must contain an extension")

        ext = file_path.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: .{ext}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        return True

    def get_audio_info(self, file_path: str) -> dict:
        """Get audio file metadata.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary containing audio metadata:
                - duration: Duration in seconds (float)
                - sample_rate: Sample rate in Hz (int)
                - channels: Number of audio channels (int)
                - format: Audio format/subtype (str)
                - frames: Total number of frames (int)

        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If unable to read audio file metadata

        Example:
            >>> processor = AudioProcessor()
            >>> info = processor.get_audio_info('song.wav')
            >>> print(f"Duration: {info['duration']:.2f}s")
            Duration: 180.50s
        """
        import os
        import soundfile as sf

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            with sf.SoundFile(file_path) as audio_file:
                frames = len(audio_file)
                sample_rate = audio_file.samplerate
                channels = audio_file.channels
                audio_format = audio_file.format
                subtype = audio_file.subtype

                duration = frames / sample_rate if sample_rate > 0 else 0.0

                return {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'format': f"{audio_format}/{subtype}",
                    'frames': frames,
                }
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file metadata: {e}") from e
