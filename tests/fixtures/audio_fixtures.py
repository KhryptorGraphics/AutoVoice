"""Audio-related pytest fixtures for AutoVoice testing.

Provides comprehensive audio generation, manipulation, and validation fixtures
for testing voice conversion, singing analysis, and audio processing pipelines.
"""

import numpy as np
import pytest
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import soundfile as sf


# ============================================================================
# Audio Factory Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_factory():
    """Factory fixture for generating various types of synthetic audio.

    Returns a callable that can generate audio with different characteristics:
    - Type: sine, harmonics, speech_like, noise, silence, chirp
    - Sample rate: customizable (default 22050)
    - Duration: customizable (default 1.0s)
    - Additional parameters based on type

    Examples:
        audio = sample_audio_factory(type='sine', frequency=440, duration=2.0)
        audio = sample_audio_factory(type='harmonics', fundamental=220, num_harmonics=5)
        audio = sample_audio_factory(type='speech_like', formants=[800, 1200, 2500])
    """
    def factory(
        audio_type: str = 'sine',
        sample_rate: int = 22050,
        duration: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """Generate synthetic audio of specified type.

        Args:
            audio_type: Type of audio to generate
            sample_rate: Sample rate in Hz
            duration: Duration in seconds
            **kwargs: Type-specific parameters

        Returns:
            Audio samples as float32 numpy array
        """
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)

        if audio_type == 'sine':
            frequency = kwargs.get('frequency', 440.0)
            amplitude = kwargs.get('amplitude', 0.5)
            audio = amplitude * np.sin(2 * np.pi * frequency * t)

        elif audio_type == 'harmonics':
            fundamental = kwargs.get('fundamental', 220.0)
            num_harmonics = kwargs.get('num_harmonics', 5)
            audio = np.zeros(num_samples, dtype=np.float32)
            for h in range(1, num_harmonics + 1):
                amplitude = 1.0 / h
                audio += amplitude * np.sin(2 * np.pi * fundamental * h * t)
            audio /= num_harmonics

        elif audio_type == 'speech_like':
            # Generate speech-like audio with formants
            formants = kwargs.get('formants', [800, 1200, 2500])
            audio = np.zeros(num_samples, dtype=np.float32)
            for i, formant in enumerate(formants):
                amplitude = 1.0 / (i + 1)
                audio += amplitude * np.sin(2 * np.pi * formant * t)
            # Add amplitude envelope
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t)
            audio *= envelope

        elif audio_type == 'noise':
            noise_type = kwargs.get('noise_type', 'white')
            if noise_type == 'white':
                audio = np.random.randn(num_samples).astype(np.float32)
            elif noise_type == 'pink':
                # Simplified pink noise
                white = np.random.randn(num_samples)
                audio = np.cumsum(white).astype(np.float32)
                audio = audio / np.max(np.abs(audio))
            else:
                audio = np.random.randn(num_samples).astype(np.float32)

        elif audio_type == 'silence':
            audio = np.zeros(num_samples, dtype=np.float32)

        elif audio_type == 'chirp':
            f0 = kwargs.get('f0', 200.0)
            f1 = kwargs.get('f1', 800.0)
            # Linear chirp
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
            audio = np.sin(phase).astype(np.float32)

        else:
            raise ValueError(f"Unknown audio type: {audio_type}")

        # Apply gain if specified
        gain = kwargs.get('gain', 1.0)
        audio *= gain

        # Normalize if requested
        if kwargs.get('normalize', True):
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9

        return audio.astype(np.float32)

    return factory


@pytest.fixture
def audio_file_factory(tmp_path: Path):
    """Factory for creating temporary audio files.

    Returns a callable that generates audio files in various formats.
    Files are automatically cleaned up after test completion.

    Examples:
        path = audio_file_factory('test.wav', audio_data, sample_rate=22050)
        path = audio_file_factory('test.flac', audio_data, format='FLAC')
    """
    created_files = []

    def factory(
        filename: str,
        audio: np.ndarray,
        sample_rate: int = 22050,
        file_format: Optional[str] = None,
        subtype: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Create audio file from array.

        Args:
            filename: Output filename
            audio: Audio data (1D or 2D array)
            sample_rate: Sample rate in Hz
            file_format: File format (WAV, FLAC, etc.)
            subtype: Audio subtype (PCM_16, FLOAT, etc.)
            **kwargs: Additional sf.write parameters

        Returns:
            Path to created audio file
        """
        filepath = tmp_path / filename

        # Ensure audio is 2D for soundfile
        if audio.ndim == 1:
            audio_2d = audio.reshape(-1, 1)
        else:
            audio_2d = audio

        # Write audio file
        sf.write(
            str(filepath),
            audio_2d,
            sample_rate,
            format=file_format,
            subtype=subtype,
            **kwargs
        )

        created_files.append(filepath)
        return filepath

    yield factory

    # Cleanup
    for file in created_files:
        if file.exists():
            file.unlink()


@pytest.fixture
def multi_channel_audio():
    """Generate multi-channel audio for testing stereo/surround processing.

    Returns a callable that generates N-channel audio with configurable
    channel relationships (identical, phase-shifted, independent, etc.)

    Examples:
        stereo = multi_channel_audio(num_channels=2, relationship='identical')
        surround = multi_channel_audio(num_channels=6, relationship='independent')
    """
    def factory(
        num_channels: int = 2,
        relationship: str = 'independent',
        sample_rate: int = 22050,
        duration: float = 1.0,
        base_frequency: float = 440.0
    ) -> np.ndarray:
        """Generate multi-channel audio.

        Args:
            num_channels: Number of audio channels
            relationship: Channel relationship type
            sample_rate: Sample rate in Hz
            duration: Duration in seconds
            base_frequency: Base frequency for sine generation

        Returns:
            Audio array of shape (num_samples, num_channels)
        """
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)

        if relationship == 'identical':
            # All channels identical
            channel = np.sin(2 * np.pi * base_frequency * t)
            audio = np.tile(channel.reshape(-1, 1), (1, num_channels))

        elif relationship == 'phase_shifted':
            # Each channel phase-shifted
            audio = np.zeros((num_samples, num_channels), dtype=np.float32)
            for ch in range(num_channels):
                phase = 2 * np.pi * ch / num_channels
                audio[:, ch] = np.sin(2 * np.pi * base_frequency * t + phase)

        elif relationship == 'independent':
            # Each channel independent random
            audio = np.random.randn(num_samples, num_channels).astype(np.float32)

        elif relationship == 'stereo_field':
            # Stereo with panning
            if num_channels != 2:
                raise ValueError("stereo_field requires num_channels=2")
            mono = np.sin(2 * np.pi * base_frequency * t)
            audio = np.zeros((num_samples, 2), dtype=np.float32)
            audio[:, 0] = mono * 0.7  # Left
            audio[:, 1] = mono * 0.3  # Right (panned)

        else:
            raise ValueError(f"Unknown relationship: {relationship}")

        return audio.astype(np.float32)

    return factory


@pytest.fixture
def audio_batch_generator():
    """Generate batches of audio for testing batch processing.

    Returns a callable that yields batches of audio samples with
    consistent or varying lengths, sample rates, and characteristics.

    Examples:
        batches = list(audio_batch_generator(batch_size=16, num_batches=10))
        for batch in audio_batch_generator(batch_size=8, variable_length=True):
            process(batch)
    """
    def factory(
        batch_size: int = 16,
        num_batches: int = 1,
        sample_rate: int = 22050,
        duration: float = 1.0,
        variable_length: bool = False,
        audio_type: str = 'harmonics',
        return_tensors: bool = False
    ):
        """Generate audio batches.

        Args:
            batch_size: Number of samples per batch
            num_batches: Number of batches to generate
            sample_rate: Sample rate in Hz
            duration: Base duration in seconds
            variable_length: If True, vary duration per sample
            audio_type: Type of audio to generate
            return_tensors: If True, return torch tensors instead of numpy

        Yields:
            Batches of audio samples
        """
        for batch_idx in range(num_batches):
            batch = []

            for sample_idx in range(batch_size):
                # Vary duration if requested
                if variable_length:
                    dur = duration * (0.5 + np.random.rand())
                else:
                    dur = duration

                # Generate audio
                num_samples = int(sample_rate * dur)
                t = np.linspace(0, dur, num_samples, dtype=np.float32)

                if audio_type == 'harmonics':
                    fundamental = 200.0 + np.random.rand() * 400.0
                    audio = np.zeros(num_samples, dtype=np.float32)
                    for h in range(1, 6):
                        audio += (1.0 / h) * np.sin(2 * np.pi * fundamental * h * t)
                    audio /= 5.0
                elif audio_type == 'noise':
                    audio = np.random.randn(num_samples).astype(np.float32)
                else:
                    freq = 440.0
                    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

                batch.append(audio)

            if return_tensors:
                # Pad to same length and convert to tensor
                max_len = max(len(x) for x in batch)
                batch_tensor = torch.zeros(batch_size, max_len)
                for i, audio in enumerate(batch):
                    batch_tensor[i, :len(audio)] = torch.from_numpy(audio)
                yield batch_tensor
            else:
                yield batch

    return factory


@pytest.fixture
def corrupted_audio_samples():
    """Generate corrupted/problematic audio samples for edge case testing.

    Returns a dict of various corrupted audio scenarios:
    - clipped: Audio with clipping artifacts
    - silent: Silent audio
    - dc_offset: Audio with DC offset
    - inf_values: Audio with inf/nan values
    - ultra_short: Very short audio (< 100 samples)
    - ultra_long: Very long audio (> 1M samples)

    Examples:
        samples = corrupted_audio_samples()
        assert handle_clipping(samples['clipped'])
        assert handle_silent(samples['silent'])
    """
    sample_rate = 22050
    duration = 1.0
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, dtype=np.float32)

    # Generate base audio
    base_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    return {
        'clipped': np.clip(base_audio * 2.0, -1.0, 1.0),
        'silent': np.zeros(num_samples, dtype=np.float32),
        'dc_offset': base_audio + 0.5,
        'inf_values': np.concatenate([
            base_audio[:num_samples//2],
            np.array([np.inf] * (num_samples//2), dtype=np.float32)
        ]),
        'nan_values': np.concatenate([
            base_audio[:num_samples//2],
            np.array([np.nan] * (num_samples//2), dtype=np.float32)
        ]),
        'ultra_short': base_audio[:50],
        'ultra_long': np.tile(base_audio, 100),  # 100 seconds
        'extreme_amplitude': base_audio * 100.0,
        'subsonic': np.sin(2 * np.pi * 5 * t).astype(np.float32),  # 5 Hz
        'ultrasonic': np.sin(2 * np.pi * 25000 * t).astype(np.float32),  # 25 kHz
    }


# ============================================================================
# Audio Feature Fixtures
# ============================================================================

@pytest.fixture
def mel_spectrogram_factory():
    """Factory for generating mel-spectrograms with various configurations.

    Examples:
        mel = mel_spectrogram_factory(audio, n_mels=80, fmax=8000)
        mel = mel_spectrogram_factory(audio, power=2.0)
    """
    def factory(
        audio: np.ndarray,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        return_tensor: bool = False
    ):
        """Generate mel-spectrogram from audio.

        Args:
            audio: Input audio samples
            sample_rate: Sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length in samples
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sr/2)
            power: Exponent for magnitude spectrogram
            return_tensor: If True, return torch tensor

        Returns:
            Mel-spectrogram (n_mels, n_frames)
        """
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=power
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if return_tensor:
            return torch.from_numpy(mel_spec_db).float()
        return mel_spec_db

    return factory


@pytest.fixture
def pitch_contour_factory():
    """Factory for generating pitch contours for testing.

    Examples:
        pitch = pitch_contour_factory(length=100, base_freq=220, vibrato=True)
        pitch = pitch_contour_factory(length=200, pattern='rising')
    """
    def factory(
        length: int = 100,
        base_freq: float = 220.0,
        pattern: str = 'constant',
        vibrato: bool = False,
        vibrato_rate: float = 5.0,
        vibrato_depth: float = 20.0,
    ) -> np.ndarray:
        """Generate pitch contour.

        Args:
            length: Number of frames
            base_freq: Base frequency in Hz
            pattern: Pattern type (constant, rising, falling, wavy)
            vibrato: Add vibrato modulation
            vibrato_rate: Vibrato rate in Hz
            vibrato_depth: Vibrato depth in cents

        Returns:
            Pitch contour in Hz
        """
        t = np.linspace(0, 1, length)

        if pattern == 'constant':
            pitch = np.ones(length) * base_freq
        elif pattern == 'rising':
            pitch = base_freq * (1 + t * 0.5)  # Rise 50%
        elif pattern == 'falling':
            pitch = base_freq * (1.5 - t * 0.5)  # Fall 50%
        elif pattern == 'wavy':
            pitch = base_freq * (1 + 0.2 * np.sin(2 * np.pi * 2 * t))
        else:
            pitch = np.ones(length) * base_freq

        if vibrato:
            # Add vibrato modulation
            vibrato_mod = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            pitch *= 2 ** (vibrato_mod / 1200.0)  # Convert cents to ratio

        return pitch.astype(np.float32)

    return factory


__all__ = [
    'sample_audio_factory',
    'audio_file_factory',
    'multi_channel_audio',
    'audio_batch_generator',
    'corrupted_audio_samples',
    'mel_spectrogram_factory',
    'pitch_contour_factory',
]
