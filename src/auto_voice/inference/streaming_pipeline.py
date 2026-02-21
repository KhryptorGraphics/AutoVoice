"""Real-time streaming voice conversion pipeline.

Implements chunked inference with overlap-add synthesis for continuous
audio streaming with minimal latency.

Architecture:
- Chunk-based processing with configurable chunk/hop sizes
- Overlap-add synthesis with crossfade windows for glitch-free output
- Latency tracking and optimization
- Audio I/O stream handling for microphone/speaker integration

Target: < 50ms end-to-end latency on Jetson Thor
"""
import time
from typing import Optional, Callable, Dict, Any, List
import torch
import torch.nn.functional as F
import numpy as np

from .sota_pipeline import SOTAConversionPipeline


class StreamingConversionPipeline:
    """Real-time streaming voice conversion with overlap-add synthesis.

    Processes audio in chunks with configurable overlap for continuous
    output without glitches. Tracks latency to ensure real-time performance.

    Args:
        chunk_size_ms: Chunk duration in milliseconds (default: 100ms)
        overlap_ratio: Overlap between chunks as ratio of chunk size (default: 0.5)
        sample_rate: Audio sample rate (default: 24000)
        device: Torch device for processing (default: auto-detect)
    """

    def __init__(
        self,
        chunk_size_ms: int = 100,
        overlap_ratio: float = 0.5,
        sample_rate: int = 24000,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.overlap_ratio = overlap_ratio

        # Calculate chunk and hop sizes in samples
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        self.hop_size = int(self.chunk_size * (1 - overlap_ratio))
        self.overlap_size = self.chunk_size - self.hop_size

        # Device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize SOTA pipeline for actual conversion
        self._pipeline = SOTAConversionPipeline(device=self.device, n_steps=1)

        # Overlap-add buffer for continuous output
        self.overlap_buffer: Optional[torch.Tensor] = None

        # Crossfade window for smooth transitions
        self.crossfade_window = self._create_crossfade_window()

        # Latency tracking
        self._latency_history: List[float] = []
        self._max_latency_history = 100

        # Session state
        self.is_running = False
        self._speaker_embedding: Optional[torch.Tensor] = None

        # Minimum chunk size for processing (10ms worth of samples)
        self._min_chunk_size = int(sample_rate * 0.01)

    def _create_crossfade_window(self) -> torch.Tensor:
        """Create Hann crossfade window for overlap-add synthesis.

        Generates fade-in and fade-out windows using Hann function to ensure
        smooth transitions between overlapping chunks without audio glitches.

        Returns:
            Tensor of shape (2, overlap_size) containing [fade_in, fade_out] windows,
            or ones(1) if no overlap is configured
        """
        if self.overlap_size <= 0:
            return torch.ones(1)

        # Fade-in and fade-out windows
        fade = torch.hann_window(self.overlap_size * 2, device=self.device)
        fade_in = fade[:self.overlap_size]
        fade_out = fade[self.overlap_size:]

        return torch.stack([fade_in, fade_out])

    def process_chunk(
        self,
        audio_chunk: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Process a single audio chunk and return converted output.

        Args:
            audio_chunk: Input audio tensor of shape (samples,) or (1, samples)
            speaker_embedding: Target speaker embedding (256-dim)

        Returns:
            Converted audio chunk with overlap-add applied

        Raises:
            RuntimeError: If chunk is too short or speaker embedding is wrong size
        """
        start_time = time.time()

        # Validate inputs
        if audio_chunk.numel() < self._min_chunk_size:
            raise RuntimeError(
                f"Chunk too short: {audio_chunk.numel()} samples, "
                f"minimum is {self._min_chunk_size}"
            )

        if speaker_embedding.shape[-1] != 256:
            raise RuntimeError(
                f"Speaker embedding must be 256-dim, got {speaker_embedding.shape[-1]}"
            )

        # Ensure correct shape (samples,)
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.squeeze(0)

        # Move to device
        audio_chunk = audio_chunk.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)

        # Convert using SOTA pipeline
        # Note: Pipeline expects full audio, we process chunk by chunk
        with torch.no_grad():
            result = self._pipeline.convert(
                audio_chunk,
                self.sample_rate,
                speaker_embedding,
            )

        converted = result['audio'].squeeze()

        # Apply overlap-add synthesis
        output = self._apply_overlap_add(converted)

        # Track latency
        elapsed_ms = (time.time() - start_time) * 1000
        self._latency_history.append(elapsed_ms)
        if len(self._latency_history) > self._max_latency_history:
            self._latency_history.pop(0)

        # Ensure bounded output [-1, 1]
        output = torch.clamp(output, -1.0, 1.0)

        return output

    def _apply_overlap_add(self, converted: torch.Tensor) -> torch.Tensor:
        """Apply overlap-add synthesis for continuous output.

        Args:
            converted: Converted audio from current chunk

        Returns:
            Output audio with crossfade applied
        """
        if self.overlap_size <= 0 or self.overlap_buffer is None:
            # No overlap - just update buffer and return
            if self.overlap_size > 0 and converted.shape[0] >= self.overlap_size:
                self.overlap_buffer = converted[-self.overlap_size:].clone()
            return converted

        # Apply crossfade in overlap region
        output_len = converted.shape[0]
        output = converted.clone()

        if output_len >= self.overlap_size:
            # Crossfade with previous chunk's tail
            fade_in = self.crossfade_window[0][:min(self.overlap_size, output_len)]
            fade_out = self.crossfade_window[1][:min(self.overlap_size, self.overlap_buffer.shape[0])]

            overlap_len = min(self.overlap_size, output_len, self.overlap_buffer.shape[0])
            output[:overlap_len] = (
                output[:overlap_len] * fade_in[:overlap_len] +
                self.overlap_buffer[:overlap_len] * fade_out[:overlap_len]
            )

            # Update buffer with current chunk's tail
            self.overlap_buffer = converted[-self.overlap_size:].clone()

        return output

    def reset(self) -> None:
        """Reset overlap buffer and latency history.

        Clears internal state to start fresh processing. Call this when
        starting a new audio stream or after stopping to prevent artifacts
        from previous sessions.
        """
        self.overlap_buffer = None
        self._latency_history.clear()

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from recent chunks.

        Returns:
            Dictionary with min_ms, max_ms, avg_ms keys
        """
        if not self._latency_history:
            return {'min_ms': 0.0, 'max_ms': 0.0, 'avg_ms': 0.0}

        return {
            'min_ms': min(self._latency_history),
            'max_ms': max(self._latency_history),
            'avg_ms': sum(self._latency_history) / len(self._latency_history),
        }

    def start_session(self, speaker_embedding: torch.Tensor) -> None:
        """Start a streaming conversion session.

        Initializes the pipeline for real-time processing with the target
        speaker embedding. Resets all internal buffers and state.

        Args:
            speaker_embedding: Target speaker embedding (256-dim) to convert to
        """
        self._speaker_embedding = speaker_embedding.to(self.device)
        self.reset()
        self.is_running = True

    def stop_session(self) -> None:
        """Stop the streaming conversion session.

        Cleans up session state, clears speaker embedding, and resets buffers.
        Call this when finishing real-time processing to free resources.
        """
        self.is_running = False
        self._speaker_embedding = None
        self.reset()


class AudioInputStream:
    """Audio input stream capture from microphone or audio interface.

    Provides buffered audio capture with callback-based chunk delivery.

    Args:
        sample_rate: Audio sample rate (default: 24000)
        buffer_size: Buffer size in samples (default: 1024)
        device_index: Audio device index (default: None for system default)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        buffer_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device_index = device_index

        self.callback: Optional[Callable[[torch.Tensor], None]] = None
        self._stream = None
        self._is_running = False

    def set_callback(self, callback: Callable[[torch.Tensor], None]) -> None:
        """Set callback function for audio chunk delivery.

        Args:
            callback: Function called with audio tensor for each buffer
        """
        self.callback = callback

    def start(self) -> None:
        """Start audio capture.

        Raises:
            RuntimeError: If no callback is set or audio device unavailable
        """
        if self.callback is None:
            raise RuntimeError("No callback set - call set_callback() first")

        try:
            import sounddevice as sd

            def _audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio input status: {status}")
                audio_tensor = torch.from_numpy(indata[:, 0].copy()).float()
                if self.callback:
                    self.callback(audio_tensor)

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                device=self.device_index,
                channels=1,
                dtype='float32',
                callback=_audio_callback,
            )
            self._stream.start()
            self._is_running = True

        except ImportError:
            raise RuntimeError("sounddevice package required for audio capture")
        except Exception as e:
            raise RuntimeError(f"Failed to start audio capture: {e}")

    def stop(self) -> None:
        """Stop audio capture and release resources.

        Closes the audio stream and frees the audio device. Safe to call
        multiple times.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if audio capture is active.

        Returns:
            True if currently capturing audio, False otherwise
        """
        return self._is_running


class AudioOutputStream:
    """Audio output stream for low-latency playback.

    Provides buffered audio output with minimal latency.

    Args:
        sample_rate: Audio sample rate (default: 24000)
        buffer_size: Buffer size in samples (default: 1024)
        device_index: Audio device index (default: None for system default)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        buffer_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device_index = device_index

        self._stream = None
        self._is_running = False
        self._buffer: List[torch.Tensor] = []

    def write(self, audio: torch.Tensor) -> None:
        """Write audio data to output stream.

        Buffers audio for playback. If stream is running, audio is flushed
        immediately. Otherwise, audio accumulates until start() is called.

        Args:
            audio: Audio tensor of shape (samples,) to play
        """
        self._buffer.append(audio)

        # If stream is running, push to output
        if self._is_running and self._stream is not None:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered audio to output device.

        Concatenates all buffered audio chunks and sends to the audio output
        device for playback. Clears the buffer after flushing. Silently fails
        if sounddevice is not available.
        """
        if not self._buffer:
            return

        try:
            import sounddevice as sd

            # Concatenate buffered audio
            audio = torch.cat(self._buffer, dim=0)
            self._buffer.clear()

            # Convert to numpy and play
            audio_np = audio.cpu().numpy()
            sd.play(audio_np, samplerate=self.sample_rate, device=self.device_index)

        except ImportError:
            pass  # Buffer audio but can't play without sounddevice

    def start(self) -> None:
        """Start audio output stream.

        Enables audio playback. Any buffered audio will be flushed immediately.

        Raises:
            RuntimeError: If sounddevice package is not installed
        """
        try:
            import sounddevice as sd
            self._is_running = True
        except ImportError:
            raise RuntimeError("sounddevice package required for audio output")

    def stop(self) -> None:
        """Stop audio output stream and clear buffer.

        Stops playback and discards any buffered audio. Safe to call
        multiple times.
        """
        self._is_running = False
        self._buffer.clear()

    @property
    def is_running(self) -> bool:
        """Check if output stream is active.

        Returns:
            True if currently playing audio, False otherwise
        """
        return self._is_running
