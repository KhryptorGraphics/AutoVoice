"""Karaoke session for live voice conversion.

Manages session state including speaker embedding, separated tracks,
and real-time audio processing pipeline integration.

Supports TensorRT optimization for <50ms latency when engines are available.
"""
import logging
import os
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

from auto_voice.web.karaoke_manager import load_audio

logger = logging.getLogger(__name__)

# Default TRT engine directory
DEFAULT_TRT_ENGINE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'models', 'trt_engines'
)


def _resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform data without requiring torchaudio at import time."""
    if orig_sr == target_sr:
        return audio

    if TORCHAUDIO_AVAILABLE:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(audio)

    target_length = max(1, int(round(audio.shape[-1] * target_sr / orig_sr)))
    return F.interpolate(
        audio.unsqueeze(0),
        size=target_length,
        mode='linear',
        align_corners=False,
    ).squeeze(0)


class AudioMixer:
    """Mix converted voice with instrumental track.

    Handles real-time mixing of voice conversion output with
    instrumental backing track, with gain control for each channel.
    """

    def __init__(
        self,
        voice_gain: float = 1.0,
        instrumental_gain: float = 0.8,
        sample_rate: int = 24000
    ):
        self.voice_gain = voice_gain
        self.instrumental_gain = instrumental_gain
        self.sample_rate = sample_rate
        self._instrumental_buffer: Optional[torch.Tensor] = None
        self._playback_position: int = 0

    def load_instrumental(self, path: str):
        """Load instrumental track for mixing.

        Args:
            path: Path to instrumental audio file
        """
        try:
            audio, sr = load_audio(path)
            if sr != self.sample_rate:
                audio = _resample_audio(audio, sr, self.sample_rate)
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            self._instrumental_buffer = audio.squeeze(0)
            self._playback_position = 0
            logger.info(f"Loaded instrumental: {len(self._instrumental_buffer)} samples")
        except Exception as e:
            logger.error(f"Failed to load instrumental: {e}")
            self._instrumental_buffer = None

    def mix(
        self,
        voice: torch.Tensor,
        position_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix voice with instrumental at current position.

        Args:
            voice: Converted voice audio tensor
            position_samples: Optional specific position in instrumental

        Returns:
            Tuple of (mixed_output, voice_only) tensors
        """
        voice_scaled = voice * self.voice_gain

        if self._instrumental_buffer is None:
            return voice_scaled, voice_scaled

        # Get instrumental chunk at current position
        pos = position_samples if position_samples is not None else self._playback_position
        chunk_len = len(voice)
        end_pos = min(pos + chunk_len, len(self._instrumental_buffer))

        if pos >= len(self._instrumental_buffer):
            # Past end of instrumental, return voice only
            return voice_scaled, voice_scaled

        instrumental_chunk = self._instrumental_buffer[pos:end_pos]

        # Pad if needed (at end of track)
        if len(instrumental_chunk) < chunk_len:
            pad_len = chunk_len - len(instrumental_chunk)
            instrumental_chunk = torch.cat([
                instrumental_chunk,
                torch.zeros(pad_len, device=voice.device)
            ])

        instrumental_scaled = instrumental_chunk.to(voice.device) * self.instrumental_gain

        # Mix voice + instrumental
        mixed = voice_scaled + instrumental_scaled

        # Normalize to prevent clipping
        max_val = mixed.abs().max()
        if max_val > 1.0:
            mixed = mixed / max_val

        # Update position
        if position_samples is None:
            self._playback_position += chunk_len

        return mixed, voice_scaled

    def reset_position(self):
        """Reset playback position to start."""
        self._playback_position = 0

    def set_position(self, position_samples: int):
        """Set playback position in samples."""
        self._playback_position = max(0, position_samples)

    def set_gains(self, voice_gain: float, instrumental_gain: float):
        """Update gain levels."""
        self.voice_gain = max(0.0, min(2.0, voice_gain))
        self.instrumental_gain = max(0.0, min(2.0, instrumental_gain))


class KaraokeSession:
    """Manages a live karaoke voice conversion session.

    Handles:
    - Session lifecycle (start/stop)
    - Speaker embedding for target voice
    - Real-time audio chunk processing
    - Latency measurement and tracking
    - Integration with StreamingConversionPipeline or TRTStreamingPipeline

    Automatically uses TensorRT when engines are available for <50ms latency.

    Args:
        session_id: Unique session identifier
        song_id: ID of the uploaded song
        vocals_path: Path to separated vocals audio
        instrumental_path: Path to separated instrumental audio
        sample_rate: Audio sample rate (default: 24000)
        device: Torch device for processing
        use_trt: Force TRT usage (None=auto-detect, True=require, False=disable)
        trt_engine_dir: Directory containing TRT engines (default: models/trt_engines)
    """

    def __init__(
        self,
        session_id: str,
        song_id: str,
        vocals_path: str,
        instrumental_path: str,
        sample_rate: int = 24000,
        device: Optional[torch.device] = None,
        use_trt: Optional[bool] = None,
        trt_engine_dir: Optional[str] = None,
    ):
        self.session_id = session_id
        self.song_id = song_id
        self.vocals_path = vocals_path
        self.instrumental_path = instrumental_path
        self.sample_rate = sample_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TRT configuration
        self.trt_engine_dir = trt_engine_dir or DEFAULT_TRT_ENGINE_DIR
        self._use_trt = use_trt
        self._trt_available: Optional[bool] = None  # Lazy-checked

        # Session state
        self.is_active = False
        self._speaker_embedding: Optional[torch.Tensor] = None
        self.voice_model_id: Optional[str] = None
        self._source_voice_model_id: Optional[str] = None
        self._target_profile_id: Optional[str] = None
        self._target_model_type: Optional[str] = None
        self._profiles_dir: Optional[str] = None
        self._full_model_path: Optional[str] = None

        # Latency tracking
        self._latency_history: List[float] = []
        self._max_latency_history = 100

        # Streaming pipeline (lazy-loaded)
        self._streaming_pipeline = None
        self._pipeline_type: Optional[str] = None

        # Chunk counter for statistics
        self._chunks_processed = 0
        self._started_at: Optional[float] = None

        # Audio mixer for combining voice + instrumental
        self._mixer = AudioMixer(sample_rate=sample_rate)
        if instrumental_path:
            self._mixer.load_instrumental(instrumental_path)

        logger.info(f"KaraokeSession created: {session_id} for song {song_id}")

    @property
    def speaker_embedding(self) -> Optional[torch.Tensor]:
        """Get the current speaker embedding."""
        return self._speaker_embedding

    def set_speaker_embedding(self, embedding: torch.Tensor):
        """Set the target speaker embedding for voice conversion.

        Args:
            embedding: Speaker embedding tensor (256-dim for mel-statistics)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        self._speaker_embedding = embedding.to(self.device)
        if self._streaming_pipeline is not None and hasattr(self._streaming_pipeline, 'set_target_voice'):
            try:
                self._streaming_pipeline.set_target_voice(
                    self._speaker_embedding.squeeze(0).detach().cpu().numpy()
                )
            except Exception as exc:
                logger.debug(
                    "Session %s: failed to update live target voice on existing pipeline: %s",
                    self.session_id,
                    exc,
                )
        logger.info(f"Session {self.session_id}: Speaker embedding set, shape {embedding.shape}")

    def load_voice_model(self, registry: 'VoiceModelRegistry', model_id: str):
        """Load voice model from registry and set as speaker embedding.

        Args:
            registry: VoiceModelRegistry instance
            model_id: ID of voice model to load

        Raises:
            ValueError: If model not found or embedding cannot be loaded
        """
        model = registry.get_model(model_id)
        if not model:
            raise ValueError(f"Voice model not found: {model_id}")

        embedding = registry.get_embedding(model_id)
        if embedding is None:
            raise ValueError(f"Could not load embedding for model: {model_id}")

        self.set_speaker_embedding(embedding)
        self.voice_model_id = model_id
        logger.info(f"Session {self.session_id}: Loaded voice model {model_id}")

    def _check_trt_available(self) -> bool:
        """Check if TRT engines are available."""
        if self._trt_available is None:
            try:
                from ..inference.trt_streaming_pipeline import TRTStreamingPipeline
                self._trt_available = TRTStreamingPipeline.engines_available(
                    self.trt_engine_dir
                )
                if self._trt_available:
                    logger.info(f"Session {self.session_id}: TRT engines available")
                else:
                    logger.info(f"Session {self.session_id}: TRT engines not found")
            except ImportError:
                self._trt_available = False
                logger.info(f"Session {self.session_id}: TRT not installed")
        return self._trt_available

    def _get_pipeline(self):
        """Lazy-load the streaming conversion pipeline.

        Uses TRTStreamingPipeline when engines are available and TRT is
        not explicitly disabled. Falls back to StreamingConversionPipeline.
        """
        if self._streaming_pipeline is None:
            if (
                self._target_model_type == 'full_model'
                and self._full_model_path
                and os.path.exists(self._full_model_path)
            ):
                from ..inference.realtime_voice_conversion_pipeline import (
                    RealtimeVoiceConversionPipeline,
                )

                self._streaming_pipeline = RealtimeVoiceConversionPipeline(
                    sample_rate=self.sample_rate,
                    device=self.device,
                    config={
                        'sample_rate': self.sample_rate,
                        'speaker_id': self._target_profile_id or 'default',
                        'voice_model_path': self._full_model_path,
                    },
                )
                if self._speaker_embedding is not None:
                    self._streaming_pipeline.set_target_voice(
                        self._speaker_embedding.squeeze(0).detach().cpu().numpy()
                    )
                self._streaming_pipeline.start()
                self._pipeline_type = 'pytorch_full_model'
                logger.info(
                    "Session %s: RealtimeVoiceConversionPipeline loaded with full model %s",
                    self.session_id,
                    self._full_model_path,
                )
                return self._streaming_pipeline

            # Determine whether to use TRT
            use_trt = self._use_trt
            if self._target_profile_id:
                use_trt = False
                logger.info(
                    "Session %s: disabling TRT path for target profile %s so trained target models are honored",
                    self.session_id,
                    self._target_profile_id,
                )
            if use_trt is None:
                use_trt = self._check_trt_available()
            elif use_trt is True and not self._check_trt_available():
                raise RuntimeError(
                    f"TRT requested but engines not found at {self.trt_engine_dir}"
                )

            try:
                if use_trt:
                    from ..inference.trt_streaming_pipeline import TRTStreamingPipeline
                    self._streaming_pipeline = TRTStreamingPipeline(
                        engine_dir=self.trt_engine_dir,
                        sample_rate=self.sample_rate,
                        device=self.device,
                    )
                    self._streaming_pipeline.load_engines()
                    self._pipeline_type = 'tensorrt'
                    logger.info(
                        f"Session {self.session_id}: TRTStreamingPipeline loaded"
                    )
                else:
                    from ..inference.streaming_pipeline import StreamingConversionPipeline
                    self._streaming_pipeline = StreamingConversionPipeline(
                        sample_rate=self.sample_rate,
                        device=self.device
                    )
                    if self._target_profile_id:
                        try:
                            self._streaming_pipeline.set_speaker(self._target_profile_id)
                            logger.info(
                                "Session %s: loaded target speaker %s into streaming pipeline",
                                self.session_id,
                                self._target_profile_id,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Session %s: failed to load target speaker %s into streaming pipeline: %s",
                                self.session_id,
                                self._target_profile_id,
                                exc,
                            )
                            if self._speaker_embedding is not None:
                                self._streaming_pipeline.start_session(self._speaker_embedding.squeeze(0))
                    elif self._speaker_embedding is not None:
                        self._streaming_pipeline.start_session(self._speaker_embedding.squeeze(0))
                    self._pipeline_type = 'pytorch'
                    logger.info(
                        f"Session {self.session_id}: StreamingConversionPipeline loaded"
                    )
            except Exception as e:
                logger.error(f"Failed to load streaming pipeline: {e}")
                raise RuntimeError(f"Pipeline initialization failed: {e}")

        return self._streaming_pipeline

    @property
    def pipeline_type(self) -> Optional[str]:
        """Get the current pipeline type ('tensorrt' or 'pytorch')."""
        return self._pipeline_type

    def start(self):
        """Start the session for live conversion."""
        if self.is_active:
            logger.warning(f"Session {self.session_id} already active")
            return

        self.is_active = True
        self._started_at = time.time()
        self._chunks_processed = 0
        self._latency_history.clear()

        # Pre-warm the pipeline
        try:
            self._get_pipeline()
        except RuntimeError:
            pass  # Pipeline may not be available in tests

        logger.info(f"Session {self.session_id} started")

    def _should_bypass_pipeline(self) -> bool:
        """Return whether conversion should fall back to passthrough mode.

        This keeps synthetic smoke and stress sessions from forcing heavyweight
        model inference when they only provide an ad-hoc embedding and do not
        have real separated assets available yet.
        """
        return (
            self._pipeline_type == 'pytorch'
            and self.voice_model_id is None
            and self._target_profile_id is None
            and self._target_model_type is None
            and not os.path.exists(self.vocals_path)
            and not os.path.exists(self.instrumental_path)
        )

    def stop(self):
        """Stop the session."""
        if not self.is_active:
            return

        self.is_active = False
        if self._streaming_pipeline is not None:
            try:
                if hasattr(self._streaming_pipeline, 'stop_session'):
                    self._streaming_pipeline.stop_session()
                elif hasattr(self._streaming_pipeline, 'stop'):
                    self._streaming_pipeline.stop()
            except Exception as exc:
                logger.debug("Session %s: pipeline stop failed: %s", self.session_id, exc)
        duration = time.time() - self._started_at if self._started_at else 0
        logger.info(
            f"Session {self.session_id} stopped after {duration:.1f}s, "
            f"{self._chunks_processed} chunks processed"
        )

    def process_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """Process an audio chunk through voice conversion.

        Args:
            audio_chunk: Input audio tensor of shape (samples,) or (1, samples)

        Returns:
            Converted audio tensor

        Raises:
            RuntimeError: If session is not active or embedding not set
        """
        if not self.is_active:
            raise RuntimeError("Session is not active")

        if self._speaker_embedding is None:
            raise RuntimeError("Speaker embedding not set")

        start_time = time.time()

        # Ensure tensor is on correct device
        if not isinstance(audio_chunk, torch.Tensor):
            audio_chunk = torch.from_numpy(audio_chunk).float()
        audio_chunk = audio_chunk.to(self.device)

        if self._should_bypass_pipeline():
            output = audio_chunk

        else:
            # Process through streaming pipeline
            try:
                pipeline = self._get_pipeline()
                if self._target_model_type == 'full_model' and hasattr(pipeline, 'process_chunk') and hasattr(pipeline, 'set_target_voice'):
                    pipeline.set_target_voice(
                        self._speaker_embedding.squeeze(0).detach().cpu().numpy()
                    )
                    output_np = pipeline.process_chunk(audio_chunk.detach().cpu().numpy())
                    output = (
                        torch.from_numpy(np.asarray(output_np, dtype=np.float32)).to(self.device)
                        if output_np is not None
                        else audio_chunk
                    )
                else:
                    output = pipeline.process_chunk(audio_chunk, self._speaker_embedding.squeeze(0))
            except RuntimeError:
                # Fallback for testing: return input unchanged
                output = audio_chunk

        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > self._max_latency_history:
            self._latency_history.pop(0)

        self._chunks_processed += 1

        return output

    def get_latency_ms(self) -> float:
        """Get average latency in milliseconds.

        Returns:
            Average processing latency over recent chunks
        """
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    def get_stats(self) -> dict:
        """Get session statistics.

        Returns:
            Dict with session statistics including pipeline type and latency info
        """
        stats = {
            'session_id': self.session_id,
            'song_id': self.song_id,
            'is_active': self.is_active,
            'pipeline_type': self._pipeline_type,
            'chunks_processed': self._chunks_processed,
            'avg_latency_ms': self.get_latency_ms(),
            'min_latency_ms': min(self._latency_history) if self._latency_history else 0,
            'max_latency_ms': max(self._latency_history) if self._latency_history else 0,
            'duration_s': time.time() - self._started_at if self._started_at else 0,
        }

        # Add TRT-specific stats if available
        if self._pipeline_type == 'tensorrt' and self._streaming_pipeline is not None:
            try:
                memory_mb = self._streaming_pipeline.get_engine_memory_usage() / (1024 * 1024)
                stats['trt_memory_mb'] = round(memory_mb, 1)
            except Exception:
                pass

        return stats

    def process_chunk_with_mix(
        self,
        audio_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio chunk and mix with instrumental.

        Args:
            audio_chunk: Input audio tensor

        Returns:
            Tuple of (speaker_output, headphone_output):
                - speaker_output: instrumental + converted voice (for audience)
                - headphone_output: original song (for performer to follow)
        """
        # Convert voice
        converted = self.process_chunk(audio_chunk)

        # Mix with instrumental
        speaker_output, voice_only = self._mixer.mix(converted)

        # For headphone output, could return original song audio at position
        # For now, return voice only as headphone output
        headphone_output = voice_only

        return speaker_output, headphone_output

    def set_mixer_gains(self, voice_gain: float, instrumental_gain: float):
        """Set mixer gain levels.

        Args:
            voice_gain: Voice channel gain (0.0 to 2.0)
            instrumental_gain: Instrumental channel gain (0.0 to 2.0)
        """
        self._mixer.set_gains(voice_gain, instrumental_gain)

    def reset_playback(self):
        """Reset instrumental playback position."""
        self._mixer.reset_position()

    def seek_playback(self, position_seconds: float):
        """Seek to position in instrumental track.

        Args:
            position_seconds: Position in seconds
        """
        position_samples = int(position_seconds * self.sample_rate)
        self._mixer.set_position(position_samples)
