"""Real-time voice conversion pipeline.

Provides low-latency streaming voice conversion for live audio input.
Uses overlapping windows and crossfade for smooth output.
"""
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Callable, Any, Union

import numpy as np
import torch
import torch.nn as nn

from ..models.adapter_manager import AdapterManager, AdapterManagerConfig

logger = logging.getLogger(__name__)


class RealtimeVoiceConversionPipeline:
    """Streaming voice conversion with low-latency processing.

    Uses a ring buffer approach with overlapping windows to provide
    continuous voice conversion with minimal delay.
    """

    def __init__(self, device=None, config: Optional[Dict[str, Any]] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}

        # Audio parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.chunk_size = self.config.get('chunk_size', 4096)  # ~186ms at 22050
        self.hop_size = self.config.get('hop_size', 2048)  # 50% overlap
        self.crossfade_size = self.config.get('crossfade_size', 512)

        # Processing state
        self._running = False
        self._target_embedding = None
        self._input_buffer = deque(maxlen=10)
        self._output_buffer = deque(maxlen=10)
        self._prev_output_tail = None

        # Thread management
        self._process_thread = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_output: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Model manager for real voice conversion
        self._model_manager = None
        self._speaker_id: str = self.config.get('speaker_id', 'default')

        # Consistency student for fast 1-step inference
        self._consistency_student = None
        self._use_consistency = self.config.get('use_consistency', False)

        # Metrics
        self._latency_samples: deque = deque(maxlen=100)
        self._chunks_processed = 0

        # Speaker management
        self._current_speaker_id: Optional[str] = None
        self._adapter_manager = AdapterManager(
            AdapterManagerConfig(device=str(self.device))
        )

        logger.info(f"RealtimeVoiceConversion initialized: "
                    f"chunk={self.chunk_size}, sr={self.sample_rate}, "
                    f"device={self.device}")

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is actively processing."""
        return self._running

    @property
    def latency_ms(self) -> float:
        """Current average processing latency in milliseconds."""
        if not self._latency_samples:
            return 0.0
        return float(np.mean(list(self._latency_samples)) * 1000)

    @property
    def buffer_latency_ms(self) -> float:
        """Theoretical minimum latency from buffer size."""
        return (self.chunk_size / self.sample_rate) * 1000

    def set_target_voice(self, embedding: np.ndarray):
        """Set the target voice embedding for conversion.

        Args:
            embedding: Speaker embedding vector (from VoiceCloner)
        """
        with self._lock:
            self._target_embedding = np.asarray(embedding, dtype=np.float32)
        logger.info("Target voice updated")

    def set_speaker(
        self,
        profile_id: str,
        profiles_dir: Union[str, Path] = "data/voice_profiles",
    ) -> None:
        """Switch to a different speaker by profile ID.

        Loads the speaker embedding from the profile and sets it as the target.
        This provides API consistency with SOTAConversionPipeline.

        Args:
            profile_id: UUID of the voice profile.
            profiles_dir: Directory containing voice profile JSON files.

        Raises:
            FileNotFoundError: If profile doesn't exist.
            ValueError: If profile has no speaker embedding.
        """
        import json

        if profile_id == self._current_speaker_id:
            logger.debug(f"Speaker {profile_id} already set, skipping")
            return

        profiles_path = Path(profiles_dir)
        profile_file = profiles_path / f"{profile_id}.json"

        if not profile_file.exists():
            raise FileNotFoundError(f"Profile not found: {profile_id}")

        with open(profile_file) as f:
            profile_data = json.load(f)

        # Get speaker embedding from profile
        embedding_data = profile_data.get("speaker_embedding")
        if embedding_data is None:
            raise ValueError(f"Profile {profile_id} has no speaker embedding")

        embedding = np.array(embedding_data, dtype=np.float32)

        # Set the target voice
        self.set_target_voice(embedding)
        self._current_speaker_id = profile_id

        logger.info(f"Switched to speaker: {profile_id}")

    def get_current_speaker(self) -> Optional[str]:
        """Get the currently loaded speaker profile ID.

        Returns:
            Profile ID of current speaker, or None if using raw embedding.
        """
        return self._current_speaker_id

    def clear_speaker(self) -> None:
        """Clear the current speaker, stopping voice conversion.

        After calling this, audio will pass through unchanged.
        """
        with self._lock:
            self._target_embedding = None
            self._current_speaker_id = None
        logger.info("Speaker cleared")

    def start(self, on_output: Optional[Callable] = None,
              on_error: Optional[Callable] = None):
        """Start the realtime processing pipeline.

        Args:
            on_output: Callback(audio_chunk: np.ndarray) for processed output
            on_error: Callback(error: Exception) for errors
        """
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._on_output = on_output
        self._on_error = on_error
        self._running = True
        self._chunks_processed = 0
        self._prev_output_tail = None

        self._process_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="realtime-voice-conversion",
        )
        self._process_thread.start()
        logger.info("Realtime pipeline started")

    def stop(self):
        """Stop the processing pipeline."""
        self._running = False
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        # Clear buffers
        self._input_buffer.clear()
        self._output_buffer.clear()
        self._prev_output_tail = None

        logger.info(f"Realtime pipeline stopped. Processed {self._chunks_processed} chunks.")

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process an audio chunk through the conversion pipeline.

        Can be called in two modes:
        1. Push mode: Call with audio, receive output via on_output callback
        2. Pull mode: Call and receive processed audio as return value

        Args:
            audio_chunk: Input audio samples (float32, mono)

        Returns:
            Processed audio chunk if in pull mode, None if using callbacks
        """
        if not self._running:
            return None

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        if self._on_output is not None:
            # Push mode - queue for background processing
            self._input_buffer.append(audio_chunk)
            return None
        else:
            # Pull mode - process synchronously
            return self._convert_chunk(audio_chunk)

    def _processing_loop(self):
        """Background processing loop for push mode."""
        while self._running:
            if self._input_buffer:
                try:
                    chunk = self._input_buffer.popleft()
                    output = self._convert_chunk(chunk)
                    if output is not None and self._on_output:
                        self._on_output(output)
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    if self._on_error:
                        self._on_error(e)
            else:
                time.sleep(0.001)  # 1ms sleep when idle

    def _convert_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Convert a single audio chunk.

        Applies STFT-based conversion with target voice characteristics.
        Uses crossfading for smooth transitions between chunks.
        """
        start_time = time.perf_counter()

        if self._target_embedding is None:
            return audio  # Pass-through if no target set

        try:
            # Store original length before padding
            original_len = len(audio)

            # Pad to chunk_size if needed
            if len(audio) < self.chunk_size:
                audio = np.pad(audio, (0, self.chunk_size - len(audio)))

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)

            with torch.no_grad():
                converted = self._apply_conversion(audio_tensor)

            output = converted.cpu().numpy()

            # Apply crossfade with previous chunk
            if self._prev_output_tail is not None and self.crossfade_size > 0:
                fade_len = min(self.crossfade_size, len(output), len(self._prev_output_tail))
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = 1 - fade_in
                output[:fade_len] = (
                    output[:fade_len] * fade_in +
                    self._prev_output_tail[-fade_len:] * fade_out
                )

            # Save tail for next crossfade
            self._prev_output_tail = output.copy()

            # Track latency
            elapsed = time.perf_counter() - start_time
            self._latency_samples.append(elapsed)
            self._chunks_processed += 1

            return output[:original_len]  # Return original length

        except Exception as e:
            logger.error(f"Chunk conversion error: {e}")
            return audio[:original_len]  # Return original length on error

    def _get_model_manager(self):
        """Get ModelManager instance. Raises if not configured."""
        if self._model_manager is None:
            from .model_manager import ModelManager
            self._model_manager = ModelManager(device=self.device, config=self.config)
            self._model_manager.load(
                hubert_path=self.config.get('hubert_path'),
                vocoder_path=self.config.get('vocoder_path'),
                vocoder_type=self.config.get('vocoder_type', 'hifigan'),
                encoder_backend=self.config.get('encoder_backend', 'hubert'),
                encoder_type=self.config.get('encoder_type', 'linear'),
                conformer_config=self.config.get('conformer_config'),
            )
            voice_model = self.config.get('voice_model_path')
            if voice_model:
                self._model_manager.load_voice_model(voice_model, self._speaker_id)
        return self._model_manager

    def load_consistency_student(self, student_model):
        """Load a ConsistencyStudent for fast 1-step inference.

        Args:
            student_model: A trained ConsistencyStudent instance.

        Raises:
            RuntimeError: If student_model is None.
        """
        if student_model is None:
            raise RuntimeError("ConsistencyStudent model cannot be None")
        self._consistency_student = student_model
        self._consistency_student.to(self.device)
        self._use_consistency = True
        logger.info("ConsistencyStudent loaded for 1-step realtime inference")

    def _apply_conversion(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply voice conversion using trained model.

        Uses consistency student (1-step) if loaded, otherwise uses
        full model manager pipeline. Raises RuntimeError if neither available.
        """
        if self._use_consistency and self._consistency_student is not None:
            return self._apply_consistency_conversion(audio)

        model_manager = self._get_model_manager()
        audio_np = audio.cpu().numpy()
        output = model_manager.infer(
            audio_np, self._speaker_id,
            self._target_embedding, self.sample_rate
        )
        return torch.from_numpy(output).float().to(self.device)

    def _apply_consistency_conversion(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply 1-step voice conversion using consistency student.

        Extracts content+pitch features, constructs conditioning tensor,
        and runs single-step inference through the student model + vocoder.
        """
        from ..models.encoder import ContentEncoder, PitchEncoder

        # Get or create lightweight encoders for feature extraction
        if not hasattr(self, '_content_enc'):
            self._content_enc = ContentEncoder(device=self.device).to(self.device)
        if not hasattr(self, '_pitch_enc'):
            self._pitch_enc = PitchEncoder().to(self.device)

        audio_1d = audio.unsqueeze(0) if audio.dim() == 1 else audio

        # Extract content features
        with torch.no_grad():
            content = self._content_enc.extract_features(audio_1d, sr=self.sample_rate)
            # [1, N, 256]

        # Simple F0 estimation (zero for speed in realtime)
        n_frames = content.shape[1]
        f0 = torch.zeros(1, n_frames, device=self.device)
        with torch.no_grad():
            pitch = self._pitch_enc(f0)  # [1, N, 256]

        # Build conditioning: content + pitch -> project to cond_dim
        cond = torch.cat([
            content.transpose(1, 2),   # [1, 256, N]
            pitch.transpose(1, 2),     # [1, 256, N]
        ], dim=1)  # [1, 512, N]

        # Project to student's expected cond_dim
        if not hasattr(self, '_cond_proj'):
            cond_input_dim = cond.shape[1]
            student_cond_dim = self._consistency_student.student.cond_proj.in_channels
            self._cond_proj = nn.Conv1d(
                cond_input_dim, student_cond_dim, 1
            ).to(self.device)

        with torch.no_grad():
            cond = self._cond_proj(cond)

            # Single-step inference
            mel = self._consistency_student.infer(cond, n_frames=n_frames)
            # [1, n_mels, T]

        # Vocoder: mel -> waveform
        if self._model_manager is not None and self._model_manager._vocoder is not None:
            with torch.no_grad():
                waveform = self._model_manager._vocoder.synthesize(mel)
            return waveform.squeeze(0)

        raise RuntimeError(
            "Vocoder not loaded. Load vocoder via model_manager before "
            "using consistency student for inference."
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        return {
            'is_running': self._running,
            'chunks_processed': self._chunks_processed,
            'avg_latency_ms': self.latency_ms,
            'buffer_latency_ms': self.buffer_latency_ms,
            'input_queue_size': len(self._input_buffer),
            'output_queue_size': len(self._output_buffer),
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'has_target_voice': self._target_embedding is not None,
        }
