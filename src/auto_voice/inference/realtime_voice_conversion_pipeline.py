"""
Real-Time Voice Conversion Streaming Pipeline
Provides live timbre transformation for singing with low-latency processing
"""
import numpy as np
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import deque
import queue

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamingBuffer:
    """Circular buffer for streaming audio processing."""

    def __init__(self, buffer_size: int = 4096, channels: int = 1):
        self.buffer_size = buffer_size
        self.channels = channels
        self.buffer = np.zeros((channels, buffer_size), dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.data_available = 0

    def write(self, audio: np.ndarray) -> int:
        """Write audio data to buffer. Returns number of samples written."""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add channel dimension
        elif audio.ndim == 2 and audio.shape[0] != self.channels:
            if audio.shape[0] == 1 and self.channels == 1:
                pass  # Already correct
            else:
                raise ValueError(f"Audio channels {audio.shape[0]} != buffer channels {self.channels}")

        samples_to_write = min(audio.shape[1], self.buffer_size - self.data_available)

        if samples_to_write <= 0:
            return 0

        # Write to buffer
        for ch in range(self.channels):
            end_pos = (self.write_pos + samples_to_write) % self.buffer_size
            if end_pos > self.write_pos:
                self.buffer[ch, self.write_pos:end_pos] = audio[ch, :samples_to_write]
            else:
                # Wrap around
                first_part = self.buffer_size - self.write_pos
                self.buffer[ch, self.write_pos:] = audio[ch, :first_part]
                self.buffer[ch, :end_pos] = audio[ch, first_part:first_part + end_pos]

        self.write_pos = (self.write_pos + samples_to_write) % self.buffer_size
        self.data_available += samples_to_write

        return samples_to_write

    def read(self, samples: int) -> np.ndarray:
        """Read audio data from buffer. Returns (channels, samples) array."""
        if samples > self.data_available:
            samples = self.data_available

        if samples <= 0:
            return np.zeros((self.channels, 0), dtype=np.float32)

        result = np.zeros((self.channels, samples), dtype=np.float32)

        for ch in range(self.channels):
            end_pos = (self.read_pos + samples) % self.buffer_size
            if end_pos > self.read_pos:
                result[ch, :] = self.buffer[ch, self.read_pos:end_pos]
            else:
                # Wrap around
                first_part = self.buffer_size - self.read_pos
                result[ch, :first_part] = self.buffer[ch, self.read_pos:]
                result[ch, first_part:] = self.buffer[ch, :end_pos]

        self.read_pos = (self.read_pos + samples) % self.buffer_size
        self.data_available -= samples

        return result

    def available_samples(self) -> int:
        """Get number of samples available for reading."""
        return self.data_available

    def clear(self):
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.read_pos = 0
        self.data_available = 0


class RealtimeVoiceConversionPipeline:
    """Streaming pipeline for real-time voice conversion during singing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Audio parameters
        self.sample_rate = config.get('sample_rate', 44100)
        self.channels = config.get('channels', 1)
        self.hop_length = config.get('hop_length', 512)
        self.buffer_size = config.get('buffer_size', 2048)
        self.overlap_samples = config.get('overlap_samples', 512)

        # Processing latency targets (in samples)
        self.target_latency_samples = config.get('target_latency_samples', 2048)  # ~46ms at 44.1kHz

        # Model components (simplified for now - would use real components)
        self.content_encoder = None
        self.pitch_estimator = None
        self.voice_converter = None
        self.vocoder = None

        # Streaming buffers
        self.input_buffer = StreamingBuffer(self.buffer_size * 4, self.channels)
        self.output_buffer = StreamingBuffer(self.buffer_size * 4, self.channels)

        # Processing state
        self.is_streaming = False
        self.thread = None
        self.stop_event = threading.Event()

        # Performance monitoring
        self.latency_stats = []
        self.processing_times = []

        # Quality settings
        self.quality_mode = config.get('quality_mode', 'balanced')  # 'fast', 'balanced', 'quality'

        # Target speaker profile (for voice conversion)
        self.current_target_profile = None

        logger.info(f"Initialized RealtimeVoiceConversionPipeline with {self.quality_mode} quality mode")

    def start_streaming(self, target_profile_id: str = None) -> bool:
        """Start real-time streaming conversion."""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return False

        try:
            # Load/set target profile
            if target_profile_id:
                self.current_target_profile = self._load_target_profile(target_profile_id)
            else:
                self.current_target_profile = None

            # Initialize model components
            self._initialize_models()

            # Clear buffers
            self.input_buffer.clear()
            self.output_buffer.clear()

            # Reset performance stats
            self.latency_stats.clear()
            self.processing_times.clear()

            # Start processing thread
            self.stop_event.clear()
            self.is_streaming = True
            self.thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.thread.start()

            logger.info("Started real-time voice conversion streaming")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self) -> bool:
        """Stop real-time streaming conversion."""
        if not self.is_streaming:
            return True

        try:
            self.stop_event.set()
            self.is_streaming = False

            if self.thread:
                self.thread.join(timeout=1.0)

            # Clean up
            self.input_buffer.clear()
            self.output_buffer.clear()

            logger.info("Stopped real-time voice conversion streaming")
            return True

        except Exception as e:
            logger.error(f"Failed to stop streaming cleanly: {e}")
            return False

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a chunk of audio in real-time.

        Args:
            audio_chunk: Input audio chunk (channels, samples)

        Returns:
            Processed output chunk if available, None if buffer needs more data
        """
        if not self.is_streaming:
            raise RuntimeError("Streaming not active")

        # Write to input buffer
        self.input_buffer.write(audio_chunk)

        # Try to read processed data
        available_samples = self.output_buffer.available_samples()
        if available_samples >= len(audio_chunk[0]) if audio_chunk.ndim > 1 else len(audio_chunk):
            chunk_size = len(audio_chunk[0]) if audio_chunk.ndim > 1 else len(audio_chunk)
            output_chunk = self.output_buffer.read(chunk_size)
            return output_chunk

        return None

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get real-time streaming performance statistics."""
        if not self.processing_times:
            return {'status': 'no_data'}

        stats = {
            'is_streaming': self.is_streaming,
            'input_buffer_samples': self.input_buffer.available_samples(),
            'output_buffer_samples': self.output_buffer.available_samples(),
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            'avg_latency_ms': np.mean(self.latency_stats) * 1000 if self.latency_stats else 0,
            'latency_samples': len(self.latency_stats),
            'target_latency_ms': (self.target_latency_samples / self.sample_rate) * 1000,
            'quality_mode': self.quality_mode,
            'current_target_profile': self.current_target_profile['profile_id'] if self.current_target_profile else None
        }

        return stats

    def update_target_profile(self, profile_id: str) -> bool:
        """Update target speaker profile during streaming."""
        try:
            new_profile = self._load_target_profile(profile_id)

            # Smooth transition (fade out old, fade in new)
            transition_samples = min(1024, self.buffer_size)  # 23ms transition at 44.1kHz

            # For now, just switch immediately (in production, implement crossfading)
            self.current_target_profile = new_profile

            logger.info(f"Switched to target profile: {profile_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update target profile: {e}")
            return False

    def _processing_loop(self):
        """Main real-time processing loop."""
        logger.info("Started real-time processing loop")

        chunk_size = self.buffer_size
        overlap = self.overlap_samples

        # Keep some history for overlap processing
        input_history = deque(maxlen=overlap * 2)

        while not self.stop_event.is_set():
            try:
                # Wait for enough input data
                while self.input_buffer.available_samples() < chunk_size and not self.stop_event.is_set():
                    time.sleep(0.001)  # 1ms sleep

                if self.stop_event.is_set():
                    break

                # Read input chunk
                start_time = time.time()
                input_chunk = self.input_buffer.read(chunk_size)
                if input_chunk.size == 0:
                    continue

                # Add to history for overlapping
                input_history.append(input_chunk)
                if len(input_history) < 2:
                    continue

                # Get overlapped input
                if len(input_history) >= 2:
                    prev_chunk = input_history[-2]
                    curr_chunk = input_history[-1]

                    # Create overlap (this is simplified - real implementation would crossfade)
                    if overlap > 0 and prev_chunk.shape[1] >= overlap:
                        overlapped_input = np.concatenate([
                            prev_chunk[:, -overlap:],
                            curr_chunk[:, :curr_chunk.shape[1]-overlap]
                        ], axis=1)
                    else:
                        overlapped_input = curr_chunk
                else:
                    overlapped_input = input_chunk

                # Process the chunk
                processed_chunk = self._process_audio_chunk_realtime(overlapped_input)

                # Write to output buffer
                if processed_chunk is not None:
                    self.output_buffer.write(processed_chunk)

                # Track performance
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Keep only recent samples
                if len(self.processing_times) > 1000:
                    self.processing_times.pop(0)

                # Estimate latency
                latency_samples = self.input_buffer.available_samples() + self.output_buffer.available_samples()
                latency_time = latency_samples / self.sample_rate
                self.latency_stats.append(latency_time)

                if len(self.latency_stats) > 1000:
                    self.latency_stats.pop(0)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.stop_event.is_set():
                    break

        logger.info("Exited real-time processing loop")

    def _process_audio_chunk_realtime(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio chunk for real-time conversion.

        Args:
            audio_chunk: Input audio chunk (channels, samples)

        Returns:
            Processed audio chunk or None if processing failed
        """
        try:
            if not self.is_streaming or self.current_target_profile is None:
                return audio_chunk  # Pass through unchanged

            # For quality modes, adjust processing depth
            downsample_factor = {'fast': 4, 'balanced': 2, 'quality': 1}[self.quality_mode]

            # Simplified real-time processing pipeline (placeholder for full implementation)

            # Step 1: Extract vocal features (simplified)
            if self.quality_mode != 'fast':
                vocal_features = self._extract_vocal_features(audio_chunk)
            else:
                # Skip feature extraction in fast mode
                vocal_features = None

            # Step 2: Apply voice conversion (simplified transformation)
            if vocal_features is not None:
                converted_audio = self._apply_voice_transformation(audio_chunk, vocal_features)
            else:
                # Simple timbre shift in fast mode
                converted_audio = self._apply_fast_timbre_shift(audio_chunk)

            # Step 3: Post-processing (noise reduction, dynamics)
            if self.quality_mode == 'quality':
                converted_audio = self._apply_post_processing(converted_audio)

            return converted_audio

        except Exception as e:
            logger.error(f"Real-time chunk processing failed: {e}")
            return audio_chunk  # Return original on error

    def _extract_vocal_features(self, audio_chunk: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Extract vocal features for voice conversion."""
        try:
            # Simplified feature extraction (placeholder)
            features = {
                'fundamental_frequency': np.random.rand(audio_chunk.shape[1]) * 200 + 100,  # Mock F0
                'spectral_centroid': np.random.rand(audio_chunk.shape[1]) * 1000 + 2000,    # Mock centroid
                'energy': np.mean(np.abs(audio_chunk), axis=0),                             # Energy
                'mfcc': np.random.rand(13, audio_chunk.shape[1])                             # Mock MFCCs
            }

            # Using placeholder pitch extraction since we can't import the actual components
            # In real implementation, would use src/auto_voice/audio/pitch_extractor.py

            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def _apply_voice_transformation(self, audio_chunk: np.ndarray,
                                   vocal_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply voice transformation using extracted features."""
        try:
            # Simplified voice transformation (placeholder)
            # In real implementation, this would use the voice conversion pipeline

            # Mock timbre transformation using simple filtering
            converted_audio = audio_chunk.copy()

            # Apply basic frequency shifting based on pitch difference
            if 'fundamental_frequency' in vocal_features:
                # Calculate pitch shift (simplified)
                target_f0 = self.current_target_profile.get('vocal_range', {}).get('mean_f0', 220.0)
                current_f0 = np.mean(vocal_features['fundamental_frequency'])
                pitch_ratio = target_f0 / max(current_f0, 1.0)

                # Simple pitch shifting using basic filtering (placeholder)
                if pitch_ratio != 1.0:
                    converted_audio = self._simple_pitch_shift(converted_audio, pitch_ratio)

            return converted_audio

        except Exception as e:
            logger.error(f"Voice transformation failed: {e}")
            return audio_chunk

    def _apply_fast_timbre_shift(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Fast timbre shifting for low-latency mode."""
        # Simple frequency filtering to change timbre
        try:
            # Apply basic high/low cut filters (simplified)
            # In real implementation, would use proper spectral filtering

            # This is a placeholder - actual implementation would use DSP filters
            converted_audio = audio_chunk * 1.05  # Slight amplification as placeholder

            # Clamp to prevent clipping
            converted_audio = np.clip(converted_audio, -1.0, 1.0)

            return converted_audio

        except Exception as e:
            return audio_chunk

    def _apply_post_processing(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply post-processing (noise reduction, dynamics) in quality mode."""
        try:
            # Simplified post-processing (placeholder)
            # In real implementation, would apply noise reduction, limiting, etc.

            processed_audio = audio_chunk.copy()

            # Apply gentle dynamics processing (compression/limiting)
            threshold = 0.8
            ratio = 4.0

            over_threshold = np.abs(processed_audio) > threshold
            processed_audio[over_threshold] = (np.sign(processed_audio[over_threshold]) *
                                             threshold + (processed_audio[over_threshold] -
                                                        np.sign(processed_audio[over_threshold]) * threshold) / ratio)

            return processed_audio

        except Exception as e:
            return audio_chunk

    def _simple_pitch_shift(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Simple pitch shifting using phase vocoder principles (placeholder)."""
        try:
            # This is a very simplified pitch shifting - actual implementation
            # would use proper STFT and phase vocoder
            if ratio == 1.0:
                return audio

            # Simple approximation using basic interpolation
            # This won't sound good but demonstrates the concept
            new_length = int(audio.shape[1] / ratio)
            indices = np.linspace(0, audio.shape[1] - 1, new_length)
            shifted_audio = np.zeros((audio.shape[0], new_length))

            for ch in range(audio.shape[0]):
                shifted_audio[ch] = np.interp(indices, np.arange(audio.shape[1]), audio[ch])

            # Resample back to original length
            final_indices = np.linspace(0, new_length - 1, audio.shape[1])
            final_audio = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                final_audio[ch] = np.interp(final_indices, np.arange(new_length),
                                          shifted_audio[ch])

            return final_audio

        except Exception as e:
            logger.warning(f"Simple pitch shift failed: {e}")
            return audio

    def _initialize_models(self):
        """Initialize model components for streaming."""
        # Placeholder for model initialization
        # In real implementation, would load content encoder, pitch estimator, etc.
        logger.debug("Initializing models for streaming (placeholder)")

    def _load_target_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load target speaker profile (placeholder)."""
        # In real implementation, would load from voice_cloner
        logger.debug(f"Loading target profile: {profile_id} (placeholder)")

        # Return mock profile data
        return {
            'profile_id': profile_id,
            'vocal_range': {
                'mean_f0': 220.0,  # A3 note
                'min_f0': 85.0,
                'max_f0': 1000.0
            },
            'timbre_features': {
                'brightness': 0.7,
                'warmth': 0.6,
                'resonance': 0.8
            }
        }


class AdvancedVocalProcessor:
    """Advanced vocal processing features for next-level voice conversion."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_features = config.get('enabled_features', [
            'emotion_injection',
            'style_transfer',
            'harmonic_enhancement',
            'noise_adaptation'
        ])

        # Model components for advanced features
        self.emotion_model = None
        self.style_transfer_model = None
        self.harmonic_enhancer = None

        logger.info(f"Initialized AdvancedVocalProcessor with features: {self.enabled_features}")

    def inject_emotion(self, audio: np.ndarray, emotion_type: str,
                      intensity: float = 0.5) -> np.ndarray:
        """Inject emotional characteristics into vocal audio."""
        try:
            # Placeholder emotion injection
            logger.info(f"Injecting emotion: {emotion_type} with intensity {intensity}")

            # Simple emotional modifications (placeholder)
            if emotion_type == 'excited':
                # Increase brightness and dynamics
                audio = audio * (1.0 + intensity * 0.3)
                # Add high-frequency emphasis
            elif emotion_type == 'sad':
                # Reduce dynamics, add warmth
                audio = audio * (1.0 - intensity * 0.2)
                # Apply gentle low-pass filtering

            # Always clamp to prevent clipping
            audio = np.clip(audio, -1.0, 1.0)

            return audio

        except Exception as e:
            logger.error(f"Emotion injection failed: {e}")
            return audio

    def transfer_style(self, audio: np.ndarray, style_source: str) -> np.ndarray:
        """Transfer vocal style from a reference source."""
        try:
            logger.info(f"Transferring style from: {style_source}")

            # Placeholder style transfer
            # In real implementation, would extract style features and apply

            return audio
        except Exception as e:
            logger.error(f"Style transfer failed: {e}")
            return audio

    def enhance_harmonics(self, audio: np.ndarray, harmonics_boost: float = 0.3) -> np.ndarray:
        """Enhance harmonic content for richer vocal tone."""
        try:
            # Placeholder harmonic enhancement
            if harmonics_boost > 0:
                logger.info(f"Enhancing harmonics with boost factor {harmonics_boost}")
                # Apply subtle saturation/clipping to increase harmonic content
                audio = np.tanh(audio * (1.0 + harmonics_boost))
                audio = audio / (1.0 + harmonics_boost * 0.5)  # Normalize

            return audio
        except Exception as e:
            logger.error(f"Harmonic enhancement failed: {e}")
            return audio

    def adapt_to_noise(self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Adapt vocal processing to background noise conditions."""
        try:
            if noise_profile is None:
                # Analyze background noise from audio
                noise_profile = self._analyze_background_noise(audio)

            # Placeholder noise adaptation
            logger.info("Adapting to noise conditions")

            return audio
        except Exception as e:
            logger.error(f"Noise adaptation failed: {e}")
            return audio

    def _analyze_background_noise(self, audio: np.ndarray) -> np.ndarray:
        """Analyze background noise profile."""
        # Simple noise analysis (placeholder)
        try:
            # Calculate noise floor using low percentile energy
            energy = np.abs(audio)
            noise_floor = np.percentile(energy, 10)  # 10th percentile as noise estimate

            return np.full_like(audio, noise_floor)
        except Exception as e:
            return np.zeros_like(audio)
