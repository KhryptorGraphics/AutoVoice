"""High-performance voice synthesis optimized for <100ms latency."""

import time
import threading
import queue
from typing import Optional, Union, Dict, Any, List, Callable
import logging
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceSynthesizer:
    """High-performance voice synthesizer optimized for real-time processing."""
    
    def __init__(self, model, audio_processor, gpu_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize synthesizer with real-time optimizations."""
        self.model = model
        self.audio_processor = audio_processor
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.get_device() if gpu_manager else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Performance settings
        self.latency_target_ms = self.config.get('latency_target_ms', 100)
        self.enable_mixed_precision = self.config.get('mixed_precision', True)
        self.enable_torch_compile = self.config.get('torch_compile', True)
        self.chunk_size = self.config.get('chunk_size', 512)
        
        # Caching and optimization
        self.text_cache = {}
        self.mel_cache = {}
        self.speaker_embeddings = {}
        
        # Real-time processing components
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_processing = False
        
        # Performance tracking
        self.synthesis_times = []
        self.total_syntheses = 0
        
        # CUDA streams for pipeline parallelism
        self.text_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.mel_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.audio_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Pre-allocated buffers
        self.input_buffers = []
        self.output_buffers = []
        self._warmed_up = False

        # Initialize model optimizations
        self._setup_model_optimizations()
        self._init_buffer_pools()
        self._warmup_synthesis()
    
    def text_to_speech(self, text: str, speaker_id: int = 0,
                      speed: float = 1.0, pitch: float = 1.0, 
                      enable_streaming: bool = False, return_metrics: bool = False):
        """Convert text to speech with real-time optimization."""
        start_time = time.time()
        
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("Required libraries not available")
            return np.zeros(22050) if NUMPY_AVAILABLE else []
        
        try:
            # Check cache first for repeated text
            cache_key = f"{text}_{speaker_id}_{speed}_{pitch}"
            if cache_key in self.text_cache and not enable_streaming:
                audio = self.text_cache[cache_key]
                if return_metrics:
                    return audio, {'latency_ms': 0.1, 'cached': True}
                return audio
            
            # Fast path for streaming
            if enable_streaming:
                audio = self._streaming_synthesis(text, speaker_id, speed, pitch)
            else:
                audio = self._standard_synthesis(text, speaker_id, speed, pitch)
            
            # Cache result for small texts
            if len(text) < 100:
                self.text_cache[cache_key] = audio
                # Limit cache size
                if len(self.text_cache) > 100:
                    # Remove oldest entry
                    oldest_key = next(iter(self.text_cache))
                    del self.text_cache[oldest_key]
            
            # Track performance
            synthesis_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(synthesis_time)
            
            if synthesis_time > self.latency_target_ms:
                logger.warning(f"Synthesis time {synthesis_time:.2f}ms exceeds target {self.latency_target_ms}ms")
            
            if return_metrics:
                return audio, {
                    'latency_ms': synthesis_time,
                    'cached': False,
                    'streaming': enable_streaming
                }
            
            return audio
            
        except Exception as e:
            logger.error(f"Text-to-speech synthesis failed: {e}")
            # Return silence on error
            silence = np.zeros(int(self.audio_processor.sample_rate * 0.5))  # 0.5 seconds of silence
            if return_metrics:
                return silence, {'latency_ms': -1, 'error': str(e)}
            return silence

    def synthesize_speech(self, text: str, speaker_id: int = 0, 
                         return_metrics: bool = False) -> Union[np.ndarray, tuple]:
        """Standardized method name for API compatibility with performance tracking.

        This is the method expected by the API blueprint.
        """
        return self.text_to_speech(text=text, speaker_id=speaker_id, return_metrics=return_metrics)
    
    def voice_conversion(self, audio_data,
                        target_speaker_id: int,
                        pitch_shift: float = 1.0):
        """Convert voice to target speaker"""
        # Convert audio to mel spectrogram
        mel_spec = self.audio_processor.to_mel_spectrogram(audio_data)
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spec.T).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            # Generate converted mel spectrogram
            output_mel = self.model(mel_tensor, target_speaker_id)
            
            # Convert to numpy
            output_mel = output_mel.cpu().numpy().squeeze()
            
            # Convert mel to audio
            audio = self.audio_processor.from_mel_spectrogram(output_mel.T)
            
            # Apply pitch shift if needed
            if pitch_shift != 1.0:
                audio = self._adjust_pitch(audio, pitch_shift)
        
        return audio
    
    def extract_speaker_embedding(self, audio):
        """Extract speaker embedding from audio"""
        # Convert audio to mel spectrogram
        mel_spec = self.audio_processor.to_mel_spectrogram(audio)
        
        # In practice, you would use a speaker encoder model
        # For now, return a random embedding
        embedding = np.random.randn(256)
        
        return embedding
    
    def text_to_speech_with_embedding(self, text: str,
                                     speaker_embedding):
        """Synthesize speech with custom speaker embedding"""
        # Similar to text_to_speech but using embedding instead of speaker_id
        mel_length = len(text) * 10
        mel_spec = torch.randn(1, mel_length, 128).to(self.device)
        
        with torch.no_grad():
            # For demonstration, use speaker_id=0
            output_mel = self.model(mel_spec, 0)
            output_mel = output_mel.cpu().numpy().squeeze()
            audio = self.audio_processor.from_mel_spectrogram(output_mel.T)
        
        return audio
    
    def _setup_model_optimizations(self) -> None:
        """Setup model optimizations for latency."""
        if not TORCH_AVAILABLE or not self.model:
            return
            
        # Move model to device and set to eval mode
        if self.device:
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Disable gradient computation globally
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Apply torch.compile for faster inference (PyTorch 2.0+)
        if self.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("Torch compile optimization applied")
            except Exception as e:
                logger.warning(f"Torch compile failed: {e}")
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _init_buffer_pools(self, pool_size: int = 4) -> None:
        """Initialize pre-allocated buffer pools."""
        if not TORCH_AVAILABLE:
            return
            
        # Common buffer shapes
        text_shape = (1, self.chunk_size)
        mel_shape = (1, 80, self.chunk_size)
        
        for _ in range(pool_size):
            self.input_buffers.append(torch.zeros(text_shape, dtype=torch.float32, device=self.device))
            self.output_buffers.append(torch.zeros(mel_shape, dtype=torch.float32, device=self.device))
    
    def _warmup_synthesis(self, warmup_steps: int = 3) -> None:
        """Warmup synthesis pipeline."""
        logger.info("Warming up synthesis pipeline...")
        
        dummy_texts = [
            "Hello world",
            "This is a test sentence",
            "Warming up the voice synthesis system"
        ]
        
        for i, text in enumerate(dummy_texts[:warmup_steps]):
            try:
                _ = self.text_to_speech(text, speaker_id=0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Warmup step {i} failed: {e}")
        
        self._warmed_up = True
        logger.info("Synthesis pipeline warmed up")
    
    def _standard_synthesis(self, text: str, speaker_id: int, speed: float, pitch: float) -> np.ndarray:
        """Standard synthesis pipeline."""
        # Text preprocessing (simplified)
        text_length = len(text)
        mel_length = max(text_length * 10, 128)  # Ensure minimum length
        
        # Generate mel spectrogram (placeholder - real implementation would use text encoder)
        mel_spec = torch.randn(1, mel_length, 80, device=self.device)
        
        # Model inference with mixed precision
        with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
            with torch.no_grad():
                if hasattr(self.model, 'synthesize'):
                    output_mel = self.model.synthesize(mel_spec, speaker_id)
                else:
                    # Fallback to direct model call
                    output_mel = self.model(mel_spec)
        
        # Convert to audio
        if hasattr(output_mel, 'cpu'):
            output_mel = output_mel.cpu().numpy().squeeze()
        
        # Ensure correct shape for audio processor
        if output_mel.ndim == 2:
            output_mel = output_mel.T
        
        audio = self.audio_processor.from_mel_spectrogram(output_mel)
        
        # Apply audio modifications
        if speed != 1.0:
            audio = self._adjust_speed(audio, speed)
        
        if pitch != 1.0:
            audio = self._adjust_pitch(audio, pitch)
        
        return audio
    
    def _streaming_synthesis(self, text: str, speaker_id: int, speed: float, pitch: float) -> np.ndarray:
        """Streaming synthesis for real-time applications."""
        # Split text into chunks for streaming
        chunk_size = self.config.get('streaming_chunk_size', 32)
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        audio_chunks = []
        
        for chunk in text_chunks:
            if not chunk.strip():
                continue
                
            # Process chunk with minimal overhead
            chunk_audio = self._synthesize_chunk(chunk, speaker_id)
            
            if chunk_audio is not None and len(chunk_audio) > 0:
                audio_chunks.append(chunk_audio)
        
        # Concatenate chunks
        if audio_chunks:
            audio = np.concatenate(audio_chunks, axis=0)
            
            # Apply post-processing
            if speed != 1.0:
                audio = self._adjust_speed(audio, speed)
            
            if pitch != 1.0:
                audio = self._adjust_pitch(audio, pitch)
            
            return audio
        
        return np.zeros(int(self.audio_processor.sample_rate * 0.1))  # 0.1 second silence
    
    def _synthesize_chunk(self, text_chunk: str, speaker_id: int) -> Optional[np.ndarray]:
        """Synthesize a small text chunk efficiently."""
        try:
            # Simplified chunk processing
            chunk_length = max(len(text_chunk) * 5, 64)
            mel_spec = torch.randn(1, chunk_length, 80, device=self.device)
            
            with torch.no_grad():
                if hasattr(self.model, 'synthesize'):
                    output_mel = self.model.synthesize(mel_spec, speaker_id)
                else:
                    output_mel = self.model(mel_spec)
            
            if hasattr(output_mel, 'cpu'):
                output_mel = output_mel.cpu().numpy().squeeze()
            
            if output_mel.ndim == 2:
                output_mel = output_mel.T
            
            return self.audio_processor.from_mel_spectrogram(output_mel)
            
        except Exception as e:
            logger.warning(f"Chunk synthesis failed: {e}")
            return None
    
    def _adjust_speed(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Optimized audio speed adjustment."""
        if factor == 1.0:
            return audio
            
        try:
            import scipy.signal
            # Fast resampling
            new_length = int(len(audio) / factor)
            resampled = scipy.signal.resample(audio, new_length)
            return resampled.astype(audio.dtype)
        except ImportError:
            logger.warning("Scipy not available, speed adjustment skipped")
            return audio

    def _adjust_pitch(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Optimized audio pitch adjustment."""
        if factor == 1.0:
            return audio
            
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available, pitch adjustment skipped")
            return audio
            
        try:
            import librosa
            # Efficient pitch shifting
            n_steps = 12 * np.log2(factor)
            pitched_audio = librosa.effects.pitch_shift(
                y=audio,
                sr=self.audio_processor.sample_rate,
                n_steps=n_steps
            )
            return pitched_audio.astype(audio.dtype)
        except Exception as e:
            logger.warning(f"Pitch adjustment failed: {e}")
            return audio
    
    def _update_performance_metrics(self, synthesis_time_ms: float) -> None:
        """Update performance tracking."""
        self.total_syntheses += 1
        self.synthesis_times.append(synthesis_time_ms)
        
        # Keep only recent history
        if len(self.synthesis_times) > 100:
            self.synthesis_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get synthesis performance statistics."""
        if not self.synthesis_times:
            return {'status': 'no_data'}
        
        return {
            'total_syntheses': self.total_syntheses,
            'avg_latency_ms': sum(self.synthesis_times) / len(self.synthesis_times),
            'min_latency_ms': min(self.synthesis_times),
            'max_latency_ms': max(self.synthesis_times),
            'target_latency_ms': self.latency_target_ms,
            'success_rate': sum(1 for t in self.synthesis_times if t <= self.latency_target_ms) / len(self.synthesis_times),
            'cache_size': len(self.text_cache),
            'warmed_up': self._warmed_up
        }
    
    def clear_cache(self) -> None:
        """Clear synthesis caches."""
        self.text_cache.clear()
        self.mel_cache.clear()
        logger.info("Synthesis caches cleared")
    
    def optimize_for_latency(self) -> None:
        """Apply additional latency optimizations."""
        logger.info("Applying synthesis latency optimizations...")
        
        # Clear caches and prepare for optimal performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Re-setup optimizations
        self._setup_model_optimizations()
        
        # Warmup again
        self._warmup_synthesis(warmup_steps=1)
        
        logger.info("Synthesis latency optimizations applied")
