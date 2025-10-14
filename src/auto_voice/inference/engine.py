"""TensorRT inference engine for voice synthesis."""
import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from contextlib import nullcontext

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TensorRTEngine:
    """TensorRT inference engine."""

    def __init__(self, engine_path: str, device_id: int = 0):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available. Please install tensorrt.")

        self.engine_path = engine_path
        self.device_id = device_id

        # TensorRT components
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None

        # Memory buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        # Load engine
        self.load_engine()

    def load_engine(self):
        """Load TensorRT engine from file."""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")

        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Allocate buffers
        self.allocate_buffers()

        logger.info(f"TensorRT engine loaded: {self.engine_path}")

    def allocate_buffers(self):
        """Allocate GPU memory buffers for inputs and outputs."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        # Copy input data to GPU
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data from GPU
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()

        # Reshape output
        output_shape = self.context.get_binding_shape(1)  # Assuming single output
        return self.outputs[0].host.reshape(output_shape)

    def get_input_shape(self) -> tuple:
        """Get input tensor shape."""
        return self.context.get_binding_shape(0)

    def get_output_shape(self) -> tuple:
        """Get output tensor shape."""
        return self.context.get_binding_shape(1)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                inp.device.free()
        if hasattr(self, 'outputs'):
            for out in self.outputs:
                out.device.free()


class HostDeviceMem:
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


class VoiceInferenceEngine:
    """High-level voice synthesis inference engine optimized for <100ms latency."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda:0'))
        
        # Performance tracking
        self.latency_target = config.get('latency_target_ms', 100)
        self.batch_size = config.get('batch_size', 1)
        self.enable_mixed_precision = config.get('mixed_precision', True)
        
        # Model components
        self.encoder_engine = None
        self.decoder_engine = None
        self.vocoder_engine = None

        # Fallback PyTorch models
        self.encoder_model = None
        self.decoder_model = None
        self.vocoder_model = None
        
        # Memory pools for zero-copy operations
        self.input_buffer_pool = []
        self.output_buffer_pool = []
        self.mel_buffer_pool = []
        
        # CUDA streams for pipeline parallelism
        self.text_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.mel_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.audio_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Warmup flags
        self._warmed_up = False
        
        # Performance counters
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.latency_history = []

        # Initialize engines
        self.init_engines()
        
        # Pre-allocate buffers for optimal performance
        self._init_buffer_pools()
        
        # Warmup for consistent performance
        self._warmup_inference()

        logger.info(f"Voice inference engine initialized - Target latency: {self.latency_target}ms")

    def init_engines(self):
        """Initialize TensorRT engines or fallback to PyTorch."""
        engine_dir = Path(self.config.get('engine_dir', 'models/engines'))

        if TRT_AVAILABLE and engine_dir.exists():
            try:
                # Load TensorRT engines
                encoder_path = engine_dir / 'encoder.trt'
                decoder_path = engine_dir / 'decoder.trt'
                vocoder_path = engine_dir / 'vocoder.trt'

                if encoder_path.exists():
                    self.encoder_engine = TensorRTEngine(str(encoder_path))
                if decoder_path.exists():
                    self.decoder_engine = TensorRTEngine(str(decoder_path))
                if vocoder_path.exists():
                    self.vocoder_engine = TensorRTEngine(str(vocoder_path))

                logger.info("TensorRT engines loaded")
            except Exception as e:
                logger.warning(f"Failed to load TensorRT engines: {e}")
                self.load_pytorch_models()
        else:
            logger.info("TensorRT not available, using PyTorch models")
            self.load_pytorch_models()

    def load_pytorch_models(self):
        """Load PyTorch models as fallback."""
        import sys
        from pathlib import Path as PathLib

        # Add models directory to path for imports
        models_dir = PathLib(__file__).parent.parent / 'models'
        sys.path.insert(0, str(models_dir))

        try:
            from transformer import VoiceTransformer
            from hifigan import HiFiGANGenerator
        except ImportError:
            logger.error("Failed to import model classes. Ensure models are in path.")
            return

        model_dir = Path(self.config.get('model_dir', 'models/pytorch'))

        # Model configuration from config
        transformer_config = self.config.get('transformer', {})
        hifigan_config = self.config.get('hifigan', {})

        try:
            # Load VoiceTransformer (encoder/decoder)
            encoder_checkpoint = model_dir / 'transformer.pth'
            if encoder_checkpoint.exists():
                logger.info(f"Loading VoiceTransformer from {encoder_checkpoint}")

                # Instantiate model with config
                self.encoder_model = VoiceTransformer(
                    input_dim=transformer_config.get('input_dim', 80),
                    d_model=transformer_config.get('d_model', 512),
                    n_heads=transformer_config.get('n_heads', 8),
                    n_layers=transformer_config.get('num_layers', 6),
                    d_ff=transformer_config.get('d_ff', 2048),
                    max_seq_len=transformer_config.get('max_seq_len', 1024),
                    dropout=transformer_config.get('dropout', 0.1)
                )

                # Load state dict
                checkpoint = torch.load(encoder_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.encoder_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.encoder_model.load_state_dict(checkpoint)

                self.encoder_model.to(self.device)
                self.encoder_model.eval()
                logger.info("✓ VoiceTransformer loaded successfully")
            else:
                logger.warning(f"Transformer checkpoint not found: {encoder_checkpoint}")

            # Load HiFiGANGenerator (vocoder)
            vocoder_checkpoint = model_dir / 'hifigan.pth'
            if vocoder_checkpoint.exists():
                logger.info(f"Loading HiFiGAN from {vocoder_checkpoint}")

                # Instantiate vocoder with config
                gen_config = hifigan_config.get('generator', {})
                self.vocoder_model = HiFiGANGenerator(
                    mel_channels=gen_config.get('mel_channels', 80),
                    upsample_rates=gen_config.get('upsample_rates', [8, 8, 2, 2]),
                    upsample_kernel_sizes=gen_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
                    upsample_initial_channel=gen_config.get('upsample_initial_channel', 512),
                    resblock_kernel_sizes=gen_config.get('resblock_kernel_sizes', [3, 7, 11]),
                    resblock_dilation_sizes=gen_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
                )

                # Load state dict
                checkpoint = torch.load(vocoder_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.vocoder_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.vocoder_model.load_state_dict(checkpoint)

                self.vocoder_model.to(self.device)
                self.vocoder_model.eval()
                logger.info("✓ HiFiGAN vocoder loaded successfully")
            else:
                logger.warning(f"HiFiGAN checkpoint not found: {vocoder_checkpoint}")

        except Exception as e:
            logger.error(f"Failed to load PyTorch models: {e}")
            raise

    def synthesize_speech(self, text: str, speaker_id: Optional[int] = None,
                         batch_size: int = 1, enable_streaming: bool = False) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            speaker_id: Optional speaker ID for multi-speaker models
            batch_size: Batch size for processing

        Returns:
            Audio waveform as numpy array
        """
        start_time = time.time()
        
        try:
            # Fast path for real-time processing
            if enable_streaming and self._warmed_up:
                return self._streaming_synthesis(text, speaker_id)
            
            # Text preprocessing with padding and batching - optimized
            text_features, attention_mask = self.preprocess_text(text, return_mask=True)

            # Use pre-allocated buffers to avoid memory allocation overhead
            if len(text_features.shape) == 2:
                text_features = np.expand_dims(text_features, 0)
                if attention_mask is not None:
                    attention_mask = np.expand_dims(attention_mask, 0)

            # Pipeline Stage 1: Text Encoding with CUDA stream
            with torch.cuda.stream(self.text_stream) if torch.cuda.is_available() else nullcontext():
                if self.encoder_engine:
                    encoded = self.encoder_engine.infer(text_features)
                elif self.encoder_model:
                    text_tensor = self._get_input_buffer(text_features.shape, text_features.dtype)
                    text_tensor.copy_(torch.from_numpy(text_features))
                    text_tensor = text_tensor.to(self.device, non_blocking=True)
                    
                    # Mixed precision for speed
                    with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                        with torch.no_grad():
                            encoded = self.encoder_model(text_tensor)
                    
                    encoded = encoded.cpu().numpy()
                else:
                    raise RuntimeError("No encoder model available")

            # Pipeline Stage 2: Mel Spectrogram Processing
            mel_spec = encoded

            # Pipeline Stage 3: Vocoding with CUDA stream
            with torch.cuda.stream(self.audio_stream) if torch.cuda.is_available() else nullcontext():
                if self.vocoder_engine:
                    audio = self.vocoder_engine.infer(mel_spec)
                elif self.vocoder_model:
                    mel_tensor = self._get_mel_buffer(mel_spec.shape, mel_spec.dtype)
                    mel_tensor.copy_(torch.from_numpy(mel_spec))
                    mel_tensor = mel_tensor.to(self.device, non_blocking=True)
                    
                    # Ensure correct shape for HiFiGAN: (batch, mel_channels, time_steps)
                    if mel_tensor.dim() == 3 and mel_tensor.shape[2] != mel_tensor.shape[1]:
                        mel_tensor = mel_tensor.transpose(1, 2)
                    
                    # Mixed precision inference
                    with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                        with torch.no_grad():
                            audio_tensor = self.vocoder_model(mel_tensor)
                    
                    audio = audio_tensor.cpu().numpy()
                else:
                    raise RuntimeError("No vocoder model available")

            # Synchronize streams for final output
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Post-processing: denormalization and clipping
            audio = self._postprocess_audio(audio)
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_performance_metrics(inference_time)
            
            if inference_time > self.latency_target:
                logger.warning(f"Inference time {inference_time:.2f}ms exceeds target {self.latency_target}ms")

            return audio
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def _postprocess_audio(self, audio: np.ndarray,
                          denormalize: bool = True,
                          clip_range: tuple = (-1.0, 1.0)) -> np.ndarray:
        """Post-process synthesized audio.

        Args:
            audio: Raw audio output from vocoder
            denormalize: Whether to denormalize from tanh range
            clip_range: Clipping range for audio values

        Returns:
            Post-processed audio
        """
        # Remove batch dimension if present
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)

        # Remove channel dimension if present
        if len(audio.shape) > 1 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        # Clip to range
        audio = np.clip(audio, clip_range[0], clip_range[1])

        return audio

    def preprocess_text(self, text: str, max_length: int = 512,
                       return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Preprocess text input to phoneme/token representation.

        Args:
            text: Input text string
            max_length: Maximum sequence length (for padding)
            return_mask: Whether to return attention mask

        Returns:
            Tokenized text features, optionally with attention mask
        """
        # Simple character-level encoding (placeholder - can be replaced with phoneme conversion)
        # This is a basic implementation; production would use proper phoneme conversion

        # Character to ID mapping (a-z, space, punctuation)
        char_to_id = {' ': 0}
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            char_to_id[c] = i + 1
        for i, c in enumerate('.,!?;:-'):
            char_to_id[c] = i + 27

        # Convert text to lowercase and encode
        text = text.lower()
        char_ids = [char_to_id.get(c, 0) for c in text]

        # Truncate or pad to max_length
        seq_length = len(char_ids)
        attention_mask = None

        if seq_length > max_length:
            char_ids = char_ids[:max_length]
            seq_length = max_length
        elif seq_length < max_length:
            # Create attention mask before padding
            if return_mask:
                attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
            # Pad with zeros
            char_ids = char_ids + [0] * (max_length - seq_length)
        else:
            if return_mask:
                attention_mask = [1] * max_length

        # Convert to numpy arrays
        features = np.array(char_ids, dtype=np.float32)

        if return_mask:
            mask = np.array(attention_mask, dtype=np.float32) if attention_mask else np.ones_like(features)
            return features, mask

        return features

    def _init_buffer_pools(self, pool_size: int = 4) -> None:
        """Initialize pre-allocated buffer pools for zero-copy operations."""
        # Common buffer shapes for optimization
        text_shape = (1, 512)  # Max text sequence length
        mel_shape = (1, 80, 256)  # Typical mel-spectrogram shape
        audio_shape = (1, 22050 * 5)  # 5 seconds at 22kHz
        
        for _ in range(pool_size):
            self.input_buffer_pool.append(torch.zeros(text_shape, dtype=torch.float32, device=self.device))
            self.mel_buffer_pool.append(torch.zeros(mel_shape, dtype=torch.float32, device=self.device))
            self.output_buffer_pool.append(torch.zeros(audio_shape, dtype=torch.float32, device=self.device))
    
    def _get_input_buffer(self, shape: tuple, dtype) -> torch.Tensor:
        """Get a pre-allocated input buffer or create new one."""
        for buf in self.input_buffer_pool:
            if buf.shape[1] >= shape[1]:  # Can accommodate sequence length
                return buf[:, :shape[1]].contiguous()
        # Fallback: create new buffer
        return torch.zeros(shape, dtype=torch.float32, device=self.device)
    
    def _get_mel_buffer(self, shape: tuple, dtype) -> torch.Tensor:
        """Get a pre-allocated mel buffer or create new one."""
        for buf in self.mel_buffer_pool:
            if buf.shape[2] >= shape[2]:  # Can accommodate time dimension
                return buf[:, :shape[1], :shape[2]].contiguous()
        # Fallback: create new buffer
        return torch.zeros(shape, dtype=torch.float32, device=self.device)
    
    def _warmup_inference(self, warmup_steps: int = 5) -> None:
        """Warmup inference pipeline for consistent performance."""
        logger.info("Warming up inference pipeline...")
        
        dummy_text = "This is a warmup sentence for optimal performance."
        
        for i in range(warmup_steps):
            try:
                _ = self.synthesize_speech(dummy_text, speaker_id=0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Warmup step {i} failed: {e}")
        
        self._warmed_up = True
        logger.info("Inference pipeline warmed up")
    
    def _streaming_synthesis(self, text: str, speaker_id: Optional[int] = None) -> np.ndarray:
        """Optimized streaming synthesis for minimal latency."""
        # Chunk text into smaller segments for streaming
        chunk_size = self.config.get('streaming_chunk_size', 32)
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        audio_chunks = []
        
        for chunk in text_chunks:
            # Process each chunk with minimal overhead
            text_features, _ = self.preprocess_text(chunk, max_length=chunk_size, return_mask=False)
            
            # Direct GPU processing without CPU roundtrips
            text_tensor = torch.from_numpy(text_features).unsqueeze(0).to(self.device, non_blocking=True)
            
            with torch.no_grad():
                # Pipeline: text -> mel -> audio
                if self.encoder_model:
                    mel_features = self.encoder_model(text_tensor)
                    if self.vocoder_model:
                        if mel_features.dim() == 3:
                            mel_features = mel_features.transpose(1, 2)
                        audio_chunk = self.vocoder_model(mel_features)
                        audio_chunks.append(audio_chunk.cpu().numpy().squeeze())
        
        # Concatenate chunks
        if audio_chunks:
            audio = np.concatenate(audio_chunks, axis=0)
            return self._postprocess_audio(audio)
        
        return np.array([])
    
    def _update_performance_metrics(self, inference_time_ms: float) -> None:
        """Update performance tracking metrics."""
        self.inference_count += 1
        self.total_inference_time += inference_time_ms
        self.latency_history.append(inference_time_ms)
        
        # Keep only recent history (last 100 inferences)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.latency_history:
            return {'status': 'no_data'}
        
        return {
            'inference_count': self.inference_count,
            'avg_latency_ms': sum(self.latency_history) / len(self.latency_history),
            'min_latency_ms': min(self.latency_history),
            'max_latency_ms': max(self.latency_history),
            'target_latency_ms': self.latency_target,
            'success_rate': sum(1 for t in self.latency_history if t <= self.latency_target) / len(self.latency_history),
            'warmed_up': self._warmed_up
        }
    
    def optimize_for_latency(self) -> None:
        """Apply latency optimizations."""
        logger.info("Applying latency optimizations...")
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set models to inference mode
        if self.encoder_model:
            self.encoder_model.eval()
            for param in self.encoder_model.parameters():
                param.requires_grad = False
        
        if self.vocoder_model:
            self.vocoder_model.eval()
            for param in self.vocoder_model.parameters():
                param.requires_grad = False
        
        logger.info("Latency optimizations applied")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'engines': {
                'encoder': self.encoder_engine is not None,
                'decoder': self.decoder_engine is not None,
                'vocoder': self.vocoder_engine is not None
            },
            'pytorch_models': {
                'encoder': self.encoder_model is not None,
                'decoder': self.decoder_model is not None,
                'vocoder': self.vocoder_model is not None
            },
            'device': str(self.device),
            'tensorrt_available': TRT_AVAILABLE,
            'performance': self.get_performance_stats(),
            'config': {
                'latency_target_ms': self.latency_target,
                'batch_size': self.batch_size,
                'mixed_precision': self.enable_mixed_precision,
                'warmed_up': self._warmed_up
            }
        }
        return info