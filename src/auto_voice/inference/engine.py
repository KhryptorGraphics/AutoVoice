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


# Import the consolidated TensorRTEngine
from .tensorrt_engine import TensorRTEngine, TRT_AVAILABLE as TRT_ENGINE_AVAILABLE

if TRT_AVAILABLE:
    if not TRT_ENGINE_AVAILABLE:
        logger.warning("TensorRT engine module not available - using fallback only")
        TRT_AVAILABLE = False


class VoiceInferenceEngine:
    """High-level voice synthesis inference engine optimized for <100ms latency."""

    def __init__(self, config: Dict[str, Any], mode: str = 'tts'):
        self.config = config
        self.mode = mode  # 'tts' or 'voice_conversion'
        self.device = torch.device(config.get('device', 'cuda:0'))

        # Performance tracking
        self.latency_target = config.get('latency_target_ms', 100)
        self.batch_size = config.get('batch_size', 1)
        self.enable_mixed_precision = config.get('mixed_precision', True)

        # Model components - different for different modes
        if mode == 'tts':
            self.encoder_engine = None
            self.decoder_engine = None
            self.vocoder_engine = None

            # Fallback PyTorch models for TTS
            self.encoder_model = None
            self.decoder_model = None
            self.vocoder_model = None
        elif mode == 'voice_conversion':
            # Voice conversion components
            self.content_encoder_engine = None
            self.pitch_encoder_engine = None
            self.flow_decoder_engine = None
            self.mel_projection_engine = None
            self.vocoder_engine = None

            # Voice conversion PyTorch model (SingingVoiceConverter)
            self.voice_converter_model = None
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'tts' or 'voice_conversion'")
        
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

        if self.mode == 'tts':
            # TTS mode initialization
            if TRT_AVAILABLE and engine_dir.exists():
                try:
                    # Load TensorRT engines for TTS
                    encoder_path = engine_dir / 'encoder.trt'
                    decoder_path = engine_dir / 'decoder.trt'
                    # Use consistent naming: vocoder.engine (not vocoder.trt)
                    vocoder_path = engine_dir / 'vocoder.engine'

                    if encoder_path.exists():
                        self.encoder_engine = TensorRTEngine(str(encoder_path))
                    if decoder_path.exists():
                        self.decoder_engine = TensorRTEngine(str(decoder_path))
                    if vocoder_path.exists():
                        self.vocoder_engine = TensorRTEngine(str(vocoder_path))

                    logger.info("TTS TensorRT engines loaded")
                except Exception as e:
                    logger.warning(f"Failed to load TTS TensorRT engines: {e}")
                    self.load_pytorch_models()
            else:
                logger.info("TensorRT not available, using PyTorch models for TTS")
                self.load_pytorch_models()

        elif self.mode == 'voice_conversion':
            # Voice conversion mode initialization
            # FIXED: Proper config key resolution with fallback chain
            # Try: config.tensorrt.voice_conversion.engine_dir → config.paths.tensorrt_engines → default
            svc_engine_dir = None

            # Try nested tensorrt.voice_conversion.engine_dir first
            if 'tensorrt' in self.config:
                tensorrt_config = self.config['tensorrt']
                if isinstance(tensorrt_config, dict) and 'voice_conversion' in tensorrt_config:
                    vc_config = tensorrt_config['voice_conversion']
                    if isinstance(vc_config, dict) and 'engine_dir' in vc_config:
                        svc_engine_dir = vc_config['engine_dir']
                        logger.info(f"Using engine_dir from config.tensorrt.voice_conversion.engine_dir: {svc_engine_dir}")

            # Fall back to config.paths.tensorrt_engines
            if svc_engine_dir is None:
                if 'paths' in self.config:
                    paths_config = self.config['paths']
                    if isinstance(paths_config, dict) and 'tensorrt_engines' in paths_config:
                        svc_engine_dir = paths_config['tensorrt_engines']
                        logger.info(f"Using engine_dir from config.paths.tensorrt_engines: {svc_engine_dir}")

            # Fall back to default
            if svc_engine_dir is None:
                svc_engine_dir = 'models/engines/voice_conversion'
                logger.info(f"Using default engine_dir: {svc_engine_dir}")

            # Validate engine directory exists
            svc_engine_path = Path(svc_engine_dir)
            if not svc_engine_path.exists():
                logger.warning(f"Engine directory does not exist: {svc_engine_dir}")

            if TRT_AVAILABLE and svc_engine_path.exists():
                try:
                    # Load TensorRT engines for voice conversion
                    content_path = Path(svc_engine_dir) / 'content_encoder.engine'
                    pitch_path = Path(svc_engine_dir) / 'pitch_encoder.engine'
                    flow_path = Path(svc_engine_dir) / 'flow_decoder.engine'
                    mel_path = Path(svc_engine_dir) / 'mel_projection.engine'

                    if content_path.exists():
                        self.content_encoder_engine = TensorRTEngine(str(content_path))
                    if pitch_path.exists():
                        self.pitch_encoder_engine = TensorRTEngine(str(pitch_path))
                    if flow_path.exists():
                        self.flow_decoder_engine = TensorRTEngine(str(flow_path))
                    if mel_path.exists():
                        self.mel_projection_engine = TensorRTEngine(str(mel_path))

                    # FIXED: Vocoder engine uses svc_engine_dir for consistency with other VC engines
                    vocoder_path = Path(svc_engine_dir) / 'vocoder.engine'
                    if vocoder_path.exists():
                        self.vocoder_engine = TensorRTEngine(str(vocoder_path))
                        logger.info(f"Vocoder engine loaded from {vocoder_path}")

                    logger.info(f"Voice conversion TensorRT engines loaded from {svc_engine_dir}")
                except Exception as e:
                    logger.warning(f"Failed to load voice conversion TensorRT engines: {e}")
                    self.load_pytorch_models()
            else:
                logger.info("TensorRT not available, using PyTorch models for voice conversion")
                self.load_pytorch_models()

    def load_pytorch_models(self):
        """Load PyTorch models as fallback."""
        import sys
        from pathlib import Path as PathLib

        # Add models directory to path for imports
        models_dir = PathLib(__file__).parent.parent / 'models'
        sys.path.insert(0, str(models_dir))

        model_dir = Path(self.config.get('model_dir', 'models/pytorch'))

        if self.mode == 'tts':
            self._load_tts_models(model_dir)
        elif self.mode == 'voice_conversion':
            self._load_voice_conversion_models(model_dir)

    def _load_tts_models(self, model_dir):
        """Load TTS models (transformer + vocoder)."""
        try:
            from transformer import VoiceTransformer
            from hifigan import HiFiGANGenerator
        except ImportError:
            logger.error("Failed to import TTS model classes. Ensure models are in path.")
            return

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
            logger.error(f"Failed to load TTS PyTorch models: {e}")
            raise

    def _load_voice_conversion_models(self, model_dir):
        """Load voice conversion models."""
        try:
            # Try importing SingingVoiceConverter - this is our main voice conversion model
            from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter
        except ImportError as e:
            logger.error(f"Failed to import SingingVoiceConverter: {e}")
            return

        svc_config = {
            'latent_dim': self.config.get('latent_dim', 192),
            'mel_channels': self.config.get('mel_channels', 80),
            'singing_voice_converter': {
                'content_encoder': {'type': 'cnn_fallback', 'output_dim': 256},
                'pitch_encoder': {'pitch_dim': 192},
                'speaker_encoder': {'embedding_dim': 256},
                'posterior_encoder': {'hidden_channels': 192},
                'flow_decoder': {'hidden_channels': 192, 'num_flows': 2},
                'vocoder': {'use_vocoder': False}
            }
        }

        try:
            # Load SingingVoiceConverter model
            svc_checkpoint = model_dir / 'singing_voice_converter.pth'
            if svc_checkpoint.exists():
                logger.info(f"Loading SingingVoiceConverter from {svc_checkpoint}")

                # Instantiate model with config
                self.voice_converter_model = SingingVoiceConverter(svc_config)

                # Load state dict
                checkpoint = torch.load(svc_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.voice_converter_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.voice_converter_model.load_state_dict(checkpoint)

                self.voice_converter_model.to(self.device)
                self.voice_converter_model.eval()
                logger.info("✓ SingingVoiceConverter loaded successfully")
            else:
                logger.warning(f"SingingVoiceConverter checkpoint not found: {svc_checkpoint}")

            # Load shared vocoder for voice conversion
            vocoder_checkpoint = model_dir.parent / 'hifigan.pth'  # Shared vocoder
            if vocoder_checkpoint.exists():
                logger.info(f"Loading shared vocoder from {vocoder_checkpoint}")

                try:
                    from hifigan import HiFiGANGenerator

                    gen_config = self.config.get('hifigan', {}).get('generator', {})
                    self.vocoder_model = HiFiGANGenerator(
                        mel_channels=gen_config.get('mel_channels', 80),
                        upsample_rates=gen_config.get('upsample_rates', [8, 8, 2, 2]),
                        upsample_kernel_sizes=gen_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
                        upsample_initial_channel=gen_config.get('upsample_initial_channel', 512),
                        resblock_kernel_sizes=gen_config.get('resblock_kernel_sizes', [3, 7, 11]),
                        resblock_dilation_sizes=gen_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
                    )

                    checkpoint = torch.load(vocoder_checkpoint, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.vocoder_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.vocoder_model.load_state_dict(checkpoint)

                    self.vocoder_model.to(self.device)
                    self.vocoder_model.eval()
                    logger.info("✓ Shared vocoder loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load shared vocoder: {e}")
            else:
                logger.warning(f"Shared vocoder checkpoint not found: {vocoder_checkpoint}")

        except Exception as e:
            logger.error(f"Failed to load voice conversion PyTorch models: {e}")
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
                    encoder_inputs = {'text_features': text_features}
                    encoded = self.encoder_engine.infer(encoder_inputs)['encoded_output']
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
                    vocoder_inputs = {'mel_input': mel_spec}
                    audio = self.vocoder_engine.infer(vocoder_inputs)['audio_output']
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

    def _interpolate_features(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate features along time axis to match target length.

        Args:
            features: Input features with shape [B, C, T]
            target_length: Target time dimension

        Returns:
            Interpolated features with shape [B, C, target_length]
        """
        if features.shape[2] == target_length:
            return features

        # Use linear interpolation along time axis
        from scipy.interpolate import interp1d

        batch_size, channels, current_length = features.shape
        result = np.zeros((batch_size, channels, target_length), dtype=features.dtype)

        # Interpolate each batch and channel
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)

        for b in range(batch_size):
            for c in range(channels):
                f = interp1d(x_old, features[b, c, :], kind='linear', fill_value='extrapolate')
                result[b, c, :] = f(x_new)

        return result

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
    
    def convert_voice(self, source_audio: np.ndarray, source_f0: np.ndarray,
                     target_embedding: Optional[np.ndarray] = None,
                     target_f0: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert voice from source to target characteristics.

        Args:
            source_audio: Source audio waveform (batch, time_steps)
            source_f0: Source F0 contour (batch, time_steps)
            target_embedding: Target speaker embedding (batch, embedding_dim) or style
            target_f0: Target F0 contour (batch, time_steps) - optional

        Returns:
            Converted audio waveform as numpy array
        """
        start_time = time.time()

        if self.mode != 'voice_conversion':
            raise ValueError("Voice conversion is only available in voice_conversion mode")

        try:
            # Ensure proper shapes
            if len(source_audio.shape) == 2:
                batch_size = source_audio.shape[0]
            else:
                batch_size = 1
                source_audio = source_audio[np.newaxis, :]

            if len(source_f0.shape) == 2:
                source_f0 = source_f0[np.newaxis, :]

            # Use target_f0 if provided, otherwise use source_f0 (no pitch conversion)
            if target_f0 is not None:
                if len(target_f0.shape) == 2:
                    target_f0 = target_f0[np.newaxis, :]
            else:
                target_f0 = source_f0

            # Create target embedding if not provided (use neutral/default)
            if target_embedding is None:
                # Create neutral speaker embedding (this would be learned from training data)
                embedding_dim = self.config.get('speaker_embedding_dim', 256)
                target_embedding = np.zeros((batch_size, embedding_dim), dtype=np.float32)
            elif len(target_embedding.shape) == 1:
                target_embedding = target_embedding[np.newaxis, :]

            # Store inputs metadata for proper reshading
            input_audio_shape = source_audio.shape
            input_f0_shape = source_f0.shape

            # Pipeline Stage 1: Content Encoding
            with torch.no_grad():
                if self.content_encoder_engine:
                    # Use TensorRT engine with proper input dict
                    # Ensure shape is (B, T) for audio
                    source_audio_np = source_audio.astype(np.float32)
                    if len(source_audio_np.shape) == 1:
                        source_audio_np = source_audio_np[np.newaxis, :]

                    # ContentEncoder ONNX export now assumes audio is already at 16kHz
                    # Input is audio only (no sample_rate parameter to avoid unsupported ops)
                    inputs = {
                        'input_audio': source_audio_np
                    }
                    content_emb = self.content_encoder_engine.infer(inputs)['content_features']

                    # ContentEncoder output is [B, T, C=256], transpose to [B, C, T]
                    if len(content_emb.shape) == 3 and content_emb.shape[2] == 256:
                        content_emb = content_emb.transpose(0, 2, 1)
                elif self.voice_converter_model:
                    # Use PyTorch model (SingingVoiceConverter has built-in content encoder)
                    audio_tensor = torch.from_numpy(source_audio).to(self.device)
                    content_emb = self.voice_converter_model.content_encoder(audio_tensor)
                    content_emb = content_emb.cpu().numpy()
                else:
                    raise RuntimeError("No content encoder available for voice conversion")

                # Pipeline Stage 2: Pitch Encoding (source and target)
                if self.pitch_encoder_engine:
                    # Use TensorRT engine with proper input dict
                    # Ensure shape is (B, T) for f0
                    source_f0_np = source_f0.astype(np.float32)
                    if len(source_f0_np.shape) == 1:
                        source_f0_np = source_f0_np[np.newaxis, :]

                    # Create voiced mask (boolean mask where f0 > 0)
                    source_voiced_mask_np = (source_f0_np > 0).astype(np.bool_)

                    # Build input dict with ONNX names
                    pitch_inputs = {
                        'f0_input': source_f0_np,
                        'voiced_mask': source_voiced_mask_np
                    }
                    source_pitch_emb = self.pitch_encoder_engine.infer(pitch_inputs)['pitch_features']

                    # PitchEncoder output is [B, T, C=192], transpose to [B, C, T]
                    if len(source_pitch_emb.shape) == 3 and source_pitch_emb.shape[2] == 192:
                        source_pitch_emb = source_pitch_emb.transpose(0, 2, 1)

                    if target_f0 is not None:
                        target_f0_np = target_f0.astype(np.float32)
                        if len(target_f0_np.shape) == 1:
                            target_f0_np = target_f0_np[np.newaxis, :]
                        target_voiced_mask_np = (target_f0_np > 0).astype(np.bool_)

                        target_pitch_inputs = {
                            'f0_input': target_f0_np,
                            'voiced_mask': target_voiced_mask_np
                        }
                        target_pitch_emb = self.pitch_encoder_engine.infer(target_pitch_inputs)['pitch_features']

                        # Transpose target pitch embedding
                        if len(target_pitch_emb.shape) == 3 and target_pitch_emb.shape[2] == 192:
                            target_pitch_emb = target_pitch_emb.transpose(0, 2, 1)
                    else:
                        target_pitch_emb = source_pitch_emb
                else:
                    # Use PyTorch pitch encoder from voice converter model
                    f0_tensor = torch.from_numpy(source_f0).to(self.device)
                    voiced_mask = torch.ones_like(f0_tensor, dtype=torch.bool)
                    source_pitch_emb = self.voice_converter_model.pitch_encoder(f0_tensor, voiced_mask)
                    source_pitch_emb = source_pitch_emb.cpu().numpy()

                    if target_f0 is not None:
                        target_f0_tensor = torch.from_numpy(target_f0).to(self.device)
                        target_voiced_mask = torch.ones_like(target_f0_tensor, dtype=torch.bool)
                        target_pitch_emb = self.voice_converter_model.pitch_encoder(target_f0_tensor, target_voiced_mask)
                        target_pitch_emb = target_pitch_emb.cpu().numpy()
                    else:
                        target_pitch_emb = source_pitch_emb

                # Pipeline Stage 3: Voice Conversion (Flow-based conversion)
                target_emb_tensor = torch.from_numpy(target_embedding).to(self.device)

                if self.flow_decoder_engine:
                    # Use TensorRT engine with proper input dict
                    # Determine target time steps from content embedding (which sets the reference length)
                    T = content_emb.shape[2] if len(content_emb.shape) == 3 else content_emb.shape[1]

                    # Interpolate pitch embeddings to match content length if needed
                    if source_pitch_emb.shape[2] != T:
                        source_pitch_emb = self._interpolate_features(source_pitch_emb, T)
                    if target_pitch_emb.shape[2] != T:
                        target_pitch_emb = self._interpolate_features(target_pitch_emb, T)

                    # Expand target speaker embedding to [B, 256, T]
                    if len(target_embedding.shape) == 2:
                        target_embedding_expanded = np.repeat(target_embedding[:, :, np.newaxis], T, axis=2)
                    else:
                        target_embedding_expanded = target_embedding

                    # Sample latent input for inverse flow (random noise matching latent dim 192)
                    latent_input = np.random.randn(batch_size, self.config.get('latent_dim', 192), T).astype(np.float32)

                    # Build conditioning: [content(256) + target_pitch(192) + speaker(256)] = 704
                    conditioning = np.concatenate([
                        content_emb[:, :256, :T],
                        target_pitch_emb[:, :192, :T],
                        target_embedding_expanded[:, :256, :T]
                    ], axis=1).astype(np.float32)

                    # Create mask (B, 1, T)
                    mask = np.ones((batch_size, 1, T), dtype=np.float32)

                    # Build input dict with ONNX names
                    flow_inputs = {
                        'latent_input': latent_input,
                        'mask': mask,
                        'conditioning': conditioning
                    }

                    # Call TensorRT flow decoder (inverse mode for conversion)
                    converted_latent = self.flow_decoder_engine.infer(flow_inputs)['output_latent']
                else:
                    # Use PyTorch model's flow decoder
                    # FIXED: Align with TensorRT path - sample random latent and build proper conditioning

                    # Determine target time steps from content embedding
                    T = content_emb.shape[2] if len(content_emb.shape) == 3 else content_emb.shape[1]

                    # Convert embeddings to PyTorch tensors
                    content_tensor = torch.from_numpy(content_emb).to(self.device)
                    source_pitch_tensor = torch.from_numpy(source_pitch_emb).to(self.device)
                    target_pitch_tensor = torch.from_numpy(target_pitch_emb).to(self.device)

                    # Interpolate pitch embeddings to match content length if needed
                    if source_pitch_tensor.shape[2] != T:
                        source_pitch_tensor = torch.nn.functional.interpolate(
                            source_pitch_tensor, size=T, mode='linear', align_corners=False
                        )
                    if target_pitch_tensor.shape[2] != T:
                        target_pitch_tensor = torch.nn.functional.interpolate(
                            target_pitch_tensor, size=T, mode='linear', align_corners=False
                        )

                    # Expand target speaker embedding to [B, 256, T]
                    if len(target_emb_tensor.shape) == 2:
                        target_embedding_expanded = target_emb_tensor.unsqueeze(2).expand(-1, -1, T)
                    else:
                        target_embedding_expanded = target_emb_tensor

                    # Sample random latent input for inverse flow: [B, 192, T]
                    temperature = self.config.get('sampling_temperature', 1.0)
                    latent_input = torch.randn(
                        batch_size, self.config.get('latent_dim', 192), T,
                        device=self.device, dtype=torch.float32
                    ) * temperature

                    # Build conditioning: [content(256) + target_pitch(192) + speaker(256)] = [B, 704, T]
                    conditioning = torch.cat([
                        content_tensor[:, :256, :T],
                        target_pitch_tensor[:, :192, :T],
                        target_embedding_expanded[:, :256, :T]
                    ], dim=1)

                    # Create mask [B, 1, T]
                    mask = torch.ones((batch_size, 1, T), dtype=torch.float32, device=self.device)

                    # Convert through flow decoder (inverse mode with proper conditioning)
                    converted_latent = self.voice_converter_model.flow_decoder(
                        latent_input, mask, cond=conditioning, inverse=True
                    ).cpu().numpy()

                # Pipeline Stage 4: Mel Projection (Optional - convert latent to mel)
                if self.mel_projection_engine:
                    mel_inputs = {'latent_input': converted_latent}
                    mel_spectrogram = self.mel_projection_engine.infer(mel_inputs)['mel_output']
                elif hasattr(self.voice_converter_model, 'mel_projection'):
                    converted_latent_tensor = torch.from_numpy(converted_latent).to(self.device)
                    mel_spectrogram = self.voice_converter_model.mel_projection(converted_latent_tensor).cpu().numpy()
                elif hasattr(self.voice_converter_model, 'latent_to_mel'):
                    # Handle both naming conventions (mel_projection and latent_to_mel)
                    converted_latent_tensor = torch.from_numpy(converted_latent).to(self.device)
                    mel_spectrogram = self.voice_converter_model.latent_to_mel(converted_latent_tensor).cpu().numpy()
                else:
                    # If no mel projection, use latent directly (assuming it's mel-like)
                    mel_spectrogram = converted_latent

                # Pipeline Stage 5: Vocoding (convert mel to audio)
                if self.vocoder_engine:
                    vocoder_inputs = {'mel_input': mel_spectrogram}
                    audio_output = self.vocoder_engine.infer(vocoder_inputs)['audio_output']
                elif self.vocoder_model:
                    mel_tensor = torch.from_numpy(mel_spectrogram).to(self.device)

                    # Ensure correct shape for HiFiGAN: (batch, mel_channels, time_steps)
                    if mel_tensor.dim() == 3 and mel_tensor.shape[1] != mel_tensor.shape[2]:
                        mel_tensor = mel_tensor.transpose(1, 2)

                    # Mixed precision inference
                    with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                        audio_tensor = self.vocoder_model(mel_tensor)

                    audio_output = audio_tensor.cpu().numpy()
                else:
                    raise RuntimeError("No vocoder available for voice conversion")

            # Synchronize streams for final output
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Post-processing: denormalization and clipping
            audio_output = self._postprocess_audio(audio_output)

            # Track performance
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_performance_metrics(inference_time)

            if inference_time > self.latency_target:
                logger.warning(f"Voice conversion time {inference_time:.2f}ms exceeds target {self.latency_target}ms")

            return audio_output

        except Exception as e:
            logger.error(f"Voice conversion failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'mode': self.mode,
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

        # TTS mode specific information
        if self.mode == 'tts':
            info['engines'] = {
                'encoder': self.encoder_engine is not None,
                'decoder': self.decoder_engine is not None,
                'vocoder': self.vocoder_engine is not None
            }
            info['pytorch_models'] = {
                'encoder': self.encoder_model is not None,
                'decoder': self.decoder_model is not None,
                'vocoder': self.vocoder_model is not None
            }

        # Voice conversion specific model info
        elif self.mode == 'voice_conversion':
            info['voice_conversion_engines'] = {
                'content_encoder': self.content_encoder_engine is not None,
                'pitch_encoder': self.pitch_encoder_engine is not None,
                'flow_decoder': self.flow_decoder_engine is not None,
                'mel_projection': self.mel_projection_engine is not None,
                'vocoder': self.vocoder_engine is not None
            }
            info['voice_conversion_models'] = {
                'singing_voice_converter': self.voice_converter_model is not None
            }

        return info
