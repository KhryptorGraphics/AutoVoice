"""
Unified inference manager coordinating all inference engines for optimal <100ms latency.
Manages VoiceInferenceEngine, TensorRTEngine, VoiceSynthesizer, RealtimeProcessor, and CUDAGraphs.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .engine import VoiceInferenceEngine
from .synthesizer import VoiceSynthesizer
from .realtime_processor import RealtimeProcessor, AsyncRealtimeProcessor
from .cuda_graphs import CUDAGraphManager, GraphOptimizedModel

# Try to import TensorRT components with fallback
try:
    from .tensorrt_engine import TensorRTEngine, TensorRTEngineBuilder
    TENSORRT_AVAILABLE = True
except ImportError:
    TensorRTEngine = None
    TensorRTEngineBuilder = None
    TENSORRT_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceManager:
    """
    Unified manager for all inference engines with <100ms latency optimization.
    
    Coordinates:
    - VoiceInferenceEngine: Main inference orchestrator
    - TensorRTEngine: TensorRT optimization
    - VoiceSynthesizer: Text-to-speech synthesis  
    - RealtimeProcessor: Stream processing
    - CUDAGraphs: CUDA graph optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize inference manager with comprehensive configuration."""
        self.config = config
        self.device = torch.device(config.get('device', 'cuda:0'))
        self.latency_target_ms = config.get('latency_target_ms', 100)
        
        # Engine components
        self.voice_engine = None
        self.tensorrt_engines = {}
        self.synthesizer = None
        self.realtime_processor = None
        self.cuda_graph_manager = None
        
        # Performance tracking
        self.total_inferences = 0
        self.successful_inferences = 0
        self.failed_inferences = 0
        self.latency_history = []
        
        # Optimization flags
        self.optimization_mode = config.get('optimization_mode', 'balanced')  # 'speed', 'balanced', 'quality'
        self.enable_tensorrt = config.get('enable_tensorrt', True)
        self.enable_cuda_graphs = config.get('enable_cuda_graphs', True)
        self.enable_realtime = config.get('enable_realtime', True)
        
        # Thread safety
        self._lock = threading.Lock()
        self._initialized = False
        
        logger.info(f"Inference manager initialized - Target latency: {self.latency_target_ms}ms")
    
    def initialize(self) -> bool:
        """Initialize all inference engines."""
        if self._initialized:
            return True
            
        logger.info("Initializing inference engines...")
        start_time = time.time()
        
        try:
            with self._lock:
                # Initialize CUDA graph manager first
                if self.enable_cuda_graphs and torch.cuda.is_available():
                    self.cuda_graph_manager = CUDAGraphManager(self.device)
                    logger.info("CUDA graph manager initialized")
                
                # Initialize main voice inference engine
                self.voice_engine = VoiceInferenceEngine(self.config)
                logger.info("Voice inference engine initialized")
                
                # Initialize TensorRT engines if available
                if self.enable_tensorrt:
                    self._init_tensorrt_engines()
                
                # Initialize synthesizer
                self._init_synthesizer()
                
                # Initialize real-time processor
                if self.enable_realtime:
                    self._init_realtime_processor()
                
                # Apply optimizations based on mode
                self._apply_optimization_mode()
                
                # Comprehensive warmup
                self._warmup_all_engines()
                
                self._initialized = True
                
            init_time = (time.time() - start_time) * 1000
            logger.info(f"All inference engines initialized in {init_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Inference manager initialization failed: {e}")
            return False
    
    def _init_tensorrt_engines(self) -> None:
        """Initialize TensorRT engines for supported models."""
        if not TENSORRT_AVAILABLE:
            logger.info("TensorRT not available, skipping TensorRT initialization")
            return
            
        engine_dir = Path(self.config.get('tensorrt_engine_dir', 'models/tensorrt'))
        
        if not engine_dir.exists():
            logger.info("TensorRT engine directory not found, skipping TensorRT initialization")
            return
        
        # Load available TensorRT engines
        engine_files = list(engine_dir.glob('*.trt'))
        
        for engine_file in engine_files:
            try:
                engine_name = engine_file.stem
                trt_engine = TensorRTEngine(engine_file)
                self.tensorrt_engines[engine_name] = trt_engine
                logger.info(f"TensorRT engine '{engine_name}' loaded")
            except Exception as e:
                logger.warning(f"Failed to load TensorRT engine {engine_file}: {e}")
    
    def _init_synthesizer(self) -> None:
        """Initialize voice synthesizer."""
        # Get models from voice engine or load separately
        if self.voice_engine and hasattr(self.voice_engine, 'vocoder_model'):
            model = self.voice_engine.vocoder_model
        else:
            # Create dummy model for demonstration
            model = torch.nn.Linear(80, 1024) if TORCH_AVAILABLE else None
        
        # Initialize audio processor (simplified)
        audio_processor = SimpleAudioProcessor(self.config.get('sample_rate', 22050))
        
        # Initialize GPU manager (simplified)
        gpu_manager = SimpleGPUManager(self.device)
        
        self.synthesizer = VoiceSynthesizer(
            model=model,
            audio_processor=audio_processor,
            gpu_manager=gpu_manager,
            config=self.config
        )
        logger.info("Voice synthesizer initialized")
    
    def _init_realtime_processor(self) -> None:
        """Initialize real-time processor."""
        if not self.synthesizer or not self.synthesizer.model:
            logger.warning("Cannot initialize real-time processor without synthesizer model")
            return
        
        self.realtime_processor = RealtimeProcessor(
            model=self.synthesizer.model,
            device=str(self.device),
            buffer_size=self.config.get('buffer_size', 1024),
            sample_rate=self.config.get('sample_rate', 22050),
            config=self.config
        )
        logger.info("Real-time processor initialized")
    
    def _apply_optimization_mode(self) -> None:
        """Apply optimizations based on selected mode."""
        logger.info(f"Applying optimization mode: {self.optimization_mode}")
        
        if self.optimization_mode == 'speed':
            # Maximum speed optimizations
            if self.voice_engine:
                self.voice_engine.optimize_for_latency()
            
            if self.synthesizer:
                self.synthesizer.optimize_for_latency()
            
            if self.realtime_processor:
                self.realtime_processor.optimize_for_latency()
                
            logger.info("Speed optimizations applied")
            
        elif self.optimization_mode == 'quality':
            # Quality-focused settings (may increase latency)
            if self.voice_engine:
                self.voice_engine.enable_mixed_precision = False
            
            logger.info("Quality optimizations applied")
            
        else:  # balanced
            # Default balanced settings
            logger.info("Balanced optimizations applied")
    
    def _warmup_all_engines(self, warmup_steps: int = 3) -> None:
        """Comprehensive warmup of all engines."""
        logger.info(f"Warming up all engines with {warmup_steps} steps...")
        
        dummy_text = "This is a warmup sentence for optimal performance."
        
        for step in range(warmup_steps):
            try:
                # Warmup voice synthesis
                if self.synthesizer:
                    _ = self.synthesizer.text_to_speech(dummy_text)
                
                # Warmup real-time processing
                if self.realtime_processor:
                    dummy_audio = np.random.randn(1024).astype(np.float32)
                    _ = self.realtime_processor.process_audio(dummy_audio)
                
                # Synchronize CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            except Exception as e:
                logger.warning(f"Warmup step {step} failed: {e}")
        
        logger.info("All engines warmed up")
    
    def synthesize_speech(self, text: str, speaker_id: int = 0, 
                         priority: str = 'normal') -> Dict[str, Any]:
        """
        Unified speech synthesis with automatic engine selection.
        
        Args:
            text: Text to synthesize
            speaker_id: Speaker identifier
            priority: 'low', 'normal', 'high', 'realtime'
            
        Returns:
            Dictionary with audio data and metadata
        """
        if not self._initialized:
            if not self.initialize():
                return {'error': 'Inference manager not initialized'}
        
        start_time = time.time()
        self.total_inferences += 1
        
        try:
            # Select synthesis method based on priority and available engines
            if priority == 'realtime' and self.realtime_processor:
                # Use real-time processor for ultra-low latency
                audio = self._realtime_synthesis(text, speaker_id)
                method = 'realtime'
            
            elif self.enable_tensorrt and 'synthesizer' in self.tensorrt_engines:
                # Use TensorRT for optimized inference
                audio = self._tensorrt_synthesis(text, speaker_id)
                method = 'tensorrt'
            
            elif self.synthesizer:
                # Use standard synthesizer
                audio, metrics = self.synthesizer.text_to_speech(
                    text, speaker_id, return_metrics=True
                )
                method = 'standard'
            
            else:
                raise RuntimeError("No synthesis engine available")
            
            # Track performance
            total_latency = (time.time() - start_time) * 1000
            self.latency_history.append(total_latency)
            if len(self.latency_history) > 100:
                self.latency_history.pop(0)
            
            # Success
            self.successful_inferences += 1
            
            return {
                'audio': audio,
                'sample_rate': self.config.get('sample_rate', 22050),
                'latency_ms': total_latency,
                'method': method,
                'target_latency_ms': self.latency_target_ms,
                'within_target': total_latency <= self.latency_target_ms
            }
            
        except Exception as e:
            self.failed_inferences += 1
            logger.error(f"Speech synthesis failed: {e}")
            return {'error': str(e)}
    
    def _realtime_synthesis(self, text: str, speaker_id: int) -> np.ndarray:
        """Real-time synthesis using streaming processor."""
        # Convert text to audio chunks for real-time processing
        # This is a simplified implementation
        text_chunks = [text[i:i+32] for i in range(0, len(text), 32)]
        audio_chunks = []
        
        for chunk in text_chunks:
            if chunk.strip():
                # Process each chunk
                chunk_audio = self.synthesizer.text_to_speech(chunk, speaker_id)
                audio_chunks.append(chunk_audio)
        
        return np.concatenate(audio_chunks) if audio_chunks else np.array([])
    
    def _tensorrt_synthesis(self, text: str, speaker_id: int) -> np.ndarray:
        """TensorRT-optimized synthesis."""
        # Use TensorRT engine for synthesis
        trt_engine = self.tensorrt_engines.get('synthesizer')
        if not trt_engine:
            # Fallback to standard synthesis
            return self.synthesizer.text_to_speech(text, speaker_id)
        
        # Prepare inputs for TensorRT
        # This would need to match the specific model's input format
        text_features = self.voice_engine.preprocess_text(text)
        
        # Run TensorRT inference
        results = trt_engine.infer({
            'text_input': text_features,
            'speaker_id': np.array([speaker_id], dtype=np.int32)
        })
        
        # Post-process TensorRT outputs
        audio_output = results.get('audio_output', np.array([]))
        return audio_output.squeeze() if audio_output.ndim > 1 else audio_output
    
    def start_realtime_processing(self) -> bool:
        """Start real-time processing thread."""
        if not self.realtime_processor:
            logger.error("Real-time processor not initialized")
            return False
        
        try:
            self.realtime_processor.start()
            logger.info("Real-time processing started")
            return True
        except Exception as e:
            logger.error(f"Failed to start real-time processing: {e}")
            return False
    
    def stop_realtime_processing(self) -> bool:
        """Stop real-time processing thread."""
        if not self.realtime_processor:
            return True
        
        try:
            self.realtime_processor.stop()
            logger.info("Real-time processing stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop real-time processing: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'manager': {
                'total_inferences': self.total_inferences,
                'successful_inferences': self.successful_inferences,
                'failed_inferences': self.failed_inferences,
                'success_rate': self.successful_inferences / max(self.total_inferences, 1),
                'optimization_mode': self.optimization_mode,
                'target_latency_ms': self.latency_target_ms
            },
            'engines': {
                'voice_engine': self.voice_engine.get_performance_stats() if self.voice_engine else None,
                'synthesizer': self.synthesizer.get_performance_stats() if self.synthesizer else None,
                'realtime_processor': self.realtime_processor.get_performance_stats() if self.realtime_processor else None,
                'tensorrt_engines': {
                    name: engine.get_performance_stats() 
                    for name, engine in self.tensorrt_engines.items()
                },
                'cuda_graphs': self.cuda_graph_manager.get_performance_stats() if self.cuda_graph_manager else None
            }
        }
        
        if self.latency_history:
            stats['manager'].update({
                'avg_latency_ms': sum(self.latency_history) / len(self.latency_history),
                'min_latency_ms': min(self.latency_history),
                'max_latency_ms': max(self.latency_history),
                'within_target_rate': sum(1 for t in self.latency_history if t <= self.latency_target_ms) / len(self.latency_history)
            })
        
        return stats
    
    def optimize_for_latency(self) -> None:
        """Apply comprehensive latency optimizations to all engines."""
        logger.info("Applying comprehensive latency optimizations...")
        
        if self.voice_engine:
            self.voice_engine.optimize_for_latency()
        
        if self.synthesizer:
            self.synthesizer.optimize_for_latency()
        
        if self.realtime_processor:
            self.realtime_processor.optimize_for_latency()
        
        for engine in self.tensorrt_engines.values():
            if hasattr(engine, 'optimize_for_latency'):
                engine.optimize_for_latency()
        
        if self.cuda_graph_manager:
            self.cuda_graph_manager.optimize_for_latency()
        
        # Clear performance history for fresh measurements
        self.latency_history.clear()
        
        logger.info("Comprehensive latency optimizations applied")
    
    def build_tensorrt_engines(self, onnx_models: Dict[str, str]) -> Dict[str, bool]:
        """Build TensorRT engines from ONNX models."""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available, cannot build engines")
            return {name: False for name in onnx_models.keys()}
            
        results = {}
        builder = TensorRTEngineBuilder()
        
        for model_name, onnx_path in onnx_models.items():
            engine_path = Path(self.config.get('tensorrt_engine_dir', 'models/tensorrt')) / f"{model_name}.trt"
            
            logger.info(f"Building TensorRT engine for {model_name}...")
            success = builder.build_from_onnx(
                onnx_path=onnx_path,
                engine_path=engine_path,
                fp16=True,
                optimize_for_latency=True
            )
            
            results[model_name] = success
            
            if success:
                # Load the built engine
                try:
                    trt_engine = TensorRTEngine(engine_path)
                    self.tensorrt_engines[model_name] = trt_engine
                    logger.info(f"TensorRT engine '{model_name}' built and loaded")
                except Exception as e:
                    logger.error(f"Failed to load built engine '{model_name}': {e}")
                    results[model_name] = False
        
        return results
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_realtime_processing()
        except:
            pass


class SimpleAudioProcessor:
    """Simplified audio processor for demonstration."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def from_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to audio (simplified)."""
        # This is a placeholder - real implementation would use vocoder
        audio_length = mel_spec.shape[0] * 256  # Typical hop length
        return np.random.randn(audio_length).astype(np.float32) * 0.1
    
    def to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel spectrogram (simplified)."""
        # This is a placeholder - real implementation would use STFT + mel filters
        mel_length = len(audio) // 256
        return np.random.randn(mel_length, 80).astype(np.float32)


class SimpleGPUManager:
    """Simplified GPU manager for demonstration."""
    
    def __init__(self, device):
        self.device = device
    
    def get_device(self):
        return self.device


# Factory functions for easy instantiation
def create_inference_manager(config: Dict[str, Any]) -> InferenceManager:
    """Create and initialize inference manager."""
    manager = InferenceManager(config)
    manager.initialize()
    return manager


def create_realtime_manager(model_path: str, config: Optional[Dict[str, Any]] = None) -> InferenceManager:
    """Create inference manager optimized for real-time processing."""
    default_config = {
        'latency_target_ms': 50,  # Ultra-low latency
        'optimization_mode': 'speed',
        'enable_realtime': True,
        'enable_cuda_graphs': True,
        'enable_tensorrt': True,
        'mixed_precision': True,
        'buffer_size': 512,
        'sample_rate': 22050
    }
    
    if config:
        default_config.update(config)
    
    return create_inference_manager(default_config)


def create_quality_manager(model_path: str, config: Optional[Dict[str, Any]] = None) -> InferenceManager:
    """Create inference manager optimized for quality."""
    default_config = {
        'latency_target_ms': 200,  # Allow higher latency for quality
        'optimization_mode': 'quality',
        'enable_realtime': False,
        'enable_cuda_graphs': True,
        'enable_tensorrt': True,
        'mixed_precision': False,
        'buffer_size': 2048,
        'sample_rate': 44100
    }
    
    if config:
        default_config.update(config)
    
    return create_inference_manager(default_config)