#!/usr/bin/env python3
"""
TensorRT Benchmarking Script for Voice Conversion Models

Comprehensive benchmarking of TensorRT-optimized voice conversion components
including performance, accuracy, and memory usage comparisons.
"""

import argparse
import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import tempfile

import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    logger.warning("TensorRT not available. Install tensorrt for benchmarking.")
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not available. Install onnxruntime for ONNX benchmarking.")
    ORT_AVAILABLE = False

# Fix import path for TensorRTEngine
from src.auto_voice.inference.tensorrt_engine import TensorRTEngine
from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter
from src.auto_voice.models.content_encoder import ContentEncoder
from src.auto_voice.models.pitch_encoder import PitchEncoder
from src.auto_voice.models.flow_decoder import FlowDecoder

# Import TensorRT converter if available
try:
    from src.auto_voice.inference.tensorrt_converter import TensorRTConverter
    CONVERTER_AVAILABLE = True
except ImportError:
    logger.warning("TensorRTConverter not available")
    CONVERTER_AVAILABLE = False


class BenchmarkConfig:
    """Configuration for benchmarking runs."""

    def __init__(self, args):
        self.model_dir = Path(args.model_dir)
        self.output_dir = Path(args.output_dir)
        self.sample_count = args.samples
        self.warmup_runs = args.warmup
        self.audio_lengths = args.audio_lengths or [1.0, 3.0, 5.0, 10.0]  # seconds
        self.batch_sizes = args.batch_sizes or [1]
        self.precision_modes = args.precision_modes or ['fp32', 'fp16']
        self.include_memory = args.include_memory
        self.include_accuracy = args.include_accuracy
        self.device = args.device
        self.seed = args.seed

        # NEW: Build and export configuration
        self.build_engines = getattr(args, 'build_engines', False)
        self.export_onnx = getattr(args, 'export_onnx', False)
        self.int8 = getattr(args, 'int8', False)
        self.calibration_data = getattr(args, 'calibration_data', None)
        self.dynamic_shapes = getattr(args, 'dynamic_shapes', False)
        self.workspace_gb = getattr(args, 'workspace_gb', 2.0)
        self.opset = getattr(args, 'opset', 13)

        # Set random seed for reproducible results
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


class BenchmarkResults:
    """Container for benchmark results."""

    def __init__(self):
        self.results = {}
        self.metadata = {}

    def add_result(self, component: str, precision: str, batch_size: int, audio_length: float,
                   inference_type: str, metrics: Dict[str, Any]) -> None:
        """Add benchmark result."""
        key = f"{component}_{precision}_b{batch_size}_l{audio_length}s_{inference_type}"
        self.results[key] = {
            'component': component,
            'precision': precision,
            'batch_size': batch_size,
            'audio_length': audio_length,
            'inference_type': inference_type,
            'metrics': metrics,
            'timestamp': time.time()
        }

    def save_to_file(self, filepath: Path) -> None:
        """Save results to JSON file."""
        data = {
            'metadata': self.metadata,
            'results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {filepath}")


class VoiceConversionBenchmark:
    """Benchmark suite for voice conversion components."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = BenchmarkResults()

        # Store model paths
        self.model_paths = {
            'pytorch': {},
            'onnx': {},
            'tensorrt': {}
        }

        self._discover_models()

    def _discover_models(self) -> None:
        """Discover available models in the model directory."""
        # Look for PyTorch checkpoints
        pytorch_dir = self.config.model_dir / 'pytorch'
        if pytorch_dir.exists():
            for component in ['content_encoder', 'pitch_encoder', 'flow_decoder', 'singing_voice_converter']:
                model_file = pytorch_dir / f"{component}.pt"
                if model_file.exists():
                    self.model_paths['pytorch'][component] = model_file

        # Look for ONNX models
        onnx_dir = self.config.model_dir / 'onnx'
        if onnx_dir.exists():
            for component in ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection']:
                onnx_file = onnx_dir / f"{component}.onnx"
                if onnx_file.exists():
                    self.model_paths['onnx'][component] = onnx_file

        # Look for TensorRT engines
        trt_dir = self.config.model_dir / 'tensorrt'
        if trt_dir.exists():
            for component in ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection']:
                engine_file = trt_dir / f"{component}.engine"
                if engine_file.exists():
                    self.model_paths['tensorrt'][component] = engine_file

        logger.info(f"Discovered models: PyTorch={len(self.model_paths['pytorch'])}, "
                   f"ONNX={len(self.model_paths['onnx'])}, TensorRT={len(self.model_paths['tensorrt'])}")

    def export_onnx_models(self) -> Dict[str, Path]:
        """Export ONNX models if missing and export flag is set."""
        if not self.config.export_onnx:
            return {}

        logger.info("Exporting ONNX models...")
        exported = {}

        # Create ONNX output directory
        onnx_dir = self.config.model_dir / 'onnx'
        onnx_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load SingingVoiceConverter on CPU for export
            logger.info("Loading SingingVoiceConverter for ONNX export...")
            config = {
                'latent_dim': 192,
                'mel_channels': 80,
                'singing_voice_converter': {
                    'content_encoder': {'type': 'cnn_fallback', 'output_dim': 256},
                    'pitch_encoder': {'pitch_dim': 192},
                    'speaker_encoder': {'embedding_dim': 256},
                    'posterior_encoder': {'hidden_channels': 192},
                    'flow_decoder': {'hidden_channels': 192, 'num_flows': 4},
                    'vocoder': {'use_vocoder': False}
                }
            }

            model = SingingVoiceConverter(config)
            model.eval()

            # Prepare for inference (loads submodules)
            model.prepare_for_inference()

            # Export components to ONNX
            logger.info("Exporting components to ONNX...")
            exported_paths = model.export_components_to_onnx(
                export_dir=str(onnx_dir),
                opset_version=self.config.opset
            )

            # Update model paths
            for component, path in exported_paths.items():
                self.model_paths['onnx'][component] = Path(path)
                exported[component] = Path(path)
                logger.info(f"  Exported {component} to {path}")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

        return exported

    def build_tensorrt_engines(self) -> Dict[str, Path]:
        """Build TensorRT engines from ONNX models."""
        if not self.config.build_engines:
            return {}

        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available - skipping engine build")
            return {}

        if not CONVERTER_AVAILABLE:
            logger.warning("TensorRT converter not available - skipping engine build")
            return {}

        logger.info("Building TensorRT engines...")
        built = {}

        # Create TensorRT output directory
        trt_dir = self.config.model_dir / 'tensorrt'
        trt_dir.mkdir(parents=True, exist_ok=True)

        # Initialize converter
        converter = TensorRTConverter(export_dir=trt_dir)

        # Build engines for each ONNX model
        for component, onnx_path in self.model_paths['onnx'].items():
            logger.info(f"Building TensorRT engine for {component}...")

            try:
                # Define dynamic shapes for each component
                dynamic_shapes = None
                if self.config.dynamic_shapes:
                    dynamic_shapes = self._get_dynamic_shapes(component)

                # Determine precision modes
                fp16 = 'fp16' in self.config.precision_modes
                int8 = self.config.int8

                # Load calibration data if INT8 is enabled
                calibration_npz = None
                if int8 and self.config.calibration_data:
                    calibration_npz = self.config.calibration_data

                # Build engine
                engine_path = converter.optimize_with_tensorrt(
                    onnx_path=str(onnx_path),
                    output_path=str(trt_dir / f"{component}.engine"),
                    fp16=fp16,
                    int8=int8,
                    dynamic_shapes=dynamic_shapes,
                    workspace_size=int(self.config.workspace_gb * (1 << 30)),
                    calibration_npz=calibration_npz,
                    component_name=component
                )

                # Update model paths
                self.model_paths['tensorrt'][component] = Path(engine_path)
                built[component] = Path(engine_path)
                logger.info(f"  Built engine for {component}: {engine_path}")

            except Exception as e:
                logger.error(f"Engine build failed for {component}: {e}")

        return built

    def _get_dynamic_shapes(self, component: str) -> Optional[Dict[str, Tuple[Tuple, Tuple, Tuple]]]:
        """Get dynamic shape specifications for a component."""
        if component == 'content_encoder':
            # Audio input: variable length
            return {
                'input_audio': (
                    (1, 8000),      # min: 0.5s at 16kHz
                    (1, 48000),     # opt: 3s at 16kHz
                    (1, 160000)     # max: 10s at 16kHz
                )
            }
        elif component == 'pitch_encoder':
            # F0 contour: variable frames
            return {
                'f0_input': (
                    (1, 25),        # min: 0.5s at 50Hz
                    (1, 150),       # opt: 3s at 50Hz
                    (1, 500)        # max: 10s at 50Hz
                ),
                'voiced_mask': (
                    (1, 25),
                    (1, 150),
                    (1, 500)
                )
            }
        elif component == 'flow_decoder':
            # Latent and conditioning: variable time dimension
            return {
                'latent_input': (
                    (1, 192, 25),   # min
                    (1, 192, 150),  # opt
                    (1, 192, 500)   # max
                ),
                'mask': (
                    (1, 1, 25),
                    (1, 1, 150),
                    (1, 1, 500)
                ),
                'conditioning': (
                    (1, 704, 25),
                    (1, 704, 150),
                    (1, 704, 500)
                )
            }
        elif component == 'mel_projection':
            # Latent input: variable time dimension
            return {
                'latent_input': (
                    (1, 192, 25),
                    (1, 192, 150),
                    (1, 192, 500)
                )
            }

        return None

    def generate_test_data(self, audio_length: float, batch_size: int = 1) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking."""
        sample_rate = 16000
        num_samples = int(audio_length * sample_rate)

        # Generate synthetic audio
        # Use a mix of sine waves and noise to simulate singing
        t = np.linspace(0, audio_length, num_samples, dtype=np.float32)

        # Fundamental frequency (sine wave oscillation)
        f0_base = 220  # A3 note
        f0_variation = 50 * np.sin(2 * np.pi * 0.5 * t)  # Slow vibrato
        f0_signal = f0_base + f0_variation

        # Generate harmonic series with formants
        harmonics = [1.0, 0.5, 0.25, 0.125]  # Fundamental + harmonics
        formant_freqs = [700, 1200, 2500]  # Vowel formants
        formant_bws = [80, 100, 120]  # Bandwidths

        audio = np.zeros(num_samples, dtype=np.float32)

        # Add harmonics
        for i, amp in enumerate(harmonics):
            freq = f0_signal * (i + 1)
            # Frequency modulation for vibrato
            phase = 2 * np.pi * np.cumsum(freq) / sample_rate
            audio += amp * np.sin(phase)

        # Add formants (resonant peaks)
        for freq, bw in zip(formant_freqs, formant_bws):
            # Simple resonant filter approximation
            omega = 2 * np.pi * freq / sample_rate
            r = 1 - (bw * 2 * np.pi / sample_rate)
            # Impulse response approximation
            ir_len = int(sample_rate * 0.01)  # 10ms
            ir = np.exp(-bw * np.pi * np.arange(ir_len) / sample_rate) * np.cos(omega * np.arange(ir_len))
            ir = ir / np.sum(ir**2)**0.5  # Normalize
            # Convolve (simplified)
            audio = np.convolve(audio, ir[:min(100, len(ir))], mode='same')

        # Add noise floor
        noise_level = 0.01
        audio += noise_level * np.random.randn(num_samples)

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Generate F0 contour
        f0_frames = int(audio_length * 50)  # 50Hz F0 rate
        f0_contour = np.zeros(f0_frames, dtype=np.float32)
        for i in range(f0_frames):
            t_frame = i / 50.0
            if t_frame < audio_length:
                f0_contour[i] = f0_base + 50 * np.sin(2 * np.pi * 0.5 * t_frame)

        # Generate speaker embedding (random for now)
        speaker_emb = np.random.randn(256).astype(np.float32)
        speaker_emb = speaker_emb / np.linalg.norm(speaker_emb)  # Normalize

        return {
            'audio': torch.from_numpy(audio).unsqueeze(0).repeat(batch_size, 1),
            'f0': torch.from_numpy(f0_contour).unsqueeze(0).repeat(batch_size, 1),
            'speaker_emb': torch.from_numpy(speaker_emb).unsqueeze(0).repeat(batch_size, 1),
            'sample_rate': sample_rate,
            'batch_size': batch_size,
            'audio_length': audio_length
        }

    def benchmark_component(self, component: str, inference_type: str, test_data: Dict[str, Any],
                          precision: str = 'fp32') -> Dict[str, Any]:
        """Benchmark a single component."""
        metrics = {
            'inference_time': [],
            'throughput': [],
            'memory_usage': []
        }

        if inference_type == 'pytorch':
            return self._benchmark_pytorch(component, test_data, metrics)
        elif inference_type == 'onnx':
            return self._benchmark_onnx(component, test_data, metrics)
        elif inference_type == 'tensorrt':
            return self._benchmark_tensorrt(component, test_data, metrics, precision)
        else:
            raise ValueError(f"Unknown inference type: {inference_type}")

        return metrics

    def _benchmark_pytorch(self, component: str, test_data: Dict[str, Any],
                          metrics: Dict[str, List]) -> Dict[str, Any]:
        """Benchmark PyTorch component."""
        if component not in self.model_paths['pytorch']:
            return {'error': 'PyTorch model not found'}

        try:
            # Load PyTorch model
            model_path = self.model_paths['pytorch'][component]
            device = self.config.device

            # Load the appropriate model
            if component == 'content_encoder':
                model = ContentEncoder(encoder_type='cnn_fallback')
                model.load_state_dict(torch.load(model_path, map_location=device))
                inputs = (test_data['audio'].to(device), test_data['sample_rate'])
            elif component == 'pitch_encoder':
                model = PitchEncoder()
                model.load_state_dict(torch.load(model_path, map_location=device))
                inputs = (test_data['f0'].to(device), None)
            elif component == 'flow_decoder':
                model = FlowDecoder(in_channels=192, cond_channels=704)
                model.load_state_dict(torch.load(model_path, map_location=device))
                batch_size, time_steps = test_data['f0'].shape
                latent = torch.randn(batch_size, 192, time_steps).to(device)
                mask = torch.ones(batch_size, 1, time_steps).to(device)
                cond = torch.randn(batch_size, 704, time_steps).to(device)
                inputs = (latent, mask, cond, True)
            else:
                return {'error': f'PyTorch benchmark not implemented for {component}'}

            model.to(device)
            model.eval()

            # Warmup runs
            with torch.no_grad():
                for _ in range(self.config.warmup_runs):
                    if component == 'flow_decoder':
                        _ = model(*inputs[:-1], inverse=inputs[-1])
                    else:
                        _ = model(*inputs)

            # Benchmark runs
            start_time = time.time()
            with torch.no_grad():
                for _ in range(self.config.sample_count):
                    if component == 'flow_decoder':
                        _ = model(*inputs[:-1], inverse=inputs[-1])
                    else:
                        _ = model(*inputs)

            # Synchronize if using CUDA
            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / self.config.sample_count
            throughput = self.config.sample_count / total_time

            return {
                'avg_inference_time': avg_time * 1000,  # ms
                'throughput': throughput,  # inferences/sec
                'total_time': total_time,
                'sample_count': self.config.sample_count
            }

        except Exception as e:
            return {'error': str(e)}

    def _benchmark_onnx(self, component: str, test_data: Dict[str, Any],
                        metrics: Dict[str, List]) -> Dict[str, Any]:
        """Benchmark ONNX component."""
        if not ORT_AVAILABLE or component not in self.model_paths['onnx']:
            return {'error': 'ONNX Runtime not available or model not found'}

        try:
            # Load ONNX model
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(self.model_paths['onnx'][component]),
                sess_options,
                providers=providers
            )

            # Prepare inputs based on component
            inputs = self._prepare_component_inputs(component, test_data)

            # Warmup runs
            for _ in range(self.config.warmup_runs):
                session.run(None, inputs)

            # Benchmark runs
            start_time = time.time()
            for _ in range(self.config.sample_count):
                session.run(None, inputs)
            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / self.config.sample_count
            throughput = self.config.sample_count / total_time

            return {
                'avg_inference_time': avg_time * 1000,  # ms
                'throughput': throughput,  # inferences/sec
                'total_time': total_time,
                'sample_count': self.config.sample_count
            }

        except Exception as e:
            return {'error': str(e)}

    def _benchmark_tensorrt(self, component: str, test_data: Dict[str, Any],
                           metrics: Dict[str, List], precision: str) -> Dict[str, Any]:
        """Benchmark TensorRT component."""
        if not TRT_AVAILABLE or component not in self.model_paths['tensorrt']:
            return {'error': 'TensorRT not available or engine not found'}

        try:
            # Load TensorRT engine (import already at top of file)
            engine = TensorRTEngine(str(self.model_paths['tensorrt'][component]))

            # Prepare inputs
            inputs = self._prepare_component_inputs(component, test_data)

            # Warmup runs
            for _ in range(self.config.warmup_runs):
                engine.infer(inputs)

            # Benchmark runs
            start_time = time.time()
            for _ in range(self.config.sample_count):
                engine.infer(inputs)
            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / self.config.sample_count
            throughput = self.config.sample_count / total_time

            # Get engine size
            engine_size = os.path.getsize(str(self.model_paths['tensorrt'][component])) / (1024 * 1024)  # MB

            return {
                'avg_inference_time': avg_time * 1000,  # ms
                'throughput': throughput,  # inferences/sec
                'total_time': total_time,
                'sample_count': self.config.sample_count,
                'engine_size_mb': engine_size
            }

        except Exception as e:
            return {'error': str(e)}

    def _prepare_component_inputs(self, component: str, test_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Prepare inputs for specific component (matches ONNX export signatures)."""
        if component == 'content_encoder':
            # ContentEncoder ONNX export now assumes audio is already at 16kHz
            # No sample_rate input - preprocessing is external
            return {
                'input_audio': test_data['audio'].numpy()
            }
        elif component == 'pitch_encoder':
            # PitchEncoder expects f0_input and voiced_mask
            f0_input = test_data['f0'].numpy()
            voiced_mask = (f0_input > 0).astype(np.bool_)  # Boolean mask where f0 > 0
            return {
                'f0_input': f0_input,
                'voiced_mask': voiced_mask
            }
        elif component == 'flow_decoder':
            batch_size, time_steps = test_data['f0'].shape
            latent_dim = 192  # Standard latent dimension
            cond_dim = 704  # content(256) + pitch(192) + speaker(256)

            # FlowDecoder ONNX export freezes inverse=True internally
            # Only inputs are: latent_input, mask, conditioning (NO inverse)
            return {
                'latent_input': np.random.randn(batch_size, latent_dim, time_steps).astype(np.float32),
                'mask': np.ones((batch_size, 1, time_steps), dtype=np.float32),
                'conditioning': np.random.randn(batch_size, cond_dim, time_steps).astype(np.float32)
            }
        elif component == 'mel_projection':
            batch_size, time_steps = test_data['f0'].shape
            latent_dim = 192

            return {
                'latent_input': np.random.randn(batch_size, latent_dim, time_steps).astype(np.float32)
            }
        else:
            raise ValueError(f"Unknown component: {component}")

    def run_benchmarks(self) -> None:
        """Run complete benchmark suite."""
        logger.info("Starting TensorRT benchmark suite")

        # Store metadata
        self.results.metadata = {
            'config': {
                'model_dir': str(self.config.model_dir),
                'sample_count': self.config.sample_count,
                'warmup_runs': self.config.warmup_runs,
                'audio_lengths': self.config.audio_lengths,
                'batch_sizes': self.config.batch_sizes,
                'precision_modes': self.config.precision_modes
            },
            'system_info': {
                'tensorrt_available': TRT_AVAILABLE,
                'onnx_runtime_available': ORT_AVAILABLE,
                'pytorch_version': torch.__version__
            },
            'discovered_models': {
                'pytorch': [k for k in self.model_paths['pytorch'].keys()],
                'onnx': [k for k in self.model_paths['onnx'].keys()],
                'tensorrt': [k for k in self.model_paths['tensorrt'].keys()]
            }
        }

        # Export ONNX models if requested and missing
        if self.config.export_onnx or (self.config.build_engines and not self.model_paths['onnx']):
            logger.info("Exporting ONNX models (required for engine building or explicitly requested)...")
            exported = self.export_onnx_models()
            if exported:
                if 'build_info' not in self.results.metadata:
                    self.results.metadata['build_info'] = {}
                self.results.metadata['build_info'].update({
                    'onnx_exported': list(exported.keys()),
                    'export_timestamp': time.time(),
                    'opset_version': self.config.opset
                })
                logger.info(f"Successfully exported {len(exported)} ONNX models")

        # Build TensorRT engines if requested
        if self.config.build_engines:
            logger.info("Building TensorRT engines...")
            built = self.build_tensorrt_engines()
            if built:
                if 'build_info' not in self.results.metadata:
                    self.results.metadata['build_info'] = {}
                self.results.metadata['build_info'].update({
                    'engines_built': list(built.keys()),
                    'precision_modes': self.config.precision_modes,
                    'int8_enabled': self.config.int8,
                    'dynamic_shapes': self.config.dynamic_shapes,
                    'workspace_gb': self.config.workspace_gb,
                    'calibration_data': self.config.calibration_data if self.config.int8 else None,
                    'build_timestamp': time.time()
                })
                logger.info(f"Successfully built {len(built)} TensorRT engines")

                # Re-discover models to include newly built engines
                self._discover_models()
                self.results.metadata['discovered_models'] = {
                    'pytorch': [k for k in self.model_paths['pytorch'].keys()],
                    'onnx': [k for k in self.model_paths['onnx'].keys()],
                    'tensorrt': [k for k in self.model_paths['tensorrt'].keys()]
                }

        # Benchmark each component and configuration
        components = ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection']

        for component in components:
            for precision in self.config.precision_modes:
                for batch_size in self.config.batch_sizes:
                    for audio_length in self.config.audio_lengths:

                        logger.info(f"Benchmarking {component} - {precision} - batch{batch_size} - {audio_length}s")

                        # Generate test data
                        test_data = self.generate_test_data(audio_length, batch_size)

                        # Benchmark different inference types (include PyTorch baseline)
                        for inference_type in ['pytorch', 'onnx', 'tensorrt']:
                            if inference_type in self.model_paths and component in self.model_paths[inference_type]:
                                logger.info(f"  Running {inference_type} benchmark")

                                metrics = self.benchmark_component(
                                    component, inference_type, test_data, precision
                                )

                                if 'error' not in metrics:
                                    self.results.add_result(
                                        component, precision, batch_size, audio_length,
                                        inference_type, metrics
                                    )
                                else:
                                    logger.warning(f"    {inference_type} failed: {metrics['error']}")

        # Save results
        results_file = self.config.output_dir / "benchmark_results.json"
        self.results.save_to_file(results_file)

        # Generate accuracy comparison if enabled
        if self.config.include_accuracy:
            self._generate_accuracy_comparison()

        logger.info("Benchmark suite completed")

    def export_and_build_engines(self, model_config: Dict[str, Any]) -> Dict[str, Path]:
        """
        Export PyTorch models to ONNX and build TensorRT engines.

        This provides end-to-end workflow from PyTorch → ONNX → TensorRT engine.
        Useful for benchmarking the complete optimization pipeline.

        Args:
            model_config: Configuration dict for SingingVoiceConverter model

        Returns:
            Dict mapping component names to built engine paths
        """
        logger.info("Starting end-to-end ONNX export and TensorRT engine building")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter

        # Create temporary directories
        onnx_dir = self.config.output_dir / "onnx_exports"
        engine_dir = self.config.output_dir / "tensorrt_engines"
        onnx_dir.mkdir(exist_ok=True)
        engine_dir.mkdir(exist_ok=True)

        # Initialize model
        model = SingingVoiceConverter(model_config)
        model.eval()

        # Initialize converter
        converter = TensorRTConverter(export_dir=onnx_dir)

        # Track results
        engine_paths = {}
        export_metrics = {}
        build_metrics = {}

        # Export components to ONNX
        logger.info("Exporting components to ONNX...")
        onnx_start = time.time()
        onnx_paths = model.export_components_to_onnx(export_dir=str(onnx_dir))
        onnx_time = time.time() - onnx_start
        export_metrics['total_onnx_export_time'] = onnx_time
        logger.info(f"  ONNX export completed in {onnx_time:.2f}s")

        # Build TensorRT engines for each component
        for component, onnx_path in onnx_paths.items():
            if component == 'mel_projection':
                # Skip mel_projection for now as it's a simple linear layer
                continue

            logger.info(f"Building TensorRT engine for {component}...")

            # Build engine with FP16 precision
            build_start = time.time()
            try:
                engine_path = converter.optimize_with_tensorrt(
                    onnx_path=Path(onnx_path),
                    engine_name=f"{component}_fp16",
                    precision='fp16',
                    workspace_size=1 << 30,  # 1GB
                    component_name=component
                )
                build_time = time.time() - build_start

                # Get engine size
                engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)

                # Store results
                engine_paths[component] = engine_path
                build_metrics[component] = {
                    'build_time_seconds': build_time,
                    'engine_size_mb': engine_size_mb,
                    'precision': 'fp16',
                    'onnx_path': str(onnx_path),
                    'engine_path': str(engine_path)
                }

                logger.info(f"  {component} engine built in {build_time:.2f}s, size: {engine_size_mb:.2f}MB")

            except Exception as e:
                logger.error(f"  Failed to build {component} engine: {e}")
                build_metrics[component] = {'error': str(e)}

        # Save metrics
        metrics_file = self.config.output_dir / "engine_build_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'export_metrics': export_metrics,
                'build_metrics': build_metrics,
                'timestamp': time.time()
            }, f, indent=2, default=str)

        logger.info(f"Engine build metrics saved to {metrics_file}")

        return engine_paths

    def create_int8_calibration_dataset(self, num_samples: int = 100) -> Path:
        """
        Create INT8 calibration dataset for quantization.

        Generates synthetic calibration data that covers the expected
        range of inputs for voice conversion components.

        Args:
            num_samples: Number of calibration samples to generate

        Returns:
            Path to saved calibration NPZ file
        """
        logger.info(f"Creating INT8 calibration dataset with {num_samples} samples")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter

        # Create mock dataset
        class MockDataset:
            def __init__(self, num_samples: int):
                self.samples = []
                for i in range(num_samples):
                    # Generate diverse audio lengths (1-5 seconds)
                    audio_length = np.random.uniform(1.0, 5.0)
                    num_audio_samples = int(audio_length * 16000)

                    # Generate audio with varied characteristics
                    audio = self._generate_varied_audio(num_audio_samples)

                    # Generate F0 contour
                    f0_length = int(audio_length * 50)  # 50Hz F0 rate
                    f0 = self._generate_varied_f0(f0_length)

                    self.samples.append(type('MockSample', (), {
                        'source_audio': audio.astype(np.float32),
                        'source_f0': f0.astype(np.float32)
                    })())

            def _generate_varied_audio(self, length: int) -> np.ndarray:
                """Generate audio with varied spectral characteristics."""
                t = np.arange(length) / 16000.0

                # Vary fundamental frequency and harmonics
                f0 = np.random.uniform(80, 400)  # Random pitch
                audio = np.zeros(length)

                # Add harmonics with random amplitudes
                for harmonic in range(1, 6):
                    amplitude = np.random.uniform(0.1, 1.0) / harmonic
                    audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)

                # Add noise
                audio += np.random.randn(length) * 0.05

                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8

                return audio

            def _generate_varied_f0(self, length: int) -> np.ndarray:
                """Generate F0 contour with varied characteristics."""
                # Random base pitch
                base_f0 = np.random.uniform(100, 300)

                # Add vibrato
                t = np.arange(length) / 50.0
                vibrato_rate = np.random.uniform(4, 7)  # Hz
                vibrato_depth = np.random.uniform(5, 20)  # Hz

                f0 = base_f0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)

                # Add pitch drift
                drift = np.random.randn(length).cumsum() * 0.5
                f0 = f0 + drift

                # Clip to valid range
                f0 = np.clip(f0, 80, 1000)

                return f0

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        # Create dataset
        dataset = MockDataset(num_samples)

        # Save calibration data
        converter = TensorRTConverter(export_dir=self.config.output_dir / "calibration")
        output_path = self.config.output_dir / "int8_calibration_data.npz"

        npz_path = converter.create_calibration_dataset(
            dataset=dataset,
            num_samples=num_samples,
            output_path=str(output_path)
        )

        logger.info(f"Calibration dataset saved to {npz_path}")

        # Log dataset statistics
        with np.load(npz_path) as data:
            logger.info("Calibration dataset statistics:")
            for key in data.keys():
                arr = data[key]
                logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
                          f"range=[{arr.min():.3f}, {arr.max():.3f}]")

        return Path(npz_path)

    def benchmark_end_to_end_latency(self, engine_paths: Dict[str, Path]) -> Dict[str, Any]:
        """
        Benchmark end-to-end latency for complete voice conversion pipeline.

        Tests the full pipeline: audio → content encoding → pitch encoding →
        flow decoding → mel projection (if available).

        Args:
            engine_paths: Dict mapping component names to TensorRT engine paths

        Returns:
            Dict containing latency metrics for full pipeline
        """
        logger.info("Benchmarking end-to-end pipeline latency")

        # Load all engines
        engines = {}
        for component, engine_path in engine_paths.items():
            try:
                engines[component] = TensorRTEngine(str(engine_path))
                logger.info(f"  Loaded {component} engine")
            except Exception as e:
                logger.error(f"  Failed to load {component} engine: {e}")
                return {'error': f'Failed to load {component} engine: {e}'}

        # Required components
        required = ['content_encoder', 'pitch_encoder', 'flow_decoder']
        if not all(comp in engines for comp in required):
            missing = [c for c in required if c not in engines]
            return {'error': f'Missing required engines: {missing}'}

        # Generate test data
        test_data = self.generate_test_data(audio_length=3.0, batch_size=1)

        # Prepare inputs for each stage
        content_inputs = {
            'input_audio': test_data['audio'].numpy()
        }

        f0_input = test_data['f0'].numpy()
        pitch_inputs = {
            'f0_input': f0_input,
            'voiced_mask': (f0_input > 0).astype(np.bool_)
        }

        # Warmup runs
        logger.info("  Running warmup...")
        for _ in range(self.config.warmup_runs):
            content_out = engines['content_encoder'].infer(content_inputs)
            pitch_out = engines['pitch_encoder'].infer(pitch_inputs)

            # Prepare flow decoder inputs
            batch_size, time_steps = test_data['f0'].shape
            latent_input = np.random.randn(batch_size, 192, time_steps).astype(np.float32)
            mask = np.ones((batch_size, 1, time_steps), dtype=np.float32)

            # Concatenate conditioning (content + pitch + speaker)
            speaker_emb = test_data['speaker_emb'].numpy()
            speaker_emb_expanded = np.repeat(speaker_emb[:, :, np.newaxis], time_steps, axis=2)

            flow_inputs = {
                'latent_input': latent_input,
                'mask': mask,
                'conditioning': np.concatenate([
                    list(content_out.values())[0][:, :256, :time_steps],  # Content: 256
                    list(pitch_out.values())[0][:, :192, :time_steps],     # Pitch: 192
                    speaker_emb_expanded[:, :256, :time_steps]              # Speaker: 256
                ], axis=1).astype(np.float32)  # Total: 704
            }

            _ = engines['flow_decoder'].infer(flow_inputs)

        # Benchmark runs
        logger.info(f"  Running {self.config.sample_count} benchmark iterations...")
        latencies = {
            'content_encoder': [],
            'pitch_encoder': [],
            'flow_decoder': [],
            'total_pipeline': []
        }

        for _ in range(self.config.sample_count):
            # Time each component
            pipeline_start = time.time()

            # Content encoding
            content_start = time.time()
            content_out = engines['content_encoder'].infer(content_inputs)
            latencies['content_encoder'].append(time.time() - content_start)

            # Pitch encoding
            pitch_start = time.time()
            pitch_out = engines['pitch_encoder'].infer(pitch_inputs)
            latencies['pitch_encoder'].append(time.time() - pitch_start)

            # Flow decoding
            flow_start = time.time()
            _ = engines['flow_decoder'].infer(flow_inputs)
            latencies['flow_decoder'].append(time.time() - flow_start)

            latencies['total_pipeline'].append(time.time() - pipeline_start)

        # Compute statistics
        metrics = {}
        for component, times in latencies.items():
            times_ms = np.array(times) * 1000  # Convert to ms
            metrics[component] = {
                'mean_ms': float(np.mean(times_ms)),
                'std_ms': float(np.std(times_ms)),
                'min_ms': float(np.min(times_ms)),
                'max_ms': float(np.max(times_ms)),
                'median_ms': float(np.median(times_ms)),
                'p95_ms': float(np.percentile(times_ms, 95)),
                'p99_ms': float(np.percentile(times_ms, 99))
            }

            logger.info(f"  {component}: {metrics[component]['mean_ms']:.2f} ± {metrics[component]['std_ms']:.2f} ms")

        # Calculate speedup vs target latency (100ms)
        target_latency_ms = 100
        total_mean_ms = metrics['total_pipeline']['mean_ms']
        speedup = target_latency_ms / total_mean_ms if total_mean_ms > 0 else 0

        metrics['summary'] = {
            'target_latency_ms': target_latency_ms,
            'achieved_latency_ms': total_mean_ms,
            'speedup_vs_target': float(speedup),
            'meets_target': total_mean_ms < target_latency_ms
        }

        logger.info(f"  Pipeline latency: {total_mean_ms:.2f}ms (target: {target_latency_ms}ms)")
        logger.info(f"  {'✓ MEETS TARGET' if metrics['summary']['meets_target'] else '✗ DOES NOT MEET TARGET'}")

        return metrics

    def _generate_accuracy_comparison(self) -> None:
        """Generate accuracy comparison between inference types."""
        logger.info("Generating accuracy comparison metrics")

        for component in ['content_encoder', 'pitch_encoder', 'flow_decoder']:
            if component not in self.model_paths['pytorch'] or component not in self.model_paths['tensorrt']:
                continue

            logger.info(f"Comparing accuracy for {component}")

            # Generate test data
            test_data = self.generate_test_data(audio_length=3.0, batch_size=1)

            try:
                # Get PyTorch output
                pytorch_result = self._get_pytorch_output(component, test_data)

                # Get TensorRT output
                tensorrt_result = self._get_tensorrt_output(component, test_data)

                # Compute accuracy metrics
                accuracy_metrics = self._compute_accuracy_metrics(pytorch_result, tensorrt_result)

                # Store in results
                self.results.add_result(
                    component, 'accuracy_comparison', 1, 3.0, 'comparison', accuracy_metrics
                )

                logger.info(f"  Max diff: {accuracy_metrics['max_diff']:.6f}")
                logger.info(f"  Mean diff: {accuracy_metrics['mean_diff']:.6f}")
                logger.info(f"  RMSE: {accuracy_metrics['rmse']:.6f}")

            except Exception as e:
                logger.warning(f"Accuracy comparison failed for {component}: {e}")

    def _compute_accuracy_metrics(
        self,
        pytorch_output: np.ndarray,
        tensorrt_output: np.ndarray
    ) -> Dict[str, float]:
        """Compute accuracy comparison metrics between two outputs."""
        # Ensure same shape
        if pytorch_output.shape != tensorrt_output.shape:
            logger.warning(f"Shape mismatch: {pytorch_output.shape} vs {tensorrt_output.shape}")
            # Flatten for comparison
            pytorch_output = pytorch_output.flatten()
            tensorrt_output = tensorrt_output.flatten()
            min_len = min(len(pytorch_output), len(tensorrt_output))
            pytorch_output = pytorch_output[:min_len]
            tensorrt_output = tensorrt_output[:min_len]

        # Compute metrics
        diff = pytorch_output - tensorrt_output
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))

        # Correlation
        correlation = np.corrcoef(pytorch_output.flatten(), tensorrt_output.flatten())[0, 1]

        # SNR (Signal-to-Noise Ratio)
        signal_power = np.mean(pytorch_output ** 2)
        noise_power = np.mean(diff ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return {
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'snr_db': float(snr_db)
        }

    def _get_pytorch_output(self, component: str, test_data: Dict[str, Any]) -> np.ndarray:
        """Get PyTorch model output."""
        # Implementation similar to _benchmark_pytorch but returns output
        model_path = self.model_paths['pytorch'][component]
        device = self.config.device

        if component == 'content_encoder':
            model = ContentEncoder(encoder_type='cnn_fallback')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(test_data['audio'].to(device), test_data['sample_rate'])
            return output.cpu().numpy()

        # Add other components as needed...
        raise NotImplementedError(f"PyTorch output not implemented for {component}")

    def _get_tensorrt_output(self, component: str, test_data: Dict[str, Any]) -> np.ndarray:
        """Get TensorRT model output."""
        engine = TensorRTEngine(str(self.model_paths['tensorrt'][component]))
        inputs = self._prepare_component_inputs(component, test_data)
        results = engine.infer(inputs)

        # Return the first output (assuming single output)
        return list(results.values())[0]


def main():
    """Main entry point for benchmarking script."""
    parser = argparse.ArgumentParser(description="TensorRT Benchmarking for Voice Conversion")

    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory containing models (pytorch/, onnx/, tensorrt/)')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Directory to save benchmark results')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of inference runs per benchmark')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup runs before benchmarking')
    parser.add_argument('--audio-lengths', type=float, nargs='+',
                       help='Audio lengths to test (seconds)')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       help='Batch sizes to test')
    parser.add_argument('--precision-modes', type=str, nargs='+',
                       help='Precision modes to test (fp32, fp16, int8)')
    parser.add_argument('--include-memory', action='store_true',
                       help='Include memory usage measurements')
    parser.add_argument('--include-accuracy', action='store_true',
                       help='Include accuracy comparison measurements')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for PyTorch models (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')

    # ONNX export and TensorRT engine building options
    parser.add_argument('--build-engines', action='store_true',
                       help='Build TensorRT engines from ONNX models')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export components to ONNX format')
    parser.add_argument('--int8', action='store_true',
                       help='Enable INT8 quantization for TensorRT engines')
    parser.add_argument('--calibration-data', type=str, default=None,
                       help='Path to calibration data NPZ file for INT8 quantization')
    parser.add_argument('--dynamic-shapes', action='store_true',
                       help='Use dynamic shapes for TensorRT engines (enables variable-length audio)')
    parser.add_argument('--workspace-gb', type=float, default=2.0,
                       help='TensorRT workspace size in GB (default: 2.0)')
    parser.add_argument('--opset', type=int, default=13,
                       help='ONNX opset version for export (default: 13)')

    args = parser.parse_args()

    # Create benchmark configuration
    config = BenchmarkConfig(args)

    # Run benchmarks
    benchmark = VoiceConversionBenchmark(config)
    benchmark.run_benchmarks()


if __name__ == '__main__':
    main()
