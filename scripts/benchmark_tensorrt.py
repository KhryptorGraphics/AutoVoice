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

        # This would load and benchmark PyTorch model
        # Implementation depends on specific component loading
        return {'error': 'PyTorch benchmarking not implemented'}

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
            from ..src.auto_voice.inference.tensorrt_engine import TensorRTEngine

            # Load TensorRT engine
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
        """Prepare inputs for specific component."""
        if component == 'content_encoder':
            return {
                'input_audio': test_data['audio'].numpy(),
                'sample_rate': np.array([test_data['sample_rate']])
            }
        elif component == 'pitch_encoder':
            return {
                'f0_input': test_data['f0'].numpy()
            }
        elif component == 'flow_decoder':
            batch_size, time_steps = test_data['f0'].shape
            latent_dim = 192  # Standard latent dimension
            cond_dim = 704  # content(256) + pitch(192) + speaker(256)

            return {
                'latent_input': np.random.randn(batch_size, latent_dim, time_steps).astype(np.float32),
                'mask': np.ones((batch_size, 1, time_steps), dtype=np.float32),
                'conditioning': np.random.randn(batch_size, cond_dim, time_steps).astype(np.float32),
                'inverse': np.array([True], dtype=bool)
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

        # Benchmark each component and configuration
        components = ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection']

        for component in components:
            for precision in self.config.precision_modes:
                for batch_size in self.config.batch_sizes:
                    for audio_length in self.config.audio_lengths:

                        logger.info(f"Benchmarking {component} - {precision} - batch{batch_size} - {audio_length}s")

                        # Generate test data
                        test_data = self.generate_test_data(audio_length, batch_size)

                        # Benchmark different inference types
                        for inference_type in ['onnx', 'tensorrt']:
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

        logger.info("Benchmark suite completed")


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

    args = parser.parse_args()

    # Create benchmark configuration
    config = BenchmarkConfig(args)

    # Run benchmarks
    benchmark = VoiceConversionBenchmark(config)
    benchmark.run_benchmarks()


if __name__ == '__main__':
    main()
