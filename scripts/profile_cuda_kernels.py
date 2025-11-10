#!/usr/bin/env python3
"""
CUDA Kernel Profiling and Benchmarking Script

This script provides comprehensive profiling capabilities for CUDA kernels,
including performance benchmarks against reference implementations,
Nsight profiling integration, and detailed metrics collection.

Usage:
    python profile_cuda_kernels.py --kernel stft --audio-file audio.wav --iterations 100
    python profile_cuda_kernels.py --kernel pitch_detection --iterations 50 --nsight
    python profile_cuda_kernels.py --kernel mel_spectrogram_singing --compare-reference --output results.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.cuda
    import cuda_kernels
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torchcrepe
    TORCHCREPE_AVAILABLE = True
except ImportError:
    TORCHCREPE_AVAILABLE = False

try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False


class CUDAKernelProfiler:
    """CUDA Kernel Profiler with benchmarking capabilities."""

    def __init__(self, device: Optional[int] = None):
        """Initialize profiler with CUDA device."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.device = device or 0
        torch.cuda.set_device(self.device)
        self.cuda_device = torch.device(f"cuda:{self.device}")

        # Profiling results
        self.results = {}

    def benchmark_kernel(self, kernel_name: str, kernel_func, *args, iterations: int = 100,
                        warmup: int = 10, **kwargs) -> Dict[str, float]:
        """Benchmark a single kernel execution."""
        print(f"Benchmarking {kernel_name}...")

        # Warmup
        for _ in range(warmup):
            _ = kernel_func(*args, **kwargs)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = kernel_func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / iterations
        throughput = 1.0 / avg_time if avg_time > 0 else 0

        result = {
            'kernel_name': kernel_name,
            'iterations': iterations,
            'total_time': end_time - start_time,
            'avg_time': avg_time,
            'throughput_ips': throughput,
            'throughput_ms': avg_time * 1000
        }

        return result

    def run_nsight_profile(self, kernel_name: str, profile_func, *args, use_ncu: bool = False, **kwargs) -> Dict[str, Any]:
        """Run Nsight profiling on a kernel.

        Args:
            kernel_name: Name of kernel to profile
            profile_func: Function to profile
            args: Arguments to pass to profile_func
            use_ncu: If True, use Nsight Compute (ncu) instead of Nsight Systems
            kwargs: Keyword arguments to pass to profile_func

        Returns:
            Dictionary with profiling results and metrics
        """
        if use_ncu:
            return self._run_nsight_compute_profile(kernel_name, profile_func, *args, **kwargs)
        else:
            return self._run_nsight_systems_profile(kernel_name, profile_func, *args, **kwargs)

    def _run_nsight_systems_profile(self, kernel_name: str, profile_func, *args, **kwargs) -> Dict[str, Any]:
        """Run Nsight Systems (nsys) profiling."""
        nsys_available = self._check_tool_available('nsys')
        if not nsys_available:
            print("Nsight Systems not available, skipping profiling")
            return {'nsys_available': False}

        try:
            # Create temporary helper script for profiling
            helper_script = f"/tmp/{kernel_name}_profile_helper.py"
            output_file = f"/tmp/{kernel_name}_profile"  # FIXED: Base path without extension

            # Write helper script with serialized args
            with open(helper_script, 'w') as f:
                f.write(f'''
import sys
import os
sys.path.insert(0, "{os.path.join(os.path.dirname(__file__), "..", "src")}")
import torch
import cuda_kernels

torch.cuda.set_device({self.device})

# Setup profiling markers
torch.cuda.nvtx.range_push("{kernel_name}")

# Execute kernel function
result = cuda_kernels.{profile_func.__name__}(*{repr(args)}, **{repr(kwargs)})
torch.cuda.synchronize()

torch.cuda.nvtx.range_pop()
print("Profiling completed successfully")
''')

            # Build nsys command
            cmd = [
                'nsys', 'profile',
                '--output', output_file,  # FIXED: nsys will add .nsys-rep automatically
                '--force-overwrite', 'true',
                '--capture-range=nvtx',  # FIXED: Changed from cudaProfilerApi to nvtx
                '--capture-range-end=stop',
                sys.executable, helper_script
            ]

            # Run profiling
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # FIXED: Check for the actual output file with extension
            output_file_with_ext = f"{output_file}.nsys-rep"
            # Check if profiling succeeded
            if result.returncode == 0 and os.path.exists(output_file_with_ext):
                # Convert to JSON stats
                stats_cmd = ['nsys', 'stats', '--format', 'json', output_file_with_ext]
                stats_result = subprocess.run(stats_cmd, capture_output=True, text=True)

                if stats_result.returncode == 0:
                    try:
                        stats_data = json.loads(stats_result.stdout)
                        return {
                            'tool': 'nsys',
                            'nsys_available': True,
                            'profiling_successful': True,
                            'stats': stats_data,
                            'output_file': output_file_with_ext  # FIXED: Return actual file with extension
                        }
                    except json.JSONDecodeError:
                        pass

            return {
                'tool': 'nsys',
                'nsys_available': True,
                'profiling_successful': False,
                'error': result.stderr if result.stderr else "Unknown error"
            }

        except subprocess.TimeoutExpired:
            return {
                'tool': 'nsys',
                'nsys_available': True,
                'profiling_successful': False,
                'error': "Profiling timeout"
            }
        except Exception as e:
            return {
                'tool': 'nsys',
                'nsys_available': True,
                'profiling_successful': False,
                'error': str(e)
            }
        finally:
            # Clean up helper script
            if os.path.exists(helper_script):
                os.remove(helper_script)

    def _run_nsight_compute_profile(self, kernel_name: str, profile_func, *args, **kwargs) -> Dict[str, Any]:
        """Run Nsight Compute (ncu) profiling with detailed metrics."""
        ncu_available = self._check_tool_available('ncu')
        if not ncu_available:
            print("Nsight Compute not available, skipping profiling")
            return {'ncu_available': False}

        try:
            # Create temporary helper script for profiling
            helper_script = f"/tmp/{kernel_name}_ncu_profile_helper.py"
            output_file = f"/tmp/{kernel_name}_ncu_profile"

            # Write helper script with NVTX ranges
            with open(helper_script, 'w') as f:
                f.write(f'''
import sys
import os
sys.path.insert(0, "{os.path.join(os.path.dirname(__file__), "..", "src")}")
import torch
import cuda_kernels

torch.cuda.set_device({self.device})

# Warmup
for _ in range(3):
    result = cuda_kernels.{profile_func.__name__}(*{repr(args)}, **{repr(kwargs)})
    torch.cuda.synchronize()

# Profile execution with NVTX range
torch.cuda.nvtx.range_push("{kernel_name}")
result = cuda_kernels.{profile_func.__name__}(*{repr(args)}, **{repr(kwargs)})
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("NCU profiling completed successfully")
''')

            # Build ncu command with key metrics
            cmd = [
                'ncu',
                '--metrics',
                'sm__throughput.avg.pct_of_peak_sustained_elapsed,'
                'dram__throughput.avg.pct_of_peak_sustained_elapsed,'
                'launch__occupancy_limit_active,'
                'sm__warps_active.avg.pct_of_peak_sustained_active,'
                'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,'
                'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
                '--export', output_file,
                '--force-overwrite',
                '--target-processes', 'all',
                sys.executable, helper_script
            ]

            # Run profiling
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse ncu output
            if result.returncode == 0:
                # Try to load CSV or JSON output
                csv_file = f"{output_file}.csv"
                json_file = f"{output_file}.ncu-rep"

                metrics = {}
                if os.path.exists(csv_file):
                    metrics = self._parse_ncu_csv(csv_file)

                return {
                    'tool': 'ncu',
                    'ncu_available': True,
                    'profiling_successful': True,
                    'metrics': metrics,
                    'output_file': output_file,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }

            return {
                'tool': 'ncu',
                'ncu_available': True,
                'profiling_successful': False,
                'error': result.stderr if result.stderr else "Unknown error",
                'stdout': result.stdout
            }

        except subprocess.TimeoutExpired:
            return {
                'tool': 'ncu',
                'ncu_available': True,
                'profiling_successful': False,
                'error': "NCU profiling timeout"
            }
        except Exception as e:
            return {
                'tool': 'ncu',
                'ncu_available': True,
                'profiling_successful': False,
                'error': str(e)
            }
        finally:
            # Clean up helper script
            if os.path.exists(helper_script):
                os.remove(helper_script)

    def _parse_ncu_csv(self, csv_file: str) -> Dict[str, Any]:
        """Parse NCU CSV output to extract metrics."""
        try:
            import csv
            metrics = {}
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract key metrics
                    if 'Metric Name' in row and 'Metric Value' in row:
                        metric_name = row['Metric Name']
                        metric_value = row['Metric Value']
                        try:
                            metrics[metric_name] = float(metric_value)
                        except ValueError:
                            metrics[metric_name] = metric_value
            return metrics
        except Exception as e:
            print(f"Error parsing NCU CSV: {e}")
            return {}

    def benchmark_with_reference(self, kernel_name: str, cuda_func, reference_func,
                               *args, **kwargs) -> Dict[str, Any]:
        """Benchmark CUDA kernel against reference implementation."""
        cuda_result = self.benchmark_kernel(f"{kernel_name}_cuda", cuda_func, *args, **kwargs)

        if reference_func is not None:
            ref_result = self.benchmark_kernel(f"{kernel_name}_reference", reference_func, *args, **kwargs)
            speedup = ref_result['avg_time'] / cuda_result['avg_time'] if cuda_result['avg_time'] > 0 else 0

            return {
                'cuda_performance': cuda_result,
                'reference_performance': ref_result,
                'speedup': speedup,
                'speedup_description': f"{speedup:.2f}x" if speedup > 0 else "N/A"
            }

        return {
            'cuda_performance': cuda_result,
            'reference_performance': None,
            'speedup': None
        }

    def _check_tool_available(self, tool_name: str) -> bool:
        """Check if a profiling tool is available."""
        try:
            result = subprocess.run(['which', tool_name], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _check_nsys_available(self) -> bool:
        """Check if Nsight Systems is available (legacy method)."""
        return self._check_tool_available('nsys')

    def _check_ncu_available(self) -> bool:
        """Check if Nsight Compute is available."""
        return self._check_tool_available('ncu')


class KernelBenchmarker:
    """High-level benchmarker for specific kernels."""

    def __init__(self, profiler: CUDAKernelProfiler, audio_file: Optional[str] = None):
        self.profiler = profiler
        self.cuda_device = profiler.cuda_device

        # Load audio if provided
        if audio_file and os.path.exists(audio_file):
            try:
                audio, sr = librosa.load(audio_file, sr=None, mono=True)
                self.audio_tensor = torch.from_numpy(audio).float().to(self.cuda_device)
                self.sample_rate = sr
                print(f"Loaded audio: {len(audio)} samples at {sr}Hz")
            except Exception as e:
                print(f"Failed to load audio file {audio_file}: {e}")
                self.audio_tensor = None
                self.sample_rate = 16000
        else:
            # Generate synthetic audio
            self.audio_tensor = torch.randn(1, 16000, device=self.cuda_device)  # 1 second at 16kHz
            self.sample_rate = 16000
            print("Using synthetic audio")

    def benchmark_pitch_detection(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark pitch detection kernel."""
        def cuda_pitch_detection():
            # Compute correct number of frames based on audio length
            audio_len = self.audio_tensor.shape[1] if len(self.audio_tensor.shape) > 1 else self.audio_tensor.shape[0]
            frame_length = 2048
            hop_length = 512
            n_frames = max(0, (audio_len - frame_length) // hop_length + 1)

            pitch_output = torch.zeros(n_frames, device=self.cuda_device)
            confidence_output = torch.zeros(n_frames, device=self.cuda_device)
            vibrato_output = torch.zeros(n_frames, device=self.cuda_device)

            cuda_kernels.launch_pitch_detection(
                self.audio_tensor, pitch_output, confidence_output, vibrato_output,
                self.sample_rate, frame_length, hop_length, 80.0, 1000.0, 0.3
            )
            return pitch_output

        def reference_pitch_detection():
            if not TORCHCREPE_AVAILABLE:
                return None

            # Estimate pitch using torchcrepe
            _, pitch = torchcrepe.predict(
                self.audio_tensor,
                self.sample_rate,
                hop_length=256,
                fmin=80.0,
                fmax=1000.0,
                model='tiny',
                return_periodicity=False,
                decoder=torchcrepe.decode.weighted_argmax,
                device=self.cuda_device
            )
            return pitch

        return self.profiler.benchmark_with_reference(
            "pitch_detection", cuda_pitch_detection, reference_pitch_detection, iterations=iterations
        )

    def benchmark_mel_spectrogram_singing(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark mel-spectrogram singing kernel."""
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        window = torch.hann_window(n_fft, device=self.cuda_device)
        mel_filterbank = torch.randn(n_mels, n_fft // 2 + 1, device=self.cuda_device)

        # Compute dimensions
        n_frames = (self.audio_tensor.shape[1] - n_fft) // hop_length + 1

        def cuda_mel_spec():
            mel_output = torch.zeros(self.audio_tensor.shape[0], n_frames, n_mels, device=self.cuda_device)
            cuda_kernels.launch_mel_spectrogram_singing(
                self.audio_tensor, window, mel_filterbank, mel_output,
                n_fft, hop_length, True  # apply_a_weighting=True
            )
            return mel_output

        def reference_mel_spec():
            if not LIBROSA_AVAILABLE:
                return None

            # Librosa reference
            audio_np = self.audio_tensor.cpu().numpy().squeeze()
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=80.0,
                fmax=8000.0
            )
            return torch.from_numpy(mel_spec).to(self.cuda_device)

        return self.profiler.benchmark_with_reference(
            "mel_spectrogram_singing", cuda_mel_spec, reference_mel_spec, iterations=iterations
        )

    def benchmark_stft_istft(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark STFT/ISTFT round-trip."""
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=self.cuda_device)

        # Pre-compute STFT for ISTFT benchmark
        n_frames = (self.audio_tensor.shape[1] - n_fft) // hop_length + 1
        stft_output = torch.zeros(self.audio_tensor.shape[0], n_frames, n_fft // 2 + 1,
                                 dtype=torch.cfloat, device=self.cuda_device)

        cuda_kernels.launch_optimized_stft(self.audio_tensor, window, stft_output, n_fft, hop_length)

        def cuda_stft():
            stft_out = torch.zeros_like(stft_output)
            cuda_kernels.launch_optimized_stft(self.audio_tensor, window, stft_out, n_fft, hop_length)
            return stft_out

        def cuda_istft():
            expected_length = (n_frames - 1) * hop_length + n_fft
            audio_out = torch.zeros(self.audio_tensor.shape[0], expected_length, device=self.cuda_device)
            cuda_kernels.launch_optimized_istft(stft_output, window, audio_out, n_fft, hop_length)
            return audio_out

        def reference_stft_istft():
            if not LIBROSA_AVAILABLE:
                return None

            audio_np = self.audio_tensor.cpu().numpy().squeeze()
            # STFT
            stft_ref = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length, window='hann')
            # ISTFT
            audio_ref = librosa.istft(stft_ref, hop_length=hop_length, window='hann')
            return torch.from_numpy(audio_ref).to(self.cuda_device)

        # Benchmark individual operations and round-trip
        stft_benchmark = self.profiler.benchmark_kernel("stft_cuda", cuda_stft, iterations=iterations)
        istft_benchmark = self.profiler.benchmark_kernel("istft_cuda", cuda_istft, iterations=iterations)

        # Round-trip benchmark (STFT + ISTFT)
        def cuda_round_trip():
            # STFT
            stft_temp = torch.zeros_like(stft_output)
            cuda_kernels.launch_optimized_stft(self.audio_tensor, window, stft_temp, n_fft, hop_length)

            # ISTFT
            expected_length = (n_frames - 1) * hop_length + n_fft
            audio_out = torch.zeros(self.audio_tensor.shape[0], expected_length, device=self.cuda_device)
            cuda_kernels.launch_optimized_istft(stft_temp, window, audio_out, n_fft, hop_length)
            return audio_out

        round_trip = self.profiler.benchmark_kernel("stft_istft_round_trip", cuda_round_trip, iterations=iterations)
        reference = self.profiler.benchmark_with_reference(
            "stft_istft_reference", cuda_round_trip, reference_stft_istft, iterations=iterations//2
        )

        return {
            'stft_performance': stft_benchmark,
            'istft_performance': istft_benchmark,
            'round_trip_performance': round_trip,
            'reference_comparison': reference
        }

    def benchmark_formant_extraction(self, iterations: int = 30) -> Dict[str, Any]:
        """Benchmark formant extraction kernel."""
        frame_length = 2048
        lpc_order = 14
        num_formants = 4

        # Create test frames
        batch_size = 1
        n_frames = 10
        audio_frames = torch.randn(batch_size, n_frames, frame_length, device=self.cuda_device)

        def cuda_formant_extraction():
            formants_output = torch.zeros(n_frames, num_formants, device=self.cuda_device)
            cuda_kernels.launch_formant_extraction(
                audio_frames, formants_output, frame_length, self.sample_rate, lpc_order, num_formants
            )
            return formants_output

        def reference_formant_extraction():
            if not PARSELMOUTH_AVAILABLE:
                return None

            # Use parselmouth for formant extraction
            audio_np = audio_frames.cpu().numpy().squeeze()
            formants_list = []

            for i in range(n_frames):
                frame_start = i * frame_length
                frame_end = min(frame_start + frame_length, len(audio_np))
                frame_audio = audio_np[frame_start:frame_end]

                # Create sound object
                sound = parselmouth.Sound(frame_audio, sampling_frequency=self.sample_rate)

                # Extract formants
                formant = sound.to_formant_burg(max_number_of_formants=4, window_length=0.025)

                # Get formant values at center of frame
                time_point = len(frame_audio) / (2 * self.sample_rate)
                f1 = formant.get_value_at_time(1, time_point) if formant.get_value_at_time(1, time_point) != None else 0
                f2 = formant.get_value_at_time(2, time_point) if formant.get_value_at_time(2, time_point) != None else 0
                f3 = formant.get_value_at_time(3, time_point) if formant.get_value_at_time(3, time_point) != None else 0
                f4 = formant.get_value_at_time(4, time_point) if formant.get_value_at_time(4, time_point) != None else 0

                formants_list.append([f1 or 0, f2 or 0, f3 or 0, f4 or 0])

            return torch.tensor(formants_list, device=self.cuda_device)

        return self.profiler.benchmark_with_reference(
            "formant_extraction", cuda_formant_extraction, reference_formant_extraction, iterations=iterations
        )


def main():
    parser = argparse.ArgumentParser(description="CUDA Kernel Profiler")
    parser.add_argument("--kernel", type=str, required=True,
                       choices=['stft', 'istft', 'stft_istft', 'mel_spectrogram_singing',
                               'pitch_detection', 'formant_extraction', 'all'],
                       help="Kernel to profile")
    parser.add_argument("--audio-file", type=str, help="Audio file to use for profiling")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="profile_results.json",
                       help="Output file for results")
    parser.add_argument("--nsight", action="store_true",
                       help="Enable Nsight profiling")
    parser.add_argument("--use-ncu", action="store_true", default=False,
                       help="Use Nsight Compute (ncu) instead of Nsight Systems for profiling")
    parser.add_argument("--compare-reference", action="store_true",
                       help="Compare against reference implementations")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index")

    args = parser.parse_args()

    if not CUDA_AVAILABLE:
        print("CUDA not available. Exiting.")
        sys.exit(1)

    # Initialize profiler and benchmarker with specified GPU
    profiler = CUDAKernelProfiler(device=args.gpu_id)
    benchmarker = KernelBenchmarker(profiler, args.audio_file)

    # Get GPU info for metadata
    gpu_name = torch.cuda.get_device_name(args.gpu_id) if CUDA_AVAILABLE else "N/A"

    results = {
        'metadata': {
            'kernels_tested': args.kernel,
            'iterations': args.iterations,
            'cuda_device': profiler.device,
            'gpu_index': args.gpu_id,
            'gpu_name': gpu_name,
            'reference_available': {
                'librosa': LIBROSA_AVAILABLE,
                'torchcrepe': TORCHCREPE_AVAILABLE,
                'parselmouth': PARSELMOUTH_AVAILABLE
            }
        },
        'results': {}
    }

    # Run benchmarks
    if args.kernel == 'pitch_detection' or args.kernel == 'all':
        print("Benchmarking pitch detection...")
        results['results']['pitch_detection'] = benchmarker.benchmark_pitch_detection(args.iterations)
        if args.nsight:
            # FIXED: Compute correct n_frames based on audio length and kernel parameters
            audio_len = benchmarker.audio_tensor.shape[1] if len(benchmarker.audio_tensor.shape) > 1 else benchmarker.audio_tensor.shape[0]
            frame_length = 2048
            hop_length = 512
            n_frames = max(0, (audio_len - frame_length) // hop_length + 1)

            results['results']['pitch_detection']['nsight'] = profiler.run_nsight_profile(
                'pitch_detection', cuda_kernels.launch_pitch_detection,
                benchmarker.audio_tensor,
                torch.zeros(n_frames, device=benchmarker.cuda_device),
                torch.zeros(n_frames, device=benchmarker.cuda_device),
                torch.zeros(n_frames, device=benchmarker.cuda_device),
                benchmarker.sample_rate, frame_length, hop_length, 80.0, 1000.0, 0.3,
                use_ncu=args.use_ncu
            )

    if args.kernel == 'mel_spectrogram_singing' or args.kernel == 'all':
        print("Benchmarking mel-spectrogram singing...")
        results['results']['mel_spectrogram_singing'] = benchmarker.benchmark_mel_spectrogram_singing(args.iterations)

    if args.kernel == 'stft_istft' or args.kernel == 'all':
        print("Benchmarking STFT/ISTFT...")
        results['results']['stft_istft'] = benchmarker.benchmark_stft_istft(args.iterations)

    if args.kernel == 'formant_extraction' or args.kernel == 'all':
        print("Benchmarking formant extraction...")
        results['results']['formant_extraction'] = benchmarker.benchmark_formant_extraction(args.iterations)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {args.output}")

    # Print summary
    print("\n=== PROFILING SUMMARY ===")
    for kernel_name, kernel_results in results['results'].items():
        print(f"\n{kernel_name}:")
        if 'cuda_performance' in kernel_results:
            perf = kernel_results['cuda_performance']
            print(f"  avg: {perf['avg_time']*1000:.3f} ms  thr: {1.0/perf['avg_time']:.1f} it/s")
        if 'speedup' in kernel_results and kernel_results['speedup'] is not None:
            print(f"  Speedup vs reference: {kernel_results['speedup']:.2f}x")


if __name__ == "__main__":
    main()
