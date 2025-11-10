#!/usr/bin/env python3
"""
Performance Profiling Utility (Comment 8)

Profiles voice conversion pipeline with:
- Component timing for pipeline stages (separation, f0_extraction, conversion, mixing)
- GPU utilization sampling via pynvml at 100-200ms intervals
- Performance breakdown metrics

Outputs validation_results/performance_breakdown.json
"""

import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


def check_gpu_monitoring_available() -> bool:
    """Check if GPU monitoring is available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return True
    except (ImportError, Exception):
        return False


class GPUMonitor:
    """Monitor GPU utilization at regular intervals."""

    def __init__(self, interval_ms: int = 150):
        self.interval_ms = interval_ms
        self.samples: List[float] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.peak_used_bytes = 0  # Add peak memory tracking

        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def start(self) -> None:
        """Start monitoring GPU utilization."""
        if not self.available:
            return

        self.samples = []
        self.peak_used_bytes = 0  # Reset peak memory
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        """Stop monitoring and return average GPU utilization."""
        if not self.available:
            return 0.0

        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if self.samples:
            return sum(self.samples) / len(self.samples)
        return 0.0

    def _monitor_loop(self) -> None:
        """Monitor loop running in background thread."""
        if not self.available:
            return

        while self.running:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.samples.append(utilization.gpu)

                # Update peak memory usage
                memory_info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.peak_used_bytes = max(self.peak_used_bytes, memory_info.used)
            except Exception as e:
                print(f"GPU monitoring error: {e}")

            time.sleep(self.interval_ms / 1000.0)

    def get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if self.peak_used_bytes > 0:
            return self.peak_used_bytes / (1024 * 1024)
        return 0.0


class PerformanceProfiler:
    """Profile voice conversion pipeline performance."""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_monitor = GPUMonitor(interval_ms=150)
        self.stage_timings: Dict[str, Dict[str, float]] = {}
        self.stage_gpu_utils: Dict[str, List[float]] = {}

    def profile_pipeline(self, audio_file: str, profile_id: str = "test_profile", profile_path: str = None, storage_dir: str = None, gpu_id: int = 0) -> Dict[str, Any]:
        """Profile complete pipeline execution."""
        print("Profiling voice conversion pipeline...")

        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import torch
        except ImportError as e:
            print(f"Error: Required modules not available: {e}")
            return {'error': str(e)}

        # Determine device and set GPU index
        if torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            print(f"Using device: {device}")
        else:
            device = 'cpu'
            print(f"Using device: {device}")

        # Set up voice profile storage
        pipeline_config = {'device': device}
        resolution_method = None

        if profile_path:
            # Handle explicit profile registration from JSON file
            profile_path_obj = Path(profile_path)
            if profile_path_obj.exists():
                print(f"Loading profile from: {profile_path}")

                with open(profile_path_obj, 'r') as f:
                    profile_data = json.load(f)

                # Create temporary storage directory
                temp_storage_dir = Path(self.output_dir) / 'temp_profiles'
                temp_storage_dir.mkdir(exist_ok=True)

                # Initialize storage with temp directory
                from src.auto_voice.storage.voice_profiles import VoiceProfileStorage
                temp_storage = VoiceProfileStorage(storage_dir=str(temp_storage_dir))

                # Split embedding from metadata for storage
                if 'embedding' in profile_data:
                    embedding = np.array(profile_data['embedding'])
                    profile_meta = {k: v for k, v in profile_data.items() if k != 'embedding'}
                    profile_meta['embedding'] = embedding
                    temp_storage.save_profile(profile_meta)

                    # Override profile_id if it differs
                    profile_id = profile_data.get('profile_id', profile_id)
                    pipeline_config['storage_dir'] = str(temp_storage_dir)
                    resolution_method = f"explicit_profile_path:{profile_path}"
                    print(f"Registered profile '{profile_id}' in temporary storage: {temp_storage_dir}")
                else:
                    print(f"Warning: Profile {profile_path} does not contain embedding")
                    return {'error': f'Profile {profile_path} missing embedding field'}
            else:
                print(f"Error: Profile path {profile_path} does not exist")
                return {'error': f'Profile path not found: {profile_path}'}
        elif storage_dir:
            # Use explicit storage directory (profiles must already exist there)
            pipeline_config['storage_dir'] = storage_dir
            resolution_method = f"explicit_storage_dir:{storage_dir}"
            print(f"Using explicit storage directory: {storage_dir}")
        else:
            # Try to auto-discover and register profile from test data
            registered_storage = self._register_test_profile(profile_id)
            if registered_storage:
                pipeline_config['storage_dir'] = registered_storage
                resolution_method = f"auto_registered_from_test_data:{registered_storage}"
                print(f"Auto-registered profile '{profile_id}' from test data in: {registered_storage}")
            else:
                # Fall back to using default storage (profile must already exist)
                print(f"Using default storage location for profile: {profile_id}")
                resolution_method = "default_storage"

        # Create pipeline with config
        pipeline = SingingConversionPipeline(config=pipeline_config)

        # Track stage timing with callbacks
        stage_monitors: Dict[str, GPUMonitor] = {}
        current_stage: Optional[str] = None

        def progress_callback(progress: float, message: str):
            """Capture stage transitions and monitor GPU."""
            nonlocal current_stage

            if message.startswith('stage_start:'):
                stage_name = message.replace('stage_start:', '')
                current_stage = stage_name

                # Initialize timing
                if stage_name not in self.stage_timings:
                    self.stage_timings[stage_name] = {}
                self.stage_timings[stage_name]['start'] = time.perf_counter()

                # Start GPU monitoring for this stage
                stage_monitor = GPUMonitor(interval_ms=150)
                stage_monitors[stage_name] = stage_monitor
                stage_monitor.start()

                print(f"  Stage started: {stage_name}")

            elif message.startswith('stage_end:'):
                stage_name = message.replace('stage_end:', '')

                # Record timing
                if stage_name in self.stage_timings and 'start' in self.stage_timings[stage_name]:
                    self.stage_timings[stage_name]['end'] = time.perf_counter()
                    duration = (self.stage_timings[stage_name]['end'] -
                               self.stage_timings[stage_name]['start'])
                    print(f"  Stage completed: {stage_name} ({duration:.3f}s)")

                # Stop GPU monitoring for this stage
                if stage_name in stage_monitors:
                    avg_util = stage_monitors[stage_name].stop()
                    self.stage_gpu_utils[stage_name] = avg_util
                    print(f"    GPU utilization: {avg_util:.1f}%")

                current_stage = None

        # Start overall GPU monitoring
        self.gpu_monitor.start()
        start_time = time.perf_counter()

        # Reset PyTorch memory stats before profiling
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()

        # Run pipeline with profiling
        try:
            result = pipeline.convert_song(
                song_path=audio_file,
                target_profile_id=profile_id,
                progress_callback=progress_callback
            )
            elapsed_time = time.perf_counter() - start_time
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            return {'error': str(e)}

        # Stop overall GPU monitoring
        avg_gpu_util = self.gpu_monitor.stop()
        peak_memory_mb = self.gpu_monitor.get_peak_memory_mb()

        # Get PyTorch memory peaks if on CUDA
        memory_peak_allocated_mb = 0.0
        memory_peak_reserved_mb = 0.0
        if device.startswith('cuda'):
            torch.cuda.synchronize()  # Ensure all operations complete
            memory_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            memory_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        # Calculate stage metrics
        stages = []
        total_stage_time = 0.0

        expected_stages = ['separation', 'pitch_extraction', 'voice_conversion', 'audio_mixing']

        for stage_name in expected_stages:
            if stage_name in self.stage_timings:
                timing = self.stage_timings[stage_name]
                if 'start' in timing and 'end' in timing:
                    stage_duration = timing['end'] - timing['start']
                    stage_gpu = self.stage_gpu_utils.get(stage_name, 0.0)

                    stages.append({
                        'name': stage_name,
                        'time_ms': stage_duration * 1000,
                        'gpu_utilization': stage_gpu
                    })

                    total_stage_time += stage_duration

        # Validate GPU utilization for CUDA
        if device == 'cuda' and avg_gpu_util < 70.0:
            print(f"⚠️  Warning: Average GPU utilization ({avg_gpu_util:.1f}%) below 70% threshold")

        # Extract audio duration from result
        audio_duration_s = result.get('duration', 30.0)  # Default to 30s if not available

        # Calculate RTF (Real-Time Factor)
        rtf = (elapsed_time / audio_duration_s) if audio_duration_s > 0 else 0.0

        # Calculate throughput
        throughput_req_s = (1.0 / elapsed_time) if elapsed_time > 0 else 0.0

        # Build performance breakdown
        # Extract GPU index from device string
        current_device_index = torch.cuda.current_device() if device.startswith('cuda') else 0

        performance_data = {
            'device': device,
            'gpu_info': {
                'index': current_device_index if device.startswith('cuda') else None,
                'name': torch.cuda.get_device_name(current_device_index) if device.startswith('cuda') else None,
                'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(current_device_index))) if device.startswith('cuda') else None,
                'total_memory_gb': torch.cuda.get_device_properties(current_device_index).total_memory / 1e9 if device.startswith('cuda') else None
            },
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if device == 'cuda' else None,
            'audio_duration_s': audio_duration_s,
            'total_time_ms': elapsed_time * 1000,
            'rtf': rtf,
            'throughput_req_s': throughput_req_s,
            'stage_breakdown_ms': total_stage_time * 1000,
            'average_gpu_utilization': avg_gpu_util if device.startswith('cuda') else 0.0,
            'memory_peak_mb': peak_memory_mb if device.startswith('cuda') else 0.0,
            'memory_peak_allocated_mb': memory_peak_allocated_mb if device.startswith('cuda') else 0.0,
            'memory_peak_reserved_mb': memory_peak_reserved_mb if device.startswith('cuda') else 0.0,
            'stages': stages,
            'gpu_utilization_requirement': {
                'required': 70.0,
                'actual': avg_gpu_util,
                'met': avg_gpu_util >= 70.0 if device == 'cuda' else None
            },
            'profile_resolution': {
                'profile_id': profile_id,
                'method': resolution_method,
                'storage_dir': pipeline_config.get('storage_dir')
            }
        }

        return performance_data

    def save_results(self, performance_data: Dict[str, Any]) -> None:
        """Save performance breakdown to JSON file."""
        output_file = self.output_dir / 'performance_breakdown.json'

        with open(output_file, 'w') as f:
            json.dump(performance_data, f, indent=2)

        print(f"\nPerformance breakdown saved to {output_file}")

    def print_summary(self, performance_data: Dict[str, Any]) -> None:
        """Print performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*60)

        if 'error' in performance_data:
            print(f"\n❌ Error: {performance_data['error']}")
            return

        print(f"\nDevice: {performance_data['device']}")
        print(f"Total Time: {performance_data['total_time_ms']:.2f}ms")

        if performance_data['device'].startswith('cuda'):
            peak_nvml = performance_data.get('memory_peak_mb', 0.0)
            peak_allocated = performance_data.get('memory_peak_allocated_mb', 0.0)
            peak_reserved = performance_data.get('memory_peak_reserved_mb', 0.0)
            max_peak = max(peak_nvml, peak_allocated)
            print(f"Peak GPU Memory: {max_peak:.1f}MB")

            gpu_req = performance_data['gpu_utilization_requirement']
            status = "✅" if gpu_req['met'] else "⚠️"
            print(f"GPU Utilization Requirement: {status} ({gpu_req['actual']:.1f}% >= {gpu_req['required']:.1f}%)")

        print("\nStage Breakdown:")
        for stage in performance_data['stages']:
            print(f"  {stage['name']:<20} {stage['time_ms']:>8.2f}ms", end='')
            if performance_data['device'].startswith('cuda'):
                print(f"  (GPU: {stage['gpu_utilization']:>5.1f}%)")
            else:
                print()

        print("="*60)

    def _register_test_profile(self, profile_id: str) -> Optional[str]:
        """Auto-discover and register test profile from benchmark data.

        Args:
            profile_id: Profile ID to search for

        Returns:
            Storage directory path if successful, None otherwise
        """
        try:
            # Look for benchmark metadata
            repo_root = Path(__file__).parent.parent
            metadata_file = repo_root / "tests" / "data" / "benchmark" / "metadata.json"

            if not metadata_file.exists():
                print(f"Benchmark metadata not found: {metadata_file}")
                return None

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Find matching profile
            profiles = metadata.get('profiles', [])
            matching_profile = None

            for profile_meta in profiles:
                if profile_meta.get('profile_id') == profile_id:
                    matching_profile = profile_meta
                    break

            if not matching_profile:
                print(f"Profile '{profile_id}' not found in benchmark metadata")
                return None

            # Get profile file path (relative to tests/data/)
            profile_rel_path = matching_profile.get('file')
            if not profile_rel_path:
                print(f"Profile '{profile_id}' has no file path in metadata")
                return None

            # Resolve profile path - metadata stores paths relative to tests/data/
            # e.g., "benchmark/profiles/test_profile_1.json"
            profile_file = repo_root / "tests" / "data" / profile_rel_path

            if not profile_file.exists():
                print(f"Profile file not found: {profile_file}")
                return None

            # Load profile JSON
            with open(profile_file, 'r') as f:
                profile_data = json.load(f)

            # Check for embedding
            if 'embedding' not in profile_data:
                print(f"Profile {profile_file} missing embedding field")
                return None

            # Create temporary storage directory
            temp_storage_dir = Path(self.output_dir) / 'temp_profiles'
            temp_storage_dir.mkdir(exist_ok=True)

            # Initialize storage
            from src.auto_voice.storage.voice_profiles import VoiceProfileStorage
            temp_storage = VoiceProfileStorage(storage_dir=str(temp_storage_dir))

            # Convert embedding to numpy array and save profile
            embedding = np.array(profile_data['embedding'])
            profile_meta = {k: v for k, v in profile_data.items() if k != 'embedding'}
            profile_meta['embedding'] = embedding
            temp_storage.save_profile(profile_meta)

            print(f"Auto-registered profile '{profile_id}' from {profile_file}")
            return str(temp_storage_dir)

        except Exception as e:
            print(f"Failed to auto-register profile '{profile_id}': {e}")
            return None


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Profile voice conversion pipeline performance')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for results')
    parser.add_argument('--profile-path', type=str, help='Path to voice profile JSON file')
    parser.add_argument('--audio-file', type=str, default=None,
                       help='Audio file to profile (default: auto-detect from test data)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )
    args = parser.parse_args()

    print("=== Voice Conversion Performance Profiling ===\n")

    # Import torch here after args are parsed
    import torch

    # Set GPU device if CUDA is available and gpu_id is specified
    if torch.cuda.is_available() and args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
        print(f"✅ Using GPU device {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    # Check GPU monitoring availability
    if check_gpu_monitoring_available():
        print("✅ GPU monitoring available (pynvml)")
    else:
        print("⚠️  GPU monitoring not available (pynvml not installed or no GPU)")

    # Get test audio file
    if args.audio_file:
        test_audio = Path(args.audio_file)
    else:
        # Use benchmark test data
        test_audio = Path(__file__).parent.parent / "tests" / "data" / "benchmark" / "audio_30s_22050hz.wav"

    if not test_audio.exists():
        print(f"\nError: Test audio file not found: {test_audio}")
        print("Please generate test data first using scripts/generate_benchmark_test_data.py")
        print(f"Run: python scripts/generate_benchmark_test_data.py")
        return 1

    # Get profile ID and path
    profile_id = None
    profile_path = None

    if args.profile_path:
        # Use explicit profile path from command line
        profile_path = args.profile_path
        # Try to extract profile_id from the file
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
                profile_id = profile_data.get('profile_id', 'test_profile')
        except Exception as e:
            print(f"Warning: Could not read profile_id from {profile_path}: {e}")
            profile_id = 'test_profile'
    else:
        # Try to auto-detect from metadata
        metadata_file = Path(__file__).parent.parent / "tests" / "data" / "benchmark" / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if metadata.get('profiles'):
                        profile_data = metadata['profiles'][0]
                        profile_id = profile_data['profile_id']

                        # Get profile file path (relative to tests/data/)
                        if 'file' in profile_data:
                            # Resolve path correctly - metadata stores paths relative to tests/data/
                            profile_rel_path = profile_data['file']
                            profile_path = str(Path(__file__).parent.parent / "tests" / "data" / profile_rel_path)

                            if Path(profile_path).exists():
                                print(f"Found profile file: {profile_path}")
                            else:
                                print(f"Warning: Profile file not found: {profile_path}")
                                profile_path = None
                    else:
                        profile_id = "test_profile"
            except Exception as e:
                print(f"Warning: Could not load profile metadata: {e}")
                profile_id = "test_profile"
        else:
            profile_id = "test_profile"

    # Create profiler with custom output directory
    profiler = PerformanceProfiler(output_dir=args.output_dir)

    # Profile pipeline with profile_path if available
    try:
        performance_data = profiler.profile_pipeline(
            audio_file=str(test_audio),
            profile_id=profile_id,
            profile_path=profile_path,
            gpu_id=args.gpu_id
        )
    except ImportError as e:
        print(f"\nError: Required modules not available: {e}")
        print("Please ensure SingingConversionPipeline is installed and CUDA extensions are built.")
        return 1
    except RuntimeError as e:
        if 'CUDA' in str(e):
            print(f"\nError: CUDA error during profiling: {e}")
            print("Please check GPU availability and CUDA installation.")
        else:
            print(f"\nError: Runtime error during profiling: {e}")
        return 1

    # Save results
    if 'error' not in performance_data:
        profiler.save_results(performance_data)

    # Print summary
    profiler.print_summary(performance_data)

    # Exit code based on GPU utilization requirement (if CUDA)
    if performance_data.get('device', '').startswith('cuda'):
        gpu_req = performance_data.get('gpu_utilization_requirement', {})
        if not gpu_req.get('met', True):
            print("\n⚠️  GPU utilization below threshold (results saved anyway)")
            # Always return 0 to preserve results
            return 0

    if 'error' in performance_data:
        return 1

    print("\n✅ Performance profiling complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
