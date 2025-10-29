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

        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except (ImportError, Exception) as e:
            print(f"GPU monitoring not available: {e}")
            self.available = False

    def start(self) -> None:
        """Start monitoring GPU utilization."""
        if not self.available:
            return

        self.samples = []
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
            except Exception as e:
                print(f"GPU monitoring error: {e}")

            time.sleep(self.interval_ms / 1000.0)

    def get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if not self.available:
            return 0.0

        try:
            memory_info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            return memory_info.used / (1024 * 1024)
        except Exception:
            return 0.0


class PerformanceProfiler:
    """Profile voice conversion pipeline performance."""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_monitor = GPUMonitor(interval_ms=150)
        self.stage_timings: Dict[str, Dict[str, float]] = {}
        self.stage_gpu_utils: Dict[str, List[float]] = {}

    def profile_pipeline(self, audio_file: str, profile_id: str = "test_profile") -> Dict[str, Any]:
        """Profile complete pipeline execution."""
        print("Profiling voice conversion pipeline...")

        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import torch
        except ImportError as e:
            print(f"Error: Required modules not available: {e}")
            return {'error': str(e)}

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Create pipeline
        pipeline = SingingConversionPipeline(config={'device': device})

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
            print(f"\n⚠️  Warning: Average GPU utilization ({avg_gpu_util:.1f}%) below 70% threshold")

        # Build performance breakdown
        performance_data = {
            'device': device,
            'total_time_ms': elapsed_time * 1000,
            'stage_breakdown_ms': total_stage_time * 1000,
            'average_gpu_utilization': avg_gpu_util if device == 'cuda' else 0.0,
            'memory_peak_mb': peak_memory_mb if device == 'cuda' else 0.0,
            'stages': stages,
            'gpu_utilization_requirement': {
                'required': 70.0,
                'actual': avg_gpu_util,
                'met': avg_gpu_util >= 70.0 if device == 'cuda' else None
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

        if performance_data['device'] == 'cuda':
            print(f"Average GPU Utilization: {performance_data['average_gpu_utilization']:.1f}%")
            print(f"Peak GPU Memory: {performance_data['memory_peak_mb']:.1f}MB")

            gpu_req = performance_data['gpu_utilization_requirement']
            status = "✅" if gpu_req['met'] else "❌"
            print(f"GPU Utilization Requirement: {status} ({gpu_req['actual']:.1f}% >= {gpu_req['required']:.1f}%)")

        print("\nStage Breakdown:")
        for stage in performance_data['stages']:
            print(f"  {stage['name']:<20} {stage['time_ms']:>8.2f}ms", end='')
            if performance_data['device'] == 'cuda':
                print(f"  (GPU: {stage['gpu_utilization']:>5.1f}%)")
            else:
                print()

        print("="*60)


def main() -> int:
    """Main entry point."""
    print("=== Voice Conversion Performance Profiling ===\n")

    # Check GPU monitoring availability
    if check_gpu_monitoring_available():
        print("✅ GPU monitoring available (pynvml)")
    else:
        print("⚠️  GPU monitoring not available (pynvml not installed or no GPU)")

    # Get test audio file
    test_audio = Path(__file__).parent.parent / "tests" / "data" / "test_song.wav"

    if not test_audio.exists():
        print(f"\nError: Test audio file not found: {test_audio}")
        print("Please create test data first using scripts/generate_test_data.py")
        return 1

    # Create profiler
    profiler = PerformanceProfiler()

    # Profile pipeline
    performance_data = profiler.profile_pipeline(
        audio_file=str(test_audio),
        profile_id="test_profile"
    )

    # Save results
    if 'error' not in performance_data:
        profiler.save_results(performance_data)

    # Print summary
    profiler.print_summary(performance_data)

    # Exit code based on GPU utilization requirement (if CUDA)
    if performance_data.get('device') == 'cuda':
        gpu_req = performance_data.get('gpu_utilization_requirement', {})
        if not gpu_req.get('met', True):
            print("\n❌ GPU utilization below threshold")
            return 1

    if 'error' in performance_data:
        return 1

    print("\n✅ Performance profiling complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
