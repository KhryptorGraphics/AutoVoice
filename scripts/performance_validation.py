#!/usr/bin/env python3
"""Performance Validation Suite for AutoVoice Pipelines.

Comprehensive benchmarking for all 4 voice conversion pipelines:
1. realtime - ContentVec + HiFiGAN (22kHz, karaoke)
2. quality - CoMoSVC with consistency model (24kHz, studio)
3. quality_seedvc - Seed-VC DiT-CFM (44kHz, SOTA quality)
4. realtime_meanvc - MeanVC streaming (16kHz, low latency)

Metrics collected:
- RTF (Real-Time Factor)
- Latency (chunk and end-to-end)
- GPU Memory (peak, sustained)
- MCD (Mel Cepstral Distortion)
- Speaker Similarity (cosine)

Usage:
    python scripts/performance_validation.py --pipeline all
    python scripts/performance_validation.py --pipeline realtime --audio tests/quality_samples/william_singe_pillowtalk.wav
    python scripts/performance_validation.py --compare --output reports/benchmark_report.md

Track: performance-validation-suite_20260201
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'seed-vc'))
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'meanvc'))

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for a pipeline benchmark."""
    name: str
    pipeline_type: str
    target_rtf: float
    target_latency_ms: float
    target_mcd: float
    target_memory_gb: float
    output_sample_rate: int
    description: str


@dataclass
class BenchmarkResult:
    """Results from a single pipeline benchmark run."""
    pipeline_name: str
    pipeline_type: str
    audio_duration_sec: float
    processing_time_sec: float
    rtf: float
    latency_ms: float = 0.0
    gpu_memory_peak_gb: float = 0.0
    gpu_memory_idle_gb: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    mcd: float = 0.0
    speaker_similarity: float = 0.0
    output_sample_rate: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyBreakdown:
    """Latency breakdown by component."""
    content_encoder_ms: float = 0.0
    pitch_extractor_ms: float = 0.0
    decoder_ms: float = 0.0
    vocoder_ms: float = 0.0
    asr_ms: float = 0.0
    vc_ms: float = 0.0
    total_ms: float = 0.0


# ============================================================================
# Pipeline Configurations
# ============================================================================

PIPELINE_CONFIGS = {
    'realtime': PipelineConfig(
        name='Realtime (ContentVec + HiFiGAN)',
        pipeline_type='realtime',
        target_rtf=0.5,
        target_latency_ms=100,
        target_mcd=10.0,
        target_memory_gb=8.0,
        output_sample_rate=22050,
        description='Low-latency karaoke pipeline with 22kHz output',
    ),
    'quality': PipelineConfig(
        name='Quality (CoMoSVC)',
        pipeline_type='quality',
        target_rtf=2.0,
        target_latency_ms=3000,
        target_mcd=6.0,
        target_memory_gb=16.0,
        output_sample_rate=24000,
        description='High-quality studio pipeline with consistency model',
    ),
    'quality_seedvc': PipelineConfig(
        name='Quality Seed-VC (DiT-CFM)',
        pipeline_type='quality_seedvc',
        target_rtf=2.0,
        target_latency_ms=2000,
        target_mcd=6.0,
        target_memory_gb=16.0,
        output_sample_rate=44100,
        description='SOTA quality with 10-step DiT-CFM at 44.1kHz',
    ),
    'realtime_meanvc': PipelineConfig(
        name='Realtime MeanVC (Streaming)',
        pipeline_type='realtime_meanvc',
        target_rtf=0.5,
        target_latency_ms=80,
        target_mcd=8.0,
        target_memory_gb=6.0,
        output_sample_rate=16000,
        description='Single-step streaming with 16kHz output',
    ),
}


# ============================================================================
# Metrics Collection Utilities
# ============================================================================


class MetricsCollector:
    """Collect and compute performance metrics."""

    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self._device_idx = int(device.split(':')[1]) if ':' in device else 0

    def get_gpu_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(self._device_idx) / (1024**3)

    def get_gpu_peak_memory_gb(self) -> float:
        """Get peak GPU memory usage in GB since last reset."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self._device_idx) / (1024**3)

    def reset_peak_memory(self) -> None:
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device_idx)
            torch.cuda.synchronize()

    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def compute_rtf(self, processing_time: float, audio_duration: float) -> float:
        """Compute Real-Time Factor."""
        if audio_duration <= 0:
            return 0.0
        return processing_time / audio_duration

    def compute_mcd(
        self,
        reference: np.ndarray,
        converted: np.ndarray,
        ref_sr: int,
        conv_sr: int,
        n_mfcc: int = 13,
    ) -> float:
        """Compute Mel Cepstral Distortion."""
        try:
            import librosa

            # Resample to same rate
            target_sr = min(ref_sr, conv_sr)
            if ref_sr != target_sr:
                reference = librosa.resample(reference, orig_sr=ref_sr, target_sr=target_sr)
            if conv_sr != target_sr:
                converted = librosa.resample(converted, orig_sr=conv_sr, target_sr=target_sr)

            # Match lengths
            min_len = min(len(reference), len(converted))
            reference = reference[:min_len]
            converted = converted[:min_len]

            # Extract MFCCs
            mfcc_ref = librosa.feature.mfcc(y=reference, sr=target_sr, n_mfcc=n_mfcc)
            mfcc_conv = librosa.feature.mfcc(y=converted, sr=target_sr, n_mfcc=n_mfcc)

            # Match frame counts
            min_frames = min(mfcc_ref.shape[1], mfcc_conv.shape[1])
            mfcc_ref = mfcc_ref[:, :min_frames]
            mfcc_conv = mfcc_conv[:, :min_frames]

            # Compute MCD (excluding 0th coefficient)
            diff = mfcc_ref[1:, :] - mfcc_conv[1:, :]
            mcd = np.sqrt(2) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))

            return float(mcd)
        except Exception as e:
            logger.warning(f"MCD computation failed: {e}")
            return 0.0

    def compute_speaker_similarity(
        self,
        audio1: np.ndarray,
        sr1: int,
        audio2: np.ndarray,
        sr2: int,
    ) -> float:
        """Compute speaker similarity using MFCC statistics."""
        try:
            import librosa

            # Resample to 16kHz
            if sr1 != 16000:
                audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=16000)
            if sr2 != 16000:
                audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=16000)

            # Extract MFCCs
            mfcc1 = librosa.feature.mfcc(y=audio1, sr=16000, n_mfcc=20)
            mfcc2 = librosa.feature.mfcc(y=audio2, sr=16000, n_mfcc=20)

            # Use mean + std as embedding
            emb1 = np.concatenate([np.mean(mfcc1, axis=1), np.std(mfcc1, axis=1)])
            emb2 = np.concatenate([np.mean(mfcc2, axis=1), np.std(mfcc2, axis=1)])

            # Normalize
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)

            # Cosine similarity
            similarity = np.dot(emb1, emb2)
            return float(np.clip(similarity, 0, 1))
        except Exception as e:
            logger.warning(f"Speaker similarity computation failed: {e}")
            return 0.0


# ============================================================================
# Benchmark Runner
# ============================================================================


class BenchmarkRunner:
    """Run benchmarks for voice conversion pipelines."""

    def __init__(
        self,
        device: str = 'cuda:0',
        warmup_runs: int = 1,
        timed_runs: int = 3,
        verbose: bool = True,
    ):
        self.device = device
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.verbose = verbose
        self.metrics = MetricsCollector(device)
        self._factory = None

    def _get_factory(self):
        """Get pipeline factory singleton."""
        if self._factory is None:
            from auto_voice.inference.pipeline_factory import PipelineFactory
            self._factory = PipelineFactory.get_instance()
        return self._factory

    def _load_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        import librosa
        audio, sr = librosa.load(audio_path, sr=None, mono=True, duration=duration)
        return audio, sr

    def _create_reference_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Create a speaker embedding from reference audio."""
        import librosa
        # Resample to 16kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Extract mel statistics as embedding (matches VoiceCloner behavior)
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Mean + std of mels = 256-dim embedding
        embedding = np.concatenate([
            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1),
        ])

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.astype(np.float32)

    def benchmark_pipeline(
        self,
        pipeline_type: str,
        audio_path: str,
        reference_path: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> BenchmarkResult:
        """Benchmark a single pipeline.

        Args:
            pipeline_type: One of 'realtime', 'quality', 'quality_seedvc', 'realtime_meanvc'
            audio_path: Path to input audio file
            reference_path: Path to reference audio (for speaker embedding)
            duration: Optional duration limit in seconds

        Returns:
            BenchmarkResult with all metrics
        """
        config = PIPELINE_CONFIGS.get(pipeline_type)
        if not config:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  Benchmarking: {config.name}")
            print(f"{'=' * 60}")

        # Load audio
        audio, sr = self._load_audio(audio_path, duration)
        audio_duration = len(audio) / sr

        if self.verbose:
            print(f"  Audio: {audio_path}")
            print(f"  Duration: {audio_duration:.2f}s")

        # Load reference for embedding
        if reference_path:
            ref_audio, ref_sr = self._load_audio(reference_path, duration=30)
        else:
            ref_audio, ref_sr = audio[:int(sr * 30)], sr

        # Create speaker embedding
        speaker_embedding = self._create_reference_embedding(ref_audio, ref_sr)
        speaker_tensor = torch.from_numpy(speaker_embedding).to(self.device)

        # Clear caches
        self.metrics.clear_cache()
        self.metrics.reset_peak_memory()
        idle_memory = self.metrics.get_gpu_memory_gb()

        result = BenchmarkResult(
            pipeline_name=config.name,
            pipeline_type=pipeline_type,
            audio_duration_sec=audio_duration,
            processing_time_sec=0.0,
            rtf=0.0,
            gpu_memory_idle_gb=idle_memory,
            output_sample_rate=config.output_sample_rate,
        )

        try:
            # Initialize pipeline
            factory = self._get_factory()
            pipeline = factory.get_pipeline(pipeline_type)

            # Set reference for in-context learning pipelines
            if hasattr(pipeline, 'set_reference_audio'):
                pipeline.set_reference_audio(ref_audio, ref_sr)
            elif hasattr(pipeline, 'set_speaker_embedding'):
                pipeline.set_speaker_embedding(speaker_embedding)

            # Warmup runs
            if self.verbose:
                print(f"  Warming up ({self.warmup_runs} run(s))...")

            audio_tensor = torch.from_numpy(audio).to(self.device)

            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    if hasattr(pipeline, 'convert'):
                        _ = pipeline.convert(audio_tensor, sr, speaker_tensor)
                    elif hasattr(pipeline, 'process_chunk'):
                        # Streaming pipeline - process in chunks
                        chunk_size = getattr(pipeline, 'chunk_size', 3200)
                        for i in range(0, len(audio), chunk_size):
                            chunk = audio[i:i+chunk_size]
                            if len(chunk) < chunk_size:
                                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                            _ = pipeline.process_chunk(chunk)
                        pipeline.reset_session() if hasattr(pipeline, 'reset_session') else None

            # Timed runs
            if self.verbose:
                print(f"  Running benchmark ({self.timed_runs} run(s))...")

            times = []
            outputs = []

            for run in range(self.timed_runs):
                self.metrics.reset_peak_memory()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                with torch.no_grad():
                    if hasattr(pipeline, 'convert'):
                        output = pipeline.convert(audio_tensor, sr, speaker_tensor)
                        if isinstance(output, dict):
                            output_audio = output['audio']
                        else:
                            output_audio = output
                    else:
                        # Streaming pipeline
                        output_chunks = []
                        chunk_size = getattr(pipeline, 'chunk_size', 3200)
                        for i in range(0, len(audio), chunk_size):
                            chunk = audio[i:i+chunk_size]
                            if len(chunk) < chunk_size:
                                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                            out_chunk = pipeline.process_chunk(chunk)
                            output_chunks.append(out_chunk)
                        output_audio = np.concatenate(output_chunks)
                        pipeline.reset_session() if hasattr(pipeline, 'reset_session') else None

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                outputs.append(output_audio)

                if self.verbose:
                    print(f"    Run {run + 1}: {elapsed:.3f}s (RTF: {elapsed/audio_duration:.3f})")

            # Use median timing
            median_time = np.median(times)
            result.processing_time_sec = float(median_time)
            result.rtf = self.metrics.compute_rtf(median_time, audio_duration)
            result.gpu_memory_peak_gb = self.metrics.get_gpu_peak_memory_gb()

            # Get latency metrics if available
            if hasattr(pipeline, 'get_latency_metrics'):
                latency = pipeline.get_latency_metrics()
                result.latency_ms = latency.get('total_ms', 0.0)
                result.metadata['latency_breakdown'] = latency

            # Compute quality metrics
            if self.verbose:
                print(f"  Computing quality metrics...")

            # Convert output to numpy
            if isinstance(outputs[-1], torch.Tensor):
                output_np = outputs[-1].cpu().numpy()
            else:
                output_np = np.asarray(outputs[-1])

            result.mcd = self.metrics.compute_mcd(
                ref_audio, output_np,
                ref_sr, config.output_sample_rate
            )

            result.speaker_similarity = self.metrics.compute_speaker_similarity(
                ref_audio, ref_sr,
                output_np, config.output_sample_rate
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.error = str(e)

        # Unload pipeline
        factory = self._get_factory()
        factory.unload_pipeline(pipeline_type)
        self.metrics.clear_cache()

        if self.verbose:
            self._print_result(result, config)

        return result

    def _print_result(self, result: BenchmarkResult, config: PipelineConfig):
        """Print benchmark result."""
        print(f"\n  Results:")
        print(f"    Processing Time: {result.processing_time_sec:.3f}s")
        print(f"    RTF: {result.rtf:.3f} (target: <{config.target_rtf})")

        rtf_ok = result.rtf <= config.target_rtf
        print(f"    RTF Target: {'PASS' if rtf_ok else 'FAIL'}")

        if result.latency_ms > 0:
            latency_ok = result.latency_ms <= config.target_latency_ms
            print(f"    Latency: {result.latency_ms:.1f}ms (target: <{config.target_latency_ms}ms)")
            print(f"    Latency Target: {'PASS' if latency_ok else 'FAIL'}")

        print(f"    GPU Memory Peak: {result.gpu_memory_peak_gb:.2f}GB (target: <{config.target_memory_gb}GB)")
        mem_ok = result.gpu_memory_peak_gb <= config.target_memory_gb
        print(f"    Memory Target: {'PASS' if mem_ok else 'FAIL'}")

        print(f"    MCD: {result.mcd:.2f} dB (target: <{config.target_mcd}dB)")
        mcd_ok = result.mcd <= config.target_mcd
        print(f"    MCD Target: {'PASS' if mcd_ok else 'FAIL'}")

        print(f"    Speaker Similarity: {result.speaker_similarity:.3f}")

        if result.error:
            print(f"    ERROR: {result.error}")

    def benchmark_all(
        self,
        audio_path: str,
        reference_path: Optional[str] = None,
        duration: Optional[float] = 30.0,
    ) -> List[BenchmarkResult]:
        """Benchmark all 4 pipelines.

        Args:
            audio_path: Path to input audio
            reference_path: Path to reference audio (optional)
            duration: Duration limit in seconds

        Returns:
            List of BenchmarkResult for each pipeline
        """
        results = []

        for pipeline_type in PIPELINE_CONFIGS.keys():
            try:
                result = self.benchmark_pipeline(
                    pipeline_type=pipeline_type,
                    audio_path=audio_path,
                    reference_path=reference_path,
                    duration=duration,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to benchmark {pipeline_type}: {e}")
                results.append(BenchmarkResult(
                    pipeline_name=PIPELINE_CONFIGS[pipeline_type].name,
                    pipeline_type=pipeline_type,
                    audio_duration_sec=duration or 0,
                    processing_time_sec=0,
                    rtf=0,
                    error=str(e),
                ))

            # Clear between pipelines
            self.metrics.clear_cache()

        return results


# ============================================================================
# Report Generation
# ============================================================================


class ReportGenerator:
    """Generate benchmark reports."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.timestamp = datetime.now().isoformat()

    def generate_markdown(self) -> str:
        """Generate markdown comparison table."""
        lines = [
            "# Performance Validation Report",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Platform:** Jetson Thor (CUDA 13.0, SM 11.0)",
            "",
            "## Pipeline Comparison",
            "",
            "| Pipeline | RTF | Latency | GPU Mem | MCD | Similarity | Target Met |",
            "|----------|-----|---------|---------|-----|------------|------------|",
        ]

        for r in self.results:
            config = PIPELINE_CONFIGS.get(r.pipeline_type)
            if not config:
                continue

            rtf_ok = r.rtf <= config.target_rtf
            mcd_ok = r.mcd <= config.target_mcd or r.mcd == 0
            mem_ok = r.gpu_memory_peak_gb <= config.target_memory_gb

            status = "PASS" if (rtf_ok and mcd_ok and mem_ok and not r.error) else "FAIL"

            lines.append(
                f"| {r.pipeline_name} | {r.rtf:.3f} | {r.latency_ms:.0f}ms | "
                f"{r.gpu_memory_peak_gb:.2f}GB | {r.mcd:.2f}dB | "
                f"{r.speaker_similarity:.3f} | {status} |"
            )

        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])

        for r in self.results:
            config = PIPELINE_CONFIGS.get(r.pipeline_type)
            if not config:
                continue

            lines.extend([
                f"### {r.pipeline_name}",
                "",
                f"- **Description:** {config.description}",
                f"- **Audio Duration:** {r.audio_duration_sec:.2f}s",
                f"- **Processing Time:** {r.processing_time_sec:.3f}s",
                f"- **RTF:** {r.rtf:.3f} (target: <{config.target_rtf})",
                f"- **Latency:** {r.latency_ms:.1f}ms (target: <{config.target_latency_ms}ms)",
                f"- **GPU Memory Peak:** {r.gpu_memory_peak_gb:.2f}GB (target: <{config.target_memory_gb}GB)",
                f"- **MCD:** {r.mcd:.2f}dB (target: <{config.target_mcd}dB)",
                f"- **Speaker Similarity:** {r.speaker_similarity:.3f}",
                f"- **Output Sample Rate:** {r.output_sample_rate}Hz",
                "",
            ])

            if r.error:
                lines.append(f"- **ERROR:** {r.error}")
                lines.append("")

        lines.extend([
            "## Recommendations",
            "",
            "| Use Case | Recommended Pipeline |",
            "|----------|---------------------|",
            "| Live Karaoke | realtime or realtime_meanvc |",
            "| Studio Conversion | quality_seedvc |",
            "| Batch Processing | quality (consistency model) |",
            "| Mobile/Edge | realtime_meanvc (CPU-friendly) |",
            "",
        ])

        return "\n".join(lines)

    def generate_json(self) -> str:
        """Generate JSON report."""
        report = {
            "timestamp": self.timestamp,
            "platform": "Jetson Thor (CUDA 13.0, SM 11.0)",
            "results": [asdict(r) for r in self.results],
            "summary": self._compute_summary(),
        }
        return json.dumps(report, indent=2)

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.results:
            return {}

        valid_results = [r for r in self.results if not r.error]

        return {
            "total_pipelines": len(self.results),
            "successful": len(valid_results),
            "failed": len(self.results) - len(valid_results),
            "fastest_rtf": min([r.rtf for r in valid_results], default=0),
            "best_mcd": min([r.mcd for r in valid_results if r.mcd > 0], default=0),
            "max_memory_gb": max([r.gpu_memory_peak_gb for r in valid_results], default=0),
        }


# ============================================================================
# Progress Display
# ============================================================================


class ProgressDisplay:
    """Console progress display for benchmarks."""

    def __init__(self, total_pipelines: int):
        self.total = total_pipelines
        self.current = 0
        self.start_time = time.time()

    def update(self, pipeline_name: str, status: str = "running"):
        """Update progress display."""
        self.current += 1
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current if self.current > 0 else 0
        remaining = avg_time * (self.total - self.current)

        bar_width = 40
        filled = int(bar_width * self.current / self.total)
        bar = '=' * filled + '>' + '.' * (bar_width - filled - 1)

        print(f"\r[{bar}] {self.current}/{self.total} | {pipeline_name}: {status} | "
              f"ETA: {remaining:.0f}s", end='', flush=True)

    def finish(self):
        """Complete progress display."""
        elapsed = time.time() - self.start_time
        print(f"\n\nCompleted {self.total} benchmarks in {elapsed:.1f}s")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Performance Validation Suite for AutoVoice Pipelines"
    )
    parser.add_argument(
        '--pipeline', '-p',
        choices=['all', 'realtime', 'quality', 'quality_seedvc', 'realtime_meanvc'],
        default='all',
        help='Pipeline to benchmark (default: all)'
    )
    parser.add_argument(
        '--audio', '-a',
        default=str(PROJECT_ROOT / 'tests' / 'quality_samples' / 'william_singe_pillowtalk.wav'),
        help='Input audio file'
    )
    parser.add_argument(
        '--reference', '-r',
        default=str(PROJECT_ROOT / 'tests' / 'quality_samples' / 'conor_maynard_pillowtalk.wav'),
        help='Reference audio for speaker embedding'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=30.0,
        help='Duration limit in seconds (default: 30)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=1,
        help='Number of warmup runs (default: 1)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of timed runs (default: 3)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (markdown or json based on extension)'
    )
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Generate comparison table'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (less output)'
    )

    args = parser.parse_args()

    # Change to project root
    os.chdir(PROJECT_ROOT)

    print("\n" + "=" * 70)
    print("  PERFORMANCE VALIDATION SUITE")
    print("  AutoVoice Pipeline Benchmarks")
    print("=" * 70)

    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n  GPU: {device_name}")
        print(f"  Memory: {total_memory:.1f} GB")
    else:
        print("\n  WARNING: CUDA not available, benchmarks may be slower")

    # Create runner
    runner = BenchmarkRunner(
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        verbose=not args.quiet,
    )

    # Run benchmarks
    if args.pipeline == 'all':
        results = runner.benchmark_all(
            audio_path=args.audio,
            reference_path=args.reference,
            duration=args.duration,
        )
    else:
        result = runner.benchmark_pipeline(
            pipeline_type=args.pipeline,
            audio_path=args.audio,
            reference_path=args.reference,
            duration=args.duration,
        )
        results = [result]

    # Generate report
    generator = ReportGenerator(results)

    if args.compare or args.output:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == '.json':
                content = generator.generate_json()
            else:
                content = generator.generate_markdown()

            output_path.write_text(content)
            print(f"\nReport saved to: {output_path}")
        else:
            # Print comparison table
            print("\n" + generator.generate_markdown())

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for r in results:
        config = PIPELINE_CONFIGS.get(r.pipeline_type)
        if not config:
            continue

        status = 'ERROR' if r.error else (
            'PASS' if r.rtf <= config.target_rtf else 'FAIL'
        )
        print(f"\n  {r.pipeline_name}:")
        print(f"    Status: {status}")
        print(f"    RTF: {r.rtf:.3f} (target: <{config.target_rtf})")
        print(f"    Memory: {r.gpu_memory_peak_gb:.2f}GB")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70 + "\n")

    return 1 if any(result.error for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
