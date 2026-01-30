"""Benchmark runner for voice conversion evaluation.

Runs conversion pipeline on benchmark samples and collects metrics.
Provides aggregation and export functionality for results.
"""
import json
import time
from typing import Dict, List, Optional, Any
import torch
import numpy as np

from .quality_metrics import QualityMetrics


class BenchmarkRunner:
    """Runs benchmarks and collects quality metrics.

    Executes voice conversion pipeline on samples, measures quality
    and latency, and aggregates results.

    Args:
        metrics: QualityMetrics instance (created if not provided)
    """

    def __init__(self, metrics: Optional[QualityMetrics] = None):
        self.metrics = metrics or QualityMetrics()

    def run_single(
        self,
        pipeline,
        source_audio: torch.Tensor,
        target_speaker: torch.Tensor,
        sample_rate: int = 24000,
        reference_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run benchmark on a single sample.

        Args:
            pipeline: Voice conversion pipeline with convert() method
            source_audio: Source audio tensor
            target_speaker: Target speaker embedding
            sample_rate: Audio sample rate
            reference_audio: Optional reference for metric computation

        Returns:
            Dictionary with converted audio, metrics, and latency
        """
        # Time the conversion
        start_time = time.time()
        result = pipeline.convert(source_audio, sample_rate, target_speaker)
        latency_ms = (time.time() - start_time) * 1000

        converted_audio = result['audio']

        # Compute metrics
        # Use source as reference if no ground truth provided
        ref_audio = reference_audio if reference_audio is not None else source_audio

        metrics_result = self.metrics.compute_all(
            reference_audio=ref_audio,
            converted_audio=converted_audio,
            target_speaker=target_speaker,
            sample_rate=sample_rate,
        )

        return {
            'converted_audio': converted_audio,
            'metrics': metrics_result,
            'latency_ms': latency_ms,
        }

    def run_dataset(
        self,
        pipeline,
        dataset,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run benchmark on entire dataset.

        Args:
            pipeline: Voice conversion pipeline
            dataset: BenchmarkDataset instance
            max_samples: Maximum number of samples to process

        Returns:
            List of result dictionaries
        """
        results = []

        for i, sample in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break

            result = self.run_single(
                pipeline=pipeline,
                source_audio=sample['source_audio'],
                target_speaker=sample['target_speaker'],
                sample_rate=sample.get('metadata', {}).get('sample_rate', 24000),
                reference_audio=sample.get('reference_audio'),
            )
            result['sample_idx'] = i
            results.append(result)

        return results

    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Aggregate metrics from multiple samples.

        Args:
            results: List of result dictionaries with metrics

        Returns:
            Aggregated statistics (mean, std) for each metric
        """
        # Extract metrics values
        metrics_lists: Dict[str, List[float]] = {}

        for result in results:
            # Handle both direct metric dicts and nested 'metrics' key
            metrics = result.get('metrics', result)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metrics_lists:
                        metrics_lists[key] = []
                    metrics_lists[key].append(value)

        # Compute statistics
        aggregated = {}
        for key, values in metrics_lists.items():
            arr = np.array(values)
            aggregated[f'{key}_mean'] = float(np.mean(arr))
            aggregated[f'{key}_std'] = float(np.std(arr))
            aggregated[f'{key}_min'] = float(np.min(arr))
            aggregated[f'{key}_max'] = float(np.max(arr))

        return aggregated

    def export_markdown(
        self,
        results: Dict[str, float],
        title: str = "Benchmark Results",
    ) -> str:
        """Export results to markdown format.

        Args:
            results: Aggregated results dictionary
            title: Report title

        Returns:
            Markdown formatted string
        """
        lines = [
            f"# {title}",
            "",
            "## Quality Metrics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|--------|------|-----|-----|-----|",
        ]

        # Group by base metric name
        base_metrics = set()
        for key in results.keys():
            for suffix in ['_mean', '_std', '_min', '_max']:
                if key.endswith(suffix):
                    base_metrics.add(key[:-len(suffix)])
                    break

        for metric in sorted(base_metrics):
            mean = results.get(f'{metric}_mean', 'N/A')
            std = results.get(f'{metric}_std', 'N/A')
            min_val = results.get(f'{metric}_min', 'N/A')
            max_val = results.get(f'{metric}_max', 'N/A')

            if isinstance(mean, float):
                mean = f"{mean:.3f}"
            if isinstance(std, float):
                std = f"{std:.3f}"
            if isinstance(min_val, float):
                min_val = f"{min_val:.3f}"
            if isinstance(max_val, float):
                max_val = f"{max_val:.3f}"

            lines.append(f"| {metric.upper()} | {mean} | {std} | {min_val} | {max_val} |")

        lines.append("")
        lines.append("## Quality Targets")
        lines.append("")
        lines.append("| Metric | Target | Status |")
        lines.append("|--------|--------|--------|")

        mcd_mean = results.get('mcd_mean', float('inf'))
        f0_mean = results.get('f0_rmse_mean', float('inf'))
        spk_mean = results.get('speaker_similarity_mean', 0)

        mcd_status = "✅" if mcd_mean < 5.0 else "❌"
        f0_status = "✅" if f0_mean < 20.0 else "❌"
        spk_status = "✅" if spk_mean > 0.85 else "❌"

        lines.append(f"| MCD | < 5.0 dB | {mcd_status} |")
        lines.append(f"| F0 RMSE | < 20 cents | {f0_status} |")
        lines.append(f"| Speaker Similarity | > 0.85 | {spk_status} |")

        return "\n".join(lines)

    def export_json(
        self,
        results: Dict[str, Any],
    ) -> str:
        """Export results to JSON format.

        Args:
            results: Results dictionary

        Returns:
            JSON formatted string
        """
        return json.dumps(results, indent=2)
