"""Benchmark execution and report generation for voice conversion evaluation."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .quality_metrics import QualityMetrics


class BenchmarkRunner:
    """Runs benchmarks, aggregates metrics, and exports report artifacts."""

    def __init__(self, metrics: Optional[QualityMetrics] = None):
        self.metrics = metrics or QualityMetrics()

    def run_single(
        self,
        pipeline,
        source_audio: torch.Tensor,
        target_speaker: torch.Tensor,
        sample_rate: int = 24000,
        reference_audio: Optional[torch.Tensor] = None,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run benchmark on a single sample."""
        start_time = time.time()
        result = pipeline.convert(source_audio, sample_rate, target_speaker)
        latency_ms = (time.time() - start_time) * 1000

        if isinstance(result, dict):
            converted_audio = result["audio"]
            converted_speaker = result.get("speaker_embedding")
        else:
            converted_audio = result
            converted_speaker = None

        ref_audio = reference_audio if reference_audio is not None else source_audio
        metrics_result = self.metrics.compute_all(
            reference_audio=ref_audio,
            converted_audio=converted_audio,
            target_speaker=target_speaker,
            sample_rate=sample_rate,
            converted_speaker=converted_speaker,
        )

        return {
            "sample_id": sample_id,
            "converted_audio": converted_audio,
            "source_audio": source_audio,
            "reference_audio": ref_audio,
            "metrics": metrics_result,
            "latency_ms": latency_ms,
            "sample_rate": sample_rate,
            "metadata": metadata or {},
        }

    def run_dataset(
        self,
        pipeline,
        dataset,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run benchmark on an entire dataset."""
        results = []

        for index, sample in enumerate(dataset):
            if max_samples is not None and index >= max_samples:
                break

            result = self.run_single(
                pipeline=pipeline,
                source_audio=sample["source_audio"],
                target_speaker=sample["target_speaker"],
                sample_rate=sample.get("metadata", {}).get("sample_rate", 24000),
                reference_audio=sample.get("reference_audio"),
                sample_id=sample.get("sample_id") or f"sample_{index:03d}",
                metadata=sample.get("metadata", {}),
            )
            result["sample_idx"] = index
            results.append(result)

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics from multiple samples."""
        metrics_lists: Dict[str, List[float]] = {}

        for result in results:
            metrics = result.get("metrics", result)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_lists.setdefault(key, []).append(float(value))

            latency_ms = result.get("latency_ms")
            if isinstance(latency_ms, (int, float)):
                metrics_lists.setdefault("latency_ms", []).append(float(latency_ms))

        aggregated: Dict[str, float] = {}
        for key, values in metrics_lists.items():
            arr = np.asarray(values, dtype=np.float64)
            aggregated[f"{key}_mean"] = float(np.mean(arr))
            aggregated[f"{key}_std"] = float(np.std(arr))
            aggregated[f"{key}_min"] = float(np.min(arr))
            aggregated[f"{key}_max"] = float(np.max(arr))

        aggregated["sample_count"] = float(len(results))
        return aggregated

    def export_markdown(
        self,
        results: Dict[str, float],
        title: str = "Benchmark Results",
    ) -> str:
        """Export aggregated results to Markdown."""
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Quality Metrics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|--------|------|-----|-----|-----|",
        ]

        base_metrics = set()
        for key in results.keys():
            for suffix in ["_mean", "_std", "_min", "_max"]:
                if key.endswith(suffix):
                    base_metrics.add(key[: -len(suffix)])
                    break

        display_names = {
            "mcd": "MCD",
            "f0_rmse": "F0_RMSE",
            "pitch_corr": "PITCH_CORR",
            "speaker_similarity": "SPEAKER_SIMILARITY",
            "mos_pred": "MOS_PRED",
            "pesq": "PESQ",
            "stoi": "STOI",
            "snr": "SNR",
            "latency_ms": "LATENCY_MS",
        }

        for metric in sorted(base_metrics):
            mean = results.get(f"{metric}_mean", "N/A")
            std = results.get(f"{metric}_std", "N/A")
            min_val = results.get(f"{metric}_min", "N/A")
            max_val = results.get(f"{metric}_max", "N/A")

            if isinstance(mean, float):
                mean = f"{mean:.3f}"
            if isinstance(std, float):
                std = f"{std:.3f}"
            if isinstance(min_val, float):
                min_val = f"{min_val:.3f}"
            if isinstance(max_val, float):
                max_val = f"{max_val:.3f}"

            display_name = display_names.get(metric, metric.upper())
            lines.append(f"| {display_name} | {mean} | {std} | {min_val} | {max_val} |")

        lines.extend(
            [
                "",
                "## Quality Targets",
                "",
                "| Metric | Target | Status |",
                "|--------|--------|--------|",
            ]
        )

        targets = {
            "mcd_mean": ("< 5.0 dB", lambda value: value < 5.0),
            "f0_rmse_mean": ("< 20 cents", lambda value: value < 20.0),
            "pitch_corr_mean": ("> 0.90", lambda value: value > 0.90),
            "speaker_similarity_mean": (">= 0.85", lambda value: value >= 0.85),
            "pesq_mean": (">= 3.5", lambda value: value >= 3.5),
            "stoi_mean": (">= 0.85", lambda value: value >= 0.85),
        }

        for key, (target, predicate) in targets.items():
            value = results.get(key)
            status = "N/A"
            if isinstance(value, (int, float)):
                status = "PASS" if predicate(float(value)) else "FAIL"
            lines.append(f"| {key.replace('_mean', '')} | {target} | {status} |")

        return "\n".join(lines)

    def export_json(self, results: Dict[str, Any]) -> str:
        return json.dumps(results, indent=2)

    def benchmark(
        self,
        pipeline,
        dataset,
        output_dir: str,
        max_samples: Optional[int] = None,
        title: str = "Benchmark Results",
    ) -> Dict[str, Any]:
        """Run the dataset benchmark and emit report artifacts."""
        results = self.run_dataset(pipeline=pipeline, dataset=dataset, max_samples=max_samples)
        artifacts = self.write_report_artifacts(results=results, output_dir=output_dir, title=title)
        return {
            "results": results,
            "summary": artifacts["summary"],
            "artifacts": artifacts,
        }

    def write_report_artifacts(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        title: str = "Benchmark Results",
    ) -> Dict[str, Any]:
        """Write JSON, CSV, Markdown, figures, and TensorBoard logs."""
        output_path = Path(output_dir)
        figures_dir = output_path / "figures"
        tensorboard_dir = output_path / "tensorboard"
        output_path.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        aggregated = self.aggregate_results(results)
        sample_rows = [self._serialize_result_row(result) for result in results]

        summary_path = output_path / "summary.json"
        metrics_csv_path = output_path / "metrics.csv"
        report_md_path = output_path / "report.md"

        summary_payload = {
            "title": title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": aggregated,
            "samples": sample_rows,
        }
        summary_path.write_text(self.export_json(summary_payload))
        self._write_metrics_csv(sample_rows, metrics_csv_path)
        report_md_path.write_text(self.export_markdown(aggregated, title=title))

        figure_paths = self._write_figures(results, figures_dir)
        self._write_tensorboard(summary=aggregated, sample_rows=sample_rows, log_dir=tensorboard_dir)

        return {
            "summary": aggregated,
            "summary_json": str(summary_path),
            "metrics_csv": str(metrics_csv_path),
            "report_md": str(report_md_path),
            "figures": figure_paths,
            "tensorboard_dir": str(tensorboard_dir),
        }

    def _serialize_result_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "sample_id": result.get("sample_id"),
            "sample_idx": result.get("sample_idx"),
            "sample_rate": result.get("sample_rate"),
            "latency_ms": result.get("latency_ms"),
        }
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[f"metadata_{key}"] = value

        for key, value in result.get("metrics", {}).items():
            row[key] = value
        return row

    def _write_metrics_csv(self, rows: List[Dict[str, Any]], path: Path) -> None:
        fieldnames = sorted({key for row in rows for key in row.keys()}) or ["sample_id"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_figures(self, results: List[Dict[str, Any]], figures_dir: Path) -> Dict[str, str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_paths: Dict[str, str] = {}
        if not results:
            return figure_paths

        aggregated = self.aggregate_results(results)
        mean_metrics = {
            key.replace("_mean", ""): value
            for key, value in aggregated.items()
            if key.endswith("_mean") and key not in {"sample_count"}
        }

        aggregate_path = figures_dir / "aggregate_metrics.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(list(mean_metrics.keys()), list(mean_metrics.values()))
        ax.set_title("Aggregate Mean Metrics")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(aggregate_path)
        plt.close(fig)
        figure_paths["aggregate_metrics"] = str(aggregate_path)

        speaker_similarity_values = [
            result["metrics"].get("speaker_similarity")
            for result in results
            if isinstance(result.get("metrics", {}).get("speaker_similarity"), (int, float))
        ]
        if speaker_similarity_values:
            speaker_path = figures_dir / "speaker_similarity_distribution.png"
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(speaker_similarity_values, bins=min(10, len(speaker_similarity_values)))
            ax.set_title("Speaker Similarity Distribution")
            ax.set_xlabel("Speaker Similarity")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(speaker_path)
            plt.close(fig)
            figure_paths["speaker_similarity_distribution"] = str(speaker_path)

        first_result = results[0]
        reference_audio = self._to_numpy_audio(first_result.get("reference_audio"))
        converted_audio = self._to_numpy_audio(first_result.get("converted_audio"))
        sample_rate = int(first_result.get("sample_rate") or 24000)

        if reference_audio.size and converted_audio.size:
            pitch_path = figures_dir / "pitch_contour_overlay.png"
            self._plot_pitch_contours(reference_audio, converted_audio, sample_rate, pitch_path)
            figure_paths["pitch_contour_overlay"] = str(pitch_path)

            spectrogram_path = figures_dir / "spectrogram_comparison.png"
            self._plot_spectrograms(reference_audio, converted_audio, sample_rate, spectrogram_path)
            figure_paths["spectrogram_comparison"] = str(spectrogram_path)

        return figure_paths

    def _write_tensorboard(
        self,
        summary: Dict[str, float],
        sample_rows: List[Dict[str, Any]],
        log_dir: Path,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"summary/{key}", float(value), 0)

        for index, row in enumerate(sample_rows):
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"samples/{key}", float(value), index)
        writer.flush()
        writer.close()

    @staticmethod
    def _to_numpy_audio(audio: Any) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy().squeeze().astype(np.float64)
        if audio is None:
            return np.asarray([], dtype=np.float64)
        return np.asarray(audio, dtype=np.float64).squeeze()

    def _plot_pitch_contours(
        self,
        reference_audio: np.ndarray,
        converted_audio: np.ndarray,
        sample_rate: int,
        output_path: Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f0_ref = self.metrics._get_pitch_metric()._extract_f0(reference_audio, sample_rate)
        f0_conv = self.metrics._get_pitch_metric()._extract_f0(converted_audio, sample_rate)

        frames = np.arange(min(len(f0_ref), len(f0_conv)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, f0_ref[: len(frames)], label="reference")
        ax.plot(frames, f0_conv[: len(frames)], label="converted")
        ax.set_title("Pitch Contour Overlay")
        ax.set_xlabel("Frame")
        ax.set_ylabel("F0 (Hz)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    def _plot_spectrograms(
        self,
        reference_audio: np.ndarray,
        converted_audio: np.ndarray,
        sample_rate: int,
        output_path: Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import librosa.display

        ref_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=reference_audio.astype(np.float32),
                sr=sample_rate,
                n_mels=128,
                hop_length=256,
            )
            + 1e-6,
            ref=np.max,
        )
        conv_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=converted_audio.astype(np.float32),
                sr=sample_rate,
                n_mels=128,
                hop_length=256,
            )
            + 1e-6,
            ref=np.max,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        librosa.display.specshow(ref_spec, sr=sample_rate, hop_length=256, x_axis="time", y_axis="mel", ax=axes[0])
        axes[0].set_title("Reference")
        librosa.display.specshow(conv_spec, sr=sample_rate, hop_length=256, x_axis="time", y_axis="mel", ax=axes[1])
        axes[1].set_title("Converted")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
