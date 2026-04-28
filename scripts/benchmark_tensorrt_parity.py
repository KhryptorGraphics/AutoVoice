#!/usr/bin/env python3
"""Benchmark checkpoint-aligned TensorRT engines against PyTorch artifacts.

This lane is intentionally narrower than the bootstrap full-pipeline TRT smoke
test. It only runs strict numerical parity when an engine sidecar identifies the
checkpoint that produced the engine, so release evidence never compares
unrelated random/bootstrap weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from auto_voice.export.tensorrt_engine import TRT_AVAILABLE, TRTEngineBuilder
from export_hq_lora_tensorrt import _find_artist, _load_adapter


DEFAULT_METADATA_GLOB = "models/*/artifacts/tensorrt/engine_metadata.json"


def _git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("artist_key", "checkpoint_path", "engine_path", "onnx_path", "precision")
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(f"{path} missing required metadata fields: {', '.join(missing)}")
    return payload


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _benchmark_pytorch(model: torch.nn.Module, content: torch.Tensor, *, warmup: int, runs: int) -> tuple[torch.Tensor, list[float]]:
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(content)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        output = model(content)
        for _ in range(runs):
            start = time.perf_counter()
            output = model(content)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
    return output, latencies


def _run_one(metadata_path: Path, *, frames: int, warmup: int, runs: int, tolerance: dict[str, float]) -> dict[str, Any]:
    metadata = _load_metadata(metadata_path)
    checkpoint_path = _resolve_path(str(metadata["checkpoint_path"]))
    engine_path = _resolve_path(str(metadata["engine_path"]))
    onnx_path = _resolve_path(str(metadata["onnx_path"]))

    missing = [
        str(path)
        for path in (checkpoint_path, engine_path, onnx_path)
        if not path.exists()
    ]
    if missing:
        return {
            "metadata_path": str(metadata_path),
            "artist_key": metadata.get("artist_key"),
            "status": "blocked",
            "reason": "missing checkpoint-aligned artifacts",
            "missing": missing,
        }

    if not TRT_AVAILABLE:
        return {
            "metadata_path": str(metadata_path),
            "artist_key": metadata.get("artist_key"),
            "status": "blocked",
            "reason": "TensorRT is not installed in this environment",
        }

    torch.manual_seed(1234)
    np.random.seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = _find_artist(str(metadata["artist_key"]))
    base_seed = int(metadata.get("base_seed", 1234))
    model, checkpoint_payload = _load_adapter(spec, base_seed=base_seed)
    model = model.to(device).eval()

    content = torch.randn(1, frames, int(metadata.get("config", {}).get("input_dim", 768)), device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pytorch_output, pytorch_latencies = _benchmark_pytorch(model, content, warmup=warmup, runs=runs)
    pytorch_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    builder = TRTEngineBuilder()
    engine = builder.load_engine(str(engine_path))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trt_inputs = {"content": content.detach().cpu().numpy().astype(np.float32)}
    trt_output_np = builder.infer(engine, trt_inputs)["adapted_content"]
    latency_stats = builder.benchmark(engine, trt_inputs, n_runs=runs, warmup=warmup)
    trt_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    trt_output = torch.from_numpy(trt_output_np).to(device=device, dtype=pytorch_output.dtype)
    diff = (pytorch_output - trt_output).float()
    mean_abs_error = float(diff.abs().mean().item())
    max_abs_error = float(diff.abs().max().item())
    rmse = float(diff.pow(2).mean().sqrt().item())
    denom = pytorch_output.float().norm().clamp(min=1e-12)
    relative_l2_error = float(diff.norm().item() / denom.item())
    cosine = float(torch.nn.functional.cosine_similarity(
        pytorch_output.float().reshape(1, -1),
        trt_output.float().reshape(1, -1),
    ).item())

    parity_passed = (
        mean_abs_error <= tolerance["mean_abs_error"]
        and relative_l2_error <= tolerance["relative_l2_error"]
        and cosine >= tolerance["cosine_similarity"]
    )
    latency_guard_passed = latency_stats.mean_ms <= max(statistics.mean(pytorch_latencies) * 1.10, 0.01)

    return {
        "metadata_path": str(metadata_path),
        "artist_key": metadata["artist_key"],
        "display_name": metadata.get("display_name"),
        "status": "passed" if parity_passed and latency_guard_passed else "failed",
        "checkpoint_aligned": True,
        "precision": metadata.get("precision"),
        "shape": {"content": list(content.shape)},
        "artifact_hashes": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "onnx_sha256": _sha256(onnx_path),
            "engine_sha256": _sha256(engine_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": metadata.get("checkpoint_epoch", checkpoint_payload.get("epoch")),
            "global_step": metadata.get("checkpoint_global_step", checkpoint_payload.get("global_step")),
            "loss": metadata.get("checkpoint_loss", checkpoint_payload.get("loss")),
            "base_seed": base_seed,
        },
        "metrics": {
            "mean_abs_error": mean_abs_error,
            "max_abs_error": max_abs_error,
            "rmse": rmse,
            "relative_l2_error": relative_l2_error,
            "cosine_similarity": cosine,
            "pytorch_latency_ms_mean": statistics.mean(pytorch_latencies),
            "pytorch_latency_ms_p95": _percentile(pytorch_latencies, 95),
            "trt_latency_ms_mean": latency_stats.mean_ms,
            "trt_latency_ms_p95": latency_stats.p95_ms,
            "pytorch_vram_mb_peak": pytorch_vram_mb,
            "trt_vram_mb_peak": trt_vram_mb,
        },
        "tolerances": tolerance,
        "parity_passed": parity_passed,
        "latency_guard_passed": latency_guard_passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata_paths = sorted(PROJECT_ROOT.glob(args.metadata_glob))
    if args.metadata:
        metadata_paths.extend(args.metadata)
    seen: set[Path] = set()
    unique_metadata_paths: list[Path] = []
    for path in metadata_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_metadata_paths.append(path)

    tolerance = {
        "mean_abs_error": args.mean_abs_error,
        "relative_l2_error": args.relative_l2_error,
        "cosine_similarity": args.cosine_similarity,
    }
    results = [
        _run_one(path, frames=args.frames, warmup=args.warmup, runs=args.runs, tolerance=tolerance)
        for path in unique_metadata_paths
    ]
    passed = [result for result in results if result.get("status") == "passed"]
    failed = [result for result in results if result.get("status") == "failed"]
    blocked = [result for result in results if result.get("status") == "blocked"]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "kind": "checkpoint_aligned_tensorrt_parity",
        "ok": bool(passed) and not failed and not blocked,
        "summary": {
            "metadata_count": len(unique_metadata_paths),
            "passed": len(passed),
            "failed": len(failed),
            "blocked": len(blocked),
        },
        "results": results,
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-glob", default=DEFAULT_METADATA_GLOB)
    parser.add_argument("--metadata", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("reports/benchmarks/tensorrt-parity/latest/parity_report.json"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--mean-abs-error", type=float, default=0.05)
    parser.add_argument("--relative-l2-error", type=float, default=0.05)
    parser.add_argument("--cosine-similarity", type=float, default=0.999)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(output)}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
