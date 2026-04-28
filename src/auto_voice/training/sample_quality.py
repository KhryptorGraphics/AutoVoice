"""Training-sample quality analysis and recommendation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import librosa
import numpy as np
import soundfile as sf


MIN_SOURCE_DURATION_SECONDS = 0.5
MIN_ACTIVE_RATIO = 0.08
MIN_VOICED_RATIO = 0.02
MIN_PEAK_AMPLITUDE = 1e-4
MIN_SPEAKER_PURITY = 0.75
MIN_TRAINABLE_SAMPLES = 5


def _merge_quality_metadata(
    analyzed: Dict[str, Any],
    provided: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(analyzed)
    if provided:
        merged.update(dict(provided))
    return merged


def _load_sidecar_metadata(audio_path: Path) -> Dict[str, Any]:
    sidecar = audio_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def analyze_training_sample(
    audio_path: str | Path,
    *,
    quality_metadata: Optional[Mapping[str, Any]] = None,
    provenance: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze one training sample and return normalized QA metadata."""
    path = Path(audio_path)
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except TypeError:
        try:
            audio, sample_rate = sf.read(str(path))
        except sf.LibsndfileError as exc:
            raise ValueError(f"Invalid training sample audio: {path}") from exc
    except sf.LibsndfileError as exc:
        raise ValueError(f"Invalid training sample audio: {path}") from exc
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1 if audio_np.shape[0] > audio_np.shape[1] else 0)
    audio_np = audio_np.squeeze()

    duration_seconds = float(len(audio_np) / sample_rate) if sample_rate > 0 else 0.0
    peak_amplitude = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    active_ratio = float(np.mean(np.abs(audio_np) >= MIN_PEAK_AMPLITUDE)) if audio_np.size else 0.0

    analysis_window = audio_np[: min(len(audio_np), int(sample_rate * 30))]
    try:
        f0, voiced, _ = librosa.pyin(
            analysis_window,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            hop_length=256,
        )
        voiced_mask = np.asarray(voiced, dtype=bool) if voiced is not None else np.zeros(0, dtype=bool)
        voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
    except Exception:
        voiced_ratio = 0.0

    issues: List[str] = []
    recommendations: List[str] = []

    if duration_seconds < MIN_SOURCE_DURATION_SECONDS:
        issues.append("sample_too_short")
        recommendations.append("Capture a longer isolated vocal segment.")
    if peak_amplitude < MIN_PEAK_AMPLITUDE:
        issues.append("effectively_silent")
        recommendations.append("Replace the sample with a louder clean vocal take.")
    if active_ratio < MIN_ACTIVE_RATIO:
        issues.append("silence_heavy")
        recommendations.append("Trim silence and breathing-only sections before retraining.")
    if voiced_ratio < MIN_VOICED_RATIO:
        issues.append("insufficient_pitch_coverage")
        recommendations.append("Add sung material with clear voiced notes across the target range.")

    sidecar_metadata = _load_sidecar_metadata(path)
    sidecar_quality = sidecar_metadata.get("quality_metadata")
    candidate_quality = (
        sidecar_quality
        if isinstance(sidecar_quality, Mapping)
        else sidecar_metadata if isinstance(sidecar_metadata, Mapping) else {}
    )
    merged = _merge_quality_metadata(
        {
            "sample_rate": int(sample_rate),
            "duration_seconds": duration_seconds,
            "peak_amplitude": peak_amplitude,
            "active_ratio": active_ratio,
            "voiced_ratio": voiced_ratio,
            "provenance": provenance or str(path),
        },
        candidate_quality,
    )
    merged = _merge_quality_metadata(merged, quality_metadata)

    speaker_purity = merged.get("speaker_purity")
    diarization_ok = merged.get("diarization_ok")
    if speaker_purity is not None and float(speaker_purity) < MIN_SPEAKER_PURITY:
        issues.append("speaker_purity_too_low")
        recommendations.append("Filter the clip to the target singer or re-extract cleaner vocals.")
    if diarization_ok is False:
        issues.append("diarization_failed")
        recommendations.append("Re-run diarization or manually assign the correct singer segment.")

    status = "pass"
    if any(
        issue
        in {
            "sample_too_short",
            "effectively_silent",
            "silence_heavy",
            "insufficient_pitch_coverage",
            "speaker_purity_too_low",
            "diarization_failed",
        }
        for issue in issues
    ):
        status = "fail"
    elif issues:
        status = "warn"

    merged["issues"] = issues
    merged["recommendations"] = recommendations
    merged["qa_status"] = status
    return merged


def summarize_training_samples(samples: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate sample QA metadata for a profile."""
    entries = list(samples)
    status_counts = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
    total_duration = 0.0
    trainable_duration = 0.0
    recommendations: List[str] = []
    failing_samples: List[str] = []

    for sample in entries:
        quality = dict(getattr(sample, "quality_metadata", {}) or {})
        status = str(quality.get("qa_status") or "unknown")
        status_counts[status if status in status_counts else "unknown"] += 1
        total_duration += float(getattr(sample, "duration", 0.0) or quality.get("duration_seconds") or 0.0)
        if status != "fail":
            trainable_duration += float(getattr(sample, "duration", 0.0) or quality.get("duration_seconds") or 0.0)
        else:
            failing_samples.append(getattr(sample, "sample_id", "unknown"))
        for recommendation in quality.get("recommendations", []) or []:
            if recommendation not in recommendations:
                recommendations.append(str(recommendation))

    trainable_samples = status_counts["pass"] + status_counts["warn"]
    if trainable_samples < MIN_TRAINABLE_SAMPLES:
        recommendations.append(
            f"Collect at least {MIN_TRAINABLE_SAMPLES - trainable_samples} more clean vocal samples before the next major retraining pass."
        )
    if status_counts["fail"] > 0:
        recommendations.append("Remove or replace failed samples before starting the next training job.")

    return {
        "sample_count": len(entries),
        "trainable_sample_count": trainable_samples,
        "failing_sample_ids": failing_samples,
        "status_counts": status_counts,
        "total_duration_seconds": total_duration,
        "trainable_duration_seconds": trainable_duration,
        "recommendations": recommendations,
        "ready_for_training": trainable_samples > 0 and status_counts["fail"] < len(entries),
    }
