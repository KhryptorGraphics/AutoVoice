"""Canonical runtime and packaged artifact contract for AutoVoice."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


PipelineStability = str
PipelineMode = str


@dataclass(frozen=True)
class PipelineDefinition:
    """One supported voice-conversion pipeline contract."""

    pipeline_type: str
    mode: PipelineMode
    stability: PipelineStability
    sample_rate: int
    latency_target_ms: int
    description: str
    runtime_backend: str
    model_family: str
    features: Tuple[str, ...] = ()
    canonical_default: bool = False


PIPELINE_DEFINITIONS: dict[str, PipelineDefinition] = {
    "realtime": PipelineDefinition(
        pipeline_type="realtime",
        mode="live",
        stability="canonical",
        sample_rate=22050,
        latency_target_ms=100,
        description="Canonical low-latency karaoke and live/offline realtime conversion path",
        runtime_backend="pytorch",
        model_family="realtime",
        canonical_default=True,
    ),
    "quality_seedvc": PipelineDefinition(
        pipeline_type="quality_seedvc",
        mode="offline",
        stability="canonical",
        sample_rate=44100,
        latency_target_ms=2000,
        description="Canonical offline high-quality Seed-VC DiT-CFM path",
        runtime_backend="pytorch",
        model_family="seed_vc",
        features=(
            "In-context learning from reference audio",
            "Conditional Flow Matching",
            "Whisper encoder for semantic content",
            "BigVGAN v2 vocoder at 44.1kHz",
        ),
        canonical_default=True,
    ),
    "quality": PipelineDefinition(
        pipeline_type="quality",
        mode="offline",
        stability="experimental",
        sample_rate=24000,
        latency_target_ms=3000,
        description="Experimental CoMoSVC offline quality path",
        runtime_backend="pytorch",
        model_family="comosvc",
    ),
    "quality_shortcut": PipelineDefinition(
        pipeline_type="quality_shortcut",
        mode="offline",
        stability="experimental",
        sample_rate=44100,
        latency_target_ms=800,
        description="Experimental Seed-VC shortcut flow path",
        runtime_backend="pytorch",
        model_family="seed_vc_shortcut",
        features=(
            "2-step shortcut flow matching",
            "Self-consistency loss training",
            "GPU-accelerated",
        ),
    ),
    "realtime_meanvc": PipelineDefinition(
        pipeline_type="realtime_meanvc",
        mode="live",
        stability="experimental",
        sample_rate=16000,
        latency_target_ms=350,
        description="Experimental CPU-friendly single-step MeanVC streaming live path",
        runtime_backend="pytorch",
        model_family="meanvc",
        features=(
            "Single-step mean flow inference",
            "Chunk-wise autoregressive processing with KV-cache",
            "CPU-optimized model",
        ),
    ),
}

CANONICAL_OFFLINE_PIPELINE = "quality_seedvc"
CANONICAL_LIVE_PIPELINE = "realtime"
TRAINED_PROFILE_SERVING_PIPELINE = "realtime"
REFERENCE_AUDIO_OFFLINE_PIPELINE = "quality_seedvc"

OFFLINE_PIPELINES = {
    pipeline.pipeline_type
    for pipeline in PIPELINE_DEFINITIONS.values()
    if pipeline.mode == "offline" or pipeline.pipeline_type == "realtime"
}
LIVE_PIPELINES = {
    pipeline.pipeline_type
    for pipeline in PIPELINE_DEFINITIONS.values()
    if pipeline.mode == "live"
}
LEGACY_PIPELINES = {"realtime", "quality"}
EXPERIMENTAL_PIPELINES = {
    pipeline.pipeline_type
    for pipeline in PIPELINE_DEFINITIONS.values()
    if pipeline.stability == "experimental"
}

REFERENCE_AUDIO_FIELDS = (
    "path",
    "source",
    "sample_id",
    "source_file",
    "duration_seconds",
    "created_at",
)


def get_pipeline_definition(pipeline_type: str) -> PipelineDefinition:
    """Return the canonical definition for one pipeline."""
    try:
        return PIPELINE_DEFINITIONS[pipeline_type]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}") from exc


def normalize_pipeline_choice(
    value: Optional[str],
    *,
    mode: str,
) -> str:
    """Normalize a pipeline choice to a valid pipeline for the requested mode."""
    if mode == "offline":
        valid = OFFLINE_PIPELINES
        default = CANONICAL_OFFLINE_PIPELINE
    elif mode == "live":
        valid = LIVE_PIPELINES
        default = CANONICAL_LIVE_PIPELINE
    else:
        raise ValueError(f"Unknown pipeline mode: {mode}")

    if value in valid:
        return str(value)
    return default


def get_pipeline_status_template() -> Dict[str, Dict[str, Any]]:
    """Return static status metadata for all pipeline definitions."""
    status: Dict[str, Dict[str, Any]] = {}
    for pipeline in PIPELINE_DEFINITIONS.values():
        entry: Dict[str, Any] = {
            "latency_target_ms": pipeline.latency_target_ms,
            "sample_rate": pipeline.sample_rate,
            "description": pipeline.description,
            "runtime_backend": pipeline.runtime_backend,
            "model_family": pipeline.model_family,
            "stability": pipeline.stability,
            "mode": pipeline.mode,
            "canonical_default": pipeline.canonical_default,
        }
        if pipeline.features:
            entry["features"] = list(pipeline.features)
        status[pipeline.pipeline_type] = entry
    return status


def normalize_reference_audio_entries(
    entries: Optional[Iterable[object]],
    *,
    require_exists: bool = False,
) -> list[Dict[str, Any]]:
    """Normalize reference-audio entries to one canonical list schema."""
    normalized: list[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    for entry in entries or ():
        payload: Dict[str, Any]
        if isinstance(entry, Mapping):
            payload = dict(entry)
        else:
            payload = {"path": entry}

        raw_path = payload.get("path") or payload.get("vocals_path")
        if raw_path is None:
            continue

        path_text = str(Path(str(raw_path)))
        if not path_text:
            continue
        if require_exists and not Path(path_text).exists():
            continue
        if path_text in seen_paths:
            continue

        duration = payload.get("duration_seconds", payload.get("duration"))
        normalized_entry: Dict[str, Any] = {"path": path_text}
        if payload.get("source"):
            normalized_entry["source"] = str(payload["source"])
        if payload.get("sample_id"):
            normalized_entry["sample_id"] = str(payload["sample_id"])
        if payload.get("source_file"):
            normalized_entry["source_file"] = str(payload["source_file"])
        if duration not in (None, ""):
            normalized_entry["duration_seconds"] = float(duration)
        if payload.get("created_at"):
            normalized_entry["created_at"] = str(payload["created_at"])

        for field_name in REFERENCE_AUDIO_FIELDS:
            if field_name in normalized_entry:
                continue
            if field_name in payload and payload[field_name] not in (None, ""):
                normalized_entry[field_name] = payload[field_name]

        normalized.append(normalized_entry)
        seen_paths.add(path_text)

    return normalized


@dataclass
class PackagedArtifactManifest:
    """Machine-readable packaged model/profile manifest."""

    schema_version: int
    compatibility_version: str
    profile_id: str
    display_name: str
    model_family: str
    canonical_pipeline: str
    sample_rate: int
    speaker_embedding_dim: int
    mel_bins: int
    artifacts: Dict[str, Optional[str]]
    compatibility: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_packaged_artifact_manifest(
    *,
    profile_id: str,
    display_name: str,
    model_family: str,
    canonical_pipeline: str,
    sample_rate: int,
    speaker_embedding_dim: int,
    mel_bins: int,
    artifacts: Mapping[str, Optional[str]],
    compatibility: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> PackagedArtifactManifest:
    """Build a validated packaged model/profile manifest."""
    pipeline = get_pipeline_definition(canonical_pipeline)
    if pipeline.stability != "canonical":
        raise ValueError(
            f"Canonical packaged artifacts must use a canonical pipeline, got {canonical_pipeline}"
        )
    if model_family != pipeline.model_family:
        raise ValueError(
            f"Packaged artifact model_family {model_family!r} does not match "
            f"pipeline {canonical_pipeline!r} model_family {pipeline.model_family!r}"
        )
    if canonical_pipeline == REFERENCE_AUDIO_OFFLINE_PIPELINE and (
        artifacts.get("adapter") or artifacts.get("full_model")
    ):
        raise ValueError(
            "quality_seedvc is reference-audio driven and does not consume trained "
            "LoRA adapter or full-model artifacts"
        )
    if speaker_embedding_dim <= 0:
        raise ValueError("speaker_embedding_dim must be positive")
    if mel_bins <= 0:
        raise ValueError("mel_bins must be positive")

    compatibility_payload = dict(compatibility or {})
    compatibility_payload.setdefault("supported_pipelines", [canonical_pipeline])
    compatibility_payload.setdefault("supported_runtime_backends", [pipeline.runtime_backend])
    compatibility_payload.setdefault("supports_tensorrt", bool(artifacts.get("tensorrt_engine")))
    if canonical_pipeline == TRAINED_PROFILE_SERVING_PIPELINE:
        compatibility_payload.setdefault("serving_contract", "trained_profile_realtime_only")
        compatibility_payload.setdefault(
            "unsupported_pipelines",
            {
                REFERENCE_AUDIO_OFFLINE_PIPELINE: (
                    "Seed-VC offline conversion uses target reference audio and does not "
                    "consume trained LoRA/full-model artifacts."
                )
            },
        )

    return PackagedArtifactManifest(
        schema_version=1,
        compatibility_version="1.0",
        profile_id=profile_id,
        display_name=display_name,
        model_family=model_family,
        canonical_pipeline=canonical_pipeline,
        sample_rate=sample_rate,
        speaker_embedding_dim=speaker_embedding_dim,
        mel_bins=mel_bins,
        artifacts=dict(artifacts),
        compatibility=compatibility_payload,
        metadata=dict(metadata or {}),
    )


def validate_packaged_artifact_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a packaged artifact manifest and return a normalized copy."""
    required = (
        "schema_version",
        "compatibility_version",
        "profile_id",
        "display_name",
        "model_family",
        "canonical_pipeline",
        "sample_rate",
        "speaker_embedding_dim",
        "mel_bins",
        "artifacts",
        "compatibility",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing packaged artifact manifest fields: {', '.join(missing)}")

    canonical_pipeline = str(payload["canonical_pipeline"])
    pipeline = get_pipeline_definition(canonical_pipeline)
    if pipeline.stability != "canonical":
        raise ValueError(
            f"Packaged artifact manifest must target a canonical pipeline, got {canonical_pipeline}"
        )
    if str(payload["model_family"]) != pipeline.model_family:
        raise ValueError(
            f"Packaged artifact model_family {payload['model_family']!r} does not match "
            f"pipeline {canonical_pipeline!r} model_family {pipeline.model_family!r}"
        )

    normalized = dict(payload)
    normalized["sample_rate"] = int(payload["sample_rate"])
    normalized["speaker_embedding_dim"] = int(payload["speaker_embedding_dim"])
    normalized["mel_bins"] = int(payload["mel_bins"])
    normalized["artifacts"] = dict(payload["artifacts"])
    if canonical_pipeline == REFERENCE_AUDIO_OFFLINE_PIPELINE and (
        normalized["artifacts"].get("adapter") or normalized["artifacts"].get("full_model")
    ):
        raise ValueError(
            "quality_seedvc manifests cannot declare trained LoRA adapter or full-model artifacts"
        )
    normalized["compatibility"] = dict(payload["compatibility"])
    normalized["metadata"] = dict(payload.get("metadata", {}))
    if "reference_audio" in normalized["metadata"]:
        normalized["metadata"]["reference_audio"] = normalize_reference_audio_entries(
            normalized["metadata"]["reference_audio"]
        )
    return normalized


def load_packaged_artifact_manifest(path: Path | str) -> Dict[str, Any]:
    """Load and validate a packaged artifact manifest from disk."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_packaged_artifact_manifest(payload)


def write_packaged_artifact_manifest(
    path: Path | str,
    manifest: PackagedArtifactManifest | Mapping[str, Any],
) -> Path:
    """Write one packaged artifact manifest to disk."""
    manifest_path = Path(path)
    payload = manifest.to_dict() if isinstance(manifest, PackagedArtifactManifest) else dict(manifest)
    normalized = validate_packaged_artifact_manifest(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return manifest_path


def choose_artifact_manifest_path(paths: Iterable[Path]) -> Optional[Path]:
    """Return the first existing manifest path from an ordered list."""
    for path in paths:
        if path.exists():
            return path
    return None
