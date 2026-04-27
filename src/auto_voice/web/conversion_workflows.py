"""Durable dual-upload conversion workflow orchestration."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency during some tests
    sf = None

from auto_voice.audio.separation import VocalSeparator
from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerDiarizer, SpeakerSegment
from auto_voice.storage.paths import (
    resolve_profiles_dir,
    resolve_samples_dir,
    resolve_trained_models_dir,
)
from auto_voice.storage.voice_profiles import (
    PROFILE_ROLE_SOURCE_ARTIST,
    PROFILE_ROLE_TARGET_USER,
    VoiceProfileStore,
)

logger = logging.getLogger(__name__)

AUTO_MATCH_THRESHOLD = 0.85
REVIEW_MATCH_THRESHOLD = 0.75
MIN_ARTIST_DURATION_SECONDS = 1.5
TERMINAL_WORKFLOW_STATUSES = {
    "awaiting_review",
    "ready_for_training",
    "ready_for_conversion",
    "training_in_progress",
    "error",
}
RECOVERABLE_WORKFLOW_STATUSES = {"queued", "processing"}
RECOVERABLE_WORKFLOW_STAGES = {
    "uploaded",
    "separating_artist_song",
    "analyzing_user_vocals",
    "diarizing_artist_song",
    "matching_profiles",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_name(raw_name: Optional[str], fallback: str) -> str:
    stem = Path(str(raw_name or fallback)).stem.strip()
    return stem or fallback


def _serialize_match(match: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not match:
        return None
    serialized = dict(match)
    serialized["similarity"] = float(serialized.get("similarity") or 0.0)
    return serialized


def _serialize_embedding(embedding: Optional[np.ndarray | List[float]]) -> Optional[List[float]]:
    if embedding is None:
        return None
    array = np.asarray(embedding, dtype=np.float32)
    if array.size == 0:
        return None
    return array.astype(np.float32).tolist()


def _serialize_speaker_segment(segment: SpeakerSegment) -> Dict[str, Any]:
    return {
        "speaker_id": segment.speaker_id,
        "start": float(segment.start),
        "end": float(segment.end),
        "duration": float(segment.duration),
        "confidence": float(segment.confidence),
    }


def _serialize_diarization_result(diarization: DiarizationResult) -> Dict[str, Any]:
    return {
        "num_speakers": int(diarization.num_speakers),
        "audio_duration": float(diarization.audio_duration),
        "segments": [_serialize_speaker_segment(segment) for segment in diarization.segments],
        "speaker_embeddings": {
            speaker_id: embedding.astype(np.float32).tolist()
            for speaker_id, embedding in (diarization.speaker_embeddings or {}).items()
        },
    }


class ConversionWorkflowManager:
    """Runs multi-step conversion intake workflows in the background."""

    def __init__(self, app):
        self.app = app
        self.state_store = app.state_store
        self.data_dir = Path(app.config["DATA_DIR"])
        self.base_dir = self.data_dir / "conversion_workflows"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="conversion-workflow")
        self._lock = threading.RLock()
        self._scheduled_workflows: set[str] = set()
        self._recover_incomplete_workflows()

    def _profile_store(self) -> VoiceProfileStore:
        return VoiceProfileStore(
            profiles_dir=str(resolve_profiles_dir(data_dir=str(self.data_dir))),
            samples_dir=str(resolve_samples_dir(data_dir=str(self.data_dir))),
            trained_models_dir=str(resolve_trained_models_dir(data_dir=str(self.data_dir))),
        )

    def _emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        socketio = getattr(self.app, "socketio", None)
        if socketio is None:
            return
        try:
            socketio.emit(event_name, payload)
        except Exception as exc:  # pragma: no cover - websocket failures shouldn't break workflow
            logger.debug("Failed to emit %s: %s", event_name, exc)

    def _workflow_dir(self, workflow_id: str) -> Path:
        path = self.base_dir / workflow_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_upload(self, file_storage, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_storage.save(str(destination))

    def list_workflows(self) -> List[Dict[str, Any]]:
        return [self._hydrate_runtime_state(item) for item in self.state_store.list_conversion_workflows()]

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        workflow = self.state_store.get_conversion_workflow(workflow_id)
        if not workflow:
            return None
        return self._hydrate_runtime_state(dict(workflow))

    def create_workflow(
        self,
        *,
        artist_song,
        user_vocals: Iterable[Any],
        target_profile_override: Optional[str] = None,
        dominant_source_profile_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        workflow_id = str(uuid.uuid4())
        workflow_dir = self._workflow_dir(workflow_id)
        artist_name = _safe_name(getattr(artist_song, "filename", None), "artist_song")
        artist_song_path = workflow_dir / f"artist_song{Path(getattr(artist_song, 'filename', '') or '.wav').suffix or '.wav'}"
        self._save_upload(artist_song, artist_song_path)

        saved_user_vocals: List[Dict[str, Any]] = []
        user_vocals_dir = workflow_dir / "user_vocals"
        for index, upload in enumerate(user_vocals, start=1):
            suffix = Path(getattr(upload, "filename", "") or ".wav").suffix or ".wav"
            dest_path = user_vocals_dir / f"{index:02d}_{_safe_name(getattr(upload, 'filename', None), f'user_vocal_{index}')}{suffix}"
            self._save_upload(upload, dest_path)
            saved_user_vocals.append({
                "filename": getattr(upload, "filename", dest_path.name) or dest_path.name,
                "path": str(dest_path),
            })

        workflow = {
            "workflow_id": workflow_id,
            "status": "queued",
            "stage": "uploaded",
            "progress": 0,
            "artist_song": {
                "filename": getattr(artist_song, "filename", artist_song_path.name) or artist_song_path.name,
                "path": str(artist_song_path),
            },
            "user_vocals": saved_user_vocals,
            "artist_vocals_path": None,
            "instrumental_path": None,
            "diarization_id": None,
            "resolved_source_profiles": [],
            "resolved_target_profile_id": target_profile_override,
            "review_items": [],
            "target_profile_override": target_profile_override,
            "dominant_source_profile_override": dominant_source_profile_override,
            "training_readiness": {"ready": False, "reason": "workflow_incomplete"},
            "conversion_readiness": {"ready": False, "reason": "workflow_incomplete"},
            "user_analysis": {"status": "pending"},
            "artist_analysis": {"status": "pending"},
            "recovery": {"resume_count": 0, "last_resume_at": None, "last_resume_reason": None},
            "current_training_job_id": None,
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
            "error": None,
        }
        self.state_store.save_conversion_workflow(workflow)
        self._schedule_workflow_run(workflow_id, reason="created")
        return self.get_workflow(workflow_id) or workflow

    def resolve_review_item(
        self,
        workflow_id: str,
        review_id: str,
        *,
        resolution: str,
        profile_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        workflow = self.state_store.get_conversion_workflow(workflow_id)
        if not workflow:
            raise KeyError(f"Workflow {workflow_id} not found")

        review_items = list(workflow.get("review_items") or [])
        matching_item = next((item for item in review_items if item.get("review_id") == review_id), None)
        if not matching_item:
            raise KeyError(f"Review item {review_id} not found")

        store = self._profile_store()
        resolved_profile_id: Optional[str] = None
        resolution = str(resolution or "").strip().lower()

        if resolution == "use_suggested":
            suggested = matching_item.get("suggested_match") or {}
            resolved_profile_id = suggested.get("profile_id")
            if not resolved_profile_id:
                raise ValueError("No suggested profile available for this review item")
        elif resolution == "use_existing":
            if not profile_id:
                raise ValueError("profile_id is required when using an existing profile")
            resolved_profile_id = profile_id
        elif resolution == "create_new":
            candidate = matching_item.get("candidate") or {}
            role = candidate.get("role") or PROFILE_ROLE_TARGET_USER
            resolved_profile_id = self._create_profile_from_candidate(
                store,
                candidate,
                forced_name=name,
                role=role,
                workflow_id=workflow_id,
            )
        else:
            raise ValueError("resolution must be one of: use_suggested, use_existing, create_new")

        candidate = matching_item.get("candidate") or {}
        if resolution in {"use_suggested", "use_existing"}:
            self._attach_candidate_to_profile(store, candidate, resolved_profile_id, workflow_id=workflow_id)

        if candidate.get("role") == PROFILE_ROLE_TARGET_USER:
            workflow["resolved_target_profile_id"] = resolved_profile_id
            user_analysis = dict(workflow.get("user_analysis") or {})
            user_analysis["status"] = "resolved"
            user_analysis["resolved_target_profile_id"] = resolved_profile_id
            user_analysis["resolution"] = resolution
            workflow["user_analysis"] = user_analysis
        else:
            resolved_sources = list(workflow.get("resolved_source_profiles") or [])
            resolved_sources.append(self._resolved_profile_entry(candidate, resolved_profile_id, store))
            workflow["resolved_source_profiles"] = resolved_sources
            artist_analysis = dict(workflow.get("artist_analysis") or {})
            speaker_assignments = list(artist_analysis.get("speaker_assignments") or [])
            updated_assignment = False
            for assignment in speaker_assignments:
                if assignment.get("speaker_id") == candidate.get("speaker_id"):
                    assignment["resolved_profile_id"] = resolved_profile_id
                    assignment["resolution"] = resolution
                    updated_assignment = True
                    break
            if not updated_assignment:
                speaker_assignments.append({
                    "speaker_id": candidate.get("speaker_id"),
                    "resolved_profile_id": resolved_profile_id,
                    "resolution": resolution,
                })
            artist_analysis["speaker_assignments"] = speaker_assignments
            workflow["artist_analysis"] = artist_analysis

        workflow["review_items"] = [item for item in review_items if item.get("review_id") != review_id]
        workflow["updated_at"] = _utcnow_iso()
        workflow["status"] = "processing"
        workflow["stage"] = "matching_profiles"
        self.state_store.save_conversion_workflow(workflow)
        self._finalize_workflow(workflow_id)
        return self.get_workflow(workflow_id) or workflow

    def attach_training_job(self, workflow_id: str, job_id: str) -> Dict[str, Any]:
        workflow = self.state_store.get_conversion_workflow(workflow_id)
        if not workflow:
            raise KeyError(f"Workflow {workflow_id} not found")
        workflow["current_training_job_id"] = job_id
        workflow["status"] = "training_in_progress"
        workflow["stage"] = "training_in_progress"
        workflow["updated_at"] = _utcnow_iso()
        self.state_store.save_conversion_workflow(workflow)
        return self.get_workflow(workflow_id) or workflow

    def create_conversion_job(self, workflow_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise KeyError(f"Workflow {workflow_id} not found")
        profile_id = workflow.get("resolved_target_profile_id")
        if not profile_id:
            raise ValueError("Workflow does not have a resolved target profile")

        store = self._profile_store()
        profile = store.load(profile_id)
        if not profile.get("has_trained_model"):
            raise ValueError("Target profile is not trained yet")

        artist_song_path = workflow.get("artist_song", {}).get("path")
        if not artist_song_path or not os.path.exists(artist_song_path):
            raise FileNotFoundError("Artist song for workflow is not available")

        job_manager = getattr(self.app, "job_manager", None)
        if job_manager is None:
            raise RuntimeError("Job manager is unavailable")

        settings_dict = dict(settings or {})
        settings_dict.setdefault("requested_pipeline", settings_dict.get("pipeline_type"))
        settings_dict.setdefault("pipeline_type", settings_dict.get("requested_pipeline"))
        settings_dict.setdefault("return_stems", True)
        settings_dict.setdefault("preset", settings_dict.get("preset", "balanced"))
        settings_dict["workflow_id"] = workflow_id
        settings_dict["auto_resolved_source_profiles"] = workflow.get("resolved_source_profiles", [])
        settings_dict["active_model_type"] = profile.get("active_model_type")

        temp_copy = tempfile.NamedTemporaryFile(
            suffix=Path(artist_song_path).suffix or ".wav",
            delete=False,
        )
        temp_copy.close()
        shutil.copy2(artist_song_path, temp_copy.name)

        job_id = job_manager.create_job(temp_copy.name, profile_id, settings_dict)
        workflow["updated_at"] = _utcnow_iso()
        self.state_store.save_conversion_workflow(workflow)
        return {
            "status": "queued",
            "job_id": job_id,
            "websocket_room": job_id,
            "message": "Conversion job created from workflow",
            "active_model_type": profile.get("active_model_type"),
            "requested_pipeline": settings_dict.get("requested_pipeline"),
            "resolved_pipeline": settings_dict.get("requested_pipeline"),
            "runtime_backend": "pytorch_full_model" if profile.get("active_model_type") == "full_model" else "pytorch",
        }

    def _schedule_workflow_run(self, workflow_id: str, *, reason: str) -> None:
        with self._lock:
            if workflow_id in self._scheduled_workflows:
                return
            self._scheduled_workflows.add(workflow_id)
        future = self._executor.submit(self._run_workflow, workflow_id)

        def _clear(_future) -> None:
            with self._lock:
                self._scheduled_workflows.discard(workflow_id)

        future.add_done_callback(_clear)

    def _workflow_needs_recovery(self, workflow: Dict[str, Any]) -> bool:
        status = str(workflow.get("status") or "")
        stage = str(workflow.get("stage") or "")
        if status in TERMINAL_WORKFLOW_STATUSES:
            return False
        if workflow.get("review_items"):
            return False
        if workflow.get("current_training_job_id"):
            return False
        return status in RECOVERABLE_WORKFLOW_STATUSES or stage in RECOVERABLE_WORKFLOW_STAGES

    def _recover_incomplete_workflows(self) -> None:
        for workflow in self.state_store.list_conversion_workflows():
            if not self._workflow_needs_recovery(workflow):
                continue
            recovery = dict(workflow.get("recovery") or {})
            recovery["resume_count"] = int(recovery.get("resume_count") or 0) + 1
            recovery["last_resume_at"] = _utcnow_iso()
            recovery["last_resume_reason"] = "startup_recovery"
            workflow["recovery"] = recovery
            workflow["status"] = "processing"
            workflow["updated_at"] = _utcnow_iso()
            self.state_store.save_conversion_workflow(workflow)
            self._schedule_workflow_run(workflow["workflow_id"], reason="startup_recovery")

    def _user_analysis_complete(self, workflow: Dict[str, Any]) -> bool:
        status = str(((workflow.get("user_analysis") or {}).get("status")) or "")
        return status in {"resolved", "awaiting_review"} or bool(workflow.get("resolved_target_profile_id"))

    def _artist_analysis_complete(self, workflow: Dict[str, Any]) -> bool:
        status = str(((workflow.get("artist_analysis") or {}).get("status")) or "")
        return status in {"resolved", "awaiting_review"} and bool(workflow.get("diarization_id"))

    def _run_workflow(self, workflow_id: str) -> None:
        with self.app.app_context():
            try:
                workflow = self.state_store.get_conversion_workflow(workflow_id) or {}
                artist_vocals_path = workflow.get("artist_vocals_path")
                instrumental_path = workflow.get("instrumental_path")

                if not artist_vocals_path or not os.path.exists(str(artist_vocals_path)):
                    self._update_workflow(
                        workflow_id,
                        status="processing",
                        stage="separating_artist_song",
                        progress=10,
                    )
                    separation_result = self._separate_artist_song(workflow_id)
                    self._update_workflow(
                        workflow_id,
                        artist_vocals_path=separation_result["artist_vocals_path"],
                        instrumental_path=separation_result["instrumental_path"],
                        stage="analyzing_user_vocals",
                        progress=30,
                    )

                workflow = self.state_store.get_conversion_workflow(workflow_id) or {}
                if not self._user_analysis_complete(workflow):
                    self._update_workflow(workflow_id, status="processing", stage="analyzing_user_vocals", progress=30)
                    self._analyze_user_vocals(workflow_id)

                workflow = self.state_store.get_conversion_workflow(workflow_id) or {}
                if workflow.get("status") == "awaiting_review":
                    return

                if not self._artist_analysis_complete(workflow):
                    self._update_workflow(workflow_id, status="processing", stage="diarizing_artist_song", progress=55)
                    self._analyze_artist_song(workflow_id)

                workflow = self.state_store.get_conversion_workflow(workflow_id) or {}
                if workflow.get("status") == "awaiting_review":
                    return

                self._finalize_workflow(workflow_id)
            except Exception as exc:
                logger.error("Conversion workflow %s failed: %s", workflow_id, exc, exc_info=True)
                workflow = self._update_workflow(
                    workflow_id,
                    status="error",
                    stage="error",
                    error=str(exc),
                )
                self._emit("conversion_workflow_error", workflow)

    def _update_workflow(self, workflow_id: str, **updates: Any) -> Dict[str, Any]:
        workflow = self.state_store.get_conversion_workflow(workflow_id)
        if not workflow:
            raise KeyError(f"Workflow {workflow_id} not found")
        workflow.update(updates)
        workflow["updated_at"] = _utcnow_iso()
        self.state_store.save_conversion_workflow(workflow)
        event_name = "conversion_workflow_progress"
        if workflow.get("status") == "awaiting_review":
            event_name = "conversion_workflow_review_required"
        elif workflow.get("status") in {"ready_for_training", "ready_for_conversion"}:
            event_name = "conversion_workflow_ready"
        self._emit(event_name, self._hydrate_runtime_state(dict(workflow)))
        return workflow

    def _hydrate_runtime_state(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        hydrated = dict(workflow)
        target_profile_id = hydrated.get("resolved_target_profile_id")
        current_training_job_id = hydrated.get("current_training_job_id")
        if current_training_job_id:
            training_job = self.state_store.get_training_job(current_training_job_id)
            if training_job and training_job.get("status") in {"pending", "running"}:
                hydrated["current_training_job_id"] = current_training_job_id
                hydrated["status"] = "training_in_progress"
                hydrated["stage"] = "training_in_progress"
            else:
                hydrated["current_training_job_id"] = None
        if target_profile_id:
            try:
                profile = self._profile_store().load(target_profile_id)
                hydrated["resolved_target_profile"] = {
                    "profile_id": profile["profile_id"],
                    "name": profile.get("name") or profile["profile_id"],
                    "profile_role": profile.get("profile_role"),
                    "has_trained_model": profile.get("has_trained_model"),
                    "active_model_type": profile.get("active_model_type"),
                    "sample_count": profile.get("sample_count"),
                    "clean_vocal_minutes": profile.get("clean_vocal_seconds", 0.0) / 60.0,
                }
                trainable_samples = self._profile_store().list_trainable_samples(target_profile_id)
                hydrated["training_readiness"] = {
                    "ready": bool(trainable_samples),
                    "reason": "ready" if trainable_samples else "no_trainable_samples",
                    "sample_count": len(trainable_samples),
                    "clean_vocal_minutes": profile.get("clean_vocal_seconds", 0.0) / 60.0,
                }
                conversion_ready = bool(profile.get("has_trained_model"))
                hydrated["conversion_readiness"] = {
                    "ready": conversion_ready,
                    "reason": "ready" if conversion_ready else "target_profile_not_trained",
                }
                hydrated["readiness"] = {
                    "training": dict(hydrated["training_readiness"]),
                    "conversion": dict(hydrated["conversion_readiness"]),
                    "live_conversion": {
                        "ready": conversion_ready,
                        "reason": "ready" if conversion_ready else "target_profile_not_trained",
                    },
                }
                if not hydrated.get("current_training_job_id"):
                    training_jobs = self.state_store.list_training_jobs(profile_id=target_profile_id)
                    active_job = next(
                        (job for job in training_jobs if job.get("status") in {"pending", "running"}),
                        None,
                    )
                    if active_job:
                        hydrated["current_training_job_id"] = active_job.get("job_id")
                        hydrated["status"] = "training_in_progress"
                        hydrated["stage"] = "training_in_progress"
                    else:
                        hydrated["status"] = "ready_for_conversion" if conversion_ready else "ready_for_training"
                        hydrated["stage"] = hydrated["status"]
            except Exception:
                hydrated["resolved_target_profile"] = None
        else:
            hydrated["resolved_target_profile"] = None
        return hydrated

    def _preferred_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch import should exist in runtime
            return "cpu"

    def _load_audio_file(self, audio_path: str) -> tuple[np.ndarray, int]:
        if sf is None:
            raise RuntimeError("soundfile is required for workflow audio processing")
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        return audio_np, int(sample_rate)

    def _write_audio_file(self, path: Path, audio: np.ndarray, sample_rate: int) -> None:
        if sf is None:
            raise RuntimeError("soundfile is required for workflow audio processing")
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        sf.write(str(path), audio_np, sample_rate)

    def _separate_artist_song(self, workflow_id: str) -> Dict[str, str]:
        workflow = self.get_workflow(workflow_id) or {}
        artist_song_path = workflow.get("artist_song", {}).get("path")
        if not artist_song_path:
            raise FileNotFoundError("Workflow artist song is missing")
        audio, sample_rate = self._load_audio_file(artist_song_path)
        separator = VocalSeparator(device=self._preferred_device())
        separated = separator.separate(audio, sample_rate)
        workflow_dir = self._workflow_dir(workflow_id)
        artist_vocals_path = workflow_dir / "artist_vocals.wav"
        instrumental_path = workflow_dir / "instrumental.wav"
        self._write_audio_file(artist_vocals_path, separated["vocals"], sample_rate)
        self._write_audio_file(instrumental_path, separated["instrumental"], sample_rate)
        return {
            "artist_vocals_path": str(artist_vocals_path),
            "instrumental_path": str(instrumental_path),
        }

    def _analyze_user_vocals(self, workflow_id: str) -> None:
        workflow = self.get_workflow(workflow_id) or {}
        store = self._profile_store()
        target_override = workflow.get("target_profile_override")
        user_vocals = list(workflow.get("user_vocals") or [])
        diarizer = SpeakerDiarizer(device=self._preferred_device())

        user_embeddings: List[np.ndarray] = []
        multi_speaker_detected = False
        for item in user_vocals:
            audio_path = item.get("path")
            if not audio_path or not os.path.exists(audio_path):
                continue
            try:
                diarization = diarizer.diarize(audio_path)
                if diarization.num_speakers > 1:
                    multi_speaker_detected = True
            except Exception:
                diarization = None
            embedding = diarizer.extract_speaker_embedding(audio_path)
            item["embedding"] = embedding.tolist()
            user_embeddings.append(embedding)

        if not user_embeddings:
            raise RuntimeError("No valid user vocal embeddings could be extracted")

        aggregate_embedding = np.mean(np.stack(user_embeddings), axis=0)
        aggregate_embedding = aggregate_embedding / (np.linalg.norm(aggregate_embedding) + 1e-8)
        suggested_matches = store.rank_speaker_embedding_matches(
            aggregate_embedding,
            profile_role=PROFILE_ROLE_TARGET_USER,
            limit=5,
        )
        user_analysis = {
            "status": "pending",
            "aggregate_embedding": aggregate_embedding.astype(np.float32).tolist(),
            "multi_speaker_detected": multi_speaker_detected,
            "suggested_matches": [_serialize_match(match) for match in suggested_matches],
        }

        if target_override:
            attachment_summary = self._attach_user_samples(
                store,
                target_override,
                user_vocals,
                aggregate_embedding,
                workflow_id=workflow_id,
            )
            user_analysis.update(
                {
                    "status": "resolved",
                    "resolution": "target_override",
                    "resolved_target_profile_id": target_override,
                    "attachment_summary": attachment_summary,
                }
            )
            self._update_workflow(
                workflow_id,
                resolved_target_profile_id=target_override,
                user_vocals=user_vocals,
                user_analysis=user_analysis,
                progress=45,
            )
            return

        best_match = suggested_matches[0] if suggested_matches else None
        if multi_speaker_detected:
            review_item = self._build_review_item(
                role=PROFILE_ROLE_TARGET_USER,
                candidate={
                    "role": PROFILE_ROLE_TARGET_USER,
                    "name": "Target User",
                    "embedding": aggregate_embedding.tolist(),
                    "sample_paths": [item.get("path") for item in user_vocals if item.get("path")],
                    "source_files": [item.get("filename") for item in user_vocals if item.get("filename")],
                },
                reason="multiple_speakers_detected",
                suggested_match=best_match,
            )
            workflow = self.get_workflow(workflow_id) or {}
            review_items = list(workflow.get("review_items") or [])
            review_items.append(review_item)
            self._update_workflow(
                workflow_id,
                status="awaiting_review",
                stage="awaiting_review",
                progress=45,
                review_items=review_items,
                user_vocals=user_vocals,
                user_analysis={**user_analysis, "status": "awaiting_review", "reason": "multiple_speakers_detected"},
            )
            return

        similarity = float(best_match.get("similarity") or 0.0) if best_match else 0.0
        if best_match and similarity >= AUTO_MATCH_THRESHOLD:
            resolved_profile_id = str(best_match["profile_id"])
            attachment_summary = self._attach_user_samples(
                store,
                resolved_profile_id,
                user_vocals,
                aggregate_embedding,
                workflow_id=workflow_id,
            )
            self._update_workflow(
                workflow_id,
                resolved_target_profile_id=resolved_profile_id,
                user_vocals=user_vocals,
                user_analysis={
                    **user_analysis,
                    "status": "resolved",
                    "resolution": "auto_match",
                    "resolved_target_profile_id": resolved_profile_id,
                    "attachment_summary": attachment_summary,
                },
                progress=45,
            )
            return

        if best_match and similarity >= REVIEW_MATCH_THRESHOLD:
            review_item = self._build_review_item(
                role=PROFILE_ROLE_TARGET_USER,
                candidate={
                    "role": PROFILE_ROLE_TARGET_USER,
                    "name": "Target User",
                    "embedding": aggregate_embedding.tolist(),
                    "sample_paths": [item.get("path") for item in user_vocals if item.get("path")],
                    "source_files": [item.get("filename") for item in user_vocals if item.get("filename")],
                },
                reason="ambiguous_match",
                suggested_match=best_match,
            )
            workflow = self.get_workflow(workflow_id) or {}
            review_items = list(workflow.get("review_items") or [])
            review_items.append(review_item)
            self._update_workflow(
                workflow_id,
                status="awaiting_review",
                stage="awaiting_review",
                progress=45,
                review_items=review_items,
                user_vocals=user_vocals,
                user_analysis={**user_analysis, "status": "awaiting_review", "reason": "ambiguous_match"},
            )
            return

        profile_id = self._create_target_profile_from_samples(
            store,
            user_vocals=user_vocals,
            aggregate_embedding=aggregate_embedding,
            workflow_id=workflow_id,
        )
        self._update_workflow(
            workflow_id,
            resolved_target_profile_id=profile_id,
            user_vocals=user_vocals,
            user_analysis={
                **user_analysis,
                "status": "resolved",
                "resolution": "created_new",
                "resolved_target_profile_id": profile_id,
            },
            progress=45,
        )

    def _analyze_artist_song(self, workflow_id: str) -> None:
        workflow = self.get_workflow(workflow_id) or {}
        artist_vocals_path = workflow.get("artist_vocals_path")
        if not artist_vocals_path:
            raise FileNotFoundError("Separated artist vocals not available")

        store = self._profile_store()
        diarizer = SpeakerDiarizer(device=self._preferred_device())
        diarization = diarizer.diarize(artist_vocals_path)
        diarization_id = str(uuid.uuid4())
        resolved_profiles: List[Dict[str, Any]] = list(workflow.get("resolved_source_profiles") or [])
        review_items: List[Dict[str, Any]] = list(workflow.get("review_items") or [])
        speaker_assignments: List[Dict[str, Any]] = []
        dominant_speaker_id = None
        dominant_duration = -1.0

        for speaker_id in diarization.get_all_speaker_ids():
            duration = diarization.get_speaker_total_duration(speaker_id)
            if duration < MIN_ARTIST_DURATION_SECONDS:
                continue
            if duration > dominant_duration:
                dominant_duration = duration
                dominant_speaker_id = speaker_id

        for speaker_id in diarization.get_all_speaker_ids():
            duration = diarization.get_speaker_total_duration(speaker_id)
            if duration < MIN_ARTIST_DURATION_SECONDS:
                continue

            speaker_segments = diarization.get_speaker_segments(speaker_id)
            candidate_embedding = diarization.speaker_embeddings.get(speaker_id)
            if candidate_embedding is None and speaker_segments:
                longest_segment = max(speaker_segments, key=lambda seg: seg.duration)
                candidate_embedding = diarizer.extract_speaker_embedding(
                    artist_vocals_path,
                    start=longest_segment.start,
                    end=longest_segment.end,
                )
            if candidate_embedding is None:
                continue

            extracted_path = diarizer.extract_speaker_audio(
                artist_vocals_path,
                diarization,
                speaker_id,
                output_path=self._workflow_dir(workflow_id) / f"{speaker_id.lower()}_artist.wav",
            )
            candidate = {
                "role": PROFILE_ROLE_SOURCE_ARTIST,
                "speaker_id": speaker_id,
                "duration_seconds": float(duration),
                "name": f"Artist {speaker_id.replace('_', ' ').title()}",
                "embedding": candidate_embedding.tolist(),
                "sample_paths": [str(extracted_path)],
                "source_files": [Path(extracted_path).name],
            }

            override_profile_id = workflow.get("dominant_source_profile_override")
            if override_profile_id and speaker_id == dominant_speaker_id:
                self._attach_candidate_to_profile(
                    store,
                    candidate,
                    override_profile_id,
                    workflow_id=workflow_id,
                )
                resolved_profiles.append(self._resolved_profile_entry(candidate, override_profile_id, store))
                speaker_assignments.append(
                    {
                        "speaker_id": speaker_id,
                        "duration_seconds": float(duration),
                        "segment_count": len(speaker_segments),
                        "resolved_profile_id": override_profile_id,
                        "resolution": "dominant_source_profile_override",
                        "sample_paths": [str(extracted_path)],
                    }
                )
                continue

            ranked_matches = store.rank_speaker_embedding_matches(
                candidate_embedding,
                profile_role=PROFILE_ROLE_SOURCE_ARTIST,
                limit=5,
            )
            best_match = ranked_matches[0] if ranked_matches else None
            similarity = float(best_match.get("similarity") or 0.0) if best_match else 0.0

            if best_match and similarity >= AUTO_MATCH_THRESHOLD:
                resolved_profile_id = str(best_match["profile_id"])
                self._attach_candidate_to_profile(
                    store,
                    candidate,
                    resolved_profile_id,
                    workflow_id=workflow_id,
                )
                resolved_profiles.append(self._resolved_profile_entry(candidate, resolved_profile_id, store, suggested_match=best_match))
                speaker_assignments.append(
                    {
                        "speaker_id": speaker_id,
                        "duration_seconds": float(duration),
                        "segment_count": len(speaker_segments),
                        "resolved_profile_id": resolved_profile_id,
                        "resolution": "auto_match",
                        "suggested_match": _serialize_match(best_match),
                        "sample_paths": [str(extracted_path)],
                    }
                )
                continue

            if best_match and similarity >= REVIEW_MATCH_THRESHOLD:
                speaker_assignments.append(
                    {
                        "speaker_id": speaker_id,
                        "duration_seconds": float(duration),
                        "segment_count": len(speaker_segments),
                        "resolution": "awaiting_review",
                        "suggested_match": _serialize_match(best_match),
                        "sample_paths": [str(extracted_path)],
                    }
                )
                review_items.append(
                    self._build_review_item(
                        role=PROFILE_ROLE_SOURCE_ARTIST,
                        candidate=candidate,
                        reason="ambiguous_match",
                        suggested_match=best_match,
                    )
                )
                continue

            new_profile_id = self._create_profile_from_candidate(
                store,
                candidate,
                role=PROFILE_ROLE_SOURCE_ARTIST,
                workflow_id=workflow_id,
            )
            resolved_profiles.append(self._resolved_profile_entry(candidate, new_profile_id, store))
            speaker_assignments.append(
                {
                    "speaker_id": speaker_id,
                    "duration_seconds": float(duration),
                    "segment_count": len(speaker_segments),
                    "resolved_profile_id": new_profile_id,
                    "resolution": "created_new",
                    "sample_paths": [str(extracted_path)],
                }
            )

        next_status = "awaiting_review" if review_items else "processing"
        next_stage = "awaiting_review" if review_items else "matching_profiles"
        self._update_workflow(
            workflow_id,
            diarization_id=diarization_id,
            resolved_source_profiles=resolved_profiles,
            review_items=review_items,
            artist_analysis={
                "status": "awaiting_review" if review_items else "resolved",
                "diarization": _serialize_diarization_result(diarization),
                "diarization_id": diarization_id,
                "dominant_speaker_id": dominant_speaker_id,
                "speaker_assignments": speaker_assignments,
            },
            status=next_status,
            stage=next_stage,
            progress=75 if review_items else 80,
        )

    def _attach_user_samples(
        self,
        store: VoiceProfileStore,
        profile_id: str,
        user_vocals: List[Dict[str, Any]],
        aggregate_embedding: np.ndarray,
        *,
        workflow_id: str,
    ) -> Dict[str, int]:
        store.save_speaker_embedding(profile_id, aggregate_embedding)
        attached = 0
        skipped_existing = 0
        for item in user_vocals:
            audio_path = item.get("path")
            if not audio_path or not os.path.exists(audio_path):
                continue
            if self._workflow_sample_exists(
                store,
                profile_id=profile_id,
                workflow_id=workflow_id,
                sample_path=audio_path,
                provenance="conversion_workflow_user_vocals",
            ):
                skipped_existing += 1
                continue
            duration_seconds = self._duration_seconds(audio_path)
            store.add_training_sample(
                profile_id=profile_id,
                vocals_path=audio_path,
                duration=duration_seconds,
                source_file=item.get("filename") or os.path.basename(audio_path),
                extra_metadata={
                    "provenance": "conversion_workflow_user_vocals",
                    "workflow_id": workflow_id,
                    "workflow_source_path": str(Path(audio_path).resolve()),
                    "workflow_auto_attached": True,
                },
            )
            attached += 1
        return {"attached": attached, "skipped_existing": skipped_existing}

    def _create_target_profile_from_samples(
        self,
        store: VoiceProfileStore,
        *,
        user_vocals: List[Dict[str, Any]],
        aggregate_embedding: np.ndarray,
        workflow_id: str,
    ) -> str:
        profile_name = _safe_name(user_vocals[0].get("filename") if user_vocals else None, "Target User")
        profile_id = store.save({
            "name": f"{profile_name} Target",
            "created_from": "conversion_workflow",
            "profile_role": PROFILE_ROLE_TARGET_USER,
            "user_id": "system",
        })
        self._attach_user_samples(
            store,
            profile_id,
            user_vocals,
            aggregate_embedding,
            workflow_id=workflow_id,
        )
        return profile_id

    def _resolved_profile_entry(
        self,
        candidate: Dict[str, Any],
        profile_id: str,
        store: VoiceProfileStore,
        *,
        suggested_match: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = store.load(profile_id)
        return {
            "profile_id": profile_id,
            "name": profile.get("name") or profile_id,
            "profile_role": profile.get("profile_role"),
            "speaker_id": candidate.get("speaker_id"),
            "duration_seconds": float(candidate.get("duration_seconds") or 0.0),
            "status": "matched" if suggested_match else "created" if candidate.get("role") == PROFILE_ROLE_SOURCE_ARTIST and not profile.get("created_from") == "manual" else "matched",
            "suggested_match": _serialize_match(suggested_match),
            "sample_paths": list(candidate.get("sample_paths") or []),
        }

    def _build_review_item(
        self,
        *,
        role: str,
        candidate: Dict[str, Any],
        reason: str,
        suggested_match: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "review_id": str(uuid.uuid4()),
            "role": role,
            "reason": reason,
            "suggested_match": _serialize_match(suggested_match),
            "candidate": candidate,
        }

    def _create_profile_from_candidate(
        self,
        store: VoiceProfileStore,
        candidate: Dict[str, Any],
        *,
        forced_name: Optional[str] = None,
        role: str,
        workflow_id: str,
    ) -> str:
        profile_id = store.save({
            "name": forced_name or candidate.get("name") or "Auto-created profile",
            "created_from": "conversion_workflow",
            "profile_role": role,
            "user_id": "system",
        })
        self._attach_candidate_to_profile(store, candidate, profile_id, workflow_id=workflow_id)
        return profile_id

    def _attach_candidate_to_profile(
        self,
        store: VoiceProfileStore,
        candidate: Dict[str, Any],
        profile_id: str,
        *,
        workflow_id: Optional[str] = None,
    ) -> None:
        embedding = np.asarray(candidate.get("embedding") or [], dtype=np.float32)
        if embedding.size:
            store.save_speaker_embedding(profile_id, embedding)
        for sample_path, source_file in zip(
            list(candidate.get("sample_paths") or []),
            list(candidate.get("source_files") or []) or [None] * len(list(candidate.get("sample_paths") or [])),
        ):
            if not sample_path or not os.path.exists(sample_path):
                continue
            if workflow_id and self._workflow_sample_exists(
                store,
                profile_id=profile_id,
                workflow_id=workflow_id,
                sample_path=sample_path,
                provenance="conversion_workflow_candidate",
                speaker_id=str(candidate.get("speaker_id") or ""),
            ):
                continue
            store.add_training_sample(
                profile_id=profile_id,
                vocals_path=sample_path,
                duration=self._duration_seconds(sample_path),
                source_file=source_file or os.path.basename(sample_path),
                extra_metadata={
                    "provenance": "conversion_workflow_candidate",
                    "workflow_id": workflow_id,
                    "workflow_source_path": str(Path(sample_path).resolve()),
                    "speaker_id": candidate.get("speaker_id"),
                    "workflow_auto_attached": True,
                },
            )

    def _workflow_sample_exists(
        self,
        store: VoiceProfileStore,
        *,
        profile_id: str,
        workflow_id: str,
        sample_path: str,
        provenance: str,
        speaker_id: Optional[str] = None,
    ) -> bool:
        resolved_path = str(Path(sample_path).resolve())
        for sample in store.list_training_samples(profile_id):
            metadata = dict(sample.extra_metadata or {})
            if metadata.get("provenance") != provenance:
                continue
            if metadata.get("workflow_id") != workflow_id:
                continue
            if str(metadata.get("workflow_source_path") or "") != resolved_path:
                continue
            if speaker_id is not None and str(metadata.get("speaker_id") or "") != str(speaker_id):
                continue
            return True
        return False

    def _duration_seconds(self, audio_path: str) -> float:
        if sf is None:
            return 0.0
        info = sf.info(audio_path)
        return float(info.frames) / float(info.samplerate or 1)

    def _finalize_workflow(self, workflow_id: str) -> None:
        workflow = self.state_store.get_conversion_workflow(workflow_id)
        if not workflow:
            return

        if workflow.get("review_items"):
            self._update_workflow(
                workflow_id,
                status="awaiting_review",
                stage="awaiting_review",
                training_readiness={"ready": False, "reason": "review_required"},
                conversion_readiness={"ready": False, "reason": "review_required"},
                progress=80,
            )
            return

        target_profile_id = workflow.get("resolved_target_profile_id")
        if not target_profile_id:
            self._update_workflow(
                workflow_id,
                status="error",
                stage="error",
                error="Workflow did not resolve a target user profile",
            )
            return

        store = self._profile_store()
        profile = store.load(target_profile_id)
        trainable_samples = store.list_trainable_samples(target_profile_id)
        training_readiness = {
            "ready": bool(trainable_samples),
            "reason": "ready" if trainable_samples else "no_trainable_samples",
            "sample_count": len(trainable_samples),
            "clean_vocal_minutes": profile.get("clean_vocal_seconds", 0.0) / 60.0,
        }
        conversion_ready = bool(profile.get("has_trained_model"))
        conversion_readiness = {
            "ready": conversion_ready,
            "reason": "ready" if conversion_ready else "target_profile_not_trained",
        }
        readiness = {
            "training": dict(training_readiness),
            "conversion": dict(conversion_readiness),
            "live_conversion": {
                "ready": conversion_ready,
                "reason": "ready" if conversion_ready else "target_profile_not_trained",
            },
        }
        next_status = "ready_for_conversion" if conversion_ready else "ready_for_training"
        self._update_workflow(
            workflow_id,
            status=next_status,
            stage=next_status,
            progress=100,
            training_readiness=training_readiness,
            conversion_readiness=conversion_readiness,
            readiness=readiness,
        )
