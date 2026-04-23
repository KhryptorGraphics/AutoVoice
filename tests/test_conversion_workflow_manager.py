from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pytest

import auto_voice.storage.voice_profiles as voice_profiles_module
import auto_voice.web.conversion_workflows as workflow_module
from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerSegment
from auto_voice.storage.voice_profiles import PROFILE_ROLE_SOURCE_ARTIST, PROFILE_ROLE_TARGET_USER
from auto_voice.web.app import create_app
from auto_voice.web.conversion_workflows import ConversionWorkflowManager


@pytest.fixture
def workflow_app(tmp_path):
    app, _socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
            "singing_conversion_enabled": False,
            "voice_cloning_enabled": False,
        }
    )
    return app


def _touch(path: Path, content: bytes = b"audio") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _write_wav(path: Path, *, sample_rate: int = 22050, duration_seconds: float = 0.5) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * duration_seconds)
    samples = bytearray()
    amplitude = int(0.25 * 32767)
    frequency_hz = 220.0
    for index in range(frames):
        t = index / sample_rate
        value = int(amplitude * np.sin(2.0 * np.pi * frequency_hz * t))
        samples.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(samples))
    return str(path)


def _workflow_payload(
    workflow_id: str,
    *,
    artist_song_path: str,
    user_vocals: list[dict] | None = None,
    artist_vocals_path: str | None = None,
    instrumental_path: str | None = None,
    status: str = "processing",
    stage: str = "uploaded",
    user_analysis: dict | None = None,
    artist_analysis: dict | None = None,
    resolved_target_profile_id: str | None = None,
    resolved_source_profiles: list[dict] | None = None,
    review_items: list[dict] | None = None,
    diarization_id: str | None = None,
    current_training_job_id: str | None = None,
) -> dict:
    return {
        "workflow_id": workflow_id,
        "status": status,
        "stage": stage,
        "progress": 0,
        "artist_song": {
            "filename": Path(artist_song_path).name,
            "path": artist_song_path,
        },
        "user_vocals": list(user_vocals or []),
        "artist_vocals_path": artist_vocals_path,
        "instrumental_path": instrumental_path,
        "diarization_id": diarization_id,
        "resolved_source_profiles": list(resolved_source_profiles or []),
        "resolved_target_profile_id": resolved_target_profile_id,
        "review_items": list(review_items or []),
        "target_profile_override": None,
        "dominant_source_profile_override": None,
        "training_readiness": {"ready": False, "reason": "workflow_incomplete"},
        "conversion_readiness": {"ready": False, "reason": "workflow_incomplete"},
        "user_analysis": dict(user_analysis or {"status": "pending"}),
        "artist_analysis": dict(artist_analysis or {"status": "pending"}),
        "recovery": {"resume_count": 0, "last_resume_at": None, "last_resume_reason": None},
        "current_training_job_id": current_training_job_id,
        "created_at": "2026-04-22T00:00:00Z",
        "updated_at": "2026-04-22T00:00:00Z",
        "error": None,
    }


def test_manager_recovers_incomplete_workflows_on_startup(workflow_app, monkeypatch, tmp_path):
    artist_song_path = _touch(tmp_path / "recover" / "artist.wav")
    recoverable_workflow = _workflow_payload(
        "wf-recover",
        artist_song_path=artist_song_path,
        status="queued",
        stage="analyzing_user_vocals",
    )
    terminal_workflow = _workflow_payload(
        "wf-ready",
        artist_song_path=artist_song_path,
        status="ready_for_training",
        stage="ready_for_training",
        resolved_target_profile_id="target-1",
        user_analysis={"status": "resolved"},
        artist_analysis={"status": "resolved"},
        diarization_id="diarization-ready",
    )
    workflow_app.state_store.save_conversion_workflow(recoverable_workflow)
    workflow_app.state_store.save_conversion_workflow(terminal_workflow)

    scheduled: list[tuple[str, str]] = []

    def fake_schedule(self, workflow_id: str, *, reason: str) -> None:
        scheduled.append((workflow_id, reason))

    monkeypatch.setattr(ConversionWorkflowManager, "_schedule_workflow_run", fake_schedule)

    ConversionWorkflowManager(workflow_app)

    restored = workflow_app.state_store.get_conversion_workflow("wf-recover")
    untouched = workflow_app.state_store.get_conversion_workflow("wf-ready")

    assert restored is not None
    assert restored["status"] == "processing"
    assert restored["recovery"]["resume_count"] == 1
    assert restored["recovery"]["last_resume_reason"] == "startup_recovery"
    assert restored["recovery"]["last_resume_at"]
    assert scheduled == [("wf-recover", "startup_recovery")]

    assert untouched is not None
    assert untouched["status"] == "ready_for_training"
    assert untouched["recovery"]["resume_count"] == 0


def test_run_workflow_resumes_from_persisted_stage_without_repeating_completed_steps(workflow_app, tmp_path):
    manager = ConversionWorkflowManager(workflow_app)
    artist_song_path = _touch(tmp_path / "resume" / "artist.wav")
    artist_vocals_path = _touch(tmp_path / "resume" / "artist_vocals.wav")
    instrumental_path = _touch(tmp_path / "resume" / "instrumental.wav")
    workflow_id = "wf-resume"

    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            workflow_id,
            artist_song_path=artist_song_path,
            artist_vocals_path=artist_vocals_path,
            instrumental_path=instrumental_path,
            status="processing",
            stage="diarizing_artist_song",
            resolved_target_profile_id="target-1",
            user_analysis={"status": "resolved", "resolved_target_profile_id": "target-1"},
            artist_analysis={"status": "pending"},
        )
    )

    calls: list[str] = []

    def fail_separation(_workflow_id: str):
        raise AssertionError("artist separation should not rerun for recovered workflow")

    def fail_user_analysis(_workflow_id: str):
        raise AssertionError("user analysis should not rerun for recovered workflow")

    def fake_artist_analysis(_workflow_id: str) -> None:
        calls.append("artist")

    def fake_finalize(_workflow_id: str) -> None:
        calls.append("finalize")

    manager._separate_artist_song = fail_separation  # type: ignore[method-assign]
    manager._analyze_user_vocals = fail_user_analysis  # type: ignore[method-assign]
    manager._analyze_artist_song = fake_artist_analysis  # type: ignore[method-assign]
    manager._finalize_workflow = fake_finalize  # type: ignore[method-assign]

    manager._run_workflow(workflow_id)

    assert calls == ["artist", "finalize"]


def test_attach_user_samples_is_idempotent_across_workflow_retries(workflow_app, monkeypatch, tmp_path):
    monkeypatch.setattr(
        voice_profiles_module,
        "analyze_training_sample",
        lambda *_args, **_kwargs: {"qa_status": "pass", "duration_seconds": 1.25},
    )

    manager = ConversionWorkflowManager(workflow_app)
    store = manager._profile_store()
    profile_id = store.save(
        {
            "name": "Target User",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_TARGET_USER,
        }
    )
    source_path = _touch(tmp_path / "samples" / "user.wav")
    user_vocals = [{"filename": "user.wav", "path": source_path}]

    manager._duration_seconds = lambda _audio_path: 1.25  # type: ignore[method-assign]

    first = manager._attach_user_samples(
        store,
        profile_id,
        user_vocals,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        workflow_id="wf-dup",
    )
    second = manager._attach_user_samples(
        store,
        profile_id,
        user_vocals,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        workflow_id="wf-dup",
    )

    samples = store.list_training_samples(profile_id)

    assert first == {"attached": 1, "skipped_existing": 0}
    assert second == {"attached": 0, "skipped_existing": 1}
    assert len(samples) == 1
    assert samples[0].extra_metadata["provenance"] == "conversion_workflow_user_vocals"
    assert samples[0].extra_metadata["workflow_id"] == "wf-dup"
    assert samples[0].extra_metadata["workflow_auto_attached"] is True


def test_artist_analysis_persists_serialized_diarization_and_assignments(workflow_app, monkeypatch, tmp_path):
    monkeypatch.setattr(
        voice_profiles_module,
        "analyze_training_sample",
        lambda *_args, **_kwargs: {"qa_status": "pass", "duration_seconds": 2.5},
    )

    manager = ConversionWorkflowManager(workflow_app)
    store = manager._profile_store()
    source_profile_id = store.save(
        {
            "name": "Artist Source",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_SOURCE_ARTIST,
        }
    )
    store.save_speaker_embedding(source_profile_id, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    artist_song_path = _touch(tmp_path / "artist-analysis" / "artist.wav")
    artist_vocals_path = _touch(tmp_path / "artist-analysis" / "artist_vocals.wav")
    workflow_id = "wf-artist-analysis"
    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            workflow_id,
            artist_song_path=artist_song_path,
            artist_vocals_path=artist_vocals_path,
            status="processing",
            stage="diarizing_artist_song",
            user_analysis={"status": "resolved", "resolved_target_profile_id": "target-1"},
        )
    )

    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00", confidence=0.97),
        ],
        num_speakers=1,
        audio_duration=2.0,
        speaker_embeddings={"SPEAKER_00": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
    )

    class FakeDiarizer:
        def __init__(self, device: str | None = None):
            self.device = device

        def diarize(self, audio_path: str) -> DiarizationResult:
            assert audio_path == artist_vocals_path
            return diarization

        def extract_speaker_embedding(self, *args, **kwargs):
            raise AssertionError("speaker embedding fallback should not run when diarization already provides one")

        def extract_speaker_audio(self, _audio_path, _diarization, _speaker_id, output_path=None):
            extracted = Path(output_path)
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_bytes(b"artist speaker")
            return str(extracted)

    monkeypatch.setattr(workflow_module, "SpeakerDiarizer", FakeDiarizer)
    manager._duration_seconds = lambda _audio_path: 2.5  # type: ignore[method-assign]

    manager._analyze_artist_song(workflow_id)

    workflow = workflow_app.state_store.get_conversion_workflow(workflow_id)
    assert workflow is not None

    artist_analysis = workflow["artist_analysis"]
    assignment = artist_analysis["speaker_assignments"][0]
    samples = store.list_training_samples(source_profile_id)

    assert workflow["diarization_id"] == artist_analysis["diarization_id"]
    assert artist_analysis["status"] == "resolved"
    assert artist_analysis["dominant_speaker_id"] == "SPEAKER_00"
    assert artist_analysis["diarization"]["num_speakers"] == 1
    assert artist_analysis["diarization"]["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert assignment["speaker_id"] == "SPEAKER_00"
    assert assignment["resolved_profile_id"] == source_profile_id
    assert assignment["resolution"] == "auto_match"
    assert workflow["resolved_source_profiles"][0]["profile_id"] == source_profile_id
    assert len(samples) == 1
    assert samples[0].extra_metadata["provenance"] == "conversion_workflow_candidate"
    assert samples[0].extra_metadata["workflow_id"] == workflow_id
    assert samples[0].extra_metadata["speaker_id"] == "SPEAKER_00"


def test_restart_preserves_attached_training_job_without_rescheduling(workflow_app, monkeypatch, tmp_path):
    manager = ConversionWorkflowManager(workflow_app)
    store = manager._profile_store()
    profile_id = store.save(
        {
            "name": "Target User",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_TARGET_USER,
        }
    )
    sample_path = _write_wav(tmp_path / "attached-training" / "user.wav")
    store.add_training_sample(
        profile_id=profile_id,
        vocals_path=sample_path,
        duration=1.0,
        source_file=Path(sample_path).name,
    )

    workflow_id = "wf-training-attached"
    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            workflow_id,
            artist_song_path=_touch(tmp_path / "attached-training" / "artist.wav"),
            status="ready_for_training",
            stage="ready_for_training",
            resolved_target_profile_id=profile_id,
            user_analysis={"status": "resolved", "resolved_target_profile_id": profile_id},
            artist_analysis={"status": "resolved"},
        )
    )
    manager.attach_training_job(workflow_id, "job-running")
    workflow_app.state_store.save_training_job(
        {
            "job_id": "job-running",
            "profile_id": profile_id,
            "status": "running",
            "sample_ids": [],
            "progress": 35,
        }
    )

    scheduled: list[tuple[str, str]] = []

    def fake_schedule(self, resumed_workflow_id: str, *, reason: str) -> None:
        scheduled.append((resumed_workflow_id, reason))

    monkeypatch.setattr(ConversionWorkflowManager, "_schedule_workflow_run", fake_schedule)

    restarted_manager = ConversionWorkflowManager(workflow_app)
    hydrated = restarted_manager.get_workflow(workflow_id)
    persisted = workflow_app.state_store.get_conversion_workflow(workflow_id)

    assert hydrated is not None
    assert hydrated["status"] == "training_in_progress"
    assert hydrated["stage"] == "training_in_progress"
    assert hydrated["current_training_job_id"] == "job-running"
    assert hydrated["training_readiness"]["ready"] is True
    assert scheduled == []
    assert persisted is not None
    assert persisted["recovery"]["resume_count"] == 0


def test_restart_hydrates_active_training_job_from_profile_when_pointer_missing(workflow_app, tmp_path):
    manager = ConversionWorkflowManager(workflow_app)
    store = manager._profile_store()
    profile_id = store.save(
        {
            "name": "Target User",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_TARGET_USER,
        }
    )
    sample_path = _write_wav(tmp_path / "hydrate-training" / "user.wav")
    store.add_training_sample(
        profile_id=profile_id,
        vocals_path=sample_path,
        duration=1.0,
        source_file=Path(sample_path).name,
    )

    workflow_id = "wf-hydrate-training"
    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            workflow_id,
            artist_song_path=_touch(tmp_path / "hydrate-training" / "artist.wav"),
            status="ready_for_training",
            stage="ready_for_training",
            resolved_target_profile_id=profile_id,
            user_analysis={"status": "resolved", "resolved_target_profile_id": profile_id},
            artist_analysis={"status": "resolved"},
        )
    )
    workflow_app.state_store.save_training_job(
        {
            "job_id": "job-pending",
            "profile_id": profile_id,
            "status": "pending",
            "sample_ids": [],
            "progress": 0,
        }
    )

    restarted_manager = ConversionWorkflowManager(workflow_app)
    hydrated = restarted_manager.get_workflow(workflow_id)

    assert hydrated is not None
    assert hydrated["status"] == "training_in_progress"
    assert hydrated["stage"] == "training_in_progress"
    assert hydrated["current_training_job_id"] == "job-pending"
    assert hydrated["training_readiness"]["ready"] is True


def test_restart_clears_completed_training_job_and_enables_conversion(workflow_app, tmp_path):
    manager = ConversionWorkflowManager(workflow_app)
    store = manager._profile_store()
    profile_id = store.save(
        {
            "name": "Target User",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_TARGET_USER,
            "has_trained_model": True,
            "has_adapter_model": True,
            "active_model_type": "adapter",
        }
    )
    sample_path = _write_wav(tmp_path / "completed-training" / "user.wav")
    store.add_training_sample(
        profile_id=profile_id,
        vocals_path=sample_path,
        duration=1.0,
        source_file=Path(sample_path).name,
    )

    workflow_id = "wf-completed-training"
    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            workflow_id,
            artist_song_path=_touch(tmp_path / "completed-training" / "artist.wav"),
            status="training_in_progress",
            stage="training_in_progress",
            resolved_target_profile_id=profile_id,
            current_training_job_id="job-complete",
            user_analysis={"status": "resolved", "resolved_target_profile_id": profile_id},
            artist_analysis={"status": "resolved"},
        )
    )
    workflow_app.state_store.save_training_job(
        {
            "job_id": "job-complete",
            "profile_id": profile_id,
            "status": "completed",
            "sample_ids": [],
            "progress": 100,
            "results": {"artifact": "adapter"},
        }
    )

    restarted_manager = ConversionWorkflowManager(workflow_app)
    hydrated = restarted_manager.get_workflow(workflow_id)

    assert hydrated is not None
    assert hydrated["current_training_job_id"] is None
    assert hydrated["status"] == "ready_for_conversion"
    assert hydrated["stage"] == "ready_for_conversion"
    assert hydrated["conversion_readiness"]["ready"] is True
    assert hydrated["conversion_readiness"]["reason"] == "ready"


def test_restart_recovers_processing_workflows_without_disturbing_parallel_training_workflows(
    workflow_app,
    monkeypatch,
    tmp_path,
):
    manager = ConversionWorkflowManager(workflow_app)
    store = manager._profile_store()

    profile_a = store.save(
        {
            "name": "Target A",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_TARGET_USER,
        }
    )
    profile_b = store.save(
        {
            "name": "Target B",
            "created_from": "manual",
            "profile_role": PROFILE_ROLE_TARGET_USER,
        }
    )
    for profile_id, stem in [(profile_a, "a"), (profile_b, "b")]:
        sample_path = _write_wav(tmp_path / "parallel-training" / f"{stem}.wav")
        store.add_training_sample(
            profile_id=profile_id,
            vocals_path=sample_path,
            duration=1.0,
            source_file=Path(sample_path).name,
        )

    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            "wf-a",
            artist_song_path=_touch(tmp_path / "parallel-training" / "artist-a.wav"),
            status="ready_for_training",
            stage="ready_for_training",
            resolved_target_profile_id=profile_a,
            user_analysis={"status": "resolved", "resolved_target_profile_id": profile_a},
            artist_analysis={"status": "resolved"},
        )
    )
    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            "wf-b",
            artist_song_path=_touch(tmp_path / "parallel-training" / "artist-b.wav"),
            status="ready_for_training",
            stage="ready_for_training",
            resolved_target_profile_id=profile_b,
            user_analysis={"status": "resolved", "resolved_target_profile_id": profile_b},
            artist_analysis={"status": "resolved"},
        )
    )
    workflow_app.state_store.save_conversion_workflow(
        _workflow_payload(
            "wf-recover",
            artist_song_path=_touch(tmp_path / "parallel-training" / "artist-recover.wav"),
            status="processing",
            stage="analyzing_user_vocals",
        )
    )
    workflow_app.state_store.save_training_job(
        {
            "job_id": "job-a",
            "profile_id": profile_a,
            "status": "running",
            "sample_ids": [],
            "progress": 42,
        }
    )
    workflow_app.state_store.save_training_job(
        {
            "job_id": "job-b",
            "profile_id": profile_b,
            "status": "pending",
            "sample_ids": [],
            "progress": 0,
        }
    )

    scheduled: list[tuple[str, str]] = []

    def fake_schedule(self, workflow_id: str, *, reason: str) -> None:
        scheduled.append((workflow_id, reason))

    monkeypatch.setattr(ConversionWorkflowManager, "_schedule_workflow_run", fake_schedule)

    restarted_manager = ConversionWorkflowManager(workflow_app)

    workflow_a = restarted_manager.get_workflow("wf-a")
    workflow_b = restarted_manager.get_workflow("wf-b")
    recoverable = workflow_app.state_store.get_conversion_workflow("wf-recover")

    assert workflow_a is not None
    assert workflow_a["current_training_job_id"] == "job-a"
    assert workflow_a["status"] == "training_in_progress"
    assert workflow_a["stage"] == "training_in_progress"

    assert workflow_b is not None
    assert workflow_b["current_training_job_id"] == "job-b"
    assert workflow_b["status"] == "training_in_progress"
    assert workflow_b["stage"] == "training_in_progress"

    assert recoverable is not None
    assert recoverable["status"] == "processing"
    assert recoverable["recovery"]["resume_count"] == 1
    assert scheduled == [("wf-recover", "startup_recovery")]
