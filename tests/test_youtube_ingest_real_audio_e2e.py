"""Real-audio YouTube ingest E2E coverage.

The deterministic lane uses tracked real-audio fixtures while mocking only the
external YouTube download and heavyweight ML models. This keeps release evidence
stable while exercising the actual ingest route, audio read/write, asset
registration, diarization persistence, profile-match suggestions, and reviewed
profile confirmation flow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_AUDIO_FIXTURE = PROJECT_ROOT / "tests" / "quality_samples" / "conor_maynard_pillowtalk.wav"


@dataclass
class _FakeSegment:
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _FakeDiarizationResult:
    segments: list[_FakeSegment]
    audio_duration: float
    num_speakers: int


def _copy_real_audio_excerpt(source: Path, destination: Path, *, seconds: float = 3.0) -> float:
    with sf.SoundFile(source) as audio_file:
        frames = min(len(audio_file), int(audio_file.samplerate * seconds))
        audio = audio_file.read(frames, dtype="float32", always_2d=False)
        sf.write(destination, audio, audio_file.samplerate)
        return frames / float(audio_file.samplerate)


def _create_app():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app
    from auto_voice.web import api as web_api

    app, socketio = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": True,
            "voice_cloning_enabled": True,
        }
    )
    app.socketio = socketio

    web_api._profile_samples.clear()
    web_api._separation_jobs.clear()
    web_api._diarization_results.clear()
    web_api._segment_assignments.clear()
    web_api._presets.clear()
    web_api._conversion_history.clear()
    web_api._youtube_downloader = None
    return app


def _create_source_profile(app, profile_id: str) -> None:
    profile = {
        "profile_id": profile_id,
        "name": "Known Fixture Artist",
        "embedding": np.ones(8, dtype=np.float32).tolist(),
        "profile_role": "source_artist",
        "created_from": "manual",
        "sample_count": 0,
        "training_sample_count": 0,
        "clean_vocal_seconds": 0.0,
        "has_trained_model": False,
        "has_adapter_model": False,
        "has_full_model": False,
        "active_model_type": "base",
    }
    app.voice_cloner.store.save(profile)
    app.voice_cloner.store.save_speaker_embedding(profile_id, np.ones(8, dtype=np.float32))


class _FixtureSeparator:
    def separate(self, audio, sr):
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=0)
        if mono.size == 0:
            mono = np.zeros(sr, dtype=np.float32)
        return {
            "vocals": mono,
            "instrumental": np.zeros_like(mono),
        }


class _FixtureDiarizer:
    def diarize(self, audio_path):
        with sf.SoundFile(audio_path) as audio_file:
            duration = len(audio_file) / float(audio_file.samplerate)
        split = max(duration / 2.0, 0.5)
        return _FakeDiarizationResult(
            segments=[
                _FakeSegment(0.0, split, "SPEAKER_00"),
                _FakeSegment(split, duration, "SPEAKER_01"),
            ],
            audio_duration=duration,
            num_speakers=2,
        )

    def extract_speaker_embedding(self, audio_path, start=None, end=None):
        del audio_path, end
        if float(start or 0.0) == 0.0:
            return np.ones(8, dtype=np.float32)
        return -np.ones(8, dtype=np.float32)

    def extract_speaker_audio(self, audio_path, diarization=None, speaker_id=None, output_path=None):
        del diarization
        target = Path(output_path) if output_path else Path(audio_path).with_name(f"{speaker_id}_review.wav")
        with sf.SoundFile(audio_path) as audio_file:
            frames = min(len(audio_file), int(audio_file.samplerate * 1.0))
            audio = audio_file.read(frames, dtype="float32", always_2d=False)
            sf.write(target, audio, audio_file.samplerate)
        return target


def _run_background_jobs_synchronously(monkeypatch):
    from auto_voice.web import api as web_api

    def run_sync(job_id, runner, payload):
        web_api._update_background_job(job_id, status="running", progress=5)
        result = runner(job_id, payload)
        web_api._update_background_job(
            job_id,
            status="completed",
            progress=100,
            result=result,
        )

    monkeypatch.setattr(web_api, "_submit_background_job", run_sync)


def _patch_ingest_models(monkeypatch):
    import auto_voice.audio.separation as separation_module
    import auto_voice.audio.speaker_diarization as diarization_module

    monkeypatch.setattr(separation_module, "VocalSeparator", _FixtureSeparator)
    monkeypatch.setattr(diarization_module, "SpeakerDiarizer", _FixtureDiarizer)


def test_youtube_ingest_real_audio_fixture_requires_review_before_profile_writes(tmp_path, monkeypatch):
    from auto_voice.web import api as web_api

    assert REAL_AUDIO_FIXTURE.exists(), f"missing real-audio fixture: {REAL_AUDIO_FIXTURE}"
    app = _create_app()
    client = app.test_client()
    web_api.YOUTUBE_DOWNLOADER_AVAILABLE = True
    _create_source_profile(app, "source-fixture-known")

    downloaded_audio = tmp_path / "downloaded-real-audio.wav"
    duration = _copy_real_audio_excerpt(REAL_AUDIO_FIXTURE, downloaded_audio)

    class FixtureDownloader:
        def download(self, url, audio_format="wav", sample_rate=44100):
            del url, audio_format, sample_rate
            return SimpleNamespace(
                success=True,
                audio_path=str(downloaded_audio),
                title="Known Fixture Artist feat New Fixture Artist - Real Fixture",
                duration=duration,
                main_artist="Known Fixture Artist",
                featured_artists=["New Fixture Artist"],
                is_cover=False,
                original_artist=None,
                song_title="Real Fixture",
                thumbnail_url=None,
                video_id="realfixture123",
                error=None,
            )

    monkeypatch.setattr(web_api, "get_youtube_downloader", lambda: FixtureDownloader())
    _run_background_jobs_synchronously(monkeypatch)
    _patch_ingest_models(monkeypatch)

    response = client.post(
        "/api/v1/youtube/ingest",
        json={
            "url": "https://www.youtube.com/watch?v=realfixture123",
            "format": "wav",
            "sample_rate": 44100,
            "consent_confirmed": True,
            "source_media_policy_confirmed": True,
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    job = client.get(f"/api/v1/youtube/ingest/{job_id}").get_json()

    assert job["status"] == "completed"
    assert job["result"]["review_required"] is True
    assert job["result"]["diarization_result"]["num_speakers"] == 2
    assert job["result"]["assets"]["vocals"]["asset_id"]
    assert job["result"]["assets"]["instrumental"]["asset_id"]
    assert app.voice_cloner.store.list_training_samples("source-fixture-known") == []

    suggestions = job["result"]["suggestions"]
    assert suggestions[0]["recommended_action"] == "assign_existing"
    assert suggestions[0]["recommended_profile_id"] == "source-fixture-known"
    assert suggestions[1]["recommended_action"] == "create_new"
    assert suggestions[1]["suggested_name"] == "New Fixture Artist"

    confirmation_response = client.post(
        f"/api/v1/youtube/ingest/{job_id}/confirm",
        json={
            "decisions": [
                {
                    "speaker_id": "SPEAKER_00",
                    "action": "assign_existing",
                    "profile_id": "source-fixture-known",
                },
                {
                    "speaker_id": "SPEAKER_01",
                    "action": "create_new",
                    "name": "New Fixture Artist",
                },
            ]
        },
    )

    assert confirmation_response.status_code == 200
    confirmation = confirmation_response.get_json()
    assert confirmation["status"] == "success"
    assert len(confirmation["applied"]) == 2
    assert app.voice_cloner.store.list_training_samples("source-fixture-known")
    assert any(profile["name"] == "New Fixture Artist" for profile in app.voice_cloner.store.list_profiles())

    assets = app.state_store.list_assets(owner_id=job_id)
    assert {asset["kind"] for asset in assets} >= {
        "youtube_audio",
        "youtube_vocals",
        "youtube_instrumental",
    }
    history = app.state_store.list_youtube_history()
    assert history and history[0]["ingestJobId"] == job_id


@pytest.mark.live_youtube
def test_live_youtube_ingest_smoke_downloads_operator_url(monkeypatch):
    from auto_voice.web import api as web_api

    live_url = os.environ.get("AUTOVOICE_LIVE_YOUTUBE_URL")
    if not live_url:
        pytest.skip("AUTOVOICE_LIVE_YOUTUBE_URL is required for the live YouTube smoke lane")

    app = _create_app()
    client = app.test_client()
    web_api.YOUTUBE_DOWNLOADER_AVAILABLE = True
    _create_source_profile(app, "source-live-known")

    _run_background_jobs_synchronously(monkeypatch)
    _patch_ingest_models(monkeypatch)

    response = client.post(
        "/api/v1/youtube/ingest",
        json={
            "url": live_url,
            "format": "wav",
            "sample_rate": 44100,
            "consent_confirmed": True,
            "source_media_policy_confirmed": True,
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    job = client.get(f"/api/v1/youtube/ingest/{job_id}").get_json()
    assert job["status"] == "completed"
    assert job["result"]["assets"]["audio"]["asset_id"]
    assert job["result"]["review_required"] is True
