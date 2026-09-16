"""Contract for creating a sing-along target profile from a name alone.

The sing-along studio creates the profile *before* recording starts, so the
endpoint under test has to produce a profile that the sample-upload route
accepts with no embedding and no reference audio.
"""

from __future__ import annotations

import io

import pytest

from tests.fixtures.audio import write_voiced_wav


@pytest.fixture
def profile_app(tmp_path):
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={"TESTING": True, "DATA_DIR": str(tmp_path)},
        testing=True,
    )
    app.socketio = socketio
    return app


@pytest.fixture
def profile_client(profile_app):
    return profile_app.test_client()


def _create(client, name):
    return client.post("/api/v1/voice/profiles", json={"name": name})


def test_create_profile_from_name_only(profile_client):
    """A name alone yields a trainable target-user profile with no samples."""
    response = _create(profile_client, "Sing Along Voice")

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["name"] == "Sing Along Voice"
    assert payload["profile_role"] == "target_user"
    assert payload["created_from"] == "singalong"
    assert payload["sample_count"] == 0
    assert payload["profile_id"]


def test_created_profile_accepts_a_recorded_take(profile_client, tmp_path):
    """The take attaches to the fresh profile — the contract the studio needs.

    Without an embedding or reference audio the profile still has to satisfy
    ensure_profile_in_store, or the recording would be dropped after capture.
    """
    profile_id = _create(profile_client, "Take Owner").get_json()["profile_id"]

    take_path = tmp_path / "take.wav"
    # Above the 3s minimum the sample route enforces on training audio.
    write_voiced_wav(take_path, duration_seconds=4.0, sample_rate=22_050, frequency_hz=330)

    response = profile_client.post(
        f"/api/v1/profiles/{profile_id}/samples",
        data={
            "audio": (io.BytesIO(take_path.read_bytes()), "take.wav", "audio/wav"),
            "metadata": '{"source": "browser_singalong_capture"}',
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert response.get_json()["profile_id"] == profile_id

    listed = profile_client.get(f"/api/v1/profiles/{profile_id}/samples")
    assert listed.status_code == 200
    assert len(listed.get_json()) == 1


def test_created_profile_is_listed_for_the_picker(profile_client):
    """The studio's profile picker reads GET /voice/profiles."""
    profile_id = _create(profile_client, "Listed Voice").get_json()["profile_id"]

    listing = profile_client.get("/api/v1/voice/profiles")

    assert listing.status_code == 200
    assert profile_id in {profile["profile_id"] for profile in listing.get_json()}


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_blank_name_is_rejected(profile_client, name):
    """An empty name must fail loudly instead of creating an unnameable profile."""
    response = _create(profile_client, name)

    assert response.status_code == 400
    assert response.get_json()["error_code"] == "profile_name_required"


def test_missing_name_field_is_rejected(profile_client):
    response = profile_client.post("/api/v1/voice/profiles", json={})

    assert response.status_code == 400
    assert response.get_json()["error_code"] == "profile_name_required"


@pytest.mark.parametrize("duplicate", ["Existing Voice", "existing voice", "  EXISTING VOICE  "])
def test_duplicate_name_is_rejected(profile_client, duplicate):
    """Case and surrounding whitespace do not make a name distinct."""
    first_id = _create(profile_client, "Existing Voice").get_json()["profile_id"]

    response = _create(profile_client, duplicate)

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error_code"] == "duplicate_profile_name"
    assert payload["profile_id"] == first_id
    assert "Existing Voice" in payload["error"]

    listing = profile_client.get("/api/v1/voice/profiles").get_json()
    assert len([p for p in listing if p["profile_id"] == first_id]) == 1
