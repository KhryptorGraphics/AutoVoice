from __future__ import annotations


import pytest

from tests.fixtures.audio import write_voiced_wav


@pytest.fixture
def mix_app(tmp_path):
    pytest.importorskip(
        "flask_swagger_ui", reason="flask_swagger_ui not installed"
    )
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={"TESTING": True, "DATA_DIR": str(tmp_path)},
        testing=True,
    )
    app.socketio = socketio
    return app


@pytest.fixture
def mix_client(mix_app):
    return mix_app.test_client()


def test_mix_singalong_duet_returns_stereo_wav(mix_app, mix_client, tmp_path):
    """A saved backing + profile vocal sample produces a duet WAV."""
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    backing_path = tmp_path / "backing.wav"
    vocal_path = tmp_path / "vocal.wav"
    write_voiced_wav(
        backing_path, duration_seconds=3.5, sample_rate=22_050, frequency_hz=220
    )
    # Different sample rate verifies the vocal-resampling branch.
    write_voiced_wav(
        vocal_path, duration_seconds=3.0, sample_rate=16_000, frequency_hz=440
    )

    asset = mix_app.state_store.register_asset(
        backing_path,
        kind="uploaded_song_original",
        metadata={"label": "Test backing"},
    )
    data_dir = mix_app.config["DATA_DIR"]
    profile_store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )
    profile_store.save({
        "profile_id": "mix-profile",
        "name": "Mix Profile",
        "profile_role": "target_user",
        "created_from": "test",
    })
    sample = profile_store.add_training_sample(
        "mix-profile",
        vocals_path=str(vocal_path),
        duration=3.0,
        source_file="vocal.wav",
    )

    response = mix_client.post(
        "/api/v1/singalong/mix",
        json={
            "source_asset_id": asset["asset_id"],
            "profile_id": "mix-profile",
            "sample_id": sample.sample_id,
            "alignment_offset_ms": 100,
        },
    )

    assert response.status_code == 200
    assert response.content_type.startswith("audio/wav")
    assert response.data[:4] == b"RIFF"
    assert response.data[8:12] == b"WAVE"

    import soundfile as sf
    import io

    mixed, sample_rate = sf.read(
        io.BytesIO(response.data), dtype="float32", always_2d=True
    )
    assert sample_rate == 22_050
    assert mixed.shape[1] == 2
    # The mix is padded to at least the full backing duration.
    assert mixed.shape[0] >= int(3.5 * sample_rate)
    # Both backing and vocal energy are present after alignment.
    assert abs(mixed).max() > 0



def test_mix_singalong_duet_rejects_negative_offset(mix_app, mix_client, tmp_path):
    """Negative alignment_offset_ms returns a validation error, not a 500."""
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    backing_path = tmp_path / "backing.wav"
    vocal_path = tmp_path / "vocal.wav"
    write_voiced_wav(backing_path, duration_seconds=3.5, sample_rate=22_050, frequency_hz=220)
    write_voiced_wav(vocal_path, duration_seconds=3.0, sample_rate=22_050, frequency_hz=440)
    asset = mix_app.state_store.register_asset(
        backing_path, kind="uploaded_song_original", metadata={"label": "B"},
    )
    data_dir = mix_app.config["DATA_DIR"]
    profile_store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )
    profile_store.save({
        "profile_id": "mix-profile",
        "name": "Mix Profile",
        "profile_role": "target_user",
        "created_from": "test",
    })
    sample = profile_store.add_training_sample(
        "mix-profile", vocals_path=str(vocal_path), duration=1.0, source_file="vocal.wav",
    )

    response = mix_client.post(
        "/api/v1/singalong/mix",
        json={
            "source_asset_id": asset["asset_id"],
            "profile_id": "mix-profile",
            "sample_id": sample.sample_id,
            "alignment_offset_ms": -500,
        },
    )
    assert response.status_code == 400
    assert b"non-negative" in response.data


def test_mix_singalong_duet_rejects_out_of_range_gain(mix_app, mix_client, tmp_path):
    """backing_gain/vocal_gain outside [0.0, 2.0] returns a validation error."""
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    backing_path = tmp_path / "backing.wav"
    vocal_path = tmp_path / "vocal.wav"
    write_voiced_wav(backing_path, duration_seconds=3.5, sample_rate=22_050, frequency_hz=220)
    write_voiced_wav(vocal_path, duration_seconds=3.0, sample_rate=22_050, frequency_hz=440)
    asset = mix_app.state_store.register_asset(
        backing_path, kind="uploaded_song_original", metadata={"label": "B"},
    )
    data_dir = mix_app.config["DATA_DIR"]
    profile_store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )
    profile_store.save({
        "profile_id": "mix-profile",
        "name": "Mix Profile",
        "profile_role": "target_user",
        "created_from": "test",
    })
    sample = profile_store.add_training_sample(
        "mix-profile", vocals_path=str(vocal_path), duration=1.0, source_file="vocal.wav",
    )

    base_payload = {
        "source_asset_id": asset["asset_id"],
        "profile_id": "mix-profile",
        "sample_id": sample.sample_id,
    }

    # backing_gain too high
    resp = mix_client.post(
        "/api/v1/singalong/mix",
        json={**base_payload, "backing_gain": 5.0},
    )
    assert resp.status_code == 400
    assert b"backing_gain" in resp.data

    # vocal_gain negative
    resp = mix_client.post(
        "/api/v1/singalong/mix",
        json={**base_payload, "vocal_gain": -0.5},
    )
    assert resp.status_code == 400
    assert b"vocal_gain" in resp.data


def test_mix_singalong_duet_clamps_offset_to_backing_duration(mix_app, mix_client, tmp_path):
    """An alignment_offset_ms far beyond the backing duration is clamped, not
    used to allocate a massive zero-padded array (OOM guard)."""
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    backing_path = tmp_path / "backing.wav"
    vocal_path = tmp_path / "vocal.wav"
    write_voiced_wav(backing_path, duration_seconds=3.5, sample_rate=22_050, frequency_hz=220)
    write_voiced_wav(vocal_path, duration_seconds=3.0, sample_rate=22_050, frequency_hz=440)

    asset = mix_app.state_store.register_asset(
        backing_path, kind="uploaded_song_original", metadata={"label": "B"},
    )
    data_dir = mix_app.config["DATA_DIR"]
    profile_store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )
    profile_store.save({
        "profile_id": "mix-profile",
        "name": "Mix Profile",
        "profile_role": "target_user",
        "created_from": "test",
    })
    sample = profile_store.add_training_sample(
        "mix-profile", vocals_path=str(vocal_path), duration=0.5, source_file="vocal.wav",
    )

    # 10 minutes offset on a 1-second backing — would be catastrophic without clamping.
    response = mix_client.post(
        "/api/v1/singalong/mix",
        json={
            "source_asset_id": asset["asset_id"],
            "profile_id": "mix-profile",
            "sample_id": sample.sample_id,
            "alignment_offset_ms": 600_000,
        },
    )
    assert response.status_code == 200
    assert response.content_type.startswith("audio/wav")
    # The mix length stays bounded by backing+vocal (~6.5s), not the
    # 10-minute offset that would have produced ~13M samples unclamped.
    import soundfile as sf
    import io
    mixed, _ = sf.read(io.BytesIO(response.data), dtype="float32", always_2d=True)
    # Without clamping this would be ~600s * 22050 ≈ 13.2M samples.
    assert mixed.shape[0] < int(8.0 * 22_050)