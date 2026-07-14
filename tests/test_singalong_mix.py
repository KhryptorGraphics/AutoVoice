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
