from __future__ import annotations

from auto_voice.runtime_contract import build_packaged_artifact_manifest
from auto_voice.storage.voice_profiles import VoiceProfileStore
from tests.fixtures.audio import write_voiced_wav


def _write_wav(path, duration_seconds: float = 1.0) -> None:
    write_voiced_wav(path, duration_seconds=duration_seconds, sample_rate=16000)


def test_voice_profile_store_persists_runtime_artifact_manifest(tmp_path):
    store = VoiceProfileStore(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained_models"),
    )
    profile_id = store.save({"name": "Test Voice"})
    sample_path = tmp_path / "sample.wav"
    _write_wav(sample_path, duration_seconds=3.5)
    training_sample = store.add_training_sample(profile_id, str(sample_path), duration=3.5)

    manifest = build_packaged_artifact_manifest(
        profile_id=profile_id,
        display_name="Test Voice",
        model_family="realtime",
        canonical_pipeline="realtime",
        sample_rate=22050,
        speaker_embedding_dim=256,
        mel_bins=80,
        artifacts={
            "profile_json": str(tmp_path / "profiles" / f"{profile_id}.json"),
            "speaker_embedding": str(tmp_path / "profiles" / f"{profile_id}.npy"),
            "adapter": str(tmp_path / "trained_models" / f"{profile_id}_adapter.pt"),
            "full_model": None,
        },
    )

    manifest_path = store.save_runtime_artifact_manifest(profile_id, manifest.to_dict())
    loaded_profile = store.load(profile_id)

    assert manifest_path.endswith("artifact_manifest.json")
    assert store.has_trained_model(profile_id) is True
    assert loaded_profile["runtime_artifact_manifest_path"] == manifest_path
    assert loaded_profile["runtime_artifact_pipeline"] == "realtime"
    assert loaded_profile["reference_audio_paths"] == [training_sample.vocals_path]

    persisted_manifest = store.load_runtime_artifact_manifest(profile_id)
    assert persisted_manifest is not None
    assert persisted_manifest["metadata"]["profile_role"] == "target_user"
    assert persisted_manifest["metadata"]["reference_audio"] == [
        {
            "path": training_sample.vocals_path,
            "source": "training_sample",
            "sample_id": training_sample.sample_id,
            "duration_seconds": 3.5,
            "created_at": training_sample.created_at,
        }
    ]


def test_voice_profile_store_falls_back_to_separated_track_reference_audio(tmp_path):
    store = VoiceProfileStore(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained_models"),
    )
    vocals = tmp_path / "isolated_vocals.wav"
    _write_wav(vocals)
    profile_id = store.save(
        {
            "name": "Separated Voice",
            "separated_tracks": {"vocals": str(vocals)},
        }
    )

    loaded_profile = store.load(profile_id)

    assert loaded_profile["reference_audio"] == [
        {
            "path": str(vocals),
            "source": "separated_track",
            "created_at": loaded_profile["created_at"],
        }
    ]
    assert loaded_profile["primary_reference_audio_path"] == str(vocals)
