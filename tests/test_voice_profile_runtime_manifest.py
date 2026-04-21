from __future__ import annotations

from auto_voice.runtime_contract import build_packaged_artifact_manifest
from auto_voice.storage.voice_profiles import VoiceProfileStore


def test_voice_profile_store_persists_runtime_artifact_manifest(tmp_path):
    store = VoiceProfileStore(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained_models"),
    )
    profile_id = store.save({"name": "Test Voice"})

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
