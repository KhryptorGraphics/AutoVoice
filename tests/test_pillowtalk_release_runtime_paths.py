from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

fake_tensorrt_engine = types.ModuleType("auto_voice.export.tensorrt_engine")


class _FakeShapeProfile:
    def __init__(self, min, opt, max):
        self.min = min
        self.opt = opt
        self.max = max


class _FakeTRTEngineBuilder:
    def __init__(self, workspace_size_gb=2.0):
        self.workspace_size_gb = workspace_size_gb

    def load_cached_engine(self, *args, **kwargs):
        return None


fake_tensorrt_engine.ShapeProfile = _FakeShapeProfile
fake_tensorrt_engine.TRTEngineBuilder = _FakeTRTEngineBuilder
sys.modules.setdefault("auto_voice.export.tensorrt_engine", fake_tensorrt_engine)

import export_hq_lora_tensorrt as export_script  # noqa: E402
import package_pillowtalk_models as package_script  # noqa: E402
import pillowtalk_release_paths  # noqa: E402
import write_pillowtalk_delivery_manifest as delivery_script  # noqa: E402


def test_release_paths_follow_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "release-data"
    models_dir = tmp_path / "release-models"
    output_dir = tmp_path / "release-output"

    paths = pillowtalk_release_paths.resolve_pillowtalk_release_paths(
        str(data_dir),
        models_dir=str(models_dir),
        output_dir=str(output_dir),
    )

    assert paths["data_dir"] == data_dir
    assert paths["models_dir"] == models_dir
    assert paths["output_dir"] == output_dir
    assert paths["profiles_dir"] == data_dir / "voice_profiles"
    assert paths["samples_dir"] == data_dir / "samples"
    assert paths["trained_models_dir"] == data_dir / "trained_models"
    assert paths["checkpoints_dir"] == data_dir / "checkpoints"
    assert paths["pillowtalk_training_dir"] == data_dir / "training" / "pillowtalk"
    assert paths["pillowtalk_dataset_manifest"] == data_dir / "training" / "pillowtalk" / "metadata.json"


def test_export_checkpoint_path_supports_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "export-data"
    spec = export_script.ARTISTS[0]

    checkpoint_path = export_script._checkpoint_path(spec, data_dir=str(data_dir))

    assert checkpoint_path == (
        data_dir / "checkpoints" / "hq" / f"{spec.checkpoint_profile_id}_hq_lora.pt"
    )


def test_package_release_uses_explicit_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "package-data"
    models_dir = tmp_path / "package-models"
    output_dir = tmp_path / "package-output"
    runtime_paths = pillowtalk_release_paths.resolve_pillowtalk_release_paths(
        str(data_dir),
        models_dir=str(models_dir),
        output_dir=str(output_dir),
    )
    monkeypatch.setattr(
        package_script,
        "resolve_pillowtalk_release_paths",
        lambda data_dir=None: runtime_paths,
    )

    spec = package_script.RELEASES[0]
    profiles_dir = runtime_paths["profiles_dir"]
    trained_models_dir = runtime_paths["trained_models_dir"]
    dataset_manifest = runtime_paths["pillowtalk_dataset_manifest"]
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (trained_models_dir / "hq").mkdir(parents=True, exist_ok=True)
    (trained_models_dir / "nvfp4").mkdir(parents=True, exist_ok=True)
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)

    (profiles_dir / f"{spec.canonical_profile_id}.json").write_text(
        json.dumps({"profile_id": spec.canonical_profile_id, "embedding": [1.0, 2.0, 3.0]})
    )
    np.save(profiles_dir / f"{spec.canonical_profile_id}.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    torch.save({"profile_id": spec.canonical_profile_id, "artist": spec.artist_key}, trained_models_dir / f"{spec.canonical_profile_id}_adapter.pt")
    torch.save({"profile_id": spec.canonical_profile_id, "global_step": 100}, trained_models_dir / "hq" / f"{spec.canonical_profile_id}_hq_lora.pt")
    torch.save({"profile_id": spec.canonical_profile_id, "global_step": 90}, trained_models_dir / "nvfp4" / f"{spec.canonical_profile_id}_nvfp4_lora.pt")
    dataset_manifest.write_text(json.dumps({"speaker_backends": {spec.artist_key: "mock-backend"}}))

    registry = package_script._package_release(
        spec,
        mirror_alias_artifacts=False,
        data_dir=str(data_dir),
    )

    release_dir = models_dir / spec.artist_key
    assert registry["dataset_manifest"] == str(dataset_manifest)
    assert registry["profile"]["json"] == str(release_dir / "profile.json")
    assert (release_dir / "artifact_manifest.json").exists()
    assert (release_dir / "registry_entry.json").exists()


def test_write_delivery_manifest_uses_explicit_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "delivery-data"
    models_dir = tmp_path / "delivery-models"
    output_dir = tmp_path / "delivery-output"
    runtime_paths = pillowtalk_release_paths.resolve_pillowtalk_release_paths(
        str(data_dir),
        models_dir=str(models_dir),
        output_dir=str(output_dir),
    )

    monkeypatch.setattr(delivery_script, "MODELS_DIR", models_dir)
    monkeypatch.setattr(delivery_script, "OUTPUT_DIR", output_dir)

    dataset_manifest = runtime_paths["pillowtalk_dataset_manifest"]
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest.write_text(json.dumps({"speaker_backends": {"william_singe": "mock"}}))

    for artist in ("william_singe", "conor_maynard"):
        release_dir = models_dir / artist
        release_dir.mkdir(parents=True, exist_ok=True)
        (release_dir / "registry_entry.json").write_text(json.dumps({"artist_key": artist}))
        (release_dir / "artifact_manifest.json").write_text(json.dumps({"artist_key": artist}))

    assert delivery_script.main(["--data-dir", str(data_dir)]) == 0

    manifest_path = output_dir / "pillowtalk_delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"]["manifest_path"] == str(dataset_manifest)
    assert manifest["models"]["william_singe"]["release_dir"] == str(models_dir / "william_singe")
