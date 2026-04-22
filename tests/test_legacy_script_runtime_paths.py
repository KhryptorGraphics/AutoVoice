from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import convert_pillowtalk  # noqa: E402
import sota_conversion_nvfp4  # noqa: E402
import train_fp16_lora  # noqa: E402
import train_nvfp4_lora  # noqa: E402
import train_pillowtalk  # noqa: E402


def test_convert_pillowtalk_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "runtime-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = convert_pillowtalk.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["profiles_dir"] == data_dir / "voice_profiles"
    assert paths["models_dir"] == data_dir / "trained_models"
    assert paths["separated_dir"] == data_dir / "separated"
    assert paths["output_dir"] == data_dir / "conversions"


def test_convert_pillowtalk_load_speaker_embedding_supports_explicit_data_dir(tmp_path):
    profile_id = "profile-convert"
    data_dir = tmp_path / "convert-data"
    profiles_dir = data_dir / "voice_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    expected = np.array([0.5, -0.25, 0.125], dtype=np.float32)
    np.save(profiles_dir / f"{profile_id}.npy", expected)

    loaded = convert_pillowtalk.load_speaker_embedding(profile_id, data_dir=str(data_dir))

    assert np.array_equal(loaded, expected)


def test_train_pillowtalk_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "training-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = train_pillowtalk.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["profiles_dir"] == data_dir / "voice_profiles"
    assert paths["separated_dir"] == data_dir / "separated"
    assert paths["models_dir"] == data_dir / "trained_models"


def test_sota_runtime_paths_use_data_dir_and_pretrained_override(monkeypatch, tmp_path):
    data_dir = tmp_path / "sota-data"
    pretrained_dir = tmp_path / "pretrained-cache"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("AUTOVOICE_PRETRAINED_DIR", str(pretrained_dir))

    paths = sota_conversion_nvfp4.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["profiles_dir"] == data_dir / "voice_profiles"
    assert paths["separated_dir"] == data_dir / "separated"
    assert paths["output_dir"] == data_dir / "conversions"
    assert paths["pretrained_dir"] == pretrained_dir


def test_sota_load_speaker_embedding_supports_explicit_data_dir(tmp_path):
    profile_id = "profile-sota"
    data_dir = tmp_path / "sota-explicit"
    profiles_dir = data_dir / "voice_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(profiles_dir / f"{profile_id}.npy", expected)

    loaded = sota_conversion_nvfp4.load_speaker_embedding(profile_id, data_dir=str(data_dir))

    assert np.array_equal(loaded, expected)


def test_train_fp16_lora_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "fp16-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = train_fp16_lora.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["separated_dir"] == data_dir / "separated_youtube"
    assert paths["diarized_dir"] == data_dir / "diarized_youtube"
    assert paths["adapters_dir"] == data_dir / "trained_models" / "fp16"
    assert paths["checkpoints_dir"] == data_dir / "checkpoints" / "fp16"


def test_train_fp16_lora_configure_runtime_paths_supports_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "fp16-explicit"

    paths = train_fp16_lora.configure_runtime_paths(str(data_dir))

    assert paths["data_dir"] == data_dir
    assert train_fp16_lora.SEPARATED_DIR == data_dir / "separated_youtube"
    assert train_fp16_lora.DIARIZED_DIR == data_dir / "diarized_youtube"
    assert train_fp16_lora.ADAPTERS_DIR == data_dir / "trained_models" / "fp16"
    assert train_fp16_lora.CHECKPOINTS_DIR == data_dir / "checkpoints" / "fp16"


def test_train_nvfp4_lora_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "nvfp4-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = train_nvfp4_lora.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["separated_dir"] == data_dir / "separated_youtube"
    assert paths["diarized_dir"] == data_dir / "diarized_youtube"
    assert paths["adapters_dir"] == data_dir / "trained_models" / "nvfp4"
    assert paths["checkpoints_dir"] == data_dir / "checkpoints" / "nvfp4"


def test_train_nvfp4_lora_configure_runtime_paths_supports_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "nvfp4-explicit"

    paths = train_nvfp4_lora.configure_runtime_paths(str(data_dir))

    assert paths["data_dir"] == data_dir
    assert train_nvfp4_lora.SEPARATED_DIR == data_dir / "separated_youtube"
    assert train_nvfp4_lora.DIARIZED_DIR == data_dir / "diarized_youtube"
    assert train_nvfp4_lora.ADAPTERS_DIR == data_dir / "trained_models" / "nvfp4"
    assert train_nvfp4_lora.CHECKPOINTS_DIR == data_dir / "checkpoints" / "nvfp4"
