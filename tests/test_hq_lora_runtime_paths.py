from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import train_hq_lora  # noqa: E402
import train_hq_lora_optimized  # noqa: E402


def test_train_hq_lora_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "hq-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = train_hq_lora.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["diarized_dir"] == data_dir / "diarized_youtube"
    assert paths["separated_dir"] == data_dir / "separated_youtube"
    assert paths["checkpoints_dir"] == data_dir / "checkpoints" / "hq"
    assert paths["output_dir"] == data_dir / "trained_models" / "hq"


def test_train_hq_lora_configure_runtime_paths_supports_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "hq-explicit"

    paths = train_hq_lora.configure_runtime_paths(str(data_dir))

    assert paths["data_dir"] == data_dir
    assert train_hq_lora.DATA_DIR == data_dir
    assert train_hq_lora.DIARIZED_DIR == data_dir / "diarized_youtube"
    assert train_hq_lora.SEPARATED_DIR == data_dir / "separated_youtube"
    assert train_hq_lora.CHECKPOINTS_DIR == data_dir / "checkpoints" / "hq"
    assert train_hq_lora.OUTPUT_DIR == data_dir / "trained_models" / "hq"


def test_train_hq_lora_optimized_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "hq-optimized-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = train_hq_lora_optimized.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["diarized_dir"] == data_dir / "diarized_youtube"
    assert paths["separated_dir"] == data_dir / "separated_youtube"
    assert paths["features_dir"] == data_dir / "features_cache"
    assert paths["checkpoints_dir"] == data_dir / "checkpoints" / "hq"
    assert paths["output_dir"] == data_dir / "trained_models" / "hq"


def test_train_hq_lora_optimized_configure_runtime_paths_supports_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "hq-optimized-explicit"

    paths = train_hq_lora_optimized.configure_runtime_paths(str(data_dir))

    assert paths["data_dir"] == data_dir
    assert train_hq_lora_optimized.DATA_DIR == data_dir
    assert train_hq_lora_optimized.DIARIZED_DIR == data_dir / "diarized_youtube"
    assert train_hq_lora_optimized.SEPARATED_DIR == data_dir / "separated_youtube"
    assert train_hq_lora_optimized.FEATURES_DIR == data_dir / "features_cache"
    assert train_hq_lora_optimized.CHECKPOINTS_DIR == data_dir / "checkpoints" / "hq"
    assert train_hq_lora_optimized.OUTPUT_DIR == data_dir / "trained_models" / "hq"
