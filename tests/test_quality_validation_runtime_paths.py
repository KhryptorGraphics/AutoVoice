from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import quality_validation  # noqa: E402


def test_quality_validation_runtime_paths_follow_env_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "quality-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = quality_validation.resolve_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["output_path"] == data_dir / "reports" / "quality_validation.json"


def test_quality_validation_runtime_paths_support_explicit_overrides(tmp_path):
    data_dir = tmp_path / "quality-explicit"
    output_path = tmp_path / "custom-report.json"

    paths = quality_validation.resolve_runtime_paths(data_dir, output_path=output_path)

    assert paths["data_dir"] == data_dir
    assert paths["output_path"] == output_path
