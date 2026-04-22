from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import audit_loras  # noqa: E402


def test_audit_loras_runtime_paths_follow_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "audit-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = audit_loras.resolve_runtime_paths()
    auditor = audit_loras.LoRAAuditor()

    assert paths["data_dir"] == data_dir
    assert paths["voice_profiles_dir"] == data_dir / "voice_profiles"
    assert paths["trained_models_dir"] == data_dir / "trained_models"
    assert paths["diarized_dir"] == data_dir / "diarized_youtube"
    assert auditor.data_dir == data_dir
    assert auditor.voice_profiles_dir == data_dir / "voice_profiles"
    assert auditor.trained_models_dir == data_dir / "trained_models"
    assert auditor.diarized_dir == data_dir / "diarized_youtube"


def test_audit_loras_runtime_paths_support_explicit_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "env-data"))
    explicit_data_dir = tmp_path / "explicit-audit-data"

    paths = audit_loras.resolve_runtime_paths(str(explicit_data_dir))
    auditor = audit_loras.LoRAAuditor(data_dir=explicit_data_dir)

    assert paths["data_dir"] == explicit_data_dir
    assert paths["voice_profiles_dir"] == explicit_data_dir / "voice_profiles"
    assert paths["trained_models_dir"] == explicit_data_dir / "trained_models"
    assert paths["diarized_dir"] == explicit_data_dir / "diarized_youtube"
    assert auditor.data_dir == explicit_data_dir
    assert auditor.voice_profiles_dir == explicit_data_dir / "voice_profiles"
    assert auditor.trained_models_dir == explicit_data_dir / "trained_models"
    assert auditor.diarized_dir == explicit_data_dir / "diarized_youtube"
