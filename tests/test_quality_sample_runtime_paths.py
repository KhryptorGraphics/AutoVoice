from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import quality_sample_paths  # noqa: E402


def test_quality_sample_runtime_paths_follow_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "quality-data"

    paths = quality_sample_paths.resolve_quality_sample_runtime_paths(str(data_dir))

    assert paths["data_dir"] == data_dir
    assert paths["william_test_audio"] == (
        data_dir / "separated_youtube" / "william_singe" / "2iVFx7f5MMU_vocals.wav"
    )
    assert paths["conor_reference_audio"] == (
        data_dir / "separated_youtube" / "conor_maynard" / "08NWh97_DME_vocals.wav"
    )
    assert paths["quality_outputs_dir"] == PROJECT_ROOT / "tests" / "quality_samples" / "outputs"
    assert paths["realtime_output"] == paths["quality_outputs_dir"] / "william_as_conor_realtime_30s.wav"
    assert paths["quality_output"] == paths["quality_outputs_dir"] / "william_as_conor_quality_30s.wav"


def test_quality_sample_runtime_paths_follow_env_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "quality-data-env"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    paths = quality_sample_paths.resolve_quality_sample_runtime_paths()

    assert paths["data_dir"] == data_dir
    assert paths["william_test_audio"].parent == data_dir / "separated_youtube" / "william_singe"
    assert paths["conor_reference_audio"].parent == data_dir / "separated_youtube" / "conor_maynard"
