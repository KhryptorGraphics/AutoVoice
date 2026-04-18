"""Tests for scripts/prepare_pillowtalk_dataset.py."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import prepare_pillowtalk_dataset as dataset_script


def _write_audio(path: Path, frequency: float) -> None:
    sample_rate = 16000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * frequency * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)


def test_prepare_dataset_builds_manifest_profiles_and_samples(tmp_path, monkeypatch):
    william_raw = tmp_path / "fixtures" / "william.wav"
    conor_raw = tmp_path / "fixtures" / "conor.wav"
    william_vocals = tmp_path / "separated" / "william" / "vocals.wav"
    william_instrumental = tmp_path / "separated" / "william" / "instrumental.wav"
    conor_vocals = tmp_path / "separated" / "conor" / "vocals.wav"
    conor_instrumental = tmp_path / "separated" / "conor" / "instrumental.wav"

    _write_audio(william_raw, 220.0)
    _write_audio(conor_raw, 330.0)
    _write_audio(william_vocals, 220.0)
    _write_audio(william_instrumental, 110.0)
    _write_audio(conor_vocals, 330.0)
    _write_audio(conor_instrumental, 165.0)

    artists = {
        "william_singe": replace(
            dataset_script.ARTISTS["william_singe"],
            profile_id="profile-william",
            raw_audio=william_raw,
        ),
        "conor_maynard": replace(
            dataset_script.ARTISTS["conor_maynard"],
            profile_id="profile-conor",
            raw_audio=conor_raw,
        ),
    }

    # Patch separated asset accessors to use the temp fixtures.
    monkeypatch.setattr(dataset_script, "ARTISTS", artists)
    monkeypatch.setattr(
        type(artists["william_singe"]),
        "separated_vocals",
        property(lambda self: william_vocals if self.key == "william_singe" else conor_vocals),
    )
    monkeypatch.setattr(
        type(artists["william_singe"]),
        "separated_instrumental",
        property(
            lambda self: william_instrumental
            if self.key == "william_singe"
            else conor_instrumental
        ),
    )

    monkeypatch.setattr(
        dataset_script,
        "compute_embedding",
        lambda audio_path, device: {
            "embedding": np.ones(256, dtype=np.float32) / 16.0,
            "backend": f"mock-{Path(audio_path).stem}",
        },
    )

    def fake_write_aligned_pair(pair_id, source_vocals, target_vocals, target_instrumental, destination_dir):
        del pair_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        aligned_audio_path = destination_dir / "source.wav"
        _write_audio(aligned_audio_path, 440.0)
        dataset_script.ensure_relative_symlink(target_vocals, destination_dir / "reference.wav")
        dataset_script.ensure_relative_symlink(target_instrumental, destination_dir / "instrumental.wav")
        metadata_path = destination_dir / "alignment.json"
        metadata_path.write_text(json.dumps({"sample_rate": 16000, "method": "test"}))
        return {
            "source_audio": aligned_audio_path,
            "reference_audio": destination_dir / "reference.wav",
            "instrumental_audio": destination_dir / "instrumental.wav",
            "alignment_metadata": metadata_path,
            "stats": {"sample_rate": 16000, "method": "test"},
        }

    monkeypatch.setattr(dataset_script, "write_aligned_pair", fake_write_aligned_pair)

    output_dir = tmp_path / "dataset"
    profiles_dir = tmp_path / "profiles"
    samples_dir = tmp_path / "samples"

    metadata_path = dataset_script.prepare_dataset(
        output_dir=output_dir,
        device="cpu",
        profiles_dir=profiles_dir,
        samples_dir=samples_dir,
    )

    manifest = json.loads(metadata_path.read_text())
    assert metadata_path.exists()
    assert manifest["dataset_name"] == "pillowtalk"
    assert set(manifest["tracks"].keys()) == {"william_singe", "conor_maynard"}
    assert set(manifest["alignments"].keys()) == {"william_to_conor", "conor_to_william"}
    assert {sample["split"] for sample in manifest["samples"]} == {"train", "holdout"}
    assert manifest["speaker_backends"]["william_singe"].startswith("mock-")

    assert (profiles_dir / "profile-william.json").exists()
    assert (profiles_dir / "profile-conor.json").exists()
    assert any((samples_dir / "profile-william").rglob("vocals.wav"))
    assert any((samples_dir / "profile-conor").rglob("vocals.wav"))
