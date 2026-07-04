"""Tests for the so-vits-svc-fork serving bridge and its ModelManager routing.

The fork engine itself runs in a separate conda env; these tests mock the
subprocess so they need neither the fork nor a GPU.
"""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from auto_voice.inference import svc_fork_bridge as bridge


@pytest.fixture(autouse=True)
def _clear_cache():
    bridge.clear_cache()
    yield
    bridge.clear_cache()


def _write_registry(data_dir, profile_id, model_path, config_path,
                    speaker="connor", **extra):
    d = os.path.join(str(data_dir), bridge.REGISTRY_DIRNAME)
    os.makedirs(d, exist_ok=True)
    entry = {"profile_id": profile_id, "speaker": speaker,
             "model_path": str(model_path), "config_path": str(config_path)}
    entry.update(extra)
    with open(os.path.join(d, f"{profile_id}.json"), "w") as f:
        json.dump(entry, f)


def test_no_registry_returns_none(tmp_path):
    assert bridge.get_fork_model("nope", str(tmp_path)) is None
    assert bridge.is_available("nope", str(tmp_path)) is False


def test_registry_pointing_at_missing_files_is_ignored(tmp_path):
    # A stale entry must fall back to the in-repo decoder, not fail a conversion.
    _write_registry(tmp_path, "p1", tmp_path / "missing_G.pth",
                    tmp_path / "missing.json")
    assert bridge.get_fork_model("p1", str(tmp_path)) is None
    assert bridge.is_available("p1", str(tmp_path)) is False


def test_registry_missing_speaker_is_invalid(tmp_path):
    mp = tmp_path / "G.pth"; mp.write_text("x")
    cp = tmp_path / "config.json"; cp.write_text("{}")
    d = os.path.join(str(tmp_path), bridge.REGISTRY_DIRNAME)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "p1.json"), "w") as f:
        json.dump({"model_path": str(mp), "config_path": str(cp)}, f)  # no speaker
    assert bridge.get_fork_model("p1", str(tmp_path)) is None


def test_valid_registry_returned(tmp_path):
    mp = tmp_path / "G.pth"; mp.write_text("x")
    cp = tmp_path / "config.json"; cp.write_text("{}")
    _write_registry(tmp_path, "p1", mp, cp, speaker="connor")
    entry = bridge.get_fork_model("p1", str(tmp_path))
    assert entry is not None and entry["speaker"] == "connor"
    assert bridge.is_available("p1", str(tmp_path))


def test_convert_invokes_fork_and_returns_input_sr(tmp_path, monkeypatch):
    mp = tmp_path / "G.pth"; mp.write_text("x")
    cp = tmp_path / "config.json"; cp.write_text("{}")
    _write_registry(tmp_path, "p1", mp, cp)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out_path = cmd[cmd.index("-o") + 1]
        # fork emits 44.1k; convert must resample back to the input sr
        sf.write(out_path, np.zeros(44100, dtype=np.float32), 44100)

        class _R:
            returncode = 0
            stderr = ""
            stdout = ""
        return _R()

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)

    audio = np.sin(np.linspace(0, 6.28, 22050, dtype=np.float32))
    out = bridge.convert(audio, 22050, "p1", str(tmp_path))

    assert out.dtype == np.float32
    assert abs(len(out) - 22050) <= 4          # 44.1k output -> 22.05k
    assert "-na" in captured["cmd"]            # melody preserved (no auto f0)
    assert "crepe" in captured["cmd"]


def test_convert_raises_when_fork_produces_no_output(tmp_path, monkeypatch):
    mp = tmp_path / "G.pth"; mp.write_text("x")
    cp = tmp_path / "config.json"; cp.write_text("{}")
    _write_registry(tmp_path, "p1", mp, cp)

    def fake_run(cmd, **kwargs):
        class _R:
            returncode = 1
            stderr = "boom"
            stdout = ""
        return _R()

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="fork infer produced no output"):
        bridge.convert(np.zeros(1000, dtype=np.float32), 22050, "p1", str(tmp_path))


def test_convert_unregistered_profile_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No fork model registered"):
        bridge.convert(np.zeros(10, dtype=np.float32), 22050, "ghost", str(tmp_path))


def test_model_manager_infer_routes_to_fork(monkeypatch):
    from auto_voice.inference.model_manager import ModelManager

    mm = ModelManager()  # no load(); fork routing precedes the encoder checks
    sentinel = np.arange(5, dtype=np.float32)
    monkeypatch.setattr(bridge, "is_available", lambda pid, dd: True)
    monkeypatch.setattr(bridge, "convert", lambda a, s, pid, dd: sentinel)

    out = mm.infer(np.zeros(100, dtype=np.float32), "pid",
                   np.zeros(256, dtype=np.float32), 22050)
    assert out is sentinel  # routed to fork, bypassed the unloaded in-repo decoder


def test_model_manager_infer_without_fork_falls_through(monkeypatch):
    from auto_voice.inference.model_manager import ModelManager

    mm = ModelManager()
    monkeypatch.setattr(bridge, "is_available", lambda pid, dd: False)
    # Falls through to the in-repo path, which raises because nothing is loaded.
    with pytest.raises(RuntimeError, match="ContentEncoder not loaded"):
        mm.infer(np.zeros(100, dtype=np.float32), "pid",
                 np.zeros(256, dtype=np.float32), 22050)
