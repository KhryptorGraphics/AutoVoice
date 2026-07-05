"""Unit tests for the uvr separation bridge (no real env/models needed)."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from auto_voice.inference import separation_bridge as sb


def _fake_run(write_outputs):
    """Build a subprocess.run fake that writes files into the --output_dir."""
    def run(cmd, **kwargs):
        out_dir = None
        for flag in ("--output_dir",):
            if flag in cmd:
                out_dir = Path(cmd[cmd.index(flag) + 1])
        if out_dir is None:  # basic-pitch call: positional outdir first arg
            out_dir = Path(cmd[1])
        write_outputs(out_dir, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return run


class TestAvailability:
    def test_unavailable_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(sb, "DEFAULT_SEPARATOR_BIN", "/nonexistent/audio-separator")
        assert sb.is_available() is False

    def test_separate_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(sb, "DEFAULT_SEPARATOR_BIN", "/nonexistent/audio-separator")
        with pytest.raises(RuntimeError, match="not found"):
            sb.separate_lead_backing(np.ones(1000, dtype=np.float32), 16000)


class TestSeparateLeadBacking:
    @pytest.fixture(autouse=True)
    def available(self, monkeypatch):
        monkeypatch.setattr(sb, "is_available", lambda: True)

    def test_parses_vocals_and_instrumental_outputs(self, monkeypatch, tmp_path):
        sr = 16000
        voc = np.random.default_rng(0).standard_normal(sr).astype(np.float32) * 0.1

        def write(out_dir, cmd):
            t = np.arange(sr) / sr
            sf.write(out_dir / "in_(Vocals)_model.wav",
                     (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
            sf.write(out_dir / "in_(Instrumental)_model.wav",
                     (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

        monkeypatch.setattr(sb.subprocess, "run", _fake_run(write))
        lead, backing = sb.separate_lead_backing(voc, sr, data_dir=str(tmp_path))
        assert lead.shape == voc.shape == backing.shape
        assert lead.dtype == np.float32
        # lead is the 220Hz file, backing the 440Hz one
        assert np.abs(lead).max() > 0.4
        assert np.abs(backing).max() < 0.3

    def test_failure_raises(self, monkeypatch, tmp_path):
        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        monkeypatch.setattr(sb.subprocess, "run", run)
        with pytest.raises(RuntimeError, match="audio-separator failed"):
            sb.separate_lead_backing(np.ones(1000, dtype=np.float32), 16000,
                                     data_dir=str(tmp_path))

    def test_unexpected_outputs_raise(self, monkeypatch, tmp_path):
        def write(out_dir, cmd):
            sf.write(out_dir / "in_(Other)_model.wav",
                     np.zeros(100, dtype=np.float32), 16000)
        monkeypatch.setattr(sb.subprocess, "run", _fake_run(write))
        with pytest.raises(RuntimeError, match="Unexpected separator outputs"):
            sb.separate_lead_backing(np.ones(1000, dtype=np.float32), 16000,
                                     data_dir=str(tmp_path))

    def test_length_fitting(self, monkeypatch, tmp_path):
        sr = 16000
        voc = np.ones(sr, dtype=np.float32)

        def write(out_dir, cmd):
            sf.write(out_dir / "in_(Vocals)_m.wav",
                     np.ones(sr + 500, dtype=np.float32), sr)   # longer
            sf.write(out_dir / "in_(Instrumental)_m.wav",
                     np.ones(sr - 500, dtype=np.float32), sr)   # shorter
        monkeypatch.setattr(sb.subprocess, "run", _fake_run(write))
        lead, backing = sb.separate_lead_backing(voc, sr, data_dir=str(tmp_path))
        assert len(lead) == len(backing) == sr


class TestPolyphonicNotes:
    def test_parses_note_csv(self, monkeypatch):
        monkeypatch.setattr(sb.os.path, "exists", lambda p: True)

        def write(out_dir, cmd):
            (out_dir / "stack_basic_pitch.csv").write_text(
                "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n"
                "0.5,2.0,57,81,1,1,1\n"
                "0.6,1.9,64,40,1,1\n"
            )
        monkeypatch.setattr(sb.subprocess, "run", _fake_run(write))
        notes = sb.polyphonic_notes(np.ones(1000, dtype=np.float32), 16000)
        assert len(notes) == 2
        assert notes[0]["pitch_midi"] == 57.0
        assert abs(notes[0]["amplitude"] - 81 / 127) < 1e-6
        assert notes[0]["start"] == 0.5 and notes[0]["end"] == 2.0

    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(sb, "DEFAULT_BASIC_PITCH_BIN", "/nonexistent/basic-pitch")
        with pytest.raises(RuntimeError, match="not found"):
            sb.polyphonic_notes(np.ones(100, dtype=np.float32), 16000)

    def test_failure_raises(self, monkeypatch):
        monkeypatch.setattr(sb.os.path, "exists", lambda p: True)

        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="nope")
        monkeypatch.setattr(sb.subprocess, "run", run)
        with pytest.raises(RuntimeError, match="basic-pitch failed"):
            sb.polyphonic_notes(np.ones(100, dtype=np.float32), 16000)
