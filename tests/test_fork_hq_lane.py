"""Tests for the fork-backed HQ (stereo, native-rate) conversion lane.

The heavy pieces (separator, fork inference, pitch) are stubbed; the real
stereo/44.1k end-to-end is covered by a live API smoke, not here.
"""
import numpy as np
import pytest
import soundfile as sf

from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
from auto_voice.inference import svc_fork_bridge


def _pipeline(tmp_path):
    return SingingConversionPipeline(device="cpu", config={"data_dir": str(tmp_path)})


def test_fork_hq_lane_is_stereo_and_mirrors_contract(tmp_path, monkeypatch):
    song = tmp_path / "song.wav"
    sf.write(str(song), (np.random.randn(44100 * 3, 2) * 0.1).astype(np.float32), 44100)
    p = _pipeline(tmp_path)

    class FakeSep:
        def separate(self, audio, sr, mono=True):
            n = audio.shape[-1]
            assert mono is False  # HQ lane must request stereo stems
            return {"vocals": (np.random.randn(2, n) * 0.1).astype(np.float32),
                    "instrumental": (np.random.randn(2, n) * 0.1).astype(np.float32)}

    class FakeMM:
        def infer(self, audio, spk, emb, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)  # mono vocal at sr

    monkeypatch.setattr(p, "_get_separator", lambda: FakeSep())
    monkeypatch.setattr(p, "_get_model_manager", lambda: FakeMM())
    monkeypatch.setattr(p, "_extract_pitch", lambda a, sr: np.zeros(8, np.float32))

    res = p._convert_song_fork_hq(str(song), "prof-1", 1.0, 0.9, True, "balanced", 0.0)

    # every key convert_song's consumers read must be present
    for k in ("mixed_audio", "sample_rate", "duration", "metadata",
              "f0_contour", "f0_original", "f0_sample_rate", "stems"):
        assert k in res, f"missing result key {k}"
    mx = res["mixed_audio"]
    assert isinstance(mx, np.ndarray) and mx.ndim == 2 and mx.shape[1] == 2  # STEREO (frames, 2)
    assert res["sample_rate"] == p._output_sample_rate
    assert float(np.abs(mx).max()) <= 0.95 + 1e-6
    md = res["metadata"]
    for k in ("active_model_type", "speaker_id", "quality_post_processing",
              "target_profile_id", "vocal_volume", "instrumental_volume"):
        assert k in md
    assert md["active_model_type"] == "svc_fork"


def test_convert_song_delegates_to_fork_lane_when_available(tmp_path, monkeypatch):
    song = tmp_path / "s.wav"; song.write_bytes(b"x")   # file exists; delegation precedes load
    p = _pipeline(tmp_path)
    sentinel = {"mixed_audio": np.zeros((4, 2), np.float32), "sample_rate": 44100}
    monkeypatch.setattr(svc_fork_bridge, "is_available", lambda pid, dd: True)
    called = {}
    def fake_lane(song_path, profile, vv, iv, rs, preset, ps):
        called["hit"] = (song_path, profile); return sentinel
    monkeypatch.setattr(p, "_convert_song_fork_hq", fake_lane)

    assert p.convert_song(str(song), "prof-1") is sentinel
    assert called["hit"][1] == "prof-1"


def test_convert_song_does_not_delegate_when_not_fork(tmp_path, monkeypatch):
    song = tmp_path / "s.wav"; song.write_bytes(b"x")
    p = _pipeline(tmp_path)
    monkeypatch.setattr(svc_fork_bridge, "is_available", lambda pid, dd: False)
    # legacy path must be taken -> it tries to librosa.load the bogus file and fails
    with pytest.raises(Exception):
        p.convert_song(str(song), "prof-1")
