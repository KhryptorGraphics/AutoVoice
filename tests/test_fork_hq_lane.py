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
    # Chain off: this asserts the lane's stereo/contract behaviour, not the
    # output-correction stages, and leaving them on runs torchcrepe on the CPU.
    return SingingConversionPipeline(device="cpu", config={
        "data_dir": str(tmp_path),
        "fork_hq_hum_suppress": False,
        "fork_hq_subharmonic_suppress": False,
        "fork_hq_voice_polish": False,
    })


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
    # Both sample peak and reconstructed inter-sample peak are constrained.
    from auto_voice.audio.true_peak import true_peak_dbfs
    assert float(np.abs(mx).max()) <= 0.95 + 1e-6
    assert true_peak_dbfs(mx) <= -1.0 + 0.01
    assert res["metadata"]["mix_true_peak_dbfs"] == -1.0
    assert res["metadata"]["mix_trim_db"] <= 0.0
    md = res["metadata"]
    for k in ("active_model_type", "speaker_id", "quality_post_processing",
              "target_profile_id", "vocal_volume", "instrumental_volume"):
        assert k in md
    assert md["active_model_type"] == "svc_fork"

def test_fork_hq_mix_is_true_peak_limited_after_summing(tmp_path, monkeypatch):
    """The instrumental + converted vocal can overshoot between samples.

    Use a phase-offset 12 kHz tone: the sample peak is clamped to -0.45 dBFS
    by the existing 0.95 guard, but the reconstructed peak is above -1 dBTP.
    The delivered mix must be trimmed after summing, not merely the vocal.
    """
    from auto_voice.audio.true_peak import true_peak_dbfs

    sr = 44100
    n = sr
    t = np.arange(n, dtype=np.float32) / sr
    wave = np.sin(2 * np.pi * 12000 * t + 0.37).astype(np.float32)
    song = tmp_path / "inter_sample_overs.wav"
    sf.write(str(song), np.stack([wave * 0.5, wave * 0.5], axis=1), sr)

    p = _pipeline(tmp_path)
    p.config.update({
        "fork_hq_match_lead_level": False,
        "fork_hq_hum_suppress": False,
        "fork_hq_subharmonic_suppress": False,
        "fork_hq_voice_polish": False,
        "fork_hq_hf_detonalize": False,
        "fork_hq_air_cap_db": 0.0,
        "fork_hq_extend_bandwidth": False,
        "fork_hq_match_source_bandwidth": False,
        "fork_hq_inharmonic_notch": False,
    })

    class FakeSep:
        def separate(self, audio, sr, mono=True):
            vocals = np.stack([wave * 1.0, wave * 1.0])
            instrumental = np.stack([wave * 0.6, wave * 0.6])
            return {"vocals": vocals, "instrumental": instrumental}

    class FakeMM:
        def infer(self, audio, spk, emb, sr):
            return np.asarray(audio, dtype=np.float32)

    monkeypatch.setattr(p, "_get_separator", lambda: FakeSep())
    monkeypatch.setattr(p, "_get_model_manager", lambda: FakeMM())
    monkeypatch.setattr(p, "_extract_pitch", lambda audio, sr: np.zeros(8, np.float32))
    monkeypatch.setattr(
        p, "_fill_lead_dropouts", lambda converted, source, sr: converted)

    result = p._convert_song_fork_hq(
        str(song), "prof-1", 1.0, 1.0, True, "balanced", 0.0)
    mixed = result["mixed_audio"]

    assert float(np.abs(mixed).max()) <= 0.95 + 1e-6
    assert true_peak_dbfs(mixed) <= -1.0 + 0.01
    assert result["metadata"]["vocal_trim_db"] < 0.0
    assert result["metadata"]["mix_trim_db"] < 0.0
    assert result["metadata"]["mix_sample_peak_trim_db"] < 0.0
    assert np.allclose(
        mixed[:, 0],
        result["stems"]["vocals"] + result["stems"]["instrumental"],
        atol=1e-6,
    )



def test_convert_song_delegates_to_fork_lane_when_available(tmp_path, monkeypatch):
    song = tmp_path / "s.wav"; song.write_bytes(b"x")   # file exists; delegation precedes load
    p = _pipeline(tmp_path)
    sentinel = {"mixed_audio": np.zeros((4, 2), np.float32), "sample_rate": 44100}
    monkeypatch.setattr(svc_fork_bridge, "is_available", lambda pid, dd: True)
    called = {}
    # The lane now also receives the per-request quality-preset overrides.
    def fake_lane(song_path, profile, vv, iv, rs, preset, ps, **kwargs):
        called["hit"] = (song_path, profile); called["kwargs"] = kwargs; return sentinel
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
