"""Regression tests for three fixes:

- AV-ksek: VoiceProfileStore.list_training_samples re-anchors stored absolute
  sample paths to the store's actual location (survives data-dir relocation /
  cross-machine sync).
- AV-e4p7: VocalSeparator.separate clamps a too-large segment to the model's
  max so htdemucs does not crash with a reshape error (>7.8s).
- AV-ua0w: trainer.resolve_precision downgrades bf16 to fp16 on pre-Ampere GPUs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# --------------------------------------------------------------------------- #
# AV-ksek: sample-path re-anchoring
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, seconds: float = 3.5, sr: int = 22050) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    # voiced-ish tone so analyze_training_sample passes QA gates
    audio = (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


def test_list_training_samples_reanchors_relocated_paths(tmp_path):
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    store = VoiceProfileStore(
        profiles_dir=str(tmp_path / "voice_profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained_models"),
    )
    pid = store.save({"name": "Reloc", "profile_role": "target_user"})

    src = tmp_path / "src.wav"
    _write_wav(src)
    sample = store.add_training_sample(profile_id=pid, vocals_path=str(src), duration=3.5)

    # Simulate relocation: rewrite metadata with a bogus absolute path that
    # only existed on the original machine.
    sample_dir = Path(store._samples_dir_for_profile(pid)) / sample.sample_id
    meta_path = sample_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["vocals_path"] = "/nonexistent/other-machine/data/samples/x/vocals.wav"
    meta_path.write_text(json.dumps(meta))

    # list_training_samples must re-anchor to the real local file.
    loaded = store.list_training_samples(pid)
    assert len(loaded) == 1
    vp = loaded[0].vocals_path
    assert Path(vp).exists(), f"re-anchored path should exist: {vp}"
    assert str(sample_dir) in vp, f"path should live under the store sample dir: {vp}"


# --------------------------------------------------------------------------- #
# AV-e4p7: separation segment clamp
# --------------------------------------------------------------------------- #
class _FakeDemucs:
    """Minimal stand-in for an htdemucs model (max segment 7.8s)."""
    sources = ["drums", "bass", "other", "vocals"]
    samplerate = 44100
    segment = 7.8

    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self


def _make_separator(segment):
    from auto_voice.audio.separation import VocalSeparator

    sep = VocalSeparator(device=torch.device("cpu"), segment=segment)
    model = _FakeDemucs()
    captured = {}

    def fake_apply(m, wav, **kw):
        captured.update(kw)
        # sources shape: (batch, n_sources, channels, samples)
        return torch.zeros(1, 4, wav.shape[1], wav.shape[2])

    sep._get_model = lambda _name: model
    sep._apply_model = fake_apply
    return sep, captured


def test_segment_clamped_to_model_max():
    sep, captured = _make_separator(10.0)
    audio = np.zeros(44100, dtype=np.float32)  # 1s mono
    sep.separate(audio, 44100)
    assert captured["segment"] == 7.8, captured  # clamped from 10.0


def test_segment_below_max_unchanged():
    sep, captured = _make_separator(5.0)
    audio = np.zeros(44100, dtype=np.float32)
    sep.separate(audio, 44100)
    assert captured["segment"] == 5.0, captured  # left as requested


# --------------------------------------------------------------------------- #
# AV-ua0w: bf16 precision gate
# --------------------------------------------------------------------------- #
def test_resolve_precision_downgrades_bf16_pre_ampere():
    from auto_voice.training.trainer import resolve_precision

    assert resolve_precision("bf16", (7, 0)) == "fp16"   # V100
    assert resolve_precision("bf16", (6, 0)) == "fp16"   # P100
    assert resolve_precision("bf16", (8, 0)) == "bf16"   # A100 keeps bf16
    assert resolve_precision("bf16", (8, 6)) == "bf16"   # RTX 3080 Ti
    assert resolve_precision("bf16", None) == "bf16"     # unknown -> unchanged
    assert resolve_precision("fp16", (7, 0)) == "fp16"   # unrelated precision
    assert resolve_precision("fp32", (7, 0)) == "fp32"
