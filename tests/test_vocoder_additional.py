"""Focused branch coverage for vocoder checkpoint and proxy paths."""
from types import SimpleNamespace

import pytest
import torch

from auto_voice.models import vocoder as vocoder_module
from auto_voice.models.vocoder import BigVGANVocoder, HiFiGANVocoder


class DummyGenerator:
    """Minimal generator stub for vocoder checkpoint tests."""

    def __init__(self):
        self.loaded_state = None
        self.removed_weight_norm = False
        self.to_device = None
        self.ups = ["up"]
        self.resblocks = ["resblock"]

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict

    def remove_weight_norm(self):
        self.removed_weight_norm = True

    def to(self, device):
        self.to_device = device
        return self


def test_hifigan_load_checkpoint_with_generator_key(monkeypatch, tmp_path):
    checkpoint = tmp_path / "hifigan.pt"
    checkpoint.write_bytes(b"checkpoint")
    generator = DummyGenerator()
    monkeypatch.setattr(vocoder_module.torch, "load", lambda *args, **kwargs: {"generator": {"w": 1}})

    vocoder = HiFiGANVocoder(device=torch.device("cpu"))
    vocoder._generator = generator

    assert vocoder.load_checkpoint(str(checkpoint)) is True
    assert generator.loaded_state == {"w": 1}
    assert generator.removed_weight_norm is True
    assert vocoder._loaded is True


def test_hifigan_load_checkpoint_with_raw_state_dict(monkeypatch, tmp_path):
    checkpoint = tmp_path / "hifigan-raw.pt"
    checkpoint.write_bytes(b"checkpoint")
    generator = DummyGenerator()
    monkeypatch.setattr(vocoder_module.torch, "load", lambda *args, **kwargs: {"weight": torch.tensor(1.0)})

    vocoder = HiFiGANVocoder(device=torch.device("cpu"))
    vocoder._generator = generator

    assert vocoder.load_checkpoint(str(checkpoint)) is True
    assert "weight" in generator.loaded_state


def test_hifigan_load_checkpoint_returns_false_on_error(monkeypatch, tmp_path):
    checkpoint = tmp_path / "broken-hifigan.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(vocoder_module.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    vocoder = HiFiGANVocoder(device=torch.device("cpu"))
    vocoder._generator = DummyGenerator()

    assert vocoder.load_checkpoint(str(checkpoint)) is False
    assert vocoder._loaded is False


def test_bigvgan_init_with_pretrained_calls_load_checkpoint(monkeypatch):
    calls = []

    def fake_load_checkpoint(self, checkpoint_path):
        calls.append(checkpoint_path)

    monkeypatch.setattr(BigVGANVocoder, "load_checkpoint", fake_load_checkpoint)
    BigVGANVocoder(pretrained="/tmp/bigvgan.pt")

    assert calls == ["/tmp/bigvgan.pt"]


def test_bigvgan_upsamples_and_resblocks_proxy_loaded_generator():
    vocoder = BigVGANVocoder(device=torch.device("cpu"))
    vocoder._generator = DummyGenerator()

    assert vocoder.upsamples == ["up"]
    assert vocoder.resblocks == ["resblock"]


def test_bigvgan_to_moves_loaded_generator():
    vocoder = BigVGANVocoder(device=torch.device("cpu"))
    generator = DummyGenerator()
    vocoder._generator = generator

    result = vocoder.to(torch.device("cpu"))

    assert result is vocoder
    assert generator.to_device == torch.device("cpu")


def test_bigvgan_load_checkpoint_with_state_dict_key(monkeypatch, tmp_path):
    checkpoint = tmp_path / "bigvgan-state.pt"
    checkpoint.write_bytes(b"checkpoint")
    generator = DummyGenerator()
    monkeypatch.setattr(vocoder_module.torch, "load", lambda *args, **kwargs: {"state_dict": {"w": 2}})

    vocoder = BigVGANVocoder(device=torch.device("cpu"))
    vocoder._generator = generator
    vocoder.load_checkpoint(str(checkpoint))

    assert generator.loaded_state == {"w": 2}
    assert generator.removed_weight_norm is True
    assert vocoder._loaded is True


def test_bigvgan_load_checkpoint_accepts_raw_state_dict(monkeypatch, tmp_path):
    checkpoint = tmp_path / "bigvgan-raw.pt"
    checkpoint.write_bytes(b"checkpoint")
    generator = DummyGenerator()
    monkeypatch.setattr(vocoder_module.torch, "load", lambda *args, **kwargs: {"weight": torch.tensor(3.0)})

    vocoder = BigVGANVocoder(device=torch.device("cpu"))
    vocoder._generator = generator
    vocoder.load_checkpoint(str(checkpoint))

    assert "weight" in generator.loaded_state


def test_bigvgan_load_checkpoint_rejects_unexpected_format(monkeypatch, tmp_path):
    checkpoint = tmp_path / "bigvgan-bad.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(vocoder_module.torch, "load", lambda *args, **kwargs: SimpleNamespace())

    vocoder = BigVGANVocoder(device=torch.device("cpu"))
    vocoder._generator = DummyGenerator()

    with pytest.raises(RuntimeError, match="Unexpected checkpoint format"):
        vocoder.load_checkpoint(str(checkpoint))


def test_bigvgan_load_pretrained_returns_loaded_vocoder(monkeypatch):
    calls = []

    def fake_load_checkpoint(self, checkpoint_path):
        calls.append(checkpoint_path)

    monkeypatch.setattr(BigVGANVocoder, "load_checkpoint", fake_load_checkpoint)

    vocoder = BigVGANVocoder.load_pretrained("/tmp/pretrained-bigvgan.pt", device=torch.device("cpu"))

    assert isinstance(vocoder, BigVGANVocoder)
    assert calls == ["/tmp/pretrained-bigvgan.pt"]
