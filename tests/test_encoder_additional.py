"""Focused branch coverage for encoder module edge cases."""
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
import builtins

import pytest
import torch

from auto_voice.models import encoder as encoder_module
from auto_voice.models.encoder import (
    ContentEncoder,
    ContentVecEncoder,
    HuBERTSoft,
    PitchEncoder,
    _ensure_optional_module_spec,
    f0_to_coarse,
)


class DummyHubert:
    """Lightweight HuBERT stub for load path tests."""

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def encode(self, audio):
        batch = audio.shape[0]
        return torch.zeros(batch, 2, 256, device=audio.device)


class FakeHubertConfig:
    """Small config stand-in for transformers.HubertConfig."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def make_fake_transformers(from_pretrained_side_effect=None):
    """Build a fake transformers module for ContentVec tests."""
    module = ModuleType("transformers")
    module.__spec__ = ModuleSpec("transformers", loader=None)

    class FakeHubertModel:
        from_pretrained_calls = []
        constructed_configs = []

        def __init__(self, config=None):
            self.config = config
            self.device = None
            self.eval_called = False
            FakeHubertModel.constructed_configs.append(config)

        @classmethod
        def from_pretrained(cls, pretrained_id, local_files_only=False):
            cls.from_pretrained_calls.append((pretrained_id, local_files_only))
            if from_pretrained_side_effect is not None:
                raise from_pretrained_side_effect
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def __call__(self, audio, output_hidden_states=False):
            batch = audio.shape[0]
            frames = 4
            hidden_states = [
                torch.full((batch, frames, 768), float(i), device=audio.device)
                for i in range(13)
            ]
            return SimpleNamespace(hidden_states=hidden_states)

    module.HubertModel = FakeHubertModel
    module.HubertConfig = FakeHubertConfig
    return module, FakeHubertModel


def test_ensure_optional_module_spec_sets_missing_spec(monkeypatch):
    stub = SimpleNamespace()
    monkeypatch.setitem(encoder_module.sys.modules, "tensorrt", stub)

    _ensure_optional_module_spec("tensorrt")

    assert isinstance(stub.__spec__, ModuleSpec)
    assert stub.__spec__.name == "tensorrt"


def test_content_encoder_load_hubert_returns_early_when_already_loaded(monkeypatch):
    encoder = ContentEncoder()
    encoder._hubert_loaded = True

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HuBERTSoft should not be reinitialized")

    monkeypatch.setattr(encoder_module, "HuBERTSoft", fail_if_called)
    encoder._load_hubert()


def test_content_encoder_load_hubert_uses_checkpoint_when_present(monkeypatch, tmp_path):
    checkpoint = tmp_path / "hubert.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(encoder_module, "HuBERTSoft", DummyHubert)

    encoder = ContentEncoder(device=torch.device("cpu"))
    encoder._load_hubert(str(checkpoint))

    assert isinstance(encoder._hubert, DummyHubert)
    assert encoder._hubert.checkpoint_path == str(checkpoint)
    assert encoder._hubert.eval_called is True
    assert encoder._hubert_loaded is True


def test_content_encoder_extract_features_raises_when_hubert_failed_to_init():
    encoder = ContentEncoder()
    encoder._hubert = None
    encoder._hubert_loaded = True

    with pytest.raises(RuntimeError, match="failed to initialize"):
        encoder.extract_features(torch.randn(1, 16000))


def test_f0_to_coarse_accepts_three_dimensional_input():
    f0 = torch.tensor([[[55.0], [220.0], [440.0]]])
    bins = f0_to_coarse(f0)
    assert bins.shape == (1, 3)
    assert bins.dtype == torch.long


def test_pitch_encoder_load_pretrained_existing_checkpoint(tmp_path):
    source = PitchEncoder()
    with torch.no_grad():
        source.residual_proj.weight.fill_(2.5)
    checkpoint = tmp_path / "pitch.pt"
    torch.save(source.state_dict(), checkpoint)

    loaded = PitchEncoder.load_pretrained(str(checkpoint), device=torch.device("cpu"))

    assert torch.allclose(loaded.residual_proj.weight, source.residual_proj.weight)


def test_pitch_encoder_load_pretrained_missing_checkpoint():
    loaded = PitchEncoder.load_pretrained("/nonexistent/pitch.pt", device=torch.device("cpu"))
    assert isinstance(loaded, PitchEncoder)


def test_hubert_soft_load_checkpoint_success(tmp_path):
    checkpoint = tmp_path / "hubert-soft.pt"
    torch.save({"weights": 1}, checkpoint)

    model = HuBERTSoft(checkpoint_path=str(checkpoint))

    assert model._loaded is True


def test_hubert_soft_load_checkpoint_warning(monkeypatch, tmp_path, caplog):
    checkpoint = tmp_path / "bad-hubert-soft.pt"
    checkpoint.write_bytes(b"bad")

    def raise_load_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(encoder_module.torch, "load", raise_load_error)

    with caplog.at_level("WARNING"):
        model = HuBERTSoft(checkpoint_path=str(checkpoint))

    assert model._loaded is False
    assert "Could not load HuBERT checkpoint" in caplog.text


def test_contentvec_load_model_returns_early_when_already_loaded():
    encoder = ContentVecEncoder(pretrained=None)
    sentinel_model = object()
    encoder._model = sentinel_model
    encoder._loaded = True

    encoder._load_model()

    assert encoder._model is sentinel_model


def test_contentvec_load_model_from_local_path(monkeypatch, tmp_path):
    transformers_module, fake_model_cls = make_fake_transformers()
    monkeypatch.setitem(encoder_module.sys.modules, "transformers", transformers_module)

    encoder = ContentVecEncoder(pretrained=str(tmp_path), device=torch.device("cpu"))
    encoder._load_model()

    assert fake_model_cls.from_pretrained_calls == [(str(tmp_path), True)]
    assert encoder._loaded is True


def test_contentvec_load_model_falls_back_to_random_weights(monkeypatch):
    transformers_module, fake_model_cls = make_fake_transformers(
        from_pretrained_side_effect=RuntimeError("network unavailable")
    )
    monkeypatch.setitem(encoder_module.sys.modules, "transformers", transformers_module)

    encoder = ContentVecEncoder(pretrained="lengyue233/content-vec-best", device=torch.device("cpu"))
    encoder._load_model()

    assert fake_model_cls.from_pretrained_calls == [("lengyue233/content-vec-best", False)]
    assert isinstance(fake_model_cls.constructed_configs[-1], FakeHubertConfig)
    assert encoder._loaded is True


def test_contentvec_load_model_raises_runtime_error_when_transformers_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers":
            raise ImportError("missing transformers")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(encoder_module.sys.modules, "transformers", raising=False)

    encoder = ContentVecEncoder(pretrained="missing", device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="transformers package required"):
        encoder._load_model()
