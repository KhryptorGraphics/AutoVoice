"""Tests for the generative EDM-diffusion mel decoder.

The decoder replaces L1-mel regression, which over-smooths (produces ~half the
target mel variance -> muffled audio). The load-bearing test is that after a
short overfit the SAMPLED mel recovers the target's variance (>=0.8x), i.e. it
does NOT collapse to the conditional mean the way L1 does.
"""
import tempfile
import os

import numpy as np
import pytest
import torch

from auto_voice.models.diffusion_decoder import DiffusionMelDecoder


def _structured_mel(B, T, device):
    """A [0,1] mel with real harmonic variance to measure smoothing against."""
    tt = torch.linspace(0, 6, T, device=device)
    bins = torch.arange(80, device=device)
    mel = 0.5 + 0.15 * torch.sin(tt[None, None] * (1 + bins[None, :, None] * 0.3))
    return mel.expand(B, 80, T).contiguous()


def test_build_forward_and_loss_keys():
    m = DiffusionMelDecoder(hidden_dim=96, n_blocks=6)
    content = torch.randn(2, 32, 768)
    pitch = torch.randn(2, 32, 768)
    speaker = torch.randn(2, 256)
    cond = m(content, pitch, speaker, spec=None)
    assert cond.shape == (2, m.cond_dim, 32)
    losses = m.compute_loss(cond, _structured_mel(2, 32, cond.device))
    assert set(losses) >= {"total_loss", "reconstruction_loss"}
    assert torch.isfinite(losses["total_loss"])


def test_infer_shape_and_range():
    m = DiffusionMelDecoder(hidden_dim=96, n_blocks=6).eval()
    content = torch.randn(1, 40, 768)
    pitch = torch.randn(1, 40, 768)
    speaker = torch.randn(1, 256)
    mel = m.infer(content, pitch, speaker, n_steps=8)
    assert mel.shape == (1, 80, 40)
    assert torch.isfinite(mel).all()
    # denormalized downstream; should sit broadly in the [0,1] training space
    assert -0.5 < mel.mean().item() < 1.5


@pytest.mark.slow
def test_overfit_recovers_mel_variance_not_smoothed():
    """The anti-smoothing guarantee: a fit diffusion decoder samples a mel with
    variance close to the target, unlike L1 regression (which halves it)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = DiffusionMelDecoder(hidden_dim=160, n_blocks=10, device=device)
    B, T = 2, 64
    content = torch.randn(B, T, 768, device=device)
    pitch = torch.randn(B, T, 768, device=device)
    speaker = torch.randn(B, 256, device=device)
    target = _structured_mel(B, T, device)

    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    m.train()
    for _ in range(300):
        opt.zero_grad()
        loss = m.compute_loss(m(content, pitch, speaker), target)["total_loss"]
        loss.backward()
        opt.step()

    m.eval()
    mel = m.infer(content, pitch, speaker, n_steps=16)
    ratio = mel.std().item() / target.std().item()
    assert ratio >= 0.8, f"sampled mel variance ratio {ratio:.2f} < 0.8 (over-smoothed)"


def test_validation_loss_is_deterministic():
    """Eval-mode loss must be reproducible so early stopping isn't fooled by the
    stochastic training objective."""
    m = DiffusionMelDecoder(hidden_dim=96, n_blocks=6).eval()
    content = torch.randn(1, 32, 768)
    pitch = torch.randn(1, 32, 768)
    speaker = torch.randn(1, 256)
    mel = _structured_mel(1, 32, content.device)
    cond = m(content, pitch, speaker)
    a = m.compute_loss(cond, mel)["total_loss"].item()
    b = m.compute_loss(cond, mel)["total_loss"].item()
    assert a == pytest.approx(b), "eval loss should be deterministic"


def test_checkpoint_tag_roundtrip_rebuilds_identical():
    """Tagged artifact rebuilds via the shared loader without key-sniffing."""
    from auto_voice.inference.model_manager import build_voice_model_from_checkpoint

    m = DiffusionMelDecoder(hidden_dim=128, n_blocks=8)
    ck = {
        "architecture": "diffusion_mel",
        "model_state_dict": m.state_dict(),
        "config": m.get_config(),
    }
    path = tempfile.mktemp(suffix=".pt")
    torch.save(ck, path)
    try:
        loaded = build_voice_model_from_checkpoint(
            torch.load(path, map_location="cpu", weights_only=False), path, "cpu"
        )
        assert type(loaded).__name__ == "DiffusionMelDecoder"
        assert loaded.hidden_dim == 128 and loaded.n_blocks == 8
        assert all(
            torch.equal(a, b)
            for a, b in zip(m.state_dict().values(), loaded.state_dict().values())
        )
    finally:
        os.unlink(path)


def test_default_architecture_is_diffusion():
    from auto_voice.training.job_manager import TrainingConfig
    assert TrainingConfig().architecture == "diffusion_mel"
