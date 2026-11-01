"""Behavioral invariants for the optional SPA cycle objective helpers."""
from __future__ import annotations

import importlib

import pytest


def test_shift_scales_the_whole_contour_and_respects_the_ceiling():
    torch = pytest.importorskip("torch")
    sample_shifted_f0 = importlib.import_module(
        "patches.svcfork_spa_cycle_runtime"
    ).sample_shifted_f0
    f0 = torch.tensor(
        [
            [250.0, 500.0, 600.0, 400.0],
            [1000.0, 850.0, 900.0, 800.0],
            [1100.0, 900.0, 700.0, 500.0],
        ]
    )
    uv = torch.tensor(
        [
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )

    shifted, eligible, semitones = sample_shifted_f0(f0, uv)

    assert eligible.tolist() == [True, True, False]
    assert 6 <= semitones[0].item() <= 18
    assert semitones[1].item() == 1
    assert semitones[2].item() == 0
    assert torch.equal(shifted[2], f0[2]), (
        "a sample at the F0 ceiling must be skipped"
    )
    assert torch.all(shifted <= 1100.0)
    # The WHOLE interpolated contour moves by ONE uniform factor per sample:
    # shifting only the voiced frames would put a step of up to 2.8x at every
    # voiced/unvoiced boundary, and the pitch embedding would read that
    # sawtooth as the singer's contour.
    for sample_index in (0, 1):
        ratios = 12.0 * torch.log2(
            shifted[sample_index] / f0[sample_index]
        )
        expected = torch.full_like(ratios, float(semitones[sample_index]))
        assert torch.allclose(ratios, expected, atol=1e-3), (
            "every frame of the contour must shift by the same amount"
        )
    # The unvoiced frame is part of the contour, so it moves with it.
    assert shifted[0, 2].item() != f0[0, 2].item(), (
        "the interpolated unvoiced frame must shift with the contour"
    )


def test_contentvec_feature_extraction_keeps_waveform_gradient():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    contentvec_features = importlib.import_module(
        "patches.svcfork_spa_cycle_runtime"
    ).contentvec_features

    class DifferentiableEncoder(torch.nn.Module):
        def forward(self, audio, output_hidden_states=False):
            latent = audio[:, ::320].unsqueeze(-1).expand(-1, -1, 8)
            if output_hidden_states:
                return {"hidden_states": tuple(latent for _ in range(10))}
            return {"last_hidden_state": latent}

    audio = torch.linspace(-0.8, 0.9, 4410).unsqueeze(0).requires_grad_()
    features = contentvec_features(
        DifferentiableEncoder(),
        audio,
        sample_rate=44_100,
        legacy_final_proj=False,
    )
    features.square().mean().backward()

    assert features.ndim == 3
    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
    assert audio.grad.abs().sum().item() > 0
