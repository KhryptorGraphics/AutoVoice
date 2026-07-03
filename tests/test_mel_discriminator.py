"""Tests for the adversarial mel discriminator (fixes L1 over-smoothing)."""
import torch
import torch.nn.functional as F

from auto_voice.models.mel_discriminator import (
    MultiScaleMelDiscriminator, discriminator_loss,
    generator_adv_loss, feature_matching_loss,
)


def test_discriminator_shapes_and_losses():
    D = MultiScaleMelDiscriminator()
    real = torch.rand(2, 80, 128) * 0.5 + 0.25
    fake = torch.rand(2, 80, 128) * 0.5 + 0.25
    ro, rf = D(real)
    fo, ff = D(fake)
    assert len(ro) == 3  # three time-resolution scales
    assert torch.isfinite(discriminator_loss(ro, fo))
    assert torch.isfinite(generator_adv_loss(fo))
    assert torch.isfinite(feature_matching_loss(rf, ff))


def test_adversarial_loop_pushes_variance_up_not_down():
    """The whole point: adversarial training must NOT collapse variance the way
    L1 does. A tiny generator trained adversarially should reach >=0.8x target
    variance (L1 alone stalls near 0.5x)."""
    torch.manual_seed(0)
    D = MultiScaleMelDiscriminator()
    gen = torch.nn.Sequential(torch.nn.Conv1d(80, 80, 1))
    optd = torch.optim.AdamW(D.parameters(), lr=1e-3)
    optg = torch.optim.AdamW(gen.parameters(), lr=1e-3)
    T = 128
    tgt = (0.5 + 0.15 * torch.sin(torch.linspace(0, 6, T)[None, None]
           * (1 + torch.arange(80)[None, :, None] * 0.3))).expand(2, 80, T).contiguous()
    inp = torch.randn(2, 80, T)
    for _ in range(200):
        g = gen(inp)
        optd.zero_grad(); ro, rf = D(tgt); fo, _ = D(g.detach())
        discriminator_loss(ro, fo).backward(); optd.step()
        optg.zero_grad(); fo2, ff2 = D(g)
        (45 * F.l1_loss(g, tgt) + generator_adv_loss(fo2)
         + 2 * feature_matching_loss(rf, ff2)).backward(); optg.step()
    ratio = gen(inp).std().item() / tgt.std().item()
    assert ratio >= 0.8, f"adversarial variance ratio {ratio:.2f} < 0.8"
