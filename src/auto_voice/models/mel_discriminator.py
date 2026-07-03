"""Multi-scale mel-spectrogram discriminator for adversarial decoder training.

L1-on-mel regression converges to the conditional mean of the one-to-many
content+F0+speaker -> mel mapping and over-smooths (muffled output). A
discriminator that judges whole mel patches as real/fake pushes the generator
to produce the sharp harmonic structure L1 averages away (the VITS/So-VITS-SVC
recipe applied at the mel level; Multi-SpectroGAN AAAI 2021, UnivNet 2106.07889).

Training-only: the trained generator (the existing decoder) serves unchanged.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MelScaleDiscriminator(nn.Module):
    """One 2D-conv discriminator over a (down-pooled) mel spectrogram."""

    def __init__(self, pool: int = 1):
        super().__init__()
        self.pool = pool
        ch = [1, 32, 64, 128, 128]
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.utils.weight_norm(nn.Conv2d(
                ch[i], ch[i + 1], kernel_size=(3, 9),
                stride=(1, 2) if i < len(ch) - 2 else (1, 1),
                padding=(1, 4))))
        self.convs = nn.ModuleList(layers)
        self.post = nn.utils.weight_norm(nn.Conv2d(ch[-1], 1, (3, 3), padding=(1, 1)))

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # mel: [B, n_mels, T] -> [B, 1, n_mels, T]
        x = mel.unsqueeze(1)
        if self.pool > 1:
            x = F.avg_pool2d(x, (1, self.pool))
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x, feats


class MultiScaleMelDiscriminator(nn.Module):
    """Judges the mel at several time resolutions (coarse=prosody, fine=detail)."""

    def __init__(self, pools: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.discs = nn.ModuleList([_MelScaleDiscriminator(pool=p) for p in pools])

    def forward(self, mel: torch.Tensor):
        outs, feats = [], []
        for d in self.discs:
            o, f = d(mel)
            outs.append(o)
            feats.append(f)
        return outs, feats


# ── losses (hinge GAN + feature matching) ────────────────────────────────────
def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss = loss + torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
    return loss / len(real_outs)


def generator_adv_loss(fake_outs) -> torch.Tensor:
    loss = 0.0
    for f in fake_outs:
        loss = loss + torch.mean(F.relu(1.0 - f))
    return loss / len(fake_outs)


def feature_matching_loss(real_feats, fake_feats) -> torch.Tensor:
    loss = 0.0
    n = 0
    for rf_list, ff_list in zip(real_feats, fake_feats):
        for rf, ff in zip(rf_list, ff_list):
            loss = loss + F.l1_loss(ff, rf.detach())
            n += 1
    return loss / max(n, 1)
