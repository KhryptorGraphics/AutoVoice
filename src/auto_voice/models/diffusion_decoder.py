"""Generative (EDM-diffusion) mel decoder for voice conversion.

Replaces L1-on-mel regression, which provably converges to the conditional
mean of the one-to-many (content+F0+speaker -> mel) mapping and produces
over-smoothed, muffled output. This decoder instead SAMPLES from
p(mel | content, F0, speaker) via an EDM diffusion model (Karras et al.,
arXiv:2206.00364), the teacher formulation of CoMoSVC (arXiv:2401.01792),
so it reproduces the full harmonic detail (variance) of real vocals.

Reuses the EDM building blocks in ``consistency.py`` (DiffusionDecoder
denoiser, Karras schedule) and adds the two missing pieces: a conditioning
adapter (content+pitch+speaker -> cond) and a multi-step Heun ODE sampler.

External contract is unchanged from the regression decoder: ``infer`` returns
[B, n_mels, T] in the pipeline's [0,1]-normalized log-mel space, which serving
denormalizes and feeds to the HiFiGAN vocoder.
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .consistency import DiffusionDecoder, KarrasNoiseSchedule


class DiffusionMelDecoder(nn.Module):
    """EDM-diffusion mel decoder conditioned on content + pitch + speaker."""

    ARCHITECTURE = "diffusion_mel"

    def __init__(
        self,
        content_dim: int = 768,
        pitch_dim: int = 768,
        speaker_dim: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
        n_blocks: int = 20,
        cond_dim: int = 256,
        sigma_data: float = 0.5,
        mel_mean: float = 0.5,
        # NOT the true mel std (~0.15). This is the scale that maps the [0,1]
        # mel to std ~= sigma_data (0.5) so EDM's noise schedule and Karras's
        # P_mean/P_std defaults are centred correctly; a mismatch makes
        # training ignore the mid/high-noise regime and sampling collapse to a
        # flat mel (0.148 / 0.30 = 0.49 ~= sigma_data).
        mel_std: float = 0.30,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_steps: int = 16,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        device=None,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.pitch_dim = pitch_dim
        self.speaker_dim = speaker_dim
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.cond_dim = cond_dim
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        self.sampler_steps = sampler_steps

        # Standardize the [0,1] mel to ~zero-mean/unit-var so EDM preconditioning
        # (which assumes data ~ N(0, sigma_data^2)) behaves. Stored as buffers so
        # they persist in the checkpoint and follow .to(device).
        self.register_buffer("mel_mean", torch.tensor(float(mel_mean)))
        self.register_buffer("mel_std", torch.tensor(float(mel_std)))

        # Conditioning adapter: fuse content+pitch+speaker -> cond_dim.
        self.cond_proj = nn.Conv1d(content_dim + pitch_dim + speaker_dim, cond_dim, 1)

        # EDM denoiser (reused, tested) + sampling schedule.
        self.denoiser = DiffusionDecoder(
            n_mels=n_mels, hidden_dim=hidden_dim, n_blocks=n_blocks,
            cond_dim=cond_dim, sigma_data=sigma_data,
        )
        self.schedule = KarrasNoiseSchedule(sigma_min=sigma_min, sigma_max=sigma_max)

        self._lora_injected = False
        if device is not None:
            self.to(device)

    # ── config / checkpoint ──────────────────────────────────────────────
    def get_config(self) -> Dict[str, Any]:
        """Dims needed to rebuild this decoder from a saved state dict."""
        return {
            "content_dim": self.content_dim,
            "pitch_dim": self.pitch_dim,
            "speaker_dim": self.speaker_dim,
            "n_mels": self.n_mels,
            "hidden_dim": self.hidden_dim,
            "n_blocks": self.n_blocks,
            "cond_dim": self.cond_dim,
            "sigma_data": self.sigma_data,
            "mel_mean": float(self.mel_mean.item()),
            "mel_std": float(self.mel_std.item()),
            "p_mean": self.p_mean,
            "p_std": self.p_std,
            "sampler_steps": self.sampler_steps,
        }

    # ── conditioning ─────────────────────────────────────────────────────
    def _build_cond(self, content: torch.Tensor, pitch: torch.Tensor,
                    speaker: torch.Tensor) -> torch.Tensor:
        """content [B,T,Dc], pitch [B,T,Dp], speaker [B,Ds] -> cond [B,cond_dim,T]."""
        content = content.transpose(1, 2)  # [B, Dc, T]
        pitch = pitch.transpose(1, 2)       # [B, Dp, T]
        T = content.shape[-1]
        if pitch.shape[-1] != T:
            pitch = F.interpolate(pitch, size=T, mode="linear", align_corners=False)
        if speaker.dim() == 2:
            speaker = speaker.unsqueeze(-1)  # [B, Ds, 1]
        speaker = speaker.expand(-1, -1, T)  # [B, Ds, T]
        fused = torch.cat([content, pitch, speaker], dim=1)
        return self.cond_proj(fused)

    # ── training ─────────────────────────────────────────────────────────
    def forward(self, content: torch.Tensor, pitch: torch.Tensor,
                speaker: torch.Tensor, spec: torch.Tensor = None) -> torch.Tensor:
        """Return the conditioning; the EDM loss is applied in compute_loss.

        (spec is accepted for trainer-call compatibility and ignored — the
        diffusion decoder has no VITS posterior branch.)
        """
        return self._build_cond(content, pitch, speaker)

    def _edm_loss(self, x: torch.Tensor, cond: torch.Tensor,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """One EDM denoising-score loss draw on standardized mel x."""
        b = x.shape[0]
        rnd = torch.randn(b, device=x.device, generator=generator)
        sigma = (rnd * self.p_std + self.p_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn(x.shape, device=x.device, generator=generator) * sigma.view(-1, 1, 1)
        denoised = self.denoiser(x + noise, sigma, cond)
        return (weight.view(-1, 1, 1) * (denoised - x) ** 2).mean()

    def compute_loss(self, outputs: torch.Tensor, target_mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """EDM loss. `outputs` is the cond from forward(); `target_mel` is the
        [0,1] mel [B, n_mels, T]. Returns the keys the trainer logs."""
        cond = outputs
        x = (target_mel - self.mel_mean) / self.mel_std  # standardize
        if self.training:
            loss = self._edm_loss(x, cond)
        else:
            # Deterministic, low-variance validation loss so early stopping is
            # not fooled by the stochastic training objective's noise.
            gen = torch.Generator(device=x.device).manual_seed(1234)
            loss = sum(self._edm_loss(x, cond, generator=gen) for _ in range(8)) / 8.0
        return {"total_loss": loss, "reconstruction_loss": loss.detach()}

    # ── inference ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _heun_sample(self, cond: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Deterministic 2nd-order (Heun) EDM sampler over the Karras schedule.

        Returns standardized mel [B, n_mels, T]."""
        b, _, t = cond.shape
        device = cond.device
        sigmas = self.schedule.get_sigmas(n_steps, device=device)  # [n_steps+1], last 0
        x = torch.randn(b, self.n_mels, t, device=device) * sigmas[0]
        for i in range(n_steps):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            denoised = self.denoiser(x, s_cur.expand(b), cond)
            d = (x - denoised) / s_cur
            x_next = x + (s_next - s_cur) * d
            if s_next > 0:  # Heun correction
                denoised_next = self.denoiser(x_next, s_next.expand(b), cond)
                d_next = (x_next - denoised_next) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d_next)
            x = x_next
        return x

    @torch.no_grad()
    def infer(self, content: torch.Tensor, pitch: torch.Tensor,
              speaker: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """Generate [B, n_mels, T] mel in [0,1] log-mel space via diffusion."""
        cond = self._build_cond(content, pitch, speaker)
        x_std = self._heun_sample(cond, int(n_steps or self.sampler_steps))
        return x_std * self.mel_std + self.mel_mean
