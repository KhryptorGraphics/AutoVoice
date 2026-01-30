"""Consistency distillation for fast 1-step inference.

Implements CoMoSVC-style two-stage training:
  Stage 1: Diffusion teacher (BiDilConv decoder with EDM preconditioning)
  Stage 2: Consistency student distilled from teacher for 1-step inference

Reference: Lu et al., "CoMoSVC" (arXiv:2401.01792)
Reference: Karras et al., "EDM" (arXiv:2206.00364)
"""
import copy
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# BiDilConv Residual Block (Non-causal WaveNet)
# ─────────────────────────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Bidirectional dilated convolution residual block with gated activation.

    Non-causal (sees past + future context), suitable for offline/batch SVC.
    Conditioned on diffusion noise level via learned projection.
    """

    def __init__(self, hidden_dim: int, n_mels: int, dilation: int,
                 kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # Non-causal (symmetric)

        self.dilated_conv = nn.Conv1d(
            hidden_dim, 2 * hidden_dim,
            kernel_size=kernel_size, dilation=dilation, padding=padding,
        )
        self.diffusion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.conditioner_proj = nn.Conv1d(n_mels, 2 * hidden_dim, 1)
        self.output_proj = nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)

    def forward(self, x: torch.Tensor, diffusion_step: torch.Tensor,
                conditioner: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, hidden_dim, T] current hidden state
            diffusion_step: [B, hidden_dim] noise level embedding
            conditioner: [B, n_mels, T] conditioning (coarse mel)

        Returns:
            Tuple of (residual output [B, hidden_dim, T], skip [B, hidden_dim, T])
        """
        h = x + self.diffusion_proj(diffusion_step).unsqueeze(-1)
        h = self.dilated_conv(h)
        h = h + self.conditioner_proj(conditioner)

        # Gated activation: sigmoid(gate) * tanh(filter)
        gate, filt = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filt)

        h = self.output_proj(h)
        residual, skip = h.chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Step Embedding
# ─────────────────────────────────────────────────────────────────────────────


class DiffusionStepEmbedding(nn.Module):
    """Sinusoidal embedding for noise level (sigma), projected to hidden_dim."""

    def __init__(self, hidden_dim: int, max_positions: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Embed noise level.

        Args:
            sigma: [B] or [B,1] noise level

        Returns:
            [B, hidden_dim] embedding
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 2:
            sigma = sigma.squeeze(-1)

        # Sinusoidal embedding from log(sigma)
        log_sigma = sigma.log().clamp(min=-10, max=10)
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device, dtype=sigma.dtype) * -emb)
        emb = log_sigma.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, hidden_dim]

        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionDecoder (BiDilConv, 20 blocks)
# ─────────────────────────────────────────────────────────────────────────────


class DiffusionDecoder(nn.Module):
    """BiDilConv diffusion decoder with EDM preconditioning.

    Architecture: 20 residual blocks across 2 dilation cycles
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] x 2 = 20 blocks.

    Uses EDM (Karras) preconditioning for stable training:
        D(x, sigma) = c_skip(sigma)*x + c_out(sigma)*F(c_in(sigma)*x; c_noise(sigma))
    """

    def __init__(self, n_mels: int = 80, hidden_dim: int = 256,
                 n_blocks: int = 20, kernel_size: int = 3,
                 dilation_cycle: int = 10, sigma_data: float = 0.5,
                 cond_dim: int = 256):
        """Initialize DiffusionDecoder.

        Args:
            n_mels: Number of mel channels (output dimension).
            hidden_dim: Hidden dimension for residual blocks.
            n_blocks: Total number of residual blocks (default 20).
            kernel_size: Convolution kernel size.
            dilation_cycle: Length of one dilation cycle (default 10).
            sigma_data: Data standard deviation for EDM preconditioning.
            cond_dim: Conditioning input dimension (content+pitch+speaker).
        """
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.sigma_data = sigma_data

        # Input projection: noisy mel -> hidden
        self.input_proj = nn.Conv1d(n_mels, hidden_dim, 1)

        # Conditioning projection: cond features -> n_mels (acts as conditioner)
        self.cond_proj = nn.Conv1d(cond_dim, n_mels, 1)

        # Noise level embedding
        self.noise_embed = DiffusionStepEmbedding(hidden_dim)

        # BiDilConv residual blocks
        dilations = [2 ** (i % dilation_cycle) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, n_mels, d, kernel_size)
            for d in dilations
        ])

        # Output projection: skip sum -> mel
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, n_mels, 1),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _raw_forward(self, x: torch.Tensor, sigma: torch.Tensor,
                     cond: torch.Tensor) -> torch.Tensor:
        """Raw network F_theta (before EDM preconditioning).

        Args:
            x: [B, n_mels, T] (already scaled by c_in)
            sigma: [B] noise level
            cond: [B, cond_dim, T] conditioning features

        Returns:
            [B, n_mels, T] predicted output
        """
        # Project conditioning to mel space
        conditioner = self.cond_proj(cond)  # [B, n_mels, T]

        # Project noisy input to hidden
        h = self.input_proj(x)  # [B, hidden_dim, T]

        # Embed noise level
        noise_emb = self.noise_embed(sigma)  # [B, hidden_dim]

        # Run through BiDilConv blocks
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, noise_emb, conditioner)
            skip_sum = skip_sum + skip

        # Output projection
        output = self.output_proj(skip_sum / math.sqrt(self.n_blocks))
        return output

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """EDM-preconditioned forward pass.

        D(x, sigma) = c_skip*x + c_out*F(c_in*x; c_noise)

        Args:
            x: [B, n_mels, T] noisy mel spectrogram
            sigma: [B] noise level (scalar per batch element)
            cond: [B, cond_dim, T] conditioning features

        Returns:
            [B, n_mels, T] denoised mel spectrogram
        """
        sigma = sigma.view(-1)  # Ensure [B]

        # EDM preconditioning coefficients
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

        # Scale input
        x_scaled = x * c_in.view(-1, 1, 1)

        # Raw network prediction
        F_x = self._raw_forward(x_scaled, sigma, cond)

        # Apply skip connection and output scaling
        D_x = c_skip.view(-1, 1, 1) * x + c_out.view(-1, 1, 1) * F_x
        return D_x


# ─────────────────────────────────────────────────────────────────────────────
# EDM Training Loss
# ─────────────────────────────────────────────────────────────────────────────


class EDMLoss(nn.Module):
    """EDM training loss with log-normal noise sampling.

    From Karras et al. (2022): samples sigma from log-normal distribution
    and applies lambda(sigma) weighting for balanced training.
    """

    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2,
                 sigma_data: float = 0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def forward(self, model: DiffusionDecoder, x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """Compute EDM training loss.

        Args:
            model: DiffusionDecoder with EDM preconditioning
            x: [B, n_mels, T] clean mel spectrogram
            cond: [B, cond_dim, T] conditioning features

        Returns:
            Scalar loss
        """
        B = x.shape[0]

        # Sample noise level from log-normal
        rnd_normal = torch.randn(B, device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # Lambda weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Add noise
        noise = torch.randn_like(x) * sigma.view(-1, 1, 1)
        x_noisy = x + noise

        # Predict clean signal
        D_x = model(x_noisy, sigma, cond)

        # Weighted MSE
        loss = weight.view(-1, 1, 1) * (D_x - x) ** 2
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Karras Noise Schedule
# ─────────────────────────────────────────────────────────────────────────────


class KarrasNoiseSchedule:
    """Karras et al. noise schedule for diffusion sampling.

    Provides sigma values for N-step sampling with optimal spacing.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 rho: float = 7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n_steps: int, device: torch.device = None) -> torch.Tensor:
        """Get sigma schedule for n_steps.

        Args:
            n_steps: Number of sampling steps
            device: Target device

        Returns:
            [n_steps + 1] tensor of sigma values (last is 0)
        """
        step_indices = torch.arange(n_steps, device=device, dtype=torch.float64)
        t = step_indices / max(n_steps - 1, 1)

        # Karras schedule: sigma_i = (sigma_max^(1/rho) + t*(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        sigmas = (
            self.sigma_max ** (1 / self.rho) +
            t * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho

        # Append sigma=0 for final step
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float64)])
        return sigmas.float()


# ─────────────────────────────────────────────────────────────────────────────
# Consistency Student (EMA teacher, single-step inference)
# ─────────────────────────────────────────────────────────────────────────────


class ConsistencyStudent(nn.Module):
    """Consistency student model distilled from diffusion teacher.

    Maintains an EMA copy of the teacher for generating consistency targets.
    At inference, produces high-quality mel spectrograms in a single step.

    Training:
        1. Sample (x_t, t) and (x_s, s) on the same ODE trajectory
        2. Student predicts f(x_t, t); EMA teacher predicts f(x_s, s)
        3. Minimize ||f_student(x_t, t) - f_ema(x_s, s)||^2

    Inference:
        Single forward pass: mel = student(noise, sigma_max, cond)
    """

    def __init__(self, n_mels: int = 80, hidden_dim: int = 256,
                 n_blocks: int = 20, cond_dim: int = 256,
                 sigma_data: float = 0.5, ema_mu: float = 0.95,
                 sigma_min: float = 0.002, sigma_max: float = 80.0):
        """Initialize ConsistencyStudent.

        Args:
            n_mels: Mel spectrogram channels.
            hidden_dim: Hidden dimension.
            n_blocks: BiDilConv blocks in decoder.
            cond_dim: Conditioning dimension.
            sigma_data: EDM data std.
            ema_mu: EMA decay rate for teacher copy.
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
        """
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.sigma_data = sigma_data
        self.ema_mu = ema_mu
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Student network (trainable)
        self.student = DiffusionDecoder(
            n_mels=n_mels, hidden_dim=hidden_dim,
            n_blocks=n_blocks, cond_dim=cond_dim,
            sigma_data=sigma_data,
        )

        # EMA teacher (frozen copy, updated via EMA)
        self.teacher = DiffusionDecoder(
            n_mels=n_mels, hidden_dim=hidden_dim,
            n_blocks=n_blocks, cond_dim=cond_dim,
            sigma_data=sigma_data,
        )
        # Initialize teacher from student
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Noise schedule
        self.schedule = KarrasNoiseSchedule(sigma_min=sigma_min, sigma_max=sigma_max)

    @torch.no_grad()
    def update_ema(self):
        """Update EMA teacher weights: theta_ema = mu*theta_ema + (1-mu)*theta_student."""
        for p_ema, p_student in zip(self.teacher.parameters(), self.student.parameters()):
            p_ema.data.mul_(self.ema_mu).add_(p_student.data, alpha=1.0 - self.ema_mu)

    def forward(self, x_clean: torch.Tensor, cond: torch.Tensor,
                n_steps: int = 20) -> Dict[str, torch.Tensor]:
        """Training forward pass: compute consistency loss targets.

        Args:
            x_clean: [B, n_mels, T] clean mel spectrogram
            cond: [B, cond_dim, T] conditioning features
            n_steps: Number of discretization steps for trajectory

        Returns:
            Dict with 'student_pred', 'teacher_pred', 'x_t', 'x_s', 'sigma_t', 'sigma_s'
        """
        B = x_clean.shape[0]
        device = x_clean.device

        # Get discretized sigma schedule
        sigmas = self.schedule.get_sigmas(n_steps, device=device)

        # Sample random adjacent indices on the schedule
        idx = torch.randint(0, n_steps - 1, (B,), device=device)
        sigma_t = sigmas[idx]      # Larger noise (further from data)
        sigma_s = sigmas[idx + 1]  # Smaller noise (closer to data)

        # Create noisy samples on the same trajectory
        noise = torch.randn_like(x_clean)
        x_t = x_clean + noise * sigma_t.view(-1, 1, 1)
        x_s = x_clean + noise * sigma_s.view(-1, 1, 1)

        # Student prediction at (x_t, sigma_t)
        student_pred = self.student(x_t, sigma_t, cond)

        # EMA teacher prediction at (x_s, sigma_s) - no grad
        with torch.no_grad():
            teacher_pred = self.teacher(x_s, sigma_s, cond)

        return {
            'student_pred': student_pred,
            'teacher_pred': teacher_pred,
            'x_t': x_t,
            'x_s': x_s,
            'sigma_t': sigma_t,
            'sigma_s': sigma_s,
        }

    @torch.no_grad()
    def infer(self, cond: torch.Tensor, n_frames: Optional[int] = None) -> torch.Tensor:
        """Single-step inference: generate mel from pure noise.

        Args:
            cond: [B, cond_dim, T] conditioning features
            n_frames: Number of output frames (default: infer from cond)

        Returns:
            [B, n_mels, T] generated mel spectrogram
        """
        B = cond.shape[0]
        T = n_frames or cond.shape[2]
        device = cond.device

        # Align conditioning to target frame count
        if cond.shape[2] != T:
            cond = F.interpolate(cond, size=T, mode='linear', align_corners=False)

        # Start from pure noise at sigma_max
        x = torch.randn(B, self.n_mels, T, device=device) * self.sigma_max
        sigma = torch.full((B,), self.sigma_max, device=device)

        # Single-step denoising
        mel = self.student(x, sigma, cond)
        return mel

    def load_teacher_weights(self, teacher_model: DiffusionDecoder):
        """Initialize student and EMA teacher from a pretrained diffusion teacher.

        Args:
            teacher_model: Pretrained DiffusionDecoder to distill from.
        """
        self.student.load_state_dict(teacher_model.state_dict())
        self.teacher.load_state_dict(teacher_model.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# CTLoss_D: Consistency Training Loss for Distillation
# ─────────────────────────────────────────────────────────────────────────────


class CTLoss_D(nn.Module):
    """Consistency Training Loss for Distillation.

    Enforces self-consistency: the student should map any point on
    a diffusion trajectory to the same clean output.

    L = ||f_student(x_t, t) - sg[f_ema(x_s, s)]||^2 + lambda_mel * L_mel

    Where:
        - (x_t, sigma_t) and (x_s, sigma_s) are adjacent points on same trajectory
        - sg[] = stop-gradient (EMA teacher is frozen)
        - L_mel = reconstruction loss on clean mel (auxiliary)
    """

    def __init__(self, lambda_mel: float = 0.1, sigma_data: float = 0.5):
        """Initialize CTLoss_D.

        Args:
            lambda_mel: Weight for mel reconstruction auxiliary loss.
            sigma_data: Data standard deviation for weighting.
        """
        super().__init__()
        self.lambda_mel = lambda_mel
        self.sigma_data = sigma_data

    def forward(self, model: ConsistencyStudent, x_clean: torch.Tensor,
                cond: torch.Tensor, n_steps: int = 20) -> Dict[str, torch.Tensor]:
        """Compute consistency distillation loss.

        Args:
            model: ConsistencyStudent model
            x_clean: [B, n_mels, T] clean mel spectrogram
            cond: [B, cond_dim, T] conditioning features
            n_steps: Discretization steps

        Returns:
            Dict with 'total_loss', 'consistency_loss', 'mel_loss'
        """
        outputs = model(x_clean, cond, n_steps=n_steps)

        student_pred = outputs['student_pred']
        teacher_pred = outputs['teacher_pred']

        # Consistency loss: student should match teacher target
        consistency_loss = F.mse_loss(student_pred, teacher_pred.detach())

        # Mel reconstruction auxiliary loss (student should produce clean mel)
        mel_loss = F.l1_loss(student_pred, x_clean)

        total_loss = consistency_loss + self.lambda_mel * mel_loss

        return {
            'total_loss': total_loss,
            'consistency_loss': consistency_loss,
            'mel_loss': mel_loss,
        }
