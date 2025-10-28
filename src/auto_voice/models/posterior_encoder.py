"""
Posterior Encoder for Singing Voice Conversion

Implements WaveNet-style residual blocks for encoding ground-truth audio
to latent distribution during training.
"""

from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WaveNetResidualBlock(nn.Module):
    """
    Single WaveNet residual block with gated activation and skip connections.

    Args:
        residual_channels: Number of residual channels
        skip_channels: Number of skip connection channels
        kernel_size: Convolution kernel size (default: 5)
        dilation: Dilation rate for dilated convolution (default: 1)
        cond_channels: Conditioning channels (default: 0)
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        cond_channels: int = 0
    ):
        super().__init__()

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        padding = (kernel_size * dilation - dilation) // 2

        # Dilated convolution (outputs 2x channels for filter and gate)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            dilation=dilation,
            padding=padding
        )
        # Apply weight normalization
        self.dilated_conv = nn.utils.weight_norm(self.dilated_conv)

        # Conditioning projection if needed
        if cond_channels > 0:
            self.cond_proj = nn.Conv1d(cond_channels, 2 * residual_channels, 1)
        else:
            self.cond_proj = None

        # Residual output
        self.res_out = nn.Conv1d(residual_channels, residual_channels, 1)

        # Skip output
        self.skip_out = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through WaveNet residual block.

        Args:
            x: Input [B, C_res, T]
            x_mask: Mask [B, 1, T]
            cond: Optional conditioning [B, C_cond, T]

        Returns:
            (residual, skip) tuple
        """
        # Apply dilated convolution
        h = self.dilated_conv(x * x_mask)

        # Add conditioning
        if cond is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond)

        # Split into filter and gate
        a, b = torch.chunk(h, 2, dim=1)

        # Gated activation: tanh(a) * sigmoid(b)
        z = torch.tanh(a) * torch.sigmoid(b)

        # Skip connection
        skip = self.skip_out(z) * x_mask

        # Residual
        residual = (self.res_out(z) + x) * x_mask

        return residual, skip


class PosteriorEncoder(nn.Module):
    """
    Encode ground-truth mel-spectrogram to latent distribution q(z|x).

    Uses WaveNet-style residual blocks for variational inference.

    Args:
        in_channels: Input mel channels (default: 80)
        out_channels: Latent dimension (default: 192)
        hidden_channels: Residual/skip channels (default: 192)
        kernel_size: Convolution kernel size (default: 5)
        num_layers: Number of WaveNet layers (default: 16)
        cond_channels: Speaker conditioning channels (default: 0)

    Example:
        >>> posterior_enc = PosteriorEncoder(in_channels=80, out_channels=192, num_layers=16)
        >>> mel = torch.randn(2, 80, 100)  # [B, mel_channels, T]
        >>> mask = torch.ones(2, 1, 100)   # [B, 1, T]
        >>> mean, log_var = posterior_enc(mel, mask)
        >>> print(mean.shape, log_var.shape)  # [2, 192, 100], [2, 192, 100]
    """

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        num_layers: int = 16,
        cond_channels: int = 0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        # WaveNet residual blocks
        self.blocks = nn.ModuleList([
            WaveNetResidualBlock(
                residual_channels=hidden_channels,
                skip_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=2 ** (i % 10),  # Cycle dilations: 1,2,4,...,512,1,2,...
                cond_channels=cond_channels
            )
            for i in range(num_layers)
        ])

        # Output projection to mean and log-variance
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2 * out_channels, 1)  # 2x for mean and log_var
        )

        # Initialize output projection weights to zero for stable training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        logger.info(f"PosteriorEncoder initialized: {in_channels}→{out_channels}, {num_layers} layers")

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode mel-spectrogram to latent distribution parameters.

        Args:
            x: Mel-spectrogram [B, mel_channels, T]
            x_mask: Mask [B, 1, T]
            cond: Optional conditioning [B, cond_channels, T]

        Returns:
            (mean, log_var) both [B, out_channels, T]
        """
        # Project input
        x = self.input_proj(x * x_mask)

        # Initialize skip sum
        skip_sum = torch.zeros_like(x)

        # Pass through WaveNet blocks
        for block in self.blocks:
            x, skip = block(x, x_mask, cond)
            skip_sum = skip_sum + skip

        # Project skip sum to mean and log-variance
        stats = self.output_proj(skip_sum * x_mask)

        # Split into mean and log_var
        mean, log_var = torch.chunk(stats, 2, dim=1)

        return mean, log_var

    def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            mean: Mean [B, out_channels, T]
            log_var: Log-variance [B, out_channels, T]

        Returns:
            Sampled latent z [B, out_channels, T]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        return z

    def remove_weight_norm(self):
        """Remove weight normalization from all conv layers."""
        for block in self.blocks:
            nn.utils.remove_weight_norm(block.dilated_conv)
        logger.info("Weight normalization removed from PosteriorEncoder")
