"""
Flow Decoder for Singing Voice Conversion

Implements normalizing flow with affine coupling layers for latent space modeling.
"""

from typing import Optional, Union, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Flip(nn.Module):
    """Channel flip operation for flow layers."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        inverse: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flip channel order.

        Args:
            x: Input [B, C, T]
            x_mask: Mask [B, 1, T]
            cond: Conditioning (unused)
            inverse: Whether inverse operation

        Returns:
            If inverse: flipped x
            Else: (flipped x, logdet=0)
        """
        x = torch.flip(x, dims=[1])
        if inverse:
            return x
        else:
            logdet = torch.zeros(x.size(0), device=x.device)
            return x, logdet


class DDSConv(nn.Module):
    """
    Depthwise-dilated separable convolution stack (WaveNet-like conditioner).

    Args:
        channels: Number of channels
        kernel_size: Kernel size (default: 5)
        num_layers: Number of conv layers (default: 4)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        num_layers: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Changed from kernel_size ** i to 2 ** i
            padding = (kernel_size * dilation - dilation) // 2

            # Depthwise conv
            layers.append(nn.Conv1d(
                channels, channels, kernel_size,
                padding=padding, dilation=dilation, groups=channels
            ))
            layers.append(nn.GELU())

            # Pointwise conv
            layers.append(nn.Conv1d(channels, channels, 1))
            layers.append(nn.GELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through DDS conv stack.

        Args:
            x: Input [B, C, T]
            x_mask: Mask [B, 1, T]
            g: Global conditioning [B, C, T]

        Returns:
            Output [B, C, T]
        """
        # Add global conditioning
        if g is not None:
            x = x + g

        # Pass through conv stack with residual
        y = self.net(x * x_mask)
        return (x + y) * x_mask


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flow.

    Args:
        in_channels: Input channels (must be even)
        hidden_channels: Intermediate dimension (default: 192)
        kernel_size: Kernel size (default: 5)
        num_layers: Number of DDS conv layers (default: 4)
        cond_channels: Conditioning channels (default: 0)
        use_only_mean: Only predict mean (more stable) (default: False)
                       WARNING: use_only_mean=True yields zero log-det which may
                       undermine flow likelihood during training.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        num_layers: int = 4,
        cond_channels: int = 0,
        use_only_mean: bool = False
    ):
        super().__init__()

        assert in_channels % 2 == 0, "in_channels must be even"

        self.half_channels = in_channels // 2
        self.use_only_mean = use_only_mean

        # Issue warning if use_only_mean is True (at init time, not runtime)
        if use_only_mean:
            logger.warning(
                "AffineCouplingLayer initialized with use_only_mean=True. "
                "This yields zero log-det and may undermine flow likelihood during training. "
                "Consider using use_only_mean=False or implementing a staged training schedule."
            )

        # Input projection
        self.in_proj = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # Conditioning projection
        if cond_channels > 0:
            self.cond_proj = nn.Conv1d(cond_channels, hidden_channels, 1)
        else:
            self.cond_proj = None

        # DDS conv conditioner
        self.dds_conv = DDSConv(hidden_channels, kernel_size, num_layers)

        # Output projection
        out_channels = self.half_channels if use_only_mean else 2 * self.half_channels
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)

        # Initialize output projection weights to zero
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        inverse: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward or inverse pass through coupling layer.

        Args:
            x: Input [B, C, T]
            x_mask: Mask [B, 1, T]
            cond: Conditioning [B, C_cond, T]
            inverse: Whether to do inverse transform

        Returns:
            If inverse: transformed x
            Else: (transformed x, logdet)
        """
        # Split input
        xa, xb = torch.split(x, [self.half_channels, self.half_channels], dim=1)

        # Project first half
        h = self.in_proj(xa) * x_mask

        # Add conditioning
        if cond is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond)

        # Apply DDS conv
        h = self.dds_conv(h, x_mask)

        # Project to stats
        stats = self.out_proj(h) * x_mask

        if self.use_only_mean:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)

        if not inverse:
            # Forward: xb -> yb
            yb = (m + xb * torch.exp(logs)) * x_mask
            y = torch.cat([xa, yb], dim=1)
            logdet = torch.sum(logs * x_mask, dim=[1, 2])
            return y, logdet
        else:
            # Inverse: yb -> xb
            xb_rec = ((xb - m) * torch.exp(-logs)) * x_mask
            x_rec = torch.cat([xa, xb_rec], dim=1)
            return x_rec


class FlowDecoder(nn.Module):
    """
    Stack of affine coupling layers with flips for normalizing flow.

    Args:
        in_channels: Latent dimension (must be even)
        hidden_channels: Intermediate dimension (default: 192)
        num_flows: Number of coupling layers (default: 4)
        kernel_size: Kernel size (default: 5)
        num_layers: Layers per coupling (default: 4)
        cond_channels: Conditioning channels (default: 0)
        use_only_mean: Only predict mean (default: False)
                       WARNING: use_only_mean=True yields zero log-det which may
                       undermine flow likelihood during training.

    Example:
        >>> flow = FlowDecoder(in_channels=192, hidden_channels=192, num_flows=4, cond_channels=512)
        >>> z = torch.randn(2, 192, 100)
        >>> mask = torch.ones(2, 1, 100)
        >>> cond = torch.randn(2, 512, 100)
        >>> u, logdet = flow(z, mask, cond=cond, inverse=False)
        >>> print(u.shape, logdet.shape)  # [2, 192, 100], [2]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 192,
        num_flows: int = 4,
        kernel_size: int = 5,
        num_layers: int = 4,
        cond_channels: int = 0,
        use_only_mean: bool = False
    ):
        super().__init__()

        assert in_channels % 2 == 0, "in_channels must be even"

        self.in_channels = in_channels
        self.num_flows = num_flows
        self.use_only_mean = use_only_mean

        # Build flow stack
        flows = []
        for _ in range(num_flows):
            flows.append(AffineCouplingLayer(
                in_channels,
                hidden_channels,
                kernel_size,
                num_layers,
                cond_channels,
                use_only_mean
            ))
            flows.append(Flip())

        self.flows = nn.ModuleList(flows)

        logger.info(f"FlowDecoder initialized: {in_channels} channels, {num_flows} flows, use_only_mean={use_only_mean}")

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        inverse: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward or inverse flow.

        Args:
            x: Input [B, C, T]
            x_mask: Mask [B, 1, T]
            cond: Conditioning [B, C_cond, T]
            inverse: Whether to do inverse transform (inference)

        Returns:
            If inverse: transformed x
            Else: (transformed x, total logdet)
        """
        if not inverse:
            # Forward: z -> u (training)
            logdet_tot = torch.zeros(x.size(0), device=x.device)
            for flow in self.flows:
                out = flow(x, x_mask, cond=cond, inverse=False)
                if isinstance(out, tuple):
                    x, logdet = out
                    logdet_tot = logdet_tot + logdet
                else:
                    x = out  # Flip returns only x
            return x, logdet_tot
        else:
            # Inverse: u -> z (inference)
            for flow in reversed(self.flows):
                x = flow(x, x_mask, cond=cond, inverse=True)
            return x

    def remove_weight_norm(self):
        """Remove weight normalization from all coupling layers."""
        # AffineCouplingLayer doesn't use weight norm currently
        # This method is here for API consistency
        logger.info("Weight normalization removal called on FlowDecoder (no-op)")
