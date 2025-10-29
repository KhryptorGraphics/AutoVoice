"""
Voice Discriminator for Adversarial Training

Multi-scale discriminator for voice conversion adversarial loss.
Uses hinge loss for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class DiscriminatorBlock(nn.Module):
    """Single discriminator block with conv layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 41,
        stride: int = 4,
        groups: int = 4,
        use_spectral_norm: bool = False
    ):
        """Initialize discriminator block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            groups: Number of groups for grouped convolution
            use_spectral_norm: Apply spectral normalization
        """
        super().__init__()

        # Build conv layer
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups
        )

        # Apply spectral norm if requested
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        self.conv = conv
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C_out, T_out]
        """
        x = self.conv(x)
        x = self.activation(x)
        return x


class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator."""

    def __init__(
        self,
        use_spectral_norm: bool = False,
        channels: int = 64
    ):
        """Initialize scale discriminator.

        Args:
            use_spectral_norm: Apply spectral normalization
            channels: Base number of channels
        """
        super().__init__()

        # Initial layer (no pooling)
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(1, channels, kernel_size=15, stride=1, groups=1, use_spectral_norm=use_spectral_norm),
            DiscriminatorBlock(channels, channels * 2, kernel_size=41, stride=4, groups=4, use_spectral_norm=use_spectral_norm),
            DiscriminatorBlock(channels * 2, channels * 4, kernel_size=41, stride=4, groups=16, use_spectral_norm=use_spectral_norm),
            DiscriminatorBlock(channels * 4, channels * 8, kernel_size=41, stride=4, groups=16, use_spectral_norm=use_spectral_norm),
            DiscriminatorBlock(channels * 8, channels * 8, kernel_size=5, stride=1, groups=1, use_spectral_norm=use_spectral_norm),
        ])

        # Final projection to logits
        final_conv = nn.Conv1d(channels * 8, 1, kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)
        self.final_conv = final_conv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input waveform [B, 1, T] or [B, T]

        Returns:
            Tuple of (logits [B, 1, T_out], feature_maps [List of tensors])
        """
        # Ensure input is [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != 1:
            # If [B, T, 1], transpose
            if x.size(2) == 1:
                x = x.transpose(1, 2)
            else:
                # Assume [B, C, T] with C != 1, take first channel
                x = x[:, :1, :]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        logits = self.final_conv(x)

        return logits, feature_maps


class VoiceDiscriminator(nn.Module):
    """Multi-scale discriminator for voice conversion.

    Uses 3 discriminators at different scales:
    - Scale 1: Original resolution (1x)
    - Scale 2: Downsampled by 2x
    - Scale 3: Downsampled by 4x
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        num_scales: int = 3,
        channels: int = 64
    ):
        """Initialize multi-scale discriminator.

        Args:
            use_spectral_norm: Apply spectral normalization for training stability
            num_scales: Number of discriminator scales (default: 3)
            channels: Base number of channels (default: 64)
        """
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=use_spectral_norm, channels=channels)
            for _ in range(num_scales)
        ])

        # Downsampling layers (average pooling)
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])

        logger.info(f"VoiceDiscriminator initialized with {num_scales} scales, "
                   f"spectral_norm={use_spectral_norm}, channels={channels}")

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass through all scales.

        Args:
            x: Input waveform [B, T] or [B, 1, T]

        Returns:
            Tuple of:
                - logits_list: List of logits from each scale [B, 1, T_i]
                - features_list: List of feature maps from each scale
        """
        # Ensure input is [B, 1, T] or [B, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        logits_list = []
        features_list = []

        for i, (downsampler, discriminator) in enumerate(zip(self.downsamplers, self.discriminators)):
            # Downsample input
            x_scaled = downsampler(x)

            # Discriminate
            logits, features = discriminator(x_scaled)

            logits_list.append(logits)
            features_list.append(features)

        return logits_list, features_list


def hinge_discriminator_loss(
    real_logits_list: List[torch.Tensor],
    fake_logits_list: List[torch.Tensor]
) -> torch.Tensor:
    """Compute hinge loss for discriminator.

    Hinge loss: D_loss = mean(ReLU(1 - D(real))) + mean(ReLU(1 + D(fake)))

    Args:
        real_logits_list: List of logits from real audio (each [B, 1, T])
        fake_logits_list: List of logits from fake audio (each [B, 1, T])

    Returns:
        Scalar discriminator loss
    """
    loss = 0.0
    for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
        # Hinge loss for real: max(0, 1 - D(real))
        loss_real = torch.mean(F.relu(1.0 - real_logits))

        # Hinge loss for fake: max(0, 1 + D(fake))
        loss_fake = torch.mean(F.relu(1.0 + fake_logits))

        loss += loss_real + loss_fake

    # Average over scales
    loss = loss / len(real_logits_list)

    return loss


def hinge_generator_loss(
    fake_logits_list: List[torch.Tensor]
) -> torch.Tensor:
    """Compute hinge loss for generator (adversarial term).

    Generator loss: G_adv = -mean(D(fake))

    Args:
        fake_logits_list: List of logits from generated audio (each [B, 1, T])

    Returns:
        Scalar generator adversarial loss
    """
    loss = 0.0
    for fake_logits in fake_logits_list:
        # Generator wants discriminator to output high values (real-like)
        loss += -torch.mean(fake_logits)

    # Average over scales
    loss = loss / len(fake_logits_list)

    return loss


def feature_matching_loss(
    real_features_list: List[List[torch.Tensor]],
    fake_features_list: List[List[torch.Tensor]]
) -> torch.Tensor:
    """Compute feature matching loss between real and fake intermediate features.

    Args:
        real_features_list: List of feature maps from real audio (list of lists)
        fake_features_list: List of feature maps from fake audio (list of lists)

    Returns:
        Scalar feature matching loss
    """
    loss = 0.0
    num_features = 0

    for real_features, fake_features in zip(real_features_list, fake_features_list):
        for real_feat, fake_feat in zip(real_features, fake_features):
            # L1 loss between feature maps
            loss += F.l1_loss(fake_feat, real_feat.detach())
            num_features += 1

    # Average over all features
    if num_features > 0:
        loss = loss / num_features

    return loss
