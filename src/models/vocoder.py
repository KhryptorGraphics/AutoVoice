"""Neural vocoder for audio synthesis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class WaveNetBlock(nn.Module):
    """WaveNet residual block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size,
                                    dilation=dilation, padding=dilation)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size,
                                  dilation=dilation, padding=dilation)
        self.conv_res = nn.Conv1d(channels, channels, 1)
        self.conv_skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        residual = x

        # Gated activation
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        x = filter_out * gate_out

        # Residual and skip connections
        skip = self.conv_skip(x)
        x = self.conv_res(x)
        x = x + residual

        return x, skip


class Vocoder(nn.Module):
    """Neural vocoder for converting features to waveform."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256,
                num_layers: int = 30, kernel_size: int = 3):
        """Initialize vocoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden channel dimension
            num_layers: Number of WaveNet layers
            kernel_size: Kernel size for convolutions
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, 1)

        # WaveNet blocks
        self.wavenet_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 10)
            self.wavenet_blocks.append(
                WaveNetBlock(hidden_dim, kernel_size, dilation)
            )

        # Output projection
        self.output_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.output_conv2 = nn.Conv1d(hidden_dim, 1, 1)

        # Upsampling layers for frame-to-sample conversion
        self.upsample = nn.ConvTranspose1d(
            input_dim, input_dim, kernel_size=512,
            stride=256, padding=128
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: Mel spectrogram (batch, input_dim, time)

        Returns:
            Waveform (batch, 1, samples)
        """
        # Upsample mel to sample rate
        x = self.upsample(mel)

        # Initial projection
        x = self.input_conv(x)

        # WaveNet processing
        skip_sum = 0
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # Output projection
        x = torch.relu(skip_sum)
        x = self.output_conv1(x)
        x = torch.relu(x)
        x = self.output_conv2(x)

        # Apply tanh to bound output
        x = torch.tanh(x)

        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator for high-quality synthesis."""

    def __init__(self, input_dim: int = 80, upsample_rates: List[int] = [8, 8, 2, 2],
                upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
                channel_sizes: List[int] = [512, 256, 128, 64, 32]):
        """Initialize HiFi-GAN generator.

        Args:
            input_dim: Input feature dimension
            upsample_rates: Upsampling rates for each layer
            upsample_kernel_sizes: Kernel sizes for upsampling
            channel_sizes: Channel sizes for each layer
        """
        super().__init__()

        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(input_dim, channel_sizes[0], 7, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    channel_sizes[i],
                    channel_sizes[i + 1],
                    kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2
                )
            )

        # Residual blocks for each upsampling layer
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channel_sizes[i + 1]
            self.resblocks.append(
                nn.ModuleList([
                    ResBlock(ch, kernel_size=3, dilation=(1, 3, 5)),
                    ResBlock(ch, kernel_size=3, dilation=(1, 3, 5)),
                    ResBlock(ch, kernel_size=3, dilation=(1, 3, 5))
                ])
            )

        # Output convolution
        self.conv_post = nn.Conv1d(channel_sizes[-1], 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            x: Mel spectrogram (batch, input_dim, time)

        Returns:
            Waveform (batch, 1, samples)
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # Apply residual blocks
            for resblock in self.resblocks[i]:
                x = resblock(x)

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN."""

    def __init__(self, channels: int, kernel_size: int = 3,
                dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d,
                     padding=kernel_size * d // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=1,
                     padding=kernel_size // 2)
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + residual
        return x


class UniversalVocoder(nn.Module):
    """Universal vocoder supporting multiple architectures."""

    def __init__(self, vocoder_type: str = 'hifigan', **kwargs):
        """Initialize universal vocoder.

        Args:
            vocoder_type: Type of vocoder ('wavenet', 'hifigan')
            **kwargs: Additional arguments for specific vocoder
        """
        super().__init__()

        self.vocoder_type = vocoder_type

        if vocoder_type == 'wavenet':
            self.vocoder = Vocoder(**kwargs)
        elif vocoder_type == 'hifigan':
            self.vocoder = HiFiGANGenerator(**kwargs)
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: Mel spectrogram

        Returns:
            Waveform
        """
        return self.vocoder(mel)

    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode for faster generation.

        Args:
            mel: Mel spectrogram

        Returns:
            Waveform
        """
        with torch.no_grad():
            return self.forward(mel)