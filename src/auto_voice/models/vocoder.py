"""Vocoder models for waveform synthesis.

Includes HiFiGAN and BigVGAN (arxiv:2206.04658) generators.
BigVGAN uses Snake periodic activations and anti-aliased multi-periodicity
composition for superior singing voice synthesis.
"""
import logging
import math
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

HIFIGAN_CONFIG = {
    'resblock_kernel_sizes': [3, 7, 11],
    'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    'upsample_rates': [8, 8, 2, 2],
    'upsample_kernel_sizes': [16, 16, 4, 4],
    'upsample_initial_channel': 512,
    'num_mels': 80,
}


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=d,
                              padding=(kernel_size * d - d) // 2)
                )
            )
            self.convs2.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=1,
                              padding=(kernel_size - 1) // 2)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """HiFiGAN generator for mel-to-waveform synthesis."""

    def __init__(self, num_mels: int = 80, upsample_rates: Optional[List[int]] = None,
                 upsample_kernel_sizes: Optional[List[int]] = None,
                 upsample_initial_channel: int = 512,
                 resblock_kernel_sizes: Optional[List[int]] = None,
                 resblock_dilation_sizes: Optional[List[List[int]]] = None):
        super().__init__()

        upsample_rates = upsample_rates or HIFIGAN_CONFIG['upsample_rates']
        upsample_kernel_sizes = upsample_kernel_sizes or HIFIGAN_CONFIG['upsample_kernel_sizes']
        resblock_kernel_sizes = resblock_kernel_sizes or HIFIGAN_CONFIG['resblock_kernel_sizes']
        resblock_dilation_sizes = resblock_dilation_sizes or HIFIGAN_CONFIG['resblock_dilation_sizes']

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(num_mels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
                )
            )
            ch = ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch_out, k, d))

        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch_out, 1, 7, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel-spectrogram.

        Args:
            mel: [B, num_mels, T]

        Returns:
            [B, 1, T*prod(upsample_rates)]
        """
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for module in self.modules():
            try:
                nn.utils.parametrize.remove_parametrizations(module, 'weight')
            except (ValueError, AttributeError):
                pass


class HiFiGANVocoder:
    """High-level vocoder interface wrapping HiFiGAN generator."""

    def __init__(self, device=None, config: Optional[dict] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or HIFIGAN_CONFIG
        self.sample_rate = 22050
        self._generator = None
        self._loaded = False

    def _ensure_loaded(self):
        """Ensure generator is initialized."""
        if self._generator is None:
            self._generator = HiFiGANGenerator(
                num_mels=self.config['num_mels'],
                upsample_rates=self.config['upsample_rates'],
                upsample_kernel_sizes=self.config['upsample_kernel_sizes'],
                upsample_initial_channel=self.config['upsample_initial_channel'],
                resblock_kernel_sizes=self.config['resblock_kernel_sizes'],
                resblock_dilation_sizes=self.config['resblock_dilation_sizes'],
            ).to(self.device)
            self._generator.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained vocoder weights."""
        self._ensure_loaded()
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Vocoder checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                if 'generator' in checkpoint:
                    self._generator.load_state_dict(checkpoint['generator'])
                elif 'state_dict' in checkpoint:
                    self._generator.load_state_dict(checkpoint['state_dict'])
                else:
                    self._generator.load_state_dict(checkpoint)
            self._generator.remove_weight_norm()
            self._loaded = True
            logger.info(f"Vocoder loaded from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vocoder: {e}")
            return False

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel-spectrogram.

        Args:
            mel: [B, num_mels, T] or [num_mels, T]

        Returns:
            Audio waveform [B, T]
        """
        self._ensure_loaded()

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)

        with torch.no_grad():
            audio = self._generator(mel)

        return audio.squeeze(1)

    def mel_to_audio(self, mel, sr: int = 22050):
        """Convert mel spectrogram to numpy audio array."""
        import numpy as np
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel).float()
        audio = self.synthesize(mel)
        return audio.cpu().numpy().squeeze()

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, device=None) -> 'HiFiGANVocoder':
        """Load a pretrained HiFiGAN vocoder."""
        vocoder = cls(device=device)
        vocoder.load_checkpoint(checkpoint_path)
        return vocoder


# ─────────────────────────────────────────────────────────────────────────────
# BigVGAN: Large-Scale Training for Universal Neural Vocoder (ICLR 2023)
# arxiv:2206.04658 - Uses Snake periodic activations + anti-aliased upsampling
# ─────────────────────────────────────────────────────────────────────────────

BIGVGAN_24KHZ_100BAND_CONFIG = {
    'num_mels': 100,
    'upsample_rates': [4, 4, 2, 2, 2, 2],
    'upsample_kernel_sizes': [8, 8, 4, 4, 4, 4],
    'upsample_initial_channel': 1536,
    'resblock_kernel_sizes': [3, 7, 11],
    'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    'sample_rate': 24000,
    'hop_size': 256,
    'activation': 'snakebeta',
    'snake_logscale': True,
}


class SnakeBeta(nn.Module):
    """Snake activation with separate beta parameter (BigVGAN v2).

    f(x) = x + (1/beta) * sin^2(alpha * x)

    Alpha and beta are trainable per-channel parameters stored in log-scale.
    """

    def __init__(self, channels: int, logscale: bool = True):
        super().__init__()
        self.logscale = logscale
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp() if self.logscale else self.log_alpha
        beta = self.log_beta.exp() if self.logscale else self.log_beta
        beta = beta.clamp(min=1e-5)
        return x + (1.0 / beta) * torch.sin(alpha * x).pow(2)


class Snake(nn.Module):
    """Snake activation (same alpha for frequency and amplitude).

    f(x) = x + (1/alpha) * sin^2(alpha * x)
    """

    def __init__(self, channels: int, logscale: bool = True):
        super().__init__()
        self.logscale = logscale
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp() if self.logscale else self.log_alpha
        alpha = alpha.clamp(min=1e-5)
        return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)


class Activation1d(nn.Module):
    """Anti-aliased 1D activation: upsample -> activate -> downsample.

    Prevents aliasing artifacts from periodic activations by applying
    the nonlinearity in an oversampled domain.
    """

    def __init__(self, activation: nn.Module, up_ratio: int = 2):
        super().__init__()
        self.activation = activation
        self.up_ratio = up_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=self.up_ratio, mode='linear',
                             align_corners=False)
        x_act = self.activation(x_up)
        x_down = F.interpolate(x_act, size=x.shape[-1], mode='linear',
                               align_corners=False)
        return x_down


class AMPBlock(nn.Module):
    """Anti-aliased Multi-Periodicity residual block (BigVGAN).

    Uses SnakeBeta activation wrapped in Activation1d for anti-aliasing.
    3 residual layers (AMPBlock1 variant).
    """

    def __init__(self, channels: int, kernel_size: int, dilations: List[int],
                 activation: str = 'snakebeta', logscale: bool = True):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.activations1 = nn.ModuleList()
        self.activations2 = nn.ModuleList()

        for d in dilations:
            act_fn = self._make_activation(channels, activation, logscale)
            self.activations1.append(Activation1d(act_fn))
            self.convs1.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=d,
                              padding=(kernel_size * d - d) // 2)
                )
            )
            act_fn = self._make_activation(channels, activation, logscale)
            self.activations2.append(Activation1d(act_fn))
            self.convs2.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=1,
                              padding=(kernel_size - 1) // 2)
                )
            )

    @staticmethod
    def _make_activation(channels: int, activation: str, logscale: bool) -> nn.Module:
        if activation == 'snakebeta':
            return SnakeBeta(channels, logscale=logscale)
        elif activation == 'snake':
            return Snake(channels, logscale=logscale)
        else:
            raise RuntimeError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, c1, act2, c2 in zip(self.activations1, self.convs1,
                                        self.activations2, self.convs2):
            xt = act1(x)
            xt = c1(xt)
            xt = act2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGANGenerator(nn.Module):
    """BigVGAN generator with Snake activations and anti-aliased upsampling.

    Architecture: Conv1d -> [N upsample blocks with AMP residuals] -> Snake -> Conv1d -> tanh

    Reference: arxiv:2206.04658 (BigVGAN, ICLR 2023)
    Config: bigvgan_v2_24khz_100band_256x (112M params)
    """

    def __init__(self, num_mels: int = 100,
                 upsample_rates: Optional[List[int]] = None,
                 upsample_kernel_sizes: Optional[List[int]] = None,
                 upsample_initial_channel: int = 1536,
                 resblock_kernel_sizes: Optional[List[int]] = None,
                 resblock_dilation_sizes: Optional[List[List[int]]] = None,
                 activation: str = 'snakebeta',
                 snake_logscale: bool = True):
        super().__init__()

        upsample_rates = upsample_rates or BIGVGAN_24KHZ_100BAND_CONFIG['upsample_rates']
        upsample_kernel_sizes = upsample_kernel_sizes or BIGVGAN_24KHZ_100BAND_CONFIG['upsample_kernel_sizes']
        resblock_kernel_sizes = resblock_kernel_sizes or BIGVGAN_24KHZ_100BAND_CONFIG['resblock_kernel_sizes']
        resblock_dilation_sizes = resblock_dilation_sizes or BIGVGAN_24KHZ_100BAND_CONFIG['resblock_dilation_sizes']

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.activation = activation
        self.snake_logscale = snake_logscale

        self.conv_pre = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(num_mels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
                )
            )
            ch = ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    AMPBlock(ch_out, k, d, activation=activation,
                             logscale=snake_logscale)
                )

        self.activation_post = Activation1d(
            SnakeBeta(ch_out, logscale=snake_logscale) if activation == 'snakebeta'
            else Snake(ch_out, logscale=snake_logscale)
        )
        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch_out, 1, 7, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel-spectrogram.

        Args:
            mel: [B, num_mels, T]

        Returns:
            [B, 1, T*prod(upsample_rates)]
        """
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for module in self.modules():
            try:
                nn.utils.parametrize.remove_parametrizations(module, 'weight')
            except (ValueError, AttributeError):
                pass


class BigVGANVocoder:
    """High-level vocoder interface wrapping BigVGAN generator.

    Drop-in replacement for HiFiGANVocoder with better singing generalization.
    Uses 100-band mel at 24kHz by default (bigvgan_v2_24khz_100band_256x config).
    """

    def __init__(self, device=None, config: Optional[dict] = None,
                 pretrained: Optional[str] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or BIGVGAN_24KHZ_100BAND_CONFIG
        self.sample_rate = self.config.get('sample_rate', 24000)
        self.n_mels = self.config.get('num_mels', 100)
        self.hop_size = self.config.get('hop_size', 256)
        self.upsample_rates = self.config.get('upsample_rates', [4, 4, 2, 2, 2, 2])
        self._generator = None
        self._loaded = False

        if pretrained is not None:
            self.load_checkpoint(pretrained)

    @property
    def upsamples(self):
        """Proxy to generator's upsample layers."""
        self._ensure_loaded()
        return self._generator.ups

    @property
    def resblocks(self):
        """Proxy to generator's residual blocks."""
        self._ensure_loaded()
        return self._generator.resblocks

    def to(self, device):
        """Move vocoder to device."""
        self.device = device
        if self._generator is not None:
            self._generator.to(device)
        return self

    def _ensure_loaded(self):
        """Ensure generator is initialized."""
        if self._generator is None:
            self._generator = BigVGANGenerator(
                num_mels=self.config['num_mels'],
                upsample_rates=self.config['upsample_rates'],
                upsample_kernel_sizes=self.config['upsample_kernel_sizes'],
                upsample_initial_channel=self.config['upsample_initial_channel'],
                resblock_kernel_sizes=self.config['resblock_kernel_sizes'],
                resblock_dilation_sizes=self.config['resblock_dilation_sizes'],
                activation=self.config.get('activation', 'snakebeta'),
                snake_logscale=self.config.get('snake_logscale', True),
            ).to(self.device)
            self._generator.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained BigVGAN weights.

        Supports NVIDIA's official checkpoint format (generator key)
        and HuggingFace-style state dicts.
        """
        self._ensure_loaded()
        path = Path(checkpoint_path)
        if not path.exists():
            raise RuntimeError(f"BigVGAN checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'generator' in checkpoint:
                self._generator.load_state_dict(checkpoint['generator'])
            elif 'state_dict' in checkpoint:
                self._generator.load_state_dict(checkpoint['state_dict'])
            else:
                self._generator.load_state_dict(checkpoint)
        else:
            raise RuntimeError("Unexpected checkpoint format for BigVGAN")
        self._generator.remove_weight_norm()
        self._loaded = True
        logger.info(f"BigVGAN vocoder loaded from {checkpoint_path}")

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel-spectrogram.

        Args:
            mel: [B, num_mels, T] or [num_mels, T]

        Returns:
            Audio waveform [B, T]
        """
        self._ensure_loaded()

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)

        with torch.no_grad():
            audio = self._generator(mel)

        return audio.squeeze(1)

    def mel_to_audio(self, mel, sr: int = 24000):
        """Convert mel spectrogram to numpy audio array."""
        import numpy as np
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel).float()
        audio = self.synthesize(mel)
        return audio.cpu().numpy().squeeze()

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, device=None) -> 'BigVGANVocoder':
        """Load a pretrained BigVGAN vocoder."""
        vocoder = cls(device=device)
        vocoder.load_checkpoint(checkpoint_path)
        return vocoder
