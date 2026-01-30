"""So-VITS-SVC (Singing Voice Conversion) model.

Combines content encoder, pitch encoder, speaker encoder, and decoder
for high-quality singing voice conversion.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Compute differentiable SSIM loss between predicted and target mel spectrograms.

    Args:
        pred: [B, C, H, W] predicted mel (treat as single-channel image)
        target: [B, C, H, W] target mel
        window_size: Gaussian window size (must be odd)
        size_average: If True, return mean SSIM loss

    Returns:
        1 - SSIM (so lower is better, suitable as loss)
    """
    channel = pred.size(1)

    # Create 2D Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) * g.unsqueeze(0)  # [window_size, window_size]
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)

    pad = window_size // 2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
    mu2 = F.conv2d(target, window, padding=pad, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1.0 - ssim_map.mean()
    else:
        return 1.0 - ssim_map.mean(dim=[1, 2, 3])


class PosteriorEncoder(nn.Module):
    """Posterior encoder using WaveNet-style dilated convolutions."""

    def __init__(self, in_channels: int = 513, hidden_channels: int = 192,
                 out_channels: int = 192, kernel_size: int = 5, n_layers: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNetBlock(hidden_channels, kernel_size, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to latent with reparameterization."""
        x = self.pre(x)
        x = self.enc(x)
        stats = self.proj(x)
        mean, logvar = torch.chunk(stats, 2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z, mean, logvar


class WaveNetBlock(nn.Module):
    """WaveNet-style dilated convolution block."""

    def __init__(self, channels: int, kernel_size: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, channels * 2, kernel_size,
                          dilation=dilation, padding=padding),
                nn.GLU(dim=1),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        return x


class FlowDecoder(nn.Module):
    """Normalizing flow for voice conversion."""

    def __init__(self, channels: int = 192, hidden_channels: int = 192,
                 kernel_size: int = 5, n_layers: int = 4, n_flows: int = 4):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(AffineCouplingLayer(channels, hidden_channels, kernel_size, n_layers))
            self.flows.append(Flip())

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x = flow(x)
        else:
            for flow in reversed(self.flows):
                x = flow(x, reverse=True)
        return x


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""

    def __init__(self, channels: int, hidden_channels: int,
                 kernel_size: int, n_layers: int):
        super().__init__()
        self.half_channels = channels // 2

        self.net = nn.Sequential(
            nn.Conv1d(self.half_channels, hidden_channels, 1),
            nn.ReLU(),
            WaveNetBlock(hidden_channels, kernel_size, n_layers),
            nn.Conv1d(hidden_channels, self.half_channels * 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        x0, x1 = x.chunk(2, dim=1)
        stats = self.net(x0)
        shift, log_scale = stats.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)

        if not reverse:
            x1 = x1 * torch.exp(log_scale) + shift
        else:
            x1 = (x1 - shift) * torch.exp(-log_scale)

        return torch.cat([x0, x1], dim=1)


class Flip(nn.Module):
    """Channel flip for flow diversity."""

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return torch.flip(x, dims=[1])


class SoVitsSvc(nn.Module):
    """So-VITS-SVC model for singing voice conversion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}

        self.content_dim = config.get('content_dim', 256)
        self.pitch_dim = config.get('pitch_dim', 256)
        self.speaker_dim = config.get('speaker_dim', 256)
        self.hidden_dim = config.get('hidden_dim', 192)
        self.n_mels = config.get('n_mels', 80)
        self.spec_channels = config.get('spec_channels', 513)
        self.ssim_weight = config.get('ssim_weight', 0.5)
        self.ssim_window_size = config.get('ssim_window_size', 11)

        self.content_proj = nn.Linear(self.content_dim, self.hidden_dim)
        self.pitch_proj = nn.Linear(self.pitch_dim, self.hidden_dim)
        self.speaker_proj = nn.Linear(self.speaker_dim, self.hidden_dim)

        self.posterior_encoder = PosteriorEncoder(
            in_channels=self.spec_channels,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
        )

        self.flow = FlowDecoder(
            channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
        )

        self.mel_decoder = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim * 2, self.n_mels, 1),
        )

    def forward(self, content: torch.Tensor, pitch: torch.Tensor,
                speaker: torch.Tensor, spec: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            content: [B, T, content_dim]
            pitch: [B, T, pitch_dim]
            speaker: [B, speaker_dim]
            spec: [B, spec_channels, T] (training only)
        """
        c = self.content_proj(content).transpose(1, 2)
        p = self.pitch_proj(pitch).transpose(1, 2)
        s = self.speaker_proj(speaker).unsqueeze(-1)

        h = c + p + s

        if spec is not None:
            z, mean, logvar = self.posterior_encoder(spec)
            z_flow = self.flow(z)
            mel_pred = self.mel_decoder(z)
            return {
                'mel_pred': mel_pred,
                'z': z,
                'z_flow': z_flow,
                'mean': mean,
                'logvar': logvar,
                'h': h,
            }
        else:
            z = self.flow(h, reverse=True)
            mel_pred = self.mel_decoder(z)
            return {
                'mel_pred': mel_pred,
                'z': z,
            }

    def infer(self, content: torch.Tensor, pitch: torch.Tensor,
              speaker: torch.Tensor) -> torch.Tensor:
        """Inference - generate mel from content+pitch+speaker."""
        device = next(self.parameters()).device
        content = content.to(device)
        pitch = pitch.to(device)
        speaker = speaker.to(device)
        with torch.no_grad():
            result = self.forward(content, pitch, speaker, spec=None)
        return result['mel_pred']

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, device=None,
                        config: Optional[Dict] = None) -> 'SoVitsSvc':
        """Load pretrained So-VITS-SVC model."""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cls(config=config)

        path = Path(checkpoint_path)
        if path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'], strict=False)
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                logger.info(f"So-VITS-SVC loaded from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

        model.to(device)
        return model

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     target_mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute training losses including SSIM for perceptual quality."""
        mel_pred = outputs['mel_pred']
        mean = outputs['mean']
        logvar = outputs['logvar']
        h = outputs['h']
        z_flow = outputs['z_flow']

        min_len = min(mel_pred.shape[-1], target_mel.shape[-1])
        mel_pred_aligned = mel_pred[..., :min_len]
        target_aligned = target_mel[..., :min_len]

        recon_loss = F.l1_loss(mel_pred_aligned, target_aligned)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        min_len_flow = min(z_flow.shape[-1], h.shape[-1])
        flow_loss = F.mse_loss(z_flow[..., :min_len_flow], h[..., :min_len_flow])

        # SSIM loss: treat mel as single-channel image [B, 1, 80, T]
        ssim_loss = _ssim_loss(
            mel_pred_aligned.unsqueeze(1),
            target_aligned.unsqueeze(1),
            window_size=self.ssim_window_size,
        )

        total_loss = recon_loss + 0.1 * kl_loss + flow_loss + self.ssim_weight * ssim_loss

        return {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'flow_loss': flow_loss,
            'ssim_loss': ssim_loss,
            'total_loss': total_loss,
        }
