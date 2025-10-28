"""HiFi-GAN vocoder implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN generator."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                dilation=dilation[0], padding=self.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                dilation=dilation[1], padding=self.get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                dilation=dilation[2], padding=self.get_padding(kernel_size, dilation[2])))
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                dilation=1, padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                dilation=1, padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                dilation=1, padding=self.get_padding(kernel_size, 1)))
        ])
        
    def remove_weight_norm(self):
        """Remove weight normalization from convolutions."""
        try:
            for conv in self.convs1:
                nn.utils.remove_weight_norm(conv)
            for conv in self.convs2:
                nn.utils.remove_weight_norm(conv)
        except:
            pass

    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class MRF(nn.Module):
    """Multi-Receptive Field Fusion."""

    def __init__(self, channels, resblock_kernel_sizes, resblock_dilation_sizes):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for i, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.resblocks.append(ResBlock(channels, k, d))

    def forward(self, x):
        xs = None
        for resblock in self.resblocks:
            if xs is None:
                xs = resblock(x)
            else:
                xs += resblock(x)
        return xs / len(self.resblocks)


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator."""

    def __init__(self, mel_channels=80, in_channels=None, upsample_rates=(8, 8, 2, 2),
                 upsample_kernel_sizes=(16, 16, 4, 4),
                 upsample_initial_channel=512,
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 sample_rate=22050):
        super().__init__()

        # Support both parameter names for compatibility
        if in_channels is not None:
            mel_channels = in_channels

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.mel_channels = mel_channels
        self.sample_rate = sample_rate

        # Pre conv
        self.conv_pre = weight_norm(nn.Conv1d(mel_channels, upsample_initial_channel, 7, 1, padding=3))

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                 upsample_initial_channel // (2 ** (i + 1)),
                                 k, u, padding=(k - u) // 2)))

        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(MRF(ch, resblock_kernel_sizes, resblock_dilation_sizes))

        # Post conv
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i](x)
                else:
                    xs += self.resblocks[i](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        try:
            for l in self.ups:
                nn.utils.remove_weight_norm(l)
        except:
            pass
        try:
            for resblock in self.resblocks:
                if hasattr(resblock, 'remove_weight_norm'):
                    resblock.remove_weight_norm()
                elif hasattr(resblock, 'resblocks'):
                    for rb in resblock.resblocks:
                        if hasattr(rb, 'remove_weight_norm'):
                            rb.remove_weight_norm()
        except:
            pass
        try:
            nn.utils.remove_weight_norm(self.conv_pre)
        except:
            pass
        try:
            nn.utils.remove_weight_norm(self.conv_post)
        except:
            pass

    def prepare_for_export(self):
        """Prepare model for ONNX export by removing weight norm and setting eval mode."""
        self.eval()
        try:
            self.remove_weight_norm()
        except Exception as e:
            import logging
            logging.warning(f"Weight norm already removed or not present: {e}")

    def export_to_onnx(self, output_path: str, mel_shape: tuple = (1, 80, 100),
                      opset_version: int = 17, verbose: bool = True):
        """Export HiFi-GAN generator to ONNX format.

        Args:
            output_path: Path to save ONNX model
            mel_shape: Input mel-spectrogram shape (batch, mel_channels, time_steps)
            opset_version: ONNX opset version (17 for TensorRT 8.6+ compatibility)
            verbose: Enable verbose logging
        """
        import os
        from pathlib import Path

        # Prepare model for export
        self.prepare_for_export()

        # Create dummy mel-spectrogram input
        dummy_mel = torch.randn(*mel_shape)

        # Dynamic axes for variable-length audio
        dynamic_axes = {
            'mel_spectrogram': {0: 'batch_size', 2: 'time_steps'},
            'waveform': {0: 'batch_size', 2: 'audio_length'}
        }

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        if verbose:
            print(f"Exporting HiFi-GAN to ONNX: {output_path}")
            print(f"  Mel shape: {mel_shape}")
            print(f"  Dynamic axes: {dynamic_axes}")
            print(f"  Opset version: {opset_version}")

        torch.onnx.export(
            self,
            dummy_mel,
            output_path,
            input_names=['mel_spectrogram'],
            output_names=['waveform'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=verbose
        )

        if verbose:
            print(f"✓ ONNX export completed: {output_path}")
            print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


class HiFiGANDiscriminator(nn.Module):
    """HiFi-GAN Multi-Period Discriminator."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.msd = MultiScaleDiscriminator()

    def forward(self, y, y_hat):
        """Forward pass for discriminator training.
        
        Args:
            y: Real audio
            y_hat: Generated audio
            
        Returns:
            Tuple of (real_outputs, fake_outputs, real_feature_maps, fake_feature_maps)
        """
        # Multi-period discriminator
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        
        for d in self.mpd.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
            
        # Multi-scale discriminator  
        for d in self.msd.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        
    def forward_single(self, x):
        """Forward pass for single audio (for testing)."""
        outputs = []
        for d in self.mpd.discriminators:
            outputs.append(d(x))
        for d in self.msd.discriminators:
            outputs.append(d(x))
        return outputs


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])

    def forward(self, x):
        ret = []
        for disc in self.discriminators:
            ret.append(disc(x))
        return ret


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pooling = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        ret = []
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))
        return ret


class ScaleDiscriminator(nn.Module):
    """Scale discriminator."""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class PeriodDiscriminator(nn.Module):
    """Period discriminator."""

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap