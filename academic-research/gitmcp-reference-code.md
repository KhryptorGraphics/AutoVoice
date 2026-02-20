# GitMCP: Reference Implementation Code Analysis

## Research Source
- GitMCP code search across CoMoSVC, BigVGAN, ContentVec repositories
- Focus: exact implementation details, parameters, architecture patterns

---

## 1. CoMoSVC / CoMoSpeech - Consistency Distillation

### Important: Correct Repository URLs
- **`Grace9994/CoMoSVC`** - The SVC (Singing Voice Conversion) version
- **`zhenye234/CoMoSpeech`** - The TTS version (ACM MM 2023)
- ~~`zhenye234/CoMoSVC`~~ - Does NOT exist

### Denoiser Network (wavenet.py)

Standard dilated Conv1d residual blocks (NOT BiDilConv as previously assumed):

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, n_mels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            hidden_dim, 2 * hidden_dim,
            kernel_size=3, dilation=dilation, padding=dilation
        )
        self.diffusion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.conditioner_proj = nn.Conv1d(n_mels, 2 * hidden_dim, 1)
        self.output_proj = nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)

    def forward(self, x, diffusion_step, conditioner):
        # Add diffusion step embedding
        h = x + self.diffusion_proj(diffusion_step).unsqueeze(-1)
        # Dilated convolution
        h = self.dilated_conv(h)
        # Add conditioner
        h = h + self.conditioner_proj(conditioner)
        # Gated activation
        gate, filter = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filter)
        # Residual + skip
        h = self.output_proj(h)
        residual, skip = h.chunk(2, dim=1)
        return (x + residual) / sqrt(2), skip
```

### EDM Preconditioning (como.py)

```python
class EDMPrecond(nn.Module):
    """Karras et al. preconditioning for consistency models."""

    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def forward(self, net, x, sigma, cond):
        # Preconditioning coefficients
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.log() / 4  # Log-scale noise conditioning

        # Preconditioned output
        F_x = net(c_in * x, c_noise, cond)
        D_x = c_skip * x + c_out * F_x
        return D_x
```

### EDM Loss

```python
class EDMLoss:
    """EDM training loss with log-normal noise sampling."""

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x, cond):
        # Sample noise level from log-normal
        rnd_normal = torch.randn(x.shape[0], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # Weight function
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2

        # Add noise and predict
        n = torch.randn_like(x) * sigma.unsqueeze(-1)
        D_x = net(x + n, sigma, cond)

        loss = weight * ((D_x - x)**2)
        return loss.mean()
```

### Consistency Training Loss (CTLoss_D)

```python
class CTLoss_D:
    """Consistency distillation loss with EMA teacher."""

    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7, N=25, mu=0.95):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.N = N          # 25 for SVC, 50 for TTS
        self.mu = mu        # EMA decay rate

    def __call__(self, student, teacher_ema, pretrained, x, cond):
        # Discretize noise schedule
        step_indices = torch.randint(0, self.N - 1, (x.shape[0],))
        t_cur = self._sigma_schedule(step_indices)
        t_next = self._sigma_schedule(step_indices + 1)

        # Add noise at t_cur
        noise = torch.randn_like(x)
        x_cur = x + t_cur.unsqueeze(-1) * noise

        # Euler step using pretrained model (teacher denoise)
        with torch.no_grad():
            d_cur = pretrained(x_cur, t_cur, cond)
            x_next = x_cur + (t_next - t_cur).unsqueeze(-1) * (x_cur - d_cur) / t_cur.unsqueeze(-1)

        # Student prediction at t_cur
        student_out = student(x_cur, t_cur, cond)

        # EMA teacher prediction at t_next
        with torch.no_grad():
            teacher_out = teacher_ema(x_next, t_next, cond)

        # Consistency loss: student(x_t) should equal teacher_ema(x_t+1)
        loss = F.mse_loss(student_out, teacher_out)
        return loss

    def _sigma_schedule(self, indices):
        """Karras noise schedule discretization."""
        rho_inv = 1.0 / self.rho
        sigmas = (
            self.sigma_min**rho_inv +
            indices / (self.N - 1) *
            (self.sigma_max**rho_inv - self.sigma_min**rho_inv)
        ) ** self.rho
        return sigmas
```

### CT Sampler (Single-Step Inference)

```python
def ct_sample(model, shape, cond, device, sigma_max=80):
    """Single-step consistency model sampling."""
    # Start from maximum noise
    x = torch.randn(shape, device=device) * sigma_max
    sigma = torch.full((shape[0],), sigma_max, device=device)

    # Single forward pass → clean prediction
    with torch.no_grad():
        x_pred = model(x, sigma, cond)

    return x_pred
```

### Key Parameters:
| Parameter | SVC Value | TTS Value |
|-----------|-----------|-----------|
| sigma_min | 0.002 | 0.002 |
| sigma_max | 80 | 80 |
| rho | 7 | 7 |
| N (steps) | 25 | 50 |
| mu (EMA) | 0.95 | 0.95 |
| sigma_data | 0.5 | 0.5 |
| P_mean | -1.2 | -1.2 |
| P_std | 1.2 | 1.2 |

---

## 2. BigVGAN - Snake Activation & Anti-Aliasing

### Snake Activation (activations.py)

```python
class Snake(nn.Module):
    """Periodic activation: x + (1/alpha) * sin^2(x * alpha)"""

    def __init__(self, channels, alpha=1.0, alpha_logscale=True):
        super().__init__()
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(1, channels, 1))  # log-scale
        else:
            self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
        self.alpha_logscale = alpha_logscale

    def forward(self, x):
        alpha = self.alpha.exp() if self.alpha_logscale else self.alpha
        return x + (1.0 / alpha) * torch.sin(x * alpha) ** 2


class SnakeBeta(nn.Module):
    """Snake with separate beta for magnitude control (recommended)."""

    def __init__(self, channels, alpha=1.0, alpha_logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.alpha_logscale = alpha_logscale

    def forward(self, x):
        alpha = self.alpha.exp() if self.alpha_logscale else self.alpha
        beta = self.beta.exp() if self.alpha_logscale else self.beta
        return x + (1.0 / beta) * torch.sin(x * alpha) ** 2
```

### Anti-Aliased Activation (activation1d.py)

```python
class Activation1d(nn.Module):
    """Anti-aliased activation: upsample -> activate -> downsample."""

    def __init__(self, activation, up_ratio=2, down_ratio=2, up_kernel_size=12):
        super().__init__()
        self.activation = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, up_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.activation(x)
        x = self.downsample(x)
        return x
```

### Generator Architecture (bigvgan.py)

```python
class BigVGAN(nn.Module):
    """BigVGAN generator with AMP blocks."""

    def __init__(self, h):
        # Initial conv
        self.conv_pre = nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, 3)

        # Upsampling blocks
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(ch // (2**i), ch // (2**(i+1)),
                                   k, u, padding=(k-u)//2)
            )

        # AMP residual blocks (per upsampling layer)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(
                    AMPBlock1(ch // (2**(i+1)), k, d,
                              activation=SnakeBeta)  # SnakeBeta recommended
                )

        # Post conv
        self.activation_post = SnakeBeta(ch // (2**len(self.ups)))
        self.conv_post = nn.Conv1d(ch // (2**len(self.ups)), 1, 7, 1, 3)
```

---

## 3. ContentVec - Feature Extraction

### Model Architecture (contentvec.py)

```python
class ContentVec(nn.Module):
    """ContentVec: content-disentangled speech representation."""

    def __init__(self, cfg):
        # 12 transformer encoder layers
        # 768 embed dim, 3072 FFN dim, 12 attention heads
        # mask_prob=0.80 (during training)
        self.encoder = TransformerEncoder(cfg)

    def extract_features(self, source, spk_emb=None, output_layer=None):
        """Extract features from specified transformer layer.

        Args:
            source: raw audio waveform
            spk_emb: optional speaker embedding (for training)
            output_layer: which layer to extract from (1-indexed)
                         Use output_layer=9 for SVC (vec256l9)

        Returns:
            features: [B, T, 768] tensor from specified layer
        """
        # CNN feature extractor
        features = self.feature_extractor(source)
        # Transformer layers (stop at output_layer if specified)
        for i, layer in enumerate(self.encoder.layers):
            features = layer(features)
            if output_layer is not None and i + 1 == output_layer:
                break
        return features
```

### Config (contentvec.yaml)

```yaml
encoder_layers: 12
encoder_embed_dim: 768
encoder_ffn_embed_dim: 3072
encoder_attention_heads: 12
contrastive_layers: [-1]
mask_prob: 0.80
```

### Usage for AutoVoice (Layer 9, 256-dim)

```python
# Option 1: Via transformers (lengyue233/content-vec-best)
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("lengyue233/content-vec-best")
model.eval()

with torch.no_grad():
    # Input: raw waveform at 16kHz
    outputs = model(input_values=waveform, output_hidden_states=True)
    # Layer 9 features (0-indexed in transformers = layer 10)
    content_features = outputs.hidden_states[9]  # [B, T, 768]

# Project to 256-dim
projection = nn.Linear(768, 256)
content_256 = projection(content_features)  # [B, T, 256]
```

```python
# Option 2: Via fairseq checkpoint (contentvec_hubert_large_top9)
import fairseq

models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    ["checkpoint_best_legacy_500.pt"]
)
model = models[0].eval()

with torch.no_grad():
    features = model.extract_features(
        source=waveform,
        output_layer=9  # 1-indexed
    )[0]  # [B, T, 768]
```

### Key Details:
- **Input**: 16kHz mono waveform
- **Output**: [B, T, 768] where T = audio_samples / 320 (hop size)
- **Layer 9**: Best content/speaker disentanglement for SVC
- **Projection**: 768→256 linear layer matches our ContentEncoder output_size
