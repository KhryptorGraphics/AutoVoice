# Vocoder & Separation Research for Singing Voice Conversion

Research findings on BigVGAN (neural vocoder) and Demucs v4 (vocal separation) for integration into the AutoVoice singing voice conversion pipeline.

---

## 1. BigVGAN: Universal Neural Vocoder

### Paper References

- **Original**: "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (ICLR 2023)
  - arXiv: [2206.04658](https://arxiv.org/abs/2206.04658)
  - Authors: Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon
- **BigVGAN-v2** (July 2024): Updated with CQT discriminator, diverse training data, CUDA kernels
- **Repository**: https://github.com/NVIDIA/BigVGAN
- **License**: MIT

### Architecture Overview

BigVGAN extends HiFi-GAN's generator with two key innovations:
1. **Snake periodic activation function** for waveform inductive bias
2. **Anti-aliased multi-periodicity composition (AMP)** module to suppress aliasing

#### Generator Structure

```
Input: Mel spectrogram [B, num_mels, T]
  |
  v
Conv1d(num_mels -> upsample_initial_channel, kernel=7)
  |
  v
[N upsampling blocks] -- each halves channels
  |  ConvTranspose1d(ch -> ch/2, kernel=upsample_kernel_size, stride=upsample_rate)
  |  M residual AMP blocks per upsampling stage
  |
  v
Snake/SnakeBeta activation
Conv1d(final_ch -> 1, kernel=7)
  |
  v
Output: Waveform [B, 1, T * hop_size]
```

#### Model Configurations

| Model | Params | Sample Rate | Mel Bands | fmax | Hop Size | Upsample Ratio | Upsample Rates |
|-------|--------|-------------|-----------|------|----------|-----------------|----------------|
| BigVGAN-base | 14M | 24 kHz | 100 | 12000 | 256 | 256x | [8, 8, 2, 2] |
| BigVGAN | 112M | 24 kHz | 100 | 12000 | 256 | 256x | [4, 4, 2, 2, 2, 2] |
| BigVGAN-v2 24kHz | 112M | 24 kHz | 100 | 12000 | 256 | 256x | [4, 4, 2, 2, 2, 2] |
| BigVGAN-v2 44kHz/256x | 112M | 44 kHz | 128 | 22050 | 512 | 256x | [4, 4, 2, 2, 2, 2]* |
| BigVGAN-v2 44kHz/512x | 122M | 44 kHz | 128 | 22050 | 512 | 512x | [4, 4, 4, 4, 2]* |

*Exact factorizations for 44kHz models inferred from total ratio and parameter counts.

#### Key Hyperparameters (BigVGAN 112M / 24kHz)

```json
{
    "upsample_initial_channel": 1536,
    "upsample_rates": [4, 4, 2, 2, 2, 2],
    "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "activation": "snakebeta",
    "snake_logscale": true,
    "num_mels": 100,
    "sampling_rate": 24000,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "fmin": 0,
    "fmax": 12000
}
```

For BigVGAN-base (14M):
```json
{
    "upsample_initial_channel": 512,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
}
```

### Snake Activation Function

#### Mathematical Formulation

```
Snake_a(x) = x + (1/a) * sin^2(a * x)
```

Where `a` (alpha) is a trainable per-channel parameter controlling the frequency of the periodic component.

**SnakeBeta variant** (used in v2 pretrained models):
```
SnakeBeta_{a,b}(x) = x + (1/b) * sin^2(a * x)
```

Here alpha controls frequency and beta controls magnitude independently.

#### PyTorch Implementation

```python
import torch
import torch.nn as nn


class SnakeBeta(nn.Module):
    """
    Snake activation with separate alpha (frequency) and beta (magnitude).
    Uses log-scale parameterization for stability.

    Reference: https://arxiv.org/abs/2006.08195
    """
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            # Initialize to log(1) = 0 for both alpha and beta
            self.alpha = nn.Parameter(torch.zeros(in_features))
            self.beta = nn.Parameter(torch.zeros(in_features))
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x):
        # x shape: [B, C, T]
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)    # [1, C, 1]

        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)

        # Snake formula: x + (1/beta) * sin^2(alpha * x)
        return x + (1.0 / (beta + self.eps)) * torch.pow(torch.sin(x * alpha), 2)


class Snake(nn.Module):
    """
    Original Snake activation: x + (1/a) * sin^2(a * x)
    Single parameter controls both frequency and magnitude.
    """
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features))
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]

        if self.alpha_logscale:
            alpha = torch.exp(alpha)

        return x + (1.0 / (alpha + self.eps)) * torch.pow(torch.sin(x * alpha), 2)
```

**Gradient**: `d/dx Snake_a(x) = 1 + sin(2ax)` -- always non-negative, ensuring monotonicity.

### Anti-Aliased Multi-Periodicity Composition (AMP)

#### Why Anti-Aliasing is Needed

Snake activations produce arbitrary high-frequency harmonics in continuous time that cannot be represented at the discrete output sample rate, causing aliasing artifacts. The AMP module suppresses this via low-pass filtering.

#### Implementation: Upsample -> Activate -> Downsample

```
Signal at rate R
    |
    v
Upsample 2x (zero-insert + low-pass filter)
    |
    v
Apply Snake/SnakeBeta activation
    |
    v
Downsample 2x (low-pass filter + decimate)
    |
    v
Signal at rate R (alias-free)
```

#### Kaiser Window Low-Pass Filter Parameters

The windowed sinc filter uses a Kaiser window with:
- Window length: `n = 6 * m` (where m=2 for 2x up/downsampling, so n=12)
- Cutoff frequency: `f_c = s / (2m)` where s = sampling rate
- Transition band half-width: `f_h = 0.6 / m`
- Maximum attenuation: `A = 2.285 * (n/2 - 1) * pi * 4 * f_h + 7.95`
- Shape parameter: `beta = 0.1102 * (A - 8.7)`

```python
import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import kaiser


def design_lowpass_filter(cutoff, num_taps=12, beta=None):
    """Design Kaiser-windowed sinc low-pass filter for anti-aliasing."""
    if beta is None:
        # Compute beta from desired attenuation
        fh = 0.6 / 2  # transition band half-width for 2x resampling
        A = 2.285 * (num_taps / 2 - 1) * np.pi * 4 * fh + 7.95
        beta = 0.1102 * (A - 8.7) if A > 50 else (
            0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21) if A >= 21 else 0.0
        )

    # Windowed sinc filter
    n = np.arange(num_taps)
    sinc = np.sinc(2 * cutoff * (n - (num_taps - 1) / 2))
    window = kaiser(num_taps, beta)
    h = sinc * window
    h /= h.sum()
    return torch.FloatTensor(h)


class Activation1d(nn.Module):
    """Anti-aliased 1D activation: upsample -> activate -> downsample."""

    def __init__(self, activation, up_ratio=2, down_ratio=2,
                 up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.activation = activation
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio

        # Design anti-aliasing filters
        cutoff = 0.5 / up_ratio
        self.up_filter = design_lowpass_filter(cutoff, up_kernel_size)
        self.down_filter = design_lowpass_filter(cutoff, down_kernel_size)

    def forward(self, x):
        # Upsample
        x = F.interpolate(x, scale_factor=self.up_ratio, mode='nearest')
        x = self._apply_filter(x, self.up_filter.to(x.device))

        # Apply activation
        x = self.activation(x)

        # Downsample
        x = self._apply_filter(x, self.down_filter.to(x.device))
        x = x[:, :, ::self.down_ratio]
        return x

    def _apply_filter(self, x, h):
        """Apply 1D FIR filter per channel."""
        B, C, T = x.shape
        padding = len(h) // 2
        h = h.view(1, 1, -1).expand(C, -1, -1)
        return F.conv1d(x, h, padding=padding, groups=C)
```

#### CUDA Fused Kernel (v2)

BigVGAN-v2 provides a fused CUDA kernel combining upsample + activation + downsample for 1.5-3x faster inference on A100 GPUs. Enable with:

```python
model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_24khz_100band_256x',
    use_cuda_kernel=True  # Requires CUDA toolkit + ninja
)
```

### Discriminator Architecture

#### Original BigVGAN: MPD + MRD

**Multi-Period Discriminator (MPD):**
- Reshape periods: [2, 3, 5, 7, 11]
- 2D convolutions per period with kernels [1x1], [3x1], [5x1]

**Multi-Resolution Discriminator (MRD):**
- STFT parameters:
  - n_fft: [1024, 2048, 512]
  - hop_length: [120, 240, 50]
  - win_length: [600, 1200, 240]

#### BigVGAN-v2: Multi-Scale Sub-Band CQT Discriminator

Replaces MRD with a CQT-based discriminator for better frequency resolution at low frequencies. Combined with multi-scale mel spectrogram loss for both coarse and fine-grained quality.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 32 (8x A100), 4 (single A100 fine-tune) |
| Segment size | 65536 samples (v2) |
| Gradient clipping | global norm <= 1000 |
| Training steps | 1M (original), 5M (v2) |
| Loss: Adversarial | Least-square GAN |
| Loss: Feature matching | lambda_fm = 2 |
| Loss: Mel spectrogram | lambda_mel = 45 |

### Performance Benchmarks

#### Quality Metrics (LibriTTS Test Set)

| Model | PESQ | M-STFT | MCD | Periodicity | V/UV F1 |
|-------|------|--------|-----|-------------|---------|
| HiFi-GAN V1 (14M) | 2.947 | 0.9303 | 0.6603 | 0.1018 | 0.9598 |
| BigVGAN-base (14M) | 3.519 | 0.8223 | 0.4564 | 0.0844 | 0.9689 |
| BigVGAN (112M) | **4.027** | **0.7997** | **0.3745** | **0.0711** | **0.9727** |

#### Subjective Quality (SMOS out of 5.0)

| Domain | HiFi-GAN | BigVGAN |
|--------|----------|---------|
| Speech (LibriTTS) | 4.15 +/- 0.09 | **4.26 +/- 0.08** |
| Music (MUSDB18-HQ) | 4.08 +/- 0.05 | **4.26 +/- 0.04** |
| Singing voice | 3.92 +/- 0.06 | **4.18 +/- 0.05** |

#### Inference Speed (RTX 8000)

| Model | Real-time Factor |
|-------|-----------------|
| HiFi-GAN V1 | 93.75x |
| BigVGAN-base (14M) | 70.18x |
| BigVGAN (112M) | 44.72x |
| BigVGAN-v2 (CUDA kernel) | ~67-134x (1.5-3x faster than without) |

### Pretrained Model URLs and Loading

#### Available Pretrained Models on Hugging Face

| Model ID | URL |
|----------|-----|
| bigvgan_v2_44khz_128band_512x | https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x |
| bigvgan_v2_44khz_128band_256x | https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_256x |
| bigvgan_v2_24khz_100band_256x | https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x |
| bigvgan_v2_22khz_80band_256x | https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x |
| bigvgan_24khz_100band | https://huggingface.co/nvidia/bigvgan_24khz_100band |
| bigvgan_base_24khz_100band | https://huggingface.co/nvidia/bigvgan_base_24khz_100band |

#### Loading and Inference Code

```python
import torch
import bigvgan
import librosa

# --- Load pretrained model ---
model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_24khz_100band_256x',
    use_cuda_kernel=False  # Set True if CUDA toolkit available
)
model.remove_weight_norm()
model = model.eval().to('cuda')

# --- Prepare mel spectrogram ---
from bigvgan.meldataset import get_mel_spectrogram

wav_path = "input_audio.wav"
wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True)
wav = torch.FloatTensor(wav).unsqueeze(0)  # [1, T]

# Compute mel using model's config
mel = get_mel_spectrogram(wav, model.h)  # [1, num_mels, T_mel]
mel = mel.to('cuda')

# --- Generate waveform ---
with torch.inference_mode():
    wav_gen = model(mel)  # [1, 1, T_audio]

# Convert to numpy
wav_gen = wav_gen.squeeze().cpu().numpy()

# Save as 16-bit PCM
import soundfile as sf
sf.write("output.wav", wav_gen, model.h.sampling_rate)
```

#### Manual Download (git-lfs)

```bash
git lfs install
git clone https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x

# Files in checkpoint:
# - bigvgan_generator.pt (generator weights)
# - bigvgan_discriminator_optimizer.pt (discriminator + optimizer states)
# - config.json (full architecture config)
```

#### Install BigVGAN Package

```bash
pip install bigvgan
# Or from source:
git clone https://github.com/NVIDIA/BigVGAN.git
cd BigVGAN
pip install -e .
```

### Comparison with HiFi-GAN for Singing Voice

| Aspect | HiFi-GAN | BigVGAN |
|--------|----------|---------|
| Parameters | 14M (V1) | 112M (large), 14M (base) |
| Periodic inductive bias | None | Snake activation |
| Anti-aliasing | None | Kaiser-window low-pass in AMP |
| OOD generalization | Poor (speech-only training) | Strong (zero-shot to music/singing) |
| Singing SMOS | 3.92 | 4.18 |
| Music SMOS | 4.08 | 4.26 |
| High-frequency artifacts | Common | Reduced (but not eliminated) |
| Speed | 93.75x RT | 44.72x RT (67-134x with CUDA kernel) |

**Key advantage for singing voice conversion**: BigVGAN's Snake activation provides natural periodicity modeling crucial for pitched singing, and anti-aliasing prevents spectral artifacts at high frequencies. The v2 model trained on diverse audio (including music) makes it directly applicable without fine-tuning.

### Key Techniques to Implement

1. **Replace HiFi-GAN vocoder with BigVGAN-v2** in the conversion pipeline
2. **Use SnakeBeta with log-scale** parameterization (best quality in ablations)
3. **Anti-aliased activation** via upsample-activate-downsample pattern
4. **Multi-scale mel loss** (lambda=45) during any fine-tuning
5. **Recommended model**: `bigvgan_v2_24khz_100band_256x` for balanced speed/quality, or `bigvgan_v2_44khz_128band_512x` for highest fidelity

---

## 2. Demucs v4 (HTDemucs): Hybrid Transformer Vocal Separation

### Paper References

- **Paper**: "Hybrid Transformers for Music Source Separation" (ICASSP 2023)
  - arXiv: [2211.08553](https://arxiv.org/abs/2211.08553)
  - Authors: Simon Rouard, Francisco Massa, Alexandre Defossez
- **Repository**: https://github.com/facebookresearch/demucs (archived Jan 2025)
- **Active fork**: https://github.com/adefossez/demucs
- **License**: MIT
- **Inference package**: https://pypi.org/project/demucs-infer/

### Architecture Overview

HTDemucs is a hybrid temporal/spectral bi-U-Net where the innermost layers are replaced by a cross-domain Transformer Encoder.

```
                     Input Waveform (44.1kHz stereo)
                            |
              +-------------+-------------+
              |                           |
       [Time Domain]              [Frequency Domain]
       Temporal U-Net              Spectral U-Net
              |                           |
     4 Encoder Layers             4 Encoder Layers
     (1D convolutions)           (2D convolutions on STFT)
              |                           |
              +---> Cross-Domain <--------+
                    Transformer
                    (5 layers)
              |                           |
     4 Decoder Layers             4 Decoder Layers
              |                           |
              +-------------+-------------+
                            |
                   4 Source Stems
            [drums, bass, other, vocals]
```

### Detailed Architecture Specifications

#### U-Net Encoder/Decoder

- **Depth**: 4 outer encoder/decoder layers per branch + 1 innermost Transformer
- **Channels**: Progressive doubling from initial channel count
- **Temporal branch**: 1D convolutions with stride for downsampling
- **Spectral branch**: 2D convolutions on complex STFT representation

#### Cross-Domain Transformer Encoder

| Parameter | Value |
|-----------|-------|
| Transformer depth | 5 layers (7 for sparse variant) |
| Hidden dimension | 384 (512 in larger variant) |
| Attention heads | 8 |
| Feed-forward hidden | 4x transformer dim (1536 or 2048) |
| Layer Scale init | epsilon = 1e-4 |
| Normalization | Layer norm + Time layer norm |
| Attention type | Self-attention (intra-domain) + Cross-attention (inter-domain) |

#### Sparse Attention (Extended Context)

- **Mechanism**: Locally Sensitive Hashing (LSH)
- **LSH rounds**: 32
- **Buckets per round**: 4
- **Sparsity**: 90% of softmax elements removed
- **Benefit**: Extends receptive field to 12.2 seconds during training
- **SDR improvement**: +0.2 dB over dense attention

#### Audio Processing

| Parameter | Value |
|-----------|-------|
| Sample rate | 44.1 kHz |
| Channels | 2 (stereo) |
| Segment length | 3.4s (dense) to 12.2s (sparse) |
| FFT size | 4096 (for spectral branch) |
| Hop length | 1024 |
| Output sources | 4: drums, bass, other, vocals |
| Output sources (6s) | 6: + piano, guitar |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (beta1=0.9, beta2=0.999) |
| Learning rate | 3e-4 (1e-4 for fine-tuning) |
| Batch size | 32 |
| Loss function | L1 on waveforms |
| Training duration | 1200 epochs x 800 batches |
| Data augmentation | Pitch shift, tempo stretch, stem remix |
| Training data | MUSDB18-HQ (150 tracks) + 800 internal songs |
| Fine-tuning | Per-source, 50 epochs |

### SDR Performance (MUSDB HQ Test Set)

#### Per-Source SDR (dB) -- with 800 extra training songs

| Model | All | Drums | Bass | Other | Vocals |
|-------|-----|-------|------|-------|--------|
| Hybrid Demucs v3 | 7.68 | 8.12 | 8.43 | 5.65 | 8.50 |
| HT Demucs | 8.80 | 10.05 | 9.78 | 6.42 | 8.93 |
| HT Demucs (fine-tuned) | 9.00 | 10.08 | 10.39 | 6.32 | 9.20 |
| Sparse HT Demucs + FT | **9.20** | **10.83** | **10.47** | **6.41** | **9.37** |

#### Vocals-Only SDR Comparison

| Model | Vocals SDR |
|-------|-----------|
| Open-Unmix | 6.32 |
| Hybrid Demucs v3 | 8.50 |
| MDX-Net | 8.90 |
| HT Demucs | 8.93 |
| HT Demucs-FT | 9.20 |
| Sparse HT Demucs-FT | **9.37** |

### Available Pretrained Models

| Model ID | Description | Quality | Speed |
|----------|-------------|---------|-------|
| `htdemucs` | Hybrid Transformer Demucs | 9.0 dB SDR | 1x |
| `htdemucs_ft` | Fine-tuned per-source | 9.0 dB SDR | 4x slower |
| `htdemucs_6s` | 6 sources (+ piano, guitar) | Slightly lower | 1x |
| `hdemucs_mmi` | Hybrid Demucs v3 | 7.7 dB SDR | 1x |
| `mdx_extra` | MDX architecture | Competitive | 1x |
| `mdx_extra_q` | Quantized MDX | Competitive | 1x, smaller |

### Python API Usage

#### Method 1: High-Level Separator API (Recommended)

```python
import torch
import demucs.api

# Initialize separator
separator = demucs.api.Separator(
    model="htdemucs_ft",  # Best quality for vocals
    segment=10,            # Segment length in seconds
    overlap=0.25,          # Overlap between segments
    device="cuda",
    shifts=1,              # Random shift averaging (higher = better but slower)
    split=True             # Enable chunked processing
)

# Separate from file
origin, separated = separator.separate_audio_file("song.wav")

# Access individual stems
vocals = separated["vocals"]      # torch.Tensor [channels, samples]
drums = separated["drums"]
bass = separated["bass"]
accompaniment = separated["other"]

# Save stems
demucs.api.save_audio(vocals, "vocals.wav", separator.samplerate)
demucs.api.save_audio(accompaniment, "accompaniment.wav", separator.samplerate)

# Separate from tensor (for pipeline integration)
import torchaudio
waveform, sr = torchaudio.load("song.wav")
origin, separated = separator.separate_tensor(waveform, sr)
```

#### Method 2: Low-Level apply_model API

```python
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Load model
model = get_model("htdemucs_ft")
model.to("cuda")
model.eval()

# Load and prepare audio
wav, sr = torchaudio.load("song.wav")

# Resample if needed (model expects 44.1kHz)
if sr != model.samplerate:
    wav = torchaudio.functional.resample(wav, sr, model.samplerate)

# Ensure stereo
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)

# Normalize
ref = wav.mean(0)
mean = ref.mean()
std = ref.std()
wav = (wav - mean) / std

# Apply model
with torch.no_grad():
    sources = apply_model(
        model,
        wav[None],           # Add batch dim: [1, channels, samples]
        device="cuda",
        shifts=1,            # Time-shift averaging
        split=True,          # Chunk processing for long audio
        overlap=0.25,        # Chunk overlap
        progress=True        # Show progress bar
    )

# Denormalize
sources = sources * std + mean
sources = sources[0]  # Remove batch dim: [num_sources, channels, samples]

# Map to source names
source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
for i, name in enumerate(source_names):
    torchaudio.save(f"{name}.wav", sources[i].cpu(), model.samplerate)

# Get just vocals
vocals_idx = source_names.index("vocals")
vocals = sources[vocals_idx]
```

#### Method 3: TorchAudio Pipeline

```python
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

# Load pre-trained model via torchaudio
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sample_rate = bundle.sample_rate  # 44100

def separate_sources(model, mix, segment=10.0, overlap=0.1, device="cuda"):
    """Process audio in overlapping chunks with fade transitions."""
    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    overlap_frames = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)
    start = 0

    while start < length - overlap_frames:
        end = min(start + chunk_len, length)
        chunk = mix[:, :, start:end]

        with torch.no_grad():
            out = model(chunk.to(device))

        out = fade(out)
        final[:, :, :, start:end] += out

        if start == 0:
            fade.fade_in_len = overlap_frames
        start += chunk_len - overlap_frames

    return final

# Load audio
waveform, sr = torchaudio.load("song.wav")
waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

# Normalize
ref = waveform.mean(0)
waveform_norm = (waveform - ref.mean()) / ref.std()

# Separate
sources = separate_sources(model, waveform_norm[None].to(device))[0]
sources = sources * ref.std() + ref.mean()

# Access stems
audios = dict(zip(model.sources, sources))
vocals = audios["vocals"]
```

#### Method 4: Using demucs-infer (Modern, Lightweight)

```bash
pip install demucs-infer  # PyTorch 2.x compatible, ~50% smaller
```

```python
import torch
import torchaudio
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import save_audio

# Load model
model = get_model("htdemucs_ft")
model.to("cuda")

# Load audio
wav, sr = torchaudio.load("song.wav")
if sr != model.samplerate:
    wav = torchaudio.functional.resample(wav, sr, model.samplerate)

# Separate
sources = apply_model(model, wav[None], device="cuda")
# sources shape: [1, 4, channels, time]

# Save individual stems
for i, name in enumerate(model.sources):
    save_audio(sources[0, i], f"{name}.wav", model.samplerate)
```

### aarch64/ARM Compatibility Considerations

#### Current Status

- **No official aarch64 pre-built wheels** for torchaudio (required dependency)
- The Demucs Python package itself is architecture-independent
- PyTorch for Jetson requires NVIDIA's custom wheels matching JetPack version

#### Setup on Jetson/aarch64

```bash
# 1. Install NVIDIA's PyTorch wheel for Jetson
# (Get URL from https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
pip install torch-2.x.x-cp312-cp312-linux_aarch64.whl

# 2. Build torchaudio from source
git clone https://github.com/pytorch/audio.git
cd audio
git checkout v2.x.x  # Match PyTorch version
BUILD_SOX=1 pip install -e . --no-build-isolation

# 3. Install audio dependencies
sudo apt-get install ffmpeg libavformat-dev libavcodec-dev libavutil-dev

# 4. Install Demucs
pip install demucs
# Or for lightweight inference only:
pip install demucs-infer

# 5. Verify
python -c "import demucs; from demucs.pretrained import get_model; print(get_model('htdemucs'))"
```

#### Memory Considerations on Embedded Devices

```python
# Reduce GPU memory usage
separator = demucs.api.Separator(
    model="htdemucs",      # Use non-FT model (less memory)
    segment=7.8,           # Shorter segments
    overlap=0.1,           # Less overlap
    device="cuda",
    split=True
)

# Or set environment variable
# PYTORCH_NO_CUDA_MEMORY_CACHING=1 python separate.py
```

#### Performance Expectations on Jetson

- GPU inference should work with reduced segment sizes
- Expect 2-5x slower than desktop GPU due to memory bandwidth
- Consider TensorRT optimization for production deployment
- Monitor GPU memory: htdemucs requires ~3-7 GB depending on segment length

### Integration Pattern for Singing Voice Conversion

```python
import torch
import torchaudio
import demucs.api
import bigvgan

class VocalSeparationPipeline:
    """
    Pipeline: Input song -> Vocal separation -> Voice conversion -> Mix back
    """

    def __init__(self, device="cuda"):
        self.device = device

        # Initialize vocal separator
        self.separator = demucs.api.Separator(
            model="htdemucs_ft",
            segment=10,
            overlap=0.25,
            device=device,
            shifts=1,
            split=True
        )

        # Initialize vocoder for synthesis
        self.vocoder = bigvgan.BigVGAN.from_pretrained(
            'nvidia/bigvgan_v2_24khz_100band_256x',
            use_cuda_kernel=False
        )
        self.vocoder.remove_weight_norm()
        self.vocoder.eval().to(device)

    def separate_vocals(self, audio_path: str):
        """Extract vocals and accompaniment from a song."""
        origin, separated = self.separator.separate_audio_file(audio_path)
        return {
            "vocals": separated["vocals"],
            "accompaniment": separated["drums"] + separated["bass"] + separated["other"],
            "drums": separated["drums"],
            "bass": separated["bass"],
            "other": separated["other"],
            "sample_rate": self.separator.samplerate  # 44100
        }

    def synthesize_with_vocoder(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform using BigVGAN."""
        mel = mel_spectrogram.to(self.device)
        with torch.inference_mode():
            waveform = self.vocoder(mel)
        return waveform.squeeze(1)  # [B, T]

    def remix(self, converted_vocals: torch.Tensor, accompaniment: torch.Tensor,
              vocal_gain: float = 1.0) -> torch.Tensor:
        """Mix converted vocals back with accompaniment."""
        # Ensure same length
        min_len = min(converted_vocals.shape[-1], accompaniment.shape[-1])
        vocals = converted_vocals[..., :min_len] * vocal_gain
        acc = accompaniment[..., :min_len]
        return vocals + acc


# Usage example
pipeline = VocalSeparationPipeline(device="cuda")

# Step 1: Separate vocals
stems = pipeline.separate_vocals("input_song.wav")
vocals = stems["vocals"]           # [2, samples] at 44.1kHz
accompaniment = stems["accompaniment"]  # [2, samples] at 44.1kHz

# Step 2: Convert vocals through your voice conversion model
# (extract features, convert, get mel spectrogram)
# converted_mel = voice_conversion_model(vocals)

# Step 3: Synthesize with BigVGAN
# converted_audio = pipeline.synthesize_with_vocoder(converted_mel)

# Step 4: Remix
# final = pipeline.remix(converted_audio, accompaniment, vocal_gain=0.9)
# torchaudio.save("output.wav", final.cpu(), 44100)
```

### Key Techniques to Implement

1. **Use `htdemucs_ft` model** for best vocal separation quality (9.20 dB SDR)
2. **Chunked processing** with overlap and fade for memory-efficient long audio
3. **Normalize before separation**, denormalize after (critical for quality)
4. **Shifts parameter** (1-5): time-shift averaging trades speed for quality
5. **Keep all stems** for flexible remixing (drums/bass/other/vocals)
6. **Use `demucs-infer`** package for production (lighter, PyTorch 2.x compatible)

---

## 3. Integration Notes for AutoVoice Pipeline

### Recommended Pipeline Order

```
Input Song (44.1kHz stereo)
    |
    v
[Demucs HTDemucs-FT] -- Vocal Separation
    |
    +-- vocals.wav (44.1kHz stereo)
    +-- accompaniment.wav (44.1kHz stereo, drums+bass+other)
    |
    v
[Resample vocals to 24kHz mono] -- for voice conversion
    |
    v
[Feature Extraction] -- HuBERT/ContentVec + F0
    |
    v
[Voice Conversion Model] -- So-VITS-SVC / your encoder
    |
    v
[Mel Spectrogram] -- converted voice representation
    |
    v
[BigVGAN-v2 24kHz] -- Neural Vocoder Synthesis
    |
    v
[Resample back to 44.1kHz stereo]
    |
    v
[Remix with accompaniment]
    |
    v
Output Song (44.1kHz stereo)
```

### Sample Rate Considerations

- Demucs operates at 44.1kHz stereo (fixed)
- BigVGAN-v2 24kHz model: good quality, faster, compatible with most SVC models
- BigVGAN-v2 44kHz model: highest quality, matches Demucs output directly
- Resampling between stages should use high-quality sinc interpolation

### GPU Memory Budget (Estimated)

| Component | VRAM Usage |
|-----------|-----------|
| Demucs HTDemucs-FT | 3-7 GB (segment-dependent) |
| BigVGAN-v2 24kHz (112M) | ~1.5 GB |
| BigVGAN-v2 44kHz (122M) | ~2.0 GB |
| Voice conversion model | ~1-3 GB |
| **Total (sequential)** | **~5-10 GB peak** |

### Dependencies

```bash
# Core
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bigvgan
pip install demucs-infer  # or: pip install demucs

# Audio processing
pip install librosa soundfile scipy numpy

# Optional: CUDA kernel acceleration for BigVGAN
# Requires: nvcc, ninja
pip install ninja
```

---

## References

1. Lee, S., Ping, W., Ginsburg, B., Catanzaro, B., Yoon, S. (2023). "BigVGAN: A Universal Neural Vocoder with Large-Scale Training." ICLR 2023. arXiv:2206.04658
2. Rouard, S., Massa, F., Defossez, A. (2023). "Hybrid Transformers for Music Source Separation." ICASSP 2023. arXiv:2211.08553
3. Ziyin, L., Hartwig, T., Ueda, M. (2020). "Neural Networks Fail to Learn Periodic Functions and How to Fix It." arXiv:2006.08195
4. Kong, J., Kim, J., Bae, J. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." NeurIPS 2020.
5. NVIDIA BigVGAN-v2 release (July 2024): https://github.com/NVIDIA/BigVGAN
6. Demucs-infer package: https://pypi.org/project/demucs-infer/
