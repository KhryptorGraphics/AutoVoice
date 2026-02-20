# DeepWiki: BigVGAN v2 Architecture Analysis

## Research Source
- DeepWiki MCP queries on `NVIDIA/BigVGAN`
- Topics: wiki structure, Snake activation, model configs, inference usage

---

## 1. Repository Structure

| Section | Contents |
|---------|----------|
| Architecture | Generator, Discriminators, Anti-Aliased Activation |
| Data Processing | Mel Spectrogram Generation, Dataset Handling |
| Usage | Training Guide, Inference Guide |
| CUDA Optimizations | Fused Kernel Implementation |
| Testing/Validation | Torch vs CUDA kernel verification |
| License/Model Card | MIT license |

---

## 2. Snake Activation & Anti-Aliased Multi-Periodicity

### Snake Activation Function
Periodic activation with trainable frequency parameter:

```python
# Snake: sin-based periodic activation
def snake_forward(x, alpha, beta):
    # alpha/beta stored in log-scale, then exponentiated
    alpha = torch.exp(log_alpha)
    beta = torch.exp(log_beta)
    return x + (1.0 / beta) * torch.sin(x * alpha) ** 2
```

- **Snake**: Uses same alpha for both frequency and amplitude
- **SnakeBeta**: Separate beta parameter (recommended for BigVGAN, improved quality)

### AMP Blocks (Anti-aliased Multi-Periodicity)

Two variants: `AMPBlock1` (3 residual layers) and `AMPBlock2` (2 residual layers).

Each integrates `Activation1d` which performs a 3-step pipeline:
```
Upsample (2x) → Snake/SnakeBeta Activation → Downsample (2x)
```

This prevents aliasing artifacts from the periodic activation by:
1. Upsampling to increase Nyquist frequency
2. Applying periodic activation in oversampled domain
3. Downsampling with anti-aliasing filter

### Fused CUDA Kernel
- Combines all 3 operations (upsample + activate + downsample) into single kernel
- Processes data in-register, eliminating intermediate memory writes
- **1.5-3x speedup** on A100 compared to PyTorch implementation
- **Inference only** (not used during training)
- Requires `nvcc` + `ninja` at runtime for JIT compilation

---

## 3. Model Configurations

| Model | Sample Rate | Mel Bands | Hop Size | Upsample Ratios | Upsample Kernels |
|-------|------------|-----------|----------|-----------------|------------------|
| `bigvgan_v2_24khz_100band_256x` | 24000 Hz | 100 | 256 | [4, 4, 2, 2, 2, 2] | [8, 8, 4, 4, 4, 4] |
| `bigvgan_v2_44khz_128band_512x` | 44100 Hz | 128 | 512 | [8, 4, 4, 2, 2, 2] | [16, 8, 8, 4, 4, 4] |
| `bigvgan_v2_44khz_128band_256x` | 44100 Hz | 128 | 256 | [4, 4, 4, 2, 2, 2] | [8, 8, 8, 4, 4, 4] |
| `bigvgan_v2_22khz_80band_256x` | 22050 Hz | 80 | 256 | [4, 4, 2, 2, 2, 2] | [8, 8, 4, 4, 4, 4] |
| `bigvgan_v2_22khz_80band_fmax8k_256x` | 22050 Hz | 80 (fmax=8k) | 256 | [4, 4, 2, 2, 2, 2] | [8, 8, 4, 4, 4, 4] |

### Recommended for AutoVoice:
- **`bigvgan_v2_24khz_100band_256x`**: Best balance of quality and speed for voice
- **`bigvgan_v2_44khz_128band_512x`**: Highest quality for final output (singing)

---

## 4. Inference Usage

### Python API:
```python
import bigvgan
import torch

# Load pretrained model from HuggingFace
model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_24khz_100band_256x',
    use_cuda_kernel=False  # Set True if nvcc+ninja available
)

# Prepare for inference
model.remove_weight_norm()
model = model.eval().to('cuda')

# Input: mel spectrogram [B, n_mels, T_frames]
mel = torch.randn(1, 100, 200).to('cuda')  # Example

# Generate waveform
with torch.inference_mode():
    wav = model(mel)  # Output: [B, 1, T_samples]
    # T_samples = T_frames * hop_size (256 for 24kHz model)

# Convert to numpy
wav_np = wav.squeeze().cpu().numpy()  # [-1, 1] float range
wav_int16 = (wav_np * 32767).astype('int16')  # For WAV file
```

### CLI (for batch processing):
```bash
python inference_e2e.py \
    --checkpoint_file nvidia/bigvgan_v2_24khz_100band_256x \
    --input_mels_dir /path/to/mel_npy_files/ \
    --output_dir /path/to/output_wavs/
```

### Input/Output Specifications:
| | Format | Shape | Range |
|-|--------|-------|-------|
| Input | Mel spectrogram | `[B, C_mel, T_frame]` | Log-scale |
| Output | Waveform | `[B, 1, T_time]` | `[-1, 1]` float |

Where `T_time = T_frame * hop_size`

---

## 5. Integration with AutoVoice (AV-006)

### BigVGANVocoder Class Design:
```python
class BigVGANVocoder(nn.Module):
    """BigVGAN v2 vocoder wrapper matching HiFiGAN interface."""

    def __init__(self, model_name: str = 'nvidia/bigvgan_v2_24khz_100band_256x',
                 use_cuda_kernel: bool = False):
        super().__init__()
        import bigvgan
        self.model = bigvgan.BigVGAN.from_pretrained(
            model_name, use_cuda_kernel=use_cuda_kernel
        )
        self.model.remove_weight_norm()
        self.sample_rate = self.model.h.sampling_rate
        self.hop_size = self.model.h.hop_size
        self.n_mels = self.model.h.num_mels

    @torch.inference_mode()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: [B, n_mels, T] mel spectrogram

        Returns:
            [B, 1, T*hop_size] waveform in [-1, 1]
        """
        return self.model(mel)

    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """Alias matching HiFiGANVocoder interface."""
        return self.forward(mel)
```

### Drop-in Replacement Strategy:
- BigVGAN uses identical input format to HiFiGAN: `[B, n_mels, T]`
- Output format also matches: `[B, 1, T*hop_size]`
- Can replace in `ModelManager` by swapping vocoder class
- Need to adjust `n_mels` (HiFiGAN: 80, BigVGAN-24k: 100) and mel computation

### TensorRT Optimization:
```bash
# Export to ONNX
python -c "
import bigvgan, torch
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x')
model.remove_weight_norm()
model.eval()
dummy_mel = torch.randn(1, 100, 100)
torch.onnx.export(model, dummy_mel, 'bigvgan.onnx',
    input_names=['mel'], output_names=['audio'],
    dynamic_axes={'mel': {2: 'time'}, 'audio': {2: 'samples'}})
"

# Convert with trtexec
/usr/src/tensorrt/bin/trtexec \
    --onnx=bigvgan.onnx \
    --saveEngine=bigvgan.trt \
    --fp16 \
    --minShapes=mel:1x100x10 \
    --optShapes=mel:1x100x100 \
    --maxShapes=mel:1x100x500
```

### Dependency:
```bash
pip install bigvgan  # or clone NVIDIA/BigVGAN
```
