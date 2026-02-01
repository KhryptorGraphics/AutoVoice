# Seed-VC Architecture Documentation

**Research Date:** 2026-01-31
**Agent:** seed-vc-researcher (Phase 1)
**Source:** [Plachta/Seed-VC](https://github.com/Plachta/Seed-VC) - arXiv:2411.09943

## Overview

Seed-VC is a zero-shot voice conversion system based on Diffusion Transformer (DiT) and Conditional Flow Matching (CFM). It supports:
- Zero-shot voice conversion (VC)
- Zero-shot singing voice conversion (SVC)
- Real-time voice conversion
- Fine-tuning on custom data (1 utterance minimum, 100 steps minimum)

## Model Variants

| Version | Name | Purpose | Sample Rate | Content Encoder | Vocoder | Params |
|---------|------|---------|-------------|-----------------|---------|--------|
| v1.0 | seed-uvit-tat-xlsr-tiny | Real-time VC | 22050 Hz | XLSR-large | HIFT | 25M |
| v1.0 | seed-uvit-whisper-small-wavenet | Offline VC | 22050 Hz | Whisper-small | BigVGAN | 98M |
| **v1.0** | **seed-uvit-whisper-base-f0-44k** | **Singing VC** | **44100 Hz** | **Whisper-small** | **BigVGAN** | **200M** |
| v2.0 | hubert-bsqvae-small | Voice + Accent | 22050 Hz | ASTRAL | BigVGAN | 67M+90M |

**For AutoVoice QUALITY_PIPELINE:** Use `seed-uvit-whisper-base-f0-44k` (DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth)

## Architecture Components

### 1. Content Encoder (Whisper)

```
Audio (16kHz) → Whisper Encoder → Semantic Features (768-dim)
```

- **Model:** `openai/whisper-small` (from HuggingFace)
- **Input:** 16kHz audio waveform
- **Output:** 768-dimensional semantic embeddings
- **Purpose:** Extract speaker-independent linguistic content

### 2. Style Encoder (CAMPPlus)

```
Reference Audio → Mel Spectrogram → CAMPPlus → Style Embedding (192-dim)
```

- **Model:** `campplus_cn_common.bin` (FunASR)
- **Architecture:** FCM + TDNN + DenseBlocks
- **Input:** 80-band mel spectrogram
- **Output:** 192-dimensional speaker style embedding
- **Purpose:** Capture speaker timbre and style

### 3. F0 Extractor (RMVPE)

```
Audio (16kHz) → RMVPE → F0 Contour
```

- **Model:** `rmvpe.pt` from `lj1995/VoiceConversionWebUI`
- **Purpose:** Extract pitch contour for singing voice
- **Bins:** 256 F0 bins for pitch conditioning
- **Used when:** `f0_condition=True` (SVC mode)

### 4. Length Regulator

```
Content Features → InterpolateRegulator → Aligned Features
```

- **Module:** `InterpolateRegulator`
- **Channels:** 768
- **Sampling ratios:** [1, 1, 1, 1]
- **F0 conditioning:** 256 bins when enabled

### 5. Diffusion Transformer (DiT)

```
Noise + Content + Style + F0 → DiT → Predicted Mel
```

- **Architecture:** Transformer with U-ViT skip connections
- **Hidden dim:** 768
- **Heads:** 12
- **Depth:** 17 layers
- **Block size:** 8192 tokens
- **Input channels:** 128 (mel bands)
- **Features:**
  - Adaptive Layer Normalization
  - RoPE (Rotary Position Embedding)
  - U-ViT skip connections (layer n/2)
  - Classifier-Free Guidance (CFG)

### 6. Conditional Flow Matching (CFM)

```
Noise → Euler Solver (n steps) → Mel Spectrogram
```

- **Sigma min:** 1e-6
- **Inference steps:** 25-50 (quality) or 4-10 (real-time)
- **CFG rate:** 0.7 (controls speaker similarity)
- **Temperature:** 1.0

### 7. Vocoder (BigVGAN)

```
Mel Spectrogram → BigVGAN → Waveform (44.1kHz)
```

- **Model:** `nvidia/bigvgan_v2_44khz_128band_512x`
- **Input:** 128-band mel spectrogram
- **Output:** 44.1kHz waveform
- **Features:** Alias-free activations (CUDA optimized)

## Data Flow (SVC Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SEED-VC SVC PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

Source Audio (any SR)          Reference Audio (1-30s)
        │                              │
        ▼                              ▼
   Resample to 16kHz              Resample + Mel
        │                              │
        ▼                              ▼
┌───────────────┐             ┌───────────────┐
│   Whisper     │             │   CAMPPlus    │
│   Encoder     │             │   (192-dim)   │
└───────────────┘             └───────────────┘
        │                              │
        ▼                              │
   768-dim content                     │
        │                              │
        ▼                              │
┌───────────────┐                      │
│    RMVPE      │                      │
│  (F0 + 256b)  │                      │
└───────────────┘                      │
        │                              │
        ▼                              │
┌───────────────┐                      │
│   Length      │                      │
│  Regulator    │                      │
└───────────────┘                      │
        │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Diffusion Transformer  │
        │   (DiT + CFM)            │
        │   - 17 layers            │
        │   - U-ViT skips          │
        │   - 25-50 steps          │
        └──────────────────────────┘
                       │
                       ▼
              128-band mel (44.1kHz)
                       │
                       ▼
        ┌──────────────────────────┐
        │        BigVGAN           │
        │   (nvidia/bigvgan_v2)    │
        └──────────────────────────┘
                       │
                       ▼
              Output: 44.1kHz Waveform
```

## Configuration Reference

From `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`:

```yaml
preprocess_params:
  sr: 44100
  spect_params:
    n_fft: 2048
    win_length: 2048
    hop_length: 512
    n_mels: 128

model_params:
  vocoder:
    type: "bigvgan"
    name: "nvidia/bigvgan_v2_44khz_128band_512x"

  speech_tokenizer:
    type: 'whisper'
    name: "openai/whisper-small"

  style_encoder:
    dim: 192
    campplus_path: "campplus_cn_common.bin"

  length_regulator:
    channels: 768
    f0_condition: true
    n_f0_bins: 256

  DiT:
    hidden_dim: 768
    num_heads: 12
    depth: 17
    in_channels: 128
    uvit_skip_connection: true
```

## Integration Points for AutoVoice

### Required Downloads (HuggingFace)

1. **Seed-VC Checkpoint:**
   - Repo: `Plachta/Seed-VC`
   - File: `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth`

2. **Whisper Encoder:**
   - Repo: `openai/whisper-small`
   - Auto-downloaded by transformers

3. **CAMPPlus Style Encoder:**
   - Repo: `funasr/campplus`
   - File: `campplus_cn_common.bin`
   - **Already present:** `/home/kp/repo2/autovoice/models/seed-vc/campplus_cn_common.bin`

4. **RMVPE Pitch Extractor:**
   - Repo: `lj1995/VoiceConversionWebUI`
   - File: `rmvpe.pt`

5. **BigVGAN Vocoder:**
   - Repo: `nvidia/bigvgan_v2_44khz_128band_512x`
   - Auto-downloaded by bigvgan module

### Python API (Inference)

```python
from models.seed_vc.inference import load_models, convert_voice

# Load with F0 conditioning for SVC
args = argparse.Namespace(
    f0_condition=True,
    checkpoint=None,  # Auto-download from HF
    config=None,
    fp16=True
)
models = load_models(args)

# Convert
output = convert_voice(
    source_audio,
    reference_audio,
    diffusion_steps=30,
    inference_cfg_rate=0.7,
    semi_tone_shift=0
)
```

### Key Inference Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| diffusion_steps | 25 | 4-50 | Quality vs speed tradeoff |
| inference_cfg_rate | 0.7 | 0.0-1.0 | Speaker similarity strength |
| semi_tone_shift | 0 | -12 to +12 | Pitch shift in semitones |
| temperature | 1.0 | 0.5-1.5 | Sampling randomness |
| length_adjust | 1.0 | 0.5-2.0 | Speed adjustment |

## Memory Requirements

- **GPU VRAM:** ~8GB for inference (FP16)
- **Model weights:** ~1.5GB
- **BigVGAN:** ~500MB
- **Whisper-small:** ~460MB

## References

- **Paper:** [arXiv:2411.09943](https://arxiv.org/abs/2411.09943)
- **Demo:** [HuggingFace Space](https://huggingface.co/spaces/Plachta/Seed-VC)
- **Weights:** [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC)

---

_Generated by seed-vc-researcher agent for AutoVoice QUALITY_PIPELINE integration_
