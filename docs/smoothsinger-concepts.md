# SmoothSinger Concepts for AutoVoice Integration

**Research Document** | Agent: smoothsinger-researcher | Date: 2026-02-01

## Executive Summary

SmoothSinger (arXiv:2506.21478v1, June 2025) introduces three key innovations for singing voice synthesis that can enhance AutoVoice's quality pipeline:

1. **Multi-Resolution (MR) Module** - Parallel low-frequency upsampling path for better pitch contours
2. **Reference-Guided Dual-Branch Architecture** - Context-aware denoising using reference audio
3. **Vocoder-Free Synthesis** - End-to-end waveform generation mitigating two-stage artifacts

These concepts address common artifacts in voice conversion and can improve our QUALITY_PIPELINE's fidelity.

---

## 1. Multi-Resolution (MR) Module

### Problem Addressed
Traditional U-Net architectures use sequential upsampling, where each stage predicts finer-resolution features from the previous coarser level. This sequential design can lose low-frequency information critical for pitch contours and tonal stability in singing voices.

### SmoothSinger Solution

The MR module introduces a **parallel, non-sequential low-frequency upsampling path**:

```
Standard U-Net (Sequential):
   Downsampled Features → LVCUp Block 1 → LVCUp Block 2 → LVCUp Block 3 → Output
                              ↑               ↑               ↑
                          Skip Connection   Skip Connection   Skip Connection

SmoothSinger MR Module (Parallel):
   Downsampled Features → LowF Block 1 ────────────────────────────→ ⊕ → Output
                       → LowF Block 2 ────────────────────────────→ ⊕
                       → LowF Block 3 ────────────────────────────→ ⊕
                       → LVCUp Blocks (standard path) ─────────────→ ⊕
```

### Key Implementation Details

```python
# Pseudocode for MR Module
class LowFrequencyBlock(nn.Module):
    """Non-sequential low-frequency upsampling block.

    Features:
    - 1D convolutions for frequency processing
    - Sliding window self-attention (O(L) complexity for long audio)
    - Upsampling via hidden size increase + reshape (8x factor)
    """
    def __init__(self, in_channels, out_channels, window_size=512):
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3)
        self.attention = SlidingWindowAttention(hidden_dim, window_size)
        self.conv2 = nn.Conv1d(hidden_dim, out_channels * 8, kernel_size=3)  # 8x upsample

    def forward(self, x, condition):
        # Process at current resolution
        x = F.leaky_relu(self.conv1(x))
        x = self.attention(x)  # Local attention, not global
        x = self.conv2(x)
        # Reshape to upsample: (B, C*8, L) -> (B, C, L*8)
        return x.reshape(x.size(0), -1, x.size(2) * 8)
```

### Benefits for AutoVoice

| Aspect | Standard U-Net | With MR Module |
|--------|---------------|----------------|
| Low-frequency preservation | Sequential loss | Direct path to output |
| Pitch contour modeling | Indirect through layers | Dedicated low-freq branch |
| Long-range dependencies | Limited by receptive field | Sliding window attention |
| Computational complexity | O(L²) for global attention | O(L) with sliding window |

### Integration Points

For AutoVoice's QUALITY_PIPELINE:
1. Add MR module as parallel branch in Seed-VC decoder
2. Concatenate MR output with standard upsampling output
3. Use zero-initialized 1×1 conv to blend (prevents disrupting pretrained weights)

---

## 2. Reference-Guided Dual-Branch Architecture

### Problem Addressed
Voice conversion systems typically lack direct acoustic grounding during synthesis. The model only sees symbolic features (content, pitch, speaker embedding) but not actual acoustic patterns.

### SmoothSinger Solution

A **duplicate downsampling branch** processes reference audio in parallel:

```
Main Branch (Diffusion):                Reference Branch:
   Noisy Audio                           Reference Audio (from baseline model)
        ↓                                      ↓
   DownBlock[0] → X₀ ←──── concat ─────→ DownBlock[0] → Y₀
        ↓                                      ↓
   DownBlock[1] → X₁ ←──── concat ─────→ DownBlock[1] → Y₁
        ↓                                      ↓
   DownBlock[2] → X₂ ←──── concat ─────→ DownBlock[2] → Y₂
        ↓                                      ↓
   Bottleneck                            (features flow to main)
        ↓
   Upsampling Path
```

### Feature Fusion Mechanism

```python
def fuse_features(main_features, ref_features):
    """Zero-initialized fusion for gradual learning."""
    # Zero-init 1x1 conv ensures ref features don't disrupt early training
    fused = zero_init_conv(torch.cat([main_features, ref_features], dim=1))
    return fused

# Propagation with reference
x_next = fuse_features(X_i, Y_i)  # Fed to next downblock AND upblock
```

### Comparison with ControlNet

| Aspect | ControlNet | SmoothSinger Reference |
|--------|------------|------------------------|
| Injection point | Upsampling only | Both downsampling AND upsampling |
| Feature richness | Limited | Richer multi-resolution features |
| Original weights | Frozen | Independent weight update |

### Integration Points

For AutoVoice:
1. Use converted audio from REALTIME_PIPELINE as reference for QUALITY_PIPELINE
2. This creates a **two-pass refinement**:
   - Pass 1: Fast conversion with REALTIME (ContentVec + HiFiGAN)
   - Pass 2: Quality refinement with reference-guided diffusion

---

## 3. Vocoder-Free Synthesis

### Problem Addressed
Two-stage pipelines (acoustic model → vocoder) suffer from:
1. Distribution mismatch between training and inference spectrograms
2. Vocoder artifacts (background noise, crackling, distortion)
3. Ground-truth spectrogram reconstruction still loses information

### SmoothSinger Solution

**End-to-end waveform synthesis** using diffusion directly on the audio signal:

```
Traditional Pipeline:
   Score → Acoustic Model → Mel-Spectrogram → Vocoder → Waveform
                                   ↑
                           (mismatch here)

SmoothSinger:
   Score → FastSpeech2 → Low-Quality Audio → SmoothSinger → High-Quality Waveform
                              ↑                     ↑
                        (as reference)        (diffusion on waveform)
```

### Training with Degraded Audio

To address temporal misalignment between reference and target:

```python
def degrade_audio(audio, regions=None):
    """Apply random degradation for training alignment."""
    if regions is None:
        num_regions = random.randint(3, 10)
        regions = [random_segment(len(audio), 500, 2000) for _ in range(num_regions)]

    degraded = audio.clone()
    for region in regions:
        # Randomly apply one or more:
        degraded[region] = add_noise(degraded[region], alpha=uniform(0.8, 1.05))
        degraded[region] = adjust_amplitude(degraded[region], beta=uniform(0.8, 1.05))
        degraded[region] = apply_distortion(degraded[region], gamma=uniform(0.9, 1.2))
        degraded[region] = change_frequency_strength(degraded[region])

    return degraded
```

**Training Schedule:**
- Steps 0-1.2M: Train with FastSpeech2 + HiFiGAN reference
- Steps 1.2M-1.6M: 50% probability use degraded ground-truth as reference

### Integration Points

For AutoVoice's unified architecture:
1. Consider replacing HQ-SVC super-resolution with SmoothSinger-style refinement
2. Use degraded training for adapter fine-tuning (more robust to input quality)

---

## 4. Architecture Summary

### Complete SmoothSinger Architecture

```
                              ┌─────────────────────────────────────┐
                              │     SMOOTHSINGER ARCHITECTURE       │
                              └─────────────────────────────────────┘
                                              │
    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    ▼                                                                       ▼
┌──────────────┐                                                   ┌──────────────┐
│  Diffusion   │                                                   │  Reference   │
│   Module     │                                                   │   Module     │
│              │                                                   │              │
│ Input: Noisy │                                                   │ Input: Ref   │
│   Waveform   │                                                   │   Audio      │
└──────┬───────┘                                                   └──────┬───────┘
       │                                                                   │
       ▼                                                                   ▼
┌──────────────┐    concat + zero-init conv                       ┌──────────────┐
│ DownBlock[0] │◄──────────────────────────────────────────────────│ DownBlock[0] │
│   stride=8   │                                                   │   stride=8   │
└──────┬───────┘                                                   └──────┬───────┘
       │                                                                   │
       ▼                                                                   ▼
┌──────────────┐                                                   ┌──────────────┐
│ DownBlock[1] │◄──────────────────────────────────────────────────│ DownBlock[1] │
│   stride=8   │                                                   │   stride=8   │
└──────┬───────┘                                                   └──────┬───────┘
       │                                                                   │
       ▼                                                                   ▼
┌──────────────┐                                                   ┌──────────────┐
│ DownBlock[2] │◄──────────────────────────────────────────────────│ DownBlock[2] │
│   stride=4   │                                                   │   stride=4   │
└──────┬───────┘                                                   └──────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              UPSAMPLING PATH                                  │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │  LVCUp[0]   │───▶│  LVCUp[1]   │───▶│  LVCUp[2]   │──┐                   │
│  │  (standard) │    │  (standard) │    │  (standard) │  │                   │
│  └─────────────┘    └─────────────┘    └─────────────┘  │                   │
│                                                          │                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │    ┌───────────┐ │
│  │  LowF[0]    │────────────────────────────────────────┼───▶│   OUTPUT  │ │
│  │  (parallel) │    │  LowF[1]    │────────────────────┼┼───▶│  (concat) │ │
│  └─────────────┘    │  (parallel) │    │  LowF[2]    │─┼┼┼──▶│           │ │
│                     └─────────────┘    │  (parallel) │ │││   └───────────┘ │
│                                        └─────────────┘ │││                  │
│                                                        │││                  │
│                              MR MODULE (non-sequential)│││                  │
└────────────────────────────────────────────────────────┴┴┴──────────────────┘
```

### Model Specifications

| Component | Specification |
|-----------|--------------|
| Downsampling strides | [8, 8, 4] → 256x compression |
| Total parameters | ~16M trainable |
| Sample rate | 24 kHz |
| Training time | ~72 hours on single RTX 4090 |
| Diffusion steps | 24 (optimal quality/speed) |
| Sliding window size | 512 samples for attention |

---

## 5. Integration Recommendations for AutoVoice

### Phase 4 Implementation Tasks

Based on the plan.md Phase 4 requirements:

#### Task 4.1: Multi-Resolution Frequency Branch in Decoder

```python
# src/auto_voice/inference/mr_module.py

class MultiResolutionModule(nn.Module):
    """Parallel low-frequency upsampling for Seed-VC decoder."""

    def __init__(self, channels_per_stage, window_size=512):
        super().__init__()
        self.stages = nn.ModuleList([
            LowFrequencyBlock(ch, ch, window_size)
            for ch in channels_per_stage
        ])
        self.fusion = nn.Conv1d(
            sum(channels_per_stage),
            channels_per_stage[-1],
            kernel_size=1
        )
        nn.init.zeros_(self.fusion.weight)  # Zero-init for gradual blending

    def forward(self, multi_scale_features, condition):
        outputs = []
        for stage, features in zip(self.stages, multi_scale_features):
            outputs.append(stage(features, condition))
        # All outputs at original resolution, concatenate and fuse
        return self.fusion(torch.cat(outputs, dim=1))
```

#### Task 4.2: Low-Frequency Upsampling Path

Key insight: Use **reshape-based upsampling** instead of transposed convolutions to avoid checkerboard artifacts:

```python
def reshape_upsample(x, factor=8):
    """Artifact-free upsampling via channel-to-time reshape.

    Instead of: ConvTranspose1d (prone to checkerboard)
    Use: Conv1d to expand channels, then reshape
    """
    B, C, L = x.shape
    # Expand channels
    x = conv_expand(x)  # (B, C, L) -> (B, C*factor, L)
    # Reshape to time domain
    return x.reshape(B, C, L * factor)  # (B, C, L*factor)
```

#### Task 4.3: Sliding Window Attention

For long audio sequences (>30s), use Longformer-style attention:

```python
class SlidingWindowAttention(nn.Module):
    """O(L) attention for long audio sequences.

    Reference: Longformer (Beltagy et al., 2020)
    """
    def __init__(self, dim, window_size=512):
        super().__init__()
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Sliding window attention
        attn_output = torch.zeros_like(x)
        half_window = self.window_size // 2

        for i in range(0, L, half_window):
            start = max(0, i - half_window)
            end = min(L, i + half_window)

            q_local = q[:, i:min(i+half_window, L), :]
            k_local = k[:, start:end, :]
            v_local = v[:, start:end, :]

            attn = torch.softmax(q_local @ k_local.transpose(-2, -1) / (C ** 0.5), dim=-1)
            attn_output[:, i:min(i+half_window, L), :] = attn @ v_local

        return self.proj(attn_output)
```

---

## 6. Expected Quality Improvements

### Ablation Study Results (from paper)

| Configuration | MOS Score | BAK MOS |
|--------------|-----------|---------|
| Full Model | **3.61 ± 0.06** | **4.08 ± 0.06** |
| Without Reference Module | 3.49 ± 0.06 | 4.05 ± 0.06 |
| Without MR Module | 3.31 ± 0.07 | 3.98 ± 0.07 |
| Without Degraded Training | 3.16 ± 0.07 | 3.72 ± 0.08 |

### Key Takeaways

1. **MR Module provides +0.30 MOS** - Significant improvement from parallel low-frequency path
2. **Reference Module provides +0.12 MOS** - Context-aware synthesis helps
3. **Degraded Training provides +0.45 MOS** - Critical for temporal alignment

### Expected AutoVoice Improvements

Integrating SmoothSinger concepts into QUALITY_PIPELINE should:
- Improve low-frequency fidelity (bass, chest voice)
- Reduce vocoder artifacts
- Handle long audio (>30s) without memory explosion
- Achieve better pitch contour preservation

---

## 7. Related Work

### HQ-SVC (AAAI 2026)

Complementary approach using **decoupled codec + diffusion**:
- arXiv:2511.08496v3 (November 2025)
- Jointly models content and speaker features
- Supports 16→44.1kHz super-resolution
- Uses differentiable signal processing

**Synergy with SmoothSinger:**
- HQ-SVC provides the codec backbone
- SmoothSinger's MR module can enhance the decoder
- Combined: HQ-SVC encoder → SmoothSinger-style decoder

### BigVGAN (2022)

Multi-band discriminators that inspired SmoothSinger's frequency-aware design:
- Low-frequency structures for tonal stability
- High-frequency details for clarity
- Already integrated in AutoVoice's QUALITY_PIPELINE

---

## References

1. **SmoothSinger** - Sui, K., Xiang, J., & Jin, F. (2025). SmoothSinger: A Conditional Diffusion Model for Singing Voice Synthesis with Multi-Resolution Architecture. arXiv:2506.21478v1

2. **HQ-SVC** - Bai, B., et al. (2025). HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios. arXiv:2511.08496v3

3. **Wave-U-Net** - Stoller, D., Ewert, S., & Dixon, S. (2018). Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation. arXiv:1806.03185

4. **Longformer** - Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150

5. **FastDiff** - Huang, R., et al. (2022). FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis. arXiv:2204.09934

---

*Document prepared by smoothsinger-researcher agent for AutoVoice SOTA dual-pipeline integration.*
