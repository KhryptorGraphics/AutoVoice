# State-of-the-Art Singing Voice Conversion (2024-2026)

## Research Sources
- Perplexity deep research (sonar-deep-research)
- HuggingFace model discovery
- DeepWiki architecture analysis (NVIDIA/BigVGAN, facebookresearch/demucs)

---

## 1. SOTA Systems Overview

### One-Step Inference Methods

| System | Approach | Speed | Quality | Repo |
|--------|----------|-------|---------|------|
| CoMoSpeech | Consistency distillation | 150x RT on A100 | Matches multi-step | [zhenye234/CoMoSpeech](https://github.com/zhenye234/CoMoSpeech) |
| ROSE-CD | Robust consistency distillation | 54x faster than teacher | Exceeds teacher | arxiv:2507.05688 |
| WaveFM | Reparameterized flow matching | Single-step | PESQ 4.362 | arxiv:2503.16689 |
| RIFT-SVC V3 | Rectified flow transformer | Single-step | Multiple CFG modes | [Pur1zumu/RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC) |
| Vevo1.5/2 | Flow matching + AR transformer | Multi-step (fast) | SVCC 2025 baseline | [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) |
| Seed-VC | DiT + BigVGAN | ~18ms/forward | Near real-time | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) |

### Key Insight: Consistency Distillation
- CoMoSpeech: Applies consistency constraint to ensure predictions remain consistent across noise-to-data trajectory
- ROSE-CD: Randomized learning trajectories + time-domain auxiliary losses → student exceeds teacher
- Critical for Jetson Thor: One-step inference eliminates multi-step iterative processes

### SVCC 2025 Challenge Results
- 26 systems evaluated across in-domain and zero-shot tasks
- **Winner: S²Voice** - FiLM-style layer-norm conditioning + style-aware cross-attention
- Top 5 systems achieved singer identity similarity indistinguishable from ground truth
- Singing style similarity (vibrato, breathiness, glissando) remains a frontier
- Reference: arxiv:2509.15629

---

## 2. Architecture Recommendations

### Content Encoder: ContentVec

**HuggingFace Checkpoint:** [lengyue233/content-vec-best](https://hf.co/lengyue233/content-vec-best)
- 388K downloads, MIT license, transformers-compatible
- HuBERT-based with content-specific fine-tuning
- Layer 6 of WavLM-Large efficiently captures content with minimal speaker bias
- Discrete unit representation provides natural information bottleneck

**ContentVec vs HuBERT-Soft:**
| Feature | ContentVec | HuBERT-Soft |
|---------|------------|-------------|
| Representation | Discrete units | Continuous soft units |
| Timbre leakage | Lower (bottleneck) | Higher (continuous) |
| Content fidelity | Good | Better nuance |
| Use case | Standard SVC | Expressive conversion |

**Recommendation:** ContentVec for production (less timbre leakage), HuBERT-Soft for research/expressive tasks.

### Conformer Encoder
- ESPnet reference: espnet/espnet Conformer
- k2-fsa/icefall Zipformer BiasNorm variant (latest)
- Combines self-attention + convolution for local+global context
- Our implementation: `src/auto_voice/models/conformer.py`

### Vocoder: BigVGAN v2

**HuggingFace Checkpoints (NVIDIA official):**
| Model | Sample Rate | Mel Bands | Upsampling | Downloads |
|-------|-------------|-----------|------------|-----------|
| [nvidia/bigvgan_v2_44khz_128band_512x](https://hf.co/nvidia/bigvgan_v2_44khz_128band_512x) | 44.1 kHz | 128 | 512x | 485K |
| [nvidia/bigvgan_v2_24khz_100band_256x](https://hf.co/nvidia/bigvgan_v2_24khz_100band_256x) | 24 kHz | 100 | 256x | 18.7K |
| [nvidia/bigvgan_v2_22khz_80band_256x](https://hf.co/nvidia/bigvgan_v2_22khz_80band_256x) | 22 kHz | 80 | 256x | 1.4M |

**Architecture Details (from DeepWiki analysis):**

1. **Snake Activation**: Periodic activation function with trainable `alpha` parameter (log-scale). `SnakeBeta` variant recommended for improved quality.

2. **Anti-Aliased Multi-Periodicity Composition**: `Activation1d` module combines:
   - Upsampling → Snake/SnakeBeta activation → Downsampling
   - Prevents aliasing artifacts in neural vocoding
   - Custom CUDA kernel fuses all three operations for 1.5-3x speedup on A100

3. **Loading Pretrained Model:**
```python
import bigvgan
import torch

# Load from HuggingFace
model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_24khz_100band_256x',
    use_cuda_kernel=False  # Set True if nvcc+ninja available
)

# Prepare for inference
model.remove_weight_norm()
model = model.eval().to('cuda')

# Generate waveform from mel spectrogram
with torch.inference_mode():
    wav_gen = model(mel)  # mel: [B, n_mels, T]
```

4. **Integration with existing HiFiGAN:**
   - BigVGAN uses same mel→waveform interface
   - Replace `HiFiGANVocoder.forward()` with BigVGAN forward
   - Same input format: `[batch, mel_channels, time_frames]`
   - Output: `[batch, 1, waveform_samples]`

### Vocal Separation: Demucs/HTDemucs

**Installation:** `pip install demucs`

**Model Variants:**
| Model | Stems | Quality | Speed | Notes |
|-------|-------|---------|-------|-------|
| `htdemucs` | 4 (drums/bass/other/vocals) | Good | Fast | Default |
| `htdemucs_ft` | 4 | Better | Slower | Fine-tuned |
| `htdemucs_6s` | 6 (+piano, +guitar) | Good | Fast | Extended |

**Python API:**
```python
from demucs.api import Separator, save_audio

separator = Separator(model="htdemucs", segment=12, device="cuda")
original, separated = separator.separate_audio_file("song.mp3")

# Get vocals only
vocals = separated["vocals"]  # Tensor
save_audio(vocals, "vocals.wav", samplerate=separator.samplerate)
```

**CLI shortcut:** `demucs --two-stems=vocals myfile.mp3`

**Architecture:** Hybrid Transformer U-Net processing both time and frequency domains with cross-domain Transformer for information exchange.

---

## 3. Real-Time Optimization for Jetson Thor

### Platform Capabilities
- CUDA 13.0, SM 11.0
- TensorRT 10.13.3.9 (verified in autovoice-thor env)
- Up to 5x speedup vs Jetson Orin for generative reasoning
- FP4 quantization + speculative decoding support
- Sub-100ms latency loops achievable

### Optimization Strategy

1. **Component-wise TensorRT Export:**
   - ContentVec encoder → ONNX → TRT (FP16, dynamic time axis)
   - Conformer encoder → ONNX → TRT (FP16)
   - Consistency student model → ONNX → TRT (FP16, dynamic time)
   - BigVGAN vocoder → ONNX → TRT (FP16, fused Snake kernel)

2. **Pipeline Latency Budget (<50ms target):**
   - ContentVec: ~8ms (small transformer, cached)
   - Conformer: ~5ms (lightweight)
   - Consistency model: ~15ms (single-step, no iteration)
   - BigVGAN: ~12ms (fused CUDA kernel)
   - Overhead: ~10ms (memory transfers, preprocessing)

3. **Streaming Inference:**
   - Overlap-add windowing: 20ms chunks with 50% overlap
   - Ring buffer for mel spectrogram accumulation
   - TensorRT async execution with CUDA streams

4. **Memory Optimization:**
   - FP16 for all inference (TRT --fp16 flag)
   - Batch size 1 (real-time constraint)
   - Pin memory for CPU↔GPU transfers
   - Pre-allocated output buffers

### TensorRT Export Commands
```bash
# ContentVec encoder
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/optimized/contentvec.onnx \
  --saveEngine=models/optimized/contentvec.trt \
  --fp16 \
  --minShapes=input:1x16000 \
  --optShapes=input:1x80000 \
  --maxShapes=input:1x320000

# Consistency student
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/optimized/consistency.onnx \
  --saveEngine=models/optimized/consistency.trt \
  --fp16 \
  --minShapes=input:1x256x10 \
  --optShapes=input:1x256x100 \
  --maxShapes=input:1x256x500

# BigVGAN vocoder
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/optimized/bigvgan.onnx \
  --saveEngine=models/optimized/bigvgan.trt \
  --fp16 \
  --minShapes=mel:1x100x10 \
  --optShapes=mel:1x100x100 \
  --maxShapes=mel:1x100x500
```

---

## 4. Key Papers & Citations

| Paper | Year | arXiv | Key Contribution |
|-------|------|-------|-----------------|
| CoMoSpeech | 2023 | 2305.06908 | Consistency model for single-step speech synthesis |
| ROSE-CD | 2025 | 2507.05688 | Robust consistency distillation exceeding teacher |
| BigVGAN v2 | 2024 | 2206.04658 | Universal vocoder with fused CUDA kernel |
| WaveFM | 2025 | 2503.16689 | Flow matching vocoder with mel-conditioned prior |
| SVCC 2025 | 2025 | 2509.15629 | Challenge results: style vs identity conversion |
| S²Voice | 2025 | 2601.13629 | SVCC 2025 winner, FiLM + style cross-attention |
| RIFT-SVC V3 | 2025 | - | Rectified flow, multi-CFG, no Whisper encoder |
| Vevo2 | 2025 | 2508.16332 | Unified speech+singing, chromagram melody |
| SAVC | 2024 | 2405.00603 | Adversarial style augmentation for HuBERT-Soft |
| ContentVec | 2022 | 2204.09224 | Content-disentangled representation |
| FreeSVC | 2025 | 2501.05586 | Multilingual SVC with SPIN clustering |
| DeCodec | 2025 | 2509.09201 | Hierarchical speech disentanglement |
| FACodec | 2025 | 2510.10785 | Factored codec: content/prosody/timbre/residual |
| Serenade | 2025 | 2503.12388 | Audio infilling for style conversion |

---

## 5. Implementation Priority for AutoVoice

### Story Execution Order (dependency-aware):

1. **AV-009: Conformer Encoder** (Low complexity, code exists)
   - Validate existing `conformer.py` implementation
   - Integrate with ContentEncoder as backend option

2. **AV-005: ContentVec Feature Extraction** (Medium)
   - Download `lengyue233/content-vec-best` checkpoint
   - Replace HuBERT-soft with ContentVec in ContentEncoder
   - Extract layer 6 features → 256-dim projection

3. **AV-006: BigVGAN Vocoder** (Medium)
   - Load `nvidia/bigvgan_v2_24khz_100band_256x` from HF
   - Create BigVGANVocoder class wrapping official model
   - Same mel→waveform interface as HiFiGANVocoder

4. **AV-008: Demucs Vocal Separation** (Medium)
   - `pip install demucs` in autovoice-thor env
   - Create VocalSeparator class using `demucs.api.Separator`
   - Integrate into SingingConversionPipeline as preprocessing step

5. **AV-010: Consistency Distillation** (High complexity)
   - Implement teacher (multi-step diffusion) and student (one-step)
   - Use CoMoSpeech approach: consistency constraint on noise-to-data trajectory
   - Add ROSE-CD improvements: randomized trajectories + time-domain auxiliary losses
   - Target: single forward pass generating mel spectrogram from content+pitch+speaker

---

## 6. Integration Architecture

```
Input Audio
    │
    ├── Demucs (htdemucs) ──→ Separated Vocals
    │
    ▼
ContentVec (lengyue233/content-vec-best)
    │ Layer 6 features [B, T, 768]
    ▼
Linear Projection [768 → 256]
    │
    ▼
Conformer Encoder [B, T, 256]
    │
    ├── + PitchEncoder (mel-quantized F0) [B, T, 256]
    ├── + SpeakerEmbedding (mel-statistics) [256]
    │
    ▼
Consistency Student (one-step)
    │ [B, T, mel_channels]
    ▼
BigVGAN v2 (nvidia/bigvgan_v2_24khz_100band_256x)
    │
    ▼
Output Waveform [B, 1, samples]
```
