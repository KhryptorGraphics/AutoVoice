# Linkup: Latest SVC Research (2024-2026 Web Search)

## Research Source
- Linkup deep web search (JS rendering enabled)
- Topics: Latest SVC systems, BigVGAN integration, ContentVec updates, TensorRT audio on Jetson

---

## 1. Latest SVC Systems (Consistency / One-Step Inference)

### LCM-SVC (Chen et al., 2024/ISCSLP)
- Latent Consistency Distillation applied to latent diffusion SVC
- Achieves one-step or few-step inference while preserving quality and timbre
- Distills pre-trained LDM-based SVC model
- **Key technique**: LCD on latent space (not waveform/mel directly)

### CoMoSVC (Lu et al., 2024)
- Consistency model-based SVC with one-step sampling
- Distills from diffusion teacher using EDM structure
- Reference: [zhenye234/CoMoSVC](https://github.com/zhenye234/CoMoSVC)

### MCF-SVC (ICIC 2025)
- Zero-shot high-fidelity SVC using multi-condition flow synthesis
- MS-iSTFT for speed optimization
- Handles unseen speakers without fine-tuning

### LHQ-SVC
- Lightweight SVC via teacher-to-student single-step distillation
- Targets edge deployment scenarios

### S2Voice (SVCC 2025 Winner)
- Won both tracks of Singing Voice Conversion Challenge 2025
- Style-aware autoregressive modeling
- Addresses incomplete disentanglement between style and timbre
- Advanced conditioning strategies for fine-grained control

### SVCC 2025 Architectures
- Top systems use: VAE-GAN, diffusion, ARLM+diffusion
- Vevo1.5 baseline: auto-regressive transformer
- Shift from identity conversion → **singing style conversion**

---

## 2. BigVGAN Integration in Real Systems

### Seed-VC + BigVGAN
- Updated singing voice model to use BigVGAN from NVIDIA
- "Large improvement to high-pitched singing voices"
- Confirmed production-quality results for SVC

### so-vits-svc 5.0 + BigVGAN
- `bigvgan-mix-v2` branch integrating BigVGAN as vocoder
- Drop-in replacement for existing HiFiGAN vocoder
- Community-validated for singing voice quality

### whisper-vits-svc + BigVGAN
- Also has `bigvgan-mix-v2` branch
- Demonstrates BigVGAN compatibility across SVC architectures

### RVC Community
- Active feature requests for BigVGAN integration
- Expected to provide higher quality and speed in Retrieval-based VC

### BigVGAN-v2 Technical Details
- Custom CUDA kernel: fused upsampling + Snake activation
- 1.5-3x faster inference on A100
- 112M parameters (universal vocoder)
- PyTorch 2.3.1 compatible, CUDA 12.1 tested
- `use_cuda_kernel=True` for optimized path

---

## 3. ContentVec Updates & Alternatives

### SVCC 2025 Usage
- ContentVec is the primary content encoder in multiple challenge baselines
- B3 baseline: ContentVec + log F0 + VUV + loudness → diffusion model
- Confirmed as industry standard for speaker-independent features

### SYKI-SVC (Jan 2025)
- Fuses ContentVec with Whisper BNF via element-wise addition
- Enhanced speaker-independent features from dual encoders
- **Insight**: Combining SSL models improves content representation

### HQ-SVC (Nov 2025)
- Compared FACodec (unified codec) vs ContentVec+CAM++ (separate encoders)
- Found unified codec better for rhythm/combined information
- ContentVec still preferred for explicit disentanglement

### RT-VC (Jun 2025)
- Uses SPARC (Speech Articulatory Coding) as alternative to ContentVec
- More interpretable disentanglement
- Suitable for real-time scenarios

### Dimension Reduction (Interspeech 2025)
- "Simple and Effective Content Encoder for SVC via SSL-Embedding Dimension Reduction"
- Explores simplifying ContentVec 768-dim → lower dimensions
- **Relevant**: Our architecture projects 768→256 via linear

### Information Perturbation
- Applied to ContentVec/HuBERT to further disentangle speaker info
- Used in SVCC 2023 T13 winning system
- Random perturbation during training removes residual speaker information

---

## 4. TensorRT Audio Optimization on Jetson

### Jetson Thor / CUDA 13.0 Support
- TensorRT repo has native aarch64 build support
- `cmake_aarch64-native.toolchain` available
- Docker files for `ubuntu-24.04-aarch64` with CUDA 13.0
- **Confirmed**: Our TensorRT 10.13.3.9 is compatible

### Standard Pipeline
```
PyTorch Model
    → torch.onnx.export() → ONNX
    → TensorRT ONNX Parser → TRT Network
    → Builder (layer fusion, precision calibration, kernel auto-tuning)
    → Serialized Engine (.trt)
```

### JEDI Framework (ACM TECS)
- TensorRT-based framework for Jetson
- Multi-threading + pipelining + buffer assignment
- Network duplication for heterogeneous CPU+GPU+NPU execution
- **Relevant**: Can pipeline ContentVec → Conformer → Consistency → BigVGAN

### Precision Modes
| Mode | Support | Notes |
|------|---------|-------|
| FP32 | All | Baseline, slowest |
| FP16 | All Jetson | Good quality/speed tradeoff |
| INT8 | Xavier+ (SM 7.2+) | Requires calibration dataset |
| NVfp4 | Latest (Blackwell) | Maximum throughput |

### Split-Inference on Jetson
- Unified memory enables concurrent CPU-GPU execution
- Up to 21.2% throughput improvement on Jetson AGX Orin
- **Useful for**: Running ContentVec on CPU while vocoder runs on GPU

### aarch64 Considerations
- PyTorch wheels not available via pip for aarch64
- Must use NVIDIA containers or build from source
- `LD_PRELOAD` for libgomp needed for "static TLS block" errors
- Our setup: conda env with source-built PyTorch (already handled)

### JDIMO (Jetson Deep-learning Inference Mapping Optimization)
- Energy efficiency + performance optimization framework
- Targets Jetson Orin NX specifically
- Layer-level scheduling decisions

---

## 5. Key Takeaways for AutoVoice

1. **LCM-SVC** is the most directly relevant consistency distillation paper for our architecture (operates in latent space like our VAE)

2. **BigVGAN integration is validated** across multiple SVC systems (Seed-VC, so-vits-svc 5.0, whisper-vits-svc) — confirmed drop-in replacement

3. **ContentVec remains the standard** for SVCC 2025, but combining with Whisper BNF (SYKI-SVC) or using information perturbation can improve disentanglement

4. **TensorRT on Jetson Thor** is fully supported with native aarch64 toolchain — JEDI framework provides a model for multi-stage pipeline optimization

5. **Split-inference** (CPU+GPU) on unified memory could pipeline our content encoder on CPU while vocoder runs on GPU for better throughput
