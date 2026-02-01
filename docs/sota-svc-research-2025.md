# SOTA Singing Voice Conversion Research (2024-2026)

Research compiled 2026-01-31 from arXiv, Semantic Scholar, and ICASSP papers.

## Key Papers by Innovation Category

### 1. Diffusion Transformer (DiT) Based Methods
- **Seed-VC** (2411.09943) - External timbre shifter + DiT with full reference context, 40 citations
  - Key innovation: In-context learning for fine-grained timbre capture
  - Uses whisper-base encoder + DiT backbone + BigVGAN vocoder
- **DiTVC** (2025) - One-shot VC with environment and speaking rate cloning
  - Replicates acoustic properties beyond just timbre
  - Speaking rate transfer via augmentation during training
- **R-VC** (2506.01014) - Rhythm-controllable VC with shortcut flow matching
  - High-quality in just 2 sampling steps
  - DiT with mask generative transformer for duration modeling
- **Speech Diffusion Transformer (SDT)** - Personalized style transfer
  - Captures spatial and temporal acoustic characteristics
  - Zero-shot + low-resource adaptation

### 2. Flow Matching Architectures
- **VoicePrompter** (2501.17612) - DiT-based conditional flow matching + latent mixup
- **MCF-SVC** - Multi-condition flow synthesis with MS-iSTFT for fast reconstruction
- **InvoxSVC** - Latent flow matching with in-context learning for temporal features
- **DAFMSVC** (2508.05978) - Dual attention + flow matching for timbre similarity
- **MeanVC** (2510.08392) - Lightweight streaming VC via mean flows
  - Zero-shot with single sampling step
  - Chunk-wise autoregressive denoising for streaming
- **TechSinger** (2502.12572) - Controllable SVS with 7 vocal techniques + flow matching

### 3. One-Step/Fast Inference
- **CoMoSVC** (2401.01792) - Consistency model distillation for one-step sampling
- **LCM-SVC** (2408.12354) - Latent consistency distillation for acceleration
- **FastVoiceGrad** (2409.02245) - Adversarial conditional diffusion distillation
- **Consistency Flow Matching** (StableTTS) - "Free lunch" training improvement

### 4. Zero-Shot & In-Context Learning
- **Seed-VC** (2411.09943) - External timbre shifter + DiT with full reference context
- **HQ-SVC** (2511.08496) - Decoupled codec + diffusion + DSP refinement
  - Also supports voice super-resolution (22kHz → 44.1kHz)
- **FreeSVC** (2501.05586) - Multilingual with ECAPA2 speaker encoder + SPIN clustering
- **Cross-Lingual F5-TTS** (2509.14579) - Voice cloning without audio prompt transcripts

### 5. Neural Source Filter (NSF) & Source-Filter Models
- **SiFiSinger** (2410.12536) - End-to-end SVS with mcep decoupling
  - Source excitation signals for accurate pitch capture
  - Differentiable mcep and F0 losses
- **FIRNet** (2024) - Source-filter vocoder for TTS→SVS transfer
- **R2-SVC** (2510.20677) - NSF for explicit harmonic/noise separation

### 6. Vocoder Innovations
- **BigVGAN v2** - Still SOTA for singing voice quality
- **SingNet** (2505.09325) - 3000 hours singing dataset + BigVGAN/NSF-HiFiGAN pretraining
- **Pupu-Vocoder** (2512.20211) - Aliasing-free neural audio synthesis
  - Anti-derivative anti-aliasing for activation functions
  - Outperforms existing systems on singing/music/audio
- **RingFormer** (2501.01182) - Ring attention + convolution-augmented transformer
  - Real-time audio generation for long sequences
- **DisCoder** (2502.12759) - High-fidelity music vocoder using DAC latent space
- **DDSP Vocoder** (2401.10460) - Ultra-lightweight (15 MFLOPS), RTF 0.003
- **WOLONet** - Comparable to BigVGAN with fewer parameters

### 7. Robustness Improvements
- **R2-SVC** (2510.20677) - F0 perturbations, separation artifact simulation
- **REF-VC** (2508.04996) - Random erasing for noise robustness
- **kNN-SVC** (2504.05686) - Additive synthesis + concatenation smoothness optimization
  - WavLM-based bijection for harmonic emphasis
- **SSL-Melody SVC** (2502.04722) - Self-supervised melody features for BGM robustness
- **SYKI-SVC** (2501.02953) - Post-processing with ContentVec + Whisper features

### 8. Pitch/F0 Innovations
- **VibE-SVC** (2505.20794) - Discrete wavelet transform for vibrato control
- **SPA-SVC** (2406.05692) - Self-supervised pitch augmentation + SSIM loss
- **Auto-F0-Adjust** - Pitch range normalization across speakers + semitone shifting

### 9. Controllable Generation
- **TTS-CtrlNet** (2507.04349) - ControlNet for flow-matching TTS
  - Time-varying emotion control without full fine-tuning
- **MM-Sonate** (2601.01568) - Multimodal flow-matching with voice cloning
  - Zero-shot voice cloning with timbre injection mechanism

---

## Key Metrics from Papers

| Model | Speaker Sim | MCD | Inference Speed | Notes |
|-------|-------------|-----|-----------------|-------|
| CoMoSVC | 0.92 | 4.23 | 1 step | Consistency distillation |
| Seed-VC | 0.94 | 3.87 | 5-10 steps | DiT + in-context |
| HQ-SVC | 0.95 | 3.52 | Progressive | Also super-resolution |
| R2-SVC | 0.93 | 3.91 | Real-time | NSF-based |
| InvoxSVC | >0.90 | - | ~10 steps | VAE-enhanced |
| R-VC | SOTA | - | 2 steps | Shortcut flow matching |
| MeanVC | SOTA | - | 1 step | Streaming capable |

---

## Architecture Recommendations for AutoVoice

### Quality Pipeline (SOTA Configuration)

**Current**: ContentVec → RMVPE → CoMoSVC → BigVGAN

**Recommended SOTA Upgrade**:
```
Source Audio
    │
    ├─→ MelBandRoFormer (vocal separation)
    │
    ├─→ Whisper-base Encoder (semantic features, from Seed-VC)
    │
    ├─→ RMVPE (F0 extraction, Seed-VC compatible)
    │
    ├─→ CAMPPlus/ECAPA2 (speaker style embedding)
    │
    ├─→ DiT with CFM (Conditional Flow Matching decoder)
    │     - 5-10 step inference
    │     - Full reference context for in-context learning
    │     - Optional: Shortcut flow matching for 2-step inference
    │
    ├─→ NSF Module (optional, from R2-SVC)
    │     - Explicit harmonic/noise separation
    │     - Better naturalness for singing
    │
    └─→ BigVGAN v2 / Pupu-Vocoder (44.1kHz output)
          - Anti-aliasing for artifact reduction
```

**Key Innovations to Integrate**:
1. **DiT-based CFM decoder** (from Seed-VC) - 5-10 step inference
2. **Shortcut flow matching** (from R-VC) - 2-step high-quality
3. **NSF harmonic modeling** (from R2-SVC/SiFiSinger) - Better singing naturalness
4. **ECAPA2 speaker encoder** (from FreeSVC) - Robust zero-shot
5. **Dual attention mechanism** (from DAFMSVC) - Better timbre similarity
6. **Pupu-Vocoder** - Aliasing-free synthesis

### Realtime Pipeline (Low-Latency Configuration)

**Current**: ContentVec → RMVPE → SimpleDecoder → HiFiGAN

**Recommended SOTA Upgrade**:
```
Source Audio (streaming chunks)
    │
    ├─→ ContentVec/WavLM (content features)
    │
    ├─→ RMVPE (streaming F0)
    │
    ├─→ MeanVC-style decoder
    │     - Chunk-wise autoregressive denoising
    │     - Single sampling step via mean flows
    │
    └─→ Causal BigVGAN / NSF-HiFiGAN
          - Streaming-compatible vocoder
```

**Key Innovations to Integrate**:
1. **MeanVC streaming architecture** - Single step, chunk-wise processing
2. **Causal BigVGAN** (2408.11842) - 0.32 PESQ improvement
3. **Random F0 perturbation training** (from R2-SVC) - Robustness to artifacts
4. **kNN-SVC smoothness optimization** - Better concatenation

---

## Implementation Priority for New Track

### Phase 1: DiT-CFM Integration (P0)
- Replace CoMoSVC with Seed-VC's DiT decoder
- Integrate Whisper-base encoder
- Configure CAMPPlus speaker encoder
- Target: 5-10 step inference at 44.1kHz

### Phase 2: Shortcut Flow Matching (P1)
- Implement shortcut flow matching from R-VC
- Enable 2-step high-quality inference
- Add diffusion adversarial post-training

### Phase 3: NSF Integration (P1)
- Add Neural Source Filter from SiFiSinger/R2-SVC
- Explicit harmonic/noise separation
- Mcep decoupling for pitch accuracy

### Phase 4: Vocoder Upgrade (P2)
- Integrate Pupu-Vocoder anti-aliasing
- Or BigVGAN v2 with anti-aliased activations
- Target: Singing-voice-optimized synthesis

### Phase 5: Streaming Pipeline (P2)
- Implement MeanVC mean-flow architecture
- Chunk-wise autoregressive processing
- Single-step inference for streaming

### Phase 6: Speaker Encoder Upgrade (P3)
- Replace CAMPPlus with ECAPA2
- Better zero-shot generalization
- Robust multilingual support

---

## Resources

### Datasets
- **SingNet Dataset**: 3000 hours of singing voices in various languages/styles
- **TORGO**: For robustness testing with varied speech

### Repositories
- Seed-VC: https://github.com/Plachtaa/seed-vc
- BigVGAN Official: https://github.com/NVIDIA/BigVGAN
- CoMoSVC: https://comosvc.github.io/
- FreeSVC: Public code + models available
- Pupu-Vocoder: High-quality pre-trained checkpoints

### Model Weights
- `DiT_seed_v2_uvit_whisper_base_f0_44k` - Seed-VC DiT decoder
- `bigvgan_v2_44khz_128band` - BigVGAN v2 vocoder
- ECAPA2 speaker encoder from FreeSVC

---

## Citation Reference

Key papers for implementation:
```
@article{seedvc2024,
  title={Zero-shot Voice Conversion with Diffusion Transformers},
  author={Liu, Songting},
  journal={arXiv:2411.09943},
  year={2024}
}

@article{hqsvc2025,
  title={HQ-SVC: High-Quality Zero-Shot Singing Voice Conversion},
  author={Bai et al.},
  journal={arXiv:2511.08496},
  year={2025}
}

@article{sifisinger2024,
  title={SiFiSinger: High-Fidelity End-to-End SVS Based on Source-Filter Model},
  author={Cui et al.},
  journal={ICASSP 2024},
  year={2024}
}

@article{meanvc2025,
  title={MeanVC: Lightweight Streaming Zero-Shot VC via Mean Flows},
  author={Ma et al.},
  journal={arXiv:2510.08392},
  year={2025}
}
```
