# Specification: SOTA Voice Conversion Innovations Implementation

**Track ID:** sota-innovations_20260131
**Created:** 2026-01-31
**Status:** [ ] Not Started
**Research:** [docs/sota-svc-research-2025.md](../../../docs/sota-svc-research-2025.md)

## Overview

Implement the most advanced SOTA singing voice conversion innovations discovered through academic research. This track upgrades both the Quality and Realtime pipelines with cutting-edge techniques from 2024-2026 papers.

## Key Innovations to Implement

### Category 1: Diffusion Transformer (DiT) Architecture
**Source Papers:** Seed-VC, DiTVC, R-VC, SDT

| Innovation | Paper | Benefit | Priority |
|------------|-------|---------|----------|
| DiT with in-context learning | Seed-VC | Fine-grained timbre capture | P0 |
| Shortcut flow matching | R-VC | 2-step high-quality inference | P1 |
| Speaking rate transfer | DiTVC | Replicate acoustic properties | P2 |
| Spatial-temporal capture | SDT | Better style adaptation | P2 |

### Category 2: Flow Matching Techniques
**Source Papers:** VoicePrompter, MCF-SVC, InvoxSVC, DAFMSVC, MeanVC

| Innovation | Paper | Benefit | Priority |
|------------|-------|---------|----------|
| Conditional Flow Matching | VoicePrompter | 5-10 step inference | P0 |
| Mean flows | MeanVC | Single-step streaming | P1 |
| Multi-condition synthesis | MCF-SVC | MS-iSTFT fast reconstruction | P2 |
| Dual attention mechanism | DAFMSVC | Better timbre similarity | P2 |

### Category 3: Neural Source Filter (NSF)
**Source Papers:** SiFiSinger, R2-SVC, FIRNet

| Innovation | Paper | Benefit | Priority |
|------------|-------|---------|----------|
| Mcep decoupling | SiFiSinger | Accurate pitch capture | P1 |
| Harmonic/noise separation | R2-SVC | Better singing naturalness | P1 |
| Source excitation signals | SiFiSinger | F0 accuracy | P2 |

### Category 4: Vocoder Improvements
**Source Papers:** BigVGAN v2, Pupu-Vocoder, RingFormer, DisCoder

| Innovation | Paper | Benefit | Priority |
|------------|-------|---------|----------|
| Anti-aliased activations | Pupu-Vocoder | Eliminate aliasing artifacts | P1 |
| Ring attention | RingFormer | Long sequence support | P2 |
| DAC latent space | DisCoder | Music synthesis quality | P3 |
| Causal BigVGAN | 2408.11842 | Streaming compatibility | P1 |

### Category 5: Robustness & Quality
**Source Papers:** kNN-SVC, REF-VC, SSL-Melody SVC, SYKI-SVC

| Innovation | Paper | Benefit | Priority |
|------------|-------|---------|----------|
| Additive synthesis | kNN-SVC | Harmonic emphasis | P2 |
| Random erasing | REF-VC | Noise robustness | P2 |
| SSL melody features | SSL-Melody SVC | BGM robustness | P2 |
| Concatenation smoothness | kNN-SVC | Better quality | P2 |

### Category 6: Speaker Encoding
**Source Papers:** FreeSVC

| Innovation | Paper | Benefit | Priority |
|------------|-------|---------|----------|
| ECAPA2 encoder | FreeSVC | Better zero-shot | P2 |
| SPIN clustering | FreeSVC | Speaker-invariant content | P3 |

## Target Architecture

### Quality Pipeline (After Implementation)
```
Source Audio
    │
    ├─→ MelBandRoFormer (separation, existing)
    │
    ├─→ Whisper-base Encoder (NEW: semantic features)
    │
    ├─→ RMVPE (F0, existing - Seed-VC compatible)
    │
    ├─→ CAMPPlus → ECAPA2 (speaker style)
    │
    ├─→ DiT-CFM Decoder (NEW: replaces CoMoSVC)
    │     ├── Full reference context
    │     ├── In-context learning
    │     └── 5-10 step inference → 2-step with shortcut
    │
    ├─→ NSF Module (NEW: optional enhancement)
    │     ├── Harmonic/noise separation
    │     └── Mcep decoupling
    │
    └─→ BigVGAN v2 / Pupu-Vocoder (enhanced)
          ├── Anti-aliased activations
          └── 44.1kHz output
```

### Realtime Pipeline (After Implementation)
```
Source Audio (chunks)
    │
    ├─→ ContentVec/WavLM (content, existing)
    │
    ├─→ RMVPE (streaming F0, existing)
    │
    ├─→ MeanVC-style Decoder (NEW: replaces SimpleDecoder)
    │     ├── Chunk-wise autoregressive
    │     └── Single-step via mean flows
    │
    └─→ Causal BigVGAN / NSF-HiFiGAN (NEW: streaming)
          └── RTF < 0.5
```

## Acceptance Criteria

### Quality Metrics
- [ ] Speaker similarity ≥ 0.94 (matching Seed-VC)
- [ ] MCD ≤ 3.9 (matching R2-SVC)
- [ ] Inference speed: 5-10 steps (CFM) or 2 steps (shortcut)
- [ ] Output sample rate: 44.1kHz

### Realtime Metrics
- [ ] Chunk latency < 100ms on Thor
- [ ] RTF < 0.5
- [ ] Single-step inference with mean flows
- [ ] No audible artifacts in streaming

### Integration
- [ ] Both pipelines selectable in web UI
- [ ] PipelineFactory routes correctly
- [ ] Trained LoRA adapters work with new architecture
- [ ] E2E tests pass for William↔Conor conversion

## Dependencies

### Existing Infrastructure
- MelBandRoFormer separator (implemented)
- RMVPE pitch extractor (implemented)
- BigVGAN vocoder (implemented)
- PipelineFactory (implemented)
- Frontend PipelineSelector (implemented)

### New Model Downloads
- `DiT_seed_v2_uvit_whisper_base_f0_44k` - Seed-VC DiT
- `bigvgan_v2_44khz_128band` - BigVGAN v2
- Whisper-base encoder weights
- CAMPPlus speaker encoder
- Optional: ECAPA2, Pupu-Vocoder weights

### Code References
- Seed-VC: https://github.com/Plachtaa/seed-vc
- BigVGAN: https://github.com/NVIDIA/BigVGAN
- FreeSVC: Models publicly available

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| DiT model too large for Thor GPU | Use FP16/INT8 quantization, gradient checkpointing |
| Incompatible with trained LoRAs | Design adapter bridge layer |
| Streaming latency too high | Fall back to existing pipeline |
| Memory pressure with dual pipelines | Lazy loading via PipelineFactory |

## Success Criteria

1. **Quality pipeline** produces better speaker similarity than current CoMoSVC
2. **Realtime pipeline** achieves single-step inference with acceptable quality
3. **Both pipelines** work with existing trained LoRA adapters
4. **Web UI** correctly routes to selected pipeline
5. **E2E tests** pass for William↔Conor voice conversion
