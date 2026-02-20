# Track: SOTA Dual-Pipeline Voice Conversion

**ID:** sota-dual-pipeline_20260130
**Status:** In Progress

## Documents

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)

## Progress

- Phases: 0/6 complete
- Tasks: 1/30 complete

## Key Components

### REALTIME_PIPELINE (Low Latency)
- ContentVec encoder (768-dim content features)
- RMVPE pitch extractor
- Simple transformer decoder
- HiFiGAN vocoder (22kHz output)
- Target: <100ms chunk latency

### QUALITY_PIPELINE (Best Quality)
- Whisper encoder (semantic features)
- Seed-VC DiT with CFM
- CAMPPlus speaker style
- BigVGAN vocoder (44kHz output)
- Optional HQ-SVC enhancement
- SmoothSinger multi-resolution concepts

## Research Sources

- [SmoothSinger Paper](https://arxiv.org/html/2506.21478v1) - Multi-resolution U-Net, vocoder-free
- [Seed-VC](https://github.com/Plachtaa/seed-vc) - DiT + Whisper + BigVGAN
- [HQ-SVC](https://github.com/ShawnPi233/HQ-SVC) - AAAI 2026, decoupled codec

## Quick Links

- [Back to Tracks](../../tracks.md)
- [Product Context](../../product.md)
- [Tech Stack](../../tech-stack.md)
