# Tech Stack

## Platform

- **Hardware**: NVIDIA Jetson Thor (aarch64)
- **GPU**: SM 11.0 (sm_110)
- **CUDA**: 13.0
- **TensorRT**: 10.13.3.9

## Languages

| Language | Version | Purpose |
|----------|---------|---------|
| Python | 3.12 | Backend, ML pipeline, inference |
| TypeScript | 5.x | Frontend UI |
| CUDA C++ | 13.0 | Custom GPU kernels |

## Backend

| Component | Technology | Notes |
|-----------|-----------|-------|
| Web framework | Flask + SocketIO | Existing web UI and WebSocket progress |
| ML API | FastAPI | Planned: async inference endpoints |
| ML framework | PyTorch 2.11+ | Model training and inference |
| Inference | TensorRT 10.x | Optimized engine execution |
| ONNX | ONNX Runtime | Intermediate format, model validation |
| Task queue | (TBD) | For async training/conversion jobs |

## Frontend

| Component | Technology |
|-----------|-----------|
| Framework | React + Next.js |
| Styling | Tailwind CSS |
| Build | (Vite currently, migrating to Next.js) |

## Database

| Database | Purpose |
|----------|---------|
| PostgreSQL | User accounts, voice profiles, job history |
| MySQL | Existing hosted services (non-destructive) |

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Deployment | Jetson Thor (on-device) |
| Containerization | Docker + docker-compose |
| Environment | Conda (autovoice-thor) |
| Package management | pip + conda |

## Key Dependencies

- **So-VITS-SVC**: Core voice conversion architecture
- **HiFiGAN / BigVGAN**: Neural vocoder for waveform generation
- **HuBERT**: Content feature extraction
- **Vocal Separator**: Source separation for extracting vocals from songs

## SOTA Techniques (Target)

| Area | SOTA Approach | Status |
|------|--------------|--------|
| Voice conversion | So-VITS-SVC v2 / RVC v2 / DDSP-SVC | Current: So-VITS-SVC |
| Content extraction | HuBERT / ContentVec / WavLM | Current: HuBERT |
| Vocoder | BigVGAN v2 / Vocos / HiFi-GAN v2 | Current: BigVGAN |
| Pitch extraction | CREPE / RMVPE / FCN-F0 | Evaluate |
| Source separation | HTDemucs v4 / BS-RoFormer | Evaluate |
| Speaker embedding | ECAPA-TDNN / WeSpeaker / ReDimNet | Evaluate |
| Training | AdamW + cosine annealing + mel loss + multi-resolution STFT | Evaluate |
| Inference | TensorRT FP16 + dynamic batching + streaming | In progress |
| Quantization | INT8 calibration / FP8 (SM 11.0) / nvfp4 | Planned |

When implementing any component, research the current SOTA first. The table above is a starting point — check recent papers and benchmarks before committing to an approach.

## Environment Isolation

- PYTHONNOUSERSITE=1 always set (prevents system package contamination)
- Dedicated conda environment per project
- Never mix system and conda packages
- CUDA packages built from source (never fallback to pip wheels)
