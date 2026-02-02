
# Agent Assignment
================================================================================
Swarm: sota-dual-pipeline
Agent: quality-assessor
Type: tester
Phase: 4
Track: conductor/tracks/sota-dual-pipeline_20260130
GPU Required: False
Dependencies: None

## Responsibility
Implement automated quality metrics

## Expected Outputs
- src/auto_voice/evaluation/quality_metrics.py
- tests/test_quality_metrics.py

## Workflow Rules
1. Follow TDD: Write tests FIRST, then implement
2. Report progress: Update beads tasks (`bd update <id> --status in_progress`)
3. Share discoveries: Write to cipher memory for cross-agent learning
4. No fallback behavior: Raise errors, never pass silently
5. Atomic commits: One feature per commit, run tests before committing

================================================================================

# Injected Context
# Agent Context Injection
# Files: 35 (17 summarized)
# Tokens: ~47,625 / 50,000 budget
# Priority breakdown: 7 critical, 96 important, 99 reference

============================================================
# CLAUDE.md
============================================================
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoVoice is a GPU-accelerated singing voice conversion and TTS system built with PyTorch, CUDA kernels, and Flask. It converts songs to a target voice while preserving pitch and timing, using the So-VITS-SVC architecture.

## Build & Development Commands

### Platform (Jetson Thor)
```bash
# Active conda environment (ALWAYS use this)
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python

# Run any python command (PYTHONNOUSERSITE prevents system package contamination)
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON <script.py>

# Run tests
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -m pytest tests/ -x --tb=short -q
```

### Environment Setup
```bash
# Create conda environment
conda create -n autovoice python=3.12 -y && conda activate autovoice

# Install PyTorch with CUDA (REQUIRED FIRST)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
pip install -e .
```

### Running the Application
```bash
# Start server (Flask + SocketIO)
python main.py --host 0.0.0.0 --port 5000

# With Docker
docker-compose up
```

### Testing
```bash
# Run complete test suite
./run_tests.sh all

# Quick smoke tests (< 30s)
./run_tests.sh smoke

# Fast tests (excludes slow)
./run_tests.sh fast

# With coverage report
./run_tests.sh coverage

# Run specific test file
pytest tests/test_voice_conversion.py -v

# Run by marker
pytest tests/ -m cuda -v
pytest tests/ -m "not slow" -v
```

### Code Quality
```bash
black src/ tests/
isort src/ tests/
mypy src/auto_voice
```

### Key Scripts
- `scripts/setup_pytorch_env.sh` - Automated PyTorch/CUDA setup
- `scripts/build_and_test.sh` - Build and verify
- `scripts/verify_bindings.py` - Verify CUDA extension build
- `scripts/download_pretrained_models.py` - Download model weights

## Architecture

### Source Structure (`src/`)
```
auto_voice/
  inference/           # Voice conversion pipeline (So-VITS-SVC)
    singing_conversion_pipeline.py  # Main conversion entry point
    voice_cloner.py                 # Speaker encoder/profile creation
    realtime_voice_conversion_pipeline.py
  models/              # Neural network architectures
    encoder.py         # Content/pitch encoders
    vocoder.py         # HiFiGAN vocoder
  audio/               # Audio processing utilities
  evaluation/          # Quality metrics (pitch RMSE, speaker similarity)
  web/                 # Flask API + SocketIO handlers
    app.py             # create_app() factory
    api.py             # REST endpoints
  training/            # Training pipeline
  gpu/                 # GPU memory management
  monitoring/          # Prometheus metrics
cuda_kernels/          # Custom CUDA kernels (.cu files)
```

### Key Classes
- `SingingConversionPipeline` - Main voice conversion class (`inference/singing_conversion_pipeline.py`)
- `VoiceCloner` - Creates voice profiles from audio samples (`inference/voice_cloner.py`)
- `create_app()` - Flask app factory (`web/app.py`)

### Frontend (`frontend/`)
React + TypeScript + Vite + Tailwind CSS
```bash
cd frontend && npm install && npm run dev
```

### API Endpoints
- `GET /health` - Health check with component status
- `POST /api/v1/voice/clone` - Create voice profile
- `POST /api/v1/convert/song` - Convert song to target voice
- `GET /api/v1/convert/status/{id}` - Check conversion status
- WebSocket events for real-time progress

## Test Markers
Tests use pytest markers defined in `pytest.ini`:
- `smoke` - Fast validation
- `cuda` - Requires GPU
- `slow` / `very_slow` - Long-running tests
- `integration` - Component interaction tests
- `performance` - Benchmarks
- `tensorrt` - TensorRT-specific tests

## Configuration
- `config/gpu_config.yaml` - GPU and server settings
- `config/logging_config.yaml` - Logging configuration
- Environment variables: `LOG_LEVEL`, `LOG_FORMAT`, `CUDA_VISIBLE_DEVICES`

## Pre-trained Models
Located in `models/pretrained/`:
- `sovits5.0_main_1500.pth` - Main So-VITS model
- `hifigan_ljspeech.ckpt` - HiFiGAN vocoder
- `hubert-soft-0d54a1f4.pt` - HuBERT feature extractor

## File Organization Rules
- Source code: `/src`
- Tests: `/tests`
- Documentation: `/docs`
- Scripts: `/scripts`
- Configuration: `/config`
- Examples: `/examples`

## Critical Coding Rules
- No fallback behavior: Always raise RuntimeError, never pass through silently
- Speaker embedding: mel-statistics (mean+std of 128 mels = 256-dim, L2-normalized)
- Frame alignment: F.interpolate(transpose(1,2), size=target) for content/pitch
- PYTHONNOUSERSITE=1 always set for python commands
- Tests must verify real behavior (shapes, non-NaN, correct types)
- Atomic commits: one feature per commit, always run full test suite first

============================================================
# PROMPT.md
============================================================
# AutoVoice Master Development Orchestrator

## Sprint: Complete SOTA Voice Conversion System + Training Workflow

### Ultimate Goal
William Singe and Conor Maynard voice swaps: each artist singing in the style and talent of the other on each other's instrumental tracks.

**Song Conversion Mode:**
- One artist's voice sings another's song **EXACTLY** as the original artist sang it
- **Pitch Correct**: Match original pitch contour exactly
- **Singing Abilities Matched**: Transfer vibrato, dynamics, articulation
- **Synced to Instrumental**: Perfect alignment with backing track

**Live Karaoke Mode:**
- Auto-morph user's live singing to match original artist's performance
- **Multiple Audio Outputs (Configurable)**:
  - **Audience Output**: Converted vocals + instrumental for speakers
  - **Headphone Output**: Original song for user to sing along with
- User can practice matching original artist's timing/pitch while audience hears converted voice

### Objectives

1. **Dual SOTA Pipelines** - REALTIME (low-latency) + QUALITY (high-fidelity)
2. **Web Interface** - Pipeline selection for conversion and live karaoke modes
3. **Progressive Training UI** - Live loss curves, audio previews, evaluation metrics
4. **Pillowtalk Training** - Start with Pillowtalk covers for both artists
5. **Voice Swap Evaluation** - User listens and evaluates quality together with system metrics
6. **Final Conversions** - William→Conor and Conor→William on each other's songs

### Memory Systems (Compaction-Resistant Stack)

| System | Status | Command to Query |
|--------|--------|------------------|
| **Cipher** | Active | `mcp__cipher__ask_cipher "AutoVoice status"` |
| **Beads** | Active | `bd list` / `bd ready` |
| **Conductor** | Active | `cat conductor/tracks.md` |
| **Serena** | Manual | `.serena/memories/sota-dual-pipeline-2026-01-30.md` |
| **PROMPT.md** | This file | `cat PROMPT.md` |
| **ORCHESTRATOR.md** | Active | `cat ORCHESTRATOR.md` |

### Orchestration Stack

```
RALPH (top-level workflow)
    ↓
BEADS (task management: bd list, bd ready, bd close)
    ↓
CONDUCTOR (track planning: conductor/tracks/{track_id}/)
    ↓
CLAUDE-FLOW SWARMS (parallel execution)
```

### Active Beads Tasks

**Epic AV-55x: SOTA Dual-Pipeline Voice Conversion**
- [x] AV-5k7 (P1): Complete REALTIME_PIPELINE - scripts/realtime_pipeline.py
- [~] AV-u6e (P1): Create QUALITY_PIPELINE with Seed-VC
- [ ] AV-508 (P2): Add HQ-SVC enhancement (blocked by AV-u6e)
- [ ] AV-8k8 (P2): Implement SmoothSinger concepts (blocked by AV-u6e)
- [ ] AV-d11 (P1): Add Web UI pipeline selector

**Epic AV-2xb: Training-to-Inference Integration**
- [ ] AV-v7p (P1): Create AdapterManager for unified adapter loading

**Epic AV-by1: End-to-End Voice Training & Swap Workflow**
- [ ] AV-4kd (P1): Download Pillowtalk covers for William and Conor
- [ ] AV-v32 (P1): Progressive training web UI with live loss display
- [ ] AV-t32 (P1): Voice quality evaluation system
- [ ] AV-3is (P1): Train William voice model on Pillowtalk (blocked)
- [ ] AV-952 (P1): Train Conor voice model on Pillowtalk (blocked)
- [ ] AV-0wn (P1): Final voice swap: William singing as Conor (blocked)
- [ ] AV-tq1 (P1): Final voice swap: Conor singing as William (blocked)

### Implementation Order (Dependency-Driven)

**Phase 1: Infrastructure** (can run in parallel)
1. AV-u6e: Create `scripts/quality_pipeline.py` with Seed-VC
2. AV-v7p: Create `src/auto_voice/models/adapter_manager.py`
3. AV-4kd: Download Pillowtalk covers
4. AV-v32: Progressive training web UI
5. AV-t32: Voice quality evaluation system
6. AV-d11: Web UI pipeline selector

**Phase 2: Training** (after Phase 1)
7. AV-3is: Train William on Pillowtalk
8. AV-952: Train Conor on Pillowtalk

**Phase 3: Final Voice Swaps** (after Phase 2)
9. AV-0wn: William→Conor conversion
10. AV-tq1: Conor→William conversion

### Artist Test Profiles

- **William Singe**: `7da05140-1303-40c6-95d9-5b6e2c3624df`
- **Conor Maynard**: `9679a6ec-e6e2-43c4-b64e-1f004fed34f9`

### Architecture

**REALTIME_PIPELINE** (scripts/realtime_pipeline.py) - COMPLETE
```
ContentVec → RMVPE → Simple Decoder → HiFiGAN
(16kHz)     (pitch)   (transformer)   (22kHz)
Target: <100ms latency for karaoke
```

**QUALITY_PIPELINE** (scripts/quality_pipeline.py) - IN PROGRESS
```
Whisper → Seed-VC DiT → BigVGAN → HQ-SVC (optional)
(16kHz)   (CFM 44kHz)   (44kHz)   (enhancement)
Target: >0.85 speaker similarity
```

### Commands

```bash
# Environment
cd /home/kp/repo2/autovoice
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON <script>

# Beads task management
bd list                           # All tasks
bd ready                          # Unblocked tasks
bd update AV-XXX --status in_progress  # Claim
bd close AV-XXX --force --reason "..."  # Complete

# Run tests
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -m pytest tests/ -x --tb=short -q

# Conductor
cat conductor/tracks.md           # View tracks
cat conductor/tracks/{track_id}/plan.md  # View plan

# Master orchestrator
claude-flow swarm "Complete AutoVoice tasks" --strategy development --background --monitor --testing --parallel --max-agents 8
```

### Completion Criteria

- [ ] Both pipelines (REALTIME + QUALITY) working
- [ ] Web UI pipeline selector on Convert and Karaoke pages
- [ ] Progressive training UI with live loss curves
- [ ] Voice quality evaluation system (>0.85 speaker similarity)
- [ ] William and Conor trained on Pillowtalk
- [ ] User evaluation of training quality
- [ ] Final voice swaps: William↔Conor on each other's songs
- [ ] All 15 beads tasks closed

---
Last Updated: 2026-01-30 11:35 CST
Master Orchestrator: Ralph → Beads → Conductor → Claude-flow swarms

============================================================
# product.md
============================================================
# Product Definition

## Project Name

AutoVoice

## Description

GPU-accelerated singing voice conversion and TTS system that converts songs to a target voice while preserving pitch and timing, using the So-VITS-SVC architecture.

## Problem Statement

Existing voice conversion tools can't preserve pitch and timing during voice swap. Current solutions either require expensive cloud APIs with latency, or produce output that loses the musical qualities of the original performance.

## Target Users

Music producers and audio engineers who need high-quality vocal transformation for production work.

## Key Goals

1. **Real-time inference on edge hardware (Jetson Thor)** - Sub-100ms latency voice conversion running entirely on-device
2. **Production-quality vocal output preserving musicality** - Output indistinguishable from the original singer's performance quality
3. **Simple API for integration into DAWs and production tools** - Easy-to-use REST/WebSocket API for third-party integration
4. **End-to-end voice cloning pipeline** - Input a song, extract the vocal track, train a model on the user's voice from training data, replace the artist's voice with the user's voice, giving the user the singing ability of the artist in the song

## Core Workflow

1. User provides a song (input audio)
2. System extracts/separates the vocal track
3. User provides training data of their own voice
4. System trains a voice model on the user's voice
5. System converts the extracted vocal to the user's voice
6. Output: the song with the user's voice, preserving the original artist's pitch and timing

============================================================
# tech-stack.md
============================================================
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

============================================================
# workflow.md
============================================================
# Workflow

## TDD Policy

**Strict** - Tests are required before implementation.

- Write failing test first (red)
- Implement minimum code to pass (green)
- Refactor while keeping tests green (refactor)
- No code merges without passing test coverage
- Tests must verify real behavior (shapes, non-NaN, correct types, actual outputs)

## Commit Strategy

**Descriptive messages** - No strict format required, but messages should clearly describe what changed and why. The existing conventional commit format (feat:, fix:) is acceptable but not enforced.

## Code Review

**Required for all changes** - Every PR must be reviewed before merge. For AI-assisted development, this means running the full test suite and verifying integration before committing.

## Verification Checkpoints

**After each task completion** - Verify every individual task before moving on. Deep integration verification is required to ensure the code is written and integrated correctly for the project. This includes:

1. Unit tests pass for the new code
2. Integration tests pass (existing tests don't break)
3. The feature actually works in context (not just in isolation)
4. Code follows project conventions (CLAUDE.md rules)
5. No silent fallback behavior introduced

## Task Lifecycle

1. **Define** - Clearly specify what the task accomplishes
2. **Research** - Use academic MCP servers to find SOTA approaches:
   - `paper-search` — Search across arXiv, PubMed, Semantic Scholar, Google Scholar
   - `arxiv-advanced` — Deep-dive into specific arXiv papers, download and read full text
   - `semantic-scholar-citations` — Trace citation graphs to find foundational and recent work
   - Focus: find current best architectures, loss functions, training recipes, and benchmarks
   - Output: document chosen approach with paper references before implementing
3. **Test** - Write failing tests that define success criteria
4. **Implement** - Write minimum code to pass tests, using SOTA techniques
5. **Verify** - Run full test suite, check integration
6. **Review** - Deep verification of correctness and integration
7. **Commit** - Only after all checks pass

## Critical Rules (from CLAUDE.md)

- No fallback behavior: Always raise RuntimeError, never pass through silently
- Speaker embedding: mel-statistics (mean+std of 128 mels = 256-dim, L2-normalized)
- Frame alignment: F.interpolate(transpose(1,2), size=target) for content/pitch
- PYTHONNOUSERSITE=1 always set for python commands
- Atomic commits: one feature per commit, always run full test suite first

============================================================
# spec.md
============================================================
# Specification: SOTA Dual-Pipeline Voice Conversion

**Track ID:** sota-dual-pipeline_20260130
**Type:** Feature
**Created:** 2026-01-30
**Status:** Active

## Summary

Implement a two-tier voice conversion system:
1. **CUTTING_EDGE_PIPELINE** - Absolute latest January 2026 research, best possible quality
2. **STABLE_PIPELINE** - Proven 2024-2025 methods, reliable and well-tested

Both pipelines REQUIRED. User can select based on their needs (experimental vs reliable).

## Context

AutoVoice needs two distinct quality tiers:
1. **Cutting-edge** - Latest research, potentially experimental, maximum quality
2. **Stable** - Battle-tested methods, reliable output, known behavior

## Research - CUTTING EDGE (Late 2025 - January 2026)
- **HQ-SVC** (AAAI 2026): Decoupled codec + diffusion, super-resolution 16->44.1kHz
- **SmoothSinger** (Jun 2025): Multi-resolution non-sequential U-Net, vocoder-free
- **VoiceCraft** (2026): Zero-shot voice editing with neural codec language models
- **Latest Seed-VC updates** (Jan 2026): Enhanced DiT with better speaker similarity
- **FlashSpeech** (2026): Efficient zero-shot speech synthesis

## Research - STABLE (2024 - Early 2025)
- **Seed-VC** (Nov 2024): DiT + Whisper + BigVGAN, 44kHz, F0-conditioned - PROVEN
- **So-VITS-SVC 5.0** (2024): Well-tested, large community, known edge cases
- **RVC v2** (2024): Lightweight, fast, widely deployed
- **ContentVec** (2023): Proven content encoder, stable embeddings

## User Story

As a music producer, I want to choose between fast real-time conversion for live performance and high-quality conversion for final production, so that I can optimize for my specific use case.

## Acceptance Criteria

- [ ] CUTTING_EDGE_PIPELINE uses latest 2026 research (HQ-SVC + SmoothSinger + VoiceCraft)
- [ ] STABLE_PIPELINE uses proven 2024-2025 methods (Seed-VC + BigVGAN)
- [ ] REALTIME_PIPELINE converts audio with <100ms chunk latency
- [ ] **Web UI has pipeline toggle (Cutting-Edge vs Stable) on Convert page**
- [ ] **Web UI has pipeline toggle on Karaoke page**
- [ ] Toggle clearly shows: "Cutting-Edge (Experimental)" vs "Stable (Reliable)"
- [ ] All pipelines support pitch shifting
- [ ] Memory usage stays within Thor's 122GB GPU limit
- [ ] Speaker embedding format is compatible between all pipelines

## Dependencies

- Seed-VC repository: models/seed-vc/ (cloned)
- HQ-SVC repository: models/hq-svc/ (cloned)
- Existing voice profiles: data/voice_profiles/
- Existing separated vocals: data/separated/

## Out of Scope

- Training new base models (use pretrained)
- Mobile/web deployment
- Multi-speaker batch processing

## Technical Notes

### CUTTING_EDGE_PIPELINE Architecture (January 2026 SOTA)
```
Audio -> HQ-SVC Encoder -> SmoothSinger Decoder -> Neural Codec -> VoiceCraft Enhancement
         (codec)           (multi-res U-Net)       (44.1kHz)       (zero-shot refine)
```
- Uses latest 2026 research papers
- Maximum quality, experimental
- All components REQUIRED

### STABLE_PIPELINE Architecture (Proven 2024-2025)
```
Audio -> ContentVec/Whisper -> Seed-VC DiT (CFM) -> BigVGAN (44kHz) -> Optional Polish
         (16kHz)               (proven DiT)         (reliable)
```
- Battle-tested components with known behavior
- Reliable output quality
- Large community support, documented edge cases

### REALTIME_PIPELINE Architecture (Low Latency Mode)
```
Audio -> ContentVec (16kHz) -> RMVPE (pitch) -> RVC Decoder -> HiFiGAN (22kHz)
         ~40ms               ~20ms             ~10ms          ~20ms
```
- Sub-100ms latency for live karaoke
- Uses proven lightweight components

**NOTE:** ALL pipeline stages in ALL pipelines are REQUIRED. No optional components.

### SmoothSinger Integration Points
1. Multi-resolution non-sequential processing - add to quality decoder
2. Reference-guided dual-branch - already in Seed-VC prompt conditioning
3. Vocoder-free concept - HQ-SVC codec diffusion achieves similar goals

---

_Generated by Conductor. Review and edit as needed._

============================================================
# plan.md
============================================================
# Implementation Plan: SOTA Dual-Pipeline Voice Conversion

**Track ID:** sota-dual-pipeline_20260130
**Spec:** [spec.md](./spec.md)
**Created:** 2026-01-30
**Status:** [~] In Progress (~75% complete - see status-audit.md for details)

## Overview

Implement two voice conversion pipelines and integrate them into the web UI. Phase 1 creates the realtime pipeline (already started), Phase 2 creates the quality pipeline with Seed-VC, Phase 3 adds HQ-SVC enhancement, Phase 4 integrates SmoothSinger concepts, Phase 5 adds web UI controls.

**Cross-Track Dependencies (2026-02-01):**
- **Phase 2 (Seed-VC):** ✅ UNBLOCKED - Seed-VC integrated in `sota-innovations_20260131` Phase 1
- **MeanVC Alternative:** ✅ AVAILABLE - MeanVC streaming pipeline from `sota-innovations_20260131` Phase 4
- **Shortcut Flow:** ✅ AVAILABLE - 2-step inference option from `sota-innovations_20260131` Phase 2
- **LoRA Bridge:** ✅ AVAILABLE - AdapterBridge working from `sota-innovations_20260131` Phase 8

## Phase 1: Realtime Pipeline

Low-latency pipeline for karaoke using ContentVec + RMVPE + HiFiGAN.

### Tasks

- [x] Task 1.1: Create scripts/realtime_pipeline.py scaffold
- [x] Task 1.2: Implement ContentVec encoder loading with FP16 (HuBERT fallback)
- [x] Task 1.3: Implement RMVPE pitch extraction with Seed-VC fallback
- [x] Task 1.4: Implement HiFiGAN vocoder loading from CosyVoice
- [x] Task 1.5: Build simple decoder (content + pitch + speaker -> mel)
- [x] Task 1.6: Implement streaming chunk processing with crossfade
- [x] Task 1.7: Test William->Conor conversion with realtime pipeline (using HQ LoRA)

### Verification

- [x] Chunk latency <100ms on Thor (achieved ~80ms average)
- [x] RTF (real-time factor) <0.5 (achieved 0.475)
- [x] Output audio plays without artifacts (william_as_conor_realtime_30s.wav generated)

## Phase 2: Quality Pipeline - Seed-VC Integration

High-quality pipeline using Seed-VC with whisper-base and BigVGAN.

**Note (2026-02-01):** Seed-VC integration completed in `sota-innovations_20260131` Phase 1. This phase verification should reference the SeedVCPipeline implementation.

### Tasks

- [x] Task 2.1: Create scripts/quality_pipeline.py scaffold
- [x] Task 2.2: Integrate Seed-VC model loading (DiT_seed_v2_uvit_whisper_base_f0_44k)
- [x] Task 2.3: Implement Whisper encoder for semantic features
- [x] Task 2.4: Implement CAMPPlus speaker style extraction
- [x] Task 2.5: Implement CFM (Conditional Flow Matching) inference
- [x] Task 2.6: Implement BigVGAN vocoder with official NVIDIA weights
- [x] Task 2.7: Add F0 conditioning with RMVPE
- [x] Task 2.8: Test William->Conor conversion with quality pipeline (using HQ LoRA)

### Verification

- [x] Output sample rate is 44.1kHz (achieved 44100Hz)
- [ ] Speaker similarity > 0.85 (MCD < 250) - requires metric calculation
- [x] Pitch tracking preserved accurately (F0 conditioning enabled)
- [x] **Cross-track verification:** PipelineFactory includes `quality_seedvc` (from sota-innovations Phase 1)

## Phase 3: HQ-SVC Enhancement (Optional)

Add HQ-SVC as post-processing for voice super-resolution.

### Tasks

- [x] Task 3.1: Create HQ-SVC wrapper for enhancement mode (hq_svc_wrapper.py, 539 lines)
- [x] Task 3.2: Implement 22kHz -> 44.1kHz super-resolution path (super_resolve method)
- [x] Task 3.3: Test combined pipeline: Seed-VC -> HQ-SVC
- [x] Task 3.4: Benchmark quality improvement vs latency cost

### Verification

- [x] Super-resolution improves high-frequency clarity (44kHz output)
- [x] No artifacts introduced by upsampling (clean synthesis, MCD 183.93)
- [x] HQ-SVC super-resolution is fast: RTF 0.102 (10x faster than realtime)
- [x] Benchmark complete: Realtime (RTF 0.475, MCD 955) vs Quality (RTF 1.981, MCD 183)

## Phase 4: SmoothSinger Concepts Integration

Apply SmoothSinger innovations to quality pipeline.

### Tasks

- [ ] Task 4.1: Implement multi-resolution frequency branch in decoder
- [ ] Task 4.2: Add low-frequency upsampling path (non-sequential)
- [ ] Task 4.3: Implement sliding window attention for long sequences
- [ ] Task 4.4: Test improved frequency representation

### Verification

- [ ] Low-frequency components (bass) improved
- [ ] Long audio (>30s) handled without memory explosion

## Phase 5: Web UI Integration

Add pipeline selection to frontend.

### Tasks

- [x] Task 5.1: Add PipelineType enum to API types (REALTIME, QUALITY)
- [x] Task 5.2: Create pipeline selector component (PipelineSelector.tsx, AdapterSelector.tsx)
- [ ] Task 5.3: No separate Convert page - conversion happens from VoiceProfilePage
- [x] Task 5.4: Integrate selector into Karaoke page (/karaoke) - UI ONLY, not wired to API
- [x] **Task 5.5: CRITICAL - Update backend /api/v1/convert/song to accept pipeline parameter**
- [x] **Task 5.6: CRITICAL - Update backend WebSocket startSession to accept pipeline parameter**
- [x] **Task 5.7: CRITICAL - Wire KaraokePage pipeline state to startSession API call**
- [x] Task 5.8: Add pipeline selector to main Convert page (App.tsx)
- [ ] Task 5.9: Add pipeline info to conversion history display

### Verification

- [x] UI shows pipeline selection dropdown (KaraokePage has it)
- [x] UI shows pipeline selection dropdown (ConvertPage in App.tsx)
- [x] **Backend correctly routes to selected pipeline (PipelineFactory created)**
- [ ] Conversion history shows which pipeline was used

### Implementation Notes (2026-01-31)

- Created `src/auto_voice/inference/pipeline_factory.py` - singleton factory with lazy loading
- Updated `api.py` convert_song() to accept `pipeline_type` and route via PipelineFactory
- Updated `karaoke_events.py` on_start_session() to accept `pipeline_type`
- Updated `audioStreaming.ts` startSession() to accept `pipelineType` parameter
- Updated `KaraokePage.tsx` to pass pipeline state to startSession()
- Updated `App.tsx` ConvertPage to include PipelineSelector UI
- Updated `api.ts` convertSong() to accept `pipeline_type` in settings

## Phase 6: Testing & Polish ✅ COMPLETE

End-to-end testing and optimization.

### Tasks

- [x] Task 6.1: Write unit tests for both pipelines
- [x] Task 6.2: Write integration tests for web UI flow (SKIP - covered by manual E2E tests)
- [x] Task 6.3: Benchmark memory usage for both pipelines
- [x] Task 6.4: Optimize GPU memory with model unloading
- [x] Task 6.5: Add progress callbacks for long conversions
- [x] Task 6.6: Document pipeline differences in Help page (DEFER - documentation task)

### Verification

- [x] All tests pass (8/8 unit tests, 100% pass rate)
- [x] Memory stays within 64GB GPU allocation (Realtime: 0.46GB, Quality: 1.79GB)
- [x] User can successfully convert songs with both pipelines (verified in Tasks 1.7, 2.8)

## Final Verification

- [x] All acceptance criteria met
- [x] Tests passing (100% unit test coverage, all benchmarks green)
- [x] Documentation updated (BENCHMARK_RESULTS.md, test scripts)
- [x] Ready for review

---

## TRACK COMPLETE ✅

**Summary:** SOTA dual-pipeline implementation complete with full testing and benchmarks.

**Deliverables:**
- Realtime pipeline: ContentVec + Simple Decoder + HiFiGAN (RTF 0.475, 22kHz)
- Quality pipeline: Seed-VC + BigVGAN (RTF 1.981, 44kHz)
- Combined pipeline: Seed-VC + HQ-SVC enhancement (RTF 2.083, 44kHz)
- Comprehensive unit tests (8 tests, 100% pass)
- Memory benchmarks (0.46GB / 1.79GB peaks, 98.7% recovery)
- Progress callbacks for WebSocket updates

**Ready for integration with Agent 1's AdapterManager work.**

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._

============================================================
# tests/test_web_api.py
============================================================
"""Tests for Flask REST API endpoints."""
import json
import io
import pytest
import numpy as np


class TestHealthEndpoint:
    """Health check endpoint tests."""

    @pytest.mark.smoke
    def test_health_returns_200(self, client):
        resp = client.get('/api/v1/health')
        assert resp.status_code == 200

    @pytest.mark.smoke
    def test_health_has_components(self, client):
        resp = client.get('/api/v1/health')
        data = resp.get_json()
        assert 'components' in data
        assert 'status' in data
        assert data['components']['api']['status'] == 'up'

    def test_health_shows_torch_status(self, client):
        data = client.get('/api/v1/health').get_json()
        assert 'torch' in data['components']
        assert data['components']['torch']['status'] == 'up'

    def test_health_with_full_app(self, client_full):
        resp = client_full.get('/api/v1/health')
        data = resp.get_json()
        assert resp.status_code == 200
        assert data['status'] == 'healthy'
        assert data['components']['singing_pipeline']['status'] == 'up'
        assert data['components']['voice_cloner']['status'] == 'up'


class TestSystemInfo:
    """System info endpoint tests."""

    @pytest.mark.smoke
    def test_system_info_returns_200(self, client):
        resp = client.get('/api/v1/system/info')
        assert resp.status_code == 200

    def test_system_info_has_python_version(self, client):
        data = client.get('/api/v1/system/info').get_json()
        assert 'system' in data
        assert 'python_version' in data['system']

    def test_system_info_shows_dependencies(self, client):
        data = client.get('/api/v1/system/info').get_json()
        assert 'dependencies' in data
        assert data['dependencies']['torch'] is True
        assert data['dependencies']['numpy'] is True


class TestGPUMetrics:
    """GPU metrics endpoint tests."""

    @pytest.mark.cuda
    def test_gpu_metrics_returns_200(self, client):
        resp = client.get('/api/v1/gpu/metrics')
        assert resp.status_code == 200

    @pytest.mark.cuda
    def test_gpu_metrics_shows_device(self, client):
        data = client.get('/api/v1/gpu/metrics').get_json()
        assert data['available'] is True
        assert data['device_count'] >= 1


class TestKernelMetrics:
    """CUDA kernel metrics endpoint tests."""

    def test_kernel_metrics_returns_200(self, client):
        resp = client.get('/api/v1/kernels/metrics')
        assert resp.status_code == 200


class TestVoiceCloneEndpoint:
    """Voice clone API tests."""

    def test_clone_no_file_returns_400(self, client_full):
        resp = client_full.post('/api/v1/voice/clone')
        assert resp.status_code == 400

    def test_clone_empty_filename_returns_400(self, client_full):
        data = {'reference_audio': (io.BytesIO(b''), '')}
        resp = client_full.post('/api/v1/voice/clone', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_clone_invalid_extension_returns_400(self, client_full):
        data = {'reference_audio': (io.BytesIO(b'data'), 'test.txt')}
        resp = client_full.post('/api/v1/voice/clone', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_clone_service_unavailable_when_disabled(self, client):
        data = {'reference_audio': (io.BytesIO(b'data'), 'test.wav')}
        resp = client.post('/api/v1/voice/clone', data=data,
                           content_type='multipart/form-data')
        assert resp.status_code == 503


class TestVoiceProfilesEndpoint:
    """Voice profiles listing tests."""

    def test_profiles_service_unavailable_when_disabled(self, client):
        resp = client.get('/api/v1/voice/profiles')
        assert resp.status_code == 503

    def test_profiles_returns_list(self, client_full):
        resp = client_full.get('/api/v1/voice/profiles')
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)


class TestConvertSongEndpoint:
    """Song conversion endpoint tests."""

    def test_convert_no_file_returns_400(self, client_full):
        resp = client_full.post('/api/v1/convert/song')
        assert resp.status_code == 400

    def test_convert_no_profile_returns_400(self, client_full):
        data = {'song': (io.BytesIO(b'audio'), 'test.wav')}
        resp = client_full.post('/api/v1/convert/song', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_convert_invalid_profile_returns_404(self, client_full):
        data = {
            'song': (io.BytesIO(b'audio'), 'test.wav'),
            'profile_id': 'nonexistent-profile-id'
        }
        resp = client_full.post('/api/v1/convert/song', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 404

    def test_convert_pipeline_unavailable_returns_503(self, client):
        data = {
            'song': (io.BytesIO(b'audio'), 'test.wav'),
            'profile_id': 'test'
        }
        resp = client.post('/api/v1/convert/song', data=data,
                           content_type='multipart/form-data')
        assert resp.status_code == 503


class TestConvertStatusEndpoint:
    """Conversion status endpoint tests."""

    def test_status_unknown_job_returns_404(self, client_full):
        resp = client_full.get('/api/v1/convert/status/nonexistent-job')
        assert resp.status_code == 404

    def test_status_service_unavailable_when_disabled(self, client):
        resp = client.get('/api/v1/convert/status/any-id')
        assert resp.status_code == 503


class TestConvertDownloadEndpoint:
    """Download endpoint tests."""

    def test_download_unknown_job_returns_404(self, client_full):
        resp = client_full.get('/api/v1/convert/download/nonexistent-job')
        assert resp.status_code == 404


class TestConvertCancelEndpoint:
    """Cancel endpoint tests."""

    def test_cancel_unknown_job_returns_404(self, client_full):
        resp = client_full.post('/api/v1/convert/cancel/nonexistent-job')
        assert resp.status_code == 404


class TestConvertMetricsEndpoint:
    """Conversion metrics endpoint tests."""

    def test_metrics_unknown_job_returns_404(self, client_full):
        resp = client_full.get('/api/v1/convert/metrics/nonexistent-job')
        assert resp.status_code == 404

============================================================
# tests/test_scripts_quality_pipeline.py
============================================================
#!/usr/bin/env python3
"""Unit tests for SOTA quality pipeline (scripts/quality_pipeline.py).

Tests the Seed-VC + BigVGAN pipeline for high-quality conversion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pytest
import torch
import numpy as np

from quality_pipeline import QualityVoiceConverter, QualityConfig


@pytest.fixture
def converter():
    """Create a quality converter instance."""
    config = QualityConfig(
        sample_rate=44100,
        diffusion_steps=10,  # Use fewer steps for testing
        fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return QualityVoiceConverter(config)


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


class TestQualityConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default values."""
        config = QualityConfig()
        assert config.sample_rate == 44100
        assert config.diffusion_steps == 30
        assert config.fp16 is True


class TestQualityConverter:
    """Test converter."""

    def test_initialization(self, converter):
        """Test init."""
        assert converter is not None
        assert converter.config.sample_rate == 44100

    def test_unload(self, converter):
        """Test unload."""
        converter.unload()
        # Models are lazily loaded, so unload should succeed even if not loaded

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_convert(self, converter, sample_audio):
        """Test conversion."""
        audio, sr = sample_audio
        # Need reference audio for Seed-VC
        reference = audio.copy()
        
        converted, out_sr = converter.convert(
            source_audio=audio,
            source_sr=sr,
            reference_audio=reference,
            reference_sr=sr,
            pitch_shift=0
        )
        
        assert len(converted) > 0
        assert out_sr == 44100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

============================================================
# tests/test_svc_decoder_sota.py
============================================================
"""Tests for SOTA SVC decoder (Phase 6).

Validates the CoMoSVC consistency model decoder with:
- Single-step inference (consistency distillation)
- BiDilConv decoder architecture
- Content (768-dim) + pitch (256-dim) + speaker (256-dim) conditioning
- Mel spectrogram output for BigVGAN vocoder
- Speaker similarity preservation
- Pitch preservation through conversion
"""
import pytest
import torch


class TestCoMoSVCDecoder:
    """Tests for CoMoSVC consistency model decoder."""

    def test_class_exists(self):
        """CoMoSVCDecoder class should exist."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        assert CoMoSVCDecoder is not None

    def test_init_default(self):
        """Default initialization."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        decoder = CoMoSVCDecoder()
        assert decoder.content_dim == 768  # ContentVec Layer 12
        assert decoder.pitch_dim == 256  # PitchEncoder output
        assert decoder.speaker_dim == 256  # mel-statistics embedding
        assert decoder.n_mels == 100  # BigVGAN input

    def test_forward_shape(self):
        """Decoder produces correct mel spectrogram shape."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 50  # 50 frames
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker = torch.randn(1, 256, device=device)  # Global speaker embedding

        mel = decoder(content, pitch, speaker)
        assert mel.shape == (1, 100, T)  # [B, n_mels, T]

    def test_single_step_inference(self):
        """Consistency model should produce output in single step."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker = torch.randn(1, 256, device=device)

        # Single-step (default) should work
        mel = decoder.infer(content, pitch, speaker, n_steps=1)
        assert mel.shape == (1, 100, T)
        assert torch.isfinite(mel).all()

    def test_multi_step_inference(self):
        """Multi-step should also work (higher quality)."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker = torch.randn(1, 256, device=device)

        mel_1step = decoder.infer(content, pitch, speaker, n_steps=1)
        mel_4step = decoder.infer(content, pitch, speaker, n_steps=4)

        # Both should produce valid output
        assert mel_1step.shape == mel_4step.shape
        assert torch.isfinite(mel_4step).all()

    def test_batch_processing(self):
        """Batched input produces batched output."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 40
        content = torch.randn(2, T, 768, device=device)
        pitch = torch.randn(2, T, 256, device=device)
        speaker = torch.randn(2, 256, device=device)

        mel = decoder(content, pitch, speaker)
        assert mel.shape == (2, 100, T)

    def test_output_finite(self):
        """All outputs should be finite."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 50
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker = torch.randn(1, 256, device=device)

        mel = decoder(content, pitch, speaker)
        assert torch.isfinite(mel).all()

    def test_device_placement(self):
        """Output on same device as input."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker = torch.randn(1, 256, device=device)

        mel = decoder(content, pitch, speaker)
        assert mel.device == content.device


class TestBiDilConv:
    """Tests for BiDilConv (Bidirectional Dilated Convolution) network."""

    def test_bidilconv_exists(self):
        """BiDilConv class should exist."""
        from auto_voice.models.svc_decoder import BiDilConv
        assert BiDilConv is not None

    def test_bidilconv_output_shape(self):
        """BiDilConv preserves temporal dimension."""
        from auto_voice.models.svc_decoder import BiDilConv
        block = BiDilConv(channels=256, kernel_size=3, n_layers=4)
        x = torch.randn(1, 256, 50)
        y = block(x)
        assert y.shape == x.shape

    def test_bidilconv_dilated(self):
        """Should use dilated convolutions with increasing dilation."""
        from auto_voice.models.svc_decoder import BiDilConv
        block = BiDilConv(channels=256, kernel_size=3, n_layers=4)
        # Should have layers with dilation 1, 2, 4, 8 (or similar pattern)
        assert len(block.layers) >= 4


class TestSpeakerConditioning:
    """Tests for speaker embedding conditioning."""

    def test_speaker_embedding_shape(self):
        """Speaker embedding should be 256-dim (mel-statistics)."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        decoder = CoMoSVCDecoder()
        assert decoder.speaker_dim == 256

    def test_different_speakers_different_output(self):
        """Different speaker embeddings should produce different mels."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker_a = torch.randn(1, 256, device=device)
        speaker_b = torch.randn(1, 256, device=device)

        mel_a = decoder(content, pitch, speaker_a)
        mel_b = decoder(content, pitch, speaker_b)

        assert not torch.allclose(mel_a, mel_b, atol=1e-3)

    def test_same_speaker_consistent(self):
        """Same speaker + same content with same seed produces same output."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)
        decoder.eval()

        T = 30
        content = torch.randn(1, T, 768, device=device)
        pitch = torch.randn(1, T, 256, device=device)
        speaker = torch.randn(1, 256, device=device)

        with torch.no_grad():
            torch.manual_seed(42)
            mel_1 = decoder(content, pitch, speaker)
            torch.manual_seed(42)
            mel_2 = decoder(content, pitch, speaker)

        assert torch.allclose(mel_1, mel_2, atol=1e-5)


class TestPitchPreservation:
    """Tests for pitch preservation through decoder."""

    def test_pitch_affects_output(self):
        """Different pitch should produce different mels."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, 768, device=device)
        pitch_low = torch.randn(1, T, 256, device=device) * 0.5
        pitch_high = torch.randn(1, T, 256, device=device) * 2.0
        speaker = torch.randn(1, 256, device=device)

        mel_low = decoder(content, pitch_low, speaker)
        mel_high = decoder(content, pitch_high, speaker)

        assert not torch.allclose(mel_low, mel_high, atol=1e-3)

============================================================
# tests/test_auto_training_trigger.py
============================================================
"""Tests for auto-triggering training on profile creation.

Phase 5: Test that profile creation automatically triggers training.

Tests verify:
- create_voice_profile triggers training job
- API endpoint triggers training
- Profile response includes training status
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for profiles and jobs."""
    profile_dir = tmp_path / "voice_profiles"
    profile_dir.mkdir()
    job_dir = tmp_path / "training_jobs"
    job_dir.mkdir()
    return {"profiles": profile_dir, "jobs": job_dir}


@pytest.fixture
def store(temp_dirs):
    """Create VoiceProfileStore with temp directory."""
    return VoiceProfileStore(profiles_dir=str(temp_dirs["profiles"]))


class TestVoiceClonerAutoTraining:
    """Tests for VoiceCloner auto-training trigger."""

    def test_voice_cloner_has_training_manager(self):
        """Task 5.1: VoiceCloner should accept training_manager."""
        from auto_voice.inference.voice_cloner import VoiceCloner

        # VoiceCloner should have training_manager parameter
        cloner = VoiceCloner.__new__(VoiceCloner)
        assert hasattr(VoiceCloner, '__init__')

    def test_create_profile_triggers_training(self, store, temp_dirs):
        """Task 5.2: create_voice_profile should trigger training job."""
        from auto_voice.inference.voice_cloner import VoiceCloner
        from auto_voice.training.job_manager import TrainingJobManager

        # Create training manager
        manager = TrainingJobManager(
            storage_path=temp_dirs["jobs"],
            require_gpu=False,
        )

        # Create cloner with training manager
        with patch.object(VoiceCloner, '__init__', lambda self, **kwargs: None):
            cloner = VoiceCloner.__new__(VoiceCloner)
            cloner.store = store
            cloner._training_manager = manager
            cloner._auto_train = True

            # Mock the _extract_embedding method
            cloner._extract_embedding = MagicMock(
                return_value=torch.randn(256).numpy()
            )

            # Add trigger_training method if not exists
            if hasattr(cloner, 'trigger_training'):
                # Create profile with samples
                profile_id = "test-profile"
                store.save({
                    "profile_id": profile_id,
                    "name": "Test",
                    "embedding": torch.randn(256).numpy(),
                })

                # Trigger training
                cloner.trigger_training(profile_id, ["sample1", "sample2"])

                # Check job was created
                assert manager.queue_size >= 0  # Manager initialized


class TestAPIAutoTraining:
    """Tests for API endpoint auto-training trigger."""

    def test_profiles_endpoint_exists(self):
        """Task 5.3: POST /api/v1/profiles endpoint should exist."""
        from auto_voice.web.app import create_app

        app, socketio = create_app()
        client = app.test_client()

        # Check endpoint exists (may return error without data, but shouldn't 404)
        response = client.post('/api/v1/profiles')
        assert response.status_code != 404, "Profiles endpoint should exist"

    def test_profile_creation_returns_status(self, store, temp_dirs):
        """Task 5.4-5.6: Profile creation should return training_status."""
        from auto_voice.web.app import create_app

        app, socketio = create_app()
        client = app.test_client()

        # Check that the profiles list endpoint returns profiles with status
        response = client.get('/api/v1/voice/profiles')
        # Should return list (may be empty)
        assert response.status_code == 200
        data = response.get_json()
        assert "profiles" in data or isinstance(data, list), "Should return profiles"


class TestProfileTrainingStatus:
    """Tests for training status in profile responses."""

    def test_profile_has_training_status_field(self, store):
        """Profile response should include training_status."""
        # Create profile
        profile_id = "status-test-profile"
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
            "training_status": "pending",
        })

        # Load and check
        profile = store.load(profile_id)
        assert "training_status" in profile or profile.get("training_status") is None

    def test_training_status_transitions(self, store):
        """Training status should transition: pending → training → ready."""
        profile_id = "transition-test"

        # Create with pending status
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
            "training_status": "pending",
        })

        # Simulate training start
        profile = store.load(profile_id)
        profile["training_status"] = "training"
        store.save(profile)

        # Verify transition
        profile = store.load(profile_id)
        assert profile.get("training_status") == "training"

        # Simulate training complete
        profile["training_status"] = "ready"
        store.save(profile)

        profile = store.load(profile_id)
        assert profile.get("training_status") == "ready"

    def test_has_trained_model_matches_status(self, store):
        """has_trained_model should match 'ready' training_status."""
        profile_id = "model-status-test"

        # Create profile without weights
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
            "training_status": "pending",
        })

        # Should not have trained model
        assert not store.has_trained_model(profile_id)

        # Add weights
        store.save_lora_weights(profile_id, {
            "test.lora_A": torch.randn(8, 256),
            "test.lora_B": torch.randn(256, 8),
        })

        # Now should have trained model
        assert store.has_trained_model(profile_id)

============================================================
# tests/test_training_websocket_events.py
============================================================
"""Tests for training WebSocket events.

Phase 7: Test WebSocket events for training progress.

Tests verify:
- training.started event is emitted when job begins
- training.progress events are emitted with epoch/loss
- training.completed/failed events are emitted at end
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from auto_voice.training.job_manager import (
    TrainingJobManager,
    TrainingJob,
    TrainingConfig,
    JobStatus,
)


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "training_jobs"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def mock_socketio():
    """Create mock SocketIO instance."""
    socketio = MagicMock()
    socketio.emit = MagicMock()
    return socketio


@pytest.fixture
def manager_with_socketio(temp_storage, mock_socketio):
    """Create TrainingJobManager with mock SocketIO."""
    return TrainingJobManager(
        storage_path=temp_storage,
        require_gpu=False,
        socketio=mock_socketio,
    )


class TestTrainingStartedEvent:
    """Tests for training.started WebSocket event."""

    def test_manager_accepts_socketio_parameter(self, temp_storage, mock_socketio):
        """Task 7.1: TrainingJobManager should accept socketio parameter."""
        manager = TrainingJobManager(
            storage_path=temp_storage,
            require_gpu=False,
            socketio=mock_socketio,
        )
        assert manager is not None

    def test_emits_started_event_on_job_start(self, manager_with_socketio, mock_socketio):
        """Task 7.2: Should emit training.started when job begins."""
        # Create and start a job
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1", "sample2"],
        )

        # Start the job
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.RUNNING.value,
            gpu_device=0,
        )

        # Check emit was called
        mock_socketio.emit.assert_called()

        # Find the training.started call
        started_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.started'
        ]
        assert len(started_calls) == 1

        # Check event data
        event_data = started_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['profile_id'] == 'test-profile'
        assert 'config' in event_data

    def test_started_event_includes_config(self, manager_with_socketio, mock_socketio):
        """Started event should include training configuration."""
        config = TrainingConfig(epochs=15, learning_rate=5e-5)
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
            config=config,
        )

        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)

        started_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.started'
        ]
        assert len(started_calls) == 1

        event_data = started_calls[0][0][1]
        assert event_data['config']['epochs'] == 15


class TestTrainingProgressEvent:
    """Tests for training.progress WebSocket event."""

    def test_emits_progress_event(self, manager_with_socketio, mock_socketio):
        """Task 7.3-7.4: Should emit training.progress during training."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)

        # Reset mock to clear started event
        mock_socketio.emit.reset_mock()

        # Emit progress
        manager_with_socketio.emit_training_progress(
            job_id=job.job_id,
            epoch=3,
            total_epochs=10,
            step=150,
            total_steps=500,
            loss=0.45,
            learning_rate=1e-4,
        )

        # Check progress event was emitted
        mock_socketio.emit.assert_called()
        progress_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.progress'
        ]
        assert len(progress_calls) == 1

        event_data = progress_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['epoch'] == 3
        assert event_data['total_epochs'] == 10
        assert event_data['loss'] == 0.45

    def test_progress_event_includes_all_fields(self, manager_with_socketio, mock_socketio):
        """Progress event should include all training metrics."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        manager_with_socketio.emit_training_progress(
            job_id=job.job_id,
            epoch=5,
            total_epochs=10,
            step=250,
            total_steps=500,
            loss=0.32,
            learning_rate=8e-5,
        )

        progress_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.progress'
        ]
        event_data = progress_calls[0][0][1]

        # Verify all fields
        assert 'epoch' in event_data
        assert 'total_epochs' in event_data
        assert 'step' in event_data
        assert 'total_steps' in event_data
        assert 'loss' in event_data
        assert 'learning_rate' in event_data
        assert 'progress_percent' in event_data


class TestTrainingCompletedEvent:
    """Tests for training.completed WebSocket event."""

    def test_emits_completed_event(self, manager_with_socketio, mock_socketio):
        """Task 7.5-7.6: Should emit training.completed when job finishes."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        # Complete the job
        results = {
            'final_loss': 0.15,
            'epochs_trained': 10,
            'training_time_seconds': 120.5,
        }
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.COMPLETED.value,
            results=results,
        )

        # Check completed event was emitted
        completed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.completed'
        ]
        assert len(completed_calls) == 1

        event_data = completed_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['profile_id'] == 'test-profile'
        assert event_data['results']['final_loss'] == 0.15

    def test_completed_event_includes_results(self, manager_with_socketio, mock_socketio):
        """Completed event should include training results."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1", "sample2"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        results = {
            'final_loss': 0.12,
            'initial_loss': 1.5,
            'epochs_trained': 10,
            'loss_curve': [1.5, 0.8, 0.5, 0.3, 0.2, 0.15, 0.13, 0.12, 0.12, 0.12],
        }
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.COMPLETED.value,
            results=results,
        )

        completed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.completed'
        ]
        event_data = completed_calls[0][0][1]

        assert 'results' in event_data
        assert event_data['results']['initial_loss'] == 1.5
        assert len(event_data['results']['loss_curve']) == 10


class TestTrainingFailedEvent:
    """Tests for training.failed WebSocket event."""

    def test_emits_failed_event(self, manager_with_socketio, mock_socketio):
        """Should emit training.failed when job fails."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        # Fail the job
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.FAILED.value,
            error="Out of GPU memory",
        )

        # Check failed event was emitted
        failed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.failed'
        ]
        assert len(failed_calls) == 1

        event_data = failed_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['error'] == "Out of GPU memory"

    def test_failed_event_includes_error_details(self, manager_with_socketio, mock_socketio):
        """Failed event should include error details."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.FAILED.value,
            error="CUDA error: device-side assert triggered",
        )

        failed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.failed'
        ]
        event_data = failed_calls[0][0][1]

        assert 'error' in event_data
        assert 'CUDA error' in event_data['error']
        assert 'profile_id' in event_data


class TestWebSocketWithoutSocketIO:
    """Tests for graceful handling when socketio is not provided."""

    def test_works_without_socketio(self, temp_storage):
        """Manager should work without socketio (no events emitted)."""
        manager = TrainingJobManager(
            storage_path=temp_storage,
            require_gpu=False,
            socketio=None,
        )

        job = manager.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )

        # These should not raise even without socketio
        manager.update_job_status(job.job_id, JobStatus.RUNNING.value)
        manager.update_job_progress(job.job_id, 50)
        manager.update_job_status(
            job.job_id,
            JobStatus.COMPLETED.value,
            results={'final_loss': 0.1},
        )

        assert manager.get_job(job.job_id).status == JobStatus.COMPLETED.value

============================================================
# tests/test_adapter_bridge.py
============================================================
"""Tests for AdapterBridge - LoRA to Seed-VC integration.

The AdapterBridge serves as the integration layer between:
1. Trained LoRA adapters (from our MLP-based decoder)
2. Seed-VC's in-context learning approach (reference audio)

Tests cover:
- Loading and caching voice references
- Loading and caching LoRA weights
- Profile mapping and fuzzy matching
- Error handling for missing/corrupt files
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestAdapterBridgeInit:
    """Test AdapterBridge initialization and configuration."""

    def test_import_succeeds(self):
        """AdapterBridge can be imported."""
        from auto_voice.inference.adapter_bridge import AdapterBridge
        assert AdapterBridge is not None

    def test_init_creates_instance(self, tmp_path):
        """AdapterBridge initializes with custom directories."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"

        profiles_dir.mkdir()
        training_dir.mkdir()
        lora_dir.mkdir()

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        assert bridge is not None
        assert bridge.profiles_dir == profiles_dir
        assert bridge.training_audio_dir == training_dir
        assert bridge.lora_dir == lora_dir
        assert bridge.device == torch.device("cpu")

    def test_init_loads_profile_mappings(self, tmp_path):
        """AdapterBridge loads profile JSON files on initialization."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create test profile
        profile_data = {
            "profile_id": "test-profile-123",
            "name": "John Doe"
        }
        with open(profiles_dir / "test-profile-123.json", "w") as f:
            json.dump(profile_data, f)

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(tmp_path / "loras"),
            device="cpu"
        )

        assert "test-profile-123" in bridge._profile_to_artist
        assert bridge._profile_to_artist["test-profile-123"] == "John Doe"

    def test_init_handles_corrupt_profile_json(self, tmp_path):
        """AdapterBridge gracefully handles corrupt profile JSON."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create corrupt JSON file
        with open(profiles_dir / "corrupt.json", "w") as f:
            f.write("{ invalid json }")

        # Should not raise - just logs warning
        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(tmp_path / "loras"),
            device="cpu"
        )

        assert bridge is not None
        assert len(bridge._profile_to_artist) == 0

    def test_init_default_directories(self):
        """AdapterBridge uses default directories when not specified."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        with patch.object(Path, 'glob', return_value=[]):
            bridge = AdapterBridge(device="cpu")

        assert "data/voice_profiles" in str(bridge.profiles_dir)
        assert "data/separated_youtube" in str(bridge.training_audio_dir)
        assert "data/trained_models/hq" in str(bridge.lora_dir)


class TestVoiceReferenceLoading:
    """Test loading voice references for Seed-VC pipeline."""

    @pytest.fixture
    def bridge_setup(self, tmp_path):
        """Create AdapterBridge with test data."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"

        profiles_dir.mkdir()
        training_dir.mkdir()
        lora_dir.mkdir()

        # Create test profile
        profile_data = {
            "profile_id": "profile-abc",
            "name": "John Artist"
        }
        with open(profiles_dir / "profile-abc.json", "w") as f:
            json.dump(profile_data, f)

        # Create artist audio directory with vocals
        artist_dir = training_dir / "john_artist"
        artist_dir.mkdir()

        # Create fake vocal files (with some content for size estimation)
        for i in range(3):
            vocal_file = artist_dir / f"song_{i}_vocals.wav"
            # Write enough bytes to simulate ~10s of audio at 44.1kHz
            with open(vocal_file, "wb") as f:
                f.write(b"\x00" * (88200 * 10))  # ~10s

        # Create speaker embedding
        embedding = np.random.randn(256).astype(np.float32)
        np.save(profiles_dir / "profile-abc.npy", embedding)

        # Create LoRA checkpoint
        lora_state = {"layer1.weight": torch.randn(64, 64)}
        torch.save({"lora_state": lora_state}, lora_dir / "profile-abc_hq_lora.pt")

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        return bridge, profiles_dir, training_dir, lora_dir

    def test_get_voice_reference_returns_dataclass(self, bridge_setup):
        """get_voice_reference returns VoiceReference dataclass."""
        from auto_voice.inference.adapter_bridge import VoiceReference

        bridge, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert isinstance(ref, VoiceReference)
        assert ref.profile_id == "profile-abc"
        assert ref.profile_name == "John Artist"

    def test_get_voice_reference_finds_audio_files(self, bridge_setup):
        """get_voice_reference finds reference audio files."""
        bridge, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert len(ref.reference_paths) > 0
        assert all(p.suffix == ".wav" for p in ref.reference_paths)
        assert all("vocals" in p.name for p in ref.reference_paths)

    def test_get_voice_reference_loads_embedding(self, bridge_setup):
        """get_voice_reference loads pre-computed speaker embedding."""
        bridge, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert ref.speaker_embedding is not None
        assert ref.speaker_embedding.shape == (256,)

    def test_get_voice_reference_finds_lora_path(self, bridge_setup):
        """get_voice_reference finds LoRA checkpoint path."""
        bridge, _, _, lora_dir = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert ref.lora_path is not None
        assert ref.lora_path.exists()
        assert "hq_lora.pt" in str(ref.lora_path)

    def test_get_voice_reference_estimates_duration(self, bridge_setup):
        """get_voice_reference estimates total audio duration."""
        bridge, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        # We created 3 files of ~10s each
        assert ref.total_duration > 20.0  # At least 20s total

    def test_get_voice_reference_max_references(self, bridge_setup):
        """get_voice_reference respects max_references parameter."""
        bridge, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc", max_references=2)

        assert len(ref.reference_paths) <= 2

    def test_get_voice_reference_caching(self, bridge_setup):
        """get_voice_reference caches results."""
        bridge, _, _, _ = bridge_setup

        ref1 = bridge.get_voice_reference("profile-abc")
        ref2 = bridge.get_voice_reference("profile-abc")

        assert ref1 is ref2  # Same object (cached)

    def test_get_voice_reference_profile_not_found(self, bridge_setup):
        """get_voice_reference raises ValueError for missing profile."""
        bridge, _, _, _ = bridge_setup

        with pytest.raises(ValueError, match="Profile not found"):
            bridge.get_voice_reference("nonexistent-profile")

    def test_get_voice_reference_no_audio_returns_empty_list(self, tmp_path):
        """get_voice_reference returns empty list when no audio available."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"
        profiles_dir.mkdir()
        training_dir.mkdir()  # Create the directory, but with no matching artist
        lora_dir.mkdir()

        # Profile exists but no audio directory for this artist
        profile_data = {"profile_id": "no-audio", "name": "No Audio Artist"}
        with open(profiles_dir / "no-audio.json", "w") as f:
            json.dump(profile_data, f)

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        ref = bridge.get_voice_reference("no-audio")
        assert len(ref.reference_paths) == 0


class TestFuzzyMatching:
    """Test fuzzy string matching for artist name variations."""

    def test_fuzzy_match_exact(self):
        """Exact strings match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("connor", "connor") is True

    def test_fuzzy_match_one_char_diff(self):
        """Strings with one character difference match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("connor", "conor") is True  # Missing 'n'
        assert AdapterBridge._fuzzy_match("john", "jonn") is True  # Wrong char

    def test_fuzzy_match_two_char_diff(self):
        """Strings with two character differences match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("steven", "stevan") is True

    def test_fuzzy_match_too_different(self):
        """Strings too different don't match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("alice", "bob") is False
        assert AdapterBridge._fuzzy_match("john", "jonathan") is False

    def test_fuzzy_match_empty_strings(self):
        """Empty strings don't match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("", "test") is False
        assert AdapterBridge._fuzzy_match("test", "") is False
        assert AdapterBridge._fuzzy_match("", "") is False


class TestLoRALoading:
    """Test LoRA weight loading functionality."""

    @pytest.fixture
    def lora_bridge(self, tmp_path):
        """Create AdapterBridge with LoRA checkpoint."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        lora_dir = tmp_path / "loras"
        profiles_dir.mkdir()
        lora_dir.mkdir()

        # Create test profile
        profile_data = {"profile_id": "lora-test", "name": "LoRA Test"}
        with open(profiles_dir / "lora-test.json", "w") as f:
            json.dump(profile_data, f)

        # Create LoRA checkpoint with state dict
        lora_state = {
            "encoder.lora_A": torch.randn(32, 64),
            "encoder.lora_B": torch.randn(64, 32),
            "decoder.lora_A": torch.randn(16, 32),
            "decoder.lora_B": torch.randn(32, 16),
        }
        checkpoint = {
            "lora_state": lora_state,
            "artist": "Test Artist",
            "epoch": 100,
            "loss": 0.0123,
            "precision": "fp16",
            "status": "completed",
            "config": {"lr": 1e-4, "rank": 32},
        }
        torch.save(checkpoint, lora_dir / "lora-test_hq_lora.pt")

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        return bridge

    def test_load_lora_returns_state_dict(self, lora_bridge):
        """load_lora returns LoRA state dictionary."""
        lora_state = lora_bridge.load_lora("lora-test")

        assert isinstance(lora_state, dict)
        assert "encoder.lora_A" in lora_state
        assert "encoder.lora_B" in lora_state
        assert "decoder.lora_A" in lora_state
        assert "decoder.lora_B" in lora_state

    def test_load_lora_tensors_on_device(self, lora_bridge):
        """load_lora moves tensors to specified device."""
        lora_state = lora_bridge.load_lora("lora-test")

        for key, tensor in lora_state.items():
            assert tensor.device == torch.device("cpu")

    def test_load_lora_caching(self, lora_bridge):
        """load_lora caches loaded weights."""
        lora1 = lora_bridge.load_lora("lora-test")
        lora2 = lora_bridge.load_lora("lora-test")

        # Tensors should be same objects (cached)
        for key in lora1:
            assert lora1[key] is lora2[key]

    def test_load_lora_no_cache(self, lora_bridge):
        """load_lora can skip caching."""
        lora1 = lora_bridge.load_lora("lora-test", use_cache=False)
        lora2 = lora_bridge.load_lora("lora-test", use_cache=False)

        # Tensors should be different objects
        for key in lora1:
            assert lora1[key] is not lora2[key]

    def test_load_lora_not_found(self, lora_bridge):
        """load_lora raises FileNotFoundError for missing LoRA."""
        with pytest.raises(FileNotFoundError, match="No LoRA found"):
            lora_bridge.load_lora("nonexistent-profile")

    def test_get_lora_metadata(self, lora_bridge):
        """get_lora_metadata returns training metadata."""
        metadata = lora_bridge.get_lora_metadata("lora-test")

        assert metadata["artist"] == "Test Artist"
        assert metadata["epoch"] == 100
        assert metadata["loss"] == 0.0123
        assert metadata["precision"] == "fp16"
        assert metadata["status"] == "completed"
        assert metadata["config"]["lr"] == 1e-4

    def test_get_lora_metadata_missing_returns_empty(self, lora_bridge):
        """get_lora_metadata returns empty dict for missing LoRA."""
        metadata = lora_bridge.get_lora_metadata("nonexistent")
        assert metadata == {}


class TestProfileListing:
    """Test listing available profiles."""

    @pytest.fixture
    def populated_bridge(self, tmp_path):
        """Create AdapterBridge with multiple profiles."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"

        profiles_dir.mkdir()
        training_dir.mkdir()
        lora_dir.mkdir()

        # Profile 1: has both LoRA and reference audio
        profile1 = {"profile_id": "profile-1", "name": "Alpha Singer"}
        with open(profiles_dir / "profile-1.json", "w") as f:
            json.dump(profile1, f)
        (training_dir / "alpha_singer").mkdir()
        with open(training_dir / "alpha_singer" / "song_vocals.wav", "wb") as f:
            f.write(b"\x00" * 88200)
        torch.save({}, lora_dir / "profile-1_hq_lora.pt")

        # Profile 2: only LoRA, no reference audio (unique name to avoid fuzzy match)
        profile2 = {"profile_id": "profile-2", "name": "Zeta Performer"}
        with open(profiles_dir / "profile-2.json", "w") as f:
            json.dump(profile2, f)
        torch.save({}, lora_dir / "profile-2_hq_lora.pt")

        # Profile 3: only reference audio, no LoRA
        profile3 = {"profile_id": "profile-3", "name": "Gamma Vocalist"}
        with open(profiles_dir / "profile-3.json", "w") as f:
            json.dump(profile3, f)
        (training_dir / "gamma_vocalist").mkdir()
        with open(training_dir / "gamma_vocalist" / "track_vocals.wav", "wb") as f:
            f.write(b"\x00" * 88200)

        return AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

    def test_list_available_profiles(self, populated_bridge):
        """list_available_profiles returns all profiles."""
        profiles = populated_bridge.list_available_profiles()

        assert len(profiles) == 3
        profile_ids = [p[0] for p in profiles]
        assert "profile-1" in profile_ids
        assert "profile-2" in profile_ids
        assert "profile-3" in profile_ids

    def test_list_profiles_shows_lora_status(self, populated_bridge):
        """list_available_profiles shows LoRA availability."""
        profiles = populated_bridge.list_available_profiles()
        profiles_dict = {p[0]: p for p in profiles}

        # Profile 1 and 2 have LoRA
        assert profiles_dict["profile-1"][2] is True
        assert profiles_dict["profile-2"][2] is True
        # Profile 3 has no LoRA
        assert profiles_dict["profile-3"][2] is False

    def test_list_profiles_shows_reference_status(self, populated_bridge):
        """list_available_profiles shows reference audio availability."""
        profiles = populated_bridge.list_available_profiles()
        profiles_dict = {p[0]: p for p in profiles}

        # Profile 1 and 3 have reference audio
        assert profiles_dict["profile-1"][3] is True
        assert profiles_dict["profile-3"][3] is True
        # Profile 2 has no reference audio
        assert profiles_dict["profile-2"][3] is False


class TestCacheManagement:
    """Test cache clearing functionality."""

    def test_clear_cache(self, tmp_path):
        """clear_cache empties both caches."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create test profile
        profile_data = {"profile_id": "cache-test", "name": "Cache Test"}
        with open(profiles_dir / "cache-test.json", "w") as f:
            json.dump(profile_data, f)

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(tmp_path / "loras"),
            device="cpu"
        )

        # Manually populate cache
        bridge._reference_cache["test"] = object()
        bridge._lora_cache["test"] = {"weight": torch.randn(10)}

        assert len(bridge._reference_cache) == 1
        assert len(bridge._lora_cache) == 1

        bridge.clear_cache()

        assert len(bridge._reference_cache) == 0
        assert len(bridge._lora_cache) == 0


class TestSingletonBehavior:
    """Test global singleton instance."""

    def test_get_adapter_bridge_singleton(self):
        """get_adapter_bridge returns singleton instance."""
        from auto_voice.inference import adapter_bridge

        # Reset singleton for test
        adapter_bridge._bridge_instance = None

        with patch.object(Path, 'glob', return_value=[]):
            bridge1 = adapter_bridge.get_adapter_bridge()
            bridge2 = adapter_bridge.get_adapter_bridge()

        assert bridge1 is bridge2

        # Clean up
        adapter_bridge._bridge_instance = None

============================================================
# tests/test_training_job_manager.py
============================================================
"""TDD tests for TrainingJobManager and incremental training jobs.

Task 4.1: Write failing tests for incremental training job creation
Task 4.2: Implement TrainingJobManager with job queue (GPU-only execution)

Tests cover:
- TrainingJob model/dataclass
- TrainingJobManager initialization
- Job creation for voice profiles
- Job states (pending, running, completed, failed)
- Job queue management
- GPU requirement enforcement
"""

import pytest
import tempfile
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_job_storage():
    """Temporary directory for job artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_profile():
    """Mock VoiceProfile for testing."""
    profile = Mock()
    profile.profile_id = "test-profile-123"
    profile.user_id = "user-456"
    profile.name = "Test Voice"
    profile.samples_count = 15
    profile.model_version = "v1"
    return profile


@pytest.fixture
def mock_training_samples():
    """Mock training samples for a profile."""
    samples = []
    for i in range(10):
        sample = Mock()
        sample.sample_id = f"sample-{i}"
        sample.profile_id = "test-profile-123"
        sample.duration_seconds = 5.0 + i * 0.5  # 5-9.5 seconds
        sample.audio_path = f"/data/samples/sample-{i}.wav"
        sample.quality_score = 0.85 + i * 0.01
        samples.append(sample)
    return samples


@pytest.fixture
def job_manager(temp_job_storage):
    """TrainingJobManager instance for testing."""
    from auto_voice.training.job_manager import TrainingJobManager
    return TrainingJobManager(storage_path=temp_job_storage)


# ============================================================================
# Test: TrainingJob Model
# ============================================================================

class TestTrainingJobModel:
    """Tests for TrainingJob dataclass/model."""

    def test_training_job_has_required_fields(self):
        """TrainingJob must have job_id, profile_id, status, created_at."""
        from auto_voice.training.job_manager import TrainingJob

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
        )

        assert job.job_id == "job-001"
        assert job.profile_id == "profile-123"
        assert job.status == "pending"  # Default status
        assert job.created_at is not None
        assert isinstance(job.created_at, datetime)

    def test_training_job_status_values(self):
        """TrainingJob status must be one of: pending, running, completed, failed, cancelled."""
        from auto_voice.training.job_manager import TrainingJob, JobStatus

        # Valid statuses
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_training_job_tracks_progress(self):
        """TrainingJob must track training progress (0-100%)."""
        from auto_voice.training.job_manager import TrainingJob

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
        )

        assert job.progress == 0
        job.update_progress(50)
        assert job.progress == 50
        job.update_progress(100)
        assert job.progress == 100

    def test_training_job_stores_config(self):
        """TrainingJob must store training configuration."""
        from auto_voice.training.job_manager import TrainingJob, TrainingConfig

        config = TrainingConfig(
            learning_rate=1e-4,
            epochs=10,
            batch_size=4,
            lora_rank=8,
            lora_alpha=16,
            use_ewc=True,
            ewc_lambda=1000.0,
        )

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
            config=config,
        )

        assert job.config.learning_rate == 1e-4
        assert job.config.lora_rank == 8
        assert job.config.use_ewc is True

    def test_training_job_tracks_sample_ids(self):
        """TrainingJob must track which samples are used for training."""
        from auto_voice.training.job_manager import TrainingJob

        sample_ids = ["sample-1", "sample-2", "sample-3"]
        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
            sample_ids=sample_ids,
        )

        assert job.sample_ids == sample_ids
        assert len(job.sample_ids) == 3

    def test_training_job_to_dict(self):
        """TrainingJob must be serializable to dict."""
        from auto_voice.training.job_manager import TrainingJob

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
            sample_ids=["s1", "s2"],
        )

        job_dict = job.to_dict()
        assert job_dict["job_id"] == "job-001"
        assert job_dict["profile_id"] == "profile-123"
        assert job_dict["status"] == "pending"
        assert "created_at" in job_dict

    def test_training_job_from_dict(self):
        """TrainingJob must be deserializable from dict."""
        from auto_voice.training.job_manager import TrainingJob

        job_dict = {
            "job_id": "job-002",
            "profile_id": "profile-456",
            "status": "completed",
            "created_at": "2026-01-25T10:00:00",
            "progress": 100,
        }

        job = TrainingJob.from_dict(job_dict)
        assert job.job_id == "job-002"
        assert job.profile_id == "profile-456"
        assert job.status == "completed"
        assert job.progress == 100


# ============================================================================
# Test: TrainingJobManager Initialization
# ============================================================================

class TestTrainingJobManagerInit:
    """Tests for TrainingJobManager initialization."""

    def test_job_manager_initialization(self, temp_job_storage):
        """TrainingJobManager initializes with storage path."""
        from auto_voice.training.job_manager import TrainingJobManager

        manager = TrainingJobManager(storage_path=temp_job_storage)
        assert manager.storage_path == temp_job_storage
        assert manager.is_initialized

    def test_job_manager_creates_storage_directory(self, temp_job_storage):
        """TrainingJobManager creates storage directory if not exists."""
        from auto_voice.training.job_manager import TrainingJobManager

        new_path = temp_job_storage / "jobs"
        manager = TrainingJobManager(storage_path=new_path)
        assert new_path.exists()

    def test_job_manager_has_empty_queue_initially(self, job_manager):
        """TrainingJobManager starts with empty job queue."""
        assert job_manager.queue_size == 0
        assert job_manager.get_pending_jobs() == []

    def test_job_manager_requires_gpu(self, temp_job_storage):
        """TrainingJobManager raises RuntimeError if CUDA unavailable."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch.object(torch.cuda, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                TrainingJobManager(storage_path=temp_job_storage, require_gpu=True)

    def test_job_manager_accepts_gpu_check_skip_for_testing(self, temp_job_storage):
        """TrainingJobManager allows GPU check skip for testing."""
        from auto_voice.training.job_manager import TrainingJobManager

        # Should not raise even if CUDA unavailable
        manager = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False  # Skip GPU check for testing
        )
        assert manager.is_initialized


# ============================================================================
# Test: Job Creation
# ============================================================================

class TestJobCreation:
    """Tests for creating training jobs."""

    def test_create_job_for_profile(self, job_manager, mock_profile, mock_training_samples):
        """Create training job for a voice profile."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=[s.sample_id for s in mock_training_samples],
        )

        assert job is not None
        assert job.job_id is not None
        assert job.profile_id == mock_profile.profile_id
        assert job.status == "pending"
        assert len(job.sample_ids) == len(mock_training_samples)

    def test_create_job_generates_unique_id(self, job_manager, mock_profile):
        """Each job gets a unique ID."""
        job1 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1", "s2"],
        )
        job2 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s3", "s4"],
        )

        assert job1.job_id != job2.job_id

    def test_create_job_with_custom_config(self, job_manager, mock_profile):
        """Create job with custom training configuration."""
        from auto_voice.training.job_manager import TrainingConfig

        config = TrainingConfig(
            learning_rate=5e-5,
            epochs=20,
            lora_rank=16,
        )

        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
            config=config,
        )

        assert job.config.learning_rate == 5e-5
        assert job.config.epochs == 20
        assert job.config.lora_rank == 16

    def test_create_job_requires_samples(self, job_manager, mock_profile):
        """Creating job without samples raises ValueError."""
        with pytest.raises(ValueError, match="sample.*required"):
            job_manager.create_job(
                profile_id=mock_profile.profile_id,
                sample_ids=[],
            )

    def test_create_job_adds_to_queue(self, job_manager, mock_profile):
        """Created job is added to pending queue."""
        assert job_manager.queue_size == 0

        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1", "s2"],
        )

        assert job_manager.queue_size == 1
        pending = job_manager.get_pending_jobs()
        assert len(pending) == 1
        assert pending[0].job_id == job.job_id


# ============================================================================
# Test: Job Queue Management
# ============================================================================

class TestJobQueueManagement:
    """Tests for job queue operations."""

    def test_get_job_by_id(self, job_manager, mock_profile):
        """Retrieve job by its ID."""
        created_job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        retrieved_job = job_manager.get_job(created_job.job_id)
        assert retrieved_job is not None
        assert retrieved_job.job_id == created_job.job_id

    def test_get_nonexistent_job_returns_none(self, job_manager):
        """Getting non-existent job returns None."""
        job = job_manager.get_job("nonexistent-job-id")
        assert job is None

    def test_list_jobs_for_profile(self, job_manager, mock_profile):
        """List all jobs for a specific profile."""
        job1 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job2 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s2"],
        )
        # Job for different profile
        job3 = job_manager.create_job(
            profile_id="other-profile",
            sample_ids=["s3"],
        )

        profile_jobs = job_manager.get_jobs_for_profile(mock_profile.profile_id)
        assert len(profile_jobs) == 2
        job_ids = [j.job_id for j in profile_jobs]
        assert job1.job_id in job_ids
        assert job2.job_id in job_ids
        assert job3.job_id not in job_ids

    def test_cancel_pending_job(self, job_manager, mock_profile):
        """Cancel a pending job."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        success = job_manager.cancel_job(job.job_id)
        assert success is True

        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == "cancelled"

    def test_cancel_running_job(self, job_manager, mock_profile):
        """Cancelling running job sets status to cancelled."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        # Simulate job starting
        job_manager._set_job_status(job.job_id, "running")

        success = job_manager.cancel_job(job.job_id)
        assert success is True

        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == "cancelled"

    def test_cannot_cancel_completed_job(self, job_manager, mock_profile):
        """Cannot cancel already completed job."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager._set_job_status(job.job_id, "completed")

        success = job_manager.cancel_job(job.job_id)
        assert success is False

        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == "completed"  # Unchanged

    def test_get_next_pending_job(self, job_manager, mock_profile):
        """Get next job from queue (FIFO order)."""
        job1 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        time.sleep(0.01)  # Ensure different timestamps
        job2 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s2"],
        )

        next_job = job_manager.get_next_pending_job()
        assert next_job.job_id == job1.job_id  # First in, first out


# ============================================================================
# Test: Job Status Updates
# ============================================================================

class TestJobStatusUpdates:
    """Tests for job status transitions."""

    def test_update_job_status_to_running(self, job_manager, mock_profile):
        """Update job status from pending to running."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        job_manager.update_job_status(job.job_id, "running")
        updated = job_manager.get_job(job.job_id)
        assert updated.status == "running"
        assert updated.started_at is not None

    def test_update_job_status_to_completed(self, job_manager, mock_profile):
        """Update job status to completed with results."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager.update_job_status(job.job_id, "running")

        results = {
            "adapter_path": "/models/profile-123/adapter_v2.pt",
            "metrics": {
                "speaker_similarity": 0.92,
                "loss_final": 0.015,
            }
        }

        job_manager.update_job_status(job.job_id, "completed", results=results)
        updated = job_manager.get_job(job.job_id)

        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.results["adapter_path"] == results["adapter_path"]
        assert updated.results["metrics"]["speaker_similarity"] == 0.92

    def test_update_job_status_to_failed(self, job_manager, mock_profile):
        """Update job status to failed with error message."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager.update_job_status(job.job_id, "running")

        error_msg = "CUDA out of memory"
        job_manager.update_job_status(job.job_id, "failed", error=error_msg)

        updated = job_manager.get_job(job.job_id)
        assert updated.status == "failed"
        assert updated.error == error_msg

    def test_update_job_progress(self, job_manager, mock_profile):
        """Update job training progress."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager.update_job_status(job.job_id, "running")

        job_manager.update_job_progress(job.job_id, 25)
        assert job_manager.get_job(job.job_id).progress == 25

        job_manager.update_job_progress(job.job_id, 75)
        assert job_manager.get_job(job.job_id).progress == 75

    def test_invalid_status_transition_raises(self, job_manager, mock_profile):
        """Invalid status transitions raise ValueError."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        # Cannot go from pending directly to completed
        with pytest.raises(ValueError, match="Invalid.*transition"):
            job_manager.update_job_status(job.job_id, "completed")


# ============================================================================
# Test: GPU Enforcement
# ============================================================================

class TestGPUEnforcement:
    """Tests for GPU-only execution requirement."""

    def test_job_execution_requires_cuda(self, temp_job_storage, mock_profile):
        """Job execution raises RuntimeError if CUDA unavailable."""
        from auto_voice.training.job_manager import TrainingJobManager

        # Create manager with GPU check disabled (for queue operations)
        manager = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        job = manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        # But execution should fail without GPU
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required.*training"):
                manager.execute_job(job.job_id)

    def test_job_tracks_gpu_device(self, job_manager, mock_profile):
        """Job records which GPU device was used."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        # Simulate job running on GPU 0
        job_manager.update_job_status(job.job_id, "running", gpu_device=0)

        updated = job_manager.get_job(job.job_id)
        assert updated.gpu_device == 0


# ============================================================================
# Test: Job Persistence
# ============================================================================

class TestJobPersistence:
    """Tests for job state persistence."""

    def test_jobs_persist_to_storage(self, temp_job_storage, mock_profile):
        """Jobs are persisted to storage directory."""
        from auto_voice.training.job_manager import TrainingJobManager

        manager1 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        job = manager1.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1", "s2"],
        )
        job_id = job.job_id

        # Create new manager instance - should load existing jobs
        manager2 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        loaded_job = manager2.get_job(job_id)
        assert loaded_job is not None
        assert loaded_job.profile_id == mock_profile.profile_id
        assert loaded_job.sample_ids == ["s1", "s2"]

    def test_job_status_updates_persist(self, temp_job_storage, mock_profile):
        """Job status updates are persisted."""
        from auto_voice.training.job_manager import TrainingJobManager

        manager1 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        job = manager1.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        manager1.update_job_status(job.job_id, "running")
        manager1.update_job_progress(job.job_id, 50)

        # Reload
        manager2 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        loaded_job = manager2.get_job(job.job_id)
        assert loaded_job.status == "running"
        assert loaded_job.progress == 50


# ============================================================================
# Test: TrainingConfig Defaults
# ============================================================================

class TestTrainingConfigDefaults:
    """Tests for TrainingConfig with sensible defaults."""

    def test_training_config_defaults(self):
        """TrainingConfig has sensible defaults from SOTA research."""
        from auto_voice.training.job_manager import TrainingConfig

        config = TrainingConfig()

        # LoRA defaults from research doc
        assert config.lora_rank == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1

        # Training defaults
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.epochs == 10

        # EWC defaults
        assert config.use_ewc is True
        assert config.ewc_lambda == 1000.0

    def test_training_config_serialization(self):
        """TrainingConfig serializes to/from dict."""
        from auto_voice.training.job_manager import TrainingConfig

        config = TrainingConfig(
            learning_rate=5e-5,
            epochs=20,
        )

        config_dict = config.to_dict()
        assert config_dict["learning_rate"] == 5e-5
        assert config_dict["epochs"] == 20

        loaded = TrainingConfig.from_dict(config_dict)
        assert loaded.learning_rate == 5e-5
        assert loaded.epochs == 20

============================================================
# tests/test_pipeline_factory.py
============================================================
"""Tests for PipelineFactory - unified pipeline management.

The PipelineFactory provides:
- Lazy loading: Pipelines only initialized when first requested
- Caching: Re-uses existing pipeline instances
- Memory management: Can unload pipelines to free GPU memory
- Unified interface: All pipelines accessible via same API

Tests cover:
- Singleton behavior
- Lazy loading verification
- Pipeline type routing
- Memory tracking
- Pipeline unloading
"""
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


class TestPipelineFactoryInit:
    """Test PipelineFactory initialization."""

    def test_import_succeeds(self):
        """PipelineFactory can be imported."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        assert PipelineFactory is not None

    def test_singleton_pattern(self):
        """get_instance returns singleton."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        # Reset singleton
        PipelineFactory.reset_instance()

        factory1 = PipelineFactory.get_instance()
        factory2 = PipelineFactory.get_instance()

        assert factory1 is factory2

        # Clean up
        PipelineFactory.reset_instance()

    def test_reset_instance_clears_singleton(self):
        """reset_instance clears the singleton."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        factory1 = PipelineFactory.get_instance()
        PipelineFactory.reset_instance()
        factory2 = PipelineFactory.get_instance()

        assert factory1 is not factory2

        # Clean up
        PipelineFactory.reset_instance()

    def test_init_with_device(self):
        """Factory accepts device parameter."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device("cpu"))

        assert factory.device == torch.device("cpu")

        PipelineFactory.reset_instance()

    def test_init_default_device_cuda_if_available(self):
        """Factory defaults to CUDA if available."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        if torch.cuda.is_available():
            assert factory.device.type == "cuda"
        else:
            assert factory.device.type == "cpu"

        PipelineFactory.reset_instance()

    def test_init_creates_empty_caches(self):
        """Factory starts with empty pipeline caches."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        assert len(factory._pipelines) == 0
        assert len(factory._memory_usage) == 0

        PipelineFactory.reset_instance()


class TestPipelineTypeRouting:
    """Test pipeline type routing in get_pipeline."""

    def test_invalid_pipeline_type_raises(self):
        """get_pipeline raises ValueError for invalid type."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with pytest.raises(ValueError, match="Unknown pipeline type"):
            factory.get_pipeline("invalid_type")

        PipelineFactory.reset_instance()

    def test_valid_pipeline_types(self):
        """get_pipeline accepts all valid pipeline types."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        valid_types = ['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Just check validation doesn't raise - actual creation mocked
        for pt in valid_types:
            # Mock the creation to avoid loading real models
            with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
                pipeline = factory.get_pipeline(pt)
                assert pipeline is not None

        PipelineFactory.reset_instance()


class TestLazyLoading:
    """Test lazy loading behavior."""

    def test_pipeline_not_loaded_until_requested(self):
        """Pipelines are not loaded on factory init."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # No pipelines loaded yet
        assert not factory.is_loaded('realtime')
        assert not factory.is_loaded('quality')
        assert not factory.is_loaded('quality_seedvc')
        assert not factory.is_loaded('realtime_meanvc')

        PipelineFactory.reset_instance()

    def test_pipeline_loaded_on_first_request(self):
        """Pipeline loads on first get_pipeline call."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        mock_pipeline = MagicMock()
        with patch.object(factory, '_create_pipeline', return_value=mock_pipeline) as mock_create:
            pipeline = factory.get_pipeline('realtime')

            assert factory.is_loaded('realtime')
            mock_create.assert_called_once_with('realtime', None)

        PipelineFactory.reset_instance()

    def test_pipeline_reused_on_second_request(self):
        """Subsequent requests return cached pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        mock_pipeline = MagicMock()
        with patch.object(factory, '_create_pipeline', return_value=mock_pipeline) as mock_create:
            pipeline1 = factory.get_pipeline('realtime')
            pipeline2 = factory.get_pipeline('realtime')

            assert pipeline1 is pipeline2
            # Only created once
            assert mock_create.call_count == 1

        PipelineFactory.reset_instance()


class TestPipelineCaching:
    """Test pipeline caching behavior."""

    def test_different_types_have_different_instances(self):
        """Different pipeline types create different instances."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        realtime_mock = MagicMock()
        quality_mock = MagicMock()

        def create_side_effect(pipeline_type, profile_store):
            if pipeline_type == 'realtime':
                return realtime_mock
            elif pipeline_type == 'quality':
                return quality_mock
            return MagicMock()

        with patch.object(factory, '_create_pipeline', side_effect=create_side_effect):
            realtime = factory.get_pipeline('realtime')
            quality = factory.get_pipeline('quality')

            assert realtime is not quality
            assert realtime is realtime_mock
            assert quality is quality_mock

        PipelineFactory.reset_instance()

    def test_is_loaded_returns_correct_status(self):
        """is_loaded correctly reports pipeline status."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            assert not factory.is_loaded('realtime')

            factory.get_pipeline('realtime')
            assert factory.is_loaded('realtime')
            assert not factory.is_loaded('quality')

        PipelineFactory.reset_instance()


class TestMemoryTracking:
    """Test GPU memory tracking."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage_tracked(self):
        """Memory usage is tracked for loaded pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance(device=torch.device("cuda"))

        # Create a pipeline that allocates some GPU memory
        def create_with_memory(pt, ps):
            mock = MagicMock()
            # Allocate some memory
            torch.cuda.empty_cache()
            _ = torch.randn(1000, 1000, device="cuda")
            return mock

        with patch.object(factory, '_create_pipeline', side_effect=create_with_memory):
            factory.get_pipeline('realtime')

            # Should have some memory recorded
            assert factory.get_memory_usage('realtime') >= 0

        PipelineFactory.reset_instance()

    def test_get_memory_usage_returns_zero_for_unloaded(self):
        """get_memory_usage returns 0 for unloaded pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        assert factory.get_memory_usage('realtime') == 0.0
        assert factory.get_memory_usage('quality') == 0.0

        PipelineFactory.reset_instance()

    def test_get_total_memory_usage(self):
        """get_total_memory_usage sums all pipeline memory."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Manually set memory usage for testing
        factory._memory_usage['realtime'] = 1.5
        factory._memory_usage['quality'] = 2.5

        assert factory.get_total_memory_usage() == 4.0

        PipelineFactory.reset_instance()


class TestPipelineUnloading:
    """Test pipeline unloading functionality."""

    def test_unload_pipeline_removes_from_cache(self):
        """unload_pipeline removes pipeline from cache."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            assert factory.is_loaded('realtime')

            result = factory.unload_pipeline('realtime')

            assert result is True
            assert not factory.is_loaded('realtime')

        PipelineFactory.reset_instance()

    def test_unload_pipeline_clears_memory_tracking(self):
        """unload_pipeline clears memory tracking."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory._memory_usage['realtime'] = 1.5

            factory.unload_pipeline('realtime')

            assert factory.get_memory_usage('realtime') == 0.0

        PipelineFactory.reset_instance()

    def test_unload_pipeline_returns_false_if_not_loaded(self):
        """unload_pipeline returns False for unloaded pipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        result = factory.unload_pipeline('realtime')
        assert result is False

        PipelineFactory.reset_instance()

    def test_unload_all(self):
        """unload_all removes all cached pipelines."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')
            factory.get_pipeline('quality')

            assert factory.is_loaded('realtime')
            assert factory.is_loaded('quality')

            factory.unload_all()

            assert not factory.is_loaded('realtime')
            assert not factory.is_loaded('quality')

        PipelineFactory.reset_instance()


class TestGetStatus:
    """Test get_status for API responses."""

    def test_get_status_returns_all_pipeline_info(self):
        """get_status returns info for all pipeline types."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert 'realtime' in status
        assert 'quality' in status
        assert 'quality_seedvc' in status
        assert 'realtime_meanvc' in status

        PipelineFactory.reset_instance()

    def test_get_status_shows_loaded_state(self):
        """get_status shows whether pipelines are loaded."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        # All should be unloaded initially
        assert status['realtime']['loaded'] is False
        assert status['quality']['loaded'] is False

        with patch.object(factory, '_create_pipeline', return_value=MagicMock()):
            factory.get_pipeline('realtime')

            status = factory.get_status()
            assert status['realtime']['loaded'] is True
            assert status['quality']['loaded'] is False

        PipelineFactory.reset_instance()

    def test_get_status_includes_sample_rates(self):
        """get_status includes sample rate info."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert status['realtime']['sample_rate'] == 22050
        assert status['quality']['sample_rate'] == 24000
        assert status['quality_seedvc']['sample_rate'] == 44100
        assert status['realtime_meanvc']['sample_rate'] == 16000

        PipelineFactory.reset_instance()

    def test_get_status_includes_latency_targets(self):
        """get_status includes latency targets."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert status['realtime']['latency_target_ms'] == 100
        assert status['quality']['latency_target_ms'] == 3000
        assert status['quality_seedvc']['latency_target_ms'] == 2000
        assert status['realtime_meanvc']['latency_target_ms'] == 80

        PipelineFactory.reset_instance()

    def test_get_status_includes_descriptions(self):
        """get_status includes pipeline descriptions."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        status = factory.get_status()

        assert 'description' in status['realtime']
        assert 'description' in status['quality']
        assert 'karaoke' in status['realtime']['description'].lower()
        assert 'features' in status['quality_seedvc']

        PipelineFactory.reset_instance()


class TestPipelineCreation:
    """Test actual pipeline creation logic."""

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_create_realtime_pipeline(self):
        """_create_pipeline creates RealtimePipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # This loads real models - slow
        pipeline = factory._create_pipeline('realtime', None)

        assert isinstance(pipeline, RealtimePipeline)

        PipelineFactory.reset_instance()

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_create_quality_pipeline(self):
        """_create_pipeline creates SOTAConversionPipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        pipeline = factory._create_pipeline('quality', None)

        assert isinstance(pipeline, SOTAConversionPipeline)

        PipelineFactory.reset_instance()

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_create_quality_seedvc_pipeline(self):
        """_create_pipeline creates SeedVCPipeline."""
        from auto_voice.inference.pipeline_factory import PipelineFactory
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        pipeline = factory._create_pipeline('quality_seedvc', None)

        assert isinstance(pipeline, SeedVCPipeline)

        PipelineFactory.reset_instance()

    def test_create_realtime_meanvc_uses_cpu(self):
        """MeanVC pipeline uses CPU by default."""
        from auto_voice.inference.pipeline_factory import PipelineFactory

        PipelineFactory.reset_instance()
        factory = PipelineFactory.get_instance()

        # Mock to avoid loading real models - patch at the source module
        with patch('auto_voice.inference.meanvc_pipeline.MeanVCPipeline') as MockMeanVC:
            mock_instance = MagicMock()
            MockMeanVC.return_value = mock_instance

            pipeline = factory._create_pipeline('realtime_meanvc', None)

            # Should be called with CPU device
            MockMeanVC.assert_called_once()
            call_kwargs = MockMeanVC.call_args[1]
            assert call_kwargs['device'] == torch.device('cpu')
            assert call_kwargs['require_gpu'] is False

        PipelineFactory.reset_instance()

============================================================
# tests/test_voice_profile_db.py
============================================================
"""Tests for VoiceProfile database models and operations.

Task 1.2: Test PostgreSQL schema for voice_profiles and training_samples tables.
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auto_voice.profiles.db.models import Base, VoiceProfileDB, TrainingSampleDB


@pytest.fixture
def test_engine():
    """Create a SQLite in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


class TestVoiceProfileDB:
    """Test VoiceProfileDB SQLAlchemy model."""

    def test_create_voice_profile(self, test_session):
        """Profile can be created and persisted."""
        profile = VoiceProfileDB(
            user_id="user-123",
            name="My Singing Voice",
        )
        test_session.add(profile)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(VoiceProfileDB).filter_by(user_id="user-123").first()
        assert retrieved is not None
        assert retrieved.user_id == "user-123"
        assert retrieved.name == "My Singing Voice"
        assert retrieved.samples_count == 0
        assert retrieved.model_version is None

    def test_profile_id_is_uuid(self, test_session):
        """Profile ID is a valid UUID string."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        # Should be valid UUID
        UUID(profile.id)  # Raises if invalid

    def test_profile_timestamps(self, test_session):
        """Profile has created and updated timestamps."""
        before = datetime.now(timezone.utc)
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()
        after = datetime.now(timezone.utc)

        assert profile.created is not None
        assert profile.updated is not None
        # Note: SQLite doesn't have timezone support, so we just check existence

    def test_profile_to_dict(self, test_session):
        """Profile can be serialized to dictionary."""
        profile = VoiceProfileDB(
            user_id="user-123",
            name="Test Profile",
            model_version="v1.0.0",
        )
        test_session.add(profile)
        test_session.commit()

        data = profile.to_dict()
        assert data["user_id"] == "user-123"
        assert data["name"] == "Test Profile"
        assert data["model_version"] == "v1.0.0"
        assert "id" in data
        assert "created" in data
        assert "updated" in data

    def test_profile_settings_json(self, test_session):
        """Profile can store JSON settings."""
        profile = VoiceProfileDB(
            user_id="user-1",
            name="Test",
            settings={"pitch_shift": 2, "formant_shift": 0.5},
        )
        test_session.add(profile)
        test_session.commit()

        retrieved = test_session.query(VoiceProfileDB).filter_by(user_id="user-1").first()
        assert retrieved.settings["pitch_shift"] == 2
        assert retrieved.settings["formant_shift"] == 0.5


class TestTrainingSampleDB:
    """Test TrainingSampleDB SQLAlchemy model."""

    def test_create_training_sample(self, test_session):
        """Training sample can be created with profile reference."""
        # Create profile first
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        # Create sample
        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=5.5,
            sample_rate=24000,
        )
        test_session.add(sample)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(TrainingSampleDB).filter_by(profile_id=profile.id).first()
        assert retrieved is not None
        assert retrieved.audio_path == "/data/sample.wav"
        assert retrieved.duration_seconds == 5.5
        assert retrieved.sample_rate == 24000

    def test_sample_quality_score(self, test_session):
        """Sample can have quality score."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
            quality_score=0.85,
        )
        test_session.add(sample)
        test_session.commit()

        assert sample.quality_score == 0.85

    def test_sample_metadata(self, test_session):
        """Sample can store extra metadata."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
            extra_metadata={
                "song_id": "song-456",
                "pitch_range": [200, 800],
                "snr_db": 25.5,
            },
        )
        test_session.add(sample)
        test_session.commit()

        retrieved = test_session.query(TrainingSampleDB).filter_by(id=sample.id).first()
        assert retrieved.extra_metadata["song_id"] == "song-456"
        assert retrieved.extra_metadata["pitch_range"] == [200, 800]

    def test_sample_to_dict(self, test_session):
        """Sample can be serialized to dictionary."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
        )
        test_session.add(sample)
        test_session.commit()

        data = sample.to_dict()
        assert data["profile_id"] == profile.id
        assert data["audio_path"] == "/data/sample.wav"
        assert data["duration_seconds"] == 3.0
        assert "id" in data
        assert "created" in data

    def test_cascade_delete(self, test_session):
        """Samples are deleted when profile is deleted."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()
        profile_id = profile.id

        # Add samples
        for i in range(3):
            sample = TrainingSampleDB(
                profile_id=profile_id,
                audio_path=f"/data/sample{i}.wav",
                duration_seconds=3.0,
                sample_rate=24000,
            )
            test_session.add(sample)
        test_session.commit()

        # Verify samples exist
        count = test_session.query(TrainingSampleDB).filter_by(profile_id=profile_id).count()
        assert count == 3

        # Delete profile
        test_session.delete(profile)
        test_session.commit()

        # Samples should be gone
        count = test_session.query(TrainingSampleDB).filter_by(profile_id=profile_id).count()
        assert count == 0

    def test_processing_status(self, test_session):
        """Sample tracks processing status."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
        )
        test_session.add(sample)
        test_session.commit()

        assert sample.processed == 0  # False
        assert sample.processed_at is None

        # Mark as processed
        sample.processed = 1
        sample.processed_at = datetime.now(timezone.utc)
        test_session.commit()

        retrieved = test_session.query(TrainingSampleDB).filter_by(id=sample.id).first()
        assert retrieved.processed == 1
        assert retrieved.processed_at is not None


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_profile_sample_relationship(self, test_session):
        """Profile has relationship to samples."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        # Add samples
        for i in range(5):
            sample = TrainingSampleDB(
                profile_id=profile.id,
                audio_path=f"/data/sample{i}.wav",
                duration_seconds=3.0 + i,
                sample_rate=24000,
            )
            test_session.add(sample)
        test_session.commit()

        # Access via relationship
        test_session.refresh(profile)
        assert profile.samples.count() == 5

    def test_multiple_profiles_per_user(self, test_session):
        """User can have multiple profiles."""
        for i in range(3):
            profile = VoiceProfileDB(
                user_id="user-1",
                name=f"Profile {i}",
            )
            test_session.add(profile)
        test_session.commit()

        profiles = test_session.query(VoiceProfileDB).filter_by(user_id="user-1").all()
        assert len(profiles) == 3

============================================================
# tests/test_shortcut_flow_matching.py
============================================================
"""
Tests for shortcut flow matching implementation.

Validates that the ShortcutFlowMatching wrapper correctly:
1. Adds step size embedding capability
2. Performs shortcut inference with configurable step counts
3. Computes self-consistency loss correctly
"""

import pytest
import torch
import sys
import os

# Add Seed-VC modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/seed-vc'))

from modules.shortcut_flow_matching import ShortcutFlowMatching, StepSizeEmbedder, enable_shortcut_cfm


class MockCFMEstimator(torch.nn.Module):
    """Mock DiT estimator for testing."""

    def __init__(self, in_channels=128, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        # Simple linear layer to simulate DiT
        self.linear = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x, prompt_x, x_lens, t, style, mu, prompt_lens=None):
        """Mock forward pass."""
        # Just return a simple transformation
        B, C, T = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, C)
        out_flat = self.linear(x_flat)
        out = out_flat.reshape(B, T, C).transpose(1, 2)
        return out

    def setup_caches(self, max_batch_size, max_seq_length):
        """Mock cache setup."""
        pass


class MockCFM(torch.nn.Module):
    """Mock CFM model for testing."""

    def __init__(self, in_channels=128, hidden_dim=256):
        super().__init__()
        self.estimator = MockCFMEstimator(in_channels, hidden_dim)
        self.in_channels = in_channels
        self.sigma_min = 1e-6
        self.zero_prompt_speech_token = False
        self.criterion = torch.nn.MSELoss()

    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Mock CFM forward (standard flow matching loss)."""
        b, _, t = x1.shape
        device = x1.device

        # Sample random time
        t_sample = torch.rand([b, 1, 1], device=device, dtype=x1.dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Noisy sample
        y = (1 - (1 - self.sigma_min) * t_sample) * z + t_sample * x1

        # Target velocity
        u = x1 - (1 - self.sigma_min) * z

        # Prepare prompt
        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            y[bib, :, :prompt_lens[bib]] = 0

        # Estimate
        estimator_out = self.estimator(y, prompt, x_lens, t_sample.squeeze(1).squeeze(1), style, mu, prompt_lens)

        # Compute loss
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]],
                u[bib, :, prompt_lens[bib]:x_lens[bib]]
            )
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z


@pytest.fixture
def mock_cfm():
    """Create mock CFM model."""
    return MockCFM(in_channels=128, hidden_dim=256)


@pytest.fixture
def shortcut_cfm(mock_cfm):
    """Create shortcut CFM wrapper."""
    return enable_shortcut_cfm(mock_cfm, hidden_dim=256)


@pytest.mark.smoke
def test_step_size_embedder():
    """Test that step size embedder produces correct shapes."""
    embedder = StepSizeEmbedder(hidden_size=256)

    # Test with various batch sizes
    for batch_size in [1, 4, 8]:
        d = torch.rand(batch_size)
        d_emb = embedder(d)

        assert d_emb.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {d_emb.shape}"
        assert not torch.isnan(d_emb).any(), "Step size embedding contains NaN"
        assert not torch.isinf(d_emb).any(), "Step size embedding contains Inf"


@pytest.mark.smoke
def test_shortcut_cfm_initialization(shortcut_cfm):
    """Test that ShortcutFlowMatching initializes correctly."""
    assert shortcut_cfm.d_embedder is not None, "Step size embedder not initialized"
    assert hasattr(shortcut_cfm, 'base_cfm'), "Base CFM not stored"
    assert shortcut_cfm.k_flow_matching == 0.7, "Incorrect batch split ratio"


@pytest.mark.smoke
def test_shortcut_inference_shapes(shortcut_cfm):
    """Test that shortcut inference produces correct output shapes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    mu = torch.randn(B, T, 256, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt = torch.randn(in_channels, 20, device=device)
    style = torch.randn(B, style_dim, device=device)
    f0 = None

    # Test with different step counts
    for n_steps in [1, 2, 5, 10]:
        output = shortcut_cfm.shortcut_inference(
            mu, x_lens, prompt, style, f0, n_timesteps=n_steps
        )

        assert output.shape == (B, in_channels, T), \
            f"Expected shape ({B}, {in_channels}, {T}), got {output.shape} for {n_steps} steps"
        assert not torch.isnan(output).any(), f"Output contains NaN for {n_steps} steps"


@pytest.mark.smoke
def test_flow_matching_loss(shortcut_cfm):
    """Test that FM loss computation works."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    x1 = torch.randn(B, in_channels, T, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt_lens = torch.tensor([20, 20], device=device)
    mu = torch.randn(B, T, 256, device=device)
    style = torch.randn(B, style_dim, device=device)

    # Compute FM loss
    loss, output = shortcut_cfm._flow_matching_loss(x1, x_lens, prompt_lens, mu, style)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert output.shape == x1.shape, f"Output shape mismatch: {output.shape} vs {x1.shape}"


@pytest.mark.smoke
def test_self_consistency_loss(shortcut_cfm):
    """Test that self-consistency loss computation works."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    x1 = torch.randn(B, in_channels, T, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt_lens = torch.tensor([20, 20], device=device)
    mu = torch.randn(B, T, 256, device=device)
    style = torch.randn(B, style_dim, device=device)

    # Compute SC loss
    loss, output = shortcut_cfm._self_consistency_loss(x1, x_lens, prompt_lens, mu, style)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert output.shape == x1.shape, f"Output shape mismatch: {output.shape} vs {x1.shape}"


@pytest.mark.smoke
def test_dual_objective_training(shortcut_cfm):
    """Test that training correctly alternates between FM and SC objectives."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    x1 = torch.randn(B, in_channels, T, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt_lens = torch.tensor([20, 20], device=device)
    mu = torch.randn(B, T, 256, device=device)
    style = torch.randn(B, style_dim, device=device)

    # Run multiple training steps and count objective types
    fm_count = 0
    sc_count = 0
    n_trials = 100

    torch.manual_seed(42)  # For reproducibility
    for _ in range(n_trials):
        loss, output, obj_type = shortcut_cfm.forward(x1, x_lens, prompt_lens, mu, style, training=True)

        assert obj_type in ["FM", "SC"], f"Unknown objective type: {obj_type}"
        assert loss.item() >= 0, "Loss should be non-negative"

        if obj_type == "FM":
            fm_count += 1
        else:
            sc_count += 1

    # Check that ratio is approximately 70/30
    fm_ratio = fm_count / n_trials
    assert 0.6 < fm_ratio < 0.8, f"FM ratio {fm_ratio} outside expected range (0.6-0.8)"


@pytest.mark.integration
def test_shortcut_vs_baseline_consistency(shortcut_cfm):
    """
    Test that shortcut inference with many steps produces similar results
    to baseline CFM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 1, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    mu = torch.randn(B, T, 256, device=device)
    x_lens = torch.tensor([T], device=device)
    prompt = torch.randn(in_channels, 20, device=device)
    style = torch.randn(B, style_dim, device=device)
    f0 = None

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Run with many steps (should be close to baseline)
    output_10 = shortcut_cfm.shortcut_inference(
        mu, x_lens, prompt, style, f0, n_timesteps=10, temperature=1.0
    )

    # Run with fewer steps
    torch.manual_seed(42)
    output_2 = shortcut_cfm.shortcut_inference(
        mu, x_lens, prompt, style, f0, n_timesteps=2, temperature=1.0
    )

    # Both should have valid outputs (but may differ in quality)
    assert not torch.isnan(output_10).any(), "10-step output contains NaN"
    assert not torch.isnan(output_2).any(), "2-step output contains NaN"
    assert output_10.shape == output_2.shape, "Shape mismatch between step counts"


if __name__ == "__main__":
    # Run smoke tests
    print("Testing StepSizeEmbedder...")
    test_step_size_embedder()
    print("✓ StepSizeEmbedder tests passed")

    print("\nTesting ShortcutFlowMatching initialization...")
    mock_cfm = MockCFM()
    shortcut_cfm = enable_shortcut_cfm(mock_cfm, hidden_dim=256)
    test_shortcut_cfm_initialization(shortcut_cfm)
    print("✓ Initialization tests passed")

    print("\nTesting shortcut inference shapes...")
    test_shortcut_inference_shapes(shortcut_cfm)
    print("✓ Inference shape tests passed")

    print("\nTesting FM loss...")
    test_flow_matching_loss(shortcut_cfm)
    print("✓ FM loss tests passed")

    print("\nTesting SC loss...")
    test_self_consistency_loss(shortcut_cfm)
    print("✓ SC loss tests passed")

    print("\nTesting dual objective training...")
    test_dual_objective_training(shortcut_cfm)
    print("✓ Dual objective tests passed")

    print("\n✅ All tests passed!")

============================================================
# tests/test_adapter_manager.py
============================================================
"""Unit tests for AdapterManager.

Tests the adapter loading, caching, validation, and application functionality
for voice conversion adapters used across both REALTIME and QUALITY pipelines.
"""
import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch

from auto_voice.models.adapter_manager import (
    AdapterManager,
    AdapterManagerConfig,
    AdapterCache,
    AdapterInfo,
    load_adapter_for_profile,
    get_trained_profiles,
    get_adapter_manager,
)


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for adapters and profiles."""
    adapters_dir = tmp_path / "trained_models"
    profiles_dir = tmp_path / "voice_profiles"
    adapters_dir.mkdir()
    profiles_dir.mkdir()
    return {
        "adapters_dir": adapters_dir,
        "profiles_dir": profiles_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def adapter_config(temp_dirs):
    """Create AdapterManagerConfig with temp directories."""
    return AdapterManagerConfig(
        adapters_dir=temp_dirs["adapters_dir"],
        profiles_dir=temp_dirs["profiles_dir"],
        cache_size=3,
        device="cpu",  # Use CPU for tests
        auto_validate=True,
    )


@pytest.fixture
def mock_adapter_state():
    """Create a mock adapter state dict with LoRA structure."""
    return {
        "lora_adapters.content_proj.lora_A": torch.randn(8, 256),
        "lora_adapters.content_proj.lora_B": torch.randn(256, 8),
        "lora_adapters.output.lora_A": torch.randn(8, 512),
        "lora_adapters.output.lora_B": torch.randn(512, 8),
    }


@pytest.fixture
def mock_profile_metadata():
    """Create mock profile metadata."""
    return {
        "name": "Test Voice",
        "created_at": "2026-01-30T12:00:00Z",
        "sample_count": 10,
        "adapter_version": "1.0",
        "adapter_target_modules": ["content_proj", "output"],
        "adapter_rank": 8,
        "adapter_alpha": 16,
        "training_epochs": 50,
        "loss_final": 0.023,
    }


@pytest.fixture
def create_test_adapter(temp_dirs, mock_adapter_state, mock_profile_metadata):
    """Helper to create test adapter files."""
    def _create(profile_id: str):
        # Save adapter weights
        adapter_path = temp_dirs["adapters_dir"] / f"{profile_id}_adapter.pt"
        torch.save(mock_adapter_state, adapter_path)

        # Save profile metadata
        profile_path = temp_dirs["profiles_dir"] / f"{profile_id}.json"
        with open(profile_path, "w") as f:
            json.dump(mock_profile_metadata, f)

        return adapter_path, profile_path

    return _create


class TestAdapterCache:
    """Test LRU cache implementation."""

    @pytest.mark.smoke
    def test_cache_init(self):
        cache = AdapterCache(max_size=3)
        assert len(cache) == 0
        assert cache.max_size == 3

    @pytest.mark.smoke
    def test_cache_put_and_get(self):
        cache = AdapterCache(max_size=3)
        state = {"key": torch.tensor([1.0])}

        cache.put("profile1", state)
        assert len(cache) == 1
        assert "profile1" in cache

        retrieved = cache.get("profile1")
        assert retrieved is not None
        assert "key" in retrieved

    def test_cache_lru_eviction(self):
        cache = AdapterCache(max_size=2)

        cache.put("p1", {"data": torch.tensor([1.0])})
        cache.put("p2", {"data": torch.tensor([2.0])})
        cache.put("p3", {"data": torch.tensor([3.0])})  # Should evict p1

        assert len(cache) == 2
        assert "p1" not in cache
        assert "p2" in cache
        assert "p3" in cache

    def test_cache_lru_order_update(self):
        cache = AdapterCache(max_size=2)

        cache.put("p1", {"data": torch.tensor([1.0])})
        cache.put("p2", {"data": torch.tensor([2.0])})

        # Access p1, making it most recent
        _ = cache.get("p1")

        # Add p3, should evict p2 (least recent)
        cache.put("p3", {"data": torch.tensor([3.0])})

        assert "p1" in cache
        assert "p2" not in cache
        assert "p3" in cache

    def test_cache_clear(self):
        cache = AdapterCache(max_size=3)
        cache.put("p1", {"data": torch.tensor([1.0])})
        cache.put("p2", {"data": torch.tensor([2.0])})

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        assert "p1" not in cache


class TestAdapterManagerInit:
    """Test AdapterManager initialization."""

    @pytest.mark.smoke
    def test_init_default_config(self, temp_dirs):
        manager = AdapterManager()
        assert manager.config is not None
        assert manager.device is not None
        assert isinstance(manager._cache, AdapterCache)

    @pytest.mark.smoke
    def test_init_custom_config(self, adapter_config):
        manager = AdapterManager(adapter_config)
        assert manager.config.cache_size == 3
        assert manager.config.device == "cpu"
        assert manager.device == torch.device("cpu")

    def test_init_creates_directories(self, adapter_config, temp_dirs):
        # Remove adapters_dir to test creation
        temp_dirs["adapters_dir"].rmdir()

        manager = AdapterManager(adapter_config)

        assert temp_dirs["adapters_dir"].exists()


class TestAdapterManagerListing:
    """Test adapter listing functionality."""

    @pytest.mark.smoke
    def test_list_available_adapters_empty(self, adapter_config):
        manager = AdapterManager(adapter_config)
        adapters = manager.list_available_adapters()
        assert adapters == []

    def test_list_available_adapters(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")
        create_test_adapter("profile2")

        manager = AdapterManager(adapter_config)
        adapters = manager.list_available_adapters()

        assert len(adapters) == 2
        assert "profile1" in adapters
        assert "profile2" in adapters

    @pytest.mark.smoke
    def test_has_adapter_false(self, adapter_config):
        manager = AdapterManager(adapter_config)
        assert manager.has_adapter("nonexistent") is False

    def test_has_adapter_true(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        assert manager.has_adapter("profile1") is True

    def test_get_adapter_path_none(self, adapter_config):
        manager = AdapterManager(adapter_config)
        path = manager.get_adapter_path("nonexistent")
        assert path is None

    def test_get_adapter_path_exists(self, adapter_config, create_test_adapter):
        adapter_path, _ = create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        path = manager.get_adapter_path("profile1")

        assert path == adapter_path


class TestAdapterManagerLoading:
    """Test adapter loading functionality."""

    def test_load_adapter_success(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        state_dict = manager.load_adapter("profile1")

        assert state_dict is not None
        assert "lora_adapters.content_proj.lora_A" in state_dict
        assert "lora_adapters.content_proj.lora_B" in state_dict
        assert isinstance(state_dict["lora_adapters.content_proj.lora_A"], torch.Tensor)

    def test_load_adapter_not_found(self, adapter_config):
        manager = AdapterManager(adapter_config)

        with pytest.raises(FileNotFoundError, match="No adapter found for profile"):
            manager.load_adapter("nonexistent")

    def test_load_adapter_caching(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        # First load
        state1 = manager.load_adapter("profile1", use_cache=True)
        assert "profile1" in manager._cache

        # Second load (should hit cache)
        state2 = manager.load_adapter("profile1", use_cache=True)

        # Should be same object (from cache)
        assert state1 is state2

    def test_load_adapter_no_cache(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        state1 = manager.load_adapter("profile1", use_cache=False)
        assert "profile1" not in manager._cache

        state2 = manager.load_adapter("profile1", use_cache=False)

        # Should be different objects (not cached)
        assert state1 is not state2


class TestAdapterValidation:
    """Test adapter validation functionality."""

    def test_validate_adapter_valid(self, adapter_config, mock_adapter_state):
        manager = AdapterManager(adapter_config)

        # Should not raise
        manager._validate_adapter(mock_adapter_state, "profile1")

    def test_validate_adapter_empty(self, adapter_config):
        manager = AdapterManager(adapter_config)

        with pytest.raises(ValueError, match="Empty adapter state dict"):
            manager._validate_adapter({}, "profile1")

    def test_validate_adapter_missing_lora_structure(self, adapter_config):
        manager = AdapterManager(adapter_config)
        invalid_state = {"some_weight": torch.randn(10, 10)}

        # Should log warning but not raise
        manager._validate_adapter(invalid_state, "profile1")

    def test_validate_adapter_disabled(self, adapter_config):
        adapter_config.auto_validate = False
        manager = AdapterManager(adapter_config)

        # Non-empty state without LoRA structure should not raise when validation disabled
        # Note: empty state always raises regardless of auto_validate
        manager._validate_adapter({"some_weight": torch.randn(10, 10)}, "profile1")


class TestAdapterInfo:
    """Test adapter info retrieval."""

    def test_get_adapter_info_not_found(self, adapter_config):
        manager = AdapterManager(adapter_config)
        info = manager.get_adapter_info("nonexistent")
        assert info is None

    def test_get_adapter_info_success(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        info = manager.get_adapter_info("profile1")

        assert info is not None
        assert isinstance(info, AdapterInfo)
        assert info.profile_id == "profile1"
        assert info.profile_name == "Test Voice"
        assert info.rank == 8
        assert info.alpha == 16
        assert info.sample_count == 10
        assert info.training_epochs == 50
        assert info.loss_final == 0.023

    def test_get_adapter_info_caching(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        info1 = manager.get_adapter_info("profile1")
        info2 = manager.get_adapter_info("profile1")

        # Should return same cached object
        assert info1 is info2


class TestAdapterApplication:
    """Test applying adapters to models."""

    @pytest.fixture
    def mock_model(self, mock_adapter_state):
        """Create a mock model with LoRA structure matching mock_adapter_state."""
        class MockLoRALayer(nn.Module):
            def __init__(self, lora_a_shape, lora_b_shape):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(*lora_a_shape))
                self.lora_B = nn.Parameter(torch.randn(*lora_b_shape))

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Match shapes from mock_adapter_state
                self.lora_adapters = nn.ModuleDict({
                    "content_proj": MockLoRALayer((8, 256), (256, 8)),
                    "output": MockLoRALayer((8, 512), (512, 8)),
                })

        return MockModel()

    def test_apply_adapter_success(self, adapter_config, mock_adapter_state, mock_model):
        manager = AdapterManager(adapter_config)

        # Apply adapter
        manager.apply_adapter(mock_model, mock_adapter_state)

        # Verify parameters were updated
        applied_param = mock_model.lora_adapters.content_proj.lora_A
        expected_param = mock_adapter_state["lora_adapters.content_proj.lora_A"]

        assert torch.allclose(applied_param, expected_param)

    def test_apply_adapter_no_matching_params(self, adapter_config, mock_adapter_state):
        manager = AdapterManager(adapter_config)
        empty_model = nn.Module()

        # Should log warning but not raise
        manager.apply_adapter(empty_model, mock_adapter_state)

    def test_remove_adapter(self, adapter_config, mock_model):
        manager = AdapterManager(adapter_config)

        # Store original values
        original_b = mock_model.lora_adapters.content_proj.lora_B.data.clone()

        # Remove adapter (zeros out lora_B)
        manager.remove_adapter(mock_model)

        # Verify lora_B is zeroed
        assert torch.allclose(
            mock_model.lora_adapters.content_proj.lora_B.data,
            torch.zeros_like(original_b)
        )


class TestAdapterSaving:
    """Test saving adapters from models."""

    @pytest.fixture
    def trained_model(self):
        """Create a model with trained LoRA adapters."""
        class MockLoRALayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(8, 256))
                self.lora_B = nn.Parameter(torch.randn(256, 8))

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_adapters = nn.ModuleDict({
                    "content_proj": MockLoRALayer(),
                    "output": MockLoRALayer(),
                })

        return MockModel()

    def test_save_adapter_success(self, adapter_config, trained_model, temp_dirs):
        manager = AdapterManager(adapter_config)

        path = manager.save_adapter("profile1", trained_model)

        assert path.exists()
        assert path.name == "profile1_adapter.pt"

        # Verify saved state can be loaded
        saved_state = torch.load(path, map_location="cpu", weights_only=False)
        assert "lora_adapters.content_proj.lora_A" in saved_state
        assert "lora_adapters.content_proj.lora_B" in saved_state

    def test_save_adapter_no_lora_params(self, adapter_config):
        manager = AdapterManager(adapter_config)
        empty_model = nn.Module()

        with pytest.raises(ValueError, match="No adapter parameters found"):
            manager.save_adapter("profile1", empty_model)

    def test_save_adapter_with_metadata(
        self, adapter_config, trained_model, temp_dirs, mock_profile_metadata
    ):
        # Create existing profile
        profile_path = temp_dirs["profiles_dir"] / "profile1.json"
        with open(profile_path, "w") as f:
            json.dump({"name": "Original"}, f)

        manager = AdapterManager(adapter_config)

        # Save with metadata update
        manager.save_adapter("profile1", trained_model, metadata={"name": "Updated"})

        # Verify metadata was updated
        with open(profile_path) as f:
            data = json.load(f)
        assert data["name"] == "Updated"


class TestCacheManagement:
    """Test cache management functionality."""

    def test_clear_cache(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        # Load to populate cache
        manager.load_adapter("profile1")
        manager.get_adapter_info("profile1")

        assert len(manager._cache) > 0
        assert len(manager._adapter_info) > 0

        # Clear cache
        manager.clear_cache()

        assert len(manager._cache) == 0
        assert len(manager._adapter_info) == 0

    def test_get_cache_stats(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")
        create_test_adapter("profile2")

        manager = AdapterManager(adapter_config)

        manager.load_adapter("profile1")
        manager.load_adapter("profile2")
        manager.get_adapter_info("profile1")

        stats = manager.get_cache_stats()

        assert stats["cached_adapters"] == 2
        assert stats["max_cache_size"] == 3
        assert stats["cached_info"] == 1


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_load_adapter_for_profile(self, adapter_config, create_test_adapter, monkeypatch):
        create_test_adapter("profile1")

        # Mock AdapterManager to use our test config
        def mock_init(self, config=None):
            self.config = adapter_config
            self.device = torch.device("cpu")
            self._cache = AdapterCache(max_size=adapter_config.cache_size)
            self._adapter_info = {}
            self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(AdapterManager, "__init__", mock_init)

        state_dict = load_adapter_for_profile("profile1", device="cpu")

        assert state_dict is not None
        assert "lora_adapters.content_proj.lora_A" in state_dict

    def test_get_trained_profiles(self, adapter_config, create_test_adapter, monkeypatch):
        create_test_adapter("profile1")
        create_test_adapter("profile2")

        # Mock AdapterManager to use our test config
        def mock_init(self, config=None):
            self.config = adapter_config
            self.device = torch.device("cpu")
            self._cache = AdapterCache(max_size=adapter_config.cache_size)
            self._adapter_info = {}
            self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(AdapterManager, "__init__", mock_init)

        profiles = get_trained_profiles()

        assert len(profiles) == 2
        profile_ids = [p[0] for p in profiles]
        assert "profile1" in profile_ids
        assert "profile2" in profile_ids

    def test_get_adapter_manager_singleton(self):
        # Import and reset global
        import auto_voice.models.adapter_manager as am
        am._global_manager = None

        manager1 = get_adapter_manager()
        manager2 = get_adapter_manager()

        # Should return same instance
        assert manager1 is manager2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_load_corrupted_adapter(self, adapter_config, temp_dirs):
        # Create corrupted adapter file
        adapter_path = temp_dirs["adapters_dir"] / "corrupt_adapter.pt"
        with open(adapter_path, "w") as f:
            f.write("not a valid pytorch file")

        manager = AdapterManager(adapter_config)

        with pytest.raises(Exception):  # torch.load will raise
            manager.load_adapter("corrupt")

    def test_profile_metadata_missing_fields(self, adapter_config, temp_dirs, mock_adapter_state):
        # Save adapter
        adapter_path = temp_dirs["adapters_dir"] / "profile1_adapter.pt"
        torch.save(mock_adapter_state, adapter_path)

        # Save incomplete profile metadata
        profile_path = temp_dirs["profiles_dir"] / "profile1.json"
        with open(profile_path, "w") as f:
            json.dump({"name": "Test"}, f)  # Missing most fields

        manager = AdapterManager(adapter_config)
        info = manager.get_adapter_info("profile1")

        # Should use defaults for missing fields
        assert info is not None
        assert info.profile_name == "Test"
        assert info.rank == 8  # Default
        assert info.sample_count == 0  # Default

    def test_device_placement(self, temp_dirs, create_test_adapter):
        create_test_adapter("profile1")

        config = AdapterManagerConfig(
            adapters_dir=temp_dirs["adapters_dir"],
            profiles_dir=temp_dirs["profiles_dir"],
            device="cpu",
        )

        manager = AdapterManager(config)
        state_dict = manager.load_adapter("profile1")

        # Verify all tensors are on correct device
        for tensor in state_dict.values():
            assert tensor.device == torch.device("cpu")


class TestTensorShapes:
    """Test that adapter tensors have correct shapes."""

    def test_adapter_shapes_valid(self, adapter_config, mock_adapter_state):
        manager = AdapterManager(adapter_config)

        # Validate shapes
        lora_a = mock_adapter_state["lora_adapters.content_proj.lora_A"]
        lora_b = mock_adapter_state["lora_adapters.content_proj.lora_B"]

        # LoRA A should be (rank, in_features)
        assert lora_a.shape[0] == 8  # rank

        # LoRA B should be (out_features, rank)
        assert lora_b.shape[1] == 8  # rank

        # A and B should be compatible
        assert lora_a.shape[0] == lora_b.shape[1]

    def test_adapter_weights_not_nan(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        state_dict = manager.load_adapter("profile1")

        # Verify no NaN values
        for name, tensor in state_dict.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {name}"

    def test_adapter_weights_finite(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        state_dict = manager.load_adapter("profile1")

        # Verify all values are finite
        for name, tensor in state_dict.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in {name}"

============================================================
# tests/__init__.py [SUMMARIZED]
============================================================

============================================================
# tests/conftest.py [SUMMARIZED]
============================================================
"""Shared test fixtures for AutoVoice."""

import os
import sys
import tempfile
import shutil
import numpy
import pytest

def sample_audio():
    """Generate a simple sine wave audio sample...."""
    ...

def sample_audio_file(sample_audio, tmp_path):
    """Create a temporary audio file...."""
    ...

def short_audio():
    """Very short audio (1 second) for edge case testing...."""
    ...

def short_audio_file(short_audio, tmp_path):
    """Create a short audio file (below minimum duration for cloning)...."""
    ...

def profiles_dir(tmp_path):
    """Temporary directory for voice profiles...."""
    ...

def flask_app():
    """Create a test Flask app with ML components disabled...."""
    ...

def flask_app_full():
    """Create a test Flask app with ML components enabled...."""
    ...

def client(flask_app):
    """Flask test client without ML components...."""
    ...

def client_full(flask_app_full):
    """Flask test client with ML components...."""
    ...

def voice_cloner(profiles_dir):
    """VoiceCloner instance with temp profile storage...."""
    ...

def singing_pipeline(voice_cloner):
    """SingingConversionPipeline with ModelManager pre-loaded (random weights)...."""
    ...

def audio_processor():
    """AudioProcessor instance...."""
    ...

def profile_store(profiles_dir):
    """VoiceProfileStore instance...."""
    ...
# ... (truncated)
============================================================
# fixtures/multi_speaker_fixtures.py [SUMMARIZED]
============================================================
"""Multi-speaker audio fixtures for E2E testing of speaker diarization.

This module provides utilities to create realistic multi-speaker test audio
by concatenating existing single-speaker samples or generating synthetic audio."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy
from scipy.io import wavfile

class SpeakerInfo:
    """Information about a speaker segment in test audio."""
    def duration(self) -> float:
        ...

class MultiSpeakerFixture:
    """A multi-speaker audio fixture with ground truth annotations."""
    def __post_init__(self):
        ...
    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerInfo]:
        """Get all segments for a specific speaker...."""
        ...
    def get_speaker_total_duration(self, speaker_id: str) -> float:
        """Get total duration for a speaker...."""
        ...
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization...."""
        ...

def create_synthetic_multi_speaker(output_path: str, durations: List[Tuple[str, float]] = None, sample_rate: int = 16000) -> MultiSpeakerFixture:
    """Create synthetic multi-speaker audio using different frequency tones...."""
    ...

def create_multi_speaker_audio(speaker_files: List[Tuple[str, str, float, float]], output_path: str, sample_rate: int = 16000, crossfade_ms: int = 50) -> MultiSpeakerFixture:
    """Create multi-speaker audio by concatenating segments from real audio files...."""
    ...

def get_quality_samples_dir() -> Path:
    """Get the path to quality samples directory...."""
    ...

def create_duet_fixture(output_dir: str = None) -> Optional[MultiSpeakerFixture]:
    """Create a duet fixture using Conor Maynard and William Singe samples...."""
    ...

def create_interview_fixture(output_dir: str = None) -> Optional[MultiSpeakerFixture]:
    """Create an interview-style fixture with longer speaker turns...."""
    ...
============================================================
# fixtures/__init__.py [SUMMARIZED]
============================================================
"""Test fixtures for AutoVoice E2E testing."""

from multi_speaker_fixtures import create_multi_speaker_audio, create_synthetic_multi_speaker, MultiSpeakerFixture, SpeakerInfo
============================================================
# auto_voice/__init__.py [SUMMARIZED]
============================================================
"""AutoVoice - GPU-accelerated singing voice conversion and TTS system."""

============================================================
# youtube/__init__.py [SUMMARIZED]
============================================================
"""YouTube integration for voice training data collection."""

from channel_scraper import VideoMetadata, YouTubeChannelScraper, scrape_artist_channel
from downloader import DownloadResult, YouTubeDownloader, download_artist_videos, download_artist_videos_async
============================================================
# youtube/downloader.py [SUMMARIZED]
============================================================
"""YouTube audio downloader with parallel processing.

Downloads audio from YouTube videos with rate limiting and progress tracking."""

import asyncio
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
from channel_scraper import VideoMetadata

class DownloadResult:
    """Result of a download operation."""

class YouTubeDownloader:
    """Downloads audio from YouTube videos with parallel processing."""
    def __init__(self, output_dir: Path, max_workers: int = 4, rate_limit: float = 1.0):
        """Initialize downloader...."""
        ...
    def _wait_rate_limit(self):
        """Wait if needed to respect rate limit...."""
        ...
    def download_audio(self, video_id: str, title: str = '') -> DownloadResult:
        """Download audio from a single video...."""
        ...
    def download_batch(self, videos: List[VideoMetadata], progress_callback: Optional[Callable[[int, int, DownloadResult], None]] = None) -> List[DownloadResult]:
        """Download audio from multiple videos in parallel...."""
        ...

def download_artist_videos(artist_key: str, output_subdir: str = None, max_videos: int = 500, max_workers: int = 4) -> List[DownloadResult]:
    """Convenience function to download all music videos for an artist...."""
    ...

async def download_artist_videos_async(artist_key: str, output_subdir: str = None, max_videos: int = 500, max_workers: int = 4) -> List[DownloadResult]:
    """Async wrapper for download_artist_videos...."""
    ...
============================================================
# youtube/channel_scraper.py [SUMMARIZED]
============================================================
"""YouTube channel scraper using yt-dlp.

Discovers and filters videos from artist channels for training data collection."""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

class VideoMetadata:
    """Metadata for a YouTube video."""
    def is_music(self) -> bool:
        """Heuristic check if video is likely music content...."""
        ...
    def is_solo_artist(self) -> bool:
        """Check if the primary artist is solo (not a collaboration)...."""
        ...
    def is_valid_for_training(self) -> bool:
        """Check if video is suitable for voice training...."""
        ...

class YouTubeChannelScraper:
    """Scrapes video metadata from YouTube channels using yt-dlp."""
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize scraper...."""
        ...
    def get_channel_videos(self, channel_url: str, max_videos: int = 1000, music_only: bool = True, solo_only: bool = True) -> List[VideoMetadata]:
        """Get all video metadata from a channel...."""
        ...
    def save_metadata(self, videos: List[VideoMetadata], artist_name: str) -> Path:
        """Save video metadata to JSON file...."""
        ...
    def load_metadata(self, artist_name: str) -> List[VideoMetadata]:
        """Load cached video metadata...."""
        ...

def scrape_artist_channel(artist_key: str, max_videos: int = 1000, solo_only: bool = True) -> List[VideoMetadata]:
    """Convenience function to scrape known artist channel...."""
    ...
============================================================
# storage/voice_profiles.py [SUMMARIZED]
============================================================
"""Voice profile storage - file-based CRUD operations."""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy
import torch
DEFAULT_PROFILES_DIR = 'data/voice_profiles'
DEFAULT_SAMPLES_DIR = 'data/samples'

class ProfileNotFoundError(Exception):
    """Raised when a voice profile is not found."""

class TrainingSample:
    """Represents a training sample for progressive voice model improvement."""
    def __init__(self, sample_id: str, vocals_path: str, instrumental_path: Optional[str] = None, source_file: Optional[str] = None, duration: float = 0.0, created_at: Optional[str] = None):
        ...
    def to_dict(self) -> Dict[str, Any]:
        ...
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        ...

class VoiceProfileStore:
    """File-based voice profile storage with progressive training sample support."""
    def __init__(self, profiles_dir: str = DEFAULT_PROFILES_DIR, samples_dir: str = DEFAULT_SAMPLES_DIR):
        ...
    def _profile_path(self, profile_id: str) -> str:
        ...
    def _embedding_path(self, profile_id: str) -> str:
        ...
    def save(self, profile_data: Dict[str, Any]) -> str:
        """Save a voice profile. Returns profile_id...."""
        ...
    def load(self, profile_id: str) -> Dict[str, Any]:
        """Load a voice profile by ID. Raises ProfileNotFoundError if not found...."""
        ...
    def list_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all profiles, optionally filtered by user_id...."""
        ...
    def delete(self, profile_id: str) -> bool:
        """Delete a profile. Returns True if deleted, False if not found...."""
        ...
    def exists(self, profile_id: str) -> bool:
        """Check if a profile exists...."""
        ...
    def _lora_weights_path(self, profile_id: str) -> str:
        """Get path to LoRA weights file for a profile...."""
        ...
    def save_lora_weights(self, profile_id: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Save LoRA adapter weights for a voice profile...."""
        ...
    def load_lora_weights(self, profile_id: str) -> Dict[str, torch.Tensor]:
        """Load LoRA adapter weights for a voice profile...."""
        ...
    def has_trained_model(self, profile_id: str) -> bool:
        """Check if a profile has trained LoRA weights...."""
# ... (truncated)
============================================================
# storage/__init__.py [SUMMARIZED]
============================================================
"""Data persistence layer."""

from voice_profiles import ProfileNotFoundError
============================================================
# models/adapter_manager.py [SUMMARIZED]
============================================================
"""Unified Adapter Manager for Voice Conversion Pipelines.

Provides a single interface for loading, caching, and applying LoRA adapters
across both REALTIME and QUALITY pipelines.

Features:
- Profile-based adapter loading
- LRU caching for frequently used adapters
- Validation of adapter compatibility
- Integration with both pipeline types"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import json
import torch
import torch.nn

class AdapterInfo:
    """Information about a loaded adapter."""

class AdapterManagerConfig:
    """Configuration for AdapterManager."""

class AdapterCache:
    """LRU cache for loaded adapters."""
    def __init__(self, max_size: int = 5):
        ...
    def get(self, profile_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get adapter from cache, moving to end (most recently used)...."""
        ...
    def put(self, profile_id: str, adapter_state: Dict[str, torch.Tensor]) -> None:
        """Add adapter to cache, evicting oldest if necessary...."""
        ...
    def clear(self) -> None:
        """Clear the cache...."""
        ...
    def __len__(self) -> int:
        ...
    def __contains__(self, profile_id: str) -> bool:
        ...

class AdapterManager:
    """Unified manager for voice adapter loading and application.

Provides a single interface for both REALTIME and QUALITY pipelines
to load and apply LoRA adapters trained on specific voice profiles.

Usa..."""
    def __init__(self, config: Optional[AdapterManagerConfig] = None):
        ...
    def list_available_adapters(self) -> List[str]:
        """List all available adapter profile IDs...."""
        ...
    def has_adapter(self, profile_id: str) -> bool:
        """Check if an adapter exists for the given profile...."""
        ...
# ... (truncated)
============================================================
# models/vocoder.py [SUMMARIZED]
============================================================
"""Vocoder models for waveform synthesis.

Includes HiFiGAN and BigVGAN (arxiv:2206.04658) generators.
BigVGAN uses Snake periodic activations and anti-aliased multi-periodicity
composition for superior singing voice synthesis."""

import logging
import math
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn
import torch.nn.functional
HIFIGAN_CONFIG = {'resblock_kernel_sizes': [3, 7, 11], 'resblock_di...

class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class HiFiGANGenerator(nn.Module):
    """HiFiGAN generator for mel-to-waveform synthesis."""
    def __init__(self, num_mels: int = 80, upsample_rates: Optional[List[int]] = None, upsample_kernel_sizes: Optional[List[int]] = None, upsample_initial_channel: int = 512, resblock_kernel_sizes: Optional[List[int]] = None, resblock_dilation_sizes: Optional[List[List[int]]] = None):
        ...
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel-spectrogram...."""
        ...
    def remove_weight_norm(self):
        """Remove weight normalization for inference...."""
        ...

class HiFiGANVocoder:
    """High-level vocoder interface wrapping HiFiGAN generator."""
    def __init__(self, device = None, config: Optional[dict] = None):
        ...
    def _ensure_loaded(self):
        """Ensure generator is initialized...."""
        ...
    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained vocoder weights...."""
        ...
    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel-spectrogram...."""
        ...
    def mel_to_audio(self, mel, sr: int = 22050):
        """Convert mel spectrogram to numpy audio array...."""
        ...
    def load_pretrained(cls, checkpoint_path: str, device = None) -> 'HiFiGANVocoder':
        """Load a pretrained HiFiGAN vocoder...."""
        ...
BIGVGAN_24KHZ_100BAND_CONFIG = {'num_mels': 100, 'upsample_rates': [4, 4, 2, 2, 2...

class SnakeBeta(nn.Module):
    """Snake activation with separate beta parameter (BigVGAN v2).

f(x) = x + (1/beta) * sin^2(alpha * x)

Alpha and beta are trainable per-channel parameters stored in log-scale."""
# ... (truncated)
============================================================
# models/hq_adapter_bridge.py [SUMMARIZED]
============================================================
"""Adapter Bridge for HQ LoRA Adapters.

Bridges the architecture gap between:
- Trained HQ adapters: Standalone 6-layer MLP with keys 'lora_0_A', 'lora_0_B', etc.
- Expected format: Layer-injection format '{module}.adapter.lora_A'

The HQVoiceLoRAAdapter is a content feature transformer that takes ContentVec
features and applies speaker-specific adaptation before the decoder.

Usage:
    bridge = HQLoRAAdapterBridge(device='cuda')
    bridge.load_adapter('profile-uuid')

    # In conversion pipeline:
    content_features = content_encoder.encode(audio)  # [B, T, 768]
    adapted_features = bridge.transform(content_features, speaker_embedding)
    mel = decoder.infer(adapted_features, pitch, speaker)"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn
import torch.nn.functional
import numpy

class LoRALayer(nn.Module):
    """High-Quality LoRA layer with scaled initialization."""
    def __init__(self, in_features: int, out_features: int, rank: int = 128, alpha: float = 256.0, dropout: float = 0.05):
        ...
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        ...
    def get_delta_weight(self) -> torch.Tensor:
        ...

class HQVoiceLoRAAdapter(nn.Module):
    """High-Quality Voice LoRA Adapter.

Architecture: 768 -> 1024 -> 1024 -> 1024 -> 1024 -> 1024 -> 768
With residual connections and layer normalization.

This is a content feature transformer that applie..."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, output_dim: int = 768, lora_rank: int = 128, lora_alpha: float = 256.0, dropout: float = 0.05, num_layers: int = 6):
        ...
    def forward(self, content: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform content features with speaker-specific adaptation...."""
        ...
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA weights in the training format...."""
        ...
    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        """Load LoRA weights from training format...."""
        ...
DEFAULT_HQ_CONFIG = {'input_dim': 768, 'hidden_dim': 1024, 'output_dim...

class AdapterBridgeConfig:
    """Configuration for adapter bridge."""

# ... (truncated)
============================================================
# models/__init__.py [SUMMARIZED]
============================================================
"""Neural network model architectures."""

from encoder import ContentEncoder, PitchEncoder, HuBERTSoft
from vocoder import HiFiGANVocoder, HiFiGANGenerator
from so_vits_svc import SoVitsSvc
from consistency import DiffusionDecoder, ConsistencyStudent, CTLoss_D, EDMLoss, KarrasNoiseSchedule, ResidualBlock, DiffusionStepEmbedding
from svc_decoder import CoMoSVCDecoder, BiDilConv, FiLMConditioning
from smoothsinger_decoder import SmoothSingerDecoder, MultiResolutionBlock, DualBranchFusion
============================================================
# models/consistency.py [SUMMARIZED]
============================================================
"""Consistency distillation for fast 1-step inference.

Implements CoMoSVC-style two-stage training:
  Stage 1: Diffusion teacher (BiDilConv decoder with EDM preconditioning)
  Stage 2: Consistency student distilled from teacher for 1-step inference

Reference: Lu et al., "CoMoSVC" (arXiv:2401.01792)
Reference: Karras et al., "EDM" (arXiv:2206.00364)"""

import copy
import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn
import torch.nn.functional

class ResidualBlock(nn.Module):
    """Bidirectional dilated convolution residual block with gated activation.

Non-causal (sees past + future context), suitable for offline/batch SVC.
Conditioned on diffusion noise level via learned proje..."""
    def __init__(self, hidden_dim: int, n_mels: int, dilation: int, kernel_size: int = 3):
        ...
    def forward(self, x: torch.Tensor, diffusion_step: torch.Tensor, conditioner: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass...."""
        ...

class DiffusionStepEmbedding(nn.Module):
    """Sinusoidal embedding for noise level (sigma), projected to hidden_dim."""
    def __init__(self, hidden_dim: int, max_positions: int = 10000):
        ...
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Embed noise level...."""
        ...

class DiffusionDecoder(nn.Module):
    """BiDilConv diffusion decoder with EDM preconditioning.

Architecture: 20 residual blocks across 2 dilation cycles
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512] x 2 = 20 blocks.

Uses EDM (Karras) preconditio..."""
    def __init__(self, n_mels: int = 80, hidden_dim: int = 256, n_blocks: int = 20, kernel_size: int = 3, dilation_cycle: int = 10, sigma_data: float = 0.5, cond_dim: int = 256):
        """Initialize DiffusionDecoder...."""
        ...
    def _raw_forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Raw network F_theta (before EDM preconditioning)...."""
        ...
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """EDM-preconditioned forward pass...."""
        ...

class EDMLoss(nn.Module):
    """EDM training loss with log-normal noise sampling.

From Karras et al. (2022): samples sigma from log-normal distribution
and applies lambda(sigma) weighting for balanced training."""
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        ...
    def forward(self, model: DiffusionDecoder, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
# ... (truncated)
============================================================
# models/conformer.py [SUMMARIZED]
============================================================
"""Conformer encoder for content feature refinement.

Replaces the linear projection in ContentEncoder with a multi-head
self-attention + Conv1D feed-forward network that captures long-range
dependencies in content features. Based on the Amphion SVC Conformer.

Architecture per layer:
    x → LayerNorm → MultiHeadAttention(relative pos) → Dropout → Residual
    x → LayerNorm → Conv1D FFN (GELU) → Dropout → Residual"""

import math
from typing import Optional
import torch
import torch.nn
import torch.nn.functional

class ConformerLayerNorm(nn.Module):
    """Channel-wise layer normalization for [B, C, T] tensors."""
    def __init__(self, channels: int, eps: float = 1e-05):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative position encoding.

Operates on [B, C, T] tensors. Uses a local window for relative
position bias, enabling efficient attention on long sequences."""
    def __init__(self, channels: int, n_heads: int = 2, window_size: int = 4, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention on [B, C, T] tensor...."""
        ...
    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        ...
    def _get_relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        ...
    def _relative_to_absolute(self, x: torch.Tensor) -> torch.Tensor:
        """Convert relative position scores to absolute...."""
        ...
    def _absolute_to_relative(self, x: torch.Tensor) -> torch.Tensor:
        """Convert absolute position attention to relative...."""
        ...

class ConformerFFN(nn.Module):
    """Feed-forward network with Conv1D and GELU activation."""
    def __init__(self, channels: int, filter_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN on [B, C, T] tensor...."""
        ...

class ConformerLayer(nn.Module):
    """Single Conformer layer: attention + FFN with pre-norm residuals."""
    def __init__(self, channels: int, filter_channels: int, n_heads: int = 2, kernel_size: int = 3, window_size: int = 4, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: pre-norm residual attention + FFN...."""
        ...

# ... (truncated)
============================================================
# models/pitch.py [SUMMARIZED]
============================================================
"""RMVPE pitch extractor for singing voice conversion.

Implements a simplified RMVPE (Robust Model for Vocal Pitch Estimation)
architecture based on the Interspeech 2023 paper. Uses a deep residual
CNN operating on mel spectrograms to produce cent-based pitch estimates
with voicing probabilities.

Key design choices:
- 20ms hop size (320 samples at 16kHz) matching ContentVec frame rate
- 360 bins per octave (10-cent resolution)
- 6 octaves coverage (C1=32.7Hz to C7=2093Hz, clipped to f0_min/f0_max)
- Weighted average of cent bins for sub-cent precision
- No fallback: raises RuntimeError on failure"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy
import torch
import torch.nn
import torch.nn.functional

class ResBlock(nn.Module):
    """Residual block for RMVPE feature extraction."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RMVPEBackbone(nn.Module):
    """Deep residual CNN backbone for RMVPE.

Processes mel spectrograms through a series of residual blocks
with progressive channel expansion, producing frame-level features
for pitch classification."""
    def __init__(self, n_mels: int = 128, n_blocks: int = 6, base_channels: int = 64):
        ...
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Process mel spectrogram...."""
        ...

class RMVPEPitchExtractor(nn.Module):
    """RMVPE-based pitch extractor for singing voice.

Extracts F0 contour directly from audio waveform using a deep
residual network operating on mel spectrograms. Outputs F0 in Hz
with voiced/unvoiced deci..."""
    def __init__(self, pretrained: Optional[str] = None, device: Optional[torch.device] = None, hop_size: int = 320, f0_min: float = 50.0, f0_max: float = 1100.0, n_mels: int = 128, sample_rate: int = 16000):
        ...
    def _load_pretrained(self, path: str):
        """Load pretrained RMVPE weights...."""
        ...
    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from audio...."""
        ...
    def _create_mel_filterbank(self, device: torch.device) -> torch.Tensor:
        """Create mel filterbank matrix...."""
        ...
    def _decode_pitch(self, logits: torch.Tensor, voicing: torch.Tensor) -> torch.Tensor:
        """Decode pitch bin logits to F0 in Hz...."""
# ... (truncated)

================================================================================
