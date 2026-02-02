
# Agent Assignment
================================================================================
Swarm: sota-dual-pipeline
Agent: smoothsinger-researcher
Type: researcher
Phase: 1
Track: conductor/tracks/sota-dual-pipeline_20260130
GPU Required: False
Dependencies: None

## Responsibility
Research SmoothSinger frequency innovations

## Expected Outputs
- docs/smoothsinger-concepts.md

## Workflow Rules
1. Follow TDD: Write tests FIRST, then implement
2. Report progress: Update beads tasks (`bd update <id> --status in_progress`)
3. Share discoveries: Write to cipher memory for cross-agent learning
4. No fallback behavior: Raise errors, never pass silently
5. Atomic commits: One feature per commit, run tests before committing

================================================================================

# Injected Context
# Agent Context Injection
# Files: 38 (30 summarized)
# Tokens: ~24,169 / 50,000 budget
# Priority breakdown: 8 critical, 0 important, 30 reference

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
# CLAUDE.md
============================================================
<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

*No recent activity*
</claude-mem-context>
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
# inference/seed_vc_pipeline.py [SUMMARIZED]
============================================================
"""Seed-VC Quality Pipeline using DiT-CFM decoder.

This pipeline provides state-of-the-art voice conversion quality using:
  - Whisper-base for semantic content extraction
  - CAMPPlus for speaker style embedding
  - DiT (Diffusion Transformer) with Conditional Flow Matching for 5-10 step inference
  - BigVGAN v2 (44kHz, 128-band) for high-quality waveform synthesis
  - RMVPE for F0 extraction (singing voice)

Key advantages over CoMoSVC:
  - In-context learning: Uses reference audio as prompt for fine-grained timbre
  - Fewer steps: 5-10 steps vs 30+ for diffusion
  - Higher sample rate: 44.1kHz output vs 24kHz

Research paper: Seed-VC (arXiv:2411.09943) - 40 citations as of 2026"""

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Union, TYPE_CHECKING
import numpy
import torch
import torch.nn.functional
SEED_VC_DIR = Path(__file__).parent.parent.parent.parent / 'mode...

class SeedVCPipeline:
    """SOTA quality pipeline using Seed-VC's DiT-CFM decoder.

This pipeline wraps the Seed-VC implementation to provide a consistent
interface with our other pipelines while leveraging the DiT architecture
..."""
    def __init__(self, device: Optional[torch.device] = None, diffusion_steps: int = 10, f0_condition: bool = True, require_gpu: bool = True):
        """Initialize Seed-VC pipeline...."""
        ...
    def _initialize(self):
        """Lazy-initialize the Seed-VC wrapper...."""
        ...
    def sample_rate(self) -> int:
        """Output sample rate (44.1kHz for F0-conditioned singing)...."""
        ...
    def set_reference_audio(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int = 44100) -> None:
        """Set reference audio for in-context learning...."""
        ...
    def set_reference_from_profile(self, profile_store: 'VoiceProfileStore', profile_id: str) -> None:
        """Load reference audio from a voice profile (legacy method)...."""
        ...
    def set_reference_from_profile_id(self, profile_id: str, reference_index: int = 0) -> None:
        """Load reference audio using the AdapterBridge...."""
        ...
    def convert(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int, speaker_embedding: Optional[torch.Tensor] = None, on_progress: Optional[Callable[[str, float], None]] = None, pitch_shift: int = 0) -> Dict[str, Any]:
        """Convert audio using Seed-VC DiT-CFM...."""
        ...
    def convert_with_separation(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int, speaker_embedding: Optional[torch.Tensor] = None, on_progress: Optional[Callable[[str, float], None]] = None, pitch_shift: int = 0) -> Dict[str, Any]:
        """Convert audio with vocal separation pre-processing...."""
        ...
============================================================
# inference/streaming_pipeline.py [SUMMARIZED]
============================================================
"""Real-time streaming voice conversion pipeline.

Implements chunked inference with overlap-add synthesis for continuous
audio streaming with minimal latency.

Architecture:
- Chunk-based processing with configurable chunk/hop sizes
- Overlap-add synthesis with crossfade windows for glitch-free output
- Latency tracking and optimization
- Audio I/O stream handling for microphone/speaker integration

Target: < 50ms end-to-end latency on Jetson Thor"""

import time
from typing import Optional, Callable, Dict, Any, List
import torch
import torch.nn.functional
import numpy
from sota_pipeline import SOTAConversionPipeline

class StreamingConversionPipeline:
    """Real-time streaming voice conversion with overlap-add synthesis.

Processes audio in chunks with configurable overlap for continuous
output without glitches. Tracks latency to ensure real-time perform..."""
    def __init__(self, chunk_size_ms: int = 100, overlap_ratio: float = 0.5, sample_rate: int = 24000, device: Optional[torch.device] = None):
        ...
    def _create_crossfade_window(self) -> torch.Tensor:
        """Create Hann crossfade window for overlap-add synthesis...."""
        ...
    def process_chunk(self, audio_chunk: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """Process a single audio chunk and return converted output...."""
        ...
    def _apply_overlap_add(self, converted: torch.Tensor) -> torch.Tensor:
        """Apply overlap-add synthesis for continuous output...."""
        ...
    def reset(self) -> None:
        """Reset overlap buffer and latency history...."""
        ...
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from recent chunks...."""
        ...
    def start_session(self, speaker_embedding: torch.Tensor) -> None:
        """Start a streaming conversion session...."""
        ...
    def stop_session(self) -> None:
        """Stop the streaming conversion session...."""
        ...

class AudioInputStream:
    """Audio input stream capture from microphone or audio interface.

Provides buffered audio capture with callback-based chunk delivery.

Args:
    sample_rate: Audio sample rate (default: 24000)
    buffe..."""
    def __init__(self, sample_rate: int = 24000, buffer_size: int = 1024, device_index: Optional[int] = None):
        ...
    def set_callback(self, callback: Callable[[torch.Tensor], None]) -> None:
# ... (truncated)
============================================================
# inference/trt_pipeline.py [SUMMARIZED]
============================================================
"""TensorRT-optimized SOTA singing voice conversion pipeline.

Provides ONNX export and TensorRT engine building for all pipeline
components, plus optimized inference using TRT engines.

Target: Jetson Thor (SM 11.0, 16GB GPU memory, CUDA 13.0)

Components exported:
- ContentVec encoder (768-dim features)
- RMVPE pitch extractor (F0 + voicing)
- CoMoSVC decoder (mel spectrogram generation)
- BigVGAN vocoder (mel -> waveform)

The separator (MelBandRoFormer) runs in PyTorch as it has complex
STFT operations that don't export cleanly to ONNX."""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Tuple, List
import torch
import torch.nn.functional
import numpy

class ONNXExporter:
    """Export PyTorch models to ONNX format with dynamic shapes."""
    def __init__(self, opset_version: int = 17):
        """Initialize ONNX exporter...."""
        ...
    def export_content_extractor(self, model: torch.nn.Module, output_path: str) -> str:
        """Export ContentVec encoder to ONNX...."""
        ...
    def export_pitch_extractor(self, model: torch.nn.Module, output_path: str) -> str:
        """Export RMVPE pitch extractor to ONNX...."""
        ...
    def export_decoder(self, model: torch.nn.Module, output_path: str) -> str:
        """Export CoMoSVC decoder to ONNX...."""
        ...
    def export_vocoder(self, model: torch.nn.Module, output_path: str) -> str:
        """Export BigVGAN vocoder to ONNX...."""
        ...

class TRTEngineBuilder:
    """Build TensorRT engines from ONNX models."""
    def __init__(self, precision: str = 'fp16', workspace_size: int = ...):
        """Initialize TRT engine builder...."""
        ...
    def supports_dynamic_shapes(self, shapes: Dict[str, List[Tuple]]) -> bool:
        """Check if builder configuration supports the given dynamic shapes...."""
        ...
    def build_engine(self, onnx_path: str, engine_path: str, dynamic_shapes: Optional[Dict] = None) -> str:
        """Build TRT engine from ONNX file...."""
        ...

class TRTInferenceContext:
    """Context manager for TRT engine inference."""
    def __init__(self, engine_path: str):
        """Load TRT engine for inference...."""
        ...
# ... (truncated)
============================================================
# inference/voice_identifier.py [SUMMARIZED]
============================================================
"""Voice Identification Pipeline.

Identifies vocalist from audio by matching against existing voice profile embeddings.

Cross-Context Dependencies:
- speaker-diarization_20260130: WavLM embeddings (256-dim)
- voice-profile-training_20260124: Profile management
- training-inference-integration_20260130: AdapterManager

Features:
- Load all profile embeddings
- Compute cosine similarity
- Return best match above threshold (0.85 default)
- Auto-route separated vocals to correct profiles"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import torch
import torch.nn.functional

class IdentificationResult:
    """Result of voice identification."""

class VoiceIdentifier:
    """Identifies voices by matching embeddings against known profiles.

Thresholds:
- speaker_similarity_min: 0.85 (from lora-lifecycle-management spec)"""
    def __init__(self, profiles_dir: Path = ..., device: str = 'cuda', embedding_model: Optional[str] = None):
        ...
    def load_all_embeddings(self) -> int:
        """Load all profile embeddings from disk...."""
        ...
    def _load_wavlm(self) -> None:
        """Lazy load WavLM model for embedding extraction...."""
        ...
    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker embedding from audio...."""
        ...
    def identify(self, audio: np.ndarray, sample_rate: int = 16000, threshold: Optional[float] = None) -> IdentificationResult:
        """Identify the speaker in the audio...."""
        ...
    def identify_from_file(self, audio_path: str, threshold: Optional[float] = None) -> IdentificationResult:
        """Identify speaker from audio file...."""
        ...
    def match_segments_to_profiles(self, segments: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Match diarization segments to known profiles...."""
        ...
    def get_loaded_profiles(self) -> List[Tuple[str, str]]:
        """Get list of loaded profiles...."""
        ...
    def create_profile_from_segment(self, audio: np.ndarray, sample_rate: int = 16000, youtube_metadata: Optional[Dict[str, Any]] = None, source_file: Optional[str] = None) -> str:
        """Create a new voice profile from an unknown speaker segment...."""
        ...
    def _generate_profile_name(self, youtube_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate profile name from YouTube metadata or default pattern...."""
# ... (truncated)
============================================================
# inference/realtime_pipeline.py [SUMMARIZED]
============================================================
"""Realtime voice conversion pipeline for live karaoke.

Architecture: Audio -> ContentVec (16kHz) -> RMVPE -> SimpleDecoder -> HiFiGAN (22kHz)

Target latency breakdown:
- ContentVec: ~40ms (content feature extraction)
- RMVPE: ~20ms (pitch extraction)
- SimpleDecoder: ~10ms (mel generation)
- HiFiGAN: ~20ms (waveform synthesis)
- Total: <100ms for live performance

This pipeline is optimized for low-latency streaming inference, using
lightweight components that maintain quality while meeting realtime constraints."""

import logging
import time
from collections import deque
from typing import Dict, Optional, Any
import numpy
import torch
import torch.nn
import torch.nn.functional

class SimpleDecoder(nn.Module):
    """Lightweight decoder for realtime voice conversion.

Converts content features (from ContentVec) and pitch features (from RMVPE)
into mel spectrograms, conditioned on speaker embedding.

Architecture o..."""
    def __init__(self, content_dim: int = 768, pitch_dim: int = 256, speaker_dim: int = 256, n_mels: int = 80, hidden_dim: int = 256):
        ...
    def forward(self, content: torch.Tensor, pitch: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Generate mel spectrogram from content, pitch, and speaker...."""
        ...

class RealtimePipeline:
    """Realtime voice conversion pipeline for live karaoke.

Orchestrates:
1. ContentVec encoder - extracts speaker-independent content (16kHz input)
2. RMVPE pitch extractor - extracts F0 contour
3. SimpleD..."""
    def __init__(self, device: Optional[torch.device] = None, contentvec_model: Optional[str] = None, vocoder_checkpoint: Optional[str] = None):
        ...
    def _init_content_encoder(self, model_id: Optional[str]):
        """Initialize ContentVec encoder with error handling...."""
        ...
    def _init_pitch_extractor(self):
        """Initialize RMVPE pitch extractor with error handling...."""
        ...
    def _init_decoder(self):
        """Initialize SimpleDecoder with error handling...."""
        ...
    def _init_vocoder(self, checkpoint: Optional[str]):
        """Initialize HiFiGAN vocoder with error handling...."""
        ...
    def set_speaker_embedding(self, embedding: np.ndarray) -> None:
        """Set target speaker embedding for voice conversion with validation...."""
        ...
# ... (truncated)
============================================================
# inference/hq_svc_wrapper.py [SUMMARIZED]
============================================================
"""HQ-SVC Wrapper for cutting-edge voice conversion and super-resolution.

HQ-SVC (AAAI 2026) provides:
  - Zero-shot singing voice conversion
  - Super-resolution: 16kHz → 44.1kHz upsampling

Architecture:
  FACodec (content extraction @ 16kHz) → DDSP (initial synthesis)
  → Diffusion (mel refinement) → NSF-HiFiGAN (vocoder @ 44.1kHz)

No fallback behavior: raises RuntimeError on failure."""

import logging
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Union
import numpy
import torch
import torch.nn.functional
HQ_SVC_ROOT = os.path.join(os.path.dirname(__file__), '..', '..'...
HQ_SVC_ROOT = os.path.abspath(HQ_SVC_ROOT)
MIN_DURATION_SAMPLES_16K = 8000

class HQSVCWrapper:
    """Wrapper for HQ-SVC cutting-edge voice conversion.

Provides two modes:
  1. Super-resolution: Upsample 16kHz audio to 44.1kHz
  2. Voice conversion: Convert source voice to target speaker

All operati..."""
    def __init__(self, device: Optional[torch.device] = None, config_path: Optional[str] = None, require_gpu: bool = True):
        """Initialize HQ-SVC wrapper with all components...."""
        ...
    def _setup_paths(self):
        """Add HQ-SVC paths to sys.path for imports...."""
        ...
    def _load_config(self, config_path: str):
        """Load HQ-SVC configuration from YAML file...."""
        ...
    def _init_models(self):
        """Initialize all HQ-SVC model components...."""
        ...
    def _resample(self, audio: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        """Resample audio between sample rates using linear interpolation...."""
        ...
    def _to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert to mono by averaging channels...."""
        ...
    def _wav_pad(self, wav: np.ndarray, multiple: int = 200) -> np.ndarray:
        """Pad waveform to multiple of given value...."""
        ...
    def _process_audio(self, audio: torch.Tensor, sample_rate: int) -> Dict:
        """Process audio to extract all features for HQ-SVC...."""
        ...
    def extract_speaker_embedding(self, audio: Union[torch.Tensor, List[torch.Tensor]], sample_rate: int) -> torch.Tensor:
        """Extract speaker embedding from reference audio...."""
        ...
    def super_resolve(self, audio: torch.Tensor, sample_rate: int, on_progress: Optional[Callable[[str, float], None]] = None) -> Dict:
# ... (truncated)
============================================================
# inference/trt_streaming_pipeline.py [SUMMARIZED]
============================================================
"""TensorRT-optimized real-time streaming voice conversion pipeline.

Combines TRT inference engines with overlap-add synthesis for
ultra-low-latency (<50ms) live voice conversion.

Target: Jetson Thor (SM 11.0, 16GB GPU memory, CUDA 13.0)"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn.functional
import numpy

class TRTStreamingPipeline:
    """TensorRT-optimized streaming voice conversion pipeline.

Uses preloaded TRT engines for minimal latency during live conversion.
Includes overlap-add synthesis for glitch-free continuous output.

Args:..."""
    def __init__(self, engine_dir: str, chunk_size_ms: int = 100, overlap_ratio: float = 0.5, sample_rate: int = 24000, device: Optional[torch.device] = None):
        ...
    def engines_available(engine_dir: str) -> bool:
        """Check if TRT engines exist in the specified directory...."""
        ...
    def _create_crossfade_window(self) -> torch.Tensor:
        """Create Hann crossfade window for overlap-add synthesis...."""
        ...
    def load_engines(self):
        """Preload TRT engines for fast inference...."""
        ...
    def _resample(self, audio: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        """Resample audio tensor...."""
        ...
    def _encode_pitch(self, f0: torch.Tensor) -> torch.Tensor:
        """Encode F0 to pitch embeddings using sinusoidal encoding...."""
        ...
    def process_chunk(self, audio_chunk: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """Process a single audio chunk using TRT engines...."""
        ...
    def _apply_overlap_add(self, converted: torch.Tensor) -> torch.Tensor:
        """Apply overlap-add synthesis for continuous output...."""
        ...
    def reset(self):
        """Reset overlap buffer and latency history...."""
        ...
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from recent chunks...."""
        ...
    def get_engine_memory_usage(self) -> int:
        """Get total memory usage of all TRT engines in bytes...."""
        ...
============================================================
# inference/singing_conversion_pipeline.py [SUMMARIZED]
============================================================
"""Singing voice conversion pipeline.

Orchestrates: audio separation -> content encoding -> voice conversion -> vocoder -> mixing."""

import logging
import os
import tempfile
import time
import uuid
from typing import Optional, Dict, Any
import numpy

class SeparationError(Exception):
    """Raised when vocal/instrumental separation fails."""

class ConversionError(Exception):
    """Raised when voice conversion fails."""
PRESETS = {'draft': {'n_steps': 10, 'denoise': 0.3}, 'fast':...

class SingingConversionPipeline:
    """Main voice conversion pipeline for singing audio."""
    def __init__(self, device = None, config: Optional[Dict] = None, voice_cloner = None):
        ...
    def _get_separator(self):
        """Lazy-load vocal separator...."""
        ...
    def _separate_vocals(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate vocals from instrumental...."""
        ...
    def _get_model_manager(self):
        """Get or create ModelManager. Raises if not configured...."""
        ...
    def _convert_voice(self, vocals: np.ndarray, target_embedding: np.ndarray, sr: int, preset: str = 'balanced') -> np.ndarray:
        """Convert vocals to target voice using trained SoVitsSvc model...."""
        ...
    def _extract_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch contour from audio...."""
        ...
    def _detect_techniques(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
        """Detect vocal techniques (vibrato, melisma) in audio...."""
        ...
    def convert_song(self, song_path: str, target_profile_id: str, vocal_volume: float = 1.0, instrumental_volume: float = 0.9, pitch_shift: float = 0.0, return_stems: bool = False, preset: str = 'balanced', preserve_techniques: bool = True) -> Dict[str, Any]:
        """Convert a song to target voice...."""
        ...
============================================================
# inference/__init__.py [SUMMARIZED]
============================================================
"""Voice conversion inference pipeline."""

============================================================
# inference/sota_pipeline.py [SUMMARIZED]
============================================================
"""SOTA Singing Voice Conversion Pipeline.

End-to-end pipeline connecting all SOTA components:
  MelBandRoFormer → ContentVec → RMVPE → CoMoSVC → BigVGAN

Sample rate flow:
  Input (any SR) → 44.1kHz (separator) → 16kHz (content+pitch) → mel → 24kHz (vocoder)

Frame alignment:
  ContentVec: 50fps at 16kHz (hop=320)
  RMVPE: 50fps at 16kHz (hop=320)
  Both produce aligned frame sequences for the decoder.

No fallback behavior: raises RuntimeError on failure."""

import logging
import time
from typing import Callable, Dict, Optional, Any, TYPE_CHECKING
import torch
import torch.nn.functional
from audio.separator import MelBandRoFormer
from models.encoder import ContentVecEncoder
from models.pitch import RMVPEPitchExtractor
from models.svc_decoder import CoMoSVCDecoder
from models.vocoder import BigVGANVocoder
from models.adapter_manager import AdapterManager, AdapterManagerConfig
from models.hq_adapter_bridge import HQLoRAAdapterBridge, AdapterBridgeConfig
MIN_DURATION_SAMPLES_24K = 2400

class SOTAConversionPipeline:
    """SOTA singing voice conversion pipeline.

Orchestrates:
  1. Vocal separation (MelBandRoFormer @ 44.1kHz)
  2. Content extraction (ContentVec @ 16kHz → 768-dim)
  3. Pitch extraction (RMVPE @ 16kHz → F..."""
    def __init__(self, device: Optional[torch.device] = None, n_steps: int = 1, profile_store: Optional['VoiceProfileStore'] = None, profile_id: Optional[str] = None, require_gpu: bool = True):
        """Initialize pipeline with all SOTA components...."""
        ...
    def _load_profile_lora(self, profile_store: 'VoiceProfileStore', profile_id: str) -> None:
        """Load LoRA weights from profile if available...."""
        ...
    def set_speaker(self, profile_id: str) -> None:
        """Dynamically switch to a different speaker by loading their LoRA adapter...."""
        ...
    def get_current_speaker(self) -> Optional[str]:
        """Get the currently loaded speaker profile ID...."""
        ...
    def get_speaker_embedding(self) -> Optional[torch.Tensor]:
        """Get the currently loaded speaker embedding...."""
        ...
    def clear_speaker(self) -> None:
        """Clear the current speaker adapter, reverting to base model...."""
        ...
    def _resample(self, audio: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        """Resample audio tensor between sample rates...."""
        ...
    def _to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono by averaging channels...."""
        ...
# ... (truncated)
============================================================
# inference/realtime_voice_conversion_pipeline.py [SUMMARIZED]
============================================================
"""Real-time voice conversion pipeline.

Provides low-latency streaming voice conversion for live audio input.
Uses overlapping windows and crossfade for smooth output."""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Callable, Any, Union
import numpy
import torch
import torch.nn
from models.adapter_manager import AdapterManager, AdapterManagerConfig

class RealtimeVoiceConversionPipeline:
    """Streaming voice conversion with low-latency processing.

Uses a ring buffer approach with overlapping windows to provide
continuous voice conversion with minimal delay."""
    def __init__(self, device = None, config: Optional[Dict[str, Any]] = None):
        ...
    def is_running(self) -> bool:
        """Whether the pipeline is actively processing...."""
        ...
    def latency_ms(self) -> float:
        """Current average processing latency in milliseconds...."""
        ...
    def buffer_latency_ms(self) -> float:
        """Theoretical minimum latency from buffer size...."""
        ...
    def set_target_voice(self, embedding: np.ndarray):
        """Set the target voice embedding for conversion...."""
        ...
    def set_speaker(self, profile_id: str, profiles_dir: Union[str, Path] = ...) -> None:
        """Switch to a different speaker by profile ID...."""
        ...
    def get_current_speaker(self) -> Optional[str]:
        """Get the currently loaded speaker profile ID...."""
        ...
    def clear_speaker(self) -> None:
        """Clear the current speaker, stopping voice conversion...."""
        ...
    def start(self, on_output: Optional[Callable] = None, on_error: Optional[Callable] = None):
        """Start the realtime processing pipeline...."""
        ...
    def stop(self):
        """Stop the processing pipeline...."""
        ...
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process an audio chunk through the conversion pipeline...."""
        ...
    def _processing_loop(self):
        """Background processing loop for push mode...."""
        ...
    def _convert_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Convert a single audio chunk...."""
        ...
    def _get_model_manager(self):
# ... (truncated)
============================================================
# inference/voice_cloner.py [SUMMARIZED]
============================================================
"""Voice cloner - speaker embeddings and profile management."""

import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import numpy
from storage.voice_profiles import VoiceProfileStore, ProfileNotFoundError, TrainingSample

class InvalidAudioError(Exception):
    """Raised when audio input is invalid (corrupt, too short, wrong format)."""

class InsufficientQualityError(Exception):
    """Raised when audio quality is too low for voice cloning."""
    def __init__(self, message: str, error_code: str = ..., details: Optional[Dict] = None):
        ...

class InconsistentSamplesError(Exception):
    """Raised when multiple audio samples are inconsistent (different speakers)."""
    def __init__(self, message: str, error_code: str = ..., details: Optional[Dict] = None):
        ...

def _get_vocal_separator(device):
    """Get or create the vocal separator (lazy singleton)...."""
    ...

class VoiceCloner:
    """Creates and manages voice profiles using speaker embeddings."""
    def __init__(self, device = None, profiles_dir: str = ..., auto_separate_vocals: bool = True):
        ...
    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from mel-spectrogram statistics...."""
        ...
    def create_speaker_embedding(self, audio_paths: List[str]) -> np.ndarray:
        """Create averaged speaker embedding from multiple singing recordings...."""
        ...
    def _estimate_vocal_range(self, audio_path: str) -> Dict[str, float]:
        """Estimate vocal range from audio...."""
        ...
    def _extract_vocals(self, audio_path: str, profile_id: str) -> Optional[Dict[str, str]]:
        """Extract vocals from audio using Demucs separation...."""
        ...
    def create_voice_profile(self, audio: str, user_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a voice profile from audio file...."""
        ...
    def load_voice_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load a voice profile. Raises ProfileNotFoundError if not found...."""
        ...
    def list_voice_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List voice profiles, optionally filtered by user_id...."""
        ...
    def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete a voice profile. Returns True if deleted...."""
        ...
    def compare_embeddings(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings...."""
        ...
    def add_vocal_sample(self, profile_id: str, audio_path: str, source_name: Optional[str] = None) -> Optional[TrainingSample]:
# ... (truncated)
============================================================
# inference/meanvc_pipeline.py [SUMMARIZED]
============================================================
"""MeanVC Streaming Pipeline for Real-Time Voice Conversion.

This pipeline provides single-step voice conversion with streaming support using:
  - FastU2++ ASR model for content feature extraction (bottleneck features)
  - WavLM + ECAPA-TDNN for speaker embeddings
  - DiT with Mean Flows for single-step inference
  - Vocos vocoder for 16kHz output

Key advantages:
  - Single-step inference (1-2 NFE) via mean flows
  - Chunk-wise autoregressive processing with KV-cache
  - <100ms chunk latency for real-time streaming
  - 14M parameters (lightweight compared to full diffusion models)

Research paper: MeanVC (arXiv:2510.08392)
GitHub: https://github.com/ASLP-lab/MeanVC"""

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Union
import numpy
import torch
import torch.nn
import torch.nn.functional
import torchaudio.compliance.kaldi
from librosa.filters import mel
MEANVC_DIR = Path(__file__).parent.parent.parent.parent / 'mode...
MEANVC_SRC = MEANVC_DIR / 'src'

def _amp_to_db(x: torch.Tensor, min_level_db: float) -> torch.Tensor:
    """Convert amplitude to decibels...."""
    ...

def _normalize(S: torch.Tensor, max_abs_value: float, min_db: float) -> torch.Tensor:
    """Normalize spectrogram...."""
    ...

class MelSpectrogramFeatures(nn.Module):
    """Mel spectrogram extractor for MeanVC."""
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, win_size: int = 640, hop_length: int = 160, n_mels: int = 80, fmin: int = 0, fmax: int = 8000, center: bool = True):
        ...
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        ...

def extract_fbanks(wav: np.ndarray, sample_rate: int = 16000, mel_bins: int = 80, frame_length: float = 25, frame_shift: float = 10) -> torch.Tensor:
    """Extract filter bank features for ASR model...."""
    ...

class MeanVCPipeline:
    """Streaming voice conversion pipeline using MeanVC.

This pipeline provides real-time voice conversion with:
- Single-step inference via mean flows (1-2 NFE)
- Chunk-wise autoregressive processing with ..."""
    def __init__(self, device: Optional[torch.device] = None, steps: int = 2, require_gpu: bool = False):
        """Initialize MeanVC pipeline...."""
        ...
# ... (truncated)
============================================================
# inference/model_manager.py [SUMMARIZED]
============================================================
"""Model manager for voice conversion inference.

Orchestrates content encoding, pitch encoding, SoVitsSvc, and HiFiGAN vocoder
with frame alignment. No fallback behavior - raises RuntimeError if models
are not loaded."""

import logging
from typing import Optional, Dict
import numpy
import torch
import torch.nn.functional

class ModelManager:
    """Manages voice models and runs frame-aligned inference.

Raises RuntimeError if any required model is not loaded or if invalid
configuration values are provided. No fallback behavior.

Supported config..."""
    def __init__(self, device = None, config: Optional[Dict] = None):
        ...
    def _validate_config(self, config: Dict) -> None:
        """Validate configuration values. Raises RuntimeError for invalid values...."""
        ...
    def load(self, hubert_path: Optional[str] = None, vocoder_path: Optional[str] = None, vocoder_type: str = 'hifigan', encoder_backend: str = 'hubert', encoder_type: str = 'linear', conformer_config: Optional[Dict] = None):
        """Load shared models. Must be called before infer()...."""
        ...
    def load_voice_model(self, model_path: str, speaker_id: str, speaker_embedding: Optional[np.ndarray] = None):
        """Load a trained per-speaker SoVitsSvc model...."""
        ...
    def infer(self, audio: np.ndarray, speaker_id: str, speaker_embedding: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Convert audio to target speaker's voice. No fallbacks...."""
        ...
============================================================
# inference/adapter_bridge.py [SUMMARIZED]
============================================================
"""LoRA Adapter Bridge for SeedVC Pipeline.

Bridges trained LoRA adapters (from our MLP-based decoder) to work with
SeedVC's in-context learning approach. Since Seed-VC uses reference audio
rather than speaker embeddings, this bridge provides reference audio paths
from voice profiles.

For the original pipeline (realtime/quality), LoRAs directly modify the decoder.
For Seed-VC, we provide reference audio that captures the voice characteristics."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy
import torch

class VoiceReference:
    """Reference audio information for a voice profile."""

class AdapterBridge:
    """Bridge between trained LoRAs and Seed-VC pipeline.

This bridge serves two purposes:
1. For original pipelines (realtime/quality): Load and apply LoRA adapters
2. For Seed-VC pipeline: Provide referen..."""
    def __init__(self, profiles_dir: Union[str, Path] = ..., training_audio_dir: Union[str, Path] = ..., lora_dir: Union[str, Path] = ..., device: str = 'cuda'):
        """Initialize the adapter bridge...."""
        ...
    def _load_profile_mappings(self) -> None:
        """Load profile ID to artist name mappings...."""
        ...
    def _find_artist_audio_dir(self, profile_id: str) -> Optional[Path]:
        """Find the audio directory for a profile's artist...."""
        ...
    def _fuzzy_match(s1: str, s2: str, max_distance: int = 2) -> bool:
        """Simple fuzzy string matching using Levenshtein distance...."""
        ...
    def get_voice_reference(self, profile_id: str, max_references: int = 5, min_duration: float = 10.0) -> VoiceReference:
        """Get voice reference information for Seed-VC pipeline...."""
        ...
    def load_lora(self, profile_id: str, use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """Load LoRA weights for the original pipeline...."""
        ...
    def get_lora_metadata(self, profile_id: str) -> Dict:
        """Get metadata from a trained LoRA checkpoint...."""
        ...
    def list_available_profiles(self) -> List[Tuple[str, str, bool, bool]]:
        """List all available voice profiles with their capabilities...."""
        ...
    def clear_cache(self) -> None:
        """Clear all caches...."""
        ...

def get_adapter_bridge() -> AdapterBridge:
    """Get the global AdapterBridge instance...."""
    ...
============================================================
# inference/gpu_enforcement.py [SUMMARIZED]
============================================================
"""GPU enforcement utilities for inference operations.

Task 7.5: Add strict GPU-only checks (RuntimeError on any CPU fallback attempt)

Provides:
- GPU requirement verification for inference
- Tensor/model device verification
- Context manager for GPU-only inference blocks
- Strict mode to catch all CPU operations"""

import functools
import logging
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, TypeVar, Any
import torch
import torch.nn
F = TypeVar('F', bound=Callable)

def require_gpu_for_inference(operation_name: str = 'Inference') -> None:
    """Verify CUDA is available for inference, raise if not...."""
    ...

def get_inference_device(device_id: Optional[int] = None) -> torch.device:
    """Get the device for inference operations...."""
    ...

def enforce_inference_gpu(func: F) -> F:
    """Decorator to enforce GPU availability before inference function execution...."""
    ...

def verify_tensor_on_gpu(tensor: torch.Tensor, name: str = 'tensor') -> None:
    """Verify a tensor is on GPU, raise if not...."""
    ...

def verify_all_tensors_on_gpu(tensors: Dict[str, torch.Tensor]) -> None:
    """Verify all tensors in a dict are on GPU...."""
    ...

def verify_model_on_gpu(model: nn.Module, name: str = 'model') -> None:
    """Verify a model's parameters are on GPU...."""
    ...

class GPUInferenceContext:
    """Context manager that ensures GPU availability for inference blocks.

Usage:
    with GPUInferenceContext("voice conversion") as ctx:
        model.to(ctx.device)
        output = model(input)

Raises:..."""
    def __init__(self, operation_name: str = 'inference', device_id: Optional[int] = None):
        """Initialize GPU inference context...."""
        ...
    def device(self) -> torch.device:
        """Get the CUDA device for this context...."""
        ...
    def __enter__(self) -> 'GPUInferenceContext':
        """Enter context, verifying CUDA availability...."""
        ...
# ... (truncated)
============================================================
# inference/pipeline_factory.py [SUMMARIZED]
============================================================
"""Pipeline factory for unified voice conversion pipeline management.

Provides lazy loading, caching, and memory management for:
- RealtimePipeline: Low-latency karaoke (22kHz, simple decoder)
- SOTAConversionPipeline: High-quality CoMoSVC (24kHz, 30-step diffusion)
- SeedVCPipeline: SOTA quality with DiT-CFM (44kHz, 5-10 step flow matching)

Usage:
    factory = PipelineFactory.get_instance()
    pipeline = factory.get_pipeline('realtime')  # or 'quality' or 'quality_seedvc'
    result = pipeline.convert(audio, sr, speaker_embedding)"""

import logging
from typing import Dict, Literal, Optional, TYPE_CHECKING, Union
import torch

class PipelineFactory:
    """Factory for creating and managing voice conversion pipelines.

Features:
- Lazy loading: Pipelines only initialized when first requested
- Caching: Re-uses existing pipeline instances
- Memory managem..."""
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize factory...."""
        ...
    def get_instance(cls, device: Optional[torch.device] = None) -> 'PipelineFactory':
        """Get singleton instance of the factory...."""
        ...
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)...."""
        ...
    def get_pipeline(self, pipeline_type: PipelineType, profile_store: Optional['VoiceProfileStore'] = None) -> Union['RealtimePipeline', 'SOTAConversionPipeline', 'SeedVCPipeline', 'MeanVCPipeline']:
        """Get or create a pipeline instance...."""
        ...
    def _create_pipeline(self, pipeline_type: PipelineType, profile_store: Optional['VoiceProfileStore'] = None) -> Union['RealtimePipeline', 'SOTAConversionPipeline', 'SeedVCPipeline', 'MeanVCPipeline']:
        """Create a new pipeline instance...."""
        ...
    def is_loaded(self, pipeline_type: PipelineType) -> bool:
        """Check if a pipeline is currently loaded...."""
        ...
    def get_memory_usage(self, pipeline_type: PipelineType) -> float:
        """Get GPU memory usage for a pipeline in GB...."""
        ...
    def get_total_memory_usage(self) -> float:
        """Get total GPU memory usage across all pipelines in GB...."""
        ...
    def unload_pipeline(self, pipeline_type: PipelineType) -> bool:
        """Unload a pipeline to free GPU memory...."""
        ...
    def unload_all(self) -> None:
        """Unload all pipelines...."""
        ...
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all pipelines...."""
        ...
============================================================
# inference/trt_rebuilder.py [SUMMARIZED]
============================================================
"""TensorRT engine rebuilding for fine-tuned models.

Task 7.4: Implement TensorRT engine rebuilding for fine-tuned models

Provides:
- Model checksum computation for version tracking
- Automatic rebuild detection after fine-tuning
- Engine caching with invalidation
- State persistence across sessions"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn

class TRTEngineManager:
    """Manages TensorRT engines with automatic rebuilding for fine-tuned models.

Tracks model checksums to detect when fine-tuning has changed parameters,
triggering automatic ONNX export and TRT engine reb..."""
    def __init__(self, cache_dir: str, precision: str = 'fp16'):
        """Initialize TRT engine manager...."""
        ...
    def compute_model_checksum(self, model: nn.Module) -> str:
        """Compute SHA-256 checksum of model parameters...."""
        ...
    def get_engine_path(self, model_name: str, model: nn.Module) -> Path:
        """Get engine path for a model, including checksum in filename...."""
        ...
    def needs_rebuild(self, model_name: str, model: nn.Module) -> bool:
        """Check if a model's TRT engine needs to be rebuilt...."""
        ...
    def register_model(self, model_name: str, model: nn.Module) -> None:
        """Register a model for engine management...."""
        ...
    def _mark_engine_built(self, model_name: str, model: nn.Module) -> None:
        """Mark an engine as successfully built for a model...."""
        ...
    def _store_engine_metadata(self, model_name: str, metadata: Dict[str, Any]) -> None:
        """Store engine metadata for a model...."""
        ...
    def _get_engine_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get stored engine metadata for a model...."""
        ...
    def cleanup_old_engines(self, keep_count: int = 3) -> List[str]:
        """Remove old engine files, keeping only the most recent...."""
        ...
    def save_state(self) -> None:
        """Save manager state to disk for persistence across sessions...."""
        ...
    def load_state(self) -> bool:
        """Load manager state from disk...."""
        ...
    def rebuild_engine(self, model_name: str, model: nn.Module, export_fn: callable, dynamic_shapes: Optional[Dict] = None) -> str:
        """Rebuild TRT engine for a model...."""
        ...
============================================================
# inference/mean_flow_decoder.py [SUMMARIZED]
============================================================
"""Mean Flow Decoder for single-step voice conversion inference.

This module implements the mean flow regression approach from MeanVC, which
enables single-step inference by directly regressing the average velocity field
instead of iteratively solving an ODE.

Key innovation: Instead of computing v(x_t, t) at each timestep and integrating,
we directly predict the mean flow: x1 = x0 + mean_flow(x0, conditions).

Research paper: MeanVC (arXiv:2510.08392)

Architecture:
    ContentVec features → DiT with Mean Flow → Mel spectrogram
    Single step: x1 = x0 + ∫₀¹ v(x_t, t) dt ≈ x0 + mean_v(x0)

This is MUCH faster than iterative CFM (10 steps) or diffusion (30+ steps)."""

import logging
from typing import Optional, Tuple
import torch
import torch.nn

class MeanFlowDecoder(nn.Module):
    """Mean flow decoder for single-step voice conversion.

Instead of iteratively solving the ODE:
    dx/dt = v(x_t, t)
    x1 = x0 + ∫₀¹ v(x_t, t) dt

We directly regress the mean flow:
    mean_v(x0) ≈ ∫..."""
    def __init__(self, content_dim: int = 512, speaker_dim: int = 256, mel_dim: int = 80, hidden_dim: int = 512, num_layers: int = 6, num_heads: int = 8):
        """Initialize mean flow decoder...."""
        ...
    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor, content: torch.Tensor, speaker: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict mean flow from x to x1...."""
        ...
    def inference_single_step(self, x0: torch.Tensor, content: torch.Tensor, speaker: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single-step inference via mean flow...."""
        ...
    def inference_two_step(self, x0: torch.Tensor, content: torch.Tensor, speaker: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Two-step inference for slightly better quality...."""
        ...

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same as DiT/CFM models)."""
    def __init__(self, dim: int, max_period: int = 10000):
        ...
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:..."""
        ...

def compute_mean_flow_loss(model: MeanFlowDecoder, x0: torch.Tensor, x1: torch.Tensor, content: torch.Tensor, speaker: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dual loss for mean flow training...."""
    ...
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
============================================================
# models/encoder.py [SUMMARIZED]
============================================================
"""Content and pitch encoders for voice conversion.

Uses HuBERT or ContentVec for content extraction (speaker-independent
linguistic features) and a pitch encoder for F0 contour processing."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy
import torch
import torch.nn
import torch.nn.functional

class ContentEncoder(nn.Module):
    """Content encoder supporting HuBERT-soft and ContentVec backends.

Extracts speaker-independent content features from audio,
preserving linguistic content while removing speaker identity.

Supports two ..."""
    def __init__(self, hidden_size: int = 256, output_size: int = 256, hubert_model: str = 'hubert-soft', device = None, encoder_type: str = 'linear', conformer_config: dict = None, encoder_backend: str = 'hubert', contentvec_model: str = ..., contentvec_layer: int = 12):
        ...
    def _load_hubert(self, checkpoint_path: Optional[str] = None):
        """Load HuBERT model for feature extraction...."""
        ...
    def extract_features(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Extract content features from audio...."""
        ...
    def forward(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Forward pass - extract content features...."""
        ...
    def load_pretrained(cls, checkpoint_path: str, device = None) -> 'ContentEncoder':
        """Load a pretrained content encoder...."""
        ...

def f0_to_coarse(f0: torch.Tensor, n_bins: int = 256, f0_min: float = 50.0, f0_max: float = 1100.0) -> torch.Tensor:
    """Convert F0 in Hz to mel-scale quantized bin indices...."""
    ...

class PitchEncoder(nn.Module):
    """Mel-quantized F0 encoder with voiced/unvoiced embedding.

Replaces LSTM-based encoding with mel-scale quantized F0 lookup (256 bins)
plus a learned voiced/unvoiced embedding. A small residual linear p..."""
    def __init__(self, input_size: int = 1, hidden_size: int = 128, output_size: int = 256, n_bins: int = 256):
        ...
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """Encode pitch contour using mel-quantized embeddings...."""
        ...
    def load_pretrained(cls, checkpoint_path: str, device = None) -> 'PitchEncoder':
        """Load pretrained pitch encoder weights...."""
        ...

class HuBERTSoft(nn.Module):
    """Simplified soft-HuBERT model for content extraction."""
    def __init__(self, checkpoint_path: Optional[str] = None):
        ...
    def _load_checkpoint(self, path: str):
        """Load hubert-soft checkpoint...."""
        ...
# ... (truncated)
============================================================
# models/so_vits_svc.py [SUMMARIZED]
============================================================
"""So-VITS-SVC (Singing Voice Conversion) model.

Combines content encoder, pitch encoder, speaker encoder, and decoder
for high-quality singing voice conversion."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn
import torch.nn.functional

def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Compute differentiable SSIM loss between predicted and target mel spectrograms...."""
    ...

class PosteriorEncoder(nn.Module):
    """Posterior encoder using WaveNet-style dilated convolutions."""
    def __init__(self, in_channels: int = 513, hidden_channels: int = 192, out_channels: int = 192, kernel_size: int = 5, n_layers: int = 16):
        ...
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to latent with reparameterization...."""
        ...

class WaveNetBlock(nn.Module):
    """WaveNet-style dilated convolution block."""
    def __init__(self, channels: int, kernel_size: int, n_layers: int):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class FlowDecoder(nn.Module):
    """Normalizing flow for voice conversion."""
    def __init__(self, channels: int = 192, hidden_channels: int = 192, kernel_size: int = 5, n_layers: int = 4, n_flows: int = 4):
        ...
    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        ...

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, n_layers: int):
        ...
    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        ...

class Flip(nn.Module):
    """Channel flip for flow diversity."""
    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        ...

class SoVitsSvc(nn.Module):
    """So-VITS-SVC model for singing voice conversion."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        ...
    def forward(self, content: torch.Tensor, pitch: torch.Tensor, speaker: torch.Tensor, spec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass...."""
        ...
    def infer(self, content: torch.Tensor, pitch: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Inference - generate mel from content+pitch+speaker...."""
        ...
# ... (truncated)
============================================================
# models/smoothsinger_decoder.py [SUMMARIZED]
============================================================
"""SmoothSinger-inspired decoder for CUTTING_EDGE_PIPELINE.

Implements January 2026 SOTA innovations:
- Multi-resolution non-sequential U-Net processing (SmoothSinger Jun 2025)
- Reference-guided dual-branch architecture for style transfer
- Vocoder-free codec diffusion output (HQ-SVC AAAI 2026 concepts)
- CFM-based diffusion with classifier-free guidance

Key design:
- Content features from Whisper (768-dim)
- Style features from CAMPPlus (192-dim)
- F0 conditioning from RMVPE (256 bins)
- Output: mel spectrogram (128 bands) or codec tokens
- LoRA support for per-voice fine-tuning

Reference: SmoothSinger, HQ-SVC, Seed-VC"""

import logging
from typing import Dict, Optional, Literal
import torch
import torch.nn
import torch.nn.functional

class MultiResolutionBlock(nn.Module):
    """Multi-resolution non-sequential processing block.

Processes input at multiple temporal resolutions in parallel,
then fuses results. Unlike sequential U-Net, all resolutions
are computed simultaneousl..."""
    def __init__(self, channels: int = 256, n_resolutions: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through parallel multi-resolution branches...."""
        ...

class DualBranchFusion(nn.Module):
    """Reference-guided dual-branch fusion module.

Implements dual-branch processing where:
- Content branch: processes source linguistic features
- Style branch: generates modulation parameters from refere..."""
    def __init__(self, content_dim: int = 512, style_dim: int = 192, expansion: int = 2):
        ...
    def forward(self, content: torch.Tensor, style: torch.Tensor, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse content features with style conditioning...."""
        ...

class SmoothSingerDecoder(nn.Module):
    """SmoothSinger-inspired decoder for CUTTING_EDGE_PIPELINE.

Generates mel spectrograms or codec tokens from content, F0, and style
features using multi-resolution processing and dual-branch fusion.

Arc..."""
    def __init__(self, content_dim: int = 768, style_dim: int = 192, f0_bins: int = 256, hidden_dim: int = 512, n_mels: int = 128, n_resolutions: int = 3, n_blocks: int = 6, output_mode: Literal['mel', 'codec'] = 'mel', n_codebooks: int = 8, codebook_size: int = 1024, enable_super_resolution: bool = False, device: Optional[torch.device] = None):
        ...
    def _diffusion_forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, style: torch.Tensor, reference_mel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Diffusion forward pass: predict x_0 from x_t...."""
        ...
    def forward(self, content: torch.Tensor, f0: torch.Tensor, style: torch.Tensor, reference_mel: Optional[torch.Tensor] = None) -> torch.Tensor:
# ... (truncated)
============================================================
# models/svc_decoder.py [SUMMARIZED]
============================================================
"""CoMoSVC consistency model decoder for singing voice conversion.

Implements a consistency model decoder based on CoMoSVC (ISCSLP 2024).
Uses Bidirectional Dilated Convolutions (BiDilConv) conditioned on
content features (768-dim ContentVec), pitch embeddings (256-dim),
and speaker embeddings (256-dim mel-statistics).

Key design:
- Consistency distillation enables 1-step inference (matches 50-step diffusion)
- BiDilConv captures long-range temporal patterns in mel spectrograms
- Speaker conditioning via FiLM (Feature-wise Linear Modulation)
- Multi-step inference available for higher quality when latency allows
- LoRA injection for per-voice fine-tuning
- No fallback: raises RuntimeError on failure"""

import logging
from typing import Dict, Optional
import torch
import torch.nn
import torch.nn.functional

class BiDilConv(nn.Module):
    """Bidirectional Dilated Convolution block.

Uses exponentially increasing dilation rates for large receptive field
while maintaining temporal resolution. Each layer doubles the dilation.

Architecture: ..."""
    def __init__(self, channels: int = 256, kernel_size: int = 3, n_layers: int = 4, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through dilated conv layers with residual connections...."""
        ...

class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation for speaker conditioning.

Produces scale (gamma) and shift (beta) from speaker embedding
to modulate intermediate features."""
    def __init__(self, speaker_dim: int, feature_dim: int):
        ...
    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning...."""
        ...

class CoMoSVCDecoder(nn.Module):
    """CoMoSVC consistency model decoder.

Generates mel spectrograms from content, pitch, and speaker features
using a consistency model approach. Supports 1-step inference for
real-time use and multi-step ..."""
    def __init__(self, content_dim: int = 768, pitch_dim: int = 256, speaker_dim: int = 256, n_mels: int = 100, hidden_dim: int = 512, n_layers: int = 8, device: Optional[torch.device] = None):
        ...
    def inject_lora(self, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        """Inject LoRA adapters into Linear layers for fine-tuning...."""
        ...
    def remove_lora(self) -> None:
        """Remove LoRA adapters and restore original Linear layers...."""
        ...
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
# ... (truncated)

================================================================================
