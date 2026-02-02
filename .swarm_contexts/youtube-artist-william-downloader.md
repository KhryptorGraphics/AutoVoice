
# Agent Assignment
================================================================================
Swarm: youtube-artist
Agent: william-downloader
Type: worker
Phase: 1
Track: conductor/tracks/youtube-artist-training_20260130
GPU Required: False
Dependencies: None

## Responsibility
Download William Singe audio

## Expected Outputs
- data/youtube/william_singe/

## Workflow Rules
1. Follow TDD: Write tests FIRST, then implement
2. Report progress: Update beads tasks (`bd update <id> --status in_progress`)
3. Share discoveries: Write to cipher memory for cross-agent learning
4. No fallback behavior: Raise errors, never pass silently
5. Atomic commits: One feature per commit, run tests before committing

================================================================================

# Injected Context
# Agent Context Injection
# Files: 53 (41 summarized)
# Tokens: ~28,923 / 50,000 budget
# Priority breakdown: 6 critical, 6 important, 41 reference

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
# Spec: YouTube Artist Training Pipeline

**Track ID:** youtube-artist-training_20260130
**Created:** 2026-01-30
**Priority:** P1
**Status:** [ ] Not Started

## Overview

Automated pipeline to download all music videos from Connor Maynard and William Singe YouTube channels, extract vocals using speaker diarization, add segments to their voice profiles, and train LoRA models with OOM protection.

## Target Artists

1. **Connor Maynard**
   - YouTube Channel: Conor Maynard
   - Genre: Pop covers, original songs
   - Expected videos: 200+

2. **William Singe**
   - YouTube Channel: William Singe
   - Genre: R&B/Pop covers
   - Expected videos: 300+

## Pipeline Stages

### Stage 1: YouTube Discovery & Download
- Use yt-dlp to list all videos from each channel
- Download audio only (best quality AAC/MP3)
- Store metadata (title, duration, upload date)
- Skip non-music content (vlogs, shorts)

### Stage 2: Audio Separation
- Run Demucs HTDemucs to separate vocals from instrumentals
- Save isolated vocal tracks
- Discard instrumentals (or save for karaoke)

### Stage 3: Speaker Diarization
- Run WavLM-based diarization on each vocal track
- Identify featured artists vs main artist
- Extract segments by speaker
- Use chunked processing for memory safety

### Stage 4: Profile Matching
- Match speaker embeddings to existing Connor/William profiles
- Create profiles if they don't exist
- Add matched segments as training samples
- Store embedding metadata

### Stage 5: LoRA Training
- Train separate LoRA adapters for each artist
- Max settings: rank=16, alpha=32, epochs=50
- OOM protection: gradient checkpointing, mixed precision
- Memory monitoring: abort if >90% GPU memory

## Acceptance Criteria

1. [ ] All music videos downloaded from both channels
2. [ ] Vocals separated with <10% bleed
3. [ ] Diarization identifies correct speaker >95% accuracy
4. [ ] Connor profile has >4 hours of clean vocals
5. [ ] William profile has >4 hours of clean vocals
6. [ ] LoRA training completes without OOM
7. [ ] Voice conversion produces recognizable output

## Technical Requirements

- yt-dlp for YouTube download
- Demucs for separation
- WavLM for diarization
- Memory-safe chunked processing
- Parallel download with rate limiting
- Progress tracking via beads

## OOM Prevention

- Max 4GB GPU memory per diarization chunk
- Gradient checkpointing for training
- Mixed precision (fp16/bf16)
- Batch size auto-reduction on OOM
- Memory monitoring hooks

============================================================
# scripts/validate_openapi.py
============================================================
#!/usr/bin/env python3
"""Validate OpenAPI specification and test Swagger UI."""
import sys
import json
import yaml
import requests
from pathlib import Path


def validate_openapi_spec(spec_url: str):
    """Validate OpenAPI spec structure."""
    print(f"Fetching OpenAPI spec from {spec_url}...")

    try:
        response = requests.get(spec_url, timeout=10)
        response.raise_for_status()
        spec = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch spec: {e}")
        return False

    # Validate required fields
    required_fields = ['openapi', 'info', 'paths']
    for field in required_fields:
        if field not in spec:
            print(f"❌ Missing required field: {field}")
            return False

    print(f"✅ OpenAPI version: {spec['openapi']}")
    print(f"✅ Title: {spec['info']['title']}")
    print(f"✅ Version: {spec['info']['version']}")

    # Count endpoints
    endpoint_count = len(spec.get('paths', {}))
    print(f"✅ Total endpoints documented: {endpoint_count}")

    # Validate schemas
    schemas = spec.get('components', {}).get('schemas', {})
    print(f"✅ Total schemas defined: {len(schemas)}")

    # List endpoint groups
    tags = set()
    for path, methods in spec.get('paths', {}).items():
        for method, details in methods.items():
            if isinstance(details, dict):
                endpoint_tags = details.get('tags', [])
                tags.update(endpoint_tags)

    print(f"\n✅ Endpoint groups:")
    for tag in sorted(tags):
        print(f"   - {tag}")

    return True


def test_swagger_ui(base_url: str):
    """Test Swagger UI accessibility."""
    swagger_url = f"{base_url}/docs"

    print(f"\nTesting Swagger UI at {swagger_url}...")

    try:
        response = requests.get(swagger_url, timeout=10)
        response.raise_for_status()

        if 'swagger-ui' in response.text.lower():
            print(f"✅ Swagger UI accessible at {swagger_url}")
            return True
        else:
            print(f"❌ Swagger UI page found but content unexpected")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to access Swagger UI: {e}")
        return False


def validate_yaml_spec(yaml_url: str):
    """Validate YAML format of spec."""
    print(f"\nValidating YAML spec at {yaml_url}...")

    try:
        response = requests.get(yaml_url, timeout=10)
        response.raise_for_status()

        # Try to parse YAML
        spec = yaml.safe_load(response.text)

        if 'openapi' in spec:
            print(f"✅ YAML spec valid and parseable")
            return True
        else:
            print(f"❌ YAML spec missing 'openapi' field")
            return False

    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch YAML spec: {e}")
        return False


def check_endpoint_coverage():
    """Check if major endpoint groups are documented."""
    required_groups = [
        'Conversion',
        'Voice Profiles',
        'Training',
        'Audio Processing',
        'System',
        'YouTube'
    ]

    print("\n✅ Expected endpoint groups:")
    for group in required_groups:
        print(f"   - {group}")

    return True


def main():
    """Run validation suite."""
    base_url = "http://localhost:5000"

    print("=" * 60)
    print("AutoVoice OpenAPI Validation")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ Server running at {base_url}")
    except requests.exceptions.RequestException:
        print(f"❌ Server not running at {base_url}")
        print("   Please start the server with: python main.py")
        return 1

    # Run validation tests
    tests = [
        ("OpenAPI JSON Spec", lambda: validate_openapi_spec(f"{base_url}/api/v1/openapi.json")),
        ("OpenAPI YAML Spec", lambda: validate_yaml_spec(f"{base_url}/api/v1/openapi.yaml")),
        ("Swagger UI", lambda: test_swagger_ui(base_url)),
        ("Endpoint Coverage", check_endpoint_coverage),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print('=' * 60)
        result = test_func()
        results.append((test_name, result))

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    # Overall result
    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 All validation tests passed!")
        print(f"\n📚 View API documentation at: {base_url}/docs")
        return 0
    else:
        print("\n⚠️  Some validation tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

============================================================
# scripts/test_pipeline_switching.py
============================================================
#!/usr/bin/env python3
"""Task 6.4: Test pipeline switching with memory unloading.

Verifies that switching between pipelines properly unloads models and recovers memory.
"""

import os
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import numpy as np
import librosa

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig
from quality_pipeline import QualityVoiceConverter, QualityConfig


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 / 1024


def main():
    print("\n" + "=" * 70)
    print("  TASK 6.4: PIPELINE SWITCHING TEST")
    print("  Verifying memory recovery when switching between pipelines")
    print("=" * 70)

    os.chdir(Path(__file__).parent.parent)

    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available - cannot test GPU memory recovery")
        print("Skipping test")
        return 0

    # Prepare test data
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=5.0)
    speaker_embedding = np.random.randn(256).astype(np.float32)
    reference_audio = audio.copy()

    # Baseline
    torch.cuda.empty_cache()
    gc.collect()
    baseline = get_gpu_memory_mb()
    print(f"\nBaseline GPU memory: {baseline:.1f} MB")

    # Test 1: Realtime → Quality
    print("\n" + "=" * 70)
    print("TEST 1: Switch from Realtime to Quality")
    print("=" * 70)

    print("\n1. Load Realtime pipeline...")
    realtime_config = RealtimeConfig(sample_rate=22050, fp16=True, device="cuda")
    realtime_converter = RealtimeVoiceConverter(realtime_config)
    realtime_converter.convert_full(audio, sr, speaker_embedding)
    realtime_peak = get_gpu_memory_mb()
    print(f"   Memory after Realtime conversion: {realtime_peak:.1f} MB")

    print("\n2. Unload Realtime pipeline...")
    realtime_converter.unload()
    torch.cuda.empty_cache()
    gc.collect()
    after_realtime_unload = get_gpu_memory_mb()
    recovered_1 = realtime_peak - after_realtime_unload
    print(f"   Memory after unload: {after_realtime_unload:.1f} MB (recovered {recovered_1:.1f} MB)")

    print("\n3. Load Quality pipeline...")
    quality_config = QualityConfig(sample_rate=44100, diffusion_steps=10, fp16=True, device="cuda")
    quality_converter = QualityVoiceConverter(quality_config)
    quality_converter.convert(audio, sr, reference_audio, sr)
    quality_peak = get_gpu_memory_mb()
    print(f"   Memory after Quality conversion: {quality_peak:.1f} MB")

    print("\n4. Unload Quality pipeline...")
    quality_converter.unload()
    torch.cuda.empty_cache()
    gc.collect()
    after_quality_unload = get_gpu_memory_mb()
    recovered_2 = quality_peak - after_quality_unload
    print(f"   Memory after unload: {after_quality_unload:.1f} MB (recovered {recovered_2:.1f} MB)")

    # Test 2: Quality → Realtime
    print("\n" + "=" * 70)
    print("TEST 2: Switch from Quality to Realtime")
    print("=" * 70)

    print("\n1. Load Quality pipeline...")
    quality_converter = QualityVoiceConverter(quality_config)
    quality_converter.convert(audio, sr, reference_audio, sr)
    quality_peak_2 = get_gpu_memory_mb()
    print(f"   Memory after Quality conversion: {quality_peak_2:.1f} MB")

    print("\n2. Unload Quality pipeline...")
    quality_converter.unload()
    torch.cuda.empty_cache()
    gc.collect()
    after_quality_unload_2 = get_gpu_memory_mb()
    recovered_3 = quality_peak_2 - after_quality_unload_2
    print(f"   Memory after unload: {after_quality_unload_2:.1f} MB (recovered {recovered_3:.1f} MB)")

    print("\n3. Load Realtime pipeline...")
    realtime_converter = RealtimeVoiceConverter(realtime_config)
    realtime_converter.convert_full(audio, sr, speaker_embedding)
    realtime_peak_2 = get_gpu_memory_mb()
    print(f"   Memory after Realtime conversion: {realtime_peak_2:.1f} MB")

    print("\n4. Unload Realtime pipeline...")
    realtime_converter.unload()
    torch.cuda.empty_cache()
    gc.collect()
    final_memory = get_gpu_memory_mb()
    recovered_4 = realtime_peak_2 - final_memory
    print(f"   Memory after unload: {final_memory:.1f} MB (recovered {recovered_4:.1f} MB)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline memory: {baseline:.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Net memory leak: {final_memory - baseline:.1f} MB")

    print("\nMemory recovery rates:")
    recovery_rate_1 = (recovered_1 / realtime_peak) * 100 if realtime_peak > 0 else 0
    recovery_rate_2 = (recovered_2 / quality_peak) * 100 if quality_peak > 0 else 0
    recovery_rate_3 = (recovered_3 / quality_peak_2) * 100 if quality_peak_2 > 0 else 0
    recovery_rate_4 = (recovered_4 / realtime_peak_2) * 100 if realtime_peak_2 > 0 else 0

    print(f"  Realtime unload #1: {recovery_rate_1:.1f}%")
    print(f"  Quality unload #1:  {recovery_rate_2:.1f}%")
    print(f"  Quality unload #2:  {recovery_rate_3:.1f}%")
    print(f"  Realtime unload #2: {recovery_rate_4:.1f}%")

    avg_recovery = (recovery_rate_1 + recovery_rate_2 + recovery_rate_3 + recovery_rate_4) / 4
    print(f"\nAverage recovery rate: {avg_recovery:.1f}%")

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    if avg_recovery > 95:
        print("\n✓ Memory recovery is excellent (>95%)")
    elif avg_recovery > 90:
        print("\n✓ Memory recovery is good (>90%)")
    else:
        print(f"\n⚠ Memory recovery is suboptimal ({avg_recovery:.1f}%)")

    if (final_memory - baseline) < 50:
        print("✓ Minimal memory leak (<50 MB)")
    else:
        print(f"⚠ Memory leak detected ({final_memory - baseline:.1f} MB)")

    print("\n" + "=" * 70)
    print("✓ TASK 6.4 COMPLETE")
    print("=" * 70)
    print("\nConclusion: Pipeline switching with unload() works correctly")
    print("Both pipelines properly release GPU memory when unloaded\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

============================================================
# scripts/test_combined_pipeline.py
============================================================
#!/usr/bin/env python3
"""Test Task 3.3: Combined pipeline - Seed-VC → HQ-SVC super-resolution.

Tests chaining the quality pipeline with HQ-SVC enhancement for super-resolution.
Pipeline: Source audio → Seed-VC (44kHz) → Downsample (22kHz) → HQ-SVC upsample (44kHz)

This tests whether HQ-SVC can enhance the quality of Seed-VC output.
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import numpy as np
import librosa
import soundfile as sf
import torchaudio

from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  TASK 3.3: COMBINED PIPELINE TEST")
    print("  Seed-VC (44kHz) → Downsample (22kHz) → HQ-SVC Super-resolution (44kHz)")
    print("=" * 70 + "\n")

    # Change to repo root
    os.chdir(Path(__file__).parent.parent)

    # Use the quality pipeline output from Task 2.8 as input
    seedvc_output = "tests/quality_samples/outputs/william_as_conor_quality_30s.wav"

    if not Path(seedvc_output).exists():
        print(f"ERROR: Seed-VC output not found: {seedvc_output}")
        print("Run test_quality_pipeline_hq.py first (Task 2.8)")
        return 1

    # Load Seed-VC output
    print(f"Loading Seed-VC output: {seedvc_output}")
    audio, sr = librosa.load(seedvc_output, sr=None, mono=True)
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Shape: {audio.shape}")

    # Downsample to 22kHz to simulate lower quality
    print("\nDownsampling to 22kHz (simulating lower quality)...")
    audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    print(f"  22kHz shape: {audio_22k.shape}")

    # Initialize HQ-SVC wrapper
    print("\nInitializing HQ-SVC wrapper for super-resolution...")
    try:
        hqsvc = HQSVCWrapper(device=torch.device("cuda"))
    except Exception as e:
        print(f"ERROR: Failed to initialize HQ-SVC: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Super-resolve 22kHz → 44kHz
    print("\n" + "=" * 70)
    print("SUPER-RESOLVING (HQ-SVC Enhancement)...")
    print("=" * 70)
    start_time = time.time()

    try:
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_22k).float()

        # Progress callback
        def progress_callback(stage: str, progress: float):
            logger.info(f"  {stage}: {progress*100:.0f}%")

        # Super-resolve
        result = hqsvc.super_resolve(
            audio=audio_tensor,
            sample_rate=22050,
            on_progress=progress_callback
        )

        enhanced_audio = result['audio'].cpu().numpy()
        enhanced_sr = result['sample_rate']

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio_22k) / 22050)

        print(f"\n✓ Super-resolution complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Output SR: {enhanced_sr}Hz")
        print(f"  Output shape: {enhanced_audio.shape}")

    except Exception as e:
        print(f"\n✗ Super-resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save outputs
    output_dir = Path("tests/quality_samples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save downsampled version (for comparison)
    downsampled_path = output_dir / "william_as_conor_22k_intermediate.wav"
    sf.write(str(downsampled_path), audio_22k, 22050)

    # Save enhanced version
    enhanced_path = output_dir / "william_as_conor_combined_30s.wav"
    sf.write(str(enhanced_path), enhanced_audio, enhanced_sr)

    print(f"\nOutputs saved:")
    print(f"  Original (Seed-VC):  {seedvc_output} (44kHz)")
    print(f"  Downsampled:         {downsampled_path} (22kHz)")
    print(f"  Enhanced (Combined): {enhanced_path} (44kHz)")

    # Cleanup
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✓ TASK 3.3 COMPLETE")
    print("=" * 70)
    print("\nCombined Pipeline Results:")
    print("  Input:  Seed-VC quality conversion (44kHz)")
    print("  Step 1: Downsample to 22kHz")
    print("  Step 2: HQ-SVC super-resolution to 44kHz")
    print(f"  Output: {enhanced_path}")
    print("\nNext: Task 3.4 - Benchmark quality improvement vs latency cost\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

============================================================
# scripts/test_progress_callbacks.py
============================================================
#!/usr/bin/env python3
"""Task 6.5: Test progress callbacks for long conversions.

Verifies that both pipelines emit progress updates during conversion.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import numpy as np
import librosa

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig
from quality_pipeline import QualityVoiceConverter, QualityConfig


def test_realtime_progress():
    """Test realtime pipeline progress callbacks."""
    print("\n" + "=" * 70)
    print("TEST 1: REALTIME PIPELINE PROGRESS CALLBACKS")
    print("=" * 70)

    # Track progress updates
    progress_updates = []

    def progress_callback(progress: float, status: str):
        progress_updates.append((progress, status))
        print(f"  [{progress*100:5.1f}%] {status}")

    # Load test audio
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=10.0)
    speaker_embedding = np.random.randn(256).astype(np.float32)

    # Initialize and convert
    config = RealtimeConfig(sample_rate=22050, fp16=True, device="cuda")
    converter = RealtimeVoiceConverter(config)

    print("\nConverting with progress callbacks...")
    converted, out_sr = converter.convert_full(
        audio, sr, speaker_embedding,
        progress_callback=progress_callback
    )

    converter.unload()

    # Verify
    print(f"\nTotal progress updates: {len(progress_updates)}")
    print(f"First update: {progress_updates[0] if progress_updates else 'None'}")
    print(f"Last update: {progress_updates[-1] if progress_updates else 'None'}")

    assert len(progress_updates) > 0, "No progress updates received"
    assert progress_updates[0][0] == 0.0, "First progress should be 0.0"
    assert progress_updates[-1][0] == 1.0, "Last progress should be 1.0"
    assert "Complete" in progress_updates[-1][1], "Last status should indicate completion"

    print("\n✓ Realtime pipeline progress callbacks working correctly")
    return True


def test_quality_progress():
    """Test quality pipeline progress callbacks."""
    print("\n" + "=" * 70)
    print("TEST 2: QUALITY PIPELINE PROGRESS CALLBACKS")
    print("=" * 70)

    # Track progress updates
    progress_updates = []

    def progress_callback(progress: float, status: str):
        progress_updates.append((progress, status))
        print(f"  [{progress*100:5.1f}%] {status}")

    # Load test audio
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=5.0)
    reference = librosa.load("data/separated_youtube/conor_maynard/08NWh97_DME_vocals.wav",
                            sr=None, mono=True, duration=5.0)[0]

    # Initialize and convert
    config = QualityConfig(sample_rate=44100, diffusion_steps=10, fp16=True, device="cuda")
    converter = QualityVoiceConverter(config)

    print("\nConverting with progress callbacks...")
    converted, out_sr = converter.convert(
        audio, sr, reference, sr,
        progress_callback=progress_callback
    )

    converter.unload()

    # Verify
    print(f"\nTotal progress updates: {len(progress_updates)}")
    print(f"First update: {progress_updates[0] if progress_updates else 'None'}")
    print(f"Last update: {progress_updates[-1] if progress_updates else 'None'}")

    assert len(progress_updates) > 0, "No progress updates received"
    assert progress_updates[0][0] <= 0.2, "First progress should be early stage"
    assert progress_updates[-1][0] == 1.0, "Last progress should be 1.0"
    assert "Complete" in progress_updates[-1][1], "Last status should indicate completion"

    print("\n✓ Quality pipeline progress callbacks working correctly")
    return True


def main():
    os.chdir(Path(__file__).parent.parent)

    print("\n" + "=" * 70)
    print("  TASK 6.5: PROGRESS CALLBACK TEST")
    print("  Verifying progress updates during long conversions")
    print("=" * 70)

    try:
        # Test both pipelines
        realtime_ok = test_realtime_progress()
        quality_ok = test_quality_progress()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if realtime_ok and quality_ok:
            print("\n✓ Both pipelines emit progress callbacks correctly")
            print("  - Realtime: Progress from 0.0 to 1.0 with status updates")
            print("  - Quality: Progress from 0.0 to 1.0 with detailed stage updates")
        else:
            print("\n✗ Some progress callbacks failed")

        print("\n" + "=" * 70)
        print("✓ TASK 6.5 COMPLETE")
        print("=" * 70)
        print("\nProgress callbacks implemented for WebSocket real-time updates")
        print("UI can now show conversion progress to users\n")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

============================================================
# scripts/test_realtime_pipeline_hq.py
============================================================
#!/usr/bin/env python3
"""Test Task 1.7: William->Conor conversion using realtime pipeline with HQ LoRA.

Tests the realtime voice conversion pipeline with trained voice profiles.
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import numpy as np
import librosa
import soundfile as sf

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig, load_speaker_embedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  TASK 1.7: REALTIME PIPELINE TEST WITH HQ LORA")
    print("  William Singe → Conor Maynard Conversion")
    print("=" * 70 + "\n")

    # Change to repo root
    os.chdir(Path(__file__).parent.parent)

    # Profile IDs from spec
    WILLIAM_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
    CONOR_ID = "c572d02c-c687-4bed-8676-6ad253cf1c91"

    # Test audio: Use a short William vocals file (first 30s for testing)
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"

    if not Path(test_audio).exists():
        print(f"ERROR: Test audio not found: {test_audio}")
        return 1

    # Load source audio (first 30s only)
    print(f"Loading source audio: {test_audio}")
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=30.0)
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Shape: {audio.shape}")

    # Load target speaker embedding (Conor)
    print(f"\nLoading target speaker: Conor (ID: {CONOR_ID})")
    try:
        target_embedding = load_speaker_embedding(CONOR_ID)
        print(f"  Embedding shape: {target_embedding.shape}")
        print(f"  Embedding dtype: {target_embedding.dtype}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize realtime converter
    print("\nInitializing realtime converter...")
    config = RealtimeConfig(
        sample_rate=22050,
        chunk_size_ms=100,
        overlap_ms=20,
        fp16=True,
        device="cuda"
    )
    converter = RealtimeVoiceConverter(config)

    # Convert
    print("\n" + "=" * 70)
    print("CONVERTING (Realtime Pipeline)...")
    print("=" * 70)
    start_time = time.time()

    try:
        converted, out_sr = converter.convert_full(
            audio=audio,
            sr=sr,
            speaker_embedding=target_embedding,
            pitch_shift=0.0  # No pitch shift for now
        )

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio) / sr)

        print(f"\n✓ Conversion complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Output SR: {out_sr}Hz")
        print(f"  Output shape: {converted.shape}")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    output_dir = Path("tests/quality_samples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "william_as_conor_realtime_30s.wav"

    print(f"\nSaving output: {output_path}")
    sf.write(str(output_path), converted, out_sr)
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Cleanup
    converter.unload()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✓ TASK 1.7 COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print("Next: Verify audio quality and proceed to Task 2.8\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

============================================================
# scripts/test_quality_pipeline_hq.py
============================================================
#!/usr/bin/env python3
"""Test Task 2.8: William->Conor conversion using quality pipeline with HQ LoRA.

Tests the Seed-VC quality voice conversion pipeline with trained voice profiles.
Compares quality vs realtime pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import numpy as np
import librosa
import soundfile as sf

from quality_pipeline import QualityVoiceConverter, QualityConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  TASK 2.8: QUALITY PIPELINE TEST WITH HQ LORA")
    print("  William Singe → Conor Maynard Conversion (Seed-VC)")
    print("=" * 70 + "\n")

    # Change to repo root
    os.chdir(Path(__file__).parent.parent)

    # Profile IDs from spec
    WILLIAM_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
    CONOR_ID = "c572d02c-c687-4bed-8676-6ad253cf1c91"

    # Test audio: Same as realtime test for comparison
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"

    if not Path(test_audio).exists():
        print(f"ERROR: Test audio not found: {test_audio}")
        return 1

    # Load source audio (first 30s for direct comparison)
    print(f"Loading source audio: {test_audio}")
    source_audio, source_sr = librosa.load(test_audio, sr=None, mono=True, duration=30.0)
    print(f"  Duration: {len(source_audio)/source_sr:.1f}s")
    print(f"  Sample rate: {source_sr}Hz")
    print(f"  Shape: {source_audio.shape}")

    # Load reference audio (Conor vocals for style)
    print(f"\nLoading reference speaker: Conor")
    reference_audio_path = "data/separated_youtube/conor_maynard/08NWh97_DME_vocals.wav"
    if not Path(reference_audio_path).exists():
        print(f"ERROR: Reference audio not found: {reference_audio_path}")
        return 1

    reference_audio, reference_sr = librosa.load(reference_audio_path, sr=None, mono=True, duration=25.0)
    print(f"  Reference: {reference_audio_path}")
    print(f"  Duration: {len(reference_audio)/reference_sr:.1f}s")
    print(f"  Sample rate: {reference_sr}Hz")

    # Initialize quality converter
    print("\nInitializing quality converter (Seed-VC)...")
    config = QualityConfig(
        sample_rate=44100,
        diffusion_steps=30,
        f0_condition=True,
        auto_f0_adjust=False,
        fp16=True,
        device="cuda"
    )
    converter = QualityVoiceConverter(config)

    # Convert
    print("\n" + "=" * 70)
    print("CONVERTING (Quality Pipeline - Seed-VC)...")
    print("=" * 70)
    start_time = time.time()

    try:
        converted, out_sr = converter.convert(
            source_audio=source_audio,
            source_sr=source_sr,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            pitch_shift=0  # No pitch shift for now
        )

        elapsed = time.time() - start_time
        rtf = elapsed / (len(source_audio) / source_sr)

        print(f"\n✓ Conversion complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Output SR: {out_sr}Hz")
        print(f"  Output shape: {converted.shape}")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    output_dir = Path("tests/quality_samples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "william_as_conor_quality_30s.wav"

    print(f"\nSaving output: {output_path}")
    sf.write(str(output_path), converted, out_sr)
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Cleanup
    converter.unload()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✓ TASK 2.8 COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print("Comparison:")
    print("  - Realtime: tests/quality_samples/outputs/william_as_conor_realtime_30s.wav (22kHz)")
    print(f"  - Quality:  {output_path} (44kHz)")
    print("\nNext: Compare audio quality and proceed to Task 3.3\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

============================================================
# scripts/train_hq_lora_optimized.py [SUMMARIZED]
============================================================
"""Optimized High-Quality LoRA Training for NVIDIA Thor.

Key optimizations for full GPU utilization:
1. Pre-extract all ContentVec features before training
2. Large batch sizes (32) for better GPU saturation
3. Multiple data loading workers with prefetching
4. Pin memory for faster CPU to GPU transfer
5. Gradient accumulation for effective larger batches

Usage:
    python scripts/train_hq_lora_optimized.py --artist conor_maynard --epochs 200
    python scripts/train_hq_lora_optimized.py --artist william_singe --epochs 200
    python scripts/train_hq_lora_optimized.py --artist both --epochs 200"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn
import torch.nn.functional
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy
import librosa
import soundfile
from tqdm import tqdm
ARTIST_PROFILES = {'conor_maynard': {'name': 'Conor Maynard', 'profi...
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DIARIZED_DIR = DATA_DIR / 'diarized_youtube'
SEPARATED_DIR = DATA_DIR / 'separated_youtube'
FEATURES_DIR = DATA_DIR / 'features_cache'
CHECKPOINTS_DIR = DATA_DIR / 'checkpoints' / 'hq'
OUTPUT_DIR = DATA_DIR / 'trained_models' / 'hq'
HQ_CONFIG = {'input_dim': 768, 'hidden_dim': 1024, 'output_dim...
TRAIN_CONFIG = {'batch_size': 32, 'gradient_accumulation': 2, 'nu...

def print_banner(text: str):
    ...

def print_gpu_memory():
    ...

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 128, alpha: float = 256.0, dropout: float = 0.05):
        ...
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        ...
    def get_delta_weight(self) -> torch.Tensor:
        ...

class HQVoiceLoRAAdapter(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, output_dim: int = 768, lora_rank: int = 128, lora_alpha: float = 256.0, dropout: float = 0.05, num_layers: int = 6):
        ...
    def forward(self, content: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
# ... (truncated)
============================================================
# scripts/train_pillowtalk.py [SUMMARIZED]
============================================================
"""Train voice models on Pillowtalk with live progress display.

This script:
1. Separates vocals from Pillowtalk for both artists
2. Trains LoRA adapters with live progress output
3. Saves trained models for voice conversion"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
import numpy
WILLIAM_PROFILE_ID = '7da05140-1303-40c6-95d9-5b6e2c3624df'
CONOR_PROFILE_ID = '9679a6ec-e6e2-43c4-b64e-1f004fed34f9'
PILLOWTALK_WILLIAM = 'tests/quality_samples/william_singe_pillowtalk.wa...
PILLOWTALK_CONOR = 'tests/quality_samples/conor_maynard_pillowtalk.wa...
PROFILES_DIR = 'data/voice_profiles'
SEPARATED_DIR = 'data/separated'
MODELS_DIR = 'data/trained_models'
TRAINING_CONFIG = {'epochs': 5, 'learning_rate': 0.0001, 'batch_size...

def print_banner(text: str):
    """Print a prominent banner...."""
    ...

def print_progress(epoch: int, step: int, loss: float, progress: int, profile_name: str):
    """Print training progress...."""
    ...

def separate_vocals(audio_path: str, profile_id: str, profile_name: str) -> dict:
    """Separate vocals from audio using Demucs...."""
    ...

def extract_mel_features(audio_path: str, device: torch.device) -> torch.Tensor:
    """Extract mel spectrogram features for training...."""
    ...

def extract_speaker_embedding(audio_path: str) -> torch.Tensor:
    """Extract speaker embedding from audio...."""
    ...

class SimpleSample:
    """Simple training sample container."""
    def __init__(self, mel_tensor: torch.Tensor, speaker_embedding: torch.Tensor):
        ...

class SimpleMelModel(torch.nn.Module):
    """Simple mel-to-embedding model for training."""
    def __init__(self, mel_channels: int = 128, embedding_dim: int = 256):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

def train_voice_model(profile_id: str, profile_name: str, vocals_path: str, output_dir: str, config: dict) -> dict:
    """Train a voice model with live progress display...."""
    ...

# ... (truncated)
============================================================
# scripts/quality_validation.py [SUMMARIZED]
============================================================
"""Quality validation script for LoRA adapters.

Task 5.1: Validates voice conversion quality across adapter types.
Compares HQ vs nvfp4 adapters using objective audio quality metrics.

Usage:
    python scripts/quality_validation.py --profile-id <id> --input audio.wav
    python scripts/quality_validation.py --all-profiles --report"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import torch
import numpy

class QualityMetrics:
    """Audio quality metrics for a single conversion."""

class ComparisonReport:
    """Quality comparison between HQ and nvfp4 adapters."""

def calculate_snr(signal: torch.Tensor, noise_floor: float = 1e-10) -> float:
    """Calculate signal-to-noise ratio in dB...."""
    ...

def calculate_spectral_centroid(audio: torch.Tensor, sample_rate: int) -> float:
    """Calculate spectral centroid (brightness) in Hz...."""
    ...

def calculate_zero_crossing_rate(audio: torch.Tensor) -> float:
    """Calculate zero-crossing rate (noisiness indicator)...."""
    ...

def load_adapter(adapter_path: Path) -> dict:
    """Load adapter state dict and extract metadata...."""
    ...

def mock_convert_audio(input_audio: torch.Tensor, sample_rate: int, adapter_info: dict) -> tuple[torch.Tensor, float]:
    """Mock audio conversion for validation...."""
    ...

def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio file with fallback methods...."""
    ...

def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """Save audio file with fallback methods...."""
    ...

def validate_single_profile(profile_id: str, input_path: Path, data_dir: Path) -> ComparisonReport:
    """Validate quality for a single voice profile...."""
    ...

def generate_report(reports: list[ComparisonReport], output_path: Path) -> None:
    """Generate JSON quality comparison report...."""
    ...
# ... (truncated)
============================================================
# scripts/verify_bindings.py [SUMMARIZED]
============================================================
"""Verify AutoVoice module bindings and CUDA extension."""

import sys
import os

def check_module(name, import_path):
    """Check if a module imports successfully...."""
    ...

def main():
    ...
============================================================
# scripts/aligned_conversion.py [SUMMARIZED]
============================================================
"""Aligned Voice Conversion Pipeline

This pipeline:
1. Aligns source vocals to match target timing using DTW
2. Adjusts pitch to match target pitch contour (optional)
3. Converts the voice timbre using Seed-VC
4. Mixes with target instrumental

This ensures the converted vocals match the target's timing perfectly."""

import argparse
import logging
import numpy
import librosa
import soundfile
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional

def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCCs for DTW alignment...."""
    ...

def align_with_dtw(source_audio: np.ndarray, source_sr: int, target_audio: np.ndarray, target_sr: int) -> Tuple[np.ndarray, int]:
    """Align source audio to match target timing using DTW...."""
    ...

def align_pitch(aligned_audio: np.ndarray, sr: int, target_audio: np.ndarray, target_sr: int) -> np.ndarray:
    """Adjust pitch of aligned audio to match target pitch contour...."""
    ...

def run_voice_conversion(source_audio: np.ndarray, source_sr: int, reference_audio: np.ndarray, reference_sr: int) -> Tuple[np.ndarray, int]:
    """Run Seed-VC voice conversion...."""
    ...

def mix_with_instrumental(vocals: np.ndarray, vocals_sr: int, instrumental: np.ndarray, instrumental_sr: int, vocal_gain: float = 1.0, inst_gain: float = 0.8) -> Tuple[np.ndarray, int]:
    """Mix vocals with instrumental...."""
    ...

def main():
    ...
============================================================
# scripts/swarm_orchestrator.py [SUMMARIZED]
============================================================
"""Claude-Flow Smart Swarm Orchestrator for AutoVoice

Launches and coordinates parallel agent swarms for:
- SOTA Dual-Pipeline implementation (P0)
- Training-Inference integration (P1)
- YouTube Artist training pipeline (P1)

Usage:
    python scripts/swarm_orchestrator.py --swarm all
    python scripts/swarm_orchestrator.py --swarm sota-dual-pipeline
    python scripts/swarm_orchestrator.py --swarm youtube-artist --parallel 4
    python scripts/swarm_orchestrator.py --status"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / 'config' / 'swarm_config.yaml'
AGENT_CONTEXTS_FILE = PROJECT_ROOT / 'config' / 'agent_contexts.yaml'

class SwarmStatus:
    """Status of a swarm execution."""

def load_config() -> dict[str, Any]:
    """Load swarm configuration...."""
    ...

def load_agent_contexts() -> dict[str, Any]:
    """Load agent context injection rules...."""
    ...

def run_command(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    """Run a shell command...."""
    ...

def check_claude_flow() -> bool:
    """Check if claude-flow is available...."""
    ...

def init_queen(config: dict[str, Any]) -> bool:
    """Initialize the Queen coordinator with full project context...."""
    ...

def build_agent_context(agent_name: str, agent_config: dict[str, Any], swarm_config: dict[str, Any], agent_contexts: dict[str, Any]) -> list[str]:
    """Build the context file list for an agent...."""
    ...
PRIORITY_TIERS = {'critical': ['CLAUDE.md', 'spec.md', 'plan.md', '...
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    """Estimate token count from text length...."""
    ...
# ... (truncated)
============================================================
# scripts/train_nvfp4_lora.py [SUMMARIZED]
============================================================
"""Train nvfp4-optimized LoRA adapters for NVIDIA Thor.

Trains LoRA voice adapters using diarized YouTube vocal data with:
- Mixed precision (fp16/bf16) training
- Gradient checkpointing for memory efficiency
- nvfp4 quantization for inference deployment
- Optimized for Jetson Thor CUDA 13.0 / SM 11.0

Usage:
    python scripts/train_nvfp4_lora.py --artist conor_maynard --epochs 100
    python scripts/train_nvfp4_lora.py --artist william_singe --epochs 100
    python scripts/train_nvfp4_lora.py --artist all --epochs 100"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy
import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import librosa
import soundfile
ARTIST_PROFILES = {'conor_maynard': {'profile_id': 'c572d02c-c687-4b...
SEPARATED_DIR = Path('data/separated_youtube')
DIARIZED_DIR = Path('data/diarized_youtube')
ADAPTERS_DIR = Path('data/trained_models/nvfp4')
CHECKPOINTS_DIR = Path('data/checkpoints/nvfp4')

def print_banner(text: str):
    ...

def print_gpu_memory():
    """Print GPU memory usage...."""
    ...

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer optimized for Thor nvfp4."""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation: base_output + scaling * (x @ A^T @ B^T)..."""
        ...
    def get_delta_weight(self) -> torch.Tensor:
        """Get the delta weight matrix for merging...."""
        ...

class VoiceLoRAAdapter(nn.Module):
    """Voice conversion LoRA adapter with nvfp4 optimization."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256, lora_rank: int = 16, lora_alpha: float = 32.0, dropout: float = 0.1, num_layers: int = 3):
        ...
    def forward(self, content: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
# ... (truncated)
============================================================
# scripts/benchmark_pipelines.py [SUMMARIZED]
============================================================
"""Task 3.4: Benchmark quality improvement vs latency cost.

Compares three pipelines:
1. Realtime: ContentVec + Simple Decoder + HiFiGAN (22kHz)
2. Quality: Seed-VC with Whisper + DiT + BigVGAN (44kHz)
3. Combined: Seed-VC + HQ-SVC super-resolution (44kHz enhanced)

Metrics:
- Processing time & RTF (Real-Time Factor)
- Speaker similarity (cosine similarity of embeddings)
- Mel Cepstral Distortion (MCD)
- Output sample rate"""

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import torch
import numpy
import librosa
import soundfile

class BenchmarkResult:
    """Results from a single pipeline benchmark."""

def compute_speaker_similarity(audio1: np.ndarray, sr1: int, audio2: np.ndarray, sr2: int) -> float:
    """Compute speaker similarity using CAMPPlus embeddings...."""
    ...

def compute_mcd(audio1: np.ndarray, sr1: int, audio2: np.ndarray, sr2: int, n_mfcc: int = 13) -> float:
    """Compute Mel Cepstral Distortion between two audio signals...."""
    ...

def main():
    ...
============================================================
# scripts/train_fp16_lora.py [SUMMARIZED]
============================================================
"""Train full fp16 LoRA adapters for quality testing and validation.

Trains LoRA voice adapters at full fp16 precision (no quantization) for:
- Quality validation and comparison with nvfp4 models
- Performance benchmarking
- A/B testing between quantized and full-precision inference

Usage:
    python scripts/train_fp16_lora.py --artist conor_maynard --epochs 100
    python scripts/train_fp16_lora.py --artist william_singe --epochs 100
    python scripts/train_fp16_lora.py --artist all --epochs 100"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy
import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
import librosa
import soundfile
ARTIST_PROFILES = {'conor_maynard': {'profile_id': 'c572d02c-c687-4b...
SEPARATED_DIR = Path('data/separated_youtube')
DIARIZED_DIR = Path('data/diarized_youtube')
ADAPTERS_DIR = Path('data/trained_models/fp16')
CHECKPOINTS_DIR = Path('data/checkpoints/fp16')

def print_banner(text: str):
    ...

def print_gpu_memory():
    """Print GPU memory usage...."""
    ...

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for fp16 training."""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation: base_output + scaling * (x @ A^T @ B^T)..."""
        ...
    def get_delta_weight(self) -> torch.Tensor:
        """Get the delta weight matrix for merging...."""
        ...

class VoiceLoRAAdapter(nn.Module):
    """Voice conversion LoRA adapter - full fp16 version."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256, lora_rank: int = 16, lora_alpha: float = 32.0, dropout: float = 0.1, num_layers: int = 3):
        ...
    def forward(self, content: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...
# ... (truncated)
============================================================
# scripts/realtime_pipeline.py [SUMMARIZED]
============================================================
"""Realtime Voice Conversion Pipeline for AutoVoice.

Optimized for low-latency karaoke applications.
Architecture: ContentVec -> RMVPE -> Simple Decoder -> HiFiGAN

Design choices for low latency:
- ContentVec (lighter than Whisper) for content extraction
- Streaming-friendly chunk processing
- HiFiGAN vocoder (faster than BigVGAN)
- FP16 inference throughout"""

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional
import numpy
import librosa
import soundfile
import torchaudio

class RealtimeConfig:
    """Configuration for realtime pipeline."""

class RealtimeVoiceConverter:
    """Low-latency voice conversion for karaoke applications.

Pipeline:
1. ContentVec extracts linguistic content (speaker-invariant)
2. RMVPE extracts pitch (F0) contour
3. Simple decoder combines content ..."""
    def __init__(self, config: Optional[RealtimeConfig] = None):
        ...
    def _load_contentvec(self):
        """Load ContentVec encoder for content extraction...."""
        ...
    def _load_rmvpe(self):
        """Load RMVPE pitch extractor...."""
        ...
    def _load_vocoder(self):
        """Load HiFiGAN vocoder for fast synthesis...."""
        ...
    def _build_simple_decoder(self):
        """Build lightweight decoder for realtime inference...."""
        ...
    def extract_content(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract content features from audio...."""
        ...
    def extract_pitch(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract F0 pitch contour...."""
        ...
    def convert_chunk(self, audio_chunk: torch.Tensor, sr: int, speaker_embedding: torch.Tensor, pitch_shift: float = 0.0) -> torch.Tensor:
        """Convert a single chunk of audio (low-latency path)...."""
        ...
    def convert_streaming(self, audio: np.ndarray, sr: int, speaker_embedding: np.ndarray, pitch_shift: float = 0.0, callback = None, progress_callback = None) -> np.ndarray:
# ... (truncated)
============================================================
# scripts/benchmark_pipelines_comprehensive.py [SUMMARIZED]
============================================================
"""Comprehensive Performance Benchmark Suite for AutoVoice Pipelines.

Benchmarks all voice conversion pipeline types with detailed metrics:
- realtime: Low-latency ContentVec + SimpleDecoder + HiFiGAN (22kHz)
- quality: SOTA CoMoSVC consistency model (24kHz)
- quality_seedvc: Seed-VC DiT-CFM with BigVGAN (44.1kHz)
- realtime_meanvc: MeanVC streaming with mean flows (16kHz)

Metrics measured:
- RTF (Real-Time Factor) - must be <1.0 for realtime pipelines
- Latency (ms) - time to first output chunk
- GPU Memory (MB) - peak memory allocation
- MCD (Mel Cepstral Distortion) - synthesis quality metric
- Speaker Similarity - cosine similarity of embeddings vs reference

Usage:
    python scripts/benchmark_pipelines_comprehensive.py
    python scripts/benchmark_pipelines_comprehensive.py --pipelines realtime quality
    python scripts/benchmark_pipelines_comprehensive.py --iterations 20 --output reports/benchmark.json"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import torch

class PipelineMetrics:
    """Comprehensive metrics for a single pipeline benchmark."""

class BenchmarkReport:
    """Complete benchmark report across all pipelines."""

def get_system_info() -> Dict[str, Any]:
    """Gather system information for reproducibility...."""
    ...

def reset_gpu_memory():
    """Reset GPU memory for accurate measurement...."""
    ...

def get_gpu_memory_mb() -> Tuple[float, float]:
    """Get current and peak GPU memory in MB...."""
    ...

def compute_speaker_similarity(audio1: np.ndarray, sr1: int, audio2: np.ndarray, sr2: int, device: torch.device = torch.device('cpu')) -> float:
    """Compute speaker similarity using mel-statistic embeddings...."""
    ...

def compute_mcd(audio1: np.ndarray, sr1: int, audio2: np.ndarray, sr2: int, n_mfcc: int = 13) -> float:
    """Compute Mel Cepstral Distortion between two audio signals...."""
    ...

# ... (truncated)
============================================================
# scripts/quality_pipeline.py [SUMMARIZED]
============================================================
"""Quality Voice Conversion Pipeline for AutoVoice.

High-fidelity pipeline for song conversion using Seed-VC.
Architecture: Whisper -> Seed-VC DiT (CFM) -> BigVGAN (44kHz)

Design choices for quality:
- Whisper encoder for robust semantic extraction
- Seed-VC DiT with Conditional Flow Matching
- BigVGAN vocoder for high-fidelity 44kHz synthesis
- CAMPPlus speaker style encoding
- Optional HQ-SVC enhancement"""

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import torch
import torch.nn.functional
import numpy
import librosa
import soundfile
import torchaudio
import yaml

class QualityConfig:
    """Configuration for quality pipeline."""

class QualityVoiceConverter:
    """High-fidelity voice conversion using Seed-VC.

Pipeline:
1. Whisper extracts semantic features (speaker-invariant)
2. CAMPPlus extracts speaker style from reference
3. RMVPE extracts F0 pitch contour
..."""
    def __init__(self, config: Optional[QualityConfig] = None):
        ...
    def _load_models(self):
        """Load all Seed-VC models...."""
        ...
    def extract_speaker_style(self, reference_audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract speaker style embedding from reference audio using CAMPPlus...."""
        ...
    def convert(self, source_audio: np.ndarray, source_sr: int, reference_audio: np.ndarray, reference_sr: int, pitch_shift: int = 0, progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[np.ndarray, int]:
        """Convert source audio to target voice style...."""
        ...
    def _crossfade(self, chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
        """Crossfade between two audio chunks...."""
        ...
    def unload(self):
        """Unload models to free GPU memory...."""
        ...

def load_speaker_embedding(profile_id: str) -> np.ndarray:
    """Load speaker embedding from profile...."""
    ...

# ... (truncated)
============================================================
# scripts/performance_validation.py [SUMMARIZED]
============================================================
"""Performance Validation Suite for AutoVoice Pipelines.

Comprehensive benchmarking for all 4 voice conversion pipelines:
1. realtime - ContentVec + HiFiGAN (22kHz, karaoke)
2. quality - CoMoSVC with consistency model (24kHz, studio)
3. quality_seedvc - Seed-VC DiT-CFM (44kHz, SOTA quality)
4. realtime_meanvc - MeanVC streaming (16kHz, low latency)

Metrics collected:
- RTF (Real-Time Factor)
- Latency (chunk and end-to-end)
- GPU Memory (peak, sustained)
- MCD (Mel Cepstral Distortion)
- Speaker Similarity (cosine)

Usage:
    python scripts/performance_validation.py --pipeline all
    python scripts/performance_validation.py --pipeline realtime --audio tests/quality_samples/william_singe_pillowtalk.wav
    python scripts/performance_validation.py --compare --output reports/benchmark_report.md

Track: performance-validation-suite_20260201"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
PROJECT_ROOT = Path(__file__).parent.parent
import numpy
import torch

class PipelineConfig:
    """Configuration for a pipeline benchmark."""

class BenchmarkResult:
    """Results from a single pipeline benchmark run."""

class LatencyBreakdown:
    """Latency breakdown by component."""
PIPELINE_CONFIGS = {'realtime': PipelineConfig(name='Realtime (Conten...

class MetricsCollector:
    """Collect and compute performance metrics."""
    def __init__(self, device: str = 'cuda:0'):
        ...
    def get_gpu_memory_gb(self) -> float:
        """Get current GPU memory usage in GB...."""
        ...
    def get_gpu_peak_memory_gb(self) -> float:
        """Get peak GPU memory usage in GB since last reset...."""
        ...
    def reset_peak_memory(self) -> None:
        """Reset peak memory stats...."""
        ...
# ... (truncated)
============================================================
# scripts/audit_loras.py [SUMMARIZED]
============================================================
"""LoRA Lifecycle Audit Script.

Examines all existing LoRAs to determine training status, quality metrics,
freshness, and retraining recommendations.

Cross-Context Dependencies:
- speaker-diarization_20260130: WavLM embeddings (256-dim)
- training-inference-integration_20260130: AdapterManager
- voice-profile-training_20260124: Profile management

Usage:
    python scripts/audit_loras.py [--json] [--markdown] [--verbose]"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

class LoRAStatus:
    """Status information for a LoRA adapter."""

class AuditSummary:
    """Summary of the LoRA audit."""

class LoRAAuditor:
    """Audits LoRA adapters across voice profiles.

Thresholds (from lora-lifecycle-management track):
- min_samples_for_training: 5
- retrain_new_samples: 3
- freshness_days: 30
- speaker_similarity_min: 0...."""
    def __init__(self, data_dir: Path = Path('data'), verbose: bool = False):
        ...
    def _log(self, msg: str) -> None:
        ...
    def _find_all_profiles(self) -> List[Dict[str, Any]]:
        """Find all voice profiles including diarized ones...."""
        ...
    def _find_adapter(self, profile_id: str) -> Tuple[Optional[Path], str]:
        """Find adapter for profile, checking all adapter types...."""
        ...
    def _count_samples(self, profile: Dict[str, Any]) -> int:
        """Count training samples for a profile...."""
        ...
    def _get_training_timestamp(self, adapter_path: Optional[Path]) -> Optional[datetime]:
        """Get training timestamp from adapter file...."""
        ...
    def _get_quality_metrics(self, profile_id: str) -> Dict[str, Optional[float]]:
        """Get quality metrics for a profile's adapter...."""
        ...
    def audit_profile(self, profile: Dict[str, Any]) -> LoRAStatus:
        """Audit a single profile...."""
        ...
    def audit_all(self) -> Tuple[List[LoRAStatus], AuditSummary]:
        """Audit all profiles and return statuses with summary...."""
# ... (truncated)
============================================================
# scripts/sota_conversion_nvfp4.py [SUMMARIZED]
============================================================
"""SOTA Voice Conversion with nvfp4 quantization for NVIDIA Thor.

Full pipeline: ContentVec → RMVPE → CoMoSVC → BigVGAN
Optimized for Jetson Thor with CUDA 13.0 and JetPack 7.2."""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn
import numpy
import librosa
import soundfile
WILLIAM_PROFILE_ID = '7da05140-1303-40c6-95d9-5b6e2c3624df'
CONOR_PROFILE_ID = '9679a6ec-e6e2-43c4-b64e-1f004fed34f9'
SEPARATED_DIR = 'data/separated'
MODELS_DIR = 'models/pretrained'
OUTPUT_DIR = 'data/conversions'

def print_banner(text: str):
    ...

def print_memory_usage():
    """Print current GPU memory usage...."""
    ...

def quantize_model_nvfp4(model: nn.Module, name: str = 'model') -> nn.Module:
    """Quantize model to nvfp4 (4-bit) for memory efficiency...."""
    ...

class SOTAVoiceConverter:
    """Full SOTA voice conversion pipeline with nvfp4 optimization."""
    def __init__(self, device = None, quantize: bool = True):
        ...
    def _load_contentvec(self):
        """Load and optionally quantize ContentVec encoder...."""
        ...
    def _load_rmvpe(self):
        """Load and optionally quantize RMVPE pitch extractor...."""
        ...
    def _load_vocoder(self):
        """Load and optionally quantize BigVGAN vocoder...."""
        ...
    def extract_content(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract content features using ContentVec...."""
        ...
    def extract_pitch(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract F0 using RMVPE...."""
        ...
    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel spectrogram using BigVGAN...."""
        ...
    def convert_voice(self, source_audio: np.ndarray, source_sr: int, target_embedding: np.ndarray, pitch_shift: float = 0.0) -> tuple:
        """Full voice conversion pipeline...."""
        ...
    def unload_models(self):
        """Unload all models to free GPU memory...."""
# ... (truncated)
============================================================
# scripts/extract_diarized_vocals.py [SUMMARIZED]
============================================================
"""Extract speaker-specific vocals from diarization results.

Reads diarization JSON files and creates separate WAV files for each speaker,
identifying the primary artist (longest total speaking time) as the target."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy
import librosa
import soundfile
from dataclasses import dataclass
from collections import defaultdict

class Segment:

def load_diarization(json_path: Path) -> Tuple[str, List[Segment]]:
    """Load diarization results from JSON...."""
    ...

def identify_primary_speaker(segments: List[Segment]) -> str:
    """Identify the primary speaker (most total speaking time)...."""
    ...

def extract_speaker_audio(audio: np.ndarray, sr: int, segments: List[Segment], speaker: str, fade_ms: float = 10.0) -> np.ndarray:
    """Extract audio for a specific speaker with crossfade...."""
    ...

def process_artist(artist_name: str, diarization_dir: Path, separated_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Process all tracks for an artist...."""
    ...

def main():
    ...
============================================================
# scripts/download_seed_vc_models.py [SUMMARIZED]
============================================================
"""Download Seed-VC pretrained models for QUALITY_PIPELINE.

Downloads:
1. DiT_seed_v2_uvit_whisper_base_f0_44k (SVC model, 200M params)
2. RMVPE pitch extractor
3. BigVGAN vocoder (auto-downloaded)
4. Whisper-small (auto-downloaded)
5. CAMPPlus (already present)

Usage:
    PYTHONNOUSERSITE=1 python scripts/download_seed_vc_models.py"""

import os
import sys
from pathlib import Path
SEED_VC_DIR = Path(__file__).parent.parent / 'models' / 'seed-vc...

def download_models():
    """Download all required Seed-VC models...."""
    ...
============================================================
# scripts/download_pretrained_models.py [SUMMARIZED]
============================================================
"""Download pretrained models for AutoVoice.

Models:
- hubert-soft-35d9f29f.pt (361MB) - HuBERT-Soft feature extractor
- generator_universal.pth.tar (55MB) - HiFiGAN universal vocoder
- sovits5.0_main_1500.pth (184MB) - Main So-VITS model (requires training)"""

import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
import shutil
MODELS_DIR = Path(__file__).parent.parent / 'models' / 'pretrai...
MODELS = {'hubert-soft-35d9f29f.pt': {'url': 'https://githu...

def download_gdrive(file_id: str, dest: Path):
    """Download from Google Drive...."""
    ...

def download_file(url: str, dest: Path, expected_size_mb: int = 0):
    """Download a file with progress...."""
    ...

def verify_model(path: Path) -> bool:
    """Verify a downloaded model is valid (non-zero, loadable)...."""
    ...

def main():
    ...
============================================================
# scripts/youtube_artist_pipeline.py [SUMMARIZED]
============================================================
"""YouTube Artist Training Pipeline.

Downloads videos, separates vocals, runs diarization, and trains LoRA models
for Connor Maynard and William Singe.

Usage:
    python scripts/youtube_artist_pipeline.py --artist conor_maynard
    python scripts/youtube_artist_pipeline.py --artist william_singe
    python scripts/youtube_artist_pipeline.py --stage download --artist conor_maynard
    python scripts/youtube_artist_pipeline.py --stage separate --artist william_singe
    python scripts/youtube_artist_pipeline.py --stage train --artist all"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
import torch
from auto_voice.youtube import download_artist_videos, scrape_artist_channel
from auto_voice.audio.separation import VocalSeparator
ARTIST_PROFILES = {'conor_maynard': {'profile_id': 'c572d02c-c687-4b...

def stage_download(artist_key: str, max_videos: int = 200, max_workers: int = 4):
    """Stage 1: Download audio from YouTube channel...."""
    ...

def stage_separate(artist_key: str, gpu_memory_limit_gb: float = 8.0):
    """Stage 2: Separate vocals from downloaded audio...."""
    ...

def stage_diarize(artist_key: str, max_memory_gb: float = 4.0):
    """Stage 3: Run speaker diarization to identify artist segments...."""
    ...

def stage_train(artist_key: str, epochs: int = 50, lora_rank: int = 16, lora_alpha: int = 32, gradient_checkpointing: bool = True):
    """Stage 4: Train LoRA adapter with OOM protection...."""
    ...

def main():
    ...
============================================================
# scripts/benchmark_memory.py [SUMMARIZED]
============================================================
"""Task 6.3: Benchmark GPU memory usage for both pipelines.

Measures peak GPU memory consumption to verify both pipelines fit in 64GB budget."""

import os
import sys
import gc
from pathlib import Path
import torch
import numpy
import librosa
from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig
from quality_pipeline import QualityVoiceConverter, QualityConfig

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB...."""
    ...

def benchmark_realtime_memory():
    """Benchmark realtime pipeline memory usage...."""
    ...

def benchmark_quality_memory():
    """Benchmark quality pipeline memory usage...."""
    ...

def main():
    ...
============================================================
# scripts/convert_pillowtalk.py [SUMMARIZED]
============================================================
"""Voice conversion: Swap vocals between William Singe and Conor Maynard.

This script:
1. Converts William's vocals to sound like Conor on Conor's instrumental
2. Converts Conor's vocals to sound like William on William's instrumental
3. Runs quality metrics and saves outputs for listening test"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
import numpy
import librosa
import soundfile
WILLIAM_PROFILE_ID = '7da05140-1303-40c6-95d9-5b6e2c3624df'
CONOR_PROFILE_ID = '9679a6ec-e6e2-43c4-b64e-1f004fed34f9'
SEPARATED_DIR = 'data/separated'
MODELS_DIR = 'data/trained_models'
OUTPUT_DIR = 'data/conversions'

def print_banner(text: str):
    """Print a prominent banner...."""
    ...

def load_speaker_embedding(profile_id: str) -> np.ndarray:
    """Load speaker embedding from profile...."""
    ...

def simple_voice_conversion(source_vocals: np.ndarray, source_sr: int, target_embedding: np.ndarray, shift_semitones: float = 0.0) -> np.ndarray:
    """Simple voice conversion using pitch shifting and spectral matching...."""
    ...

def mix_vocals_with_instrumental(vocals: np.ndarray, instrumental: np.ndarray, vocals_sr: int, inst_sr: int, vocal_level: float = 0.8, inst_level: float = 1.0) -> tuple:
    """Mix converted vocals with instrumental track...."""
    ...

def compute_quality_metrics(converted: np.ndarray, reference: np.ndarray, sr: int) -> dict:
    """Compute quality metrics for converted audio...."""
    ...

def run_conversion(source_name: str, target_name: str, source_profile_id: str, target_profile_id: str) -> dict:
    """Run a single voice conversion...."""
    ...

def main():
    """Main conversion script...."""
    ...
============================================================
# scripts/train_hq_lora.py [SUMMARIZED]
============================================================
"""High-Quality LoRA Training for NVIDIA Thor.

Optimized for maximum voice quality while maintaining real-time inference.
Configuration: 768->1024->768 with 6 layers, rank=128 (~1.5M params)

Usage:
    python scripts/train_hq_lora.py --artist conor_maynard --epochs 200
    python scripts/train_hq_lora.py --artist william_singe --epochs 200"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn
import torch.nn.functional
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy
import librosa
import soundfile
ARTIST_PROFILES = {'conor_maynard': {'name': 'Conor Maynard', 'profi...
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DIARIZED_DIR = DATA_DIR / 'diarized_youtube'
SEPARATED_DIR = DATA_DIR / 'separated_youtube'
CHECKPOINTS_DIR = DATA_DIR / 'checkpoints' / 'hq'
OUTPUT_DIR = DATA_DIR / 'trained_models' / 'hq'
HQ_CONFIG = {'input_dim': 768, 'hidden_dim': 1024, 'output_dim...

def print_banner(text: str):
    ...

def print_gpu_memory():
    ...

class LoRALayer(nn.Module):
    """High-Quality LoRA layer with scaled initialization."""
    def __init__(self, in_features: int, out_features: int, rank: int = 128, alpha: float = 256.0, dropout: float = 0.05):
        ...
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        ...
    def get_delta_weight(self) -> torch.Tensor:
        ...

class HQVoiceLoRAAdapter(nn.Module):
    """High-Quality Voice LoRA Adapter for Thor.

Architecture: 768 -> 1024 -> 1024 -> 1024 -> 1024 -> 1024 -> 768
With residual connections and layer normalization."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, output_dim: int = 768, lora_rank: int = 128, lora_alpha: float = 256.0, dropout: float = 0.05, num_layers: int = 6):
        ...
    def forward(self, content: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        ...
# ... (truncated)
============================================================
# scripts/setup_sota_models.py [SUMMARIZED]
============================================================
"""Download and setup SOTA voice conversion models.

Downloads:
- BigVGAN v2 24kHz 100band (NVIDIA)
- ContentVec (lengyue233)
- RMVPE pitch extractor"""

import os
import sys
import logging
from pathlib import Path
import torch
MODELS_DIR = Path(__file__).parent.parent / 'models' / 'pretrai...

def print_banner(text: str):
    ...

def download_bigvgan():
    """Download BigVGAN v2 24kHz 100band from HuggingFace...."""
    ...

def download_contentvec():
    """Download ContentVec encoder from HuggingFace...."""
    ...

def download_rmvpe():
    """Download RMVPE pitch extractor...."""
    ...

def verify_models():
    """Verify all models are loadable...."""
    ...

def main():
    ...
============================================================
# audio/multi_artist_separator.py [SUMMARIZED]
============================================================
"""Multi-Artist Separation and Profile Routing.

Phase 5 of LoRA Lifecycle Management:
- Demucs vocal separation
- Pyannote/WavLM diarization for speaker segments
- WavLM embedding extraction per segment
- Cluster by speaker similarity (0.85 threshold)
- Match clusters to known profiles
- Create profiles for unknown artists

Cross-Context Dependencies:
- speaker-diarization_20260130: WavLM embeddings (256-dim), speaker diarization
- training-inference-integration_20260130: AdapterManager, JobManager
- voice-profile-training_20260124: VoiceProfileStore
- sota-dual-pipeline_20260130: Demucs separation

Ultimate Goal:
Voice-to-voice conversion where one artist sings another's song EXACTLY as the
original artist sang it - pitch correct, singing abilities matched, synced to instrumental."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import torch

class ArtistSegment:
    """A segment of audio belonging to a specific artist."""
    def duration(self) -> float:
        ...

class SeparationResult:
    """Result of multi-artist separation."""

class MultiArtistSeparator:
    """Separates multi-artist tracks and routes to voice profiles.

Pipeline:
1. Demucs vocal/instrumental separation
2. WavLM speaker diarization
3. Cluster embeddings by similarity
4. Match to existing pro..."""
    def __init__(self, profiles_dir: Path = ..., device: str = 'cuda', auto_create_profiles: bool = True, auto_queue_training: bool = True):
        """Initialize the multi-artist separator...."""
        ...
    def _load_separator(self):
        """Lazy load Demucs vocal separator...."""
        ...
    def _load_diarizer(self):
        """Lazy load speaker diarizer...."""
        ...
    def _load_identifier(self):
        """Lazy load voice identifier...."""
        ...
    def _load_job_manager(self):
        """Lazy load training job manager...."""
        ...
    def separate_vocals(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
# ... (truncated)
============================================================
# audio/diarization_extractor.py [SUMMARIZED]
============================================================
"""Diarization-based speaker extraction for multi-artist vocal tracks.

This module extracts speaker-isolated vocal tracks from diarized audio:
- Each speaker gets a FULL-LENGTH track where they are audible and others are SILENCED
- Automatically creates voice profiles for each detected speaker
- Enables per-artist voice conversion with later remixing

Workflow:
1. Load diarization JSON (speaker segments with timestamps)
2. Load separated vocals WAV
3. For EACH speaker detected:
   - Create full-length track with only that speaker audible
   - Create/update voice profile for that speaker
   - Save to profile's training data directory
4. Expose via web interface for training"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import uuid
import numpy
import librosa
import soundfile

class SpeakerSegment:
    """A segment of audio belonging to a single speaker."""
    def duration(self) -> float:
        ...

class ExtractionResult:
    """Result of extracting speaker-isolated tracks."""

class SpeakerExtractionInfo:
    """Information about a single speaker's extraction."""

class DiarizationExtractor:
    """Extract speaker-isolated vocal tracks from diarized audio.

For each detected speaker, creates a full-length track where:
- That speaker's segments are audible
- All other speakers are silenced (zero ..."""
    def __init__(self, fade_ms: float = 10.0, min_segment_duration: float = 0.5, profiles_dir: Optional[Path] = None, training_vocals_dir: Optional[Path] = None):
        """Initialize the extractor...."""
        ...
    def load_diarization(self, json_path: Path) -> Tuple[str, List[SpeakerSegment]]:
        """Load diarization results from JSON file...."""
        ...
    def get_speaker_durations(self, segments: List[SpeakerSegment]) -> Dict[str, float]:
        """Calculate total speaking duration for each speaker...."""
        ...
    def identify_primary_speaker(self, segments: List[SpeakerSegment]) -> Optional[str]:
        """Identify the primary speaker (longest total speaking time)...."""
        ...
    def extract_speaker_track(self, audio: np.ndarray, sr: int, segments: List[SpeakerSegment], target_speaker: str) -> np.ndarray:
        """Create a full-length track with only target speaker audible...."""
        ...
    def get_or_create_profile(self, artist_name: str, speaker_id: str, is_primary: bool) -> str:
# ... (truncated)
============================================================
# audio/technique_detector.py [SUMMARIZED]
============================================================
"""Vocal technique detection for singing voice analysis.

Implements detection of singing techniques:
- Vibrato: periodic pitch modulation (typically 4-7 Hz, ±20-50 cents)
- Melisma: rapid pitch transitions across multiple notes

Phase 5: Advanced Vocal Technique Preservation
- Task 5.2: Vibrato detector (frequency modulation analysis)
- Task 5.4: Melisma detector (rapid pitch transitions)
- Task 5.6: Technique-aware pitch extraction"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy
import torch

class VibratoSegment:
    """A segment of audio containing vibrato."""

class VibratoResult:
    """Result of vibrato detection."""

class MelismaSegment:
    """A segment of audio containing melisma/vocal run."""

class MelismaResult:
    """Result of melisma detection."""

class PitchExtractionResult:
    """Result of technique-aware pitch extraction."""

class TechniqueFlags:
    """Flags for passing technique information through the pipeline."""
    def has_vibrato(self) -> bool:
        """Check if any frames are flagged as vibrato...."""
        ...
    def has_melisma(self) -> bool:
        """Check if any frames are flagged as melisma...."""
        ...
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization...."""
        ...
    def from_dict(cls, data: Dict[str, Any]) -> 'TechniqueFlags':
        """Create from dictionary...."""
        ...

class VibratoDetector:
    """Detect vibrato in audio using frequency modulation analysis.

Vibrato is characterized by:
- Periodic pitch modulation at 4-7 Hz
- Depth of ±20-50 cents (sometimes up to 100 cents)
- Consistent rate a..."""
    def __init__(self, sample_rate: int = 16000, frame_size: int = 512, hop_size: int = 128, min_rate: float = 4.0, max_rate: float = 8.0, min_depth_cents: float = 15.0):
        """Initialize vibrato detector...."""
        ...
    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 contour from audio using autocorrelation...."""
        ...
# ... (truncated)
============================================================
# audio/youtube_metadata.py [SUMMARIZED]
============================================================
"""YouTube metadata parsing and fetching for featured artist detection.

This module parses YouTube video titles and descriptions to identify:
- Main artist performing the song
- Featured/collaborating artists (ft., feat., vs., with, &, x patterns)
- Cover song detection and original artist identification

It also provides yt-dlp integration for fetching metadata from YouTube.

Usage:
    from auto_voice.audio.youtube_metadata import YouTubeMetadataFetcher

    fetcher = YouTubeMetadataFetcher()
    metadata = fetcher.fetch_metadata("dQw4w9WgXcQ")
    featured = parse_featured_artists(metadata.title)"""

import re
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
FEATURED_PATTERNS = ['\\bft\\.?\\s+([^(\\[\\]|,&-]+?)(?:\\s*[(\\[\\]|,...
PAREN_FEATURED_PATTERNS = ['\\(ft\\.?\\s+([^)]+)\\)', '\\(feat\\.?\\s+([^)]+...
EXCLUDE_PATTERNS = ['\\bprod\\.?\\s+(?:by\\s+)?', '\\bproduced\\s+by\...
COVER_PATTERNS = ['\\(([^)]+)\\s+cover\\)', '\\bcover\\s+of\\s+([^(...
DESCRIPTION_FEATURED_PATTERNS = ['featuring\\s+(?:vocals?\\s+by\\s+)?([^.!\\n]+)',...

def _clean_artist_name(name: str) -> str:
    """Clean and normalize an artist name...."""
    ...

def _split_multiple_artists(artist_str: str) -> List[str]:
    """Split a string containing multiple artists separated by , & and...."""
    ...

def _is_producer_credit(text: str) -> bool:
    """Check if text is a producer credit rather than a featured artist...."""
    ...

def parse_featured_artists(title: str, description: Optional[str] = None) -> List[str]:
    """Parse featured artists from YouTube video title and description...."""
    ...

def extract_main_artist(title: str) -> Optional[str]:
    """Extract the main artist from a YouTube video title...."""
    ...

def detect_cover_song(title: str, description: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Detect if a video is a cover song and identify the original artist...."""
    ...

def parse_youtube_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Parse YouTube video metadata to extract artist information...."""
    ...

class VideoMetadata:
    """YouTube video metadata."""

# ... (truncated)
============================================================
# audio/file_organizer.py [SUMMARIZED]
============================================================
"""File organizer for speaker-identified vocal files.

This module re-organizes extracted vocal files from UUID-based directories
to named artist directories after speaker identification and clustering.

Usage:
    from auto_voice.audio.file_organizer import organize_by_identified_artist

    stats = organize_by_identified_artist()"""

import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

class FileOrganizer:
    """Re-organize vocal files by identified artist name."""
    def __init__(self, training_vocals_dir: Optional[Path] = None, voice_profiles_dir: Optional[Path] = None):
        """Initialize the file organizer...."""
        ...
    def get_cluster_assignments(self) -> Dict[str, Dict[str, Any]]:
        """Get cluster assignments from database...."""
        ...
    def find_profile_for_tracks(self, track_ids: List[str], speaker_id: str) -> Optional[str]:
        """Find the profile UUID used for a specific speaker in tracks...."""
        ...
    def normalize_artist_name(self, name: str) -> str:
        """Normalize artist name for use as directory name...."""
        ...
    def organize_by_cluster(self, dry_run: bool = True) -> Dict[str, Any]:
        """Re-organize files based on cluster assignments...."""
        ...
    def create_speaker_profiles_json(self, artist_name: str, dry_run: bool = True) -> Dict[str, Any]:
        """Create speaker_profiles.json for an artist directory...."""
        ...
    def generate_all_profiles(self, dry_run: bool = True) -> Dict[str, Any]:
        """Generate speaker_profiles.json for all artist directories...."""
        ...

def organize_by_identified_artist(dry_run: bool = True) -> Dict[str, Any]:
    """Run full organization pipeline...."""
    ...
============================================================
# audio/effects.py [SUMMARIZED]
============================================================
"""Audio effects - pitch shifting, volume adjustment."""

import logging
import numpy

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones...."""
    ...

def volume_adjust(audio: np.ndarray, gain: float) -> np.ndarray:
    """Adjust audio volume...."""
    ...

def fade_in(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """Apply linear fade-in...."""
    ...

def fade_out(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """Apply linear fade-out...."""
    ...
============================================================
# audio/__init__.py [SUMMARIZED]
============================================================
"""Audio processing utilities."""

============================================================
# audio/speaker_diarization.py [SUMMARIZED]
============================================================
"""Speaker diarization for multi-speaker audio segmentation.

This module provides speaker diarization capabilities using:
- WavLM for speaker embeddings (512-dim from wavlm-base-sv)
- Agglomerative clustering for speaker segmentation
- Energy-based Voice Activity Detection (VAD)

The pipeline identifies different speakers in audio and extracts their segments."""

import gc
import logging
import psutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import numpy
import torch
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

def get_available_memory_gb() -> float:
    """Get available system memory in GB...."""
    ...

def get_gpu_memory_gb() -> Tuple[float, float]:
    """Get GPU memory (used, total) in GB. Returns (0, 0) if no GPU...."""
    ...

class SpeakerSegment:
    """A segment of audio belonging to a single speaker."""
    def duration(self) -> float:
        """Duration of the segment in seconds...."""
        ...

class DiarizationResult:
    """Complete diarization result for an audio file."""
    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker...."""
        ...
    def get_speaker_total_duration(self, speaker_id: str) -> float:
        """Get total speaking duration for a speaker...."""
        ...
    def get_all_speaker_ids(self) -> List[str]:
        """Get list of all unique speaker IDs...."""
        ...

class SpeakerDiarizer:
    """Speaker diarization using WavLM embeddings and clustering.

This provides a simpler alternative to pyannote.audio that works with
the latest torchaudio versions on Jetson platforms."""
    def __init__(self, device: Optional[str] = None, model_name: str = ..., min_segment_duration: float = 0.5, max_speakers: int = 10, max_memory_gb: Optional[float] = None, chunk_duration_sec: float = 60.0):
        """Initialize the speaker diarizer...."""
        ...
    def _load_model(self):
        """Lazy load the speaker embedding model...."""
        ...
    def _check_memory(self, warn_threshold: float = 0.9) -> bool:
# ... (truncated)
============================================================
# audio/processor.py [SUMMARIZED]
============================================================
"""Audio processing utilities."""

import logging
from typing import Optional, Tuple
import numpy

class AudioProcessor:
    """Core audio processing operations."""
    def __init__(self, sample_rate: int = 22050):
        ...
    def load(self, path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file...."""
        ...
    def save(self, path: str, audio: np.ndarray, sr: Optional[int] = None):
        """Save audio to file...."""
        ...
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate...."""
        ...
    def normalize(self, audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
        """Normalize audio to peak amplitude...."""
        ...
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Trim silence from beginning and end...."""
        ...
    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono...."""
        ...
============================================================
# audio/speaker_matcher.py [SUMMARIZED]
============================================================
"""Cross-track speaker matching and clustering for AutoVoice.

This module provides:
- Speaker embedding extraction using WavLM
- Cross-track speaker clustering using cosine similarity
- Auto-matching clusters to featured artist names from metadata

Usage:
    from auto_voice.audio.speaker_matcher import SpeakerMatcher

    matcher = SpeakerMatcher()
    matcher.extract_embeddings_for_artist('conor_maynard')
    clusters = matcher.cluster_speakers(threshold=0.85)
    matcher.auto_match_clusters_to_artists()"""

import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy

class SpeakerMatcher:
    """Cross-track speaker matching and clustering.

Uses WavLM-based embeddings to identify the same speaker across
different tracks, enabling consistent voice profile assignment."""
    def __init__(self, similarity_threshold: float = 0.85, min_cluster_duration: float = 30.0, device: str = 'cuda'):
        """Initialize the speaker matcher...."""
        ...
    def _get_encoder(self):
        """Lazy-load the WavLM encoder...."""
        ...
    def extract_embedding_from_audio(self, audio_path: Path, start_sec: Optional[float] = None, end_sec: Optional[float] = None) -> np.ndarray:
        """Extract speaker embedding from audio file or segment...."""
        ...
    def extract_embeddings_for_artist(self, artist_name: str, separated_dir: Optional[Path] = None, diarized_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Extract and store embeddings for all speakers in an artist's tracks...."""
        ...
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings...."""
        ...
    def cluster_speakers(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Cluster all unclustered speaker embeddings...."""
        ...
    def auto_match_clusters_to_artists(self) -> Dict[str, Any]:
        """Automatically match clusters to featured artist names from metadata...."""
        ...
    def get_cluster_sample_audio(self, cluster_id: str, max_duration: float = 10.0) -> Tuple[np.ndarray, int]:
        """Get a sample audio clip for a speaker cluster...."""
        ...

def run_speaker_matching(artists: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run full speaker matching pipeline for specified artists...."""
    ...
============================================================
# audio/separation.py [SUMMARIZED]
============================================================
"""Vocal/instrumental separation using Demucs HTDemucs model.

No fallback behavior - raises RuntimeError if Demucs is unavailable."""

import logging
from typing import Dict, Optional
import numpy
import torch

class VocalSeparator:
    """Separates vocals from instrumental using Demucs HTDemucs.

Uses the pretrained HTDemucs model for high-quality source separation.
Raises RuntimeError if Demucs cannot be loaded - no silent fallback."""
    def __init__(self, device = None, model_name: str = 'htdemucs', segment: Optional[float] = None):
        """Initialize VocalSeparator...."""
        ...
    def _load_model(self):
        """Lazy-load Demucs model...."""
        ...
    def model_sample_rate(self) -> int:
        """Return the model's expected sample rate...."""
        ...
    def sources(self):
        """Return list of source names the model separates...."""
        ...
    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate audio into vocals and instrumental...."""
        ...
============================================================
# audio/separator.py [SUMMARIZED]
============================================================
"""Mel-Band RoFormer vocal separator for source separation.

Implements a simplified Mel-Band RoFormer architecture for vocal/
instrumental separation, based on the SDX'23 winning approach.

Key design choices:
- Mel-scale frequency band splitting (non-uniform bandwidth)
- RoPE (Rotary Position Embeddings) in transformer layers
- Complex-valued mask estimation for phase-aware separation
- 44.1kHz processing with 2048 n_fft, 512 hop
- No fallback: raises RuntimeError on failure"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy
import torch
import torch.nn
import torch.nn.functional

def compute_mel_band_splits(n_fft: int, sample_rate: int, n_bands: int = 32) -> list:
    """Compute mel-scale frequency band boundaries...."""
    ...

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for temporal modeling."""
    def __init__(self, dim: int, max_len: int = 8192):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor...."""
        ...

class BandTransformerBlock(nn.Module):
    """Transformer block for processing a frequency band."""
    def __init__(self, dim: int, n_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process band features...."""
        ...

class MelBandRoFormer(nn.Module):
    """Mel-Band RoFormer for vocal/instrumental separation.

Architecture (SDX'23):
- Input: Complex STFT at 44.1kHz
- Band splitting: 32 mel-scale frequency bands
- Per-band processing: Transformer with RoP..."""
    def __init__(self, pretrained: Optional[str] = None, device: Optional[torch.device] = None, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 32, hidden_dim: int = 128, n_layers: int = 6, n_heads: int = 4):
        ...
    def _load_pretrained(self, path: str):
        """Load pretrained weights...."""
        ...
    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute STFT...."""
        ...
    def _istft(self, stft: torch.Tensor, length: int) -> torch.Tensor:
        """Compute inverse STFT...."""
        ...
    def _process_bands(self, stft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process STFT through band transformer...."""
# ... (truncated)
============================================================
# audio/youtube_downloader.py [SUMMARIZED]
============================================================
"""YouTube audio downloader with metadata extraction and diarization integration.

Uses yt-dlp to download audio from YouTube videos and extracts metadata
for featured artist detection."""

import logging
import os
import subprocess
import tempfile
import json
import uuid
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from youtube_metadata import parse_youtube_metadata, parse_featured_artists

def _find_ytdlp() -> str:
    """Find yt-dlp executable, checking common locations...."""
    ...

class YouTubeDownloadResult:
    """Result of a YouTube download operation."""

class YouTubeDownloader:
    """Downloads audio from YouTube videos with metadata extraction."""
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the downloader...."""
        ...
    def download(self, url: str, output_filename: Optional[str] = None, audio_format: str = 'wav', sample_rate: int = 44100) -> YouTubeDownloadResult:
        """Download audio from a YouTube video...."""
        ...
    def get_video_info(self, url: str) -> YouTubeDownloadResult:
        """Get video information without downloading...."""
        ...
    def _get_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch video metadata using yt-dlp...."""
        ...
    def _download_audio(self, url: str, output_path: str, audio_format: str, sample_rate: int) -> bool:
        """Download audio using yt-dlp...."""
        ...
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string for use as a filename...."""
        ...

def get_downloader(output_dir: Optional[str] = None) -> YouTubeDownloader:
    """Get or create a YouTubeDownloader instance...."""
    ...
============================================================
# audio/training_filter.py [SUMMARIZED]
============================================================
"""Training data filtering for speaker-specific audio extraction.

This module provides functionality to filter training audio to only include
segments from a target speaker, based on diarization results and profile matching."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy
import torch
from scipy.io import wavfile
from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerDiarizer, SpeakerSegment, compute_speaker_similarity, match_speaker_to_profile

class TrainingDataFilter:
    """Filter training audio to extract only target speaker vocals."""
    def __init__(self, diarizer: Optional[SpeakerDiarizer] = None, device: Optional[str] = None):
        """Initialize the training data filter...."""
        ...
    def diarizer(self) -> SpeakerDiarizer:
        """Lazy-load the diarizer...."""
        ...
    def filter_training_audio(self, audio_path: Union[str, Path], target_embedding: np.ndarray, output_path: Optional[Union[str, Path]] = None, similarity_threshold: float = 0.7, min_segment_duration: float = 0.5, diarization_result: Optional[DiarizationResult] = None) -> Tuple[Path, Dict]:
        """Filter audio to only include segments matching target speaker...."""
        ...
    def filter_with_profile_matching(self, audio_path: Union[str, Path], profile_embeddings: Dict[str, np.ndarray], target_profile_id: str, output_path: Optional[Union[str, Path]] = None, **kwargs) -> Tuple[Path, Dict]:
        """Filter audio to match a specific profile from a set of profiles...."""
        ...
    def auto_split_by_speakers(self, audio_path: Union[str, Path], output_dir: Union[str, Path], diarization_result: Optional[DiarizationResult] = None, min_segment_duration: float = 0.5) -> Dict[str, Tuple[Path, float]]:
        """Split audio into separate files per detected speaker...."""
        ...

def filter_training_audio(audio_path: Union[str, Path], target_embedding: np.ndarray, output_path: Optional[Union[str, Path]] = None, **kwargs) -> Tuple[Path, Dict]:
    """Convenience function for filtering training audio...."""
    ...
============================================================
# audio/augmentation.py [SUMMARIZED]
============================================================
"""Data augmentation pipeline for training.

Provides pitch shifting, time stretching, and EQ augmentation
to increase effective training data diversity."""

import logging
from typing import Dict, Optional
import numpy

class AugmentationPipeline:
    """Configurable audio augmentation pipeline for training.

Each augmentation has an independent probability of being applied.
Multiple augmentations can be applied to the same sample.

Args:
    pitch_sh..."""
    def __init__(self, pitch_shift_prob: float = 0.5, pitch_shift_range: float = 2.0, time_stretch_prob: float = 0.3, time_stretch_range: float = 0.1, eq_prob: float = 0.3, eq_bands: int = 3, eq_gain_range: float = 6.0):
        ...
    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random augmentations to audio...."""
        ...
    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random pitch shift within ±pitch_shift_range semitones...."""
        ...
    def _time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random time stretch within ±time_stretch_range...."""
        ...
    def _eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random bandpass EQ emphasis/attenuation...."""
        ...
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

================================================================================
