# AutoVoice - SOTA Singing Voice Conversion

## Platform Target
- **GPU**: NVIDIA Thor (Blackwell, SM 11.0, Compute Capability 11.0)
- **CUDA**: 13.0 (V13.0.48)
- **JetPack**: 7.2 (R38.4.0)
- **Architecture**: aarch64
- **Python**: 3.12.12
- **PyTorch**: 2.11.0.dev20260113+cu130
- **Conda Env**: autovoice-thor

## Project Description
GPU-accelerated SOTA singing voice conversion system. Converts any song to a target voice
while preserving pitch, timing, and expression. Architecture based on So-VITS-SVC with
Amphion CoMoSVC-inspired improvements.

## Current State (2026-01-23)
- 8-phase no-fallback implementation COMPLETE
- 231 model/inference/training tests passing
- ModelManager orchestrates: HuBERT→content, pyin→pitch, mel-stats→speaker, SoVitsSvc→mel, HiFiGAN→audio
- Research: 14 papers analyzed, Amphion architecture studied
- Beads: 11 tasks created (3 P1, 4 P2, 2 P3, 1 epic)

## Environment Setup
```bash
conda activate autovoice-thor
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST="11.0"
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python
PYTEST="$PYTHON -m pytest"
```

## Beads Task Management
```bash
# CRITICAL: Repo on CIFS mount. Always use --no-daemon.
BD="bd --no-daemon --db /home/kp/.beads-local/autovoice/beads.db"
$BD list                                    # Show all tasks
$BD update AV-xxx --status in_progress      # Start work
$BD close AV-xxx --force --reason "done"    # Complete task
```

## Cross-Compaction Context Protocol
**MANDATORY before each task:**
```python
TaskCreate(
    subject="[AV-xxx] Working on: [description]",
    description="Context: [what was done before, what's needed, key files]\n"
                "Beads: $BD list output\n"
                "Next: [specific next steps]",
    activeForm="[Present participle]"
)
```
**On completion:** TaskUpdate(status="completed")

This ensures ralph-loop maintains context across automatic summarization.

---

## Ralph Orchestrator Loop

### Loop Structure
```
WHILE tasks remain:
  1. $BD list → find highest priority open task
  2. $BD update AV-xxx --status in_progress
  3. TaskCreate with full context
  4. IF task needs research:
     a. Search arxiv for latest papers (2024-2026)
     b. Read relevant Amphion code at /home/kp/repo2/Amphion/
     c. Save findings to academic-research/
  5. Implement the feature/fix
  6. Run tests: $PYTEST tests/ -x --tb=short -q
  7. Fix any failures
  8. $BD close AV-xxx --force --reason "[summary]"
  9. TaskUpdate(status="completed")
  10. Save context to Serena memory (mcp__serena__write_memory)
```

### Phase 1: Stability (P1 - Do First)
| ID | Task | Key Files |
|----|------|-----------|
| AV-gmv | Fix remaining test failures | tests/*.py |
| AV-ge9 | Download pretrained weights | models/pretrained/, scripts/ |
| AV-3lx | SSIM loss for training | models/so_vits_svc.py |
| AV-1q3 | Mel-quantized F0 + UV embedding | models/encoder.py |

### Phase 2: Quality (P2)
| ID | Task | Key Files |
|----|------|-----------|
| AV-87t | ContentVec features | models/encoder.py |
| AV-7a5 | BigVGAN vocoder | models/vocoder.py |
| AV-2u7 | Data augmentation | audio/augmentation.py (NEW) |
| AV-1il | Vocal separation (Demucs) | audio/separation.py |

### Phase 3: Speed (P3)
| ID | Task | Key Files |
|----|------|-----------|
| AV-0gj | Conformer encoder | models/conformer.py (NEW) |
| AV-3ka | Consistency distillation | models/consistency.py (NEW) |

---

## Architecture

### Source Structure
```
src/auto_voice/
  inference/
    model_manager.py         # Central inference orchestrator
    singing_conversion_pipeline.py
    voice_cloner.py          # Mel-stats speaker embedding
    realtime_voice_conversion_pipeline.py
  models/
    encoder.py               # ContentEncoder, PitchEncoder, HuBERTSoft
    vocoder.py               # HiFiGANVocoder, HiFiGANGenerator
    so_vits_svc.py           # SoVitsSvc, FlowDecoder
  audio/
    processor.py             # AudioProcessor
    effects.py               # Pitch shift, volume
    separation.py            # Vocal separation
  evaluation/metrics.py      # Pitch RMSE, speaker similarity
  web/
    app.py                   # create_app() factory
    api.py                   # REST endpoints
    job_manager.py           # Async job processing
  training/trainer.py        # Trainer with real encoder features
  gpu/                       # GPU memory management
  monitoring/                # Prometheus metrics
  storage/                   # Voice profiles
```

### Research Resources
```
academic-research/
  bibliography.md            # 14 papers with arxiv IDs and relevance
  amphion-analysis.md        # Amphion CoMoSVC deep-dive
/home/kp/repo2/Amphion/      # Reference implementation
  models/svc/comosvc/        # Consistency model SVC
  modules/encoder/           # Condition encoder (multi-modal)
  modules/diffusion/         # BiDilConv diffusion decoder
  utils/f0.py                # F0 extraction utilities
```

## API Contracts (from existing api.py)
1. `VoiceCloner.create_voice_profile(audio, user_id)` → dict
2. `SingingConversionPipeline.convert_song(song_path, target_profile_id, ...)` → dict
3. `JobManager.create_job(file_path, profile_id, settings)` → job_id
4. `GET /health` → 200 with component status
5. `POST /api/v1/voice/clone` → voice profile
6. `POST /api/v1/convert/song` → conversion job

## Test Commands
```bash
# Quick validation
$PYTEST tests/ -x --tb=short -q
# Full suite
$PYTEST tests/ -v --tb=short
# Specific areas
$PYTEST tests/test_models.py tests/test_model_manager.py -v
$PYTEST tests/ -m smoke -v
$PYTEST tests/ -m "not slow" -v
```

## Success Criteria
- [ ] >95% test pass rate across all test files
- [ ] Training loss decreases over 50+ epochs with real features
- [ ] Different speakers produce measurably different outputs
- [ ] Pretrained weights loaded (HuBERT, vocoder)
- [ ] Realtime pipeline <50ms per chunk
- [ ] Vocal separation functional
- [ ] SSIM loss improves perceptual quality
- [ ] Mel-quantized F0 preserves pitch accuracy
- [ ] ContentVec improves speaker disentanglement
- [ ] BigVGAN improves audio quality vs HiFiGAN
