# Transfer Plan: AutoVoice Network Mount to Local Storage

## Task

Comprehensive project context for continuing development after moving from network mount to local storage on Jetson Thor.

---

## 1. Environment Setup

```bash
# Conda environment
CONDA_ENV=autovoice-thor
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python

# CUDA configuration (Jetson Thor)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="11.0"
export PYTHONNOUSERSITE=1

# Run any command
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON <script.py>
```

---

## 2. Hardcoded Paths to Update

After moving to new location, update `PROJECT_ROOT` in these files:

| File | Lines | Pattern |
|------|-------|---------|
| `src/auto_voice/web/api.py` | 1500, 1532 | `/home/kp/repo2/autovoice` |
| `src/auto_voice/training/job_manager.py` | 627, 647 | `/home/kp/repo2/autovoice` |
| `src/auto_voice/audio/youtube_downloader.py` | 32-33 | yt-dlp paths |
| `CLAUDE.md` | 14 | Python path |
| `PROMPT.md` | 104-105 | cd and Python paths |
| `ORCHESTRATOR.md` | 58, 130 | Project path |
| `scripts/ralph/ralph.sh` | 51 | Python path |
| `scripts/ralph/progress.txt` | 6-7 | Conda env paths |
| `scripts/ralph/resume-prompt.md` | 32 | cd path |

**Find all occurrences:**
```bash
grep -rn "/home/kp/repo2/autovoice" --include="*.py" --include="*.md" --include="*.sh"
```

---

## 3. Active Work in Progress

### Conductor Tracks (conductor/tracks.md)

| Status | Track ID | Phase | Next Tasks |
|--------|----------|-------|------------|
| [~] | training-inference-integration_20260130 | Phase 2 | Task 2.3-2.5, then Phase 3-6 |
| [~] | sota-dual-pipeline_20260130 | Phase 1 | Task 1.2-1.7, then Phase 2-6 |
| [~] | frontend-lora-integration_20260130 | Phase 1 | Task 1.5-1.6 (HQ LoRA training) |
| [ ] | youtube-artist-training_20260130 | Not started | All phases |
| [x] | speaker-diarization_20260130 | Complete | N/A |

### Beads Tasks

```bash
bd list  # Current open tasks
```

- `AV-k3u` [P1 epic] - Frontend LoRA Integration Epic (open)
- `AV-28i` [P1 task] - Train HQ LoRA adapters (in_progress)

### Uncommitted Git Changes

```
Modified:
- src/auto_voice/audio/separation.py
- src/auto_voice/audio/speaker_diarization.py
- src/auto_voice/inference/*.py
- src/auto_voice/training/*.py
- src/auto_voice/web/api.py
- frontend/src/components/*.tsx
- frontend/src/pages/*.tsx
- tests/*.py

Untracked:
- scripts/train_*.py (training scripts)
- scripts/quality_pipeline.py
- scripts/realtime_pipeline.py
- src/auto_voice/youtube/ (new module)
- models/ (local model files)
- test_audio/ (test fixtures)
```

---

## 4. Data Inventory

### Directory Sizes
```
models/           1.5GB   (pretrained weights)
data/            18GB    (all runtime data)
frontend/node_modules/  148MB
```

### data/ Subdirectories
```
data/
├── checkpoints/          # Training checkpoints
├── cipher-sessions.db    # Session database
├── conversions/          # Converted audio output
├── diarized_youtube/     # Diarized YouTube audio
├── features_cache/       # Cached audio features
├── samples/              # Raw audio samples
├── separated/            # Demucs-separated tracks
├── separated_youtube/    # YouTube separated tracks
├── trained_models/       # Trained LoRA adapters
│   ├── nvfp4/           # NvFP4 quantized adapters
│   └── hq/              # High-quality adapters (empty)
├── voice_profiles/       # Profile JSON + embeddings
├── youtube_audio/        # Downloaded YouTube audio
├── youtube_downloads/    # Raw YouTube downloads
└── youtube_metadata/     # Video metadata cache
```

### models/pretrained/
```
bigvgan_generator.pt      450MB
generator_universal.pth.tar 56MB
hubert-soft-35d9f29f.pt   378MB
rmvpe.pt                  181MB
content-vec-best/         (directory)
```

---

## 5. Voice Profiles

| Name | Profile ID | Speaker Embedding | nvfp4 Adapter | HQ Adapter |
|------|------------|-------------------|---------------|------------|
| William Singe | `7da05140-1303-40c6-95d9-5b6e2c3624df` | ✓ | ✓ | - |
| Connor | `c572d02c-c687-4bed-8676-6ad253cf1c91` | ✓ | ✓ | - |

**Profile Details:**
- William: 195s audio, 51-962Hz range, mean 356Hz
- Connor: 207s audio, 54-945Hz range, mean 255Hz

---

## 6. Dependencies

### Python (requirements.txt)
```
torch>=2.0, torchaudio>=2.0
flask>=3.0, flask-socketio>=5.3
transformers>=4.30, resemblyzer>=0.1.3, demucs>=4.0
sqlalchemy>=2.0, alembic>=1.13
prometheus_client>=0.19
```

### Frontend (package.json)
```
react@18.2, react-router-dom@6.20
@tanstack/react-query@5.0
socket.io-client@4.7, wavesurfer.js@7.4
tailwindcss@3.4, vite@5.0, typescript@5.3
```

---

## 7. Post-Move Checklist

```bash
# 1. Update all hardcoded paths
grep -rn "/home/kp/repo2/autovoice" --include="*.py" --include="*.md" | \
  xargs sed -i 's|/home/kp/repo2/autovoice|<NEW_PATH>|g'

# 2. Verify conda environment
conda activate autovoice-thor
which python  # Should be conda env

# 3. Test basic import
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -c "import auto_voice; print('OK')"

# 4. Run quick tests
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -m pytest tests/ -x --tb=short -q -m "not slow"

# 5. Verify models load
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -c "
from auto_voice.inference.voice_cloner import VoiceCloner
vc = VoiceCloner()
print('Voice cloner OK')
"

# 6. Start server
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON main.py --host 0.0.0.0 --port 5000

# 7. Verify frontend builds
cd frontend && npm install && npm run build

# 8. Check beads
bd list
bd stats
```

---

## 8. Resume Development

### Priority 1: Complete HQ LoRA Training
```bash
# Continue from frontend-lora-integration track
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON scripts/train_hq_lora_optimized.py
```

### Priority 2: Training-Inference Integration
```bash
# Phase 3-6 of training-inference-integration track
# See conductor/tracks/training-inference-integration_20260130/plan.md
```

### Priority 3: SOTA Dual-Pipeline
```bash
# Phase 2+ of sota-dual-pipeline track
# See conductor/tracks/sota-dual-pipeline_20260130/plan.md
```

---

## 9. Critical Coding Rules (from CLAUDE.md)

- **No fallback behavior:** Always raise RuntimeError, never pass through silently
- **Speaker embedding:** mel-statistics (mean+std of 128 mels = 256-dim, L2-normalized)
- **Frame alignment:** F.interpolate(transpose(1,2), size=target) for content/pitch
- **PYTHONNOUSERSITE=1** always set for python commands
- **Tests must verify real behavior** (shapes, non-NaN, correct types)
- **Atomic commits:** one feature per commit, always run full test suite first
