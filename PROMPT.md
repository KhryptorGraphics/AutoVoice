# AutoVoice Master Development Orchestrator

## Sprint: Complete SOTA Voice Conversion System + Training Workflow

### Ultimate Goal
William Singe and Conor Maynard voice swaps: each artist singing in the style and talent of the other on each other's instrumental tracks.

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
