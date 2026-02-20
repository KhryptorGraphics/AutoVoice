# Coverage Gap Analysis & Test Creation Roadmap to 95%

**Date:** 2026-02-02
**Agent:** Test Automation Engineer (Coverage Gap Analyzer)
**Status:** 🔴 **ACTIVE** - Roadmap for 63% → 95% coverage

---

## Executive Summary

### Current State
- **Overall Coverage:** 63% (9,467 / 15,063 lines)
- **Target Coverage:** 95% (14,310 / 15,063 lines)
- **Gap:** 32 percentage points
- **Lines to Cover:** ~4,843 additional lines

### Effort Estimate
- **Total Tests to Create:** ~1,268 tests
- **Productivity Assumption:** 10 tests/day/agent
- **Timeline:**
  - With 1 agent: ~126 days
  - With 3 agents: ~42 days
  - With 6 agents: ~21 days

---

## Coverage Targets by Category

| Category | Current % | Target % | Gap % | Gap Lines | Tests Needed | Priority |
|----------|-----------|----------|-------|-----------|--------------|----------|
| **Inference** | 68% | **95%** | 27% | 793 | 196 | P0 |
| **Audio** | 55% | **90%** | 35% | 1,262 | 312 | P0 |
| **Web API** | 60% | **90%** | 30% | 853 | 212 | P0/P1 |
| **Evaluation** | ~15% | **90%** | 75% | 765 | 191 | P0/P1 |
| **Database** | 87% | **95%** | 8% | 21 | 7 | ✅ P2 |
| **Storage** | 78% | **95%** | 17% | 64 | 16 | P1 |
| **Models** | ~50% | **90%** | 40% | 435 | 108 | P1 |
| **Training** | ~45% | **90%** | 45% | 678 | 169 | P1 |
| **Export/TRT** | 24% | **85%** | 61% | 103 | 25 | P1 |
| **Monitoring** | 33% | **90%** | 57% | 129 | 32 | P1 |

---

## Priority P0 - CRITICAL (Inference Core & Audio Core)

**Target:** Cover critical inference and audio modules blocking 80% overall coverage
**Total:** 10 modules, 1,601 lines, ~396 tests
**Timeline:** 39 days (1 agent) → 6.5 days (6 agents)

### P0.1 - Inference Modules (Target: 95%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `inference.voice_identifier` | 206 | 0% | 95% | 195 | 48 | AV-xxx |
| `inference.mean_flow_decoder` | 101 | 0% | 95% | 95 | 23 | AV-xxx |
| `inference.trt_pipeline` | 246 | 23% | 95% | 177 | 44 | AV-xxx |
| `inference.trt_streaming_pipeline` | 140 | 38% | 95% | 80 | 20 | AV-xxx |

**Test Focus:**
- Voice identification from embeddings (speaker matching, profile lookup)
- Mean flow decoder initialization, forward pass, GPU enforcement
- TensorRT pipeline: engine loading, optimization, inference, fallback paths
- TensorRT streaming: chunk processing, state management, real-time constraints
- Error paths: missing models, GPU OOM, invalid inputs
- Boundary conditions: empty audio, single sample, very long sequences

### P0.2 - Audio Processing Modules (Target: 90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `audio.multi_artist_separator` | 194 | 0% | 90% | 174 | 43 | AV-xxx |
| `audio.separation` | 285 | 40% | 90% | 142 | 35 | AV-xxx |
| `audio.diarization_extractor` | 389 | 50% | 90% | 156 | 39 | AV-xxx |
| `audio.speaker_matcher` | 397 | 45% | 90% | 179 | 44 | AV-xxx |

**Test Focus:**
- Multi-artist separation: track splitting, overlap handling, artist identification
- Vocal separation: demucs integration, stem extraction, quality validation
- Diarization: speaker segmentation, timeline extraction, overlap regions
- Speaker matching: embedding similarity, threshold tuning, profile selection
- Mock expensive ML models (demucs, pyannote) for fast tests
- Test with synthetic multi-speaker audio

### P0.3 - Evaluation & Web API Critical Gaps (Target: 90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `evaluation.conversion_quality_analyzer` | 268 | 0% | 90% | 241 | 60 | AV-xxx |
| `web.speaker_api` | 225 | 18% | 90% | 162 | 40 | AV-xxx |

**Test Focus:**
- Quality analyzer: MCD, F0 RMSE, PESQ, STOI metrics computation
- Speaker API: profile CRUD, speaker registration, embedding storage
- Error handling: invalid audio formats, missing profiles, validation failures
- Integration with database and storage layers

---

## Priority P1 - HIGH (80% → 90% Push)

**Target:** Fill remaining gaps to reach 90% overall coverage
**Total:** 22 modules, 3,481 lines, ~865 tests
**Timeline:** 86 days (1 agent) → 14 days (6 agents)

### P1.1 - Web API Modules (Target: 90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `web.karaoke_api` | 793 | 30% | 90% | 476 | 119 | AV-xxx |
| `web.audio_router` | 180 | 50% | 90% | 72 | 18 | AV-xxx |
| `web.karaoke_manager` | 117 | 32% | 90% | 68 | 17 | AV-xxx |
| `web.voice_model_registry` | 150 | 40% | 90% | 75 | 18 | AV-xxx |

**Test Focus:**
- Karaoke API: WebSocket events, session management, real-time synchronization
- Audio router: stream routing, format conversion, buffering
- Voice model registry: model CRUD, version management, metadata storage
- Fix async_mode configuration for SocketIO tests
- Test concurrent sessions and race conditions

### P1.2 - Training & Job Management (Target: 90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `training.job_manager` | 1,113 | 40% | 90% | 556 | 139 | AV-xxx |
| `training.trainer` | 306 | 50% | 90% | 122 | 30 | AV-xxx |

**Test Focus:**
- Job manager: job scheduling, queue management, status tracking, cancellation
- Trainer: training loop, checkpointing, early stopping, GPU memory management
- Mock expensive training operations for fast tests
- Test failure recovery and resume functionality

### P1.3 - Audio Processing Extensions (Target: 90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `audio.technique_detector` | 499 | 0% | 90% | 449 | 112 | AV-xxx |
| `audio.file_organizer` | 192 | 30% | 90% | 115 | 28 | AV-xxx |
| `audio.youtube_downloader` | 91 | 38% | 90% | 47 | 11 | AV-xxx |

**Test Focus:**
- Technique detector: vibrato, belting, falsetto detection algorithms
- File organizer: directory structure creation, file naming, metadata extraction
- YouTube downloader: URL parsing, metadata extraction, download handling
- Mock yt-dlp for fast, network-free tests

### P1.4 - Models (Target: 90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `models.vocoder` | 458 | 45% | 90% | 206 | 51 | AV-xxx |
| `models.encoder` | 349 | 50% | 90% | 140 | 35 | AV-xxx |
| `models.so_vits_svc` | 254 | 55% | 90% | 89 | 22 | AV-xxx |

**Test Focus:**
- Vocoder: HiFiGAN forward pass, upsampling, residual blocks
- Encoder: content/pitch encoding, feature extraction, normalization
- So-VITS-SVC: complete model forward pass, loss computation, gradient flow
- Test with small dummy models for speed
- Verify output shapes and non-NaN values

### P1.5 - Inference Extensions (Target: 95%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `inference.realtime_pipeline` | 358 | 75% | 95% | 72 | 18 | AV-xxx |
| `inference.streaming_pipeline` | 221 | 71% | 95% | 53 | 13 | AV-xxx |
| `inference.voice_cloner` | 277 | 79% | 95% | 45 | 11 | AV-xxx |
| `inference.singing_conversion_pipeline` | 224 | 75% | 95% | 44 | 11 | AV-xxx |
| `inference.trt_rebuilder` | 229 | 81% | 95% | 32 | 8 | AV-xxx |

**Test Focus:**
- Realtime pipeline: latency constraints, buffer management, dropout handling
- Streaming pipeline: chunk processing, state persistence, context window
- Voice cloner: profile creation, embedding extraction, similarity scoring
- Singing conversion: pitch preservation, timing alignment, quality metrics
- TRT rebuilder: model conversion, optimization levels, fallback logic
- Edge cases: very short/long audio, silence, extreme pitch shifts

### P1.6 - Evaluation & Export (Target: 85-90%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `evaluation.quality_metrics` | 449 | 0% | 90% | 404 | 101 | AV-xxx |
| `evaluation.benchmark_dataset` | 200 | 30% | 90% | 120 | 30 | AV-xxx |
| `export.tensorrt_engine` | 169 | 24% | 85% | 103 | 25 | AV-xxx |

**Test Focus:**
- Quality metrics: PESQ, STOI, MCD, F0 RMSE implementations
- Benchmark dataset: data loading, augmentation, train/val splits
- TensorRT engine: model export, optimization, calibration, inference
- Mock TensorRT operations where possible for speed

### P1.7 - Monitoring & Storage (Target: 90-95%)

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `monitoring.quality_monitor` | 226 | 33% | 90% | 129 | 32 | AV-xxx |
| `storage.voice_profiles` | 376 | 78% | 95% | 64 | 16 | AV-xxx |

**Test Focus:**
- Quality monitor: metric collection, alerting, threshold violations
- Storage: profile CRUD, embedding storage, metadata indexing
- Test error conditions: disk full, permission errors, corrupted data

---

## Priority P2 - MEDIUM (Polish to 95%)

**Target:** Polish already-strong modules to 95%
**Total:** 2 modules, 21 lines, ~7 tests
**Timeline:** <1 day

| Module | Lines | Curr % | Target % | Gap Lines | Tests | Beads Issue |
|--------|-------|--------|----------|-----------|-------|-------------|
| `db.operations` | 616 | 91% | 95% | 25 | 6 | AV-xxx |
| `db.schema` | 206 | 97% | 95% | -4 | 1 | AV-xxx |

**Test Focus:**
- Database operations: edge case error handling, transaction rollback, cascade deletes
- Schema: constraint validation, index creation, migration scenarios

---

## Test Creation Strategy

### Test Types Distribution

For each module, aim for:

| Test Type | Percentage | Focus |
|-----------|------------|-------|
| **Unit Tests** | 60% | Individual functions, error paths, boundary conditions |
| **Integration Tests** | 25% | Component interaction, database integration, API calls |
| **Edge Case Tests** | 10% | Empty inputs, extreme values, malformed data |
| **Error Path Tests** | 5% | Exception handling, fallback logic, recovery |

### Coverage-Driven Development Workflow

1. **Analyze uncovered lines** (use `htmlcov/index.html`)
2. **Write failing tests** for uncovered code paths
3. **Verify tests fail** for the right reason
4. **Mock dependencies** to make tests fast and isolated
5. **Run tests** and confirm coverage increase
6. **Refactor** tests for clarity and maintainability

### Test Performance Targets

- **Unit tests:** <100ms per test
- **Integration tests:** <1s per test
- **E2E tests:** <5s per test
- **Full suite:** <30 minutes

### Mocking Strategy

**Always mock:**
- Network calls (yt-dlp, API requests)
- Heavy ML models (demucs, pyannote, HuBERT)
- File system operations (use in-memory alternatives)
- GPU operations (use CPU fallback for most tests)
- TensorRT operations (mock engine creation)

**Use real implementations for:**
- Core inference logic (with small models)
- Audio processing algorithms (with short synthetic audio)
- Database operations (use in-memory SQLite)

---

## Beads Issues Created

### P0 - Critical (10 issues)

| ID | Module | Lines | Tests | Description |
|----|--------|-------|-------|-------------|
| AV-xxx | `inference.voice_identifier` | 206 | 48 | Add tests for voice identification (0% → 95%) |
| AV-xxx | `inference.mean_flow_decoder` | 101 | 23 | Add tests for mean flow decoder (0% → 95%) |
| AV-xxx | `inference.trt_pipeline` | 246 | 44 | Add tests for TensorRT pipeline (23% → 95%) |
| AV-xxx | `inference.trt_streaming_pipeline` | 140 | 20 | Add tests for TRT streaming pipeline (38% → 95%) |
| AV-xxx | `audio.multi_artist_separator` | 194 | 43 | Add tests for multi-artist separation (0% → 90%) |
| AV-xxx | `audio.separation` | 285 | 35 | Add tests for vocal separation (40% → 90%) |
| AV-xxx | `audio.diarization_extractor` | 389 | 39 | Add tests for speaker diarization (50% → 90%) |
| AV-xxx | `audio.speaker_matcher` | 397 | 44 | Add tests for speaker matching (45% → 90%) |
| AV-xxx | `evaluation.conversion_quality_analyzer` | 268 | 60 | Add tests for quality analyzer (0% → 90%) |
| AV-xxx | `web.speaker_api` | 225 | 40 | Add tests for speaker API (18% → 90%) |

**P0 Total:** 10 issues, 2,451 lines, 396 tests

### P1 - High (22 issues)

*See detailed breakdown in P1 sections above*

**P1 Total:** 22 issues, 4,481 lines, 865 tests

### P2 - Medium (2 issues)

**P2 Total:** 2 issues, 822 lines, 7 tests

**GRAND TOTAL:** 34 issues, 5,103 lines, 1,268 tests

---

## Execution Plan

### Phase 1 - P0 Critical (Weeks 1-2)
- **Focus:** Inference core + Audio core
- **Agents:** 6 agents
- **Output:** +10% overall coverage (63% → 73%)
- **Tests:** 396 tests created

### Phase 2 - P1 Web API & Training (Weeks 3-4)
- **Focus:** Web API, Training, Monitoring
- **Agents:** 6 agents
- **Output:** +8% overall coverage (73% → 81%)
- **Tests:** 450 tests created

### Phase 3 - P1 Models & Evaluation (Weeks 5-6)
- **Focus:** Models, Evaluation, Export
- **Agents:** 6 agents
- **Output:** +7% overall coverage (81% → 88%)
- **Tests:** 415 tests created

### Phase 4 - P2 Polish (Week 7)
- **Focus:** Polish to 95%
- **Agents:** 2 agents
- **Output:** +7% overall coverage (88% → 95%)
- **Tests:** 7 tests created + bug fixes

---

## Success Criteria

✅ **Overall coverage ≥ 95%**
✅ **Inference coverage ≥ 95%**
✅ **Audio coverage ≥ 90%**
✅ **Web API coverage ≥ 90%**
✅ **Database coverage ≥ 95%**
✅ **All tests passing (>98% pass rate)**
✅ **Test suite runtime < 30 minutes**
✅ **No test failures due to missing dependencies**
✅ **Comprehensive edge case and error path coverage**

---

## Next Steps

1. **Create beads issues** for all 34 modules (P0, P1, P2)
2. **Launch testing agents**:
   - Agent 1-3: P0 Inference modules
   - Agent 4-6: P0 Audio modules
3. **Track progress** via beads dashboard
4. **Daily sync** on coverage metrics
5. **Adjust targets** based on test failure patterns

---

**Report Generated:** 2026-02-02
**Master Orchestrator:** Standing by for agent deployment
**Roadmap Status:** ✅ **READY FOR EXECUTION**
