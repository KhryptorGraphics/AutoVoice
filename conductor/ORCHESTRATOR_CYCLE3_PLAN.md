# Orchestration Cycle 3 Plan - Coverage Gap-Filling

**Date:** 2026-02-02
**Status:** DRAFT (Pending Cycle 2 validation results)
**Trigger Condition:** Overall project coverage <80% after Cycle 2

---

## Executive Summary

If Cycle 2 results show coverage below 80% target (projected ~67%), execute Cycle 3 with 4-5 parallel agents targeting remaining high-priority coverage gaps. Focus on completing P0 (TensorRT pipelines) and highest-impact P1 modules (audio processing, web API).

---

## Pre-Execution Checklist

- [ ] Verify Cycle 2 coverage results: `coverage <80%`
- [ ] Confirm all Cycle 2 beads issues closed (currently: 44 closed, 0 open) ✅
- [ ] Review pytest pass rate (current: 86.2%, target: 95%+)
- [ ] Verify dependencies installed: local-attention ✅, PyTorch CUDA ✅
- [ ] Check available agent slots (recommended: 4-5 concurrent)

---

## Coverage Gap Analysis

### Current State (Post-Cycle-2 Projection)
- **Baseline (Cycle 1):** 63%
- **Cycle 2 Impact:** +3.9% (voice_identifier 81%, mean_flow_decoder 85%, conversion_quality_analyzer 98%)
- **Projected:** ~67%
- **Target:** 80%
- **Gap:** 13 percentage points

### Priority Targets for Cycle 3

#### P0 (Critical - Remaining)
| Module | Current | Target | Lines | Impact |
|--------|---------|--------|-------|--------|
| `inference/trt_pipeline.py` | 23% | 70% | +115 | +0.8% |

#### P1 (High Priority - Selected)
| Module | Current | Target | Lines | Impact |
|--------|---------|--------|-------|--------|
| `audio/multi_artist_separator.py` | 0% | 70% | +194 | +1.3% |
| `web/speaker_api.py` | 18% | 80% | +185 | +1.2% |
| `audio/file_organizer.py` | 30% | 70% | +80 | +0.5% |
| `inference/trt_streaming_pipeline.py` | 38% | 70% | +45 | +0.3% |

#### P2 (Medium Priority - Selected)
| Module | Current | Target | Lines | Impact |
|--------|---------|--------|-------|--------|
| `export/tensorrt_engine.py` | 24% | 70% | +129 | +0.9% |
| `monitoring/quality_monitor.py` | 33% | 70% | +151 | +1.0% |

### Expected Cycle 3 Impact
- **P0:** +0.8%
- **P1:** +3.3%
- **P2:** +1.9%
- **Total:** +6.0%
- **Projected Final:** 67% + 6% = **73%**

**⚠️ Note:** 73% is still below 80% target. May require Cycle 4 or strategic test expansion beyond listed modules.

---

## Agent Allocation Strategy

### Parallel Wave 1 (4 Testing Agents)

#### Agent 1: TensorRT Pipeline Tests (AV-c3t1)
**Priority:** P0 + P1
**Modules:**
- `src/auto_voice/inference/trt_pipeline.py` (115 lines to cover)
- `src/auto_voice/inference/trt_streaming_pipeline.py` (45 lines to cover)

**Tests to Create:**
- TensorRT engine initialization and caching
- Model compilation for different batch sizes
- FP16/INT8 quantization paths
- Streaming buffer management
- Error handling for unsupported ops

**Expected Deliverable:** `tests/inference/test_trt_pipeline.py` (~400 lines, 25-30 tests)
**Duration:** ~5 minutes

#### Agent 2: Audio Processing Tests (AV-c3t2)
**Priority:** P1
**Modules:**
- `src/auto_voice/audio/multi_artist_separator.py` (194 lines to cover)
- `src/auto_voice/audio/file_organizer.py` (80 lines to cover)

**Tests to Create:**
- Multi-track audio separation (requires demucs or mocking)
- Speaker assignment to tracks
- File organization by artist/date
- Duplicate detection
- Metadata extraction

**Expected Deliverable:** `tests/audio/test_multi_artist_processing.py` (~500 lines, 35-40 tests)
**Duration:** ~6 minutes

#### Agent 3: Web API Tests (AV-c3t3)
**Priority:** P1
**Modules:**
- `src/auto_voice/web/speaker_api.py` (185 lines to cover)

**Tests to Create:**
- Speaker profile CRUD operations
- Embedding storage and retrieval
- Speaker matching endpoints
- Error cases (invalid IDs, missing data)
- Authentication/authorization tests

**Expected Deliverable:** `tests/web/test_speaker_api_comprehensive.py` (~450 lines, 30-35 tests)
**Duration:** ~5 minutes

#### Agent 4: Export & Monitoring Tests (AV-c3t4)
**Priority:** P2
**Modules:**
- `src/auto_voice/export/tensorrt_engine.py` (129 lines to cover)
- `src/auto_voice/monitoring/quality_monitor.py` (151 lines to cover)

**Tests to Create:**
- ONNX export from PyTorch models
- TensorRT engine optimization
- Quality metric tracking (MOS, PESQ, STOI)
- Performance benchmarking
- Alert thresholds

**Expected Deliverable:** `tests/export/test_tensorrt_engine.py`, `tests/monitoring/test_quality_monitor.py` (~550 lines, 40-45 tests)
**Duration:** ~7 minutes

### Sequential Wave 2 (1 Quality Agent)

#### Agent 5: Fix Failing Tests (AV-c3q1)
**Priority:** Quality assurance
**Dependencies:** Agents 1-4 complete

**Tasks:**
- Fix import errors in new test files
- Resolve any dependency issues (demucs, TensorRT)
- Add pytest marks for conditional tests
- Verify all tests collecting correctly
- Run full suite and categorize remaining failures

**Expected Improvements:**
- Test pass rate: 86.2% → 92%+
- New tests passing: 130-150 tests (100%)
- Zero import/collection errors

**Duration:** ~8 minutes

---

## Execution Timeline

**Total Duration:** ~25 minutes
**Agents:** 5 (4 parallel + 1 sequential)

| Phase | Time | Agents | Activity |
|-------|------|--------|----------|
| **Wave 1** | 0-7 min | 4 parallel | Create test files for P0/P1/P2 modules |
| **Wave 2** | 7-15 min | 1 | Fix failing tests, verify collection |
| **Verification** | 15-25 min | 1 | Run full coverage analysis |

---

## Beads Task Structure

### Epic
- **AV-c3** (Epic): Orchestration Cycle 3 - Coverage Gap-Filling
  - Type: epic
  - Priority: 0 (critical)
  - Status: blocked (by coverage validation)

### Testing Stack
- **AV-c3t1**: TensorRT Pipeline Tests (trt_pipeline.py, trt_streaming_pipeline.py)
- **AV-c3t2**: Audio Processing Tests (multi_artist_separator.py, file_organizer.py)
- **AV-c3t3**: Web API Tests (speaker_api.py)
- **AV-c3t4**: Export & Monitoring Tests (tensorrt_engine.py, quality_monitor.py)

### Quality Stack
- **AV-c3q1**: Fix Failing Tests (dependency resolution, pytest configuration)

### Dependencies
```
AV-c3 (Epic)
├── AV-c3t1 (blocks AV-c3q1)
├── AV-c3t2 (blocks AV-c3q1)
├── AV-c3t3 (blocks AV-c3q1)
├── AV-c3t4 (blocks AV-c3q1)
└── AV-c3q1 (blocks coverage validation)
```

---

## Success Criteria

### Coverage Targets
- [ ] trt_pipeline.py: 23% → 70% (minimum 65%)
- [ ] multi_artist_separator.py: 0% → 70% (minimum 60%)
- [ ] speaker_api.py: 18% → 80% (minimum 70%)
- [ ] Overall project coverage: 67% → 73%+ (stretch: 75%)

### Quality Targets
- [ ] Test pass rate: 86.2% → 92%+
- [ ] All new tests passing (100% of ~150 new tests)
- [ ] Zero import/collection errors
- [ ] No regression in existing test coverage

### Delivery Targets
- [ ] 4 new test files created (~1,900 lines, 130-150 tests)
- [ ] All beads issues closed
- [ ] Coverage report generated
- [ ] Orchestrator sync report updated

---

## Risk Mitigation

### Risk 1: TensorRT Dependencies
**Issue:** TensorRT tests may fail if runtime not available on ARM64
**Mitigation:** Use skipif decorators: `@pytest.mark.skipif(not has_tensorrt)`

### Risk 2: Demucs Dependency
**Issue:** multi_artist_separator.py requires demucs (not installed)
**Mitigation:** Install demucs OR mock separation functionality in tests

### Risk 3: Still Below 80% After Cycle 3
**Issue:** Projected 73% coverage, need 80%
**Mitigation:**
- Execute Cycle 4 targeting P2 modules
- Expand test coverage beyond unit tests (more integration tests)
- Consider lowering target to 75% if 80% requires excessive effort

### Risk 4: Agent Context Errors
**Issue:** Previous cycles had 2-3 agent resumptions due to directory context
**Mitigation:**
- Use absolute paths in all agent prompts
- Verify directory context before spawning agents
- Include explicit `cd /home/kp/repo2/autovoice` in agent instructions

---

## Post-Cycle Actions

### If Coverage ≥80%
1. Close Cycle 3 beads issues
2. Update `conductor/tracks.md`
3. Mark `coverage-report-generation_20260201` complete
4. Begin `production-deployment-prep_20260201` track
5. Commit and push all changes

### If Coverage <80%
1. Analyze coverage gaps
2. Determine if Cycle 4 needed or adjust target
3. Plan Cycle 4 with remaining P2 modules + strategic expansions
4. Consider integration test expansion (E2E workflows)

---

## Agent Prompts (Ready to Execute)

### Agent 1 Prompt
```
Create comprehensive tests for TensorRT inference pipelines:

Files to test:
- src/auto_voice/inference/trt_pipeline.py (current: 23%, target: 70%)
- src/auto_voice/inference/trt_streaming_pipeline.py (current: 38%, target: 70%)

Create: tests/inference/test_trt_pipeline.py

Tests needed:
1. TensorRT engine initialization (FP16/INT8 modes)
2. Model compilation with different batch sizes
3. Inference execution with sample audio
4. Streaming buffer management
5. Error handling (unsupported ops, OOM)
6. Engine caching and persistence

Use skipif for TensorRT availability:
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

Target: 25-30 tests, ~400 lines, all passing
```

### Agent 2 Prompt
```
Create comprehensive tests for audio processing modules:

Files to test:
- src/auto_voice/audio/multi_artist_separator.py (current: 0%, target: 70%)
- src/auto_voice/audio/file_organizer.py (current: 30%, target: 70%)

Create: tests/audio/test_multi_artist_processing.py

Tests needed:
1. Multi-track separation (mock demucs if not installed)
2. Speaker assignment to separated tracks
3. File organization by artist/date
4. Duplicate detection via audio fingerprinting
5. Metadata extraction and tagging

Use skipif for demucs:
@pytest.mark.skipif(not has_demucs, reason="requires demucs")

Target: 35-40 tests, ~500 lines, all passing
```

### Agent 3 Prompt
```
Create comprehensive tests for speaker API endpoints:

Files to test:
- src/auto_voice/web/speaker_api.py (current: 18%, target: 80%)

Create: tests/web/test_speaker_api_comprehensive.py

Tests needed:
1. GET /api/speakers (list all)
2. GET /api/speakers/{id} (retrieve)
3. POST /api/speakers (create with audio sample)
4. PUT /api/speakers/{id} (update)
5. DELETE /api/speakers/{id}
6. POST /api/speakers/match (find similar speakers)
7. Error cases (404, 400, missing data)

Use Flask test client fixtures.

Target: 30-35 tests, ~450 lines, all passing
```

### Agent 4 Prompt
```
Create comprehensive tests for export and monitoring:

Files to test:
- src/auto_voice/export/tensorrt_engine.py (current: 24%, target: 70%)
- src/auto_voice/monitoring/quality_monitor.py (current: 33%, target: 70%)

Create:
- tests/export/test_tensorrt_engine.py
- tests/monitoring/test_quality_monitor.py

Tests needed:
1. ONNX export from PyTorch models
2. TensorRT optimization (FP16, INT8 calibration)
3. Quality metric computation (MOS prediction)
4. PESQ/STOI calculation
5. Performance benchmarking
6. Alert threshold monitoring

Target: 40-45 tests, ~550 lines, all passing
```

### Agent 5 Prompt
```
Fix failing tests and verify test suite health:

Tasks:
1. Run pytest to identify any import/collection errors
2. Fix TensorRT import guards (conditional imports)
3. Add pytest marks for demucs tests
4. Verify all new tests collecting correctly
5. Categorize remaining failures by type
6. Run full coverage analysis

Expected:
- Test pass rate: 86.2% → 92%+
- Zero import/collection errors
- All new tests (130-150) passing

Target: Complete in ~8 minutes
```

---

**Status:** Ready for execution pending Cycle 2 coverage validation
**Next Action:** Wait for pytest (task b3ad36d) to complete, then make go/no-go decision
