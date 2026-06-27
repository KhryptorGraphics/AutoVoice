# AutoVoice Master Orchestration Plan
## Generated: 2026-02-01

> Historical artifact: this plan captured a February 2026 coordination snapshot. It is not the current backlog or execution source of truth. Use `bd`, [docs/current-truth.md](docs/current-truth.md), and current repo automation instead.

## Executive Summary

**Current Status:**
- 1,588 tests collected across codebase
- 4 voice conversion pipelines operational (realtime, quality, quality_seedvc, realtime_meanvc)
- SOTA innovations track COMPLETE (DiT-CFM, Shortcut CFM, MeanVC all implemented)
- 3 in-progress tracks need completion: testing coverage, performance validation, E2E workflows

**Completion Target:** All P0/P1 tracks complete within 3-4 days

---

## Track Status Overview

### ✅ Completed Tracks (11)
1. sota-pipeline_20260124 - SOTA Pipeline Refactor
2. live-karaoke_20260124 - Live Karaoke Voice Conversion
3. voice-profile-training_20260124 - Voice Profile & Continuous Training
4. frontend-parity_20260129 - Frontend-Backend Parity
5. codebase-audit_20260130 - Comprehensive Codebase Audit
6. training-inference-integration_20260130 - Training-to-Inference Integration
7. speaker-diarization_20260130 - Speaker Diarization & Auto-Profile
8. youtube-artist-training_20260130 - YouTube Artist Training Pipeline
9. **sota-innovations_20260131** - SOTA Innovations (JUST MARKED COMPLETE)
10. frontend-complete-integration_20260201 - Frontend Complete Integration
11. api-documentation-suite_20260201 - API Documentation Suite

### 🔄 In-Progress Tracks (3)

#### 1. comprehensive-testing-coverage_20260201 (P0)
**Status:** Phase 1/6 complete, 90% done
**Tests:** 1,588 collected
**Remaining Work:**
- Generate pytest-cov HTML report
- Identify modules <70% coverage
- Add audio processing tests (Phase 2)
- Add database/storage tests (Phase 3)
- Add web API tests (Phase 4-5)
- Optimize slow tests

**Beads Task:** AV-4tg (Generate pytest coverage report)
**Estimated:** 1 day for coverage report + gap analysis

#### 2. performance-validation-suite_20260201 (P1)
**Status:** 0/7 phases complete
**Dependencies:** SOTA innovations Phase 4 (MeanVC) ✅ COMPLETE
**Remaining Work:**
- Create benchmark infrastructure (scripts/performance_validation.py)
- Benchmark all 4 pipelines (RTF, latency, memory, quality)
- Memory profiling and leak detection
- Latency analysis (component-level breakdown)
- Quality validation (MCD, speaker similarity)
- Concurrent load testing
- Generate comprehensive report

**Beads Tasks:**
- AV-4kt (Create benchmark infrastructure)
- AV-maw (Benchmark all pipelines) - blocks on AV-4kt

**Estimated:** 2.5 days

#### 3. voice-profile-training-e2e_20260201 (P0)
**Status:** 0/6 phases complete
**Dependencies:** All upstream tracks complete ✅
**Remaining Work:**
- Web UI flow validation (VoiceProfilePage, sample upload, diarization)
- LoRA training flow (config, job creation, progress monitoring)
- YouTube multi-artist flow (detection, auto-profile creation)
- Adapter integration (loading, switching, nvfp4 vs hq)
- Error handling (insufficient samples, invalid formats, cancellation)
- Integration tests (automated E2E scripts)

**Beads Task:** AV-39t (E2E validation workflows)
**Estimated:** 1.5 days

### ⏸️ Deferred Track (1)
- production-deployment-prep_20260201 - Awaiting completion of above tracks

---

## Orchestration Strategy

### Phase 1: Immediate Wins (Today)
**Parallel Execution:**

1. **Coverage Report Generation** (AV-4tg)
   - Run: `pytest --cov=src/auto_voice --cov-report=html`
   - Analysis: Identify modules <70% coverage
   - Document: Create gap analysis report
   - **Owner:** Test automation agent or immediate execution
   - **Duration:** 2 hours

2. **SOTA Track Finalization**
   - Update plan.md with completion status ✅ DONE
   - Mark metadata.json as complete ✅ DONE
   - Update conductor/tracks.md ✅ DONE

### Phase 2: Testing Infrastructure (Day 1-2)
**Sequential + Parallel:**

1. **Performance Benchmark Infrastructure** (AV-4kt)
   - Create scripts/performance_validation.py
   - Implement RTF calculator, memory profiler, latency timer
   - Setup test fixtures (5s, 30s, 3min audio clips)
   - Validate JSON/CSV output
   - **Owner:** Performance engineer agent
   - **Duration:** 4 hours

2. **E2E Test Automation Setup** (AV-39t - Part 1)
   - Create test fixtures for voice profiles
   - Setup mock YouTube downloads
   - Create automated test scripts (pytest)
   - **Owner:** Test automation agent
   - **Duration:** 4 hours (parallel with above)

### Phase 3: Comprehensive Validation (Day 2-3)
**Parallel Execution:**

1. **Pipeline Benchmarks** (AV-maw)
   - Benchmark realtime pipeline (RTF, latency, memory)
   - Benchmark quality pipeline (quality metrics)
   - Benchmark quality_seedvc (speed vs quality tradeoff)
   - Benchmark realtime_meanvc (CPU-only streaming)
   - Generate comparison table + report
   - **Owner:** Performance engineer agent
   - **Duration:** 6 hours

2. **E2E Workflow Validation** (AV-39t - Part 2)
   - Test profile creation from audio
   - Test YouTube multi-artist detection + training
   - Test adapter loading in all pipeline types
   - Test error handling scenarios
   - **Owner:** Test automation agent
   - **Duration:** 6 hours (parallel with benchmarks)

3. **Test Gap Filling** (comprehensive-testing-coverage Phase 2-5)
   - Audio processing tests (diarization, separation, YouTube)
   - Database tests (CRUD, schema, sessions)
   - Web API tests (60+ endpoints)
   - Integration tests (E2E flows)
   - **Owner:** Test automation agent
   - **Duration:** 1 day (can parallelize across modules)

### Phase 4: Documentation & Finalization (Day 3-4)
**Sequential:**

1. **Performance Report Generation**
   - Comprehensive benchmark report (markdown)
   - Pipeline selection guide for users
   - Update CLAUDE.md with performance targets
   - CI integration tests
   - **Duration:** 4 hours

2. **Testing Report Generation**
   - Coverage report analysis
   - Test strategy documentation
   - Update CLAUDE.md with test patterns
   - **Duration:** 2 hours

3. **Track Completion**
   - Mark all 3 tracks complete in conductor/tracks.md
   - Update metadata.json for each track
   - Run `bd sync --flush-only`
   - Close beads tasks
   - **Duration:** 1 hour

---

## Agent Spawning Plan

### Recommended Agent Assignments

**Option 1: Minimal Agents (Sequential)**
- 1 test-automator agent: Handles all testing tasks sequentially
- 1 performance-engineer agent: Handles all performance tasks
- **Timeline:** 4-5 days

**Option 2: Parallel Execution (Recommended)**
- 2 test-automator agents:
  - Agent A: Coverage report + audio/database tests
  - Agent B: E2E workflow validation
- 1 performance-engineer agent: Benchmark infrastructure + pipeline validation
- **Timeline:** 2-3 days

**Option 3: Maximum Parallelization**
- 3 test-automator agents:
  - Agent A: Coverage report generation
  - Agent B: Audio/database/API tests
  - Agent C: E2E workflow validation
- 2 performance-engineer agents:
  - Agent A: Benchmark infrastructure
  - Agent B: Pipeline benchmarks + reporting
- **Timeline:** 1.5-2 days (aggressive)

### Cross-Dependencies Map
```
AV-4tg (Coverage Report) → Independent, can run immediately
AV-4kt (Benchmark Infra) → Independent, can run immediately
AV-maw (Pipeline Benchmarks) → Blocks on AV-4kt
AV-39t (E2E Workflows) → Independent, can run immediately

comprehensive-testing-coverage Phase 2-5 → Can run in parallel with all
```

---

## Beads Task Summary

| ID | Title | Priority | Status | Blocks | Track |
|----|-------|----------|--------|--------|-------|
| AV-4tg | Generate pytest coverage report | P0 | open | - | comprehensive-testing-coverage |
| AV-4kt | Create performance benchmark infrastructure | P1 | open | - | performance-validation-suite |
| AV-maw | Benchmark all 4 pipelines | P1 | open | AV-4kt | performance-validation-suite |
| AV-39t | E2E validation workflows | P0 | open | - | voice-profile-training-e2e |

---

## Success Criteria

### comprehensive-testing-coverage_20260201
- [x] 1,588 tests collected
- [ ] Overall coverage ≥80%
- [ ] Inference coverage ≥85%
- [ ] All critical modules tested
- [ ] HTML coverage report generated
- [ ] Gap analysis documented

### performance-validation-suite_20260201
- [ ] All 4 pipelines benchmarked
- [ ] Comparison table published
- [ ] Memory profiling complete (9.5GB / 64GB = 15%)
- [ ] RTF targets met (realtime <0.5, quality <1.0)
- [ ] Latency targets met (streaming <100ms)
- [ ] Quality targets met (MCD <6dB, similarity >0.85)
- [ ] Comprehensive report in reports/

### voice-profile-training-e2e_20260201
- [ ] Profile creation workflow validated
- [ ] YouTube multi-artist workflow validated
- [ ] Training + adapter loading workflow validated
- [ ] All pipeline types tested with adapters
- [ ] Error handling validated
- [ ] E2E test suite created

---

## Risk Assessment

### Low Risk
- Coverage report generation (automated, fast)
- Benchmark infrastructure creation (clear requirements)
- E2E test creation (fixtures available)

### Medium Risk
- Achieving 80% coverage target (may require significant gap filling)
- Performance targets (RTF, latency) - may reveal optimization needs
- E2E workflows (complex dependencies, potential UI issues)

### Mitigation
- Start with coverage report to quantify actual gap
- Use existing quality_report.json as baseline for performance
- Create mock fixtures for E2E to avoid network dependencies

---

## Next Steps (Immediate Action)

1. **NOW:** Generate coverage report
   ```bash
   PYTHONNOUSERSITE=1 PYTHONPATH=src /home/kp/anaconda3/envs/autovoice-thor/bin/python -m pytest \
     --cov=src/auto_voice \
     --cov-report=html \
     --cov-report=term \
     tests/
   ```

2. **Spawn Agents** (if using Task tool):
   - test-automator for AV-4tg (coverage analysis)
   - performance-engineer for AV-4kt (benchmark infra)
   - test-automator for AV-39t (E2E workflows)

3. **Track Progress:**
   - Update metadata.json as phases complete
   - Run `bd sync --flush-only` after each milestone
   - Close beads tasks as they finish

---

## Timeline Summary

**Optimistic (Parallel Execution):** 2 days
**Realistic (Partial Parallel):** 3 days
**Conservative (Sequential):** 4-5 days

**Recommended Approach:** Option 2 (Parallel Execution) → 3 day timeline

---

_Generated by Master Swarm Orchestrator - 2026-02-01_
