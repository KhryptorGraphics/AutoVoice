# Implementation Plan: Coverage Report Generation

**Track ID:** coverage-report-generation_20260201
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-01
**Status:** [ ] Pending

## Phase 1: Generate Initial Coverage Report

### Tasks

- [ ] Task 1.1: Install pytest-cov if missing
  - `pip install pytest-cov`

- [ ] Task 1.2: Run pytest with coverage
  - `PYTHONNOUSERSITE=1 PYTHONPATH=src pytest --cov=src/auto_voice --cov-report=html --cov-report=term`

- [ ] Task 1.3: Analyze HTML report in `htmlcov/index.html`
  - Identify overall coverage percentage
  - List modules <70% coverage
  - Identify untested critical modules

- [ ] Task 1.4: Generate coverage gap report
  - Create `reports/coverage_gaps_20260201.md`
  - List uncovered branches and functions
  - Prioritize by criticality (P0: inference, web, audio)

### Verification

- [ ] Coverage report generated successfully
- [ ] Gaps identified and prioritized
- [ ] No test failures

## Phase 2: Fill Inference Coverage Gaps

Target: 85% coverage for `src/auto_voice/inference/`

### Tasks

- [ ] Task 2.1: Add tests for uncovered error paths
  - Model loading failures
  - GPU OOM handling
  - Invalid adapter paths

- [ ] Task 2.2: Add edge case tests
  - Empty audio input
  - None values in parameters
  - Boundary conditions (0-length, max-length)

- [ ] Task 2.3: Test all pipeline types
  - Verify each pipeline in PipelineFactory
  - Test lazy loading and caching
  - Test pipeline switching

### Verification

- [ ] Inference coverage ≥85%
- [ ] All critical error paths tested
- [ ] Tests pass in <5 minutes

## Phase 3: Fill Web API Coverage Gaps

Target: 80% coverage for `src/auto_voice/web/`

### Tasks

- [ ] Task 3.1: Add missing endpoint tests
  - Missing from Phase 4 of comprehensive-testing-coverage
  - Focus on error responses (400, 404, 500)

- [ ] Task 3.2: Test authentication/authorization (if applicable)
  - API key validation
  - Rate limiting

- [ ] Task 3.3: Test WebSocket error handling
  - Disconnection recovery
  - Invalid message formats
  - Session cleanup on error

### Verification

- [ ] Web API coverage ≥80%
- [ ] All error codes tested
- [ ] WebSocket edge cases covered

## Phase 4: Fill Audio Processing Coverage Gaps

Target: 70% coverage for `src/auto_voice/audio/`

### Tasks

- [ ] Task 4.1: Test speaker_diarization.py uncovered branches
  - Multi-speaker edge cases
  - Overlapping speech handling

- [ ] Task 4.2: Test separation.py error paths
  - GPU vs CPU fallback
  - Invalid audio format handling

- [ ] Task 4.3: Test youtube_downloader.py errors
  - Network failures
  - Invalid URLs
  - Geo-blocking

### Verification

- [ ] Audio coverage ≥70%
- [ ] Critical modules fully tested
- [ ] No network calls in tests (use mocks)

## Phase 5: Database and Storage Tests

Target: 70% coverage for `src/auto_voice/db/` and storage modules

### Tasks

- [ ] Task 5.1: Test db/operations.py CRUD operations
  - Profile creation, retrieval, update, deletion
  - Transaction rollback on error

- [ ] Task 5.2: Test db/schema.py validation
  - Schema creation
  - Foreign key constraints
  - Unique constraints

- [ ] Task 5.3: Test storage/voice_profiles.py
  - File storage and retrieval
  - Cleanup on profile deletion

### Verification

- [ ] Database coverage ≥70%
- [ ] Tests use in-memory SQLite
- [ ] No file system side effects

## Phase 6: Final Coverage Report - ✅ COMPLETE

### Tasks

- [x] Task 6.1: Re-run pytest with coverage
  - ✅ Complete: 63% overall (9,467 / 15,063 lines), 1,791 tests passing

- [x] Task 6.2: Generate final HTML report
  - ✅ Complete: `htmlcov/index.html` generated

- [x] Task 6.3: Create coverage summary document
  - ✅ Complete: `reports/coverage_summary_20260202.md`
  - Overall percentage: 63%
  - Per-module breakdown included
  - Remaining gaps documented with priority

- [x] Task 6.4: Update CLAUDE.md with coverage patterns
  - ✅ Complete: Test fixtures documented
  - ✅ Complete: Test strategy guidance added
  - ✅ Complete: Module-specific patterns included

- [ ] Task 6.5: Add coverage CI check
  - ⏳ Deferred: `.github/workflows/coverage.yml` (requires CI setup)
  - Recommended threshold: 60% (current baseline)

### Verification

- [~] Overall coverage ≥80% - **Current: 63%** (17pp below target)
- [~] Inference coverage ≥85% - **Current: ~68%** (17pp below target)
- [x] Documentation updated - ✅ Complete
- [ ] CI integration ready - ⏳ Deferred (no CI pipeline configured yet)

### Results

**Coverage Achievement:**
- Database: 87% ✅ (exceeds 70% target)
- Storage: 78% ✅ (exceeds 70% target)
- Inference Core: 68% ⚠️ (17pp below 85% target)
- Audio Processing: 55% ⚠️ (15pp below 70% target)
- Web API: 60% ⚠️ (20pp below 80% target)

**Test Suite Health:**
- 1,984 tests total
- 1,791 passing (90.3%)
- 147 failing (7.4%)
- 47 errors (2.4%)
- 39 skipped (2.0%)
- Runtime: 27 minutes

**Gap Analysis:**
- Need ~900 additional lines of coverage to reach 80%
- Priority P0: 690 lines (inference + evaluation)
- Priority P1: 536 lines (TensorRT + audio + web)
- Priority P2: 416 lines (export + monitoring + youtube)
- Estimated effort: 7 days to reach 80% target

**Next Steps:**
1. Fix test failures (194 total, mostly missing dependencies)
2. Add tests for P0 modules (voice_identifier, mean_flow_decoder)
3. Improve TensorRT pipeline coverage
4. Fill remaining gaps to reach 80% target

## Final Verification

- [ ] All acceptance criteria met
- [ ] Coverage report committed
- [ ] No regression in existing tests
- [ ] Test suite completes in <20 minutes

---

**Estimated Timeline:** 2 days
**Dependencies:** comprehensive-testing-coverage_20260201 Phases 1-5
**Blocks:** Production deployment, CI/CD validation

---

_Generated by Gap Analysis Watcher._
