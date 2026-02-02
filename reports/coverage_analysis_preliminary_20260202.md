# Coverage Analysis Report (Preliminary)

**Date:** 2026-02-02
**Status:** BLOCKED - Waiting for Phase 5 completion
**Last Coverage Run:** 2026-02-01 12:26 PM
**Agent:** Phase 6 - Coverage Report Generation

## Executive Summary

**Current State:**
- Overall Coverage: **60.2%** (8,889 / 14,769 lines)
- Inference Coverage: **68.4%**
- Target: **80% overall**, **85% inference**
- Gap: **19.8 percentage points overall**, **16.6 pp inference**

**Test Suite:**
- 2,017 tests collected (up from 1,562)
- 105 test files
- Mature test infrastructure

## Critical Coverage Gaps

### Modules with 0% Coverage (HIGH PRIORITY)

These modules have **NO test coverage** and should be prioritized:

#### Audio Processing (4 modules, 802 lines)
1. **`audio/diarization_extractor.py`** - 196 lines
   - Speaker segment isolation
   - Critical for profile creation workflows

2. **`audio/speaker_matcher.py`** - 220 lines
   - Embedding-based speaker identification
   - Critical for multi-speaker workflows

3. **`audio/multi_artist_separator.py`** - 194 lines
   - Multi-artist track separation
   - Used in complex audio processing

4. **`audio/file_organizer.py`** - 192 lines
   - File management and cleanup
   - Important for storage management

#### Database (3 modules, 179 lines)
5. **`db/operations.py`** - 116 lines
   - CRUD operations for profiles
   - **CRITICAL** - Core data persistence

6. **`db/schema.py`** - 60 lines
   - Database schema definition
   - **CRITICAL** - Data model validation

7. **`db/__init__.py`** - 3 lines
   - Module initialization

#### Evaluation (1 module, 268 lines)
8. **`evaluation/conversion_quality_analyzer.py`** - 268 lines
   - Quality metrics computation
   - Used for benchmarking and validation

#### Inference (2 modules, 307 lines)
9. **`inference/mean_flow_decoder.py`** - 101 lines
   - MeanVC decoder implementation
   - Part of realtime pipeline

10. **`inference/voice_identifier.py`** - 206 lines
    - Voice recognition and matching
    - Used in profile workflows

**Total 0% Coverage: 10 modules, 1,556 lines (10.5% of codebase)**

### Modules with <50% Coverage (MEDIUM PRIORITY)

These modules have partial coverage but need significant improvement:

| Module | Coverage | Missing Lines | Priority |
|--------|----------|---------------|----------|
| `audio/training_filter.py` | 13.0% | 120/138 | P1 |
| `web/speaker_api.py` | 17.8% | 185/225 | P1 |
| `evaluation/benchmark_dataset.py` | 22.9% | 37/48 | P2 |
| `inference/trt_pipeline.py` | 23.2% | 189/246 | P1 |
| `export/tensorrt_engine.py` | 23.7% | 129/169 | P2 |
| `web/karaoke_manager.py` | 31.6% | 80/117 | P1 |
| `monitoring/quality_monitor.py` | 33.2% | 151/226 | P2 |
| `web/job_manager.py` | 37.5% | 100/160 | P1 |
| `inference/trt_streaming_pipeline.py` | 37.9% | 87/140 | P1 |
| `youtube/downloader.py` | 38.5% | 56/91 | P2 |

**Total <50% Coverage: 10 additional modules, 1,074 missing lines**

## Inference Module Deep Dive

**Current Inference Coverage: 68.4%**
**Target: 85%**
**Gap: 16.6 percentage points**

### Inference Modules Performance

| Module | Coverage | Status |
|--------|----------|--------|
| `adapter_bridge.py` | 97.2% | ✅ Excellent |
| `pipeline_factory.py` | 93.7% | ✅ Excellent |
| `gpu_enforcement.py` | 92.3% | ✅ Excellent |
| `meanvc_pipeline.py` | 91.3% | ✅ Excellent |
| `sota_pipeline.py` | 88.6% | ✅ Good |
| `model_manager.py` | 87.8% | ✅ Good |
| `realtime_voice_conversion_pipeline.py` | 87.7% | ✅ Good |
| `hq_svc_wrapper.py` | 87.1% | ✅ Good |
| `seed_vc_pipeline.py` | 84.8% | ⚠️ Near Target |
| `trt_rebuilder.py` | 80.5% | ⚠️ Acceptable |
| `voice_cloner.py` | 78.8% | ⚠️ Below Target |
| `realtime_pipeline.py` | 75.4% | ❌ Needs Work |
| `singing_conversion_pipeline.py` | 74.6% | ❌ Needs Work |
| `streaming_pipeline.py` | 70.6% | ❌ Needs Work |
| `trt_streaming_pipeline.py` | 37.9% | ❌ Critical Gap |
| `trt_pipeline.py` | 23.2% | ❌ Critical Gap |
| `voice_identifier.py` | 0.0% | ❌ Critical Gap |
| `mean_flow_decoder.py` | 0.0% | ❌ Critical Gap |

### Inference Coverage Strategy

To reach **85% inference coverage**, focus on:

1. **Critical Gaps (0% coverage, 307 lines)**
   - Add tests for `mean_flow_decoder.py` (101 lines)
   - Add tests for `voice_identifier.py` (206 lines)

2. **Low Coverage TensorRT Modules (276 missing lines)**
   - Improve `trt_pipeline.py` from 23.2% to 70%+ (add 115 lines)
   - Improve `trt_streaming_pipeline.py` from 37.9% to 70%+ (add 45 lines)

3. **Below Target Modules (push to 85%+)**
   - `voice_cloner.py`: 78.8% → 85% (add 15 lines)
   - `realtime_pipeline.py`: 75.4% → 85% (add 22 lines)
   - `singing_conversion_pipeline.py`: 74.6% → 85% (add 23 lines)
   - `streaming_pipeline.py`: 70.6% → 85% (add 32 lines)

**Estimated Lines to Add: ~560 lines of test coverage**

## Coverage by Module Category

| Category | Modules | Total Lines | Covered | Coverage | Target | Gap |
|----------|---------|-------------|---------|----------|--------|-----|
| Inference | 18 | 2,939 | 2,012 | 68.4% | 85% | -16.6pp |
| Audio | 11 | 2,150 | 821 | 38.2% | 70% | -31.8pp |
| Web | 8 | 1,247 | 682 | 54.7% | 80% | -25.3pp |
| Database | 3 | 179 | 0 | 0.0% | 70% | -70.0pp |
| Evaluation | 3 | 392 | 86 | 21.9% | 70% | -48.1pp |
| Export | 2 | 263 | 97 | 36.9% | 70% | -33.1pp |
| Training | 5 | 842 | 548 | 65.1% | 75% | -9.9pp |
| Monitoring | 3 | 358 | 171 | 47.8% | 60% | -12.2pp |
| YouTube | 2 | 152 | 83 | 54.6% | 70% | -15.4pp |
| Other | 40 | 6,247 | 4,389 | 70.3% | 60% | +10.3pp ✅ |

## Estimated Work to Reach 80% Overall Coverage

**Current:** 8,889 / 14,769 lines = 60.2%
**Target:** 11,815 / 14,769 lines = 80.0%
**Need to Cover:** 2,926 additional lines

### Prioritized Coverage Plan

#### Phase 1: Database (Critical) - 179 lines
- `db/operations.py` - 116 lines
- `db/schema.py` - 60 lines
- `db/__init__.py` - 3 lines
- **Impact:** +1.2pp coverage

#### Phase 2: Audio Processing - ~500 lines
- `diarization_extractor.py` - 196 lines
- `speaker_matcher.py` - 220 lines
- Focus on critical user workflows
- **Impact:** +3.4pp coverage

#### Phase 3: Inference Gaps - ~560 lines
- Complete `mean_flow_decoder.py` - 101 lines
- Complete `voice_identifier.py` - 206 lines
- Improve TensorRT pipelines - 160 lines
- Improve streaming pipelines - 93 lines
- **Impact:** +3.8pp coverage

#### Phase 4: Web API - ~300 lines
- `speaker_api.py` - 185 missing lines
- `karaoke_manager.py` - 80 missing lines
- `job_manager.py` - partial improvement
- **Impact:** +2.0pp coverage

#### Phase 5: Evaluation & Export - ~400 lines
- `conversion_quality_analyzer.py` - 268 lines
- `tensorrt_engine.py` - 130 lines
- **Impact:** +2.7pp coverage

#### Phase 6: Training & Monitoring - ~300 lines
- Training edge cases and error paths
- Monitoring uncovered branches
- **Impact:** +2.0pp coverage

**Total Estimated Impact: +15.1pp** → 75.3% coverage

**Additional Optimizations Needed: ~687 lines** → 80% target

## Recommendations for Phase 6 Execution

When unblocked (after Phase 5 completion):

### Immediate Actions (Day 1)
1. ✅ Re-run pytest with coverage to get fresh baseline
   ```bash
   PYTHONNOUSERSITE=1 PYTHONPATH=src pytest --cov=src/auto_voice \
     --cov-report=html --cov-report=term --cov-report=json -v
   ```

2. ✅ Verify all prior phases' tests are passing
   - Phase 2: Audio Processing Tests
   - Phase 3: Database & Storage Tests
   - Phase 4: Web API Tests
   - Phase 5: E2E Integration Tests

3. ✅ Prioritize Database coverage (0% → 70%)
   - Write tests for `db/operations.py` (CRITICAL)
   - Write tests for `db/schema.py` (CRITICAL)
   - Use in-memory SQLite for speed

### Critical Path (Days 1-2)
4. ✅ Fill inference gaps to reach 85% target
   - Add tests for `mean_flow_decoder.py` (0% → 70%)
   - Add tests for `voice_identifier.py` (0% → 70%)
   - Improve TensorRT pipeline coverage

5. ✅ Fill audio processing gaps
   - Add tests for `diarization_extractor.py`
   - Add tests for `speaker_matcher.py`

### Optimization (Day 3)
6. ✅ Add remaining Web API tests
7. ✅ Add evaluation and export tests
8. ✅ Optimize slow tests
   - Cache model loading in fixtures
   - Use smaller audio clips
   - Mock expensive operations
   - Parallelize with pytest-xdist

### Documentation (Day 3)
9. ✅ Generate final coverage report
10. ✅ Create coverage summary document
11. ✅ Update CLAUDE.md with test patterns
12. ✅ Document remaining gaps with justification

## Blocking Issues

**Cannot Proceed Until:**
- ✗ AV-6w9: E2E Integration Tests - Phase 5 Complete (OPEN)
  - ✗ AV-a9j: Audio Processing Tests - Phase 2 Complete (OPEN)
  - ✗ AV-cht: Database & Storage Tests - Phase 3 Complete (OPEN)
  - ✗ AV-plm: Web API Tests - Phase 4 Complete (OPEN)

**Estimated Time Once Unblocked:** 3 days
- Day 1: Database + Inference gaps (reach 70%)
- Day 2: Audio + Web API gaps (reach 75-78%)
- Day 3: Remaining gaps + optimization + docs (reach 80%)

## Next Steps

1. **Wait for Phase 5 completion signal**
2. **Run fresh coverage analysis**
3. **Execute prioritized coverage plan**
4. **Generate final report**
5. **Update beads issues:**
   - Close AV-k7j (Coverage Report + Gap Analysis)
   - Close AV-pio (Coverage Report Generation Track)

---

**Report Status:** Preliminary (based on Feb 1 coverage data)
**Will Update:** When Phase 5 completes and fresh coverage run is performed
