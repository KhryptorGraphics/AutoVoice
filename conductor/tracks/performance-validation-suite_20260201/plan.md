# Implementation Plan: Performance Validation Suite

**Track ID:** performance-validation-suite_20260201
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-01
**Status:** [x] Phase 1 & 2 Complete

## Overview

Create comprehensive performance benchmarks for all 4 voice conversion pipelines. Measure RTF, latency, memory usage, and quality metrics. Generate comparison report to guide users and validate SLAs.

## Phase 1: Benchmark Infrastructure

Create automated benchmark runner and metrics collection.

### Tasks

- [x] Task 1.1: Create `scripts/performance_validation.py` scaffold
  - Argument parsing (pipeline type, audio file, profile)
  - Environment setup (GPU warmup, cache clearing)
  - Results output (JSON, CSV, console)

- [x] Task 1.2: Implement metrics collection utilities
  - RTF calculator (processing time / audio duration)
  - Memory profiler (peak GPU/CPU allocation)
  - Latency timer (end-to-end, chunk-level)
  - Quality metrics wrapper (MCD, speaker similarity)

- [x] Task 1.3: Create test fixtures
  - 5s test clip (quick validation) - uses --duration 5
  - 30s test clip (standard benchmark) - uses --duration 30
  - 3min test clip (long audio stress test) - uses full audio files
  - William/Conor adapter paths - tests/quality_samples/*.wav

- [x] Task 1.4: Implement benchmark runner class
  - Pipeline initialization (lazy loading)
  - Warmup run (exclude from measurements)
  - Multiple trials (median of 3 runs)
  - Error handling (GPU OOM, model missing)

- [x] Task 1.5: Add progress display
  - Console output (pipeline, progress bar)
  - Estimated time remaining
  - Real-time metrics (RTF, memory)

### Verification

- [x] Benchmark script runs without errors
- [x] Metrics collected correctly
- [x] Progress display works
- [x] JSON output valid

## Phase 2: Pipeline Benchmarks (All 4 Types)

Benchmark each pipeline type with standard test audio.

### Tasks

- [x] Task 2.1: Benchmark `realtime` pipeline
  - Run with 30s test clip
  - Measure RTF, chunk latency, GPU memory
  - Calculate MCD, speaker similarity
  - **Results: RTF=0.456, Latency=33ms, Memory=0.47GB - PASS**

- [x] Task 2.2: Benchmark `quality` pipeline (CoMoSVC consistency model)
  - Run with 30s test clip
  - Measure RTF, GPU memory
  - Calculate MCD, speaker similarity
  - **Results: RTF=0.206, Memory=1.86GB - PASS**

- [x] Task 2.3: Benchmark `quality_seedvc` pipeline (10-step DiT-CFM)
  - Run with 30s test clip
  - Measure RTF, GPU memory
  - **Results: RTF=0.742, Memory=3.25GB, Similarity=0.983 - PASS**

- [x] Task 2.4: Benchmark `realtime_meanvc` pipeline
  - Run with 30s test clip
  - Measure streaming chunk latency
  - Calculate RTF, GPU memory
  - **Results: RTF=2.765 (CPU mode), Memory=0.01GB - NEEDS OPTIMIZATION**
  - Note: MeanVC runs on CPU; needs GPU acceleration for realtime

- [x] Task 2.5: Generate comparison table
  - All 4 pipelines side-by-side
  - Columns: RTF, latency, GPU mem, MCD, similarity
  - Markdown table for docs - reports/BENCHMARK_RESULTS.md
  - JSON export for analysis - reports/benchmark_results.json

### Verification

- [x] All 4 pipelines benchmarked
- [x] Comparison table generated
- [x] Results reproducible (variance <10%)
- [x] Targets met or gaps identified (MeanVC needs GPU acceleration)

## Phase 3: Memory Profiling

Measure GPU/CPU memory usage under various conditions.

### Tasks

- [ ] Task 3.1: Profile single pipeline memory
  - Idle memory (model loaded, no inference)
  - Peak memory (during inference)
  - Sustained memory (10 conversions)
  - Cleanup verification (memory released)

- [ ] Task 3.2: Profile concurrent pipelines
  - Load 2 pipelines simultaneously
  - Measure total GPU memory
  - Test conversion with both active
  - Verify 64GB limit not exceeded

- [ ] Task 3.3: Test memory leak detection
  - Run 10 consecutive conversions
  - Monitor GPU memory after each
  - Verify no growth (leak detection)
  - Test cleanup on error

- [ ] Task 3.4: Profile model loading overhead
  - Measure first load time (cold start)
  - Measure cached load time (warm start)
  - Amortize loading cost (10+ conversions)
  - Document model size (GB on disk)

- [ ] Task 3.5: Generate memory report
  - Memory usage by pipeline (table)
  - Concurrent capacity (max sessions)
  - Recommendations for production

### Verification

- [ ] Memory profiling complete
- [ ] No memory leaks detected
- [ ] Concurrent capacity determined
- [ ] 64GB limit respected

## Phase 4: Latency Analysis

Measure end-to-end and component-level latency.

### Tasks

- [ ] Task 4.1: Measure API endpoint latency
  - POST `/api/v1/convert/song` response time
  - Include model loading (amortized)
  - Exclude queue wait time
  - Report p50, p90, p99

- [ ] Task 4.2: Measure streaming chunk latency
  - WebSocket `audioChunk` → `convertedChunk` time
  - Realtime pipelines only
  - Target: <100ms p90
  - Identify bottlenecks (encoding, inference, vocoding)

- [ ] Task 4.3: Profile component latency breakdown
  - Encoder: feature extraction time
  - Decoder: mel generation time
  - Vocoder: audio synthesis time
  - F0 extraction: pitch estimation time

- [ ] Task 4.4: Test long audio scaling
  - 5s, 30s, 3min clips
  - Verify linear scaling (RTF constant)
  - Identify memory constraints
  - Test sliding window attention (if applicable)

- [ ] Task 4.5: Generate latency report
  - Latency by pipeline (table)
  - Component breakdown (pie chart)
  - Bottleneck identification

### Verification

- [ ] Latency measurements complete
- [ ] Realtime pipelines <100ms p90
- [ ] Linear scaling verified
- [ ] Bottlenecks identified

## Phase 5: Quality Validation

Validate quality metrics across pipelines and test conditions.

### Tasks

- [ ] Task 5.1: Calculate MCD for all pipelines
  - Use reference audio (ground truth)
  - Calculate mel-cepstral distortion
  - Compare to baselines
  - Verify quality targets met

- [ ] Task 5.2: Calculate speaker similarity
  - Extract embeddings (source, target, converted)
  - Compute cosine similarity
  - Target: >0.80 for all pipelines
  - Identify quality regressions

- [ ] Task 5.3: Shortcut flow quality comparison
  - 2-step vs 10-step CFM
  - Measure quality degradation (MCD delta)
  - Determine acceptable threshold
  - Document tradeoff recommendation

- [ ] Task 5.4: Audio quality subjective tests (optional)
  - Generate sample outputs (all pipelines)
  - Manual listening comparison
  - Note artifacts (robotic, muffled, glitches)
  - Correlate with objective metrics

- [ ] Task 5.5: Generate quality report
  - Quality by pipeline (table)
  - Shortcut flow tradeoff analysis
  - Recommendations for use cases

### Verification

- [ ] Quality metrics calculated
- [ ] Targets met or gaps identified
- [ ] Shortcut flow validated
- [ ] Quality report generated

## Phase 6: Concurrent Load Testing

Test performance under concurrent load.

### Tasks

- [ ] Task 6.1: Test concurrent karaoke sessions
  - Spawn 5 WebSocket sessions
  - Send audio chunks simultaneously
  - Measure latency degradation
  - Verify no crashes or deadlocks

- [ ] Task 6.2: Test concurrent conversion jobs
  - Queue 10 conversion jobs
  - Measure queue processing rate
  - Test job prioritization
  - Verify fair scheduling

- [ ] Task 6.3: Test mixed load (karaoke + conversion)
  - 3 karaoke sessions + 5 conversion jobs
  - Measure resource contention
  - Verify karaoke latency maintained
  - Document capacity limits

- [ ] Task 6.4: Test GPU saturation
  - Max concurrent sessions until OOM
  - Measure graceful degradation
  - Test error handling (queue full)
  - Document production limits

- [ ] Task 6.5: Generate load testing report
  - Concurrent capacity (max sessions)
  - Latency under load (degradation)
  - Recommendations for scaling

### Verification

- [ ] Load tests complete
- [ ] Capacity limits documented
- [ ] No crashes under load
- [ ] Graceful degradation verified

## Phase 7: Report Generation and Documentation

Create final benchmark report and update docs.

### Tasks

- [ ] Task 7.1: Generate comprehensive benchmark report
  - `reports/performance_validation_20260201.md`
  - Executive summary (key findings)
  - Comparison table (all pipelines)
  - Memory profiling results
  - Latency analysis
  - Quality validation
  - Load testing results

- [ ] Task 7.2: Create pipeline selection guide
  - `docs/pipeline-comparison.md`
  - Use case recommendations (karaoke, offline, quality)
  - Quality-speed tradeoff matrix
  - Memory requirements
  - User-facing language (no jargon)

- [ ] Task 7.3: Update CLAUDE.md
  - Add benchmark results
  - Document performance targets
  - Add regression test patterns

- [x] Task 7.4: Create CI integration tests
  - `tests/test_performance_benchmarks.py`
  - Smoke tests (quick validation)
  - Regression thresholds (RTF, memory)
  - Mark as `@pytest.mark.performance`
  - **Result: 13 tests passing, 3 skipped (MeanVC deps)**

- [x] Task 7.5: Export data for analysis
  - JSON export (structured data) - `reports/performance_report.json`
  - Markdown report - `reports/BENCHMARK_RESULTS.md`
  - Git-track in `reports/`

### Verification

- [x] Benchmark report complete - `reports/BENCHMARK_RESULTS.md`
- [ ] Pipeline selection guide published
- [ ] CLAUDE.md updated
- [x] CI tests integrated - `tests/test_performance_benchmarks.py`

## Final Verification

- [ ] All acceptance criteria met
- [ ] All 4 pipelines benchmarked
- [ ] Reports published
- [ ] Documentation updated
- [ ] Ready for review

---

**Estimated Timeline:**
- Phase 1: 0.5 day (infrastructure)
- Phase 2: 0.5 day (pipeline benchmarks)
- Phase 3: 0.25 day (memory profiling)
- Phase 4: 0.25 day (latency analysis)
- Phase 5: 0.25 day (quality validation)
- Phase 6: 0.25 day (load testing)
- Phase 7: 0.5 day (reports and docs)
- **Total:** 2.5 days

**Dependencies:**
- Upstream: MeanVC pipeline (sota-innovations Phase 4) - COMPLETE
- Upstream: Shortcut flow (sota-innovations Phase 2) - COMPLETE
- Upstream: Trained adapters (William, Conor)

**Blocks:**
- SLA commitments
- User pipeline selection
- Optimization prioritization

---

_Generated by Conductor._
