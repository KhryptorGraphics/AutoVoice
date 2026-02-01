# Specification: Performance Validation Suite

**Track ID:** performance-validation-suite_20260201
**Type:** test
**Priority:** P1 (BLOCKING SLAs)
**Created:** 2026-02-01
**Status:** Pending

## Problem Statement

Performance benchmarks are incomplete. We have partial data for Seed-VC and HQ-SVC, but lack comprehensive measurements across all 4 pipeline types. Cannot guarantee latency SLAs, memory usage constraints, or quality-speed tradeoffs without complete validation.

**Current Benchmark Coverage:**
- ✅ Seed-VC quality pipeline: RTF 1.981, MCD 183.93, GPU 3.49GB
- ✅ HQ-SVC enhancement: RTF 0.102, super-resolution working
- ✅ Realtime pipeline: RTF 0.475, latency ~80ms
- ❌ MeanVC streaming: No measurements
- ❌ Shortcut flow matching: No quality comparison (2-step vs 10-step)
- ❌ Concurrent sessions: No load testing
- ❌ Memory profiling: Incomplete data

**Impact Without Validation:**
- Cannot promise <100ms latency for karaoke
- Cannot guarantee 64GB GPU fits all pipelines
- Cannot advise users on pipeline selection
- Risk of production performance degradation

## Acceptance Criteria

### Must Have (P0)

1. **Pipeline Benchmarks (All 4 Types)**
   - [ ] `realtime` - ContentVec + HiFiGAN baseline
   - [ ] `quality` - Seed-VC with 10-step CFM
   - [ ] `quality_seedvc` - Seed-VC shortcut (2-step CFM)
   - [ ] `realtime_meanvc` - MeanVC streaming

   **Metrics per Pipeline:**
   - Real-time factor (RTF)
   - Chunk latency (ms)
   - GPU memory peak (GB)
   - MCD (Mel-Cepstral Distortion)
   - Speaker similarity (cosine)
   - Processing time (seconds)

2. **Quality-Speed Tradeoff Matrix**
   - [ ] Generate comparison table (RTF vs MCD vs similarity)
   - [ ] Identify sweet spots (best quality/speed balance)
   - [ ] Document use case recommendations

3. **Memory Profiling**
   - [ ] Measure GPU memory per pipeline (idle, peak, sustained)
   - [ ] Measure CPU memory usage
   - [ ] Test concurrent pipelines (2-3 simultaneous)
   - [ ] Verify no memory leaks (10+ conversions)

4. **Latency Validation**
   - [ ] Measure end-to-end latency (API request → response)
   - [ ] Measure chunk processing latency (karaoke mode)
   - [ ] Validate <100ms target for realtime pipelines
   - [ ] Identify bottlenecks (model loading, inference, I/O)

5. **GPU Utilization Verification**
   - [ ] Confirm GPU offload working (not CPU fallback)
   - [ ] Measure GPU vs CPU time ratio
   - [ ] Test TensorRT optimizations (if applicable)
   - [ ] Validate CUDA kernel usage

### Should Have (P1)

6. **Concurrent Session Testing**
   - [ ] Test 5 concurrent karaoke sessions
   - [ ] Test 10 concurrent conversion jobs
   - [ ] Measure performance degradation under load
   - [ ] Test queue saturation behavior

7. **Shortcut Flow Validation**
   - [ ] Compare 2-step vs 10-step CFM quality (MCD)
   - [ ] Measure speedup (2.83x claimed)
   - [ ] Determine quality threshold acceptability

8. **Long Audio Handling**
   - [ ] Test 3min song conversion
   - [ ] Test 10min song conversion
   - [ ] Verify no memory explosion
   - [ ] Measure linear scaling

### Nice to Have (P2)

9. **Audio Quality Metrics**
   - [ ] PESQ (Perceptual Evaluation of Speech Quality)
   - [ ] STOI (Short-Time Objective Intelligibility)
   - [ ] Spectrogram analysis (visual comparison)

10. **Benchmark Report Generation**
    - [ ] Automated HTML report with charts
    - [ ] CSV export for analysis
    - [ ] Comparison graphs (RTF vs quality)
    - [ ] Historical tracking (regression detection)

## Benchmark Methodology

### Test Audio
- **Source:** 3 test clips (5s, 30s, 3min)
- **Profiles:** William, Conor (trained adapters)
- **Conversions:** William → Conor, Conor → William
- **Repetitions:** 3 runs per configuration (median reported)

### Environment
- **Platform:** Jetson Thor (CUDA 13.0, SM 11.0)
- **GPU:** 64GB available
- **Isolation:** No concurrent jobs during benchmarks
- **Warmup:** 1 conversion run before measurements

### Metrics Collection

**Real-Time Factor (RTF):**
```
RTF = processing_time / audio_duration
RTF < 1.0 = faster than realtime
RTF > 1.0 = slower than realtime
```

**Mel-Cepstral Distortion (MCD):**
- Lower is better (typical range: 4-8 dB)
- Measures spectral difference from target

**Speaker Similarity:**
- Cosine similarity of embeddings
- Range: 0.0-1.0 (higher is better)
- Target: >0.80 for good conversion

**GPU Memory:**
- Peak allocation during inference
- Measured with `torch.cuda.max_memory_allocated()`

**Latency:**
- Measured with `time.perf_counter()`
- Includes model loading (amortized)

## Performance Targets

### Realtime Pipelines (karaoke)
- RTF: <0.5
- Chunk latency: <100ms
- GPU memory: <8GB
- Quality: MCD <10 dB

### Quality Pipelines (offline)
- RTF: <2.0
- GPU memory: <16GB
- Quality: MCD <6 dB
- Speaker similarity: >0.85

### Shortcut Flow (quality_seedvc)
- RTF: <1.0 (2-3x faster than quality)
- Quality: MCD <7 dB (slight degradation acceptable)
- GPU memory: Same as quality (~4GB)

### MeanVC Streaming
- RTF: <0.5
- Chunk latency: <80ms
- GPU memory: <6GB
- Quality: MCD <8 dB

## Out of Scope

- Network latency benchmarking (focus on compute)
- Browser/frontend performance testing
- Database query performance (covered by testing track)
- CUDA kernel micro-benchmarks (if needed, separate track)

## Dependencies

**Upstream:**
- `sota-innovations_20260131` Phase 4 (MeanVC) - COMPLETE
- `sota-innovations_20260131` Phase 2 (Shortcut) - COMPLETE
- Trained adapters (William, Conor)

**Downstream:**
- Blocks SLA commitments
- Informs user documentation
- Guides optimization priorities

## Estimated Effort

- **Size:** Medium
- **Duration:** 2 days with 1 agent
- **Complexity:** Low (measurement-focused, not implementation)

## Deliverables

1. **Benchmark Script:** `scripts/performance_validation.py`
   - Automated benchmark runner
   - JSON output for CI integration
   - Console progress display

2. **Results Report:** `reports/performance_validation.md`
   - Comparison table (all pipelines)
   - Quality-speed tradeoff analysis
   - Memory profiling results
   - Recommendations

3. **Test Suite:** `tests/test_performance_benchmarks.py`
   - Regression tests (RTF, memory)
   - CI integration (smoke tests only)
   - Performance threshold validation

4. **Documentation Update:** `docs/pipeline-comparison.md`
   - User-facing pipeline selection guide
   - Use case recommendations
   - Performance expectations

## Success Metrics

1. **Data Completeness:**
   - All 4 pipelines benchmarked
   - All metrics collected
   - Reproducible results (variance <10%)

2. **Actionable Insights:**
   - Clear pipeline recommendations
   - Identified bottlenecks
   - Optimization opportunities documented

3. **Validation:**
   - Performance targets met or gaps identified
   - No memory leaks detected
   - GPU utilization confirmed

## Risks

1. **Hardware variability:** Thor performance may differ from other platforms
   - Mitigation: Document platform in report

2. **Model loading overhead:** First run slower than subsequent
   - Mitigation: Warmup run + amortized timing

3. **Benchmark data storage:** Results need version control
   - Mitigation: JSON format, git-tracked in `reports/`

## References

- Existing benchmark data: `tests/quality_samples/outputs/quality_report.json`
- RTF calculation: `src/auto_voice/evaluation/performance_profiler.py`
- Quality metrics: `src/auto_voice/evaluation/quality_metrics.py`
- Memory profiling: `src/auto_voice/gpu/memory_manager.py`
