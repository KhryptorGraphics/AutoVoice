# Specification: Comprehensive Testing Coverage

**Track ID:** comprehensive-testing-coverage_20260201
**Type:** test
**Priority:** P0 (BLOCKING DEPLOYMENT)
**Created:** 2026-02-01
**Status:** Pending

## Problem Statement

95% of source modules (73 out of 77) lack dedicated unit tests. Critical inference pipelines, audio processing modules, database operations, and web APIs are untested. This creates severe production risk and prevents confident deployment.

**Current Coverage:**
- Total source modules: 77
- Modules with tests: 4
- Untested modules: 73
- Test coverage: ~5%

**Risk Impact:**
- Cannot verify voice conversion quality
- Cannot guarantee adapter loading works
- Cannot validate database integrity
- Cannot ensure API error handling
- Production failures will cause user-facing crashes

## Acceptance Criteria

### Must Have (P0)

1. **Inference Pipeline Tests (CRITICAL)**
   - [ ] `adapter_bridge.py` - LoRA → Seed-VC integration tests
   - [ ] `pipeline_factory.py` - Pipeline routing and lazy loading
   - [ ] `seed_vc_pipeline.py` - Quality pipeline end-to-end
   - [ ] `meanvc_pipeline.py` - Realtime streaming pipeline
   - [ ] `hq_svc_wrapper.py` - Enhancement pipeline
   - [ ] `model_manager.py` - Model loading and caching

2. **Audio Processing Tests**
   - [ ] `diarization_extractor.py` - Speaker isolation accuracy
   - [ ] `speaker_matcher.py` - Speaker identification correctness
   - [ ] `separation.py` - Vocal extraction quality
   - [ ] `youtube_downloader.py` - Download and format handling

3. **Database Tests**
   - [ ] `db/operations.py` - CRUD operation correctness
   - [ ] `db/schema.py` - Data model validation
   - [ ] `db/session.py` - Connection lifecycle

4. **Web API Tests (60+ endpoints)**
   - [ ] All `/api/v1/convert/*` endpoints
   - [ ] All `/api/v1/voice/*` endpoints
   - [ ] All `/api/v1/training/*` endpoints
   - [ ] WebSocket events (karaoke, training)

5. **Integration Tests (E2E Flows)**
   - [ ] Train profile → Load adapter → Convert song
   - [ ] Download YouTube → Diarize → Auto-create profile → Train
   - [ ] Upload samples → Train → Convert with each pipeline
   - [ ] Karaoke session with profile switching

### Should Have (P1)

6. **Evaluation Module Tests**
   - [ ] `benchmark_runner.py` - Benchmark execution
   - [ ] `quality_metrics.py` - MCD, speaker similarity calculations
   - [ ] `performance_profiler.py` - RTF and latency measurement

7. **Export Module Tests**
   - [ ] `tensorrt_engine.py` - TensorRT engine building
   - [ ] `onnx_export.py` - ONNX export correctness

8. **GPU Module Tests**
   - [ ] `memory_manager.py` - GPU allocation and cleanup
   - [ ] `latency_profiler.py` - Timing accuracy

### Nice to Have (P2)

9. **Training Module Tests**
   - [ ] `trainer.py` - Training loop correctness
   - [ ] `job_manager.py` - Job queue and status
   - [ ] `training_scheduler.py` - Concurrent job handling

10. **Coverage Metrics**
    - [ ] Line coverage > 80% for inference modules
    - [ ] Line coverage > 70% for audio modules
    - [ ] Line coverage > 60% overall

## Test Strategy

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies (models, GPU, filesystem)
- Fast execution (<30s per module)
- Use fixtures for common test data

### Integration Tests
- Test component interactions
- Use real models (lightweight versions where possible)
- Test error propagation between layers
- Validate data flow end-to-end

### E2E Tests
- Test complete user workflows
- Use real audio samples (5-10s clips)
- Verify output quality (not just shape/dtype)
- Test both success and failure paths

## Test Infrastructure

### Fixtures Required
- `tests/fixtures/audio/` - 5s clips (vocals, instrumental, mixed)
- `tests/fixtures/adapters/` - Minimal valid LoRA adapters
- `tests/fixtures/models/` - Lightweight model checkpoints
- `tests/fixtures/profiles/` - Test voice profiles with embeddings

### Mocking Strategy
- Mock GPU when not required for correctness
- Mock filesystem for path validation tests
- Mock network for YouTube download tests
- Use real models for inference accuracy tests

### Performance Constraints
- Smoke tests: <30s total
- Fast tests: <5min total
- Full suite: <20min total
- E2E tests marked `@pytest.mark.slow`

## Success Metrics

1. **Coverage Increase:**
   - From 5% → 80%+ coverage
   - All inference modules tested
   - All API endpoints tested

2. **Defect Detection:**
   - Catch at least 5 real bugs during implementation
   - Prevent regression in future changes

3. **Confidence:**
   - Can run full test suite before deployment
   - CI/CD pipeline can validate PRs
   - Safe to refactor code with test coverage

## Out of Scope

- Performance benchmarking (covered by `performance-validation-suite_20260201`)
- Load testing (deferred)
- Security testing (deferred)
- Browser UI testing (deferred)

## Dependencies

**Upstream:**
- None (can start immediately)

**Downstream:**
- Blocks production deployment
- Enables confident refactoring

## Estimated Effort

- **Size:** Large
- **Duration:** 3-4 days with 1 agent
- **Complexity:** Medium (straightforward but volume-heavy)

## Risks

1. **Test data requirements:** Need representative audio samples
   - Mitigation: Use existing test_audio/ samples

2. **Model loading overhead:** Tests may be slow with real models
   - Mitigation: Cache models in fixtures, use lightweight versions

3. **GPU availability:** Some tests require GPU
   - Mitigation: Mark with `@pytest.mark.cuda`, skip if unavailable

## References

- Current test files: `tests/test_*.py` (75 files)
- Untested modules list: See Phase 1-4 tasks
- Test markers: `pytest.ini` (smoke, cuda, slow, integration)
