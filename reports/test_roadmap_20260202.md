# Test Roadmap - Coverage Gap Analysis
**Date:** 2026-02-02
**Current Coverage:** 63% (9,467/15,063 lines)
**Target Coverage:** 80% (12,050 lines)
**Gap:** 2,583 lines needed

---

## Executive Summary

This roadmap provides a prioritized plan to increase test coverage from 63% to 80% by adding targeted tests for 30 modules currently below the 70% threshold.

### Coverage Targets by Module Category

| Category | Modules | Missing Lines | Target | Priority | Impact |
|----------|---------|---------------|--------|----------|--------|
| **Inference** | 5 | 542 | 85% | P0 | +3.6% |
| **Web API** | 8 | 1,606 | 80% | P0 | +10.7% |
| **Audio** | 7 | 673 | 70% | P1 | +4.5% |
| **Training** | 2 | 303 | 70% | P1 | +2.0% |
| **Export/GPU** | 3 | 248 | 70% | P2 | +1.6% |
| **Monitoring/Other** | 5 | 373 | 70% | P2 | +2.5% |
| **TOTAL** | **30** | **3,745** | **80%** | - | **~25%** |

**Note:** Total missing lines exceed gap because we only need to cover ~70% of missing lines to reach 80% target.

---

## Priority 0 (Critical) - 13 Modules - Required for 80% Target

### Inference Modules (Target: 85%) - 5 modules, 542 missing lines

| Module | Lines | Current | Missing | Target | Impact | Est. Tests |
|--------|-------|---------|---------|--------|--------|------------|
| `inference/trt_pipeline.py` | 246 | 20% | 196 | 85% | +1.3% | 15-20 |
| `inference/hq_svc_wrapper.py` | 209 | 31% | 145 | 85% | +1.0% | 12-15 |
| `inference/trt_streaming_pipeline.py` | 140 | 38% | 87 | 85% | +0.6% | 10-12 |
| `inference/singing_conversion_pipeline.py` | 134 | 57% | 58 | 85% | +0.4% | 8-10 |
| `inference/voice_cloner.py` | 170 | 67% | 56 | 85% | +0.4% | 6-8 |
| **Subtotal** | **899** | **39%** | **542** | **85%** | **+3.6%** | **51-65** |

**Test Focus:**
- TensorRT pipeline initialization and inference
- HQ-SVC model loading and conversion
- Streaming pipeline chunk processing
- Voice profile creation and embedding extraction
- Error handling for GPU OOM and invalid inputs

---

### Web API Modules (Target: 80%) - 8 modules, 1,606 missing lines

| Module | Lines | Current | Missing | Target | Impact | Est. Tests |
|--------|-------|---------|---------|--------|--------|------------|
| `web/api.py` | 2,026 | 49% | 1,030 | 80% | +6.8% | 40-50 |
| `web/karaoke_api.py` | 406 | 49% | 209 | 80% | +1.4% | 15-20 |
| `web/speaker_api.py` | 225 | 18% | 185 | 80% | +1.2% | 12-15 |
| `web/job_manager.py` | 160 | 38% | 100 | 80% | +0.7% | 8-10 |
| `web/karaoke_events.py` | 214 | 51% | 104 | 80% | +0.7% | 8-10 |
| `web/karaoke_manager.py` | 117 | 32% | 80 | 80% | +0.5% | 6-8 |
| `web/karaoke_session.py` | 201 | 64% | 72 | 80% | +0.5% | 5-7 |
| `web/voice_model_registry.py` | 84 | 68% | 27 | 80% | +0.2% | 3-5 |
| **Subtotal** | **3,433** | **47%** | **1,807** | **80%** | **+12.0%** | **97-125** |

**Test Focus:**
- All REST API endpoints (success + error cases)
- WebSocket event handlers (karaoke session management)
- Request validation and error responses
- File upload handling
- Job queue management
- Speaker profile API
- Model registry operations

---

## Priority 1 (High) - 9 Modules - Required for 70% Audio/Training Target

### Audio Processing Modules (Target: 70%) - 7 modules, 673 missing lines

| Module | Lines | Current | Missing | Target | Impact | Est. Tests |
|--------|-------|---------|---------|--------|--------|------------|
| `audio/speaker_matcher.py` | 220 | 15% | 186 | 70% | +1.2% | 12-15 |
| `audio/multi_artist_separator.py` | 194 | 25% | 146 | 70% | +1.0% | 10-12 |
| `audio/training_filter.py` | 138 | 13% | 120 | 70% | +0.8% | 8-10 |
| `audio/youtube_metadata.py` | 224 | 57% | 97 | 70% | +0.6% | 6-8 |
| `audio/file_organizer.py` | 192 | 53% | 90 | 70% | +0.6% | 6-8 |
| `audio/diarization_extractor.py` | 196 | 64% | 70 | 70% | +0.5% | 4-6 |
| `audio/separation.py` | 81 | 21% | 64 | 70% | +0.4% | 5-7 |
| **Subtotal** | **1,245** | **35%** | **773** | **70%** | **+5.1%** | **51-66** |

**Test Focus:**
- Speaker embedding similarity matching
- Multi-artist audio separation
- Training data filtering and validation
- YouTube metadata extraction and normalization
- File organization and naming
- Speaker diarization track extraction
- Vocal separation (mock demucs if not installed)

---

### Training Modules (Target: 70%) - 2 modules, 303 missing lines

| Module | Lines | Current | Missing | Target | Impact | Est. Tests |
|--------|-------|---------|---------|--------|--------|------------|
| `training/job_manager.py` | 544 | 44% | 303 | 70% | +2.0% | 20-25 |
| `profiles/api.py` | 197 | 47% | 104 | 70% | +0.7% | 8-10 |
| **Subtotal** | **741** | **45%** | **407** | **70%** | **+2.7%** | **28-35** |

**Test Focus:**
- Training job queue management
- Job status tracking and updates
- Error handling for failed jobs
- Profile API CRUD operations
- Profile validation and constraints

---

## Priority 2 (Medium) - 8 Modules - Nice to Have

### Export/GPU Modules (Target: 70%) - 3 modules, 248 missing lines

| Module | Lines | Current | Missing | Target | Impact | Est. Tests |
|--------|-------|---------|---------|--------|--------|------------|
| `export/tensorrt_engine.py` | 169 | 5% | 160 | 70% | +1.1% | 12-15 |
| `gpu/memory_manager.py` | 199 | 69% | 62 | 70% | +0.4% | 3-5 |
| `gpu/cuda_kernels.py` | 62 | 58% | 26 | 70% | +0.2% | 3-4 |
| **Subtotal** | **430** | **44%** | **248** | **70%** | **+1.6%** | **18-24** |

**Test Focus:**
- TensorRT engine build and optimization
- GPU memory allocation and cleanup
- CUDA kernel loading and execution
- Memory pool management

---

### Monitoring/YouTube Modules (Target: 70%) - 5 modules, 373 missing lines

| Module | Lines | Current | Missing | Target | Impact | Est. Tests |
|--------|-------|---------|---------|--------|--------|------------|
| `monitoring/quality_monitor.py` | 226 | 61% | 88 | 70% | +0.6% | 6-8 |
| `youtube/channel_scraper.py` | 98 | 41% | 58 | 70% | +0.4% | 5-7 |
| `youtube/downloader.py` | 91 | 38% | 56 | 70% | +0.4% | 5-7 |
| `web/audio_router.py` | 78 | 69% | 24 | 80% | +0.2% | 2-3 |
| **Subtotal** | **493** | **52%** | **226** | **70%** | **+1.5%** | **18-25** |

**Test Focus:**
- Quality metrics collection and reporting
- YouTube channel scraping and parsing
- Video download and conversion
- Audio routing and streaming

---

## Implementation Strategy

### Phase 1: Quick Wins (Week 1, Days 1-2)
**Goal:** +5% coverage (63% → 68%)
**Focus:** P0 inference modules + fix existing test failures

1. **Fix Test Failures** (194 failures/errors)
   - Install missing dependencies (demucs, local-attention)
   - Fix validation schema issues
   - Fix audio processing test bugs
   - **Impact:** +2% (enables existing tests)

2. **Inference Module Tests** (542 missing lines)
   - Add TensorRT pipeline tests (mock TRT if needed)
   - Add HQ-SVC wrapper tests (mock local_attention)
   - Add streaming pipeline tests
   - **Impact:** +2.5% (covers P0 inference gaps)

3. **Voice Cloner Tests** (56 missing lines)
   - Test profile creation workflow
   - Test embedding extraction
   - Test error cases
   - **Impact:** +0.4%

**Deliverables:** 51-65 new tests, ~598 lines covered

---

### Phase 2: Web API Coverage (Week 1, Days 3-5)
**Goal:** +8% coverage (68% → 76%)
**Focus:** P0 web API modules

1. **Core API Endpoints** (`web/api.py` - 1,030 missing lines)
   - Test all REST endpoints
   - Test request validation
   - Test error responses
   - Test file uploads
   - **Impact:** +5% (partial coverage, focus on critical paths)

2. **Karaoke System** (565 missing lines)
   - Test WebSocket events
   - Test session management
   - Test job queue
   - **Impact:** +2%

3. **Speaker API** (185 missing lines)
   - Test speaker CRUD operations
   - Test speaker search and filtering
   - **Impact:** +1%

**Deliverables:** 97-125 new tests, ~1,200 lines covered

---

### Phase 3: Audio & Training (Week 2, Days 1-3)
**Goal:** +4% coverage (76% → 80%)
**Focus:** P1 audio and training modules

1. **Audio Processing** (673 missing lines)
   - Test speaker matching
   - Test multi-artist separation
   - Test training data filtering
   - Test metadata extraction
   - **Impact:** +3%

2. **Training Management** (303 missing lines)
   - Test job manager
   - Test profile API
   - **Impact:** +1%

**Deliverables:** 79-101 new tests, ~600 lines covered

---

### Phase 4: Optimization (Week 2, Days 4-5)
**Goal:** Maintain 80%+, optimize performance
**Focus:** Test suite optimization and maintenance

1. **Performance Optimization**
   - Enable pytest-xdist parallel execution
   - Reduce test runtime to <15 minutes
   - Cache expensive fixtures

2. **Quality Improvements**
   - Review and refactor brittle tests
   - Add missing edge cases
   - Improve test documentation

---

## Estimated Effort

| Phase | Days | Tests | Lines | Coverage Gain |
|-------|------|-------|-------|---------------|
| Phase 1: Quick Wins | 2 | 51-65 | 598 | +5% (63% → 68%) |
| Phase 2: Web API | 3 | 97-125 | 1,200 | +8% (68% → 76%) |
| Phase 3: Audio/Training | 3 | 79-101 | 600 | +4% (76% → 80%) |
| Phase 4: Optimization | 2 | 0-20 | 200 | +2% (80% → 82%) |
| **TOTAL** | **10 days** | **227-311** | **2,598** | **+19% (63% → 82%)** |

---

## Test Templates and Patterns

### Inference Module Test Pattern
```python
@pytest.mark.cuda
def test_pipeline_inference():
    """Test pipeline converts audio correctly"""
    pipeline = Pipeline(config)
    audio = generate_test_audio(duration=5.0)

    result = pipeline.convert(audio, profile_id='test')

    assert result.shape[0] > 0
    assert not torch.isnan(result).any()
    assert result.device.type == 'cuda'

@pytest.mark.parametrize("error_case", ["missing_adapter", "invalid_audio"])
def test_pipeline_error_handling(error_case):
    """Test pipeline handles errors gracefully"""
    # Test error paths
```

### Web API Test Pattern
```python
def test_api_endpoint_success(client):
    """Test endpoint returns 200 with valid data"""
    response = client.post('/api/v1/endpoint', json={...})

    assert response.status_code == 200
    assert 'result' in response.json

def test_api_endpoint_validation(client):
    """Test endpoint validates input"""
    response = client.post('/api/v1/endpoint', json={})

    assert response.status_code == 400
    assert 'error' in response.json
```

### Audio Processing Test Pattern
```python
def test_audio_processing():
    """Test audio processor with synthetic audio"""
    audio = generate_sine_wave(freq=440, duration=5.0)

    result = process_audio(audio, sr=44100)

    assert len(result) > 0
    assert result.dtype == np.float32
    assert not np.isnan(result).any()
```

---

## Success Criteria

### Coverage Targets
- [ ] Overall coverage ≥ 80%
- [ ] Inference coverage ≥ 85%
- [ ] Web API coverage ≥ 80%
- [ ] Audio coverage ≥ 70%
- [ ] Training coverage ≥ 70%

### Test Quality
- [ ] All tests passing (0 failures/errors)
- [ ] Test runtime < 15 minutes
- [ ] No flaky tests
- [ ] All tests use generated data (no file dependencies)
- [ ] All tests documented with clear descriptions

### Documentation
- [ ] CLAUDE.md updated with test patterns
- [ ] All new tests follow existing conventions
- [ ] Coverage report generated and reviewed
- [ ] Roadmap progress tracked in beads

---

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing dependencies (demucs, TRT) | High | Mock dependencies or mark tests as optional |
| Flaky integration tests | Medium | Use deterministic fixtures, avoid time.sleep() |
| Long test runtime | Medium | Enable parallel execution, optimize fixtures |
| TensorRT not available on platform | High | Mock TensorRT engine, test logic only |
| Complex web API validation | Medium | Start with simple success cases, add complexity |

---

## Next Actions

1. **Create beads issues** for each module category (6 issues)
2. **Assign priorities** based on this roadmap
3. **Link all issues** to parent epic AV-w3a
4. **Begin Phase 1** with test failure fixes and inference tests
5. **Track progress** via beads status updates

---

## Appendix: Full Module List

### All Modules Below 70% Coverage (30 modules)

| Module | Statements | Missing | Current | Target | Priority |
|--------|------------|---------|---------|--------|----------|
| web/api.py | 2,026 | 1,030 | 49% | 80% | P0 |
| inference/trt_pipeline.py | 246 | 196 | 20% | 85% | P0 |
| audio/speaker_matcher.py | 220 | 186 | 15% | 70% | P1 |
| web/speaker_api.py | 225 | 185 | 18% | 80% | P0 |
| export/tensorrt_engine.py | 169 | 160 | 5% | 70% | P2 |
| audio/multi_artist_separator.py | 194 | 146 | 25% | 70% | P1 |
| inference/hq_svc_wrapper.py | 209 | 145 | 31% | 85% | P0 |
| audio/training_filter.py | 138 | 120 | 13% | 70% | P1 |
| profiles/api.py | 197 | 104 | 47% | 70% | P1 |
| web/karaoke_events.py | 214 | 104 | 51% | 80% | P0 |
| web/job_manager.py | 160 | 100 | 38% | 80% | P0 |
| audio/youtube_metadata.py | 224 | 97 | 57% | 70% | P1 |
| audio/file_organizer.py | 192 | 90 | 53% | 70% | P1 |
| monitoring/quality_monitor.py | 226 | 88 | 61% | 70% | P2 |
| inference/trt_streaming_pipeline.py | 140 | 87 | 38% | 85% | P0 |
| web/karaoke_manager.py | 117 | 80 | 32% | 80% | P0 |
| web/karaoke_session.py | 201 | 72 | 64% | 80% | P0 |
| audio/diarization_extractor.py | 196 | 70 | 64% | 70% | P1 |
| audio/separation.py | 81 | 64 | 21% | 70% | P1 |
| gpu/memory_manager.py | 199 | 62 | 69% | 70% | P2 |
| youtube/channel_scraper.py | 98 | 58 | 41% | 70% | P2 |
| inference/singing_conversion_pipeline.py | 134 | 58 | 57% | 85% | P0 |
| inference/voice_cloner.py | 170 | 56 | 67% | 85% | P0 |
| youtube/downloader.py | 91 | 56 | 38% | 70% | P2 |
| web/voice_model_registry.py | 84 | 27 | 68% | 80% | P0 |
| gpu/cuda_kernels.py | 62 | 26 | 58% | 70% | P2 |
| web/audio_router.py | 78 | 24 | 69% | 80% | P2 |
| web/karaoke_api.py | 406 | 209 | 49% | 80% | P0 |
| training/job_manager.py | 544 | 303 | 44% | 70% | P1 |
| training/training_scheduler.py | 83 | 0 | 100% | 70% | P1 |

---

**Report Generated:** 2026-02-02
**Generated By:** Coverage Gap Analyzer
**Next Review:** After Phase 1 completion
