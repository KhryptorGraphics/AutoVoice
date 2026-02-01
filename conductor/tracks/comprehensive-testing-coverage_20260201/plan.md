# Implementation Plan: Comprehensive Testing Coverage

**Track ID:** comprehensive-testing-coverage_20260201
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-01
**Status:** [ ] Pending

## Overview

Add unit tests, integration tests, and E2E tests for 73 untested modules. Focus on inference pipelines (P0), audio processing, database, and web APIs. Target 80%+ code coverage to enable confident deployment.

## Phase 1: Inference Pipeline Tests (P0 - CRITICAL)

Test all voice conversion pipelines and adapter loading.

### Tasks

- [ ] Task 1.1: Test `adapter_bridge.py` - LoRA → Seed-VC integration
  - Load valid/invalid adapters
  - Verify embedding injection into Seed-VC
  - Test adapter format validation
  - Test error handling (missing files, corrupt adapters)

- [ ] Task 1.2: Test `pipeline_factory.py` - Pipeline routing
  - Test lazy loading (models not loaded until first use)
  - Test pipeline type routing (realtime, quality, quality_seedvc, realtime_meanvc)
  - Test singleton behavior
  - Test caching and model reuse

- [ ] Task 1.3: Test `seed_vc_pipeline.py` - Quality pipeline
  - Test inference with valid audio (5s clip)
  - Verify output shape, sample rate (44.1kHz)
  - Test F0 conditioning correctness
  - Test speaker embedding application
  - Test error handling (GPU OOM, invalid input)

- [ ] Task 1.4: Test `meanvc_pipeline.py` - Realtime streaming
  - Test streaming chunk processing (512 samples)
  - Verify chunk latency <100ms
  - Test crossfade continuity
  - Test speaker switching mid-stream

- [ ] Task 1.5: Test `hq_svc_wrapper.py` - Enhancement pipeline
  - Test 22kHz → 44.1kHz super-resolution
  - Verify no artifacts introduced
  - Test RTF <0.2 (enhancement is fast)
  - Test combined Seed-VC → HQ-SVC flow

- [ ] Task 1.6: Test `model_manager.py` - Model loading
  - Test model caching (load once, reuse)
  - Test model unloading (GPU cleanup)
  - Test concurrent load requests
  - Test error handling (missing checkpoint)

### Verification

- [ ] All inference tests pass
- [ ] Tests run in <5min total
- [ ] Coverage >80% for inference/ directory

## Phase 2: Audio Processing Tests

Test speaker diarization, separation, and YouTube download.

### Tasks

- [ ] Task 2.1: Test `diarization_extractor.py` - Speaker isolation
  - Test segment extraction from timestamps
  - Verify segment audio quality (no clipping)
  - Test multiple speakers (2-3 speakers)
  - Test edge cases (overlapping speech, silence)

- [ ] Task 2.2: Test `speaker_matcher.py` - Speaker identification
  - Test embedding-based matching
  - Verify correct speaker assignment
  - Test similarity threshold tuning
  - Test unknown speaker detection

- [ ] Task 2.3: Test `separation.py` - Vocal extraction
  - Test Demucs separation (vocals, drums, bass, other)
  - Verify output stems (4 files)
  - Test separation quality (SDR metric)
  - Test GPU vs CPU execution

- [ ] Task 2.4: Test `youtube_downloader.py` - Download handling
  - Test successful download (5s clip)
  - Test format extraction (audio-only)
  - Test error handling (404, geo-block, invalid URL)
  - Test metadata extraction (title, artist)

- [ ] Task 2.5: Test `youtube_metadata.py` - Metadata parsing
  - Test artist detection from title
  - Test featured artist extraction
  - Test title cleaning
  - Test genre classification (if applicable)

- [ ] Task 2.6: Test `file_organizer.py` - File management
  - Test directory creation
  - Test file naming conventions
  - Test cleanup of old files

- [ ] Task 2.7: Test `speaker_diarization.py` - Diarization accuracy
  - Test speaker count detection
  - Test timestamp accuracy (±0.5s)
  - Test pyannote integration

### Verification

- [ ] All audio tests pass
- [ ] Tests use fixtures (not network downloads)
- [ ] Coverage >70% for audio/ directory

## Phase 3: Database and Storage Tests

Test CRUD operations, schema validation, and storage.

### Tasks

- [ ] Task 3.1: Test `db/operations.py` - CRUD operations
  - Test profile creation (insert)
  - Test profile retrieval (select)
  - Test profile update
  - Test profile deletion (soft delete)
  - Test transaction rollback on error

- [ ] Task 3.2: Test `db/schema.py` - Data model validation
  - Test schema creation (tables, indexes)
  - Test foreign key constraints
  - Test unique constraints
  - Test default values

- [ ] Task 3.3: Test `db/session.py` - Connection lifecycle
  - Test session creation
  - Test session cleanup
  - Test connection pooling
  - Test error recovery

- [ ] Task 3.4: Test `storage/voice_profiles.py` - File storage
  - Test profile directory creation
  - Test sample file storage
  - Test adapter file storage
  - Test cleanup on profile deletion

- [ ] Task 3.5: Test `profiles/sample_collector.py` - Sample collection
  - Test sample validation (format, duration)
  - Test duplicate detection
  - Test sample organization

### Verification

- [ ] All database tests pass
- [ ] Tests use in-memory SQLite (fast)
- [ ] Coverage >70% for db/ and storage/

## Phase 4: Web API Tests (60+ Endpoints)

Test all REST API endpoints and WebSocket events.

### Tasks

- [ ] Task 4.1: Test `/api/v1/convert/*` endpoints (7 endpoints)
  - POST `/convert/song` - Job creation
  - GET `/convert/status/{job_id}` - Status polling
  - GET `/convert/download/{job_id}` - Result download
  - POST `/convert/cancel/{job_id}` - Job cancellation
  - GET `/convert/metrics/{job_id}` - Quality metrics
  - GET `/convert/history` - History list
  - DELETE `/convert/history/{id}` - History cleanup

- [ ] Task 4.2: Test `/api/v1/voice/*` endpoints (10 endpoints)
  - POST `/voice/clone` - Profile creation
  - GET `/voice/profiles` - Profile list
  - GET `/voice/profiles/{id}` - Profile detail
  - DELETE `/voice/profiles/{id}` - Profile deletion
  - GET `/voice/profiles/{id}/adapters` - Adapter list
  - GET `/voice/profiles/{id}/model` - Model status
  - POST `/voice/profiles/{id}/adapter/select` - Adapter selection
  - GET `/voice/profiles/{id}/adapter/metrics` - Adapter metrics
  - GET `/voice/profiles/{id}/training-status` - Training status
  - POST `/voice/profiles/{id}/speaker-embedding` - Embedding update

- [ ] Task 4.3: Test `/api/v1/training/*` endpoints (4 endpoints)
  - GET `/training/jobs` - Job list
  - POST `/training/jobs` - Job creation
  - GET `/training/jobs/{id}` - Job status
  - POST `/training/jobs/{id}/cancel` - Job cancellation

- [ ] Task 4.4: Test `/api/v1/profiles/*` endpoints (8 endpoints)
  - GET `/profiles/{id}/samples` - Sample list
  - POST `/profiles/{id}/samples` - Sample upload
  - POST `/profiles/{id}/samples/from-path` - Sample from path
  - GET `/profiles/{id}/samples/{sid}` - Sample detail
  - DELETE `/profiles/{id}/samples/{sid}` - Sample deletion
  - POST `/profiles/{id}/samples/{sid}/filter` - Sample filtering
  - GET `/profiles/{id}/segments` - Diarization segments
  - GET `/profiles/{id}/checkpoints` - Checkpoint list

- [ ] Task 4.5: Test `/api/v1/audio/*` endpoints (3 endpoints)
  - POST `/audio/diarize` - Diarization job
  - POST `/audio/diarize/assign` - Segment assignment
  - POST `/profiles/auto-create` - Auto-profile creation

- [ ] Task 4.6: Test utility endpoints (10 endpoints)
  - GET `/health` - Health check
  - GET `/gpu/metrics` - GPU stats
  - GET `/system/info` - System info
  - GET `/devices/list` - Device list
  - POST `/youtube/info` - YouTube metadata
  - POST `/youtube/download` - YouTube download
  - GET `/models/loaded` - Loaded models
  - POST `/models/load` - Model loading
  - POST `/models/tensorrt/rebuild` - TensorRT rebuild
  - GET `/kernels/metrics` - CUDA kernel metrics

- [ ] Task 4.7: Test WebSocket events (karaoke)
  - `startSession` - Session initialization
  - `audioChunk` - Streaming audio
  - `stopSession` - Session cleanup
  - `convertedChunk` - Output streaming
  - `error` - Error propagation

- [ ] Task 4.8: Test WebSocket events (training)
  - `training_started` - Job start notification
  - `training_progress` - Progress updates
  - `training_complete` - Completion notification
  - `training_failed` - Error notification

### Verification

- [ ] All API endpoint tests pass
- [ ] Tests use Flask test client (no server needed)
- [ ] Coverage >80% for web/ directory
- [ ] Error responses tested (400, 404, 500)

## Phase 5: Integration Tests (E2E Flows)

Test complete user workflows end-to-end.

### Tasks

- [ ] Task 5.1: E2E - Train and convert flow
  - Create profile
  - Upload 3 samples (5s each)
  - Start training job
  - Wait for completion
  - Load trained adapter
  - Convert 10s song
  - Verify output quality

- [ ] Task 5.2: E2E - YouTube to trained profile
  - Download YouTube video (or mock)
  - Run diarization
  - Auto-create profile from segments
  - Collect 30s of vocals
  - Train adapter
  - Convert test song
  - Verify speaker similarity >0.8

- [ ] Task 5.3: E2E - Multi-pipeline comparison
  - Upload same source audio
  - Convert with each pipeline type:
    - realtime
    - quality
    - quality_seedvc
    - realtime_meanvc (if available)
  - Compare RTF, MCD, speaker similarity
  - Verify quality > realtime tradeoff

- [ ] Task 5.4: E2E - Karaoke session
  - Start WebSocket session
  - Send audio chunks (512 samples each)
  - Receive converted chunks
  - Verify latency <100ms
  - Switch profiles mid-session
  - Verify graceful cleanup

- [ ] Task 5.5: E2E - Error recovery
  - Test conversion with missing adapter (404 error)
  - Test GPU OOM handling (graceful degradation)
  - Test WebSocket disconnection recovery
  - Test training job cancellation cleanup

### Verification

- [ ] All E2E tests pass
- [ ] Tests marked `@pytest.mark.slow` (can skip in CI)
- [ ] Tests use real audio (from fixtures)
- [ ] Tests verify quality (not just shapes)

## Phase 6: Coverage Analysis and Gaps

Generate coverage report and fill remaining gaps.

### Tasks

- [ ] Task 6.1: Run pytest-cov for coverage report
  - `pytest --cov=src/auto_voice --cov-report=html`
  - Generate HTML report in `htmlcov/`
  - Identify modules <70% coverage

- [ ] Task 6.2: Add tests for uncovered branches
  - Focus on error handling paths
  - Test edge cases (empty input, None values)
  - Test boundary conditions

- [ ] Task 6.3: Add tests for remaining modules
  - `evaluation/benchmark_runner.py`
  - `evaluation/quality_metrics.py`
  - `export/tensorrt_engine.py`
  - `gpu/memory_manager.py`
  - `training/trainer.py`

- [ ] Task 6.4: Optimize slow tests
  - Cache model loading in fixtures
  - Use smaller audio clips
  - Mock expensive operations
  - Parallelize independent tests

- [ ] Task 6.5: Document test strategy
  - Update CLAUDE.md with test patterns
  - Add docstrings to test files
  - Create test fixture documentation

### Verification

- [ ] Overall coverage >80%
- [ ] Inference coverage >85%
- [ ] No critical modules untested
- [ ] Full test suite <20min

## Final Verification

- [ ] All acceptance criteria met
- [ ] Coverage report generated
- [ ] Tests integrated into CI/CD
- [ ] Documentation updated
- [ ] Ready for review

---

**Estimated Timeline:**
- Phase 1: 1 day (inference tests - critical)
- Phase 2: 1 day (audio processing)
- Phase 3: 0.5 day (database/storage)
- Phase 4: 1 day (web API - 60+ endpoints)
- Phase 5: 0.5 day (E2E flows)
- Phase 6: 0.5 day (coverage analysis)
- **Total:** 4.5 days

**Dependencies:**
- None (can start immediately)

**Blocks:**
- Production deployment
- Confident refactoring
- CI/CD validation

---

_Generated by Conductor._
