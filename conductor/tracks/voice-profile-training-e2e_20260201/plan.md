# Implementation Plan: Voice Profile Training E2E Validation

**Track ID:** voice-profile-training-e2e_20260201
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-01
**Status:** [x] In Progress

## Overview

End-to-end validation of voice profile creation and LoRA training from web UI. Tests complete user workflows with diarization, separation, multi-artist detection, and training.

## Cross-Context Dependencies

**From Completed Tracks:**
- `speaker-diarization_20260130`: Pyannote diarization, segment extraction, profile matching
- `youtube-artist-training_20260130`: Featured artist detection, auto-profile creation
- `training-inference-integration_20260130`: AdapterManager, LoRA loading, API endpoints
- `frontend-complete-integration_20260201`: VoiceProfilePage, TrainingConfigPanel, DiarizationTimeline

**Key APIs:**
- POST /api/v1/voice/clone - Create profile
- POST /api/v1/profiles/{id}/samples - Add samples
- POST /api/v1/audio/diarize - Run diarization
- POST /api/v1/profiles/auto-create - Auto-create from diarization
- POST /api/v1/training/jobs - Start training
- GET /api/v1/profiles/{id}/model - Check adapter status

## Phase 1: Web UI Flow Validation

### Tasks

- [ ] Task 1.1: Test VoiceProfilePage profile creation flow
  - Navigate to /profiles
  - Click "Create Profile"
  - Enter name, upload sample audio
  - Verify profile appears in list

- [ ] Task 1.2: Test sample upload and validation
  - Select existing profile
  - Upload multiple audio samples
  - Verify samples appear in profile detail
  - Check audio duration validation (>5s required)

- [ ] Task 1.3: Test diarization UI flow
  - Upload multi-speaker audio
  - Click "Run Diarization"
  - Verify DiarizationTimeline shows speaker segments
  - Test segment playback

- [ ] Task 1.4: Test segment assignment to profiles
  - Click speaker segment
  - Assign to existing or new profile
  - Verify segment moves to profile's samples
  - Check speaker embedding updates

## Phase 2: LoRA Training Flow

### Tasks

- [ ] Task 2.1: Test training configuration UI
  - Select profile with samples
  - Open TrainingConfigPanel
  - Configure epochs, learning rate, batch size
  - Verify validation warnings for insufficient samples

- [ ] Task 2.2: Test training job creation
  - Click "Start Training"
  - Verify job appears in training queue
  - Check WebSocket training_started event

- [ ] Task 2.3: Test training progress monitoring
  - Monitor LiveTrainingMonitor component
  - Verify training_progress WebSocket events
  - Check loss curve updates
  - Verify checkpoint saves

- [ ] Task 2.4: Test training completion
  - Wait for training_complete event
  - Verify adapter file created in profiles/{id}/adapters/
  - Check adapter loads without error
  - Verify "Trained" badge appears on profile

## Phase 3: YouTube Multi-Artist Flow

### Tasks

- [ ] Task 3.1: Test YouTube URL input and metadata detection
  - Navigate to YouTube download page
  - Enter URL with featured artists (e.g., "Artist A ft. Artist B")
  - Verify FeaturedArtistCard shows detected artists

- [ ] Task 3.2: Test download and separation
  - Click Download
  - Monitor ExtractionPanel progress
  - Verify separated vocals created

- [ ] Task 3.3: Test multi-speaker diarization
  - Run diarization on separated vocals
  - Verify each detected artist has segments
  - Check speaker embeddings distinguish artists

- [ ] Task 3.4: Test auto-profile creation
  - Click "Create Profiles for All Artists"
  - Verify profile created for each artist
  - Check segments assigned correctly
  - Confirm speaker embeddings saved

## Phase 4: Adapter Integration

### Tasks

- [ ] Task 4.1: Test adapter loading in realtime pipeline
  - Select trained profile in KaraokePage
  - Start session with profile's adapter
  - Verify adapter applies to conversion

- [ ] Task 4.2: Test adapter loading in quality pipeline
  - Convert song with profile_id parameter
  - Verify AdapterManager loads correct adapter
  - Check converted audio has target voice characteristics

- [ ] Task 4.3: Test adapter switching
  - Convert with Profile A adapter
  - Switch to Profile B adapter
  - Verify distinct voice characteristics

- [ ] Task 4.4: Test nvfp4 vs hq adapter selection
  - Train with hq adapter type
  - Train with nvfp4 adapter type
  - Verify both load correctly
  - Compare conversion speed

## Phase 5: Error Handling

### Tasks

- [ ] Task 5.1: Test insufficient samples error
  - Try training with <3 samples
  - Verify clear error message
  - Check suggested minimum

- [ ] Task 5.2: Test invalid audio format error
  - Upload non-audio file
  - Verify rejection with message

- [ ] Task 5.3: Test training cancellation
  - Start training
  - Click cancel
  - Verify cleanup (no partial adapter)

- [ ] Task 5.4: Test network error recovery
  - Simulate WebSocket disconnect during training
  - Verify reconnection and progress recovery

## Phase 6: Integration Tests

### Tasks

- [x] Task 6.1: Create E2E test: Profile → Train → Convert
  - Automated test script ✓
  - Creates profile from fixture audio ✓
  - Trains LoRA (quick 10 epochs) ✓
  - Converts test song ✓
  - Verifies output ✓

- [x] Task 6.2: Create E2E test: YouTube → Multi-Profile → Train
  - Mock YouTube download ✓
  - Test full multi-artist flow ✓
  - Verify multiple profiles created and trainable ✓

- [x] Task 6.3: Create E2E test: Karaoke with trained adapter
  - WebSocket session test (browser tests) ✓
  - Uses trained profile ✓
  - Verifies realtime conversion ✓

## Final Verification

- [x] Test suite created with comprehensive coverage
- [x] API integration tests implemented (12 test methods)
- [x] Browser automation tests implemented (20+ test methods)
- [x] Test execution scripts created
- [x] Documentation and guides written
- [ ] All 3 user workflows succeed (pending execution)
- [ ] LoRA training produces valid adapters (pending execution)
- [ ] Adapters work in all pipeline types (pending execution)
- [ ] Error messages helpful (pending execution)
- [ ] Progress updates work throughout (pending execution)

## Test Artifacts Created

### Test Files
- `tests/test_voice_profile_training_e2e.py` - API integration tests
- `tests/test_browser_voice_profile_workflow.py` - Browser automation tests
- `scripts/run_voice_profile_e2e_tests.sh` - Test execution script

### Documentation
- `E2E_TEST_GUIDE.md` - Complete testing guide
- `IMPLEMENTATION_SUMMARY.md` - Test suite summary

### Test Coverage
- **Total Tests:** 32+ test methods
- **Code Lines:** ~1,713 lines of tests and docs
- **Coverage:** Profile creation, training, diarization, adapters, errors

### Next Steps for Execution
1. Start Flask app: `python main.py --host 0.0.0.0 --port 5000`
2. Run smoke tests: `./scripts/run_voice_profile_e2e_tests.sh smoke`
3. Run full suite: `./scripts/run_voice_profile_e2e_tests.sh all`
4. Document results and fix any gaps

---

_Generated by Conductor._
