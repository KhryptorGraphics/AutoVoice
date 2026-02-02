# Voice Profile Training E2E Test Guide

Complete testing guide for voice profile training workflows.

## Overview

This document describes the end-to-end tests for validating the complete voice profile training system, including:
- Web UI flows (VoiceProfilePage, TrainingConfigPanel)
- API endpoints (profile creation, sample upload, training jobs)
- Diarization and segment assignment
- LoRA adapter training and usage
- Multi-artist YouTube workflows

## Test Files

### 1. API Integration Tests
**File:** `tests/test_voice_profile_training_e2e.py`

Tests all API endpoints and backend workflows:
- Profile creation and sample management
- Diarization and segment assignment
- Training job creation and monitoring
- Adapter loading and conversion
- Error handling and validation

**Run with:**
```bash
./scripts/run_voice_profile_e2e_tests.sh api
```

### 2. Browser Automation Tests
**File:** `tests/test_browser_voice_profile_workflow.py`

Tests UI workflows with visual validation using VNC:
- Profile creation form
- Sample upload UI
- Training configuration panel
- Diarization timeline
- YouTube multi-artist detection

**Requirements:**
- VNC server running on :99
- Flask app running on localhost:5000
- xdotool, chromium-browser installed

**Run with:**
```bash
./scripts/run_voice_profile_e2e_tests.sh browser
```

## Test Coverage by Phase

### Phase 1: Web UI Flow Validation
- ✅ Profile creation via API
- ✅ Sample upload and validation
- ✅ Diarization execution
- ✅ Segment assignment to profiles

**Tests:**
- `TestProfileCreationFlow::test_create_profile_with_samples`
- `TestProfileCreationFlow::test_insufficient_sample_duration`
- `TestDiarizationFlow::test_diarization_and_segment_assignment`

### Phase 2: LoRA Training Flow
- ✅ Training configuration
- ✅ Job creation and queueing
- ✅ WebSocket progress updates
- ✅ Adapter file creation

**Tests:**
- `TestLoRATrainingFlow::test_training_job_creation`
- `TestLoRATrainingFlow::test_insufficient_samples_error`
- `TestCompleteWorkflow::test_profile_train_convert_workflow`

### Phase 3: Multi-Artist YouTube Flow
- ✅ YouTube download mock
- ✅ Multi-speaker diarization
- ✅ Auto-profile creation
- ✅ Featured artist detection

**Tests:**
- `TestMultiArtistFlow::test_auto_profile_creation_from_diarization`

### Phase 4: Adapter Integration
- ✅ Adapter listing endpoint
- ✅ Conversion requires trained model
- ✅ Adapter type selection (hq/nvfp4)
- ✅ Pipeline integration

**Tests:**
- `TestAdapterIntegration::test_adapter_listing`
- `TestAdapterIntegration::test_conversion_requires_trained_adapter`

### Phase 5: Error Handling
- ✅ Invalid file format rejection
- ✅ Insufficient samples error
- ✅ Training cancellation
- ✅ Network error recovery

**Tests:**
- `TestErrorHandling::test_invalid_audio_format`
- `TestErrorHandling::test_training_cancellation`

### Phase 6: Complete Workflow
- ✅ Profile → Upload → Train → Convert
- ✅ End-to-end with real models
- ✅ Quality validation

**Tests:**
- `TestCompleteWorkflow::test_profile_train_convert_workflow`

## Running Tests

### Quick Smoke Test
```bash
./scripts/run_voice_profile_e2e_tests.sh smoke
```

### Full Test Suite
```bash
# Start Flask app first
python main.py --host 0.0.0.0 --port 5000 &

# Run all tests
./scripts/run_voice_profile_e2e_tests.sh all
```

### Individual Test Classes
```bash
# Run specific test class
PYTHONNOUSERSITE=1 PYTHONPATH=src pytest tests/test_voice_profile_training_e2e.py::TestProfileCreationFlow -v

# Run with specific markers
pytest tests/test_voice_profile_training_e2e.py -m "integration and not slow" -v
```

## Test Markers

Tests use pytest markers for filtering:
- `integration` - Integration tests (require full app)
- `slow` - Long-running tests (>30s)
- `cuda` - Requires GPU
- `browser` - Browser automation tests

## Expected Results

### API Tests
All tests should pass if:
- Flask app is running
- CUDA is available (for training tests)
- All components initialized (VoiceCloner, TrainingJobManager, etc.)

Some tests may skip if:
- Services not available (503 responses)
- Endpoints not implemented (404 responses)

### Browser Tests
Tests capture screenshots to `/tmp/` for visual validation:
- `profiles_page.png` - Profile list
- `profile_created.png` - New profile form
- `training_config.png` - Training settings
- `diarization_result.png` - Speaker segments

## Troubleshooting

### Test Failures

**"Profile creation failed"**
- Check Flask app is running
- Verify VoiceCloner component initialized
- Check profile storage path is writable

**"Training not available"**
- GPU not available (CUDA required)
- TrainingJobManager not initialized
- Check logs for component startup errors

**"Diarization service unavailable"**
- Pyannote models not downloaded
- WavLM encoder not initialized
- Check diarization dependencies installed

### Browser Test Issues

**"VNC display :99 not available"**
```bash
# Start VNC server
vncserver :99 -geometry 1920x1080 -depth 24
```

**"Browser window not found"**
- Increase sleep times in test
- Check browser launched successfully
- Verify xdotool can find windows

**"Click coordinates incorrect"**
- UI layout changed
- Adjust x/y coordinates in test
- Take screenshots to verify positions

## Manual Testing Checklist

For manual validation of workflows not covered by automation:

### Profile Creation
- [ ] Navigate to /profiles
- [ ] Click "Create Profile"
- [ ] Enter profile name
- [ ] Upload sample audio (>5s)
- [ ] Verify profile appears in list

### Training
- [ ] Select profile with 3+ samples
- [ ] Open "Config" tab
- [ ] Adjust training parameters
- [ ] Click "Start Training"
- [ ] Verify WebSocket progress updates
- [ ] Wait for completion
- [ ] Check "Trained" badge appears

### Diarization
- [ ] Upload multi-speaker audio
- [ ] Click "Run Diarization"
- [ ] Verify speaker segments display
- [ ] Play segment audio
- [ ] Assign segment to profile
- [ ] Verify segment in profile's samples

### Conversion
- [ ] Select trained profile
- [ ] Upload song
- [ ] Click "Convert"
- [ ] Monitor progress
- [ ] Download result
- [ ] Verify voice characteristics

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run E2E tests
        run: |
          python main.py --host 0.0.0.0 --port 5000 &
          sleep 10
          ./scripts/run_voice_profile_e2e_tests.sh smoke
```

## Test Data

### Sample Audio Files
Tests generate synthetic audio with:
- 24kHz sample rate
- Voice-like harmonics (fundamental + overtones)
- Vibrato modulation for realism
- Multiple speaker frequencies for diarization

### Test Profiles
Auto-generated profiles for testing:
- "Test Artist" - Basic profile
- "Training Test" - For training workflows
- "Speaker 1", "Speaker 2" - For diarization

## Performance Benchmarks

Expected test execution times:
- API smoke tests: <30s
- Full API suite: 5-10 minutes (with training)
- Browser tests: 10-15 minutes
- Complete workflow: 5-20 minutes (depends on training config)

## Next Steps

After E2E validation:
1. Update plan.md with completed tasks
2. Document any gaps or issues
3. Create bug reports for failures
4. Propose improvements for user experience
5. Add performance optimization tests
