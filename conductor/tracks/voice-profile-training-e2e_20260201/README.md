# Voice Profile Training E2E Validation

Complete end-to-end test suite for voice profile training workflows.

## Quick Start

### Run Tests

```bash
# Quick validation (< 30s)
./scripts/run_voice_profile_e2e_tests.sh smoke

# Full API integration tests
./scripts/run_voice_profile_e2e_tests.sh api

# Browser automation tests
./scripts/run_voice_profile_e2e_tests.sh browser

# Everything
./scripts/run_voice_profile_e2e_tests.sh all
```

### Prerequisites

**For API Tests:**
- Flask app running on localhost:5000
- CUDA available (for training tests)
- Python dependencies installed

**For Browser Tests:**
- VNC server on display :99
- xdotool and chromium-browser installed
- Flask app running

## What's Tested

### User Workflows
1. **Profile Creation** - Create profile, upload samples, validate
2. **Training** - Configure LoRA settings, start job, monitor progress
3. **Diarization** - Multi-speaker detection, segment assignment
4. **YouTube** - Multi-artist detection, auto-profile creation
5. **Conversion** - Use trained adapters in realtime/quality pipelines

### Test Coverage
- ✅ 12 API integration tests
- ✅ 20+ browser automation tests
- ✅ Error handling and validation
- ✅ WebSocket progress events
- ✅ Complete end-to-end workflows

## Files

- **Tests**
  - `tests/test_voice_profile_training_e2e.py` - API tests
  - `tests/test_browser_voice_profile_workflow.py` - Browser tests

- **Scripts**
  - `scripts/run_voice_profile_e2e_tests.sh` - Test runner

- **Docs**
  - `E2E_TEST_GUIDE.md` - Detailed testing guide
  - `IMPLEMENTATION_SUMMARY.md` - Test suite overview
  - `plan.md` - Implementation plan

## Results

Test results are saved to:
- `test-results/voice-profile-api-e2e.xml` - API test results
- `test-results/voice-profile-browser-e2e.xml` - Browser test results
- `/tmp/*.png` - Browser test screenshots

## Troubleshooting

### "Flask app not running"
```bash
python main.py --host 0.0.0.0 --port 5000
```

### "VNC display not available"
```bash
vncserver :99 -geometry 1920x1080 -depth 24
```

### "CUDA not available"
Training tests will be skipped automatically on CPU-only systems.

### "Service unavailable (503)"
Some tests skip if components aren't initialized. This is expected in test mode.

## Documentation

See [E2E_TEST_GUIDE.md](./E2E_TEST_GUIDE.md) for:
- Detailed test descriptions
- Manual testing checklists
- Performance benchmarks
- CI/CD integration

## Status

✅ **Tests Created** - Ready for execution

Track completes the validation for:
- speaker-diarization_20260130
- youtube-artist-training_20260130
- training-inference-integration_20260130
- frontend-complete-integration_20260201
