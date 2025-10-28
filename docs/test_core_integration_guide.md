# AutoVoice Core Integration Test Suite Guide

## Overview

The `test_core_integration.py` test suite provides comprehensive integration testing for the AutoVoice project, validating component interactions, data flow, and end-to-end pipeline execution.

**File Location**: `/home/kp/autovoice/tests/test_core_integration.py`

**Total Tests**: 13 test functions across 5 test classes

**Lines of Code**: 940

---

## Test Coverage

### 1. Component Integration Tests

#### TestVocalSeparatorPitchExtractorIntegration

Tests the integration between vocal separation and pitch extraction components.

**Tests:**
- `test_separate_and_extract_pitch`: Validates that separated vocals can be analyzed for pitch
  - Separates vocals using VocalSeparator
  - Extracts F0 contour using SingingPitchExtractor
  - Validates data format and sample rate alignment
  - Checks for finite values and proper dimensions

- `test_sample_rate_alignment`: Validates sample rate consistency
  - Tests sample rate communication between components
  - Validates resampling when needed
  - Checks for data corruption during rate conversion

#### TestVoiceClonerConverterIntegration

Tests voice profile creation and usage for voice conversion.

**Tests:**
- `test_create_profile_and_convert`: End-to-end profile workflow
  - Creates voice profile from audio
  - Loads profile with embedding
  - Validates embedding format (256-dim)
  - Checks all required fields

- `test_profile_compatibility`: Validates profile format
  - Checks embedding dimensions
  - Validates normalization
  - Ensures all required fields present

#### TestEndToEndPipeline

Tests complete pipeline execution from input to output.

**Tests:**
- `test_full_pipeline`: Complete workflow validation
  - Separates vocals from audio
  - Extracts pitch from separated vocals
  - Uses speaker embedding for conversion
  - Validates all outputs

---

### 2. Data Flow Tests

#### TestDataFlow

Validates data format compatibility across components.

**Tests:**
- `test_audio_format_mono_stereo`: Mono/stereo handling
  - Tests mono audio preservation
  - Validates stereo downmixing
  - Checks for shape errors

- `test_sample_rate_consistency`: Sample rate handling
  - Tests different sample rates (16kHz, 44.1kHz)
  - Validates resampling correctness
  - Checks for data corruption

- `test_data_type_consistency`: Data type handling
  - Tests float32, float64, int16
  - Validates conversions
  - Checks for overflow/underflow

---

### 3. Performance Tests

#### TestPerformance

Performance benchmarks and stress tests.

**Tests:**
- `test_gpu_memory_tracking`: GPU memory usage monitoring
  - Tracks memory allocation during processing
  - Validates reasonable memory consumption (<2GB)
  - Tests cleanup

- `test_memory_leak_detection`: Memory leak detection
  - Runs 10 iterations of pipeline
  - Tracks CPU and GPU memory
  - Reports leaks above threshold (CPU: 100MB, GPU: 50MB)

- `test_stress_concurrent_operations`: Concurrent stress test
  - Runs 10 concurrent pitch extraction tasks
  - Validates no race conditions
  - Checks all operations complete successfully

- `test_processing_time_benchmark`: Processing time benchmarks
  - Measures separation time
  - Measures pitch extraction time
  - Reports total pipeline time
  - Validates performance (<60s for 3s audio)

---

## Test Fixtures

### Audio Fixtures

1. **song_file_mono**: 3-second mono synthetic song
   - Multiple harmonics (A3 fundamental)
   - Amplitude modulation for speech-like quality
   - Noise component for realism

2. **song_file_stereo**: 3-second stereo synthetic song
   - Different content in L/R channels
   - Used for stereo handling tests

### Component Fixtures

3. **pipeline_instance**: Full pipeline with all components
   - VocalSeparator
   - SingingPitchExtractor
   - VoiceCloner
   - AudioProcessor

4. **voice_profile_fixture**: Synthetic voice profile
   - 256-dim speaker embedding
   - Vocal range data
   - Timbre features

### Utility Fixtures

5. **concurrent_executor**: ThreadPoolExecutor with 4 workers
   - Used for concurrent stress tests

6. **memory_leak_detector**: Memory leak detection utility
   - Tracks CPU and GPU memory
   - Reports leaks above threshold
   - Supports before/after measurements

---

## Test Markers

All tests are marked with appropriate pytest markers for filtering:

- **@pytest.mark.integration**: All tests in this suite
- **@pytest.mark.slow**: Tests taking >1 second
- **@pytest.mark.cuda**: GPU-specific tests
- **@pytest.mark.performance**: Performance benchmarks

---

## Running Tests

### Run All Integration Tests
```bash
pytest tests/test_core_integration.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_core_integration.py::TestVocalSeparatorPitchExtractorIntegration -v
```

### Run Specific Test
```bash
pytest tests/test_core_integration.py::TestPerformance::test_gpu_memory_tracking -v
```

### Filter by Marker
```bash
# Run only integration tests
pytest tests/test_core_integration.py -m integration

# Run fast tests only (exclude slow)
pytest tests/test_core_integration.py -m "integration and not slow"

# Run GPU tests only
pytest tests/test_core_integration.py -m cuda

# Run performance benchmarks
pytest tests/test_core_integration.py -m performance
```

### Run with Coverage
```bash
pytest tests/test_core_integration.py --cov=src/auto_voice --cov-report=html
```

### Run with Verbose Logging
```bash
pytest tests/test_core_integration.py -v -s --log-cli-level=INFO
```

---

## Expected Test Behavior

### Success Criteria

1. **Component Integration**: All components work together seamlessly
2. **Data Flow**: Sample rates and formats align correctly
3. **Performance**: Processing completes within reasonable time
4. **Memory**: No significant memory leaks detected
5. **Concurrency**: Handles concurrent operations without errors

### Common Skip Reasons

Tests may be skipped if:
- Required components not available (Demucs, torchcrepe, etc.)
- CUDA not available (for GPU tests)
- Model loading fails
- Test data invalid

### Expected Warnings

Some warnings are expected:
- HuBERT-Soft loading skipped (requires network)
- Component initialization warnings
- Memory leak warnings (if above threshold)

---

## Troubleshooting

### Test Failures

1. **"Required components not available"**
   - Install missing dependencies: `pip install demucs torchcrepe`
   - Check imports in test output

2. **"CUDA not available"**
   - GPU tests require CUDA-capable hardware
   - Run with `-m "not cuda"` to skip GPU tests

3. **"Audio validation failed"**
   - Check test audio generation
   - Verify sample rates match expectations

4. **"Memory leak detected"**
   - Review component cleanup code
   - Check for circular references
   - Run with smaller test data

### Performance Issues

If tests are too slow:
1. Run with `-m "not slow"` to skip long tests
2. Reduce iterations in stress tests
3. Use smaller test audio files

---

## Test Maintenance

### Adding New Tests

1. Add test function to appropriate class
2. Use existing fixtures when possible
3. Add appropriate markers (@pytest.mark.*)
4. Include docstring with test description
5. Validate all assertions are meaningful

### Updating Fixtures

1. Keep fixtures simple and reusable
2. Document fixture purpose in docstring
3. Clean up resources in teardown
4. Use session/module scope for expensive fixtures

### Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Always clean up resources
3. **Assertions**: Use descriptive assertion messages
4. **Documentation**: Add docstrings to all tests
5. **Markers**: Use appropriate markers for filtering

---

## Performance Benchmarks

### Typical Performance (3-second audio)

| Operation | CPU | GPU (CUDA) |
|-----------|-----|------------|
| Vocal Separation | 20-40s | 5-15s |
| Pitch Extraction | 1-3s | 0.5-1s |
| Voice Conversion | 5-10s | 1-3s |
| **Total Pipeline** | 26-53s | 6.5-19s |

### Memory Usage

| Component | CPU RAM | GPU VRAM |
|-----------|---------|----------|
| VocalSeparator | 500MB-1GB | 1-2GB |
| PitchExtractor | 200MB | 100-500MB |
| VoiceCloner | 300MB | 200-800MB |
| **Total Pipeline** | 1-2GB | 1.3-3.3GB |

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run integration tests
        run: |
          pytest tests/test_core_integration.py \
            -m "integration and not slow and not cuda" \
            --cov=src/auto_voice \
            --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Related Documentation

- **Unit Tests**: See individual component test files
- **E2E Tests**: See `test_end_to_end.py`
- **Performance Tests**: See `test_performance.py`
- **API Tests**: See `test_web_interface.py`

---

## Contact & Support

For issues or questions about the integration test suite:
1. Check test output for detailed error messages
2. Review this guide for common issues
3. Check component-specific documentation
4. Open an issue on GitHub with test output

---

**Last Updated**: 2025-10-27
**Test Suite Version**: 1.0
**AutoVoice Version**: 0.1.0
