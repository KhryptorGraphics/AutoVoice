# System Validation Test Suite - Implementation Summary

**Created**: 2025-10-28  
**Addresses**: Comments 1, 2, 3, 9

## Files Created

### 1. Test Data Generator
**File**: `/home/kp/autovoice/tests/data/validation/generate_test_data.py`

- **Purpose**: Generate diverse synthetic audio for system validation
- **Features**:
  - 5 genres (pop, rock, jazz, classical, rap) with genre-specific characteristics
  - Variable durations (10s-30s)
  - Multiple languages (en, es, fr, de, ja)
  - Pitch ranges (220-392 Hz)
  - Reproducible generation (--seed parameter)
  - CLI interface
- **Usage**: `python tests/data/validation/generate_test_data.py --samples-per-genre 5`
- **Output**: 25+ WAV files + test_set.json metadata
- **Status**: ✅ Tested and working

### 2. System Validation Test Suite
**File**: `/home/kp/autovoice/tests/test_system_validation.py`

**Test Classes**:

#### TestMetadataDrivenValidation (Comment 1)
- `test_diverse_genres_conversion`: End-to-end validation
  - Metadata-driven test iteration from test_set.json
  - Uses `SingingConversionPipeline` for conversion
  - Uses `QualityMetricsAggregator` for evaluation
  - Asserts: pitch RMSE < 10 Hz, speaker similarity > 0.85, latency targets
  - Saves per-sample metrics to validation_results/
  - Generates aggregated validation report

#### TestTensorRTLatency (Comment 2)
- `test_latency_target_trt_fast_30s`: TensorRT latency enforcement
  - Synthesizes 30s WAV
  - Uses preset='fast', use_tensorrt=True, tensorrt_precision='fp16'
  - Asserts wall time < 5.0 seconds
  - Uses `pytest.importorskip('tensorrt')`
  - Skips when CUDA/TensorRT unavailable
  - Documents GPU requirements (RTX 30xx+)
  - Marked: `@pytest.mark.requires_trt`

#### TestEdgeCases (Comment 9)
- `test_short_audio_under_10s`: 7s clip validation
- `test_long_audio_over_5min`: 5+ min with memory tracking (@pytest.mark.very_slow)
- `test_acappella_input`: Pre-separated vocals, skip separation
- `test_heavily_processed_vocals`: Autotune-like effects, robustness

#### TestGenreSpecificValidation
- Parameterized tests for each genre (pop, rock, jazz, classical, rap)

#### TestPerformanceValidation
- `test_latency_scaling`: Linear scaling validation
- `test_latency_target_30s_input`: TensorRT latency (duplicate of TestTensorRTLatency)
- `test_gpu_utilization_monitoring`: GPU utilization > 70%
- `test_component_level_timing`: Component timing breakdown

### 3. Documentation

#### `/home/kp/autovoice/tests/data/validation/README.md`
- Test data overview
- Generation instructions
- Quality targets
- Running validation tests
- Test markers
- Validation results format
- Requirements
- Troubleshooting

#### `/home/kp/autovoice/docs/SYSTEM_VALIDATION_SUITE.md`
- Comprehensive suite documentation
- Test structure and purpose
- Running tests (all variations)
- Quality targets table
- Validation results format
- Test utilities documentation
- CI/CD integration example
- Troubleshooting guide
- Future enhancements

### 4. Pytest Configuration
**File**: `/home/kp/autovoice/pytest.ini` (updated)

**New Markers**:
- `system_validation`: Comprehensive system validation tests (Comments 1, 2, 3, 9)
- `tensorrt`: Tests requiring TensorRT
- `requires_trt`: Tests requiring TensorRT (strict requirement)
- `edge_cases`: Edge case tests
- `genre_specific`: Genre-specific validation tests
- `very_slow`: Very slow tests (>5 minutes)
- `performance`: Performance benchmarks and latency validation (updated)

## Test Coverage

### Comment Requirements Met

✅ **Comment 1** - System Validation Test Suite:
- Metadata-driven test iteration from test_set.json
- Assert pitch RMSE < 10 Hz, speaker similarity > 0.85, latency < 5s/30s
- Uses `SingingConversionPipeline` and `QualityMetricsAggregator`
- Saves per-sample metrics JSON to validation_results/
- Marked with `@pytest.mark.system_validation`

✅ **Comment 2** - TensorRT Latency Test:
- `test_latency_target_trt_fast_30s()` in TestTensorRTLatency
- Synthesizes 30s WAV
- Uses preset='fast', use_tensorrt=True, tensorrt_precision='fp16'
- Asserts wall time < 5.0 seconds
- Uses `pytest.importorskip('tensorrt')`
- Skips when CUDA/TRT unavailable
- Documents GPU requirements (RTX 30xx+)
- Marked with `@pytest.mark.performance` and `@pytest.mark.requires_trt`

✅ **Comment 3** - Test Data Generator:
- `/home/kp/autovoice/tests/data/validation/generate_test_data.py`
- Generates diverse synthetic audio (genres: pop, rock, jazz, classical, rap)
- Outputs `tests/data/validation/test_set.json` with metadata
- CLI interface with --samples-per-genre, --output, --seed
- 25+ test cases with varying characteristics

✅ **Comment 9** - Edge Case Tests:
- `test_short_audio_under_10s()` - 7s clip validation
- `test_long_audio_over_5min()` - 5+ min with memory tracking (marked @pytest.mark.very_slow)
- `test_acappella_input()` - pre-separated vocals, skip separation
- `test_heavily_processed_vocals()` - autotune-like effects
- All in TestEdgeCases class

## Quality Targets

| Metric | Target | Implementation |
|--------|--------|----------------|
| Pitch RMSE | < 10 Hz | `assert pitch_rmse_hz < 10.0` |
| Speaker Similarity | > 0.85 | `assert speaker_similarity > 0.85` |
| Latency (30s) | < 5.0s | `assert wall_time < 5.0` |
| RTF | < 5.0x | `assert latency / duration < 5.0` |

## Validation Results Structure

### Per-Sample Metrics
Location: `validation_results/<test_id>_metrics.json`

```json
{
  "test_id": "pop_001",
  "genre": "pop",
  "metrics": {
    "pitch_rmse_hz": 8.5,
    "speaker_similarity": 0.87,
    "latency_seconds": 2.3,
    "memory_usage_mb": 1024.5
  },
  "targets": {...},
  "passed": true
}
```

### Aggregated Summary
Location: `validation_results/validation_summary.json`

```json
{
  "timestamp": "2025-10-28 16:00:00",
  "total_tests": 25,
  "passed_tests": 24,
  "failed_tests": 1,
  "aggregate_metrics": {
    "pitch_rmse_hz": {
      "mean": 8.2,
      "std": 1.3,
      "min": 5.1,
      "max": 9.8
    },
    ...
  },
  "individual_results": [...]
}
```

## Running the Suite

### Generate Test Data
```bash
python tests/data/validation/generate_test_data.py --samples-per-genre 5
```

### Run All System Validation Tests
```bash
pytest tests/test_system_validation.py -v -m system_validation
```

### Run Specific Test Classes
```bash
# Metadata-driven validation (Comment 1)
pytest tests/test_system_validation.py::TestMetadataDrivenValidation -v

# TensorRT latency (Comment 2)
pytest tests/test_system_validation.py::TestTensorRTLatency -v -m requires_trt

# Edge cases (Comment 9)
pytest tests/test_system_validation.py::TestEdgeCases -v
```

### Skip Slow Tests
```bash
# Skip slow tests
pytest tests/test_system_validation.py -v -m "system_validation and not slow"

# Skip very slow tests (>5 min)
pytest tests/test_system_validation.py -v -m "system_validation and not very_slow"
```

## Dependencies

### Core Testing
- pytest >= 6.0
- torch >= 1.10
- numpy
- soundfile
- scipy
- psutil

### TensorRT Testing
- tensorrt >= 8.5
- CUDA >= 11.8
- NVIDIA GPU with Tensor Cores (RTX 2060+)

## Existing Utilities Used

The test suite integrates with existing AutoVoice components:

1. **src/auto_voice/utils/quality_metrics.py**
   - `QualityMetricsAggregator`: Quality evaluation

2. **examples/evaluate_voice_conversion.py**
   - Reference implementation for evaluation patterns

3. **src/auto_voice/inference/singing_conversion_pipeline.py**
   - `SingingConversionPipeline`: End-to-end conversion

4. **src/auto_voice/inference/voice_cloner.py**
   - `VoiceCloner`: Profile creation for testing

## Testing Status

✅ **Test Data Generator**: Tested and working
- Generated 25 test cases successfully
- test_set.json created with proper structure
- All 5 genres represented

⏳ **System Validation Tests**: Ready for execution
- All test functions implemented
- Proper markers and fixtures
- Validation results directory structure defined

⚠️ **TensorRT Tests**: Requires hardware
- Will skip automatically if TensorRT unavailable
- Properly marked with `@pytest.mark.requires_trt`

## Notes

1. **Test Execution**: Tests reference components that need proper imports configured
2. **Results Directory**: `validation_results/` will be created automatically on first run
3. **Memory Monitoring**: Uses psutil for cross-platform memory tracking
4. **GPU Utilization**: Requires nvidia-smi (installed with CUDA drivers)
5. **Edge Cases**: Long audio test marked `@pytest.mark.very_slow` for optional execution

## Next Steps

1. ✅ Test data generation - Complete
2. ✅ Test suite implementation - Complete
3. ✅ Documentation - Complete
4. ⏳ Execute tests on target hardware
5. ⏳ Validate results format
6. ⏳ CI/CD integration

## File Locations

```
/home/kp/autovoice/
├── tests/
│   ├── data/
│   │   └── validation/
│   │       ├── generate_test_data.py  ✅ Created
│   │       ├── test_set.json          ✅ Generated
│   │       ├── *.wav                  ✅ Generated (25 files)
│   │       └── README.md              ✅ Created
│   └── test_system_validation.py      ✅ Created
├── docs/
│   ├── SYSTEM_VALIDATION_SUITE.md     ✅ Created
│   └── VALIDATION_SUITE_SUMMARY.md    ✅ This file
├── pytest.ini                         ✅ Updated
└── validation_results/                📁 Created on test run
    ├── <test_id>_metrics.json        (Generated during tests)
    ├── validation_summary.json       (Generated during tests)
    ├── latency_tensorrt.json         (Generated during tests)
    ├── gpu_utilization.json          (Generated during tests)
    └── component_timing.json         (Generated during tests)
```

## Summary

**Comprehensive system validation test suite successfully created**, addressing all requirements from Comments 1, 2, 3, and 9:

- ✅ Metadata-driven end-to-end testing with quality validation
- ✅ TensorRT latency enforcement for fast preset
- ✅ Diverse test data generator (25+ samples, 5 genres)
- ✅ Edge case tests (short/long/a cappella/processed)
- ✅ Per-sample metrics saved to validation_results/
- ✅ Aggregated validation reports
- ✅ Comprehensive documentation
- ✅ Pytest markers and configuration

The suite is ready for execution on hardware with proper dependencies installed.
