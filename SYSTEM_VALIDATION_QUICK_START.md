# System Validation Test Suite - Quick Start

**Created**: 2025-10-28  
**Addresses**: Comments 1, 2, 3, 9

## ✅ What Was Created

1. **Test Data Generator** (389 lines)
   - `/home/kp/autovoice/tests/data/validation/generate_test_data.py`
   - Generates 25+ synthetic audio samples (5 genres × 5 samples)
   - CLI: `python tests/data/validation/generate_test_data.py --samples-per-genre 5`

2. **System Validation Tests** (997 lines)
   - `/home/kp/autovoice/tests/test_system_validation.py`
   - 5 test classes, 15+ test functions
   - Covers: metadata-driven validation, TensorRT latency, edge cases, performance

3. **Documentation**
   - `tests/data/validation/README.md` - Test data guide
   - `docs/SYSTEM_VALIDATION_SUITE.md` - Comprehensive documentation
   - `docs/VALIDATION_SUITE_SUMMARY.md` - Implementation summary
   - `SYSTEM_VALIDATION_QUICK_START.md` - This file

4. **Pytest Configuration**
   - Updated `pytest.ini` with 7 new markers

## 🚀 Quick Start (5 Steps)

### 1. Generate Test Data (30 seconds)
```bash
cd /home/kp/autovoice
python tests/data/validation/generate_test_data.py
```

**Output**: 25 WAV files + test_set.json in `tests/data/validation/`

### 2. Run All System Validation Tests
```bash
pytest tests/test_system_validation.py -v -m system_validation
```

### 3. Run TensorRT Latency Test (Requires CUDA + TensorRT)
```bash
pytest tests/test_system_validation.py::TestTensorRTLatency::test_latency_target_trt_fast_30s -v
```

### 4. Run Edge Case Tests
```bash
pytest tests/test_system_validation.py::TestEdgeCases -v
```

### 5. Check Results
```bash
ls -lh validation_results/
cat validation_results/validation_summary.json | python -m json.tool
```

## 📊 Test Coverage Summary

| Comment | Requirement | Implementation | Status |
|---------|------------|----------------|--------|
| **1** | Metadata-driven system validation | `TestMetadataDrivenValidation` | ✅ |
| | Assert pitch RMSE < 10 Hz | `assert pitch_rmse_hz < 10.0` | ✅ |
| | Assert speaker similarity > 0.85 | `assert speaker_similarity > 0.85` | ✅ |
| | Assert latency < 5s per 30s | `assert latency / duration < 5.0` | ✅ |
| | Save per-sample metrics | `validation_results/*.json` | ✅ |
| **2** | TensorRT fast preset latency | `test_latency_target_trt_fast_30s` | ✅ |
| | Assert < 5.0s for 30s audio | `assert elapsed < 5.0` | ✅ |
| | FP16 precision | `tensorrt_precision='fp16'` | ✅ |
| | Skip if TensorRT unavailable | `pytest.importorskip('tensorrt')` | ✅ |
| **3** | Diverse test data generator | `generate_test_data.py` | ✅ |
| | 5 genres | pop, rock, jazz, classical, rap | ✅ |
| | Variable durations | 10s-30s | ✅ |
| | test_set.json metadata | Generated with metadata | ✅ |
| **9** | Short audio (<10s) | `test_short_audio_under_10s` | ✅ |
| | Long audio (>5min) | `test_long_audio_over_5min` | ✅ |
| | A cappella input | `test_acappella_input` | ✅ |
| | Processed vocals | `test_heavily_processed_vocals` | ✅ |

## 🎯 Key Test Classes

### TestMetadataDrivenValidation (Comment 1)
- **Purpose**: End-to-end validation with quality metrics
- **Test**: `test_diverse_genres_conversion`
- **Markers**: `@pytest.mark.system_validation`, `@pytest.mark.slow`
- **Run**: `pytest tests/test_system_validation.py::TestMetadataDrivenValidation -v`

### TestTensorRTLatency (Comment 2)
- **Purpose**: TensorRT acceleration latency enforcement
- **Test**: `test_latency_target_trt_fast_30s`
- **Markers**: `@pytest.mark.requires_trt`, `@pytest.mark.performance`
- **Run**: `pytest tests/test_system_validation.py::TestTensorRTLatency -v`
- **Requires**: CUDA + TensorRT + RTX GPU

### TestEdgeCases (Comment 9)
- **Purpose**: Edge case handling validation
- **Tests**: short audio, long audio, a cappella, processed vocals
- **Markers**: `@pytest.mark.edge_cases`, `@pytest.mark.very_slow` (long audio)
- **Run**: `pytest tests/test_system_validation.py::TestEdgeCases -v`

### TestPerformanceValidation
- **Purpose**: Performance benchmarks and monitoring
- **Tests**: latency scaling, GPU utilization, component timing
- **Markers**: `@pytest.mark.performance`
- **Run**: `pytest tests/test_system_validation.py::TestPerformanceValidation -v`

## 🔧 Useful Commands

```bash
# Skip slow tests
pytest tests/test_system_validation.py -v -m "system_validation and not slow"

# Skip very slow tests (>5 min)
pytest tests/test_system_validation.py -v -m "system_validation and not very_slow"

# Run only TensorRT tests
pytest tests/test_system_validation.py -v -m tensorrt

# Run only edge case tests
pytest tests/test_system_validation.py -v -m edge_cases

# Run specific genre
pytest tests/test_system_validation.py::TestGenreSpecificValidation -v -k "pop"

# List all test functions
pytest tests/test_system_validation.py --collect-only

# Verbose output with stdout
pytest tests/test_system_validation.py -v -s

# Generate coverage report
pytest tests/test_system_validation.py --cov=src/auto_voice --cov-report=html
```

## 📁 File Structure

```
/home/kp/autovoice/
├── tests/
│   ├── data/validation/
│   │   ├── generate_test_data.py    ✅ 389 lines
│   │   ├── test_set.json            ✅ Generated
│   │   ├── *.wav                    ✅ 25 files (43MB)
│   │   └── README.md                ✅ Documentation
│   └── test_system_validation.py    ✅ 997 lines
├── docs/
│   ├── SYSTEM_VALIDATION_SUITE.md   ✅ Full documentation
│   └── VALIDATION_SUITE_SUMMARY.md  ✅ Implementation summary
├── pytest.ini                       ✅ Updated (7 new markers)
└── validation_results/              📁 Created on test run
```

## 📦 Dependencies

```bash
# Core testing
pip install pytest torch numpy soundfile scipy psutil

# TensorRT testing (optional)
pip install tensorrt  # Requires CUDA 11.8+, RTX GPU
```

## 🐛 Troubleshooting

### Test data not found
```bash
python tests/data/validation/generate_test_data.py --samples-per-genre 5
```

### Import errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or: pip install -e .
```

### TensorRT tests skipped
```bash
# Check TensorRT
python -c "import tensorrt; print(tensorrt.__version__)"

# Check CUDA
nvidia-smi

# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 📝 Quality Targets

| Metric | Target | Test Assertion |
|--------|--------|----------------|
| Pitch RMSE | < 10 Hz | `pitch_rmse_hz < 10.0` |
| Speaker Similarity | > 0.85 | `speaker_similarity > 0.85` |
| Latency (30s audio) | < 5.0s | `elapsed < 5.0` |
| RTF | < 5.0x | `latency / duration < 5.0` |
| Memory (long audio) | < 2GB | `memory_usage < 2048` MB |
| GPU Utilization | > 70% | `avg_utilization > 70.0` % |

## 🎓 Documentation

- **README**: `tests/data/validation/README.md` - Test data guide
- **Full Suite Guide**: `docs/SYSTEM_VALIDATION_SUITE.md` - Comprehensive documentation
- **Summary**: `docs/VALIDATION_SUITE_SUMMARY.md` - Implementation overview
- **Quick Start**: `SYSTEM_VALIDATION_QUICK_START.md` - This file

## ✨ Next Steps

1. ✅ Test data generation - **Complete**
2. ✅ Test suite implementation - **Complete**
3. ✅ Documentation - **Complete**
4. ⏳ Execute tests on target hardware
5. ⏳ Validate results format
6. ⏳ CI/CD integration

## 📞 Support

For issues or questions:
1. Check `docs/SYSTEM_VALIDATION_SUITE.md` troubleshooting section
2. Review test output with `-v -s` flags
3. Check `tests/data/validation/README.md` for test data issues
