# System Validation Test Suite

Comprehensive test suite for AutoVoice singing conversion system addressing Comments 1, 2, 3, and 9.

## Overview

The system validation suite provides end-to-end testing of the complete singing conversion pipeline with:

- **Metadata-driven validation**: Automated quality checks against targets
- **Diverse test coverage**: 25+ samples across genres, styles, and languages
- **Performance benchmarks**: TensorRT acceleration and latency validation
- **Edge case handling**: Short/long audio, a cappella, processed vocals
- **Per-sample metrics**: JSON output for aggregation and reporting

## Test Structure

### 1. Test Data Generator (`tests/data/validation/generate_test_data.py`)

**Comment 3**: Generates diverse synthetic audio samples.

**Features**:
- 5 genres: pop, rock, jazz, classical, rap
- Variable durations: 10s to 30s
- Multiple languages: en, es, fr, de, ja (metadata)
- Pitch ranges: 220 Hz to 392 Hz
- CLI interface with reproducibility (--seed)

**Generation**:
```bash
python tests/data/validation/generate_test_data.py --samples-per-genre 5
```

**Output**:
- 25+ WAV files (`tests/data/validation/*.wav`)
- Metadata file (`tests/data/validation/test_set.json`)

### 2. System Validation Tests (`tests/test_system_validation.py`)

#### TestMetadataDrivenValidation

**Comment 1**: End-to-end conversion with quality validation.

**Tests**:
- `test_diverse_genres_conversion`: Metadata-driven test iteration
  - Loads test cases from `test_set.json`
  - Runs `SingingConversionPipeline` for each sample
  - Uses `QualityMetricsAggregator` for evaluation
  - Asserts quality targets:
    - Pitch RMSE < 10 Hz
    - Speaker similarity > 0.85
    - Latency < 5s per 30s audio (RTF < 5.0x)
  - Saves per-sample metrics to `validation_results/`
  - Generates aggregated validation report

**Markers**: `@pytest.mark.system_validation`, `@pytest.mark.slow`

#### TestTensorRTLatency

**Comment 2**: TensorRT acceleration latency validation.

**Tests**:
- `test_latency_target_trt_fast_30s`:
  - Synthesizes 30s WAV file
  - Uses `preset='fast'`, `use_tensorrt=True`, `tensorrt_precision='fp16'`
  - Asserts wall time < 5.0 seconds
  - Uses `pytest.importorskip('tensorrt')`
  - Skips when CUDA/TensorRT unavailable
  - Documents GPU requirements (RTX 30xx+)

**Markers**: `@pytest.mark.system_validation`, `@pytest.mark.performance`, `@pytest.mark.requires_trt`

**Requirements**:
- NVIDIA GPU with Tensor Cores (RTX 2060+)
- CUDA 11.8+
- TensorRT 8.5+

#### TestEdgeCases

**Comment 9**: Edge case handling tests.

**Tests**:
- `test_short_audio_under_10s`: Validates 7s clip conversion
- `test_long_audio_over_5min`: 5+ min with memory tracking (`@pytest.mark.very_slow`)
- `test_acappella_input`: Pre-separated vocals, skip separation
- `test_heavily_processed_vocals`: Autotune-like effects, robustness validation

**Markers**: `@pytest.mark.system_validation`, `@pytest.mark.edge_cases`

#### TestGenreSpecificValidation

Genre-specific parameterized tests for each genre.

**Markers**: `@pytest.mark.system_validation`, `@pytest.mark.genre_specific`

#### TestPerformanceValidation

Performance and latency validation tests.

**Tests**:
- `test_latency_scaling`: Linear scaling validation
- `test_latency_target_30s_input`: TensorRT latency enforcement
- `test_gpu_utilization_monitoring`: GPU utilization > 70%
- `test_component_level_timing`: Component timing breakdown

**Markers**: `@pytest.mark.system_validation`, `@pytest.mark.performance`

## Running Tests

### All System Validation Tests

```bash
pytest tests/test_system_validation.py -v -m system_validation
```

### Specific Test Classes

```bash
# Metadata-driven validation
pytest tests/test_system_validation.py::TestMetadataDrivenValidation -v

# Edge cases
pytest tests/test_system_validation.py::TestEdgeCases -v

# Performance validation
pytest tests/test_system_validation.py::TestPerformanceValidation -v
```

### TensorRT Tests (Requires CUDA + TensorRT)

```bash
pytest tests/test_system_validation.py -v -m tensorrt
pytest tests/test_system_validation.py -v -m requires_trt
```

### Skip Slow Tests

```bash
# Skip all slow tests
pytest tests/test_system_validation.py -v -m "system_validation and not slow"

# Skip very slow tests (>5 min)
pytest tests/test_system_validation.py -v -m "system_validation and not very_slow"
```

### Genre-Specific Tests

```bash
pytest tests/test_system_validation.py::TestGenreSpecificValidation -v -k "pop"
pytest tests/test_system_validation.py::TestGenreSpecificValidation -v -k "rock"
```

## Quality Targets

All tests enforce the following quality targets:

| Metric | Target | Description |
|--------|--------|-------------|
| **Pitch RMSE** | < 10 Hz | Root mean square error in pitch tracking |
| **Speaker Similarity** | > 0.85 | Cosine similarity of speaker embeddings |
| **Latency (30s)** | < 5.0s | Wall time for 30s audio conversion |
| **RTF** | < 5.0x | Real-time factor (wall time / audio duration) |

## Validation Results

Results are saved to `validation_results/`:

### Per-Sample Metrics

JSON files for each test case:
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
  "targets": {
    "max_pitch_rmse_hz": 10.0,
    "min_speaker_similarity": 0.85,
    "max_latency_seconds": 5.0
  },
  "passed": true
}
```

### Aggregated Summary

`validation_results/validation_summary.json`:
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
    "speaker_similarity": {
      "mean": 0.88,
      "std": 0.02,
      "min": 0.85,
      "max": 0.92
    },
    "latency_seconds": {
      "mean": 2.5,
      "std": 0.4,
      "min": 1.8,
      "max": 3.2
    }
  },
  "individual_results": [...]
}
```

## Test Utilities

### Quality Metrics Aggregator

Used for evaluation:
```python
from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

aggregator = QualityMetricsAggregator(sample_rate=44100)
result = aggregator.evaluate(
    source_audio=source_tensor,
    target_audio=converted_tensor,
    align_audio=False,
    target_speaker_embedding=embedding
)

# Access metrics
pitch_rmse = result.pitch_accuracy.rmse_hz
speaker_sim = result.speaker_similarity.cosine_similarity
overall_quality = result.overall_quality_score
```

### Singing Conversion Pipeline

Main conversion interface:
```python
from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

pipeline = SingingConversionPipeline(config={
    'device': 'cuda',
    'preset': 'fast',
    'use_tensorrt': True,
    'tensorrt_precision': 'fp16'
})

result = pipeline.convert_song(
    song_path='audio.wav',
    target_profile_id='profile_id',
    pitch_shift=0.0
)

mixed_audio = result['mixed_audio']
```

## Dependencies

### Core Testing
- pytest >= 6.0
- torch >= 1.10
- numpy
- soundfile
- scipy

### Performance Monitoring
- psutil (memory monitoring)
- nvidia-smi (GPU utilization, installed with CUDA drivers)

### TensorRT Testing
- tensorrt >= 8.5
- CUDA >= 11.8
- NVIDIA GPU with Tensor Cores

## Continuous Integration

### GitHub Actions Example

```yaml
name: System Validation

on: [push, pull_request]

jobs:
  validation:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Generate test data
        run: |
          python tests/data/validation/generate_test_data.py
      
      - name: Run system validation
        run: |
          pytest tests/test_system_validation.py \
            -v -m "system_validation and not (tensorrt or very_slow)" \
            --cov=src/auto_voice \
            --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Test Data Not Found

```bash
# Regenerate test data
python tests/data/validation/generate_test_data.py --samples-per-genre 5
```

### TensorRT Tests Skipped

```bash
# Check TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"

# Verify CUDA
nvidia-smi

# Check GPU capabilities
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Memory Issues (Long Audio Tests)

- Tests marked `@pytest.mark.very_slow` require significant memory
- Skip with: `pytest -m "not very_slow"`
- Ensure ~2GB+ RAM available for 5+ minute tests

### Import Errors

```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

## Future Enhancements

- [ ] Real audio dataset integration
- [ ] Multi-GPU testing
- [ ] Distributed inference validation
- [ ] A/B quality comparison framework
- [ ] Automated regression detection
- [ ] Performance profiling integration
- [ ] Cross-platform validation (Linux, Windows, macOS)

## References

- Comment 1: End-to-end conversion with quality validation
- Comment 2: TensorRT latency target enforcement
- Comment 3: Diverse test data generation
- Comment 9: Edge case handling (short/long/a cappella/processed)

## Contact

For issues or questions about the validation suite, please open an issue on GitHub.
