# System Validation Test Suite

Comprehensive validation testing for AutoVoice singing voice conversion system.

## Overview

The system validation test suite provides end-to-end testing with automated quality checks, diverse test samples, and edge case handling. It addresses Comments 1, 3, and 10 from the implementation requirements.

## Features

### ✅ End-to-End Conversion Tests (Comment 1)
- Complete pipeline testing: separation → pitch extraction → conversion → mixing
- Automated quality checks:
  - **Pitch RMSE < 10 Hz**
  - **Speaker similarity > 0.85**
  - **Latency < 5s per 30s audio** (RTF < 5.0x)
- Memory usage monitoring
- Validation report generation

### ✅ Diverse Test Data (Comment 3)
- **5 Genres**: pop, rock, jazz, classical, rap
- **Multiple styles** per genre (5-10 samples each)
- **Language variety**: en, es, fr, de, ja (simulated in metadata)
- **F0 range diversity**: 220 Hz - 523 Hz (A3 - C5)
- **Duration variety**: 10s - 30s
- Metadata-driven test execution via `test_set.json`

### ✅ Edge Case Tests (Comment 10)
- **Very short audio** (<10s): 5-second test samples
- **Very long audio** (>5 minutes): 6-minute test with memory monitoring
- **A cappella inputs**: Vocals-only with graceful handling
- **Heavily processed vocals**: Reverb + distortion simulation
- Memory leak detection and resource usage tracking

## Quick Start

### 1. Generate Test Data

```bash
# Generate diverse test samples across all genres
cd /home/kp/autovoice
python tests/data/validation/generate_test_data.py \
    --output tests/data/validation \
    --samples-per-genre 5 \
    --seed 42
```

This creates:
- 25 test audio files (5 genres × 5 samples)
- `test_set.json` metadata file
- Genre-specific samples with varied F0, duration, and styles

### 2. Run System Validation Tests

```bash
# Run all system validation tests
pytest tests/test_system_validation.py -v -m system_validation

# Run specific test categories
pytest tests/test_system_validation.py -v -m edge_cases
pytest tests/test_system_validation.py -v -m genre_specific
pytest tests/test_system_validation.py -v -m performance

# Run with specific genre
pytest tests/test_system_validation.py -v -k "test_genre_conversion[pop]"
```

### 3. View Validation Report

```bash
# Validation report is auto-generated
cat tests/reports/system_validation_report.json
```

## Test Organization

### Test Classes

#### `TestSystemValidation`
Main validation tests enforcing quality targets.

**Key Tests:**
- `test_diverse_genres_conversion`: Tests all samples from test_set.json
  - Validates pitch RMSE < 10 Hz
  - Validates speaker similarity > 0.85
  - Validates latency RTF < 5.0x
  - Monitors memory usage
  - Generates validation report

#### `TestEdgeCases`
Edge case handling tests.

**Key Tests:**
- `test_very_short_audio`: 5-second audio conversion
- `test_very_long_audio`: 6-minute audio with memory monitoring
- `test_a_cappella_input`: Vocals-only (no instrumental separation)
- `test_heavily_processed_vocals`: Reverb + distortion effects

#### `TestGenreSpecificValidation`
Genre-specific validation tests.

**Key Tests:**
- `test_genre_conversion[genre]`: Parametrized test for each genre
  - Tests: pop, rock, jazz, classical, rap

#### `TestPerformanceValidation`
Performance and latency validation.

**Key Tests:**
- `test_latency_scaling`: Validates linear latency scaling with duration

## Test Data Structure

### test_set.json Schema

```json
{
  "test_cases": [
    {
      "id": "pop_001",
      "source_audio": "tests/data/validation/pop_001.wav",
      "target_profile_id": "profile_pop_1",
      "metadata": {
        "genre": "pop",
        "duration_sec": 10.0,
        "f0_range": {"min": 209.0, "max": 231.0},
        "language": "en",
        "base_freq_hz": 220.0,
        "sample_rate": 44100,
        "synthetic": true
      }
    },
    // ... more test cases
  ],
  "generation_config": {
    "samples_per_genre": 5,
    "genres": ["pop", "rock", "jazz", "classical", "rap"],
    "seed": 42,
    "sample_rate": 44100,
    "synthetic": true
  },
  "statistics": {
    "total_samples": 25,
    "genres": {"pop": 5, "rock": 5, ...},
    "duration_range": {"min": 10.0, "max": 30.0},
    "f0_range": {"min": 220.0, "max": 523.0}
  }
}
```

## Validation Report

### Report Structure

Generated at `tests/reports/system_validation_report.json`:

```json
{
  "timestamp": "2025-01-15 14:30:00",
  "total_tests": 25,
  "passed_tests": 23,
  "failed_tests": 2,
  "aggregate_metrics": {
    "pitch_rmse_hz": {
      "mean": 7.2,
      "std": 1.8,
      "min": 4.5,
      "max": 9.8
    },
    "speaker_similarity": {
      "mean": 0.88,
      "std": 0.03,
      "min": 0.84,
      "max": 0.92
    },
    "latency_seconds": {
      "mean": 45.2,
      "std": 12.1,
      "min": 28.5,
      "max": 68.3
    }
  },
  "individual_results": [
    {
      "test_case_id": "pop_001",
      "genre": "pop",
      "pitch_rmse_hz": 6.8,
      "speaker_similarity": 0.89,
      "latency_seconds": 42.1,
      "memory_usage_mb": 512.3,
      "passed": true
    },
    // ... more results
  ]
}
```

## Quality Targets

All tests enforce these quality targets:

| Metric | Target | Validation |
|--------|--------|------------|
| Pitch RMSE | < 10 Hz | ✓ Automated |
| Speaker Similarity | > 0.85 | ✓ Automated |
| Latency (30s audio) | < 5s | ✓ Automated |
| Memory Usage (6min audio) | < 2 GB | ✓ Monitored |
| Output Quality | No NaN/Inf | ✓ Checked |

## Genre Characteristics

### Pop
- Clear melody with steady vibrato (5 Hz)
- Clean harmonics (fundamental + 2 overtones)
- Moderate noise level (0.02)

### Rock
- Aggressive sound with distortion
- Strong harmonics with soft clipping
- Higher noise level (0.05)

### Jazz
- Variable pitch with swing feel
- Complex harmonics (4 overtones)
- Pitch bends for improvisation

### Classical
- Pure tone with subtle vibrato (5.5 Hz)
- Dynamic expression (crescendo/decrescendo)
- Minimal noise (0.01)

### Rap
- Monotone with minimal pitch variation
- Rhythmic patterns at 2 Hz
- Percussive elements

## Pytest Markers

Use these markers to run specific test categories:

```bash
# All system validation tests
pytest -m system_validation

# Edge case tests only
pytest -m edge_cases

# Genre-specific tests only
pytest -m genre_specific

# Performance tests only
pytest -m performance

# Slow tests (>30s each)
pytest -m slow
```

## Fixtures

### `test_metadata_loader`
Loads `test_set.json` metadata for test execution.

### `validation_pipeline`
Creates `SingingConversionPipeline` instance for conversions.

### `quality_evaluator`
Creates `VoiceConversionEvaluator` for quality metrics.

### `memory_monitor`
Monitors memory usage during test execution.

## Integration with CI/CD

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
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Generate test data
        run: |
          python tests/data/validation/generate_test_data.py
      - name: Run system validation
        run: |
          pytest tests/test_system_validation.py -v \
            -m system_validation \
            --junitxml=junit.xml \
            --cov=auto_voice
      - name: Upload validation report
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: tests/reports/system_validation_report.json
```

## Troubleshooting

### Test Data Not Found
```bash
# Error: test_set.json not found
# Solution: Generate test data first
python tests/data/validation/generate_test_data.py
```

### Memory Issues
```bash
# For systems with limited memory, reduce test scope
pytest tests/test_system_validation.py -v -m "not slow"
```

### CUDA Out of Memory
```bash
# Run tests on CPU
pytest tests/test_system_validation.py -v --device=cpu
```

### Long Test Duration
```bash
# Run subset of tests
pytest tests/test_system_validation.py -v -k "pop or rock"
```

## Development

### Adding New Genres

1. Add genre generator to `GenreAudioGenerator` class
2. Add genre to `genres` list in `generate_test_dataset()`
3. Update documentation with genre characteristics

### Adding New Edge Cases

1. Add test method to `TestEdgeCases` class
2. Implement edge case scenario
3. Add validation assertions
4. Document expected behavior

### Customizing Quality Targets

Quality targets are defined in test assertions. To modify:

```python
# In test_system_validation.py
assert pitch_rmse_hz < 10.0  # Modify threshold here
assert speaker_similarity > 0.85  # Modify threshold here
assert latency / duration_sec < 5.0  # Modify RTF threshold here
```

## References

- **Comment 1**: End-to-end validation requirements
- **Comment 3**: Diverse test data generation requirements
- **Comment 10**: Edge case testing requirements
- `test_end_to_end.py`: Existing E2E test patterns
- `evaluate_voice_conversion.py`: Quality evaluation example
- `generate_test_data.py`: Synthetic data generation reference

## Support

For issues or questions:
1. Check test output and validation report
2. Review error messages and stack traces
3. Verify test data was generated correctly
4. Check GPU/memory availability
5. Review pytest markers and test selection
