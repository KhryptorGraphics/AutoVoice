# System Validation Test Suite

Quick reference for running system validation tests.

## Quick Start

```bash
# 1. Generate test data (one-time setup)
python tests/data/validation/generate_test_data.py

# 2. Run all validation tests
pytest tests/test_system_validation.py -v -m system_validation

# 3. View results
cat tests/reports/system_validation_report.json
```

## Test Categories

### All Validation Tests
```bash
pytest tests/test_system_validation.py -v -m system_validation
```

### Edge Cases Only
```bash
pytest tests/test_system_validation.py -v -m edge_cases
```

### Genre-Specific Tests
```bash
pytest tests/test_system_validation.py -v -m genre_specific
```

### Performance Tests
```bash
pytest tests/test_system_validation.py -v -m performance
```

### Specific Genre
```bash
pytest tests/test_system_validation.py -v -k "pop"
pytest tests/test_system_validation.py -v -k "jazz"
```

## Requirements Addressed

- ✅ **Comment 1**: End-to-end conversion tests with automated quality checks
- ✅ **Comment 3**: Diverse test data generation (5 genres, multiple styles)
- ✅ **Comment 10**: Edge case tests (short, long, a cappella, processed vocals)

## Quality Targets

- Pitch RMSE: < 10 Hz
- Speaker Similarity: > 0.85
- Latency: < 5s per 30s audio (RTF < 5.0x)
- Memory Usage: < 2 GB for 6-minute audio

## Output

- Test results: Terminal output with pass/fail status
- Validation report: `tests/reports/system_validation_report.json`
- Aggregate metrics: Mean, std, min, max for all quality metrics

## Documentation

See `docs/system_validation_guide.md` for comprehensive documentation.
