# AutoVoice CUDA Bindings Test Suite

Quick reference for running and maintaining the CUDA bindings test suite.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov numpy torch torchaudio librosa

# Build CUDA extension
pip install -e .

# Run smoke tests (fast check)
pytest tests/test_bindings_smoke.py -v

# Run all tests
pytest tests/ -v
```

## Test Files

| File | Purpose | Runtime | Tests |
|------|---------|---------|-------|
| `test_bindings_smoke.py` | Basic functionality validation | < 30s | 7 |
| `test_bindings_integration.py` | End-to-end workflows | 1-5 min | 9 |
| `test_bindings_performance.py` | Performance benchmarks | 2-10 min | 9 |

## Common Commands

```bash
# Run by category
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m performance -s    # Performance tests with output
pytest -m cuda             # CUDA tests only
pytest -m "not slow"       # Skip slow tests

# Run specific test
pytest tests/test_bindings_smoke.py::test_function_callable -v

# Run with coverage
pytest tests/ --cov=src/cuda_kernels --cov-report=html

# Run smoke test script directly
python tests/test_bindings_smoke.py
```

## Test Markers

- `unit` - Fast, isolated tests
- `integration` - Component interaction tests
- `e2e` - Complete workflow tests
- `slow` - Tests > 1 second
- `cuda` - CUDA-dependent tests
- `performance` - Benchmarking tests
- `audio` - Audio processing tests

## Expected Performance

| Metric | Target |
|--------|--------|
| Short audio (1s) | < 5 ms, > 200x real-time |
| Medium audio (10s) | < 20 ms, > 500x real-time |
| Long audio (60s) | < 100 ms, > 600x real-time |
| CUDA vs CPU speedup | 10-30x |

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_bindings_smoke.py         # Smoke tests
├── test_bindings_integration.py   # Integration tests
└── test_bindings_performance.py   # Performance tests
```

## Troubleshooting

**Import fails**: `pip install -e . --force-reinstall`

**CUDA OOM**: Clear cache with `torch.cuda.empty_cache()`

**Tests hang**: Run with `CUDA_LAUNCH_BLOCKING=1 pytest tests/`

**See also**: [Complete Testing Guide](../docs/testing_guide.md)

## Documentation

- 📖 [Testing Guide](../docs/testing_guide.md) - Comprehensive testing documentation
- 📊 [Test Suite Summary](../docs/test_suite_summary.md) - Test coverage and statistics
- 🔧 [Implementation Summary](../docs/IMPLEMENTATION_SUMMARY.md) - CUDA implementation details

## CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run CUDA tests
  run: |
    pytest tests/test_bindings_smoke.py -v
    pytest tests/ -v -m "integration and not slow"
```

## Contributing

When adding tests:
1. Use fixtures from `conftest.py`
2. Add appropriate markers (`@pytest.mark.cuda`, etc.)
3. Include docstrings explaining what is tested
4. Use meaningful assertion messages
5. Clean up GPU memory after tests

## Test Coverage

Current targets:
- Overall: > 80%
- Bindings: > 90%
- Kernels: > 75%

View coverage:
```bash
pytest tests/ --cov=src/cuda_kernels --cov-report=html
firefox htmlcov/index.html
```
