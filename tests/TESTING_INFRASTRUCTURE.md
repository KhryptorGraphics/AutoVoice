# AutoVoice Testing Infrastructure

## Overview

Comprehensive pytest-based testing infrastructure for the AutoVoice voice conversion system. Provides fixtures, mocks, utilities, and integration test suites with >90% code coverage support.

## Directory Structure

```
tests/
├── fixtures/              # Pytest fixtures (NEW)
│   ├── __init__.py       # Fixture exports
│   ├── audio_fixtures.py # Audio generation & manipulation
│   ├── model_fixtures.py # Model mocks & checkpoints
│   ├── gpu_fixtures.py   # GPU/CUDA testing utilities
│   ├── mock_fixtures.py  # External dependency mocks
│   ├── integration_fixtures.py  # E2E workflow testing
│   └── performance_fixtures.py  # Benchmarking & profiling
├── utils/                # Testing utilities (NEW)
│   ├── __init__.py
│   └── test_helpers.py   # Assertion helpers
├── conftest.py           # Global pytest configuration (EXISTING)
├── sample_data_generator.py  # Test data generation (EXISTING)
└── test_*.py             # Test modules (EXISTING)
```

## New Fixtures

### Audio Fixtures (`audio_fixtures.py`)

Comprehensive audio generation for testing:

```python
# Audio factory - generate various audio types
def test_conversion(sample_audio_factory):
    sine_wave = sample_audio_factory('sine', frequency=440, duration=2.0)
    harmonics = sample_audio_factory('harmonics', fundamental=220, num_harmonics=5)
    speech_like = sample_audio_factory('speech_like', formants=[800, 1200, 2500])

# File factory - create temporary audio files
def test_file_io(audio_file_factory):
    audio_path = audio_file_factory('test.wav', audio_data, sample_rate=22050)

# Multi-channel audio
def test_stereo(multi_channel_audio):
    stereo = multi_channel_audio(num_channels=2, relationship='stereo_field')
    surround = multi_channel_audio(num_channels=6, relationship='independent')

# Batch generator
def test_batching(audio_batch_generator):
    for batch in audio_batch_generator(batch_size=16, num_batches=10):
        process_batch(batch)

# Edge cases
def test_edge_cases(corrupted_audio_samples):
    samples = corrupted_audio_samples
    assert handle_clipping(samples['clipped'])
    assert handle_silent(samples['silent'])
    assert handle_dc_offset(samples['dc_offset'])

# Mel-spectrograms
def test_features(mel_spectrogram_factory):
    mel = mel_spectrogram_factory(audio, n_mels=80, fmax=8000)

# Pitch contours
def test_pitch(pitch_contour_factory):
    pitch = pitch_contour_factory(length=100, pattern='rising', vibrato=True)
```

### Model Fixtures (`model_fixtures.py`)

Mock models and checkpoints:

```python
# Mock models - no weight loading
def test_pipeline(mock_voice_model, mock_encoder, mock_decoder, mock_vocoder):
    embedding = mock_encoder(mel_spec)
    audio = mock_decoder(embedding)
    waveform = mock_vocoder(mel)

# Checkpoint factory
def test_loading(trained_model_checkpoint):
    ckpt = trained_model_checkpoint(
        model_type='voice_transformer',
        epoch=100,
        include_optimizer=True
    )
    model.load_checkpoint(ckpt)

# Config factory
def test_config(model_config_factory):
    config = model_config_factory('transformer', hidden_size=512)

# Forward pass testing
def test_model(model_forward_tester):
    results = model_forward_tester.test_forward(
        model, input_tensor,
        expected_shape=(16, 100, 80),
        check_gradients=True
    )

# Parameter counting
def test_params(parameter_counter):
    stats = parameter_counter(model)
    assert stats['trainable'] > 1_000_000
```

### GPU Fixtures (`gpu_fixtures.py`)

CUDA memory tracking and multi-GPU testing:

```python
# GPU context manager
def test_gpu_ops(gpu_context_manager):
    with gpu_context_manager() as ctx:
        tensor = torch.randn(1000, 1000, device='cuda')
        result = model(tensor)
    print(f"Peak memory: {ctx.peak_memory_mb}MB")

# Memory tracker
def test_memory(cuda_memory_tracker):
    tracker = cuda_memory_tracker
    tracker.start()

    # ... GPU operations ...
    tracker.checkpoint('after_forward')

    stats = tracker.stop()
    assert stats['leaked_mb'] < 10

# Multi-GPU
def test_distributed(multi_gpu_config):
    if multi_gpu_config.num_gpus > 1:
        model = multi_gpu_config.data_parallel(model)

# Stress testing
def test_limits(gpu_stress_tester):
    max_batch = gpu_stress_tester.find_max_batch_size(model, input_shape)
```

### Mock Fixtures (`mock_fixtures.py`)

Isolated testing without external dependencies:

```python
# File system mock
def test_io(mock_file_system):
    mock_file_system.write('config.json', '{}')
    content = mock_file_system.read('config.json')

# Audio loader mock
def test_loading(mock_audio_loader):
    audio, sr = mock_audio_loader.load('song.wav')  # Returns synthetic audio

# Network client mock
def test_api(mock_network_client):
    mock_network_client.set_response('/api/models', {'status': 'ok'})
    response = mock_network_client.get('/api/models')

# Cache manager mock
def test_caching(mock_cache_manager):
    mock_cache_manager.set('key', value, ttl=60)
    cached = mock_cache_manager.get('key')

# Database mock
def test_db(mock_database):
    mock_database.insert('users', {'name': 'Alice'})
    users = mock_database.select('users', where={'name': 'Alice'})
```

### Integration Fixtures (`integration_fixtures.py`)

End-to-end workflow testing:

```python
# Pipeline test suite
def test_pipeline(pipeline_test_suite):
    suite = pipeline_test_suite
    suite.add_test_case('test1', audio, profile, expected_metrics)

    results = suite.run_pipeline(conversion_pipeline)
    validation = suite.validate_results()

# E2E workflow
def test_workflow(end_to_end_workflow):
    workflow = end_to_end_workflow
    workflow.setup_test_data(duration=3.0)
    workflow.run_conversion()
    assert workflow.validate_quality()

# Concurrent testing
def test_concurrent(concurrent_pipeline_tester):
    results = concurrent_pipeline_tester.run_concurrent(
        pipeline, test_data, num_workers=4
    )

# Data flow validation
def test_dataflow(data_flow_validator):
    validator = data_flow_validator
    validator.add_checkpoint('encoder', output, expected_shape=(16, 256))
    assert validator.validate_all()
```

### Performance Fixtures (`performance_fixtures.py`)

Benchmarking and profiling:

```python
# Benchmarker
def test_perf(performance_benchmarker):
    bench = performance_benchmarker

    with bench.measure('inference'):
        output = model(input)

    stats = bench.get_statistics('inference')
    assert stats['mean'] < 0.1  # < 100ms

# Resource profiler
def test_resources(resource_profiler):
    profiler = resource_profiler

    with profiler.profile():
        # ... operations ...

    summary = profiler.get_summary()
    assert summary['memory']['delta_mb'] < 100

# Throughput tester
def test_throughput(throughput_tester):
    rtf = throughput_tester.measure_rtf(process_func, audio, sample_rate)
    assert rtf['mean_rtf'] < 0.5  # Faster than real-time

# Regression testing
def test_regression(regression_tester):
    regression_tester.set_baseline('inference_time', 0.05)
    current = regression_tester.measure('inference_time', lambda: model(input))
    assert regression_tester.check_regression('inference_time')
```

## Test Utilities (`utils/test_helpers.py`)

Reusable assertion and validation functions:

```python
from tests.utils import (
    assert_audio_equal,
    assert_audio_normalized,
    compute_snr,
    compute_similarity,
    assert_model_outputs_valid,
    assert_gradients_exist,
    assert_gpu_memory_efficient,
    assert_performance_threshold,
    assert_realtime_factor,
)

def test_audio_comparison():
    assert_audio_equal(output, expected, rtol=1e-5)
    assert_audio_normalized(audio, max_value=1.0)

    snr = compute_snr(signal, noise)
    similarity = compute_similarity(audio1, audio2)

def test_model_validation():
    assert_model_outputs_valid(
        output,
        check_nan=True,
        check_range=(-1.0, 1.0)
    )
    assert_gradients_exist(model)

def test_performance():
    assert_performance_threshold(elapsed, max_time=0.1)
    assert_realtime_factor(processing_time, audio_duration, max_rtf=1.0)
```

## Usage Examples

### Unit Testing with Mocks

```python
import pytest

def test_voice_conversion_unit(mock_voice_model, sample_audio_factory):
    """Unit test with mocked model."""
    audio = sample_audio_factory('harmonics', fundamental=220)
    output = mock_voice_model(torch.from_numpy(audio))

    assert output.shape[-1] == 80  # Mel bins
```

### Integration Testing

```python
def test_full_pipeline(
    pipeline_test_suite,
    sample_audio_factory,
    cuda_memory_tracker
):
    """Integration test with memory tracking."""
    # Setup
    suite = pipeline_test_suite
    audio = sample_audio_factory('speech_like', duration=3.0)

    # Track memory
    cuda_memory_tracker.start()

    # Run pipeline
    suite.add_test_case('case1', audio, target_profile={})
    results = suite.run_pipeline(conversion_pipeline)

    # Validate
    memory_stats = cuda_memory_tracker.stop()
    assert memory_stats['leaked_mb'] < 10
    assert suite.validate_results()['success_rate'] > 0.9
```

### Performance Testing

```python
def test_inference_performance(
    performance_benchmarker,
    resource_profiler,
    sample_audio_factory
):
    """Benchmark inference performance."""
    audio = sample_audio_factory('sine', duration=5.0)

    # Benchmark
    bench = performance_benchmarker
    stats = bench.benchmark(
        lambda: model(audio),
        iterations=100,
        warmup=10
    )

    # Resource profiling
    with resource_profiler.profile() as prof:
        model(audio)

    # Assertions
    assert stats['mean'] < 0.1  # < 100ms
    assert prof.get_summary()['cpu']['max_percent'] < 80
```

## Test Organization

### Test Markers

Use pytest markers to organize tests:

```python
@pytest.mark.unit
def test_unit():
    """Fast, isolated unit tests."""

@pytest.mark.integration
def test_integration():
    """Component interaction tests."""

@pytest.mark.e2e
def test_e2e():
    """End-to-end workflow tests."""

@pytest.mark.slow
def test_slow():
    """Long-running tests (>1s)."""

@pytest.mark.cuda
def test_cuda():
    """GPU-specific tests."""

@pytest.mark.performance
def test_performance():
    """Performance benchmarks."""
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run by marker
pytest tests/ -m unit              # Fast unit tests
pytest tests/ -m integration       # Integration tests
pytest tests/ -m "not slow"        # Skip slow tests
pytest tests/ -m cuda              # GPU tests only

# With coverage
pytest tests/ --cov=src --cov-report=html

# Parallel execution
pytest tests/ -n auto

# Verbose output
pytest tests/ -v --tb=short
```

## Coverage Goals

### Target Coverage: >90%

- **Unit tests**: 95%+ for core modules
- **Integration tests**: 85%+ for pipelines
- **Edge cases**: 100% for error handling

### Coverage Report

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View in browser
open htmlcov/index.html
```

## Best Practices

### 1. Test Isolation

- Use fixtures for setup/teardown
- Mock external dependencies
- Clear CUDA cache between tests
- Use `tmp_path` for file operations

### 2. Deterministic Tests

- Set random seeds: `np.random.seed(42)`
- Use synthetic data instead of random
- Mock time-dependent operations

### 3. Performance Testing

- Warmup iterations before benchmarking
- Multiple iterations for statistics
- Compare against baselines
- Test on both CPU and GPU

### 4. Memory Safety

- Track GPU memory leaks
- Validate memory cleanup
- Test with large batches
- Monitor resource usage

### 5. Error Handling

- Test edge cases explicitly
- Validate error messages
- Test recovery mechanisms
- Check boundary conditions

## Maintenance

### Adding New Fixtures

1. Create fixture in appropriate `fixtures/*.py` file
2. Add to `fixtures/__init__.py` exports
3. Document usage with examples
4. Add tests for the fixture itself

### Updating Baselines

```bash
# Update performance baselines
pytest tests/ --update-baselines

# Review changes
git diff tests/performance_baseline.json
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --junitxml=junit.xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Support

For testing questions or issues:
- Check existing test examples in `tests/test_*.py`
- Review fixture documentation
- Consult pytest documentation: https://docs.pytest.org

---

**Last Updated**: 2025-01-09
**Coverage Target**: >90%
**Test Count**: 50+ test modules, 1000+ test cases
