# API E2E Testing Guide

## Overview

Comprehensive API contract validation E2E tests for AutoVoice, implementing Comment 12 requirements.

## Test Coverage

### 1. Health Endpoints (`TestAPIHealthEndpoints`)
- **Liveness probe** (`/health/live`) - Application running check
- **Readiness probe** (`/health/ready`) - Ready to serve traffic check
- **API health** (`/api/v1/health`) - Detailed component status
- **GPU status** (`/api/v1/gpu_status`) - GPU availability and metrics

### 2. Voice Cloning Workflow (`TestVoiceCloningWorkflow`)
- Create voice profile from reference audio
- List all voice profiles
- Get specific profile details
- Delete voice profile
- Full lifecycle validation

### 3. Conversion Workflow (`TestConversionAPIWorkflow`)
- Create voice profile
- Convert song with target profile
- Validate conversion results
- Quality metrics verification
- Audio format validation

### 4. Quality Metrics (`TestQualityMetricsValidation`)
- Audio analysis endpoint testing
- Pitch extraction validation
- Voice activity detection (VAD)
- Spectrogram generation
- Audio statistics verification

### 5. Error Handling (`TestErrorHandlingAndRecovery`)
- Invalid audio format handling
- Missing required fields
- Nonexistent profile errors
- Invalid parameter ranges
- Comprehensive error responses

### 6. Concurrent Requests (`TestConcurrentRequests`)
- Multiple simultaneous health checks
- Load testing capabilities
- Thread safety validation

## Running Tests

### Basic Execution
```bash
# Run all API E2E tests
pytest tests/test_api_e2e_validation.py -v

# Run specific test class
pytest tests/test_api_e2e_validation.py::TestAPIHealthEndpoints -v

# Run with detailed output
pytest tests/test_api_e2e_validation.py -v -s --tb=short
```

### With Coverage
```bash
pytest tests/test_api_e2e_validation.py --cov=src/auto_voice/web --cov-report=html
```

### Continuous Integration
```bash
# Run with pytest markers
pytest -m "api and e2e" tests/test_api_e2e_validation.py
```

## Test Fixtures

### `flask_server`
- **Scope**: Module-level
- **Purpose**: Starts Flask server for E2E testing
- **Port**: 5001
- **Startup timeout**: 30 seconds
- **Cleanup**: Automatic via daemon thread

### `sample_audio_file`
- **Purpose**: Generate 30-second WAV for voice cloning
- **Format**: 22050 Hz, mono, 16-bit PCM
- **Content**: Sine wave at 440 Hz

### `sample_song_file`
- **Purpose**: Generate 3-second WAV for conversion
- **Format**: 22050 Hz, mono, 16-bit PCM
- **Content**: Mixed sine waves (vocals + instrumental)

## Validation Results

All tests save results to `validation_results/` directory:

```
validation_results/
├── liveness_results.json
├── readiness_results.json
├── api_health_results.json
├── gpu_status_results.json
├── voice_cloning_workflow_results.json
├── conversion_workflow_results.json
├── audio_analysis_results.json
├── process_audio_quality_results.json
├── error_invalid_format_results.json
├── error_missing_fields_results.json
├── error_nonexistent_profile_results.json
├── error_invalid_volumes_results.json
└── concurrent_health_checks_results.json
```

### Result Format
```json
{
  "status_code": 200,
  "response": {
    "status": "success",
    "data": {}
  },
  "passed": true
}
```

## Quality Metrics Validation

### Audio Analysis Metrics
- **Duration**: Audio length in seconds
- **Sample rate**: Audio sampling frequency
- **Channels**: Number of audio channels
- **Statistics**:
  - Mean amplitude
  - Standard deviation
  - Min/max values
  - RMS (Root Mean Square)

### Pitch Analysis
- **Mean pitch**: Average fundamental frequency (Hz)
- **Std pitch**: Pitch variability (Hz)
- **Min/max pitch**: Pitch range (Hz)

### Voice Activity Detection (VAD)
- **Voice ratio**: Percentage of voiced frames
- **Segments**: Voice activity segments

### Conversion Quality
- **Target profile match**: Profile ID verification
- **Volume levels**: Vocal/instrumental balance
- **F0 statistics**: Pitch transformation metrics

## Error Scenarios

### Expected Error Responses

#### 400 Bad Request
- Invalid audio format
- Missing required fields
- Invalid parameter values
- Malformed JSON

#### 404 Not Found
- Nonexistent profile ID
- Invalid endpoint

#### 503 Service Unavailable
- Service not initialized
- Component failures

## Performance Benchmarks

### Response Time Targets
- **Health endpoints**: < 100ms
- **Audio analysis**: < 500ms
- **Voice cloning**: < 5s (30s audio)
- **Song conversion**: < 10s (3s audio)

### Concurrent Request Handling
- **Health checks**: 10+ concurrent requests
- **API endpoints**: 5+ concurrent requests

## Integration with Validation Report

Test results integrate into the validation report:

```python
# In validation report generation
with open('validation_results/conversion_workflow_results.json') as f:
    api_results = json.load(f)

report['api_validation'] = {
    'health_checks': api_results.get('health', {}),
    'conversion_quality': api_results.get('conversion', {}),
    'error_handling': api_results.get('errors', {})
}
```

## Troubleshooting

### Server Startup Issues
```bash
# Check port availability
lsof -i :5001

# Increase startup timeout in fixture
max_attempts = 60  # 60 seconds
```

### Test Failures
```bash
# Run with verbose output
pytest tests/test_api_e2e_validation.py -vv -s

# Check validation results
cat validation_results/conversion_workflow_results.json
```

### Service Unavailable
```python
# Tests handle service unavailability gracefully
if response.status_code == 503:
    pytest.skip("Service unavailable")
```

## Best Practices

1. **Always check health endpoints first**
   ```python
   response = requests.get(f'{API_BASE_URL}/health/ready')
   if response.status_code != 200:
       pytest.skip("Service not ready")
   ```

2. **Clean up resources**
   ```python
   # Always delete created profiles
   requests.delete(f'/api/v1/voice/profiles/{profile_id}')
   ```

3. **Validate all response fields**
   ```python
   assert 'profile_id' in response.json()
   assert 'duration' in response.json()
   ```

4. **Save validation results**
   ```python
   save_validation_results('test_name', {
       'status_code': response.status_code,
       'passed': True
   })
   ```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run API E2E Tests
  run: |
    pytest tests/test_api_e2e_validation.py -v --junit-xml=test-results.xml

- name: Upload Validation Results
  uses: actions/upload-artifact@v2
  with:
    name: validation-results
    path: validation_results/
```

## Future Enhancements

1. **WebSocket E2E Tests**
   - Real-time conversion progress
   - Status polling
   - Cancellation handling

2. **Performance Profiling**
   - Response time percentiles
   - Memory usage tracking
   - GPU utilization monitoring

3. **Load Testing**
   - Concurrent conversion requests
   - Throughput benchmarks
   - Stress testing

4. **Security Testing**
   - Authentication validation
   - Rate limiting tests
   - Input sanitization
