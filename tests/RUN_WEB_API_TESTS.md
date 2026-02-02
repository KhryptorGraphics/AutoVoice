# Running Web API Tests

Quick reference for executing Web API test suites.

## Quick Start

```bash
# Run all web API tests (202 tests)
pytest tests/test_web_api_*.py -v

# Run with coverage report
pytest tests/test_web_api_*.py --cov=src/auto_voice/web --cov-report=html

# Run specific test file
pytest tests/test_web_api_comprehensive.py -v

# Run specific test class
pytest tests/test_web_api_comprehensive.py::TestTrainingJobsListEndpoint -v

# Run specific test
pytest tests/test_web_api_comprehensive.py::TestTrainingJobsListEndpoint::test_list_training_jobs_returns_array -v
```

## Test Files

### test_web_api_comprehensive.py (92 tests)
**Coverage:** Tasks 4.3-4.6 comprehensive endpoint testing

Run:
```bash
pytest tests/test_web_api_comprehensive.py -v
```

**Endpoints tested:**
- Training job management (12 tests)
- Profile sample management (19 tests)
- Audio diarization (10 tests)
- Utility endpoints (19 tests)
- Error handling (10 tests)
- Parameter validation (8 tests)
- File uploads (6 tests)
- Content-type validation (8 tests)

### test_web_api_training.py (16 tests)
**Coverage:** Training job endpoints (all passing)

Run:
```bash
pytest tests/test_web_api_training.py -v
```

**Endpoints:**
- GET /api/v1/training/jobs
- POST /api/v1/training/jobs
- GET /api/v1/training/jobs/{id}
- POST /api/v1/training/jobs/{id}/cancel

### test_web_api_profiles.py (17 tests)
**Coverage:** Profile sample management (16/17 passing)

Run:
```bash
pytest tests/test_web_api_profiles.py -v
```

**Endpoints:**
- GET /api/v1/profiles/{id}/samples
- POST /api/v1/profiles/{id}/samples
- POST /api/v1/profiles/{id}/samples/from-path
- GET /api/v1/profiles/{id}/samples/{sid}
- DELETE /api/v1/profiles/{id}/samples/{sid}

### test_web_api_audio.py (15 tests)
**Coverage:** Audio processing endpoints

Run:
```bash
pytest tests/test_web_api_audio.py -v
```

**Endpoints:**
- POST /api/v1/audio/diarize
- POST /api/v1/audio/diarize/assign
- POST /api/v1/profiles/auto-create

### test_web_api_utility.py (20 tests)
**Coverage:** System utility endpoints

Run:
```bash
pytest tests/test_web_api_utility.py -v
```

**Endpoints:**
- GET /api/v1/health, /ready
- GET /api/v1/gpu/metrics, /system/info
- GET /api/v1/devices/list
- POST /api/v1/youtube/info, /youtube/download
- GET /api/v1/models/loaded
- POST /api/v1/models/load
- POST /api/v1/models/tensorrt/rebuild
- GET /api/v1/kernels/metrics

### test_web_api_edge_cases.py (30 tests)
**Coverage:** Edge cases and validation

Run:
```bash
pytest tests/test_web_api_edge_cases.py -v
```

**Focus:**
- Empty/invalid filenames
- Large file uploads
- Parameter boundary values
- Content-type edge cases

### test_web_api.py (12 tests - legacy)
**Coverage:** Original API tests

Run:
```bash
pytest tests/test_web_api.py -v
```

## Coverage Analysis

### Generate HTML Coverage Report
```bash
# Generate full coverage report
pytest tests/test_web_api_*.py \
  --cov=src/auto_voice/web \
  --cov-report=html \
  --cov-report=term-missing

# Open report in browser
xdg-open htmlcov/index.html
```

### Current Coverage (as of 2026-02-02)
```
Module                    Coverage  Lines
─────────────────────────────────────────
web/api.py                    35%   2026
web/app.py                    85%     81
web/openapi_spec.py           81%    118
web/utils.py                 100%      6
web/job_manager.py            30%    160
web/audio_router.py           23%     78
web/karaoke_api.py            19%    406
─────────────────────────────────────────
TOTAL                         32%   3753
```

## Test Execution Tips

### Run Fast (Skip Slow Tests)
```bash
pytest tests/test_web_api_*.py -v -m "not slow"
```

### Run Failed Tests Only
```bash
# First run to generate failure cache
pytest tests/test_web_api_*.py -v

# Re-run only failures
pytest tests/test_web_api_*.py -v --lf
```

### Run in Parallel (requires pytest-xdist)
```bash
pip install pytest-xdist
pytest tests/test_web_api_*.py -v -n auto
```

### Verbose Output with Captured Print
```bash
pytest tests/test_web_api_*.py -v -s
```

### Stop on First Failure
```bash
pytest tests/test_web_api_*.py -v -x
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Web API Tests
  run: |
    pytest tests/test_web_api_*.py \
      --cov=src/auto_voice/web \
      --cov-report=xml \
      --junitxml=junit.xml \
      -v

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Jenkins
```groovy
stage('Web API Tests') {
    steps {
        sh '''
            pytest tests/test_web_api_*.py \
              --cov=src/auto_voice/web \
              --cov-report=html \
              --junitxml=results.xml \
              -v
        '''
    }
    post {
        always {
            junit 'results.xml'
            publishHTML([
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
    }
}
```

## Troubleshooting

### Import Errors
```bash
# Ensure PYTHONPATH includes src/
export PYTHONPATH=/home/kp/repo2/autovoice:$PYTHONPATH
pytest tests/test_web_api_*.py -v
```

### Fixture Not Found
```bash
# Check conftest.py is present
ls tests/conftest.py

# Run with explicit conftest
pytest tests/test_web_api_*.py -v --co
```

### Mock Setup Failures
```bash
# Run single test with full traceback
pytest tests/test_web_api_comprehensive.py::TestTrainingJobsListEndpoint::test_list_training_jobs_returns_array -vvs --tb=long
```

### Coverage Not Working
```bash
# Install pytest-cov
pip install pytest-cov

# Verify installation
pytest --version
```

## Test Maintenance

### Adding New Endpoint Tests

1. Choose appropriate test file:
   - **test_web_api_comprehensive.py** - For new endpoints
   - **test_web_api_<domain>.py** - For domain-specific endpoints

2. Follow existing test structure:
```python
class TestNewEndpoint:
    """Test POST /api/v1/new/endpoint."""

    def test_success_case(self, client, app_with_mocks):
        """Test successful request."""
        response = client.post('/api/v1/new/endpoint', json={...})
        assert response.status_code == 200

    def test_missing_parameter(self, client):
        """Returns 400 when parameter missing."""
        response = client.post('/api/v1/new/endpoint', json={})
        assert response.status_code == 400

    def test_not_found(self, client, app_with_mocks):
        """Returns 404 when resource not found."""
        # Mock to return None
        response = client.post('/api/v1/new/endpoint', json={...})
        assert response.status_code == 404
```

3. Add mocks in fixtures:
```python
@pytest.fixture
def app_with_mocks():
    """Mock dependencies."""
    with patch('auto_voice.module.Class') as MockClass:
        mock_instance = MagicMock()
        mock_instance.method.return_value = {...}
        MockClass.return_value = mock_instance
        # ... rest of fixture
```

### Updating Tests After API Changes

1. Check what changed:
```bash
git diff main src/auto_voice/web/api.py
```

2. Find affected tests:
```bash
grep -r "api/v1/changed/endpoint" tests/
```

3. Update test expectations:
   - Parameter names
   - Response structure
   - Error codes

4. Re-run tests:
```bash
pytest tests/test_web_api_*.py -v -k "changed_endpoint"
```

## Documentation

- **Test Summary:** [WEB_API_TEST_SUMMARY.md](./WEB_API_TEST_SUMMARY.md)
- **Test Plan:** [../conductor/tracks/comprehensive-testing-coverage_20260201/plan.md](../conductor/tracks/comprehensive-testing-coverage_20260201/plan.md)
- **API Documentation:** [../src/auto_voice/web/api_docs.py](../src/auto_voice/web/api_docs.py)

## Support

For issues or questions:
1. Check test output for specific error messages
2. Review WEB_API_TEST_SUMMARY.md for known limitations
3. Check git history for recent test changes
4. Review Flask test client docs: https://flask.palletsprojects.com/en/latest/testing/
