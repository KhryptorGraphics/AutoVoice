# Validation Results Directory

This directory contains automated validation results from API E2E quality tests.

## Quick Start

### Run API E2E Quality Validation Tests

```bash
# Navigate to project root
cd /home/kp/autovoice

# Run all API E2E quality validation tests
pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation -v -s

# Run single conversion quality test only
pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation::test_api_e2e_conversion_quality_validation -v -s

# Run batch quality validation test only
pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation::test_api_e2e_batch_quality_validation -v -s
```

## Generated Files

### Single Conversion Quality Test

**File:** `api_e2e_quality_metrics.json`
- Detailed quality metrics for single conversion
- Includes pitch RMSE, speaker similarity, spectral distortion, MOS, STOI, PESQ

**File:** `api_e2e_quality_validation_results.json`
- Summary of quality validation pass/fail status
- Quality targets comparison

### Batch Quality Validation Test

**Files:** `api_e2e_batch_conversion_{1,2,3}.json`
- Individual metrics for each conversion in batch

**File:** `api_e2e_batch_summary.json`
- Aggregate statistics across all conversions
- Mean, standard deviation for each metric
- Overall pass/fail status

## Quality Targets (Comment 10)

| Metric | Target | Status |
|--------|--------|--------|
| Pitch RMSE (Hz) | < 10.0 Hz | Required ✅ |
| Speaker Similarity | > 0.85 | Required ✅ |
| Overall Quality | > 0.0 | Informational ℹ️ |

## Example Metrics Output

```json
{
  "pitch_rmse_hz": 8.45,
  "pitch_rmse_log2": 0.12,
  "pitch_correlation": 0.92,
  "speaker_similarity": 0.88,
  "spectral_distortion": 6.3,
  "mos_estimation": 3.8,
  "stoi_score": 0.85,
  "pesq_score": 2.9,
  "overall_quality_score": 0.79
}
```

## Interpreting Results

### Pitch RMSE (Hz)
- **Good:** < 5 Hz (excellent pitch accuracy)
- **Acceptable:** 5-10 Hz (meets target)
- **Poor:** > 10 Hz (fails target)

### Speaker Similarity
- **Good:** > 0.90 (excellent voice match)
- **Acceptable:** 0.85-0.90 (meets target)
- **Poor:** < 0.85 (fails target)

### Overall Quality Score
- **Excellent:** > 0.85
- **Good:** 0.75-0.85
- **Acceptable:** 0.65-0.75
- **Poor:** < 0.65

### MOS Estimation
- **5:** Excellent (imperceptible artifacts)
- **4:** Good (perceptible but not annoying)
- **3:** Fair (slightly annoying)
- **2:** Poor (annoying)
- **1:** Bad (very annoying)

### STOI Score
- **Good:** > 0.85 (high intelligibility)
- **Acceptable:** 0.70-0.85
- **Poor:** < 0.70 (low intelligibility)

## Troubleshooting

### Tests Skipped

**"Voice cloning service unavailable"**
- Ensure Flask server is running on `http://localhost:5001`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify model weights are loaded

**"Quality metrics not available"**
- Install missing dependencies:
  ```bash
  pip install pystoi pesq nisqa
  ```

**"Profile not found"**
- Voice profile creation may have failed
- Check server logs for errors

### Low Quality Scores

**High Pitch RMSE (> 10 Hz)**
- Check pitch extractor configuration
- Verify source audio has clear pitch contours
- Ensure proper audio alignment

**Low Speaker Similarity (< 0.85)**
- Reference audio may be too short (need ≥30s)
- Reference audio quality issues
- Speaker encoder model may need retraining

## Documentation

- **Full Documentation:** `/home/kp/autovoice/docs/API_E2E_QUALITY_VALIDATION.md`
- **Test Implementation:** `/home/kp/autovoice/tests/test_api_e2e_validation.py`
- **Quality Metrics:** `/home/kp/autovoice/src/auto_voice/utils/quality_metrics.py`

## CI/CD Integration

### GitHub Actions Example

```yaml
name: API E2E Quality Validation

on: [push, pull_request]

jobs:
  quality-validation:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pystoi pesq

    - name: Run API E2E quality tests
      run: |
        pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation -v

    - name: Upload validation results
      uses: actions/upload-artifact@v3
      with:
        name: validation-results
        path: validation_results/*.json
```

## Contact

For questions or issues:
- Check documentation in `/home/kp/autovoice/docs/`
- Review test implementation in `/home/kp/autovoice/tests/`
- Consult quality metrics source in `/home/kp/autovoice/src/auto_voice/utils/quality_metrics.py`
