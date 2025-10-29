# API E2E Quality Validation Tests

**Implementation:** `/home/kp/autovoice/tests/test_api_e2e_validation.py`
**Requirements:** Comment 10 - API end-to-end quality validation

## Overview

This document describes the comprehensive API end-to-end validation tests that verify the complete voice conversion workflow through the REST API with quality metric validation.

## Test Structure

### Test Class: `TestAPIE2EQualityValidation`

Location: `/home/kp/autovoice/tests/test_api_e2e_validation.py` (lines 587-940)

This test class implements Comment 10 requirements for API E2E quality validation with the following workflow:

1. **Create Voice Profile** - POST reference audio to `/api/v1/voice/clone`
2. **Convert Song** - POST song to `/api/v1/convert/song` with `target_profile_id`
3. **Validate Response** - Assert HTTP 200 and response structure
4. **Download Audio** - Decode base64-encoded converted audio
5. **Compute Quality Metrics** - Use `QualityMetricsAggregator` locally
6. **Assert Quality Targets** - Validate against thresholds

## Quality Targets (Comment 10)

| Metric | Target | Requirement |
|--------|--------|-------------|
| **Pitch RMSE (Hz)** | < 10 Hz | Pitch accuracy in Hz domain |
| **Speaker Similarity** | > 0.85 | Cosine similarity to target embedding |
| Overall Quality Score | > 0.0 | Informational (not enforced) |

## Test Methods

### 1. `test_api_e2e_conversion_quality_validation`

**Purpose:** Test single conversion with quality validation

**Workflow:**
```
1. Create voice profile from 30s reference audio
2. Convert 3s song using /api/v1/convert/song
3. Download and decode converted audio (base64 → WAV)
4. Load source audio for comparison
5. Compute quality metrics using QualityMetricsAggregator
   - Pitch RMSE (Hz and log2)
   - Pitch correlation
   - Speaker similarity (cosine similarity)
   - Spectral distortion
   - MOS estimation
   - STOI and PESQ scores
   - Overall quality score
6. Assert quality targets:
   - Pitch RMSE < 10 Hz
   - Speaker similarity > 0.85
7. Save metrics to validation_results/api_e2e_quality_metrics.json
8. Cleanup: Delete voice profile
```

**Key Features:**
- Uses Flask test server with real components (not mocked)
- Retrieves target speaker embedding from profile for accurate similarity evaluation
- Saves comprehensive metrics to JSON for analysis
- Validates both primary targets and additional quality metrics

**Output:**
- `validation_results/api_e2e_quality_metrics.json` - Detailed metrics
- `validation_results/api_e2e_quality_validation_results.json` - Pass/fail summary

### 2. `test_api_e2e_batch_quality_validation`

**Purpose:** Test batch conversion consistency and quality

**Workflow:**
```
1. Create voice profile
2. Convert same song 3 times
3. Evaluate quality for each conversion
4. Compute aggregate statistics:
   - Average metrics
   - Standard deviation (consistency)
   - Pass/fail for all conversions
5. Save individual and summary results
6. Assert:
   - Average pitch RMSE < 10 Hz
   - Average speaker similarity > 0.85
   - All conversions meet targets
```

**Key Features:**
- Tests conversion consistency across multiple runs
- Detects quality regressions through variance analysis
- Validates batch processing stability

**Output:**
- `validation_results/api_e2e_batch_conversion_1.json` - Conversion 1 metrics
- `validation_results/api_e2e_batch_conversion_2.json` - Conversion 2 metrics
- `validation_results/api_e2e_batch_conversion_3.json` - Conversion 3 metrics
- `validation_results/api_e2e_batch_summary.json` - Aggregate statistics

## Quality Metrics Evaluated

### Pitch Accuracy
- **RMSE (Hz):** Root-mean-square error in Hz domain (primary target < 10 Hz)
- **RMSE (log2):** Root-mean-square error in log2 domain (semitones)
- **Correlation:** Pearson correlation between source and converted F0 contours
- **Voiced Accuracy:** Percentage of frames within quarter-tone deviation
- **Octave Errors:** Count of coarse pitch errors (≥1 octave)

### Speaker Similarity
- **Cosine Similarity:** Cosine similarity between speaker embeddings (target > 0.85)
- **Embedding Distance:** Euclidean distance between embeddings
- **Confidence Score:** Overall similarity confidence

### Naturalness
- **Spectral Distortion:** Log-magnitude spectrogram difference (dB)
- **Harmonic-to-Noise Ratio:** Voice quality metric
- **MOS Estimation:** Mean Opinion Score estimate (1-5 scale)

### Intelligibility
- **STOI Score:** Short-Time Objective Intelligibility (0-1)
- **ESTOI Score:** Extended STOI (0-1)
- **PESQ Score:** Perceptual Evaluation of Speech Quality

### Overall Quality
- **Weighted Combination:**
  - Pitch Accuracy: 30%
  - Speaker Similarity: 30%
  - Naturalness: 25%
  - Intelligibility: 15%

## Implementation Details

### Audio Decoding
```python
def decode_audio_from_base64(self, audio_base64: str) -> tuple:
    """Decode base64-encoded WAV audio to numpy array."""
    audio_bytes = base64.b64decode(audio_base64)
    buffer = io.BytesIO(audio_bytes)

    with wave.open(buffer, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

        # Convert to float32 numpy array
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0

    return audio_float, sample_rate
```

### Quality Evaluation
```python
def evaluate_conversion_quality(self, source_audio, converted_audio,
                               sample_rate, target_profile_id=None):
    """Evaluate conversion quality using QualityMetricsAggregator."""
    from auto_voice.utils.quality_metrics import QualityMetricsAggregator
    from auto_voice.inference.voice_cloner import VoiceCloner

    # Convert to torch tensors
    source_tensor = torch.from_numpy(source_audio).float()
    converted_tensor = torch.from_numpy(converted_audio).float()

    # Get target speaker embedding
    target_embedding = None
    if target_profile_id:
        voice_cloner = VoiceCloner()
        target_embedding = voice_cloner.get_embedding(target_profile_id)

    # Initialize and evaluate
    aggregator = QualityMetricsAggregator(sample_rate=sample_rate)
    result = aggregator.evaluate(
        source_tensor,
        converted_tensor,
        align_audio=True,
        target_speaker_embedding=target_embedding
    )

    return metrics_dict
```

### Validation Results Format

**Individual Conversion Metrics:**
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

**Quality Validation Summary:**
```json
{
  "conversion": {
    "status_code": 200,
    "profile_id": "uuid-profile-id",
    "duration": 3.0
  },
  "metrics": { ... },
  "quality_targets": {
    "pitch_rmse_hz_target": 10.0,
    "pitch_rmse_hz_actual": 8.45,
    "pitch_rmse_hz_pass": true,
    "speaker_similarity_target": 0.85,
    "speaker_similarity_actual": 0.88,
    "speaker_similarity_pass": true
  },
  "passed": true
}
```

**Batch Summary:**
```json
{
  "num_conversions": 3,
  "avg_pitch_rmse_hz": 8.32,
  "avg_speaker_similarity": 0.87,
  "avg_overall_quality": 0.78,
  "pitch_rmse_std": 0.42,
  "speaker_similarity_std": 0.02,
  "all_conversions_meet_targets": true
}
```

## Running the Tests

### Prerequisites
- Flask server running on `http://localhost:5001`
- All model dependencies installed
- GPU available (recommended for quality)

### Execute All E2E Tests
```bash
pytest tests/test_api_e2e_validation.py -v -s
```

### Execute Only Quality Validation Tests
```bash
pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation -v -s
```

### Execute Specific Test
```bash
# Single conversion quality test
pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation::test_api_e2e_conversion_quality_validation -v -s

# Batch quality test
pytest tests/test_api_e2e_validation.py::TestAPIE2EQualityValidation::test_api_e2e_batch_quality_validation -v -s
```

### Test Markers
- `@pytest.mark.api` - API integration tests
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.integration` - Integration tests (requires real components)
- `@pytest.mark.slow` - Tests that take significant time

### Skip Tests on Service Unavailability
Tests automatically skip if:
- Flask server fails to start (30s timeout)
- Voice cloning service unavailable (503)
- Conversion service unavailable (503)
- Profile not found (404)
- Quality metrics dependencies missing

## Validation Results Directory

All validation results are saved to:
```
/home/kp/autovoice/validation_results/
├── api_e2e_quality_metrics.json              # Single conversion detailed metrics
├── api_e2e_quality_validation_results.json   # Single conversion summary
├── api_e2e_batch_conversion_1.json          # Batch conversion 1 metrics
├── api_e2e_batch_conversion_2.json          # Batch conversion 2 metrics
├── api_e2e_batch_conversion_3.json          # Batch conversion 3 metrics
└── api_e2e_batch_summary.json               # Batch aggregate statistics
```

## Integration with Existing Tests

The API E2E quality validation tests extend the existing test suite:

### Existing Test Classes
1. **TestAPIHealthEndpoints** - Health check endpoints
2. **TestVoiceCloningWorkflow** - Voice profile CRUD operations
3. **TestConversionAPIWorkflow** - Conversion workflow (no quality validation)
4. **TestQualityMetricsValidation** - Audio analysis endpoints
5. **TestErrorHandlingAndRecovery** - Error handling scenarios
6. **TestConcurrentRequests** - Concurrent request handling

### New Test Class
7. **TestAPIE2EQualityValidation** - **Comment 10** quality validation
   - Adds quality metric computation using `QualityMetricsAggregator`
   - Asserts quality targets (pitch RMSE < 10 Hz, similarity > 0.85)
   - Saves detailed metrics to `validation_results/`

## Dependencies

### Required Modules
```python
import pytest
import requests
import base64
import io
import wave
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# AutoVoice modules
from auto_voice.utils.quality_metrics import QualityMetricsAggregator
from auto_voice.inference.voice_cloner import VoiceCloner
from auto_voice.audio.pitch_extractor import SingingPitchExtractor
from auto_voice.models.speaker_encoder import SpeakerEncoder
```

### Optional Dependencies
- `torchaudio` - Audio I/O operations
- `pystoi` - STOI intelligibility metrics
- `pesq` - PESQ quality metrics
- `nisqa` - NISQA MOS prediction

## Error Handling

### Automatic Test Skipping
```python
if clone_response.status_code == 503:
    pytest.skip("Voice cloning service unavailable")

if convert_response.status_code == 404:
    pytest.skip(f"Profile not found: {profile_id}")

try:
    from auto_voice.utils.quality_metrics import QualityMetricsAggregator
except ImportError as e:
    pytest.skip(f"Quality metrics not available: {e}")
```

### Cleanup on Failure
```python
try:
    # Test operations...
finally:
    # Always cleanup
    requests.delete(f'{API_BASE_URL}/api/v1/voice/profiles/{profile_id}')
```

## Future Enhancements

### Potential Improvements
1. **Real Audio Files** - Test with real singing audio from `test_data/`
2. **Asynchronous Conversion** - Test async workflow with status polling
3. **Stems Validation** - Validate separated vocals and instrumental quality
4. **Edge Cases** - Very short audio, silence, extreme pitch ranges
5. **Performance Benchmarks** - Conversion time, memory usage
6. **Quality Regression Detection** - Compare against baseline metrics
7. **Multi-Profile Testing** - Test with different voice profiles
8. **Parallel Conversions** - Test concurrent conversion requests

## References

- **Comment 10:** Original requirement for API E2E quality validation
- **Quality Metrics:** `/home/kp/autovoice/src/auto_voice/utils/quality_metrics.py`
- **Evaluator:** `/home/kp/autovoice/src/auto_voice/evaluation/evaluator.py`
- **API Routes:** `/home/kp/autovoice/src/auto_voice/web/api.py`
- **Web App:** `/home/kp/autovoice/src/auto_voice/web/app.py`

## Summary

The API E2E quality validation tests provide comprehensive verification of the complete voice conversion workflow through the REST API, ensuring:

✅ **Functional Correctness** - API endpoints work as specified
✅ **Quality Validation** - Conversions meet pitch and speaker similarity targets
✅ **Consistency** - Multiple conversions produce consistent quality
✅ **Comprehensive Metrics** - All quality dimensions evaluated
✅ **Automated Validation** - Tests can run in CI/CD pipelines
✅ **Detailed Reporting** - Metrics saved to JSON for analysis

**Comment 10 Requirements: ✅ Complete**
