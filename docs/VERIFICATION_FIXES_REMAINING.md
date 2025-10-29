# Remaining Verification Comment Fixes

## Summary of Completed Fixes (Comments 1-10)

✅ **Comment 1**: Fixed pitch RMSE to use Hz domain
- Added `rmse_hz` field to `PitchAccuracyResult`
- Updated `evaluate_pitch_accuracy()` to compute Hz RMSE
- Updated config and validation thresholds

✅ **Comment 2**: Fixed `SingingPitchExtractor` method calls
- Changed from `extract_pitch()` to `extract_f0_contour()`
- Properly extract arrays using dictionary keys

✅ **Comment 3**: Fixed speaker similarity computation
- Updated API to accept `target_speaker_embedding`
- Now compares converted audio against target profile

✅ **Comment 4-10**: Various metric and config fixes
- Fixed STOI resampling
- Fixed spectral distortion calculation
- Implemented normalized overall quality score
- Updated config auto-loading
- Updated Hz RMSE thresholds

## Remaining Implementation: Comment 4 - Test Metadata Support

### Required Changes

#### 1. Add `evaluate_test_set()` Method to `VoiceConversionEvaluator`

```python
# In src/auto_voice/evaluation/evaluator.py

def evaluate_test_set(
    self,
    metadata_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None
) -> EvaluationResults:
    """
    Evaluate test set by running conversions from metadata.

    Args:
        metadata_path: Path to test metadata JSON file
        output_dir: Optional directory for converted audio
        pipeline_config: Optional pipeline configuration

    Returns:
        EvaluationResults with conversion and metrics
    """
    # Load test metadata
    with open(metadata_path, 'r') as f:
        test_data = json.load(f)

    # Import pipeline
    from ..inference.singing_conversion_pipeline import SingingConversionPipeline

    # Initialize pipeline
    pipeline = SingingConversionPipeline(**pipeline_config or {})

    samples = []
    for idx, test_item in enumerate(test_data):
        try:
            source_audio_path = test_item['source_audio']
            target_profile_id = test_item['target_profile_id']
            reference_audio = test_item.get('reference_audio')

            # Load source audio
            source_audio = self._load_audio(source_audio_path)

            # Run conversion
            converted_audio = pipeline.convert_song(
                audio=source_audio,
                target_speaker_id=target_profile_id,
                sample_rate=self.sample_rate
            )

            # Get target speaker embedding
            target_embedding = pipeline.get_speaker_embedding(target_profile_id)

            # Create evaluation sample
            sample = EvaluationSample(
                id=f"test_{idx}_{target_profile_id}",
                source_audio_path=source_audio_path,
                source_audio=source_audio,
                target_audio=converted_audio,
                metadata={
                    'target_profile_id': target_profile_id,
                    'reference_audio': reference_audio
                }
            )

            # Evaluate with target embedding
            result = self.metrics_aggregator.evaluate(
                source_audio, converted_audio,
                align_audio=self.config['align_audio'],
                target_speaker_embedding=target_embedding
            )
            sample.result = result
            samples.append(sample)

        except Exception as e:
            logger.error(f"Failed to process test item {idx}: {e}")
            continue

    # Compute summary statistics
    summary_stats = self._compute_batch_summary_statistics(samples)

    return EvaluationResults(
        samples=samples,
        summary_stats=summary_stats,
        evaluation_config=self.config,
        evaluation_timestamp=time.time(),
        total_evaluation_time=time.time() - start_time
    )
```

#### 2. Update `examples/evaluate_voice_conversion.py`

```python
# Add new argument
parser.add_argument(
    '--test-metadata',
    type=str,
    default=None,
    help='Path to test metadata JSON file for conversion evaluation'
)

# In main() function, before creating samples
if args.test_metadata:
    logger.info(f"Running evaluation from test metadata: {args.test_metadata}")
    results = evaluator.evaluate_test_set(
        metadata_path=args.test_metadata,
        output_dir=output_dir
    )
else:
    # Existing directory-based evaluation
    samples = evaluator.create_test_samples_from_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir
    )
    results = evaluator.evaluate_conversions(samples)
```

#### 3. Test Metadata Format

```json
[
    {
        "source_audio": "data/test/source/song1.wav",
        "target_profile_id": "speaker_001",
        "reference_audio": "data/test/reference/speaker_001_sample.wav"
    },
    {
        "source_audio": "data/test/source/song2.wav",
        "target_profile_id": "speaker_002"
    }
]
```

## Remaining: Comment 5 - CI and Synthetic Data

### Files to Create

#### `.github/workflows/quality_checks.yml`

```yaml
name: Quality Checks

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-validation:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Generate synthetic test data
        run: |
          python scripts/generate_test_data.py --output data/test_synthetic

      - name: Run quality validation tests
        run: |
          pytest tests/ -m quality -v --cov=src/auto_voice

      - name: Run evaluation with validation
        run: |
          python examples/evaluate_voice_conversion.py \
            --test-metadata data/test_synthetic/metadata.json \
            --output-dir results/ci_validation \
            --validate-targets

      - name: Upload test artifacts
        uses: actions/upload-artifact@v3
        with:
          name: quality-reports
          path: results/ci_validation/
```

#### `scripts/generate_test_data.py`

```python
#!/usr/bin/env python3
"""Generate synthetic test data for quality validation"""

import argparse
import json
import numpy as np
import torch
import librosa
from pathlib import Path

def generate_synthetic_audio(duration_sec=2.0, sample_rate=22050, f0_hz=220.0):
    """Generate synthetic singing audio with known F0"""
    # Generate time array
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))

    # Generate fundamental frequency with vibrato
    vibrato_rate = 5.0  # Hz
    vibrato_depth = 0.02  # 2% depth
    f0_contour = f0_hz * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))

    # Generate harmonics
    audio = np.zeros_like(t)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic
        audio += amplitude * np.sin(2 * np.pi * harmonic * f0_contour * t)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output)
    (output_dir / 'source').mkdir(parents=True, exist_ok=True)
    (output_dir / 'converted').mkdir(parents=True, exist_ok=True)

    metadata = []

    for i in range(args.num_samples):
        # Generate source audio
        f0_source = 220.0 + i * 10  # Vary pitch
        source_audio = generate_synthetic_audio(f0_hz=f0_source)
        source_path = output_dir / 'source' / f'test_{i:03d}.wav'
        librosa.output.write_wav(str(source_path), source_audio, 22050)

        # Generate "converted" audio (slightly pitch-shifted for testing)
        f0_target = f0_source + 5.0  # 5 Hz shift
        converted_audio = generate_synthetic_audio(f0_hz=f0_target)
        converted_path = output_dir / 'converted' / f'test_{i:03d}.wav'
        librosa.output.write_wav(str(converted_path), converted_audio, 22050)

        metadata.append({
            'id': f'test_{i:03d}',
            'source_audio': str(source_path),
            'converted_audio': str(converted_path),
            'expected_rmse_hz': 5.0,
            'target_profile_id': 'synthetic_speaker'
        })

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {args.num_samples} synthetic test samples in {output_dir}")

if __name__ == '__main__':
    main()
```

## Remaining: Comment 6 - Quality Validation Tests

### Files to Create/Update

#### `tests/test_quality_metrics.py`

```python
import pytest
import torch
import numpy as np
from src.auto_voice.utils.quality_metrics import (
    PitchAccuracyMetrics,
    SpeakerSimilarityMetrics,
    NaturalnessMetrics,
    IntelligibilityMetrics,
    QualityMetricsAggregator
)

@pytest.mark.quality
def test_pitch_accuracy_hz_rmse():
    """Test that pitch RMSE is computed in Hz domain"""
    metrics = PitchAccuracyMetrics(sample_rate=22050)

    # Generate test audio with known pitch
    duration = 2.0
    sample_rate = 22050
    t = torch.linspace(0, duration, int(duration * sample_rate))

    # Source: 220 Hz (A3)
    source_audio = torch.sin(2 * np.pi * 220 * t)
    # Target: 225 Hz (5 Hz difference)
    target_audio = torch.sin(2 * np.pi * 225 * t)

    result = metrics.evaluate_pitch_accuracy(source_audio, target_audio)

    # Hz RMSE should be close to 5.0 Hz
    assert result.rmse_hz < 10.0, f"Hz RMSE {result.rmse_hz} exceeds 10 Hz threshold"
    assert 4.0 < result.rmse_hz < 6.0, f"Expected ~5 Hz, got {result.rmse_hz}"

@pytest.mark.quality
def test_speaker_similarity_with_embedding():
    """Test speaker similarity uses target embedding"""
    metrics = SpeakerSimilarityMetrics()

    # Generate dummy audio and embedding
    audio = torch.randn(1, 22050)
    target_embedding = np.random.randn(256).astype(np.float32)

    result = metrics.evaluate_speaker_similarity(
        audio, target_speaker_embedding=target_embedding
    )

    assert result.cosine_similarity > -1.0
    assert result.cosine_similarity < 1.0
    assert result.target_embedding is not None

@pytest.mark.quality
def test_overall_quality_score_normalized():
    """Test overall quality score uses normalized metrics"""
    aggregator = QualityMetricsAggregator(sample_rate=22050)

    source = torch.randn(1, 22050)
    target = torch.randn(1, 22050)

    result = aggregator.evaluate(source, target, align_audio=False)

    # Overall score should be between 0 and 1
    assert 0.0 <= result.overall_quality_score <= 1.0
```

#### Update `tests/test_end_to_end.py`

```python
@pytest.mark.quality
def test_conversion_quality_validation():
    """Test full conversion pipeline with quality validation"""
    from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
    from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

    # Initialize pipeline
    pipeline = SingingConversionPipeline()
    metrics = QualityMetricsAggregator(sample_rate=22050)

    # Load test audio
    source_audio = torch.randn(1, 44100)

    # Run conversion
    converted_audio = pipeline.convert_song(source_audio, target_speaker_id='test_speaker')

    # Evaluate quality
    result = metrics.evaluate(source_audio, converted_audio)

    # Assert quality targets
    assert result.pitch_accuracy.rmse_hz < 10.0, "Pitch accuracy below target"
    assert result.speaker_similarity.cosine_similarity > 0.85, "Speaker similarity below target"
    assert result.overall_quality_score > 0.75, "Overall quality below target"
```

## Remaining: Comment 11 - Visualization Generation

This requires implementing visualization utilities which are referenced but not yet created. Implementation is documented in the quality evaluation guide.

## Next Steps

1. Implement test metadata evaluation support (Comment 4)
2. Create CI workflow and synthetic data generator (Comment 5)
3. Add quality validation tests (Comment 6)
4. Implement visualization generation (Comment 11)
