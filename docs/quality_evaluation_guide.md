# Voice Conversion Quality Evaluation Guide

This comprehensive guide explains how to use AutoVoice's quality evaluation system for assessing the performance of singing voice conversion models.

## Overview

The quality evaluation system provides objective metrics to measure the quality of voice conversion outputs across multiple dimensions:

- **Pitch Accuracy**: How well the pitch contour is preserved
- **Speaker Similarity**: How similar the voice characteristics are
- **Naturalness**: Audio quality and naturalness of the output
- **Intelligibility**: Speech clarity and comprehension

## Quick Start

### Basic Evaluation

```python
from auto_voice.evaluation import VoiceConversionEvaluator
from auto_voice.utils import QualityMetricsAggregator

# Create evaluator
evaluator = VoiceConversionEvaluator()

# Evaluate single conversion
results = evaluator.evaluate_single_conversion(source_audio, target_audio)
print(f"Overall quality score: {results.overall_quality_score:.3f}")
print(f"Pitch correlation: {results.pitch_accuracy.correlation:.3f}")
```

### Batch Evaluation

```python
# Create evaluation samples
samples = evaluator.create_test_samples_from_directory(
    source_dir="/path/to/source/audio",
    target_dir="/path/to/converted/audio"
)

# Evaluate all samples
results = evaluator.evaluate_conversions(samples)

# Generate reports
evaluator.generate_reports(results, "./evaluation_results")
```

### Command Line Tool

```bash
# Evaluate voice conversions
python examples/evaluate_voice_conversion.py \
    --source-dir data/test/source \
    --target-dir data/test/converted \
    --output-dir results/evaluation \
    --formats markdown json

# Validate against quality targets
python examples/evaluate_voice_conversion.py \
    --source-dir data/test/source \
    --target-dir data/test/converted \
    --validate-targets \
    --min-pitch-correlation 0.8 \
    --max-pitch-rmse-hz 10.0
```

## Quality Metrics

### Overview

AutoVoice provides comprehensive quality evaluation across four dimensions:

1. **Pitch Accuracy**: Melodic preservation and F0 tracking
2. **Speaker Similarity**: Voice characteristic matching
3. **Naturalness**: Audio quality, perceptual quality (MOS), and spectral distortion (MCD)
4. **Intelligibility**: Speech clarity (STOI, ESTOI, PESQ)

### Pitch Accuracy Metrics

- **RMSE (Hz)**: Root mean square error in Hertz domain
- **RMSE (log2)**: Root mean square error in semitones (provided for reference)
- **Correlation**: Pearson correlation between source and target pitch contours
- **Voiced Accuracy**: Percentage of frames with quarter-tone accuracy
- **Octave Errors**: Count of coarse pitch errors
- **Confidence Score**: Overall pitch accuracy confidence

**Interpretation:**
- RMSE (Hz) < 10 Hz: Excellent pitch preservation (primary metric for quality gating)
- RMSE (log2) < 0.1 semitones: Excellent melodic accuracy (provided for reference)
- Correlation > 0.8: Good pitch tracking
- Voiced Accuracy > 0.8: High melodic accuracy

### Speaker Similarity Metrics

- **Cosine Similarity**: Similarity between speaker embeddings
- **Embedding Distance**: Euclidean distance in embedding space
- **Confidence Score**: Overall speaker preservation confidence

**Interpretation:**
- Cosine Similarity > 0.8: High speaker similarity
- Embedding Distance < 0.5: Very similar speakers

### Naturalness Metrics

- **Spectral Distortion**: Difference in spectral characteristics (dB)
- **MOS Estimation**: Mean opinion score prediction (1-5 scale)
  - Can use heuristic estimation or NISQA model for more accurate prediction
  - Supports both methods for comparison
- **MCD (Mel-Cepstral Distortion)**: Spectral distance using MFCCs (dB)
- **Confidence Score**: Audio quality confidence

**Interpretation:**
- Spectral Distortion < 5 dB: Very natural sounding
- MOS > 4.0: Excellent perceived quality
- MCD < 6.0 dB: Low spectral distortion (high quality)
- MCD 6-8 dB: Moderate spectral distortion
- MCD > 8 dB: High spectral distortion (quality degradation)

### Intelligibility Metrics

- **STOI**: Short-Time Objective Intelligibility measure (0-1)
- **ESTOI**: Extended STOI with better correlation to human perception
- **PESQ**: Perceptual Evaluation of Speech Quality (-0.5 to 4.5)

**Interpretation:**
- STOI > 0.9: Nearly perfect intelligibility
- PESQ > 4.0: Excellent voice quality

## Configuration

The evaluation system supports extensive configuration through YAML files:

```yaml
# config/evaluation_config.yaml
audio:
  sample_rate: 44100
  normalize_audio: true
  target_rms_db: -12.0

alignment:
  align_audio: true
  max_delay_sec: 0.2

quality_targets:
  min_pitch_accuracy_correlation: 0.8
  max_pitch_accuracy_rmse_hz: 10.0
  min_speaker_similarity: 0.75
  max_spectral_distortion: 10.0
  min_stoi_score: 0.7
  min_mos_estimation: 4.0
  max_mcd: 6.0
  min_overall_quality_score: 0.75

# MOS and MCD configuration
naturalness:
  mos_method: 'heuristic'  # Options: 'heuristic', 'nisqa', 'both'
  compute_mcd: true  # Enable Mel-Cepstral Distortion computation
```

## Advanced Usage

### Custom Evaluation Classes

```python
from auto_voice.evaluation.evaluator import VoiceConversionEvaluator
from auto_voice.utils.quality_metrics import QualityMetricsAggregator, QualityTargets

# Create custom evaluator with specific configuration
evaluator = VoiceConversionEvaluator(
    sample_rate=48000,
    evaluation_config_path="path/to/custom/config.yaml"
)

# Add progress callbacks
def progress_callback(current, total, message):
    print(f"[{current}/{total}] {message}")

evaluator.add_progress_callback(progress_callback)

# Create quality aggregator with MCD and NISQA MOS
quality_aggregator = QualityMetricsAggregator(
    sample_rate=44100,
    mos_method='both',  # Use both heuristic and NISQA
    compute_mcd=True     # Enable MCD computation
)

# Evaluate with comprehensive metrics
metrics = quality_aggregator.evaluate(source_audio, converted_audio)

print(f"MOS (heuristic): {metrics.naturalness.mos_heuristic:.2f}")
print(f"MOS (NISQA): {metrics.naturalness.mos_nisqa:.2f}")
print(f"MCD: {metrics.naturalness.mcd:.2f} dB")
print(f"STOI: {metrics.intelligibility.stoi_score:.3f}")
```

### Audio Preprocessing

The system handles audio alignment automatically:

```python
# Manual audio preprocessing
from auto_voice.utils.quality_metrics import AudioAligner, AudioNormalizer

aligner = AudioAligner(sample_rate=44100)
normalizer = AudioNormalizer()

# Align and normalize audio
aligned_result = aligner.align_audio(source_audio, target_audio)
source_normalized = normalizer.normalize_audio(aligned_result.source_audio)
target_normalized = normalizer.normalize_audio(aligned_result.aligned_target)
```

### Visualization

```python
from auto_voice.utils.visualization import PitchContourVisualizer, QualityDashboardGenerator

# Pitch contour visualization
pitch_viz = PitchContourVisualizer()
pitch_data = PitchContourData(
    time=time_axis,
    f0_source=source_pitch,
    f0_target=target_pitch,
    sample_id="sample_001"
)
fig = pitch_viz.plot_pitch_contour(pitch_data)
plt.show()

# Quality dashboard
dashboard = QualityDashboardGenerator()
dashboard.create_summary_dashboard(summary_stats, sample_results)
```

## Output Formats

### Markdown Reports

Comprehensive human-readable reports with:
- Summary statistics
- Individual sample results
- Quality metric breakdowns
- Evaluation metadata

### JSON Reports

Machine-readable detailed results:
```json
{
  "evaluation_timestamp": 1234567890.123,
  "total_evaluation_time": 45.67,
  "summary_stats": {...},
  "samples": [
    {
      "id": "sample_001",
      "results": {
        "pitch_accuracy": {...},
        "speaker_similarity": {...},
        "naturalness": {...},
        "intelligibility": {...}
      }
    }
  ]
}
```

### HTML Dashboards

Interactive web-based reports with:
- Quality score distributions
- Pitch analysis plots
- Speaker similarity heatmaps
- Spectrogram comparisons

## Automated Quality Validation

### Quality Targets

Set minimum performance requirements:

```python
targets = QualityTargets(
    min_pitch_accuracy_correlation=0.8,
    max_pitch_accuracy_rmse_hz=10.0,
    min_speaker_similarity=0.75,
    min_overall_quality_score=0.8
)

validation_results = evaluator.validate_quality_targets(results, targets)

if validation_results['overall_pass']:
    print("✓ All quality targets met")
else:
    print(f"✗ Failed targets: {validation_results['failed_targets']}")
```

### CI/CD Integration

Automate quality regression detection in CI pipelines:

```bash
# In GitHub Actions or similar CI
python examples/evaluate_voice_conversion.py \
    --source-dir test_data/source \
    --target-dir test_output \
    --validate-targets \
    --config .github/workflows/evaluation_config.yaml

# Script exits with code 1 if quality targets not met
```

## Best Practices

### Audio Preparation

1. **Sample Rate**: Ensure all audio is at the same sample rate (default: 44.1kHz)
2. **Normalization**: RMS normalization to -12dBFS recommended
3. **Mono/Stereo**: Convert stereo to mono if needed
4. **Silence Trimming**: Trim leading/trailing silence for better alignment

### Evaluation Workflow

1. **Paired Data**: Ensure source and target audio are properly paired
2. **Alignment**: Enable audio alignment for temporal differences
3. **Batch Processing**: Use batch evaluation for large datasets
4. **Progress Tracking**: Monitor long-running evaluations

### Interpretation Guidelines

- **Perfect Conversion**: All metrics > 0.95, overall score > 0.9
- **Good Conversion**: Most metrics > 0.8, overall score > 0.75
- **Acceptable Conversion**: Mixed results, overall score > 0.6
- **Poor Conversion**: Multiple metrics < 0.5, overall score < 0.4

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install pystoi pesq seaborn matplotlib
   ```

2. **Audio Format Issues**: Convert audio to WAV/PCM format for best compatibility

3. **Memory Issues**: For large datasets, reduce batch size or process sequentially

4. **GPU Acceleration**: Enable GPU acceleration for better performance with CUDA-compatible hardware

### Performance Optimization

- Use GPU acceleration for CUDA-compatible systems
- Process audio in batches for large evaluations
- Disable resource-intensive visualizations for quick checks
- Cache results for repeated evaluations

## API Reference

### VoiceConversionEvaluator

Main evaluation interface:

```python
class VoiceConversionEvaluator:
    def __init__(self, sample_rate=44100, device='auto', evaluation_config_path=None)
    def evaluate_single_conversion(self, source_audio, target_audio) -> QualityMetricsResult
    def evaluate_conversions(self, samples, max_workers=None) -> EvaluationResults
    def validate_quality_targets(self, results, targets=None) -> Dict[str, Any]
    def generate_reports(self, results, output_dir, formats=None) -> Dict[str, str]
    def create_test_samples_from_directory(self, source_dir, target_dir) -> List[EvaluationSample]
```

### Quality Metrics Classes

Individual metric evaluators:

- `PitchAccuracyMetrics`: Pitch contour analysis
- `SpeakerSimilarityMetrics`: Speaker embedding comparison
- `NaturalnessMetrics`: Spectral quality analysis
- `IntelligibilityMetrics`: Speech clarity evaluation
- `QualityMetricsAggregator`: Coordinate all metrics

### Visualization Classes

Result visualization:

- `PitchContourVisualizer`: Pitch curve plotting
- `SpectrogramVisualizer`: Spectral comparison
- `QualityDashboardGenerator`: Comprehensive dashboards

## Examples

### Complete Evaluation Pipeline

```python
import torch
from pathlib import Path
from auto_voice.evaluation import VoiceConversionEvaluator
from auto_voice.utils.logging_config import setup_logging

# Setup logging
setup_logging()

# Initialize evaluator
evaluator = VoiceConversionEvaluator(
    sample_rate=44100,
    evaluation_config_path="config/evaluation_config.yaml"
)

# Load evaluation data
source_dir = Path("data/evaluation/source")
target_dir = Path("data/evaluation/converted")

samples = evaluator.create_test_samples_from_directory(source_dir, target_dir)

# Run comprehensive evaluation
print(f"Evaluating {len(samples)} audio pairs...")
results = evaluator.evaluate_conversions(samples)

# Validate against quality standards
validation = evaluator.validate_quality_targets(results)
print(f"Quality validation: {'PASSED' if validation['overall_pass'] else 'FAILED'}")

# Generate detailed reports
output_dir = Path("results/evaluation_run")
reports = evaluator.generate_reports(results, output_dir, ['markdown', 'json', 'html'])

print(f"Reports generated in: {output_dir}")
for fmt, path in reports.items():
    print(f"  {fmt.upper()}: {path}")

# Access detailed results
print("
Summary Statistics:")
for category, stats in results.summary_stats.items():
    if 'mean' in stats:
        print(f"  {category}: {stats['mean']:.3f} ± {stats['std']:.3f}")
```

This evaluation pipeline provides a complete quality assessment of singing voice conversion systems with comprehensive metrics, visualizations, and automated validation.
