# Mel-Cepstral Distortion (MCD) Computation Guide

## Overview

Mel-Cepstral Distortion (MCD) is an objective metric for measuring spectral distance between two audio signals. It is widely used in voice conversion research to quantify how similar the converted speech is to the target speech in the spectral domain.

## What is MCD?

MCD measures the distance between Mel-Frequency Cepstral Coefficients (MFCCs) of two audio signals. MFCCs are a compact representation of the spectral envelope of audio, making them ideal for comparing voice quality.

### Formula

```
MCD = (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))
```

Where:
- `c1`, `c2` are MFCC vectors from source and target audio
- The constant `10 / ln(10)` converts to dB scale
- The factor `sqrt(2)` accounts for squared differences

## Implementation in AutoVoice

### Basic Usage

```python
from auto_voice.utils.quality_metrics import NaturalnessMetrics

# Create naturalness metrics evaluator with MCD enabled
naturalness = NaturalnessMetrics(
    sample_rate=44100,
    compute_mcd=True
)

# Evaluate naturalness including MCD
result = naturalness.evaluate_naturalness(source_audio, converted_audio)

print(f"MCD: {result.mcd:.2f} dB")
print(f"MOS: {result.mos_estimation:.2f}")
print(f"Spectral Distortion: {result.spectral_distortion:.2f} dB")
```

### Integration with Quality Aggregator

```python
from auto_voice.utils.quality_metrics import QualityMetricsAggregator

# Create aggregator with MCD computation
aggregator = QualityMetricsAggregator(
    sample_rate=44100,
    compute_mcd=True
)

# Comprehensive evaluation
metrics = aggregator.evaluate(source_audio, converted_audio)

# Access MCD from naturalness results
mcd_value = metrics.naturalness.mcd
```

## Interpretation Guidelines

### MCD Value Ranges

| MCD (dB) | Quality Level | Interpretation |
|----------|--------------|----------------|
| < 4.0    | Excellent    | Near-identical spectral characteristics |
| 4.0-6.0  | Good         | High-quality conversion with minor differences |
| 6.0-8.0  | Fair         | Noticeable spectral differences but acceptable |
| 8.0-10.0 | Poor         | Significant spectral distortion |
| > 10.0   | Very Poor    | Severe quality degradation |

### Quality Threshold

**Recommended threshold for production systems: MCD < 6.0 dB**

This threshold ensures:
- High perceptual quality
- Minimal spectral distortion
- Good speaker similarity preservation
- Professional voice conversion quality

## Technical Details

### MFCC Extraction

AutoVoice uses the following MFCC configuration:

```python
n_mfcc = 13          # Standard number of coefficients
n_fft = 2048         # FFT window size
hop_length = 512     # Frame hop length
exclude_c0 = True    # Exclude energy coefficient
```

### Frame Alignment

MCD computation automatically handles:
- Different audio lengths (aligns to minimum frame count)
- Frame synchronization
- Per-frame distance calculation
- Averaging over all frames

### Computation Steps

1. **MFCC Extraction**: Extract 13 MFCC coefficients (excluding C0 energy term)
2. **Frame Alignment**: Align source and target MFCC matrices to same frame count
3. **Distance Calculation**: Compute Euclidean distance for each frame
4. **dB Conversion**: Apply scaling factor and convert to dB scale
5. **Averaging**: Take mean MCD across all frames

## Best Practices

### When to Use MCD

✅ Use MCD for:
- Objective voice quality assessment
- Comparing different conversion models
- Quality regression testing in CI/CD
- Research benchmarking
- Automatic quality gating

❌ Don't rely solely on MCD for:
- Subjective quality assessment (use MOS in addition)
- Intelligibility evaluation (use STOI/PESQ)
- Pitch accuracy (use pitch RMSE/correlation)

### Combining with Other Metrics

MCD works best when combined with complementary metrics:

```python
# Comprehensive quality evaluation
metrics = aggregator.evaluate(source_audio, converted_audio)

# Check multiple dimensions
quality_pass = (
    metrics.naturalness.mcd < 6.0 and           # Spectral quality
    metrics.naturalness.mos_estimation > 4.0 and # Perceived quality
    metrics.intelligibility.stoi_score > 0.9 and # Intelligibility
    metrics.pitch_accuracy.rmse_hz < 10.0        # Pitch accuracy
)
```

### Performance Considerations

MCD computation involves:
- MFCC extraction: O(n log n) due to FFT
- Distance calculation: O(n * m) where n=frames, m=coefficients
- Memory: Requires storing MFCC matrices

For large batch evaluations, consider:
- Processing in parallel across samples
- Caching MFCC computations if needed multiple times
- Using GPU acceleration for MFCC extraction (if available)

## Research Background

MCD is derived from:
- **Kubichek, R. (1993)**: "Mel-cepstral distance measure for objective speech quality assessment"
- Widely adopted in voice conversion literature
- Strong correlation with subjective quality scores (MOS)
- Used in international voice conversion challenges (VCC)

## Validation and Testing

AutoVoice includes comprehensive tests for MCD:

```bash
# Run MCD-specific tests
pytest tests/test_end_to_end.py::TestQualityValidation::test_comprehensive_quality_metrics -v

# Run full quality validation suite
pytest tests/test_end_to_end.py::TestQualityValidation -v --tb=short
```

## Troubleshooting

### Common Issues

1. **MCD returns 0.0**: Check that `compute_mcd=True` is set in evaluator configuration

2. **MCD is None**: Audio may be too short or MFCC extraction failed
   - Ensure audio is at least 0.5 seconds
   - Check sample rate compatibility

3. **Very high MCD values (>15 dB)**:
   - Audio may be severely misaligned
   - Check audio normalization
   - Verify sample rates match

4. **MCD computation fails**:
   - Ensure librosa is installed: `pip install librosa`
   - Check audio format (mono/stereo conversion)
   - Verify audio contains valid samples (no NaN/Inf)

## References

- [Librosa MFCC Documentation](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)
- Voice Conversion Challenge (VCC) evaluation protocols
- AutoVoice Quality Evaluation Guide: `quality_evaluation_guide.md`

## See Also

- **MOS Estimation**: `quality_evaluation_guide.md#naturalness-metrics`
- **STOI/PESQ**: `quality_evaluation_guide.md#intelligibility-metrics`
- **Comprehensive Testing**: `test_end_to_end.py::TestQualityValidation`
