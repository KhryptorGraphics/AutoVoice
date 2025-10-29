# NISQA Support Implementation Summary

## Overview
Successfully added optional NISQA (Non-Intrusive Speech Quality Assessment) model support to the naturalness metrics evaluation system.

## Changes Made

### 1. Core Implementation: `src/auto_voice/utils/quality_metrics.py`

#### Import and Availability Check
- Added NISQA import with try/except block (lines 35-40)
- Set `nisqa_available` flag for graceful fallback
- Warning logged when NISQA is unavailable

#### Enhanced `NaturalnessResult` dataclass
- Added `mos_method: str` field to track which method was used
- Added `mos_nisqa: Optional[float]` for NISQA-predicted MOS score
- Added `mos_heuristic: Optional[float]` for heuristic-based MOS score

#### Updated `NaturalnessMetrics` class
- New `mos_method` parameter in `__init__()`: accepts 'heuristic', 'nisqa', or 'both'
- Automatic NISQA model loading when method is 'nisqa' or 'both'
- Graceful fallback to heuristic if NISQA unavailable or fails to load
- Detailed logging of model loading and method selection

#### Enhanced `evaluate_naturalness()` method
- Calculates heuristic MOS when method is 'heuristic' or 'both'
- Calculates NISQA MOS when model available and method is 'nisqa' or 'both'
- Resamples audio to 48kHz for NISQA (NISQA requirement)
- Automatic fallback to heuristic on NISQA prediction failures
- Prefers NISQA score when method is 'both' and both scores available
- Confidence score boosted when NISQA is used (more reliable)

#### Updated `QualityMetricsAggregator` class
- Added `mos_method` parameter to `__init__()`
- Passes `mos_method` to `NaturalnessMetrics` initialization

### 2. Configuration: `config/evaluation_config.yaml`

#### Added naturalness.mos_method configuration
```yaml
naturalness:
  enabled: true
  spectral_distortion_threshold: 10.0
  n_fft: 2048
  hop_length: 512
  mos_method: 'heuristic'  # NEW: Options: 'heuristic', 'nisqa', 'both'
```

### 3. Evaluator Integration: `src/auto_voice/evaluation/evaluator.py`

#### Updated `VoiceConversionEvaluator.__init__()`
- Moved config loading before metrics aggregator initialization
- Reads `mos_method` from config: `metrics.naturalness.mos_method`
- Passes `mos_method` to `QualityMetricsAggregator`
- Defaults to 'heuristic' if not specified in config

## Usage Examples

### Configuration-Based Usage
```yaml
# config/evaluation_config.yaml
metrics:
  naturalness:
    mos_method: 'nisqa'  # Use NISQA for MOS prediction
```

```python
from auto_voice.evaluation.evaluator import VoiceConversionEvaluator

# Automatically uses mos_method from config
evaluator = VoiceConversionEvaluator(
    sample_rate=44100,
    evaluation_config_path='config/evaluation_config.yaml'
)
```

### Direct API Usage
```python
from auto_voice.utils.quality_metrics import NaturalnessMetrics
import torch

# Create evaluator with NISQA
metrics = NaturalnessMetrics(sample_rate=44100, mos_method='nisqa')

# Evaluate audio
source = torch.randn(1, 44100)
target = torch.randn(1, 44100)
result = metrics.evaluate_naturalness(source, target)

# Access scores
print(f"MOS (NISQA): {result.mos_nisqa}")
print(f"MOS (Heuristic): {result.mos_heuristic}")
print(f"Method used: {result.mos_method}")
```

### Using 'both' method
```python
# Calculate both heuristic and NISQA scores
metrics = NaturalnessMetrics(sample_rate=44100, mos_method='both')
result = metrics.evaluate_naturalness(source, target)

# Result contains both scores
# mos_estimation will be NISQA score (preferred when available)
print(f"Primary MOS: {result.mos_estimation}")
print(f"NISQA MOS: {result.mos_nisqa}")
print(f"Heuristic MOS: {result.mos_heuristic}")
```

## Fallback Behavior

### Graceful Degradation
1. **NISQA not installed**: Falls back to heuristic method with warning
2. **NISQA model fails to load**: Falls back to heuristic method with warning
3. **NISQA prediction fails**: Uses heuristic score with warning
4. **Both method with NISQA unavailable**: Only calculates heuristic

### Logging
- INFO: Model loading success and method selection
- WARNING: Fallback events with reasons
- DEBUG: Individual MOS predictions for monitoring

## Testing

### Verification Tests Passed
✅ NaturalnessMetrics accepts mos_method parameter  
✅ Graceful fallback to heuristic when NISQA unavailable  
✅ NaturalnessResult has mos_method, mos_heuristic, mos_nisqa fields  
✅ Configuration file updated with mos_method setting  
✅ QualityMetricsAggregator passes mos_method through  
✅ VoiceConversionEvaluator reads mos_method from config  

## Installation Notes

### To Enable NISQA
```bash
pip install nisqa
```

### NISQA Requirements
- Python 3.7+
- PyTorch
- librosa (for audio resampling)
- NISQA expects 48kHz audio (automatic resampling handled)

## Benefits

1. **More Accurate MOS Prediction**: NISQA uses deep learning for better quality assessment
2. **Zero Breaking Changes**: Defaults to heuristic method, existing code works unchanged
3. **Flexible Configuration**: Easy to switch methods via config or API
4. **Production Ready**: Graceful fallbacks ensure reliability
5. **Transparency**: Results include method used and individual scores

## Files Modified

- ✅ `/home/kp/autovoice/src/auto_voice/utils/quality_metrics.py`
- ✅ `/home/kp/autovoice/config/evaluation_config.yaml`
- ✅ `/home/kp/autovoice/src/auto_voice/evaluation/evaluator.py`

## Documentation

Users can now configure NISQA support by:
1. Setting `metrics.naturalness.mos_method` in `config/evaluation_config.yaml`
2. Passing `mos_method` parameter to `NaturalnessMetrics` or `QualityMetricsAggregator`
3. Accessing individual scores via `result.mos_nisqa` and `result.mos_heuristic`

Default behavior unchanged: uses heuristic method for backward compatibility.
