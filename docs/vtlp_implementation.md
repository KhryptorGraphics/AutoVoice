# VTLP (Vocal Tract Length Perturbation) Implementation

## Overview

VTLP augmentation has been successfully implemented in the AutoVoice voice conversion training pipeline. This augmentation technique simulates different vocal tract lengths by warping the mel-spectrogram frequency axis, improving model robustness to speaker variations.

## Implementation Details

### 1. Dataset Integration (`src/auto_voice/training/dataset.py`)

**Function**: `create_paired_train_val_datasets()`

- **New Parameter**: `enable_vtlp: bool = False`
- **Behavior**: When `enable_vtlp=True`, adds `SingingAugmentation.vocal_tract_length_perturbation` to the default training transforms
- **Backward Compatible**: Defaults to `False` to maintain existing behavior

```python
def create_paired_train_val_datasets(
    data_dir: Union[str, Path],
    train_metadata: str,
    val_metadata: str,
    audio_config: Optional[AudioConfig] = None,
    train_transforms: Optional[List[Callable]] = None,
    augmentation_prob: float = 0.5,
    pitch_time_stretch_strict: bool = False,
    enable_vtlp: bool = False,  # NEW PARAMETER
    **dataset_kwargs
) -> Tuple[PairedVoiceDataset, PairedVoiceDataset]:
```

### 2. Training Script Update (`examples/train_voice_conversion.py`)

**Configuration Reading**:
```python
# Read VTLP configuration from config file
vtlp_enabled = config.get('augmentation', {}).get('vtlp', {}).get('enabled', False)

# Pass to dataset creation
train_dataset, val_dataset = create_paired_train_val_datasets(
    ...,
    enable_vtlp=vtlp_enabled
)
```

### 3. Configuration File (`config/model_config.yaml`)

**VTLP Section**:
```yaml
# Data augmentation
augmentation:
  # ... other augmentations ...

  # Vocal tract length perturbation (VTLP)
  # Simulates different vocal tract lengths by warping mel-spectrogram frequency axis
  vtlp:
    enabled: true  # Enable/disable VTLP augmentation
    alpha_range: [0.9, 1.1]  # Warping factor range
                             # Values < 1.0: Simulate shorter vocal tract (higher formants)
                             # Values > 1.0: Simulate longer vocal tract (lower formants)
                             # Recommended: [0.9, 1.1] for singing voice conversion
```

### 4. VTLP Augmentation Method (`SingingAugmentation.vocal_tract_length_perturbation`)

**Implementation** (already exists in `dataset.py`):
- **Input**: Mel-spectrogram data dict
- **Processing**: Warps frequency axis using linear interpolation
- **Alpha Range**: Default `[0.9, 1.1]` (±10% warping)
- **Optimization**: Skips processing when alpha ≈ 1.0 (< 0.01 difference)
- **Shape Preservation**: Maintains original mel-spectrogram dimensions
- **Alignment Preservation**: Applies same warping to both source and target

**Key Features**:
- Works directly on mel-spectrograms (no audio recomputation needed)
- Vectorized interpolation for efficiency
- Handles variable-length sequences correctly
- Contiguous array output for CUDA compatibility

## Testing

### Unit Tests (`tests/test_vtlp_augmentation.py`)

**5 Comprehensive Tests** (All Passing ✅):

1. **`test_vtlp_preserves_shape`**: Verifies mel-spectrogram shape is preserved
2. **`test_vtlp_maintains_alignment`**: Confirms temporal alignment between source and target
3. **`test_vtlp_applies_warping`**: Validates frequency warping is applied correctly
4. **`test_vtlp_skips_small_alpha`**: Checks optimization for alpha ≈ 1.0
5. **`test_vtlp_handles_variable_length`**: Tests handling of padded sequences

**Test Results**:
```
tests/test_vtlp_augmentation.py::TestVTLPAugmentation::test_vtlp_preserves_shape PASSED [ 20%]
tests/test_vtlp_augmentation.py::TestVTLPAugmentation::test_vtlp_maintains_alignment PASSED [ 40%]
tests/test_vtlp_augmentation.py::TestVTLPAugmentation::test_vtlp_applies_warping PASSED [ 60%]
tests/test_vtlp_augmentation.py::TestVTLPAugmentation::test_vtlp_skips_small_alpha PASSED [ 80%]
tests/test_vtlp_augmentation.py::TestVTLPAugmentation::test_vtlp_handles_variable_length PASSED [100%]

======================== 5 passed, 1 warning in 18.24s ========================
```

## Usage

### Enabling VTLP in Training

**Method 1: Configuration File**
```yaml
# In config/model_config.yaml
augmentation:
  vtlp:
    enabled: true
    alpha_range: [0.9, 1.1]
```

**Method 2: Programmatic**
```python
from src.auto_voice.training import create_paired_train_val_datasets

train_dataset, val_dataset = create_paired_train_val_datasets(
    data_dir='data/voice_conversion',
    train_metadata='train_pairs.json',
    val_metadata='val_pairs.json',
    enable_vtlp=True  # Enable VTLP augmentation
)
```

## Benefits

1. **Speaker Robustness**: Model learns to handle speakers with different vocal tract lengths
2. **Data Augmentation**: Increases effective training data diversity
3. **Formant Variation**: Simulates natural variations in formant frequencies
4. **Singing Voice Quality**: Particularly effective for singing voice conversion where timbre variations are important

## Performance Characteristics

- **Computation**: Efficient (vectorized interpolation, skips when alpha ≈ 1.0)
- **Memory**: No additional audio storage (operates on mel-spectrograms)
- **Training Impact**: Applied probabilistically based on `augmentation_prob` (default: 0.5)
- **Inference**: Not applied during inference (training-only augmentation)

## Configuration Recommendations

### Conservative (Default)
```yaml
vtlp:
  enabled: true
  alpha_range: [0.9, 1.1]  # ±10% warping
```

### Aggressive (More Diversity)
```yaml
vtlp:
  enabled: true
  alpha_range: [0.85, 1.15]  # ±15% warping
```

### Disabled
```yaml
vtlp:
  enabled: false
```

## Integration Status

✅ **Completed**:
- Dataset function signature updated
- Training script integration
- Configuration file documentation
- Comprehensive unit tests (5/5 passing)
- Implementation documentation

⏳ **Pending**:
- Integration testing with full training pipeline
- Performance benchmarking with/without VTLP
- Ablation study on optimal alpha_range values

## Files Modified

1. `/home/kp/autovoice/src/auto_voice/training/dataset.py`
   - Added `enable_vtlp` parameter to `create_paired_train_val_datasets()`
   - VTLP transform added to default augmentations when enabled

2. `/home/kp/autovoice/examples/train_voice_conversion.py`
   - Added VTLP configuration reading
   - Pass `enable_vtlp` to dataset creation

3. `/home/kp/autovoice/config/model_config.yaml`
   - Enhanced VTLP documentation
   - Added detailed parameter descriptions

4. `/home/kp/autovoice/tests/test_vtlp_augmentation.py` (NEW)
   - Comprehensive unit test suite for VTLP

5. `/home/kp/autovoice/docs/vtlp_implementation.md` (NEW)
   - Implementation documentation

## References

- Original VTLP method: `SingingAugmentation.vocal_tract_length_perturbation` (already implemented)
- Used in: So-VITS-SVC and similar singing voice conversion models
- Technique: Frequency axis warping via linear interpolation
