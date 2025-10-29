# Loss Fallbacks Implementation Summary

## Overview
Implemented graceful fallbacks for `PitchConsistencyLoss` and `SpeakerSimilarityLoss` classes to handle situations where voice conversion components are unavailable or fail to initialize.

## Changes Made

### 1. PitchConsistencyLoss Updates
**File**: `/home/kp/autovoice/src/auto_voice/training/trainer.py`

Added internal state management:
- `_extractor_available`: Boolean flag tracking if pitch extractor is available
- `_warned`: Boolean flag ensuring warnings are logged only once

**Initialization**:
```python
def __init__(self, device: Optional[str] = None):
    super().__init__()
    self.device = device
    self._extractor_available = True
    self._warned = False

    if VC_COMPONENTS_AVAILABLE:
        try:
            self.pitch_extractor = SingingPitchExtractor(device=device)
        except Exception as e:
            self._extractor_available = False
            self.pitch_extractor = None
            if not self._warned:
                logger.warning(f"Pitch extractor initialization failed: {e}")
                self._warned = True
    else:
        self._extractor_available = False
        self.pitch_extractor = None
        if not self._warned:
            logger.warning("SingingPitchExtractor not available, pitch consistency loss will be disabled")
            self._warned = True
```

**Forward Pass**:
```python
def forward(self, pred_audio, source_f0, sample_rate=44100):
    if not self._extractor_available:
        if not self._warned:
            logger.warning("Pitch loss returning zero (extractor unavailable)")
            self._warned = True
        return torch.tensor(0.0, device=pred_audio.device)

    # Normal computation...
```

### 2. SpeakerSimilarityLoss Updates
**File**: `/home/kp/autovoice/src/auto_voice/training/trainer.py`

Added internal state management:
- `_encoder_available`: Boolean flag tracking if speaker encoder is available
- `_warned`: Boolean flag ensuring warnings are logged only once

**Initialization**:
```python
def __init__(self, device: Optional[str] = None):
    super().__init__()
    self.device = device
    self._encoder_available = True
    self._warned = False

    if VC_COMPONENTS_AVAILABLE:
        try:
            self.speaker_encoder = SpeakerEncoder(device=device)
        except Exception as e:
            self._encoder_available = False
            self.speaker_encoder = None
            if not self._warned:
                logger.warning(f"Speaker encoder initialization failed: {e}")
                self._warned = True
    else:
        self._encoder_available = False
        self.speaker_encoder = None
        if not self._warned:
            logger.warning("SpeakerEncoder not available, speaker similarity loss will be disabled")
            self._warned = True
```

**Forward Pass**:
```python
def forward(self, pred_audio, target_speaker_emb, sample_rate=44100):
    if not self._encoder_available:
        if not self._warned:
            logger.warning("Speaker loss returning zero (encoder unavailable)")
            self._warned = True
        return torch.tensor(0.0, device=pred_audio.device)

    # Normal computation...
```

### 3. Syntax Fixes
Fixed escaped quotes in `getattr()` calls throughout the file:
- Changed `getattr(self.config, \'local_rank\', 0)` to `getattr(self.config, 'local_rank', 0)`
- Applied to 9 occurrences in the file

## Test Coverage

### Test File
**Created**: `/home/kp/autovoice/tests/test_loss_fallbacks.py`

### Test Cases (14 total, all passing)

**Unavailable Components**:
1. `test_pitch_loss_with_unavailable_components` - Verifies graceful handling when VC components unavailable
2. `test_speaker_loss_with_unavailable_components` - Verifies graceful handling when VC components unavailable

**Initialization Failures**:
3. `test_pitch_loss_with_initialization_failure` - Tests exception handling during initialization
4. `test_speaker_loss_with_initialization_failure` - Tests exception handling during initialization

**Warning Behavior**:
5. `test_pitch_loss_warning_logged_once_init` - Ensures warning logged once during init
6. `test_speaker_loss_warning_logged_once_init` - Ensures warning logged once during init
7. `test_pitch_loss_warning_logged_once_forward` - Ensures warning logged once during forward pass
8. `test_speaker_loss_warning_logged_once_forward` - Ensures warning logged once during forward pass
9. `test_pitch_loss_multiple_forward_calls_no_repeated_warnings` - Verifies no repeated warnings
10. `test_speaker_loss_multiple_forward_calls_no_repeated_warnings` - Verifies no repeated warnings

**Device Handling**:
11. `test_pitch_loss_returns_tensor_on_correct_device` - Verifies tensor device correctness
12. `test_speaker_loss_returns_tensor_on_correct_device` - Verifies tensor device correctness

**Exception Safety**:
13. `test_pitch_loss_no_exception_on_forward` - Ensures no exceptions with various tensor sizes
14. `test_speaker_loss_no_exception_on_forward` - Ensures no exceptions with various tensor sizes

### Test Results
```
14 passed, 1 warning in 1.16s
```

## Behavior

### When Components Available
- Normal operation with full functionality
- Pitch extraction and speaker encoding work as expected

### When Components Unavailable
1. **Initialization**: Warning logged once explaining unavailability
2. **Forward Pass**: Returns `torch.tensor(0.0, device=pred_audio.device)` without exceptions
3. **Warning Logging**: One-time warning on first forward call (if not already warned during init)
4. **Training Continuity**: Training continues without interruption

### Benefits
- **Robustness**: Training doesn't crash when optional components are missing
- **Clear Communication**: Users are warned about disabled features
- **Silent Operation**: After initial warning, no log spam
- **Correct Device**: Returns zero tensor on the same device as input

## Files Modified
1. `/home/kp/autovoice/src/auto_voice/training/trainer.py`
   - Updated `PitchConsistencyLoss` class
   - Updated `SpeakerSimilarityLoss` class
   - Fixed syntax errors with escaped quotes

## Files Created
1. `/home/kp/autovoice/tests/test_loss_fallbacks.py`
   - Comprehensive test suite for fallback behavior
   - 14 test cases covering all scenarios

## Compliance
✅ All requirements from Comment 6 implemented:
- Internal availability flags (`_extractor_available`, `_encoder_available`)
- Warning flag to log once (`_warned`)
- Updated `forward()` methods with graceful fallbacks
- Tests simulating unavailable components
- Tests asserting loss returns zero without exceptions
- Tests verifying warning is logged once

## Validation
- All 14 tests passing
- No syntax errors
- Proper device handling
- One-time warning logging
- Zero loss fallback behavior confirmed
