# Source Separator Verification Comments - October 27, 2025

**Status:** ✅ All 5 comments implemented

---

## Comment 1: Fallback Model Restoration ✓

**File**: `src/auto_voice/audio/source_separator.py`

**Implementation**: Updated `_try_fallback()` method to store and restore both `self.backend` and `self.model`.

**Changes**:
- Line 733: Store `original_model = self.model` before loading fallback (Demucs→Spleeter)
- Line 738: Restore `self.model = original_model` after fallback completes
- Line 748: Store `original_model = self.model` before loading fallback (Spleeter→Demucs)
- Line 753: Restore `self.model = original_model` after fallback completes

**Result**: Both backend and model are now consistently restored after fallback operations, preventing model mismatch.

---

## Comment 2: Lazy Loading Test Update ✓

**File**: `tests/test_source_separator.py`

**Implementation**: Updated `test_separator_backend_detection()` to reflect lazy loading behavior.

**Changes**:
- Line 160: Assert backend is in `['demucs', 'spleeter']`
- Line 162: Updated assertion to allow `model` to be `None` or loaded
- Lines 165-173: Added new test `test_separator_eager_load_behavior()` to test eager loading when `defer_model_load=False`

**Result**: Tests now correctly validate both lazy and eager loading behaviors.

---

## Comment 3: Broader Exception Handling ✓

**File**: `tests/test_source_separator.py`

**Implementation**: Broadened exception handling in `setup_method` fixture to catch `ModelLoadError` and other exceptions.

**Changes**:
- Line 25: Import `ModelLoadError` from source_separator module
- Lines 90-92: Added `except Exception` block to catch `ModelLoadError` or other model loading issues
- Updated pytest.skip message to be more descriptive

**Result**: Test suite is now robust to environments where Demucs/Spleeter aren't installed or model loading fails.

---

## Comment 4: Spleeter Dependency Documentation ✓

**File**: `requirements.txt`

**Implementation**: Added comprehensive comments about Spleeter's TensorFlow dependency.

**Changes**:
- Lines 43-48: Added multi-line comment block explaining:
  - Spleeter pulls heavy TensorFlow dependencies
  - Compatible TensorFlow versions (2.5.x - 2.13.x)
  - Potential conflicts with PyTorch CUDA setup
  - Optional TensorFlow pinning suggestion
  - Recommendation to consider making it optional via extras_require

**Result**: Clear documentation for developers about Spleeter's heavy dependency and potential conflicts.

---

## Comment 5: Multi-Channel Downmixing ✓

**Files**:
- `src/auto_voice/audio/source_separator.py`
- `tests/test_source_separator.py`

**Implementation**: Added explicit multi-channel (>2 channels) downmixing to stereo.

**Changes in source_separator.py**:
- Lines 75-79: Updated docstring to document multi-channel downmixing behavior
- Lines 396-418: Added multi-channel handling logic:
  - Mono (1 channel): duplicate to stereo
  - Stereo (2 channels): keep as-is
  - Multi-channel (>2 channels): average all channels to mono, then duplicate to stereo
  - Added logging when downmixing occurs
  - Implemented for both PyTorch tensors and NumPy arrays

**Changes in test_source_separator.py**:
- Lines 863-933: Updated `test_multi_channel_downmix()` to validate stereo output
- Lines 955-1077: Fixed `test_silent_audio()` and `test_noise_only_audio()` to remove missing fixture dependencies

**Result**: Multi-channel audio (>2 channels) is now properly downmixed to stereo before separation, with comprehensive test coverage.

---

## Files Modified

1. `src/auto_voice/audio/source_separator.py` - Comments 1 & 5
2. `tests/test_source_separator.py` - Comments 2, 3 & 5
3. `requirements.txt` - Comment 4

---

## Testing Recommendations

```bash
# Run all source separator tests
pytest tests/test_source_separator.py -v

# Run specific tests
pytest tests/test_source_separator.py::TestVocalSeparator::test_separator_backend_detection -v
pytest tests/test_source_separator.py::TestVocalSeparator::test_separator_eager_load_behavior -v
pytest tests/test_source_separator.py::TestMultiChannelHandling::test_multi_channel_downmix -v
pytest tests/test_source_separator.py::TestFallbackBehavior::test_demucs_to_spleeter_fallback -v
```

**Implementation Date**: October 27, 2025
