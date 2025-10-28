# Comment 1 Implementation: Device Source Unification for AMP Enablement

## Summary
Fixed device source inconsistency in `_separate_with_demucs()` method where AMP enablement was using a different device source than audio/model placement.

## Changes Made

### File: `/src/auto_voice/audio/source_separator.py`

**Location:** `_separate_with_demucs()` method (lines 534-558)

**Before:**
- AMP enablement computed device using `device_obj = torch.device(self.device)`
- This mixed two device sources: the yielded context device and `self.device`
- Model device check was less robust

**After:**
- All device-dependent decisions now use the single `device` variable from context
- AMP enablement: `use_amp = (getattr(device, 'type', 'cpu') == 'cuda' ...)`
- Improved model device checking with `hasattr(self.model, 'to')` guard
- Removed redundant `torch.device(self.device)` construction

## Technical Details

### Device Flow
1. **Context Entry**: Device yielded from `GPUManager.device_context()` or fallback to `torch.device(self.device)`
2. **Audio Placement**: `audio.to(device)` using the context device
3. **Model Placement**: Model moved to same `device` if needed
4. **AMP Decision**: Uses `device.type` attribute directly from context device
5. **Processing**: All operations use consistent device

### Key Improvements

1. **Unified Device Source**
   - Single source of truth for device in the method
   - No mixing of context-yielded device and `self.device`

2. **Safer Model Movement**
   - Added `hasattr(self.model, 'to')` check before moving model
   - Prevents errors if model doesn't support `.to()` method

3. **Clearer Logic**
   - AMP enablement directly tied to the device variable in scope
   - Comment clarifies this is the "same device variable that was yielded/assigned above"

## Benefits

1. **Consistency**: All device operations derive from the same source
2. **Maintainability**: Future changes to device logic are centralized
3. **Correctness**: Prevents subtle bugs during device fallbacks
4. **Clarity**: Code intent is more obvious to readers

## Testing Notes

Due to PyTorch installation issues in the test environment, static code review was performed:

- Verified no other `torch.device(self.device)` calls exist in `_separate_with_demucs()`
- Confirmed device variable is used consistently throughout the method
- Logic flow matches the verification comment requirements exactly

## Verification Checklist

- [x] AMP enablement uses context-yielded `device` variable
- [x] Model placement uses same `device` variable
- [x] No redundant `torch.device(self.device)` in method
- [x] Improved model `.to()` safety check
- [x] Clear comments explain device source
- [x] All device decisions centralized to single source

## Related Files
- `/src/auto_voice/audio/source_separator.py` (modified)
- GPU manager integration (context yields device)
- Demucs separation pipeline

## Next Steps
- Run full test suite when PyTorch environment is fixed
- Verify AMP behavior on both CUDA and CPU devices
- Consider similar pattern for other device-dependent methods
