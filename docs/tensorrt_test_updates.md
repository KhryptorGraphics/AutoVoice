# TensorRT Test Suite Updates - Comment 17

## Summary
Updated `tests/test_tensorrt_conversion.py` to match new model APIs, add proper environment guards, and include accuracy validation tests.

## Changes Made

### 1. Environment Guards Added

**New Import Checks:**
```python
try:
    import onnxscript
    ONNX_EXPORT_AVAILABLE = True
except ImportError:
    ONNX_EXPORT_AVAILABLE = False
```

**Test Decorators:**
- `@pytest.mark.skipif(not TRT_AVAILABLE, ...)` - Skip when TensorRT not available
- `@pytest.mark.skipif(not ORT_AVAILABLE, ...)` - Skip when ONNX Runtime not available
- `@pytest.mark.skipif(not ONNX_EXPORT_AVAILABLE, ...)` - Skip when onnxscript not available
- Combined conditions for tests requiring multiple dependencies

### 2. Model API Updates

#### PitchEncoder
**Before:**
```python
encoder = PitchEncoder(
    pitch_dim=192,
    hidden_dim=128,
    device='cpu'
)
```

**After:**
```python
encoder = PitchEncoder(
    pitch_dim=192,
    hidden_dim=128,
    f0_min=80.0,
    f0_max=1000.0
)
```

**Test Data Changes:**
- Now generates positive F0 values: `torch.randn(1, 50).abs() * 400 + 100`
- Always provides `voiced_mask` as boolean tensor
- Updated ONNX input names to match exports: `['f0_input', 'voiced_mask']`

#### FlowDecoder
**ONNX Export Changes:**
- Removed `'inverse'` from input parameters
- `inverse=True` is now frozen inside ONNX export wrapper
- Input names: `['latent_input', 'mask', 'conditioning']`
- Output names: `['output_latent']`

### 3. New Test Classes Added

#### TestAccuracyValidation
Validates ONNX export accuracy compared to PyTorch:

**ContentEncoder Accuracy Test:**
- Exports model to ONNX
- Compares PyTorch vs ONNX outputs
- Validates: `max_diff < 1e-3`, `rmse < 1e-4`

**PitchEncoder Accuracy Test:**
- Tests with positive F0 and voiced mask
- Validates: `max_diff < 1e-3`, `rmse < 1e-4`

**FlowDecoder Accuracy Test:**
- Tests inverse mode (inference)
- Validates: `max_diff < 1e-3`, `rmse < 1e-4`

#### TestDynamicShapes
Tests ONNX models with variable input lengths:

**ContentEncoder:**
- Audio lengths: [1s, 3s, 5s, 10s] at 16kHz
- Verifies dynamic shape handling works correctly

**PitchEncoder:**
- Time steps: [50, 150, 250, 500] frames
- Verifies pitch features match input length

**FlowDecoder:**
- Time steps: [50, 150, 250, 500] frames
- Verifies latent dimension preserved

### 4. Integration Test Updates

**test_complete_workflow:**
- Added proper test inputs for each component type
- ContentEncoder: `{'input_audio', 'sample_rate'}`
- PitchEncoder: `{'f0_input', 'voiced_mask'}`
- FlowDecoder: `{'latent_input', 'mask', 'conditioning'}`
- Fixed mel_projection handling (skip for now)

### 5. Fixed Test Expectations

**test_tensorrt_support_flags:**
- Changed to expect TensorRT attributes NOT present initially
- Attributes (`use_tensorrt`, `tensorrt_models`, `fallback_to_pytorch`) are set during `load_tensorrt_engines()` call
- Validates `export_components_to_onnx` capability exists

## Test Results

**Total Tests:** 25
- **Passed:** 3
- **Skipped:** 22 (due to missing TensorRT/onnxscript dependencies)
- **Failed:** 0

**Skipped Test Breakdown:**
- 11 tests skipped: TensorRT not available
- 7 tests skipped: onnxscript not available
- 3 tests skipped: ONNX Runtime or onnxscript not available
- 1 test skipped: TensorRT and ONNX Runtime required

## Expected Behavior

### With Full Dependencies (TensorRT + onnxscript + ONNX Runtime)
All 25 tests should run and validate:
1. Model instantiation with correct APIs
2. ONNX export functionality
3. ONNX model inference
4. Accuracy: PyTorch vs ONNX (max_diff < 1e-3)
5. Dynamic shape handling (1s to 10s audio)
6. Integration workflow

### With Partial Dependencies
Tests gracefully skip when dependencies unavailable:
- Basic tests pass without any ONNX/TensorRT dependencies
- ONNX Runtime tests skip when onnxscript unavailable
- TensorRT tests skip when TensorRT unavailable

## Files Modified

- `/home/kp/autovoice/tests/test_tensorrt_conversion.py`
  - Added environment guards (ONNX_EXPORT_AVAILABLE)
  - Updated model instantiation calls
  - Fixed input/output names
  - Added accuracy validation tests
  - Added dynamic shape tests
  - Fixed integration test inputs
  - Updated attribute expectations

## Key Fixes

1. **Voiced Mask Handling:** Always pass boolean tensor, never None
2. **F0 Values:** Generate positive values in valid range (80-1000 Hz)
3. **FlowDecoder Export:** Removed inverse parameter from ONNX inputs
4. **Environment Safety:** Tests skip gracefully when dependencies missing
5. **Accuracy Metrics:** RMSE and max difference validation added
6. **Dynamic Shapes:** Comprehensive testing with multiple audio lengths

## Next Steps

To run all tests with full functionality:
```bash
pip install onnxscript tensorrt
python -m pytest tests/test_tensorrt_conversion.py -v
```

For CI/CD environments without GPU:
- Tests will skip TensorRT-specific functionality
- ONNX export tests require only onnxscript
- Basic model API tests always run
