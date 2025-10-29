# TensorRT Implementation Progress

## Completed Fixes

### ✅ Comment 1: Content Encoder ONNX Export
- Added `export_to_onnx()` method to `ContentEncoder`
- Added `prepare_for_export()` with HuBERT rejection
- Proper dynamic axes: `input_audio` (batch, time) and `content_features` (batch, time)
- CNN fallback only (HuBERT not exportable to ONNX)

### ✅ Comment 2: PitchEncoder ONNX Export
- Added `export_to_onnx()` method to `PitchEncoder`
- Always passes boolean tensor for `voiced_mask`
- Dynamic axes for both `f0_input`, `voiced_mask`, and `pitch_features`

### ✅ Comment 3: FlowDecoder ONNX Export
- Added `export_to_onnx()` method to `FlowDecoder`
- Uses wrapper to freeze `inverse=True` internally
- Does NOT expose `inverse` as input
- Proper dynamic axes for all inputs and outputs

### ✅ Comment 4: ContentEncoder dynamic_axes Fix
- Fixed in `tensorrt_converter.py`
- Changed to `'input_audio': {0: 'batch_size', 1: 'audio_length'}`
- Changed to `'content_features': {0: 'batch_size', 1: 'time_frames'}`
- Removed dynamic axes for `sample_rate`
- Ensured input shape is [1, T] not [1, C, T]

### ✅ Comment 5: PitchEncoder voiced_mask Boolean Tensor
- Updated `export_pitch_encoder()` in `tensorrt_converter.py`
- Always passes boolean tensor (all-ones if None provided)
- Included `voiced_mask` in dynamic_axes

### ✅ Comment 6: FlowDecoder inverse Input Removal
- Modified `export_flow_decoder()` in `tensorrt_converter.py`
- Wrapper freezes `inverse=True` internally
- Removed `inverse` from input_names and inputs tuple

### ✅ Comment 7: Dynamic Shape Handling in TensorRTEngine.infer()
- Added `context.set_binding_shape()` for all dynamic inputs
- Allocates buffers using resolved shapes from `context.get_binding_shape()`
- Proper memory management per inference call

### ✅ Comment 8: TensorRTEngine Destructor Fix
- Removed references to undefined `input_buffers` and `output_buffers`
- Safely cleans up `context` and `engine` with `hasattr()` checks
- Memory now freed per inference call, not in destructor

## Remaining Work

### 🔄 Comment 9: INT8 Calibration
**Status:** In Progress
**Files:** `src/auto_voice/inference/tensorrt_engine.py`, `tensorrt_converter.py`
**Tasks:**
- Implement `IInt8EntropyCalibrator2` class
- Create calibration dataset generator (NPZ format)
- Wire calibrator into `optimize_with_tensorrt()`
- Add cache file for reuse

### 🔄 Comment 10: Benchmark Script Import Path
**Status:** Pending
**Files:** `scripts/benchmark_tensorrt.py`
**Tasks:**
- Change import to `from src.auto_voice.inference.tensorrt_engine import TensorRTEngine`
- OR add project root to `sys.path`

### 🔄 Comment 11: PyTorch Baseline Benchmarking
**Status:** Pending
**Files:** `scripts/benchmark_tensorrt.py`
**Tasks:**
- Implement `_benchmark_pytorch()` method
- Load PyTorch components or full `SingingVoiceConverter`
- Time inference and report latency/throughput alongside TensorRT results

### 🔄 Comment 12: I/O Tensor Name Mismatches
**Status:** Pending
**Files:** `src/auto_voice/models/singing_voice_converter.py`
**Tasks:**
- Update TensorRT path to use exact ONNX output names
- Match: `content_features`, `pitch_features`, `output_latent`, `mel_output`
- Match input names: `input_audio`, `sample_rate`, `f0_input`, `voiced_mask`

### 🔄 Comment 13: sample_rate Dtype Fix
**Status:** Pending
**Files:** `src/auto_voice/models/singing_voice_converter.py`
**Tasks:**
- Create `sample_rate` as NumPy int32 array before passing to TensorRT
- Consider removing as model input and fixing as constant

### 🔄 Comment 14: Duplicate TensorRTEngine in engine.py
**Status:** Pending
**Files:** `src/auto_voice/inference/engine.py`
**Tasks:**
- Remove duplicate `TensorRTEngine` class
- Import from `tensorrt_engine.py`
- Add `voice_conversion` mode initialization

### 🔄 Comment 15: TensorRT Config in YAML
**Status:** Pending
**Files:** `config/model_config.yaml`
**Tasks:**
- Add `tensorrt` block for voice conversion
- Include dynamic shape min/opt/max for audio, F0, flow, mel
- Add precision flags, workspace, calibration settings

### 🔄 Comment 16: Accuracy Comparison Utilities
**Status:** Pending
**Files:** `scripts/benchmark_tensorrt.py`
**Tasks:**
- Add accuracy comparison function
- Compute max diff, mean diff, RMSE, correlation, SNR
- Emit metrics alongside latency/throughput

### 🔄 Comment 17: Test Suite Updates
**Status:** Pending
**Files:** `tests/test_tensorrt_conversion.py` and related tests
**Tasks:**
- Update to match new model APIs
- Add missing export methods
- Fix constructor args
- Align input/output names with ONNX exports
- Add skips/guards for environments without TRT/ORT
- Validate accuracy and performance

## Next Steps

1. Implement INT8 calibration (Comment 9)
2. Fix benchmark script imports and add PyTorch comparison (Comments 10, 11)
3. Fix tensor name mismatches in SingingVoiceConverter (Comments 12, 13)
4. Refactor engine.py (Comment 14)
5. Add TensorRT config to YAML (Comment 15)
6. Add accuracy comparison utilities (Comment 16)
7. Update test suite (Comment 17)
