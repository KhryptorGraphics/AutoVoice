# INT8 Calibration Pipeline Fixes

## Overview

This document details the comprehensive fixes applied to the INT8 calibration pipeline for TensorRT in AutoVoice. The fixes ensure proper calibration data handling, correct dtypes, device buffer management, and calibration cache persistence.

## Issues Fixed

### 1. Calibrator Data Override Bug

**Problem:** In `_create_calibrator()`, line 804 was overwriting the provided `calibration_data` parameter with an empty list:

```python
# WRONG - overwrites parameter
calibration_data = []
logger.info("INT8 calibrator created with entropy minimization")
return INT8EntropyCalibrator(calibration_data, calibration_cache_file)
```

**Fix:** Honor the provided `calibration_data` parameter:

```python
# CORRECT - honors parameter
if calibration_data is None:
    calibration_data = []
    logger.warning("No calibration data provided, using empty calibration dataset")

logger.info(f"INT8 calibrator created with {len(calibration_data)} calibration samples")
return INT8EntropyCalibrator(calibration_data, calibration_cache_file)
```

### 2. Missing Math Import

**Problem:** `create_calibration_dataset()` used `math.ceil()` without importing the `math` module.

**Fix:** Added `import math` at the top of `tensorrt_converter.py`.

### 3. Incorrect Data Types

**Problem:** Calibration data was not ensuring correct dtypes for TensorRT:
- `sample_rate` must be `int32`
- `voiced_mask` must be `bool` (not `float32`)
- Other tensors should be `float32`

**Fix in `_load_calibration_data()`:**

```python
# Content encoder
calibration_data.append({
    'input_audio': audio_data[i].astype(np.float32),
    'sample_rate': sample_rate_data[i:i+1].astype(np.int32)
})

# Pitch encoder
calibration_data.append({
    'f0_input': f0_data[i].astype(np.float32),
    'voiced_mask': voiced_data[i].astype(np.bool_)  # Ensure bool dtype
})
```

**Fix in `create_calibration_dataset()`:**

```python
# Ensure correct dtype for each component
if 'content/sample_rate' in key:
    calibration_samples[key] = np.array(samples, dtype=np.int32)
elif 'pitch/voiced_mask' in key:
    calibration_samples[key] = np.array(samples, dtype=np.bool_)
else:
    calibration_samples[key] = np.array(samples, dtype=np.float32)
```

### 4. Device Buffer Allocation

**Problem:** The `INT8EntropyCalibrator` class did not properly allocate per-binding device buffers based on input names, and didn't handle different dtypes correctly.

**Fix:** Complete rewrite of `get_batch()` method:

```python
def get_batch(self, names):
    """Get next calibration batch with proper device buffer management."""
    if self.current_index >= len(self.calibration_data):
        # No more calibration data - free buffers
        for buf in self.device_buffers.values():
            try:
                buf.free()
            except:
                pass
        return None

    # Get current batch data
    batch_data = self.calibration_data[self.current_index]
    self.current_index += 1

    # Allocate device buffers on first call based on batch data
    if not self.device_buffers:
        for name in names:
            if name in batch_data:
                data = batch_data[name]
                data = np.ascontiguousarray(data)
                size = data.nbytes
                try:
                    device_mem = cuda.mem_alloc(size)
                    self.device_buffers[name] = device_mem
                    logger.debug(f"Allocated {size} bytes for input '{name}', dtype={data.dtype}, shape={data.shape}")
                except Exception as e:
                    logger.error(f"Failed to allocate device buffer for {name}: {e}")

    # Copy data to device with proper dtype handling
    device_ptrs = []
    for name in names:
        if name in batch_data and name in self.device_buffers:
            data = batch_data[name]
            # Ensure proper dtype
            if name == 'sample_rate':
                data = data.astype(np.int32)
            elif name == 'voiced_mask':
                data = data.astype(np.bool_)
            else:
                data = data.astype(np.float32)

            data = np.ascontiguousarray(data)

            try:
                cuda.memcpy_htod(self.device_buffers[name], data)
                device_ptrs.append(int(self.device_buffers[name]))
            except Exception as e:
                logger.error(f"Failed to copy data for {name}: {e}")
                device_ptrs.append(0)
        else:
            if name not in batch_data:
                logger.warning(f"Calibration data missing for input: {name}")
            device_ptrs.append(0)

    return device_ptrs
```

### 5. Calibrator Initialization

**Problem:** The calibrator constructor didn't handle `None` values gracefully.

**Fix:**

```python
def __init__(self, calibration_data: List[Dict[str, np.ndarray]], cache_file: str):
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.calibration_data = calibration_data if calibration_data is not None else []
    self.cache_file = cache_file
    self.current_index = 0
    self.device_buffers = {}
```

## Component-Specific Requirements

### Content Encoder
- **Inputs:**
  - `input_audio`: float32 [B, T]
  - `sample_rate`: int32 [1]
- **Output:**
  - `content_features`: float32 [B, T', D]

### Pitch Encoder
- **Inputs:**
  - `f0_input`: float32 [B, T]
  - `voiced_mask`: bool [B, T]
- **Output:**
  - `pitch_features`: float32 [B, T, D]

### Flow Decoder
- **Inputs:**
  - `latent_input`: float32 [B, 192, T]
  - `mask`: float32 [B, 1, T]
  - `conditioning`: float32 [B, 704, T]
- **Output:**
  - `output_latent`: float32 [B, 192, T]

### Mel Projection
- **Input:**
  - `latent_input`: float32 [B, 192, T]
- **Output:**
  - `mel_output`: float32 [B, 80, T]

## Usage Example

### 1. Create Calibration Dataset

```python
from src.auto_voice.inference.tensorrt_converter import TensorRTConverter

# Initialize converter
converter = TensorRTConverter(export_dir="./models", device='cpu')

# Create calibration dataset from training data
calibration_npz = converter.create_calibration_dataset(
    dataset=training_dataset,
    num_samples=100,  # Use 100 samples for calibration
    output_path="./models/calibration.npz"
)
```

### 2. Build INT8 Engine with Calibration

```python
# Export ONNX first
onnx_path = converter.export_content_encoder(
    content_encoder=model.content_encoder,
    model_name="content_encoder"
)

# Optimize with INT8 and calibration
engine_path = converter.optimize_with_tensorrt(
    onnx_path=onnx_path,
    engine_path="./models/content_encoder_int8.engine",
    fp16=False,
    int8=True,
    component_name='content_encoder',
    calibration_npz=calibration_npz
)
```

### 3. Use INT8 Engines for Inference

```python
# Load INT8 engines
model.load_tensorrt_engines(engine_dir="./models")

# Convert with INT8 acceleration
audio = model.convert_with_tensorrt(
    source_audio=source_audio,
    target_speaker_embedding=target_emb,
    source_sample_rate=16000,
    output_sample_rate=44100
)
```

## Testing

Run the INT8 calibration tests:

```bash
pytest tests/test_int8_calibration.py -v
```

Or run standalone:

```bash
python tests/test_int8_calibration.py
```

Expected output:
```
Running INT8 Calibration Tests
==================================================
✓ Calibration dataset created with correct dtypes
✓ Calibration data loaded with correct dtypes
✓ INT8 calibrator created and functioning correctly
✓ Calibrator correctly honors provided calibration_data
==================================================
✓ All INT8 calibration tests passed!
==================================================
```

## Verification Checklist

- [x] `math` module imported
- [x] `_create_calibrator()` honors `calibration_data` parameter
- [x] `_load_calibration_data()` returns correct dtypes
- [x] `create_calibration_dataset()` saves correct dtypes to NPZ
- [x] `INT8EntropyCalibrator.get_batch()` allocates device buffers correctly
- [x] `INT8EntropyCalibrator.get_batch()` handles dtypes per binding
- [x] `INT8EntropyCalibrator.get_batch()` validates input names
- [x] `INT8EntropyCalibrator.get_batch()` frees buffers when done
- [x] `INT8EntropyCalibrator.read_calibration_cache()` implemented
- [x] `INT8EntropyCalibrator.write_calibration_cache()` implemented
- [x] Calibrator passed to `config.int8_calibrator` (TRT ≥ 8)
- [x] Calibrator passed to `builder.int8_calibrator` (legacy TRT)
- [x] `SingingVoiceConverter.create_tensorrt_engines()` propagates `calibration_npz`

## Performance Impact

With proper INT8 calibration:

- **Expected speedup:** 2-4x compared to FP16
- **Accuracy loss:** < 1% with good calibration data
- **Memory usage:** ~4x reduction compared to FP32

## Troubleshooting

### Issue: "No calibration data provided"

**Cause:** `calibration_npz` not passed or file doesn't exist.

**Solution:** Create calibration dataset first and pass path to `optimize_with_tensorrt()`.

### Issue: "Calibration data missing for input: X"

**Cause:** NPZ file doesn't contain required keys for component.

**Solution:** Verify NPZ structure matches component requirements.

### Issue: "Failed to allocate device buffer"

**Cause:** GPU memory insufficient or CUDA not initialized.

**Solution:** Reduce calibration batch count or free GPU memory.

### Issue: INT8 engine produces incorrect results

**Cause:** Insufficient or non-representative calibration data.

**Solution:** Use more samples (100-500) covering diverse inputs.

## Future Improvements

1. **Dynamic Shape Calibration:** Vary sequence lengths across batches to calibrate dynamic shapes better.
2. **Per-Layer Calibration:** Allow different calibration strategies per layer.
3. **Automatic Sample Selection:** Intelligently select most representative calibration samples.
4. **Multi-GPU Calibration:** Distribute calibration across multiple GPUs for speed.

## References

- [TensorRT INT8 Calibration Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- [ONNX INT8 Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
- AutoVoice TensorRT Integration: `src/auto_voice/inference/tensorrt_converter.py`
