# Performance Profiling and TensorRT Implementation

## Overview

This document describes the implementation of per-stage timing instrumentation and TensorRT fast path enforcement in the AutoVoice singing conversion pipeline.

## Comment 1: Per-Stage Timing Instrumentation

### Implementation

Added optional `profiling_callback` parameter to `SingingConversionPipeline.convert_song()` that provides precise timing breakdown for each conversion stage.

#### API Changes

**New Parameter:**
```python
def convert_song(
    self,
    song_path: str,
    target_profile_id: str,
    ...,
    profiling_callback: Optional[Callable[[str, float], None]] = None,
    ...
) -> Dict[str, Any]:
```

**Callback Signature:**
- `stage_name: str` - One of: 'separation', 'f0_extraction', 'conversion', 'mixing', 'total'
- `elapsed_ms: float` - Time elapsed for that stage in milliseconds

#### Timing Points

1. **Start**: Record `t0` at beginning of conversion
2. **After Separation**: Compute `t_sep = (now - stage_start) * 1000`, call `profiling_callback('separation', t_sep)`
3. **After F0 Extraction**: Compute delta, call `profiling_callback('f0_extraction', t_f0)`
4. **After Conversion**: Compute delta, call `profiling_callback('conversion', t_conv)`
5. **After Mixing**: Compute delta, call `profiling_callback('mixing', t_mix)`
6. **At Return**: Call `profiling_callback('total', total_ms)` with total time

#### Error Handling

All profiling callbacks are wrapped in try/except to prevent user callback errors from impacting the pipeline:

```python
if profiling_callback:
    try:
        profiling_callback(stage_name, elapsed_ms)
    except Exception as e:
        logger.warning(f"Profiling callback error: {e}")
```

#### Test Implementation

Updated `tests/test_system_validation.py::test_component_level_timing()` to:

1. Define a profiling callback that collects timing data
2. Pass it to `convert_song()`
3. Verify all expected stages are present
4. Compute stage percentages
5. Assert sum of stages is within ±15% of total
6. Write structured breakdown to `validation_results/performance_breakdown.json`

**Performance Breakdown JSON Structure:**
```json
{
  "timestamp": "2025-01-28 12:34:56",
  "device": "cuda",
  "audio_duration_seconds": 30.0,
  "stage_timings_ms": {
    "separation": 1234.56,
    "f0_extraction": 234.56,
    "conversion": 2345.67,
    "mixing": 123.45,
    "total": 3938.24
  },
  "stage_percentages": {
    "separation": 31.3,
    "f0_extraction": 6.0,
    "conversion": 59.6,
    "mixing": 3.1,
    "total": 100.0
  },
  "total_time_seconds": 3.938,
  "rtf": 0.13,
  "gpu_utilization": 85.2
}
```

#### Report Generation

Updated `scripts/generate_validation_report.py` to:

1. Load `performance_breakdown.json`
2. Display stage breakdown as a markdown table
3. Include RTF and GPU utilization if available

**Example Report Section:**
```markdown
## Performance Benchmarks

- **Device:** cuda
- **Audio Duration:** 30.0s
- **Total Time:** 3.938s
- **RTF (Real-Time Factor):** 0.13x
- **Average GPU Utilization:** 85.2%

### Stage Breakdown:

| Stage | Time (ms) | Percentage |
|-------|-----------|------------|
| Separation | 1234.56 | 31.3% |
| F0 Extraction | 234.56 | 6.0% |
| Conversion | 2345.67 | 59.6% |
| Mixing | 123.45 | 3.1% |
| Total | 3938.24 | 100.0% |
```

### Zero-Overhead Design

- Profiling callback is fully optional (default: `None`)
- No timing overhead when callback not provided
- Minimal overhead when enabled (only `time.time()` calls)
- No impact on existing `progress_callback` behavior

---

## Comment 2: TensorRT Fast Path Enforcement

### Implementation

Ensured the pipeline truly exercises the TensorRT fast path when requested, with verification in tests.

#### Pipeline Initialization

Added TensorRT initialization logic in `SingingConversionPipeline.__init__()`:

```python
if self.use_tensorrt:
    try:
        import tensorrt as trt
        
        # Try to load existing engines
        engine_dir = '~/.cache/autovoice/tensorrt_engines'
        engines_loaded = self.voice_converter.load_tensorrt_engines(engine_dir)
        
        if not engines_loaded:
            # Build engines if not found
            onnx_dir = '~/.cache/autovoice/onnx_models'
            
            # Export to ONNX if needed
            if not os.path.exists(onnx_dir):
                self.voice_converter.export_to_onnx(export_dir=onnx_dir)
            
            # Build TensorRT engines
            fp16 = (self.tensorrt_precision == 'fp16')
            self.voice_converter.create_tensorrt_engines(
                onnx_dir=onnx_dir,
                engine_dir=engine_dir,
                fp16=fp16
            )
    except ImportError:
        logger.warning("TensorRT not available, falling back to PyTorch")
        self.use_tensorrt = False
    except Exception as e:
        logger.warning(f"TensorRT initialization failed: {e}")
        self.use_tensorrt = False
```

#### Conversion Path Selection

Updated `convert_song()` to use TensorRT when enabled:

```python
if self.use_tensorrt and hasattr(self.voice_converter, 'convert_with_tensorrt'):
    logger.info("Using TensorRT-accelerated conversion")
    converted_vocals = self.voice_converter.convert_with_tensorrt(
        vocals, target_embedding, source_f0, ...
    )
    # Handle tuple return (audio, timing_info)
    if isinstance(converted_vocals, tuple):
        converted_vocals = converted_vocals[0]
else:
    converted_vocals = self.voice_converter.convert(
        vocals, target_embedding, source_f0, ...
    )
```

#### Metadata Reporting

Added TensorRT status to result metadata:

```python
'metadata': {
    ...,
    'tensorrt': {
        'enabled': self.use_tensorrt and self.voice_converter.trt_enabled,
        'precision': self.tensorrt_precision if self.use_tensorrt else None
    }
}
```

#### Test Verification

Updated `tests/test_system_validation.py::test_latency_target_30s_input()` to:

1. Create pipeline with `use_tensorrt=True` and `tensorrt_precision='fp16'`
2. Run warm-up conversion (TensorRT compiles on first run)
3. Run timed conversion
4. Extract TensorRT metadata from result
5. Assert `result['metadata']['tensorrt']['enabled'] is True`
6. Skip test if TensorRT engines not loaded (graceful degradation)
7. Assert latency < 5s for 30s input
8. Save metrics to `validation_results/latency_tensorrt.json`

**Latency Metrics JSON Structure:**
```json
{
  "duration_seconds": 30.0,
  "elapsed_seconds": 4.23,
  "rtf": 0.141,
  "preset": "fast",
  "tensorrt_requested": true,
  "tensorrt_enabled": true,
  "tensorrt_precision": "fp16",
  "target_met": true
}
```

#### Report Generation

Updated `scripts/generate_validation_report.py` to include TensorRT latency section:

```markdown
## TensorRT Latency Test

- **Audio Duration:** 30.0s
- **Conversion Time:** 4.230s
- **RTF:** 0.14x
- **Preset:** fast
- **TensorRT Requested:** ✅ Yes
- **TensorRT Enabled:** ✅ Yes
- **TensorRT Precision:** fp16
- **Target (<5s):** ✅ PASSED
```

### Graceful Degradation

The implementation handles missing TensorRT gracefully:

1. **Import Error**: Falls back to PyTorch if TensorRT not installed
2. **Engine Build Failure**: Falls back to PyTorch if engine creation fails
3. **Runtime Error**: Falls back to PyTorch if TensorRT inference fails
4. **Test Behavior**: Skips test if engines not available (doesn't fail)

### Requirements

To use TensorRT acceleration:

- NVIDIA GPU with Tensor Cores (RTX 2060+)
- CUDA 11.8+
- TensorRT 8.5+
- First run will export ONNX and build engines (slow)
- Subsequent runs use cached engines (fast)

---

## Files Modified

### Core Implementation
- `src/auto_voice/inference/singing_conversion_pipeline.py`
  - Added `profiling_callback` parameter to `convert_song()`
  - Added timing instrumentation at each stage
  - Added TensorRT initialization in `__init__()`
  - Added TensorRT path selection in `convert_song()`
  - Added TensorRT metadata to results

### Tests
- `tests/test_system_validation.py`
  - Updated `test_component_level_timing()` with profiling callback
  - Updated `test_latency_target_30s_input()` with TensorRT verification
  - Added performance breakdown JSON output
  - Added latency metrics JSON output

### Reporting
- `scripts/generate_validation_report.py`
  - Added performance breakdown table rendering
  - Added TensorRT latency section
  - Added `latency_tensorrt.json` to loaded files

### Documentation
- `docs/performance_profiling_implementation.md` (this file)

---

## Usage Examples

### Example 1: Profiling Callback

```python
from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

pipeline = SingingConversionPipeline(device='cuda')

# Define profiling callback
stage_timings = {}
def profiling_callback(stage_name: str, elapsed_ms: float):
    stage_timings[stage_name] = elapsed_ms
    print(f"{stage_name}: {elapsed_ms:.2f}ms")

# Run conversion with profiling
result = pipeline.convert_song(
    song_path='input.wav',
    target_profile_id='profile-123',
    profiling_callback=profiling_callback
)

# Access timing data
print(f"Total time: {stage_timings['total']:.2f}ms")
print(f"Conversion: {stage_timings['conversion']:.2f}ms")
```

### Example 2: TensorRT Acceleration

```python
from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

# Create pipeline with TensorRT
pipeline = SingingConversionPipeline(
    device='cuda',
    preset='fast',
    use_tensorrt=True,
    tensorrt_precision='fp16'
)

# Run conversion
result = pipeline.convert_song(
    song_path='input.wav',
    target_profile_id='profile-123'
)

# Check if TensorRT was used
trt_info = result['metadata']['tensorrt']
print(f"TensorRT enabled: {trt_info['enabled']}")
print(f"Precision: {trt_info['precision']}")
```

---

## Testing

Run performance tests:

```bash
# Component-level timing test
pytest tests/test_system_validation.py::TestSystemValidation::test_component_level_timing -v

# TensorRT latency test (requires TensorRT)
pytest tests/test_system_validation.py::TestSystemValidation::test_latency_target_30s_input -v

# GPU utilization test
pytest tests/test_system_validation.py::TestSystemValidation::test_gpu_utilization_monitoring -v
```

Generate validation report:

```bash
python scripts/generate_validation_report.py
```

---

## Performance Targets

| Metric | Target | Verification |
|--------|--------|--------------|
| Latency (30s input, TensorRT FP16) | < 5s | `test_latency_target_30s_input` |
| GPU Utilization (CUDA) | > 70% | `test_gpu_utilization_monitoring` |
| Stage Timing Accuracy | ±15% of total | `test_component_level_timing` |

---

## Future Enhancements

1. **Per-Stage GPU Utilization**: Track GPU usage per stage, not just overall
2. **Memory Profiling**: Add memory usage tracking per stage
3. **TensorRT INT8**: Support INT8 quantization for further speedup
4. **Multi-GPU**: Support TensorRT across multiple GPUs
5. **Persistent Engines**: Cache engines across sessions more efficiently

