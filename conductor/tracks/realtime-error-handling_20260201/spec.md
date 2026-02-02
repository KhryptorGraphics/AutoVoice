# Realtime Pipeline Error Handling

**Track ID:** realtime-error-handling_20260201
**Priority:** P0 (Critical)
**Type:** Bugfix / Gap Remediation
**Status:** Pending
**Origin:** Gap Analysis - Missing error handling in production code

## Problem Statement

Gap analysis revealed that `RealtimePipeline` (used for live karaoke) lacks error handling for critical failure modes:

1. **GPU OOM** - No graceful degradation when GPU memory exhausted
2. **Model Loading Failures** - Pipeline crashes if ContentVec/RMVPE/HiFiGAN fail to load
3. **Audio Processing Errors** - No handling for malformed audio input
4. **Speaker Embedding Errors** - No validation of embedding dimensions/format

**Impact:** Production crashes during live karaoke sessions, poor user experience.

## Files Requiring Error Handling

### Primary
- `src/auto_voice/inference/realtime_pipeline.py` (335 lines, 0 error handling)
- `src/auto_voice/inference/streaming_pipeline.py` (if exists)

### Secondary (for consistency)
- `src/auto_voice/inference/realtime_voice_conversion_pipeline.py`
- `src/auto_voice/inference/trt_streaming_pipeline.py`

## Error Handling Requirements

### Phase 1: Model Initialization Errors

**File:** `realtime_pipeline.py`

**Methods to update:**
1. `_init_content_encoder()` - Catch model loading errors
2. `_init_pitch_extractor()` - Catch RMVPE loading errors
3. `_init_decoder()` - Catch decoder initialization errors
4. `_init_vocoder()` - Catch HiFiGAN checkpoint loading errors

**Error types:**
- `FileNotFoundError` - Model files missing
- `RuntimeError` - GPU OOM during model loading
- `ValueError` - Invalid model configuration

**Recovery strategy:**
```python
try:
    # Load model
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise RuntimeError(f"Failed to initialize {component}: model file missing") from e
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"GPU OOM during {component} loading")
    torch.cuda.empty_cache()
    raise RuntimeError(f"Insufficient GPU memory for {component}") from e
except Exception as e:
    logger.error(f"Unexpected error loading {component}: {e}")
    raise RuntimeError(f"Failed to initialize {component}") from e
```

### Phase 2: Runtime Processing Errors

**Method:** `process_chunk()`

**Error types:**
1. **Invalid audio input**
   - Empty audio
   - Wrong sample rate
   - Non-float32 dtype
   - NaN/Inf values

2. **GPU errors during inference**
   - CUDA OOM during forward pass
   - Device-side assertion failures

3. **Numerical errors**
   - NaN in encoder output
   - Inf in decoder output
   - Division by zero in normalization

**Recovery strategy:**
```python
def process_chunk(self, audio: np.ndarray) -> np.ndarray:
    """Process audio with comprehensive error handling."""
    # Input validation
    if audio.size == 0:
        logger.warning("Empty audio chunk received, returning silence")
        return np.zeros(4096, dtype=np.float32)

    if not np.isfinite(audio).all():
        logger.error("Non-finite values in input audio")
        raise ValueError("Input audio contains NaN or Inf")

    try:
        # ... existing processing code ...

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM during chunk processing")
        torch.cuda.empty_cache()
        # Return passthrough audio as fallback
        return audio.astype(np.float32)

    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error during processing: {e}")
            torch.cuda.empty_cache()
            return audio.astype(np.float32)
        raise

    except Exception as e:
        logger.error(f"Unexpected error in process_chunk: {e}", exc_info=True)
        # Return passthrough audio to keep stream alive
        return audio.astype(np.float32)
```

### Phase 3: Speaker Embedding Validation

**Method:** `set_speaker_embedding()`

**Validations:**
1. Check dimension (must be 256)
2. Check dtype (float32)
3. Check for NaN/Inf
4. Verify L2 normalization (optional: auto-normalize)

**Error handling:**
```python
def set_speaker_embedding(self, embedding: np.ndarray) -> None:
    """Set speaker embedding with validation."""
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim != 1 or embedding.shape[0] != 256:
        raise ValueError(
            f"Speaker embedding must be 1D array of length 256, got shape {embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError("Speaker embedding contains NaN or Inf")

    # Auto-normalize if not normalized
    norm = np.linalg.norm(embedding)
    if not np.isclose(norm, 1.0, atol=0.01):
        logger.warning(f"Speaker embedding not L2-normalized (norm={norm:.3f}), normalizing")
        embedding = embedding / (norm + 1e-8)

    self._speaker_embedding = torch.from_numpy(embedding[np.newaxis, :]).to(self.device)
    logger.info("Speaker embedding set and validated")
```

## Testing Requirements

### Unit Tests
Create `tests/test_realtime_pipeline_error_handling.py`:

1. **Test model loading failures**
   - Invalid model paths
   - Corrupted checkpoints
   - GPU OOM simulation

2. **Test process_chunk error recovery**
   - Empty audio input
   - Audio with NaN/Inf
   - GPU OOM during inference

3. **Test speaker embedding validation**
   - Wrong dimensions
   - NaN/Inf values
   - Unnormalized embeddings

### Integration Tests
Update `tests/test_karaoke_integration.py`:

1. Test graceful degradation during live session
2. Test recovery from transient GPU errors
3. Test passthrough fallback when conversion fails

## Success Criteria

1. ✅ All 4 init methods have try-except blocks with proper error messages
2. ✅ `process_chunk()` validates input and handles GPU errors
3. ✅ `set_speaker_embedding()` validates embedding format
4. ✅ Pipeline never crashes - always returns audio (passthrough if conversion fails)
5. ✅ All errors logged with context for debugging
6. ✅ Unit tests achieve 100% coverage of error paths
7. ✅ Integration tests verify graceful degradation

## Out of Scope

- Error handling in ContentVec/RMVPE/HiFiGAN (handle in separate track)
- Performance optimization
- UI error messaging (frontend responsibility)

## Implementation Notes

**Error Handling Philosophy for Realtime:**
- **Never crash** - Always return audio (passthrough if necessary)
- **Log everything** - Errors, warnings, and recovery actions
- **Fail gracefully** - Degrade to passthrough rather than silence
- **Fast failure** - Detect errors early (input validation)

**CUDA OOM Strategy:**
```python
def _handle_cuda_oom(component: str) -> None:
    """Standard CUDA OOM handler."""
    logger.error(f"GPU OOM in {component}")
    torch.cuda.empty_cache()
    # Log GPU memory state
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.error(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

## Verification Checklist

Before marking complete:

- [ ] `realtime_pipeline.py` has error handling in all 4 init methods
- [ ] `process_chunk()` validates input and catches GPU errors
- [ ] `set_speaker_embedding()` validates embedding
- [ ] `tests/test_realtime_pipeline_error_handling.py` exists with 100% error path coverage
- [ ] All error messages include component name and context
- [ ] Graceful degradation verified in integration tests
- [ ] No crashes during GPU OOM simulation
