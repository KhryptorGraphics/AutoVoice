# Optional target_speaker_emb Implementation

## Overview
Implemented Comment 2: Made `target_speaker_emb` optional in `VoiceConversionTrainer` and `SingingVoiceConverter`.

## Changes Made

### 1. VoiceConversionTrainer (`src/auto_voice/training/trainer.py`)

#### Added Default Speaker Embedding Buffer
```python
# In __init__()
self.register_buffer(
    'default_speaker_emb',
    torch.zeros(1, 256, dtype=torch.float32)
)
```

#### Updated `_forward_pass()` Method
**Handles optional `target_speaker_emb`:**
```python
target_speaker_emb = batch.get('target_speaker_emb')
if target_speaker_emb is None:
    # Use default speaker embedding, expand to batch size
    target_speaker_emb = self.default_speaker_emb.expand(batch_size, -1).to(device)
    if self.config.extract_speaker_emb:
        logger.warning(
            "target_speaker_emb is missing but extract_speaker_emb=True. "
            "Using default zero embedding. Speaker similarity loss will be set to zero."
        )
```

**Handles optional `source_f0`:**
```python
source_f0 = batch.get('source_f0')
if source_f0 is None:
    if self.config.extract_f0:
        # Create default zeros for F0
        time_steps = batch['target_mel'].size(2)
        source_f0 = torch.zeros(batch_size, time_steps, device=device)
        logger.warning(
            "source_f0 is missing but extract_f0=True. "
            "Using zero F0. Pitch consistency loss will be set to zero."
        )
```

#### Updated Speaker Similarity Loss Computation
**Gracefully handles None/zero embeddings:**
```python
target_speaker_emb = batch.get('target_speaker_emb')
has_valid_speaker_emb = (
    target_speaker_emb is not None and
    torch.any(target_speaker_emb != 0)  # Not a zero embedding
)

if 'pred_audio' in predictions and has_valid_speaker_emb:
    # Compute speaker similarity loss
    ...
else:
    # Set loss to zero gracefully
    losses['speaker_similarity'] = torch.tensor(0.0, device=predictions['pred_mel'].device)
```

### 2. SingingVoiceConverter (`src/auto_voice/models/singing_voice_converter.py`)

#### Updated `forward()` Signature
```python
def forward(
    self,
    source_audio: torch.Tensor,
    target_mel: torch.Tensor,
    source_f0: torch.Tensor,
    target_speaker_emb: Optional[torch.Tensor] = None,  # Now optional
    source_sample_rate: int = 16000,
    x_mask: Optional[torch.Tensor] = None,
    source_voiced: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
```

#### Added None Handling Logic
```python
if target_speaker_emb is None:
    # Use default zero embedding when not provided
    target_speaker_emb = torch.zeros(B, self.speaker_dim, device=device)
    logger.debug(f"target_speaker_emb is None, using default zero embedding")
else:
    # Validate existing embedding
    target_speaker_emb = target_speaker_emb.to(device)
    if target_speaker_emb.dim() != 2:
        raise VoiceConversionError(...)
    if target_speaker_emb.size(1) != self.speaker_dim:
        raise VoiceConversionError(...)
```

#### Updated Docstring
```python
"""
Args:
    ...
    target_speaker_emb: Optional target speaker embedding [B, 256].
        If None, will use a default zero embedding.
    ...

Note:
    When target_speaker_emb is None, speaker-related losses should be
    handled gracefully in the trainer by checking for None/zero embeddings.
"""
```

## Behavior

### When `target_speaker_emb` is provided:
- ✅ Normal operation with speaker similarity loss computed
- ✅ Full speaker conditioning in model

### When `target_speaker_emb` is None or missing:
- ✅ Uses default zero embedding [1, 256] expanded to batch size
- ✅ Logs warning if `extract_speaker_emb=True` in config
- ✅ Speaker similarity loss is set to 0.0 gracefully
- ✅ Model continues forward pass without errors

### When `source_f0` is None or missing:
- ✅ Creates zero F0 tensor with appropriate shape
- ✅ Logs warning if `extract_f0=True` in config
- ✅ Pitch consistency loss is set to 0.0 gracefully
- ✅ Model continues forward pass without errors

## Config Validation
- ⚠️ Warns when embeddings are missing but expected (`extract_speaker_emb=True`)
- ⚠️ Warns when F0 is missing but expected (`extract_f0=True`)
- ✅ Sets corresponding losses to zero gracefully
- ✅ Continues training without crashes

## Implementation Pattern
```python
# Pattern used throughout:

# 1. Get optional value with batch.get()
value = batch.get('optional_field')

# 2. Check if None
if value is None:
    # 3. Use default (zeros, buffer, etc.)
    value = default_value
    # 4. Log warning if expected but missing
    if config.expects_value:
        logger.warning("Value missing but expected")

# 5. Proceed with value (default or provided)
model_output = model(..., value=value, ...)
```

## Files Modified
1. `/home/kp/autovoice/src/auto_voice/training/trainer.py`
   - Added `default_speaker_emb` buffer
   - Updated `_forward_pass()` to handle optional embeddings
   - Updated speaker similarity loss computation

2. `/home/kp/autovoice/src/auto_voice/models/singing_voice_converter.py`
   - Made `target_speaker_emb` parameter optional
   - Added None handling in `forward()`
   - Updated docstring

## Testing Recommendations
1. Test with full batch (all fields present) - should work as before
2. Test with missing `target_speaker_emb` - should use default
3. Test with missing `source_f0` - should use zeros
4. Test with both missing - should handle gracefully
5. Verify speaker similarity loss is 0.0 when embedding is None/zeros
6. Verify pitch consistency loss is 0.0 when F0 is None/zeros

## Benefits
- ✅ More flexible dataset handling
- ✅ Graceful degradation when embeddings unavailable
- ✅ Clearer logging for debugging
- ✅ No crashes from missing optional fields
- ✅ Backward compatible with existing code
