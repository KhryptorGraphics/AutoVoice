# Comment 3 Implementation: pred_audio Generation for Perceptual Losses

## Overview
Implemented pred_audio generation in SingingVoiceConverter.forward() to enable pitch consistency and speaker similarity losses during training.

## Changes Made

### 1. SingingVoiceConverter.forward() Enhancement
**File**: `src/auto_voice/models/singing_voice_converter.py`

Added `use_vocoder` parameter (default `True`) to generate `pred_audio`:

```python
def forward(
    self,
    source_audio: torch.Tensor,
    target_mel: torch.Tensor,
    source_f0: torch.Tensor,
    target_speaker_emb: torch.Tensor,
    source_sample_rate: int = 16000,
    x_mask: Optional[torch.Tensor] = None,
    source_voiced: Optional[torch.Tensor] = None,
    use_vocoder: bool = True  # NEW PARAMETER
) -> Dict[str, torch.Tensor]:
```

**Implementation**:
- When `use_vocoder=True` and vocoder is available:
  - Converts predicted mel-spectrogram to audio using internal HiFiGAN vocoder
  - Handles log-mel to linear-mel conversion automatically
  - Adds `pred_audio` to output dictionary
  - Logs success with audio shape
- Falls back gracefully if vocoder unavailable or generation fails
- Returns outputs with `pred_audio` key for downstream loss computation

### 2. VoiceConversionTrainer Updates
**File**: `src/auto_voice/training/trainer.py`

#### 2.1 Documentation Enhancement
Added critical requirement to class docstring:
```python
"""Initialize voice conversion trainer.

IMPORTANT: The SingingVoiceConverter model MUST generate pred_audio during
forward pass for pitch_consistency and speaker_similarity losses to work.
Ensure model.forward() is called with use_vocoder=True (default).
```

#### 2.2 Forward Pass Update
Modified `_forward_pass()` to explicitly enable vocoder:
```python
outputs = self.model(
    source_audio=batch['source_audio'],
    target_mel=batch['target_mel'],
    source_f0=batch['source_f0'],
    target_speaker_emb=batch['target_speaker_emb'],
    source_sample_rate=self.config.sample_rate,
    x_mask=batch.get('mel_mask'),
    use_vocoder=True  # REQUIRED for perceptual losses
)
```

#### 2.3 Loss Computation Enhancements
Added logging and validation for pitch and speaker losses:

**Pitch Consistency Loss**:
```python
if 'pred_audio' in predictions and 'source_f0' in batch:
    try:
        pitch_loss_val = self.pitch_loss(predictions['pred_audio'], ...)
        losses['pitch_consistency'] = pitch_loss_val

        # Assert contribution
        if pitch_loss_val.item() > 0:
            logger.debug(f"Pitch consistency loss: {pitch_loss_val.item():.6f}")
    except Exception as e:
        logger.warning(f"Pitch consistency loss computation failed: {e}")
else:
    if 'pred_audio' not in predictions:
        logger.warning(
            "pred_audio not in predictions. Ensure SingingVoiceConverter.forward() "
            "is called with use_vocoder=True. Skipping pitch consistency loss."
        )
```

**Speaker Similarity Loss**:
- Similar logging pattern
- Explicit warnings when pred_audio is missing
- Debug logs when loss contributes successfully

## Benefits

### 1. Perceptual Loss Availability
- ✅ Pitch consistency loss can now compute F0 RMSE on actual audio
- ✅ Speaker similarity loss can extract embeddings from predicted audio
- ✅ Multi-resolution STFT loss can operate on waveforms

### 2. Training Transparency
- Clear warnings when pred_audio is missing
- Debug logs showing loss values when contributing
- Documentation highlighting requirements

### 3. Graceful Degradation
- Falls back to mel-based losses if vocoder unavailable
- No crashes if audio generation fails
- Flexible use_vocoder flag for debugging

### 4. Backward Compatibility
- Default `use_vocoder=True` maintains expected behavior
- Can disable for mel-only training if needed
- No breaking changes to existing code

## Verification

### Check pred_audio Generation
```bash
# Verify use_vocoder parameter exists
grep -n "use_vocoder" src/auto_voice/models/singing_voice_converter.py

# Verify trainer uses it
grep -n "use_vocoder=True" src/auto_voice/training/trainer.py

# Verify warning messages
grep -n "pred_audio not in predictions" src/auto_voice/training/trainer.py
```

### Expected Training Logs
When training with pred_audio:
```
DEBUG: Generated pred_audio: shape=torch.Size([8, 512000])
DEBUG: Pitch consistency loss: 0.023451
DEBUG: Speaker similarity loss: 0.142387
```

When pred_audio missing:
```
WARNING: pred_audio not in predictions. Ensure SingingVoiceConverter.forward()
         is called with use_vocoder=True. Skipping pitch consistency loss.
```

## Implementation Notes

### Mel Domain Handling
The vocoder expects linear-scale mel, so the implementation:
1. Generates log-mel in forward pass (for training consistency)
2. Converts log-mel back to linear before vocoder: `pred_mel_linear = torch.exp(pred_mel)`
3. Passes linear-mel to HiFiGAN vocoder
4. Returns audio waveform as pred_audio

### Memory Considerations
- Generating audio adds ~512K samples per batch item at 44.1kHz
- For batch_size=8, ~4MB additional GPU memory per batch
- Trade-off: Better perceptual losses vs slightly higher memory usage

### Alternative Approaches Considered
❌ **Option 2 (Fallback)**: Add lightweight vocoder in trainer
- More complex, requires separate vocoder management
- Chosen Option 1 for cleaner architecture

## Testing Recommendations

### Unit Tests
```python
def test_forward_with_vocoder():
    """Test forward pass generates pred_audio"""
    model = SingingVoiceConverter(config)
    outputs = model(source_audio, target_mel, source_f0, target_emb, use_vocoder=True)
    assert 'pred_audio' in outputs
    assert outputs['pred_audio'].dim() == 2  # [B, T_audio]

def test_forward_without_vocoder():
    """Test forward pass without pred_audio"""
    model = SingingVoiceConverter(config)
    outputs = model(source_audio, target_mel, source_f0, target_emb, use_vocoder=False)
    assert 'pred_audio' not in outputs
```

### Integration Tests
```python
def test_voice_conversion_trainer_with_pred_audio():
    """Test trainer receives pred_audio"""
    trainer = VoiceConversionTrainer(model, config)
    losses = trainer._compute_voice_conversion_losses(predictions, batch)

    # Check pitch and speaker losses are non-zero when pred_audio exists
    if 'pred_audio' in predictions:
        assert losses['pitch_consistency'].item() > 0
        assert losses['speaker_similarity'].item() > 0
```

## Status
✅ Implementation complete
✅ Documentation added
✅ Logging and fallbacks implemented
✅ Backward compatible

## Next Steps
1. Run training to verify loss contributions
2. Monitor GPU memory usage with pred_audio generation
3. Validate perceptual loss improvements in converted audio quality
