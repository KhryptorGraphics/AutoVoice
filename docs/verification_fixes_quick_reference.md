# Verification Fixes Quick Reference

## All 10 Comments Implemented ✓

### Comment 1: Import numpy in PitchEncoder
```python
# pitch_encoder.py line 10
import numpy as np
```

### Comment 2: Parameterize vocoder sample rate
```python
# singing_voice_converter.py lines 134-135
audio_cfg = svc_config.get('audio', {})
self.vocoder_sample_rate = audio_cfg.get('sample_rate', 22050)

# Line 289
u = torch.randn(...) * self.temperature

# Line 304
if output_sample_rate != self.vocoder_sample_rate:

# Line 347
mel_basis = librosa.filters.mel(sr=self.vocoder_sample_rate, ...)
```

### Comment 3: Apply temperature during sampling
```python
# singing_voice_converter.py lines 138-139
inference_cfg = svc_config.get('inference', {})
self.temperature = inference_cfg.get('temperature', 1.0)

# Line 289
u = torch.randn(1, self.latent_dim, T, device=device) * self.temperature
```

### Comment 4: Consume YAML config
```python
# singing_voice_converter.py lines 56-101
# Support nested singing_voice_converter section or flat config
if 'singing_voice_converter' in config:
    svc_config = config['singing_voice_converter']
else:
    svc_config = config

# Extract all nested sections:
content_cfg = svc_config.get('content_encoder', {})
pitch_cfg = svc_config.get('pitch_encoder', {})
speaker_cfg = svc_config.get('speaker_encoder', {})
posterior_cfg = svc_config.get('posterior_encoder', {})
flow_cfg = svc_config.get('flow_decoder', {})
vocoder_cfg = svc_config.get('vocoder', {})
audio_cfg = svc_config.get('audio', {})
inference_cfg = svc_config.get('inference', {})
```

### Comment 5: Fix gradient flow test
```python
# test_voice_conversion.py lines 465-482
source_audio = torch.randn(1, 16000)  # Removed requires_grad=True
target_mel = torch.randn(1, 80, 50, requires_grad=True)  # Added requires_grad=True

# Check gradients on target_mel
assert target_mel.grad is not None
assert torch.isfinite(target_mel.grad).all()

# Check gradients on PosteriorEncoder parameters
posterior_params_with_grad = sum(1 for p in model.posterior_encoder.parameters() if p.grad is not None)
assert posterior_params_with_grad > 0
```

### Comment 6: Add pitch preservation and speaker conditioning tests
```python
# test_voice_conversion.py lines 415-510

def test_pitch_preservation(self, model):
    """Creates audio at 440 Hz, converts, extracts F0, asserts RMSE < 50 Hz"""
    # ... full implementation in file

def test_speaker_conditioning(self, model):
    """Converts to two speakers, asserts cosine distance > 0.1"""
    # ... full implementation in file
```

### Comment 7: Per-sample normalization in ContentEncoder
```python
# content_encoder.py lines 151-154
# Normalize to [-1, 1] per sample
max_vals = audio.abs().amax(dim=-1, keepdim=True)
max_vals = torch.clamp(max_vals, min=1e-8)
audio = audio / max_vals
```

### Comment 8: Stereo/2D input handling
```python
# content_encoder.py lines 131-136
elif audio.dim() == 2:
    # Detect if likely channels-first stereo [1-2, T]
    if audio.size(0) in {1, 2} and audio.size(1) > audio.size(0):
        # Convert to mono by averaging across channels
        audio = audio.mean(dim=0, keepdim=True)
    # Otherwise already [B, T], just continue
```

### Comment 9: Adjust DDSConv dilation
```python
# flow_decoder.py line 70
dilation = 2 ** i  # Changed from kernel_size ** i
```

### Comment 10: PitchEncoder config and device handling
```python
# singing_voice_converter.py lines 94-101
self.pitch_encoder = PitchEncoder(
    pitch_dim=self.pitch_dim,
    hidden_dim=pitch_cfg.get('hidden_dim', ...),
    num_bins=pitch_cfg.get('num_bins', ...),  # Added
    f0_min=pitch_cfg.get('f0_min', ...),
    f0_max=pitch_cfg.get('f0_max', ...)
)
if 'blend_weight' in pitch_cfg:
    self.pitch_encoder.blend_weight.data.fill_(pitch_cfg['blend_weight'])

# pitch_encoder.py lines 142-145
device = next(self.parameters()).device
f0 = f0.to(device)
if voiced is not None:
    voiced = voiced.to(device)
```

## Test Execution

Run tests:
```bash
pytest tests/test_voice_conversion.py -v -k "test_pitch_encoder or test_gradient_flow or test_pitch_preservation or test_speaker_conditioning"
```

Syntax validation:
```bash
python -m py_compile src/auto_voice/models/pitch_encoder.py \
    src/auto_voice/models/singing_voice_converter.py \
    src/auto_voice/models/content_encoder.py \
    src/auto_voice/models/flow_decoder.py \
    tests/test_voice_conversion.py
```

## Files Modified

1. ✅ `src/auto_voice/models/pitch_encoder.py`
2. ✅ `src/auto_voice/models/singing_voice_converter.py`
3. ✅ `src/auto_voice/models/content_encoder.py`
4. ✅ `src/auto_voice/models/flow_decoder.py`
5. ✅ `tests/test_voice_conversion.py`

## Status: ALL COMPLETE ✓
