# Adversarial Loss Implementation for Voice Conversion

## Overview

This document describes the implementation of adversarial loss and discriminator for the VoiceConversionTrainer, following Comment 1 requirements.

## Implementation Summary

### 1. VoiceDiscriminator Module (`src/auto_voice/models/discriminator.py`)

**Architecture:**
- Multi-scale discriminator with 3 scales: 1x, 2x, 4x downsampling
- Each scale: Conv1d blocks with LeakyReLU(0.2) and optional spectral normalization
- Input: Audio waveform [batch, time] or [batch, 1, time]
- Output: List of logits from each scale

**Loss Functions:**
- `hinge_discriminator_loss()`: D_loss = ReLU(1 - D(real)) + ReLU(1 + D(fake))
- `hinge_generator_loss()`: G_loss = -mean(D(fake))
- `feature_matching_loss()`: Optional L1 loss between discriminator features

### 2. TrainingConfig Updates

**New Parameter:**
```python
vc_loss_weights = {
    'mel_reconstruction': 45.0,
    'kl_divergence': 1.0,
    'pitch_consistency': 10.0,
    'speaker_similarity': 5.0,
    'flow_likelihood': 1.0,
    'stft': 2.5,
    'adversarial': 0.1  # NEW - Adversarial loss weight
}
```

### 3. VoiceConversionTrainer Modifications

**New Components:**
- `_setup_discriminator()`: Initializes VoiceDiscriminator and separate AdamW optimizer
- `discriminator`: Multi-scale discriminator module
- `discriminator_optimizer`: Separate optimizer for discriminator (same LR as generator)

**Updated Methods:**

**`_compute_voice_conversion_losses()`:**
- Added adversarial loss computation (step 7)
- Forward through discriminator with `pred_audio`
- Compute generator adversarial loss using hinge loss
- Weight by `vc_loss_weights['adversarial']`

**`train_epoch()` - Two-Step GAN Training:**

**Step 1: Update Discriminator**
```python
# Zero discriminator gradients
discriminator_optimizer.zero_grad()

# Forward pass with detached generator outputs
predictions = model(...)
real_audio = batch['target_audio']
fake_audio = predictions['pred_audio'].detach()  # Detach!

# Discriminator forward
real_logits_list, _ = discriminator(real_audio)
fake_logits_list, _ = discriminator(fake_audio)

# Compute discriminator loss (hinge)
disc_loss = hinge_discriminator_loss(real_logits_list, fake_logits_list)

# Backward + optimizer step
disc_loss.backward()
discriminator_optimizer.step()
```

**Step 2: Update Generator**
```python
# Zero generator gradients
optimizer.zero_grad()

# Forward pass (no detaching)
predictions = model(...)

# Compute all losses including adversarial
losses = _compute_voice_conversion_losses(predictions, batch)

# Backward + optimizer step
losses['total'].backward()
optimizer.step()
```

**Key Features:**
- Respects mixed precision training (autocast)
- Respects gradient accumulation
- Gradient clipping applied to both optimizers
- Discriminator only updates if `adversarial` weight > 0

### 4. Examples Update

**`examples/train_voice_conversion.py`:**
- Adversarial weight loaded from config `losses.adversarial` (default: 0.1)
- Trainer internally creates discriminator
- No manual discriminator construction needed

## Usage

### Training Configuration

```yaml
# config/voice_conversion_training.yaml
losses:
  mel_reconstruction: 45.0
  kl_divergence: 1.0
  pitch_consistency: 10.0
  speaker_similarity: 5.0
  flow_likelihood: 1.0
  stft: 2.5
  adversarial: 0.1  # Enable adversarial training
```

### Training Command

```bash
python examples/train_voice_conversion.py --config config/voice_conversion_training.yaml
```

## Requirements

**Pred_audio Availability:**
- Adversarial loss requires `pred_audio` in model predictions
- SingingVoiceConverter.forward() should include vocoder synthesis or Griffin-Lim
- If `pred_audio` not available, adversarial loss is skipped (set to 0)

## Loss Scheduling (Optional Future Enhancement)

For better training stability, consider:
1. Start with adversarial weight = 0 for first N epochs (warm-up)
2. Gradually increase adversarial weight (ramp-up schedule)
3. Example: `adversarial_weight = min(0.1, epoch / 50 * 0.1)`

## Testing

The implementation can be tested with:
```bash
# Synthetic dataset demo
python examples/train_voice_conversion.py --use-synthetic-data --epochs 5
```

## Notes

- Discriminator operates on raw audio waveforms (pred_audio), not mel spectrograms
- Two separate optimizers ensure clean gradient separation
- Hinge loss provides more stable training than BCE
- Optional spectral normalization can be enabled for more stable discriminator training
