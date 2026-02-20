# HuggingFace: Consistency Models for Voice Conversion

## Research Source
- HuggingFace paper_search queries across 5 topic areas
- Focus: consistency distillation, one-step inference, flow matching for audio

---

## Key Papers

### Direct Voice Conversion

| Paper | arXiv | Year | Contribution |
|-------|-------|------|--------------|
| **CoMoSVC** | 2401.01792 | 2024 | Consistency model distillation for singing voice conversion, one-step sampling |
| **CoMoSpeech** | 2305.06908 | 2023 | Consistency model for speech + singing synthesis, 150x faster than real-time |
| **FastVoiceGrad** | 2409.02245 | 2024 | One-step diffusion VC via adversarial conditional diffusion distillation |

### Consistency Training Methods

| Paper | arXiv | Year | Contribution |
|-------|-------|------|--------------|
| **FlashSpeech** | 2404.14700 | 2024 | Adversarial consistency training from scratch (no teacher), 20x faster than SOTA |
| **CM-TTS** | 2404.00569 | 2024 | Consistency models with weighted samplers for real-time TTS |
| **RapFlow-TTS** | 2506.16741 | 2025 | Velocity consistency constraints in flow matching for few-step TTS |

### Audio Generation / Vocoding

| Paper | arXiv | Year | Contribution |
|-------|-------|------|--------------|
| **EDMSound** | 2311.08667 | 2023 | EDM preconditioning framework for spectrogram-domain audio synthesis |
| **FlashAudio** | 2410.12266 | 2024 | Rectified flows with bifocal samplers, 400x faster than real-time |
| **PeriodWave-Turbo** | 2408.08019 | 2024 | Adversarial flow matching optimization, 2-4 step waveform generation |

---

## Key Techniques for AutoVoice Implementation

### 1. Consistency Distillation (CoMoSVC / CoMoSpeech)
- Train multi-step diffusion teacher on mel spectrograms
- Distill to student with self-consistency property: `f(x_t, t) = f(x_s, s)` for any points on same trajectory
- Student generates in single forward pass
- Loss: consistency loss + mel reconstruction + adversarial (optional)
- **Result**: 150x real-time on A100, comparable quality to 30-step teacher

### 2. Adversarial Consistency Training (FlashSpeech)
- Train consistency model from scratch without pre-trained teacher
- Uses adversarial loss (GAN discriminator) to enforce output quality
- Eliminates two-phase training (teacher → student)
- **Result**: 20x faster than diffusion SOTA, single training phase

### 3. EDM Preconditioning (EDMSound)
- Karras et al. (2022) noise scheduling applied to audio
- Separates noise schedule design from network parameterization
- Better training stability for spectrogram-domain generation
- Applicable to both training and distillation phases

### 4. Rectified Flows (FlashAudio)
- Straight ODE trajectories → fewer steps needed
- Bifocal samplers: coarse trajectory + refined endpoints
- **Result**: 400x real-time for audio generation

### 5. Adversarial Flow Matching (PeriodWave-Turbo)
- Combine flow matching with GAN discriminator
- 2-4 step generation (vs 30-100 for pure diffusion)
- Period-aware discriminator for waveform quality

### 6. Velocity Consistency (RapFlow-TTS)
- Enforce consistent velocity fields along ODE trajectory
- Enables few-step (2-4) generation from flow matching models
- Compatible with pre-trained flow matching teachers

---

## Recommended Approach for AV-010

Based on this research, the optimal strategy for AutoVoice consistency distillation:

1. **Phase 1 - Teacher**: Train a small diffusion model (4-layer BiDilConv, ~2M params) on mel spectrograms conditioned on content+pitch+speaker embeddings. Use EDM preconditioning for stable training.

2. **Phase 2 - Distillation**: Apply CoMoSVC-style consistency distillation:
   - Self-consistency loss on trajectory pairs
   - Mel reconstruction auxiliary loss
   - Optional: adversarial loss (FlashSpeech-style) for quality

3. **Phase 3 - Optimization**: Export student to ONNX → TensorRT FP16. Target: single forward pass <15ms on Jetson Thor.

### Architecture (from CoMoSVC):
```
Input: content_embed [B, T, 256] + pitch_embed [B, T, 256] + speaker [B, 256]
       + noise_level (scalar)
→ Concat + Linear → [B, T, hidden]
→ BiDilConv blocks (4 layers, dilation [1,2,4,8])
→ Linear → mel_output [B, T, n_mels]
```

### Loss Function:
```python
# Consistency loss
L_consistency = ||f_theta(x_t, t) - f_theta_ema(x_s, s)||^2

# Reconstruction
L_mel = ||mel_pred - mel_target||_1

# Combined
L_total = L_consistency + lambda_mel * L_mel
```

---

## HuggingFace Model Hub Status
- No dedicated audio/speech consistency model checkpoints found on HF
- Image-domain consistency models available (diffusers library)
- CoMoSVC/CoMoSpeech must be trained from source repos
- Reference implementations: [zhenye234/CoMoSpeech](https://github.com/zhenye234/CoMoSpeech)
