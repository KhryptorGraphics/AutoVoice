# Amphion SVC Architecture Analysis

## Source: /home/kp/repo2/Amphion

## Key Architecture: CoMoSVC (Consistency Model SVC)

### Encoder: Conformer (6 layers)
- Input: Multi-modal features (Whisper 1024d + ContentVec 256d + MERT 256d)
- Architecture: Self-attention + FFN + Conv1D
- Relative position embeddings (window=4)
- Output: [B, T, 384] → projected to [B, T, n_mel]

### Decoder: EDM Diffusion + Consistency Distillation
- **Teacher**: BiDilConv (20 residual blocks, dilated convolutions)
  - FILM conditioning from encoder output
  - Time embeddings (128→384)
  - Karras noise schedule (σ_min=0.002, σ_max=80, ρ=7)
  - 40-50 sampling steps
- **Student**: Same architecture, trained with CTLoss_D
  - EMA teacher (μ=0.95)
  - 1-step inference

### Pitch Handling
```python
# Mel-scale quantization (50-1100 Hz → 256 bins)
f0_mel = 1127 * log(1 + f0/700)
f0_coarse = normalize_to_bins(f0_mel, 256)
# Lookup embedding: [256, 384]
pitch_emb = embedding_table[f0_coarse]  # [B, T, 384]
# Separate UV (unvoiced) embedding: [2, 384]
uv_emb = uv_embedding_table[uv_flag]   # [B, T, 384]
```

### Speaker Embedding
- Discrete lookup table: [512 speakers, 384 dims]
- Broadcast to sequence length
- Zero-shot: Reference encoder (not yet implemented)

### Content Feature Extraction
- **Whisper** (1024d, 20ms frames): Best for linguistic content
- **ContentVec** (256d, 20ms frames): Speaker-disentangled
- **MERT** (256d, 13.3ms frames): Music-specific features
- Resolution alignment via upsample + average pooling

### Loss Functions
1. **Prior Loss**: L2(encoder_output, mel) - reconstruction
2. **SSIM Loss**: Structural similarity (perceptual)
3. **EDM Loss**: Weighted MSE (teacher diffusion)
4. **CT Loss**: MSE(student, EMA_student) - consistency distillation

### Training Pipeline
1. Stage 1: Train encoder + diffusion teacher (~50 epochs)
   - Skip diffusion loss for first N steps (fast_steps)
   - LR: 4e-4, ReduceLROnPlateau (factor=0.8, patience=10)
   - Gradient clip: max_norm=1.0 per-layer
2. Stage 2: Distill student from frozen teacher
   - Only optimize student_denoise_fn
   - EMA update of teacher copy

### Data Preprocessing
- Sample rate: 24000 Hz
- Hop size: 256 (10.67ms frames)
- N_mel: 100 channels
- F0: ParselMouth/CREPE extraction
- Energy: Log-scale bucketing (256 bins)
- Data augmentation: pitch shift, formant shift, EQ, time stretch

### Inference
- Long audio: 10s segments with 1s overlap + crossfade
- Teacher: 50 EDM sampling steps
- Student: 1-step consistency sampling

## Key Differences from AutoVoice Current Implementation

1. **Multi-modal content** vs single HuBERT
2. **Mel-quantized F0** vs raw F0 → LSTM
3. **Diffusion decoder** vs VAE+Flow
4. **Conformer encoder** vs linear projection
5. **SSIM + prior + diffusion loss** vs reconstruction + KL + flow
6. **100 mel channels** vs 80 mel channels
7. **24kHz sample rate** vs 22.05kHz
8. **Two-stage training** vs single-stage
9. **Explicit loudness encoding** vs none
10. **Data augmentation pipeline** vs none

## Adoptable Improvements (ranked by effort/impact)

### Low Effort, High Impact
- Mel-quantized F0 (replace LSTM with embedding table)
- SSIM loss addition (few lines of code)
- Energy/loudness conditioning
- Gradient clipping per-layer

### Medium Effort, High Impact
- ContentVec features (pretrained model, swap for HuBERT)
- BigVGAN vocoder (pretrained, replace HiFiGAN)
- Data augmentation pipeline
- 100 mel channels + 24kHz

### High Effort, High Impact
- Conformer encoder (replace linear projection)
- Diffusion decoder (replace VAE+Flow)
- Consistency distillation (requires trained teacher)
- Multi-modal content fusion
