# AutoVoice Context Restoration - 2026-01-23 (Session 2)

## What You Were Doing
Planning the REAL implementation of singing voice conversion (no fallbacks). In plan mode.

## CRITICAL USER REQUIREMENTS
1. Train voice model from target person's SINGING recordings (any songs they can sing)
2. At inference: take any song by any artist, extract their performance (content + pitch + expression), re-synthesize with target person's trained voice
3. **NO STFT FALLBACKS** - user explicitly rejected fallback methodology
4. Must be a REAL working implementation using actual ML models
5. Use academic research MCP servers to inform proper implementation
6. Result: target person sounds like they have the original artist's singing talent

## Plan File
Active plan at: `/home/kp/.claude/plans/zazzy-twirling-squirrel.md`
- Contains 8 phases of implementation
- User rejected the plan because it included STFT fallback methodology
- Need to revise: remove ALL fallback patterns, implement real model pipeline only

## Research Completed (Key Papers)

### 1. "Self-Supervised Representations for SVC" (arxiv 2303.12197)
- **Best approach found**: ASR fine-tuned Wav2Vec2.0/HuBERT → HiFi-GAN vocoder
- F0 harmonic generation with PBTC (parallel bank of transposed convolutions)
- Circumvents disentanglement training
- Can make a spoken voice sing
- Simple f0 shifting during inference helps retain singer identity

### 2. "Unified Voice and Accent Conversion" (arxiv 2412.08312)
- HuBERT encoder + HiFi-GAN decoder + f0 features + singer embeddings
- Encoder-decoder architecture preserves pitch/tone and vocal identity

### 3. RVC (Context7: /rvc-project/retrieval-based-voice-conversion-webui)
- HuBERT extracts 256-dim features, saved as .npy
- Pitch via harvest/parselmouth/pyworld
- Feature retrieval (FAISS index) prevents voice leakage
- SynthesizerTrn model (VITS-based)

### 4. GPT-SoVITS (Context7: /rvc-boss/gpt-sovits)
- Two-stage: GPT semantic → SoVITS acoustic
- V3 uses BigVGAN v2 24kHz vocoder
- Few-shot capable (1 min of audio)
- Pretrained models: s1v3.ckpt, s2Gv3.pth

### 5. LDM-SVC (arxiv 2406.05325)
- Latent diffusion in VITS latent space (trained on So-VITS-SVC)
- Singer guidance (classifier-free) suppresses source timbre

### 6. FreeSVC (ICASSP 2025)
- SPIN (Speaker-invariant Clustering) for content representation
- ECAPA2 speaker encoder
- Trainable language embeddings for multilingual

## Confirmed Architecture Pattern
```
TRAINING (per-speaker, from their singing recordings):
  singing_audio → HuBERT → content_features [B, T, 256]
  singing_audio → pyin/harvest → F0 [B, T]
  singing_audio → mel_stats → speaker_embedding [256] (fixed per speaker)
  singing_audio → STFT → spectrogram [B, 513, T] (posterior encoder input)
  Model learns: (content, F0, speaker, spec) → reconstruct mel-spectrogram

INFERENCE (converting any song to target voice):
  source_vocals → HuBERT → content_features (WHAT is being sung)
  source_vocals → pyin → F0 (HOW it's sung - original artist's melody)
  target_speaker_embedding (WHO should sing - from training)
  SoVitsSvc.infer(content, pitch, speaker) → mel_pred [B, 80, T]
  HiFiGAN.synthesize(mel_pred) → audio waveform
  Resample to match input length
```

## Current Code State
- 185 tests, 100% pass (structural only)
- 20/20 beads closed, 2 commits on main
- Models EXIST in src/auto_voice/models/ but are NOT wired into pipeline
- Pipeline uses placeholder STFT (to be REMOVED, not kept as fallback)
- Training uses random tensors (to be replaced with real encoder outputs)
- VoiceCloner generates random noise (to be replaced with mel-based embeddings)

## Environment
```
Conda env: autovoice-thor
Python: 3.12.12
PyTorch: 2.11.0.dev20260113+cu130
CUDA: 13.0 (V13.0.48)
Device: NVIDIA Thor (SM 11.0, Blackwell, aarch64)
JetPack: 7.2 (R38.4.0)
```

## Key Files to Modify
| File | What |
|------|------|
| `src/auto_voice/inference/model_manager.py` | CREATE - frame-aligned inference |
| `src/auto_voice/inference/singing_conversion_pipeline.py` | REPLACE _convert_voice() with real model |
| `src/auto_voice/inference/realtime_voice_conversion_pipeline.py` | REPLACE _apply_conversion() |
| `src/auto_voice/inference/voice_cloner.py` | Mel-based embeddings, multi-file avg |
| `src/auto_voice/training/trainer.py` | Real encoder outputs in training loop |
| `src/auto_voice/models/encoder.py` | ContentEncoder (HuBERT), PitchEncoder (LSTM) |
| `src/auto_voice/models/vocoder.py` | HiFiGAN - already complete |
| `src/auto_voice/models/so_vits_svc.py` | SoVitsSvc - already complete |

## Resume Instructions
1. Read this file and the plan at `/home/kp/.claude/plans/zazzy-twirling-squirrel.md`
2. Query Cipher memory: `mcp__cipher__ask_cipher` about "autovoice singing voice conversion"
3. Continue research using Context7:
   - Query `/rvc-project/retrieval-based-voice-conversion-webui` for SynthesizerTrn model details
   - Query `/rvc-boss/gpt-sovits` for training pipeline details
4. Revise plan to REMOVE all fallback methodology
5. The implementation should ERROR if models aren't available, not fall back
6. Implement the real pipeline, test thoroughly, iterate until working

## Research Still Needed
- How RVC's SynthesizerTrn model handles the content→pitch→speaker→mel flow internally
- Exact HuBERT feature extraction dimensions and preprocessing
- How to properly compute spectrogram for PosteriorEncoder input (n_fft, hop, matching mel frames)
- F0 normalization scheme (log scale? Hz? semitones?)
- Training hyperparameters that work for ~10-30 min of singing data
- BigVGAN vs HiFi-GAN for vocoder quality on Jetson Thor
