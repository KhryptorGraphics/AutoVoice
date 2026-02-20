# AutoVoice PRD: Singing Voice Conversion System

## Vision
A production-grade singing voice conversion system that takes any song and re-synthesizes
it in a target person's voice, preserving the original artist's melody, timing, and
expression while replacing their timbre with the target's vocal characteristics.

## Use Cases

### Primary: Voice Cloning for Singing
1. **Training**: Target person provides 10-30 minutes of singing recordings (any songs)
2. **Inference**: System receives any song → separates vocals → extracts content+pitch → re-synthesizes with target voice
3. **Result**: Song sounds like the target person singing with the original artist's skill

### Secondary: Real-time Voice Conversion
1. User sings into microphone
2. System converts voice in real-time (<50ms latency)
3. Output sounds like target speaker singing

## Technical Architecture

### Pipeline: Training
```
target_singing_audio
  → HuBERT/ContentVec → content_features [B, T, 256]  (speaker-agnostic phonetics)
  → pyin → F0 → mel_quantize → PitchEmbedding [B, T, 256]  (melody)
  → mel_statistics → speaker_embedding [256]  (fixed identity)
  → STFT → spectrogram [B, 513, T]  (posterior encoder input)

SoVitsSvc learns: (content, pitch, speaker, spec) → reconstruct mel
Loss: reconstruction + KL + flow + SSIM
```

### Pipeline: Inference
```
source_song
  → Demucs → vocals + instrumental  (separation)
  → HuBERT/ContentVec → content [B, T, 256]  (what is sung)
  → pyin → F0 → PitchEmbedding [B, T, 256]  (how it's sung)
  → target_speaker_embedding [256]  (who should sing)

ModelManager.infer():
  content + pitch → frame_align → SoVitsSvc.infer() → mel [B, 80, T]
  mel → BigVGAN/HiFiGAN → audio [B, T_audio]
  audio → resample to input length → normalize

Output: converted_vocals + original_instrumental → mixed result
```

## Models (Research-Backed)

### Content Encoder
- **Base**: HuBERT-Soft (arxiv:2106.07447) - self-supervised speech units
- **Upgrade**: ContentVec (arxiv:2204.09224) - speaker-disentangled features
- **Future**: Multi-modal fusion (Whisper + ContentVec + MERT)
- Paper: "ContentVec: Improved Self-Supervised Speech Representation by Disentangling Speakers"

### Pitch Encoder
- **Base**: F0 extraction (pyin) → LSTM → [B, T, 256]
- **Upgrade**: Mel-scale quantization (256 bins, 50-1100 Hz) → Embedding lookup + UV flag
- Based on Amphion's approach: perceptually meaningful pitch representation
- Paper: Amphion (arxiv:2312.09911)

### Decoder (Voice Model)
- **Base**: SoVitsSvc (VAE + Normalizing Flow)
  - Posterior encoder: WaveNet-style dilated convolutions
  - Prior: Gaussian with learned mean/variance
  - Flow decoder: 4 affine coupling layers
- **Future**: Diffusion decoder (BiDilConv, 20 blocks) + Consistency distillation
- Papers: VITS (arxiv:2106.06103), CoMoSVC (arxiv:2401.01792), Consistency Models (arxiv:2303.01469)

### Vocoder
- **Base**: HiFiGAN (arxiv:2010.05646) - multi-period/multi-scale discriminators
- **Upgrade**: BigVGAN (arxiv:2206.04658) - periodic activations, anti-aliased, 112M params
- Key advantage: BigVGAN generalizes to singing without retraining

### Speaker Embedding
- **Current**: Mel-statistics (mean+std of 128 mel bands = 256-dim, L2-normalized)
- **Future**: ECAPA-TDNN (arxiv:2005.07143) for more discriminative embeddings
- Deterministic: same audio → same embedding every time

### Vocal Separation
- **Target**: Demucs v4 or similar
- Separates: vocals, drums, bass, other
- Quality: SDR > 7dB for vocals

## Quality Metrics
1. **Pitch RMSE**: Measures pitch preservation (lower = better)
2. **Speaker Similarity**: Cosine similarity of output vs target embedding
3. **STOI**: Short-Time Objective Intelligibility
4. **PESQ**: Perceptual Evaluation of Speech Quality
5. **MCD**: Mel Cepstral Distortion (lower = better)

## Performance Requirements
- **Training**: 3-5 hours for 30 min of audio on Thor GPU
- **Inference (static)**: Process 3-min song in <30 seconds
- **Inference (realtime)**: <50ms latency per chunk (4096 samples)
- **Memory**: <8GB GPU memory for inference

## Platform Constraints
- NVIDIA Jetson Thor (Blackwell architecture)
- CUDA 13.0, SM 11.0 (sm_110)
- aarch64 architecture
- JetPack 7.2 (R38.4.0)
- All packages must build from source for aarch64/CUDA 13.0

## API Design
```
POST /api/v1/voice/clone
  Input: audio files (WAV/FLAC/MP3, 10-30 min total)
  Output: { profile_id, speaker_embedding, quality_score }

POST /api/v1/convert/song
  Input: { song_path, target_profile_id, preset, options }
  Output: { job_id }  (async processing)

GET /api/v1/convert/status/{job_id}
  Output: { status, progress, eta, result_path }

WebSocket: /ws/realtime
  Input: audio chunks (4096 samples, 22050 Hz)
  Output: converted audio chunks
```

## Development Phases

### Phase 1: Foundation (COMPLETE)
- [x] So-VITS-SVC architecture
- [x] No-fallback enforcement
- [x] ModelManager inference orchestrator
- [x] Real encoder features in training
- [x] Mel-statistics speaker embedding
- [x] 231 tests passing

### Phase 2: Stability (IN PROGRESS)
- [ ] Fix all test failures (>95% pass rate)
- [ ] Download pretrained weights
- [ ] SSIM loss addition
- [ ] Mel-quantized F0

### Phase 3: Quality
- [ ] ContentVec content features
- [ ] BigVGAN vocoder
- [ ] Data augmentation pipeline
- [ ] Vocal separation (Demucs)

### Phase 4: Speed
- [ ] Conformer encoder
- [ ] Consistency distillation (1-step inference)
- [ ] TensorRT optimization
- [ ] CUDA kernel optimization

### Phase 5: Production
- [ ] Full API integration
- [ ] Frontend connected
- [ ] Docker deployment
- [ ] Monitoring/metrics
- [ ] Documentation
