# Singing Voice Conversion Research Summary

## 🎯 Objective
Build a professional singing voice conversion system that replaces one person's singing with another artist's voice while **perfectly preserving** the original artist's pitch accuracy, vibrato, expression, and singing talent.

## 🔬 State-of-the-Art Methods (2024-2025)

### 1. RVC (Retrieval-based Voice Conversion)
- **Repository**: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **Stars**: 32.9k+ (Most popular open-source SVC)
- **Architecture**: VITS-based with Faiss retrieval
- **Key Features**:
  - Top-1 retrieval to prevent timbre leakage
  - Fast training on consumer GPUs
  - Works with <10 minutes of training data
  - RMVPE pitch extraction (InterSpeech 2023)
  - Real-time conversion (90ms latency with ASIO)
  - WebUI with training and inference

### 2. So-VITS-SVC 5.0
- **Repository**: https://github.com/svc-develop-team/so-vits-svc
- **Architecture**: Soft-VC + VITS
- **Key Components**:
  - **Content Encoder**: HuBERT-Soft (speaker-independent features)
  - **Pitch Encoder**: F0 extraction with CREPE/DIO
  - **Speaker Encoder**: Resemblyzer embeddings
  - **Decoder**: Flow-based variational decoder
  - **Vocoder**: HiFi-GAN for high-quality synthesis

### 3. Key Technical Papers

#### CREPE (2018)
- **Paper**: "CREPE: A Convolutional Representation for Pitch Estimation"
- **Accuracy**: <10 cents error (sub-semitone)
- **Method**: CNN trained on 1000+ hours of labeled pitch data
- **Use Case**: Gold standard for singing pitch extraction

#### RMVPE (2023)
- **Paper**: "RMVPE: Robust Model for Vocal Pitch Estimation" (InterSpeech 2023)
- **Advantages**: Faster than CREPE, more robust to noise
- **Performance**: Comparable accuracy to CREPE "full" model
- **Implementation**: Available in RVC project

#### HuBERT (2021)
- **Paper**: "HuBERT: Self-Supervised Speech Representation Learning"
- **Purpose**: Extract speaker-independent content features
- **Output**: 256-dimensional vectors at 50 Hz
- **Key Property**: No pitch information (perfect for SVC)

## 🏗️ Recommended Architecture

### Pipeline Overview
```
Input Song
    ↓
[1] Vocal Separation (Demucs/UVR5)
    ↓
[2] Pitch Extraction (CREPE/RMVPE) → F0 contour + vibrato
    ↓
[3] Content Encoding (HuBERT-Soft) → Speaker-independent features
    ↓
[4] Speaker Embedding (Resemblyzer) → Target voice timbre
    ↓
[5] Voice Conversion (So-VITS-SVC) → Combine content + pitch + speaker
    ↓
[6] Vocoder (HiFi-GAN) → High-quality audio synthesis
    ↓
[7] Audio Mixing → Combine with instrumental
    ↓
Output Song (Converted)
```

### Critical Design Decisions

#### 1. Pitch Preservation Strategy
**Problem**: Most voice conversion systems alter pitch accuracy  
**Solution**: Extract and preserve original F0 contour
```python
# Extract original pitch with high accuracy
f0_original = crepe.predict(
    vocals,
    model='full',  # Highest accuracy
    viterbi=True,  # Smooth pitch tracking
    step_size=10   # 10ms frames
)

# Use original F0 in conversion (do NOT re-estimate)
converted = svc_model.convert(
    content=hubert_features,
    f0=f0_original,  # PRESERVE ORIGINAL
    speaker_emb=target_speaker
)
```

#### 2. Vibrato Transfer
**Problem**: Vibrato patterns get smoothed out during conversion  
**Solution**: Detect and explicitly transfer vibrato
```python
# Detect vibrato (4-8 Hz modulation)
vibrato_rate, vibrato_depth = detect_vibrato(f0_original)

# Apply to converted audio
f0_with_vibrato = apply_vibrato(
    f0_base=f0_converted,
    rate=vibrato_rate,
    depth=vibrato_depth,
    phase=vibrato_phase
)
```

#### 3. Feature Disentanglement
**Key Principle**: Separate content, pitch, and timbre
- **Content**: HuBERT features (what is sung)
- **Pitch**: F0 contour (how it's sung)
- **Timbre**: Speaker embedding (who sings it)

This separation allows changing timbre while preserving pitch and expression.

## 📊 Performance Benchmarks

### Pitch Accuracy
- **CREPE "full"**: <10 cents error (professional quality)
- **CREPE "tiny"**: ~20 cents error (real-time capable)
- **RMVPE**: ~12 cents error (best speed/accuracy tradeoff)

### Processing Speed (RTX 3090)
- **Vocal Separation**: ~5 seconds per minute of audio
- **Pitch Extraction**: ~2 seconds per minute (CREPE full)
- **Voice Conversion**: ~10 seconds per minute
- **Vocoder**: ~3 seconds per minute
- **Total**: ~20-30 seconds for 3-minute song

### Training Requirements
- **RVC**: 10-60 minutes of clean vocals
- **So-VITS-SVC**: 30-120 minutes recommended
- **Training Time**: 2-6 hours on RTX 3090

## 🎨 Frontend Best Practices

### Inspiration from RVC WebUI
- **Drag-and-drop** file upload
- **Real-time progress** bars for each pipeline stage
- **Waveform visualization** (before/after comparison)
- **Pitch graph overlay** showing F0 contour
- **Voice profile management** with preview samples
- **Batch processing** for multiple songs

### Recommended Tech Stack
- **Frontend**: React + TypeScript + Tailwind CSS
- **Waveform Display**: Wavesurfer.js
- **Charts**: Chart.js or D3.js for pitch visualization
- **File Upload**: react-dropzone
- **State Management**: Zustand or Redux Toolkit
- **API Client**: Axios with progress tracking

## 🔧 Implementation Roadmap

### Phase 1: Core Pipeline (Week 1-2)
- [ ] Integrate Demucs for vocal separation
- [ ] Implement CREPE pitch extraction with vibrato detection
- [ ] Add HuBERT-Soft content encoder
- [ ] Build So-VITS-SVC model wrapper
- [ ] Integrate HiFi-GAN vocoder
- [ ] Create end-to-end pipeline

### Phase 2: Web Interface (Week 2-3)
- [ ] Build React frontend with modern UI
- [ ] Implement drag-and-drop upload
- [ ] Add real-time progress via WebSocket
- [ ] Create waveform visualization
- [ ] Build voice profile management
- [ ] Add pitch comparison graphs

### Phase 3: Optimization (Week 3-4)
- [ ] GPU batch processing
- [ ] CUDA kernel optimization
- [ ] TensorRT conversion for vocoder
- [ ] Caching and job queue (Celery + Redis)
- [ ] Load balancing for concurrent users

### Phase 4: Quality Assurance (Week 4)
- [ ] A/B testing with professional singers
- [ ] Pitch accuracy validation (<5 cents target)
- [ ] Vibrato preservation testing
- [ ] Audio quality metrics (PESQ, STOI)
- [ ] User acceptance testing

## 📚 Key Resources

### GitHub Repositories
1. **RVC-Project/Retrieval-based-Voice-Conversion-WebUI** (32.9k stars)
2. **svc-develop-team/so-vits-svc** (25k+ stars)
3. **maxrmorrison/torchcrepe** (PyTorch CREPE implementation)
4. **facebookresearch/fairseq** (HuBERT models)
5. **jik876/hifi-gan** (Official HiFi-GAN)

### Pre-trained Models
- **HuBERT-Soft**: https://github.com/bshall/hubert (PyTorch Hub)
- **CREPE**: Included in torchcrepe package
- **HiFi-GAN**: https://github.com/jik876/hifi-gan/releases
- **Demucs**: https://github.com/facebookresearch/demucs

### Research Papers
1. Kim et al. "CREPE: A Convolutional Representation for Pitch Estimation" (ICASSP 2018)
2. Yoneyama et al. "RMVPE: Robust Model for Vocal Pitch Estimation" (InterSpeech 2023)
3. Hsu et al. "HuBERT: Self-Supervised Speech Representation Learning" (TASLP 2021)
4. Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (NeurIPS 2020)

---

**Next Step**: Use `CLAUDE_CODE_SWARM_PROMPT.md` to deploy parallel agent swarms for implementation.

