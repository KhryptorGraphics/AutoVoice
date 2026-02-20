# SOTA Architecture Decisions for AutoVoice Pipeline

**Track:** sota-pipeline_20260124
**Date:** 2026-01-24
**Status:** Final

This document records architecture decisions for each pipeline component, backed by academic research with paper citations and benchmark comparisons.

---

## 1. Content Feature Extraction: ContentVec (768-dim, Layer 12)

### Decision
Use **ContentVec** (768-dimensional, Layer 12 output) as the content feature extractor.

### Candidates Evaluated

| Model | Dimensions | Paper | Year | Citations | Speaker Disentanglement |
|-------|-----------|-------|------|-----------|------------------------|
| **ContentVec** | 768 | "ContentVec: An Improved Self-Supervised Speech Representation..." (ICML 2022) | 2022 | 143 | **Best** — trained with speaker-conditioned masking |
| WavLM | 768-1024 | "WavLM: Large-Scale Self-Supervised Pre-Training..." | 2021 | 2683 | Good for noise, not optimized for disentanglement |
| HuBERT-Soft | 256 | "A Comparison of Discrete and Soft Speech Units..." | 2022 | - | Moderate — used by original So-VITS-SVC |

### Rationale

1. ContentVec achieves the best speaker disentanglement among SSL models by design — its training objective explicitly conditions on speaker identity, forcing content features to be speaker-independent.
2. Used by all major SVC systems: RVC, So-VITS-SVC 4.1+, CoMoSVC, NeuCoSVC.
3. LinearVC (2025) confirms SSL features have linearly separable content/speaker subspaces; ContentVec's 768-dim Layer 12 provides the best subspace for VC.
4. Interspeech 2025 dimension reduction paper validates random channel subset selection works for ContentVec in SVC.
5. Checkpoint: `lengyue233/content-vec-best` (checkpoint_best_legacy_500.pt)

### Key Papers
- Qian et al., "ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers" (ICML 2022)
- Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (IEEE JSTSP 2021)
- SaMoye (2024): Validates ContentVec as standard content extractor across 6367 speakers
- NeuCoSVC (ICASSP 2023): Demonstrates concatenative SVC with SSL features

---

## 2. Pitch Extraction: RMVPE

### Decision
Use **RMVPE** (Robust Model for Vocal Pitch Estimation in Polyphonic Music) as the pitch extractor.

### Candidates Evaluated

| Model | Paper | Year | Citations | Polyphonic Support | Accuracy |
|-------|-------|------|-----------|-------------------|----------|
| **RMVPE** | "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" (Interspeech 2023) | 2023 | 42 | **Yes — direct extraction** | Best RPA/RCA across all SNR |
| CREPE | "CREPE: A Convolutional Representation for Pitch Estimation" (ICASSP 2018) | 2018 | 438 | No — monophonic only | SOTA for clean monophonic |
| DJCM | "Joint Cascade Model for Singing Voice Separation and Pitch Estimation" (ICASSP 2024) | 2024 | - | Yes (via joint sep+pitch) | Promising but complex |
| FCN-F0 | "Vocal Pitch Extraction in Polyphonic Music Using Convolutional Residual Network" (Interspeech 2019) | 2019 | 10 | Partial | 5% OA improvement over baseline |

### Rationale

1. **Polyphonic robustness**: RMVPE extracts vocal pitch directly from polyphonic music without requiring source separation first. This is critical because:
   - Source separation introduces artifacts that degrade pitch accuracy
   - Our pipeline may receive mixed audio (vocals + accompaniment)
   - Eliminates a dependency on the separation module for pitch

2. **Accuracy**: Superior RPA (Raw Pitch Accuracy) and RCA (Raw Chroma Accuracy) across all SNR levels compared to separation-then-estimate approaches.

3. **Robustness across SNR**: Experimental results show RMVPE is robust at all signal-to-noise ratios, making it reliable regardless of accompaniment loudness.

4. **Ecosystem adoption**: Used by RVC (Retrieval-based Voice Conversion) project, which is the most deployed SVC system.

5. **Open source**: Available at https://github.com/Dream-High/RMVPE

### Integration Notes
- Output: F0 contour at frame-level (10ms hop)
- Integrate with existing mel-quantized F0 pipeline (UV embeddings for unvoiced frames)
- Frame alignment via F.interpolate to match content features

### Key Papers
- Wei et al., "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" (Interspeech 2023)
- Kim et al., "CREPE: A Convolutional Representation for Pitch Estimation" (ICASSP 2018)

---

## 3. Neural Vocoder: BigVGAN v2 (112M, 24kHz)

### Decision
Use **BigVGAN v2** (112M parameters, 24kHz, Snake activation) as the neural vocoder.

### Candidates Evaluated

| Model | Params | Paper | Year | Citations | MOS | Speed | Singing Quality |
|-------|--------|-------|------|-----------|-----|-------|-----------------|
| **BigVGAN v2** | 112M | "BigVGAN: A Universal Neural Vocoder..." (ICLR 2023) + v2 update (2024) | 2023/2024 | 386 | **4.3+** | Moderate | **Best — zero-shot generalization** |
| Vocos | ~13M | "Vocos: Closing the gap between time-domain and Fourier-based..." (ICLR 2024) | 2023 | 187 | 4.1 | **10x faster than HiFi-GAN** | Good but less validated for singing |
| HiFTNet | ~14M | "HiFTNet: A Fast High-Quality Neural Vocoder..." | 2023 | 6 | 4.2 | 4x faster than BigVGAN | Uses F0 input (harmonic-plus-noise) |
| HiFi-GAN v1 | 14M | "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" | 2020 | 2000+ | 4.0 | Fast | Speech-focused, not singing-optimized |

### Rationale

1. **Universal zero-shot generalization**: BigVGAN trained only on LibriTTS generalizes to unseen singing voices, music, and instruments without fine-tuning. This is unique and critical for SVC where target speakers are unseen.

2. **Snake periodic activation**: Provides inductive bias for periodic waveform generation — fundamental for singing voice which has strong periodicity (sustained notes, vibrato).

3. **Anti-aliased multi-periodicity composition (AMP)**: Suppresses aliasing artifacts common in GAN vocoders during upsampling, especially important for high-frequency singing harmonics.

4. **BigVGAN v2 improvements** (2024):
   - CQT-based discriminator for better harmonic accuracy
   - Diverse training data including singing
   - CUDA kernels for Snake activation (performance)
   - Better results on singing voice specifically

5. **Quality metrics**: Achieves SOTA in both objective metrics (PESQ, MCD) and subjective MOS across out-of-distribution conditions.

6. **Model availability**: MIT license, pretrained weights at https://github.com/NVIDIA/BigVGAN

### Configuration for AutoVoice
- Model: BigVGAN-v2 24kHz (112M params)
- Input: 100-band mel spectrogram
- Output: 24kHz waveform
- Hop size: 256 samples (10.67ms frames)
- fmax: 12000 Hz

### Vocos as Future Alternative
Vocos operates in frequency domain (STFT coefficients) and is 10x faster. Consider as TensorRT-optimized alternative if BigVGAN proves too slow for real-time streaming on Jetson Thor.

### Key Papers
- Lee et al., "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (ICLR 2023)
- Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders" (ICLR 2024)

---

## 4. Source Separation: BS-RoFormer (Mel-Band variant)

### Decision
Use **Mel-Band RoFormer** (Mel-RoFormer) for vocal separation from mixed audio.

### Candidates Evaluated

| Model | Paper | Year | Citations | Vocals SDR (MUSDB18HQ) | Approach |
|-------|-------|------|-----------|------------------------|----------|
| **BS-RoFormer** | "Music Source Separation With Band-Split RoPE Transformer" (ICASSP 2024) | 2023 | 63 | **9.80 dB avg (won SDX'23)** | Band-split + RoPE Transformer |
| Mel-RoFormer | "Mel-Band RoFormer for Music Source Separation" | 2023 | 16 | **>9.80 dB** (improves on BS-RoFormer) | Mel-scale band-split |
| HTDemucs | "Hybrid Transformers for Music Source Separation" (ICASSP 2023) | 2022 | 234 | 9.20 dB | Hybrid temporal/spectral bi-U-Net |
| BSRNN | "Music Source Separation With Band-Split RNN" (IEEE TASLP) | 2022 | 179 | 8.9 dB (pre-fine-tune) | Band-split + interleaved RNN |

### Rationale

1. **Best separation quality**: BS-RoFormer won the Sound Demixing Challenge 2023 (SDX'23) with 9.80 dB average SDR, surpassing HTDemucs (9.20 dB) by 0.6 dB.

2. **Mel-Band variant improves further**: Mel-RoFormer adopts mel-scale band splitting which is acoustically motivated (matches human perception), outperforming the heuristic band-split scheme of original BS-RoFormer on vocals, drums, and other stems.

3. **Architecture advantages**:
   - Band-split module projects spectrogram into subband-level representations
   - Hierarchical Transformer with RoPE models both inner-band and inter-band sequences
   - RoPE (Rotary Position Embedding) provides better positional encoding than absolute positions

4. **vs HTDemucs**: While HTDemucs pioneered cross-domain attention (temporal + spectral), BS-RoFormer's explicit band-split modeling is more parameter-efficient and achieves better SDR. HTDemucs requires 800 extra training songs to match performance.

5. **Practical considerations**: BS-RoFormer operates in frequency domain (no waveform decoder needed), producing clean spectrograms that feed directly into our content/pitch extractors.

### Integration Notes
- Input: Mixed audio (vocals + accompaniment) at any sample rate
- Output: Isolated vocal track
- Fallback: If input is already acapella, passthrough (detect via energy ratio)
- Error: If no vocals detected, raise RuntimeError (no silent degradation)

### Key Papers
- Lu et al., "Music Source Separation With Band-Split RoPE Transformer" (ICASSP 2024)
- Wang et al., "Mel-Band RoFormer for Music Source Separation" (2023)
- Rouard et al., "Hybrid Transformers for Music Source Separation" (ICASSP 2023)
- Luo & Yu, "Music Source Separation With Band-Split RNN" (IEEE TASLP 2022)

---

## 5. SVC Decoder: Consistency Model (CoMoSVC Architecture)

### Decision
Use a **consistency model** distilled from a diffusion teacher, following the CoMoSVC architecture with BiDilConv decoder.

### Candidates Evaluated

| Approach | Paper | Year | Citations | Steps | Quality | RTF |
|----------|-------|------|-----------|-------|---------|-----|
| **CoMoSVC (Consistency)** | "CoMoSVC: Consistency Model-based Singing Voice Conversion" (ISCSLP 2024) | 2024 | 16 | **1** | **Matches 50-step diffusion** | <0.01 |
| LCM-SVC (Latent Consistency) | "LCM-SVC: Latent Diffusion Model Based SVC..." (ISCSLP 2024) | 2024 | 3 | 1-4 | Comparable to CoMoSVC | <0.01 |
| DiffSVC (Diffusion) | "DiffSVC: Diffusion Model for Singing Voice Conversion" | 2022 | - | 100 | High quality | ~0.5-1.0 |
| Flow-matching (RASVC) | "Real-Time and Accurate: Zero-shot High-Fidelity SVC..." | 2024 | 2 | ~10 | SOTA-comparable | ~0.1 |
| LDM-SVC (Latent Diffusion) | "LDM-SVC: Latent Diffusion Model Based Zero-Shot Any-to-Any SVC" (Interspeech 2024) | 2024 | 15 | 50 | High quality, good disentanglement | ~0.3 |
| CoMoSpeech | "CoMoSpeech: One-Step Speech and Singing Voice Synthesis..." (ACM MM 2023) | 2023 | 55 | **1** | 150x faster than real-time | <0.01 |

### Rationale

1. **One-step inference**: CoMoSVC achieves quality matching 50-100 step diffusion models with a single forward pass. This is critical for:
   - Real-time streaming mode (Phase 9) where latency budget is <50ms
   - TensorRT optimization (single inference = simple engine, no iterative loops)
   - Jetson Thor deployment where memory constrains model loading

2. **Proven architecture for SVC**: CoMoSVC specifically validates the consistency approach for singing voice (not just speech), showing +0.05 similarity improvement over DiffSVC and SoVITS-Diff baselines.

3. **Two-stage distillation**:
   - Stage 1: Train a diffusion teacher model with Conformer encoder + BiDilConv decoder
   - Stage 2: Distill to consistency model using self-consistency property
   - This matches our existing Conformer encoder architecture

4. **BiDilConv decoder**: Non-causal bidirectional dilated CNN with:
   - 256 residual channels
   - Dilation cycle [1, 2, 4, 8, ..., 512] × 2 cycles = 20 blocks
   - Gated activation (sigmoid × tanh)
   - Lightweight compared to attention-based decoders

5. **Feature compatibility**: CoMoSVC uses ContentVec 768-dim + F0 + loudness — exactly matching our pipeline choices.

6. **LCM-SVC as enhancement**: Can additionally apply Latent Consistency Distillation for even better quality at 2-4 steps if 1-step quality is insufficient.

### Training Strategy
1. Train diffusion teacher (Conformer encoder → BiDilConv decoder) with:
   - Multi-resolution STFT loss
   - Mel-spectrogram L1 loss
   - Adversarial loss (multi-period discriminator)
2. Distill consistency student using self-consistency property
3. Fine-tune with one-step objective

### Key Papers
- Lu et al., "CoMoSVC: Consistency Model-based Singing Voice Conversion" (ISCSLP 2024)
- Chen et al., "LCM-SVC: Latent Diffusion Model Based SVC with Inference Acceleration via LCD" (ISCSLP 2024)
- Ye et al., "CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model" (ACM MM 2023)
- Chen et al., "LDM-SVC: Latent Diffusion Model Based Zero-Shot Any-to-Any SVC" (Interspeech 2024)

---

## 6. Real-time Streaming: Chunked Inference with Overlap-Add

### Decision
Implement streaming via **chunked inference** with overlap-add crossfading, targeting <50ms end-to-end latency.

### Reference Systems

| System | Paper | Year | Citations | Latency | Approach |
|--------|-------|------|-----------|---------|----------|
| **StreamVC** | "StreamVC: Real-Time Low-Latency Voice Conversion" (ICASSP 2024) | 2024 | 32 | Low (mobile-capable) | Causal SoundStream codec + soft units |
| DualVC 3 | "DualVC 3: Leveraging LM Generated Pseudo Context..." (Interspeech 2024) | 2024 | 5 | **50ms** | End-to-end, pseudo context from LM |
| SynthVC | "SynthVC: Leveraging Synthetic Data for End-to-End Low Latency Streaming VC" | 2025 | 0 | **77.1ms** | Neural codec + synthetic parallel data |
| DarkStream | "DarkStream: real-time speech anonymization with low latency" | 2025 | 1 | Low | Causal encoder + GAN speaker embedding |

### Architecture Design

```
Microphone Input (48kHz)
    |
    v
[Ring Buffer: chunk_size + lookahead]
    |
    v
[Resample to 24kHz]
    |
    v
[Source Separation] (if mixed input)
    |
    v
[Content Extraction] (ContentVec, causal/chunked)
    |
    v
[Pitch Extraction] (RMVPE, chunked)
    |
    v
[Consistency Decoder] (1-step, TensorRT)
    |
    v
[BigVGAN Vocoder] (TensorRT)
    |
    v
[Overlap-Add Crossfade]
    |
    v
Audio Output (48kHz)
```

### Key Design Parameters
- **Chunk size**: 20-40ms (480-960 samples at 24kHz)
- **Lookahead**: 10-20ms for context
- **Crossfade**: Linear or Hanning window, 5-10ms overlap
- **Total latency budget**: <50ms (chunk + processing + output buffer)

### Streaming Considerations

1. **Causal content encoding**: ContentVec is non-causal by default. Options:
   - Use causal fine-tuned variant
   - Accept lookahead penalty (~30ms) for better quality
   - StreamVC shows soft speech units can be learned causally

2. **Pitch tracking latency**: RMVPE needs context window. Use:
   - Minimum 2-3 pitch periods (~20ms at 100Hz fundamental)
   - Can run at lower frame rate than content features

3. **Consistency model advantage**: 1-step inference means no iterative diffusion loop, enabling predictable latency per chunk.

4. **TensorRT for all components**: Each component runs as a single TRT engine call per chunk — no Python overhead in the streaming loop.

### Key Papers
- Yang et al., "StreamVC: Real-Time Low-Latency Voice Conversion" (ICASSP 2024)
- Ning et al., "DualVC 3: Leveraging Language Model Generated Pseudo Context..." (Interspeech 2024)
- Guo et al., "SynthVC: Leveraging Synthetic Data for End-to-End Low Latency Streaming VC" (2025)

---

## Summary: Final Pipeline Architecture

```
Input Audio
    |
    v
[Mel-Band RoFormer] ─── Vocal Separation (if mixed)
    |
    v
Isolated Vocals
    |
    +──────────────────────┐
    |                      |
    v                      v
[ContentVec]          [RMVPE]
768-dim features      F0 contour
    |                      |
    +──────────┬───────────+
               |
               v
    [Speaker Embedding]
    (mel-statistics 256-dim)
               |
               v
    [Conformer Encoder]
    (6 layers, content + F0 + speaker)
               |
               v
    [Consistency Decoder]
    (BiDilConv, 1-step inference)
               |
               v
    Predicted Mel Spectrogram
               |
               v
    [BigVGAN v2]
    (112M, Snake activation)
               |
               v
    Output Waveform (24kHz)
```

### Deployment Modes
1. **Batch mode**: Full file processing, highest quality
2. **Streaming mode**: Chunked inference, <50ms latency, real-time performance
3. **TensorRT mode**: All components as TRT engines, optimized for Jetson Thor

### Memory Budget (Jetson Thor, 16GB)
| Component | Estimated Memory (FP16) |
|-----------|------------------------|
| Mel-RoFormer | ~200MB |
| ContentVec | ~300MB |
| RMVPE | ~50MB |
| Conformer Encoder | ~100MB |
| Consistency Decoder | ~150MB |
| BigVGAN v2 | ~250MB |
| **Total** | **~1.05GB** |
| Available for activations | ~14.9GB |

---

_All decisions based on papers published 2022-2025. Architecture validated against published benchmarks and production deployments (RVC, So-VITS-SVC 4.1+, CoMoSVC, StreamVC)._
