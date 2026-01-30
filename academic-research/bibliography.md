# AutoVoice Research Bibliography

## Core Papers for Singing Voice Conversion

### 1. CoMoSVC: Consistency Model-based Singing Voice Conversion
- **ArXiv:** 2401.01792
- **Authors:** Yiwen Lu, Zhen Ye, Wei Xue, Xu Tan, Qifeng Liu, Yike Guo (2024)
- **Relevance:** Direct architecture upgrade path. Distills diffusion teacher into one-step student for 40x inference speedup. Part of Amphion project.
- **Key Technique:** EDM preconditioning + consistency distillation

### 2. Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
- **ArXiv:** 2206.00364
- **Authors:** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (2022)
- **Relevance:** Unified diffusion framework. Noise schedule design (sigma_min=0.002, sigma_max=80, rho=7). Foundation for CoMoSVC.

### 3. Consistency Models
- **ArXiv:** 2303.01469
- **Authors:** Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023)
- **Relevance:** Core theory for one-step generation. Maps noise directly to data without iterative sampling. Enables real-time SVC.

### 4. HuBERT: Self-Supervised Speech Representation Learning
- **ArXiv:** 2106.07447
- **Authors:** Wei-Ning Hsu, Benjamin Bolte, et al. (2021)
- **Relevance:** Current content encoder in AutoVoice. BERT-like masked prediction with offline clustering. Extracts speaker-agnostic phonetic features.

### 5. ContentVec: Improved Self-Supervised Speech Representation
- **ArXiv:** 2204.09224
- **Authors:** Kaizhi Qian, Yang Zhang, et al. (2022)
- **Relevance:** Upgrade path for content extraction. Better speaker disentanglement than HuBERT. 256-dim features, directly compatible.

### 6. Soft Speech Units for Voice Conversion (Soft-VC)
- **ArXiv:** 2111.02392
- **Authors:** Benjamin van Niekerk, Marc-Andre Carbonneau, et al. (2021)
- **Relevance:** Foundation of So-VITS-SVC architecture. Soft units (distributions over discrete tokens) preserve more content than hard quantization.

### 7. HiFi-GAN: High Fidelity Speech Synthesis
- **ArXiv:** 2010.05646
- **Authors:** Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (2020)
- **Relevance:** Current vocoder in AutoVoice. Multi-period + multi-scale discriminators. 167.9x faster than real-time.

### 8. BigVGAN: Universal Neural Vocoder
- **ArXiv:** 2206.04658
- **Authors:** Sang-gil Lee, Wei Ping, Boris Ginsburg, et al. (2022)
- **Relevance:** Vocoder upgrade path. Periodic activations + anti-aliasing. 112M params. Strong zero-shot generalization to singing.

### 9. VITS: Conditional VAE with Adversarial Learning
- **ArXiv:** 2106.06103
- **Authors:** Jaehyeon Kim, Jungil Kong, Juhee Son (2021)
- **Relevance:** Backbone of So-VITS-SVC. VAE + normalizing flows + adversarial training. End-to-end parallel synthesis.

### 10. ECAPA-TDNN: Speaker Verification
- **ArXiv:** 2005.07143
- **Authors:** Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck (2020)
- **Relevance:** Speaker embedding upgrade. Res2Net + SE blocks + multi-layer aggregation. More discriminative than mel-statistics.

### 11. Neural Source-Filter (NSF) Waveform Models
- **ArXiv:** 1904.12088
- **Authors:** Xin Wang, Shinji Takaki, Junichi Yamagishi (2019)
- **Relevance:** Source-filter decomposition for pitch-aware synthesis. Sine excitation + dilated conv filtering. Better pitch preservation.

### 12. Amphion: Open-Source Audio Generation Toolkit
- **ArXiv:** 2312.09911
- **Authors:** Xueyao Zhang, Liumeng Xue, et al. (2023)
- **Relevance:** Reference implementation for CoMoSVC. Unified framework with vocoders, metrics, and SVC baselines.

### 13. DiffSinger: Singing Voice Synthesis via Shallow Diffusion
- **ArXiv:** 2105.02446
- **Authors:** Jinglin Liu, Chengxi Li, Yi Ren, et al. (2021)
- **Relevance:** Shallow diffusion mechanism starts generation at intersection of diffusion trajectories. More natural singing than L1/GAN.

### 14. ProDiff: Progressive Fast Diffusion
- **ArXiv:** 2207.06389
- **Authors:** Rongjie Huang, Zhou Zhao, et al. (2022)
- **Relevance:** Progressive knowledge distillation halves diffusion steps iteratively. Precursor to consistency models. 2-step synthesis.

---

## Architecture Comparison: Current vs Target

| Component | Current (AutoVoice) | Target (Amphion-inspired) |
|-----------|---------------------|---------------------------|
| Content Encoder | HuBERT (single) | ContentVec + HuBERT (multi-modal) |
| Pitch | pyin → LSTM | Mel-quantized F0 (256 bins) + UV embedding |
| Speaker | Mel-statistics [256] | ECAPA-TDNN or learned lookup |
| Decoder | VAE + Flow (So-VITS) | Conformer + Diffusion/Consistency |
| Vocoder | HiFiGAN (basic) | BigVGAN or NSF-HiFiGAN |
| Training | Single-stage | Two-stage (teacher/student) |
| Inference | Single forward pass | 1-step consistency or multi-step diffusion |

## Priority Improvements (ordered by impact)

1. **Mel-quantized F0 with UV embedding** - Better pitch representation (Amphion uses 256 mel-scale bins)
2. **ContentVec features** - Better speaker disentanglement than HuBERT alone
3. **BigVGAN vocoder** - Better generalization to singing voice
4. **SSIM loss** - Perceptual quality improvement during training
5. **Consistency distillation** - Fast inference for realtime pipeline
6. **ECAPA-TDNN speaker encoder** - More discriminative speaker embeddings
7. **Conformer encoder** - Better acoustic modeling than simple projection
8. **Data augmentation** - Pitch shift, formant shift, EQ, time stretch
