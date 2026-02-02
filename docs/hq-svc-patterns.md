# HQ-SVC Enhancement Patterns

Research compiled 2026-02-01 by hq-svc-researcher agent.
Paper: "HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios" (AAAI 2026)
arXiv: 2511.08496

## Overview

HQ-SVC is an efficient framework for high-quality zero-shot singing voice conversion (SVC) that uniquely combines:
1. **Decoupled codec** (FACodec) for joint content/speaker feature extraction
2. **Enhanced Voice Adaptation (EVA)** module for multi-feature fusion
3. **DDSP + Diffusion** progressive optimization for high-fidelity synthesis
4. **NSF-HiFiGAN** vocoder for 44.1kHz output

Key advantages over existing methods:
- Zero-shot conversion without fine-tuning for new speakers
- Low-resource training: single RTX 3090 GPU, <80 hours data, ~11 hours training
- Dual capabilities: voice conversion + voice super-resolution (16kHz → 44.1kHz)
- Superior naturalness and speaker similarity vs SOTA methods

---

## Architecture Patterns

### Pattern 1: Unified Decoupled Codec (FACodec)

**Problem:** Separate content/speaker encoders lose essential acoustic information and complicate feature alignment.

**Solution:** Use FACodec as a unified disentangler that extracts both content (`x_con`) and speaker (`x_spk`) features simultaneously from the same encoder-decoder architecture.

```
Audio (44.1kHz) → FACodec Encoder → Decoder Intermediate Layers
                                    ├─→ x_con (content features, 256-dim)
                                    └─→ x_spk (speaker features, 256-dim)
```

**Implementation:**
```python
# FACodec uses factorized vector quantization (FVQ)
# Q_c: 2 quantizers for content
# Q_p: 1 quantizer for prosody
# Q_d: 3 quantizers for detail
# All codebook size = 1024

fa_encoder, fa_decoder = load_facodec(device)
# Freeze all layers - no fine-tuning needed
```

**Benefits:**
- Zero-shot disentanglement for unseen speakers
- Reduced information loss vs separate modeling
- Easier feature alignment
- Pre-trained on large speech data (transfer learning)

---

### Pattern 2: Enhanced Voice Adaptation (EVA) Module

**Problem:** Content and speaker features alone don't capture melody richness and energy dynamics crucial for singing.

**Solution:** EVA module introduces additional acoustic features and fuses them effectively:

```
                    ┌─────────────────────────────────────────────────────┐
                    │                     EVA Module                       │
                    │                                                      │
  x_spk ──────────▶│ MLP → e_spk ──┐                                     │
                    │               │                                      │
  x_spk ──────────▶│ MLP → e_sty ──┤    ┌─────────┐                      │
                    │ (residual)    │    │         │    Conformer         │
  x_f0 ───────────▶│ MLP → e_f0 ───┼──▶│ FiLM    │──▶ (8-head attn) ──▶ e│
                    │               │    │         │                      │
  x_vol ──────────▶│ MLP → e_vol ──┤    └─────────┘                      │
                    │               │        ▲                            │
  x_pha ──────────▶│ MLP → e_pha ──┘        │                            │
                    │                        │                            │
  x_con ──────────▶│─────── Conv + Norm ────┘                            │
                    │                                                      │
                    └─────────────────────────────────────────────────────┘
```

**Key Innovation - Mel-scaled F0 transformation:**
```python
# Transform F0 for better melody capture
x_f0 = log_e(f0 / 700 + 1)  # Mel-scale mapping
```

**Style Embedding Fusion:**
```python
# Combine speaker + F0 (close relationship between pitch and timbre)
# Then concatenate with style, volume, phase
e_s = Concat(e_spk + e_f0, e_sty, e_vol, e_pha)  # 1024-dim
e_s = Conv1D(e_s)  # Compress to 256-dim
```

**FiLM Conditioning:**
```python
# Feature-wise Linear Modulation for content-style fusion
def FiLM(e_c, e_s):
    return f_alpha(e_s) * e_c + f_beta(e_s)
```

**Benefits:**
- Multi-feature fusion captures vocal dynamics
- Pitch-timbre correlation properly modeled
- Style residual preserves speaker-specific characteristics

---

### Pattern 3: Speaker-F0 Predictor (SFP)

**Problem:** In zero-shot scenarios, we can't access target speaker's pitch statistics for natural pitch range matching.

**Solution:** Train an MLP to predict F0 mean and variance from speaker embeddings.

```python
class SpeakerF0Predictor(nn.Module):
    def __init__(self, embed_dim=256):
        self.shared = nn.Linear(embed_dim, embed_dim)
        self.mean_head = nn.Linear(embed_dim, 1)
        self.var_head = nn.Linear(embed_dim, 1)

    def forward(self, e_spk):
        h = self.shared(e_spk)
        mu = self.mean_head(h)
        sigma2 = self.var_head(h)
        return mu, sigma2

# Loss: L1 between predicted and actual F0 statistics
L_f0 = E[|mu_x_f0 - mu_hat| + |sigma2_x_f0 - sigma2_hat|]
```

**Benefits:**
- Enables F0 normalization for cross-speaker conversion
- Automatic pitch range adaptation to target speaker
- No need for target speaker pitch data

---

### Pattern 4: Progressive Synthesis (DDSP → Diffusion)

**Problem:** Single-stage synthesis either lacks control (pure neural) or lacks detail (pure DSP).

**Solution:** Two-stage progressive refinement:

**Stage 1: DDSP (Differentiable Digital Signal Processing)**
```
EVA output (e) → DDSP Model → Audio → Mel spectrogram (x̂_ddsp)

Loss: L_ddsp = MSE(x, x̂_ddsp)
```

DDSP provides:
- Harmonic synthesizer (periodic components)
- Noise synthesizer (aperiodic components)
- Strong inductive bias for audio fidelity
- Fine-grained control over timbre/loudness

**Stage 2: Diffusion Model**
```
EVA output (e) → WaveNet Denoiser → Refined Mel spectrogram

Loss: L_diff = E[||ε - ε_θ(√α̅_t·x_0 + √(1-α̅_t)·ε, t, e)||²]
```

Diffusion provides:
- Gap-filling for DDSP limitations
- Enhanced acoustic detail
- Better harmonic richness

**Configuration:**
```yaml
diffusion:
  denoiser: WaveNet (FastSpeech2 variant)
  input_dim: 128
  residual_layers: 20
  conv_channels: 512
  encoder_hidden: 256
  inference_steps: 100
  speedup: 10  # DPM-Solver++
  method: 'dpm-solver'
```

---

### Pattern 5: Speaker Contrastive Loss (InfoNCE)

**Problem:** Zero-shot conversion requires robust speaker discrimination.

**Solution:** InfoNCE-based speaker loss to pull same-speaker embeddings together and push different speakers apart.

```python
def speaker_loss(e_spk_batch, speaker_ids, temperature=0.1):
    """
    e_spk_batch: [batch_size, embed_dim]
    speaker_ids: [batch_size] - ground truth speaker IDs
    """
    N = e_spk_batch.size(0)
    loss = 0

    for i in range(N):
        # Positive: same speaker ID
        positives = e_spk_batch[speaker_ids == speaker_ids[i]]
        # Negatives: different speaker IDs
        negatives = e_spk_batch[speaker_ids != speaker_ids[i]]

        pos_sim = torch.exp(F.cosine_similarity(e_spk_batch[i], positives) / temperature)
        neg_sim = torch.exp(F.cosine_similarity(e_spk_batch[i], negatives) / temperature)

        loss -= torch.log(pos_sim.sum() / (pos_sim.sum() + neg_sim.sum()))

    return loss / N
```

**Benefits:**
- Better speaker discrimination for unseen speakers
- Helps CAM++ speaker verification identify converted audio
- Improves timbre consistency

---

### Pattern 6: Voice Super-Resolution Mode

**Problem:** Low-quality 16kHz audio needs enhancement to 44.1kHz.

**Solution:** HQ-SVC's architecture naturally supports super-resolution because:
1. Input features extracted at 16kHz (encoder_sr)
2. Output aligned with 44.1kHz Mel spectrograms (sample_rate)

```python
def super_resolve(audio_16k, sample_rate=16000):
    """Zero-shot voice super-resolution."""
    # Extract features at 16kHz
    data = process_audio(audio_16k, sample_rate)

    # Use source's OWN speaker embedding (reconstruction mode)
    spk_emb = data['spk']

    # Generate at 44.1kHz
    mel_g = net_g(data['vq_post'], data['f0'], data['vol'], spk_emb,
                  infer=True, infer_speedup=10)

    # Vocoder synthesis
    audio_44k = vocoder.infer(mel_g, data['f0'])
    return audio_44k
```

**Performance:**
- LSD: 1.842 (better than AudioSR's 2.087)
- NISQA: 4.193 (higher naturalness than ground truth degraded)
- SECS: 0.766 (better speaker similarity than AudioSR)

---

### Pattern 7: Optimal Inference Configuration

**Sampler Comparison:**

| Sampler | Speed | SECS | F0 RMSE | NISQA | Best Speedup |
|---------|-------|------|---------|-------|--------------|
| DPM-Solver++ | 0.065 RTF | 0.627 | 8.681 | 3.841 | **10x** |
| UniPC | 0.061 RTF | 0.598 | 8.699 | 3.737 | 10x |
| DDIM | 0.060 RTF | 0.612 | 8.754 | 3.763 | 10x |

**Recommended Configuration:**
```yaml
inference:
  sampler: dpm-solver++
  speedup: 10  # Best quality/speed tradeoff
  steps: 100   # Base steps (effective: 10 after speedup)
  vocoder: nsf-hifigan
  chunk_processing: true  # Reduce latency
```

---

## Integration with AutoVoice

### Current Implementation: `hq_svc_wrapper.py`

```python
class HQSVCWrapper:
    """Wrapper for HQ-SVC cutting-edge voice conversion."""

    def __init__(self, device=None, config_path=None, require_gpu=True):
        # Components loaded:
        # - FACodec encoder/decoder
        # - RMVPE F0 extractor
        # - Volume extractor
        # - DDSP + Diffusion generator (net_g)
        # - NSF-HiFiGAN vocoder

    def super_resolve(self, audio, sample_rate) -> Dict:
        """16kHz → 44.1kHz enhancement."""

    def convert(self, source_audio, source_sr,
                target_audio=None, speaker_embedding=None,
                pitch_shift=0, auto_pitch=False) -> Dict:
        """Voice conversion with optional pitch adjustment."""
```

### Pipeline Integration

HQ-SVC integrates with the quality pipeline as an optional enhancement stage:

```
                          ┌─────────────────────┐
                          │   QUALITY PIPELINE   │
                          │                      │
Audio ──▶ Whisper ──▶ Seed-VC DiT ──▶ BigVGAN ──┼──▶ [HQ-SVC Enhancement]
                          │                      │
                          └─────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │    HQ-SVC (Optional) │
                          │                      │
                          │  Super-resolution    │
                          │  Additional polish   │
                          │  44.1kHz output      │
                          │                      │
                          └─────────────────────┘
```

### Benchmark Results (from Phase 3)

| Mode | RTF | MCD | Sample Rate |
|------|-----|-----|-------------|
| Realtime Pipeline | 0.475 | 955 | 22kHz |
| Quality Pipeline (Seed-VC) | 1.981 | 183 | 44kHz |
| Quality + HQ-SVC | 2.083 | 183.93 | 44kHz |
| HQ-SVC Super-Res Only | 0.102 | N/A | 44kHz |

---

## Training Insights

### Loss Function

```python
L_total = L_ddsp + L_diff + L_spk + L_f0

# Where:
# L_ddsp = MSE(mel_gt, mel_ddsp)
# L_diff = Diffusion denoising loss
# L_spk = InfoNCE speaker contrastive loss (τ=0.1)
# L_f0 = L1(μ_f0_pred, μ_f0_gt) + L1(σ²_f0_pred, σ²_f0_gt)
```

### Training Configuration

```yaml
optimizer: AdamW
  beta1: 0.9
  beta2: 0.999
  lr: 1.5e-4

batch_size: 64
temperature: 0.1  # Speaker loss
steps: 250k
gpu_memory: <6GB
training_time: ~11 hours (RTX 3090)
data_requirements: <80 hours singing
```

### Data Preprocessing

```python
# 1. Resample all audio to 44.1kHz
# 2. Extract 128-dim Mel spectrograms (hop_size=512)
# 3. Extract energy features (same hop_size)
# 4. Downsample to 16kHz for feature extraction
# 5. RMVPE for pitch (F0)
# 6. FACodec for content + speaker (256-dim each)
# 7. Build speaker ID lookup table for contrastive loss
```

---

## Ablation Insights

### Critical Components

| Component Removed | Effect |
|-------------------|--------|
| Diffusion | SECS drops 0.42→0.20, NISQA drops 3.841→3.175 |
| DDSP | Higher STOI but worse SECS (0.608), poor conversion |
| L_spk | Worse SECS, F0 RMSE, NISQA |
| L_f0 | Worse F0 tracking, slightly better SMOS |

### Key Finding: DDSP + Diffusion Synergy

Removing diffusion makes speech clearer but destroys speaker matching.
Removing DDSP improves some metrics but the model just reconstructs source audio (no actual conversion).

**Conclusion:** Both stages are essential for high-quality zero-shot SVC.

---

## Future Directions

1. **Singing Style Control** - Vibrato, dynamics, articulation
2. **Pitch Perturbation Training** - R2-SVC-style robustness
3. **Streaming Mode** - Chunk-wise processing for real-time
4. **ECAPA2 Speaker Encoder** - Better zero-shot generalization
5. **Shortcut Flow Matching** - 2-step inference (R-VC pattern)

---

## References

```bibtex
@article{bai2025hq,
  title={HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios},
  author={Bai, Bingsong and Geng, Yizhong and Wang, Fengping and Wang, Cong and Guo, Puyuan and Gao, Yingming and Li, Ya},
  journal={arXiv preprint arXiv:2511.08496},
  year={2025}
}

@article{ju2024naturalspeech3,
  title={NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models},
  author={Ju et al.},
  journal={ICML 2024}
}

@misc{ddsp-svc,
  title={DDSP-SVC: Real-time end-to-end singing voice conversion based on DDSP},
  author={yxlllc},
  url={https://github.com/yxlllc/DDSP-SVC}
}
```

---

_Last Updated: 2026-02-01_
_Generated by: hq-svc-researcher agent (sota-dual-pipeline swarm)_
