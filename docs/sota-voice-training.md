# SOTA Voice Training Research: Incremental Learning & Technique Preservation

**Track:** voice-profile-training_20260124
**Tasks:** 2.1-2.4
**Date:** 2026-01-24

## Executive Summary

This document synthesizes state-of-the-art research for building a continuous voice profile training system. The key findings support three architectural approaches:

1. **LoRA/Adapter Fine-tuning** - Parameter-efficient adaptation for speaker-specific models
2. **Elastic Weight Consolidation (EWC)** - Prevent catastrophic forgetting during incremental training
3. **Style/Technique Encoding** - Explicit representation of singing techniques (vibrato, melisma)

## 1. Incremental/Continual Learning for Voice Models

### 1.1 Core Challenge: Catastrophic Forgetting

When fine-tuning neural networks on new data, models tend to "forget" previously learned knowledge. For voice profiles, this means:
- Speaker similarity degradation after training on new samples
- Loss of singing technique capabilities
- Reduced generalization across pitch ranges

### 1.2 SOTA Solutions

#### Speech-FT: Two-Stage Fine-tuning [arXiv:2502.12672]

**Key Innovation:** Reduce representational drift during fine-tuning, then interpolate weights with pre-trained model.

```
Stage 1: Fine-tune with drift-reducing constraints
Stage 2: Weight-space interpolation → θ_final = α*θ_pretrained + (1-α)*θ_finetuned
```

**Results on HuBERT:**
- Phone error rate: 5.17% → 3.94%
- Speaker identification: 81.86% → 84.11%

**Applicability:** Directly applicable to our speaker encoder fine-tuning.

#### EVCL: Elastic Variational Continual Learning [arXiv:2406.15972]

**Key Innovation:** Hybrid of Variational Continual Learning (VCL) + Elastic Weight Consolidation (EWC).

```python
# EWC Loss Component
L_ewc = λ * Σ_i F_i * (θ_i - θ*_i)²

# Where:
# F_i = Fisher information matrix (importance of parameter i)
# θ*_i = optimal parameter values from previous tasks
# λ = regularization strength
```

**Recommendation:** Use EWC to protect critical encoder weights during incremental training.

#### Stable-TTS: Prior-Preservation Loss [arXiv:2412.20155]

**Key Innovation:** Keep a small set of high-quality "prior samples" during fine-tuning to maintain synthesis ability.

```python
L_total = L_target + β * L_prior

# L_target: Loss on new user samples
# L_prior: Loss on preserved quality samples (prevents overfitting)
```

**Applicability:** Maintain a curated set of high-quality singing samples to preserve technique generation.

### 1.3 Recommended Architecture: Hybrid Approach

```
┌─────────────────────────────────────────────────────┐
│              Continuous Learning Engine              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────────┐    ┌──────────────────────────┐ │
│  │  Base Model   │    │   Speaker LoRA Adapters   │ │
│  │  (Frozen)     │───▶│   (Per-profile, ~4MB)    │ │
│  └───────────────┘    └──────────────────────────┘ │
│         │                        │                 │
│         ▼                        ▼                 │
│  ┌───────────────┐    ┌──────────────────────────┐ │
│  │  EWC Fisher   │    │   Technique Adapters     │ │
│  │  Matrix       │───▶│   (Shared, fine-tuned)   │ │
│  └───────────────┘    └──────────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 2. Speaker Adaptation & Few-Shot Voice Cloning

### 2.1 Zero-Shot Approaches

#### TCSinger: Multi-Level Style Control [arXiv:2409.15977]

**Key Components:**
1. **Clustering Style Encoder** - Vector quantization to compress style into compact latent space
2. **Style & Duration LM** - Joint prediction improves both style transfer and timing
3. **Mel-Style Adaptive Normalization** - Dynamic adaptation layer for style injection

**Technique:** Uses clustering VQ to create discrete style tokens that can be mixed/transferred.

#### StyleSinger: Residual Style Adaptor [arXiv:2312.10741]

**Key Innovations:**
1. **Residual Quantization** - Captures hierarchical style information
2. **Uncertainty Modeling Layer Normalization (UMLN)** - Perturbs style during training for better generalization

```python
# UMLN: Adds noise to style embeddings during training
style_perturbed = style + ε * σ_style  # Where ε ~ N(0, 1)
```

**Applicability:** UMLN can help our model generalize to new singing styles from limited samples.

#### DS-TTS: Dual-Style Encoding [arXiv:2506.01020]

**Architecture:**
- Two parallel style encoders capture complementary aspects of vocal identity
- Style Gating-FiLM mechanism for dynamic style injection

**Recommendation:** Consider dual-encoder for separating timbre vs. singing style.

### 2.2 Few-Shot Fine-tuning

#### LoRA for Voice Models

Based on federated LoRA research [arXiv:2501.19389, arXiv:2411.14961]:

```python
# LoRA configuration for voice adaptation
lora_config = {
    "r": 8,           # Rank (8-16 for voice models)
    "alpha": 16,      # Scaling factor (α/r ≈ 2 works well)
    "target_modules": ["q_proj", "v_proj", "content_encoder"],
    "dropout": 0.1,
}
```

**Storage per profile:** ~2-8MB depending on rank and target modules.

## 3. Singing Technique Preservation

### 3.1 Vibrato Detection & Synthesis

#### VibE-SVC: Discrete Wavelet Transform for Vibrato [arXiv:2505.20794]

**Key Innovation:** Decompose F0 contour into frequency components using DWT.

```python
# F0 decomposition for vibrato extraction
import pywt

def extract_vibrato(f0_contour, wavelet='db4', level=4):
    """Extract vibrato using discrete wavelet transform."""
    coeffs = pywt.wavedec(f0_contour, wavelet, level=level)

    # High-frequency components contain vibrato
    vibrato_coeffs = coeffs[1:3]  # Levels 1-2 typically capture 4-8 Hz vibrato

    # Low-frequency = base pitch contour
    base_pitch_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]

    return {
        'vibrato': pywt.waverec(vibrato_coeffs, wavelet),
        'base_pitch': pywt.waverec(base_pitch_coeffs, wavelet)
    }
```

**Vibrato characteristics:**
- Rate: 4-8 Hz (typical singing vibrato)
- Extent: ±50-200 cents
- Onset delay: Often starts 100-300ms into sustained notes

### 3.2 Technique-Controllable Synthesis

#### TechSinger: Multi-Technique Control [arXiv:2502.12572]

**Supported Techniques:**
1. Intensity control
2. Mixed voice
3. Falsetto
4. Bubble/Vocal fry
5. Breathy tones

**Architecture:**
- Flow-matching generative model
- Phoneme-level technique labels
- Prompt-based technique prediction

**Key Insight:** Technique detection model can auto-annotate training data.

```python
# Technique label format
technique_labels = {
    "phoneme_id": "a",
    "techniques": {
        "vibrato": 0.8,      # Intensity 0-1
        "falsetto": 0.0,
        "breathy": 0.2,
        "mixed_voice": 0.5,
    }
}
```

### 3.3 Melisma & Vocal Run Detection

**Detection approach:**
1. Identify rapid pitch transitions (>50 cents/50ms)
2. Track note density (notes per second)
3. Classify as melisma when density > 8 notes/sec

```python
def detect_melisma(f0_contour, hop_length_ms=10):
    """Detect melisma/vocal runs from F0 contour."""
    # Calculate pitch velocity
    pitch_velocity = np.abs(np.diff(f0_contour)) / hop_length_ms

    # High velocity regions indicate rapid pitch changes
    melisma_threshold = 5.0  # cents per ms
    melisma_regions = pitch_velocity > melisma_threshold

    # Group consecutive high-velocity frames
    return find_contiguous_regions(melisma_regions, min_length=50)  # >500ms
```

## 4. Recommended Implementation Strategy

### 4.1 Training Pipeline

```
Phase 1: Base Model Training
├── Train on large multi-speaker dataset
├── Include singing technique annotations
└── Establish baseline quality

Phase 2: Profile-Specific Adaptation
├── Initialize LoRA adapters per profile
├── Fine-tune on user's accumulated samples
├── Apply EWC regularization to prevent forgetting
└── Store ~4MB adapter per profile

Phase 3: Continuous Improvement
├── Collect new samples from karaoke sessions
├── Quality filter (SNR > 20dB, pitch stability)
├── Incremental training with prior preservation
└── Auto-schedule training after N new samples
```

### 4.2 Model Versioning Strategy

```python
class ProfileModelVersion:
    """Track model versions for each profile."""

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.versions = []  # List of (version, metrics, adapter_path)
        self.max_versions = 5  # Keep last 5 versions for rollback

    def add_version(self, adapter_path: str, metrics: dict):
        """Add new model version with quality metrics."""
        version = {
            "version": f"v{len(self.versions) + 1}",
            "created": datetime.now().isoformat(),
            "adapter_path": adapter_path,
            "metrics": {
                "speaker_similarity": metrics["speaker_sim"],
                "mos_estimate": metrics["mos"],
                "technique_preservation": metrics["tech_score"],
            }
        }
        self.versions.append(version)
        self._prune_old_versions()

    def rollback(self, version: str):
        """Rollback to a previous version if quality regresses."""
        pass
```

### 4.3 Training Scheduler

```python
class TrainingScheduler:
    """Auto-trigger training based on accumulated samples."""

    def __init__(self, config: dict):
        self.min_samples_for_training = config.get("min_samples", 10)
        self.min_total_duration_sec = config.get("min_duration", 60)
        self.max_samples_between_training = config.get("max_samples", 50)

    def should_trigger_training(self, profile: VoiceProfile) -> bool:
        """Check if profile has enough new data for training."""
        new_samples = profile.get_unprocessed_samples()
        total_duration = sum(s.duration_seconds for s in new_samples)

        return (
            len(new_samples) >= self.min_samples_for_training and
            total_duration >= self.min_total_duration_sec
        ) or len(new_samples) >= self.max_samples_between_training
```

## 5. Key Paper References

### Continual Learning
- **Speech-FT** [2502.12672] - Two-stage fine-tuning with drift reduction
- **EVCL** [2406.15972] - EWC + Variational Continual Learning hybrid
- **Stable-TTS** [2412.20155] - Prior-preservation loss for speaker adaptation

### Singing Voice Synthesis/Conversion
- **TCSinger** [2409.15977] - Zero-shot style transfer with clustering encoder
- **StyleSinger** [2312.10741] - Residual style adaptor with UMLN
- **ConSinger** [2410.15342] - Consistency model for efficient synthesis
- **NaturalSpeech 2** [2304.09116] - Latent diffusion for zero-shot singing

### Technique Preservation
- **VibE-SVC** [2505.20794] - DWT-based vibrato extraction and transfer
- **TechSinger** [2502.12572] - Multi-technique controllable synthesis
- **FreeSVC** [2501.05586] - Multilingual zero-shot SVC with ECAPA2

### Speaker Adaptation
- **DS-TTS** [2506.01020] - Dual-style encoding for voice cloning
- **FlashSpeech** [2404.14700] - Latent consistency model for fast synthesis
- **Everyone-Can-Sing** [2501.13870] - Unified SVS/SVC with speech reference

## 6. Implementation Priorities

### High Priority (Phase 4)
1. LoRA adapter framework for per-profile fine-tuning
2. EWC regularization for catastrophic forgetting prevention
3. Training job scheduler with quality thresholds

### Medium Priority (Phase 5)
1. Vibrato detection using DWT decomposition
2. Technique-aware F0 extraction
3. Melisma/vocal run detection

### Lower Priority (Future)
1. Multi-technique control interface
2. Cross-lingual technique transfer
3. Real-time technique analysis feedback

---

_Generated for AutoVoice voice-profile-training track. Research conducted 2026-01-24._
