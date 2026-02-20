# Encoder Research: ContentVec, Conformer Variants, and Speaker Disentanglement

## Comprehensive findings for the AutoVoice singing voice conversion encoder pipeline.

---

## Table of Contents

1. [ContentVec: Speaker-Disentangled Content Features](#1-contentvec-speaker-disentangled-content-features)
2. [Conformer Encoder Variants](#2-conformer-encoder-variants)
3. [Speaker Disentanglement Techniques](#3-speaker-disentanglement-techniques)
4. [Implementation Recommendations for AutoVoice](#4-implementation-recommendations-for-autovoice)

---

## 1. ContentVec: Speaker-Disentangled Content Features

### 1.1 Paper Reference

- **Title:** ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers
- **ArXiv:** [2204.09224](https://arxiv.org/abs/2204.09224)
- **Published:** ICML 2022
- **Authors:** Kaizhi Qian et al.

### 1.2 Core Architecture

ContentVec has the same architecture as HuBERT Base:
- **7 temporal convolutional feature extraction blocks** (CNN encoder)
- **12 transformer layers** with model dimension 768
- **Final projection layer** projects 768 -> 256 dimensions

The key innovation is three disentanglement mechanisms applied during pre-training:
1. **Disentanglement in teachers:** Offline clustering uses voice-converted audio to generate speaker-invariant pseudo-labels
2. **Disentanglement in students:** SimCLR-inspired contrastive loss enforces timbre invariance
3. **Speaker conditioning:** Auxiliary speaker information prevents content degradation

### 1.3 The 256-Dimensional Feature Pipeline

ContentVec outputs come in two variants used in voice conversion:

| Variant | Layer | Dimensions | Config Key | Usage |
|---------|-------|-----------|------------|-------|
| vec256l9 | Layer 9 | 256 | `"ssl_dim": 256, "speech_encoder": "vec256l9"` | So-VITS-SVC 4.0, more disentangled |
| Vec768-Layer12 | Layer 12 | 768 | `"ssl_dim": 768` | So-VITS-SVC 4.1+, richer features |

The 256-dim output uses a **final projection layer** (`nn.Linear(768, 256)`) applied after layer 9 of the transformer stack. This bottleneck forces the model to retain only the most content-relevant information.

### 1.4 Loading checkpoint_best_legacy_500.pt via Fairseq

The "legacy" checkpoint contains only the representation module (no task head), loadable with plain fairseq:

```python
import fairseq
import torch

# Load via fairseq (legacy format)
ckpt_path = "models/pretrained/checkpoint_best_legacy_500.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
model.eval()

# Extract features from audio
# audio: [B, T] waveform at 16kHz
with torch.no_grad():
    # Get all hidden states
    results = model.extract_features(
        source=audio,
        padding_mask=None,
        output_layer=9  # Layer 9 for vec256l9
    )
    features = results["x"]  # [B, N_frames, 768]

    # Apply final projection for 256-dim output
    # The final_proj layer maps 768 -> 256
    content_features = model.final_proj(features)  # [B, N_frames, 256]
```

**Important PyTorch 2.6+ fix:** In `torch.load`, set `weights_only=False` to avoid deserialization errors:
```python
# In fairseq/checkpoint_utils.py, modify the torch.load call:
state = torch.load(path, map_location="cpu", weights_only=False)
```

### 1.5 HuggingFace Loading (Alternative)

The HuggingFace version (`lengyue233/content-vec-best`) provides a cleaner API:

```python
from transformers import HubertModel
import torch
import torch.nn as nn

class HubertModelWithFinalProj(HubertModel):
    """ContentVec with 256-dim final projection layer."""
    def __init__(self, config):
        super().__init__(config)
        # Projects hidden_size (768) -> classifier_proj_size (256)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

# Load pretrained
model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
model.eval()

# Extract features
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
with torch.no_grad():
    outputs = model(audio, output_hidden_states=True)

    # Option 1: Layer 9 + final projection (vec256l9)
    layer_9 = outputs.hidden_states[9]  # [B, N, 768]
    content_256 = model.final_proj(layer_9)  # [B, N, 256]

    # Option 2: Layer 12 raw output (768-dim)
    layer_12 = outputs.hidden_states[12]  # [B, N, 768]
```

### 1.6 Pretrained Model Availability

| Model | Source | Size | Notes |
|-------|--------|------|-------|
| checkpoint_best_legacy_500.pt | [ContentVec GitHub](https://github.com/auspicious3000/contentvec) | ~360MB | Fairseq format, 500 classes |
| checkpoint_best_legacy_500.pt | [HuggingFace wok000/vcclient_modules](https://huggingface.co/wok000/vcclient_modules) | ~360MB | Mirror |
| content-vec-best | [HuggingFace lengyue233](https://huggingface.co/lengyue233/content-vec-best) | ~360MB | HF Transformers format |
| hubert_base.pt | ContentVec GitHub | ~199MB | Smaller, same effect (rename to checkpoint_best_legacy_500.pt) |

### 1.7 Recent Improvements and Alternatives (2024-2026)

#### GenVC (2025) - arxiv:2502.04519
- Uses ContentVec as SSL feature extractor for phonetic tokens
- Simpler architecture without external supervised models
- Self-supervised zero-shot voice conversion

#### RT-VC (2025) - arxiv:2506.10289
- Identifies ContentVec as a "black-box" approach
- Proposes SPARC (Speech Articulatory Coding) as more interpretable alternative
- Content represented as vocal tract kinematics in speaker-agnostic space

#### vec2wav 2.0 (2024) - arxiv:2409.01995
- Notes that excessive speaker disentanglement in discrete tokens can harm prosody preservation
- Uses discrete token vocoders for higher quality

#### AVENet (2025) - arxiv:2504.05833
- Disentangles content by approximating average features across speakers
- Novel approach to evaluating timbre retention in SSL features

#### Simple Content Encoder for SVC (Interspeech 2025)
- SSL-embedding dimension reduction technique
- Direct bottleneck on SSL features specifically for singing voice conversion
- Reference: Zhou et al., "Simple and effective content encoder for singing voice conversion via SSL-embedding dimension reduction," Proc. Interspeech 2025

### 1.8 Layer Selection Guidelines

Research consensus on SSL model layer usage for voice conversion:

| Layers | Content | Observations |
|--------|---------|--------------|
| 1-4 | Acoustic/speaker features | High speaker identity information |
| 5-8 | Mixed content/speaker | Transitional |
| 9-10 | Content-dominant | Best for VC (vec256l9 uses layer 9) |
| 11-12 | High-level linguistic | Richer but more speaker leakage |

**Key finding (Dec 2024):** Voice identification accuracy: HuBERT 73.7%, ContentVec 37.7% -- confirming ContentVec's superior disentanglement (lower = better for VC content features).

---

## 2. Conformer Encoder Variants

### 2.1 Original Conformer

- **Paper:** [arxiv:2005.08100](https://arxiv.org/abs/2005.08100) (Gulati et al., 2020)
- **Key idea:** Combines CNN (local patterns) and Transformer (global dependencies)
- **Architecture:** Macaron-style FFN-Attention-Convolution-FFN per layer

### 2.2 E-Branchformer

- **Paper:** [arxiv:2210.00077](https://arxiv.org/abs/2210.00077) (Peng et al., 2022)
- **Implementation:** [ESPnet e_branchformer_encoder.py](https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/e_branchformer_encoder.py)

#### Architecture Details

```
Input -> [Branch 1: Self-Attention] -\
                                      -> Depth-wise Conv Merge -> Output
Input -> [Branch 2: Conv-Gating MLP] -/
```

Each E-Branchformer layer contains:
- `self_att`: Multi-headed attention (RelPos or standard)
- `feed_forward`: Macaron-style pre-FFN
- `feed_forward_macaron`: Post-FFN
- `conv_mod`: Convolutional gating MLP (cgMLP)
- `depthwise_conv_mod`: Merge mechanism (depth-wise convolution)

#### Configuration Specifications

| Config | Layers | Hidden Dim | Heads | cgMLP Kernel | Merge Kernel |
|--------|--------|-----------|-------|--------------|--------------|
| Base | 16 | 256 | 4 (d/64) | 31 | 15 |
| Large | 17 | 512 | 8 (d/64) | 31 | 15 |
| WSJ | 12 | 256 | 4 | 31 | 15 |

**Results:** 1.81% / 3.65% WER on LibriSpeech test-clean/test-other (SOTA without external data)

#### Supported Attention Types
- `MultiHeadedAttention` (absolute position)
- `RelPositionMultiHeadedAttention` (relative position)
- `LegacyRelPositionMultiHeadedAttention`
- `FastSelfAttention` (linear complexity)

### 2.3 Zipformer (ICLR 2024)

- **Paper:** [arxiv:2310.11230](https://arxiv.org/abs/2310.11230) (Yao et al., ICLR 2024)
- **Implementation:** [k2-fsa/icefall zipformer.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py)
- **Key advantage:** 50%+ FLOP reduction vs. Conformer with better or equal accuracy

#### Novel Components

**1. U-Net-like Multi-Resolution Structure:**
```
Stack 1: 50Hz   (2 layers, dim=192)
Stack 2: 25Hz   (2 layers, dim=256)  [downsample]
Stack 3: 12.5Hz (3 layers, dim=384)  [downsample]
Stack 4: 6.25Hz (4 layers, dim=512)  [downsample, center]
Stack 5: 12.5Hz (3 layers, dim=384)  [upsample]
Stack 6: 25Hz   (2 layers, dim=256)  [upsample]
```
Downsampling uses learnable weighted averaging; upsampling repeats frames.

**2. BiasNorm (replaces LayerNorm):**
```python
def bias_norm(x, bias, gamma, eps=1e-5):
    """BiasNorm: simpler than LayerNorm, retains length information."""
    # x: [B, T, C], bias: [C], gamma: scalar
    rms = torch.sqrt(((x - bias) ** 2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * torch.exp(gamma)
```
- Retains vector length information via learnable bias
- Prevents "dead" modules that LayerNorm can cause

**3. SwooshR and SwooshL Activations (replace Swish):**
```python
def swoosh_r(x):
    """Non-vanishing slopes for negative inputs."""
    return torch.log(1.0 + torch.exp(x - 1.0)) - 0.08 * x - 0.313261687

def swoosh_l(x):
    """Left-shifted variant."""
    return torch.log(1.0 + torch.exp(x - 4.0)) - 0.08 * x - 0.035
```
- Asymmetric: non-zero slopes for negative inputs prevent gradient death

**4. Attention Weight Reuse:**
```
Block Input -> MHAW (compute attention weights)
                |
                +-> NLA (Non-Linear Attention, reuses weights)
                +-> SA1 (Self-Attention 1, reuses weights)
                +-> SA2 (Self-Attention 2, reuses weights)
```
- Computes attention weights once, shared across 3 modules
- 2x more efficient per block

**5. ScaledAdam Optimizer:**
```python
# Update rule (parameter-scale-invariant):
# delta_t = -alpha_t * r_{t-1} * (1 - beta2^t)/(1 - beta1^t) * m_t / (sqrt(v_t) + eps)
# where r_{t-1} is the RMS of the parameter values
```

#### Zipformer Configuration Details

**Medium (default for LibriSpeech):**
```python
zipformer_config = {
    "num_encoder_layers": "2,2,3,4,3,2",
    "encoder_dim": "192,256,384,512,384,256",
    "num_heads": "4,4,4,8,4,4",
    "query_head_dim": 32,
    "value_head_dim": 12,
    "pos_head_dim": 4,
    "downsampling_factor": "1,2,4,8,4,2",
    "feedforward_dim": "512,768,1536,2048,1536,768",
    "cnn_module_kernel": "31,31,31,31,31,31",
}
```

**Large:**
```python
zipformer_large = {
    "num_encoder_layers": "2,2,4,5,4,2",
    "encoder_dim": "192,256,512,768,512,256",
    "encoder_unmasked_dim": "192,192,256,320,256,192",
    "feedforward_dim": "512,768,1536,2048,1536,768",
}
```

### 2.4 Splitformer (2025) - arxiv:2506.18035

- Adds parallel processing branches to Zipformer for early-exit on edge devices
- Parallel layers process downsampled inputs
- Improves ASR performance with minimal parameter increase

### 2.5 Position Encoding: RelPos vs. RoPE

#### Relative Position Encoding (Original Conformer)
- Default in Conformer, E-Branchformer
- Modifies attention scores with learnable relative position bias
- **Limitation:** Quartic time complexity, incompatible with Flash Attention

#### Rotary Position Embedding (RoPE) - arxiv:2107.05907, benchmarked 2025

- **Paper:** [arxiv:2501.06051](https://arxiv.org/abs/2501.06051) - "Benchmarking Rotary Position Embeddings for ASR" (Jan 2025)
- **Advantages:**
  - Linear time complexity (vs. quartic for RelPos)
  - Compatible with GPU-accelerated attention (Flash Attention)
  - Competitive or superior ASR accuracy
  - Better for variable-length sequences

```python
def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to queries and keys."""
    # x: [B, H, T, D]
    x_rot = x[..., :x.shape[-1]//2]
    x_pass = x[..., x.shape[-1]//2:]
    x_rot = torch.cat([
        x_rot * cos - x_pass * sin,
        x_pass * cos + x_rot * sin
    ], dim=-1)
    return x_rot
```

**Recommendation for AutoVoice:** Use RoPE for new implementations -- it's faster, compatible with Flash Attention, and shows competitive results per the Jan 2025 benchmark.

### 2.6 Conformer for Singing Voice Conversion (Specific Applications)

#### RobustSVC (2024) - arxiv:2409.06237
- Uses **Conformer encoder with CTC loss** to extract 256-dim BNFs (bottleneck features)
- Conformer combines attention (global) + CNN (local) for singing content
- BNFs from last hidden layer provide effective disentanglement

#### Karaoker-SSL (2024) - arxiv:2402.01520
- **4-layer Conformer** for pitch prediction:
  - 128-dim feedforward
  - 8 attention heads
  - Kernel size 15 for depthwise convolution

#### Voice Conversion with Conformer (2025) - arxiv:2506.08348
- Content encoder: Conformer blocks with **Instance Normalization** + AveragePooling1D
- Speaker encoder: Conformer blocks from MFA-Conformer **without** IN (preserves speaker info)

---

## 3. Speaker Disentanglement Techniques

### 3.1 Information Bottleneck Approaches

#### NaturalSpeech 3 / FACodec (2024) - arxiv:2403.03100

**Architecture: Factorized Codec (FACodec)**
```
Speech Encoder -> Timbre Extractor (global vector)
              |-> FVQ-Content (Factorized VQ for content)
              |-> FVQ-Prosody (Factorized VQ for prosody)
              |-> FVQ-Acoustic-Detail (Factorized VQ for details)
              -> Speech Decoder (reconstruction)
```

**Disentanglement Techniques:**
1. **Information Bottleneck:** Limits information through FVQ layers
2. **Gradient Reversal Layers (GRL):** Prevents feature leakage between subspaces
3. **Detail Dropout:** Randomly masks acoustic details during training

**Code:** [lifeiteng/naturalspeech3_facodec](https://github.com/lifeiteng/naturalspeech3_facodec)

#### FreeCodec (2024, Interspeech 2025) - arxiv:2412.01053

**Architecture:**
```
Input -> Content Encoder (50 Hz, WavLM-Large target) -> Content VQ
     |-> Prosody Encoder (long stride, lower rate)   -> Prosody VQ
     |-> Timbre Extractor (global vector)             -> Global embedding
     -> Decoder (reconstruction)
```

**Key results:**
- 0.45 kbps significantly outperforms FACodec at 2.4 kbps
- Uses WavLM-Large last layer as semantic learning target
- **Code:** [exercise-book-yq/FreeCodec](https://github.com/exercise-book-yq/FreeCodec)

#### VQ-VAE Information Bottleneck for VC

The VQ module acts as a discrete information bottleneck that forces the encoder to learn only phonetic content:

```python
class VQBottleneck(nn.Module):
    """Vector Quantization bottleneck for speaker disentanglement."""
    def __init__(self, dim=256, codebook_size=512, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, dim)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # z: [B, T, D]
        # Find nearest codebook entry
        distances = torch.cdist(z, self.embedding.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.embedding(indices)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Losses
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())

        return quantized_st, commitment_loss + self.commitment_cost * codebook_loss
```

**Typical bottleneck dimensions for VC:**
- Latent dim: 64 (aggressive bottleneck)
- Codebook size: 512-16384
- Lower dims = more disentanglement but less content preservation

### 3.2 Contrastive Learning Approaches

#### CTVC (2024) - arxiv:2401.08096

**Contrastive Similarity Loss:**
```python
def contrastive_similarity_loss(encoder, frames, phoneme_labels):
    """Push same-phoneme frames together, different-phoneme frames apart."""
    features = encoder(frames)  # [B, T, D]

    loss = 0
    for i in range(T):
        for j in range(T):
            cos_sim = F.cosine_similarity(features[:, i], features[:, j], dim=-1)
            if phoneme_labels[i] == phoneme_labels[j]:
                loss -= cos_sim  # Attract same phoneme
            else:
                loss += cos_sim  # Repel different phoneme
    return loss
```

**Time-Invariant Retrieval for Speaker:**
- Extracts speaker embedding from multiple random segments
- Maximizes mutual information between segments via InfoNCE:
```python
# InfoNCE for speaker consistency across segments
def info_nce_loss(seg1_emb, seg2_emb, temperature=0.07):
    logits = torch.matmul(seg1_emb, seg2_emb.T) / temperature
    labels = torch.arange(len(seg1_emb), device=logits.device)
    return F.cross_entropy(logits, labels)
```

**Combined Loss:** `L = L_recon + 0.01*L_sim + (-0.1)*L_speaker + 0.5*L_adv`

#### Discrete Unit Masking (2024) - arxiv:2409.11560

Novel approach that masks specific phonetic units before speaker encoding:

```python
class DiscreteUnitMasking:
    """Mask phonetic units to improve speaker-content disentanglement."""
    def __init__(self, mask_ratio=0.2, n_clusters=100):
        self.mask_ratio = mask_ratio
        self.n_clusters = n_clusters

    def apply(self, audio_features, unit_labels):
        # unit_labels: K-means cluster IDs from HuBERT (K=100)
        unique_units = unit_labels.unique()
        n_mask = int(len(unique_units) * self.mask_ratio)
        mask_units = unique_units[torch.randperm(len(unique_units))[:n_mask]]

        # Mask all frames belonging to selected units
        mask = torch.zeros_like(unit_labels, dtype=torch.bool)
        for u in mask_units:
            mask |= (unit_labels == u)

        # Zero out masked frames for speaker encoder input only
        masked_features = audio_features.clone()
        masked_features[mask] = 0
        return masked_features
```

**Results:** 50% relative WER reduction, 44% intelligibility improvement with TriAAN-VC at 20% masking.

### 3.3 Adversarial Training

#### Gradient Reversal Layer (GRL) for Disentanglement

```python
class GradientReversal(torch.autograd.Function):
    """Reverses gradients during backprop for adversarial training."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class SpeakerClassifier(nn.Module):
    """Adversarial speaker classifier with GRL."""
    def __init__(self, input_dim=256, n_speakers=100, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_speakers)
        )

    def forward(self, content_features):
        # Apply GRL - content encoder learns to fool this classifier
        reversed_features = GradientReversal.apply(content_features, self.alpha)
        return self.classifier(reversed_features)
```

**Used in:** W2VC, RobustSVC, MPFM-VC, NaturalSpeech 3

### 3.4 Singing-Specific Disentanglement

#### SaMoye (2024) - arxiv:2407.07728

**Multi-encoder content extraction with compression:**
```python
class SaMoyeContentEncoder:
    """Combines multiple ASR models and compresses to reduce timbre leakage."""
    def __init__(self):
        self.hubert_soft = load_hubert_soft()
        self.contentvec = load_contentvec()
        self.whisper = load_whisper_encoder()

    def extract(self, audio):
        # Extract from multiple models
        feat_hubert = self.hubert_soft(audio)    # [B, T, 256]
        feat_cv = self.contentvec(audio)          # [B, T, 256]
        feat_whisper = self.whisper(audio)         # [B, T, 512]

        # Concatenate
        combined = torch.cat([feat_hubert, feat_cv, feat_whisper], dim=-1)

        # Compress via VQ (reduces timbre leakage)
        compressed, vq_loss = self.vq_layer(combined)
        return compressed, vq_loss
```

**Timbre Enhancement via Unfreezing:**
- Unfreezes speaker encoder during SVC training (unlike TTS)
- Mixes speaker embedding with top-3 similar speakers from the dataset
- Finding: "Training the speaker encoder on SVC tasks makes a clearer difference between speaker embeddings from speech and singing"

**Code:** [CarlWangChina/SaMoye-SVC](https://github.com/CarlWangChina/SaMoye-SVC)

#### YingMusic-SVC (2025) - arxiv:2512.04793

- F0-aware timbre adaptor: refines global timbre embeddings into pitch-sensitive representations
- Energy-balance flow matching loss for high-frequency fidelity
- Addresses the unique challenge that timbre in singing varies with pitch

#### Everyone-Can-Sing (2025) - arxiv:2501.13870

- Assumes voice, content, and singing styles are naturally disentangled
- Uses pre-trained disentangled representations (speaker embedding + content embedding)
- Zero-shot SVC with speech reference

### 3.5 ECAPA-TDNN for Speaker Embedding

The standard speaker encoder for voice conversion systems:

```python
# Using SpeechBrain's pretrained ECAPA-TDNN
from speechbrain.pretrained import EncoderClassifier

speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}
)

# Extract 192-dim speaker embedding
embedding = speaker_encoder.encode_batch(audio)  # [B, 192]
```

**Architecture:**
- SE-Res2Net blocks with multi-scale feature aggregation
- Attentive statistical pooling
- Trained with AAM-Softmax loss on VoxCeleb2
- Output: 192-dim speaker embedding

**Limitation for singing:** ECAPA-TDNN trained on speech may not capture singing-specific timbre variations (pitch-dependent formant shifts). SaMoye addresses this by unfreezing during SVC training.

---

## 4. Implementation Recommendations for AutoVoice

### 4.1 Recommended Content Encoder Pipeline

Based on the research findings, the optimal content encoder for AutoVoice should:

```
Audio (16kHz) -> ContentVec (layer 9, 256-dim)
             -> [Optional: VQ Compression (codebook=512, reduce timbre)]
             -> Conformer Encoder (6 layers, 384-dim, RoPE)
             -> Content Features [B, T, 256]
```

**Key architectural choices:**

```python
# Recommended configuration for AutoVoice ContentEncoder
content_encoder_config = {
    # ContentVec backbone
    "ssl_model": "contentvec",
    "ssl_layer": 9,
    "ssl_dim": 256,

    # Optional VQ bottleneck for timbre reduction
    "use_vq_bottleneck": True,
    "vq_codebook_size": 512,
    "vq_dim": 256,

    # Conformer refinement
    "conformer_layers": 6,
    "conformer_hidden_dim": 384,
    "conformer_heads": 4,
    "conformer_ffn_dim": 1536,  # 4x expansion
    "conformer_kernel_size": 31,  # Large kernel for singing
    "position_encoding": "rope",  # RoPE for efficiency
    "activation": "swoosh_r",     # From Zipformer
}
```

### 4.2 Recommended Conformer Improvements

Upgrades to apply to the existing `conformer.py`:

1. **Replace LayerNorm with BiasNorm** (from Zipformer)
2. **Replace GELU with SwooshR/SwooshL** (non-vanishing negative slopes)
3. **Switch from RelPos to RoPE** (Flash Attention compatible)
4. **Add depthwise separable convolution module** (from original Conformer)
5. **Add Macaron-style FFN** (half-step FFN before and after attention)
6. **Consider multi-resolution processing** (Zipformer U-Net style) for long singing segments

### 4.3 Speaker Disentanglement Strategy

Recommended multi-pronged approach:

1. **ContentVec as base** (inherent disentanglement from pre-training)
2. **VQ bottleneck** after ContentVec (information bottleneck, codebook=512)
3. **GRL-based adversarial training** on content features (speaker classifier)
4. **Discrete unit masking** during speaker encoder training (20% mask ratio)
5. **Unfreeze speaker encoder** during SVC fine-tuning (SaMoye finding)

### 4.4 Training Loss Components

```python
total_loss = (
    reconstruction_loss          # Mel-spectrogram MSE
    + 0.01 * contrastive_sim     # Same-phoneme attraction (CTVC)
    + 0.5 * adversarial_loss     # GRL speaker classifier
    + 0.25 * vq_commitment       # VQ bottleneck regularization
    + 1.0 * pitch_loss           # F0 reconstruction
)
```

### 4.5 Pretrained Models Summary

| Component | Model | Source | Dim | Notes |
|-----------|-------|--------|-----|-------|
| Content | ContentVec (legacy 500) | [HuggingFace](https://huggingface.co/lengyue233/content-vec-best) | 256 | Layer 9 output |
| Speaker | ECAPA-TDNN | [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 192 | VoxCeleb2 trained |
| Pitch | RMVPE | Various | - | Robust F0 extraction for singing |
| Codec | FreeCodec | [GitHub](https://github.com/exercise-book-yq/FreeCodec) | Multi | Disentangled speech codec |
| Codec | FACodec | [GitHub](https://github.com/lifeiteng/naturalspeech3_facodec) | Multi | NaturalSpeech 3 |

---

## References

### Core Papers
1. ContentVec - [arxiv:2204.09224](https://arxiv.org/abs/2204.09224) - Qian et al., ICML 2022
2. Conformer - [arxiv:2005.08100](https://arxiv.org/abs/2005.08100) - Gulati et al., 2020
3. E-Branchformer - [arxiv:2210.00077](https://arxiv.org/abs/2210.00077) - Peng et al., 2022
4. Zipformer - [arxiv:2310.11230](https://arxiv.org/abs/2310.11230) - Yao et al., ICLR 2024

### Speaker Disentanglement
5. CTVC - [arxiv:2401.08096](https://arxiv.org/abs/2401.08096) - Contrastive Learning + Time-Invariant Retrieval, 2024
6. NaturalSpeech 3 / FACodec - [arxiv:2403.03100](https://arxiv.org/abs/2403.03100) - Ju et al., 2024
7. FreeCodec - [arxiv:2412.01053](https://arxiv.org/abs/2412.01053) - Zheng et al., Interspeech 2025
8. Discrete Unit Masking - [arxiv:2409.11560](https://arxiv.org/abs/2409.11560) - 2024
9. SaMoye - [arxiv:2407.07728](https://arxiv.org/abs/2407.07728) - Wang et al., 2024

### Singing Voice Conversion
10. RobustSVC - [arxiv:2409.06237](https://arxiv.org/abs/2409.06237) - 2024
11. YingMusic-SVC - [arxiv:2512.04793](https://arxiv.org/abs/2512.04793) - 2025
12. Everyone-Can-Sing - [arxiv:2501.13870](https://arxiv.org/abs/2501.13870) - 2025
13. Singing Voice Conversion Challenge 2025 - [arxiv:2509.15629](https://arxiv.org/abs/2509.15629)

### Position Encoding
14. RoPE for Conformer - [arxiv:2107.05907](https://arxiv.org/abs/2107.05907) - 2021
15. RoPE Benchmark for ASR - [arxiv:2501.06051](https://arxiv.org/abs/2501.06051) - Jan 2025

### Voice Conversion Systems
16. GenVC - [arxiv:2502.04519](https://arxiv.org/abs/2502.04519) - 2025
17. RT-VC - [arxiv:2506.10289](https://arxiv.org/abs/2506.10289) - 2025
18. vec2wav 2.0 - [arxiv:2409.01995](https://arxiv.org/abs/2409.01995) - 2024
19. W2VC (WavLM-based) - [SpringerOpen](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-023-00312-8) - 2023
20. AVENet - [arxiv:2504.05833](https://arxiv.org/abs/2504.05833) - 2025
21. Unified Voice/Accent Conversion - [arxiv:2412.08312](https://arxiv.org/abs/2412.08312) - Dec 2024

### Implementation Resources
22. [ContentVec GitHub](https://github.com/auspicious3000/contentvec)
23. [k2-fsa/icefall Zipformer](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py)
24. [ESPnet E-Branchformer](https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/e_branchformer_encoder.py)
25. [SaMoye-SVC](https://github.com/CarlWangChina/SaMoye-SVC)
26. [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
27. [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
