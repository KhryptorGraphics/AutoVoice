# Background Vocal Separation Experiments - Updated Results

## Experiment 1: Diarization Accuracy ✅ COMPLETED

**Result**: Diarization correctly detects backing vocals on tracks that have them:
- **Hotline Bling**: 2 speakers (95% lead / 5% backing)
- **Cry Me A River**: 2 speakers (99% lead / 1% backing)
- Other tracks: Single speaker (solo covers)

**Key Finding**: Diarization works on Demucs vocal stems! Segments align with musical phrases.

**Blocker**: Embedding incompatibility between diarization (wavlm-base-sv, 512-dim) and profile matching (wavlm-base-plus, 256-dim).

---

## Experiment 2: Per-Speaker Source Separation Quality 🔄 IN PROGRESS

**Next**: Test extracting individual speaker tracks using diarization segments + Demucs or masking.

---

## Experiment 3: Conversion Quality on Isolated Tracks 🔄 PENDING

**Next**: Convert each speaker track independently through svc_fork_bridge and recombine.

---

## Experiment 4: Recombination Strategy 🔄 PENDING

**Next**: Test cross-fade vs volume-matched recombination.

---

## Experiment 5: End-to-End Latency 🔄 PENDING

---

## Recommended Fix for Embedding Compatibility

### Option A: Use wavlm-base-plus for Diarization (Recommended)
Modify SpeakerDiarizer to extract embeddings from `last_hidden_state` (mean pooled) instead of XVector head. This gives 768-dim embeddings that can be truncated to 256-dim to match profiles.

```python
# In SpeakerDiarizer._load_model():
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
self._model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
# In embedding extraction:
hidden_states = outputs.last_hidden_state
embedding = hidden_states.mean(dim=1).squeeze()  # [768]
embedding = embedding[:256]  # Truncate to match profiles
```

### Option B: Re-extract Profile Embeddings with wavlm-base-sv
Re-process all training data through wavlm-base-sv to get 512-dim profile embeddings.

### Option C: Learn Projection Layer
Train a linear projection from 512-256 using paired embeddings.

**Recommendation**: Option A (simplest, maintains profile compatibility).

---

## Next Actions

1. **Fix embedding compatibility** (Option A - ~2 hours)
2. **Run Experiment 2** - Extract per-speaker vocal tracks using diarization segments
3. **Run Experiment 3** - Convert each track independently through svc_fork_bridge
4. **Run Experiment 4** - Test recombination strategies