# Specification: ECAPA2 Speaker Encoder Upgrade

**Track ID:** ecapa2-speaker-encoder_20260201
**Created:** 2026-02-01
**Status:** [ ] Not Started
**Priority:** P2 - Quality Improvement
**Research:** [docs/sota-svc-research-2025.md](../../../docs/sota-svc-research-2025.md)

## Overview

Upgrade the speaker style encoder from CAMPPlus to ECAPA2 (Enhanced Context Attentive Pooling, version 2) for better zero-shot voice conversion and speaker embedding quality. ECAPA2 provides improved speaker discrimination and robustness to noise/background music.

**Source Paper:** FreeSVC (arXiv:2501.05586) - Multilingual SVC with ECAPA2 speaker encoder

## Problem Statement

Current CAMPPlus encoder (192-dim) has limitations:
- Trained primarily on Mandarin Chinese speech
- Less robust to background music in singing vocals
- Limited multilingual support
- Lower speaker discrimination in challenging cases

ECAPA2 addresses this via:
- Better context modeling with attention mechanisms
- Robust to noise and background music
- Multilingual training (better zero-shot generalization)
- Higher speaker discrimination (ECAPA2 > CAMPPlus > ECAPA)

## User Story

As a user converting vocals from diverse artists and languages, I want the speaker encoder to accurately capture vocal characteristics even with background music and diverse languages, so that voice conversion works reliably on real-world music tracks.

## Acceptance Criteria

### Quality Metrics
- [ ] Speaker similarity ≥ 0.92 (improvement from CAMPPlus ~0.87)
- [ ] EER (Equal Error Rate) ≤ 2.0% for speaker verification
- [ ] Robust to SNR down to 5dB (background music tolerance)
- [ ] Works on multilingual vocals (English, Spanish, Mandarin, Korean)

### Integration
- [ ] ECAPA2 encoder integrated into Seed-VC pipeline
- [ ] Config flag: `speaker_encoder: 'campplus' | 'ecapa2'`
- [ ] Both encoders produce 192-dim embeddings (backward compatible)
- [ ] E2E tests pass with ECAPA2 encoder

### Performance
- [ ] ECAPA2 inference time ≤ 1.5x CAMPPlus
- [ ] GPU memory increase ≤ 500MB
- [ ] Compatible with existing voice profiles

## Technical Architecture

### Current Pipeline
```
Reference Audio → Mel Spectrogram → CAMPPlus → 192-dim Style Embedding
```

### Enhanced Pipeline
```
Reference Audio → Mel Spectrogram → Speaker Encoder → 192-dim Style Embedding
                                         ↓
                                    [User selectable]
                                    ├─→ CAMPPlus (default)
                                    └─→ ECAPA2 (robust)
```

## Implementation Plan

### Phase 1: ECAPA2 Integration (P0)
**Files:**
- `src/auto_voice/models/ecapa2_encoder.py` - ECAPA2 wrapper
- `models/ecapa2/` - Model weights

**Tasks:**
1. Download ECAPA2 pretrained weights from FreeSVC
2. Create ECAPA2Encoder class
3. Ensure output is 192-dim (project if needed)
4. Test on voice profile fixtures

### Phase 2: Pipeline Integration (P0)
**Files:**
- `src/auto_voice/inference/seed_vc_pipeline.py` - Add encoder selection
- `scripts/quality_pipeline.py` - Config for speaker encoder

**Tasks:**
1. Add `speaker_encoder` parameter to QualityConfig
2. Lazy load CAMPPlus or ECAPA2 based on config
3. Ensure backward compatibility with existing profiles
4. Test with William/Conor voice profiles

### Phase 3: Voice Profile Migration (P1)
**Files:**
- `scripts/migrate_speaker_embeddings.py` - Re-encode profiles with ECAPA2
- `src/auto_voice/storage/voice_profiles.py` - Support dual embeddings

**Tasks:**
1. Re-encode existing profiles with ECAPA2
2. Store both CAMPPlus and ECAPA2 embeddings
3. Auto-select encoder based on profile metadata
4. Test conversion quality with re-encoded profiles

### Phase 4: Evaluation (P1)
**Files:**
- `tests/test_speaker_encoder_comparison.py` - CAMPPlus vs ECAPA2
- `scripts/evaluate_speaker_embeddings.py` - Quality metrics

**Tasks:**
1. Compare speaker similarity: CAMPPlus vs ECAPA2
2. Measure EER on speaker verification task
3. Test robustness to background music
4. Benchmark inference speed

## Dependencies

### Models
- **ECAPA2 weights:** From FreeSVC or speechbrain
- **Seed-VC DiT:** Already implemented
- **Voice profiles:** Existing CAMPPlus embeddings

### Research Code
- **FreeSVC:** https://github.com/freesvc/freesvc
- **SpeechBrain:** https://github.com/speechbrain/speechbrain (ECAPA2 implementation)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ECAPA2 slower than CAMPPlus | Medium | Profile and optimize, make it optional |
| Embedding dimension mismatch | High | Project to 192-dim if needed |
| Existing profiles incompatible | Medium | Store dual embeddings, auto-migrate |
| Quality improvement marginal | Low | A/B test extensively on diverse vocals |

## Technical Details

### ECAPA2 Architecture

**Key improvements over ECAPA:**
- Deeper channel attention layers
- Multi-scale feature aggregation
- Attentive statistical pooling
- Residual connections for better gradient flow

**Architecture:**
```
Input (80-dim mel) → Frame-level features (1D Conv + Res2Net blocks)
                   → Channel attention
                   → Temporal context (SE-Res2Net blocks)
                   → Attentive statistical pooling
                   → Fully connected → 192-dim embedding
```

**Training:**
- AAM-Softmax loss for speaker discrimination
- Trained on VoxCeleb + multilingual speech datasets
- Robust to noise via augmentation during training

### Backward Compatibility

To ensure existing voice profiles work:
```python
class SpeakerEncoderManager:
    """Manage multiple speaker encoders."""

    def __init__(self):
        self.campplus = None  # Lazy loaded
        self.ecapa2 = None    # Lazy loaded

    def encode(self, audio: np.ndarray, encoder_type: str = 'auto'):
        """Encode audio with specified encoder.

        Args:
            audio: Audio waveform
            encoder_type: 'campplus', 'ecapa2', or 'auto'

        Returns:
            embedding: 192-dim speaker embedding
        """
        if encoder_type == 'auto':
            # Auto-detect from profile metadata
            encoder_type = self.detect_encoder_type()

        if encoder_type == 'ecapa2':
            return self.ecapa2.encode(audio)
        else:
            return self.campplus.encode(audio)
```

## Success Criteria

1. **Quality improvement verified**: Speaker similarity ≥0.92 (vs CAMPPlus ~0.87)
2. **Robustness improved**: Works at SNR ≥5dB
3. **Multilingual support**: English, Spanish, Mandarin, Korean
4. **Performance acceptable**: ≤1.5x slower than CAMPPlus
5. **Tests passing**: E2E tests with ECAPA2 encoder
6. **Profiles migrated**: Dual embeddings stored for all existing profiles

## Out of Scope

- Training custom ECAPA2 models (use pretrained)
- ECAPA2 for Realtime Pipeline (quality-only feature)
- Multi-encoder ensemble (single encoder selection only)
- Speaker diarization with ECAPA2 (separate track)

## Alternative Approach

If ECAPA2 integration is complex:

**Plan B: SpeechBrain ECAPA-TDNN**
- Use SpeechBrain's pretrained ECAPA-TDNN (original version)
- Easier integration via pip install speechbrain
- Similar benefits, slightly lower performance than ECAPA2
- Well-documented API

Implementation:
```python
from speechbrain.pretrained import EncoderClassifier

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/speechbrain_ecapa"
)
embedding = encoder.encode_batch(audio_tensor)
```

## Completion Checklist

- [ ] ECAPA2 weights downloaded
- [ ] ECAPA2Encoder class implemented
- [ ] Output dimensionality matches CAMPPlus (192-dim)
- [ ] Quality Pipeline supports encoder selection
- [ ] Config flag `speaker_encoder` works
- [ ] Existing voice profiles re-encoded with ECAPA2
- [ ] Quality comparison shows improvement (speaker sim ≥0.92)
- [ ] Robustness tested with background music (SNR ≥5dB)
- [ ] Performance benchmarks completed (≤1.5x slower)
- [ ] E2E tests pass with ECAPA2 encoder
- [ ] Documentation updated with encoder selection guide
