# Specification: Neural Source Filter (NSF) Harmonic Modeling

**Track ID:** nsf-harmonic-modeling_20260201
**Created:** 2026-02-01
**Status:** [ ] Not Started
**Priority:** P1 - Important Quality Upgrade
**Research:** [docs/sota-svc-research-2025.md](../../../docs/sota-svc-research-2025.md)

## Overview

Integrate Neural Source Filter (NSF) technology for explicit harmonic/noise separation in singing voice conversion. NSF provides better pitch accuracy, naturalness, and control over voice characteristics by explicitly modeling the harmonic and noise components of speech.

**Source Papers:**
- **SiFiSinger** (ICASSP 2024): End-to-end SVS with mcep decoupling
- **R2-SVC** (2510.20677): NSF for explicit harmonic/noise separation
- **FIRNet** (2024): Source-filter vocoder for TTS→SVS transfer

## Problem Statement

Current pipelines treat audio as a monolithic signal:
- No explicit separation of harmonic (pitched) vs noise (breathiness, consonants)
- Pitch errors propagate through entire synthesis
- Difficult to control naturalness vs artifact balance

NSF addresses this by:
- Explicit harmonic/noise source modeling
- Differentiable mcep (mel-cepstrum) and F0 losses
- Better singing voice naturalness (especially breathy vocals, vibrato)

## User Story

As a user converting singing vocals, I want the converted voice to preserve the natural characteristics of singing (vibrato, breathiness, vocal techniques) while maintaining pitch accuracy, so that the output sounds like a real human singing performance.

## Acceptance Criteria

### Quality Metrics
- [ ] Pitch RMSE ≤ 15 cents (improvement from current ~20-25 cents)
- [ ] Harmonic coherence ≥ 0.90 (measured via spectral harmonicity)
- [ ] Naturalness MOS ≥ 4.0 (subjective evaluation)
- [ ] Vibrato preservation: F0 variance matches source ±10%

### Integration
- [ ] NSF module available as optional component in Quality Pipeline
- [ ] Can be enabled/disabled via config flag
- [ ] Compatible with existing Seed-VC DiT architecture
- [ ] E2E tests pass with NSF enabled

### Performance
- [ ] NSF adds ≤30% processing time overhead
- [ ] GPU memory increase ≤2GB
- [ ] Supports real-time factor (RTF) ≤ 1.5 for Quality Pipeline

## Technical Architecture

### Without NSF (Current)
```
Audio → Content Encoder → DiT Decoder → Vocoder → Output
        (monolithic)       (monolithic)  (BigVGAN)
```

### With NSF (Enhanced)
```
Audio → Content Encoder → F0 Extractor (RMVPE)
                ↓               ↓
            DiT Decoder  →  NSF Module
                            ├─→ Harmonic Generator (F0-driven sine waves)
                            ├─→ Noise Generator (filtered noise)
                            └─→ Source Filter (combine with mcep envelope)
                                    ↓
                                Vocoder → Output
                                (BigVGAN)
```

## Implementation Plan

### Phase 1: NSF Module Implementation (P0)
**Files:**
- `src/auto_voice/models/nsf_module.py` - NSF harmonic/noise generator
- `src/auto_voice/models/source_filter.py` - Source-filter model

**Tasks:**
1. Implement harmonic generator (F0-driven sinusoids)
2. Implement noise generator (filtered white noise)
3. Implement mcep envelope extraction
4. Implement source-filter combination
5. Add differentiable F0 and mcep losses

### Phase 2: Pipeline Integration (P0)
**Files:**
- `src/auto_voice/inference/seed_vc_pipeline.py` - Add NSF processing
- `scripts/quality_pipeline.py` - Config flag for NSF

**Tasks:**
1. Insert NSF module after Seed-VC DiT decoder
2. Route mel spectrogram through NSF before vocoder
3. Add config parameter: `enable_nsf_modeling`
4. Test with/without NSF on singing fixtures

### Phase 3: Training Support (P1)
**Files:**
- `src/auto_voice/training/trainer.py` - Add NSF losses
- `config/training_config.yaml` - NSF training parameters

**Tasks:**
1. Add mcep loss to training objective
2. Add F0 loss to training objective
3. Add harmonic coherence regularization
4. Test on William/Conor voice models

### Phase 4: Evaluation & Tuning (P1)
**Files:**
- `tests/test_nsf_quality.py` - NSF quality metrics
- `scripts/evaluate_nsf_naturalness.py` - Subjective evaluation

**Tasks:**
1. Compare pitch accuracy: baseline vs NSF
2. Measure harmonic coherence improvement
3. Collect naturalness MOS ratings
4. Tune mcep/F0 loss weights for best quality

## Dependencies

### Models
- **RMVPE:** Already integrated for F0 extraction
- **Seed-VC DiT:** Already implemented
- **BigVGAN vocoder:** Already integrated

### Research Code
- **SiFiSinger:** https://github.com/sifisinger/sifisinger (reference implementation)
- **NSF-HiFiGAN:** https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| NSF too slow for practical use | High | Profile and optimize, make optional |
| Harmonic artifacts in output | Medium | Tune harmonic/noise balance, extensive testing |
| Incompatible with trained adapters | Medium | Design adapter-compatible architecture |
| Mcep extraction unstable | Medium | Use stable libraries (pyworld, librosa) |

## Technical Details

### NSF Architecture Components

**1. Harmonic Generator**
```python
def generate_harmonics(f0: torch.Tensor, num_harmonics: int = 8) -> torch.Tensor:
    """Generate harmonic sinusoids from F0.

    Args:
        f0: [B, T] fundamental frequency contour
        num_harmonics: number of harmonics to generate

    Returns:
        harmonics: [B, T, num_harmonics] harmonic signals
    """
    # f0 -> phase accumulation -> sine waves for each harmonic
```

**2. Noise Generator**
```python
def generate_noise(aperiodicity: torch.Tensor, sr: int) -> torch.Tensor:
    """Generate filtered noise component.

    Args:
        aperiodicity: [B, T, F] aperiodicity spectrum
        sr: sample rate

    Returns:
        noise: [B, T] filtered noise signal
    """
    # white noise -> spectral filtering via aperiodicity
```

**3. Source-Filter Combination**
```python
def apply_source_filter(
    source: torch.Tensor,  # harmonics + noise
    mcep: torch.Tensor,    # mel-cepstrum envelope
) -> torch.Tensor:
    """Apply source-filter model.

    Args:
        source: [B, T] combined harmonic and noise source
        mcep: [B, T, mcep_dim] spectral envelope

    Returns:
        filtered: [B, T] filtered output
    """
    # source * envelope (via LPC or FFT-based filtering)
```

### Loss Functions

**Mcep Loss (Spectral Envelope)**
```python
mcep_loss = F.l1_loss(predicted_mcep, target_mcep)
```

**F0 Loss (Pitch Accuracy)**
```python
f0_loss = F.mse_loss(predicted_f0, target_f0, reduction='none')
f0_loss = f0_loss[voiced_mask].mean()  # Only on voiced frames
```

**Harmonic Coherence Regularization**
```python
harmonic_coherence = compute_spectral_harmonicity(output)
coherence_loss = -harmonic_coherence.mean()  # Maximize
```

## Success Criteria

1. **Pitch accuracy improved**: RMSE ≤15 cents (vs current ~22 cents)
2. **Naturalness improved**: Subjective MOS ≥4.0
3. **Vibrato preserved**: F0 variance ±10% of source
4. **Performance acceptable**: Overhead ≤30%
5. **Tests passing**: E2E tests with NSF enabled
6. **Compatible with adapters**: Trained LoRAs work with NSF

## Out of Scope

- NSF for Realtime Pipeline (quality-only feature)
- Multi-speaker NSF training (use pretrained F0/mcep)
- NSF-based vocoders (NSF-HiFiGAN) - use existing BigVGAN
- Voice style transfer via NSF parameters

## Reference Implementation

SiFiSinger architecture:
```
models/sifisinger/
├── modules/
│   ├── nsf.py              # NSF harmonic generator
│   ├── source_filter.py    # Source-filter model
│   └── mcep.py             # Mcep extraction
└── losses/
    ├── f0_loss.py          # Differentiable F0 loss
    └── mcep_loss.py        # Differentiable mcep loss
```

Key integration points:
- Extract F0 using existing RMVPE
- Extract mcep using pyworld or differentiable approximation
- Generate harmonics + noise via NSF
- Apply source-filter before vocoder
- Add mcep + F0 losses to training

## Completion Checklist

- [ ] NSF harmonic generator implemented
- [ ] NSF noise generator implemented
- [ ] Source-filter combination implemented
- [ ] Mcep extraction integrated
- [ ] F0 loss added to training
- [ ] Mcep loss added to training
- [ ] Seed-VC pipeline supports NSF module
- [ ] Config flag `enable_nsf_modeling` works
- [ ] Quality metrics show improvement (pitch RMSE ≤15 cents)
- [ ] Performance overhead acceptable (≤30%)
- [ ] E2E tests pass with NSF
- [ ] Documentation updated with NSF details
