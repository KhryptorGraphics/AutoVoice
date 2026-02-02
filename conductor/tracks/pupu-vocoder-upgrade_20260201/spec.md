# Specification: Pupu-Vocoder Anti-Aliasing Upgrade

**Track ID:** pupu-vocoder-upgrade_20260201
**Created:** 2026-02-01
**Status:** [ ] Not Started
**Priority:** P2 - Nice to Have Quality Improvement
**Research:** [docs/sota-svc-research-2025.md](../../../docs/sota-svc-research-2025.md)

## Overview

Upgrade the Quality Pipeline vocoder from BigVGAN v2 to Pupu-Vocoder (or integrate anti-aliasing techniques from Pupu into BigVGAN). Pupu-Vocoder introduces anti-derivative anti-aliasing for neural audio synthesis, eliminating spectral artifacts that cause harsh/metallic sound.

**Source Paper:** arXiv:2512.20211 - "Pupu-Vocoder: Aliasing-free Neural Audio Synthesis"

## Problem Statement

Current BigVGAN vocoder (even v2) can introduce:
- High-frequency aliasing artifacts (especially >16kHz)
- Metallic/harsh timbre in some cases
- Spectral discontinuities at transients

Pupu-Vocoder addresses this via:
- Anti-derivative anti-aliasing (ADAA) for activation functions
- Aliasing-free upsampling filters
- Better preservation of high-frequency content

**Performance from paper:**
- Outperforms BigVGAN on singing/music/audio benchmarks
- Better PESQ scores on 44.1kHz synthesis
- Cleaner spectrograms with fewer artifacts

## User Story

As a music producer using AutoVoice for professional vocals, I want the highest fidelity voice synthesis with no audible artifacts, so that the converted vocals integrate seamlessly into my production without post-processing cleanup.

## Acceptance Criteria

### Quality Metrics
- [ ] PESQ ≥ 4.3 at 44.1kHz (vs BigVGAN ~4.1)
- [ ] Spectral flatness improved (fewer discontinuities)
- [ ] High-frequency aliasing reduced by ≥50% (measured via spectral analysis)
- [ ] Subjective MOS ≥ 4.2 for overall quality

### Integration
- [ ] Pupu-Vocoder available as alternative to BigVGAN
- [ ] Config flag: `vocoder_type: 'bigvgan' | 'pupu'`
- [ ] Both vocoders work with Seed-VC DiT output
- [ ] E2E tests pass with Pupu vocoder

### Performance
- [ ] Pupu-Vocoder inference speed comparable to BigVGAN (≤1.2x slower)
- [ ] GPU memory usage comparable (≤10% increase)
- [ ] Supports streaming mode for karaoke pipeline

## Technical Architecture

### Current Pipeline
```
Seed-VC DiT → 128-band mel (44kHz) → BigVGAN v2 → Audio
```

### Enhanced Pipeline
```
Seed-VC DiT → 128-band mel (44kHz) → Vocoder → Audio
                                         ↓
                                    [User selectable]
                                    ├─→ BigVGAN v2 (default)
                                    └─→ Pupu-Vocoder (anti-aliased)
```

## Implementation Plan

### Phase 1: Pupu-Vocoder Integration (P0)
**Files:**
- `src/auto_voice/models/pupu_vocoder.py` - Pupu-Vocoder wrapper
- `models/pupu-vocoder/` - Model weights and config

**Tasks:**
1. Clone Pupu-Vocoder repository
2. Download pretrained 44.1kHz weights
3. Create PupuVocoder class wrapping inference
4. Implement mel→waveform conversion
5. Test on Seed-VC outputs

### Phase 2: Pipeline Integration (P0)
**Files:**
- `src/auto_voice/inference/seed_vc_pipeline.py` - Add vocoder selection
- `scripts/quality_pipeline.py` - Config for vocoder type

**Tasks:**
1. Add `vocoder_type` parameter to QualityConfig
2. Lazy load Pupu or BigVGAN based on config
3. Ensure both vocoders handle same mel format
4. Test quality with both vocoders on fixtures

### Phase 3: Quality Comparison (P1)
**Files:**
- `tests/test_vocoder_comparison.py` - Compare BigVGAN vs Pupu
- `scripts/benchmark_vocoders.py` - Performance benchmarking

**Tasks:**
1. Compare PESQ: BigVGAN vs Pupu
2. Compare spectral flatness and aliasing
3. Benchmark inference speed
4. Collect subjective MOS ratings

### Phase 4: Web UI Integration (P2)
**Files:**
- `frontend/src/components/PipelineSelector.tsx` - Vocoder selector
- `src/auto_voice/web/api.py` - API parameter for vocoder

**Tasks:**
1. Add vocoder dropdown: "BigVGAN (fast)" vs "Pupu (anti-aliased)"
2. Pass vocoder_type to backend
3. Update quality comparison panel
4. Add tooltip explaining tradeoffs

## Dependencies

### Models
- **Pupu-Vocoder weights:** Download from official repository
- **BigVGAN v2:** Already integrated (fallback)
- **Seed-VC DiT:** Already implemented

### Research Code
- **Pupu-Vocoder repo:** (search on GitHub/arXiv for official implementation)
- May need to extract core ADAA modules and integrate into AutoVoice

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pupu-Vocoder slower than BigVGAN | Medium | Profile and optimize, make it optional |
| No pretrained 44.1kHz weights available | High | Train on singing dataset or use 22kHz version |
| Implementation complexity high | Medium | Start with simple integration, iterate |
| Quality improvement marginal | Low | A/B test extensively before full rollout |

## Technical Details

### Anti-Derivative Anti-Aliasing (ADAA)

**Problem:** Standard activation functions (tanh, LeakyReLU) introduce aliasing when applied to upsampled signals.

**Solution:** Use anti-derivative of activation function:
```python
def adaa_activation(x, activation='tanh'):
    """Anti-derivative anti-aliased activation.

    Standard: y = tanh(x)
    ADAA: y = antiderivative(tanh(x))
          then differentiate to get aliasing-free output
    """
    # Integrate activation function
    # Apply to signal
    # Differentiate (via finite differences)
    # Result: same shape as activation but aliasing-free
```

**Key activations with ADAA:**
- Tanh → anti-derivative: x * tanh(x) + log(cosh(x))
- ReLU → anti-derivative: 0.5 * x^2 for x > 0
- LeakyReLU → similar piecewise integration

### Aliasing-Free Upsampling

**Problem:** Transposed convolutions can introduce spectral imaging artifacts.

**Solution:** Use polyphase decomposition + low-pass filtering:
```python
def aliasing_free_upsample(x, scale=2):
    """Upsample without aliasing artifacts."""
    # Zero-pad in frequency domain
    # Apply ideal low-pass filter (cutoff at Nyquist/scale)
    # IFFT back to time domain
```

## Success Criteria

1. **Quality improvement verified**: PESQ ≥4.3 (vs BigVGAN ~4.1)
2. **Artifacts reduced**: High-frequency aliasing ≥50% reduction
3. **Performance acceptable**: ≤1.2x slower than BigVGAN
4. **Integration complete**: Config flag works, both vocoders supported
5. **Tests passing**: E2E tests with Pupu vocoder
6. **User preference**: A/B testing shows preference for Pupu output

## Out of Scope

- Training custom Pupu models (use pretrained)
- Pupu for Realtime Pipeline (quality-only feature)
- Hybrid BigVGAN+Pupu architectures
- ADAA for other components (DiT, encoders) - vocoder only

## Alternative Approach

If Pupu-Vocoder integration is complex, consider:

**Plan B: Anti-Aliased BigVGAN**
- Extract ADAA activation functions from Pupu paper
- Modify BigVGAN activation layers to use ADAA
- Keep BigVGAN architecture but with anti-aliased activations
- Lower integration complexity, similar benefits

Implementation:
```python
# Replace BigVGAN activations
from models.pupu_vocoder.adaa import ADAAActivation

# In BigVGAN generator
self.activation = ADAAActivation('snake')  # Instead of Snake()
```

## Completion Checklist

- [ ] Pupu-Vocoder repository cloned and explored
- [ ] Pretrained 44.1kHz weights downloaded
- [ ] PupuVocoder wrapper class implemented
- [ ] Quality Pipeline supports vocoder selection
- [ ] Config flag `vocoder_type` works
- [ ] Quality comparison shows improvement (PESQ ≥4.3)
- [ ] Performance benchmarks completed (≤1.2x slower)
- [ ] Spectral analysis confirms aliasing reduction
- [ ] E2E tests pass with Pupu vocoder
- [ ] Documentation updated with vocoder selection guide
- [ ] (Optional) Web UI has vocoder selector dropdown
