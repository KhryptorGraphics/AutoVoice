# Specification: HQ-SVC Voice Enhancement & Super-Resolution

**Track ID:** hq-svc-enhancement_20260201
**Created:** 2026-02-01
**Status:** [ ] Not Started
**Priority:** P1 - Important Quality Upgrade
**Research:** [docs/sota-svc-research-2025.md](../../../docs/sota-svc-research-2025.md)

## Overview

Integrate HQ-SVC (High-Quality Singing Voice Conversion) as a post-processing enhancement layer for the Quality Pipeline. HQ-SVC provides:
- Decoupled codec architecture for better separation of content and style
- Voice super-resolution (22kHz → 44.1kHz upsampling)
- DSP refinement for artifact reduction
- Progressive diffusion enhancement

**Source Paper:** arXiv:2511.08496 (AAAI 2026)

## Problem Statement

Current Quality Pipeline (Seed-VC) achieves:
- Speaker similarity: ~0.85-0.90
- MCD: 4.0-4.5
- Some artifacts in challenging cases (breathy vocals, high notes)

HQ-SVC research shows:
- Speaker similarity: 0.95
- MCD: 3.52
- Better handling of vocal artifacts
- Superior super-resolution quality

## User Story

As a music producer using AutoVoice for professional recordings, I want the highest possible quality voice conversion with minimal artifacts, so that the converted vocals sound indistinguishable from the target artist's natural voice.

## Acceptance Criteria

### Quality Metrics
- [ ] Speaker similarity ≥ 0.94 (improvement from current ~0.87)
- [ ] MCD ≤ 3.6 (improvement from current ~4.2)
- [ ] Super-resolution quality: PESQ ≥ 4.0 for 22→44.1kHz
- [ ] Artifact reduction: 30% fewer spectral discontinuities

### Integration
- [ ] HQ-SVC enhancement available as optional post-processing in Quality Pipeline
- [ ] Web UI checkbox: "Enable HQ Enhancement (slower, higher quality)"
- [ ] Pipeline routes correctly: Seed-VC → HQ-SVC → output
- [ ] Fallback to Seed-VC-only if HQ-SVC fails
- [ ] E2E tests pass with HQ enhancement enabled

### Performance
- [ ] HQ-SVC adds ≤2x processing time to Seed-VC baseline
- [ ] GPU memory increase ≤3GB (within Thor's 122GB limit)
- [ ] Supports same audio lengths as base pipeline (up to 5 minutes)

## Technical Architecture

### Base Pipeline (Current)
```
Audio → Whisper → Seed-VC DiT (CFM) → BigVGAN (44kHz) → Output
        (16kHz)   (5-10 steps)         (vocoder)
```

### Enhanced Pipeline (New)
```
Audio → Whisper → Seed-VC DiT (CFM) → HQ-SVC Enhancement → Output
        (16kHz)   (5-10 steps)         │
                                       ├─→ Decoupled Codec
                                       ├─→ Diffusion Refinement
                                       └─→ DSP Post-Processing
                                            (44.1kHz)
```

## Implementation Plan

### Phase 1: HQ-SVC Model Integration (P0)
**Files:**
- `src/auto_voice/inference/hq_svc_wrapper.py` - HQ-SVC inference wrapper
- `models/hq-svc/` - Model weights and config (already cloned)

**Tasks:**
1. Create HQSVCEnhancer class wrapping HQ-SVC model
2. Load pretrained weights from `models/hq-svc/`
3. Implement enhance() method: mel_spec → enhanced_mel
4. Add super-resolution mode (22→44.1kHz)

### Phase 2: Pipeline Integration (P0)
**Files:**
- `src/auto_voice/inference/seed_vc_pipeline.py` - Modify to support HQ enhancement
- `src/auto_voice/inference/pipeline_factory.py` - Add 'quality_seedvc_hq' option

**Tasks:**
1. Add `enable_hq_enhancement` parameter to SeedVCPipeline
2. Integrate HQSVCEnhancer after Seed-VC DiT output
3. Handle dual-mode: with/without enhancement
4. Test memory footprint and performance

### Phase 3: Web UI Integration (P1)
**Files:**
- `frontend/src/pages/ConvertPage.tsx` - Add HQ enhancement checkbox
- `frontend/src/components/PipelineSelector.tsx` - Show HQ option for Quality pipeline
- `src/auto_voice/web/api.py` - Add `enable_hq_enhancement` parameter

**Tasks:**
1. Add UI toggle: "Enable HQ Enhancement (2x slower)"
2. Pass enhancement flag to backend API
3. Update quality comparison panel to show HQ vs non-HQ
4. Add tooltip explaining quality/speed tradeoff

### Phase 4: Evaluation & Benchmarking (P1)
**Files:**
- `tests/test_hq_enhancement_quality.py` - Quality metrics comparison
- `scripts/benchmark_hq_enhancement.py` - Performance benchmarking

**Tasks:**
1. Compare speaker similarity: Seed-VC vs Seed-VC+HQ
2. Compare MCD: Seed-VC vs Seed-VC+HQ
3. Benchmark inference time overhead
4. Generate quality comparison samples

## Dependencies

### Models
- **HQ-SVC weights:** `models/hq-svc/` (already cloned)
- **Seed-VC:** Already implemented in quality_pipeline.py
- **BigVGAN:** Already integrated

### Infrastructure
- PipelineFactory (implemented)
- QualityMetrics evaluator (implemented)
- Frontend PipelineSelector (implemented)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| HQ-SVC too slow (>5x overhead) | High | Make enhancement optional, profile and optimize |
| Memory pressure on Thor GPU | Medium | Lazy loading, unload Seed-VC after HQ phase |
| Quality degradation in some cases | High | Fallback to Seed-VC-only, extensive testing |
| Model compatibility issues | Medium | Version pin dependencies, test on fixtures |

## Success Criteria

1. **Quality improvement verified**: Speaker similarity ≥0.94, MCD ≤3.6
2. **Performance acceptable**: ≤2x slower than Seed-VC baseline
3. **Integration complete**: Web UI toggle works, API accepts parameter
4. **Tests passing**: E2E tests with HQ enhancement enabled
5. **User feedback positive**: A/B testing shows preference for HQ output

## Out of Scope

- Training custom HQ-SVC models (use pretrained)
- HQ enhancement for Realtime Pipeline (quality-only feature)
- Multi-stage progressive enhancement (single-pass only)
- Voice super-resolution beyond 44.1kHz (48kHz+)

## Reference Implementation

HQ-SVC repository structure:
```
models/hq-svc/
├── gradio_app.py           # Reference implementation
├── inference.py            # Core inference logic
├── utils/
│   ├── models/
│   │   ├── diffusion.py    # Diffusion enhancement
│   │   └── models_v2_beta.py  # Codec architecture
│   └── ddsp/
│       └── vocoder.py      # DSP post-processing
```

Key classes to integrate:
- `utils.models.diffusion.DiffusionSVC` - Main enhancement model
- `utils.models.models_v2_beta.SynthesizerTrn` - Codec model
- `utils.ddsp.vocoder.CombSubMinimumNoisedPhase` - DSP refinement

## Completion Checklist

- [ ] HQSVCEnhancer class implemented and tested
- [ ] Seed-VC pipeline supports optional HQ enhancement
- [ ] PipelineFactory routes 'quality_seedvc_hq' correctly
- [ ] Web UI has HQ enhancement toggle
- [ ] API accepts and processes enable_hq_enhancement parameter
- [ ] Quality metrics show improvement (speaker sim ≥0.94, MCD ≤3.6)
- [ ] Performance benchmarks completed (overhead ≤2x)
- [ ] E2E tests pass with HQ enhancement
- [ ] Documentation updated with HQ enhancement details
- [ ] User guide includes HQ enhancement usage instructions
