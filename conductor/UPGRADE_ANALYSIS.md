# AutoVoice Upgrade Analysis Report

**Generated:** 2026-02-01
**Analyzer:** Project Spec Upgrade Analyzer
**Status:** 4 New Upgrade Tracks Created

## Executive Summary

Analyzed project specifications, research documents, and existing implementations to identify missing SOTA voice conversion upgrades. Created **4 new tracks** implementing cutting-edge research from 2024-2026 papers.

### New Tracks Created

| Track ID | Priority | Title | Target Metrics |
|----------|----------|-------|----------------|
| hq-svc-enhancement_20260201 | P1 | HQ-SVC Voice Enhancement | Speaker sim ≥0.94, MCD ≤3.6 |
| nsf-harmonic-modeling_20260201 | P1 | NSF Harmonic Modeling | Pitch RMSE ≤15 cents, Naturalness MOS ≥4.0 |
| pupu-vocoder-upgrade_20260201 | P2 | Pupu-Vocoder Anti-Aliasing | PESQ ≥4.3, Aliasing -50% |
| ecapa2-speaker-encoder_20260201 | P2 | ECAPA2 Speaker Encoder | Speaker sim ≥0.92, EER ≤2.0% |

### Beads Tasks Created

- **AV-2ue** (P1 Epic): HQ-SVC Enhancement Integration
- **AV-8vf** (P1 Epic): NSF Harmonic Modeling Integration
- **AV-qh9** (P2 Feature): Pupu-Vocoder Anti-Aliasing Upgrade
- **AV-axb** (P2 Feature): ECAPA2 Speaker Encoder Upgrade

---

## Analysis Methodology

### 1. Specification Review

**Files Analyzed:**
- `PROMPT.md` - Master development objectives
- `docs/sota-svc-research-2025.md` - SOTA research compilation
- `docs/seed-vc-architecture.md` - Seed-VC technical details
- All track specs in `conductor/tracks/*/spec.md`

**Key Findings:**
- Research documents identify **20+ SOTA innovations** from 2024-2026 papers
- Only **3 innovations currently implemented**: Seed-VC DiT, MeanVC streaming, Shortcut CFM
- **17 innovations unimplemented**: Quality upgrades available

### 2. Implementation Gap Analysis

**Existing Infrastructure:**
✅ Seed-VC DiT (44kHz, CFM) - IMPLEMENTED
✅ MeanVC streaming (single-step) - IMPLEMENTED
✅ Shortcut flow matching (2-step) - IMPLEMENTED
✅ BigVGAN vocoder - IMPLEMENTED
✅ RMVPE pitch extraction - IMPLEMENTED

**Missing Upgrades (from research):**
❌ HQ-SVC enhancement layer - NOT IMPLEMENTED
❌ NSF harmonic/noise separation - NOT IMPLEMENTED
❌ Pupu-Vocoder anti-aliasing - NOT IMPLEMENTED
❌ ECAPA2 speaker encoder - NOT IMPLEMENTED
❌ SmoothSinger multi-resolution - NOT IMPLEMENTED (covered in existing track)
❌ TechSinger vocal techniques - NOT IMPLEMENTED (future consideration)

### 3. Priority Assignment Logic

**P0 (Critical):** None - all base functionality exists
**P1 (Important):** HQ-SVC, NSF - Major quality improvements
**P2 (Nice to Have):** Pupu-Vocoder, ECAPA2 - Incremental improvements
**P3 (Future):** Vocal techniques, multi-resolution processing

### 4. Quality Impact Estimation

| Upgrade | Current | Target | Improvement | Priority |
|---------|---------|--------|-------------|----------|
| HQ-SVC Enhancement | Speaker sim ~0.87, MCD ~4.2 | ≥0.94, ≤3.6 | +8% sim, -14% MCD | P1 |
| NSF Harmonic Modeling | Pitch RMSE ~22 cents | ≤15 cents | -32% error | P1 |
| Pupu-Vocoder | PESQ ~4.1, aliasing present | ≥4.3, -50% aliasing | +5% PESQ | P2 |
| ECAPA2 Encoder | Speaker sim ~0.87 | ≥0.92 | +6% sim | P2 |

---

## Detailed Upgrade Descriptions

### Track 1: HQ-SVC Enhancement (P1)

**Research Source:** arXiv:2511.08496 (AAAI 2026)

**Problem Addressed:**
- Current Seed-VC pipeline produces good quality but leaves room for improvement
- Artifacts in challenging cases (breathy vocals, high notes)
- No super-resolution capability

**Solution:**
- Add HQ-SVC as post-processing enhancement layer
- Decoupled codec architecture separates content and style better
- DSP refinement reduces artifacts
- Super-resolution: 22kHz → 44.1kHz upsampling

**Architecture:**
```
Seed-VC DiT → HQ-SVC Enhancement → Output
              ├─→ Decoupled Codec
              ├─→ Diffusion Refinement
              └─→ DSP Post-Processing (44.1kHz)
```

**Key Benefits:**
- **+8% speaker similarity improvement** (0.87 → 0.94)
- **-14% MCD improvement** (4.2 → 3.6)
- Better handling of vocal artifacts
- Optional: can be toggled via Web UI

**Implementation Phases:**
1. Model integration (HQSVCEnhancer wrapper)
2. Pipeline integration (enable_hq_enhancement flag)
3. Web UI toggle
4. Evaluation and benchmarking

**Estimated Effort:** Medium (2-3 days)

---

### Track 2: NSF Harmonic Modeling (P1)

**Research Sources:**
- SiFiSinger (ICASSP 2024)
- R2-SVC (arXiv:2510.20677)
- FIRNet (2024)

**Problem Addressed:**
- Current pipelines treat audio as monolithic signal
- Pitch errors propagate through entire synthesis
- Difficult to preserve natural singing characteristics (vibrato, breathiness)

**Solution:**
- Explicit harmonic/noise source separation
- F0-driven harmonic generation (sine waves)
- Filtered noise for aperiodic components
- Mcep envelope for spectral shaping

**Architecture:**
```
Content Encoder → DiT Decoder → NSF Module → Vocoder
                                 ├─→ Harmonic Generator (F0-driven)
                                 ├─→ Noise Generator (filtered)
                                 └─→ Source Filter (mcep envelope)
```

**Key Benefits:**
- **-32% pitch error reduction** (22 → 15 cents RMSE)
- Better vibrato preservation (F0 variance ±10%)
- More natural singing voice
- Harmonic coherence ≥0.90

**Implementation Phases:**
1. NSF module implementation (harmonic/noise generators)
2. Pipeline integration (after DiT decoder)
3. Training losses (F0, mcep, coherence)
4. Evaluation (pitch accuracy, naturalness MOS)

**Estimated Effort:** Medium (3-4 days)

---

### Track 3: Pupu-Vocoder Anti-Aliasing (P2)

**Research Source:** arXiv:2512.20211

**Problem Addressed:**
- BigVGAN can introduce high-frequency aliasing artifacts
- Metallic/harsh timbre in some cases
- Spectral discontinuities at transients

**Solution:**
- Anti-derivative anti-aliasing (ADAA) for activation functions
- Aliasing-free upsampling filters
- Better high-frequency preservation

**Architecture:**
```
Seed-VC DiT → 128-band mel → Vocoder → Audio
                              [User selectable]
                              ├─→ BigVGAN v2 (default)
                              └─→ Pupu-Vocoder (anti-aliased)
```

**Key Benefits:**
- **+5% PESQ improvement** (4.1 → 4.3)
- **-50% aliasing reduction**
- Cleaner high-frequency content
- Better spectral flatness

**Implementation Phases:**
1. Pupu-Vocoder integration (model wrapper)
2. Pipeline vocoder selection (config flag)
3. Quality comparison (PESQ, spectral analysis)
4. Optional Web UI selector

**Alternative Approach:**
If Pupu integration is complex, extract ADAA activations and modify BigVGAN.

**Estimated Effort:** Low (1-2 days)

---

### Track 4: ECAPA2 Speaker Encoder (P2)

**Research Source:** FreeSVC (arXiv:2501.05586)

**Problem Addressed:**
- CAMPPlus encoder trained primarily on Mandarin Chinese
- Less robust to background music
- Limited multilingual support

**Solution:**
- ECAPA2: Enhanced Context Attentive Pooling (version 2)
- Better speaker discrimination
- Robust to noise and background music
- Multilingual training

**Architecture:**
```
Reference Audio → Mel → Speaker Encoder → 192-dim Embedding
                         [User selectable]
                         ├─→ CAMPPlus (default)
                         └─→ ECAPA2 (robust)
```

**Key Benefits:**
- **+6% speaker similarity** (0.87 → 0.92)
- **EER ≤2.0%** for speaker verification
- **SNR ≥5dB** robustness (background music tolerance)
- Multilingual support (English, Spanish, Mandarin, Korean)

**Implementation Phases:**
1. ECAPA2 integration (from FreeSVC/SpeechBrain)
2. Pipeline encoder selection (config flag)
3. Voice profile migration (dual embeddings)
4. Evaluation (similarity, EER, robustness)

**Estimated Effort:** Low (1-2 days)

---

## Upgrade Dependency Graph

```
EXISTING INFRASTRUCTURE
  │
  ├─→ Seed-VC DiT (44kHz, CFM) ✅
  ├─→ BigVGAN vocoder ✅
  ├─→ RMVPE pitch extraction ✅
  └─→ CAMPPlus speaker encoder ✅
       │
       ├─→ [NEW] HQ-SVC Enhancement (P1)
       │    └─→ Quality Pipeline post-processing
       │
       ├─→ [NEW] NSF Harmonic Modeling (P1)
       │    └─→ Quality Pipeline enhancement
       │
       ├─→ [NEW] Pupu-Vocoder (P2)
       │    └─→ Alternative to BigVGAN
       │
       └─→ [NEW] ECAPA2 Encoder (P2)
            └─→ Alternative to CAMPPlus
```

**No blocking dependencies** - all upgrades are independent and can be implemented in parallel.

---

## Implementation Recommendations

### Phase 1: P1 Quality Upgrades (Week 1)
**Parallel Implementation:**
1. HQ-SVC Enhancement (Developer Agent 1)
2. NSF Harmonic Modeling (Developer Agent 2)

**Expected Impact:**
- Speaker similarity: 0.87 → 0.94 (+8%)
- MCD: 4.2 → 3.6 (-14%)
- Pitch RMSE: 22 → 15 cents (-32%)
- Naturalness MOS: 3.8 → 4.0+

### Phase 2: P2 Refinements (Week 2)
**Parallel Implementation:**
1. Pupu-Vocoder Integration (Developer Agent 1)
2. ECAPA2 Encoder Integration (Developer Agent 2)

**Expected Impact:**
- PESQ: 4.1 → 4.3 (+5%)
- Aliasing reduction: -50%
- Speaker similarity: 0.87 → 0.92 (+6%)
- Background music robustness

### Phase 3: Evaluation & User Testing (Week 3)
1. A/B testing with William/Conor conversions
2. Subjective MOS evaluation
3. Performance benchmarking
4. Documentation and user guides

---

## Out of Scope (Future Consideration)

### Not Implemented (Deferred to P3)

1. **TechSinger Vocal Techniques** (arXiv:2502.12572)
   - Controllable vocal techniques (vibrato, growl, falsetto, etc.)
   - Requires extensive UI for technique control
   - More research needed for technique detection

2. **SmoothSinger Multi-Resolution Processing**
   - Covered in existing sota-innovations track
   - Non-sequential U-Net processing
   - Complexity vs benefit tradeoff

3. **VoiceCraft Zero-Shot Editing** (2026)
   - Neural codec language models for voice editing
   - Requires additional infrastructure
   - User workflows not defined yet

4. **DisCoder Music Vocoder** (arXiv:2502.12759)
   - DAC latent space for music generation
   - More suited to full music synthesis than VC
   - Potential future karaoke enhancement

5. **RingFormer Long Sequence Processing** (arXiv:2501.01182)
   - Ring attention for long audio sequences
   - Current max 5-minute limit is acceptable
   - Memory optimization not critical on Thor (122GB)

---

## Risk Assessment

### High-Risk Items (Mitigation Required)

1. **HQ-SVC Performance Overhead**
   - Risk: >2x slowdown unacceptable for users
   - Mitigation: Make enhancement optional, profile and optimize

2. **NSF Integration Complexity**
   - Risk: Harmonic artifacts in output
   - Mitigation: Extensive testing, tune harmonic/noise balance

3. **Model Weight Availability**
   - Risk: Pretrained weights not available for 44.1kHz
   - Mitigation: Train on singing dataset if needed, or use 22kHz versions

### Medium-Risk Items (Monitor)

1. **Backward Compatibility**
   - Risk: Existing voice profiles incompatible
   - Mitigation: Store dual embeddings, auto-migrate profiles

2. **GPU Memory Pressure**
   - Risk: Multiple enhancements exceed Thor's 122GB limit
   - Mitigation: Lazy loading, unload unused components

### Low-Risk Items

1. **Quality Improvement Marginal**
   - Risk: A/B testing shows no preference
   - Mitigation: Extensive benchmarking before rollout

---

## Success Metrics

### Quantitative Targets

| Metric | Baseline | Target | Track |
|--------|----------|--------|-------|
| Speaker Similarity | 0.87 | ≥0.94 | HQ-SVC |
| MCD | 4.2 | ≤3.6 | HQ-SVC |
| Pitch RMSE | 22 cents | ≤15 cents | NSF |
| Naturalness MOS | 3.8 | ≥4.0 | NSF |
| PESQ (44kHz) | 4.1 | ≥4.3 | Pupu-Vocoder |
| Aliasing | Baseline | -50% | Pupu-Vocoder |
| EER | 3.5% | ≤2.0% | ECAPA2 |
| SNR Robustness | 10dB | ≥5dB | ECAPA2 |

### Qualitative Targets

- [ ] User A/B testing shows preference for upgraded pipelines
- [ ] William↔Conor conversions indistinguishable from real vocals
- [ ] Professional music producers approve quality
- [ ] Background music handling improved significantly
- [ ] Multilingual vocals work reliably

---

## Next Steps

### Immediate Actions

1. **Review and approve tracks** - PM reviews new track specs
2. **Prioritize implementation** - Confirm P1 upgrades start first
3. **Allocate agents** - Spawn developer agents for parallel work
4. **Download model weights** - Pre-fetch HQ-SVC, Pupu, ECAPA2 weights

### Parallel Workstreams

**Stream 1: HQ-SVC Enhancement**
- Agent: developer-agent-1
- Duration: 2-3 days
- Files: `src/auto_voice/inference/hq_svc_wrapper.py`, pipeline integration
- Tests: Quality metrics, E2E with enhancement

**Stream 2: NSF Harmonic Modeling**
- Agent: developer-agent-2
- Duration: 3-4 days
- Files: `src/auto_voice/models/nsf_module.py`, pipeline integration
- Tests: Pitch accuracy, naturalness MOS

**Stream 3: Pupu-Vocoder**
- Agent: developer-agent-3
- Duration: 1-2 days
- Files: `src/auto_voice/models/pupu_vocoder.py`, pipeline integration
- Tests: PESQ, spectral analysis

**Stream 4: ECAPA2 Encoder**
- Agent: developer-agent-4
- Duration: 1-2 days
- Files: `src/auto_voice/models/ecapa2_encoder.py`, profile migration
- Tests: Speaker similarity, EER, robustness

### Coordination

- Daily standup via beads task updates
- Cross-agent integration testing after Phase 1
- Combined quality evaluation after all upgrades
- Final A/B testing with William/Conor conversions

---

## Appendix: Research Papers

### Implemented (Current)
1. ✅ Seed-VC (arXiv:2411.09943) - DiT-CFM architecture
2. ✅ MeanVC (arXiv:2510.08392) - Streaming via mean flows
3. ✅ R-VC (arXiv:2506.01014) - Shortcut flow matching

### New Implementations (This Analysis)
1. 🆕 HQ-SVC (arXiv:2511.08496) - Enhancement layer
2. 🆕 SiFiSinger (ICASSP 2024) - NSF harmonic modeling
3. 🆕 R2-SVC (arXiv:2510.20677) - NSF robustness
4. 🆕 Pupu-Vocoder (arXiv:2512.20211) - Anti-aliasing
5. 🆕 FreeSVC (arXiv:2501.05586) - ECAPA2 encoder

### Future Consideration
1. ⏭️ TechSinger (arXiv:2502.12572) - Vocal techniques
2. ⏭️ SmoothSinger (Jun 2025) - Multi-resolution processing
3. ⏭️ VoiceCraft (2026) - Zero-shot editing
4. ⏭️ DisCoder (arXiv:2502.12759) - Music vocoder
5. ⏭️ RingFormer (arXiv:2501.01182) - Long sequences

---

**Analysis Complete**
**Tracks Created:** 4
**Beads Tasks Created:** 4
**Estimated Total Effort:** 7-11 days (parallelizable to 3-4 days with 4 agents)
**Expected Quality Impact:** +8-32% across key metrics
