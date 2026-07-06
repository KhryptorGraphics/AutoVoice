# Background Vocal Separation Experiments

## Context
Current state: 97% toward perfect voice conversion. The remaining issue: background vocals (harmonies, backing singers) get mixed with lead vocals during Demucs separation, causing distortion when the model tries to convert the "lead" voice.

## Problem
When a song has multiple vocalists (lead + backing), Demucs HTDemucs outputs a single "vocals" stem containing ALL vocal content. The conversion model then tries to convert this mixed signal as if it were one voice, producing artifacts.

## Proposed Solution
Split the vocals stem into separate tracks per speaker (lead vs backing) using speaker diarization + source separation, then convert each track independently and recombine.

---

## Experiment 1: Speaker Diarization Accuracy on Vocal Stems

**Assumption**: Pyannote/WavLM diarization can reliably separate lead vs backing vocal segments from a Demucs-extracted vocals stem.

**Experiment**: 
- Take 10 songs with known multi-vocalist structure (e.g., duets, songs with prominent backing vocals)
- Run: Demucs vocal separation → Pyannote diarization on vocals stem
- Manual annotation of ground truth (lead vs backing segments)
- Measure: Speaker diarization accuracy (DER - Diarization Error Rate)

**Metric**: DER < 15% on vocal stems

**Success Threshold**: If DER < 15%, proceed to Experiment 2. If > 15%, investigate better diarization or alternative approaches.

---

## Experiment 2: Source Separation Quality Per-Speaker

**Assumption**: After diarization, we can extract per-speaker vocal tracks with sufficient quality for conversion.

**Experiment**:
- Use the diarization segments from Experiment 1
- Apply speaker-attributed source separation (e.g., SepFormer, or mask-based extraction using diarization boundaries)
- Evaluate: SDR (Signal-to-Distortion Ratio) for extracted lead vs backing tracks
- Compare against original mixed vocal stem

**Metric**: SDR > 10 dB for extracted tracks

**Success Threshold**: Extracted tracks have SDR > 10 dB and sound clean to human listeners.

---

## Experiment 3: Conversion Quality on Isolated Tracks

**Assumption**: Converting separated lead/backing tracks independently and recombining produces fewer artifacts than converting the mixed stem.

**Experiment**:
- Take 5 test songs with known backing vocal issues
- For each song:
  1. **Baseline**: Current pipeline (Demucs → single vocal stem → conversion)
  2. **Experimental**: Demucs → diarization → separate tracks → independent conversion → recombine
- Blind A/B listening test with 5 raters
- Rate: Naturalness, artifact presence, vocal clarity (1-5 scale)

**Metric**: Mean opinion score (MOS) improvement ≥ 0.5 points

**Success Threshold**: Experimental pipeline scores ≥ 0.5 MOS higher than baseline.

---

## Experiment 4: Recombination Strategy

**Assumption**: Simple mixing of converted tracks preserves musical balance.

**Experiment**:
- Test recombination methods:
  A. Simple sum (converted_lead + converted_backing + instrumental)
  B. Volume-matched mix (match RMS of each converted track to original segment RMS)
  C. Dynamic mixing (preserve original lead/backing ratio per segment)
- Evaluate on 3 songs with prominent harmonies
- Check for: phase issues, level imbalance, unnatural blend

**Metric**: Subjective blend quality score (1-5), no audible artifacts

**Success Threshold**: Method C scores ≥ 4/5 with no phase artifacts.

---

## Experiment 5: End-to-End Latency Impact

**Assumption**: The additional diarization + separation steps add acceptable latency.

**Experiment**:
- Measure pipeline latency for:
  1. Current pipeline (baseline)
  2. Multi-track pipeline (with diarization + per-speaker separation)
- Test on CPU and GPU
- Target: < 2x baseline latency

**Metric**: Total pipeline time (seconds)

**Success Threshold**: < 2x baseline latency on GPU; < 3x on CPU.

---

## Experiment 6: Training Data Requirements for Multi-Speaker Models

**Assumption**: Current training pipeline can handle multi-speaker training data if provided as separate samples.

**Experiment**:
- Create a test profile with deliberately mixed lead/backing training samples
- Train using current LoRA pipeline
- Compare conversion quality vs. profile trained on clean single-speaker samples
- Measure: MOS on held-out test songs

**Metric**: MOS difference between clean vs mixed training

**Success Threshold**: < 0.3 MOS degradation from mixed training data.

---

## Implementation Priority

| Experiment | Effort | Risk | Dependency |
|------------|--------|------|------------|
| 1. Diarization Accuracy | Low (1-2 days) | Low | None |
| 2. Source Separation Quality | Medium (3-5 days) | Medium | Exp 1 |
| 3. Conversion Quality A/B | Medium (1 week) | Low | Exp 1, 2 |
| 4. Recombination Strategy | Low (2 days) | Low | Exp 2 |
| 5. Latency Impact | Low (1 day) | Low | Exp 3 |
| 6. Training Data | Medium (1 week) | Medium | Profile training pipeline |

---

## Quick Win: Diarization-Enhanced Separation (MVP)

If Experiments 1-2 succeed, the MVP approach:
1. Demucs vocal separation (existing)
2. Pyannote diarization on vocals stem (new)
3. Segment vocals by speaker (new)
4. Convert each speaker segment independently through existing fork HQ pipeline
5. Recombine with original timing

This leverages existing infrastructure (svc_fork_bridge, pipeline) with minimal new code.

---

## Risk Mitigation

- **False diarization splits**: Use minimum segment duration (2s) to avoid over-segmentation
- **Phase issues**: Use cross-fade at segment boundaries (50ms)
- **Quality regression**: Feature flag the multi-speaker path; fallback to single-stem on error
- **Latency**: Run diarization in parallel with Demucs separation
