# Automatic co-vocal time-range detection — investigated, NOT shipped

**Date:** 2026-09-15 · **Status:** validated hypothesis, not a calibration. Do not
build on this without the two measurements in "What is missing" below.
**Scripts:** `scratchpad/covocal/{detect,smooth,crosscheck}.py`
**Ground truth:** job `d85f96e6` (Beauty and the Beast), operator ear-marked male
spans 39-77 / 116-135 / 193-232 s — 96.0 s of a 232.4 s stem (41 % male).

## Why this was tried

`scratchpad/registercal/FINDINGS.md` established that duet routing is bounded by
SPAN PURITY, not by the statistic: given true per-singer spans, median f0
separates the two singers of this song by 114 Hz, but the diarizer's clusters are
~40 % impure so no per-cluster statistic survives. The obvious alternative is to
skip clustering entirely — classify short windows of the separated vocal stem by
pitch and let the time ranges fall out.

## Result on the ground-truth song

2 s windows, 1 s hop, `librosa.pyin` (fmin=110), median f0 per window, 2-means on
log f0 (no hand threshold — thresholds were disproved in the register work).

| method | routed-group precision | frame accuracy |
|---|---|---|
| raw diarizer clusters | 39.7 % / 43.3 % — **chance** (base rate 41 %) | — |
| raw 2 s windows | 64 % | 71 % |
| + median filter 15 win | 81 % | 76 % |
| + median filter 21 win | **94 %** | 78 % |
| + median filter 31 win | 95 % | 72 % |
| Viterbi, switch penalty 4.0 | 82 % | 73 % |

**Precision is not the whole story. Recall is ~49 %** — 47 s detected of 96 s
actual. The filter width is a precision/recall dial, not a free win: the group
gets purer as it shrinks.

Detected blocks `(67,73) (113,122) (196,228)` against truth `(39,77) (116,135)
(193,232)`: blocks 2 and 3 land well, **block 1 is almost entirely missed** —
28 s of the male's first entry never detected.

## Why it was not shipped

1. **Recall produces a worse artefact than the current defect.** At ~49 % recall
   the male part would alternate between the male model and the lead model every
   time detection drops out — the same voice switching identity mid-phrase. That
   is plausibly more jarring than today's consistently-wrong render.
2. **The comparison above is a straw man.** "Diarizer at chance" describes RAW
   clusters. The shipped path is diarizer + centroid gate + **WavLM reassignment**,
   and WavLM was added specifically for this song. The honest baseline is the
   post-WavLM spans, which were NOT measured. If those are already 75-80 % pure,
   "94 vs 43" becomes "94 vs 78 at half the recall" — a different decision.
3. **The filter width was swept on the only song with ground truth.** n=1. That is
   the exact error the register calibration documents. 21 and 31 both reaching
   94-95 % is a plateau rather than a spike, which is mildly reassuring, no more.

## Second song: it does NOT generalise (measured 2026-09-15)

All of Me (Zendaya & Max Schneider), separated SOURCE stem, identical method:

| | Beauty and the Beast | All of Me |
|---|---|---|
| 2-means separation | 188 Hz | **49 Hz** |
| detected low-register total | 47 s (truth 96 s) | 74 s (production `preserved_s` 43.7) |
| blocks | 3, clean | **5, fragmented** — (61,98) (134,138) (157,171) (173,187) (237,242) |

49 Hz is about a major third: that is comfortably within one singer's range, so
the split is as likely to be verse-vs-chorus as singer-vs-singer. It overshoots
production's preserved region by 70 % and fragments it. No corroboration.

**The mechanism, which explains every result in this file and the register one.**
A pitch split finds PITCH boundaries, not SINGER boundaries. On Beauty the two
happen to correlate — the male genuinely sings lower, true medians 221.3 vs
335.4 Hz. Note the 2-means found 145/332 = 188 Hz, *wider* than the true 114 Hz
singer separation, i.e. it was partly splitting low notes from high notes rather
than man from woman. On All of Me the singers overlap in range (cluster medians
254.2 / 260.1 Hz) and the method has nothing to find.

This is the same wall as `registercal/FINDINGS.md`: pop singers of different
sexes occupy the same absolute f0 band. Windowing and smoothing change the
granularity of the question, not the answer to it.

**Conclusion: pitch-based co-vocal detection works only when the two singers
happen not to overlap in range, and there is no way to know in advance whether
a given song is that case. Not shippable as automatic behaviour.**

## What is missing (do these before building)

- **Score the post-WavLM spans** against the same ground truth. Cheap: jobs
  `90772105` ("male preserved via embedding") and `89c58bf8` — if either kept a
  `backing.wav`, its energy-active windows ARE the spans production preserved.
- **Diagnose the block-1 miss.** Print window f0 across 39-67 s. If both singers
  are active there, that is the f0 ceiling and no smoothing width recovers it
  (the duet-keying study found 95 % of co-vocal time overlaps the lead on All of
  Me). If the male is simply singing high there, it is a threshold problem.
  Different fixes — know which before choosing one.
- **A second ground-truth song.** All of Me has no ear-marked spans. A separation
  alone is not proof: a 2-means split can be verse-vs-chorus rather than
  singer-vs-singer. Convergent evidence available without labels: job `a9a6f84d`
  recorded `preserved_s: 43.7`; if detected low-register blocks total ~44 s and
  land where that cluster did, that corroborates.

## The upgrade path if it holds

Fuse per-window **WavLM timbre embeddings** (already in this codebase, already
used by `_select_speaker_spans`) with the f0 window statistic, rather than f0
alone. Pitch cannot separate singers whose registers overlap; timbre is what the
existing embedding pass uses for exactly that reason. That is the next
experiment, not a build.
