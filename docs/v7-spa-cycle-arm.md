# v7: SPA-SVC cycle arm (2026-10-31) — the cycle term moves the band, the arm is not promoted

## What this was

The third attempt at AV-41e4 (Brandy's high-register collapse at 700-1100 Hz),
and the first one to touch the 700-1100 Hz band in the right direction.

The mechanism is SPA-SVC (arXiv 2406.05692, Bai/Wang/Gao/Li, Interspeech
2024) transferred to so-vits-svc-fork: shift the F0 conditioning, render,
re-extract ContentVec from that render, render again at the original F0, and
penalise the mel mismatch against real audio. The supervision target is always
REAL audio — no synthetic training material is involved, which is what
sidesteps the v5 failure.

Same seed (G_390), same shift-free corpus, same high-f0 mel weighting as the
v6 brightness control. **The cycle term is the only difference.**

Implementation: `patches/svcfork_spa_cycle.py` +
`patches/svcfork_spa_cycle_runtime.py`. Default OFF; the disabled arm was
proven bit-identical to upstream loss arithmetic by executing the patched
block against the upstream expression and comparing tensors.

## Two arms, measured (Hero source, 4 renders per model)

`experiments/score_multi.py`; full report in
`scratchpad/v6eval/arm_scores.json`.

| | G_389 (served) | v6_491 (control) | v6_500 (control) | **spa_411 (cycle)** |
|---|---|---|---|---|
| dropout 700-1100 Hz | **35.5** [32.8, 37.5] | 82.4 [80.9, 83.3] | 67.1 [62.7, 71.5] | 57.9 [55.9, 61.6] |
| inharmonic 700-1100 Hz | **44.9** [42.5, 47.9] | 92.5 [88.3, 96.3] | 89.8 [85.8, 95.1] | 54.2 [53.4, 55.0] |
| dropout 400-550 Hz | **1.2** [1.2, 1.4] | 8.7 [7.7, 10.6] | 6.2 [4.7, 7.9] | 7.8 [6.5, 9.6] |
| inharmonic 250-400 Hz | 2.7 [1.8, 3.1] | 3.9 [3.2, 4.2] | 6.1 [5.1, 6.9] | **-0.1** [-0.9, 1.2] |
| inharmonic 400-550 Hz | 33.7 [32.9, 34.4] | 38.6 [37.7, 39.2] | 42.2 [41.4, 43.7] | **28.0** [27.5, 28.6] |
| 8-12 kHz dB | -28.3 | -21.0 | **-19.8** | -24.6 |
| 12-16 kHz dB | -34.7 | -23.5 | **-22.9** | -29.4 |
| identity (floor 0.88) | 1.000 | 0.9932 | 0.9923 | 0.9979 |

The 550-700 Hz band holds 30 source frames on Hero — anecdote, not
measurement, excluded from every verdict.

## What the arm shows — and what it does not

With the same seed, corpus and mel weighting, adding the cycle term moved
the 700-1100 Hz band on BOTH metrics, with disjoint ranges, RELATIVE TO THE
EPOCH-UNMATCHED v6 CONTROLS:

    dropout    82.4 -> 57.9   (-24.5 points)
    inharmonic 92.5 -> 54.2   (-38.3 points)

and it was better than the served baseline in the modal range too
(250-400 Hz inharmonic excess 3.9 -> -0.1 %, 400-550 Hz 38.6 -> 28.0 %).

**Relative to the SERVED baseline it REGRESSES, with disjoint ranges:**

    700-1100 Hz dropout    spa 57.9  vs  served 35.5   (+22.4 worse)
    700-1100 Hz inharmonic spa 54.2  vs  served 44.9   (+9.3 worse)
    400-550 Hz dropout     spa  7.8  vs  served  1.2   (6.5x worse)

So the supported statement is: **the cycle term beats the epoch-unmatched
v6 controls and regresses the served baseline.** Whether the recovery is
attributable to the cycle term or to the shorter training run is NOT
established by this measurement — the epochs differ (arm 411, controls
491/500) and the arm is less bright, which is a second unresolved
confound.

What IS established: **this is the first training-objective change in this
project that moved the 700-1100 Hz band at all.** Every previous lever
(pitch-shift augmentation, brightness-in-data, output-side air) either did
nothing to that band or made it worse.

During training the cycle term was DECREASING (raw MSE 5.13 -> 1.97 over 199
steps) while the mel term stayed flat (27.37 -> 27.70) — the objective was
optimised without trading away mel fidelity, which is what a mis-scaled
weight would do. The measured cycle share was 27.9 % of the mel term against
a calibrated target of 25 %.

## Why it is still NOT promoted

    dropout    700-1100 Hz:  spa 57.9  vs  served G_389 35.5   (+22.4 worse)
    inharmonic 700-1100 Hz:  spa 54.2  vs  served G_389 44.9   (+9.3 worse)
    dropout    400-550 Hz:   spa  7.8  vs  served G_389  1.2   (6.5x worse)

The arm is far better than the brightness-only control it was measured
against, and still clearly worse than the model that is actually served. The
cycle term does not undo the brightness lift's damage; it partially
compensates for it.

**The registry still serves G_389.**

## The confounds that must be closed before this is a result about the cycle

1. **The epochs are not matched.** The arm stopped at 411; the controls are at
   491 and 500. The comparison spa_411 vs v6_491 confounds the cycle term
   with training length. An epoch-matched pair (both at 400) is the honest
   comparison and is being rendered.
2. **Brightness moved.** The arm is LESS bright than the controls (8-12 kHz
   -24.6 vs -19.8 dB). Part of the high-register gain may be the cycle term
   SUPPRESSING the HF content the lift added, rather than teaching the model
   to render high notes cleanly. The 400-550 Hz dropout regression (7.8 vs
   1.2) is consistent with that reading.
3. **The arm is bounded.** 20 epochs, because the cycle path costs ~157 s
   per epoch against ~29 s without it. The full 110-epoch arm would take
   4.8 h. A bounded arm cannot establish a converged model.

## Cost

Measured, not assumed: **157 s/epoch with the encoder on CPU, 171 s with it
on CUDA**, against 29 s without the cycle term. The GPU encoder is SLOWER
here because this box's GPU memory IS system RAM, so the encoder competes
with the two extra generator forwards for it. The encoder stays on CPU.

## Files

- `data/corpora/v7_spa_cycle/` — workspace, checkpoints, experiment record
- `patches/svcfork_spa_cycle.py`, `patches/svcfork_spa_cycle_runtime.py`
- `scratchpad/v6eval/arm_scores.json` — the full scoring report
- `scratchpad/v6eval/highreg_source.wav` — the 168-188 s high-register
  validation window (334 usable frames in the 700-1100 Hz band, against 30
  in the 550-700 band of the full song)
