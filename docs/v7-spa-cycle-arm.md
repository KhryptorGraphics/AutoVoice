# v7: SPA-SVC cycle arm (2026-10-31) — NEGATIVE at matched epochs

## What this was

The third attempt at AV-41e4 (Brandy's high-register collapse at 700-1100 Hz).

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

## The epoch-matched result (Hero source, 4 renders per model)

Both arms at epoch 400. `experiments/score_multi.py`; full report in
`scratchpad/v6eval/matched_scores.json`.

| | G_389 (served) | v6_400 (control) | **spa_400 (cycle)** |
|---|---|---|---|
| dropout 700-1100 Hz | **35.5** [32.8, 37.5] | 49.2 [44.8, 53.3] | 74.1 [70.8, 78.3] |
| inharmonic 700-1100 Hz | **44.9** [42.5, 47.9] | 64.3 [60.3, 71.6] | **56.2** [48.4, 61.3] |
| dropout 400-550 Hz | **1.2** [1.2, 1.4] | 6.2 [5.3, 6.9] | 2.6 [2.0, 2.9] |
| inharmonic 250-400 Hz | 2.7 [1.8, 3.1] | 5.2 [5.0, 5.4] | **-0.4** [-1.2, 0.4] |
| inharmonic 400-550 Hz | 33.7 [32.9, 34.4] | 40.3 [39.8, 41.0] | **29.4** [28.6, 30.4] |
| 3-8 kHz dB | -15.9 | **-14.4** | -15.9 |
| 8-12 kHz dB | -28.3 | **-20.8** | -25.3 |
| 12-16 kHz dB | -34.7 | **-24.8** | -29.6 |
| centroid Hz | 748 | **872** | 776 |
| identity (floor 0.88) | 1.000 | 0.9951 | 0.9976 |

The 550-700 Hz band holds 30 source frames on Hero — anecdote, not
measurement, excluded from every verdict.

## Verdict: NEGATIVE. The cycle loss does not fix the high register here.

At matched epochs the cycle arm is **worse** than the brightness control on
700-1100 Hz dropout: 74.1 [70.8, 78.3] against 49.2 [44.8, 53.3] — the
ranges are DISJOINT (gap 17.5 points, p=0.029 at 4 renders per arm).

On inharmonic excess the two arms **OVERLAP** (spa 56.2 [48.4, 61.3] against
v6 64.3 [60.3, 71.6], gap −1.0), so the cycle arm's advantage there is
not established by this measurement. Against the served baseline the
cycle arm is worse on both, with disjoint ranges (dropout gap 33.3,
inharmonic gap 0.5).

The cycle arm is better in the modal range (400-550 Hz dropout 2.6 against
the control's 6.2; 250-400 Hz inharmonic -0.4 against 5.2).

**The earlier, epoch-UNMATCHED comparison was misleading.** spa_411 against
v6_491/v6_500 appeared to show the cycle term recovering 24.5 points of
700-1100 Hz dropout. With the epochs matched it does not: the apparent
recovery was the control's extra 80-90 epochs of training, not the cycle
term. Recorded here because the earlier reading was published in this repo
and pushed.

## What the cycle term does appear to do

It preserves the MODAL range (400-550 Hz dropout 2.6 % against the
control's 6.2 %) and it **suppresses the HF content the brightness lift
added** (8-12 kHz -25.3 against -20.8 dB, centroid 776 against 872 Hz). It
is DARKER, and darker is closer to the served baseline's spectrum
(-28.3 dB). So it partially undoes the brightness lift's HF residue — the
same residue that fills a dropout — **without** teaching the model to render
high notes cleanly. That is a real effect, but it is not the effect SPA-SVC
was chosen for.

The mechanism itself worked as designed, which is worth stating: the cycle
term DECREASED during training (raw MSE 5.13 -> 1.97 over 199 steps) while
the mel term stayed flat (27.37 -> 27.70), so the objective was optimised
without trading away mel fidelity. The weight was calibrated on real batches
(raw cycle MSE 4.23, mel L1 0.608 -> mel term 27.35), putting the cycle at
25 % of it; measured in training, 27.9 %.

## Consequence: the pre-registered revisit trigger fires

`docs/research/synthetic-data-ideation-20260917.md` set the trigger: *"If
(1) and (2) both fail to move the 700-1100 Hz band, the working hypothesis
that this is a data-coverage problem should be abandoned in favour of an
architectural one."*

(1) was the brightness-only corpus and (2) is this cycle loss. **Both have
now failed to move the band.** The data-coverage hypothesis must be
abandoned. The next question is whether a 16 kHz content encoder and an NSF
excitation fitted on a 291 Hz median can represent this register at all.

That also retires the remaining untried ideas from the ideation doc that
depend on synthetic high-register audio, and it re-ranks AV-9hc5
(pitch-tracked make-up gain + keep-source-spans above ~550 Hz), which is a
SIGNAL-processing lever and does not depend on the training corpus.

**The registry still serves G_389. Nothing was promoted.**

## Cost

Measured, not assumed: **157 s/epoch with the encoder on CPU, 171 s with it
on CUDA**, against 29 s without the cycle term. The GPU encoder is SLOWER
here because this box's GPU memory IS system RAM, so the encoder competes
with the two extra generator forwards for it. The encoder stays on CPU.

## Files

- `data/corpora/v7_spa_cycle/` — workspace, checkpoints, experiment record
- `patches/svcfork_spa_cycle.py`, `patches/svcfork_spa_cycle_runtime.py`
- `scratchpad/v6eval/matched_scores.json` — the epoch-matched scoring report
- `scratchpad/v6eval/arm_scores.json` — the epoch-unmatched report
- `scratchpad/v6eval/highreg_source.wav` — the 168-188 s high-register
  validation window (334 usable frames in the 700-1100 Hz band, against 30
  in the 550-700 band of the full song)
