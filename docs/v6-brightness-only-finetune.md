# v6: brightness-only fine-tune from G_390 (2026-10-31) — FAILED, not promoted

## What this was

Corpus: `experiments/build_corpus_v4.py` with `SHIFTS=[]` — Brandy's own
recordings only, no pitch-shifted copies, plus the 8-12 kHz / 12-16 kHz
brightness targets measured from the source. 78 source samples, 488 plain
chunks, **zero** shifted chunks.

Why no shifts: v5 (WORLD +4..+20 st) FAILED its pre-registered bar — it raised
inharmonicity at exactly the pitches it supplied (700-1100 Hz: 54.4 % -> 64.4 %
excess) and regressed the modal range (250-400 Hz dropout 9.4 % -> 12.3 %).
The literature agreed in advance (SingAug Table 5: pitch augmentation alone
degraded MCD and semitone accuracy). Brightness-in-data was the ONE win v5
banked, and this run takes it alone so it is cleanly attributable.

Seed: `data/fork_models/_hif0_20260917/G_389.pth` -> resumed at epoch 390.
Loss weighting: the existing high-f0 mel weighting, 4.0 with a 400-800 Hz
ramp, WITH the uv mask (the first canary, without the mask, was quarantined —
see `experiment.json`'s `training_canary`).

## Training

    epoch 500  loss/g/total 35.319  mel 24.830  fm 7.414  kl 0.829  lf0 0.000
               loss/d/total 2.430

Checkpoints kept: 389, 390, 400, 419, 437, 455, 473, 491, 500.

## Result: FAILED the pre-registered bar. NOT promoted.

Bar, set before the numbers were seen (the same bar v5 failed): beat G_389 on
BOTH dropout and inharmonic excess at 700-1100 Hz, identity >= 0.88, and the
difference must exceed the within-model render spread. 4 renders per model, so
disjoint ranges would be significant at p=0.029.

Measured on the Hero source (`scratchpad/dropout/hero_src.wav`), scored with
`experiments/score_multi.py`; full report in `scratchpad/v6eval/v6_scores.json`:

| | G_389 (baseline) | v6 G_500 | v6 G_473 |
|---|---|---|---|
| dropout 250-400 Hz | 2.2 [1.9, 2.6] | **1.8** [1.4, 2.0] | 2.4 [2.2, 2.5] |
| dropout 400-550 Hz | **1.2** [1.2, 1.4] | 6.2 [4.7, 7.9] | 5.0 [3.9, 7.1] |
| dropout 700-1100 Hz | **35.5** [32.8, 37.5] | 67.1 [62.7, 71.5] | 56.8 [55.2, 58.3] |
| inharmonic 700-1100 Hz | **44.9** [42.5, 47.9] | 89.8 [85.8, 95.1] | 85.0 [83.2, 88.8] |
| 8-12 kHz dB | -28.3 | **-19.8** | -20.6 |
| 12-16 kHz dB | -34.7 | **-22.9** | -22.8 |
| centroid Hz | 748 | 913 | 843 |
| identity | 1.000 | 0.9923 | 0.9917 |

The 550-700 Hz band holds only 30 source frames on Hero — anecdote, not
measurement, and it is excluded from the verdict.

### What happened

**Brightness WON.** The lift transferred from data to model exactly as v5
predicted: 8-12 kHz -28.3 -> -19.8 dB, 12-16 kHz -34.7 -> -22.9 dB, centroid
748 -> 913 Hz, identity still 0.992 (floor 0.88).

**The high register LOST BADLY.** Dropout 35.5 -> 67.1 % and inharmonic
excess 44.9 -> 89.8 % at 700-1100 Hz. Both ranges are disjoint from the
baseline's, so this is not render noise.

**400-550 Hz regressed**: dropout 1.2 -> 6.2 %, inharmonic 33.7 -> 42.2 %.

**250-400 Hz held** (2.2 -> 1.8 %).

### Reading

The brightness lift raised the HF content of the whole corpus, and the model
learned it — including at the pitches where it has almost no training data.
The extra HF energy at 700-1100 Hz is inharmonic residue, so a frame that was
already losing energy now loses it to a noisier band: dropout nearly doubles
while the voice gets brighter.

This is the SAME failure mode v5 showed ("brighter but worse dropouts") and
the SAME reason the output-side air restore was rejected: an HF lift raises
the inharmonic residue that fills a dropout. Fixing darkness in the DATA
helps the timbre but costs the high register, and the two cannot be
separated by a corpus-level lift.

**Implication for serving: do NOT promote v6.** The registry still serves
G_389. Brightness is worth having, but not at the cost of doubling the
high-register dropout — the user hears the dropouts, not the brightness
metric. If brightness is wanted, it has to come from the SOURCE material
(the corpus is 10-18 dB darker than the source at 8-16 kHz because her
recordings are), not from a lift applied to material that already lacks
high-register energy.

## The other arm: SPA-SVC cycle loss

`data/corpora/v7_spa_cycle/` — the same shift-free corpus, seeded from the
SAME G_390 checkpoint and run to the SAME epoch 500, so the cycle term is the
only difference between the arms. See `patches/README.md` (`svcfork_spa_cycle`)
and `data/corpora/v7_spa_cycle/experiment.json`.

The cycle weight is CALIBRATED, not guessed: measured on 4 real corpus
batches, the mean raw cycle MSE is 4.2262 and the mean mel L1 is 0.6078, so
the mel term is 27.35 and the weight 0.035957 puts the cycle term at 25 % of
it. A weight of 1.0 would put the cycle at 190 against a mel term of 27 —
seven times the mel loss, i.e. the cycle would BE the objective rather than a
regulariser on it.

## Files

- `data/corpora/v6_brightness_only/` — corpus, configs, logs, experiment record
- `data/corpora/v7_spa_cycle/` — the SPA arm's workspace
- `patches/svcfork_spa_cycle.py` — installs the cycle loss into the fork
- `patches/svcfork_spa_cycle_runtime.py` — the cycle objective itself
- `scratchpad/v6eval/render_candidates.py` — renders N takes per checkpoint
- `scratchpad/v6eval/v6_scores.json` — the full scoring report
