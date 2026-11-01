# What the SERVED path produces today, measured (2026-11-01)

**Read this before comparing any candidate.** The served pipeline is NOT the
raw fork render: the output-correction chain (`_correct_fork_output`) changes
the numbers by more than any model change measured in this session.

## The served model

`data/fork_models/_hif0_20260917/G_389.pth`, profile
`fb17af66-8415-4ffe-81b3-600efe75b6d7`. Registry:
crepe f0, no auto-predict f0, noise_scale 0.4, db_thresh -100, pad 1.0 s,
chunk 30 s, max chunk 40 s, uv contract ON, crepe periodicity threshold 0.3.

## THE KEY MEASUREMENT: raw model vs served path

Both measured on the same source (`scratchpad/dropout/hero_src.wav`, the Hero
vocal stem, 424 usable frames in the 700-1100 Hz band) with
`experiments/score_multi.py`.

**Both columns are n=4.** The raw column is 4 raw fork renders of G_389; the
served column is 4 independent end-to-end conversions of the same original
Hero track through the live service, scored on their paired vocal stems
(jobs `9c2b253b`, `b1d3295a`, `adfde124`, `50696749` — all frame-aligned to
the source at 11402241 frames / 258.55 s).

In this scorer, `identity` means cosine similarity of a mel-centroid vector
to the first reference group; it is **not** speaker verification and cannot
certify target-voice identity.

| metric | raw G_389 (n=4) | SERVED path (n=4) |
|---|---|---|
| dropout 250-400 Hz | 1.9 % | **0.3 %** |
| dropout 400-550 Hz | 1.2 % | **0.5 %** |
| **dropout 700-1100 Hz** | **35.50 % [32.78, 37.50]** | **4.36 % [3.07, 6.37]** |
| inharmonic 700-1100 Hz | 44.94 % [42.51, 47.88] | **40.69 % [37.14, 45.54]** |
| 8-12 kHz dB | -28.3 | **-22.5** |
| 12-16 kHz dB | -34.8 | **-23.2** |
| centroid Hz | 754 | 966 |
| mel-centroid cosine to raw reference | 0.9917 | 0.9999 |

**The chain removes 31.1 points of high-register dropout** (35.50 % ->
4.36 %), a reduction of 87.7 %, and the two distributions do not overlap:
the worst served draw (6.37 %) is below the best raw draw (32.78 %) by more
than 26 points. The air deficit also closes by ~6 dB in both 8-12 kHz and
12-16 kHz.

This is a MEASURED DISTRIBUTION now, not a single draw, so it is safe to
state. It is not safe to state that the high register is *solved* — see the
inharmonic column below, which barely moves.

**The mechanism is confirmed in the chain's own log** for every draw:
"Lead dropout fill: 8.8-9.5 % of frames lifted, mean 1.8-2.0 dB, max
11.3-11.5 dB". The level collapse in the high register is what the dropout
fill repairs.

## What the chain does NOT repair

The inharmonic excess at 700-1100 Hz barely moves: 44.94 % -> 40.69 %. The
served path recovers the LEVEL of the high register but not its TIMBRE. The
decoder cannot render Brandy above ~550 Hz, so it emits narrowband residue
where harmonics should be, and no amount of gain or fill removes a wrong
spectrum.

`docs/av9hc5-level-vs-timbre.md` measures this directly: a pitch-tracked
make-up gain removes the residual level dropout (3.77 % -> 0.47 %) and leaves
the inharmonic excess untouched (40.73 -> 40.92), while keeping the source's
own audio above 550 Hz removes it outright (40.73 -> 1.90).

**So the served path is fixed for the level defect and NOT fixed for the
timbre defect.** The timbre defect is in the model (AV-41e4's architectural
question), and it is what the user would hear as "ringing"/"buzz".

## The high-register window, n=4

The 168-188 s window (334 usable frames in the 700-1100 Hz band) cut from
each of the four served stems and scored against the same window of the
source. The raw column is the previously measured raw-model baseline for
the same window.

| | raw G_389 | SERVED (n=4) |
|---|---|---|
| dropout 700-1100 Hz | 11.38 % | **2.54 % [1.50, 4.19]** |
| inharmonic excess 700-1100 Hz | 69.92 % | **45.08 % [41.90, 50.54]** |
| centroid Hz | — | 1316-1381 |
| mel-centroid cosine to reference | — | 0.9994-1.0000 |

The same pattern as the full song, and sharper: the chain takes the level
dropout down by 4.5x, and takes the inharmonic excess down by only 24.8
points (69.92 -> 45.08). A third of the residue survives the chain.

**This is the measurement that says the high register is not solved.** On the
window where she is actually belting, the served output still carries 45 %
excess inharmonic energy against the source.

## How the served conversion was produced

Four independent jobs were submitted to the LIVE service on 2026-10-31:
`9c2b253b-3c71-4859-9044-9678d990daaf`, `b1d3295a-e708-4d4d-b3a8-1f5bc4b9f4a4`,
`adfde124-e134-4a3f-afd5-2441f1068be8`,
`50696749-66fa-4d05-abf2-d1dc283f8600`. All four are the same conversion of
the same original track with the same profile; the only thing that varies is
the fork render's stochastic noise (noise_scale 0.4), which is what makes
them independent draws.

**Do not submit conversions concurrently.** Three at once starved the
service's event loop: the API stopped answering for ~10 minutes (load
average 11-13) while all three jobs ran. One at a time, or the status API
becomes useless exactly when you are waiting on it.

The first of them (draw 1) was produced like this:

- input: `data/conversions/originals/fb17af66-.../3d0922b0-..._Hero_Mariah_Carey.wav`
  — the ORIGINAL track, 12410602 frames at 48 kHz, stereo
- profile `fb17af66-8415-4ffe-81b3-600efe75b6d7`, preset `studio`,
  pipeline `quality`, `enable_multi_speaker=false`
- output vocals: 11402241 frames at 44.1 kHz mono — **sample-aligned with
  the source vocal stem**, so the frame-aligned metrics are valid
- the service log records the chain: lead level match 0.66x (-3.6 dB),
  contour follow (22 % of frames corrected >3 dB), hum suppression
  (source unvoiced 22 %, hum frames 3.3 %), subharmonic suppression (26 %
  of frames, up to 20 dB), voice polish (-4..+4 dB across 1200-5000 Hz),
  HF de-tonalize (2.7 dB mean cut above 7500 Hz on 100 % of frames),
  air restore (+16.0 dB above 5 kHz, capped at the render's own 16051 Hz
  wall), bandwidth extend (skipped, fmax 16051 Hz needs no repair), and
  the dropout fill LAST (8.8 % of frames lifted, mean 1.8 dB, max 11.4 dB)

## Why an earlier note in this repo was misleading

`docs/served-models.md` and older measurements quote the served model's
700-1100 Hz dropout as ~35 %. Those figures came from RAW fork renders, not
from the served path. The distinction matters because a candidate that
"beats" 35 % may still be far worse than what is already served (4.36 %
at n=4).

**Any comparison against the served path must be made against THESE numbers,
not against the raw-model numbers.**


## The remaining defect after the current n=4 run

Fresh 2026-11-01 draws used the current checkout, with the notch enabled and
the shared vocal/mix true-peak guard. `score_multi.py` scored their aligned
vocal stems against the same Hero source:

| band | without notch (n=4) | current checkout (n=4) |
|---|---|---|
| 250-400 Hz inharmonic excess | 2.44 % [2.02, 3.08] | 2.23 % [1.81, 2.83] |
| 400-550 Hz inharmonic excess | 31.09 % [30.31, 31.79] | 31.04 % [29.92, 31.93] |
| 550-700 Hz inharmonic excess | 61.38 % [58.19, 65.06] | −0.32 % [−1.25, 1.09]† |
| **700-1100 Hz inharmonic excess** | **40.69 % [37.14, 45.54]** | **11.15 % [9.61, 12.61]** |
| dropout 250-400 Hz | 0.31 % [0.22, 0.48] | 0.37 % [0.29, 0.46] |
| dropout 400-550 Hz | 0.40 % [0.00, 0.59] | 0.54 % [0.20, 0.98] |
| **dropout 700-1100 Hz** | **4.36 % [3.07, 6.37]** | **1.06 % [0.71, 1.18]** |

† Only 30 usable frames; the 550-700 Hz inharmonic figure is anecdotal.
At 700-1100 Hz the distributions do not overlap: the worst current draw
(12.61 %) is 24.53 points below the best no-notch draw (37.14 %). All four
draws improve. This is substantial, but 11.15 % residual inharmonic excess
and 1.06 % level dropout are not near-perfect.

The high-register window (168-188 s) also improved across all four draws:

| 168-188 s metric | without notch (n=4) | current checkout (n=4) |
|---|---|---|
| dropout 700-1100 Hz | 2.54 % [1.50, 4.19] | 0.00 % [0.00, 0.00] |
| inharmonic excess 700-1100 Hz | 45.1 % [41.9, 50.5] | 11.6 % [9.1, 14.0] |
| spectral centroid | 1343 Hz [1317, 1381] | 1525 Hz [1409, 1654] |
| mel-centroid cosine to reference | 0.9998 [0.9997, 0.9999] | 0.9976 [0.9966, 0.9989] |

All four current mixes measured −1.5 dBTP with zero clipped samples. Their
integrated loudness ranged from −19.2 to −17.8 LUFS; vocal stems ranged from
−21.8 to −19.7 LUFS and −4.3 to −3.8 dBTP, also with zero clipped samples.
The original track measured −15.3 LUFS, +0.2 dBTP, and one clipped sample.
The mix and vocal trim amounts are reported separately:
`mix_sample_peak_trim_db`, `mix_trim_db`, and `vocal_trim_db`.

The notch is a measured output-side reduction, not a harmonic generator.
The model is trained almost entirely below this register (AV-41e4 records
0.48 % of target training frames at or above 550 Hz and less than 0.01 % at
or above 700 Hz, while the source reaches 1031 Hz). The filter can remove
off-harmonic energy; it cannot synthesize the missing target-voice
harmonics.

The correction strength was not increased: on one matched full-song draw,
raising `max_cut_db` from 12 to 24 dB changed inharmonic excess from 11.9 %
to 11.7 % and left dropout at 1.2 %. In the 20 s high-register A/B, the
12/24 dB variants had cosine 1.00 and no measurable feature difference.
This is one draw, not an n=4 strength comparison; it does not justify
changing the 12 dB / 3-pass default.

## What the current earshot diagnostics say

Earshot A/B on draw 1's 168-188 s vocal window found the processed and
no-notch stems nearly identical (cosine 0.98; current stem about 1.5 dB
quieter). The corresponding mix A/B was also nearly identical (cosine
0.987; current mix about 0.8 dB quieter). Realness estimates were p_real
0.956 for the vocal and 0.944 for the mix, with no window leaning
synthesized; the calibration is weak and does not prove absence of a
synthetic layer. The quality tags were weak and similar for both arms.
Nobody has listened to a render. AV-4aqs remains open for that human check.

## What this means for the two arms that failed

The brightness and cycle arms were judged on raw fork renders, before the
output-correction chain. The current served n=4 results do not change their
raw verdicts. The brightness arm's 700-1100 Hz dropout was 49.2 %; the cycle
arm's was 74.1 %. Both are worse than the current served mean of 1.06 %.
Neither raw candidate is a reason to replace the current checkpoint.

## Files

- No-notch draws: jobs `9c2b253b`, `b1d3295a`, `adfde124`, `50696749`.
- Current-checkout draws: jobs `126dec49`, `1beb9645`, `2b82af9c`,
  `ba27fa99`.
- `scratchpad/v6eval/current_truepeak_guard_n4_aggregate.json` — full-song
  current vs no-notch n=4 scoring.
- `scratchpad/v6eval/current_truepeak_guard_highreg_n4.json` — 168-188 s
  current vs no-notch n=4 scoring.
- `scratchpad/v6eval/current_truepeak_guard_n4_scores.json` — per-draw rows.
- `scratchpad/v6eval/highreg_current_guard_n4/` — current 168-188 s crops.

## Listening and next requirement

No human has listened to any current render. The output-side notch leaves
11.15 % full-song high-register inharmonic excess and 1.06 % dropout; the
user's goal of near-perfect conversion is therefore not met. The remaining
architectural path requires real high-register recordings of the target
voice or a changed content/decoder model. The existing target corpus has
insufficient coverage above 700 Hz; a synthetic pitch-shifted copy is not
evidence of authentic target-voice articulation.

Use a current-checkout listening candidate, for example:

    data/conversions/126dec49-bc8f-48c3-8d5d-a23a5de671b8/mix.wav
