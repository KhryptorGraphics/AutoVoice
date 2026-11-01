# What the SERVED path produces today, measured (2026-10-31)

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

| metric | raw G_389 (n=4) | SERVED path (n=4) |
|---|---|---|
| dropout 250-400 Hz | 1.9 % | **0.3 %** |
| dropout 400-550 Hz | 1.2 % | **0.5 %** |
| **dropout 700-1100 Hz** | **35.50 % [32.78, 37.50]** | **4.36 % [3.07, 6.37]** |
| inharmonic 700-1100 Hz | 44.94 % [42.51, 47.88] | **40.69 % [37.14, 45.54]** |
| 8-12 kHz dB | -28.3 | **-22.5** |
| 12-16 kHz dB | -34.8 | **-23.2** |
| centroid Hz | 754 | 966 |
| identity | 0.9917 | 0.9999 |

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
| identity | — | 0.9994-1.0000 |

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

## The remaining defect

Everything measured is now clean on the served path:

- **Level**: dropout 0.3 % at 250-400 Hz, 0.5 % at 400-550 Hz. The old
  "the voice isn't steady in volume" complaint is fixed.
- **Loudness and peak**: 0 clipped samples, true peak -4.1 dBTP, integrated
  LUFS matched to the source (-18.4 vs -18.4).
- **Bandwidth**: `effective_fmax` 16051 Hz on the served output against
  22050 Hz on the source. Not a narrow wall.
- **The high register**: 4.36 % dropout [3.07, 6.37] and 40.69 % inharmonic
  excess [37.14, 45.54] at 700-1100 Hz. The level dropout is fixed; what
  remains is the inharmonic excess — narrowband residue where harmonics
  should be.

That residue is the "double ringing around the voice" complaint and the
same material that would fill a dropout if the level were not repaired. It
is tonal: the served output's tonal-peak ratio is 46.0 dB against the
source's 35.4 dB (earshot, full song), and 55.2 against 49.5 on the
high-register window.

## What this means for the two arms that failed

**The verdicts on the two arms are about the MODEL, not about the served
output.** Both were judged on RAW fork renders, where the chain's dropout
fill is absent. Re-read against the served path:

- The brightness arm raised the raw model's 8-12 kHz from -28.3 to -19.8 dB
  and the raw 700-1100 Hz dropout from 35.5 % to 49.2 %. The served path
  already delivers -22.5 dB and 4.36 %. So the brightness arm's brightness
  gain is roughly what the served chain ALREADY restores, and its dropout
  cost is real.
- The cycle arm's raw 700-1100 Hz dropout was 74.1 % at matched epochs.
  Nothing about that changes.

Both remain rejected, and the reason is now sharper: **the served path
already does the high-register repair the model was being asked to learn.**
A model that is worse on raw renders is also worse on the served path - the
chain repairs a level collapse, not a spectral residue.

**The user therefore hears a much better result than the raw-render numbers
imply.** The served experience is 4.36 % dropout at 700-1100 Hz, not 35 %.

## The open question that survives

The inharmonic excess at 700-1100 Hz (40.69 %) and the tonal-peak excess
(+10.6 dB) are NOT repaired by the served chain, and they are what the
user hears as the ring. That is a spectral residue problem, not a level
problem.

A measured output-side lever now exists for it (below). The open question
is whether it is what the user wants, and that is a listening question,
not a measurement.

`docs/av9hc5-level-vs-timbre.md` establishes why a plain make-up gain
cannot fix it: the gain moves the level and leaves the spectrum untouched.

**But it CAN be fixed output-side, and the fix is now IN THE PIPELINE.**
`docs/inharmonic-notch-lever.md` documents it: a notch keyed on the
SOURCE's f0 that cuts the off-harmonic energy down to the source's own
off-harmonic fraction. `src/auto_voice/inference/fork_inharmonic_notch.py`,
wired into `_correct_fork_output` as stage 7.

Measured on all four served draws, full song:

| band | inharmonic excess (served → lever) |
|---|---|
| 250-400 Hz | 2.44 % → 2.44 % |
| 400-550 Hz | 31.09 % → 31.06 % |
| 550-700 Hz | 61.38 % → 0.53 % |
| **700-1100 Hz** | **40.69 % → 11.27 %** |

with level dropout at 700-1100 Hz going 4.36 % → 1.18 %, the low register
untouched, identity unchanged at 1.000, and every stem at exactly -1.50
dBTP with zero clipped samples.

**And end-to-end through the live service**, two conversions of the same
original Hero track with the stage on and off (jobs `baa9153b` and
`c0b293f1`): inharmonic excess at 700-1100 Hz 40.32 % → 14.64 %, at
550-700 Hz 59.8 % → 0.3 %, at 250-400 Hz 2.02 % → 2.00 % (untouched),
level dropout 5.90 % → 1.18 %, identity 1.0000 both ways.

**The registry still serves G_389 unchanged** - the model is the same, only
the output chain changed. And nobody has listened to it: the ear is the
only thing that can say whether the remaining 11-15 % is audible.

## Files

- jobs `9c2b253b`, `b1d3295a`, `adfde124`, `50696749` — the four fresh
  conversions
- `scratchpad/v6eval/served_n4_scores.json` — the n=4 scoring report
- `scratchpad/v6eval/served_fresh_scores.json` — the first (n=1) report
- `scratchpad/v6eval/submit_served.py` — the submission script
- `scratchpad/v6eval/av9hc5_probe.py` + `scratchpad/v6eval/av9hc5_scores.json`
  — the level-vs-timbre probe
- `scratchpad/v6eval/highreg_source.wav` — the 168-188 s high-register
  validation window (334 usable frames in the 700-1100 Hz band)
- `scratchpad/v6eval/highreg_n4_scores.json` — the n=4 high-register window
  report
- `scratchpad/v6eval/cut_highreg.py` — cuts the window from each draw

## Note on what was NOT done this session

Nobody listened to any render. Every number here is a measurement, not a
judgement, and the user judges by ear. Four served conversions exist and
any of them is a natural candidate for a listening pass, e.g.:

    data/conversions/9c2b253b-3c71-4859-9044-9678d990daaf/mix.wav

The remaining defect - the inharmonic excess at 700-1100 Hz and the tonal
residue that makes the "double ringing" - is a SPECTRAL problem that the
correction chain does not touch. That, not the training corpus, is where the
next effort belongs.
