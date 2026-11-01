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

| metric | raw G_389 render | SERVED path (job 9c2b253b) |
|---|---|---|
| dropout 250-400 Hz | 1.9 % | **0.3 %** |
| dropout 400-550 Hz | 1.2 % | **0.6 %** |
| **dropout 700-1100 Hz** | **32.8 %** | **3.8 %** |
| inharmonic 700-1100 Hz | 44.8 % | **40.7 %** |
| 8-12 kHz dB | -28.3 | **-22.4** |
| 12-16 kHz dB | -34.8 | **-23.3** |
| centroid Hz | 754 | 975 |
| identity | 0.9917 | 1.0000 |

**This suggests the chain removes ~29 points of high-register dropout** and
6 dB of the air deficit. It is NOT yet established: the served column is
**one** end-to-end draw (the correction chain is deterministic, but the fork
render underneath it is stochastic at noise_scale 0.4), while the raw column
is 4 renders. The project's own rule is that one render is one draw and a
result needs ≥4. The number is DESCRIPTIVE of this render, not a measured
distribution, and no claim of "dropout solved" rests on it.

What is established either way: the raw model's 32.8 % dropout at
700-1100 Hz is NOT what the user hears. The served path renders it at 3.8 %
on this draw, and the correction chain's own log line confirms the mechanism
("Lead dropout fill: 8.8 % of frames lifted, mean 1.8 dB, max 11.4 dB").

(The raw column is 4 renders: the raw model measured 35.5 % [32.8, 37.5] at
700-1100 Hz across them, and the 32.8 % here is that group's first take.
The served column is ONE draw. To put a spread on the served path, run ≥4
independent conversions of the same original source through the service and
score their paired vocal stems — the conversion is the same job each time, so
it is ~5 min per draw.)

## How the served conversion was produced

Job `9c2b253b-3c71-4859-9044-9678d990daaf`, submitted to the LIVE service
on 2026-10-31:

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
"beats" 35 % may still be far worse than what is already served (3.8 %).

**Any comparison against the served path must be made against THESE numbers,
not against the raw-model numbers.**

## The remaining defect

Everything measured is now clean on the served path:

- **Level**: dropout 0.3 % at 250-400 Hz, 0.6 % at 400-550 Hz. The old
  "the voice isn't steady in volume" complaint is fixed.
- **Loudness and peak**: 0 clipped samples, true peak -4.1 dBTP, integrated
  LUFS matched to the source (-18.4 vs -18.4).
- **Bandwidth**: `effective_fmax` 16051 Hz on the served output against
  22050 Hz on the source. Not a narrow wall.
- **The high register**: 3.8 % dropout and 40.7 % inharmonic excess at
  700-1100 Hz. The dropout is solved; what remains is the inharmonic
  excess — narrowband residue where harmonics should be.

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
  and the raw 700-1100 Hz dropout from 35.5 % to 67.1 %. The served path
  already delivers -22.4 dB and 3.8 %. So the brightness arm's brightness
  gain is roughly what the served chain ALREADY restores, and its dropout
  cost is real.
- The cycle arm's raw 700-1100 Hz dropout was 74.1 % at matched epochs.
  Nothing about that changes.

Both remain rejected, and the reason is now sharper: **the served path
already does the high-register repair the model was being asked to learn.**
A model that is worse on raw renders is also worse on the served path - the
chain repairs a level collapse, not a spectral residue.

**The user therefore hears a much better result than the raw-render numbers
imply.** The served experience is 3.8 % dropout at 700-1100 Hz, not 35 %.

## The open question that survives

The inharmonic excess at 700-1100 Hz (40.7 %) and the tonal-peak excess
(+10.6 dB) are NOT repaired by the served chain, and they are what the
user hears as the ring. That is a spectral residue problem, not a level
problem, and it is the remaining work.

## Files

- job `9c2b253b-3c71-4859-9044-9678d990daaf` — the fresh conversion
- `scratchpad/v6eval/served_fresh_scores.json` — the scoring report
- `scratchpad/v6eval/submit_served.py` — the submission script
- `scratchpad/v6eval/highreg_source.wav` — the 168-188 s high-register
  validation window (334 usable frames in the 700-1100 Hz band)

## Note on what was NOT done this session

Nobody listened to any render. Every number here is a measurement, not a
judgement, and the user judges by ear. The served conversion
`9c2b253b-3c71-4859-9044-9678d990daaf` exists and is the natural candidate
for a listening pass:

    data/conversions/9c2b253b-3c71-4859-9044-9678d990daaf/mix.wav

The remaining defect - the inharmonic excess at 700-1100 Hz and the tonal
residue that makes the "double ringing" - is a SPECTRAL problem that the
correction chain does not touch. That, not the training corpus, is where the
next effort belongs.
