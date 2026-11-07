# What the SERVED path produces today, measured (2026-11-01)

**Read this before comparing any candidate.** The served pipeline is NOT the
raw fork render: the output-correction chain (`_correct_fork_output`) changes
the numbers by more than any model change measured in this session.

## The served model

`data/fork_models/_hif0_20260917/G_389.pth`, profile
`fb17af66-8415-4ffe-81b3-600efe75b6d7`. Registry:
crepe f0, no auto-predict f0, noise_scale 0.4, db_thresh -100, pad 1.0 s,
chunk 30 s, max chunk 40 s, uv contract ON, crepe periodicity threshold 0.3.

## Historical raw-model vs served comparison (librosa 0.10.2)

This analysis used the SERVING interpreter (`autovoice-thor`: Python 3.12 /
librosa 0.10.2 / SciPy 1.17.1), so it is runtime-matched: it finds 424
usable source frames in the 700-1100 Hz band. The sections below labelled
"librosa 0.11.0" were scored with the offline interpreter (Python 3.13 /
librosa 0.11.0), whose pyin keeps 509 frames there - consistent within
themselves, but not the serving gate (see "Correction: the serving runtime
is librosa 0.10.2" at the end).

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
The metadata distinguishes measured `vocal_true_peak_dbfs` and
`mix_true_peak_dbfs` from the configured `*_ceiling_dbfs` fields. It reports
the vocal-only trim as `vocal_pre_mix_trim_db`, cumulative vocal attenuation
as `vocal_trim_db`, and the sample-peak and mix true-peak trims separately as
`mix_sample_peak_trim_db` and `mix_trim_db`.
On the 13-ballad set (2026-11-06) that vocal-only trim was 0 to -6.1 dB, one
constant gain per song set by a few transients, and it left the delivered vocal
3.5-10.8 dB under the original song's vocal-to-backing balance on four songs.
The 13 served mixes were also 0.5-9.1 LU quieter than the original songs.
docs/ballad-replay-20261103.md section 5, AV-gamn.

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
## Offline-interpreter served comparison (librosa 0.11.0 - not the serving runtime)

Four sequential end-to-end conversions with the retained notch-before-fill
order used `f0_lo=400`; four served 550 Hz-gate renders are the control.
Scores use the OFFLINE interpreter (Python 3.13.5 / librosa 0.11.0 / NumPy
2.2.6 / SciPy 1.15.3), not the serving one. The source gate has 2,223 /
212 / 26 / 509 frames in bands 250-400 / 400-550 / 550-700 / 700-1100 Hz.

| metric | served f0_lo=550 (n=4) | served f0_lo=400 (n=4) |
|---|---:|---:|
| 400-550 Hz inharmonic excess | 23.9 % [22.5, 24.6] | **13.5 % [9.9, 15.6]** |
| 400-550 Hz dropout | 0.7 % [0.0, 1.4] | 0.1 % [0.0, 0.5] |
| 250-400 Hz inharmonic excess | 1.7 % [1.4, 2.2] | 1.1 % [0.6, 2.1] |
| 550-700 Hz inharmonic excess† | 7.3 % [3.5, 9.3] | 2.3 % [1.0, 3.2] |
| **700-1100 Hz inharmonic excess** | **12.5 % [10.2, 14.1]** | **10.3 % [9.1, 11.5]** |
| 700-1100 Hz dropout | 1.1 % [1.0, 1.2] | 0.4 % [0.0, 0.8] |

The 400-550 Hz inharmonic ranges are disjoint (p=0.029). The 700-1100 Hz
inharmonic ranges overlap, so that reduction is unresolved. The dropout
ranges at 700-1100 Hz are disjoint but differ by only 0.7 percentage points.
†Only 26 source frames; treat the 550-700 Hz cell as anecdotal.

All four 400 Hz-gate mixes measured -1.5 dBTP with zero clipped samples.
Their source files are byte-identical to the 550 Hz controls (SHA-256
`831b9b06554bc6af68fad6e6e5b66a7d97f02cad5060f4f77cf255905a93d344`).
The 400 Hz setting improves the mid register; 10.3 % high-register
inharmonic excess remains and no human has listened.

An after-fill order experiment was rejected under the same librosa 0.11.0
scorer. Its 400-550 Hz inharmonic excess was 18.6 % [17.2, 20.9], worse
than retained order's 13.5 % [9.9, 15.6]; 700-1100 Hz was 10.0 %
[8.6, 11.0] versus 10.3 % [9.1, 11.5] (overlapping). It does not justify
moving the stage.

The 15.9 % / 7.7 % served values (and the 5.4 % offline value) came from
librosa 0.10.2 - the serving runtime - and are the runtime-matched ones (see
the correction at the end). Conversion metadata records the library versions
and the effective notch threshold.

The local sample audit found 71 scorer-valid 700-1100 Hz frames across
seven singing-labeled profile recordings (~0.82 s); 64 short local-video
clips contributed none. This is authentic material, but too sparse to
establish high-register articulation coverage. The v6 fine-tune candidate
trained on 78 samples and still failed its high-register acceptance gate.

## Listening and current-order status

The f0_lo=400 served n=4, scored with the serving interpreter (librosa
0.10.2), leaves 7.7 % [6.5, 9.2] full-song 700-1100 Hz inharmonic excess and
0.83 % [0.71, 0.94] dropout. The 10.3 % / 0.4 % figures were the offline
librosa 0.11.0 scorer.

The user listened to the retained-order candidate
(`data/conversions/720b64b0-4681-4f6a-bfa7-9992660bc0a1/mix.wav`) on
2026-11-02: "better but not perfect", "minorly staticy around the voice
itself", and the source's soft-versus-belted contrast is easier to hear
than the conversion's. The near-perfect goal therefore remains unmet;
AV-4aqs stays open.

## Correction: the serving runtime is librosa 0.10.2 (2026-11-02)

The service is systemd `autovoice`, which runs
`/home/kp/anaconda3/envs/autovoice-thor/bin/python` - Python 3.12.12,
librosa 0.10.2, NumPy 2.2.6, SciPy 1.17.1 - and its jobs record
`librosa_version: 0.10.2` in their conversion metadata. An earlier note in
this repo claimed the service ran librosa 0.11.0; that was the offline
interpreter (`/home/kp/anaconda3/bin/python`, Python 3.13.5), checked by
mistake. The pyin gate - in the scorer and in the notch stage itself -
depends on the version, so served numbers must be scored with the serving
interpreter. `requirements-runtime.txt` is back to `librosa>=0.10` (the
`==0.11.0` pin had broken the dependency-contract hash).

Served n=4 per cohort, all byte-identical uploads of the same original
(SHA-256 `831b9b06554bc6af...`), studio preset, scored with the serving
interpreter (`scratchpad/runtime010/scores.json`):

| metric | no notch | notch, 400 Hz (retained) |
|---|---|---|
| 700-1100 Hz inharmonic excess | 40.7 % [37.1, 45.5] | **7.7 % [6.5, 9.2]** |
| 700-1100 Hz level dropout | 4.36 % [3.07, 6.37] | **0.83 % [0.71, 0.94]** |
| 400-550 Hz inharmonic excess | 31.1 % [30.3, 31.8] | **15.9 % [14.8, 16.8]** |
| 250-400 Hz inharmonic excess | 2.4 % [2.0, 3.1] | 2.3 % [1.7, 2.7] |

## Effort follower (stage 8) and the served dropout gap (2026-11-02)

Stage 8, `fork_effort_follow.follow_effort`, runs after the notch and before
the dropout fill (`fork_hq_effort_follow_strength`, default 1.0; 0 disables
it). Its docstring carries the full offline and served-path tables.

Served n=4 per cohort, unpaired, serving interpreter
(`scratchpad/runtime010_v2/scores4.json`), 700-1100 Hz level dropout:
retained (no follower) 0.83 % [0.71, 0.94], follower first version 2.12 %
[1.42, 2.83], current follower 1.77 % [1.42, 2.12]. Paired, the follower
accounts for a small part of that gap (below); the cause of the rest is not
established.

- The extra frames are the same few events in every cohort: 154.90-154.91,
  168.93-169.19, 177.02-177.08 and 192.17-192.26 s
  (`scratchpad/effort/dropout_frames.py`).
- The uploads are byte-identical (all 46 Hero uploads, SHA-256
  `831b9b06...a93d344`). The historical jobs' vocal stems were not saved,
  so their separations cannot be compared directly. Indirect evidence only:
  their instrumental stems (same separation pass) agree within 0.3 dB at
  those frames, four fresh htdemucs_ft separations - the service's
  separator, `config/gpu_config.yaml` - match `hero_src.wav` within 0.1 dB
  there (`scratchpad/effort/sep_ft.py`), and every render is time-aligned
  (lag 0, `lag_probe.py`).
- Every served job draws its own separation (demucs `shifts=1` takes a
  random offset per call; htdemucs_ft run to run, frame level |diff| median
  0.075 dB, p99 1.6 dB, max 9.9 dB) and its own render (noise_scale 0.4),
  so served cohorts cannot be paired.

Paired served-path replay (`scratchpad/effort/ft_replay.py`,
`score_ftab.sh`): four htdemucs_ft separations of the original upload, each
rendered through the fork bridge, then the production `_match_lead_level` +
`_correct_fork_output` (studio preset) on that SAME render with the follower
off and on. `chain_ab.py`'s replay and `_correct_fork_output` are
bit-identical on the same input. Serving interpreter, render minus source:

| metric | follower off | follower on |
|---|---|---|
| 700-1100 Hz level dropout | 2.06 % [1.65, 2.36] | 2.24 % [1.65, 2.83] |
| dropout frames per render (of 424) | 8 / 7 / 10 / 10 | 8 / 7 / 11 / 12 |
| 700-1100 Hz inharmonic excess | 11.4 % [8.9, 12.4] | 7.3 % [4.5, 10.2] |
| 250-400 Hz inharmonic excess, score_multi window | 2.4 % [2.1, 2.6] | 5.0 % [4.7, 5.5] |
| 250-400 Hz off-harmonic share, 1-bin window (source 0.310) | 0.406 [0.404, 0.408] | 0.386 [0.383, 0.388] |
| harmonic tilt contrast ratio | 0.57 | **1.00** |
| sung-frame centroid | +231 Hz | **+12 Hz** |
| 10-16 kHz share, soft / belt frames | +6.8 / +5.3 dB | **+0.4 / +1.0 dB** |
| 6-10 kHz share, soft / belt frames | -8.5 / +0.2 dB | -3.7 / -1.7 dB |
| long-term 10-16 kHz | +2.3 dB | -0.1 dB |

- On the same render the follower adds 0, 0, 1 and 2 dropout frames
  (+0.18 pp; the scorer calls it noise): frames close to the scorer's 6 dB
  gate at 192.20-192.26 s tip over. Its frame RMS change through the fill
  is p99 0.37 dB, max 1.9 dB; the first version's was p99 1.1 dB, max
  3.4 dB (`norm_v1_vs_cur.py`).
- With the follower OFF the replays still drop 7-10 frames, at 154.90-154.91
  s in three of four renders: that event follows the render draw (raw dip
  about -8 dB on some draws, -5 dB on others). The fill's 5-frame median
  erases a 1-2 frame deficit entirely and its 0.15 s smoothing delivers
  4.6 of 12 dB to a 3-frame one (AV-a4dv).
- The retained cohort's 0.71-0.94 % is the low end of the 16 notch-on
  renders measured so far (the other twelve, served and replayed:
  1.42-2.83 %). It was the only cohort run one job at a time; neither
  concurrency nor the historical stems were tested as a cause.
- The 250-400 Hz result depends on the scorer's harmonic window.
  score_multi counts +-3 % of k*f0 as harmonic, narrower than one FFT bin
  (21.5 Hz) for the lowest harmonics at these pitches, so most of their
  energy reads as off-harmonic (the source itself scores 0.533 off). On
  that window the follower reads 2.4 % -> 5.0 % (offline paired 1.6 % ->
  4.2 %, served unpaired 2.3 % -> 4.5 %). Floored at one bin, the
  off-harmonic share FALLS toward the source's: served-path 0.406 -> 0.386,
  offline 0.402 -> 0.383, source 0.310 (`ftab_floor.py`). Not a confirmed
  cost, and not checked by ear (AV-n6o4).
- The match bands are what remove the 10-16 kHz bed: without them the soft
  frames stay +3.7 dB over the source and the centroid +49 Hz.
- A post-ISTFT frame-RMS lock (`lock_proto.py`) was prototyped and not
  shipped: it put the replays' dropout back to 8/7/10/10 but cut the frame
  RMS change only from p99 0.37 to 0.30 dB - a frame-rate gain cannot
  remove a change that sits inside the frame.

Nobody has listened to the follower yet. The served candidates are jobs
`4623a6cc`, `8493aa3d`, `094d5f8f` and `c926d400`.

## Hum stage (stage 1): crepe grid drift and the duck guard (2026-11-02)

The residual 700-1100 Hz dropouts on the effort-follower chain were not the
fill's blind spot first: per-stage tracing (`scratchpad/effort/stage_trace.py`)
showed the hum stage's 14 dB duck deepening 154.90, 168.93 and 169.17 s on
every draw. Two separate causes, both measured:

- **Grid drift (a bug).** `_voicing` runs crepe in 20 s chunks with
  `pad=True`, which yields one frame more per full chunk than its duration.
  Stitched as-is, frame j inside chunk k sat at (j - k) x 10 ms while
  `_to_samples` and `_silence_mask` read it as j x 10 ms: Hero gave 25868
  frames for 25856 frames of time, so the duck landed up to 120 ms late - on
  the note after each gap. Verified on a synthetic burst (7004 frames for
  7001, centre 2.5 frames late) and on Hero: the full-length array read k
  frames later equals a single-chunk excerpt in all 101 event rows
  (`scratchpad/effort/crepe_drift.py`). 168.93 and 169.17 s were drift alone.
- **crepe misses some high belts.** On the true grid the source's crepe
  periodicity is 0.13-0.24 at 154.80-154.88 s (~990 Hz) where pyin's voicing
  probability is 0.95-1.00, and 0.00 at 192.24-192.29 s (~970 Hz) against
  0.89-0.98.

Fix: each full chunk's duplicated boundary frame is dropped, and the duck
never fires where pyin hears the source sing. The pyin pass is the notch's
own (`fork_inharmonic_notch.source_pitch`), computed once in
`_correct_fork_output` and shared; the notch's voiced gate and output are
bit-identical to before (full Hero gate, 140-200 s stage output).

Paired served-path replay, the same four htdemucs_ft service separations and
raw fork renders through the production chain (studio preset, follower on),
serving interpreter (`scratchpad/effort/hum_fix_ab.py`, `hf_events.py`,
`score_hf.sh`). 700-1100 Hz dropout frames of 424 per render, 250-400 Hz
frames, and the hum stage's own target (10 ms RMS over source crepe-unvoiced
and not-sung frames minus sung frames, one true-grid definition for every
arm):

| draw | 700-1100 stock | 700-1100 shipped | 250-400 stock | 250-400 shipped | hum vs sung, stock -> shipped |
|---|---|---|---|---|---|
| 0 | 8 | 4 | 11 | 9 | -23.2 -> -23.5 dB |
| 1 | 7 | 3 | 18 | 11 | -24.8 -> -25.0 dB |
| 2 | 11 | 4 | 15 | 9 | -24.7 -> -25.1 dB |
| 3 | 12 | 5 | 19 | 9 | -22.9 -> -23.4 dB |

score_multi means: 700-1100 Hz dropout 2.24 % -> 0.94 %, 250-400 Hz 0.35 %
-> 0.21 %; inharmonic excess, HF bands, centroid, identity and the static
metrics all within render noise. Every draw improves on both dropout bands
and none raises the hum. The draws are paired, so the evidence is the four
paired deltas and the per-event attribution; an exact two-sided sign test on
four pairs cannot go below p = 0.125 (score_multi's "REAL" is its unpaired
disjoint-range criterion, p = 0.029 at 4 vs 4).

Grid fix alone (duck unguarded) also reaches 0.94 % mean but keeps 154.90-
154.91 s; the guard removes those. Its one apparent cost, 192.20-192.21 s on
draw 3, is the fill: unguarded, the duck also cut 192.24-192.29 s, the wider
dip got 3.8 dB of fill instead of 2.9 dB, and the two raw-render frames
(deficit +17.5 dB before any stage) fell under the threshold by accident
(`stage_trace_hf.py`).

What survives, 4 frames per render on average:

- 177.02-177.08 s, 3 frames on every draw and every arm: no duck fires
  there. The scorer's reference separation (`hero_src.wav`) holds a quiet
  sung note there (-23.7 to -24.7 dB re its loud median, pyin voicing
  0.76-0.95, ~790 Hz); every htdemucs_ft service key has the same frames
  4-6 dB quieter (-28.2 to -31.1 dB) and unvoiced (0.03-0.09), and the fork
  render is already 9-12 dB under the reference before any stage. The hum
  stage's source gate (-30/-42 dB re the source's peak) then takes that
  quiet key region a further 6-10 dB down. On its own input the gate is
  gating a quiet unvoiced region, not misreading a sung frame; the frames
  count as dropouts because the reference separation kept the note
  (AV-08tj).
- 192.20-192.26 s on some draws: dips already in the raw render that the
  fill only partly lifts (AV-a4dv, closed - see below).

Rejected: a faster dropout fill (2026-11-02). Taking the larger of the
shipped lift and a fast one (median 3, max 5, box 5) on the same four
pre-fill signals (`prefill_cache.py`, bit-exact to the shipped renders) gave
700-1100 Hz dropout 4/3/4/5 -> 3/3/3/3 frames and 250-400 Hz 9/11/9/9 ->
4/2/3/1, and on the true time grid (`fill_mod.py`: each live run band-passed
on its own) cut the output's 3-8 Hz level error against the source 1.20 ->
1.10 dB on every draw, with no measurable new overshoot, peak or inharmonic
excess (`fill_variants.py`, `fill_overshoot.py`). It is not shipped because
of what it lifts (`fill_harm.py`): the frames it newly fills carry an
off-harmonic share of 0.63-0.74 where the source sings cleanly (0.19-0.25),
against 0.44-0.45 on the frames the shipped fill lifts and 0.33 on sung
frames overall, and about 220 more per render have no source f0 at all. The
short dips are where the decoder emits junk; lifting them trades a dip for
louder noise.

Served confirmation (service restarted on `44236b55`): four conversions of
the original through the live service, one at a time - jobs `f2dd8582`,
`d32130ca`, `4451ea80` and `c7931b4f`, in the web history; the service log
shows the guard ("0.7% spared where pyin hears singing"). Against the
previous served cohort (effort follower: `4623a6cc`, `8493aa3d`, `094d5f8f`,
`c926d400`). These are independent draws, so score_multi's unpaired
disjoint-range test is the right one here (p = 0.029 at 4 vs 4):

| metric | previous served | this change |
|---|---|---|
| 700-1100 Hz dropout | 1.77 % [1.42, 2.12] | **1.00 % [0.94, 1.18]** (disjoint) |
| 250-400 Hz dropout | 0.31 % [0.24, 0.42] | **0.16 % [0.15, 0.18]** (disjoint) |
| 700-1100 Hz inharmonic excess | 5.06 % [3.52, 6.57] | 5.86 % [3.52, 7.42] (overlap) |
| 400-550 Hz inharmonic excess | 14.01 % [13.37, 14.93] | 14.51 % [13.53, 15.38] (overlap) |
| off-harmonic share, soft / belt | +0.045 / +0.089 | +0.042 / +0.087 |
| 10-16 kHz on unvoiced frames vs source | +0.31 dB | +0.34 dB (overlap) |

The first attempt submitted all four at once and failed: the fork processes
exhausted GPU memory on this shared box and every job hit the 900 s svc
timeout (`e40ab8ca`, `fa9db1a5`, `2a5d80c5`, `15aad00a`, also in the
history; AV-auyr).

Nobody has listened to this change.
