# The inharmonic notch: an output-side reduction for high-register residue (2026-11-01)

**The source-referenced notch reduces high-register residue but does not close it.**
The October 31 offline n=4 results below are historical; fresh current-checkout
n=4 results are included later.

## The result (offline lever, f0_lo=550 Hz, 2026-10-31)

Settings: cut 12 dB max, make-up gain 12 dB max, 3 correction passes,
true-peak ceiling -1.5 dBTP.

| band | inharmonic excess (served → lever) | level dropout (served → lever) |
|---|---|---|
| 250-400 Hz | 2.44 % → **2.44 %** | 0.32 % → 0.32 % |
| 400-550 Hz | 31.09 % → **31.06 %** | 0.39 % → 0.39 % |
| 550-700 Hz | 61.38 % → **0.53 %** | 0.00 % → 0.00 % |
| **700-1100 Hz** | **40.69 % → 11.27 %** | 4.36 % → **1.18 %** |

Per draw, inharmonic excess at 700-1100 Hz:
served 40.73 / 45.54 / 39.34 / 37.14 → lever 11.92 / 12.99 / 11.51 / 8.68.
Every draw improves by 25-33 points; none regresses. Level dropout after
the lever: 1.18 / 1.18 / 1.42 / 0.94 %.

Other metrics: spectral centroid 966 → 1147 Hz, 3-8 kHz -15.6 → -13.6 dB,
8-12 kHz -22.5 → -21.1 dB, 12-16 kHz -23.2 → -21.4 dB. The score's
mel-centroid cosine to its reference group was 1.000; this is not a
speaker-identity measure.

**The low register is untouched** (250-400 Hz identical to two decimals,
400-550 Hz within 0.03 points) even though the lever runs over the whole
song. That is by construction, not by luck - the notch is gated on the
source's pitch.

**No clipping.** All four levered stems: true peak exactly -1.50 dBTP,
0 samples at or above 0.999, sample peak -1.51 to -1.87 dBFS. The served
stems they came from were at -1.50 to -2.88 dBTP.

## Why a notch, and not the existing de-tonalizer

`fork_voice_polish.detonalize_hf` already removes tonal ringing, but it is
masked off below 7.5 kHz and could not simply be moved down: that band
carries identity, and its mechanism (a bin standing above its own local
21-bin median floor) is blind to this defect anyway. The high-register
defect is not one tone sticking up - it is the whole spectrum in the band
being misplaced relative to the source's harmonic series.

The fix is to define the cut against the SOURCE's f0 rather than against a
local floor. The source is a clean voice: its off-harmonic energy in the
band is breath noise, and that is exactly the amount of off-harmonic energy
a correct render should have.

`experiments/score_multi.py` already defines "inharmonic excess" this way -
the fraction of in-band magnitude that does not fall on a harmonic of f0
(+-3 % around harmonics 1..12, band f0/2 to 6 kHz). So the lever aims at
exactly the quantity the metric measures.

## The correction

Per frame, the metric computes `frac = off / (on + off)` from STFT
magnitudes. Requiring the render's `frac` to equal the source's `sf` and
writing the attenuated off-harmonic energy as `off * g`:

    off*g / (on + off*g) = sf   =>   g = sf*on / (off*(1 - sf))

`g < 1` is the cut; `g > 1` would be a boost and is never wanted. `g` is
bounded below by `10**(-max_cut_db/20)` and never exceeds 1.

## Four things that had to be right, each found by measurement

**1. The f0 must be the METRIC's f0.** The metric gates frames with
`voiced_flag`, `voiced_prob > 0.8`, a level gate of `esn > -25` dB, AND an
octave-error rejection (a frame is kept only if the peak at 2*f0 is not
above the peak at f0). A first version used a different gate; its mask
landed on different frames and a different pitch than the mask being
scored, and the correction plateaued at 16.6 % excess no matter how deep
the cut went.

**2. Iterate on the re-analysis.** The correction is computed on an
in-memory STFT, but the metric measures a file read back from disk, and
`STFT -> iSTFT -> file -> STFT` is not the identity. Each pass therefore
re-analyses the signal it actually produced and corrects again. The mean cut
converges 5.17 -> 1.58 -> 0.93 dB over three passes; more passes do not
help.

**3. The cut must be exact, not proportional.** A proportional depth
(excess dB mapped to a dB cut) overshoots: measured, it drove the excess to
**-6.15 %**, i.e. the render became cleaner than the source, which means
real voice had been cut. The exact formula above cannot overshoot, because
it targets the source's fraction rather than a fixed depth.

**4. And it must be gated on the source's pitch.** Without an `f0_lo` gate
the notch also fired on low frames, and the 250-400 Hz band went from
+27.5 % to **-0.8 %** excess on the high-register window - it was eating
the singer's breath noise in the low register. With the gate at 550 Hz the
low band is bit-identical.

## The collateral is structurally zero — but only after a fix

Run on the SOURCE itself (source as both key and input), the notch changes
the source's 550-6000 Hz band energy by **-0.00 dB** and its harmonic
energy by **-0.00 dB**. That is not tuning: the correction is referenced to
the source's own off-harmonic fraction, so a clean source is already at its
target and the required gain is exactly 1.

**This was only true of the NOTCH. The make-up gain had a bug that the
offline measurement could not see**, because the offline lever and the
production stage share the bug and therefore agreed with each other.

`np.convolve` zero-pads, so smoothing a gain array of all ones with a
9-tap Hannian kernel returns values as low as **0.625** in the first and
last four frames — a 4 dB ATTENUATION of audio that needed no change at
all. The gain exists to restore level the notch took; a value below 1.0
means it is removing level that was never lost.

The fix is edge padding: `np.pad(gain, pad, mode="edge")` before the
convolution, so an all-ones gain array stays exactly all ones.

Two things are worth stating plainly:

- **The real-data measurements were unaffected.** Re-measured on all four
  served draws after the fix, every metric is identical to within 0.005
  (inharm_700-1100 11.272 -> 11.272, inharm_550-700 0.533 -> 0.531,
  inharm_250-400 2.440 -> 2.435, dropout_700-1100 1.179 -> 1.179). The
  dilution only affected the first and last ~90 ms of the song, where
  there is no scored high-register frame.
- **A synthetic test found it, and a synthetic test is what it took.**
  The real-data measurement could not see it, because the defect is
  invisible in an aggregate and the offline/production agreement test
  compared two implementations that shared the defect.

This is the benchmark `detonalize_hf` set for itself ("clean-source
residual"), and the stage now meets it exactly — for the notch AND for the
gain.

## What the correction cannot reach

The correction targets the source's off-harmonic fraction, but the
achievable floor is set by the STFT round trip: attenuating off-harmonic
bins redistributes some energy into the harmonic bins, so the result lands
ABOVE the target rather than exactly on it. Measured on a synthetic case
with a pure off-harmonic tone at 1.5*f0: the off-harmonic energy falls
3.7x (660.9 -> 177.6) and the harmonic energy RISES by 53 % (1225 ->
1879), leaving the fraction at 1.26x the source's instead of 1.0x.

The offline October 31 result was 11.27 %; the fresh current-checkout n=4
result is 11.15 %. The stage removes residue but cannot restore harmonics.

## The headroom problem, and why the ceiling is a TRUE peak

The make-up gain lifts the level, and the served stem already sits at
-2.80 dBFS sample peak. With no headroom rule the levered stems reached
**0.00 dBFS with up to 4 samples at or above 0.999** - they clipped.

A sample-peak ceiling of -1.0 dBFS is not enough. The waveform between
samples is what a DAC reconstructs, and the repo already measures that
(`auto_voice.audio.true_peak`). With a -1.0 dBFS sample ceiling the levered
stems measured -1.00 dBFS sample peak but **-0.63 dBTP** - above the repo's
own -1.0 dBTP vocal ceiling.

So the ceiling is a true peak, applied with the repo's own
`limit_true_peak`. All four levered stems measure exactly **-1.50 dBTP**,
0 clipped samples, sample peak -1.51 to -1.87 dBFS.

The trim is a single constant, not a limiter: the defect is a handful of
samples at full scale, and a dynamic limiter would change the level
envelope of the entire vocal to remove them. Same reasoning as the repo's
existing `fork_hq_vocal_true_peak_dbfs`.

In the current API path, the vocal is limited before summing; the mix's
sample-peak and post-sum true-peak trims then scale the returned stems.
`vocal_true_peak_dbfs` and `mix_true_peak_dbfs` report the measured exported
signals; the separate `*_ceiling_dbfs` fields report the configured targets.
`vocal_pre_mix_trim_db` is the vocal-only trim; `vocal_trim_db` is the
cumulative vocal trim including the post-sum gains. Final mixes measured
-1.5 dBTP; vocal stems measured -3.8 to -4.3 dBTP.

The retained production notch reduces residue, but the runtime-matched
librosa 0.11.0 served n=4 still has 10.3 % [9.1, 11.5] full-song
700-1100 Hz inharmonic excess and 0.4 % [0.0, 0.8] dropout. It cannot
synthesize the missing target-voice harmonics.

## Production stage at the former 550 Hz gate

The notch remains in `_correct_fork_output` BEFORE the dropout fill. The
table below compares the offline stage to the then-current production order,
at `f0_lo=550`; the current pipeline gate is 400 Hz and its served results
are reported below. The stage is enabled by default and
`fork_hq_inharmonic_notch: false` disables it.

**The historical production stage reproduced the offline lever exactly.**
Measured side by side on all four draws:

| metric | offline lever | historical production stage | delta |
|---|---|---|---|
| inharm_700-1100 | 11.272 | 11.272 | -0.000 |
| inharm_550-700 | 0.533 | 0.531 | -0.002 |
| inharm_250-400 | 2.440 | 2.435 | -0.005 |
| dropout_700-1100 | 1.179 | 1.179 | +0.000 |

The helper function's direct-call default remains `f0_lo=550` for existing
offline callers. The production pipeline passes its own explicit configured
threshold, whose fallback is 400 Hz; the two defaults are intentionally
different.

## Historical analysis at the former f0_lo=550 Hz gate

These saved 550 Hz results used the `autovoice-thor` scoring environment
(librosa 0.10.2 / scipy 1.17.1; 424 source frames in 700-1100 Hz). The
production runtime used librosa 0.11.0 / scipy 1.15.3 and yields 509 such
frames. The old table remains historical; use the runtime-matched results
below for current comparisons.

| band | no-notch baseline (n=4) | current checkout (n=4) |
|---|---|---|
| 250-400 Hz inharmonic excess | 2.44 % [2.02, 3.08] | 2.23 % [1.81, 2.83] |
| 400-550 Hz inharmonic excess | 31.09 % [30.31, 31.79] | 31.04 % [29.92, 31.93] |
| 550-700 Hz inharmonic excess | 61.38 % [58.19, 65.06] | −0.32 % [−1.25, 1.09]† |
| **700-1100 Hz inharmonic excess** | **40.69 % [37.14, 45.54]** | **11.15 % [9.61, 12.61]** |
| 700-1100 Hz level dropout | 4.36 % [3.07, 6.37] | **1.06 % [0.71, 1.18]** |

† Only 30 source frames; this band is anecdotal. At 700-1100 Hz the
current distribution does not overlap the baseline; every draw improves,
but 11.15 % residual is not near-perfect.

The 168-188 s high-register window also improves:

| 168-188 s metric | no-notch baseline (n=4) | current checkout (n=4) |
|---|---|---|
| 700-1100 Hz dropout | 2.54 % [1.50, 4.19] | 0.00 % [0.00, 0.00] |
| 700-1100 Hz inharmonic excess | 45.1 % [41.9, 50.5] | 11.6 % [9.1, 14.0] |
| spectral centroid | 1343 Hz [1317, 1381] | 1525 Hz [1409, 1654] |
| mel-centroid cosine to no-notch reference | 0.9998 [0.9997, 0.9999] | 0.9976 [0.9966, 0.9989] |

Every current mix measured -1.5 dBTP with zero clipped samples. Vocal
stems measured -3.8 to -4.3 dBTP, also without clipping.

| draw | mix LUFS | mix dBTP | sample-peak trim | post-sum TP trim | vocal LUFS | vocal dBTP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | -17.8 | -1.5 | -0.98 dB | -1.06 dB | -19.7 | -3.8 |
| 2 | -19.2 | -1.5 | -1.50 dB | -1.06 dB | -21.8 | -4.0 |
| 3 | -19.1 | -1.5 | -1.83 dB | -1.06 dB | -21.5 | -4.3 |
| 4 | -18.1 | -1.5 | -1.10 dB | -1.27 dB | -20.0 | -4.3 |

The source measured -15.3 LUFS, +0.2 dBTP and one clipped sample. The
true-peak ceiling is a safety limit, not a loudness-match target.

## Diagnostics and remaining limit

Earshot A/B on draw 1 found current and baseline high-register vocals nearly
identical (cosine 0.98; current about 1.5 dB quieter); the corresponding
mixes were also nearly identical (cosine 0.987; current about 0.8 dB
quieter). Realness estimates were p_real 0.956 for the vocal and 0.944 for
the mix, but the calibration is weak. These are machine diagnostics, not a
human listening result.

A one-draw full-song cap check (same pre-notch input) changed inharmonic
excess from 11.9 % at a 12 dB cap to 11.7 % at 24 dB, with dropout 1.2 %
in both outputs. The 20 s high-register A/B had cosine 1.00 and no
measurable feature change. This is not an n=4 strength test and does not
justify raising the 12 dB / 3-pass default.

## Runtime-matched served comparison (librosa 0.11.0)

Runtime-matched scores were produced with Python 3.13.5, librosa 0.11.0,
NumPy 2.2.6, and SciPy 1.15.3. `requirements-runtime.txt` pins librosa
0.11.0; score and conversion metadata record the installed versions.
The source gate yields 2,223 / 212 / 26 / 509 frames in the 250-400 /
400-550 / 550-700 / 700-1100 Hz bands.

| band / metric | served f0_lo=550 (n=4) | served f0_lo=400 (n=4) |
|---|---:|---:|
| 250-400 Hz inharmonic excess | 1.7 % [1.4, 2.2] | 1.1 % [0.6, 2.1] |
| 400-550 Hz inharmonic excess | 23.9 % [22.5, 24.6] | **13.5 % [9.9, 15.6]** |
| 400-550 Hz dropout | 0.7 % [0.0, 1.4] | 0.1 % [0.0, 0.5] |
| 550-700 Hz inharmonic excess† | 7.3 % [3.5, 9.3] | 2.3 % [1.0, 3.2] |
| **700-1100 Hz inharmonic excess** | **12.5 % [10.2, 14.1]** | **10.3 % [9.1, 11.5]** |
| 700-1100 Hz dropout | 1.1 % [1.0, 1.2] | 0.4 % [0.0, 0.8] |

The 400-550 Hz inharmonic ranges are disjoint (rank-sum p=0.029); the
700-1100 Hz inharmonic ranges overlap, so that improvement is unresolved.
The 700-1100 Hz dropout ranges are disjoint (p=0.029), but the absolute
reduction is 0.7 percentage points. †The 550-700 Hz band contains only 26
source frames and remains anecdotal. This is a mid-register improvement,
not a near-perfect high-register conversion.

All four 400 Hz-gate mixes measured -1.5 dBTP with zero clipped samples.
Their source WAVs are byte-identical to the 550 Hz group
(SHA-256 `831b9b06554bc6af68fad6e6e5b66a7d97f02cad5060f4f77cf255905a93d344`).

## Offline probe and rejected after-fill order

With the same librosa 0.11.0 runtime, applying the helper once to the four
no-notch finished stems changed 400-550 Hz inharmonic excess from 24.3 %
[22.5, 27.2] to 6.1 % [3.3, 9.2] (disjoint; p=0.029); 700-1100 Hz stayed
10.9 % [9.5, 12.0]. This remains a stem-level probe using `hero_src.wav`
as the source key, not a replacement for the served result.

The attempted after-fill production order was also rescored with librosa
0.11.0. It yielded 18.6 % [17.2, 20.9] inharmonic excess at 400-550 Hz,
worse than the retained order's 13.5 % [9.9, 15.6]. At 700-1100 Hz it was
10.0 % [8.6, 11.0] vs 10.3 % [9.1, 11.5] (overlapping). The code therefore
retains notch-before-fill. The earlier librosa 0.10.2 output (26.7 %)
was not a valid production comparison; version mismatch changed the pyin
mask and frame counts.

The helper API keeps its direct-call default at `f0_lo=550`; the pipeline
explicitly supplies its 400 Hz fallback. Job reports record `librosa_version`
and `inharmonic_notch_f0_lo_hz` so future service runs identify the gate used.

Retained pre-fill 400 Hz jobs: `720b64b0`, `cafcddda`, `bd7731b2`,
`3fde1c2d`. Retained 550 Hz jobs: `126dec49`, `1beb9645`, `2b82af9c`,
`ba27fa99`. Rejected after-fill jobs: `7a929665`, `5153aa77`, `505d9406`,
`9bc1d2cc`. Runtime-matched reports:
`scratchpad/v6eval/runtime_librosa011_stem_probe/scores.json`,
`served_pair.json`, `offline_pair.json`,
`scratchpad/v6eval/post_fill_order_n4/scores_base_librosa011.json`.

The historical G_389 training audit (AV-41e4) covered 10 singing files and
168,079 voiced frames: 0.48 % were at or above 550 Hz and less than 0.01 %
at or above 700 Hz, while the source reaches 1031 Hz. A separate local
profile-sample audit found seven singing-labeled recordings with 71 scorer-
valid 700-1100 Hz frames total (about 0.82 s); the 64 short local-video clips
contained none. The v6 fine-tune corpus included 78 samples and still failed
the high-register acceptance gate. Existing authentic material is sparse;
the notch only attenuates off-harmonic energy and cannot synthesize missing
target-voice harmonics.

Historical 550 Hz jobs: `126dec49`, `1beb9645`, `2b82af9c`, `ba27fa99`.
Retained 400 Hz served jobs: `720b64b0`, `cafcddda`, `bd7731b2`,
`3fde1c2d`. Rejected after-fill jobs: `7a929665`, `5153aa77`, `505d9406`,
`9bc1d2cc`.

Runtime-matched reports (Python 3.13.5, librosa 0.11.0):

- `scratchpad/v6eval/runtime_librosa011_stem_probe/scores.json`
- `scratchpad/v6eval/runtime_librosa011_stem_probe/served_pair.json`
- `scratchpad/v6eval/runtime_librosa011_stem_probe/offline_pair.json`
- `scratchpad/v6eval/runtime_librosa011_stem_probe/order_pair.json`
- `scratchpad/v6eval/post_fill_order_n4/scores_base_librosa011.json`

**Status: the retained notch runs before fill with a 400 Hz pipeline fallback.**
Runtime-matched served n=4 still has 10.3 % [9.1, 11.5] 700-1100 Hz
inharmonic excess and 0.4 % [0.0, 0.8] dropout. The user's no-residual /
no-artifact goal is unmet, and no human has listened. AV-4aqs remains open.
