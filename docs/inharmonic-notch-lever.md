# The inharmonic notch: an output-side reduction for high-register residue (2026-11-01)

**The source-referenced notch reduces high-register residue but does not close it.**
The October 31 offline n=4 results below are historical; fresh current-checkout
n=4 results are included later.

## The result

Settings: cut 12 dB max, make-up gain 12 dB max, f0_lo 550 Hz, 3
correction passes, true-peak ceiling -1.5 dBTP.

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

In the current API path, the standalone vocal ceiling runs before summing;
the sample-peak and post-sum true-peak trims then scale the returned stems.
The metadata keeps `vocal_trim_db`, `mix_sample_peak_trim_db`, and
`mix_trim_db` separate. Final mixes measured -1.5 dBTP; final vocal stems
measured -3.8 to -4.3 dBTP.

## What it does NOT fix

The lever removes some energy between harmonics. It cannot synthesize the
missing target-voice harmonics, and the fresh n=4 result still has 11.15 %
full-song high-register inharmonic excess.

## Shipped: the stage is in the pipeline

`src/auto_voice/inference/fork_inharmonic_notch.py`, wired into
`_correct_fork_output` as stage 7, before the dropout fill. Default on;
`fork_hq_inharmonic_notch: false` turns it off, and the cut depth, make-up
gain, f0 threshold and pass count are all config keys.

**The production stage reproduces the offline lever exactly.** Measured
side by side on all four draws:

| metric | offline lever | production stage | delta |
|---|---|---|---|
| inharm_700-1100 | 11.272 | 11.272 | -0.000 |
| inharm_550-700 | 0.533 | 0.531 | -0.002 |
| inharm_250-400 | 2.440 | 2.435 | -0.005 |
| dropout_700-1100 | 1.179 | 1.179 | +0.000 |

So the numbers in this document describe the shipped code.

## Current-checkout end-to-end result (2026-11-01)

Four fresh full-song draws used the current checkout, notch enabled and the
shared vocal/mix true-peak guard. Baseline figures are the four no-notch
draws from 2026-10-31; all vocals are aligned to the same Hero source.

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

A separate one-draw `f0_lo=400 Hz` probe reduced the 400-550 Hz inharmonic
metric from 30.9 % to 5.4 % and dropout from 0.6 % to 0.0 %, while leaving
700-1100 Hz at 11.9 % and 250-400 Hz at 3.1 %. This is not an n=4
validation and is not shipped; the production gate remains 550 Hz.

The target profile's corpus has 0.48 % of frames at or above 550 Hz and
less than 0.01 % at or above 700 Hz (AV-41e4), while the source reaches
1031 Hz. The notch only attenuates off-harmonic energy; the architecture
needs genuine target-voice high-register training data or a different
content/decoder model to restore missing harmonics.

Current-checkout jobs: `126dec49`, `1beb9645`, `2b82af9c`, `ba27fa99`.
No-notch baseline jobs: `9c2b253b`, `b1d3295a`, `adfde124`, `50696749`.
Reports:

- `scratchpad/v6eval/current_truepeak_guard_n4_aggregate.json`
- `scratchpad/v6eval/current_truepeak_guard_highreg_n4.json`
- `scratchpad/v6eval/current_truepeak_guard_n4_scores.json`
- `scratchpad/v6eval/notch/cut24/full_draw1_matched_scores.json`
- `scratchpad/v6eval/notch/f0lo400/full_draw1_scores.json`

**Status: the notch is shipped, but the user's no-residual / no-artifact
goal is not met and no human has listened.** AV-4aqs remains open.
