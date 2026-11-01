# The inharmonic notch: an output-side fix for the high-register timbre defect (2026-10-31)

**This is the lever that closes the defect `docs/av9hc5-level-vs-timbre.md`
proved was spectral.** Measured on all four served draws (n=4) of the
original Hero track, on the FULL song.

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
8-12 kHz -22.5 → -21.1 dB, 12-16 kHz -23.2 → -21.4 dB, identity vector
unchanged at 1.000.

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

This is why the real-data result is 11.27 % and not 0 %. The stage removes
the residue; it cannot put back the harmonics.

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

## What it does NOT fix

The lever removes the RESIDUE - energy sitting between the harmonics. It
cannot put back the harmonics themselves. The 11 % that remains is the
decoder's high register still not matching the source's harmonic
structure, and a notch can only remove, never synthesise.

**And nobody has listened to it.** Every number here is a measurement. The
user judges by ear, and the ear is the only thing that can say whether the
remaining residue is audible as the "double ringing" they complained
about.

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

## End-to-end proof through the live service

**n=4 both ways.** Four independent end-to-end conversions of the same
original Hero track through the live service, with the stage and without,
scored on their paired vocal stems. Same profile, same preset; the only
variable is whether the stage ran.

| band | inharmonic excess WITHOUT (n=4) | WITH (n=4) |
|---|---|---|
| 250-400 Hz | 2.44 % [2.02, 3.08] | **1.84 % [1.62, 2.14]** |
| 400-550 Hz | 31.09 % [30.31, 31.79] | **30.36 % [29.32, 31.35]** |
| 550-700 Hz | 61.38 % [58.19, 65.06] | **−0.80 % [−3.29, 0.65]** |
| **700-1100 Hz** | **40.69 % [37.14, 45.54]** | **11.17 % [9.62, 13.45]** |

| band | level dropout WITHOUT | WITH |
|---|---|---|
| 250-400 Hz | 0.32 % | 0.33 % [0.22, 0.48] |
| 400-550 Hz | 0.39 % | 0.39 % [0.00, 0.79] |
| 700-1100 Hz | 4.36 % | **1.83 % [0.94, 2.36]** |

Spectral centroid 966 → 1086 Hz, 8-12 kHz -22.5 → -21.6 dB, identity
1.000 both ways.

Per draw, inharmonic excess at 700-1100 Hz:
**without** 40.73 / 45.54 / 39.34 / 37.14 →
**with** 9.62 / 11.88 / 13.45 / 9.73.

**Every draw improves, and the two distributions do not overlap**: the
worst draw with the stage (13.45 %) is 23.7 points below the best draw
without it (37.14 %). That is a 72.5 % reduction in the defect.

The 550-700 Hz band reads −0.80 % with the stage, i.e. marginally cleaner
than the source. That band has only 30 usable source frames, so
`score_multi` itself warns it is anecdote, not measurement; read it as
"the residue there is gone", not as a precise number. The band with
enough frames to measure is 700-1100 Hz, 424 frames.

Jobs without the stage: `9c2b253b`, `b1d3295a`, `adfde124`, `50696749`.
Jobs with it: `fb8e5c1e`, `be420860`, `9403b4a5`, `3c58e282`. The service
log for each reads:

    fork inharmonic notch: 2.4 dB mean cut on 6% of scored frames
    (3 passes, f0 >= 550 Hz); make-up gain on 2% of frames, max 12.0 dB

and the stage runs BEFORE the dropout fill, as intended.

## Files

- `src/auto_voice/inference/fork_inharmonic_notch.py` - the shipped stage
- `scratchpad/v6eval/inharm_notch2.py` - the offline lever it was
  developed in (metric-exact f0, exact correction, iterated, true-peak
  limited)
- `scratchpad/v6eval/notch/full/scores_lever3.json` - the n=4 full-song
  report for the shipped settings
- `scratchpad/v6eval/notch/full/lever3_draw{1..4}.wav` - the levered stems
- `scratchpad/v6eval/notch/exact/sweep.json` - the parameter sweep
- `scratchpad/v6eval/pipeline_equiv/scores.json` - the offline-vs-production
  equivalence report
- `scratchpad/v6eval/e2e_notch_scores.json` - the end-to-end report
- `scratchpad/v6eval/served_n4_notch/scores.json` - the n=4 served
  with-vs-without report
- `scratchpad/v6eval/score_served_n4_notch.py` - the n=4 scorer
- `scratchpad/v6eval/full_song_check.py` - the full-song runner

## Status

**Shipped and measured at n=4, but NOT LISTENED TO.** The stage is in the
served pipeline. The registry still serves G_389 unchanged - the model is
the same, only the output chain changed.

Nobody has listened to a render with the stage in it. Every number here is
a measurement. The 11 % of residual inharmonic excess it leaves is still
there, and only an ear can say whether the remaining residue is audible as
the "double ringing" the user complained about.

The listening candidates are the served mixes:
`data/conversions/fb8e5c1e-4bf9-4d93-bd9b-51651949fabd/mix.wav` (with the
stage) against `data/conversions/9c2b253b-3c71-4859-9044-9678d990daaf/mix.wav`
(without).
