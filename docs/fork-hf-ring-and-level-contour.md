# The G_250 Hero defects: "double ringing/buzzing", dropout, unstable volume

User report (2026-09-16), on the G_250 Hero render `79bc2143`:
> "hero is 50% better but not perfect yet. her converted voice has a kind of
> like a double runging sound to the converted voice, or it could be described
> as a double buzzing around the voice. If we can fix that output we might be
> close to perfect for hero. voice still drops out a little and voice output
> volume isnt constant"

## Verdict: three complaints, TWO defects. Neither is the model.

---

## 1. "Double ringing / double buzzing AROUND the voice" = HF decoder resonance

**ContentVec runs at 16 kHz -> Nyquist 8 kHz. Above 8 kHz the decoder has no
content input at all**; it invents the band from the speaker embedding and the
upsampling stack, and what it invents is narrowband and tonal. "Around the
voice" is the accurate part of the report: it is a metallic halo above her,
not the voice itself.

Tonal persistence (share of loud frames where a bin stands >6 dB above its own
local spectral floor), 8-10.5 kHz:

| signal | tonal % |
|---|---|
| source stem | 8.6 |
| converted G_250 | 10.1 |
| converted G_480 (old served) | 10.8 |
| **after the new stage** | **8.7** |

**Not new to G_250** (G_480 is worse). It is the user's ORIGINAL complaint from
the start of this workstream - *"a small high band resonence or someting"* -
now audible because the belt fuzz that masked it was fixed.

### Why the band mean understates it, and the selection-bias trap
The excess concentrates in a few drifting bins near **8.9-9.4 kHz** reaching
**+44..+52 points**, i.e. **5.0 sd** of a per-bin difference whose sd is 8.9.
Ranking bins by (converted - source) and quoting the top 14 manufactures a huge
effect out of noise - I did exactly that once mid-investigation and had to
retract it. The honest test is reproducibility across independent renders:
**r = 0.84**, 13/20 hottest bins shared. It is a real fixed resonance.

### Why `fork_voice_polish` cannot catch it (do not retune it)
That notcher reads the **LTAS - a whole-file average**. These tones drift, so
they smear into a broad bump: LTAS excess at 8.6-9.6 kHz is **3.63 dB** against
a 3.0 dB threshold => a **0.6 dB** notch, while per frame the same bins are
6 dB peaks in 55% of frames. **A stationary-tone notcher structurally cannot
see a moving tone.** Hence a separate per-frame stage.

**Fix**: `fork_voice_polish.detonalize_hf()`, chain stage 4, flag
`fork_hq_hf_detonalize`. Per-frame, compares each frame with its own local
spectral floor so the cut follows the drift. Collateral **-0.000 dB** below
7.5 kHz (masked, so exact), +0.00 dB broadband, silence stays silent.

---

## 2. "Volume isnt constant" AND "drops out" = ONE defect, the level contour

`_match_lead_level` applies **one scalar for the whole song**. It fixes the
average and can do nothing about a contour that drifts.

Converted-minus-source level error on loud frames:

| | sd | p10..p90 |
|---|---|---|
| before | 3.77 dB | -5.5 .. +3.2 dB (8.7 dB span) |
| **after** | **2.34 dB** | **-2.6 .. +2.3 dB (4.9 dB span)** |

**The dropout is the same defect.** The instrumental on this render averages
**0.5 dB ABOVE** the vocal, so the lead only has to sag slightly to sink under
the band. There are no silences to fill: **exact-zero runs >20 ms = 0**.

**Fix**: `_follow_lead_contour()`, flag `fork_hq_follow_lead_contour`. Slow
(0.75 s) and bounded (+-6 dB) - it corrects conversion drift, not phrasing; a
fast follower would be a compressor. Gated at -42 dB on the source so gaps are
never lifted and the hum stage's work survives.

---

## Eliminated - do not re-investigate

- **Original-singer bleed** (the literal "two voices" reading): instrumental
  tracks the vocal's pitch in **3%** of frames = noise floor. Dead.
- **Chunk overlap-add doubling**: the fork **butt-joins** (`np.concatenate`),
  it does not overlap-add. It cannot double. Dead.
- **Halo / reverb doubler**: no such stage exists; only a comment references one.
- **The correction chain adding artifacts**: level jitter 0.59/0.59 dB and
  spectral ripple 1.60/1.58 dB, chain on vs off.
- **Phrase tails zeroed to digital silence** (the `np.zeros_like` non-speech
  path): real, but `db_thresh=-100` prevents it here. Zero runs >20 ms = 0.

## Live but unfixed

### `SVCFORK_CHUNK_FADE`: dead code, and LEAVE IT DEAD

The patch IS applied to the installed fork (core.py:310) but the env var is set
NOWHERE in `src/` or `scripts/`, so the fade never runs. Chunks are butt-joined.
Measured level jump at 30 s seams **6.24 dB vs 3.03 dB elsewhere**.

**Do not simply set the variable.** The patch fades each chunk in at its start
AND out at its end, then `np.concatenate`s them - there is no overlap. At a
**speech->speech** seam chunk A fades out to zero and chunk B fades in from
zero, so the join becomes a V-shaped **dip to digital silence**. The fade is
only correct where the neighbour is an already-zeroed non-speech chunk.
`max_chunk_seconds=40` forces mid-phrase splits in a 258 s song, so
speech->speech seams certainly exist: enabling this would **manufacture the
"voice drops out" symptom** we just removed.

The real fix, if seams are ever audible, is a true crossfade (overlap the
inference by `fade_len` and sum), not this fade. The contour follower already
absorbs most of the level step, so this is not currently costing anything.
### RESOLVED 2026-09-16: do NOT retune the subharmonic gate

Measured, octave-clean, 5957 frames, PEAK ratio (the discriminating one):

| | p50 | p75 | p90 | p95 |
|---|---|---|---|---|
| source | 0.007 | 0.016 | 0.053 | 0.088 |
| converted | 0.018 | 0.032 | 0.060 | 0.092 |

**AUC(converted > source) = 0.757.** Better than chance, nowhere near
separable. The best threshold over the whole sweep (0.012) catches 70 % of
converted frames and **30 % of CLEAN SOURCE frames** (Youden J = 0.40).

The decisive number: **median excess 8.9 dB, p90 excess 1.1 dB.** The excess
lives at the MEDIAN, not in the tail - the converted voice's worst frames are
no worse than the source's worst frames. A gate cuts hardest where the ratio is
highest, which is exactly where there is nothing wrong. **The gate is
structurally aimed at the wrong frames**, and the documented per-frame
peak-ratio upgrade does not rescue that - it was the best available
discriminator and it still only reaches AUC 0.757.

This explains the module's own history: the 0.04/0.25 thresholds cut below the
source at 450-600 Hz not because they were mistuned but because **no threshold
on this statistic has the separating power**. Shipping a retune would remove
real voice at 30 % of the rate it removes the defect, for a component already
~35 dB down. Left alone deliberately.

# ponytail: if this ever has to move, it needs a different STATISTIC (one that
# separates), not a different threshold. A proportional always-on reduction
# scaled by the median excess is the only shape that fits a median shift.

- ~~**Residual period doubling**~~, 150-450 Hz (superseded by the above):
  converted is **~3x** the source (0.018-0.035 vs 0.006-0.009). The suppressor
  **idles** here - its gate needs an internal ratio >0.15 and sees 0.030-0.066,
  so it applies **0 dB** across 150-600 Hz. It only fires at 600-900 Hz, where
  Hero has **19 valid frames**. Thresholds were calibrated on G_480, whose
  ratio was 0.26. The documented upgrade (per-frame PEAK ratio, which separates
  the bands where the internal ratio compresses them) is the fix. Floor any
  retune on the SOURCE column: the earlier 0.04/0.25 thresholds cut BELOW the
  source at 450-600 Hz, i.e. removed real voice.

## Measurement notes

- **`yin`/`pyin` octave-halve badly here: 40% of voiced frames.** Reject any
  frame where `peak(2*f0) > peak(f0)` before computing harmonic ratios. An
  earlier pass that skipped this reported 600-900 Hz as the worst band; with
  octave errors removed that band has 19 frames and the real content is
  150-450 Hz. It inverted the conclusion.
- **Never compare two renders sample-wise.** `noise_scale=0.4` makes each
  render stochastic; `rms(a-b)/rms(a)` came out **+1.9 dB** (near-decorrelated)
  between two renders of the same audio. Statistical comparisons are fine;
  to isolate a stage, toggle it on ONE render.
