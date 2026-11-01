# AV-9hc5: is the high-register collapse level or timbre? (2026-10-31)

**Answer: timbre.** The level is already handled; the residue is not.

## The probe

Two output-side levers, applied offline to the real served stem
(job `9c2b253b`, the fresh conversion of the original Hero track), then
scored with `experiments/score_multi.py` against
`scratchpad/dropout/hero_src.wav`:

- **A. Pitch-tracked make-up gain.** Restore the source's 550-4000 Hz level
  per frame, keyed on the SOURCE's f0, bounded at 12 dB, applied in the STFT
  domain so the lift is band-limited. Only frames where the source is above
  550 Hz and the render is quieter get lifted: **6.4 % of frames**, max 12 dB.
- **B. Keep-source-spans above 550 Hz.** Splice the SOURCE's own audio back
  in wherever the source sings above 550 Hz (25 ms crossfades): **10.1 % of
  frames** are high-pitched in the source.

Source band frames: 250-400 Hz 4554, 400-550 Hz 509, 550-700 Hz 30,
700-1100 Hz 424.

## The measurement

| | served | A: make-up gain | B: keep-source |
|---|---|---|---|
| dropout 700-1100 Hz | 3.77 | **0.47** | **0.00** |
| inharmonic excess 700-1100 Hz | **40.73** | 40.92 | **1.90** |
| inharmonic excess 550-700 Hz | 63.69 | 63.82 | **0.00** |
| inharmonic excess 250-400 Hz | 3.1 | 3.1 | 3.1 |
| identity (mel-stat centroid) | 1.0000 | 1.0000 | 0.9995 |
| centroid Hz | 975 | 1033 | 885 |

(n=1 per arm — the served stem is one draw. The levers are deterministic
given that stem, so the COMPARISON between arms is exact even though each
arm is a single draw.)

## What it means

**A, the make-up gain, does not touch the defect.** It removes the residual
level dropout (3.77 % -> 0.47 %) and the inharmonic excess is UNCHANGED
(40.73 -> 40.92). More level in a band that contains narrowband junk gives
more junk. So the collapse is NOT a level problem at this point — the chain
had already fixed that, and what remains is spectral.

**B, keeping the source's own audio, removes the defect completely.**
Inharmonic excess at 700-1100 Hz goes 40.73 % -> **1.90 %**, and the 550-700 Hz
band (where the source has only 30 frames, so read it as anecdote) goes
63.69 % -> 0.00 %. Dropout goes to 0.00 %.

And the identity cost is **1.0000 -> 0.9995** — the mel-stat centroid barely
moves, because the mel-stat centroid is dominated by the low/mid bands where
the converted voice is unchanged.

## The catch, stated plainly

B splices **the original singer's** audio into the output. That is not a
conversion — it is the source leaking through wherever she sings high. The
identity metric does not see it (0.9995 is a centroid, not a timbre
fingerprint), and the inharmonic metric certainly does not: real harmonics
from the wrong singer score as "clean".

**So B proves the defect is in the MODEL's timbre at high pitch, and it
proves the fix is not an output-side filter.** It does not prove B is
shippable. Whether a 10 % splice of the original singer is acceptable to
the user's ear is a listening question, not a metric question — and nobody
has listened to any of this.

What B *does* establish, and this is the load-bearing part:

> There is no gain, filter or chain stage that can make the decoder render
> her high register cleanly. The decoder emits narrowband residue there
> because Brandy's corpus has essentially nothing above 700 Hz
> (0.00 % of frames above 700 Hz, p99 496 Hz, max 844 Hz), so the NSF
> excitation and the upsample stack are extrapolating outside the
> distribution they were fitted on. Only more real high-register material
> — or a different architecture — changes that.

That is the architectural question AV-41e4 was re-scoped to, and this probe
is direct evidence for it rather than an inference from failed
training-data levers.

## The practical option B does open

If the user finds a 10 % source splice acceptable (or if the source is
herself, so the splice is a no-op in timbre), B is a working output-side
lever with a measured effect. It needs:

- a threshold sweep (550 Hz is a first guess; the source's p99 is 496 Hz and
  its max 844 Hz, so the useful band is narrow),
- an n=4 served measurement behind it, and
- **a listening pass**, because the metric cannot see a foreign voice.

If the user will not accept a source splice, then the honest answer is that
the remaining defect needs either real high-register recordings of Brandy
(the AV-41e4 data route, which failed twice) or an architecture change
(16 kHz content encoder, or an NSF excitation fitted above 700 Hz).

## Files

- `scratchpad/v6eval/av9hc5_probe.py` — the probe
- `scratchpad/v6eval/av9hc5/{served,makeup_gain,keep_source_high}.wav`
- `scratchpad/v6eval/av9hc5_scores.json` — the scoring report
