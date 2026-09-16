# Duet keying of the output-correction chain — measured, 2026-09-15

**Verdict: keying the chain per converted track is not a precaution, it is a bug fix.
It removes MORE of the defect AND does zero damage to the other singer.**
Scripts: `scratchpad/duetkey/exp{1,2,3}_*.py` (re-runnable, no GPU).

## What the code does today

`_convert_multi_speaker` builds `combined = converted_primary + backing` and returns it.
`_convert_song_fork_hq` then runs all three correction stages on that sum, keyed on
`hum_key = voc_mono` — the whole duet vocal. Two things are wrong with that:

1. **The pitch key describes neither singer.** On a synthetic duet of a 262 Hz female
   and a 165 Hz male, `_source_f0` (yin) returns a median of **87 Hz** — not 262, not
   165. Two simultaneous pitches drive yin to a common sub-period. Every notch
   `suppress_subharmonics` places is then computed from a pitch nobody sang.
2. **The preserved co-vocal is corrected as if it were decoder output.** It is the
   ORIGINAL singer's audio; it has no decoder defects to fix. It can only be damaged.

## Measured, faithful reproduction (exp2)

Female lead 262 Hz carrying injected half-integer fuzz (the defect) + clean male
co-vocal at 165 Hz preserved verbatim. Change vs the untouched mix:

| | lead fuzz @393 | lead fuzz @655 | male f0 165 | male h2 330 | male h3 495 | male h4 660 |
|---|---|---|---|---|---|---|
| **today** (keyed on duet) | −17.4 | −7.4 | −0.0 | −0.1 | −1.5 | **−7.4** |
| **per-track** (keyed on lead) | **−20.0** | +0.1 | −0.0 | −0.0 | −0.0 | **+0.1** |

Worst damage to the preserved male voice: **−7.4 dB today, 0.0 dB per-track.**
Per-track also removes 2.6 dB more of the lead's actual defect, because its pitch
key is right (262 Hz, not 87 Hz).

## How much of a real song is affected (exp3)

Zendaya & Max Schneider "All of Me" (`b81cc29e`), from the exported stems:
lead active 75.5 % of frames, co-vocal active 10.9 %, **both at once 10.3 %** —
i.e. **95 % of the co-vocal's singing overlaps the lead.** Simultaneity is the
normal case for a duet, not an edge case. (Partly structural: `backing` also
carries `simul_backing` from the karaoke split, which overlaps by construction.)

## The design

Run the chain **per converted track, keyed on that track's own source, before mixing.**

```
converted_primary = infer(primary_track);   corrected_primary = correct(primary_track,   converted_primary)
converted_second  = infer(secondary_track); corrected_second  = correct(secondary_track, converted_second)
preserved_backing -> NOT corrected (original audio, no decoder defects)
combined = corrected_primary + corrected_second + preserved_backing
```

`build_speaker_track(voc, sr, spans)` already returns exactly the right source for a
track: full length, zero outside the speaker's spans, 50 ms linear fades at true
silence edges, overlapping spans unioned. Use it for the key — do not hand-roll masking.

**Each stage behaves correctly on a span-masked source; verified against the code:**
- `suppress_hum` — `_source_gate` sees digital silence outside the spans and closes to
  its −28 dB floor there. For that track's own converted output that is correct and
  desirable (the model should be emitting nothing there). It never hard-zeros.
- `suppress_subharmonics` — `_source_f0` gates f0 to NaN below −60 dBFS absolute and
  40 dB under peak, so zeroed regions produce no notches. (This gate exists because
  `librosa.yin` returns a confident ~1000 Hz on silence; see
  `tests/test_fork_subharmonics.py::test_silent_source_is_a_no_op`.)
- `polish_voice` — its LTAS already uses loud frames only (>−30 dB), so zeroed regions
  are excluded. Per-track is strictly better than today, where the second singer
  pollutes the reference spectrum the correction is computed from.

## Residual risk, stated honestly

Within the 10.3 % of frames where both sing, the separated lead stem still contains
co-vocal bleed, so the lead's own pitch key is imperfect there. Per-track keying
reduces this from "the key describes neither singer" to "the key describes the lead
with some bleed" — a large improvement, not a guarantee. If it ever matters, the next
lever is excluding overlap frames from the f0 key (they are already computable:
both-active frames of the two span tracks).

## Regression test to add

`exp2_realistic.py` is the test, minus the printing: assert that a preserved co-vocal
summed with a corrected lead loses < 1 dB at its own harmonics, and that the same
signal keyed on the mixed duet loses > 5 dB. That pins the bug so it cannot come back.
