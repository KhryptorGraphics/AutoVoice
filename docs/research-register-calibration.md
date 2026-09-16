# Absolute register classification from median pyin f0 — calibration

**Date:** 2026-09-15 · **Scope:** measurement only. Nothing under `src/`, `tests/`,
`frontend/` touched, no service restart, no conversion jobs submitted.
**Scripts:** `scratchpad/registercal/{measure,stage2,stage3,stage4,stage5,stage6}.py`
**Data:** `results.json` (fmin=65), `stage2.json`, `stage3.json`,
`stage4.json` (fmin=110, canonical), `stage5.json` (centroid replication),
`stage6.json` (ear-marked ground truth)

---

## VERDICT

**Not reliable to route on as specified, and the 165-175 Hz band is wrong by
about 60-90 Hz.** But the failure is more specific than "f0 cannot do it", and
the distinction changes what to build:

1. **Per diarization cluster — no.** The clusters are not singer-pure. On Beauty
   and the Beast, against hand-marked ground truth, both forced-2 clusters are
   roughly 40% male. The cluster medians read 320.2 and 287.0 Hz when the true
   singer medians are 335.4 and 221.3. A 33 Hz "separation" that should be 114 Hz
   is noise, not signal.
2. **Over true per-singer spans — the statistic does carry information.** Given
   the operator's ear-marked spans, Beauty separates by **114.1 Hz**.
3. **But no fixed threshold near 165-175 Hz works, at any granularity.** Every
   singer measured sits above it. Male medians are 221-232 Hz. At 175 Hz, **36 of
   40** male training samples classify as female.
4. **A retuned threshold near 260 Hz separates the five singers whose spans could
   be established**, with a 58.6 Hz whole-singer margin. That is five singers, and
   per-span variance is large enough (25% of Leroy's voiced frames exceed 329.6 Hz)
   that single-span calls will still misfire. Treat it as a hypothesis, not a
   calibrated threshold.

Prerequisite for any of this: singer-pure spans, which the diarizer does not
currently produce on either duet.

---

## 1. Beauty and the Beast — ear-marked ground truth

Job `d85f96e6` notes record the operator's by-ear male spans:
`male spans 39-77/116-135/193-232s kept original`. That is ground truth
independent of both diarization and f0. Measured directly on the separated stem
at fmin=110:

| singer | span source | seconds | median | p25 | p75 | voiced frac | sex |
|---|---|---|---|---|---|---|---|
| Leroy Sanchez | ear-marked male ranges | 96.0 | **221.3** | 111.3 | 329.6 | 0.595 | M |
| Lorea Turner (+ duet sections) | complement | 136.4 | **335.4** | 258.6 | 401.2 | 0.741 | F |

**Margin: 114.1 Hz, correct direction.** The statistic works here.

The quartiles still overlap heavily: Leroy's p75 (329.6) is above Lorea's p25
(258.6), so roughly a quarter of his voiced frames sit inside her range. The
114 Hz is a whole-singer margin, not a per-span one.

### The diarizer destroys it

Overlap of the forced-2 clusters with the ear-marked male ranges:

| cluster | seconds | fraction inside male ranges |
|---|---|---|
| SPEAKER_00 | 191.5 | 39.7% |
| SPEAKER_01 | 55.4 | 43.3% |

Both clusters are roughly a 40/60 mix of the two singers. Measured per-cluster
medians are therefore 320.2 and 287.0 Hz — a 33.2 Hz gap where the truth is
114.1 Hz. Default (unforced) diarization was worse: 3 clusters, one holding 191.5 s
of a 232.4 s stem, both large clusters reporting the same 293.5 Hz median.

### The centroid gate does not rescue the assignment here

Replicating `_mix_centroid` verbatim (`singing_conversion_pipeline.py` ~line 677:
median spectral centroid of the **original mix** over the cluster's spans, loud
frames only, first 25 s) and applying the shipped ratio of 2.0:

| duet | primary centroid | other cluster | ratio | gate fires |
|---|---|---|---|---|
| All of Me | 3130.5 Hz | 573.9 Hz | 5.46x darker | yes |
| Beauty | 2495.9 Hz | 938.6 Hz | 2.66x darker | yes |

On Beauty the gate fires on SPEAKER_01, which ground truth shows is **57%
female**. So it fires, but not on the male singer. That is consistent with the
docstring's recorded collapse on this song and with the fact that a *mix*
centroid is dominated by instrumentation, not voice: 938 Hz is a sparse-backing
passage, not a male register. **Nothing here is grounds to change the calibrated
gate or its threshold** — it is reported only because it does not settle the
assignment and cannot be used as an independent register signal on this song.

---

## 2. All of Me — unresolved

No ear-marked spans exist for this song (jobs `a9a6f84d` and `b81cc29e` carry no
notes), so no per-singer ground truth could be established.

| cluster | seconds | median | p25 | p75 | voiced frac |
|---|---|---|---|---|---|
| SPEAKER_00 | 208.0 | 254.2 | 212.5 | 278.8 | 0.819 |
| SPEAKER_01 | 43.7 | 260.1 | 233.1 | 277.2 | 0.413 |

The 43.7 s cluster is almost certainly production's preserved cluster: job
`a9a6f84d` recorded `preserved_s: 43.7`, matching to the tenth of a second. Its
centroid is 5.46x darker than the primary's, well past the 2.0 gate.

That still does not license attaching a sex to either row. Given Beauty's
clusters turned out ~40% impure, and given `b81cc29e` had to reassign 8 further
spans out of this primary cluster by embedding, both rows here are mixtures of
unknown proportion. The 5.9 Hz between them is indistinguishable either way.

---

## 3. Trained target voices

Twenty samples per profile, `data/samples/<pid>/sample_*/vocals.wav` (already
separated stems), first 30 s of each, pooled voiced frames.

| singer | profile | sex | median | p25 | p75 | voiced frac |
|---|---|---|---|---|---|---|
| Brandy | `fb17af66` | F | 290.3 | 264.7 | 329.6 | 0.636 |
| Conor Maynard | `f16189d9` | M | 231.7 | 193.7 | 277.2 | 0.807 |
| William Singe | `c8232ba1` | M | 229.1 | 182.9 | 277.2 | 0.607 |

Pooled-median margin female vs nearest male: 58.6 Hz. That averages away the
variance a live decision faces. Per sample:

| | per-sample medians (Hz) |
|---|---|
| Brandy (F) | 274, 277, 279, 282, 284, 286, 287, 287, 287, 287, 288, 289, 290, 292, 294, 295, 297, 306, 308, 315 |
| Conor (M) | 137, 176, 187, 197, 198, 208, 209, 210, 211, 233, 233, 236, 236, 237, 244, 263, **292**, **295**, **341**, **392** |
| William (M) | 128, 140, 147, 179, 186, 188, 201, 203, 209, 213, 223, 244, 253, 253, 254, 265, 265, 271, **317**, **379** |

Six male samples (bold) exceed the lowest female sample (274.0). Worst per-sample
margin: **-118 Hz**, Conor's 392 Hz sample against Brandy's 274 Hz sample.

Threshold sweep over these 60 samples:

| threshold | female wrong | male wrong |
|---|---|---|
| 165 Hz | 0/20 | 36/40 |
| 175 Hz | 0/20 | 36/40 |
| 260 Hz | 0/20 | 10/40 |
| **271 Hz (optimum)** | **0/20** | **6/40** |

### All singers whose spans could be established

| singer | median | sex |
|---|---|---|
| Leroy Sanchez | 221.3 | M |
| William Singe | 229.1 | M |
| Conor Maynard | 231.7 | M |
| Brandy | 290.3 | F |
| Lorea Turner | 335.4 | F |

Clean separation at the whole-singer level, gap 58.6 Hz, any threshold in
232-290 Hz. Five singers is a small sample and the per-span distributions still
overlap, so this is a hypothesis worth testing on more material, not a
calibration.

---

## 4. Method notes and deviations

- **pyin ran at fmin=110, not 65.** At fmin=65 both stems showed a spurious low
  mode (99 Hz All of Me, 110 Hz Beauty). Spectral check on those windows: median
  energy at the claimed f0 was **0.19x** (All of Me) / **0.51x** (Beauty) the
  spectral floor, so there is no partial at the claimed fundamental and pyin had
  halved the octave. 38% / 58% of low-mode windows met a strict halving test. All
  headline numbers use fmin=110; the fmin=65 run is kept in `results.json`.
  **fmin=65 manufactures a fake low mode on this material. Use 110.**
- One pyin pass per stem at 22050 Hz (frame 2048, hop 512, 43 fps); spans index
  into it by frame-centre time rather than concatenating audio, which avoids
  concatenation seams. Same statistic, one pass instead of N.
- Voiced fraction is pyin's `voiced_flag`, never "f0 is not NaN". These voiced
  fractions are **not** comparable to `multi_speaker_merge_voiced_min` (0.65):
  different sample rate, fmin and hop from `_voiced_fraction`.
- Speaker count forced to 2 on both duets after the default gave 3 clusters with
  one holding ~85% of the song. Forced-2 segments overlap slightly (All of Me
  251.7 s of spans on a 244.2 s stem; Beauty 246.9 s on 232.4 s), so a few frames
  count toward both clusters.
- A WavLM-embedding cross-check (KMeans on 2 s windows, f0 not involved) split
  All of Me into groups 66.1 Hz apart, agreeing with the f0 split 85% of the time;
  on Beauty it ran at 0.598 agreement, which is chance.
- Duet job dirs' `vocals.wav` is the **converted** stem
  (`singing_conversion_pipeline.py:2093`), not the source, so it was not used.
  Beauty's job dirs contain only `mix.wav` in any case.

---

## 5. Recommendation

**Do not ship an absolute register gate at 165-175 Hz.** It misclassifies 90% of
male training material and every singer measured here sits above it.

1. **Fix the spans before the statistic.** This is the load-bearing finding. On
   Beauty, true spans give a 114.1 Hz margin and diarizer clusters give 33.2 Hz
   of noise, purely because both clusters are ~40% male. No per-cluster statistic
   — f0, centroid, or embedding — can work on a 40/60 mixture. Any duet register
   routing should be judged on span purity first, and span purity is measurable
   now against job `d85f96e6`'s ear-marked ranges.
2. **The named fallback is centroid relative to the target's register, and this
   measurement does not test it.** What it does show is that the *mix* centroid
   is not a register signal on a dense arrangement: it fired on a 57%-female
   cluster on Beauty. A centroid computed on the separated stem rather than the
   mix would be a different and untested proposition. Relative *f0* is not the
   fallback and would not work: Brandy sits at 290.3 while both duet males and
   Zendaya's cluster sit 30-70 Hz below her, so distance-from-target does not
   distinguish a male co-vocal from a female one.
3. **Brandy's register is a measured fact worth keeping**: median 290.3 Hz,
   p25-p75 264.7-329.6, spread 41 Hz across 20 samples. Tight and reliable.
4. **Operator choice remains the honest fallback for identity.** The docstring's
   2026-07-04 negative result (SV embeddings cannot do identity on sung audio),
   the centroid gate firing on the wrong cluster here, and the diarizer's 40/60
   clusters all point the same way. The per-request `preserve` choice is doing
   real work that automation has not earned the right to replace, and the
   span-range form of it (`39-77` style) is what produced the only trustworthy
   per-singer measurement in this report.

---

## 6. For backend-routing: split, margin, and what to expect

**Interim number if you must ship one: `register_split_f0_hz = 261`,
`register_margin_hz = 29`.** That is the midpoint and half-gap between the
highest male and lowest female singer measured at whole-singer granularity
(20-sample pool each). Do not read it as calibrated.

| quantity | Hz |
|---|---|
| highest male singer, Conor Maynard | 231.7 |
| lowest female singer, Brandy | 290.3 |
| gap | 58.6 |
| midpoint | 261.0 |
| half-gap | 29.3 |

Per-source medians behind it: Leroy Sanchez 221.3 M, William Singe 229.1 M,
Conor Maynard 231.7 M, Brandy 290.3 F, Lorea Turner 335.4 F.

**The dead zone does not rescue it.** A cluster span is closer in length to one
training sample than to a 20-sample pool, so the 60 per-sample medians are the
right granularity to score against. Swept over splits 230-300 and margins 0-30:

| split | margin | correct | wrong | unknown |
|---|---|---|---|---|
| 261 | 0 | 50 | 10 | 0 |
| 261 | 15 | 46 | 6 | 8 |
| 261 | 29 | 28 | **6** | 26 |
| 271 | 0 | 54 | 6 | 0 |
| 295 | 30 | 33 | **3** | 24 |

**No (split, margin) pair in that range reaches zero errors.** Widening the
margin from 0 to 29 Hz at split 261 converts correct answers into unknowns
without removing a single error, because the classes interleave rather than
merely sit close: Conor has samples at 292, 295, 341 and 392 Hz while Brandy's
lowest is 274 Hz. The best cell found abstains on 40% of inputs and is still
wrong on 5%.

**Use `fmin=110`, not 80.** At 80 the artefact from section 4 still shows:
William Singe's lowest two sample medians read 86.2 and 102.6 Hz, which become
127.5 and 140.2 at fmin=110. It barely moves the threshold, so it is not a
blocker, but it costs nothing to remove a known octave artefact.

**The larger risk is upstream of your constant.** On Beauty and the Beast the
diarizer's clusters are roughly 40% male each, so your classifier will read
287.0 and 320.2 Hz where the true singer medians are 221.3 and 335.4. Both of
those land above any split recommended here, so both clusters classify female
and the male singer is routed to the female model. Gate the feature on span
purity, not on the f0 constant.

---

## 7. Span purity is the binding constraint on duet routing

This holds **independent of which statistic routes**, and it survives the move to
operator-elimination routing. Elimination removes the classifier. It does not
change where the spans come from, and the spans are what is wrong.

**Ground truth, reusable.** Job `d85f96e6`'s notes record the operator's by-ear
male spans for Beauty and the Beast:

```
39-77 / 116-135 / 193-232 s
```

96.0 s of male against a 232.4 s stem. Any future change to diarization, the
centroid gate, or the WavLM pass can be scored against these three ranges in
seconds. No listening required. That is the cheapest regression check available
for this work, and it is the only per-singer ground truth in this report.

**What the current span machinery does with it.**

| measurement | value |
|---|---|
| SPEAKER_00 (191.5 s) inside male ranges | 39.7% |
| SPEAKER_01 (55.4 s) inside male ranges | 43.3% |
| centroid gate fires on | SPEAKER_01, which is 57% female |
| true singer medians | 221.3 M / 335.4 F |
| medians the clusters report | 287.0 / 320.2 |

Both clusters are roughly a 40/60 mixture, and the gate preserves the one that
is majority female. Under elimination routing that means **majority-female audio
is handed to the model assigned to the male slot**, with no statistic anywhere in
the path that could catch it.

**Why no automated purity check is offered here.** The obvious candidate, a
per-cluster f0 spread test, does not survive contact with the data. The impure
Beauty clusters have interquartile ranges of about 130 and 175 Hz, while the
correctly-carved male span has an IQR of 218 Hz. A pure single-singer span is
*wider* than the impure clusters, so spread does not separate pure from mixed.
Inventing a threshold here would repeat exactly the error section 1 documents.

**What to do instead.** The span-range preserve form (`39-77` style) bypasses
clustering entirely and produced the only trustworthy measurement in this
report. Where an operator has already identified the co-vocal's timing, prefer
their ranges over cluster-derived spans. Where they have not, the cluster
boundaries and the centroid values are already logged by `_select_speaker_spans`,
so a bad split is at least visible after a render rather than silent.
