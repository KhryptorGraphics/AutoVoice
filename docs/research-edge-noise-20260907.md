# Edge noise & belt-register artifacts: research findings (2026-09-07)

Read-only research ticket. Scope: why the so-vits-svc-fork decoder distorts around the
edges of the vocal and fuzzes on belted notes, what is already measured, what the code
actually does, and what to try next. Every claim carries a `file:line` or URL. Numbers in
**bold** are quoted exactly from their source. Offline measurements in §1.4/§2.2 were run
with `PYTHONNOUSERSITE=1 PYTHONPATH=src /home/kp/anaconda3/envs/autovoice-thor/bin/python`
using librosa/numpy only on existing files (no GPU inference, no re-renders).

---

## 1. Evidence baseline — what is already measured and shipped

### 1.1 The shipped mix-stage chain (order and keys)

The fork HQ chain runs in `singing_conversion_pipeline.py` in this order: hum suppression
→ subharmonic suppression → voice polish → halo
(`src/auto_voice/inference/singing_conversion_pipeline.py:1619-1650`), each gated on config
keys `fork_hq_hum_suppress` / `fork_hq_subharmonic_suppress` / `fork_hq_voice_polish` (all
default on). The halo is applied last with dryness-scaled wet
(`singing_conversion_pipeline.py:1730-1802`; log line `Fork HQ halo adaption: source dryness %.1f dB -> wet scale %.2f` at `:1767`).

The engine itself is invoked per registered profile through the subprocess bridge
(`src/auto_voice/inference/svc_fork_bridge.py:141-160`): `-na` (no auto-predict f0) is
load-bearing (`docs/served-models.md:714-715`), `noise_scale` defaults to `0.4` in the
bridge call (`svc_fork_bridge.py:157`), and chunking keys (`chunk_seconds`,
`max_chunk_seconds`, `pad_seconds`, `db_thresh`) are per-model registry entries
(`svc_fork_bridge.py:171-190`). Live registry values: `chunk_seconds: 30.0,
max_chunk_seconds: 40.0, pad_seconds: 1.0, db_thresh: -40` (`docs/served-models.md:36-37`).

### 1.2 Belting rasp: period doubling (`fork_subharmonics`) — `docs/served-models.md:766`

User report: *"her voice fuzzes out when she puts any umph or effort in belting out
notes."* (`docs/served-models.md:768-769`). The measured table, quoted exactly
(`docs/served-models.md:771-776`):

| f0 band | source (Mariah) | converted | corrected |
|---|---|---|---|
| 250–350 Hz | 0.013 | 0.063 | 0.033 |
| 350–450 Hz | 0.015 | 0.060 | 0.037 |
| 450–600 Hz | 0.065 | 0.206 | 0.079 |
| 600–900 Hz | **0.144** | **0.515** | **0.172** |

"The decoder emits 3–5× the source's period doubling at every pitch and it explodes in the
belt range, because above ~450 Hz it is extrapolating past Brandy's training corpus"
(`docs/served-models.md:783-786`). Correction is subtractive: notches on the half-integer
partials placed on the **source's** f0, 15 dB deep, ±16 % wide, depth scaled per frame
(`docs/served-models.md:788-789`; implementation `fork_subharmonics.py:97-168`).

Key parameter facts (`docs/served-models.md:796-807`):
- Depth is cheaper than width (15 dB/±8 % beats 12 dB/±12 %).
- First gate thresholds (0.04/0.25) over-cut 450–600 Hz to 0.050, *below* the source's
  0.065 — removing real voice; 0.15/0.40 lands it at 0.079.
- Collateral on a clean voice: −24.9 dB residual.
- "The per-frame gate is honest but not sharp — it acts on **68 %** of converted frames and
  **39 %** of clean ones; the peak-based ratio in the table is the better detector and is
  the upgrade path."
- The polish made the rasp **22 %** worse (0.511 → 0.622) when run first — hence the stage
  order (`docs/served-models.md:793-796`; same rationale in `fork_subharmonics.py:51-55`).

### 1.3 End-of-chain subharmonics on the delivered renders — `docs/served-models.md:815`

Quoted exactly (`docs/served-models.md:820-825`):

| f0 band | Hero source | Hero live | AOM source | AOM live |
|---|---|---|---|---|
| 250–350 Hz | 0.013 | 0.038 | 0.006 | 0.041 |
| 350–450 Hz | 0.015 | 0.040 | 0.009 | 0.031 |
| 450–600 Hz | 0.063 | 0.101 | 0.008 | 0.027 |
| 600–900 Hz | **0.147** | **0.281** | **1.538** | **0.842** |

Hero job `c23927f0`, All of Me job `39ec9b96` (`docs/served-models.md:826`). "All of Me's
source belts so hard its own 600–900 ratio is 1.538; the render halves it to 0.842, which is
the most a subtractive stage can do against a source that already carries the rasp"
(`docs/served-models.md:828-830`).

### 1.4 My offline reproduction of these numbers (this session, librosa/numpy only)

Mirroring `scratchpad/rerender/measure_live.py:12-23` (the exact peak-ratio metric that
produced the docs tables) on the files listed in the ticket:

| file | 250–350 | 350–450 | 450–600 | 600–900 |
|---|---|---|---|---|
| Hero source (`scratchpad/rerender/src_vocal_full.wav`) | **0.013** | **0.015** | **0.063** | **0.147** |
| Hero live (`data/conversions/c23927f0-…/vocals.wav`) | 0.036 | 0.041 | 0.097 | 0.193 |
| AOM source (`scratchpad/rerender/aom_src_vocal.wav`) | 0.006 | 0.009 | 0.008 | **1.538** |
| AOM raw fork (`scratchpad/rerender/aom_raw_fork.wav`) | 0.038 | 0.035 | 0.033 | 0.779 |
| AOM delivered (`data/conversions/464badee-…/vocals.wav`) | 0.033 | 0.028 | 0.029 | 0.533 |

- **Source rows reproduce the doc's table exactly** (0.013/0.015/0.063/0.147 and 1.538), so
  the metric is implemented identically.
- The on-disk AOM delivery (job `464badee`, the post-edge-fix render) measures **0.533** in
  the belt band vs the doc's **0.842** for the first AOM render `39ec9b96`
  (`docs/served-models.md:824,826`) — the edge-fix delivery is measurably cleaner too, or
  render-to-render spread is large in this band. [INFERENCE on the cause]
- The Hero live discrepancy (0.193 here vs **0.281** in the doc for the same job
  `c23927f0`) is unexplained by metric choice; treat belt-band numbers as having meaningful
  render spread. [INFERENCE]
- The AOM raw-fork render carries **0.779** at 600–900 vs its source's **1.538** — the raw
  decoder *halves* AOM's belt rasp already (same "subtractive ceiling" as noted in the
  doc), while on Hero it *multiplies* (0.147 → 0.281+).

### 1.5 Hum suppression and the edge fix — `docs/served-models.md:885`

Module docstring (`src/auto_voice/inference/fork_hum_suppression.py:1-30`): the decoder
voices ~14 % of frames where the source is unvoiced (13.7 % measured on Hero/ep480); three
steps — (1) HPSS strip to percussive-only, (2) −14 dB duck on voiced-while-source-unvoiced,
(3) soft source-keyed gate with a floor. Order before the halo is load-bearing (+4 dB hum
spread if reverb first).

The edge-artifact incident (`docs/served-models.md:887-903`): crepe labels every consonant
lead unvoiced 40–80 ms before the vowel; on AOM **99 %** of attack windows sat at full
HPSS replacement, the percussive residue carried **5.5 dB** less attack energy, delivered
pre-attack energy read **−62.2 dB** against raw fork's **−52.8** ("every 't', 's' and onset
was being hollowed into a click"). Frame split: **9 %** unvoiced-and-silent (real hum) vs
**18 %** unvoiced-but-audible (consonants). Fix: strip mask now requires unvoiced **AND**
silent (`_silence_mask`, −55 dB of source peak, 10 ms grid;
`fork_hum_suppression.py:118-131` and `:157-168`). Delivered `464badee`: attack windows
**−35.1 dB** (raw fork −35.2), hum in source-silent frames **−84.2 dB**, post-phrase decay
**12.9 dB** vs source 12.0 (`docs/served-models.md:905-910`).

### 1.6 Garage/adaptive halo — `docs/served-models.md:846`

Fixed 30 %-wet 880 ms halo held AOM's phrase tails up ~6.7 dB at 150 ms (decay table at
`docs/served-models.md:856-860`: raw fork 4.0/12.0/14.0 dB @80/150/300 ms; delivered halo
0.30: **−0.5/5.3/8.7**; Hero source −0.3/−0.3/4.4). Fix: wet scaled by
`clip((12 − source_dryness)/12, 0.15, 1.0)`; AOM dryness 17.8 dB → scale 0.15; re-render
`ff720f92` decay @150 ms 5.3 → **11.4** (`docs/served-models.md:871-882`).

### 1.7 Chunk-boundary behavior (patch present, never active)

`patches/svcfork_chunk_fade.patch` documents and restores the fade upstream disabled:
non-speech chunks are replaced with **exact zeros** and hard-butted; with `ref=np.max` and
`top_db=40`, phrase tails/reverb live in the zeroed band; "Measured **2.06 %** exact zeros
on a 4:19 render" (`patches/svcfork_chunk_fade.patch:8-17`). The fade activates only when
env `SVCFORK_CHUNK_FADE` is set to a seconds value (`patches/svcfork_chunk_fade.patch:20-32`).

Search result: **`SVCFORK_CHUNK_FADE` is set nowhere** — not in `data/fork_models/*.json`,
not in `src/`, `scripts/`, `tests/`, nor in `data/app_state/conversion_history.json`
(greps returned zero hits; only `docs`/`patches` mention it). The bridge forwards only
`-ch/-mc/-p/-db/-ab` flags and never passes any fade env
(`src/auto_voice/inference/svc_fork_bridge.py:171-190`), and `_clean_env` copies the parent
environment (`svc_fork_bridge.py:95-132`), so upstream zero-replacement behavior is what
serves today.

> **Provenance gap:** the contract's "measured seam HF-click 0.23–0.43 vs 0.20 baseline on
> delivered" numbers could not be located anywhere in the repo (docs, scratchpad,
> conversion-history notes, server log, git log — searched `hf_click|seam|click|0.23|0.43`
> across `docs/`, `scratchpad/`, `data/app_state/*.json`, backups, and git history; only
> unrelated hits). Treat them as un-locatable on disk; the locatable seam evidence is the
> patch's 2.06 % exact-zeros figure above and the history note "2.06 % of the previous
> render was digital silence; single-chunk inference gives **0.17 %**"
> (`data/app_state/conversion_history.json`, "Full 4:18…" note).

### 1.8 Retrain-side context — `docs/served-models.md:673`

"Open, tracked as `AV-41e4`" (`docs/served-models.md:673-683`): Loss 2 (level collapse with
pitch, **−10.9 dB** at 800–1600 Hz, table at `docs/served-models.md:615-618`) has no
render-side knob. Corpus pitch coverage: median 292 Hz, **p99 496 Hz**, **max 844 Hz**,
**0.48 %** of frames ≥ 550 Hz, **<0.01 %** ≥ 700 Hz, vs a source reaching **1031 Hz**
(`docs/served-models.md:620-624`). "The collapse begins where coverage thins (p99 496 Hz),
not at any threshold or flag" (`docs/served-models.md:625-626`). Cheapest untried lever:
formant-preserving pitch-shift augmentation (+3…+12 st via WORLD) of the existing
`real_sing_sample_07x` files (`docs/served-models.md:675-678`).

---

## 2. Root-cause physics — why the decoder distorts on belts

### 2.1 Where the half-integer partials can enter

The fork's NSF excitation (`SourceModuleHnNSF` → `SineGen`,
`so_vits_svc_fork/modules/decoders/hifigan/_models.py:157-238`) generates **only integer
harmonics of f0**: `fn = torch.multiply(f0, torch.arange(1, self.harmonic_num + 2))`
(`_models.py:134`), i.e. f0, 2f0, … 9f0 (harmonic_num=8, `_models.py:221`). The noise branch
is additive white Gaussian noise: `noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3; noise = noise_amp * torch.randn_like(sine_waves)` (`_models.py:147-148`), plus a
per-stage `noise_convs` injection of noise into each upsampling stage
(`_models.py:247-258`). Therefore:

- **The f0/2 sub-octave cannot come from the sine source.** With f0 as instantaneous
  frequency, only integer multiples exist (this is the design from the NSF paper itself:
  source module generates "a sine-based excitation signal … by treating f_t as the
  instantaneous frequency", Eq. (1), https://arxiv.org/abs/1810.11946).
- It can only appear by (a) the filter/upsampling stack (ConvTranspose1d resblocks)
  resolving a sub-octave when driven out-of-distribution, or (b) the noise branch being
  shaped into periodic-looking bands by the filter. Both require the decoder to
  extrapolate — which is exactly the belt-register condition (no training data above
  ~450 Hz, corpus max 844 Hz vs source 1031 Hz, §1.8).
- The measured signature agrees: doubling grows with pitch (0.206 → 0.515 across 450–900
  Hz, §1.2) and the half-integer energy is phase-coherent enough to be notched effectively
  (−15 dB notches land: corrected 0.172, §1.2), which favors a filter-stack (deterministic)
  mechanism over a noise-branch one.

### 2.2 Why edges artifact even after the hum fix

- The hum incident proved **broad masks damage consonant leads** (§1.5). The same physics
  applies to any per-frame mask keyed on source voicing: crepe labels consonant leads
  unvoiced 40–80 ms early (`docs/served-models.md:893-894`).
- My measurement this session: the **subharmonic stage's own gate is active during ~9 of 10
  onsets** — 83 % (Hero src), 91 % (Hero live), 87 % (AOM src), 91 % (AOM raw), 89 % (AOM
  live) of `librosa.onset.onset_detect` frames have gate depth > 0.05 within −20/+60 ms
  (offline librosa run, §1.4 env). The notches ride the source's f0, which on attack frames
  is transitioning; a 15 dB × ±16 % notch family sweeping through a consonant band is the
  same class of damage the hum strip caused.
- Chunk seams: every phrase boundary is a hard butt-join against either zeros (non-speech
  chunks, §1.7) or an independently-converted chunk (30 s cap). No fade is active (§1.7).

---

## 3. Improvement proposals

Each: mechanism · expected gain · implementation sketch · risk · test plan. Files are
relative to repo root.

### (a) Per-frame peak-ratio gate upgrade (the module's own documented upgrade path)

- **Mechanism.** Replace the gate statistic — currently the mean half-integer bin energy
  over mean integer-bin energy (`fork_subharmonics.py:143-150`) — with the peak-based
  ratio already used for the measured table: `max(P(f0/2), P(3f0/2), P(5f0/2)) / P(f0)` per
  frame, where P is the max magnitude within ±25 Hz (the exact code is
  `scratchpad/rerender/measure_live.py:14-17`). The docstring already flags this:
  "ponytail: per-frame peak ratio … is the upgrade if the residual ever has to move"
  (`fork_subharmonics.py:48-49`), and the docs: "the peak-based ratio in the table is the
  better detector and is the upgrade path" (`docs/served-models.md:806-807`).
- **Why it is sharper — measured this session.** The current mean-ratio statistic barely
  separates clean from doubled on belts (median over loud voiced frames, 600–900 Hz bin):
  Hero source **0.313**, Hero live **0.306**, AOM source **0.666**, AOM raw **0.652**,
  AOM delivered **0.524** — clean belts score *above* converted ones. The peak ratio
  separates by 3.5–10× (0.147/1.538 clean vs 0.515/0.779 converted, §1.2/§1.4). The
  docstring says the same in its own numbers: clean median 0.11, clean belted frames 0.26,
  overlapping converted 0.26/0.52 (`fork_subharmonics.py:31-47`).
- **Current acting rates (measured, mirror of `suppress_subharmonics` with shipped
  0.15/0.40):** Hero src 39 % of all frames (= the doc's 39 % clean figure,
  `docs/served-models.md:806`), Hero live 49 %, AOM raw 47 %, AOM delivered 38 %; on
  belt-band frames it acts on **77–100 %** (Hero live 77 %, n=431; AOM 100 %, n=385-521);
  on clean low-register frames (250–450 Hz) it still acts on **24–50 %** — that is
  collateral where no doubling exists.
- **Expected gain.** Two-sided: (i) collateral on clean frames drops (the −24.9 dB residual
  improves; docs quote the gate as buying "2 dB of residual" today,
  `docs/served-models.md:806`); (ii) because detection is sharper, `max_cut_db` can rise
  from 15 toward 20 dB on true doubling frames without touching clean ones — the belt-band
  end-of-chain target moves from 0.281 (Hero live) toward the source's 0.147.
- **Implementation sketch.** In `suppress_subharmonics` (`fork_subharmonics.py:97-165`),
  after computing `mag`, replace the mean-based `frame_ratio` with a per-frame peak search:
  for each frame k with finite f0, take `max(|S|` in ±25 Hz windows around f0/2, 3f0/2,
  5f0/2`)/(|S|` near f0`)`. This is O(frames × 3 small window maxima) and can reuse the
  already-materialized `S[band, a:b]`; keep the 0.15/0.40 depth ramp on the new statistic
  (peak-ratio clean median 0.11–0.15, converted 0.5+ per the table).
- **Risk.** Peak maxima are noisier than means; a strong breath consonant could read as a
  "subharmonic peak". Mitigate with the existing voicing gate (`_source_f0`,
  `fork_subharmonics.py:61-72`) plus the loudness gate already applied in the metric
  (frames above −24 dB, `measure_live.py:18`). Worst case equals current behavior.
- **Test plan.** Extend `tests/test_fork_subharmonics.py` (existing conventions: synthetic
  voice at `F0`, band-energy assertions at `:32-64`): (1) a doubled synthetic belt (fund +
  3f0/2 at −6 dB) must notch ≥ 6 dB at 3f0/2 (existing
  `test_half_integer_partials_are_removed` pattern, `:32-41`); (2) a clean synthetic voice
  with a strong 2nd harmonic must show < 1 dB movement on all harmonics (existing
  `test_clean_voice_is_left_alone` pattern, `:55-64`) **and** a strictly higher clean
  residual than the mean-gate version; (3) a sweep asserting acting-rate on clean frames
  drops below the measured 39 % baseline.

### (b) Belt-band-specific depth: pitch-keyed depth curve

- **Mechanism.** Depth is currently flat 15 dB regardless of f0
  (`fork_subharmonics.py:97,151-156`). The measured table says the *needed* correction
  varies by band: corrected lands at **0.079** for 450–600 (just above source 0.065) but
  **0.172** for 600–900 vs source 0.144 — i.e. the 450–600 band is already at the
  collateral floor while the belt band still has headroom to cut deeper.
- **Expected gain.** Deeper cuts exactly where the rasp is (600–900: from 0.172 toward
  source 0.144) with *less* collateral at 450–600 (current 0.079 vs source 0.065 could
  relax to ~0.065, recovering real voice). Quantified anchor: the shipped flat depth leaves
  the belt band 20 % above source (0.172/0.144) while 450–600 sits 22 % above (0.079/0.065)
  — re-balancing depth can move both toward 0.
- **Implementation sketch.** In `suppress_subharmonics`, scale `max_cut_db` per frame by a
  curve keyed on the frame's source f0, e.g. `cut(f0) = clip((f0 − 450)/150, 0.5, 1.15) ×
  15 dB` — calibrated so 450–600 gets ~0.6× (→ lands at source level) and 600–900 gets
  ~1.15×. Keep it a pure function of f0 so it is testable.
- **Risk.** Low: it only modulates an existing mask. Watch the 250–450 bands — they are
  near-source already (0.033/0.037 vs 0.013/0.015) and should get ≤ current depth.
- **Test plan.** Synthetic belt at 500 Hz vs 700 Hz: assert the 700 Hz case notches deeper
  than the 500 Hz case by the curve ratio, and the 500 Hz case leaves a smaller residual
  than today's flat version.

### (c) Transient/attack-preservation guard

- **Mechanism.** Skip or fade-in the notch during attack frames. The hum-stage fix proved
  consonant damage is audible and measurable (§1.5); my onset-overlap measurement shows the
  subharmonic gate is active on **83–91 %** of onsets (§2.2) — currently the notch family
  sweeps attack frames whenever f0 is read there.
- **Expected gain.** Unquantified in the repo (no attack metric exists for this stage); the
  hum-stage analog is quantified: fixing the equivalent problem moved attack windows from
  **−62.2 dB** to **−35.1 dB** vs raw fork's −35.2, i.e. a 27 dB recovery on the damaged
  frames (`docs/served-models.md:897-905`). Note the polish section measured the decoder's
  attack softening as only 0.41 dB with polish recovering 0.23 dB
  (`docs/served-models.md:756-758`) — so the *level* cost is small; the risk here is
  spectral, not level: a notch sweeping through an attack can leave a time-varying color.
- **Implementation sketch.** In `suppress_subharmonics`, compute an attack mask from the
  **source** envelope (frames where local RMS rises; e.g. `librosa.onset.onset_detect` on
  the source at `_HOP`), and multiply the per-frame depth by a guard that ramps 0 → 1 over
  ~40 ms after each onset (mirror `_silence_mask`'s 10 ms-grid interpolation pattern in
  `fork_hum_suppression.py:118-131`).
- **Risk.** Belts often *are* attacks (the umph lands on the onset). If a belt onset is
  doubled, the guard defers the correction to steady state — acceptable, since the fuzz is
  worst on sustained belts; but verify the belt-band ratio does not regress on the first
  100 ms of notes.
- **Test plan.** Synthetic doubled note with an amplitude ramp onset: assert (1) harmonic
  movement < 1 dB during the ramp (attack untouched), (2) the steady-state subharmonic
  still drops ≥ 6 dB, (3) belt-band ratio on a real AOM belt excerpt unchanged beyond
  noise (compare against the §1.4 AOM raw numbers).

### (d) Harmonic+noise residual model at belts (comb-filter decomposition)

- **Mechanism.** Replace magnitude notching with an explicit two-component model: use the
  source f0 track to build a comb (attenuate only non-integer partials, leave integer
  partials' phase untouched). This is the harmonic-plus-noise decomposition used in NSF
  follow-up work (HPN-NSF merges harmonic and noise streams in the frequency domain with
  FIR filters; described in the NSF author's follow-up material included with
  https://arxiv.org/abs/1810.11946) and avoids the collateral of broad STFT magnitude
  edits.
- **Expected gain.** Bounded by the same subtractive ceiling as (a)+(b): belt band toward
  source level (Hero 0.281 → ~0.15-0.19; AOM delivered 0.533 vs its own source 1.538 is
  already near the subtractive floor). The real gain is **collateral**: comb filtering on
  the f0 grid leaves integer partials and attack transients intact by construction,
  addressing the measured consonant damage class (§1.5) in one mechanism.
- **Implementation sketch.** New function in `fork_subharmonics.py` (keep
  `suppress_subharmonics` as the shipped path until A/B'd): per frame, compute the
  magnitude spectrum, apply `gain(bin) = 1 − depth·(1 − comb(bin, f0))` where `comb` is a
  periodic raised-cosine lattice on integer multiples of f0 (i.e. suppress energy
  *between* harmonics, not at half-integers only — this also catches 1.25f0-type
  inharmonicity), istft back.
- **Risk.** Higher complexity; comb width must exceed the 50-cent tracking error (the
  module's own analysis: ±12 % notches already tolerate 3 % error,
  `fork_subharmonics.py:35-40`). A too-wide comb dulls breath; too-narrow misses
  mistuned doubling. Also f0/2 doubling means the *fundamental* pattern repeats at f0/2 —
  a comb at f0 does not distinguish f0/2 energy from real low harmonics of a hypothetical
  f0/2 pitch; keep the existing per-frame gate regardless.
- **Test plan.** Same harness as (a) plus a phase-coherence check: integer-partial phase
  (not just magnitude) preserved within tolerance; collateral on the clean synthetic voice
  strictly better than the notch version.

### (e) Literature techniques (searched; primary sources)

- **NSF (neural source-filter) waveform model** — Wang, Takaki, Yamagishi,
  https://arxiv.org/abs/1810.11946. Establishes the source design the fork inherits:
  sine-based excitation at integer multiples of f0 (7 harmonics, σ=0.003, α=0.1 — the same
  constants as `SineGen(sine_amp=0.1, noise_std=0.003)` at
  `_models.py:43-44`), with the unvoiced branch emitting scaled noise (Eq. (1)). This is
  the primary evidence for §2.1's claim that the sine source cannot emit f0/2 —
  the distortion is filter-stack/residual noise shaping, not excitation design.
- **Harmonic-plus-noise NSF** (HPN-NSF; same group's follow-up, described in the
  presentation material bundled with the arXiv PDF above): merges harmonic and noise
  streams in the frequency domain via FIR filters with voicing-dependent crossover — the
  direct literature precedent for proposal (d).
- **Crepe periodicity gating**: already implemented in this repo's own fork patches —
  `patches/svcfork_crepe_periodicity_uv.patch` and `patches/svcfork_uv_contract.patch`
  zero f0 where crepe periodicity < threshold
  (`so_vits_svc_fork/f0.py:143-152`, gated by env `SVCFORK_CREPE_UV_THRESHOLD`; bridge opt-in
  `svc_fork_bridge.py:113-121`). Measured irrelevance for the belt problem: crepe
  periodicity at 800–1100 Hz is **0.717** with only **3.9 %** below 0.3 — "it is *confident*
  in her top register" (`docs/served-models.md:648-651`). So periodicity gating cannot
  separate the belt distortion; the pitch-keyed and peak-ratio detectors above are the
  discriminators that can.
- Not found in this session's searches: an AES/DAFx paper specifically on belt-register
  synthesis artifacts. The literature search was truncated (one primary source verified);
  treat "periodicity-based suppression of subharmonics in neural vocoders" as an open
  search item rather than a cited technique. [INFERENCE: nothing more was verified]

### (f) Chunk-boundary artifact mitigation

- **Mechanism.** Upstream `infer_silence` replaces non-speech chunks with exact zeros and
  butt-joins everything (`so_vits_svc_fork/inference/core.py:267-285`, the patch diff at
  `patches/svcfork_chunk_fade.patch:4-34`); the restore-the-fade patch exists but the env
  key is never set (§1.7).
- **Expected gain.** Eliminates the mid-decay digital-silence steps ("2.06 % exact zeros on
  a 4:19 render", `patches/svcfork_chunk_fade.patch:16-17`) and softens every 30 s chunk
  seam. The 0.23–0.43-vs-0.20 seam HF-click figures cited in the ticket were **not
  locatable in the repo** (§1.7) — re-derive them with a seam-click metric (HF energy in
  ±20 ms windows at chunk boundaries) before/after enabling the patch.
- **Implementation sketch (zero code).** Set `SVCFORK_CHUNK_FADE=0.01` in the serving
  environment (the systemd unit or the service's env file), restart the service, re-render
  one song. The patch reads the env per chunk (`patches/svcfork_chunk_fade.patch:20-32`)
  and falls back to upstream behavior when unset. If adopted permanently, bake the fade
  into `patches/` default (fade_seconds 0.01) so registry entries cannot silently lose it —
  the same class of bug as the chunk-seconds drop (`src/auto_voice/training/svc_fork_trainer.py:54-58`,
  `tests/test_fork_registry_preserves_tuning.py:5-7`).
- **Risk.** A 10 ms fade shaves 5 ms off each chunk end — inaudible against 30 s chunks;
  the real risk is forgetting it is env-gated (config drift), not DSP.
- **Test plan.** Seam metric: locate chunk boundaries from the fork log (`Chunk: ...`
  lines, `core.py:268`), measure 3–6 kHz energy in ±20 ms windows; compare unpatched vs
  patched on the same source stem. Success = boundary HF excess drops to ≤ baseline-level
  with no new decay truncation (check post-phrase decay ≥ 12.0 dB per §1.6).

### (g) Retrain-side levers (out of mix-stage scope; listed for completeness)

Tracked as `AV-41e4` (`docs/served-models.md:673-683`):
1. **High-register data augmentation** — formant-preserving pitch-shift of existing
   `real_sing_sample_07x` files +3…+12 st via WORLD (`docs/served-models.md:675-678`).
   Note the corpus coverage numbers that motivate it: median 292 Hz / p99 496 Hz /
   max 844 Hz / 0.48 % ≥ 550 Hz (`docs/served-models.md:620-624`) vs source 1031 Hz.
   Caution: the 09-04 "synthetic data" run failed differently (SeedVC clips brick-walled
   at 13.1 kHz; `st3` with 6 fresh files was −4.9 dB at the top, among the worst —
   `docs/served-models.md:654-657`), so the augmentation must be formant-preserving and
   full-band.
2. **`-t` transpose trick** — move belts into the trained range and shift back
   (`docs/served-models.md:678-679`).
3. **Keep-source-spans above ~550 Hz** — precedent exists ("PIPELINE #1 already has a
   keep-source-spans precedent", `docs/served-models.md:679-680`).
4. **Discriminator/loss penalizing subharmonic energy** — no repo measurement exists; the
   MRD discriminator work (`patches/svcfork_mrd_discriminator.patch`) is the nearest
   precedent, and the fmax-tracks-fresh-data finding (`docs/served-models.md:681-683`)
   suggests data, not loss, is the binding constraint. [INFERENCE]

---

## 4. Recommended ranking (gain × confidence / cost)

| # | Proposal | Gain | Confidence | Cost | Note |
|---|---|---|---|---|---|
| 1 | **(a) peak-ratio gate** | High (sharper detector enables deeper safe cuts; kills clean-frame collateral) | High — module's own documented upgrade path (`fork_subharmonics.py:48-49`, `docs/served-models.md:806-807`) + my A/B measurement showing the mean gate cannot separate belts | Low — one function's statistic | The single next experiment (below) |
| 2 | **(c) attack guard** | Medium-high on the exact complaint ("edges") | High — measured 83–91 % onset overlap; the identical fix class already shipped for hum (`docs/served-models.md:905-910`) | Low | Measure belt-onset ratio regression as part of experiment |
| 3 | **(b) pitch-keyed depth** | Medium (0.079→~0.065 low band, 0.172→lower belt) | Medium-high — table-driven curve | Low | Stack on top of (a) |
| 4 | **(f) chunk fade env** | Small but free (2.06 % exact zeros → 0; seams softened) | High — patch measured its own numbers | ~Zero (env var) | Do opportunistically with any re-render; re-derive the seam metric |
| 5 | **(d) comb/H+N residual** | Collateral win, similar belt-band ceiling | Medium — literature-backed (NSF/HPN-NSF), untested here | Medium-high | Only if (a)+(b)+(c) leave audible fuzz |
| 6 | **(g) retrain levers** | Removes cause at source | High per docs (`docs/served-models.md:673-683`) | High (training cycle) | Parallel track, not mix-stage |

### The single next experiment (runnable as-is)

**Per-frame peak-ratio gate A/B, offline, no GPU.** Copy `suppress_subharmonics`
(`src/auto_voice/inference/fork_subharmonics.py:97-165`) to a scratch script; replace only
the gate statistic with the peak-based ratio (the ±25 Hz max-window code from
`scratchpad/rerender/measure_live.py:14-17`, per frame over frames with finite gated f0).
Run three variants on `scratchpad/rerender/aom_raw_fork.wav` keyed on
`scratchpad/rerender/aom_src_vocal.wav` (44.1 kHz mono float32):

1. current mean-ratio gate (baseline),
2. peak-ratio gate, same 0.15/0.40 ramp,
3. peak-ratio gate with `max_cut_db=20`.

Report for each: (i) belt-band peak-ratio on the output vs **0.779** raw / **0.533**
delivered / **1.538** AOM source (§1.4); (ii) collateral — run the same stage on the clean
Hero source `scratchpad/rerender/src_vocal_full.wav` as its own "conversion" and report the
residual dB vs the shipped **−24.9 dB** baseline (`docs/served-models.md:805`); (iii)
acting rates (all frames / belt frames / 250–450 Hz frames) vs the measured baselines in
§3(a) (39/47/38 % all; 77–100 % belt); (iv) onset overlap (target: materially below the
83–91 % baseline). Decision rule: adopt the peak gate if belt-band ratio ≤ current mean-gate
output AND clean residual ≤ −30 dB (better than −24.9) AND low-band acting rate drops.
Everything runs in
`PYTHONNOUSERSITE=1 PYTHONPATH=src /home/kp/anaconda3/envs/autovoice-thor/bin/python` with
librosa/numpy only — the same harness used for every number in §1.4 of this file.
