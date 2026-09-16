> **SUPERSEDED** — see `../handoff.md` (2026-09-07 01:25). Kept for history only.

# Handover — AutoVoice fork output chain (Brandy / Hero / All of Me)

You are taking over mid-task on **AutoVoice** (`/home/kp/thordrive/autovoice`,
branch `svc-fork-integration`). Read this whole file before touching anything.

## The job

Convert songs into the trained voice of **Brandy** (profile
`fb17af66-8415-4ffe-81b3-600efe75b6d7`, checkpoint `sota_ep480`) via
so-vits-svc-fork. The voice must **track the original artist's performance** —
Brandy singing as if she had Mariah Carey's range and articulation — not
re-sing it in her own style. Two reference songs: Mariah Carey's *Hero*, and
*All of Me* (Samantha Harvey, 1 Mic 1 Take).

The user judges by ear and reports in plain language. Their reports so far, in
order, each of which turned out to name a real, measurable defect:

1. "electronic distortion in the dark space between words" → fixed
2. "edges of the converted is not always crisp and sharp" → fixed
3. "these outputs are impressive" ✅
4. "her actual voice has a small high band resonence… her voice needs cleaning,
   better formed edges" + "does not come through as clear as I know is possible"
   → fixed
5. "that last conversion was good, but her voice fuzzes out when she puts any
   umph or effort in belting out notes. investigate and fix it" → fixed and
   wired; live re-render + ear verdict outstanding

## Environment

```bash
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON <script>          # always both vars
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -m pytest tests/ -k fork -q
```

- The live service is systemd `autovoice` on **port 10600**, working dir
  `/home/kp/thordrive/autovoice`. Restart it to pick up code changes.
- **`curl` is intercepted by a hook.** Use python `requests` **from a script
  file**, never inline via `python -c` — inline HTTP is also intercepted.
- `data/` is gitignored. The live checkpoint choice and tuned serving settings
  live in `data/fork_models/<profile>.json` and appear in no diff. Read
  `docs/served-models.md` before "correcting" anything there.
- Thor's GPU telemetry is unreadable: `nvidia-smi` reports 0% and N/A under full
  load. Never infer idle from it. Use `py-spy dump --pid <pid>` (needs
  `sudo env "PATH=$PATH"`) to see what the service is actually doing.
- Conversions sit at `in_progress 10` for their whole run — progress is not
  reported during separation/inference. That is **not** a stall; confirm with
  py-spy before diagnosing one. A full song takes ~7 min alone, longer if two
  jobs share the box.

## The output chain as it stands

In `singing_conversion_pipeline._convert_song_fork_hq` (~line 1618 onward),
after inference and level match, before the halo/stereo stage. All keyed on
`hum_key` — the source vocal, set on both the multi-speaker and single-stem
paths (lines 1603 / 1616):

| stage | module | config flag | status |
|---|---|---|---|
| between-word hum | `fork_hum_suppression.suppress_hum` | `fork_hq_hum_suppress` | live, user-approved |
| decoder colouring | `fork_voice_polish.polish_voice` | `fork_hq_voice_polish` | live, live-verified |
| period doubling | `fork_subharmonics.suppress_subharmonics` | `fork_hq_subharmonic_suppress` | wired, tested, awaiting live verification |

All three default **on** when wired. Flags are read straight from
`self.config`; they are deliberately not in `PIPELINE_SETTING_KEYS` (that
contract needs four-place registration — runtime_contract, api_runtime,
frontend api.ts, SystemConfigPanel — only do it if a UI control is wanted).

Everything above is **uncommitted**. `git status --short -- src tests docs`.

## What is already established — do not re-derive

- **`svc infer` needs `-na`.** The CLI defaults `auto_predict_f0=True`; without
  `-na` the model discards the source pitch entirely. This broke every hand
  render on 2026-09-07 (in tune 7.0% → 99.4% once fixed). The production bridge
  has it.
- **Decoder colouring** (converted minus source, 1/3-octave, both anchored to
  their 300–3000 Hz mean): scoops 1260 Hz −3.6 dB, 1587 −2.8, 2000 −2.1; bumps
  2520 +3.1, 3175 +3.9; notches 4000 −6.3. `polish_voice` closes all of these to
  under 1.5 dB. Live-verified in the service log:
  `fork voice polish: -4.0..+4.0 dB across 1200-5000 Hz, deepest notch -12.0 dB
  at 17916 Hz`.
- **Above 5 kHz has no defensible target.** The decoder sits 10–13 dB under the
  source but 5–9 dB *brighter* than Brandy's own recordings. Matching the source
  undoes the conversion; matching her catalogue chases a mastering difference.
  Only narrowband peaks are touched up there. `polish_voice(air_tame_db=…)`
  exists to test the second by ear; it is off and should stay off until the user
  has A/B'd it.
- **Fixed tones at 17916 / 18605 Hz** are decoder artifacts — 6 Hz wander across
  a whole song where real partials wander 60–130 Hz, absent from source and from
  her recordings. Notched.
- **Edges are only slightly soft.** Measured like for like (same gate on both
  sides): source 2.72, converted 2.31, +polish 2.54 dB/26 ms. ~0.18 dB residual.
  Do **not** add a transient shaper for this. `suppress_hum` is cleared of eating
  attacks (+0.8 % attack harmonics when fed a real voice as both inputs).
- **Brandy has no high-register training data.** Nothing above ~700 Hz in any
  local source; thin above 450 Hz. Verified exhaustively — do not go looking
  again. This is the root cause of the current defect.

## The belting fuzz — DONE, pending live verification

Diagnosis, fix, tests, wiring and docs are complete; see the 2026-09-07 section
of `docs/served-models.md` for the measured tables. Summary:

- `fork_subharmonics.suppress_subharmonics` is **wired in** behind
  `fork_hq_subharmonic_suppress` (default on), BEFORE `polish_voice`. Belt
  ratio 0.515 → 0.172 (source 0.144); clean-source collateral −24.9 dB.
- Shipped defaults: 15 dB, width 0.16, gate 0.15/0.40. Swept, not guessed.
- `tests/test_fork_subharmonics.py`: 6 tests, incl. the silent-source no-op.
- Fork suite: 333 passed, 1 skipped.

Two things the next person must know that are not obvious from the code:

1. **`librosa.yin` never returns NaN.** On silence it emits ~fmax with full
   confidence. Any stage keyed on a tracked f0 needs a voicing gate with BOTH a
   relative and an absolute floor (a silent source has no meaningful peak, so a
   relative-only gate passes everything). Ungated, this stage cut voiced output
   −10.8 dB on quiet source frames.
2. **The gate's internal ratio is not the metric's ratio.** The stage gates on
   mean-over-bins (cheap); the defect is measured on peaks (discriminating). They
   overlap badly on clean voice — a clean belt sits at 0.26 internally, above the
   old fuzzy threshold. The gate reduces collateral; it does not exempt clean
   frames. Do not describe it as "adaptive" without that caveat.

### Remaining steps (in order — do not reorder)

1. **Wait for All of Me to finish before restarting anything.** A restart kills
   in-flight jobs; two All of Me attempts have died that way already.
2. Restart `autovoice`, re-render BOTH songs through the live API (field name
   `song`), and confirm all three chain lines appear in `data/server.log` for
   each run: `fork hum suppression:`, `fork subharmonic suppression:`,
   `fork voice polish:`.
3. Measure the LIVE render's converted vocal against the source with the metric
   in served-models.md — that is the true end-of-chain number (de-sub + polish
   together); the tables so far are de-sub alone. Append it to served-models.md.
   Submit with stems so `vocals.wav` is written (check `api_conversion.py` for
   the field — `stems=False` shows in the log for plain submits).
4. The user's ear is the gate. If they still hear rasp on the belt, the upgrade
   path is written in the module's `ponytail:` comment.

### Also outstanding

- **All of Me has never rendered successfully.** Job `dd5faf52-bd63-4655-8926-
  90550560b2e3` was in progress at handover; an earlier attempt
  (`a0abf166`) died when the service restarted mid-run. Source:
  `data/conversions/originals/fb17af66-.../d6850afe-…_All_of_Me_Samantha_Harvey_PRODUCTION_ep480.wav`.
  Submit with field name **`song`** (not `file`), `profile_id`,
  `output_quality=studio`, `pipeline_type=quality`.
- The completed Hero render `d06bd311-8d17-4240-86cf-20ceaa2bb0ad` shows in the
  GUI as `Hero_polish_verify.wav`. Worth renaming in
  `data/app_state/conversion_history.json` — but only while no job is running,
  or the service will overwrite your edit from its in-memory copy.
- Two ear-A/B renders exist at `data/conversions/hero-polish/` and
  `hero-polish_air/` (the latter with 4 dB off 5–10 kHz). The user has not ruled
  on `air_tame_db`. Both fetch fine at
  `/api/v1/convert/download/<dirname>` — the endpoint resolves by directory.
- `suppress_hum` runs HPSS over the **entire** track but uses it on only the
  22 % of frames where the source is unvoiced. It is the slowest thing in the
  chain. Straightforward optimisation if turnaround starts to matter.

## Measurement discipline — this is where the time gets lost

Every one of these cost real hours in this session. They are not hypothetical.

- **Anchor an LTAS to the 300–3000 Hz mean, never to its peak or a narrow band.**
  Anchored to the 8–10.5 kHz mean, a convincing 900 Hz "resonance" appeared at
  8.0–8.9 kHz that does not exist — it was the top of an ordinary decline. Peak
  normalisation separately doubled a real 1.2–2.4 kHz hole from −4 to −9 dB.
- **Split the deviation.** `conv − src` is the model's own colouring and is yours
  to fix. `src − reference` is the song, and EQ'ing it fights the performance.
  Never act on a difference without splitting it first.
- **Use a median for a trend, never a mean.** A mean over a log-spaced window
  sits ~28 dB below a steeply falling curve and reported the whole top octave as
  "excess", cutting 830 bins by 8 dB before I caught it.
- **Compare like for like through the chain.** `suppress_hum` inflates any
  level-based onset metric by ~2.4 dB. Comparing a gated conversion against an
  ungated source flipped the sign of my edge finding. Gate both sides or gate
  neither.
- **Never score against a full mix.** A file named like a vocal may not be one.
  `hi20.wav` was a full mix (vocals 0.9 dB above backing) and invalidated an
  entire checkpoint ranking.
- **Get a clean source stem from the project's own separator**, not by
  subtracting the instrumental (cymbal residue inflates the 8–16 kHz band) and
  not from the `demucs` CLI (it needs torchcodec, which is not installed):
  ```python
  from auto_voice.audio.separation import VocalSeparator
  out = VocalSeparator(model_name='htdemucs_ft').separate(y, 44100)  # y mono, sr positional
  ```
- **Validate every new metric against a known-good control** before trusting it.
  Two metrics were built, believed, and discarded this session: harmonic skirt
  (it rewarded the exact tonal pathology being hunted, and a checkpoint shipped
  on it that the user rejected by ear) and ornament d-corr (it scored an
  identical performance shifted in time at ≈0, so it could not tell invented
  ornament from frame misalignment).

## Working style the user expects

- Measure before changing anything. Every fix in this chain is justified by a
  number in a docstring, and every docstring says what was ruled out and why.
- Report honestly, including your own regressions — the polish making the fuzz
  22 % worse is stated plainly above because the user needs to know what they
  are hearing.
- Renders go through the **live API** so they land in the GUI automatically, not
  hand-built into `data/conversions/`.
- The user's ear is the final gate. Measurements decide what to build; they do
  not decide whether it shipped well.
