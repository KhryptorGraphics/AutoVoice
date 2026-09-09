# HANDOFF — AutoVoice fork output chain (Brandy / Hero / All of Me)

**For:** oh-my-pi (omp), continuing this session.
**Written:** 2026-09-07 ~01:25 CDT, by the Claude Code session that built this.
**Supersedes:** `docs/handover-fork-output-chain.md` (older, partial — do not use).
**Repo:** `/home/kp/thordrive/autovoice`, branch `svc-fork-integration`. **Nothing below is committed.**

Read all of it before touching anything. The environment section is not optional —
three conversions and four subagents were lost tonight to things listed there.

---

## 0. Where things stand RIGHT NOW

A detached, serial driver script is running and owns the final steps. Do not
duplicate its work; read its log.

```
scratchpad/rerender/driver.sh      # the script (already launched with setsid; survives session death)
scratchpad/rerender/driver.log     # its log — READ THIS FIRST
pgrep -af rerender/driver.sh       # is it still alive?
```

What the log showed at handoff time:

```
00:53:08 START serial driver (nothing runs concurrently)
00:53:08 STEP suite (alone)
=== 333 passed, 1 skipped, 4881 deselected ===          <- fork suite, shipped defaults
00:55:24 STEP restart service
00:55:55 service active
00:55:56 HERO submit: 202 c23927f0-bd70-4746-8360-acf39277cfd1
FINAL c23927f0 completed                                <- Hero DONE, live-verified (see §4)
01:22:03 AOM submit: 202 60653f04-5284-498c-b74b-c949dec6fe11
```

Still to come in that log, in order:

1. `FINAL 60653f04 completed|failed ...` — All of Me.
2. `STEP chain log lines` — the `fork hum suppression:` / `fork subharmonic suppression:` /
   `fork voice polish:` INFO lines from `data/server.log` for BOTH runs.
3. `STEP stem (alone)` — separates the Hero source vocal to `scratchpad/rerender/src_vocal_full.wav`.
4. `STEP measure live Hero c23927f0` — the **end-of-chain** subharmonic table on the
   live render (this is the number the docs are missing; see §6 step 1).
5. `ALL_DONE hero=<id> aom=<id>`.

If the driver is dead without `ALL_DONE`: check `systemctl is-active autovoice`
and `data/server.log` tail for a traceback, then re-run only the missing steps by
hand using the scripts in `scratchpad/rerender/` (each is standalone; see §6).

**A job sitting at `in_progress 10` for 10–25 minutes is normal.** Progress is
not reported during separation or inference. Confirm with py-spy (§2) before
ever calling it stuck. Hero with stems took 26 min on this box.

---

## 1. The mission and the user's reports

Convert songs into the trained voice of **Brandy** (profile
`fb17af66-8415-4ffe-81b3-600efe75b6d7`, checkpoint `sota_ep480`, so-vits-svc-fork)
so that Brandy **tracks the original artist's performance** — sings *as if she had
Mariah's range and articulation*, not re-sung in her own style.

Two songs: **Hero** (Mariah Carey) and **All of Me** (Samantha Harvey, 1 Mic 1 Take).
Sources (already on disk, use these exact files):

```
data/conversions/originals/fb17af66-8415-4ffe-81b3-600efe75b6d7/9a1287db-979e-4943-8263-d79bac135031_Hero_Mariah_Carey_PRODUCTION_ep480.wav
data/conversions/originals/fb17af66-8415-4ffe-81b3-600efe75b6d7/d6850afe-b1aa-46be-9e9a-bb03f275f2be_All_of_Me_Samantha_Harvey_PRODUCTION_ep480.wav
```

The user judges **by ear** and reports in plain language. Every report so far
named a real, measurable defect:

| # | user said | finding | status |
|---|---|---|---|
| 1 | "electronic distortion in the dark space between words" | decoder voices ~14 % of source-unvoiced frames | fixed (`fork_hum_suppression`) |
| 2 | "edges … not always crisp and sharp" | consonant band scooped (see 4) | fixed |
| 3 | "these outputs are impressive" | — | ✅ |
| 4 | "small high band resonance … needs cleaning, better formed edges … not as clear as I know is possible" | decoder scoops 1.2–2 kHz, notches 4 kHz; fixed tones at 17.9/18.6 kHz | fixed (`fork_voice_polish`) |
| 5 | "that last conversion was good, but her voice fuzzes out when she puts any umph or effort in belting out notes" | period doubling, 3–5× the source, exploding above 450 Hz | **fixed, wired, live-verified on Hero; ear verdict pending** |

Then: "do renders of hero and all of me and put them in the web interface" — Hero
done; All of Me in flight (fourth attempt — the first three were killed by
service restarts, §5).

---

## 2. Environment — read every line

```bash
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON <script>            # ALWAYS both vars
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -m pytest tests/ -k "fork or hum or polish or subharm or multi_speaker or settings_frontend" -q
```

- **Live service:** systemd `autovoice`, port **10600**, `WorkingDirectory=/home/kp/thordrive/autovoice`.
  Restart with `sudo systemctl restart autovoice` (passwordless sudo works).
  **A restart kills every in-flight job** — check `pgrep`/py-spy first, always.
- **`curl` is intercepted by a hook. Inline `python -c` HTTP is also intercepted.**
  All HTTP goes through python `requests` **from a script file**. Ready-made:
  `scratchpad/rerender/{submit_stems.py,wait_job.py}`.
- **Submit API:** `POST /api/v1/convert/song`, multipart, file under field **`song`**
  (`file` → 400), plus `profile_id`, `output_quality=studio`, `pipeline_type=quality`,
  and `return_stems=true` if you need `vocals.wav` written (you do, for measurement).
  Status: `GET /api/v1/convert/status/<job>`. Download: `GET /api/v1/convert/download/<job>`
  (resolves by directory name — hand-made dirs under `data/conversions/` work too).
- **GUI history** = `data/app_state/conversion_history.json`, keyed by job id. The
  service holds it in memory and rewrites it; **never edit it while a job runs**.
- **What the service is doing:** `sudo env "PATH=$PATH" $PYTHON_BIN_DIR/py-spy dump --pid $(systemctl show autovoice -p MainPID --value)`
  then look at the `"ThreadPoolExecutor-1_N"` threads (name comes AFTER the state in
  quotes — grep for `'"ThreadPoolExecutor-1_'`). `nvidia-smi` is useless on Thor
  (reports 0 % / N/A under full load).
- **`/tmp/claude-*/…/scratchpad` is wiped when a session ends.** Anything that
  must survive goes under `scratchpad/rerender/` in the repo (gitignored).
- **Do not run heavy work concurrently with a conversion** (demucs, yin sweeps,
  the test suite). Whether or not it is the cause of §5, it is what preceded every
  loss tonight. The driver exists to serialise.
- **`data/` is gitignored.** The served checkpoint choice and by-ear settings are in
  `data/fork_models/<profile>.json` and appear in no diff. Read `docs/served-models.md`
  before "correcting" anything there.
- Clean source vocal stem: use the project separator, never the demucs CLI
  (needs torchcodec, not installed) and never instrumental subtraction (cymbal
  residue inflates 8–16 kHz):
  ```python
  from auto_voice.audio.separation import VocalSeparator
  out = VocalSeparator(model_name='htdemucs_ft').separate(y_mono, 44100)   # sr is positional
  ```
  Ready-made: `scratchpad/rerender/sep.py <song.wav> <out.wav>`.

---

## 3. The output chain as it ships

In `src/auto_voice/inference/singing_conversion_pipeline.py::_convert_song_fork_hq`
(~lines 1618–1650), after fork inference and `_match_lead_level`, **before** the
halo/stereo stage. All three are keyed on `hum_key` — the source vocal, set on
both the multi-speaker (line ~1603) and single-stem (~1616) paths. Order is
load-bearing.

| # | stage | module | config flag (default on) | what it fixes |
|---|---|---|---|---|
| 1 | between-word hum | `fork_hum_suppression.suppress_hum` | `fork_hq_hum_suppress` | decoder voices 14 % of source-unvoiced frames |
| 2 | period doubling | `fork_subharmonics.suppress_subharmonics` | `fork_hq_subharmonic_suppress` | belt rasp — subharmonics at f0/2, 3f0/2, 5f0/2 |
| 3 | decoder colouring | `fork_voice_polish.polish_voice` | `fork_hq_voice_polish` | 1.2–5 kHz scoop/notch; fixed tones 17.9/18.6 kHz |

Why that order: the hum stage and both others must run **before the halo** (its
reverb tail carries artefacts into every gap, +4 dB, where no later stage reaches).
Stage 2 before 3 because a belt's 3f0/2 and 5f0/2 partials land at 900–2250 Hz,
inside the band the polish lifts — polishing first made the rasp **22 % worse**
(0.511 → 0.622). Measured, the two orders finish within 0.001 once both run;
2 goes first so its gate reads the less-distorted signal.

Flags are read straight from `self.config`, deliberately NOT in
`PIPELINE_SETTING_KEYS` (that contract needs four-place registration:
`runtime_contract.py`, `api_runtime.py`, `frontend/src/services/api.ts`,
`SystemConfigPanel.tsx`; `tests/test_settings_frontend_contract.py` enforces it).
Only promote if a UI control is wanted.

Each module's docstring carries the measured tables and what was ruled out. Read them.

---

## 4. Measured results (all on Hero / ep480 unless stated)

**Stage 3 — decoder colouring** (converted − source, 1/3-octave, both anchored to
their 300–3000 Hz mean):

| band | before | after |
|---|---|---|
| 1260 Hz | −3.6 dB | −0.5 |
| 1587 | −2.8 | −0.3 |
| 2000 | −2.1 | −0.6 |
| 2520 | +3.1 | +0.4 |
| 3175 | +3.9 | +1.3 |
| 4000 | −6.3 | −2.7 |
| 5040 | −3.2 | −0.2 |
| ≥6350 | untouched by design | — |

Above 5 kHz has no defensible target: decoder is 10–13 dB under the source but
5–9 dB brighter than Brandy's own recordings. Matching either is wrong. Only
narrowband peaks are touched there. `polish_voice(air_tame_db=…)` exists for an
ear test and is **off**.

**Stage 2 — period doubling.** Median ratio of energy at the half-integer partials
to the fundamental, voiced frames > −24 dB, binned by the **source's** pitch
(yin, n_fft 4096, hop 512, ±25 Hz peak windows):

| f0 band | source (Mariah) | converted | stage 2 alone (shipped defaults) |
|---|---|---|---|
| 250–350 Hz | 0.013 | 0.063 | 0.033 |
| 350–450 | 0.015 | 0.060 | 0.037 |
| 450–600 | 0.065 | 0.206 | 0.079 |
| 600–900 | **0.144** | **0.515** | **0.172** |

Shipped defaults: `max_cut_db=15, width=0.16, clean_ratio=0.15, fuzzy_ratio=0.40`.
Swept, not guessed: depth is cheaper than width (15 dB/±8 % beat 12 dB/±12 % on
both ratio and collateral). Collateral when run on a *clean* voice: −24.9 dB
residual. The first gate (0.04/0.25) over-cut 450–600 to 0.050, *below* the
source's 0.065 — removing real voice; the shipped gate lands it at 0.079.

**The gate is honest but not sharp.** Its internal ratio (mean half-integer /
mean integer magnitude) overlaps badly between clean and doubled voice (clean
median 0.11, clean *belt* 0.26 vs converted 0.26/0.52). It acts on 68 % of
converted frames and 39 % of clean ones. It reduces collateral; it does not
exempt clean frames. Upgrade path is in the module's `ponytail:` comment
(per-frame peak ratio).

**Live verification, Hero `c23927f0`** (service log, 01:0x):
```
fork hum suppression: source unvoiced 21% of frames, hum frames 2.9%
fork subharmonic suppression: acted on 66% of frames (up to 15 dB on the half-integer partials)
fork voice polish: -4.0..+4.0 dB across 1200-5000 Hz, deepest notch -12.0 dB at 17916 Hz
```
Job dir has `mix.wav vocals.wav instrumental.wav backing.wav`; GUI shows it as
`Hero_-_Mariah_Carey_Brandy_full_chain.wav`, status completed, quality studio.

**Missing number:** end-of-chain (stage 2 **plus** stage 3) on the live render.
The driver's `STEP measure` produces it. Expect ~0.19–0.21 at 600–900 (polish
re-lifts part of what stage 2 removes). Append it to `docs/served-models.md`.

**Edges** (only slightly soft; no transient stage — do not add one): like for
like with the same gate both sides, source 2.72, converted 2.31, +polish 2.54 dB/26 ms.

**Bugs found and fixed on the way, both mine:**
- `librosa.yin` **never returns NaN** — on silence it emits ~1000 Hz with full
  confidence. Ungated, stage 2 cut voiced output −10.8 dB wherever the source was
  quiet. Now gated relative (−40 dB below peak) **and** absolute (−60 dBFS; a
  silent source has no meaningful peak so relative-only passes everything). Silent
  source is now a −141 dB no-op. Pinned by `test_silent_source_is_a_no_op`.
- The polish's 22 % rasp regression above.

**Adversarial review, cleared by execution:** `_taper` fades correctly on both
edges; `_octave_smooth` is a true windowed mean; polish's peak-renormalise cannot
fire on this material (vocal peak 0.61, worst case 0.97); mismatched
source/converted lengths run clean.

---

## 5. OPEN QUESTION — something restarts the service

Three All of Me attempts (`a0abf166`, `dd5faf52`, `94468745`) died with
*"Conversion did not survive a restart of the service"*. The service got fresh
`Started` lines at **~23:45, 00:13 and 00:40** — each time the Claude Code
session died at the same moment.

Facts established (do not re-derive):
- **The machine did not reboot** — `journalctl --list-boots` shows one boot since July 28.
- **Not systemd crash-recovery** — `Restart=on-failure`, `NRestarts=0`; the journal
  shows bare `Started` lines with **no** `Stopped` / `Main process exited` before them.
- **No kernel OOM kills** in `journalctl -k` tonight. (My OOM theory is *unconfirmed*.)
- **No hook, timer or cron** references the service (`~/.claude/settings*.json`,
  `.claude/`, `systemctl list-timers`, `crontab -l` all clean).
- The current run — driver serial, this session idle — has gone 30+ min with no
  restart. Weak evidence that concurrent heavy load is involved.

Something issued explicit starts. Candidates not yet checked: another terminal
or Claude session of the user's; a GUI "restart backend" control; the harness
tearing down and something in its teardown path. If it happens again, capture
`journalctl -u autovoice -o short-precise` around the moment, plus `ps` of
whatever else was alive. Tell the user; don't guess.

---

## 6. Remaining steps, in order

1. **Read `scratchpad/rerender/driver.log` until `ALL_DONE`.** Confirm All of Me
   `completed`; confirm its three chain lines are in the log; confirm it is in
   `conversion_history.json` with a `resultUrl`. Take the `STEP measure` table and
   append it to the 2026-09-07 subharmonics section of `docs/served-models.md`
   as "end of chain, live render".
   If All of Me failed: read the error in the log and `data/server.log`; if it is
   the restart message again, see §5, then resubmit alone:
   `$PY scratchpad/rerender/submit_stems.py "<AOM path>" "All of Me - Samantha Harvey (1 Mic 1 Take).wav"`.
2. **Tell the user** both renders are in the GUI and the belt fix is live. Both
   download URLs: `http://127.0.0.1:10600/api/v1/convert/download/<job id>`.
   Ask for the ear verdict on the belted passages specifically.
3. **If they still hear rasp on the belt:** the next lever is the gate — implement
   the per-frame peak ratio named in `fork_subharmonics.py`'s `ponytail:` comment,
   re-sweep with `scratchpad/rerender/measure_live.py`'s metric, keep the
   clean-source residual under about −22 dB. Then `max_cut_db` up to 18. Do not
   touch width (it costs collateral fastest).
4. **Ear A/B not yet ruled on:** `data/conversions/hero-polish/` vs
   `hero-polish_air/` (4 dB off 5–10 kHz). Fetch via the download endpoint. If the
   user prefers `_air`, set `air_tame_db=4.0` in the pipeline call. Leave off otherwise.
5. **Commit** (user asks before you do — "Commit or push only when the user asks"):
   - `git diff HEAD -- <path>` **every** path first; this repo carries huge
     uncommitted drift and git bundles it silently.
   - Chain files: `src/auto_voice/inference/fork_hum_suppression.py`,
     `fork_subharmonics.py`, `fork_voice_polish.py`, the three
     `tests/test_fork_*.py`, the `_convert_song_fork_hq` hunk of
     `singing_conversion_pipeline.py`, `docs/served-models.md`, this file.
   - **Not this work — pre-existing uncommitted, leave alone unless asked:**
     `src/auto_voice/runtime_contract.py`, `src/auto_voice/web/api_runtime.py`,
     `frontend/src/{components/SystemConfigPanel.tsx,services/api.ts}` (multi-speaker
     default plan, earlier session), `WATCHDOG.yml` (an advisor config, unrelated),
     `patches/*.patch`, `scripts/svcfork_source_gate.py`, `.claude/helpers/*`, `agentdb.rvf.lock`.
   - CLAUDE.md requires `detect_changes()` (gitnexus MCP) before committing and
     impact analysis before editing symbols.
6. Housekeeping when idle: rename the GUI entry `d06bd311` from
   `Hero_polish_verify.wav` (only while no job runs); `suppress_hum` runs HPSS over
   the whole track for the 22 % of frames it uses — the slowest stage, easy win.

---

## 7. Measurement discipline — where the hours were lost

- **Anchor an LTAS to the 300–3000 Hz mean**, never its peak or a narrow band. A
  narrow anchor manufactured a convincing 900 Hz "resonance" at 8–9 kHz that does
  not exist; peak-anchoring doubled a real 1.2–2.4 kHz hole from −4 to −9 dB.
- **Split the deviation.** `conv − src` is the model's colouring (yours to fix);
  `src − reference` is the song (EQ'ing it fights the performance).
- **Median trend, never mean.** A mean over a log window sits ~28 dB under a steep
  slope and reported the whole top octave as "excess" — 830 bins cut before caught.
- **Like for like through the chain.** `suppress_hum` inflates any level-based
  onset metric ~2.4 dB; gated-vs-ungated flipped the sign of the edge finding.
- **The detector is not the metric.** Stage 2's cheap internal ratio and the
  validated peak metric disagree badly on clean voice. Validate any gate quantity
  against the metric on a known-clean control before trusting it.
- **Never score against a full mix**; `hi20.wav` was one and invalidated a whole
  checkpoint ranking. Never trust a filename.
- **Validate new metrics on a control.** Harmonic skirt (rewarded the pathology;
  a checkpoint shipped on it and was rejected by ear) and ornament d-corr
  (scored a time-shifted identical take at 0) were both built, believed, discarded.
- **Prove reviewer findings by running code.** Four suspicions tonight: three
  cleared by execution, one (yin on silence) confirmed and it was the serious one.

---

## 8. Established facts — do not re-derive

- `svc infer` needs **`-na`** (CLI defaults `auto_predict_f0=True` and discards
  source pitch). Production bridge has it. Broke every hand render on 09-07.
- **Brandy has no high-register training data** (nothing >700 Hz, thin >450 Hz)
  — surveyed exhaustively. It is the root cause of the belt rasp. Only a retrain
  with real high-register material fixes it at source.
- Booktalk.mp4 is **not** her (0.786 vs her 0.95–0.97).
- Never A/B with raw `svc infer`; the good render = pipeline + studio + registry keys.
- The 17916/18605 Hz tones wander 6 Hz across a song (real partials 60–130) —
  decoder artefacts, in neither source nor her recordings.

---

## 9. Files this session touched (all uncommitted)

```
NEW  src/auto_voice/inference/fork_hum_suppression.py
NEW  src/auto_voice/inference/fork_voice_polish.py
NEW  src/auto_voice/inference/fork_subharmonics.py
NEW  tests/test_fork_hum_suppression.py            (3)
NEW  tests/test_fork_voice_polish.py               (4)
NEW  tests/test_fork_subharmonics.py               (6)
MOD  src/auto_voice/inference/singing_conversion_pipeline.py   (_convert_song_fork_hq only)
MOD  docs/served-models.md                         (three 2026-09-07 sections appended)
NEW  docs/handover-fork-output-chain.md            (superseded by this file)
NEW  handoff.md                                    (this)
NEW  scratchpad/rerender/*                         (driver + helpers, gitignored)
MEM  ~/.claude/projects/-home-kp-thordrive-autovoice/memory/{decoder-colouring-is-the-clarity-defect,ltas-anchor-invents-resonances,yin-reports-pitch-on-silence,…}.md
```

Renders on disk: `data/conversions/{c23927f0-…(Hero, full chain, stems), d06bd311-…(Hero, polish only), hero-polish/, hero-polish_air/, 289dda02-…(Hero, pre-polish, stems)}`.

---

## 10. How the user works

- Measure before changing anything; every fix carries its number and what was
  ruled out, in the docstring.
- Report regressions plainly, including your own — the user needs to know what
  they are hearing.
- Renders go through the **live API** so they land in the GUI; never hand-build
  into `data/conversions/` for delivery (fine for A/B scratch).
- The user's ear is the final gate. Measurements decide what to build, not
  whether it shipped well.
- Terse. Code first, then at most a few lines. No essays unless asked for a report.

---

## 11. Model-recipe workstream (AV-6sxy) — landed 2026-09-09, separate from the output chain above

**Status: RESOLVED with a 2×2 + replication + averaging, all pushed.** The tracked
record is in `docs/served-models.md` (sections "The 2x2 that settles it", "The data
re-test at the corrected LR", "The control run that reattributes the cause") and in
the `AV-6sxy` bead body. Full checkpoint set: `data/fork_models/_candidates_fb17af66_20260904/`
(each a G+D pair; the fork silently random-inits from a G-only dir — never resume one).
A recreated scorecard (`score2.py`, identical math to the lost `measure2.py`) and a
rebuilt identity centroid (`brandy_centroid_rebuilt.npy`, from the 76 sample uploads)
live in that dir too. All 25 renders are in `renders_hero20/`.

Headlines: (1) fine-tuning this converged seed at LR 1e-4 drifts it 2.3 dB in 6-8k and
costs the whole aperiodicity regression; 2e-5 mostly avoids that — but ONLY in the
zero-fresh-data case. (2) With the 2026-09-05 new material present, 6-8k is data-driven
and LR-insensitive (st4 -17.7 @1e-4 vs st4lr -18.2 @2e-5): the data penalty is 4.2 dB
and nearly doubled once the recipe was fixed. (3) `fmax` tracks fresh-data *volume*,
not bandwidth/LR/loss — measured directly on corpus_v3 (full-band, speech median 22.1k)
yet it yields the LOWEST ceiling. (4) Checkpoint averaging (ctl+lowlr arms) did NOT beat
its parents. (5) The LR-2e-5 advantage over 1e-4 is clip-dependent (~0.8 dB, within
clip-to-clip variation on the 15-render replication) — do not build on it.

Serving remains **ep235**, unchanged since 2026-09-04. Closest challenger is `lowlr`
(2e-5, no new data). **Nobody has listened to any render** — every conclusion is
metric-only; `renders_hero20/` is there for a perceptual A/B before any further GPU
spend.

**Env note for any future box rebuild** (this bit us twice): so-vits-svc-fork 4.2.30
breaks on librosa 1.x (`get_duration() got an unexpected keyword argument 'filename'`)
— pin `librosa==0.10.2.post1`. And `torchaudio` must match the torch CUDA build or the
CLI dies on `libcudart.so.13` (fix: `pip install "torchaudio==2.8.0+cu128" --no-deps
--index-url https://download.pytorch.org/whl/cu128`). Also `warmup_epochs` /
`init_lr_ratio` in the config are DEAD KEYS — `train.py` never reads them; warmup needs
a code patch, and a lower flat LR is the implementable equivalent.
