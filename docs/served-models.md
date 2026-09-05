# What the live system actually serves

`data/` is gitignored, so **the two things that most determine how a conversion
sounds — which checkpoint is loaded, and the multi-speaker settings — are not in
this repository.** They live only on the serving box:

| what | where | tracked? |
|---|---|---|
| checkpoint + f0 method per profile | `data/fork_models/<profile_id>.json` | **yes** — predates the ignore rule, so commit changes to it |
| live pipeline settings | `data/app_state/app_settings.json` (via `PATCH /api/v1/settings/app`) | no |
| the checkpoints themselves | `/home/kp/thordrive/autofusion/autovoice/checkpoints/…` | no (multi-GB) |

Note the asymmetry: the registry entry *is* version-controlled (it was committed
before `data/` was ignored, and gitignore does not apply to already-tracked
files), but the settings that reach the pipeline are not. `git add` will refuse
the path as ignored; `git commit -- <path>` works.

This file is the tracked record of that state. **Update it whenever a model is
promoted or a serving default is changed by ear.** The cost of not doing so is
concrete: a serving-side change went unrecorded once, and a later session
diffing the registry against `HEAD` had no way to tell an intentional tuning
decision from accidental drift. That ambiguity is what a whole debugging session
was eventually spent resolving.

---

## Current state (2026-09-01, profile `f16189d9-f6a1-4eac-813a-2f89e6a92f65`, "Conor")

```json
{
  "model_path":  ".../checkpoints/svcfork_conor_fullband_20260828_mrd_uvfix/G_197.pth",
  "config_path": ".../checkpoints/svcfork_conor_fullband_20260828_mrd_uvfix/config.json",
  "f0_method": "crepe",
  "trained_epochs": 197,
  "noise_scale": 0.2,
  "chunk_seconds": 30.0, "max_chunk_seconds": 40.0,
  "pad_seconds": 1.0, "db_thresh": -40,
  "requires_uv_contract": false
}
```

### Why this checkpoint

`G_197` is the MRD-discriminator retrain. It was chosen **by ear** in a blind-ish
A/B against the previously served `G_166` and several alternatives, and was the
clear winner. It is served with `f0_method: crepe`, which is what it was trained
with — no train/serve mismatch, which is why it was preferred over a
higher-scoring variant that needed `harvest`.

`requires_uv_contract` is **false** deliberately: `G_197` trained with crepe, and
crepe emits no genuine unvoiced frames, so the uv-contract patch was a no-op
throughout its training. Serving it with the contract on would be a fresh
mismatch. See `patches/svcfork_uv_contract.patch`.

### Why `f0_method: crepe` specifically

Do not "fix" this to `harvest`. `G_197`/`G_166` trained on crepe, whose f0 is
never truly zero, so `emb_uv` row 0 (the unvoiced embedding) never received a
gradient. Serving with harvest routes real unvoiced frames — 12% of a typical
vocal — through that untrained row.

Caveat worth knowing: correcting this did **not** audibly fix the "buzz" it was
expected to. `interpolate_f0` fills every unvoiced frame before the vocoder sees
it, so `SineGen` measures `uv == 1` on every frame regardless of f0 method, and
the NSF noise branch never fires either way. The fix is correct; it just is not
the lever it looks like.

### Rollback

Restore a backup beside the registry entry and restart:

```bash
cp data/fork_models/<id>.json.pre_g197_backup data/fork_models/<id>.json
sudo systemctl restart autovoice.service
curl -s localhost:10600/api/v1/profiles/<id>/adapters   # confirm fork_engine
```

Backups on the box: `.pre_g197_backup` (G_166 + crepe), `.pre_f0fix_backup`
(G_166 + harvest, the buzzing configuration), `.pre_G166_backup.json` (G_100).

**A restart is required.** `svc_fork_bridge._CACHE` is process-lifetime and
`clear_cache()` only fires automatically after training. Editing the JSON alone
changes nothing on a running server, and the `Applied stored app setting` boot
log covers a *different* subsystem — it will not mention the fork registry.

---

## Live multi-speaker settings

Tuned by ear on *One Last Time*; **not defaults**. Anything not listed is at its
code default.

| setting | live | code default | why |
|---|---|---|---|
| `multi_speaker_separator` | `karaoke_model` | `diarization` | splits simultaneous harmony from under the lead |
| `multi_speaker_convert_backing` | `true` | off | converts harmony lines to the target voice |
| `multi_speaker_backing_gain` | `1.6` | `1.0` | harmonies sat too quiet |
| `multi_speaker_line_harmonics` | `24` | `24` | |
| `multi_speaker_line_onset_ms` | `30` | `30` | |
| `multi_speaker_backing_voiced_min` | `0.5` | `0.65` | |
| `multi_speaker_backing_whole_voiced_min` | `0.92` | `0.7` | high on purpose: forces per-line decomposition instead of flattening the stack to one voice |
| `multi_speaker_karaoke_leak_voiced_min` | `0.82` | `0.65` | this song's backing measures lead-like; the default rejected the split outright |
| `multi_speaker_bleed_suppression` | `ls` | `off` | **see caveat below** |

`multi_speaker_bleed_suppression` is currently `ls` on this box **only because it
was left on after an A/B**. It measured `0.00 dB removed` at `0.26` mean
coherence — this separator's split is already ~99% clean, so there is nothing
coherent to cancel and it is an expensive no-op here. Setting it back to `off`
is safe and is the shipped default.

### Thresholds that are load-bearing and fragile

- `multi_speaker_unison_semitones` / `_note_frac` (`1.0` / `0.5`) decide whether a
  line is a double-tracked lead (folded into the lead, converted once) or a real
  harmony. On the reference song the same content measured **49%** on one run and
  **53%** on another against a 50% cutoff — it fired on one and not the other.
  **Treat this as unreliable across songs**; a better discriminator is still
  wanted.
- `multi_speaker_line_concentration_min` (`1.2`) is now an ENRICHMENT ratio over
  chance, not a raw captured fraction — `1.0` means "no better concentrated than
  noise". It was a raw fraction until it silently rejected genuine harmony lines
  twice and then admitted pure noise once, all because the raw value scales with
  how wide the comb mask is. Enrichment is width-invariant, so changing
  `line_harmonics` no longer re-tunes this gate by accident. Margins are real but
  not generous: measured noise `0.99`, real lines `1.39`.

---

## Conventions

- Promote a checkpoint by **editing the registry JSON directly**, not through the
  training UI, unless the trainer produced it. `svc_fork_trainer.train_svc_fork`
  rebuilds the entry on promotion, so tuning that lives only in the registry can
  be overwritten by a retrain — check the entry after any training run.

  (An earlier version of this file claimed a `_PRESERVED_INFERENCE_KEYS`
  constant carries that tuning forward. It does not exist in this checkout, in
  the sibling one, or in any commit — the claim came from a stale context, and
  the tests referencing it fail to import. Treat promotion as destructive to
  registry tuning until something actually preserves it.)
- Always leave a `.pre_<change>_backup` beside the entry.
- Verify via `GET /api/v1/profiles/<id>/adapters` after restarting — it reads the
  same cache the conversion path uses.
- Then update this file.

## Brandy (`fb17af66`) — 2026-09-04: `db_thresh: -40`, `pad_seconds: 1.0` added

The entry had neither key, so inference fell back to the fork's `-db -20`, which
gates everything quieter than -20 dB relative to peak. Measured on the real
"Hero" vocal: converted output sat **-31 dB below the source on quiet voiced
frames** and -24 dB in the 150 ms after phrase ends — the user heard it as "the
voice falls off around the edges". `-35` and `-50` both restored the edges
(+4 dB, i.e. fully); `-40` matches Conor's proven value. `pad_seconds: 1.0`
likewise matches Conor (longer chunk crossfade).

Weights unchanged (ep235). Two 2026-09-04 retrains did not measurably beat it:
the synthetic-data run was 9-15 dB dark above 6 kHz (SeedVC clips brick-walled at
13.1 kHz) and the real-only control was level on brightness but noisier in
8-12 kHz. Candidates: `data/fork_models/_candidates_fb17af66_20260904/`.
Rollback: `fb17af66-...json.pre_edgefix_20260904`.

## Brandy (`fb17af66`) — 2026-09-04 evening: Conor's inference keys applied

`noise_scale: 0.2`, `chunk_seconds: 30`, `max_chunk_seconds: 40` (plus the
`pad_seconds: 1.0` / `db_thresh: -40` from the morning). The entry never had
them, so Brandy ran at the fork defaults — `noise_scale 0.4` and **`chunk_seconds
0.5`**, i.e. the vocal converted in half-second slices — while Conor's good
"One Last Time" renders used 30 s chunks. Rollback: `*.pre_conorkeys_20260904`.

Pipeline finding on "Hero" (job `e2711bfe`): the karaoke separator measured
0.1 % backing energy (One Last Time: 19.5 %), so the song is handled as a solo
via diarization; the diarizer put 25.6 s of quiet phrase tails/onsets into a
"backing" speaker and the `multi_speaker_backing_voiced_min` gate (0.50) kept
them unconverted at voiced 0.49 — audible as the source singer bleeding at the
edges of the converted lead. Gate lowered to 0.45 at runtime for the re-run.

## Brandy (`fb17af66`) — 2026-09-05: six new-data training levers tried against a v3 candidate; live registry unchanged

**What is actually serving right now** (verified via
`GET /api/v1/profiles/fb17af66-.../adapters` and the registry file, not assumed, and
corroborated by `_candidates_fb17af66_20260904/README.md:1`, written 09-04):
`trained_epochs: 235`, `model_path: fb17af66-...-svcfork/G.pth` — the original pre-v3
checkpoint. Registry mtime 2026-09-04 20:06, unchanged since. Full history from the
backup files beside the registry entry: `v3 ep135` *was* promoted at 14:30 that day
(`.v3_ep135_serving_snapshot`), then superseded by the MRD-discriminator experiments
that evening, which were themselves rolled back by 20:06 — landing back on this same
ep235 checkpoint. `v3 ep135` has not been re-promoted since. **Do not read "v3 ep135"
below as the live model** — it is a checkpoint from the 09-04 `v3_uv_bright` run, used
here only as the seed and comparison baseline because the prior session recommended it
as a starting point. That run's own README (line 12) documents **`G_112` — not
`G_135`** — as its best epoch ("6-8k at her real level... `>=ep202` loses the top
octave again"); `G_135` was carried forward only as an available resume point, not
because it was independently confirmed best. Nothing in today's session was measured
against `G_112`, nor against the live ep235 — every comparison below is new experiments
vs. one render of `v3 ep135`, nothing more.

This investigation spans two batches of new source material, six training runs total
(matching every render saved in `renders_hero20/`):
- **st2 / st2b**: an earlier batch of 4 files (3 band-limited + 1 full-band, found and
  separated in a prior session) folded into the existing corpus, in two stages: Stage 1
  (5000 steps, full corpus, seeded from `v3 ep135`) produced an intermediate checkpoint
  (`st1`, epoch 143). Stage 2 fine-tuned that on the full-band subset only: st2 at
  ≈2000 steps (this session's config target; underperformed), st2b re-running the same
  Stage 2 at 10000 steps (`max_epochs: 257` in the training log) after st2 underperformed.
- **st3 / st4 / st5 / st6**: 6 new phone videos (~23 min separated vocals after demucs,
  the batch explicitly requested this session) added on top of the st2 corpus (so they
  carry the earlier 4-file batch too). st3 = full corpus, new files x3; st4 = same,
  band-limited files at x1; st5 = fidelity tail on st4; st6 = the single cleanest new
  file only, x3.

Every run was seeded from the `v3 ep135` checkpoint — directly for st3/st4/st6
(5000-step budget each, live-confirmed in their training logs), st5 as a shorter tail
on st4's output (not a fresh 5000-step run), or via the intermediate `st1` checkpoint
for st2/st2b (see above) — same recipe throughout (crepe f0, `SVCFORK_UV_CONTRACT=1`,
`SVCFORK_CREPE_UV_THRESHOLD=0.3`, LR 1e-4, batch 16), scored on
the same hero20.wav render with the same scorecard (`measure2.py`: 6-8kHz/8-12kHz band
levels relative to 300-1000Hz, `fmax`, D4C aperiodicity) plus Resemblyzer identity
against her singing centroid. **This scorecard is metric-only — no one listened to any
of these renders.** Treat every number below as a screening signal, not a substitute
for hearing them; all 7 renders (the `v3 ep135` baseline plus the six runs) are saved at
`data/fork_models/_candidates_fb17af66_20260904/renders_hero20/` for a future session to
A/B by ear before acting on this table.

| run | corpus | 6-8k dB | fmax | identity |
|---|---|---|---|---|
| **v3 ep135 (seed/baseline, NOT live)** | — | **-13.2** | 17.9k | **0.929** |
| st2 | existing corpus + earlier 4-file batch, 2k-step fidelity tail | -17.2 | 15.5k | 0.923 |
| st2b | same corpus, 10k-step fidelity tail | -17.3 | 17.9k | 0.929 |
| st3 | + 6 new videos x3 | -23.3 | 22.1k | 0.900 |
| st4 | same, 5 band-limited new videos at x1 | -17.7 | 17.9k | 0.925 |
| st5 | fidelity tail on st4 (full-band-only subset) | -21.5 | 14.4k | 0.911 |
| st6 | single cleanest new video only, x3 | -18.5 | 14.9k | 0.915 |

**Every configuration scored worse than the `v3 ep135` seed/baseline on 6-8k level**,
and none beat it on identity either — st2b ties it exactly (0.929, see below); nothing
exceeded it. Two things ruled out along the way:
- **Not a bandwidth-extension problem**: 5 of 6 new files are band-limited (11.8-14.6 kHz),
  yet the runs *containing* them hold the highest output ceilings (st4 17.9k, st3 22.1k)
  while the 1 full-band file alone (st6, `b4_6` at 22.1 kHz) gave only 14.9k. Added-file
  bandwidth is anti-correlated with output `fmax` — see the 2x2 below, which identifies
  fresh-data *volume* as what actually drives it.
- **Not simply a "need more fidelity steps" problem**: st2 (≈2k steps) → st2b (10k
  steps, 5x more) recovered `fmax` and identity to exact parity with the baseline
  (17.9k/0.929 both ways) but left the 6-8k deficit essentially unchanged (-17.2 →
  -17.3). More steps fixes bandwidth and identity here, not brightness — a narrower,
  more specific problem than plain undertraining. st5's fidelity tail (on st4, a
  different corpus) underperformed outright, for the speech-dominance reason below.
  The "full-band" subset in this corpus is ~60% speech by duration, so a fidelity tail
  re-anchors toward *speech* spectra, not singing — the tail direction was wrong for
  that corpus, not just under-trained.

### The 2x2 that settles it: LR and fresh-data volume act on different axes

Every one of the six runs above changed *two* things at once versus the seed — it added
new data **and** it continued training — so none of them could attribute cause. Three
further runs close that: a control (seed's own corpus, LR 1e-4), the same corpus at LR
2e-5, and st4's corpus at LR 2e-5. All at 5000 steps from the same `v3 ep135` seed, same
hero20 render, same scorecard. `warmup_epochs` is vestigial in this fork (`train.py`
never reads it), so a lower flat LR is the implementable equivalent; the arm was verified
by reading the runtime optimizer state out of the checkpoint
(`optimizer.param_groups[0]["lr"] = 1.9967e-05`), not the config value.

| run | LR | fresh data | 6-8k | 8-12k | fmax | aper 2-6k | aper 6-12k | identity |
|---|---|---|---|---|---|---|---|---|
| **v3 ep135 (seed)** | — | — | **-13.2** | -32.2 | 17.9k | 0.664 | **0.835** | **0.929** |
| ctl | 1e-4 | none | -15.5 | -34.8 | 14.6k | 0.599 | 0.799 | 0.921 |
| **lowlr** | **2e-5** | none | **-14.0** | **-32.4** | 15.0k | **0.675** | 0.827 | 0.922 |
| st4 | 1e-4 | +6 videos | -17.7 | -33.7 | 17.9k | 0.549 | 0.786 | 0.925 |
| st4lr | 2e-5 | +6 videos | -18.2 | -33.7 | 17.9k | 0.555 | 0.753 | 0.919 |

**The two effects do not decompose.** An additive model (2.3 dB recipe + 2.2 dB data)
predicts st4lr at ≈-16.2. It measured **-18.2** — *worse* than st4 at 5x the LR. So the
LR term is not a constant you can subtract out of the new-data runs.

**6-8k — LR matters only with no fresh data.**
- no fresh data: -15.5 (1e-4) → **-14.0** (2e-5). Real, 1.5 dB.
- with fresh data: -17.7 (1e-4) → -18.2 (2e-5). Nil; 8-12k is identical (-33.7) both ways.

So continuing to train on already-fit data drifts ~2.3 dB and a gentler LR mostly avoids
that — but once new material is present the 6-8k loss is data-driven and **LR-insensitive**.
A future session must not fine-tune new material at 2e-5 expecting brightness to hold.

**Aperiodicity — same shape.** With no fresh data, 2e-5 fully recovers both bands (2-6k
0.675, above the seed's 0.664). With fresh data it does not (6-12k 0.753, the worst value
in the table). Aperiodicity is the buzz metric this line of work started from, so on the
seed's own corpus the buzz was substantially self-inflicted by the fine-tuning LR.

**`fmax` — tracks fresh-data *volume*, not bandwidth, not LR, not the loss.** Measured
directly, `corpus_v3` is **not** band-limited: speech median 22.1 kHz (n=69, only 6%
below 16k), singing all ≥22.0 kHz. Yet it yields the *lowest* output ceiling. Ruling
each candidate out:
- not LR: 14.6k → 15.0k across a 5x change, vs a 3 kHz corpus-conditional split.
- not the added files' bandwidth: st6's single *full-band* file gave 14.9k, while st4's
  five *band-limited* files (11.8-14.6 kHz) gave 17.9k — anti-correlated.
- not mel-loss HF blindness / MRD: `c_mel=45` and the discriminator are identical in all
  four cells, so neither can produce a corpus-conditional split. **This refutes the
  earlier MRD suspicion for this gap** — worth knowing before re-running that experiment.

What it does track is how much *not-already-fit* material is in the corpus, monotonically:
0 new files → 14.6-15.0k · 1 file x3 (st6) → 14.9k · earlier 4-file batch → 15.5k at 2k
steps, 17.9k at 10k · 6 files x1 → 17.9k · 6 files x3 (st3) → 22.1k. Candidate mechanism
(hypothesis, not established): data the model has already converged on yields little
useful gradient, so the decoder's output distribution narrows; fresh material sustains it.
st2→st2b shows steps substitute for volume within a fixed corpus, consistent with
"sufficient effective learning signal" rather than volume alone.

**Correcting the earlier framing in this file.** The recipe is a co-equal,
previously-unattributed cause — not the dominant one. Against the *best* new-data run the
split is 2.3 dB recipe (seed→ctl) vs 2.2 dB data (ctl→st4), a dead heat; against the
others data dominates outright (delta from ctl: st2 1.7, st2b 1.8, st6 3.0, st5 6.0, st3
7.8 dB). And the 6-8k loss is **incurred early and then plateaus** — st2→st2b at 5x steps
left it flat (-17.2 → -17.3) — so it is not damage that accumulates with training.

**Conclusion.** Two real, separately-scoped findings: (1) fine-tuning this converged seed
at 1e-4 drifts it ~2.3 dB in 6-8k and costs all of the aperiodicity regression, avoidable
with 2e-5 — but only in the zero-fresh-data case; (2) this new material costs 4.2-4.5 dB
of 6-8k regardless of LR, while being the only thing that holds `fmax` up. Nothing tested
beats `v3 ep135` overall. Closest challenger is `lowlr` (within 0.8 dB on 6-8k, level on
8-12k, better on aper 2-6k; loses on `fmax` and identity). Tracked as `AV-6sxy`.

**Caveats on all of the above.** Every number comes from one 20-second clip (hero20) and
no per-epoch noise floor was established, so deltas under ~1 dB should not be trusted as
signal — and `v3 ep135` was itself *selected* as the best of a noisy per-epoch search, so
some regression toward the mean is expected in any continuation from it. Nobody has
listened to any render.

### Noise floor and replication (2026-09-05, later session)

The caveats above were addressed directly: every retained intermediate checkpoint of the
two no-fresh-data 2x2 arms was rendered on hero20 and scored on the same basis, and the
headline ordering was re-rendered on three clips it had never been measured on.

**A1 — ctl (LR 1e-4, no fresh data) per-epoch trajectory, 11 checkpoints:**

| epoch | 6-8k | 8-12k | fmax | aper 2-6k | aper 6-12k | identity |
|---|---|---|---|---|---|---|
| seed (G_135) | -13.2 | -32.2 | 17.9k | 0.664 | 0.835 | 0.929 |
| ep14 | -13.7 | -31.7 | 17.9k | 0.673 | 0.836 | 0.917 |
| ep28 | -13.3 | -33.1 | 15.0k | 0.677 | 0.832 | 0.913 |
| ep41 | -14.1 | -33.3 | 17.9k | 0.636 | 0.818 | 0.921 |
| ep55 | -15.4 | -33.1 | 15.2k | 0.649 | 0.813 | 0.923 |
| ep69 | -16.8 | -35.5 | 14.9k | 0.612 | 0.810 | 0.923 |
| ep82 | -13.6 | -31.3 | 15.2k | 0.678 | 0.830 | 0.922 |
| ep96 | -13.8 | -32.0 | 16.9k | 0.658 | 0.822 | 0.922 |
| ep109 | -16.1 | -33.5 | 14.9k | 0.612 | 0.800 | 0.924 |
| ep123 | -15.2 | -33.3 | 14.5k | 0.637 | 0.810 | 0.926 |
| ep137 | -13.6 | -32.9 | 14.6k | 0.650 | 0.822 | 0.920 |
| ep143 | -15.6 | -34.9 | 14.6k | 0.607 | 0.806 | 0.925 |

Shape: **oscillation, not early shock and not monotonic drift.** 6-8k swings between
-13.3 and -16.8 with no trend; ep14 sits at -13.7, essentially the seed's -13.2, so the
loss is *not* front-loaded — a warmup schedule has nothing to absorb, and warmup was
**not tested** (its trigger condition failed). Adjacent retained epochs swing 6-8k by up
to **3.2 dB** (mean 1.4; 8-12k up to 4.2, mean 1.3). `fmax` flickers between two basins
(14.5-15.2k vs 16.9-17.9k) from epoch to epoch with no trend. The endpoint the 2x2
happened to land on (ep143) is a simultaneous trough on 6-8k, fmax, and both aperiodicity
bands.

**A2 — lowlr (LR 2e-5, no fresh data), 8 retained checkpoints:**

| epoch | 6-8k | 8-12k | fmax | aper 2-6k | aper 6-12k | identity |
|---|---|---|---|---|---|---|
| ep55 | -14.1 | -33.3 | 17.9k | 0.671 | 0.829 | 0.917 |
| ep69 | -14.1 | -33.0 | 17.7k | 0.657 | 0.822 | 0.929 |
| ep82 | -14.8 | -32.9 | 16.9k | 0.641 | 0.813 | 0.924 |
| ep96 | -14.4 | -32.7 | 17.9k | 0.658 | 0.823 | 0.929 |
| ep109 | -13.8 | -32.1 | 17.9k | 0.659 | 0.829 | 0.921 |
| ep123 | -14.6 | -32.5 | 15.2k | 0.648 | 0.817 | 0.926 |
| ep137 | -13.5 | -31.6 | 17.9k | 0.673 | 0.835 | 0.914 |
| ep143 | -14.5 | -32.7 | 15.0k | 0.656 | 0.820 | 0.918 |

At 2e-5 the same oscillation exists but is 3x tighter: adjacent swings <= **1.1 dB**
6-8k (mean 0.7), <= 1.1 dB 8-12k (mean 0.5), fmax >= 17.7k in 5 of 8 epochs (vs 2 of 11
at 1e-4). The LR-1e-4 arm's extra motion, not a systematic LR penalty, is what the 2x2's
single endpoint measured.

**The noise floor, stated.** Adjacent retained epochs of the *same run* swing 6-8k by up
to 3.2 dB (mean 1.4 dB) at LR 1e-4 and up to 1.1 dB (mean 0.7 dB) at 2e-5; a re-render of
the identical G_143 differs by 0.1-0.5 dB. **Deltas under ~1.5 dB between single
checkpoints are noise at either LR**, including across arms. Consequently the 2x2's
"1.5 dB LR effect" (ctl -15.5 vs lowlr -14.0) is **endpoint luck, not signal**: the two
full trajectories average -14.65 (ctl) vs -14.25 (lowlr) — 0.4 dB apart, inside noise.
Both arms' *means* sit ~1 dB below the seed's -13.2, so a small real continued-training
cost may exist but no LR conclusion survives the noise floor.

**A3 — replication on three clips the study never measured** (20s excerpts: hero
source resampled 44.1k mono, conor vocal, full-song 60-80s; five checkpoints each):

| 6-8k (dB) | seed | ctl | lowlr | st4 | st4lr |
|---|---|---|---|---|---|
| clip_hero | -11.6 | -13.0 | -12.3 | -14.3 | -17.4 |
| clip_conor | -17.3 | -21.2 | -20.9 | -27.0 | -28.7 |
| clip_fullsong | -9.4 | -11.0 | -10.2 | -10.1 | -10.4 |

| fmax | seed | ctl | lowlr | st4 | st4lr |
|---|---|---|---|---|---|
| clip_hero | 22.1k | 15.1k | 14.6k | 17.9k | 14.7k |
| clip_conor | 16.7k | 16.7k | 16.7k | 16.7k | 16.7k |
| clip_fullsong | 22.1k | 18.0k | 18.0k | 18.0k | 22.1k |

Identity (vs singing centroid) stays 0.79-0.93 across all 15 renders; aperiodicity
follows 6-8k as before. Readouts:
- **`ctl < lowlr < seed` on 6-8k held on all three clips** — but every inter-arm delta
  is 0.3-0.8 dB, inside the noise floor above. Direction replicates; the 2x2's *sizes*
  do not.
- **The fresh-data 6-8k cost is clip-dependent, not universal.** st4lr-lowlr = -5.1 dB
  on clip_hero, ~-7 dB on clip_conor, but **+0.2/-0.2 dB on clip_fullsong** (st4 -10.1
  actually beats ctl -11.0 there). The "this new material costs 4.2 dB" headline is an
  average over clips that respond differently; on some material it costs nothing.
- **`fmax` does not replicate as a checkpoint property.** On clip_hero the seed holds
  22.1k while st4lr falls to 14.7k; on clip_conor all five arms are pinned at 16.7k;
  on clip_fullsong every arm sits at 18k+. A large share of what the 2x2 attributed to
  training arms is controlled by the *source clip's own spectrum* (16.7k = conor clip's
  apparent ceiling; 22.1k = hero/fullsong material). The corpus-volume trend in the 2x2
  stands only *on hero20*; treat any single-clip fmax comparison across clips as
  uninterpretable.

Revised conclusion: the 2x2's per-arm magnitudes were amplified by endpoint luck and
single-clip fmax aliasing, but its direction survives — the seed remains the best
overall; `lowlr` remains the closest challenger; the fresh-data cost is real but
clip-dependent. The warmup lever is closed (trigger failed: no early shock). Nobody has
listened to any render.

### Fresh-data quantity boundary and checkpoint averaging (2026-09-05, same later session)

**Phase C — 2-file fresh-data run (`st2f`, LR 2e-5, 5000 steps, seed G_135/D_135).**
Corpus: the 79 originals plus exactly 2 of the 6 `sing_b4` files (`sing_b4_1_r1`,
band-limited, and `sing_b4_6_r1`, full-band). If the fmax volume trend were already
satisfied by any fresh data, 2 files would hold the ceiling. It does not:

| st2f (2 files) | 6-8k | 8-12k | fmax | aper 2-6k | aper 6-12k | identity |
|---|---|---|---|---|---|---|
| hero20 | -19.8 | -34.6 | **14.6k** | 0.551 | 0.753 | 0.911 |
| clip_hero | -16.5 | -32.4 | 16.9k | 0.612 | 0.761 | 0.896 |
| clip_conor | -24.8 | -35.7 | 16.7k | 0.595 | 0.749 | 0.822 |
| clip_fullsong | -11.8 | -29.1 | 18.0k | 0.704 | 0.827 | 0.845 |

**The minimum fresh volume that holds `fmax` is between 2 and 6 files.** st2f's hero20
fmax is 14.6k — squarely in the no-data cluster (ctl 14.6k, lowlr 15.0k), not st4's
17.9k. Not bisected further, per plan. Two second-order readings:
- **Less fresh data costs MORE brightness, not less**: st2f's 6-8k (-19.8) is worse
  than st4lr's 6-file value (-18.2) on the same clip, LR, and steps. The fresh-data
  6-8k cost is not a fixed "you touched new material" penalty; it scales with how
  *dilute* the fresh signal is.
- **fmax does not flip to the no-data basin on every clip** (clip_hero 16.9k, fullsong
  18.0k), consistent with the replication section: single-clip fmax is partly
  clip-controlled. The cluster assignment above rests on hero20, the same clip the
  trend was established on.

**Phase D — checkpoint averaging (ctl G_143/137/123 [/109/96], key-wise float32 mean,
optimizer stripped, iteration 0).** Rendered on hero20 and all three replication clips:

| avg vs G_143 | hero20 6-8k | clip_hero 6-8k | clip_conor 6-8k | clip_fullsong 6-8k |
|---|---|---|---|---|
| ctl G_143 | -15.6 | -13.0 | -21.2 | -11.0 |
| ctl_avg3 | -15.5 | -12.7 | -20.3 | -10.4 |
| ctl_avg5 | -15.6 | -12.8 | -21.0 | -10.8 |
| seed | -13.2 | -11.6 | -17.3 | -9.4 |

**Averaging doesn't move it.** Every avg-vs-G_143 delta is 0.1-0.9 dB — inside the
noise floor — and nothing approaches the seed. Aperiodicity and identity likewise
track G_143 (avg5 hero20: aper 0.616/0.805, identity 0.924). Averaging is not the cheap
improvement; it is a no-op on this run's 6-8k oscillation (expected: the oscillation is
not noise *around* a good solution that averaging could recover — the arm's mean itself
sits ~1 dB below the seed).

**No registry change — live model is still ep235, exactly as it was at the start of
this session.** The `v3 ep135` promote/rollback from 09-04 was not revisited or
re-decided today; whether to promote anything from today's candidates (or `G_112`
from the original run) over the current ep235 is still an open question this session
did not touch.

Candidate checkpoints (not served), each a complete `G_*`+`D_*` pair — the fork's
`train.py` silently starts from **random init** if only `G` is present, so never seed a
continuation from a G-only directory:
`v3_uv_bright/` (the seed), `ctl_recipe_only/`, `lowlr_2e5/` (best challenger),
`st4lr_2e5/`, `st3_b4/`, `st4_b4x1/`, `st6_b46only/`, `two_stage_band/`,
`st2f_2file/` (Phase C), and render-only `ctl_avg3/` + `ctl_avg5/` (Phase D, G-only
by construction — never train from them).

**These live under `data/`, which is gitignored — they are local to this machine only**,
as are `renders_hero20/` (all 10 A/B renders) and the `RECIPE_FINDINGS_20260905.md`
summary beside them. Nothing in that directory survives a fresh clone; this file and
`AV-6sxy` are the durable record.

## Brandy (`fb17af66`) — 2026-09-05 late: first listen. Two top-end losses; gate, decay and UV exonerated

**The first time anybody listened to a render.** The report: the voice's **quiet edges cut
off around her converted voice**, and dropouts **worst in her highest registers** — "it's
both". Both are real and they are **two distinct losses**, both at the top of her voice but
in different domains. No serving change was made.

This entry was rewritten twice before this version; the retractions are listed at the end
because each was a plausible mechanism that measurement removed.

### Measured on the served path

All numbers from the served pipeline's own isolated lead stem
(`data/conversions/5ba195c0-b439-41ec-b036-4a1cd46f8f8a/vocals.wav`, "HERO via PIPELINE #5 —
ep235", i.e. **the served weights**) against the source vocal stem, full 258 s.

**Loss 1 — spectral air and presence.** Band levels relative to each file's own 300-1000 Hz
reference, so a global gain cannot produce it:

| band | source | ep235 (served) | delta |
|---|---|---|---|
| 3-6k (presence) | -21.0 | -24.9 | **-3.9 dB** |
| 6-8k | -22.2 | -19.0 | +3.2 dB |
| 8-12k (air) | -22.7 | -31.3 | **-8.6 dB** |
| 12-16k (sheen) | -27.0 | -40.6 | **-13.6 dB** |
| `fmax` | 20.0k | 17.9k | **-2.1 kHz** |

The halo of air and sheen that sits around a voice in a mix (8-16 kHz) is 9-14 dB down and
the spectrum stops 2.1 kHz early, while 6-8k is actually *brighter* than the source. Verified
as real vocal content rather than separator bleed: 12-16k is +20.4 dB in-phrase versus
between-phrase and correlates +0.82 with her envelope (8-12k: +23.1 dB, +0.85). This is also
the same top-end deficit the 2026-09-05 programme above was circling with its 6-8k/`fmax`
metrics.

**Loss 1b — the stereo halo, removed by a live setting.** `fork_hq_stereo_width` is
explicitly **`0.0`** in `data/app_state/app_settings.json`, and the knob is a documented
runtime setting accepting 0.0-1.0 (`runtime_contract.py:86`, `api_runtime.py:482`). So this
is configuration, not a model limit. The served lead is **1 channel**:
pipeline #5's own note records "mono centred vocal (stereo width 0.0 like every approved OLT
render)". Mariah's own vocal stem is 2 channels carrying side content at **-15.7 dB**
relative to centre, and at the mix level the delivered render has **-13.7 dB** side/mid
against the original's **-11.9 dB**. So everything spatially *around* her voice is discarded
before the metrics above ever run — every one of them sums to mono first, which is why none
could see it. This is present in every approved render by construction, which fits "the
*best* conversion still has it".

**A width-0.35 render of these exact weights already exists** — `herofix-d9e90fc4`,
"HERO FULL SONG - ep235 + FIXES (edges kept, stereo 0.35)", whose own note says it was built
"for your complaints" and asks for a comparison against the serving render. The pipeline's
widening is real model decorrelation, not a pseudo-stereo trick: L and R are converted
**separately** and their difference becomes side, scaled by `stereo_width`
(`singing_conversion_pipeline.py:1648-1676`). Measured against the served render:

| | side/mid | 8-12k | 12-16k | `fmax` |
|---|---|---|---|---|
| ORIGINAL Mariah mix | -11.9 | -21.8 | -24.2 | 20.0k |
| served #5 (width 0.0) | -13.7 | -25.7 | -27.4 | 22.1k |
| ep235 + FIXES (width 0.35) | **-10.9** | -27.4 | -31.4 | 18.0k |

Width 0.35 overshoots the original's side/mid slightly and is darker up top (that render also
used `-db -35` and a different instrumental, so width is not its only difference). The point
stands: **the spatial halo is one setting away**, and a prior session already shipped a render
with it.

**Not the bandwidth-match filter.** `fork_hq_match_source_bandwidth` defaults on and
low-passes the converted vocal to the source's measured wall, which made it a suspect for the
`fmax` gap — but `_detect_bandwidth_hz` on this material returns **20000 Hz**, above the
render's own rolloff, so it is inert here. The 17.9-18k ceiling is the decoder, not this
filter.

**Which of these her words name is not yet established.** "Edges cut off around her
converted voice" fits Loss 1 (presence/air) and Loss 1b (spatial halo) equally well on
paper; attributing it without a listen would repeat the mistakes retracted below. Both are
now restorable offline and registered for comparison — see the listening set.

**Loss 2 — level collapse with pitch.** Render minus source by source F0, loud in-phrase
frames, aligned (verified 0 samples length delta, 0 ms envelope lag):

| source F0 | 200-300 | 300-400 | 400-500 | 500-600 | 600-800 | 800-1600 |
|---|---|---|---|---|---|---|
| ep235 (served) | +1.3 | +1.7 | -2.1 | **-5.6** | **-5.5** | **-10.9** |

Monotonic from 400 Hz up, reaching **-10.9 dB** on her top notes. MRD ep41 measured -7.8 dB
on the same window, but these are single renders of different checkpoints and 3.1 dB sits
inside the ~3 dB checkpoint-to-checkpoint noise floor measured this session, so no checkpoint
ranking is claimed. Cause is corpus pitch
coverage: her training corpus (10 singing files, 168,079 voiced frames, harvest
`f0_ceil=1600`) has median 292 Hz, **p99 496 Hz**, **max 844 Hz**, 0.48% of frames >= 550 Hz
and <0.01% >= 700 Hz, against a source that reaches **1031 Hz**. The collapse begins where
coverage thins (p99 496 Hz), not at any threshold or flag.

**The two losses are independent.** Her top register does *not* sit at phrase ends, so
Loss 2 cannot be what cuts the edges: phrase-final 300 ms frames are **0.8%** >= 500 Hz
(median 324 Hz) versus **4.3%** in-phrase (median 334 Hz) — the high notes sit *inside*
phrases. The user's "it's both" was correct and an earlier "one mechanism" claim here was
wrong.

### Ruled out by measurement

- **The silence gate / `db_thresh`.** Swept `-40/-45/-50/-55/-60/-65/-70/-80` relative and
  `-75/-85` absolute through the exact served command on the full vocal: **0 tail frames of
  real decaying voice zeroed at every value, including the live -40**, and 0 frames more than
  20 dB below source. Tail-minus-body measured within each file (gain-invariant) is -10.1 dB
  for the source and **-9.9 dB** for ep235: tails are not attenuated at all. `db_thresh`
  stays -40.
- **Decay truncation.** With phrase ends derived from the source and applied identically to
  every file, the 20 dB fall time is source **330 ms** (n=39) vs ep235 **360 ms** (n=44) —
  ep235 decays *slower*, and is faster than the source on 49% of offsets, a coin flip.
- **The crepe UV threshold.** Not even active in serving: the registry entry has no
  `requires_uv_contract` and no `crepe_uv_threshold`, and `svc_fork_bridge.py:113-121` only
  sets those env vars when the entry carries them. Where it *was* active, UV 0.1 and UV-off
  gave no improvement, and crepe periodicity at 800-1100 Hz is 0.717 with only 3.9% below
  0.3 — it is *confident* in her top register.
- **The `f0_max` ceiling.** `f0.py` pins 1100 Hz; the song's highest-register 20 s window
  peaks at 1031 Hz (p95 982), so it never fires. `f0_max` also feeds `f0_to_coarse`'s mel
  quantisation (`f0.py:238-239`), so raising it would reshape pitch tokens for every frame.
- **Checkpoint choice, for Loss 2.** On hero20 across all ten arms, every arm is
  flat-or-positive to 400 Hz and **every arm is 3.9-5.9 dB down at 500-600 Hz**. `st3` — 6
  fresh singing files x3, the only run that reached 22.1k `fmax` — is -4.9 dB at the top,
  among the worst, so the fresh material carried no high notes either.

### Retracted from earlier versions of this entry

1. **"`db_thresh` -40 -> -60 fixes the cut edges" (commit 275f903e).** Measured on the wrong
   path: `G_135` plus `SVCFORK_UV_CONTRACT=1` and `SVCFORK_CREPE_UV_THRESHOLD=0.3` on 20 s
   clips, none of which serving uses, and a 20 s clip is a single chunk that never exercises
   the split. Also `-db` is dB **below the clip peak**, not absolute dBFS
   (`core.py:251,254`, `ref=1 if absolute_thresh else np.max`, `absolute_thresh` default
   False and absent here), so -60 lands below the stem floor and gates nothing — the "gate
   off" case the same commit rejected. Set to -60 briefly, rolled back from
   `.pre_dbthresh60_20260905`; the live entry is byte-identical to its pre-session state.
2. **"One mechanism, not two."** Refuted by the phrase-final F0 test above.
3. **"ep235 truncates decays 30% faster than the source."** An artifact of letting each file
   define its own phrase spans (116 vs 134 phrases). The matched-window control reverses it.

### Open, tracked as `AV-41e4`

Loss 2 has no render-side knob (out-of-range F0). Cheapest untried lever is
formant-preserving pitch-shift augmentation of her existing `real_sing_sample_07x` files
(+3…+12 semitones via WORLD) — note the 09-04 "synthetic data" run was SeedVC clips
brick-walled at 13.1 kHz, a different thing. Others: the fork's `-t` transpose to move her
belts into the trained range and shift back; keeping the source's own signal above ~550 Hz
(PIPELINE #1 already has a keep-source-spans precedent); or documenting the limit. A
pitch-tracked make-up gain in the pipeline would test whether Loss 2 is level or timbre.
Loss 1 is the project's long-running brightness thread and interacts with the
`fmax`-tracks-fresh-data-volume finding above.

### The listening set (this is the instrument that settles the attribution)

GUI History tab. Nothing here changes the model or the registry; the restorations are the
delivered mix plus a measured delta, so only the tested variable differs.

- Tagged **`edges-ab`**, 40-60 s, full mixes, six entries: original Mariah / served as
  delivered / **+air** (offline EQ match above 3 kHz, +8 dB cap, 8-12k -25.7 -> -23.1) /
  **+synthesized halo** (labelled an approximation — it is *not* how the pipeline makes
  width) / **+both** / **real pipeline width 0.35** on the same ep235 weights. The last one
  is the only spatially-restored variant that is shippable as-is, because it came from the
  pipeline's own code path; the offline ones exist to isolate the variable. No retraining is
  involved in any of them.
- Tagged **`highreg`**, 168-188 s, vocals only: source / served ep235 / MRD ep41, for Loss 2.

Registration helper: `scripts/register_render.py` (argparse, `DATA_DIR`-aware, uuid5-keyed so
re-runs replace rather than duplicate, and it preserves anything you set in the GUI).
