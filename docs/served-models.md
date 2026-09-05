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
- **Not a bandwidth-extension problem**: 5 of 6 new files are band-limited (11.8-14.6 kHz);
  the 1 full-band file (`b4_6`, 22.1 kHz) alone (st6) still lost on every axis, including
  its own bandwidth — the model didn't even fully learn that file's ceiling at 5k steps.
- **Not simply a "need more fidelity steps" problem**: st2 (≈2k steps) → st2b (10k
  steps, 5x more) recovered `fmax` and identity to exact parity with the baseline
  (17.9k/0.929 both ways) but left the 6-8k deficit essentially unchanged (-17.2 →
  -17.3). More steps fixes bandwidth and identity here, not brightness — a narrower,
  more specific problem than plain undertraining. st5's fidelity tail (on st4, a
  different corpus) underperformed outright, for the speech-dominance reason below.
  The "full-band" subset in this corpus is ~60% speech by duration, so a fidelity tail
  re-anchors toward *speech* spectra, not singing — the tail direction was wrong for
  that corpus, not just under-trained.

### The control run that reattributes the cause (run after the six above)

Every one of the six runs changed *two* things at once versus the seed: it added new
data **and** it continued training. Nobody had run the control that separates them —
continue training from `v3 ep135` on **exactly the data `v3 ep135` was already trained
on** (`corpus_v3`: 071/072/075 x3 + 074 x1 + speech x1, 79 files, verified to contain
zero new material), with **exactly the same recipe** (crepe f0, UV contract, LR 1e-4,
batch 16, 5000 steps). Same hero20 render, same scorecard:

| run | data vs seed | 6-8k dB | fmax | identity |
|---|---|---|---|---|
| **v3 ep135 (seed)** | — | **-13.2** | **17.9k** | **0.929** |
| **ctl (control)** | **none — identical corpus** | **-15.5** | **14.6k** | 0.921 |
| st4 (best new-data run) | + 6 new videos x1 | -17.7 | 17.9k | 0.925 |

**The control degraded on its own.** With zero new data, 5000 steps of this recipe cost
2.3 dB of 6-8k and 3.3 kHz of `fmax` — a *larger* bandwidth loss than st4's. So roughly
half of st4's 6-8k deficit (-4.5 dB vs seed) is reproduced by the recipe alone, and the
seed's `fmax` is not even preserved by training on its own data.

**This corrects the attribution above.** The dominant cause is not the new data; it is
that continuing to train from this converged seed at this recipe walks away from it.
New data adds a further penalty on top (st4 -17.7 vs ctl -15.5, ≈2.2 dB attributable to
data), but it is the secondary effect, not the primary one. It also explains cleanly why
more steps never recovered brightness (st2/st2b): more steps means more of whatever is
doing the damage.

Two mechanism candidates, both consistent with the numbers and worth testing before any
further data work:
- **LR too high for fine-tuning a converged checkpoint.** Flat 1e-4 with
  `lr_decay=0.999875` is nearly constant. Note `warmup_epochs` and `init_lr_ratio` exist
  in the config but this fork's `train.py` never reads them — warmup is *not*
  implemented, so a lower flat LR is the implementable equivalent.
- **The mel loss barely sees 6-8k.** `c_mel=45` dominates the loss, and the mel scale
  compresses high frequencies into few bins, so that band is constrained mostly by the
  discriminator. That is the same gap the 09-04 MRD (multi-resolution discriminator)
  experiment was aimed at.

### LR was the cause — confirmed (run after the control)

Re-ran the control corpus unchanged, with the single variable being peak LR: 2e-5
instead of 1e-4 (5x lower). `warmup_epochs` is vestigial in this fork so a lower flat
LR is the implementable equivalent. Verified the arm actually took effect by reading the
runtime optimizer state out of the saved checkpoint —
`optimizer.param_groups[0]["lr"] = 1.9967e-05`, i.e. 2e-5 after `ExponentialLR`'s
`gamma=0.999875` over 490 steps — not just the config value.

| run | LR | 6-8k | 8-12k | fmax | aper 2-6k | aper 6-12k | identity |
|---|---|---|---|---|---|---|---|
| **v3 ep135 (seed)** | — | **-13.2** | -32.2 | **17.9k** | 0.664 | 0.835 | **0.929** |
| ctl | 1e-4 | -15.5 | -34.8 | 14.6k | 0.599 | 0.799 | 0.921 |
| **lowlr** | **2e-5** | **-14.0** | **-32.4** | 15.0k | **0.675** | **0.827** | 0.922 |

Dropping the LR 5x recovers most of the damage: 6-8k regains 1.5 of the 2.3 dB lost
(now within 0.8 dB of the seed), 8-12k returns to the seed's level (-32.4 vs -32.2),
and **both aperiodicity bands recover fully** — 2-6k actually exceeds the seed
(0.675 vs 0.664). Aperiodicity is the buzz metric this whole line of work started from,
so that is the headline: the buzz signature was substantially self-inflicted by the
fine-tuning LR.

Two things do **not** recover with LR alone and are therefore a separate mechanism:
`fmax` (15.0k vs the seed's 17.9k) and identity (0.922 vs 0.929). The mel-loss
HF-blindness candidate above is the obvious suspect for the bandwidth ceiling and is
still untested.

### The data re-test at the corrected LR — data is worse than first measured

Ran st4's exact 88-file corpus at LR 2e-5, same 5000-step/100-epoch horizon as st4, so
LR is the only difference from st4 and data is the only difference from `lowlr`:

| run | LR | new data | 6-8k | 8-12k | fmax | aper 2-6k | aper 6-12k | identity |
|---|---|---|---|---|---|---|---|---|
| **v3 ep135 (seed)** | — | — | **-13.2** | -32.2 | **17.9k** | 0.664 | **0.835** | **0.929** |
| ctl | 1e-4 | none | -15.5 | -34.8 | 14.6k | 0.599 | 0.799 | 0.921 |
| **lowlr** | **2e-5** | **none** | **-14.0** | **-32.4** | 15.0k | **0.675** | 0.827 | 0.922 |
| st4 | 1e-4 | +6 videos | -17.7 | -33.7 | **17.9k** | 0.549 | 0.786 | 0.925 |
| st4lr | 2e-5 | +6 videos | -18.2 | -33.7 | **17.9k** | 0.555 | 0.753 | 0.919 |

Isolating the data penalty at each LR:
- at **1e-4**: ctl -15.5 → st4 -17.7 = **2.2 dB**
- at **2e-5**: lowlr -14.0 → st4lr -18.2 = **4.2 dB**

**The data penalty nearly doubles once the recipe is fixed.** The broken LR was doing
enough damage of its own to partly mask the data's contribution; removing it reveals the
data cost as larger, not smaller. Aperiodicity says the same: st4lr's 6-12k drops to
0.753, its worst value anywhere in this table. So the earlier "adding this material
hurts" conclusion was correct in direction and *understated* in magnitude — this
re-test closes that question rather than reopening it.

One thing the new data does buy, and LR cannot: **`fmax` 17.9k**, matching the seed,
where both no-data runs top out at 14.6-15.0k. So the two factors act on different axes
— LR governs brightness and aperiodicity, the new material governs the bandwidth
ceiling — which is why no single-lever run has beaten the seed on everything at once.

**Conclusion**: two independent, quantified findings. (1) The recipe was genuinely
broken: LR ~5x too high for fine-tuning this converged seed, costing 2.3 dB of 6-8k and
all of the aperiodicity regression, fixable by dropping to 2e-5. (2) The new material
genuinely hurts, by 4.2 dB of 6-8k at the corrected LR, while being the only thing that
restores `fmax`. Nothing tested beats `v3 ep135` overall; `lowlr` is the closest
challenger (within 0.8 dB on 6-8k, level on 8-12k, better on aper 2-6k) and loses only
on `fmax` and identity. Tracked as `AV-6sxy`.

**No registry change — live model is still ep235, exactly as it was at the start of
this session.** The `v3 ep135` promote/rollback from 09-04 was not revisited or
re-decided today; whether to promote anything from today's candidates (or `G_112`
from the original run) over the current ep235 is still an open question this session
did not touch. Candidate checkpoints (not served) under
`data/fork_models/_candidates_fb17af66_20260904/` in
`v3_uv_bright/`, `st3_b4/`, `st4_b4x1/`, `st6_b46only/`, `two_stage_band/`.
