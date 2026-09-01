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
- `multi_speaker_line_concentration_min` (`0.15`) silently rejected genuine
  harmony lines twice while the extractor was being changed. If harmonies go
  missing after any change to mask construction, check this first.

---

## Conventions

- Promote a checkpoint by **editing the registry JSON directly**, not through the
  training UI, unless the trainer produced it. `_PRESERVED_INFERENCE_KEYS` in
  `svc_fork_trainer.py` carries `f0_method`/`noise_scale`/chunking forward from
  whatever is on disk at promotion time, so a retrain promoted while the registry
  holds a wrong value will inherit it.
- Always leave a `.pre_<change>_backup` beside the entry.
- Verify via `GET /api/v1/profiles/<id>/adapters` after restarting — it reads the
  same cache the conversion path uses.
- Then update this file.
