---
name: earshot-conversion-ears
description: Use whenever judging an AutoVoice voice conversion by measurement instead of by guesswork - is this render band-limited, did a fork_hq_* setting help or hurt, is the mix at a sane loudness/true peak, does a render fit the corpus of approved renders, and what does the spectrogram actually show. Covers earshot measure/sheet/ab/compare/realness on data/conversions, data/samples and data/fork_models renders.
---

# Listening to AutoVoice conversions

You cannot hear. This project has ~40 `fork_hq_*` conversion settings and no measurement of the rendered output at
all: `src/` has no LUFS, no true-peak, no output-clipping and no bandwidth meter on a conversion result
(`grep -E "lufs|pyloudnorm|true_peak|dbtp" src/` returns nothing). `data/conversions/<job_id>/` holds only WAVs.
So a render is judged by ear, and "it sounds brighter" is a guess.

earshot gives you the measurements. The server is registered in both harnesses as `earshot` (16 tools: `measure`,
`describe`, `match`, `rank`, `find`, `compare`, `ab`, `realness`, `timeline`, `sheet`, `critique`, `doctor`,
`llm_critique`, `sfx_list`, `sfx_render`, `kit_render`). Always pass absolute paths. `measure` needs no models;
everything else loads CLAP + AST.

**Before quoting a number, read the trust rules at the bottom.** Two of earshot's outputs look authoritative and are
not, and one AutoVoice quirk silently undoes the change you are trying to evaluate.

## 1. The band-limit authority is `effective_fmax`, not earshot's `bandwidth_hz`

`effective_fmax` (`src/auto_voice/inference/fork_voice_polish.py:367`) finds the **cliff**: the steepest drop in
dB/octave above 3 kHz, requiring `min_drop_db=18.0`. It returns `sr/2` when nothing walls. This is the number that
decides "is this recording band-limited".

earshot's `measure` → `bandwidth_hz` is *"highest 1/3-octave band within 60 dB of the loudest band"*
(`earshot/features.py:220-228`) — a 60 dB-relaxed **level threshold**, not a cliff. It over-reports and saturates at
the 1/3-octave grid. Measured on the same files:

| file | `effective_fmax` (authority) | earshot `bandwidth_hz` |
|---|---|---|
| `data/samples/a24de6f2-…/sample_003/vocals.wav` | **6034 Hz** | 8127 Hz |
| `data/conversions/917dda7d-…/vocals.wav` | **6939 Hz** | 8127 Hz |
| `data/conversions/917dda7d-…/vocals_bwe.wav` | **22050 Hz** | 16254 Hz |

Use `effective_fmax` for "is it band-limited". Use earshot's `bands_db` (`air`, `presence`, `top`), `rolloff95_hz`
and `brightness_hz` for HF *balance*. Never feed earshot's `bandwidth_hz` into a decision or a threshold.

To compute the authority yourself (the earshot venv can import AutoVoice — it inherits torch/librosa):

```bash
cd /home/kp/thordrive/autovoice && PYTHONPATH=src PYTHONNOUSERSITE=1 \
  /home/kp/tools/earshot/.venv/bin/python -c "
import numpy as np, soundfile as sf
from auto_voice.inference.fork_voice_polish import effective_fmax
y, sr = sf.read('<file.wav>', dtype='float32')
if y.ndim > 1: y = y.mean(axis=1)
print('effective_fmax =', round(effective_fmax(y[:int(sr*60)], sr)), 'Hz')"
```

## 2. Never optimise "air" on its own — half of it is noise

`extend_bandwidth` copies one octave of source material up, blends it with random noise at `noise_mix=0.5`, and
randomises the phase above the wall (`fork_voice_polish.py:470-478`). The result *measures* as more air whether or
not it is more content.

Measured on `data/conversions/917dda7d-…/`, `vocals.wav` → `vocals_bwe.wav`:

| | vocals.wav | vocals_bwe.wav | Δ |
|---|---|---|---|
| air band (8–16 kHz) | −55.6 dB | **−36.4 dB** | +19.2 dB |
| `harmonic_ratio` | 0.718 | **0.718** | unchanged |
| `tonal_peak_ratio_db` | 83.2 dB | **60.3 dB** | −23 dB (noisier spectrum) |
| `noise_floor_dbfs` | −92.4 dBFS | **−88.4 dBFS** | +4 dB (a hiss bed) |

Identical harmonic share, 19 dB more "air", 23 dB less tonal dominance and a 4 dB higher noise floor = a noise bed,
not brightness.

**Always report `harmonic_ratio`, `tonal_peak_ratio_db` and `noise_floor_dbfs` next to any air-band gain.** The target
is a *band*: air up, `harmonic_ratio` and `tonal_peak_ratio_db` not down, noise floor not up. A knob whose only
measured effect is "+dB of air" is adding hiss.

## 3. Stage ordering can erase the extension you are evaluating

`extend_bandwidth` runs inside `_correct_fork_output` (`singing_conversion_pipeline.py:1920`, called at `:2515`).
`fork_hq_match_source_bandwidth` — **default `True`** (`:2585`) — low-passes at `:2601-2603`, i.e. **after** it.

Measured in that order on a real render:

```
converted vocal wall              =  6939 Hz
after extend_bandwidth            = 22050 Hz   <- the +19 dB
after match_source_bandwidth@7000 =  7980 Hz   <- cut back
```

So whenever the source mix walls below 16 kHz — i.e. every band-limited capture such as Patrick's Bluetooth-mic
recordings — the match filter re-cuts the band the extension just synthesised, and the served output shows none of
it. The extension only survives when the source is full-band.

Before crediting or blaming `extend_bandwidth`, check the job's own metadata in
`data/app_state/conversion_history.json` → `conversion_metadata`: `source_bandwidth_hz` and `vocal_bandwidth_hz`.
Job `917dda7d` shows `20000.0` / `20000.0`, which is why that job escaped it. A job with `source_bandwidth_hz`
near 8000 did not.

## 4. The listening loop for one conversion

Run in this order; stop early if `measure` already answers the question.

1. **`measure` the output.** `vocals.wav` and `mix.wav` in `data/conversions/<job_id>/`:
   `lufs`, `true_peak_dbtp`, `lra_lu`, `noise_floor_dbfs`, `clipped_samples`, `dynamic_range_db`,
   `bands_db` (sub/low/lowmid/mid/presence/air/top), `harmonic_ratio`, `tonal_peak_ratio_db`, `brightness_hz`,
   `rolloff95_hz`, `noise_floor_dbfs`. Add `music=True` for tempo/beats/key on the mix, and `light=True` to skip
   the slow parts.
2. **`sheet` it, and open the PNG.** `sheet(path, out_png="/tmp/<name>.png")` returns the path *and* an inline image
   you can read directly. The spectrogram lane is where a band limit is unmistakable: a hard horizontal ceiling
   with pure black above it, across the whole file. It also shows dropouts, the loudness curve, gaps and the beat
   grid. `sheet` takes ~90 s on a 3-minute file — pass `detail="low"` when you only need the picture.
3. **`compare` against approved renders.** `compare(path, refs="<dir>")` returns the nearest references, a
   typicality z, and a per-feature `robust_z` table (loudness, peak, brightness, bandwidth, attack, decay, length,
   noise floor, width, low end). Use `data/fork_models/_candidates_fb17af66_20260904/renders_hero20/` (33 renders,
   different settings) as the existing sweep corpus, and build a refs dir of the renders you already approve.
4. **`ab` two renders.** `ab(a, b, query=..., refs=...)` says which is closer and how they differ in measured words
   ("louder", "brighter", "longer decay"). Use it for a before/after of one setting change, not to pick a winner.
5. **`realness`** as a regression guard: a change that drops `p_real` makes the render read more synthetic.

`data/conversions` holds 732 job directories. `data/samples/<profile_id>/sample_*/vocals.wav` holds training
samples; `data/fork_models/<profile_id>*/renders_*/*.wav` holds candidate renders.

For the model-backed tools, the first call in a fresh process loads CLAP + AST (~20–60 s). The MCP server is
registered with `--warm`, so it preloads in the background at startup.

## 5. Trust rules

- **`measure` is exact.** BS.1770 meter, true peak (4× oversampled), spectrum, envelope. Use it as ground truth.
- **`compare`'s `robust_z` is a MAD z, not sigma.** `scale = 1.4826 * MAD`, floored per key by `MIN_SCALE`
  (`earshot/compare.py:200-207`). With 2–6 refs it means "candidate vs the refs' median, in MAD units". Rank
  candidates on `feature_distance` (the median |robust_z| across features) plus the per-feature medians — not on
  typicality z alone.
- **`typicality.z` is near-blind to HF changes.** It compares CLAP k-NN cosines, and the cosine between two AutoVoice
  renders differing by **19 dB of air band** measured **0.96**. Treat it as a timbral-similarity check only.
- **Never present `match` / `describe` on a singing render as a quality gate.** earshot's own vocabulary marks the
  whole `voice` category `UNVALIDATED` (`earshot/vocab.py:49`), and every singing-quality adjective measured `no
  match` on AutoVoice renders — `'a thin sound'` z −1.1, `'an airy sound'` z +0.6, `'a muffled sound'` z +0.7, whole
  file and on 3-second trims. The description is always `'singing vocals'`. Use `measure`, never the vocabulary, to
  judge a voice.
- **One render is one draw.** The fork renders at `noise_scale=0.4`, so renders are stochastic, and no setting in
  `renders_hero20/` is repeated on the same source clip. Any z-score from a single render is descriptive, not a
  result. `experiments/score_multi.py::_verdict` implements the right test (disjoint ranges + exact rank-sum
  p-value) and needs ≥2 renders per setting; `CONTINUATION_PROMPT.md` asks for ≥4.
- **The models hear only to 14 kHz** (CLAP) and 8 kHz (AudioSet). Band-limit questions go to `measure` and
  `effective_fmax`, never to the models.
- **Keep the candidate out of the refs directory.** `compare` excludes only the candidate's own exact path, so a
  near-duplicate in `refs` reads as "fits the set".
