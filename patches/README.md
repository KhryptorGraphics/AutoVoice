# Patches to the installed `so-vits-svc-fork`

Inference and training both run as a **subprocess** against the `svc` CLI in the
`svcfork` conda env (`/home/kp/anaconda3/envs/svcfork`), so these cannot live in
this repo's source tree — they patch site-packages directly. A `pip install -U`
of so-vits-svc-fork will silently revert them; `tests/test_svcfork_patches.py`
fails when that happens.

## svcfork_uv_contract.patch

`SineGen`'s docstring requires `f0 == 0` on unvoiced steps, and
`_f02uv` computes `uv = (f0 > voiced_threshold)` with `voiced_threshold=0`. But
`interpolate_f0()` fills **every** unvoiced frame, so `uv == 1` everywhere and

    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
              = 1 * 0.003     + 0 * 0.0333

is pinned at 0.003 on every frame — a **20.9 dB cut** to the aperiodic
excitation. Verified by executing the installed module: zero-free f0 gives
`noise.std() == 0.00302`; f0 with real zeros gives voiced 0.00292 vs unvoiced
0.03346. This is faithful to Xin Wang's NSF paper (arXiv:1904.12088, Eq. 13) in
every respect except the input, which the caller corrupts.

The patch masks f0 by the real `uv` **only where it feeds the decoder** (the
`f0_to_coarse` pitch embedding still gets the interpolated contour, which is
what it wants).

**Guarded, default OFF.** `SVCFORK_UV_CONTRACT=1` enables it. Off, the helper
returns `f0` unchanged, so the serving path is bit-identical to unpatched.

It is off by default because preprocessing (`preprocess_hubert_f0.py`) also
interpolates, so the model was *trained* against the violated contract.
Enabling it at inference alone is a train/test mismatch. It is intended for a
retrain that masks during training too.

Reapply after an upgrade:

    cd /home/kp/anaconda3/envs/svcfork/lib/python3.12/site-packages/so_vits_svc_fork
    cp modules/synthesizers.py modules/synthesizers.py.orig
    patch -p0 < <repo>/patches/svcfork_uv_contract.patch

## svcfork_cluster_torch_load.patch

`cluster/__init__.py:get_cluster_model` calls bare `torch.load(f,
map_location="cpu")`. Since torch 2.6 the default `weights_only` is `True`,
which refuses the plain dict of numpy arrays this checkpoint format is
(`n_features_in_`, `_n_threads`, `cluster_centers_` — see
`cluster/train_cluster.py`'s own writer). This is upstream's own file format,
not specific to any one producer of it — a genuine `svc train-cluster` run
hits the identical failure on torch 2.10 (installed here). Verified: crashed
loading a checkpoint from this session's own cluster fit before the patch.

Fixed by passing `weights_only=False` explicitly at this one load site. The
checkpoint is always produced locally (by `train-cluster` or the equivalent
script in this session), never fetched over the network, so this does not
introduce the arbitrary-code-execution risk the flag's docs warn about.

Reapply after an upgrade:

    cd /home/kp/anaconda3/envs/svcfork/lib/python3.12/site-packages/so_vits_svc_fork
    cp cluster/__init__.py cluster/__init__.py.orig
    patch -p0 < <repo>/patches/svcfork_cluster_torch_load.patch

## svcfork_numpy_fromstring.patch (remote-only, RTX 5090 gpuhub box)

`utils.py`'s `plot_spectrogram_to_numpy`/`plot_data_to_numpy` (used by
`train.py`'s periodic TensorBoard image logging) call
`np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep="")`. numpy >=2.0
removed the binary mode of `fromstring` entirely (`ValueError: The binary mode
of fromstring is removed, use frombuffer instead`). Hit on the gpuhub RTX 5090
box (numpy 2.5.2) on the first `log_interval` boundary of a from-scratch
`svc train` run — would affect ANY training run on that environment, unrelated
to this session's MRD/uv-contract changes.

Same `so-vits-svc-fork==4.2.30` version string on both machines, but the LOCAL
Thor install's `utils.py` already used `frombuffer` at this exact line - the
PyPI wheel was evidently rebuilt under an unchanged version tag at some point
between the two installs. Fixed by copying the already-correct pattern:
`np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)` (frombuffer takes
no `sep` kwarg). Not carried as a `.patch` file since it only reproduces what
a normal `pip install so-vits-svc-fork` already gives on this repo's own Thor
env - reapply by diffing against a fresh install if a from-scratch remote box
pulls the older wheel variant again.

## svcfork_crepe_periodicity_uv.patch (2026-09-04)

Target: `so_vits_svc_fork/f0.py` in the `svcfork` env. Crepe never emits
`f0 == 0`, so `interpolate_f0()` derives `uv == 1` on every frame and the
uv-contract mask above is a no-op for crepe-trained models. With
`SVCFORK_CREPE_UV_THRESHOLD=<float>` set, frames whose median-filtered crepe
periodicity is below the threshold are zeroed at pre-hubert and at inference.
Calibrated on Brandy's singing: 0.3 gives ~13.5 % unvoiced on active frames
(harvest gives 13.4 %). Unset = original behaviour. Per-model at serving time via
the registry key `crepe_uv_threshold` (see `svc_fork_bridge._clean_env`); must
match how the checkpoint was trained. Apply with
`patch <env>/site-packages/so_vits_svc_fork/f0.py < patches/svcfork_crepe_periodicity_uv.patch`.

## svcfork_mrd_discriminator.patch (rebuilt 2026-09-04)

Targets `so_vits_svc_fork/modules/descriminators.py` and `train.py`. Adds
`DiscriminatorR` (UnivNet-style STFT-magnitude sub-discriminator: 5x weight-normed
Conv2d, 32 ch) at resolutions (1024,120,600) (2048,240,1200) (512,50,240), appended
to MultiPeriodDiscriminator's own `discriminators` ModuleList so an MPD-only
`D_*.pth` (111 keys) loads key-for-key and only indices 6-8 start fresh.
`train.py` builds it when `SVCFORK_MRD=1`; unset = original fork. Conor's served
`G_197` (`svcfork_conor_fullband_20260828_mrd_uvfix`) was trained with exactly this
(its `D_197` has 165 keys and strict-loads into the rebuilt class). The original
patch was lost with the box; this copy was recovered from the session transcript
and verified against `D_197`. Apply: `patch -p0` per file, or copy the two files.

## svcfork_lora.patch (2026-09-05)

Targets `so_vits_svc_fork/modules/synthesizers.py` and `train.py`. Adds low-rank
adaptation: `inject_lora()` attaches a parallel `A -> B` Conv1d path to every
Conv1d under `dec.`, `flow.` and `enc_p.` via a **forward hook**, so each base
parameter keeps its exact state_dict key and an ordinary checkpoint still loads
key-for-key (verified: 751 -> 1109 keys, all 751 originals preserved, `safe_load`
of a non-LoRA checkpoint succeeds). `lora_B` is zero-initialised, so training
starts bit-identical to the base.

Deliberately excludes `enc_q`: the posterior encoder runs during training but
never in `SynthesizerTrn.infer`, so adapting it spends capacity the served model
cannot use. `ConvTranspose1d` (the decoder upsamplers) is skipped - its (in,out,k)
layout needs a different delta construction.

`train.py` freezes the base and narrows `optim_g` to the LoRA deltas plus
`emb_g` when `SVCFORK_LORA_RANK` is set: 2.39M of 48.0M generator params (5.0%)
at rank 16. `SVCFORK_LORA_ALPHA` defaults to `2 * rank`. Unset = original
behaviour, all parameters train.

Rationale: every full fine-tune of this ~80 min corpus degraded after ~epoch 150
(four runs, 2026-09-04/05). Constraining the update is the standard remedy.
Serving a LoRA checkpoint needs this patch present at inference too.

## svcfork_spa_cycle (2026-10-31)

Targets `so_vits_svc_fork/train.py` and installs
`so_vits_svc_fork/autovoice_spa_cycle.py` beside it. A research transfer of
SPA-SVC (arXiv 2406.05692, Bai/Wang/Gao/Li, Interspeech 2024) to this fork -
not a production fix, and the paper's own evidence is MOS-only from 13
listeners.

The mechanism, per training step:

1. Sample a semitone shift per training sample (6-18 st, capped so the
   shifted contour stays under 1100 Hz). The WHOLE interpolated F0 contour
   moves by one uniform factor, not only its voiced frames: the fork feeds
   the interpolated contour to `f0_to_coarse`, so shifting only the voiced
   frames would put a step of up to 2.8x at every voiced/unvoiced boundary
   and the pitch embedding would read that sawtooth as the singer's contour.
2. First pass: `net_g.infer` with the SHIFTED f0, `noice_scale=0`.
3. Re-extract ContentVec from that render. This needs the audio gradient, so
   it cannot go through `utils.get_content` (which wraps its model call in
   `torch.no_grad`); the helper calls the encoder directly. The encoder is
   kept on CPU and OUTSIDE the module registry - a registered child would be
   moved to the accelerator by the trainer and would appear in
   `.parameters()`.
4. Second pass: `net_g.infer` at the ORIGINAL f0 with the re-extracted
   content.
5. Cycle loss = MSE between the cycle mel and the real target mel. It is
   weighted `raw * SVCFORK_SPA_CYCLE_WEIGHT * c_mel`, so the weight has to be
   calibrated against the measured mel term or the cycle becomes the
   objective instead of a regulariser.

**Default OFF.** `SVCFORK_SPA_CYCLE_WEIGHT` unset (or 0) leaves the
objective unconstructed and the loss arithmetic exactly upstream - verified
by executing the patched block's disabled arm against the upstream
expression and comparing the tensors for equality, not by reading the source.

Env vars:

| var | default | meaning |
|---|---|---|
| `SVCFORK_SPA_CYCLE_WEIGHT` | 0 (off) | weight of the cycle term |
| `SVCFORK_SPA_F0_CEILING_HZ` | 1100 | upper bound for the shifted contour |
| `SVCFORK_SPA_MAX_SAMPLES` | 4 | cycle samples per step (memory bound) |
| `SVCFORK_SPA_ENCODER_DEVICE` | cpu | device the frozen ContentVec encoder runs on |

### Two deviations from the paper, and why

**MSE instead of SSIM.** The paper takes `1 - SSIM(mel(original), mel(cycled))`.
This uses MSE. The paper's own table has its MSE variant beating SSIM on the
metric that matters here - the large-range-gap target, 3.60 vs 3.56 (a
difference WITHIN the stated CI, so "no SSIM advantage there" rather than
"MSE wins"). SSIM also costs more per step and adds a second hyperparameter
to a run whose weight was already the delicate part.

**The encoder device is measured, not chosen.** A canary epoch took **157 s**
with the encoder on CPU and **171 s** with it on CUDA; without the cycle term
an epoch is 29 s. The GPU encoder is SLOWER here because this box's GPU
memory IS system RAM, so the encoder competes with the two extra generator
forwards for it. The encoder therefore stays on CPU by default.

### The gradient really crosses the bridge

The first generator pass reaches the loss ONLY through the ContentVec
re-extraction. If that bridge carried no gradient, the cycle loss would be a
function of the second pass alone and the F0 shift would be pointless.

The input CONTENT's own gradient is the clean proof: content feeds the
FIRST pass only, so a non-zero gradient on it can only have arrived through
the bridge. Measured on a real batch: **13.25**. The decoder also receives
gradient (all 305 decoder parameters, norm sum 1222).

(`enc_p` is NOT evidence either way: it runs in BOTH generator passes, so
its gradient does not isolate the bridge.)

The cycle term was also observed DECREASING during training (raw MSE
5.13 -> 1.97 over 199 steps) while the mel term stayed flat (27.37 ->
27.70): the objective is optimised without trading away mel fidelity, which
is what a mis-scaled weight would do.

The sample cap is not cosmetic: with Brandy's corpus nearly every sample has
positive headroom, so an uncapped cycle path would run two extra generator
forwards with the graph retained for the whole batch of 48.

Apply:

    python patches/svcfork_spa_cycle.py \
        /home/kp/anaconda3/envs/svcfork/lib/python3.12/site-packages/so_vits_svc_fork/train.py

The patcher compiles the patched source before writing it, and on a re-run it
compares the installed helper BYTE FOR BYTE with
`patches/svcfork_spa_cycle_runtime.py` and refreshes it when they differ. A
changed helper would otherwise leave the installed fork training different
semantics than this repo describes, with no error anywhere.
