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
