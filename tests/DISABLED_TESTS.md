# Disabled test modules

These are renamed `*.py.disabled` so pytest does not collect them. Each imports
something that exists in **no commit and no checkout**, so it fails at import
and takes the whole suite's collection down with it — `pytest tests/` aborts
before running anything rather than reporting a normal failure.

They are kept, not deleted: each encodes a real contract someone intended, and
the content is the clearest specification of what that feature should do.

| file | missing symbol | what it wanted |
|---|---|---|
| `test_fork_metrics_gpu_and_steps.py.disabled` | `svc_fork_metrics._GPU_ALLOC_TAGS` | GPU allocation accounting during training |
| `test_shortcut_flow_matching.py.disabled` | top-level `modules` package | the Seed-VC shortcut flow lane |

To revive one: implement the symbol it imports (the test body is the spec),
then drop the `.disabled` suffix and run it.

`test_fork_registry_preserves_tuning.py` was revived this way and is now
enabled: `_PRESERVED_INFERENCE_KEYS` is implemented, so a retrain no longer
silently reverts serving-side tuning. Worth noting how it was written — the
tests reimplement the promotion block locally rather than calling
`train_svc_fork`, which is fast but cannot catch a fault in the shipped code.
It didn't: a `logger.info` was added to the real function while the module had
no logger, and every test still passed. `TestTheRealFunctionNotACopy` closes
that gap.

`test_svc_fork_env_contract.py` was a fourth case of this kind and is now
**resolved**: the feature it specified was implemented rather than disabled,
because it documents a measured failure worth guarding — `expandable_segments:True`
grew the allocator reserve from 4 GB to 91 GB over 162 steps on Thor while live
tensors stayed flat, degrading training from ~1 s/step to 180 s/step. On Jetson
GPU memory is system RAM, so that drifts the whole box toward OOM.
`svc_fork_trainer._clean_env` now pins `PYTORCH_CUDA_ALLOC_CONF` instead of
inheriting whatever launched gunicorn.
