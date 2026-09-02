# Remote training environment (gpuhub / AutoDL RTX 5090 boxes)

Rebuilds the so-vits-svc-fork training environment on a fresh cloud box without
repeating the slow parts. Built after paying the ~15-minute from-scratch install
three times.

## Use

```bash
scp -P <port> -r scripts/remote_train_env root@<host>:/root/autodl-tmp/
ssh -p <port> root@<host> 'bash /root/autodl-tmp/remote_train_env/setup_remote_env.sh'
```

`scp` takes `-P` for the port, `ssh` takes `-p`. Mixing them makes scp treat the
port number as a local filename and fail with `stat local "<port>"`.

## What this actually saves

Only **one** dependency has no upstream wheel: `pyworld` (the WORLD vocoder),
which compiles from source and dominates the install. `wheels/` holds a prebuilt
`pyworld-0.3.5-cp310-cp310-linux_x86_64.whl`, so `pip` uses it instead of
compiling. Everything else already ships wheels and installs at CDN speed.

`requirements-gpuhub-lock.txt` is a `pip freeze` (118 packages) of a known-good
environment. The setup script does NOT install from it - it installs the pins
that matter and lets pip resolve the rest. Use the lock only when you need to
reproduce that exact environment.

## Wheel compatibility

`wheels/` is **cp310 / linux_x86_64**. On a box with a different Python or
architecture pip silently ignores it and compiles from source - correct, just
slow again. Check `python3 -V` first; these boxes ship Python 3.10.

## The pins, and why

- **`torchaudio==2.8.0`** - 2.9+ delegates decoding to torchcodec and
  `svc pre-hubert` dies with `ImportError: TorchCodec is required`. Installing
  torchcodec does not fix it: its `.so` wants a CUDA runtime lib that is absent.
- **cu128** - Blackwell (RTX 5090) needs `sm_120`; verify with
  `torch.cuda.get_arch_list()`.
- **fp32 at training time** - not set here, but `fp16_run` hits an experimental
  ComplexHalf STFT path that stalls the GPU to near-zero utilisation. svc-fork
  has no bf16 path at all.

## This is not a full snapshot

It removes the compile, not the ~3 GB torch download. The complete fix is the
provider's own **"save image"** feature in the web console, which captures the
whole disk so a new instance boots ready. That is a console action - it cannot
be triggered over SSH, so it has to be done by hand once.

Do not try to keep the environment on the box between runs: `/root/autodl-tmp`
is instance-local and disappears when the instance is released. The box used for
the William Singe run was deallocated the moment training finished.

## Related

- `docs/served-models.md` - what each profile serves and why
- Training recipe, measured rates, and traps: the `gpuhub-5090-remote-training`
  note in the project memory directory.
