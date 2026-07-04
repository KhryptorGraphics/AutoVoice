#!/usr/bin/env bash
# Create the isolated `svcfork` conda env used by the so-vits-svc-fork engine
# (serving: auto_voice.inference.svc_fork_bridge; training: svc_fork_trainer).
#
# Reproduces the env by hand-built recipe:
#  1. clone the working autovoice-thor env (keeps the proven Jetson CUDA torch;
#     pyworld / praat-parselmouth / torchcrepe already build there),
#  2. pip install so-vits-svc-fork,
#  3. install a sitecustomize shim routing torchaudio.load/save through
#     soundfile -- torchaudio 2.10 otherwise dispatches to TorchCodec, which is
#     unavailable for the FFmpeg on this box and breaks every `svc` call.
#
# Idempotent. Override env names via AUTOVOICE_BASE_ENV / AUTOVOICE_SVCFORK_ENV.
set -euo pipefail

BASE_ENV="${AUTOVOICE_BASE_ENV:-autovoice-thor}"
FORK_ENV="${AUTOVOICE_SVCFORK_ENV:-svcfork}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "/${FORK_ENV}\$"; then
  echo "Cloning $BASE_ENV -> $FORK_ENV (preserves CUDA torch)..."
  conda create -y --clone "$BASE_ENV" -n "$FORK_ENV"
else
  echo "Env $FORK_ENV already exists; ensuring fork + shim are present."
fi

PY="$CONDA_ROOT/envs/$FORK_ENV/bin/python"
"$PY" -m pip install "so-vits-svc-fork"

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SITE/sitecustomize.py" <<'PY'
"""Route torchaudio.load/save through soundfile (torchaudio 2.10 needs TorchCodec)."""
try:
    import torch
    import torchaudio
    import soundfile as sf

    def _load(filepath, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, format=None, buffer_size=4096, backend=None):
        start = int(frame_offset) if frame_offset else 0
        stop = None if (num_frames is None or int(num_frames) < 0) else start + int(num_frames)
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True,
                           start=start, stop=stop)
        t = torch.from_numpy(data)
        return (t.t().contiguous() if channels_first else t.contiguous()), sr

    def _save(filepath, src, sample_rate, channels_first=True, format=None,
              encoding=None, bits_per_sample=None, buffer_size=4096, backend=None,
              compression=None):
        arr = src.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T if channels_first else arr
        sf.write(str(filepath), arr, int(sample_rate))

    torchaudio.load = _load
    torchaudio.save = _save
except Exception:
    pass
PY

"$PY" -c "import torch, so_vits_svc_fork, torchaudio; print('svcfork OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "svcfork ready. If non-default, set AUTOVOICE_SVCFORK_BIN=$CONDA_ROOT/envs/$FORK_ENV/bin/svc"
