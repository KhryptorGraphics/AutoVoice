#!/usr/bin/env bash
# Create the isolated `uvr` conda env used by the lead/backing vocal
# separation bridge (auto_voice.inference.separation_bridge).
#
# Same recipe as setup_svcfork_env.sh: clone the working autovoice-thor env
# (keeps the proven Jetson CUDA torch — the Mel-Band RoFormer karaoke models
# run on torch CUDA), then layer the tools on top:
#   - audio-separator: runs UVR-family separation models (MIT; models are
#     community checkpoints, typically non-commercial — same posture as our
#     so-vits-svc-fork usage)
#   - basic-pitch: Spotify's polyphonic note tracker (Apache-2.0), used to
#     decompose backing-vocal harmony stacks into per-voice lines
#
# Idempotent. Override env names via AUTOVOICE_BASE_ENV / AUTOVOICE_UVR_ENV.
set -euo pipefail

BASE_ENV="${AUTOVOICE_BASE_ENV:-autovoice-thor}"
UVR_ENV="${AUTOVOICE_UVR_ENV:-uvr}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "/${UVR_ENV}\$"; then
  echo "Cloning $BASE_ENV -> $UVR_ENV (preserves CUDA torch)..."
  conda create -y --clone "$BASE_ENV" -n "$UVR_ENV"
else
  echo "Env $UVR_ENV already exists; ensuring packages are present."
fi

PY="$CONDA_ROOT/envs/$UVR_ENV/bin/python"
"$PY" -m pip install audio-separator

# basic-pitch gets its own py3.10 env: on Linux/py>=3.11 it hard-requires
# tensorflow<2.15.1 (no Jetson aarch64 wheels); py<3.11 uses tflite-runtime.
BP_ENV="${AUTOVOICE_BASICPITCH_ENV:-basicpitch}"
if ! conda env list | grep -qE "/${BP_ENV}\$"; then
  conda create -y -n "$BP_ENV" python=3.10
fi
"$CONDA_ROOT/envs/$BP_ENV/bin/pip" install basic-pitch
"$CONDA_ROOT/envs/$BP_ENV/bin/basic-pitch" --help >/dev/null

echo "Verifying CUDA torch survived the install..."
"$PY" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), (
    "CUDA torch was clobbered by pip; reinstall the Jetson torch wheel")
print("torch", torch.__version__, "cuda OK")
PYEOF

"$CONDA_ROOT/envs/$UVR_ENV/bin/audio-separator" --version
echo "uvr env ready"
