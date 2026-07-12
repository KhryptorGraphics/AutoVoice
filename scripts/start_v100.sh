#!/usr/bin/env bash
# Launch the AutoVoice training/web server on the V100 host (e.g. kp@192.168.1.53).
#
# This host has three GPUs (P100 sm60, RTX 3080 Ti sm86, Tesla V100 sm70). The
# CUDA runtime's default "fastest-first" device ordering does NOT match
# nvidia-smi's PCI order, so a numeric CUDA_VISIBLE_DEVICES=N can silently bind
# the wrong card. We therefore pin the V100 by UUID and also set
# CUDA_DEVICE_ORDER=PCI_BUS_ID, then hard-assert the bound GPU is sm_70 (7,0)
# before serving so a wrong-GPU launch fails loudly instead of training on the
# P100/3080Ti (which would also invalidate the no-bf16 assumption).
#
# The env is self-contained (PYTHONNOUSERSITE=1): user-site (~/.local) packages
# must never shadow the conda env. Uses the threading Socket.IO server
# (scripts/serve_local_threading.py) — no eventlet dependency.
#
# V100 has no bf16: training jobs must use precision fp32 or fp16 (never bf16).
#
# Usage:  scripts/start_v100.sh                # foreground
#         nohup scripts/start_v100.sh &        # background
# Overrides: AUTOVOICE_ENV, AUTOVOICE_HOST, AUTOVOICE_PORT, AUTOVOICE_DATA_DIR,
#            CUDA_VISIBLE_DEVICES (a GPU UUID for a different box/card)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${AUTOVOICE_ENV:-autovoice-v100-cu126}"
HOST="${AUTOVOICE_HOST:-0.0.0.0}"
PORT="${AUTOVOICE_PORT:-10600}"
DATA_DIR="${AUTOVOICE_DATA_DIR:-$REPO/data}"

# Tesla V100 (PG500-216) UUID on kp@192.168.1.53. Override for another card.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-f888d35b-222d-bde2-c0b1-362756ddd49f}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONNOUSERSITE=1

# Source conda from a known path (conda is not on PATH in a non-interactive
# SSH shell, so `conda info --base` can't bootstrap itself). Override with
# CONDA_SH for a non-default install location.
# shellcheck disable=SC1091
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
  for cand in "$HOME/miniconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
    [ -f "$cand" ] && CONDA_SH="$cand" && break
  done
fi
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO"

# Hard GPU guard: refuse to serve unless the bound device is a V100 (sm_70).
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("start_v100: CUDA not available in this env")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"start_v100: bound GPU0 = {name} capability={cap} arch_list={torch.cuda.get_arch_list()}")
if cap != (7, 0):
    sys.exit(
        f"start_v100: expected V100 sm_70 (7,0) but bound {cap} ({name}). "
        f"Fix CUDA_VISIBLE_DEVICES (must be the V100 UUID)."
    )
PY

echo "start_v100: launching on ${HOST}:${PORT} (env=${ENV_NAME}, V100 sm70, threading, fp16-capable)"
exec python scripts/serve_local_threading.py --host "$HOST" --port "$PORT" --data-dir "$DATA_DIR"
