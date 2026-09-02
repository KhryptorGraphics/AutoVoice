#!/bin/bash
# Rebuild the so-vits-svc-fork training environment on a fresh gpuhub/AutoDL box.
#
# Usage (from a machine holding this directory):
#   scp -P <port> -r scripts/remote_train_env root@<host>:/root/autodl-tmp/
#   ssh -p <port> root@<host> 'bash /root/autodl-tmp/remote_train_env/setup_remote_env.sh'
#                                     ^ scp uses -P for port, ssh uses -p. Mixing
#                                       them makes scp read the port as a filename.
#
# Why this exists: building from scratch takes ~15 min, and `pyworld` is the only
# dependency with no upstream wheel - it compiles from source and dominates that
# time. The prebuilt wheel in ./wheels skips it. Everything else installs from
# prebuilt wheels at CDN speed.
#
# Wheel compatibility: ./wheels holds cp310 / linux_x86_64 builds. On a box with a
# different Python or arch pip ignores them and compiles from source again (correct,
# just slower). Check with `python3 -V` first; these boxes ship Python 3.10.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/root/autodl-tmp/venv}"
export DEBIAN_FRONTEND=noninteractive

echo "== system packages =="
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq python3 python3-venv python3-pip ffmpeg libsndfile1 rsync >/dev/null 2>&1

echo "== venv =="
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip

echo "== torch (cu128; Blackwell needs sm_120) =="
# torchaudio pinned to 2.8.0 deliberately: >=2.9 delegates decoding to torchcodec
# and `svc pre-hubert` dies with "TorchCodec is required". Installing torchcodec
# does NOT fix it (its .so wants a CUDA runtime lib that is not present).
pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "== pyworld from prebuilt wheel (skips the source build) =="
pip install -q --find-links "$HERE/wheels" pyworld==0.3.5

echo "== so-vits-svc-fork =="
pip install -q so-vits-svc-fork

echo "== verify =="
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0));print('sm_120:',any('120' in a for a in torch.cuda.get_arch_list()))"
python -c "import pyworld;print('pyworld OK')"
command -v svc >/dev/null && echo "svc OK" || { echo "svc MISSING"; exit 1; }
