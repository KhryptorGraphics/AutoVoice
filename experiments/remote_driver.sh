#!/bin/bash
# Self-driving arm runner. Lives on the GPU box and is launched with setsid, so
# it survives the operator's ssh dropping, the controlling session ending, and
# the laptop closing. Everything it needs is already on disk before it starts.
exec > /root/driver.log 2>&1
set -x
ARM="${1:-baseline}"
ROOT=/root/arms/$ARM
echo "=== ARM $ARM  START $(date) ==="

# 1. keyshift conditioning patch, applied to the INSTALLED package.
PKG=$(python3 -c "import so_vits_svc_fork,os;print(os.path.dirname(so_vits_svc_fork.__file__))")
echo "package at $PKG"
if ! grep -q emb_keyshift "$PKG/modules/synthesizers.py" 2>/dev/null; then
  cd "$PKG" && patch -p1 --forward < /root/svcfork_keyshift_cond.patch && echo "PATCH_APPLIED" || echo "PATCH_FAILED"
else
  echo "PATCH_ALREADY_PRESENT"
fi
python3 -c "
import so_vits_svc_fork.modules.synthesizers as s, inspect
print('keyshift in synthesizer:', 'emb_keyshift' in inspect.getsource(s))"

# 2. preprocessing. -n 2 on purpose: the default worker count OOM'd a previous
#    box during hubert extraction and silently dropped 65 of 438 files. This box
#    has 15 GB of RAM, so it is tighter again.
cd "$ROOT" || { echo "FATAL no corpus at $ROOT"; exit 1; }
ls dataset_raw/brandy/*.wav | wc -l | sed 's/^/items: /'
svc pre-resample                     || { echo "FATAL pre-resample"; exit 1; }
svc pre-config                       || { echo "FATAL pre-config"; exit 1; }
svc pre-hubert -n 2                  || { echo "FATAL pre-hubert"; exit 1; }
echo "=== PREPROCESS_OK $(date) ==="
find dataset/44k -name '*.soft.pt' | wc -l | sed 's/^/hubert files: /'

# 3. train
svc train -t
echo "=== TRAIN_EXIT $? $(date) ==="
ls -la logs/44k/G_*.pth 2>/dev/null | tail -5
echo "=== ARM $ARM DONE $(date) ==="
