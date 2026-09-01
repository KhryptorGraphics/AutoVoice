"""The fork patches live in site-packages and a pip upgrade silently reverts them.

Inference/training run as a subprocess against the `svc` CLI in the `svcfork`
env, so the patches cannot live in this repo's source tree. These tests fail
loudly if the installed package drifts back to unpatched.
"""
from pathlib import Path

import pytest

SYNTH = Path(
    "/home/kp/anaconda3/envs/svcfork/lib/python3.12/site-packages/"
    "so_vits_svc_fork/modules/synthesizers.py"
)

pytestmark = pytest.mark.skipif(
    not SYNTH.exists(), reason="svcfork env not installed on this machine"
)


def test_uv_contract_patch_is_applied():
    assert "SVCFORK_UV_CONTRACT" in SYNTH.read_text(), (
        "svcfork_uv_contract.patch is missing from the installed package - a pip "
        "upgrade likely reverted it. Reapply per patches/README.md."
    )


def test_uv_masking_reaches_the_decoder_in_both_paths():
    """Masked f0 must feed the decoder in training AND inference."""
    text = SYNTH.read_text()
    assert "_uv_masked_f0(f0, uv), spec_lengths" in text, "forward() path unpatched"
    assert "f0=_uv_masked_f0(f0, uv)" in text, "infer() path unpatched"


def test_pitch_embedding_still_gets_the_interpolated_contour():
    """f0_to_coarse wants the continuous contour - masking it would be a bug."""
    text = SYNTH.read_text()
    assert "f0_to_coarse(_uv_masked_f0" not in text, (
        "the coarse pitch embedding must keep the interpolated f0, not the masked one"
    )


def test_default_is_off_so_serving_is_unchanged():
    """Off by default: the model was trained against the violated contract."""
    import os, subprocess
    py = "/home/kp/anaconda3/envs/svcfork/bin/python"
    env = {k: v for k, v in os.environ.items() if k != "SVCFORK_UV_CONTRACT"}
    env.pop("PYTHONPATH", None)          # keep the serving env's path out of it
    out = subprocess.run(
        [py, "-c",
         "import torch;"
         "from so_vits_svc_fork.modules.synthesizers import _uv_masked_f0;"
         "f0=torch.tensor([[100.,150.]]);uv=torch.tensor([[1.,0.]]);"
         "print(torch.equal(_uv_masked_f0(f0,uv),f0))"],
        capture_output=True, text=True, env=env,
    )
    assert out.stdout.strip() == "True", (
        f"default must be a no-op; got {out.stdout!r} {out.stderr[-300:]!r}")


CLUSTER_INIT = Path(
    "/home/kp/anaconda3/envs/svcfork/lib/python3.12/site-packages/"
    "so_vits_svc_fork/cluster/__init__.py"
)


@pytest.mark.skipif(not CLUSTER_INIT.exists(), reason="svcfork env not installed")
def test_cluster_loader_torch_load_patch_is_applied():
    """torch>=2.6 defaults weights_only=True and refuses this checkpoint's
    plain numpy-array dict; a genuine `svc train-cluster` output hits this
    identically, so the loader needs weights_only=False explicitly."""
    assert "weights_only=False" in CLUSTER_INIT.read_text(), (
        "svcfork_cluster_torch_load.patch is missing - a pip upgrade likely "
        "reverted it. Reapply per patches/README.md."
    )
