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

F0 = SYNTH.parent.parent / "f0.py"


def test_crepe_periodicity_uv_patch_is_applied():
    """Other half of the uv contract: crepe never emits f0==0, so without this
    patch uv is 1 on every frame and the SVCFORK_UV_CONTRACT mask is a no-op."""
    text = F0.read_text()
    assert "SVCFORK_CREPE_UV_THRESHOLD" in text, (
        "svcfork_crepe_periodicity_uv.patch is missing from the installed package - "
        "a pip upgrade likely reverted it. Reapply per patches/README.md."
    )
    assert "return_periodicity=True" in text
    # env-gated: unset must keep the original crepe path so serving is unaffected
    assert '_os.environ.get("SVCFORK_CREPE_UV_THRESHOLD"' in text

DISC = SYNTH.parent / "descriminators.py"
TRAIN = SYNTH.parent.parent / "train.py"


def test_mrd_discriminator_patch_is_applied():
    """The MRD discriminator (Conor's served G_197 was trained with it) lived only
    on a rented box until 2026-09-04; this guards the rebuilt copy. Env-gated:
    SVCFORK_MRD=1 at training time, otherwise the fork's plain MPD."""
    text = DISC.read_text()
    assert "class MultiPeriodDiscriminatorWithMRD" in text and "class DiscriminatorR" in text
    assert "self.discriminators.extend(" in text, "MRD must extend the SAME ModuleList so MPD-only D checkpoints load key-for-key"
    assert 'os.environ.get("SVCFORK_MRD"' in TRAIN.read_text()

def test_lora_patch_is_applied():
    """LoRA on the fork: low-rank deltas for fine-tuning ~80 min of audio, where
    every full fine-tune degraded after ~epoch 150.

    The injection must be additive (a forward hook plus new parameters), never a
    module replacement, or an ordinary checkpoint stops loading key-for-key.
    """
    text = SYNTH.read_text()
    assert "def inject_lora" in text and "def freeze_base_for_lora" in text
    assert "_lora_forward_hook" in text, "must hook, not replace the module"
    assert "register_forward_hook" in text
    # targets the inference path only - enc_q never runs at infer()
    assert '_LORA_TARGET_PREFIXES = ("dec.", "flow.", "enc_p.")' in text
    assert "enc_q" not in text.split("_LORA_TARGET_PREFIXES")[1][:400]
    # zero-init on B, so the adapted model starts identical to the base
    assert "zeros_(b.weight)" in text
    # collect-then-mutate, or named_modules() recurses into what it just added
    assert "victims = [" in text
    assert 'os.environ.get("SVCFORK_LORA_RANK"' in TRAIN.read_text(), (
        "train.py must gate LoRA on SVCFORK_LORA_RANK and narrow the optimiser")


def test_lora_keeps_an_ordinary_checkpoint_loadable():
    """The compatibility guarantee, exercised rather than asserted from source.

    Runtime check, so it only runs where the fork is importable. The fork lives
    in an isolated env and is normally driven as a subprocess, so this skips in
    the main test env - test_lora_patch_is_applied covers the same patch from
    source there.
    """
    import torch
    pytest.importorskip("so_vits_svc_fork",
                        reason="fork package is installed only in the svcfork env")
    from so_vits_svc_fork.modules.synthesizers import SynthesizerTrn, inject_lora

    hps = dict(inter_channels=192, hidden_channels=192, filter_channels=768,
               n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
               resblock="1", resblock_kernel_sizes=[3, 7, 11],
               resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
               upsample_rates=[8, 8, 2, 2, 2], upsample_initial_channel=512,
               upsample_kernel_sizes=[16, 16, 4, 4, 4], gin_channels=256,
               ssl_dim=768, n_speakers=200, sampling_rate=44100)
    m = SynthesizerTrn(513, 10240 // 512, **hps)
    before = set(m.state_dict())
    inject_lora(m, rank=4, alpha=8.0)
    after = set(m.state_dict())
    assert before < after, "LoRA must only ADD keys"
    assert before.issubset(after), "an original key was renamed or dropped"
    # a base-shaped state dict still loads
    m2 = SynthesizerTrn(513, 10240 // 512, **hps)
    from so_vits_svc_fork.utils import safe_load
    safe_load(m, m2.state_dict())

def test_lora_is_injected_at_inference_too():
    """A LoRA checkpoint carries 358 extra `_lora_*` tensors. inference/core.py
    must inject the same side-paths BEFORE load_checkpoint, or safe_load copies
    only matching keys and every delta is silently discarded - the model then
    serves as the unadapted base and the whole LoRA run measures as a no-op.
    Caught during the first LoRA evaluation, before scoring rather than after.
    """
    core = SYNTH.parent.parent / "inference" / "core.py"
    text = core.read_text()
    assert 'os.environ.get("SVCFORK_LORA_RANK"' in text
    inject_at = text.index("inject_lora(")
    load_at = text.index("utils.load_checkpoint(")
    assert inject_at < load_at, (
        "inject_lora must run BEFORE load_checkpoint or the deltas load as nothing")

