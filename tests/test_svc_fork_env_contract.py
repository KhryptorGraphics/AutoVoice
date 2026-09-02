"""Guard the CUDA allocator config handed to svc-fork training subprocesses.

``expandable_segments:True`` never returns the segments it grows. Measured on
Thor over 162 steps: live tensors flat at 2,401.8 MiB while the allocator
reserve climbed 4 GB -> 91 GB. GPU memory is system RAM on Jetson, so the run
degrades from ~1 s/step to 180 s/step and the box drifts toward OOM.

``_clean_env`` used to pass ``os.environ`` straight through, so a training
run's memory behaviour depended on the shell that launched gunicorn. These
tests pin the override so that cannot come back.
"""
import os
from unittest import mock

from auto_voice.training.svc_fork_trainer import _clean_env


class TestCleanEnvAllocator:
    def test_allocator_is_pinned(self):
        env = _clean_env()
        assert "PYTORCH_CUDA_ALLOC_CONF" in env, (
            "_clean_env must pin the allocator, not leave it to the environment"
        )
        assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"], (
            "expandable_segments grows the reserve without bound on Jetson"
        )

    @mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    def test_a_hostile_inherited_value_is_overridden(self):
        """The exact regression: the serving env exported the ruinous value."""
        env = _clean_env()
        assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"], (
            "inherited expandable_segments:True must not reach the trainer"
        )

    def test_still_drops_pythonpath_and_pins_nousersite(self):
        """The pre-existing contract this function already had."""
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/serving/path"}):
            env = _clean_env()
        assert "PYTHONPATH" not in env
        assert env["PYTHONNOUSERSITE"] == "1"


class TestInferenceAllocatorMatchesTrainer:
    """Training and inference share one GPU with no guard between them, so they
    must not run it under opposite allocator policies.

    The systemd unit exports PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
    The trainer overrode it; the inference bridge did not, so infer subprocesses
    inherited the setting that never returns grown segments - and inference is
    the one running continuously.
    """

    def test_inference_pins_the_allocator_too(self):
        from auto_voice.inference.svc_fork_bridge import _clean_env
        env = _clean_env(None)
        assert "PYTORCH_CUDA_ALLOC_CONF" in env
        assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]

    @mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512"})
    def test_the_units_inherited_value_is_overridden(self):
        """The exact value the systemd unit exports today."""
        from auto_voice.inference.svc_fork_bridge import _clean_env
        assert "expandable_segments" not in _clean_env(None)["PYTORCH_CUDA_ALLOC_CONF"]

    def test_both_sides_agree(self):
        from auto_voice.inference.svc_fork_bridge import _clean_env as infer_env
        from auto_voice.training.svc_fork_trainer import _clean_env as train_env
        assert infer_env(None)["PYTORCH_CUDA_ALLOC_CONF"] == train_env()["PYTORCH_CUDA_ALLOC_CONF"]
