"""Cluster-model wiring in svc_fork_bridge.convert.

`svc infer` supports `-k/--cluster-model-path` and `-r/--cluster-infer-ratio`
to pull out-of-distribution content vectors toward the training speaker's
distribution before the flow inverts them - a treatment for the VITS
prior/posterior gap traced this session (posterior decode holds up
cross-singer; the flow-based inference path degrades substantially more).
Neither flag was ever wired from the registry entry, because no cluster model
existed before now. These tests pin the wiring, not the flags themselves.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from auto_voice.inference import svc_fork_bridge as bridge


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    bridge.clear_cache()
    yield
    bridge.clear_cache()


def _registry_entry(tmp_path, **overrides):
    model = tmp_path / "G.pth"; model.write_bytes(b"x")
    config = tmp_path / "config.json"; config.write_text("{}")
    entry = {
        "engine": "so-vits-svc-fork", "speaker": "connor",
        "model_path": str(model), "config_path": str(config),
        "svc_bin": "/bin/true", "f0_method": "harvest",
    }
    entry.update(overrides)
    data_dir = tmp_path / "data"
    (data_dir / "fork_models").mkdir(parents=True)
    (data_dir / "fork_models" / "p1.json").write_text(json.dumps(entry))
    return data_dir


def _mock_run(out_wav_path):
    def _run(cmd, **kwargs):
        # svc_fork_bridge writes the input wav; simulate the fork writing output.
        import shutil
        src = [a for a in cmd if a.endswith("in.wav")][0]
        shutil.copy(src, out_wav_path)
        return MagicMock(returncode=0, stdout="", stderr="")
    return _run


class TestClusterFlagWiring:
    def test_no_cluster_path_means_no_cluster_flags(self, tmp_path):
        """Unset must reproduce the fork's own default exactly - no -k/-r at all,
        not `-r 0`, since callers may rely on the flag's absence."""
        data_dir = _registry_entry(tmp_path)
        captured = {}

        def _run(cmd, **kwargs):
            captured["cmd"] = cmd
            import shutil, tempfile, os
            out = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"][0]
            shutil.copy([a for a in cmd if a.endswith("in.wav")][0], out)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p1", str(data_dir))
        assert "-k" not in captured["cmd"]
        assert "-r" not in captured["cmd"]

    def test_cluster_path_set_adds_both_flags(self, tmp_path):
        cluster = tmp_path / "kmeans.pt"; cluster.write_bytes(b"x")
        data_dir = _registry_entry(tmp_path, cluster_model_path=str(cluster),
                                    cluster_infer_ratio=0.5)
        captured = {}

        def _run(cmd, **kwargs):
            captured["cmd"] = cmd
            import shutil
            out = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"][0]
            shutil.copy([a for a in cmd if a.endswith("in.wav")][0], out)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p1", str(data_dir))
        cmd = captured["cmd"]
        assert cmd[cmd.index("-k") + 1] == str(cluster)
        assert cmd[cmd.index("-r") + 1] == "0.5"

    def test_cluster_path_without_explicit_ratio_defaults_to_zero(self, tmp_path):
        """A cluster model can be registered without immediately blending it in -
        ratio 0.0 means the flag is present but inert, useful for A/B via a
        single registry field flip rather than adding/removing -k."""
        cluster = tmp_path / "kmeans.pt"; cluster.write_bytes(b"x")
        data_dir = _registry_entry(tmp_path, cluster_model_path=str(cluster))
        captured = {}

        def _run(cmd, **kwargs):
            captured["cmd"] = cmd
            import shutil
            out = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"][0]
            shutil.copy([a for a in cmd if a.endswith("in.wav")][0], out)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p1", str(data_dir))
        cmd = captured["cmd"]
        assert cmd[cmd.index("-r") + 1] == "0.0"


def test_trained_kmeans_file_is_a_valid_sklearn_cluster_dump():
    """Sanity-check the artifact this session's train-cluster run produces,
    if it has landed on durable disk yet."""
    path = Path(
        "/home/kp/thordrive/autofusion/autovoice/checkpoints/"
        "svcfork_conor_fullband_20260828/kmeans.pt"
    )
    if not path.exists():
        pytest.skip("cluster training has not completed on this machine yet")
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    assert "connor" in obj, f"expected a 'connor' speaker key, got {list(obj)}"
    centers = obj["connor"]["cluster_centers_"]
    assert centers.shape[0] == 2000, f"expected 2000 clusters, got {centers.shape}"
    # This model runs contentvec_final_proj: false (config.json), so content
    # vectors are ssl_dim-wide (768), not the 256-dim final-projection output.
    config = json.loads((path.parent / "config.json").read_text())
    expected_dim = config["model"]["ssl_dim"]
    assert centers.shape[1] == expected_dim, (
        f"expected {expected_dim}-dim vectors (ssl_dim from config.json), got {centers.shape}"
    )


class TestUvContractFlagWiring:
    """requires_uv_contract must be per-model: a checkpoint trained without
    the uv-masking fix would face a fresh train/serve mismatch if served with
    it forced on, and vice versa - see mrd-uvfix-retrain-outcome memory."""

    def test_default_off_sets_no_env_var(self, tmp_path):
        data_dir = _registry_entry(tmp_path)
        captured = {}

        def _run(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            import shutil
            out = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"][0]
            shutil.copy([a for a in cmd if a.endswith("in.wav")][0], out)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p1", str(data_dir))
        assert "SVCFORK_UV_CONTRACT" not in captured["env"]

    def test_requires_uv_contract_sets_env_var(self, tmp_path):
        data_dir = _registry_entry(tmp_path, requires_uv_contract=True)
        captured = {}

        def _run(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            import shutil
            out = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"][0]
            shutil.copy([a for a in cmd if a.endswith("in.wav")][0], out)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p1", str(data_dir))
        assert captured["env"]["SVCFORK_UV_CONTRACT"] == "1"

    def test_other_profiles_are_unaffected(self, tmp_path):
        """Flipping one model's flag must not leak into a sibling profile's
        subprocess env within the same process."""
        data_dir = _registry_entry(tmp_path, requires_uv_contract=True)
        model2 = tmp_path / "G2.pth"; model2.write_bytes(b"x")
        config2 = tmp_path / "config2.json"; config2.write_text("{}")
        (data_dir / "fork_models" / "p2.json").write_text(json.dumps({
            "engine": "so-vits-svc-fork", "speaker": "other",
            "model_path": str(model2), "config_path": str(config2),
            "svc_bin": "/bin/true", "f0_method": "harvest",
        }))
        captured = []

        def _run(cmd, **kwargs):
            captured.append(kwargs.get("env", {}))
            import shutil
            out = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"][0]
            shutil.copy([a for a in cmd if a.endswith("in.wav")][0], out)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p1", str(data_dir))
            bridge.convert(np.zeros(1600, dtype=np.float32), 16000, "p2", str(data_dir))
        assert captured[0]["SVCFORK_UV_CONTRACT"] == "1"
        assert "SVCFORK_UV_CONTRACT" not in captured[1]
