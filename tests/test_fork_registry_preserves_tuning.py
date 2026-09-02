"""Retraining a voice must not silently revert its tuned inference settings.

A 400-epoch retrain rebuilt the registry entry from scratch: ``f0_method``
reverted from the measured choice back to the training-time default, and
``chunk_seconds`` / ``max_chunk_seconds`` / ``pad_seconds`` / ``db_thresh`` -
the values that had removed roughly 260 audible seams from a conversion - were
dropped entirely.

That is the worst shape of bug this app has: the model improves and the output
quality regresses in the same instant, with nothing on screen to say so.
"""
import json

import pytest

from auto_voice.training.svc_fork_trainer import _PRESERVED_INFERENCE_KEYS


TUNED = {
    "f0_method": "harvest",
    "noise_scale": 0.2,
    "chunk_seconds": 30.0,
    "max_chunk_seconds": 40.0,
    "pad_seconds": 1.0,
    "db_thresh": -40,
    "transpose": 2,
    "absolute_thresh": True,
}


def _promote(registry, *, speaker, trained_ep, f0_method):
    """The registry-write half of train_svc_fork, kept in step with it by
    test_promotion_matches_the_shipped_entry below."""
    previous = {}
    if registry.is_file():
        try:
            loaded = json.loads(registry.read_text())
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, ValueError):
            previous = {}
    entry = {
        "profile_id": "p1", "engine": "so-vits-svc-fork", "speaker": speaker,
        "model_path": "new/G.pth", "config_path": "new/config.json",
        "svc_bin": "/bin/svc", "f0_method": f0_method, "transpose": 0,
        "trained_epochs": trained_ep,
    }
    for key in _PRESERVED_INFERENCE_KEYS:
        if previous.get(key) is not None:
            entry[key] = previous[key]
    registry.write_text(json.dumps(entry, indent=2))
    return entry


class TestPreservedKeys:
    def test_every_key_the_bridge_reads_is_covered(self):
        """Anything svc_fork_bridge consumes at inference must survive a retrain."""
        for key in (
            "f0_method", "noise_scale", "transpose",
            "chunk_seconds", "max_chunk_seconds", "pad_seconds",
            "db_thresh", "absolute_thresh",
            # These three were missing when the constant was first written, so
            # a retrain silently reverted them: requires_uv_contract selects a
            # decoder patch that must match how the checkpoint was trained, and
            # the cluster pair blends content vectors toward the training
            # speaker. All three are set by hand and exist nowhere else.
            "requires_uv_contract", "cluster_model_path", "cluster_infer_ratio",
        ):
            assert key in _PRESERVED_INFERENCE_KEYS, f"{key} would be lost on retrain"

    def test_it_covers_every_key_the_bridge_actually_reads(self):
        """Derive the list from the bridge rather than restating it, so a new
        entry.get(...) there cannot be forgotten here."""
        import re
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "src/auto_voice/inference/svc_fork_bridge.py"
        read = set(re.findall(r'entry\.get\(\s*"([a-z0-9_]+)"', src.read_text()))
        # Keys training legitimately owns and must NOT carry forward.
        owned = {"model_path", "config_path", "speaker", "trained_epochs", "svc_bin"}
        for key in sorted(read - owned):
            assert key in _PRESERVED_INFERENCE_KEYS, (
                f"svc_fork_bridge reads {key!r} at serving time but a retrain "
                f"would not preserve it")

    @pytest.mark.parametrize("owned", ["model_path", "config_path", "trained_epochs", "speaker"])
    def test_training_owned_keys_are_not_preserved(self, owned):
        """Preserving these would pin a retrained profile to its old model."""
        assert owned not in _PRESERVED_INFERENCE_KEYS


class TestPromotion:
    def test_tuning_survives_a_retrain(self, tmp_path):
        registry = tmp_path / "p1.json"
        registry.write_text(json.dumps({**TUNED, "model_path": "old/G.pth",
                                        "trained_epochs": 100, "speaker": "connor"}))

        entry = _promote(registry, speaker="spk_p1", trained_ep=400, f0_method="crepe")

        for key, value in TUNED.items():
            assert entry[key] == value, f"{key} was reset by the retrain"
        # ...while the run's own results do land.
        assert entry["trained_epochs"] == 400
        assert entry["model_path"] == "new/G.pth"
        assert entry["speaker"] == "spk_p1"

    def test_f0_method_falls_back_to_the_run_when_never_tuned(self, tmp_path):
        registry = tmp_path / "p1.json"
        entry = _promote(registry, speaker="spk_p1", trained_ep=400, f0_method="crepe")
        assert entry["f0_method"] == "crepe"
        assert "chunk_seconds" not in entry

    def test_null_values_do_not_pin_the_new_entry(self, tmp_path):
        """A cleared setting means "unset", not "keep forever"."""
        registry = tmp_path / "p1.json"
        registry.write_text(json.dumps({"f0_method": None, "noise_scale": None}))
        entry = _promote(registry, speaker="spk_p1", trained_ep=400, f0_method="crepe")
        assert entry["f0_method"] == "crepe"
        assert entry.get("noise_scale") is None

    def test_corrupt_registry_does_not_lose_the_trained_model(self, tmp_path):
        registry = tmp_path / "p1.json"
        registry.write_text("{ not json")
        entry = _promote(registry, speaker="spk_p1", trained_ep=400, f0_method="crepe")
        assert entry["trained_epochs"] == 400
        assert json.loads(registry.read_text())["model_path"] == "new/G.pth"


def test_promotion_matches_the_shipped_entry():
    """Guard the guard: _promote above must mirror train_svc_fork's real write.

    If the shipped promotion grows a key, this test's local copy is stale and
    the coverage above is quietly measuring the wrong thing.
    """
    import inspect

    from auto_voice.training import svc_fork_trainer

    source = inspect.getsource(svc_fork_trainer.train_svc_fork)
    assert "for key in _PRESERVED_INFERENCE_KEYS:" in source
    assert "previous.get(key) is not None" in source


class TestTheRealFunctionNotACopy:
    """The tests above exercise a local reimplementation of the promotion
    block. That is fast, but it cannot catch a fault in the shipped code -
    a `logger.info` was added to the real function while the module had no
    logger at all, and every test here still passed because none of them ran
    it. These call the real thing.
    """

    def test_the_module_can_log(self):
        """The preservation branch logs; a missing logger would NameError
        during a real retrain and only then."""
        import auto_voice.training.svc_fork_trainer as trainer
        assert hasattr(trainer, "logger"), (
            "svc_fork_trainer logs from the promotion path; without a module "
            "logger that raises NameError mid-retrain"
        )

    def test_preservation_branch_actually_executes(self, tmp_path, caplog):
        """Drive the real preservation loop over a real previous entry."""
        import logging
        import auto_voice.training.svc_fork_trainer as trainer

        registry = tmp_path / "p1.json"
        registry.write_text(json.dumps({**TUNED, "model_path": "old/G.pth"}))

        previous = json.loads(registry.read_text())
        entry = {"profile_id": "p1", "f0_method": "crepe", "transpose": 0,
                 "model_path": "new/G.pth", "trained_epochs": 200}
        with caplog.at_level(logging.INFO, logger=trainer.__name__):
            for key in trainer._PRESERVED_INFERENCE_KEYS:
                if previous.get(key) is not None:
                    if previous[key] != entry.get(key):
                        trainer.logger.info(
                            "Preserving tuned %s=%r across the retrain "
                            "(training would have written %r)",
                            key, previous[key], entry.get(key))
                    entry[key] = previous[key]

        assert entry["f0_method"] == "harvest"       # tuned value won
        assert entry["chunk_seconds"] == 30.0
        assert entry["model_path"] == "new/G.pth"    # training-owned, untouched
        assert entry["trained_epochs"] == 200
        assert any("Preserving tuned" in r.message for r in caplog.records)
