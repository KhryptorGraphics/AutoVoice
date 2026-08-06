"""has_trained_model must recognise engines that store artifacts elsewhere.

A fully trained, serving so-vits-svc-fork voice reported
`has_trained_model: false` from GET /profiles/<id>/training-status while the
same response said `training_status: "ready"` - because the fork lane
registers into data/fork_models/<id>.json and writes nothing under
trained_models_dir, which is the only place the check looked.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def store(tmp_path):
    return VoiceProfileStore(
        profiles_dir=str(tmp_path / "voice_profiles"),
        samples_dir=str(tmp_path / "samples"),
    )


def _make_profile(store, profile_id, **fields):
    profile = {"profile_id": profile_id, "name": "T", "user_id": "operator"}
    profile.update(fields)
    store.save(profile)
    return profile


class TestForkRegisteredModels:
    def test_recorded_manifest_outside_profile_dir_counts(self, store, tmp_path):
        """The fork registry is that lane's manifest; the record points at it."""
        registry = tmp_path / "fork_models" / "p-fork.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps({"engine": "so-vits-svc-fork"}))

        _make_profile(
            store, "p-fork",
            runtime_artifact_manifest_path=str(registry),
        )
        assert store.has_trained_model("p-fork") is True

    def test_recorded_path_that_does_not_exist_is_not_trusted(self, store, tmp_path):
        """A dangling pointer must not claim a model exists."""
        _make_profile(
            store, "p-dangling",
            runtime_artifact_manifest_path=str(tmp_path / "gone.json"),
        )
        assert store.has_trained_model("p-dangling") is False

    def test_untrained_profile_still_false(self, store):
        _make_profile(store, "p-empty")
        assert store.has_trained_model("p-empty") is False

    def test_missing_profile_false(self, store):
        assert store.has_trained_model("does-not-exist") is False

    def test_empty_recorded_path_ignored(self, store):
        _make_profile(store, "p-blank", runtime_artifact_manifest_path="")
        assert store.has_trained_model("p-blank") is False


class TestExistingBehaviourPreserved:
    def test_derived_artifact_manifest_still_wins(self, store):
        """The in-repo lanes write trained_models/<id>/artifact_manifest.json."""
        _make_profile(store, "p-native")
        manifest = Path(store._artifact_manifest_path("p-native"))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        assert store.has_trained_model("p-native") is True

    def test_lora_weights_still_count(self, store):
        _make_profile(store, "p-lora")
        weights = Path(store._lora_weights_path("p-lora"))
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_text("x")
        assert store.has_trained_model("p-lora") is True
