"""A so-vits-svc-fork voice must be *labelled* as one, everywhere.

Two independent regressions, one root cause: the fork lane was bolted onto
predicates that only knew about ``full_model``, so *routing* learned about forks
and *labelling* did not.

1. ``VoiceProfileStore`` derives ``active_model_type`` from artifacts under
   ``trained_models_dir``. A fork checkpoint lives in
   ``data/fork_models/<id>.json`` instead, so no ``has_*`` flag can see it and
   the derivation downgraded a fork voice to "adapter" - the profile then
   advertised an engine it does not use. ``save()`` compounded it by dropping
   any value outside its allow-list, so the correct value could only ever be
   destroyed, never kept.

2. The conversion ``runtime_backend`` label tested ``== 'full_model'`` while the
   routing line beside it had been widened to accept ``fork_backed`` too
   (commit 53888599). A fork profile cannot satisfy the former, so every fork
   conversion was stamped plain ``pytorch`` and the history read as though
   so-vits-svc-fork had never run.
"""
import json
import re
from pathlib import Path

import pytest

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def store(tmp_path):
    return VoiceProfileStore(
        profiles_dir=str(tmp_path / "voice_profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained_models"),
    )


def test_explicit_svc_fork_survives_normalization(store):
    """The derivation must not downgrade a fork voice to 'adapter'.

    A fork profile legitimately reports a trained model that is not under
    trained_models_dir, which is the exact state that produced "adapter".
    """
    normalized = store._normalize_profile({
        "profile_id": "fork-voice",
        "name": "Fork Voice",
        "active_model_type": "svc_fork",
        "has_trained_model": True,
    })
    assert normalized["active_model_type"] == "svc_fork"


def test_svc_fork_survives_a_save_load_round_trip(store):
    """save() must persist it.

    It previously popped anything off its allow-list, so the fork label was
    destroyed by the next save of any unrelated field - which is how Brandy's
    profile came to claim "adapter" after her legacy artifacts were deleted.
    """
    store.save({
        "profile_id": "fork-voice",
        "name": "Fork Voice",
        "active_model_type": "svc_fork",
        "has_trained_model": True,
    })
    loaded = store.load("fork-voice")
    assert loaded["active_model_type"] == "svc_fork", (
        "the fork label did not survive a save/load round trip")


def test_ordinary_derivation_is_untouched(store, tmp_path):
    """Blast-radius guard.

    _normalize_profile has 63 dependent symbols across training, web and
    inference, so the fork branch must change nothing for profiles that never
    asked for it.

    Note ``has_full_model`` is recomputed from disk and ignores the input dict,
    so the full_model case needs a real artifact; ``has_trained_model`` is
    honoured from input.
    """
    models = tmp_path / "trained_models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "withfull_full_model.pt").write_bytes(b"stub")

    assert store._normalize_profile(
        {"profile_id": "withfull"})["active_model_type"] == "full_model"
    assert store._normalize_profile(
        {"profile_id": "p", "has_trained_model": True})["active_model_type"] == "adapter"
    assert store._normalize_profile({"profile_id": "p"})["active_model_type"] == "base"
    # an explicit choice whose artifact is absent still falls back, as before
    assert store._normalize_profile(
        {"profile_id": "p", "active_model_type": "full_model"})["active_model_type"] == "base"


@pytest.mark.parametrize("path,routing_predicate", [
    ("src/auto_voice/web/job_manager.py", "fork_backed"),
    ("src/auto_voice/web/api_conversion.py", "has_fork_model"),
])
def test_the_backend_label_uses_the_same_predicate_as_the_routing(path, routing_predicate):
    """Guards the literal asymmetry that caused this.

    Routing said ``active_model_type == 'full_model' or <fork predicate>``;
    labelling said only the former. Asserted against the source because the
    defect is the divergence between two adjacent lines, not a runtime value.
    """
    src = Path(path).read_text()
    block = re.search(
        r"if resolved_pipeline == 'quality':(.{0,800}?)runtime_backend = 'pytorch_full_model'",
        src, re.S)
    assert block, f"{path}: the runtime_backend decision is not in the expected shape"
    body = block.group(1)
    assert routing_predicate in body, (
        f"{path}: the runtime_backend label ignores `{routing_predicate}`, so fork "
        "conversions get stamped 'pytorch' and read as base-model renders")
    assert "so_vits_svc_fork" in body, (
        f"{path}: a fork-backed conversion must be labelled 'so_vits_svc_fork'")
