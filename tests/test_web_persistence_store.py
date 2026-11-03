"""Round-trip unit tests for the JSON-backed AppStateStore.

These cover the store's CRUD surfaces and normalizer edge branches directly:
they are the CPU-only coverage source for src/auto_voice/web/persistence.py
in the remaining-module coverage gate (the API suites only graze it).
"""
import json

import pytest

from auto_voice.web.persistence import (
    AppStateStore,
    _coerce_bool,
    _coerce_float,
    _coerce_optional_int,
    _normalize_app_settings,
    _normalize_audio_router_config,
    _normalize_device_config,
    _normalize_karaoke_session,
    _normalize_pitch_config,
    _normalize_separation_config,
)


@pytest.fixture
def store(tmp_path):
    return AppStateStore(data_dir=str(tmp_path))


class TestCoercions:
    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("YES", True), ("on", True),
        ("false", False), ("0", False), ("No", False), ("off", False),
    ])
    def test_coerce_bool_strings(self, value, expected):
        assert _coerce_bool(value, not expected) is expected

    def test_coerce_bool_fallbacks(self):
        assert _coerce_bool(None, True) is True
        assert _coerce_bool("maybe", False) is True  # non-empty string -> bool()
        assert _coerce_bool(0, True) is False

    def test_coerce_float_invalid_and_clamped(self):
        assert _coerce_float("not-a-number", 0.5) == 0.5
        assert _coerce_float(None, 0.25) == 0.25
        assert _coerce_float(9.0, 0.5) == 2.0  # clamped to maximum
        assert _coerce_float(-1.0, 0.5) == 0.0  # clamped to minimum

    def test_coerce_optional_int(self):
        assert _coerce_optional_int(None) is None
        assert _coerce_optional_int("") is None
        assert _coerce_optional_int("7") == 7
        with pytest.raises(ValueError):
            _coerce_optional_int(True)


class TestNormalizers:
    def test_app_settings_legacy_realtime(self):
        settings = _normalize_app_settings({"preferred_pipeline": "realtime"})
        assert settings["preferred_offline_pipeline"] == "realtime"
        assert settings["preferred_live_pipeline"] == "realtime"
        assert settings["preferred_pipeline"] == "realtime"

    def test_audio_router_invalid_values_fall_back(self):
        config = _normalize_audio_router_config({
            "speaker_gain": "loud",          # invalid float -> default
            "speaker_enabled": "off",
            "speaker_device": "not-an-int",  # invalid -> default
            "headphone_device": "3",
            "sample_rate": "broken",         # invalid -> keep default
        })
        assert config["speaker_enabled"] is False
        assert config["headphone_device"] == 3
        assert config["speaker_device"] == _normalize_audio_router_config({})["speaker_device"]
        assert config["sample_rate"] == _normalize_audio_router_config({})["sample_rate"]

    def test_device_config_bad_sample_rate_ignored(self):
        config = _normalize_device_config({
            "input_device_id": 4,
            "output_device_id": "",
            "sample_rate": "junk",
        })
        assert config["input_device_id"] == "4"
        assert config["output_device_id"] is None
        assert config["sample_rate"] == _normalize_device_config({})["sample_rate"]

    def test_separation_config_invalid_segment_and_shifts(self):
        default = _normalize_separation_config({})
        config = _normalize_separation_config({
            "segment_length": "oops",
            "shifts": "nope",
            "device": "tpu",  # not accepted -> default retained
        })
        assert config["segment_length"] == default["segment_length"]
        assert config["shifts"] == default["shifts"]
        assert config["device"] == default["device"]
        assert _normalize_separation_config({"segment_length": -2})["segment_length"] is None
        assert _normalize_separation_config({"segment_length": ""})["segment_length"] is None

    def test_pitch_config_device_and_invalid_ints(self):
        default = _normalize_pitch_config({})
        config = _normalize_pitch_config({"hop_length": "bad", "device": "cuda"})
        assert config["hop_length"] == default["hop_length"]
        assert config["device"] == "cuda"
        assert config["use_gpu"] is True
        cpu = _normalize_pitch_config({"device": "cpu"})
        assert cpu["use_gpu"] is False

    def test_karaoke_session_normalization(self):
        assert _normalize_karaoke_session(None) is None
        assert _normalize_karaoke_session({"session_id": "  "}) is None
        session = _normalize_karaoke_session({
            "session_id": " s1 ",
            "requested_pipeline": " REALTIME ",
            "is_active": "yes",
            "speaker_embedding": [0.1, 0.2],
            "stats": {"frames": 5},
        })
        assert session["session_id"] == "s1"
        assert session["requested_pipeline"] == "realtime"
        assert session["resolved_pipeline"] == "realtime"
        assert session["is_active"] is True
        assert session["speaker_embedding"] == [0.1, 0.2]
        assert session["stats"] == {"frames": 5}
        no_embedding = _normalize_karaoke_session({"session_id": "s2",
                                                   "speaker_embedding": "oops"})
        assert no_embedding["speaker_embedding"] is None


class TestStoreReadWrite:
    def test_corrupt_json_returns_default(self, store):
        store._files["presets"].write_text("{not json", encoding="utf-8")
        assert store.list_presets() == []

    def test_training_jobs_roundtrip_and_filter(self, store):
        store.save_training_job({"job_id": "j1", "profile_id": "p1", "created_at": "2026-01-01"})
        store.save_training_job({"job_id": "j2", "profile_id": "p2", "created_at": "2026-01-02"})
        assert store.get_training_job("j1")["profile_id"] == "p1"
        assert [j["job_id"] for j in store.list_training_jobs()] == ["j2", "j1"]
        assert [j["job_id"] for j in store.list_training_jobs("p1")] == ["j1"]

    def test_background_jobs_roundtrip_and_filter(self, store):
        store.save_background_job({"job_id": "b1", "job_type": "export", "created_at": "1"})
        store.save_background_job({"job_id": "b2", "job_type": "import", "created_at": "2"})
        assert store.get_background_job("b2")["job_type"] == "import"
        assert [j["job_id"] for j in store.list_background_jobs("export")] == ["b1"]

    def test_presets_crud(self, store):
        store.save_preset({"id": "pr1", "created_at": "1"})
        assert store.get_preset("pr1") is not None
        assert [p["id"] for p in store.list_presets()] == ["pr1"]
        assert store.delete_preset("pr1") is True
        assert store.delete_preset("pr1") is False

    def test_conversion_history_crud(self, store):
        store.save_conversion_record({"id": "c1", "profile_id": "p1", "created_at": "1"})
        store.save_conversion_record({"id": "c2", "profile_id": "p2", "created_at": "2"})
        assert store.get_conversion_record("c1")["profile_id"] == "p1"
        assert [r["id"] for r in store.list_conversion_history("p2")] == ["c2"]
        assert store.delete_conversion_record("c1") is True
        assert store.delete_conversion_record("missing") is False

    def test_conversion_workflows_crud(self, store):
        store.save_conversion_workflow({"workflow_id": "w1", "status": "done", "created_at": "1"})
        store.save_conversion_workflow({"workflow_id": "w2", "status": "running", "updated_at": "2"})
        assert store.get_conversion_workflow("w2")["status"] == "running"
        assert [w["workflow_id"] for w in store.list_conversion_workflows("done")] == ["w1"]
        assert [w["workflow_id"] for w in store.list_conversion_workflows()] == ["w2", "w1"]
        assert store.delete_conversion_workflow("w1") is True
        assert store.delete_conversion_workflow("w1") is False

    def test_diarization_results_crud(self, store):
        store.save_diarization_result({"diarization_id": "d1", "created_at": 1})
        store.save_diarization_result({"diarization_id": "d2", "created_at": 2})
        assert store.get_diarization_result("d1") is not None
        assert [d["diarization_id"] for d in store.list_diarization_results()] == ["d2", "d1"]
        assert store.delete_diarization_result("d2") is True
        assert store.delete_diarization_result("d2") is False

    def test_diarization_segment_assignments(self, store):
        assert store.get_diarization_segment_assignments("p1") == {}
        saved = store.save_diarization_segment_assignment("p1", "seg-0", "/a.wav")
        assert saved == {"seg-0": "/a.wav"}
        assert store.get_diarization_segment_assignments("p1") == {"seg-0": "/a.wav"}
        assert store.delete_diarization_segment_assignments("p1") is True
        assert store.delete_diarization_segment_assignments("p1") is False

    def test_checkpoints_crud(self, store):
        store.save_checkpoint("p1", {"id": "ck1", "created_at": "1"})
        store.save_checkpoint("p1", {"id": "ck2", "created_at": "2"})
        assert store.get_checkpoint("p1", "ck1") is not None
        assert [c["id"] for c in store.list_checkpoints("p1")] == ["ck2", "ck1"]
        assert store.delete_checkpoint("p1", "missing") is False
        assert store.delete_checkpoint("p1", "ck1") is True
        assert store.delete_checkpoint("p1", "ck2") is True  # empties the profile map
        assert store.list_checkpoints("p1") == []

    def test_webhooks_crud(self, store):
        store.save_webhook({"id": "wh1", "created_at": "1"})
        assert store.get_webhook("wh1") is not None
        assert [w["id"] for w in store.list_webhooks()] == ["wh1"]
        assert store.delete_webhook("wh1") is True
        assert store.delete_webhook("wh1") is False

    def test_loaded_models_crud(self, store):
        store.save_loaded_model("fork", {"model_type": "fork", "loaded_at": "1"})
        assert store.get_loaded_model("fork") is not None
        assert len(store.list_loaded_models()) == 1
        assert store.delete_loaded_model("fork") is True
        assert store.delete_loaded_model("fork") is False
        store.save_loaded_model("hq", {"model_type": "hq"})
        store.clear_loaded_models()
        assert store.list_loaded_models() == []

    def test_config_updates_roundtrip(self, store):
        sep = store.update_separation_config({"model": "htdemucs_ft"})
        assert sep["model"] == "htdemucs_ft"
        assert store.get_separation_config()["model"] == "htdemucs_ft"
        pitch = store.update_pitch_config({"device": "cuda"})
        assert pitch["use_gpu"] is True
        router = store.update_audio_router_config({"speaker_gain": 1.5})
        assert router["speaker_gain"] == 1.5
        device = store.update_device_config({"input_device_id": 2})
        assert device["input_device_id"] == "2"


class TestYoutubeHistoryAndAssets:
    def test_youtube_history_crud(self, store):
        store.save_youtube_history_item({"id": "y1", "timestamp": "1"})
        store.save_youtube_history_item({"id": "y2", "timestamp": "2"})
        store.save_youtube_history_item({"id": "y1", "timestamp": "3"})  # replaces
        assert store.get_youtube_history_item("y1")["timestamp"] == "3"
        assert store.get_youtube_history_item("missing") is None
        assert [i["id"] for i in store.list_youtube_history(limit=1)] == ["y1"]
        assert store.delete_youtube_history_item("y2") is True
        assert store.delete_youtube_history_item("y2") is False
        export = store.export_youtube_history()
        assert export["count"] == 1 and export["items"][0]["id"] == "y1"

    def test_purge_youtube_history_deletes_managed_assets(self, store, tmp_path):
        managed = tmp_path / "managed"
        managed.mkdir()
        inside = managed / "song.wav"
        inside.write_bytes(b"x")
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"y")
        store.save_youtube_history_item({
            "id": "y1", "timestamp": "1",
            "audioPath": str(inside),
            "filteredPath": str(outside),
            "audio_path": "\x00bad\x00path",  # unresolvable -> skipped
        })
        result = store.purge_youtube_history(delete_assets=True, managed_roots=[managed])
        assert result["purged_items"] == 1
        assert str(inside) in result["deleted_files"]
        assert str(outside) in result["skipped_files"]
        assert not inside.exists()
        assert outside.exists()
        assert store.list_youtube_history() == []

    def test_purge_without_delete_keeps_files(self, store, tmp_path):
        wav = tmp_path / "keep.wav"
        wav.write_bytes(b"x")
        store.save_youtube_history_item({"id": "y1", "timestamp": "1", "audioPath": str(wav)})
        result = store.purge_youtube_history(delete_assets=False)
        assert result["deleted_files"] == []
        assert wav.exists()

    def test_asset_registry_crud(self, store, tmp_path):
        record = store.register_asset(tmp_path / "a.wav", kind="song", owner_id="p1")
        again = store.register_asset(tmp_path / "a.wav", kind="song", owner_id="p1")
        assert record["asset_id"] == again["asset_id"]  # stable id per path
        assert store.get_asset(record["asset_id"])["kind"] == "song"
        assert store.get_asset("missing") is None
        store.register_asset(tmp_path / "b.wav", kind="song", owner_id="p2")
        assert len(store.list_assets()) == 2
        assert [a["owner_id"] for a in store.list_assets(owner_id="p2")] == ["p2"]
        assert store.delete_asset(record["asset_id"]) is True
        assert store.delete_asset(record["asset_id"]) is False


class TestKaraokeSessions:
    def test_karaoke_session_store_roundtrip(self, store):
        saved = store.save_karaoke_session({"session_id": "s1", "last_activity": 2})
        assert saved["session_id"] == "s1"
        store.save_karaoke_session({"session_id": "s2", "last_activity": 5})
        sessions = store.list_karaoke_sessions()
        assert [s["session_id"] for s in sessions] == ["s2", "s1"]
        assert store.get_karaoke_session("s1")["session_id"] == "s1"
        assert store.get_karaoke_session("missing") is None
        assert store.delete_karaoke_session("s1") is True
        assert store.delete_karaoke_session("s1") is False

    def test_karaoke_session_requires_id(self, store):
        with pytest.raises(ValueError):
            store.save_karaoke_session({"song_id": "x"})

    def test_list_skips_malformed_snapshots(self, store):
        store._write("karaoke_sessions", {"bad": {"song_id": "no-session-id"},
                                          "ok": {"session_id": "s9"}})
        assert [s["session_id"] for s in store.list_karaoke_sessions()] == ["s9"]
