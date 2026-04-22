from __future__ import annotations

import pytest

from auto_voice.web.persistence import (
    AppStateStore,
    DEFAULT_AUDIO_ROUTER_CONFIG,
    DEFAULT_PITCH_CONFIG,
    DEFAULT_SEPARATION_CONFIG,
    _coerce_bool,
    _coerce_float,
    _coerce_optional_int,
    _normalize_app_settings,
    _normalize_audio_router_config,
    _normalize_device_config,
    _normalize_karaoke_session,
    _normalize_pitch_config,
    _normalize_separation_config,
    resolve_data_dir,
)


def test_normalize_app_settings_preserves_legacy_realtime():
    normalized = _normalize_app_settings({"preferred_pipeline": "realtime"})

    assert normalized["preferred_offline_pipeline"] == "realtime"
    assert normalized["preferred_live_pipeline"] == "realtime"
    assert normalized["preferred_pipeline"] == "realtime"
    assert normalized["last_updated"] is None


def test_coercion_helpers_cover_string_and_invalid_paths():
    assert _coerce_bool(None, True) is True
    assert _coerce_bool("off", True) is False
    assert _coerce_bool("yes", False) is True
    assert _coerce_bool(0, True) is False

    assert _coerce_float("1.7", 0.5) == pytest.approx(1.7)
    assert _coerce_float("9", 0.5, minimum=0.0, maximum=2.0) == pytest.approx(2.0)
    assert _coerce_float("bad", 0.5) == pytest.approx(0.5)

    assert _coerce_optional_int("") is None
    assert _coerce_optional_int("7") == 7
    with pytest.raises(ValueError):
        _coerce_optional_int(True)


def test_normalize_audio_router_config_coerces_invalid_inputs():
    normalized = _normalize_audio_router_config(
        {
            "speaker_gain": "1.4",
            "headphone_gain": "nan-ish",
            "speaker_enabled": "false",
            "headphone_enabled": "yes",
            "speaker_device": True,
            "headphone_device": "3",
            "sample_rate": "invalid",
        }
    )

    assert normalized["speaker_gain"] == pytest.approx(1.4)
    assert normalized["headphone_gain"] == DEFAULT_AUDIO_ROUTER_CONFIG["headphone_gain"]
    assert normalized["speaker_enabled"] is False
    assert normalized["headphone_enabled"] is True
    assert normalized["speaker_device"] is None
    assert normalized["headphone_device"] == 3
    assert normalized["sample_rate"] == DEFAULT_AUDIO_ROUTER_CONFIG["sample_rate"]


def test_normalize_device_separation_and_pitch_configs():
    device = _normalize_device_config(
        {"input_device_id": 2, "output_device_id": "", "sample_rate": "44100"}
    )
    assert device == {
        "input_device_id": "2",
        "output_device_id": None,
        "sample_rate": 44100,
    }

    separation = _normalize_separation_config(
        {
            "model": "htdemucs_ft",
            "stems": ["vocals", "", 5],
            "overlap": "9",
            "segment_length": "-1",
            "shifts": "bad",
            "device": "cuda",
        }
    )
    assert separation["model"] == "htdemucs_ft"
    assert separation["stems"] == ["vocals", "5"]
    assert separation["overlap"] == 1.0
    assert separation["segment_length"] is None
    assert separation["shifts"] == DEFAULT_SEPARATION_CONFIG["shifts"]
    assert separation["device"] == "cuda"

    pitch = _normalize_pitch_config(
        {
            "method": "harvest",
            "hop_length": "bad",
            "f0_min": 0,
            "f0_max": "2200",
            "threshold": "2.0",
            "use_gpu": "true",
        }
    )
    assert pitch["method"] == "harvest"
    assert pitch["hop_length"] == DEFAULT_PITCH_CONFIG["hop_length"]
    assert pitch["f0_min"] == DEFAULT_PITCH_CONFIG["f0_min"]
    assert pitch["f0_max"] == 2200
    assert pitch["threshold"] == 1.0
    assert pitch["use_gpu"] is True
    assert pitch["device"] == "cuda"


def test_normalize_karaoke_session_requires_session_id_and_normalizes_fields():
    assert _normalize_karaoke_session(None) is None
    assert _normalize_karaoke_session({"session_id": "   "}) is None

    normalized = _normalize_karaoke_session(
        {
            "session_id": " session-1 ",
            "song_id": 42,
            "requested_pipeline": "REALTIME",
            "resolved_pipeline": "QUALITY_SEEDVC",
            "sample_rate": "32000",
            "is_active": "1",
            "collect_samples": "off",
            "sample_collection_enabled": "yes",
            "audio_router_targets": {"speaker_device": "5", "headphone_enabled": "false"},
            "speaker_embedding": [0.1, 0.2],
        }
    )

    assert normalized["session_id"] == "session-1"
    assert normalized["song_id"] == "42"
    assert normalized["requested_pipeline"] == "realtime"
    assert normalized["resolved_pipeline"] == "quality_seedvc"
    assert normalized["sample_rate"] == 32000
    assert normalized["is_active"] is True
    assert normalized["collect_samples"] is False
    assert normalized["sample_collection_enabled"] is True
    assert normalized["audio_router_targets"]["speaker_device"] == 5
    assert normalized["audio_router_targets"]["headphone_enabled"] is False
    assert normalized["speaker_embedding"] == [0.1, 0.2]


def test_resolve_data_dir_prefers_explicit_then_env(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "env-data")
    assert resolve_data_dir("explicit-data").name == "explicit-data"
    assert resolve_data_dir().name == "env-data"

    monkeypatch.delenv("DATA_DIR")
    assert resolve_data_dir().name == "data"


def test_app_state_store_read_invalid_json_returns_default(tmp_path):
    store = AppStateStore(str(tmp_path))
    store._files["app_settings"].write_text("{not-json", encoding="utf-8")

    settings = store.get_app_settings()

    assert settings["preferred_offline_pipeline"] == "quality_seedvc"
    assert settings["preferred_live_pipeline"] == "realtime"


def test_app_state_store_training_background_preset_and_conversion_crud(tmp_path):
    store = AppStateStore(str(tmp_path))

    older_job = {"job_id": "job-1", "profile_id": "profile-a", "created_at": "2024-01-01T00:00:00Z"}
    newer_job = {"job_id": "job-2", "profile_id": "profile-b", "created_at": "2024-01-02T00:00:00Z"}
    store.save_training_job(older_job)
    store.save_training_job(newer_job)
    assert [job["job_id"] for job in store.list_training_jobs()] == ["job-2", "job-1"]
    assert [job["job_id"] for job in store.list_training_jobs("profile-a")] == ["job-1"]
    assert store.get_training_job("job-1") == older_job

    bg_one = {"job_id": "bg-1", "job_type": "tensorrt", "created_at": "2024-01-01T00:00:00Z"}
    bg_two = {"job_id": "bg-2", "job_type": "extract", "created_at": "2024-01-03T00:00:00Z"}
    store.save_background_job(bg_one)
    store.save_background_job(bg_two)
    assert [job["job_id"] for job in store.list_background_jobs("tensorrt")] == ["bg-1"]
    assert store.get_background_job("bg-2") == bg_two

    preset = {"id": "preset-1", "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-04T00:00:00Z"}
    store.save_preset(preset)
    assert store.get_preset("preset-1") == preset
    assert [item["id"] for item in store.list_presets()] == ["preset-1"]
    assert store.delete_preset("missing") is False
    assert store.delete_preset("preset-1") is True

    record = {"id": "record-1", "profile_id": "profile-a", "created_at": "2024-01-05T00:00:00Z"}
    store.save_conversion_record(record)
    assert store.get_conversion_record("record-1") == record
    assert [item["id"] for item in store.list_conversion_history("profile-a")] == ["record-1"]
    assert store.delete_conversion_record("missing") is False
    assert store.delete_conversion_record("record-1") is True


def test_app_state_store_checkpoint_and_youtube_history_crud(tmp_path):
    store = AppStateStore(str(tmp_path))

    checkpoint = {"id": "ckpt-1", "created_at": "2024-01-01T00:00:00Z"}
    store.save_checkpoint("profile-a", checkpoint)
    assert store.get_checkpoint("profile-a", "ckpt-1") == checkpoint
    assert [item["id"] for item in store.list_checkpoints("profile-a")] == ["ckpt-1"]
    assert store.delete_checkpoint("profile-a", "missing") is False
    assert store.delete_checkpoint("profile-a", "ckpt-1") is True
    assert store.list_checkpoints("profile-a") == []

    for idx in range(105):
        store.save_youtube_history_item(
            {"id": f"item-{idx}", "timestamp": f"2024-01-{idx:02d}T00:00:00Z"}
        )
    limited = store.list_youtube_history(limit=3)
    assert len(store.list_youtube_history()) == 100
    assert len(limited) == 3
    assert limited[0]["id"] == "item-99"
    assert store.delete_youtube_history_item("missing") is False
    assert store.delete_youtube_history_item("item-99") is True
    store.clear_youtube_history()
    assert store.list_youtube_history() == []


def test_app_state_store_settings_models_configs_and_karaoke_sessions(tmp_path):
    store = AppStateStore(str(tmp_path))

    settings = store.update_app_settings({"preferred_pipeline": "realtime"})
    assert settings["preferred_offline_pipeline"] == "quality_seedvc"
    assert settings["preferred_live_pipeline"] == "realtime"
    settings = store.update_app_settings(
        {
            "preferred_offline_pipeline": "realtime",
            "preferred_live_pipeline": "realtime",
        }
    )
    assert settings["preferred_offline_pipeline"] == "realtime"
    assert settings["preferred_live_pipeline"] == "realtime"

    encoder = {"model_type": "encoder", "loaded_at": "2024-01-02T00:00:00Z"}
    decoder = {"model_type": "decoder", "loaded_at": "2024-01-01T00:00:00Z"}
    store.save_loaded_model("encoder", encoder)
    store.save_loaded_model("decoder", decoder)
    assert [item["model_type"] for item in store.list_loaded_models()] == ["decoder", "encoder"]
    assert store.get_loaded_model("encoder") == encoder
    assert store.delete_loaded_model("missing") is False
    assert store.delete_loaded_model("decoder") is True
    store.clear_loaded_models()
    assert store.list_loaded_models() == []

    separation = store.update_separation_config({"segment_length": "12.5", "shifts": 2, "device": "cuda"})
    assert separation["segment_length"] == 12.5
    assert separation["shifts"] == 2
    assert separation["device"] == "cuda"

    pitch = store.update_pitch_config({"device": "cuda"})
    assert pitch["use_gpu"] is True
    assert pitch["device"] == "cuda"

    router = store.update_audio_router_config({"speaker_device": "7", "headphone_enabled": "false"})
    assert router["speaker_device"] == 7
    assert router["headphone_enabled"] is False

    devices = store.update_device_config({"input_device_id": 12, "sample_rate": 48000})
    assert devices["input_device_id"] == "12"
    assert devices["sample_rate"] == 48000

    snapshot = store.save_karaoke_session(
        {
            "session_id": "session-1",
            "song_id": "song-1",
            "last_activity": 20,
            "started_at": 10,
            "audio_router_targets": {"speaker_device": "9"},
            "speaker_embedding": [1.0],
        }
    )
    assert snapshot["audio_router_targets"]["speaker_device"] == 9
    assert store.get_karaoke_session("session-1")["speaker_embedding"] == [1.0]
    assert [item["session_id"] for item in store.list_karaoke_sessions()] == ["session-1"]
    assert store.delete_karaoke_session("missing") is False
    assert store.delete_karaoke_session("session-1") is True

    with pytest.raises(ValueError):
        store.save_karaoke_session({"song_id": "missing-session-id"})
