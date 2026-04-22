from __future__ import annotations

from pathlib import Path

from auto_voice.runtime_contract import (
    CANONICAL_LIVE_PIPELINE,
    CANONICAL_OFFLINE_PIPELINE,
    build_packaged_artifact_manifest,
    get_pipeline_status_template,
    load_packaged_artifact_manifest,
    normalize_reference_audio_entries,
    normalize_pipeline_choice,
    write_packaged_artifact_manifest,
)


def test_pipeline_defaults_are_canonical():
    assert normalize_pipeline_choice(None, mode="offline") == CANONICAL_OFFLINE_PIPELINE
    assert normalize_pipeline_choice(None, mode="live") == CANONICAL_LIVE_PIPELINE


def test_pipeline_status_template_marks_canonical_and_experimental():
    status = get_pipeline_status_template()

    assert status["quality_seedvc"]["stability"] == "canonical"
    assert status["realtime"]["canonical_default"] is True
    assert status["quality_shortcut"]["stability"] == "experimental"
    assert status["realtime_meanvc"]["mode"] == "live"


def test_packaged_artifact_manifest_round_trip(tmp_path: Path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    manifest = build_packaged_artifact_manifest(
        profile_id="profile-123",
        display_name="Test Artist",
        model_family="seed_vc",
        canonical_pipeline="quality_seedvc",
        sample_rate=44100,
        speaker_embedding_dim=256,
        mel_bins=80,
        artifacts={
            "profile_json": "models/test/profile.json",
            "speaker_embedding": "models/test/speaker_embedding.npy",
            "adapter": "models/test/artifacts/adapter.pt",
            "tensorrt_engine": None,
        },
        metadata={
            "reference_audio": [
                {"path": reference, "duration": 12.5, "source": "training_sample"}
            ]
        },
    )

    path = tmp_path / "artifact_manifest.json"
    write_packaged_artifact_manifest(path, manifest)
    loaded = load_packaged_artifact_manifest(path)

    assert loaded["profile_id"] == "profile-123"
    assert loaded["canonical_pipeline"] == "quality_seedvc"
    assert loaded["compatibility"]["supported_pipelines"] == ["quality_seedvc"]
    assert loaded["metadata"]["reference_audio"] == [
        {
            "path": str(reference),
            "source": "training_sample",
            "duration_seconds": 12.5,
        }
    ]


def test_normalize_reference_audio_entries_deduplicates_and_normalizes(tmp_path: Path):
    first = tmp_path / "one.wav"
    second = tmp_path / "two.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    entries = normalize_reference_audio_entries(
        [
            {"path": first, "duration": 4.0, "sample_id": "sample-1"},
            str(first),
            {"vocals_path": second, "source": "training_sample", "created_at": "2026-01-01T00:00:00Z"},
        ],
        require_exists=True,
    )

    assert entries == [
        {
            "path": str(first),
            "sample_id": "sample-1",
            "duration_seconds": 4.0,
        },
        {
            "path": str(second),
            "source": "training_sample",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]
