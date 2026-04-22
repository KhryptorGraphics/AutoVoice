"""Focused coverage for TrainingJobManager Phase 4/4.4 logic."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_voice.training.job_manager import (
    JobStatus,
    TrainingConfig,
    TrainingJobManager,
)


def _sample(
    sample_id: str,
    *,
    created_at: str = "2026-01-01T00:00:00",
    duration: float = 60.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        created_at=created_at,
        duration=duration,
        vocals_path=f"/tmp/{sample_id}.wav",
    )


@pytest.fixture
def manager(tmp_path):
    return TrainingJobManager(storage_path=tmp_path / "jobs", require_gpu=False)


class TestProgressAndProfilePersistence:
    def test_emit_training_progress_updates_job_and_emits_events(self, manager):
        job = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        manager._emit_event = Mock()

        manager.emit_training_progress(
            job_id=job.job_id,
            epoch=1,
            total_epochs=2,
            step=2,
            total_steps=4,
            loss=0.25,
            learning_rate=1e-4,
        )

        assert manager._emit_event.call_count == 2
        assert manager.get_job(job.job_id).progress == 25

    def test_emit_training_progress_ignores_missing_job(self, manager):
        manager._emit_event = Mock()

        manager.emit_training_progress(
            job_id="missing",
            epoch=1,
            total_epochs=1,
            step=1,
            total_steps=1,
            loss=0.1,
            learning_rate=1e-4,
        )

        manager._emit_event.assert_not_called()

    def test_update_profile_training_state_marks_adapter_ready(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "embedding": [1, 2, 3]}

        with patch.object(manager, "_get_profile_store", return_value=store):
            manager._update_profile_training_state(
                profile_id="profile-1",
                results={
                    "adapter_path": "/tmp/profile-1_adapter.pt",
                    "epochs_completed": 12,
                    "final_loss": 0.12,
                    "artifact_type": "adapter",
                },
                sample_count=7,
            )

        saved = store.save.call_args.args[0]
        assert saved["training_status"] == "ready"
        assert saved["has_trained_model"] is True
        assert saved["model_version"] == "1.0"
        assert saved["model_path"] == "/tmp/profile-1_adapter.pt"
        assert saved["training_epochs"] == 12
        assert saved["loss_final"] == 0.12
        assert saved["sample_count"] == 7
        assert saved["selected_adapter"] == "unified"
        assert saved["active_model_type"] == "adapter"
        assert "embedding" not in saved

    def test_update_profile_training_state_marks_full_model_ready(self, manager):
        store = Mock()
        store.load.return_value = {
            "profile_id": "profile-2",
            "model_version": "2.0",
            "embedding": [1, 2, 3],
        }

        with patch.object(manager, "_get_profile_store", return_value=store):
            manager._update_profile_training_state(
                profile_id="profile-2",
                results={"artifact_type": "full_model", "final_loss": 0.05},
                sample_count=10,
            )

        saved = store.save.call_args.args[0]
        assert saved["model_version"] == "2.0"
        assert saved["active_model_type"] == "full_model"
        assert "selected_adapter" not in saved

    def test_mark_profile_training_failed_persists_error(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "embedding": [1, 2, 3]}

        with patch.object(manager, "_get_profile_store", return_value=store):
            manager._mark_profile_training_failed("profile-1", "boom")

        saved = store.save.call_args.args[0]
        assert saved["training_status"] == "failed"
        assert saved["last_training_error"] == "boom"
        assert "embedding" not in saved

    def test_mark_profile_training_failed_swallows_store_errors(self, manager):
        store = Mock()
        store.load.side_effect = RuntimeError("missing")

        with patch.object(manager, "_get_profile_store", return_value=store):
            manager._mark_profile_training_failed("profile-1", "boom")

    def test_mark_job_completed_sets_terminal_state(self, manager):
        job = manager.create_job(profile_id="profile-1", sample_ids=["s1"])

        manager._mark_job_completed(job.job_id, results={"loss": 0.1})

        updated = manager.get_job(job.job_id)
        assert updated.status == JobStatus.COMPLETED.value
        assert updated.progress == 100
        assert updated.results == {"loss": 0.1}

    def test_cleanup_completed_jobs_keeps_latest_and_get_completed_jobs_is_sorted(self, manager):
        first = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        second = manager.create_job(profile_id="profile-2", sample_ids=["s2"])
        third = manager.create_job(profile_id="profile-3", sample_ids=["s3"])

        for offset, job in enumerate((first, second, third), start=1):
            job.status = JobStatus.COMPLETED.value
            job.completed_at = datetime(2026, 1, offset, 12, 0, 0)

        completed = manager.get_completed_jobs()
        assert [job.job_id for job in completed] == [third.job_id, second.job_id, first.job_id]

        removed = manager.cleanup_completed_jobs(keep_count=1)
        assert removed == [second.job_id, first.job_id]
        assert manager.get_job(third.job_id) is not None


class TestNeedsTrainingLogic:
    def test_check_needs_training_profile_not_found(self, manager):
        store = Mock()
        store.load.side_effect = FileNotFoundError("missing")

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_training("missing")

        assert result == {
            "needs_training": False,
            "reason": "profile_not_found",
            "sample_count": 0,
            "has_adapter": False,
        }

    def test_check_needs_training_insufficient_samples(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1"}
        store.list_training_samples.return_value = [_sample("s1")]

        with patch.object(manager, "_get_profile_store", return_value=store), patch(
            "auto_voice.models.adapter_manager.AdapterManager"
        ) as adapter_manager_cls:
            adapter_manager_cls.return_value.load_adapter.return_value = None
            result = manager.check_needs_training("profile-1", min_samples=2)

        assert result["needs_training"] is False
        assert result["reason"].startswith("insufficient_samples")
        assert result["sample_count"] == 1

    def test_check_needs_training_without_adapter(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1"}
        store.list_training_samples.return_value = [_sample("s1")] * 5

        with patch.object(manager, "_get_profile_store", return_value=store), patch(
            "auto_voice.models.adapter_manager.AdapterManager"
        ) as adapter_manager_cls:
            adapter_manager_cls.return_value.load_adapter.side_effect = RuntimeError("bad adapter")
            result = manager.check_needs_training("profile-1", min_samples=5)

        assert result["needs_training"] is True
        assert result["reason"] == "no_adapter"
        assert result["has_adapter"] is False

    def test_check_needs_training_honors_flagged_status(self, manager):
        store = Mock()
        store.load.return_value = {
            "profile_id": "profile-1",
            "status": "needs_training",
        }
        store.list_training_samples.return_value = [_sample(f"s{i}") for i in range(5)]

        with patch.object(manager, "_get_profile_store", return_value=store), patch(
            "auto_voice.models.adapter_manager.AdapterManager"
        ) as adapter_manager_cls:
            adapter_manager_cls.return_value.load_adapter.return_value = {"layer.lora_A": 1}
            result = manager.check_needs_training("profile-1", min_samples=5)

        assert result["needs_training"] is True
        assert result["reason"] == "flagged_needs_training"
        assert result["has_adapter"] is True

    def test_check_needs_training_rejects_non_target_profile(self, manager):
        store = Mock()
        store.load.return_value = {
            "profile_id": "profile-1",
            "profile_role": "source_artist",
        }
        store.list_training_samples.return_value = [_sample(f"s{i}") for i in range(6)]

        with patch.object(manager, "_get_profile_store", return_value=store), patch(
            "auto_voice.models.adapter_manager.AdapterManager"
        ) as adapter_manager_cls:
            adapter_manager_cls.return_value.load_adapter.return_value = {"layer.lora_A": 1}
            result = manager.check_needs_training("profile-1", min_samples=5)

        assert result["needs_training"] is False
        assert result["reason"] == "unsupported_profile_role (source_artist)"


class TestNeedsRetrainingLogic:
    def test_check_needs_retraining_profile_not_found(self, manager):
        store = Mock()
        store.load.side_effect = FileNotFoundError("missing")

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_retraining("missing")

        assert result == {
            "needs_retraining": False,
            "reasons": [],
            "new_sample_count": 0,
            "days_since_training": 0,
            "quality_metrics": {},
        }

    def test_check_needs_retraining_detects_new_samples_staleness_and_quality_drop(self, manager):
        last_trained = datetime.now() - timedelta(days=40)
        old_sample_time = (last_trained - timedelta(days=1)).isoformat()
        new_sample_time = (last_trained + timedelta(days=1)).isoformat()

        store = Mock()
        store.load.return_value = {
            "profile_id": "profile-1",
            "last_trained_at": last_trained.isoformat(),
            "quality_metrics": {"speaker_similarity": 0.7, "mcd": 5.2},
        }
        store.list_training_samples.return_value = [
            _sample("old", created_at=old_sample_time),
            _sample("new-1", created_at=new_sample_time),
            _sample("new-2", created_at=new_sample_time),
            _sample("new-3", created_at=new_sample_time),
        ]

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_retraining("profile-1")

        assert result["needs_retraining"] is True
        assert any(reason.startswith("new_samples") for reason in result["reasons"])
        assert any(reason.startswith("stale_adapter") for reason in result["reasons"])
        assert any("speaker_sim=0.700" in reason for reason in result["reasons"])
        assert any("mcd=5.20" in reason for reason in result["reasons"])
        assert result["new_sample_count"] == 3
        assert result["days_since_training"] >= 40

    def test_check_needs_retraining_without_previous_training_uses_all_samples(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "quality_metrics": {}}
        store.list_training_samples.return_value = [_sample("s1"), _sample("s2")]

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_retraining("profile-1")

        assert result["needs_retraining"] is False
        assert result["new_sample_count"] == 2
        assert result["days_since_training"] == 0


class TestAutoQueueTraining:
    def test_auto_queue_training_returns_none_when_profile_missing(self, manager):
        store = Mock()
        store.load.side_effect = FileNotFoundError("missing")

        with patch.object(manager, "_get_profile_store", return_value=store):
            assert manager.auto_queue_training("missing") is None

    def test_auto_queue_training_skips_non_target_profiles(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "profile_role": "source_artist"}

        with patch.object(manager, "_get_profile_store", return_value=store):
            assert manager.auto_queue_training("profile-1") is None

    def test_auto_queue_training_skips_when_job_already_pending(self, manager):
        existing = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1"}

        with patch.object(manager, "_get_profile_store", return_value=store):
            job = manager.auto_queue_training("profile-1")

        assert job is None
        assert manager.get_job(existing.job_id) is existing

    def test_auto_queue_training_creates_initial_training_job(self, manager):
        store = Mock()
        profile = {"profile_id": "profile-1", "status": "new"}
        samples = [_sample("s1"), _sample("s2"), _sample("s3"), _sample("s4"), _sample("s5")]
        store.load.return_value = profile
        store.list_training_samples.return_value = samples

        with patch.object(manager, "_get_profile_store", return_value=store), patch.object(
            manager,
            "check_needs_training",
            return_value={"needs_training": True, "reason": "no_adapter", "sample_count": 5, "has_adapter": False},
        ), patch.object(
            manager,
            "check_needs_retraining",
            return_value={"needs_retraining": False, "reasons": []},
        ):
            job = manager.auto_queue_training("profile-1")

        assert job is not None
        assert job.profile_id == "profile-1"
        assert job.sample_ids == [sample.sample_id for sample in samples]
        assert store.save.call_args.args[0]["status"] == "training_queued"

    def test_auto_queue_training_creates_retraining_job(self, manager):
        store = Mock()
        profile = {"profile_id": "profile-1"}
        samples = [_sample("s1"), _sample("s2"), _sample("s3")]
        store.load.return_value = profile
        store.list_training_samples.return_value = samples

        with patch.object(manager, "_get_profile_store", return_value=store), patch.object(
            manager,
            "check_needs_training",
            return_value={"needs_training": False, "reason": "none", "sample_count": 3, "has_adapter": True},
        ), patch.object(
            manager,
            "check_needs_retraining",
            return_value={"needs_retraining": True, "reasons": ["new_samples (3 >= 3)"]},
        ):
            job = manager.auto_queue_training("profile-1")

        assert job is not None
        assert job.sample_ids == [sample.sample_id for sample in samples]
        store.save.assert_not_called()

    def test_auto_queue_training_returns_none_when_no_conditions_match(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1"}

        with patch.object(manager, "_get_profile_store", return_value=store), patch.object(
            manager,
            "check_needs_training",
            return_value={"needs_training": False, "reason": "none", "sample_count": 0, "has_adapter": True},
        ), patch.object(
            manager,
            "check_needs_retraining",
            return_value={"needs_retraining": False, "reasons": []},
        ):
            assert manager.auto_queue_training("profile-1") is None

    def test_auto_queue_all_profiles_collects_created_jobs(self, manager):
        store = Mock()
        store.list_profiles.return_value = [
            {"profile_id": "profile-1"},
            {"name": "skip-no-id"},
            {"profile_id": "profile-2"},
        ]
        queued_job = manager.create_job(profile_id="profile-queued", sample_ids=["s1"])

        with patch.object(manager, "_get_profile_store", return_value=store), patch.object(
            manager,
            "auto_queue_training",
            side_effect=[queued_job, None],
        ):
            result = manager.auto_queue_all_profiles()

        assert result == [queued_job]


class TestFullModelUpgradeLogic:
    def test_check_needs_full_model_profile_not_found(self, manager):
        store = Mock()
        store.load.side_effect = FileNotFoundError("missing")

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_full_model("missing")

        assert result["reason"] == "profile_not_found"
        assert result["remaining_seconds"] == manager.FULL_MODEL_UNLOCK_SECONDS

    def test_check_needs_full_model_requires_target_role_and_duration(self, manager):
        store = Mock()
        store.load.return_value = {
            "profile_id": "profile-1",
            "profile_role": "source_artist",
            "clean_vocal_seconds": 2400.0,
        }
        store.list_training_samples.return_value = [_sample("s1", duration=2400.0)]

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_full_model("profile-1")

        assert result["needs_full_model"] is False
        assert result["reason"] == "unsupported_profile_role (source_artist)"

    def test_check_needs_full_model_reports_insufficient_audio(self, manager):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "clean_vocal_seconds": 1200.0}
        store.list_training_samples.return_value = [_sample("s1", duration=1200.0)]

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_full_model("profile-1")

        assert result["needs_full_model"] is False
        assert result["reason"].startswith("insufficient_clean_vocals")
        assert result["remaining_seconds"] == pytest.approx(600.0)

    @pytest.mark.parametrize(
        ("relative_path", "expected_type", "needs_full_model", "expected_reason"),
        [
            ("profile-1_full_model.pt", "full_model", False, "already_full_model"),
            ("hq/profile-1_hq_lora.pt", "hq_lora", True, "upgrade_recommended"),
            ("nvfp4/profile-1_nvfp4_lora.pt", "nvfp4_lora", True, "upgrade_recommended"),
            ("profile-1_adapter.pt", "standard_lora", True, "upgrade_recommended"),
        ],
    )
    def test_check_needs_full_model_detects_existing_artifact_types(
        self,
        manager,
        relative_path,
        expected_type,
        needs_full_model,
        expected_reason,
    ):
        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "clean_vocal_seconds": 2400.0}
        store.list_training_samples.return_value = [_sample("s1", duration=2400.0)]

        artifact_path = manager._resolve_trained_models_dir() / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("artifact")

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager.check_needs_full_model("profile-1")

        assert result["current_adapter_type"] == expected_type
        assert result["needs_full_model"] is needs_full_model
        assert expected_reason in result["reason"]

    def test_create_full_model_job_raises_for_insufficient_audio(self, manager):
        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={
                "needs_full_model": False,
                "clean_vocal_seconds": 1200.0,
                "reason": "insufficient_clean_vocals",
            },
        ):
            with pytest.raises(ValueError, match="Need at least"):
                manager.create_full_model_job("profile-1")

    def test_create_full_model_job_raises_for_other_terminal_reason(self, manager):
        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={
                "needs_full_model": False,
                "clean_vocal_seconds": 2400.0,
                "reason": "unsupported_profile_role (source_artist)",
            },
        ):
            with pytest.raises(ValueError, match="unsupported_profile_role"):
                manager.create_full_model_job("profile-1")

    def test_create_full_model_job_uses_default_full_training_config(self, manager):
        store = Mock()
        store.list_training_samples.return_value = [_sample("s1"), _sample("s2")]

        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={
                "needs_full_model": True,
                "clean_vocal_seconds": 2400.0,
                "reason": "upgrade_recommended",
            },
        ), patch.object(manager, "_get_profile_store", return_value=store):
            job = manager.create_full_model_job("profile-1")

        assert job.config.training_mode == "full"
        assert job.config.epochs == 50
        assert job.config.learning_rate == 5e-5
        assert job.config.batch_size == 8
        assert job.config.lora_rank == 0
        assert job.results["job_type"] == "full_model"
        assert job.results["sample_count"] == 2

    def test_create_full_model_job_overrides_custom_config_to_full_mode(self, manager):
        store = Mock()
        store.list_training_samples.return_value = [_sample("s1")]
        config = TrainingConfig(training_mode="lora", lora_rank=8, lora_alpha=16)

        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={
                "needs_full_model": True,
                "clean_vocal_seconds": 2400.0,
                "reason": "upgrade_recommended",
            },
        ), patch.object(manager, "_get_profile_store", return_value=store):
            job = manager.create_full_model_job("profile-1", config=config)

        assert job.config.training_mode == "full"
        assert job.config.lora_rank == 0
        assert job.config.lora_alpha == 0

    def test_create_full_model_job_allows_manual_retraining_when_full_model_exists(self, manager):
        store = Mock()
        store.list_training_samples.return_value = [_sample("s1")]

        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={
                "needs_full_model": False,
                "clean_vocal_seconds": 2400.0,
                "reason": "already_full_model",
            },
        ), patch.object(manager, "_get_profile_store", return_value=store):
            job = manager.create_full_model_job(
                "profile-1",
                initialization_mode="continue",
            )

        assert job.config.training_mode == "full"
        assert job.config.initialization_mode == "continue"

    def test_auto_queue_full_model_training_skips_when_not_needed(self, manager):
        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={"needs_full_model": False, "reason": "insufficient_clean_vocals"},
        ):
            assert manager.auto_queue_full_model_training("profile-1") is None

    def test_auto_queue_full_model_training_skips_when_pending_full_model_exists(self, manager):
        job = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        job.results = {"job_type": "full_model"}

        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={"needs_full_model": True, "reason": "upgrade_recommended"},
        ):
            assert manager.auto_queue_full_model_training("profile-1") is None

    def test_auto_queue_full_model_training_creates_job_when_eligible(self, manager):
        created = manager.create_job(profile_id="profile-1", sample_ids=["s1"])

        with patch.object(
            manager,
            "check_needs_full_model",
            return_value={"needs_full_model": True, "reason": "upgrade_recommended"},
        ), patch.object(manager, "create_full_model_job", return_value=created) as create_job:
            result = manager.auto_queue_full_model_training("profile-1")

        assert result is created
        create_job.assert_called_once_with("profile-1")
