"""Focused coverage for TrainingJobManager execution and persistence branches."""

from __future__ import annotations

import json
import logging
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from auto_voice.training.job_manager import (
    JobStatus,
    TrainingConfig,
    TrainingJob,
    TrainingJobManager,
)


def _sample(sample_id: str, path: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        sample_id=sample_id,
        vocals_path=str(path),
    )


class _ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target()


def _fake_training_modules(*, fail_in_train: bool = False):
    trainer_instances = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._lora_injected = False
            self.lora_calls = []

        def inject_lora(self, rank, alpha, dropout):
            self._lora_injected = True
            self.lora_calls.append((rank, alpha, dropout))

        def state_dict(self):
            return {"weights": torch.tensor([1.0])}

        def get_lora_state_dict(self):
            return {
                "layer.lora_A": torch.ones(1),
                "layer.lora_B": torch.ones(1),
            }

    class FakeTrainer:
        def __init__(self, model, config, device):
            self.model = model
            self.config = config
            self.device = device
            self.checkpoint_dir = Path(config["checkpoint_dir"])
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (self.checkpoint_dir / "final.pth").write_bytes(b"final")
            self.train_losses = []
            self.best_loss = float("inf")
            self.speaker_embedding = torch.nn.functional.normalize(torch.ones(256), dim=0)
            self.embedding_dir = None
            self._train_epoch = lambda _loader, epoch: 0.5 - (epoch * 0.1)
            trainer_instances.append(self)

        def set_speaker_embedding(self, training_dir):
            self.embedding_dir = training_dir

        def train(self, training_dir):
            if fail_in_train:
                raise RuntimeError("train failed")
            for epoch in range(self.config["epochs"]):
                loss = self._train_epoch(None, epoch)
                self.train_losses.append(loss)
                self.best_loss = min(self.best_loss, loss)

    decoder_module = types.ModuleType("auto_voice.models.svc_decoder")
    decoder_module.CoMoSVCDecoder = FakeModel
    trainer_module = types.ModuleType("auto_voice.training.trainer")
    trainer_module.Trainer = FakeTrainer

    return {
        "auto_voice.models.svc_decoder": decoder_module,
        "auto_voice.training.trainer": trainer_module,
    }, trainer_instances


@pytest.fixture
def manager(tmp_path):
    return TrainingJobManager(storage_path=tmp_path / "jobs", require_gpu=False)


class TestTrainingJobHelpers:
    def test_training_job_start_complete_fail_and_round_trip(self):
        job = TrainingJob(job_id="job-1", profile_id="profile-1")

        job.start(gpu_device=2)
        assert job.status == JobStatus.RUNNING.value
        assert job.gpu_device == 2
        assert job.started_at is not None

        job.complete({"loss": 0.1})
        assert job.status == JobStatus.COMPLETED.value
        assert job.progress == 100
        assert job.results == {"loss": 0.1}

        data = job.to_dict()
        restored = TrainingJob.from_dict(data)
        assert restored.completed_at == job.completed_at
        assert restored.gpu_device == 2
        assert restored.results == {"loss": 0.1}

        failed = TrainingJob(job_id="job-2", profile_id="profile-2")
        failed.fail("boom")
        assert failed.status == JobStatus.FAILED.value
        assert failed.error == "boom"
        assert failed.completed_at is not None

    @pytest.mark.parametrize(
        ("kwargs", "expected_data_dir"),
        [
            ({"profiles_dir": "/tmp/data/profiles"}, Path("/tmp/data")),
            ({"samples_dir": "/tmp/data/samples"}, Path("/tmp/data")),
            ({"storage_path": Path("/tmp/root/app_state")}, Path("/tmp/root")),
        ],
    )
    def test_manager_data_dir_resolution_variants(self, tmp_path, kwargs, expected_data_dir):
        params = {"storage_path": tmp_path / "jobs", "require_gpu": False}
        params.update(kwargs)
        manager = TrainingJobManager(**params)
        assert manager._data_dir == expected_data_dir

    def test_load_jobs_warns_on_invalid_json(self, tmp_path, caplog):
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        (jobs_dir / TrainingJobManager.JOBS_FILENAME).write_text("{not-json")

        with caplog.at_level(logging.WARNING):
            TrainingJobManager(storage_path=jobs_dir, require_gpu=False)

        assert "Failed to load jobs" in caplog.text

    def test_save_jobs_logs_errors(self, manager, caplog):
        manager._jobs["job-1"] = TrainingJob(job_id="job-1", profile_id="profile-1")

        with patch("builtins.open", side_effect=OSError("disk full")), caplog.at_level(logging.ERROR):
            manager._save_jobs()

        assert "Failed to save jobs" in caplog.text

    def test_list_jobs_filters_and_sorts(self, manager):
        first = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        second = manager.create_job(profile_id="profile-2", sample_ids=["s2"])
        first.created_at = datetime(2026, 1, 1)
        second.created_at = datetime(2026, 1, 2)

        assert [job.job_id for job in manager.list_jobs(profile_id="profile-1")] == [first.job_id]
        assert [job.job_id for job in manager.list_jobs()] == [second.job_id, first.job_id]

    def test_cancel_job_returns_false_when_missing(self, manager):
        assert manager.cancel_job("missing") is False

    def test_update_job_status_raises_for_missing_job(self, manager):
        with pytest.raises(ValueError, match="Job missing not found"):
            manager.update_job_status("missing", JobStatus.RUNNING.value)

    def test_emit_event_swallows_socketio_errors(self, tmp_path, caplog):
        socketio = MagicMock()
        socketio.emit.side_effect = RuntimeError("emit failed")
        manager = TrainingJobManager(storage_path=tmp_path / "jobs", require_gpu=False, socketio=socketio)

        with caplog.at_level(logging.DEBUG):
            manager._emit_event("training.started", {"job_id": "job-1"})

        assert "Failed to emit training.started" in caplog.text

    def test_mark_job_completed_ignores_missing_job(self, manager):
        manager._mark_job_completed("missing", results={"loss": 0.1})
        assert manager.get_job("missing") is None

    def test_get_profile_store_uses_resolved_paths(self, manager):
        fake_store = object()
        with patch("auto_voice.storage.voice_profiles.VoiceProfileStore", return_value=fake_store) as store_cls:
            result = manager._get_profile_store()

        assert result is fake_store
        store_cls.assert_called_once_with(
            profiles_dir=str(manager._resolve_profiles_dir()),
            samples_dir=str(manager._resolve_samples_dir()),
        )


class TestSaveTrainedAdapter:
    def test_save_trained_adapter_raises_when_lora_mode_missing_injected_adapters(self, manager):
        speaker_embedding = torch.ones(256)
        trainer = types.SimpleNamespace(
            model=types.SimpleNamespace(
                _lora_injected=False,
                state_dict=lambda: {"weights": torch.tensor([1.0])},
            ),
            speaker_embedding=speaker_embedding,
        )

        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "name": "Profile 1"}

        with patch.object(manager, "_get_profile_store", return_value=store), \
             pytest.raises(RuntimeError, match="requested LoRA output but the model has no injected LoRA adapters"):
            manager._save_trained_adapter(trainer=trainer, profile_id="profile-1", job_id="job-1")

    def test_save_trained_adapter_saves_explicit_full_model_manifest(self, manager, tmp_path):
        speaker_embedding = torch.ones(256)
        trainer = types.SimpleNamespace(
            model=types.SimpleNamespace(
                _lora_injected=False,
                state_dict=lambda: {"weights": torch.tensor([1.0])},
                n_mels=80,
            ),
            speaker_embedding=speaker_embedding,
            sample_rate=22050,
            checkpoint_dir=tmp_path / "checkpoints",
            config={"n_mels": 80},
        )
        trainer.checkpoint_dir.mkdir(parents=True)
        (trainer.checkpoint_dir / "final.pth").write_bytes(b"final")

        store = Mock()
        store.load.return_value = {"profile_id": "profile-1", "name": "Profile 1"}
        store.save_runtime_artifact_manifest.return_value = str(tmp_path / "artifact_manifest.json")

        with patch.object(manager, "_get_profile_store", return_value=store):
            result = manager._save_trained_adapter(
                trainer=trainer,
                profile_id="profile-1",
                job_id="job-1",
                training_mode="full",
            )

        assert result["artifact_type"] == "full_model"
        assert Path(result["adapter_path"]).exists()
        embedding = np.load(result["embedding_path"])
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
        store.save_runtime_artifact_manifest.assert_called_once()

    def test_save_trained_adapter_saves_and_validates_lora_artifacts(self, manager, tmp_path):
        speaker_embedding = torch.nn.functional.normalize(torch.ones(256), dim=0)
        trainer = types.SimpleNamespace(
            model=types.SimpleNamespace(
                _lora_injected=True,
                _lora_config={"rank": 8, "alpha": 16, "target_modules": ["q_proj"]},
                n_mels=80,
                get_lora_state_dict=lambda: {
                    "layer.lora_A": torch.ones(1),
                    "layer.lora_B": torch.ones(1),
                },
            ),
            speaker_embedding=speaker_embedding,
            sample_rate=22050,
            checkpoint_dir=tmp_path / "checkpoints",
            config={"n_mels": 80},
        )
        trainer.checkpoint_dir.mkdir(parents=True)
        (trainer.checkpoint_dir / "final.pth").write_bytes(b"final")

        store = Mock()
        store.load.return_value = {"profile_id": "profile-2", "name": "Profile 2"}
        store.save_runtime_artifact_manifest.return_value = str(tmp_path / "artifact_manifest.json")

        with patch.object(manager, "_get_profile_store", return_value=store), \
             patch("auto_voice.models.adapter_manager.AdapterManager") as manager_cls:
            manager_cls.return_value.load_adapter.return_value = {
                "layer.lora_A": torch.ones(1),
                "layer.lora_B": torch.ones(1),
            }
            result = manager._save_trained_adapter(trainer=trainer, profile_id="profile-2", job_id="job-2")

        assert result["artifact_type"] == "adapter"
        assert Path(result["adapter_path"]).exists()
        assert Path(result["embedding_path"]).exists()
        saved_payload = torch.load(result["adapter_path"], map_location="cpu", weights_only=False)
        assert "__autovoice_lora_metadata__" in saved_payload
        store.save_runtime_artifact_manifest.assert_called_once()

    def test_save_trained_adapter_wraps_validation_errors(self, manager):
        trainer = types.SimpleNamespace(
            model=types.SimpleNamespace(
                _lora_injected=True,
                _lora_config={},
                n_mels=80,
                get_lora_state_dict=lambda: {
                    "layer.lora_A": torch.ones(1),
                    "layer.lora_B": torch.ones(1),
                },
            ),
            speaker_embedding=torch.ones(128),
            sample_rate=22050,
            checkpoint_dir=Path("/tmp/checkpoints"),
            config={"n_mels": 80},
        )

        store = Mock()
        store.load.return_value = {"profile_id": "profile-3", "name": "Profile 3"}

        with patch.object(manager, "_get_profile_store", return_value=store), \
             pytest.raises(RuntimeError, match="Adapter save failed"):
            manager._save_trained_adapter(trainer=trainer, profile_id="profile-3", job_id="job-3")


class TestExecuteJobPaths:
    def test_execute_job_raises_for_missing_job(self, manager):
        with patch("torch.cuda.is_available", return_value=True):
            with pytest.raises(ValueError, match="Job missing not found"):
                manager.execute_job("missing")

    def test_execute_job_raises_for_non_pending_job(self, manager):
        job = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        job.status = JobStatus.RUNNING.value

        with patch("torch.cuda.is_available", return_value=True):
            with pytest.raises(ValueError, match="is not in pending state"):
                manager.execute_job(job.job_id)

    def test_execute_job_success_runs_inline_and_falls_back_to_lora_for_unknown_mode(self, manager, tmp_path):
        source = tmp_path / "sample.wav"
        source.write_bytes(b"wav")
        job = manager.create_job(
            profile_id="profile-1",
            sample_ids=["sample-1"],
            config=TrainingConfig(training_mode="mystery", epochs=2, batch_size=3, learning_rate=2e-4),
        )
        store = Mock()
        store.list_training_samples.return_value = [_sample("sample-1", source)]

        fake_modules, trainer_instances = _fake_training_modules()
        saved_artifacts = {
            "adapter_path": str(tmp_path / "profile-1_adapter.pt"),
            "embedding_path": str(tmp_path / "profile-1.npy"),
            "artifact_type": "adapter",
        }

        with patch.object(manager, "_get_profile_store", return_value=store), \
             patch.object(manager, "_save_trained_adapter", return_value=saved_artifacts) as save_adapter, \
             patch.object(manager, "_update_profile_training_state") as update_profile, \
             patch.object(manager, "_emit_started_event") as emit_started, \
             patch.object(manager, "_emit_completed_event") as emit_completed, \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "training")), \
             patch("threading.Thread", _ImmediateThread), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.dict(sys.modules, fake_modules):
            (tmp_path / "training").mkdir(exist_ok=True)
            manager.execute_job(job.job_id)

        assert job.status == JobStatus.COMPLETED.value
        assert job.results["job_type"] == "lora"
        assert job.results["epochs_completed"] == 2
        assert trainer_instances[0].model.lora_calls == [(8, 16, 0.1)]
        save_adapter.assert_called_once()
        update_profile.assert_called_once_with(
            profile_id="profile-1",
            results=job.results,
            sample_count=1,
        )
        assert emit_started.call_count == 1
        assert emit_completed.call_count == 1

    def test_execute_job_fails_when_no_training_samples_exist(self, manager):
        job = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        store = Mock()
        store.list_training_samples.return_value = []

        with patch.object(manager, "_get_profile_store", return_value=store), \
             patch.object(manager, "_emit_failed_event") as emit_failed, \
             patch("torch.cuda.is_available", return_value=True), \
             pytest.raises(ValueError, match="No training samples found"):
            manager.execute_job(job.job_id)

        assert job.status == JobStatus.FAILED.value
        emit_failed.assert_called_once()

    def test_execute_job_fails_when_selected_audio_files_do_not_exist(self, manager, tmp_path):
        job = manager.create_job(profile_id="profile-1", sample_ids=["s1"])
        store = Mock()
        store.list_training_samples.return_value = [_sample("s1", tmp_path / "missing.wav")]

        with patch.object(manager, "_get_profile_store", return_value=store), \
             patch.object(manager, "_emit_failed_event") as emit_failed, \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "training")), \
             patch("torch.cuda.is_available", return_value=True), \
             pytest.raises(ValueError, match="No valid audio samples could be loaded"):
            (tmp_path / "training").mkdir(exist_ok=True)
            manager.execute_job(job.job_id)

        assert job.status == JobStatus.FAILED.value
        emit_failed.assert_called_once()

    def test_execute_job_marks_profile_failed_when_training_thread_errors(self, manager, tmp_path):
        source = tmp_path / "sample.wav"
        source.write_bytes(b"wav")
        job = manager.create_job(profile_id="profile-1", sample_ids=["sample-1"], config=TrainingConfig(epochs=1))
        store = Mock()
        store.list_training_samples.return_value = [_sample("sample-1", source)]
        fake_modules, _ = _fake_training_modules(fail_in_train=True)

        with patch.object(manager, "_get_profile_store", return_value=store), \
             patch.object(manager, "_mark_profile_training_failed") as mark_failed, \
             patch.object(manager, "_emit_failed_event") as emit_failed, \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "training")), \
             patch("threading.Thread", _ImmediateThread), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.dict(sys.modules, fake_modules):
            (tmp_path / "training").mkdir(exist_ok=True)
            manager.execute_job(job.job_id)

        assert job.status == JobStatus.FAILED.value
        assert job.error == "train failed"
        mark_failed.assert_called_once_with("profile-1", "train failed")
        emit_failed.assert_called_once()
