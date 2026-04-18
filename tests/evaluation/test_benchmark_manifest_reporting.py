import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from auto_voice.evaluation import BenchmarkDataset, BenchmarkRunner


def _mock_torchaudio_load(path_str, *args, **kwargs):
    audio, sample_rate = sf.read(path_str)
    tensor = torch.tensor(audio, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.T
    return tensor, sample_rate


@pytest.fixture(autouse=True)
def mock_torchaudio_load():
    with patch("torchaudio.load", side_effect=_mock_torchaudio_load):
        yield


@pytest.fixture
def manifest_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "pillowtalk"
    holdout_dir = dataset_dir / "holdout" / "william_to_conor"
    train_dir = dataset_dir / "train" / "conor_to_william"
    speakers_dir = dataset_dir / "speakers"
    holdout_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    speakers_dir.mkdir(parents=True)

    sample_rate = 24000
    duration = 1.0
    time_axis = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    william = 0.5 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
    conor = 0.5 * np.sin(2 * np.pi * 330 * time_axis).astype(np.float32)

    sf.write(holdout_dir / "source.wav", william, sample_rate)
    sf.write(holdout_dir / "reference.wav", conor, sample_rate)
    sf.write(train_dir / "source.wav", conor, sample_rate)
    sf.write(train_dir / "reference.wav", william, sample_rate)

    np.save(speakers_dir / "conor.npy", np.ones(256, dtype=np.float32))
    np.save(speakers_dir / "william.npy", np.full(256, 0.5, dtype=np.float32))

    manifest = {
        "dataset_name": "pillowtalk",
        "defaults": {"sample_rate": sample_rate},
        "samples": [
            {
                "sample_id": "william_to_conor_holdout",
                "split": "holdout",
                "source_audio": "holdout/william_to_conor/source.wav",
                "reference_audio": "holdout/william_to_conor/reference.wav",
                "target_speaker_embedding": "speakers/conor.npy",
                "metadata": {
                    "song": "Pillowtalk",
                    "source_artist": "william_singe",
                    "target_artist": "conor_maynard",
                },
            },
            {
                "sample_id": "conor_to_william_train",
                "split": "train",
                "source_audio": "train/conor_to_william/source.wav",
                "reference_audio": "train/conor_to_william/reference.wav",
                "target_speaker_embedding": "speakers/william.npy",
                "metadata": {
                    "song": "Pillowtalk",
                    "source_artist": "conor_maynard",
                    "target_artist": "william_singe",
                },
            },
        ],
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(manifest))
    return dataset_dir


def test_manifest_dataset_filters_split_and_loads_embeddings(manifest_dataset_dir):
    dataset = BenchmarkDataset(
        data_dir=str(manifest_dataset_dir),
        sample_rate=24000,
        split="holdout",
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["sample_id"] == "william_to_conor_holdout"
    assert sample["metadata"]["song"] == "Pillowtalk"
    assert sample["reference_path"].endswith("holdout/william_to_conor/reference.wav")
    assert isinstance(sample["target_speaker"], torch.Tensor)
    assert sample["target_speaker"].shape[0] == 256


class DummyPipeline:
    def convert(self, source_audio, sample_rate, target_speaker):
        del target_speaker
        return {"audio": source_audio * 0.98}


def test_benchmark_runner_writes_report_bundle(manifest_dataset_dir, tmp_path):
    dataset = BenchmarkDataset(
        data_dir=str(manifest_dataset_dir),
        sample_rate=24000,
        split="holdout",
    )
    runner = BenchmarkRunner()

    report = runner.benchmark(
        pipeline=DummyPipeline(),
        dataset=dataset,
        output_dir=str(tmp_path / "benchmark_results" / "run_001"),
        title="Pillowtalk Holdout",
    )

    artifacts = report["artifacts"]
    assert Path(artifacts["summary_json"]).exists()
    assert Path(artifacts["metrics_csv"]).exists()
    assert Path(artifacts["report_md"]).exists()

    figure_paths = artifacts["figures"]
    assert Path(figure_paths["aggregate_metrics"]).exists()
    assert Path(figure_paths["speaker_similarity_distribution"]).exists()
    assert Path(figure_paths["pitch_contour_overlay"]).exists()
    assert Path(figure_paths["spectrogram_comparison"]).exists()

    tensorboard_dir = Path(artifacts["tensorboard_dir"])
    assert any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir())

    summary = report["summary"]
    assert "mcd_mean" in summary
    assert "pitch_corr_mean" in summary
    assert "speaker_similarity_mean" in summary
