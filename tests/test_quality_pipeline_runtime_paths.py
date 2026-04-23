from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "quality_pipeline.py"


def _install_quality_pipeline_stubs(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.device = lambda value: value
    fake_torch.tensor = lambda value: value
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_nn_functional = types.ModuleType("torch.nn.functional")
    fake_torch.nn = fake_torch_nn
    fake_torch_nn.functional = fake_torch_nn_functional

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.load = lambda path, sr=None, mono=True: (np.zeros(1, dtype=np.float32), 16000)
    fake_librosa.resample = lambda audio, orig_sr, target_sr: audio

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.write = lambda path, data, sample_rate: Path(path).write_bytes(b"stub-audio")

    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.functional = types.SimpleNamespace(resample=lambda audio, source_sr, target_sr: audio)
    fake_torchaudio.compliance = types.SimpleNamespace(
        kaldi=types.SimpleNamespace(fbank=lambda *args, **kwargs: np.zeros((1, 80), dtype=np.float32))
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fake_torch_nn_functional)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)


def _load_quality_pipeline(monkeypatch):
    _install_quality_pipeline_stubs(monkeypatch)

    module_name = "quality_pipeline_runtime_paths_test_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_quality_pipeline_runtime_paths_default_to_repo_root(monkeypatch, tmp_path):
    quality_pipeline = _load_quality_pipeline(monkeypatch)
    monkeypatch.chdir(tmp_path)

    paths = quality_pipeline.resolve_runtime_paths()

    assert paths["project_root"] == PROJECT_ROOT
    assert paths["src_dir"] == PROJECT_ROOT / "src"
    assert paths["seed_vc_dir"] == PROJECT_ROOT / "models" / "seed-vc"
    assert paths["seed_vc_checkpoints_dir"] == PROJECT_ROOT / "models" / "seed-vc" / "checkpoints"
    assert quality_pipeline.resolve_cli_path("fixtures/reference.wav") == (
        PROJECT_ROOT / "fixtures" / "reference.wav"
    ).resolve()


def test_quality_pipeline_reference_audio_resolution_stays_repo_root_relative(monkeypatch, tmp_path):
    quality_pipeline = _load_quality_pipeline(monkeypatch)
    external_cwd = tmp_path / "external"
    external_cwd.mkdir()
    monkeypatch.chdir(external_cwd)

    captured: dict[str, Path] = {}

    def fake_load_audio_file(path: Path):
        captured["path"] = path
        return np.zeros(8, dtype=np.float32), 16000

    monkeypatch.setattr(quality_pipeline, "load_audio_file", fake_load_audio_file)

    audio, sample_rate, resolved_path = quality_pipeline.resolve_reference_audio(
        "fixtures/reference.wav",
        None,
    )

    expected = (PROJECT_ROOT / "fixtures" / "reference.wav").resolve()
    assert captured["path"] == expected
    assert resolved_path == expected
    assert sample_rate == 16000
    assert audio.shape == (8,)


def test_quality_pipeline_seed_vc_downloads_use_repo_local_cache(monkeypatch, tmp_path):
    quality_pipeline = _load_quality_pipeline(monkeypatch)
    external_cwd = tmp_path / "external"
    external_cwd.mkdir()
    monkeypatch.chdir(external_cwd)

    captured: list[dict[str, str]] = []

    fake_huggingface_hub = types.ModuleType("huggingface_hub")

    def fake_hf_hub_download(*, repo_id: str, filename: str, cache_dir: str):
        captured.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "cache_dir": cache_dir,
            }
        )
        return str(tmp_path / filename.replace("/", "_"))

    fake_huggingface_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    model_path, config_path = quality_pipeline._download_seed_vc_model_from_hf(
        "seed-vc/test",
        "weights.pth",
        "config.yml",
    )

    expected_cache_dir = str(PROJECT_ROOT / "models" / "seed-vc" / "checkpoints")
    assert model_path.endswith("weights.pth")
    assert config_path.endswith("config.yml")
    assert [call["cache_dir"] for call in captured] == [expected_cache_dir, expected_cache_dir]


def test_quality_pipeline_main_preserves_caller_cwd(monkeypatch, tmp_path):
    quality_pipeline = _load_quality_pipeline(monkeypatch)
    source_path = tmp_path / "source.wav"
    reference_path = tmp_path / "reference.wav"
    output_path = tmp_path / "converted.wav"
    report_dir = tmp_path / "reports"

    captured: dict[str, object] = {}
    audio = np.linspace(-0.5, 0.5, 16, dtype=np.float32)

    class DummyConverter:
        def __init__(self, config):
            del config
            self.last_run_metadata = {}

        def convert(self, source_audio, source_sr, reference_audio, reference_sr, pitch_shift, progress_callback):
            del source_sr, reference_audio, reference_sr, pitch_shift
            progress_callback(1.0, "Complete")
            return source_audio, 16000

        def unload(self):
            captured["unloaded"] = True

    def fake_report_writer(**kwargs):
        captured["report_dir"] = kwargs["report_dir"]

    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    monkeypatch.setattr(
        quality_pipeline,
        "load_audio_file",
        lambda path: (audio, 16000),
    )
    monkeypatch.setattr(quality_pipeline, "QualityVoiceConverter", DummyConverter)
    monkeypatch.setattr(quality_pipeline, "write_quality_report", fake_report_writer)

    exit_code = quality_pipeline.main(
        [
            "--source-audio",
            str(source_path),
            "--reference-audio",
            str(reference_path),
            "--output",
            str(output_path),
            "--report-dir",
            str(report_dir),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert Path.cwd() == foreign_cwd
    assert output_path.exists()
    assert captured["report_dir"] == report_dir.resolve()
    assert captured["unloaded"] is True
