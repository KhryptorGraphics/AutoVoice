from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_youtube_pipeline_stubs(monkeypatch):
    monkeypatch.syspath_prepend(str(SRC_DIR))

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_youtube = types.ModuleType("auto_voice.youtube")
    fake_youtube.download_artist_videos = lambda *args, **kwargs: []
    fake_youtube.scrape_artist_channel = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "auto_voice.youtube", fake_youtube)

    fake_separator = types.ModuleType("auto_voice.audio.separation")

    class VocalSeparator:
        def __init__(self, *args, **kwargs):
            pass

    fake_separator.VocalSeparator = VocalSeparator
    monkeypatch.setitem(sys.modules, "auto_voice.audio.separation", fake_separator)

    fake_storage_pkg = types.ModuleType("auto_voice.storage")
    fake_storage_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "auto_voice.storage", fake_storage_pkg)

    fake_storage_paths = types.ModuleType("auto_voice.storage.paths")
    def resolve_data_dir(explicit_data_dir=None):
        return Path(explicit_data_dir or "data")

    fake_storage_paths.resolve_data_dir = resolve_data_dir
    fake_storage_paths.resolve_youtube_audio_dir = (
        lambda explicit_dir=None, *, data_dir=None, artist_name=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "youtube_audio" / artist_name
            if artist_name
            else resolve_data_dir(data_dir) / "youtube_audio"
        )
    )
    fake_storage_paths.resolve_separated_audio_dir = (
        lambda explicit_dir=None, *, data_dir=None, artist_name=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "separated_youtube" / artist_name
            if artist_name
            else resolve_data_dir(data_dir) / "separated_youtube"
        )
    )
    fake_storage_paths.resolve_diarized_audio_dir = (
        lambda explicit_dir=None, *, data_dir=None, artist_name=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "diarized_youtube" / artist_name
            if artist_name
            else resolve_data_dir(data_dir) / "diarized_youtube"
        )
    )
    fake_storage_paths.resolve_training_vocals_dir = (
        lambda explicit_dir=None, *, data_dir=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "training_vocals"
        )
    )
    monkeypatch.setitem(sys.modules, "auto_voice.storage.paths", fake_storage_paths)


def _install_extract_stubs(monkeypatch):
    monkeypatch.syspath_prepend(str(SRC_DIR))

    fake_librosa = types.ModuleType("librosa")
    fake_soundfile = types.ModuleType("soundfile")
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    fake_storage_pkg = types.ModuleType("auto_voice.storage")
    fake_storage_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "auto_voice.storage", fake_storage_pkg)

    fake_storage_paths = types.ModuleType("auto_voice.storage.paths")
    def resolve_data_dir(explicit_data_dir=None):
        return Path(explicit_data_dir or "data")

    fake_storage_paths.resolve_data_dir = resolve_data_dir
    fake_storage_paths.resolve_diarized_audio_dir = (
        lambda explicit_dir=None, *, data_dir=None, artist_name=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "diarized_youtube" / artist_name
            if artist_name
            else resolve_data_dir(data_dir) / "diarized_youtube"
        )
    )
    fake_storage_paths.resolve_separated_audio_dir = (
        lambda explicit_dir=None, *, data_dir=None, artist_name=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "separated_youtube" / artist_name
            if artist_name
            else resolve_data_dir(data_dir) / "separated_youtube"
        )
    )
    fake_storage_paths.resolve_training_vocals_dir = (
        lambda explicit_dir=None, *, data_dir=None: (
            Path(explicit_dir)
            if explicit_dir
            else resolve_data_dir(data_dir) / "training_vocals"
        )
    )
    monkeypatch.setitem(sys.modules, "auto_voice.storage.paths", fake_storage_paths)


def test_youtube_artist_pipeline_runtime_paths_default_to_repo_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _install_youtube_pipeline_stubs(monkeypatch)

    module = _load_module(
        "test_youtube_artist_pipeline_runtime_paths",
        SCRIPTS_DIR / "youtube_artist_pipeline.py",
    )
    paths = module.resolve_runtime_paths()
    artist_paths = module.resolve_artist_paths("conor_maynard")

    assert paths["data_dir"] == PROJECT_ROOT / "data"
    assert paths["audio_root"] == PROJECT_ROOT / "data" / "youtube_audio"
    assert paths["separated_root"] == PROJECT_ROOT / "data" / "separated_youtube"
    assert paths["diarized_root"] == PROJECT_ROOT / "data" / "diarized_youtube"
    assert artist_paths["audio_dir"] == PROJECT_ROOT / "data" / "youtube_audio" / "conor_maynard"


def test_extract_diarized_vocals_runtime_paths_default_to_repo_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _install_extract_stubs(monkeypatch)

    module = _load_module(
        "test_extract_diarized_vocals_runtime_paths",
        SCRIPTS_DIR / "extract_diarized_vocals.py",
    )
    paths = module.resolve_runtime_paths()

    assert paths["data_dir"] == PROJECT_ROOT / "data"
    assert paths["diarized_root"] == PROJECT_ROOT / "data" / "diarized_youtube"
    assert paths["separated_root"] == PROJECT_ROOT / "data" / "separated_youtube"
    assert paths["training_vocals_dir"] == PROJECT_ROOT / "data" / "training_vocals"
