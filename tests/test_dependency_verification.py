"""Tests for dependency verification helpers."""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path

from auto_voice.utils.dependency_verification import (
    DependencySpec,
    check_dependency,
    check_python_environment,
    default_dependencies,
    infer_env_name,
)


def test_infer_env_name_from_conda_path():
    assert (
        infer_env_name("/home/kp/anaconda3/envs/autovoice-thor/bin/python")
        == "autovoice-thor"
    )


def test_infer_env_name_returns_none_without_env_segment():
    assert infer_env_name("/usr/bin/python3") is None


def test_check_python_environment_prefers_configured_env_prefix(monkeypatch, tmp_path):
    expected_prefix = tmp_path / "envs" / "custom-autovoice"
    expected_python = expected_prefix / "bin" / "python"
    expected_python.parent.mkdir(parents=True, exist_ok=True)
    expected_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("AUTOVOICE_ENV_NAME", "custom-autovoice")
    monkeypatch.setenv("AUTOVOICE_ENV_PREFIX", str(expected_prefix))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "custom-autovoice")

    status = check_python_environment(executable=str(expected_python))

    assert status["expected_env_name"] == "custom-autovoice"
    assert status["expected_env_prefix"] == str(expected_prefix)
    assert status["expected_executable"] == str(expected_python)
    assert status["matches_expected_env"] is True
    assert status["matches_expected_executable"] is True


def test_check_python_environment_uses_current_prefix_when_path_matches(monkeypatch, tmp_path):
    executable = tmp_path / "miniconda3" / "envs" / "autovoice-thor" / "bin" / "python"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("", encoding="utf-8")

    monkeypatch.delenv("AUTOVOICE_ENV_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    status = check_python_environment(executable=str(executable))

    assert status["expected_env_prefix"] == str(Path(executable).parent.parent)
    assert status["matches_expected_env"] is True
    assert status["matches_expected_executable"] is True


def test_default_dependencies_make_tensorrt_optional_by_default():
    tensorrt = next(dep for dep in default_dependencies() if dep.name == "tensorrt")
    assert tensorrt.required is False


def test_default_dependencies_can_require_tensorrt():
    tensorrt = next(
        dep for dep in default_dependencies(require_tensorrt=True) if dep.name == "tensorrt"
    )
    assert tensorrt.required is True


def test_export_imports_when_tensorrt_is_missing(monkeypatch):
    """TensorRT is optional; ONNX exports must remain importable without it."""
    module_names = ["auto_voice.export", "auto_voice.export.tensorrt_engine"]
    original_modules = {name: sys.modules.pop(name, None) for name in module_names}
    auto_voice_pkg = sys.modules.get("auto_voice")
    original_export_attr = (
        getattr(auto_voice_pkg, "export", None) if auto_voice_pkg else None
    )
    had_export_attr = auto_voice_pkg is not None and hasattr(auto_voice_pkg, "export")
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorrt":
            raise ImportError("No module named 'tensorrt'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    try:
        tensorrt_engine = importlib.import_module("auto_voice.export.tensorrt_engine")
        export_package = importlib.import_module("auto_voice.export")

        assert tensorrt_engine.TRT_AVAILABLE is False
        assert tensorrt_engine.TRT_LOGGER is None
        assert export_package._TENSORRT_AVAILABLE is False
        assert export_package.export_content_encoder is not None
        assert export_package.TRTEngineBuilder is tensorrt_engine.TRTEngineBuilder
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module
        if auto_voice_pkg is not None:
            if had_export_attr:
                setattr(auto_voice_pkg, "export", original_export_attr)
            elif hasattr(auto_voice_pkg, "export"):
                delattr(auto_voice_pkg, "export")


def test_check_dependency_reports_success(monkeypatch):
    fake_module = types.SimpleNamespace(__version__="1.2.3")

    monkeypatch.setattr(
        "auto_voice.utils.dependency_verification.import_module",
        lambda _name: fake_module,
    )

    result = check_dependency(DependencySpec(name="demo", import_path="demo"))
    assert result["ok"] is True
    assert result["version"] == "1.2.3"
    assert result["error"] is None


def test_check_dependency_reports_probe_failure(monkeypatch):
    fake_module = types.SimpleNamespace(__version__="9.9.9")

    monkeypatch.setattr(
        "auto_voice.utils.dependency_verification.import_module",
        lambda _name: fake_module,
    )

    result = check_dependency(
        DependencySpec(
            name="demo",
            import_path="demo",
            probe=lambda _module: (_ for _ in ()).throw(RuntimeError("probe failed")),
        )
    )

    assert result["ok"] is False
    assert "probe failed" in result["error"]
