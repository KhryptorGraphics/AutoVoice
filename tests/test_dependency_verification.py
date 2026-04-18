"""Tests for dependency verification helpers."""

from __future__ import annotations

import types

from auto_voice.utils.dependency_verification import (
    DependencySpec,
    check_dependency,
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


def test_default_dependencies_make_tensorrt_optional_by_default():
    tensorrt = next(dep for dep in default_dependencies() if dep.name == "tensorrt")
    assert tensorrt.required is False


def test_default_dependencies_can_require_tensorrt():
    tensorrt = next(
        dep for dep in default_dependencies(require_tensorrt=True) if dep.name == "tensorrt"
    )
    assert tensorrt.required is True


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
