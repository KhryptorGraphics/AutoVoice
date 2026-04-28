"""Dependency and runtime verification helpers for the AutoVoice environment."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import os
import sys


DEFAULT_EXPECTED_ENV_NAME = "autovoice-thor"


@dataclass(frozen=True)
class DependencySpec:
    """Specification for one dependency check."""

    name: str
    import_path: str
    required: bool = True
    version_attr: str = "__version__"
    probe: Optional[Callable[[Any], None]] = None
    support_boundary: str = "supported"
    owner: str = "backend-runtime"
    action: str = "Install the runtime dependency in the canonical AutoVoice environment."


def _probe_pyworld(module: Any) -> None:
    """Run a tiny PyWorld inference to catch broken ARM64 builds."""

    module.dio(np.zeros(16000, dtype=np.float64), 16000)


def _probe_tensorrt(module: Any) -> None:
    """Touch a common TensorRT symbol so import-only stubs are caught."""

    getattr(module, "Logger")


def default_dependencies(require_tensorrt: bool = False) -> List[DependencySpec]:
    """Return the default dependency matrix for AutoVoice."""

    return [
        DependencySpec("torch", "torch"),
        DependencySpec("torchaudio", "torchaudio"),
        DependencySpec("librosa", "librosa"),
        DependencySpec("soundfile", "soundfile"),
        DependencySpec("transformers", "transformers"),
        DependencySpec("demucs", "demucs"),
        DependencySpec("flask", "flask"),
        DependencySpec("flask_socketio", "flask_socketio"),
        DependencySpec("flask_swagger_ui", "flask_swagger_ui"),
        DependencySpec("sqlalchemy", "sqlalchemy"),
        DependencySpec("pynvml", "pynvml"),
        DependencySpec("pyworld", "pyworld", probe=_probe_pyworld),
        DependencySpec("pystoi", "pystoi"),
        DependencySpec("pesq", "pesq"),
        DependencySpec(
            "fairseq",
            "fairseq",
            required=False,
            support_boundary="experimental:hq_svc",
            owner="model-runtime",
            action=(
                "Install the HQ-SVC experimental dependency set and rerun with "
                "AUTOVOICE_HQSVC_FULL=1 on CUDA hardware."
            ),
        ),
        DependencySpec(
            "local_attention",
            "local_attention",
            required=False,
            support_boundary="experimental:hq_svc",
            owner="model-runtime",
            action=(
                "Install the HQ-SVC experimental dependency set before enabling "
                "AUTOVOICE_HQSVC_FULL=1."
            ),
        ),
        DependencySpec(
            "tensorrt",
            "tensorrt",
            required=require_tensorrt,
            probe=_probe_tensorrt,
        ),
    ]


def infer_env_name(executable: str) -> Optional[str]:
    """Infer the conda environment name from an absolute interpreter path."""

    path = Path(executable)
    parts = path.parts
    if "envs" not in parts:
        return None

    idx = parts.index("envs")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def resolve_expected_env_name(expected_env_name: Optional[str] = None) -> str:
    """Resolve the canonical AutoVoice environment name."""

    candidate = (
        expected_env_name
        or os.environ.get("AUTOVOICE_ENV_NAME")
        or DEFAULT_EXPECTED_ENV_NAME
    )
    return str(candidate).strip() or DEFAULT_EXPECTED_ENV_NAME


def resolve_expected_env_prefix(
    expected_env_name: Optional[str] = None,
    *,
    executable: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the expected interpreter prefix without assuming one machine layout."""

    configured_prefix = os.environ.get("AUTOVOICE_ENV_PREFIX")
    if configured_prefix:
        return Path(configured_prefix).expanduser()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
    resolved_env_name = resolve_expected_env_name(expected_env_name)
    if conda_prefix and conda_env_name == resolved_env_name:
        return Path(conda_prefix).expanduser()

    resolved_executable = Path(executable or sys.executable).resolve()
    inferred_name = infer_env_name(str(resolved_executable))
    if inferred_name and inferred_name == resolved_env_name:
        return resolved_executable.parent.parent

    home = Path.home()
    if home:
        return home / "anaconda3" / "envs" / resolved_env_name
    return None


def check_python_environment(
    executable: Optional[str] = None,
    expected_env_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect environment facts for the current Python interpreter."""

    resolved = Path(executable or sys.executable).resolve()
    resolved_env_name = resolve_expected_env_name(expected_env_name)
    inferred_name = infer_env_name(str(resolved))
    expected_prefix = resolve_expected_env_prefix(resolved_env_name, executable=str(resolved))
    expected_executable = expected_prefix / "bin" / "python" if expected_prefix else None
    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")
    matches_expected_env = inferred_name == resolved_env_name or conda_default_env == resolved_env_name
    matches_expected_executable = (
        resolved == expected_executable.resolve()
        if expected_executable and expected_executable.exists()
        else matches_expected_env
    )

    return {
        "executable": str(resolved),
        "version": sys.version.split()[0],
        "expected_env_name": resolved_env_name,
        "expected_env_prefix": str(expected_prefix) if expected_prefix else None,
        "expected_executable": str(expected_executable) if expected_executable else None,
        "conda_default_env": conda_default_env,
        "inferred_env_name": inferred_name,
        "matches_expected_env": matches_expected_env,
        "matches_expected_executable": matches_expected_executable,
    }


def check_dependency(spec: DependencySpec) -> Dict[str, Any]:
    """Import and verify one dependency."""

    try:
        module = import_module(spec.import_path)
        version = getattr(module, spec.version_attr, None)
        if spec.probe is not None:
            spec.probe(module)
        return {
            "name": spec.name,
            "import_path": spec.import_path,
            "required": spec.required,
            "support_boundary": spec.support_boundary,
            "owner": spec.owner,
            "action": spec.action,
            "ok": True,
            "version": version,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - exercised via tests with fakes
        return {
            "name": spec.name,
            "import_path": spec.import_path,
            "required": spec.required,
            "support_boundary": spec.support_boundary,
            "owner": spec.owner,
            "action": spec.action,
            "ok": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def get_cuda_status() -> Dict[str, Any]:
    """Collect Torch/CUDA runtime status."""

    try:
        torch = import_module("torch")
    except Exception as exc:  # pragma: no cover - import failure path
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_name": None,
            "compute_capability": None,
            "memory_free_gb": None,
            "memory_total_gb": None,
        }

    status: Dict[str, Any] = {
        "ok": True,
        "error": None,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_name": None,
        "compute_capability": None,
        "memory_free_gb": None,
        "memory_total_gb": None,
    }

    if status["cuda_available"]:
        try:
            status["device_name"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            status["compute_capability"] = f"{capability[0]}.{capability[1]}"
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gb = 1024 ** 3
            status["memory_free_gb"] = round(free_bytes / gb, 2)
            status["memory_total_gb"] = round(total_bytes / gb, 2)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            status["ok"] = False
            status["error"] = f"{type(exc).__name__}: {exc}"

    return status


def run_dependency_audit(require_tensorrt: bool = False) -> Dict[str, Any]:
    """Run the full dependency audit."""

    python_status = check_python_environment()
    dependencies = [
        check_dependency(spec) for spec in default_dependencies(require_tensorrt=require_tensorrt)
    ]

    failed_required = [
        item["name"] for item in dependencies if item["required"] and not item["ok"]
    ]
    failed_optional = [
        item["name"] for item in dependencies if not item["required"] and not item["ok"]
    ]

    return {
        "python": python_status,
        "cuda": get_cuda_status(),
        "dependencies": dependencies,
        "failed_required": failed_required,
        "failed_optional": failed_optional,
        "ok": not failed_required,
    }


def format_audit(audit: Dict[str, Any]) -> str:
    """Format an audit report for CLI output."""

    python_status = audit["python"]
    cuda = audit["cuda"]
    lines = [
        "=== AutoVoice Dependency Verification ===",
        "",
        "Python environment:",
        f"  Executable: {python_status['executable']}",
        f"  Version: {python_status['version']}",
        f"  Inferred env: {python_status['inferred_env_name'] or 'unknown'}",
        f"  Expected env: {python_status['expected_env_name']}",
        f"  Env match: {'yes' if python_status['matches_expected_env'] else 'no'}",
        "",
        "CUDA runtime:",
        f"  Torch: {cuda['torch_version'] or 'unavailable'}",
        f"  CUDA available: {cuda['cuda_available']}",
        f"  CUDA version: {cuda['cuda_version'] or 'unknown'}",
        f"  Device: {cuda['device_name'] or 'n/a'}",
        f"  Compute capability: {cuda['compute_capability'] or 'n/a'}",
    ]

    if cuda["memory_free_gb"] is not None and cuda["memory_total_gb"] is not None:
        lines.append(
            f"  GPU memory free/total: {cuda['memory_free_gb']} GiB / {cuda['memory_total_gb']} GiB"
        )

    lines.extend(["", "Dependencies:"])
    for item in audit["dependencies"]:
        status = "OK" if item["ok"] else "FAIL"
        required = "required" if item["required"] else "optional"
        detail = item["version"] or item["error"] or ""
        boundary = item.get("support_boundary", "supported")
        suffix = f" [{boundary}]"
        if not item["required"] and not item["ok"]:
            suffix += f" owner={item.get('owner', 'unassigned')} action={item.get('action', '')}"
        lines.append(f"  [{status}] {item['name']} ({required}) {detail}{suffix}".rstrip())

    lines.extend(
        [
            "",
            f"Required failures: {len(audit['failed_required'])}",
            f"Optional failures: {len(audit['failed_optional'])}",
        ]
    )
    return "\n".join(lines)
