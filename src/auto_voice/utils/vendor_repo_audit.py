"""Vendor model-repo contract and hygiene checks."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class VendorRepoSpec:
    """One nested model repository tracked as a gitlink/submodule."""

    name: str
    path: str
    url: str
    runtime_artifact_prefixes: tuple[str, ...] = ()


def default_vendor_repos() -> List[VendorRepoSpec]:
    """Return the canonical nested model repositories."""

    return [
        VendorRepoSpec(
            name="hq-svc",
            path="models/hq-svc",
            url="https://github.com/ShawnPi233/HQ-SVC.git",
        ),
        VendorRepoSpec(
            name="meanvc",
            path="models/meanvc",
            url="https://github.com/ASLP-lab/MeanVC.git",
            runtime_artifact_prefixes=("src/ckpt/", "src/runtime/speaker_verification/ckpt/"),
        ),
        VendorRepoSpec(
            name="seed-vc",
            path="models/seed-vc",
            url="https://github.com/Plachtaa/seed-vc.git",
            runtime_artifact_prefixes=(".claude-flow/", "__pycache__/", "=3.20.2"),
        ),
    ]


def _run_git(repo_root: Path, args: Iterable[str], *, cwd: Optional[Path] = None) -> CompletedProcess[str]:
    return run(
        ["git", *args],
        cwd=str(cwd or repo_root),
        check=False,
        capture_output=True,
        text=True,
    )


def parse_gitmodules(gitmodules_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse .gitmodules into a path-keyed mapping."""

    if not gitmodules_path.exists():
        return {}

    parser = ConfigParser()
    parser.read(gitmodules_path, encoding="utf-8")
    mapping: Dict[str, Dict[str, str]] = {}
    for section in parser.sections():
        path = parser.get(section, "path", fallback="").strip()
        if not path:
            continue
        mapping[path] = {
            "url": parser.get(section, "url", fallback="").strip(),
            "section": section,
        }
    return mapping


def read_gitlinks(repo_root: Path) -> Dict[str, str]:
    """Return gitlink paths and SHAs from the parent repository index."""

    result = _run_git(repo_root, ["ls-files", "-s"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files -s failed")

    gitlinks: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        metadata, _, path = line.partition("\t")
        if not metadata or not path:
            continue
        parts = metadata.split()
        if len(parts) < 3:
            continue
        mode, sha = parts[0], parts[1]
        if mode == "160000":
            gitlinks[path] = sha
    return gitlinks


def read_repo_status(repo_root: Path, repo_path: str) -> Dict[str, Any]:
    """Collect porcelain status for one nested repo."""

    full_path = repo_root / repo_path
    if not full_path.exists():
        return {
            "present": False,
            "tracked_changes": [],
            "untracked_files": [],
            "clean": False,
        }

    result = _run_git(repo_root, ["status", "--short"], cwd=full_path)
    if result.returncode != 0:
        return {
            "present": True,
            "tracked_changes": [],
            "untracked_files": [],
            "clean": False,
            "error": result.stderr.strip() or "git status failed",
        }

    tracked_changes: List[str] = []
    untracked_files: List[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("?? "):
            untracked_files.append(line[3:])
        elif line.strip():
            tracked_changes.append(line)

    return {
        "present": True,
        "tracked_changes": tracked_changes,
        "untracked_files": untracked_files,
        "clean": not tracked_changes and not untracked_files,
    }


def _runtime_artifacts(paths: Iterable[str], prefixes: Iterable[str]) -> List[str]:
    artifacts: List[str] = []
    prefix_list = tuple(prefixes)
    for path in paths:
        if any(path == prefix or path.startswith(prefix) for prefix in prefix_list):
            artifacts.append(path)
    return sorted(artifacts)


def audit_vendor_repo(
    repo_root: Path,
    spec: VendorRepoSpec,
    *,
    gitmodules: Optional[Dict[str, Dict[str, str]]] = None,
    gitlinks: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Audit one vendor repo for contract correctness and hygiene state."""

    gitmodules = gitmodules if gitmodules is not None else parse_gitmodules(repo_root / ".gitmodules")
    gitlinks = gitlinks if gitlinks is not None else read_gitlinks(repo_root)
    status = read_repo_status(repo_root, spec.path)

    gitmodule_entry = gitmodules.get(spec.path)
    gitmodule_url = gitmodule_entry["url"] if gitmodule_entry else None
    gitlink_sha = gitlinks.get(spec.path)
    runtime_artifacts = _runtime_artifacts(
        status.get("untracked_files", []),
        spec.runtime_artifact_prefixes,
    )

    notes: List[str] = []
    if gitmodule_url != spec.url:
        notes.append("gitmodules URL does not match canonical vendor URL")
    if gitlink_sha is None:
        notes.append("path is not tracked as a gitlink in the parent repo")
    if not status.get("present", False):
        notes.append("nested repo path is missing from the workspace")
    if status.get("tracked_changes"):
        notes.append("nested repo has tracked modifications")
    if runtime_artifacts:
        notes.append("nested repo has untracked runtime artifacts")
    elif status.get("untracked_files"):
        notes.append("nested repo has untracked files")

    contract_ok = bool(gitlink_sha) and gitmodule_url == spec.url

    return {
        "name": spec.name,
        "path": spec.path,
        "expected_url": spec.url,
        "gitmodule_url": gitmodule_url,
        "gitlink_sha": gitlink_sha,
        "present": status.get("present", False),
        "contract_ok": contract_ok,
        "clean": bool(status.get("clean", False)),
        "tracked_changes": status.get("tracked_changes", []),
        "untracked_files": status.get("untracked_files", []),
        "runtime_artifacts": runtime_artifacts,
        "notes": notes,
    }


def run_vendor_repo_audit(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run the full vendor-repo contract audit."""

    resolved_root = (repo_root or Path.cwd()).resolve()
    gitmodules = parse_gitmodules(resolved_root / ".gitmodules")
    gitlinks = read_gitlinks(resolved_root)
    repos = [
        audit_vendor_repo(
            resolved_root,
            spec,
            gitmodules=gitmodules,
            gitlinks=gitlinks,
        )
        for spec in default_vendor_repos()
    ]

    return {
        "repo_root": str(resolved_root),
        "repos": repos,
        "contract_ok": all(item["contract_ok"] for item in repos),
        "clean": all(item["clean"] for item in repos),
    }


def format_vendor_repo_audit(audit: Dict[str, Any]) -> str:
    """Render the vendor audit in a human-readable form."""

    lines = [
        "=== AutoVoice Vendor Repo Audit ===",
        f"Repo root: {audit['repo_root']}",
        "",
    ]

    for item in audit["repos"]:
        lines.append(f"{item['name']} ({item['path']})")
        lines.append(f"  Contract OK: {'yes' if item['contract_ok'] else 'no'}")
        lines.append(f"  Clean: {'yes' if item['clean'] else 'no'}")
        lines.append(f"  Gitlink SHA: {item['gitlink_sha'] or 'missing'}")
        lines.append(f"  .gitmodules URL: {item['gitmodule_url'] or 'missing'}")
        if item["tracked_changes"]:
            lines.append(f"  Tracked changes: {len(item['tracked_changes'])}")
        if item["untracked_files"]:
            lines.append(f"  Untracked files: {len(item['untracked_files'])}")
        if item["runtime_artifacts"]:
            lines.append("  Runtime artifacts:")
            for path in item["runtime_artifacts"]:
                lines.append(f"    - {path}")
        for note in item["notes"]:
            lines.append(f"  Note: {note}")
        lines.append("")

    lines.append(f"Overall contract OK: {'yes' if audit['contract_ok'] else 'no'}")
    lines.append(f"Overall clean: {'yes' if audit['clean'] else 'no'}")
    return "\n".join(lines)
