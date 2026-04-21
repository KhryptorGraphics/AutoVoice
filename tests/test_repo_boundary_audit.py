from __future__ import annotations

from auto_voice.utils.repo_boundary_audit import run_repo_boundary_audit


def test_repo_boundary_audit_flags_forbidden_prefixes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / ".git").mkdir()
    (repo / "node_modules").mkdir()
    (repo / "reports").mkdir()
    (repo / "output" / "reports").mkdir(parents=True)

    tracked = "\n".join(
        [
            "src/app.py",
            "node_modules/pkg/index.js",
            "reports/build.md",
            "output/reports/summary.json",
        ]
    )

    def fake_ls_files(_repo_root):
        return tracked.splitlines()

    from auto_voice.utils import repo_boundary_audit as audit_module

    original = audit_module._tracked_files
    audit_module._tracked_files = fake_ls_files
    try:
        audit = run_repo_boundary_audit(repo)
    finally:
        audit_module._tracked_files = original

    assert audit["ok"] is False
    assert audit["violation_count"] == 3


def test_repo_boundary_audit_passes_when_prefixes_absent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    from auto_voice.utils import repo_boundary_audit as audit_module

    original = audit_module._tracked_files
    audit_module._tracked_files = lambda _repo_root: ["src/app.py", "frontend/src/App.tsx"]
    try:
        audit = run_repo_boundary_audit(repo)
    finally:
        audit_module._tracked_files = original

    assert audit["ok"] is True
    assert audit["violation_count"] == 0
