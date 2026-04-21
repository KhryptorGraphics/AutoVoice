"""Tests for nested vendor-repo contract auditing."""

from __future__ import annotations

from pathlib import Path

from auto_voice.utils import vendor_repo_audit as audit_module


def test_parse_gitmodules_returns_path_mapping(tmp_path):
    gitmodules = tmp_path / ".gitmodules"
    gitmodules.write_text(
        '\n'.join(
            [
                '[submodule "models/hq-svc"]',
                'path = models/hq-svc',
                'url = https://github.com/ShawnPi233/HQ-SVC.git',
                '',
                '[submodule "models/seed-vc"]',
                'path = models/seed-vc',
                'url = https://github.com/Plachtaa/seed-vc.git',
            ]
        ),
        encoding="utf-8",
    )

    parsed = audit_module.parse_gitmodules(gitmodules)

    assert parsed["models/hq-svc"]["url"] == "https://github.com/ShawnPi233/HQ-SVC.git"
    assert parsed["models/seed-vc"]["section"] == 'submodule "models/seed-vc"'


def test_audit_vendor_repo_reports_contract_and_runtime_artifacts(monkeypatch, tmp_path):
    repo_root = tmp_path
    (repo_root / "models" / "seed-vc").mkdir(parents=True)
    spec = audit_module.VendorRepoSpec(
        name="seed-vc",
        path="models/seed-vc",
        url="https://github.com/Plachtaa/seed-vc.git",
        runtime_artifact_prefixes=(".claude-flow/", "__pycache__/"),
    )

    monkeypatch.setattr(
        audit_module,
        "read_repo_status",
        lambda _repo_root, _repo_path: {
            "present": True,
            "clean": False,
            "tracked_changes": [" M app.py"],
            "untracked_files": [".claude-flow/daemon.log", "__pycache__/seed_vc.pyc", "notes.txt"],
        },
    )

    result = audit_module.audit_vendor_repo(
        repo_root,
        spec,
        gitmodules={spec.path: {"url": spec.url, "section": 'submodule "models/seed-vc"'}},
        gitlinks={spec.path: "deadbeef"},
    )

    assert result["contract_ok"] is True
    assert result["clean"] is False
    assert result["runtime_artifacts"] == [".claude-flow/daemon.log", "__pycache__/seed_vc.pyc"]
    assert any("tracked modifications" in note for note in result["notes"])


def test_run_vendor_repo_audit_aggregates_repo_results(monkeypatch, tmp_path):
    repo_root = tmp_path
    for rel_path in ("models/hq-svc", "models/meanvc", "models/seed-vc"):
        (repo_root / rel_path).mkdir(parents=True)

    monkeypatch.setattr(
        audit_module,
        "parse_gitmodules",
        lambda _path: {
            spec.path: {"url": spec.url, "section": f'submodule "{spec.path}"'}
            for spec in audit_module.default_vendor_repos()
        },
    )
    monkeypatch.setattr(
        audit_module,
        "read_gitlinks",
        lambda _repo_root: {
            spec.path: f"sha-{spec.name}"
            for spec in audit_module.default_vendor_repos()
        },
    )
    monkeypatch.setattr(
        audit_module,
        "read_repo_status",
        lambda _repo_root, repo_path: {
            "present": True,
            "clean": repo_path != "models/meanvc",
            "tracked_changes": [] if repo_path != "models/meanvc" else [" M src/eval/utils.py"],
            "untracked_files": [],
        },
    )

    audit = audit_module.run_vendor_repo_audit(repo_root)

    assert audit["contract_ok"] is True
    assert audit["clean"] is False
    meanvc = next(item for item in audit["repos"] if item["name"] == "meanvc")
    assert meanvc["clean"] is False
    assert meanvc["gitlink_sha"] == "sha-meanvc"


def test_format_vendor_repo_audit_includes_overall_summary():
    rendered = audit_module.format_vendor_repo_audit(
        {
            "repo_root": "/repo",
            "contract_ok": True,
            "clean": False,
            "repos": [
                {
                    "name": "seed-vc",
                    "path": "models/seed-vc",
                    "contract_ok": True,
                    "clean": False,
                    "gitlink_sha": "abc123",
                    "gitmodule_url": "https://github.com/Plachtaa/seed-vc.git",
                    "tracked_changes": [],
                    "untracked_files": ["__pycache__/seed_vc.pyc"],
                    "runtime_artifacts": ["__pycache__/seed_vc.pyc"],
                    "notes": ["nested repo has untracked runtime artifacts"],
                }
            ],
        }
    )

    assert "Overall contract OK: yes" in rendered
    assert "Overall clean: no" in rendered
    assert "Runtime artifacts:" in rendered
