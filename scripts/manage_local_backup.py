#!/usr/bin/env python3
"""Export or import local AutoVoice backup bundles.

Examples:
  python scripts/manage_local_backup.py export --output-dir data/backups
  python scripts/manage_local_backup.py import data/backups/autovoice-backup.zip --dry-run
  python scripts/manage_local_backup.py import data/backups/autovoice-backup.zip --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_sources(data_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("data/voice_profiles", data_dir / "voice_profiles"),
        ("data/samples", data_dir / "samples"),
        ("data/app_state", data_dir / "app_state"),
        ("data/trained_models", data_dir / "trained_models"),
        ("reports/benchmarks/latest", PROJECT_ROOT / "reports" / "benchmarks" / "latest"),
        ("reports/release-evidence/latest", PROJECT_ROOT / "reports" / "release-evidence" / "latest"),
    ]


def export_backup(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / f"autovoice-backup-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.zip"
    files: list[dict[str, object]] = []
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for label, source in backup_sources(data_dir):
            if not source.exists():
                continue
            for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
                arcname = f"payload/{label}/{file_path.relative_to(source).as_posix()}"
                bundle.write(file_path, arcname)
                files.append({
                    "path": arcname,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256(file_path),
                })
        manifest = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": git_sha(),
            "included_paths": [label for label, source in backup_sources(data_dir) if source.exists()],
            "files": files,
            "checksums": {entry["path"]: entry["sha256"] for entry in files},
            "restore_warnings": [
                "Import defaults to dry-run. Use --apply only after reviewing this manifest.",
                "Restore overwrites files under DATA_DIR profiles, samples, app_state, trained_models, and local reports/latest.",
            ],
        }
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps({"status": "success", "backup_path": str(bundle_path), "manifest": manifest}, indent=2))


def restore_targets(data_dir: Path) -> dict[str, Path]:
    return {
        "payload/data/voice_profiles": data_dir / "voice_profiles",
        "payload/data/samples": data_dir / "samples",
        "payload/data/app_state": data_dir / "app_state",
        "payload/data/trained_models": data_dir / "trained_models",
        "payload/reports/benchmarks/latest": PROJECT_ROOT / "reports" / "benchmarks" / "latest",
        "payload/reports/release-evidence/latest": PROJECT_ROOT / "reports" / "release-evidence" / "latest",
    }


def import_backup(args: argparse.Namespace) -> None:
    bundle_path = Path(args.bundle).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        restored: list[str] = []
        if args.apply:
            targets = restore_targets(data_dir)
            for member in bundle.infolist():
                if member.is_dir() or member.filename == "manifest.json":
                    continue
                matched_prefix = next((prefix for prefix in targets if member.filename.startswith(f"{prefix}/")), None)
                if not matched_prefix:
                    continue
                output_path = targets[matched_prefix] / Path(member.filename).relative_to(matched_prefix)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, output_path.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                restored.append(str(output_path))
    print(json.dumps({
        "status": "applied" if args.apply else "dry_run",
        "dry_run": not args.apply,
        "manifest": manifest,
        "restored_count": len(restored),
        "restored_paths": restored,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    export_parser = subcommands.add_parser("export")
    export_parser.add_argument("--data-dir", default="data")
    export_parser.add_argument("--output-dir", default="data/backups")
    export_parser.set_defaults(func=export_backup)

    import_parser = subcommands.add_parser("import")
    import_parser.add_argument("bundle")
    import_parser.add_argument("--data-dir", default="data")
    mode = import_parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--apply", action="store_true")
    import_parser.set_defaults(func=import_backup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
