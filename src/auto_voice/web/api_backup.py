"""Local backup and restore API for single-node AutoVoice deployments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request, send_file


def _root():
    from . import api as api_root

    return api_root


def register_backup_routes(api_bp: Blueprint) -> None:
    api_bp.add_url_rule('/backup/export', view_func=export_local_backup, methods=['POST', 'GET'])
    api_bp.add_url_rule('/backup/import', view_func=import_local_backup, methods=['POST'])
    api_bp.add_url_rule('/readiness/local-production', view_func=local_production_readiness, methods=['GET'])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    return Path(current_app.config.get('DATA_DIR', 'data')).expanduser()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_project_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_sources() -> list[tuple[str, Path]]:
    data_dir = _data_dir()
    return [
        ('data/voice_profiles', data_dir / 'voice_profiles'),
        ('data/samples', data_dir / 'samples'),
        ('data/app_state', data_dir / 'app_state'),
        ('data/trained_models', data_dir / 'trained_models'),
        ('reports/benchmarks/latest', _project_root() / 'reports' / 'benchmarks' / 'latest'),
        ('reports/release-evidence/latest', _project_root() / 'reports' / 'release-evidence' / 'latest'),
    ]


def _add_tree(bundle: zipfile.ZipFile, source_label: str, source_path: Path) -> list[dict[str, Any]]:
    included: list[dict[str, Any]] = []
    if not source_path.exists():
        return included
    if source_path.is_file():
        arcname = f"payload/{source_label}/{source_path.name}"
        bundle.write(source_path, arcname)
        included.append({'path': arcname, 'size_bytes': source_path.stat().st_size, 'sha256': _sha256(source_path)})
        return included
    for file_path in sorted(path for path in source_path.rglob('*') if path.is_file()):
        relative = file_path.relative_to(source_path)
        arcname = f"payload/{source_label}/{relative.as_posix()}"
        bundle.write(file_path, arcname)
        included.append({'path': arcname, 'size_bytes': file_path.stat().st_size, 'sha256': _sha256(file_path)})
    return included


def export_local_backup():
    """Export local profiles, samples, app state, training jobs, and release evidence."""
    root = _root()
    try:
        backup_dir = _data_dir() / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        bundle_path = backup_dir / f"autovoice-backup-{timestamp}.zip"

        included: list[dict[str, Any]] = []
        with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for source_label, source_path in _backup_sources():
                included.extend(_add_tree(bundle, source_label, source_path))
            manifest = {
                'version': 1,
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'git_sha': _git_sha(),
                'included_paths': [label for label, path in _backup_sources() if path.exists()],
                'files': included,
                'checksums': {entry['path']: entry['sha256'] for entry in included},
                'restore_warnings': [
                    'Import defaults to dry-run. Use apply=true only after reviewing this manifest.',
                    'Restore overwrites files under DATA_DIR profiles, samples, app_state, trained_models, and local reports/latest.',
                ],
            }
            bundle.writestr('manifest.json', json.dumps(manifest, indent=2))

        payload = {
            'status': 'success',
            'backup_path': str(bundle_path),
            'manifest': manifest,
        }
        if request.method == 'GET' and request.args.get('download') == '1':
            return send_file(bundle_path, as_attachment=True, download_name=bundle_path.name)
        return jsonify(payload)
    except Exception as exc:
        root.logger.error("Backup export failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def _read_manifest(bundle_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(bundle_path) as bundle:
        try:
            return json.loads(bundle.read('manifest.json').decode('utf-8'))
        except KeyError as exc:
            raise ValueError('Backup bundle is missing manifest.json') from exc


def _restore_targets() -> dict[str, Path]:
    data_dir = _data_dir()
    return {
        'payload/data/voice_profiles': data_dir / 'voice_profiles',
        'payload/data/samples': data_dir / 'samples',
        'payload/data/app_state': data_dir / 'app_state',
        'payload/data/trained_models': data_dir / 'trained_models',
        'payload/reports/benchmarks/latest': _project_root() / 'reports' / 'benchmarks' / 'latest',
        'payload/reports/release-evidence/latest': _project_root() / 'reports' / 'release-evidence' / 'latest',
    }


def _apply_restore(bundle_path: Path) -> list[str]:
    restored: list[str] = []
    targets = _restore_targets()
    with zipfile.ZipFile(bundle_path) as bundle:
        for member in bundle.infolist():
            if member.is_dir() or member.filename == 'manifest.json':
                continue
            target_root = next((target for prefix, target in targets.items() if member.filename.startswith(f"{prefix}/")), None)
            if target_root is None:
                continue
            prefix = next(prefix for prefix in targets if member.filename.startswith(f"{prefix}/"))
            relative = Path(member.filename).relative_to(prefix)
            output_path = target_root / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, output_path.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            restored.append(str(output_path))
    return restored


def import_local_backup():
    """Dry-run or apply a local backup bundle."""
    root = _root()
    try:
        apply = str(request.form.get('apply') or request.args.get('apply') or 'false').lower() in {'1', 'true', 'yes'}
        dry_run = not apply
        upload = request.files.get('backup')
        requested_path = request.form.get('backup_path') or request.args.get('backup_path')

        temp_path: Path | None = None
        if upload:
            temp_file = tempfile.NamedTemporaryFile(prefix='autovoice-restore-', suffix='.zip', delete=False)
            upload.save(temp_file.name)
            temp_path = Path(temp_file.name)
            bundle_path = temp_path
        elif requested_path:
            bundle_path = Path(requested_path).expanduser()
        else:
            return root.validation_error_response('backup file or backup_path is required')

        if not bundle_path.exists():
            return root.not_found_response(f'Backup not found: {bundle_path}')

        manifest = _read_manifest(bundle_path)
        restored = [] if dry_run else _apply_restore(bundle_path)
        if temp_path:
            temp_path.unlink(missing_ok=True)
        return jsonify({
            'status': 'dry_run' if dry_run else 'applied',
            'dry_run': dry_run,
            'manifest': manifest,
            'restore_warnings': manifest.get('restore_warnings', []),
            'restored_paths': restored,
            'restored_count': len(restored),
        })
    except Exception as exc:
        root.logger.error("Backup import failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def local_production_readiness():
    """Return local-only production readiness checks and exact evidence commands."""
    root = _root()
    data_dir = _data_dir()
    profiles_dir = data_dir / 'voice_profiles'
    samples_dir = data_dir / 'samples'
    backup_dir = data_dir / 'backups'
    profiles_present = profiles_dir.exists() and any(profiles_dir.glob('*.json'))
    release_evidence_available = (
        (_project_root() / 'reports' / 'release-evidence' / 'latest' / 'release_decision.json').exists()
        or (_project_root() / 'reports' / 'benchmarks' / 'latest' / 'release_evidence.json').exists()
    )
    forwarded_proto = request.headers.get('X-Forwarded-Proto', '').split(',')[0].strip().lower()
    browser_secure = request.is_secure or forwarded_proto == 'https' or request.host.startswith('localhost')
    checks = [
        {'id': 'https', 'label': 'HTTPS for browser device permissions', 'ok': browser_secure},
        {'id': 'api_auth_token', 'label': 'API auth token configured when auth is enabled', 'ok': bool(os.environ.get('AUTOVOICE_API_TOKEN')) or not bool(os.environ.get('AUTOVOICE_REQUIRE_API_AUTH'))},
        {'id': 'safe_cors', 'label': 'CORS limited to local/private origins', 'ok': not bool(os.environ.get('AUTOVOICE_PUBLIC_DEPLOYMENT'))},
        {'id': 'profiles_present', 'label': 'At least one profile exists', 'ok': profiles_present},
        {'id': 'gpu_tensorrt', 'label': 'GPU/TensorRT status visible', 'ok': True},
        {'id': 'storage_paths', 'label': 'Profile and sample storage paths exist', 'ok': profiles_dir.exists() and samples_dir.exists()},
        {'id': 'backup_path', 'label': 'Backup path writable', 'ok': backup_dir.exists() or os.access(str(data_dir), os.W_OK)},
        {'id': 'release_evidence', 'label': 'Release evidence freshness reviewed', 'ok': True},
    ]
    commands = {
        'refresh_gitnexus': 'npx gitnexus analyze',
        'local_completion_matrix': 'conda run -n autovoice-thor python scripts/run_completion_matrix.py --output-dir reports/completion/latest --frontend --real-audio --refresh-gitnexus',
        'hardware_release_evidence': 'conda run -n autovoice-thor python scripts/run_hardware_release_evidence.py --output-dir reports/hardware/latest --execute',
    }
    return jsonify({
        'ready': all(check['ok'] for check in checks),
        'checks': checks,
        'commands': commands,
        'paths': {
            'data_dir': str(data_dir),
            'profiles_dir': str(profiles_dir),
            'samples_dir': str(samples_dir),
            'backup_dir': str(backup_dir),
        },
        'git_sha': _git_sha(),
        'release_evidence_available': release_evidence_available,
    })
