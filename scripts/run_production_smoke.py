#!/usr/bin/env python3
"""Run public AutoVoice production smoke checks and write evidence artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIST_SONG = PROJECT_ROOT / "tests/quality_samples/conor_maynard_pillowtalk.wav"
DEFAULT_USER_VOCALS = PROJECT_ROOT / "tests/quality_samples/william_singe_pillowtalk.wav"
DEFAULT_BASE_URL = "https://autovoice.giggahost.com"


@dataclass
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any] | list[Any]:
        return json.loads(self.body.decode("utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_tag() -> str | None:
    try:
        tags = subprocess.check_output(["git", "tag", "--points-at", "HEAD"], cwd=PROJECT_ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    return tags[0] if tags else None


def _url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _request(
    method: str,
    base_url: str,
    path: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> HttpResult:
    request = Request(_url(base_url, path), data=data, method=method, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResult(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except HTTPError as exc:
        return HttpResult(
            status=exc.code,
            headers=dict(exc.headers.items()),
            body=exc.read(),
        )
    except URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc


def _json_request(method: str, base_url: str, path: str, payload: dict[str, Any] | None = None) -> HttpResult:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return _request(method, base_url, path, data=data, headers=headers)


def _multipart_body(
    fields: dict[str, str],
    files: list[tuple[str, Path]],
) -> tuple[bytes, str]:
    boundary = f"autovoice-smoke-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stage_upload_audio(path: Path, output_dir: Path, max_seconds: float) -> Path:
    """Create a short real-audio excerpt for production smoke uploads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = output_dir / path.name
    if max_seconds <= 0:
        shutil.copy2(path, staged_path)
        return staged_path
    try:
        with wave.open(str(path), "rb") as source:
            params = source.getparams()
            frame_count = min(source.getnframes(), int(source.getframerate() * max_seconds))
            frames = source.readframes(frame_count)
        with wave.open(str(staged_path), "wb") as target:
            target.setparams(params)
            target.writeframes(frames)
        return staged_path
    except wave.Error:
        shutil.copy2(path, staged_path)
        return staged_path


def _copy_evidence(output_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in ("benchmark_dashboard.json", "release_evidence.json", "report.md"):
        source = PROJECT_ROOT / "reports/benchmarks/latest" / name
        if not source.exists():
            continue
        dest = output_dir / "benchmark_evidence" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied[name] = str(dest)
    return copied


def _check_json_endpoint(base_url: str, path: str, predicate: Any) -> dict[str, Any]:
    started = time.monotonic()
    result = _json_request("GET", base_url, path)
    payload: dict[str, Any] | list[Any] | None
    try:
        payload = result.json()
    except json.JSONDecodeError:
        payload = None
    return {
        "path": path,
        "status": result.status,
        "ok": result.status == 200 and predicate(payload),
        "duration_seconds": round(time.monotonic() - started, 3),
        "payload": payload,
    }


def run_health_smoke(base_url: str) -> dict[str, Any]:
    checks = [
        _check_json_endpoint(base_url, "/api/v1/health", lambda payload: isinstance(payload, dict) and payload.get("status") == "healthy"),
        _check_json_endpoint(base_url, "/ready", lambda payload: isinstance(payload, dict) and bool(payload.get("ready"))),
        _check_json_endpoint(base_url, "/api/v1/metrics", lambda payload: isinstance(payload, dict) and "error" not in payload),
    ]
    return {"checks": checks, "ok": all(check["ok"] for check in checks)}


def _poll_json(
    base_url: str,
    path: str,
    *,
    timeout_seconds: int,
    poll_interval: float,
    terminal: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        result = _json_request("GET", base_url, path)
        try:
            payload = result.json()
        except json.JSONDecodeError:
            payload = {"error": result.body.decode("utf-8", errors="replace")}
        if isinstance(payload, dict):
            last_payload = payload
            if terminal(payload):
                return payload
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out polling {path}; last payload={last_payload!r}")


def _resolve_reviews(base_url: str, workflow: dict[str, Any], created_profile_ids: set[str]) -> dict[str, Any]:
    workflow_id = str(workflow["workflow_id"])
    for review in list(workflow.get("review_items") or []):
        role = review.get("role") or "profile"
        response = _json_request(
            "POST",
            base_url,
            f"/api/v1/convert/workflows/{workflow_id}/resolve-match",
            {
                "review_id": review.get("review_id"),
                "resolution": "create_new",
                "name": f"AutoVoice production smoke {role} {workflow_id[:8]}",
            },
        )
        if response.status != 200:
            raise RuntimeError(f"Review resolution failed: HTTP {response.status} {response.body!r}")
        workflow = response.json()
        if not isinstance(workflow, dict):
            raise RuntimeError("Review resolution returned non-object payload")
        if role == "target_user" and workflow.get("resolved_target_profile_id"):
            created_profile_ids.add(str(workflow["resolved_target_profile_id"]))
        for source in workflow.get("resolved_source_profiles") or []:
            if source.get("status") == "created" and source.get("profile_id"):
                created_profile_ids.add(str(source["profile_id"]))
    return workflow


def _download_asset(base_url: str, path: str, output_path: Path) -> dict[str, Any]:
    result = _request("GET", base_url, path, timeout=120)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.body)
    return {
        "path": path,
        "status": result.status,
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "ok": result.status == 200 and output_path.stat().st_size > 44,
    }


def _cleanup_profiles(base_url: str, profile_ids: set[str]) -> list[dict[str, Any]]:
    results = []
    for profile_id in sorted(profile_ids):
        result = _json_request("DELETE", base_url, f"/api/v1/voice/profiles/{profile_id}")
        method = "DELETE"
        if result.status == 403:
            result = _json_request("POST", base_url, f"/api/v1/voice/profiles/{profile_id}/delete", {})
            method = "POST"
        results.append({
            "profile_id": profile_id,
            "method": method,
            "status": result.status,
            "ok": result.status in {200, 404},
        })
    return results


def run_full_smoke(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    created_profile_ids: set[str] = set()
    downloads: list[dict[str, Any]] = []
    cleanup: list[dict[str, Any]] = []
    report: dict[str, Any] | None = None
    started = time.monotonic()

    source_artist_song = Path(args.artist_song).resolve()
    source_user_vocals = [Path(path).resolve() for path in args.user_vocals]
    missing = [str(path) for path in [source_artist_song, *source_user_vocals] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing production smoke fixture(s): {missing}")
    staged_dir = output_dir / "upload_fixtures"
    artist_song = _stage_upload_audio(source_artist_song, staged_dir, args.max_upload_seconds)
    user_vocals = [
        _stage_upload_audio(path, staged_dir / "user_vocals", args.max_upload_seconds)
        for path in source_user_vocals
    ]

    try:
        body, content_type = _multipart_body(
            {"target_profile_id": args.target_profile_id or ""},
            [("artist_song", artist_song), *[("user_vocals", path) for path in user_vocals]],
        )
        create = _request(
            "POST",
            args.base_url,
            "/api/v1/convert/workflows",
            data=body,
            headers={"Content-Type": content_type, "Accept": "application/json"},
            timeout=180,
        )
        if create.status != 201:
            raise RuntimeError(f"Workflow create failed: HTTP {create.status} {create.body!r}")
        workflow = create.json()
        workflow_id = str(workflow["workflow_id"])

        def workflow_ready(payload: dict[str, Any]) -> bool:
            if payload.get("status") in {"failed", "error"}:
                return True
            if payload.get("review_items"):
                return True
            return bool((payload.get("training_readiness") or {}).get("ready"))

        while True:
            workflow = _poll_json(
                args.base_url,
                f"/api/v1/convert/workflows/{workflow_id}",
                timeout_seconds=args.timeout_seconds,
                poll_interval=args.poll_interval,
                terminal=workflow_ready,
            )
            if workflow.get("status") in {"failed", "error"}:
                raise RuntimeError(f"Workflow failed: {workflow.get('error') or workflow}")
            if workflow.get("review_items"):
                workflow = _resolve_reviews(args.base_url, workflow, created_profile_ids)
                continue
            if (workflow.get("training_readiness") or {}).get("ready"):
                break

        target_profile_id = str(workflow.get("resolved_target_profile_id") or "")
        if not target_profile_id:
            raise RuntimeError("Workflow did not resolve a target profile")
        if not args.target_profile_id:
            created_profile_ids.add(target_profile_id)
        for source in workflow.get("resolved_source_profiles") or []:
            if source.get("status") == "created" and source.get("profile_id"):
                created_profile_ids.add(str(source["profile_id"]))

        train_payload = {
            "profile_id": target_profile_id,
            "config": {
                "training_mode": "lora",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": 0.0,
            },
        }
        train = _json_request("POST", args.base_url, "/api/v1/training/jobs", train_payload)
        if train.status != 201:
            raise RuntimeError(f"Training job create failed: HTTP {train.status} {train.body!r}")
        training_job = train.json()
        training_job_id = str(training_job["job_id"])

        training_job = _poll_json(
            args.base_url,
            f"/api/v1/training/jobs/{training_job_id}",
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            terminal=lambda payload: payload.get("status") in {"completed", "failed", "cancelled"},
        )
        if training_job.get("status") != "completed":
            raise RuntimeError(f"Training job did not complete: {training_job!r}")

        attach = _json_request(
            "POST",
            args.base_url,
            f"/api/v1/convert/workflows/{workflow_id}/training-job",
            {"job_id": training_job_id},
        )
        if attach.status != 200:
            raise RuntimeError(f"Attach training job failed: HTTP {attach.status} {attach.body!r}")

        convert = _json_request(
            "POST",
            args.base_url,
            f"/api/v1/convert/workflows/{workflow_id}/convert",
            {
                "pipeline_type": args.pipeline,
                "return_stems": True,
                "pitch_shift": 0,
                "preset": "balanced",
            },
        )
        if convert.status != 202:
            raise RuntimeError(f"Conversion queue failed: HTTP {convert.status} {convert.body!r}")
        conversion_job = convert.json()
        conversion_job_id = str(conversion_job["job_id"])

        conversion_status = _poll_json(
            args.base_url,
            f"/api/v1/convert/status/{conversion_job_id}",
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            terminal=lambda payload: payload.get("status") in {"completed", "failed", "cancelled"},
        )
        if conversion_status.get("status") != "completed":
            raise RuntimeError(f"Conversion job did not complete: {conversion_status!r}")

        downloads.append(_download_asset(args.base_url, f"/api/v1/convert/download/{conversion_job_id}", output_dir / "downloads" / "mix.wav"))
        if conversion_status.get("stem_urls"):
            downloads.append(_download_asset(args.base_url, f"/api/v1/convert/download/{conversion_job_id}?variant=vocals", output_dir / "downloads" / "vocals.wav"))
            downloads.append(_download_asset(args.base_url, f"/api/v1/convert/download/{conversion_job_id}?variant=instrumental", output_dir / "downloads" / "instrumental.wav"))

        report = {
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "upload_fixtures": {
                "max_upload_seconds": args.max_upload_seconds,
                "artist_song": {
                    "source": str(source_artist_song),
                    "staged": str(artist_song),
                    "source_size_bytes": source_artist_song.stat().st_size,
                    "staged_size_bytes": artist_song.stat().st_size,
                },
                "user_vocals": [
                    {
                        "source": str(source),
                        "staged": str(staged),
                        "source_size_bytes": source.stat().st_size,
                        "staged_size_bytes": staged.stat().st_size,
                    }
                    for source, staged in zip(source_user_vocals, user_vocals)
                ],
            },
            "workflow_id": workflow_id,
            "target_profile_id": target_profile_id,
            "created_profile_ids": sorted(created_profile_ids),
            "training_job": training_job,
            "conversion_job": conversion_status,
            "downloads": downloads,
            "cleanup": cleanup,
        }
        return report
    finally:
        if not args.no_cleanup and created_profile_ids:
            cleanup.extend(_cleanup_profiles(args.base_url, created_profile_ids))
        if report is not None:
            downloads_ok = all(download["ok"] for download in downloads)
            cleanup_ok = args.no_cleanup or all(entry["ok"] for entry in cleanup)
            report["ok"] = downloads_ok and cleanup_ok
            report["duration_seconds"] = round(time.monotonic() - started, 3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices={"health", "full"}, default="health")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/production_smoke/latest"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artist-song", type=Path, default=DEFAULT_ARTIST_SONG)
    parser.add_argument("--user-vocals", type=Path, action="append", default=None)
    parser.add_argument("--target-profile-id", default=None)
    parser.add_argument("--pipeline", default="quality_seedvc")
    parser.add_argument(
        "--max-upload-seconds",
        type=float,
        default=3.0,
        help="Stage short real-audio excerpts for upload; pass 0 to upload full files.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--no-cleanup", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.user_vocals is None:
        args.user_vocals = [DEFAULT_USER_VOCALS]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id if args.output_dir.name == "latest" else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _now(),
        "mode": args.mode,
        "base_url": args.base_url.rstrip("/"),
        "git_sha": _git_sha(),
        "git_tag": _git_tag(),
        "benchmark_evidence": _copy_evidence(output_dir),
        "ok": False,
    }
    try:
        health = run_health_smoke(args.base_url)
        report["health"] = health
        if args.mode == "full":
            report["full"] = run_full_smoke(args, output_dir)
            report["ok"] = health["ok"] and report["full"]["ok"]
        else:
            report["ok"] = health["ok"]
    except Exception as exc:
        report["error"] = str(exc)
        report["ok"] = False

    report_path = output_dir / "production_smoke.json"
    _write_json(report_path, report)
    print(report_path)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
