from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)


class _SmokeHandler(BaseHTTPRequestHandler):
    resolved_reviews = False
    deleted_profiles: list[str] = []

    def log_message(self, format, *args):  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self) -> None:
        body = b"RIFF" + b"\x00" * 128
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/v1/health":
            self._send_json(200, {"status": "healthy"})
        elif self.path == "/ready":
            self._send_json(200, {"ready": True})
        elif self.path == "/api/v1/metrics":
            self._send_json(200, {"requests": 1})
        elif self.path == "/api/v1/convert/workflows/wf-smoke":
            if self.resolved_reviews:
                self._send_json(
                    200,
                    {
                        "workflow_id": "wf-smoke",
                        "status": "ready_for_training",
                        "resolved_target_profile_id": "target-smoke",
                        "resolved_source_profiles": [{"profile_id": "source-smoke", "status": "created"}],
                        "review_items": [],
                        "training_readiness": {"ready": True},
                    },
                )
            else:
                self._send_json(
                    200,
                    {
                        "workflow_id": "wf-smoke",
                        "status": "awaiting_review",
                        "review_items": [
                            {
                                "review_id": "review-target",
                                "role": "target_user",
                                "candidate": {"role": "target_user"},
                            }
                        ],
                        "training_readiness": {"ready": False},
                    },
                )
        elif self.path == "/api/v1/training/jobs/train-smoke":
            self._send_json(200, {"job_id": "train-smoke", "status": "completed"})
        elif self.path == "/api/v1/convert/status/convert-smoke":
            self._send_json(
                200,
                {
                    "job_id": "convert-smoke",
                    "status": "completed",
                    "stem_urls": {
                        "vocals": "/api/v1/convert/download/convert-smoke?variant=vocals",
                        "instrumental": "/api/v1/convert/download/convert-smoke?variant=instrumental",
                    },
                },
            )
        elif self.path.startswith("/api/v1/convert/download/convert-smoke"):
            self._send_wav()
        else:
            self._send_json(404, {"error": self.path})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path == "/api/v1/convert/workflows":
            self._send_json(201, {"workflow_id": "wf-smoke"})
        elif self.path == "/api/v1/convert/workflows/wf-smoke/resolve-match":
            type(self).resolved_reviews = True
            self._send_json(
                200,
                {
                    "workflow_id": "wf-smoke",
                    "status": "ready_for_training",
                    "resolved_target_profile_id": "target-smoke",
                    "resolved_source_profiles": [{"profile_id": "source-smoke", "status": "created"}],
                    "review_items": [],
                    "training_readiness": {"ready": True},
                },
            )
        elif self.path == "/api/v1/training/jobs":
            self._send_json(201, {"job_id": "train-smoke", "status": "training"})
        elif self.path == "/api/v1/convert/workflows/wf-smoke/training-job":
            self._send_json(200, {"workflow_id": "wf-smoke", "current_training_job_id": "train-smoke"})
        elif self.path == "/api/v1/convert/workflows/wf-smoke/convert":
            self._send_json(202, {"job_id": "convert-smoke", "status": "queued"})
        elif self.path.startswith("/api/v1/voice/profiles/") and self.path.endswith("/delete"):
            profile_id = self.path.split("/")[-2]
            type(self).deleted_profiles.append(profile_id)
            self._send_json(200, {"status": "success", "profile_id": profile_id})
        else:
            self._send_json(404, {"error": self.path})

    def do_DELETE(self):  # noqa: N802
        if self.path.startswith("/api/v1/voice/profiles/"):
            self._send_json(403, {"error": "method blocked"})
        else:
            self._send_json(404, {"error": self.path})


def _serve(handler_class):
    server = HTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_production_smoke_health_mode_writes_report(tmp_path):
    server, base_url = _serve(_SmokeHandler)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_production_smoke.py",
                "--mode",
                "health",
                "--base-url",
                base_url,
                "--output-dir",
                str(tmp_path / "latest"),
                "--run-id",
                "unit",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((tmp_path / "latest" / "unit" / "production_smoke.json").read_text())
    assert report["ok"] is True
    assert [check["path"] for check in report["health"]["checks"]] == [
        "/api/v1/health",
        "/ready",
        "/api/v1/metrics",
    ]


def test_production_smoke_full_mode_exercises_workflow_and_cleanup(tmp_path):
    _SmokeHandler.resolved_reviews = False
    _SmokeHandler.deleted_profiles = []
    artist_song = tmp_path / "artist.wav"
    user_vocals = tmp_path / "user.wav"
    _write_wav(artist_song)
    _write_wav(user_vocals)
    server, base_url = _serve(_SmokeHandler)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_production_smoke.py",
                "--mode",
                "full",
                "--base-url",
                base_url,
                "--artist-song",
                str(artist_song),
                "--user-vocals",
                str(user_vocals),
                "--output-dir",
                str(tmp_path / "latest"),
                "--run-id",
                "full",
                "--poll-interval",
                "0.01",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((tmp_path / "latest" / "full" / "production_smoke.json").read_text())
    assert report["ok"] is True
    assert report["full"]["workflow_id"] == "wf-smoke"
    assert report["full"]["training_job"]["job_id"] == "train-smoke"
    assert report["full"]["conversion_job"]["job_id"] == "convert-smoke"
    assert {download["path"] for download in report["full"]["downloads"]} == {
        "/api/v1/convert/download/convert-smoke",
        "/api/v1/convert/download/convert-smoke?variant=vocals",
        "/api/v1/convert/download/convert-smoke?variant=instrumental",
    }
    assert report["full"]["upload_fixtures"]["fixture_tier"] == "smoke"
    assert report["full"]["stem_assertions"] == {
        "requested": True,
        "required": False,
        "reported_by_api": True,
        "downloaded_variants": ["instrumental", "mix", "vocals"],
        "ok": True,
    }
    assert report["full"]["quality_metrics"]["download_count"] == 3
    assert report["full"]["quality_metrics"]["all_downloads_ok"] is True
    assert all(download["inspection"]["sha256"] for download in report["full"]["downloads"])
    assert {entry["method"] for entry in report["full"]["cleanup"]} == {"POST"}
    assert all(entry["ok"] for entry in report["full"]["cleanup"])
    assert set(_SmokeHandler.deleted_profiles) == {"target-smoke", "source-smoke"}


def test_production_smoke_cleanup_503_is_nonblocking_warning(tmp_path):
    class CleanupUnavailableHandler(_SmokeHandler):
        def do_DELETE(self):  # noqa: N802
            if self.path.startswith("/api/v1/voice/profiles/"):
                self._send_json(503, {"error": "temporary cleanup unavailable"})
            else:
                self._send_json(404, {"error": self.path})

        def do_POST(self):  # noqa: N802
            if self.path.startswith("/api/v1/voice/profiles/") and self.path.endswith("/delete"):
                self._send_json(503, {"error": "temporary cleanup unavailable"})
                return
            super().do_POST()

    CleanupUnavailableHandler.resolved_reviews = False
    CleanupUnavailableHandler.deleted_profiles = []
    artist_song = tmp_path / "artist.wav"
    user_vocals = tmp_path / "user.wav"
    _write_wav(artist_song)
    _write_wav(user_vocals)
    server, base_url = _serve(CleanupUnavailableHandler)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_production_smoke.py",
                "--mode",
                "full",
                "--base-url",
                base_url,
                "--artist-song",
                str(artist_song),
                "--user-vocals",
                str(user_vocals),
                "--output-dir",
                str(tmp_path / "latest"),
                "--run-id",
                "cleanup-503",
                "--poll-interval",
                "0.01",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((tmp_path / "latest" / "cleanup-503" / "production_smoke.json").read_text())
    assert report["ok"] is True
    assert report["full"]["ok"] is True
    assert report["full"]["cleanup"]
    assert all(entry["status"] == 503 for entry in report["full"]["cleanup"])
    assert all(entry["ok"] is False for entry in report["full"]["cleanup"])
    assert all(entry["blocking"] is False for entry in report["full"]["cleanup"])


def test_production_smoke_full_mode_fails_when_required_stems_missing(tmp_path):
    class NoStemHandler(_SmokeHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/api/v1/convert/status/convert-smoke":
                self._send_json(200, {"job_id": "convert-smoke", "status": "completed"})
            else:
                super().do_GET()

    NoStemHandler.resolved_reviews = False
    NoStemHandler.deleted_profiles = []
    artist_song = tmp_path / "artist.wav"
    user_vocals = tmp_path / "user.wav"
    _write_wav(artist_song)
    _write_wav(user_vocals)
    server, base_url = _serve(NoStemHandler)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_production_smoke.py",
                "--mode",
                "full",
                "--base-url",
                base_url,
                "--artist-song",
                str(artist_song),
                "--user-vocals",
                str(user_vocals),
                "--output-dir",
                str(tmp_path / "latest"),
                "--run-id",
                "no-stems",
                "--poll-interval",
                "0.01",
                "--require-stems",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()

    assert result.returncode == 1
    report = json.loads((tmp_path / "latest" / "no-stems" / "production_smoke.json").read_text())
    assert report["ok"] is False
    assert "without stem_urls" in report["error"]


def test_production_monitoring_workflow_has_expected_schedules():
    workflow = (PROJECT_ROOT / ".github/workflows/production-monitoring.yml").read_text(encoding="utf-8")
    assert 'cron: "30 8 * * *"' in workflow
    assert 'cron: "0 10 * * 1"' in workflow
    assert "scripts/run_production_smoke.py" in workflow
    assert "scripts/run_rollback_drill.py" in workflow


def test_rollback_drill_dry_run_builds_safe_command_plan(monkeypatch, tmp_path):
    module = _load_script_module("rollback_drill_script", PROJECT_ROOT / "scripts/run_rollback_drill.py")
    monkeypatch.setattr(module, "_current_tags", lambda: ["v1.0.0"])
    monkeypatch.setattr(module, "_current_ref", lambda: "50cae249")
    monkeypatch.setattr(module, "_previous_tag", lambda _current_tag: "rc-2026-04-27")
    monkeypatch.setattr(
        module,
        "_health_checks",
        lambda _base_url: [
            {"url": "http://example.test/api/v1/health", "ok": True},
            {"url": "http://example.test/ready", "ok": True},
            {"url": "http://example.test/api/v1/metrics", "ok": True},
        ],
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **_kwargs: {
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "ok": True,
        },
    )
    args = module.parse_args([
        "--base-url",
        "http://example.test",
        "--output",
        str(tmp_path / "rollback.json"),
    ])

    report = module.run_drill(args)

    assert report["ok"] is True
    assert report["mode"] == "dry-run"
    assert report["target_ref"] == "rc-2026-04-27"
    assert report["executed"] == []
    assert ["git", "checkout", "rc-2026-04-27"] in report["command_plan"]
