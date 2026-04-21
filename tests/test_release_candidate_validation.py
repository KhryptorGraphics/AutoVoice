from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        payload = None
        status = 200
        if self.path == "/health":
            payload = {"status": "healthy"}
        elif self.path == "/ready":
            payload = {"ready": True}
        elif self.path == "/api/v1/metrics":
            payload = {"total_conversions": 1}
        else:
            status = 404
            payload = {"error": "not found"}

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        return


def test_validate_release_candidate_script(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report_dir = tmp_path / "reports"
        env = os.environ.copy()
        pythonpath = str(PROJECT_ROOT / "src")
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--skip-compose",
                "--report-dir",
                str(report_dir),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        report_path = Path(result.stdout.strip())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["compose"]["ok"] is True
        assert report["repo_boundaries"]["ok"] is True
        assert all(check["ok"] for check in report["checks"])
    finally:
        server.shutdown()
        server.server_close()
