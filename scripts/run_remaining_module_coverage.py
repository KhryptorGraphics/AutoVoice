#!/usr/bin/env python3
"""Run the AV-isz.20 remaining-module coverage gate.

The autovoice-thor environment trips a torch import bug when coverage resolves
importable source packages directly. Using coverage's timid tracer together
with directory-based source paths in .coveragerc avoids the duplicate-docstring
failure while still counting unexecuted files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE_JSON = ROOT / "coverage.remaining.json"

TEST_PATHS = [
    "tests/test_training_job_manager.py",
    "tests/test_training_job_manager_phase4.py",
    "tests/test_training_job_manager_phase5.py",
    "tests/test_training_websocket_events.py",
    "tests/test_trainer_comprehensive.py",
    "tests/test_trainer_phase6.py",
    "tests/test_gpu_enforcement.py",
    "tests/test_model_versioning.py",
    "tests/test_training_scheduler.py",
    "tests/test_training_ui_routes.py",
    "tests/test_web_api_comprehensive.py",
    "tests/test_web_api_training.py",
    "tests/test_web_api_remaining_endpoints.py",
    "tests/test_web_training_status.py",
    "tests/test_platform_scripts.py",
    "tests/test_quality_upgrades.py",
    "tests/test_web_app_lifecycle.py",
    "tests/test_web_utils.py",
    "tests/test_web_api_config_endpoints.py",
    "tests/test_web_api_utility.py",
    "tests/test_secret_key_security.py",
    "tests/test_frontend_playwright_smoke.py",
]

MODULE_THRESHOLDS = {
    "src/auto_voice/models/pupu_vocoder.py": 80.0,
    "src/auto_voice/training/gpu_enforcement.py": 80.0,
    "src/auto_voice/training/job_manager.py": 80.0,
    "src/auto_voice/training/model_versioning.py": 80.0,
    "src/auto_voice/training/trainer.py": 80.0,
    "src/auto_voice/training/training_scheduler.py": 80.0,
    "src/auto_voice/web/api.py": 70.0,
    "src/auto_voice/web/app.py": 80.0,
    "src/auto_voice/web/persistence.py": 80.0,
    "src/auto_voice/web/training_ui.py": 70.0,
    "src/auto_voice/web/utils.py": 80.0,
}


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    pythonpath = env.get("PYTHONPATH", "")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{pythonpath}" if pythonpath else src_path
    env.setdefault("COVERAGE_RCFILE", str(ROOT / ".coveragerc"))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def load_percent_covered(path: str, files: dict[str, dict[str, object]]) -> float | None:
    entry = files.get(path)
    if entry is None:
        return None
    summary = entry.get("summary", {})
    return float(summary.get("percent_covered", 0.0))


def main() -> int:
    run([sys.executable, "-m", "coverage", "erase"])
    run([sys.executable, "-m", "coverage", "run", "-m", "pytest", "-p", "no:cov", *TEST_PATHS, "-q"])
    run([sys.executable, "-m", "coverage", "json", "-o", str(COVERAGE_JSON)])
    report_include = ",".join(MODULE_THRESHOLDS)
    run([sys.executable, "-m", "coverage", "report", "-m", f"--include={report_include}"])

    data = json.loads(COVERAGE_JSON.read_text())
    files = data.get("files", {})

    failures: list[str] = []
    print("\nRemaining-module coverage thresholds:")
    for path, minimum in MODULE_THRESHOLDS.items():
        covered = load_percent_covered(path, files)
        if covered is None:
            failures.append(f"{path}: missing from coverage report")
            print(f"  FAIL {path}: missing")
            continue
        status = "PASS" if covered >= minimum else "FAIL"
        print(f"  {status} {path}: {covered:.1f}% (target {minimum:.1f}%)")
        if covered < minimum:
            failures.append(f"{path}: {covered:.1f}% < {minimum:.1f}%")

    if failures:
        print("\nCoverage gate failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nRemaining-module coverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
