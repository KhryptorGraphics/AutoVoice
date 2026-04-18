from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@pytest.mark.browser
@pytest.mark.slow
def test_frontend_playwright_smoke_suite():
    if shutil.which("npm") is None:
        pytest.skip("npm is required for the frontend Playwright smoke suite")

    if not (FRONTEND_DIR / "node_modules" / "@playwright" / "test").exists():
        pytest.skip("@playwright/test is not installed")

    result = subprocess.run(
        ["npm", "run", "test:e2e", "--", "--reporter=line"],
        cwd=FRONTEND_DIR,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
