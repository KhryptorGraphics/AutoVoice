"""Catch relative imports that reach above the top-level package.

``api_runtime.py`` (package ``auto_voice.web``) carried
``from ...runtime_contract import MULTI_SPEAKER_SETTING_KEYS``. Three dots
escapes ``auto_voice`` entirely, so the app-settings update endpoint raised
``ImportError: attempted relative import beyond top-level package`` - but only
when that endpoint was actually called, because the import sits inside the
function body. Nothing at import time, no linter here, and no test that
exercised the route caught it.

An AST sweep is the cheap general guard: it does not care whether the import
is at module scope or buried in a branch that only production reaches.
"""
import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "auto_voice"
MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))


def _max_level(module: Path) -> int:
    """Relative-import levels this module may legally use.

    level 1 is the module's own package, each extra dot climbs one more. A
    module in ``auto_voice.web`` may therefore use at most 2.
    """
    rel = module.relative_to(PACKAGE_ROOT.parent)
    parts = list(rel.parts[:-1])  # drop the filename; keep package dirs
    return len(parts)


def test_package_root_is_discoverable():
    """Guard the guard: an empty sweep would pass vacuously forever."""
    assert PACKAGE_ROOT.is_dir(), PACKAGE_ROOT
    assert len(MODULES) > 50, f"expected a real package, found {len(MODULES)} modules"


@pytest.mark.parametrize("module", MODULES, ids=lambda p: str(p.name))
def test_no_relative_import_escapes_the_package(module):
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    except (SyntaxError, UnicodeDecodeError) as exc:
        pytest.fail(f"{module} does not parse: {exc}")

    limit = _max_level(module)
    offenders = [
        (node.lineno, node.level, node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > limit
    ]
    assert not offenders, (
        f"{module.relative_to(PACKAGE_ROOT.parent)} has relative imports above the "
        f"package root (max {limit} dots here): "
        + ", ".join(f"line {ln}: {'.' * lvl}{mod}" for ln, lvl, mod in offenders)
    )
