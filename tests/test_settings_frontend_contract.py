"""Every backend pipeline setting must be reachable from the GUI.

Eight of twelve `PIPELINE_SETTING_KEYS` were accepted by the API but declared
nowhere on the client, so operators had tuned values they could neither see nor
change without editing source - including two (`fork_hq_stereo_width`,
`multi_speaker_backing_whole_voiced_min`) that were already persisted with
non-default values on the live box.

This is the same class of defect the project plan calls out: a knob that
silently changes what a conversion does, reachable only by reading source.
"""
import re
from pathlib import Path

import pytest

from auto_voice.runtime_contract import PIPELINE_SETTING_KEYS

ROOT = Path(__file__).resolve().parents[1]
API_TS = ROOT / "frontend" / "src" / "services" / "api.ts"
PANEL = ROOT / "frontend" / "src" / "components" / "SystemConfigPanel.tsx"


def _app_settings_block() -> str:
    src = API_TS.read_text()
    m = re.search(r"export interface AppSettings \{(.*?)\n\}", src, re.S)
    assert m, "AppSettings interface not found in api.ts"
    return m.group(1)


@pytest.mark.parametrize("key", PIPELINE_SETTING_KEYS)
def test_key_is_declared_in_frontend_type(key):
    """A key the client cannot name is a key the GUI can never send."""
    assert re.search(rf"^\s*{re.escape(key)}\??:", _app_settings_block(), re.M), (
        f"{key} is in PIPELINE_SETTING_KEYS but missing from AppSettings in "
        f"{API_TS.relative_to(ROOT)} - the GUI cannot send it"
    )


@pytest.mark.parametrize("key", PIPELINE_SETTING_KEYS)
def test_key_has_a_control_in_the_settings_panel(key):
    """Declared-but-unrendered is the same defect one layer up."""
    assert key in PANEL.read_text(), (
        f"{key} has no control in {PANEL.relative_to(ROOT)} - it is unreachable "
        f"for an operator who is not editing source"
    )


def test_no_frontend_key_is_unknown_to_the_backend():
    """The reverse drift: a control that PATCHes a key the API will reject."""
    declared = set(re.findall(r"^\s*(\w+)\??:", _app_settings_block(), re.M))
    # Keys the client legitimately owns that are not pipeline knobs.
    client_only = {
        "preferred_pipeline", "preferred_offline_pipeline",
        "preferred_live_pipeline", "last_updated",
    }
    unknown = declared - set(PIPELINE_SETTING_KEYS) - client_only
    assert not unknown, (
        f"AppSettings declares {sorted(unknown)}, which the backend does not "
        f"accept - a PATCH would be rejected as unsupported"
    )
