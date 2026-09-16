"""The duet routing ledger must stay readable by the GUI that renders it.

The ledger is the feature's safety mitigation: `register-validate` measured the
diarizer's clusters at ~40% impure on a dense mix, so an operator assigning both
registers in good faith can get majority-female audio sung by the male model.
No automatic gate earned its place on that data, so the split is surfaced
instead of blocked - which only works while the frontend can actually read the
rows.

A renamed ledger key does not raise anywhere. The pipeline keeps emitting rows,
the panel keeps rendering, and every value silently becomes `undefined` - the
mitigation goes quiet in exactly the case it exists for. This pins the two
halves together.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "auto_voice" / "inference" / "singing_conversion_pipeline.py"
API_TS = ROOT / "frontend" / "src" / "services" / "api.ts"
PANEL = ROOT / "frontend" / "src" / "components" / "RegisterRoutingPanel.tsx"


def _ledger_row_keys() -> set[str]:
    """Keys of the ledger row dict the pipeline appends, read as text.

    Importing the pipeline drags in torch; the settings contract test reads
    source the same way for the same reason.
    """
    src = PIPELINE.read_text()
    match = re.search(r"ledger\.append\(\{(.*?)\n\s*\}\)", src, re.S)
    assert match, "ledger.append({...}) not found in the pipeline"
    return set(re.findall(r"'(\w+)':", match.group(1)))


def _ts_interface_fields(name: str) -> set[str]:
    src = API_TS.read_text()
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", src, re.S)
    assert match, f"{name} interface not found in api.ts"
    body = re.sub(r"//[^\n]*", "", match.group(1))
    return set(re.findall(r"^\s*(\w+)\??:", body, re.M))


def test_every_ledger_key_is_declared_in_the_frontend_type():
    """A key the client cannot name renders as undefined, silently."""
    missing = _ledger_row_keys() - _ts_interface_fields("RegisterRoutingRow")
    assert not missing, (
        f"the pipeline's register_routing rows carry {sorted(missing)}, which "
        f"RegisterRoutingRow in {API_TS.relative_to(ROOT)} does not declare"
    )


def test_frontend_type_invents_no_ledger_field():
    """The reverse drift: a panel reading a key the pipeline never writes."""
    extra = _ts_interface_fields("RegisterRoutingRow") - _ledger_row_keys()
    assert not extra, (
        f"RegisterRoutingRow declares {sorted(extra)}, which the pipeline's "
        f"ledger row does not contain - those render as undefined"
    )


def test_panel_renders_the_nothing_routed_case():
    """The silent no-op is the whole reason this panel exists."""
    panel = PANEL.read_text()
    assert "register-routing-noop" in panel, (
        "the assigned-but-nothing-routed banner is gone from "
        f"{PANEL.relative_to(ROOT)}; that case looks identical to success "
        "without it"
    )


def test_panel_does_not_present_the_measurement_as_a_verdict():
    """The pitch classifier was disproved; the copy must not imply otherwise."""
    panel = PANEL.read_text()
    assert "register-routing-disagreement" in panel, (
        "the measured-vs-assigned disagreement is no longer rendered"
    )
    assert "118 Hz" in panel, (
        "the disagreement copy no longer states the overlap that makes the "
        "measurement a hint rather than a verdict"
    )
    assert "not detected from the audio" in panel, (
        "the panel no longer says registers come from the operator's slot "
        "assignment, which is the one thing that stops it reading as automatic"
    )
