"""Socket event contract: every event the GUI subscribes to must be emitted.

This exists because of a one-word mismatch. The svc_fork training lane - the
app's actual serving path - emitted ``training.progress`` while the client
subscribed to ``training_progress``, so the live training card sat frozen for
every user. TypeScript could not see it (both sides are string literals in
different languages), the backend suite could not see it (no test asserted the
underscore name), and the e2e suite could not see it (``mockApi.ts`` aborts all
socket.io traffic). A static cross-language scan is the only place this class
of defect is visible.

Three properties this scan has to have, each learned the hard way:

* **AST, not regex, on the backend.** A line-oriented scan reports
  ``conversion_cancelled`` as dead. It is not: it is the second argument of a
  *multiline* ``_emit_socket_events('job_failed', 'conversion_cancelled', ...)``
  in ``web/job_manager.py``, sitting on a line that does not contain "emit".
* **Key on (namespace, event), not bare names.** ``separation_progress`` is
  emitted only on ``/karaoke``. Keyed on bare names the two namespaces'
  vocabularies contaminate each other and the test stops catching anything.
* **Assert frontend is a subset, not equal.** The backend deliberately
  double-emits dotted and underscore aliases (``training.progress`` /
  ``training_progress``, ``job_completed`` / ``conversion_complete``). Extra
  backend events are correct, not drift.
"""
import ast
import re
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "src" / "auto_voice"
FRONTEND_ROOT = REPO_ROOT / "frontend" / "src"

DEFAULT_NAMESPACE = "/"

# ---------------------------------------------------------------------------
# KNOWN-DEAD SUBSCRIPTIONS - do not extend this list to make a failure go away.
#
# These three are subscribed in frontend/src/services/api.ts and are emitted by
# nothing in the backend, so `wsManager.onGPUMetrics()` is a public method that
# can never fire. (No line numbers here on purpose - the failure message below
# computes the live ones.)
#
# TODO: either implement the emits or delete the subscriptions and the
# onGPUMetrics accessor. They are listed here only so that a *new* mismatch
# still fails this test loudly instead of hiding behind a pre-existing one.
# ---------------------------------------------------------------------------
KNOWN_DEAD_SUBSCRIPTIONS = {
    (DEFAULT_NAMESPACE, "gpu_metrics"),
    (DEFAULT_NAMESPACE, "model_loaded"),
    (DEFAULT_NAMESPACE, "model_unloaded"),
}

# socket.io raises these on the client itself; no backend emit exists or should.
CLIENT_LIFECYCLE_EVENTS = {
    "connect",
    "connect_error",
    "connect_timeout",
    "disconnect",
    "reconnect",
    "reconnect_attempt",
    "reconnect_error",
    "reconnect_failed",
}

# ponytail: skipped by name - websocket.ts has zero importers, so its
# subscriptions can never receive anything and are not part of the contract.
SKIPPED_FRONTEND_FILES = {"websocket.ts"}

# Helpers that forward their first positional argument to socketio.emit().
EMIT_WRAPPERS = {"emit", "_emit", "_emit_event"}
# Helpers taking (primary_event, alias_event, ...) - both names go out.
EMIT_PAIR_WRAPPERS = {"_emit_socket_events"}


# ---------------------------------------------------------------------------
# Backend scan (ast)
# ---------------------------------------------------------------------------

def _base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _namespace_class_default(node):
    """Default namespace of a ``flask_socketio.Namespace`` subclass, if any.

    ``KaraokeNamespace(Namespace)`` declares ``__init__(self, namespace='/karaoke')``,
    which makes every bare ``emit()`` in its body a ``/karaoke`` emit.
    """
    if not isinstance(node, ast.ClassDef):
        return None
    if not any(_base_name(base) == "Namespace" for base in node.bases):
        return None
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            args = child.args.args
            defaults = child.args.defaults
            for arg, default in zip(reversed(args), reversed(defaults)):
                if arg.arg == "namespace" and isinstance(default, ast.Constant):
                    return default.value
    return None


def _call_namespaces(tree):
    """Map each Call inside a Namespace subclass to that class's namespace."""
    namespaces = {}
    for node in ast.walk(tree):
        namespace = _namespace_class_default(node)
        if namespace is None:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                namespaces[id(child)] = namespace
    return namespaces


def _call_scopes(tree):
    """Map each Call to the string constants assigned to local names around it.

    ``conversion_workflows._update_workflow`` picks its event name through an
    if/elif chain into a local ``event_name`` before emitting it - the only
    non-literal emit in the backend.
    """
    scopes = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {}
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            if not (isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.setdefault(target.id, set()).add(stmt.value.value)
        if not names:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                merged = scopes.setdefault(id(child), {})
                for name, values in names.items():
                    merged.setdefault(name, set()).update(values)
    return scopes


def _literals(arg, scope):
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return {arg.value}
    if isinstance(arg, ast.Name):
        return set(scope.get(arg.id, ()))
    return set()


@lru_cache(maxsize=1)
def backend_emits():
    """(namespace, event) -> {"file:line"} for every literal backend emit."""
    emits = {}
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        namespaces = _call_namespaces(tree)
        scopes = _call_scopes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = _base_name(node.func)
            if name in EMIT_WRAPPERS:
                args = node.args[:1]
            elif name in EMIT_PAIR_WRAPPERS:
                args = node.args[:2]
            else:
                continue

            namespace = namespaces.get(id(node), DEFAULT_NAMESPACE)
            for keyword in node.keywords:
                if keyword.arg == "namespace" and isinstance(keyword.value, ast.Constant):
                    namespace = keyword.value.value

            scope = scopes.get(id(node), {})
            where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            for arg in args:
                for event in _literals(arg, scope):
                    emits.setdefault((namespace, event), set()).add(where)
    return emits


# ---------------------------------------------------------------------------
# Frontend scan (regex - every subscription is a string literal)
# ---------------------------------------------------------------------------

ON_CALL_RE = re.compile(r"\.on\(\s*['\"]([^'\"]+)['\"]")
# api.ts subscribes by .forEach()-ing a literal array cast to WSEventType[].
EVENT_ARRAY_RE = re.compile(r"\[([^\]]*)\]\s*as\s+WSEventType\[\]", re.S)
STRING_RE = re.compile(r"['\"]([^'\"]+)['\"]")
# The namespace is the tail of the io() url; a non-literal url means default.
IO_URL_RE = re.compile(r"\bio\(\s*(`[^`]*`|'[^']*'|\"[^\"]*\")")
NAMESPACE_TAIL_RE = re.compile(r"(/[A-Za-z0-9_-]+)['\"`]$")


def _frontend_namespace(source):
    match = IO_URL_RE.search(source)
    if not match:
        return DEFAULT_NAMESPACE
    tail = NAMESPACE_TAIL_RE.search(match.group(1))
    return tail.group(1) if tail else DEFAULT_NAMESPACE


@lru_cache(maxsize=1)
def frontend_subscriptions():
    """(namespace, event) -> {"file:line"} for every socket.io subscription."""
    subscriptions = {}
    paths = sorted(
        set(FRONTEND_ROOT.rglob("*.ts")) | set(FRONTEND_ROOT.rglob("*.tsx"))
    )
    for path in paths:
        if path.name in SKIPPED_FRONTEND_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        if "socket.io-client" not in source:
            continue
        namespace = _frontend_namespace(source)
        relative = path.relative_to(REPO_ROOT)

        def record(event, offset):
            if event in CLIENT_LIFECYCLE_EVENTS:
                return
            line = source.count("\n", 0, offset) + 1
            subscriptions.setdefault((namespace, event), set()).add(f"{relative}:{line}")

        for match in ON_CALL_RE.finditer(source):
            record(match.group(1), match.start(1))
        for array in EVENT_ARRAY_RE.finditer(source):
            for match in STRING_RE.finditer(array.group(1)):
                record(match.group(1), array.start(1) + match.start(1))
    return subscriptions


def _hint(namespace, event, emits):
    """Why a subscription looks dead - usually a namespace or alias mismatch."""
    elsewhere = sorted(ns for (ns, name) in emits if name == event and ns != namespace)
    if elsewhere:
        return f"backend emits it on {', '.join(elsewhere)} instead"
    for variant in (event.replace("_", "."), event.replace(".", "_")):
        if variant != event and (namespace, variant) in emits:
            return f"backend emits '{variant}' instead"
    return "emitted nowhere"


@pytest.mark.smoke
class TestSocketEventContract:
    def test_scan_is_not_vacuous(self):
        """Guard the scanners themselves: a broken extractor passes silently.

        Each pinned pair is a distinct extraction path - lose one and the
        subset assertion below quietly stops testing anything.
        """
        emits = backend_emits()
        subscriptions = frontend_subscriptions()

        # Multiline _emit_socket_events second argument - the reason for `ast`.
        assert (DEFAULT_NAMESPACE, "conversion_cancelled") in emits
        # Namespace attribution, on both sides.
        assert ("/karaoke", "separation_progress") in emits
        assert ("/karaoke", "separation_progress") in subscriptions
        # The `as WSEventType[]` array - 13 subscriptions vanish if this breaks.
        assert (DEFAULT_NAMESPACE, "training_progress") in subscriptions
        # Local-name resolution through the if/elif chain in _update_workflow.
        assert (DEFAULT_NAMESPACE, "conversion_workflow_ready") in emits

    def test_every_frontend_subscription_is_emitted(self):
        emits = backend_emits()
        missing = {
            pair: files
            for pair, files in frontend_subscriptions().items()
            if pair not in emits and pair not in KNOWN_DEAD_SUBSCRIPTIONS
        }
        assert not missing, (
            "Socket events the frontend subscribes to but no backend emit produces:\n"
            + "\n".join(
                f"  {namespace} '{event}' - {_hint(namespace, event, emits)}"
                f" (subscribed at {', '.join(sorted(files))})"
                for (namespace, event), files in sorted(missing.items())
            )
        )

    def test_known_dead_allowlist_has_not_rotted(self):
        emits = backend_emits()
        subscriptions = frontend_subscriptions()
        revived = sorted(pair for pair in KNOWN_DEAD_SUBSCRIPTIONS if pair in emits)
        assert not revived, (
            f"Now emitted by the backend - drop from KNOWN_DEAD_SUBSCRIPTIONS: {revived}"
        )
        unsubscribed = sorted(
            pair for pair in KNOWN_DEAD_SUBSCRIPTIONS if pair not in subscriptions
        )
        assert not unsubscribed, (
            f"No longer subscribed - drop from KNOWN_DEAD_SUBSCRIPTIONS: {unsubscribed}"
        )
