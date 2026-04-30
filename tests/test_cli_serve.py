from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_voice import cli


def test_serve_ssl_context_requires_cert_and_key(tmp_path: Path):
    cert_path = tmp_path / "local.crt"
    cert_path.write_text("cert", encoding="utf-8")
    args = cli.parse_args(["serve", "--ssl-cert", str(cert_path)])

    with pytest.raises(SystemExit, match="Both --ssl-cert and --ssl-key"):
        cli._resolve_ssl_context(args)


def test_serve_ssl_context_resolves_existing_cert_and_key(tmp_path: Path):
    cert_path = tmp_path / "local.crt"
    key_path = tmp_path / "local.key"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")
    args = cli.parse_args(["serve", "--ssl-cert", str(cert_path), "--ssl-key", str(key_path)])

    assert cli._resolve_ssl_context(args) == (str(cert_path), str(key_path))


def test_serve_main_passes_ssl_context_to_socketio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cert_path = tmp_path / "local.crt"
    key_path = tmp_path / "local.key"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")
    captured = {}

    class FakeSocketIO:
        def run(self, app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

    fake_app = SimpleNamespace()
    monkeypatch.setattr(cli, "create_app", lambda config: (fake_app, FakeSocketIO()))
    monkeypatch.setattr(cli, "setup_signal_handlers", lambda app, socketio: None)

    result = cli.main(
        [
            "serve",
            "--config",
            "",
            "--host",
            "127.0.0.1",
            "--port",
            "5443",
            "--ssl-cert",
            str(cert_path),
            "--ssl-key",
            str(key_path),
        ]
    )

    assert result == 0
    assert captured["app"] is fake_app
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 5443
    assert captured["kwargs"]["ssl_context"] == (str(cert_path), str(key_path))
