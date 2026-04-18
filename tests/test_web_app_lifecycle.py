"""Lifecycle and graceful-shutdown coverage for the web app and CLI."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def app_lifecycle():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": False,
            "voice_cloning_enabled": False,
        }
    )
    app.socketio = socketio
    return app


def test_wait_for_requests_returns_true_after_drain(app_lifecycle):
    with app_lifecycle._request_lock:
        app_lifecycle._active_requests = 1

    def _finish_request():
        time.sleep(0.1)
        with app_lifecycle._request_lock:
            app_lifecycle._active_requests = 0

    worker = threading.Thread(target=_finish_request)
    worker.start()
    try:
        assert app_lifecycle.wait_for_requests(timeout=1.0) is True
    finally:
        worker.join()


def test_wait_for_requests_times_out_when_stuck(app_lifecycle):
    with app_lifecycle._request_lock:
        app_lifecycle._active_requests = 1

    assert app_lifecycle.wait_for_requests(timeout=0.05) is False


def test_http_exception_releases_request_counter(app_lifecycle):
    client = app_lifecycle.test_client()
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.get_json()["status_code"] == 404
    assert app_lifecycle._active_requests == 0


def test_generic_exception_releases_request_counter_once(app_lifecycle):
    def _boom():
        raise RuntimeError("boom")

    app_lifecycle.add_url_rule("/boom", "boom", _boom)
    client = app_lifecycle.test_client()
    response = client.get("/boom")

    assert response.status_code == 500
    assert response.get_json()["status_code"] == 500
    assert app_lifecycle._active_requests == 0


def test_setup_signal_handlers_drains_requests_and_cleans_up():
    from auto_voice.cli import setup_signal_handlers

    fake_app = SimpleNamespace(
        wait_for_requests=MagicMock(return_value=True),
        job_manager=SimpleNamespace(stop_cleanup_thread=MagicMock()),
    )
    fake_socketio = SimpleNamespace(stop=MagicMock())

    handlers = {}

    def _capture_signal(signum, handler):
        handlers[signum] = handler

    with patch("signal.signal", side_effect=_capture_signal), patch(
        "torch.cuda.is_available", return_value=True
    ), patch("torch.cuda.empty_cache") as empty_cache:
        setup_signal_handlers(fake_app, fake_socketio)

        with pytest.raises(SystemExit) as exc_info:
            handlers[15](15, None)

    assert exc_info.value.code == 0
    fake_app.wait_for_requests.assert_called_once_with(timeout=30.0)
    fake_app.job_manager.stop_cleanup_thread.assert_called_once()
    fake_socketio.stop.assert_called_once()
    empty_cache.assert_called_once()
