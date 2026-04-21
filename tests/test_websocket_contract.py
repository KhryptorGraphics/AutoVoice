"""Socket.IO contract coverage for default and karaoke namespaces."""

from __future__ import annotations

from flask_socketio import SocketIOTestClient
import pytest


@pytest.fixture
def app_with_socketio_contract():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": False,
            "voice_cloning_enabled": False,
            "karaoke_enabled": False,
        },
        testing=True,
    )
    app.socketio = socketio
    return app, socketio


def _event_names(events):
    return [event["name"] for event in events]


def test_join_job_acknowledges_subscription(app_with_socketio_contract):
    app, socketio = app_with_socketio_contract
    client = SocketIOTestClient(app, socketio)

    try:
        client.connect()
        client.emit("join_job", {"job_id": "job-123"})

        received = client.get_received()
        assert "joined_job" in _event_names(received)
        joined = next(event for event in received if event["name"] == "joined_job")
        assert joined["args"][0] == {"job_id": "job-123"}
    finally:
        client.disconnect()


def test_join_job_requires_job_id(app_with_socketio_contract):
    app, socketio = app_with_socketio_contract
    client = SocketIOTestClient(app, socketio)

    try:
        client.connect()
        client.emit("join_job", {})

        received = client.get_received()
        assert "job_subscription_error" in _event_names(received)
        error_event = next(
            event for event in received if event["name"] == "job_subscription_error"
        )
        assert error_event["args"][0]["message"] == "job_id is required"
    finally:
        client.disconnect()


def test_joined_job_room_receives_room_scoped_conversion_events(app_with_socketio_contract):
    app, socketio = app_with_socketio_contract
    subscribed_client = SocketIOTestClient(app, socketio)
    other_client = SocketIOTestClient(app, socketio)

    try:
        subscribed_client.connect()
        other_client.connect()

        subscribed_client.emit("join_job", {"job_id": "job-123"})
        subscribed_client.get_received()
        other_client.get_received()

        socketio.emit(
            "conversion_progress",
            {
                "job_id": "job-123",
                "progress": 45,
                "stage": "encoding",
                "message": "Converting vocals...",
            },
            room="job-123",
        )

        subscribed_events = subscribed_client.get_received()
        other_events = other_client.get_received()

        assert "conversion_progress" in _event_names(subscribed_events)
        assert "conversion_progress" not in _event_names(other_events)
    finally:
        subscribed_client.disconnect()
        other_client.disconnect()


def test_leave_job_stops_room_updates(app_with_socketio_contract):
    app, socketio = app_with_socketio_contract
    client = SocketIOTestClient(app, socketio)

    try:
        client.connect()
        client.emit("join_job", {"job_id": "job-123"})
        client.get_received()

        client.emit("leave_job", {"job_id": "job-123"})
        leave_events = client.get_received()
        assert "left_job" in _event_names(leave_events)

        socketio.emit(
            "conversion_progress",
            {
                "job_id": "job-123",
                "progress": 80,
                "stage": "mixing",
                "message": "Finalizing output...",
            },
            room="job-123",
        )

        post_leave_events = client.get_received()
        assert "conversion_progress" not in _event_names(post_leave_events)
    finally:
        client.disconnect()
