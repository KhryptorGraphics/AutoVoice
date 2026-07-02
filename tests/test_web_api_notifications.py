"""Tests for the notification webhook API and dispatcher."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def app(tmp_path):
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
            "singing_conversion_enabled": True,
            "voice_cloning_enabled": True,
        }
    )
    app.socketio = socketio
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def _payload(**overrides):
    payload = {
        "name": "CI hook",
        "url": "https://example.com/hook",
        "events": ["training_complete"],
    }
    payload.update(overrides)
    return payload


class TestWebhookCrud:
    def test_crud_roundtrip(self, client):
        created = client.post("/api/v1/notifications/webhooks", json=_payload())
        assert created.status_code == 201
        record = created.get_json()
        webhook_id = record["id"]
        assert record["name"] == "CI hook"
        assert record["url"] == "https://example.com/hook"
        assert record["events"] == ["training_complete"]
        assert record["enabled"] is True
        assert record["created_at"]

        listed = client.get("/api/v1/notifications/webhooks").get_json()
        assert listed["count"] == 1
        assert listed["webhooks"][0]["id"] == webhook_id

        updated = client.post(
            "/api/v1/notifications/webhooks",
            json=_payload(id=webhook_id, name="Renamed", enabled=False),
        )
        assert updated.status_code == 201
        updated_record = updated.get_json()
        assert updated_record["id"] == webhook_id
        assert updated_record["name"] == "Renamed"
        assert updated_record["enabled"] is False
        assert updated_record["created_at"] == record["created_at"]

        listed = client.get("/api/v1/notifications/webhooks").get_json()
        assert listed["count"] == 1

        assert client.delete(f"/api/v1/notifications/webhooks/{webhook_id}").status_code == 204
        assert client.delete(f"/api/v1/notifications/webhooks/{webhook_id}").status_code == 404
        assert client.get("/api/v1/notifications/webhooks").get_json()["count"] == 0

    @pytest.mark.parametrize(
        "payload",
        [
            _payload(url="ftp://example.com/hook"),
            _payload(url="not-a-url"),
            _payload(name=""),
            _payload(name="x" * 101),
            _payload(events=[]),
            _payload(events=["unknown_event"]),
            _payload(events="training_complete"),
        ],
    )
    def test_validation_errors(self, client, payload):
        response = client.post("/api/v1/notifications/webhooks", json=payload)
        assert response.status_code == 400


class TestWebhookTestFire:
    def _create(self, client):
        return client.post("/api/v1/notifications/webhooks", json=_payload()).get_json()["id"]

    def test_test_fire_delivers(self, client):
        webhook_id = self._create(client)
        with patch("auto_voice.web.api_notifications.requests.post") as mock_post:
            mock_post.return_value = Mock(status_code=200)
            response = client.post(f"/api/v1/notifications/webhooks/{webhook_id}/test")

        assert response.status_code == 200
        body = response.get_json()
        assert body == {"status": "delivered", "delivered": True, "error": None}
        args, kwargs = mock_post.call_args
        assert args[0] == "https://example.com/hook"
        assert kwargs["json"]["event"] == "test"
        assert kwargs["json"]["webhook_id"] == webhook_id
        assert kwargs["timeout"] == 5

    def test_test_fire_reports_failure(self, client):
        webhook_id = self._create(client)
        with patch(
            "auto_voice.web.api_notifications.requests.post",
            side_effect=ConnectionError("refused"),
        ):
            response = client.post(f"/api/v1/notifications/webhooks/{webhook_id}/test")

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "failed"
        assert body["delivered"] is False
        assert "refused" in body["error"]

    def test_test_fire_unknown_webhook(self, client):
        assert client.post("/api/v1/notifications/webhooks/missing/test").status_code == 404


class TestDispatcher:
    def test_dispatch_posts_only_to_enabled_subscribed_webhooks(self, app):
        from auto_voice.web.api_notifications import dispatch_webhooks

        app.state_store.save_webhook({
            "id": "hook-enabled",
            "name": "enabled",
            "url": "https://example.com/enabled",
            "events": ["training_complete"],
            "enabled": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        app.state_store.save_webhook({
            "id": "hook-disabled",
            "name": "disabled",
            "url": "https://example.com/disabled",
            "events": ["training_complete"],
            "enabled": False,
            "created_at": "2026-01-01T00:00:01+00:00",
        })

        with patch("auto_voice.web.api_notifications.requests.post") as mock_post:
            dispatch_webhooks(
                "training_complete",
                {"job_id": "job-1"},
                app.state_store.data_dir,
                wait=True,
            )

        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        assert args[0] == "https://example.com/enabled"
        assert kwargs["json"]["event"] == "training_complete"
        assert kwargs["json"]["data"] == {"job_id": "job-1"}
        assert kwargs["timeout"] == 5

    def test_dispatch_skips_unsubscribed_events(self, app):
        from auto_voice.web.api_notifications import dispatch_webhooks

        app.state_store.save_webhook({
            "id": "hook-conversion",
            "name": "conversion only",
            "url": "https://example.com/conversion",
            "events": ["conversion_complete"],
            "enabled": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        })

        with patch("auto_voice.web.api_notifications.requests.post") as mock_post:
            dispatch_webhooks(
                "training_complete",
                {"job_id": "job-1"},
                app.state_store.data_dir,
                wait=True,
            )

        mock_post.assert_not_called()

    def test_dispatch_logs_delivery_failure(self, app):
        from auto_voice.web.api_notifications import dispatch_webhooks

        app.state_store.save_webhook({
            "id": "hook-broken",
            "name": "broken",
            "url": "https://example.com/broken",
            "events": ["job_failed"],
            "enabled": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        })

        with patch(
            "auto_voice.web.api_notifications.requests.post",
            side_effect=ConnectionError("refused"),
        ) as mock_post:
            # Must not raise.
            dispatch_webhooks(
                "job_failed",
                {"job_id": "job-1"},
                app.state_store.data_dir,
                wait=True,
            )

        assert mock_post.call_count == 1
