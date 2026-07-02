"""Notification webhook API and dispatcher for training/conversion events."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

VALID_EVENTS = {'training_complete', 'conversion_complete', 'job_failed'}


def _root():
    from . import api as api_root

    return api_root


def register_notification_routes(api_bp: Blueprint) -> None:
    api_bp.add_url_rule('/notifications/webhooks', view_func=list_webhooks, methods=['GET'])
    api_bp.add_url_rule('/notifications/webhooks', view_func=save_webhook, methods=['POST'])
    api_bp.add_url_rule('/notifications/webhooks/<webhook_id>', view_func=delete_webhook, methods=['DELETE'])
    api_bp.add_url_rule('/notifications/webhooks/<webhook_id>/test', view_func=test_webhook, methods=['POST'])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_webhooks():
    """List all configured notification webhooks."""
    webhooks = _root()._get_state_store().list_webhooks()
    return jsonify({'webhooks': webhooks, 'count': len(webhooks)})


def save_webhook():
    """Create or update a notification webhook."""
    root = _root()
    data = request.get_json(silent=True) or {}

    name = str(data.get('name') or '').strip()
    if not name or len(name) > 100:
        return root.validation_error_response('name is required and must be at most 100 characters')

    url = str(data.get('url') or '').strip()
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        return root.validation_error_response('url must be a valid http or https URL')

    events = data.get('events')
    if not isinstance(events, list) or not events or not set(events) <= VALID_EVENTS:
        return root.validation_error_response(
            f"events must be a non-empty subset of {sorted(VALID_EVENTS)}"
        )

    state_store = root._get_state_store()
    webhook_id = str(data.get('id') or uuid.uuid4())
    existing = state_store.get_webhook(webhook_id)
    record = {
        'id': webhook_id,
        'name': name,
        'url': url,
        'events': sorted(set(events)),
        'enabled': bool(data.get('enabled', True)),
        'created_at': (existing or {}).get('created_at') or _utc_now_iso(),
    }
    state_store.save_webhook(record)
    return jsonify(record), 201


def delete_webhook(webhook_id: str):
    """Delete a notification webhook."""
    root = _root()
    if not root._get_state_store().delete_webhook(webhook_id):
        return root.not_found_response('Webhook not found')
    return '', 204


def test_webhook(webhook_id: str):
    """Synchronously deliver a test payload to a webhook."""
    root = _root()
    webhook = root._get_state_store().get_webhook(webhook_id)
    if not webhook:
        return root.not_found_response('Webhook not found')

    payload = {'event': 'test', 'webhook_id': webhook_id, 'timestamp': _utc_now_iso()}
    try:
        response = requests.post(webhook['url'], json=payload, timeout=5)
        response.raise_for_status()
        return jsonify({'status': 'delivered', 'delivered': True, 'error': None})
    except Exception as exc:
        logger.warning("Webhook %s test delivery failed: %s", webhook_id, exc)
        return jsonify({'status': 'failed', 'delivered': False, 'error': str(exc)})


def dispatch_webhooks(
    event_name: str,
    payload: dict[str, Any],
    data_dir: str | Path,
    wait: bool = False,
) -> None:
    """POST an event to every enabled webhook subscribed to it.

    Fire-and-forget: delivery runs in a daemon thread and failures are logged,
    never raised. Structured audit is skipped here because the background
    thread has no Flask app context.
    # ponytail: no retry/queue; add a queue if delivery guarantees ever matter.
    """

    def _deliver() -> None:
        try:
            from .persistence import AppStateStore

            webhooks = [
                webhook
                for webhook in AppStateStore(str(data_dir)).list_webhooks()
                if webhook.get('enabled', True) and event_name in (webhook.get('events') or [])
            ]
            body = {'event': event_name, 'timestamp': _utc_now_iso(), 'data': payload}
            for webhook in webhooks:
                try:
                    requests.post(webhook['url'], json=body, timeout=5)
                except Exception as exc:
                    logger.warning(
                        "Webhook %s delivery failed for event %s: %s",
                        webhook.get('id'), event_name, exc,
                    )
        except Exception as exc:
            logger.warning("Webhook dispatch failed for event %s: %s", event_name, exc)

    thread = threading.Thread(target=_deliver, daemon=True, name='webhook-dispatch')
    thread.start()
    if wait:
        thread.join(timeout=30)
