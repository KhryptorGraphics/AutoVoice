"""Public/commercial launch gate contract tests."""

from __future__ import annotations


def test_public_commercial_readiness_reports_blockers(client):
    response = client.get("/api/v1/public-commercial/readiness")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ready"] is False
    assert payload["status"] == "blocked"
    assert payload["scope"] == "public_commercial"
    assert payload["closure_rule"]

    blocker_ids = {blocker["id"] for blocker in payload["blockers"]}
    assert {
        "account_auth",
        "tenant_isolation",
        "persistent_quotas",
        "abuse_review",
        "hosted_evidence",
        "legal_approval",
        "public_ingress_review",
    } <= blocker_ids
    assert {blocker["bead_id"] for blocker in payload["blockers"]} >= {
        "AV-3rfd.18.1",
        "AV-3rfd.18.3",
        "AV-3rfd.18.4",
        "AV-3rfd.18.5",
        "AV-3rfd.18.6",
    }


def test_public_commercial_readiness_accepts_only_persistent_quota_backend(
    client,
    monkeypatch,
    tmp_path,
):
    evidence = tmp_path / "evidence.json"
    legal = tmp_path / "legal-approval.md"
    review = tmp_path / "public-ingress-review.md"
    for path in (evidence, legal, review):
        path.write_text("approved\n", encoding="utf-8")

    monkeypatch.setenv("AUTOVOICE_ACCOUNT_AUTH_PROVIDER", "oidc")
    monkeypatch.setenv("AUTOVOICE_TENANT_ISOLATION_ENABLED", "true")
    monkeypatch.setenv("AUTOVOICE_ABUSE_REVIEW_ENABLED", "true")
    monkeypatch.setenv("AUTOVOICE_HOSTED_PUBLIC_EVIDENCE_PATH", str(evidence))
    monkeypatch.setenv("AUTOVOICE_LEGAL_APPROVAL_PATH", str(legal))
    monkeypatch.setenv("AUTOVOICE_PUBLIC_INGRESS_REVIEW_PATH", str(review))

    monkeypatch.setenv("AUTOVOICE_QUOTA_BACKEND", "memory")
    blocked_response = client.get("/api/v1/public-commercial/readiness")
    blocked_payload = blocked_response.get_json()
    assert blocked_payload["ready"] is False
    assert [blocker["id"] for blocker in blocked_payload["blockers"]] == ["persistent_quotas"]

    monkeypatch.setenv("AUTOVOICE_QUOTA_BACKEND", "postgres")
    ready_response = client.get("/api/v1/public-commercial/readiness")
    ready_payload = ready_response.get_json()
    assert ready_payload["ready"] is True
    assert ready_payload["status"] == "ready"
    assert ready_payload["blockers"] == []
    assert {item["id"] for item in ready_payload["satisfied"]} >= {
        "account_auth",
        "tenant_isolation",
        "persistent_quotas",
        "abuse_review",
        "hosted_evidence",
        "legal_approval",
        "public_ingress_review",
    }


def test_public_commercial_readiness_is_authenticated_in_public_mode(
    flask_app,
    monkeypatch,
):
    monkeypatch.setenv("AUTOVOICE_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    client = flask_app.test_client()

    unauthorized = client.get("/api/v1/public-commercial/readiness")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/api/v1/public-commercial/readiness",
        headers={"Authorization": "Bearer unit-token"},
    )
    assert authorized.status_code == 200
    assert authorized.get_json()["public_deployment_mode"] is True
