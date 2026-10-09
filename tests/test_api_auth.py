"""Public API token: open when unset, required when PUSH_API_TOKEN is set."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import server.main as main

    monkeypatch.setattr(main.vector_service, "load", lambda: None)
    main.vector_service.index = object()

    async def _no_redis():
        main.cache_service.redis = None

    monkeypatch.setattr(main.cache_service, "connect", _no_redis)
    with TestClient(main.app) as test_client:
        yield test_client


def test_feedback_open_without_token(client, monkeypatch):
    monkeypatch.delenv("PUSH_API_TOKEN", raising=False)
    response = client.post("/api/v1/feedback", json={"item_id": "i_001", "event": "no_click"})
    assert response.status_code == 200


def test_feedback_requires_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("PUSH_API_TOKEN", "secret")
    missing = client.post("/api/v1/feedback", json={"item_id": "i_001", "event": "no_click"})
    assert missing.status_code == 401
    ok = client.post(
        "/api/v1/feedback",
        json={"item_id": "i_001", "event": "dwell_end", "watch_duration_sec": 40, "item_duration_sec": 60},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200
    assert ok.json()["reason"] == "dwell_above_threshold"
    wrong = client.post(
        "/api/v1/feedback",
        json={"item_id": "i_001", "event": "no_click"},
        headers={"Authorization": "Bearer nope"},
    )
    assert wrong.status_code == 403


def test_rate_limit(client, monkeypatch):
    from server import ratelimit

    ratelimit._buckets.clear()
    monkeypatch.setenv("PUSH_API_TOKEN", "limit-secret")
    monkeypatch.setenv("PUSH_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("PUSH_TRUSTED_PROXIES", "testclient")
    headers = {"Authorization": "Bearer limit-secret", "X-Forwarded-For": "203.0.113.50"}
    body = {"item_id": "i_001", "event": "no_click"}
    first = client.post("/api/v1/feedback", json=body, headers=headers)
    second = client.post("/api/v1/feedback", json=body, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 429


def test_spoofed_forwarded_for_does_not_reset_limit(client, monkeypatch):
    from server import ratelimit

    ratelimit._buckets.clear()
    monkeypatch.delenv("PUSH_TRUSTED_PROXIES", raising=False)
    monkeypatch.setenv("PUSH_API_TOKEN", "limit-secret")
    monkeypatch.setenv("PUSH_RATE_LIMIT_PER_MIN", "1")
    body = {"item_id": "i_001", "event": "no_click"}
    auth = {"Authorization": "Bearer limit-secret"}
    first = client.post("/api/v1/feedback", json=body, headers={**auth, "X-Forwarded-For": "203.0.113.50"})
    second = client.post("/api/v1/feedback", json=body, headers={**auth, "X-Forwarded-For": "203.0.113.51"})
    assert first.status_code == 200
    assert second.status_code == 429


def test_production_refuses_in_process_training(client, monkeypatch):
    monkeypatch.setenv("PUSH_ENV", "production")
    monkeypatch.setenv("PUSH_ADMIN_TOKEN", "admin-secret")
    monkeypatch.delenv("TRAINING_IN_PROCESS", raising=False)
    response = client.post(
        "/admin/training/run",
        json={"force": False},
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert response.status_code == 403


def test_ready_requires_index(client):
    import server.main as main

    main.vector_service.index = None
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
