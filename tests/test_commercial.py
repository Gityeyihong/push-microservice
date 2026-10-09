"""Session locking, production gate, and scheduler decision (no API server)."""

from datetime import datetime, timedelta, timezone

from policy.state import StateManager
from server.security import production_config_errors
from selftrain.scheduler import cron_matches, should_run


def test_urge_increments_are_not_lost(tmp_path):
    path = tmp_path / "urge.json"
    first = StateManager(state_file=str(path))
    start = first.get("urge")
    second = StateManager(state_file=str(path))
    first.bump_urge(0.1)
    second.bump_urge(0.1)
    assert abs(first.get("urge") - (start + 0.2)) < 1e-9


def test_recent_pushes_count_each_timestamp(tmp_path):
    path = tmp_path / "pushes.json"
    state = StateManager(state_file=str(path))
    state.note_push(1_000, 10_000)
    state.note_push(2_000, 10_000)
    assert state.count_recent_pushes(2_500, 10_000) == 2


def test_session_updates_do_not_clobber(tmp_path):
    path = tmp_path / "session.json"
    first = StateManager(state_file=str(path))
    second = StateManager(state_file=str(path))
    first.set("tau", 0.4)
    second.set("urge", 0.9)
    assert first.get("urge") == 0.9
    assert second.get("tau") == 0.4


def test_rate_limit_key_uses_forwarded_for_only_from_trusted_proxy(monkeypatch):
    from server.ratelimit import client_key

    monkeypatch.setenv("PUSH_TRUSTED_PROXIES", "10.0.0.1,10.0.0.9")
    forwarded = client_key("203.0.113.8, 10.0.0.2", "10.0.0.1")
    same_client = client_key("203.0.113.8", "10.0.0.9")
    proxy_only = client_key(None, "10.0.0.1")
    assert forwarded == same_client
    assert forwarded != proxy_only


def test_rate_limit_ignores_untrusted_forwarded_for(monkeypatch):
    from server.ratelimit import client_address

    monkeypatch.delenv("PUSH_TRUSTED_PROXIES", raising=False)
    assert client_address("203.0.113.8, 10.0.0.2", "10.0.0.1") == "10.0.0.1"


def test_production_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("PUSH_ENV", "production")
    monkeypatch.setenv("HMAC_SECRET_KEY", "change-me-in-production")
    monkeypatch.setenv("PUSH_API_TOKEN", "")
    monkeypatch.setenv("PUSH_ADMIN_TOKEN", "")
    errors = production_config_errors()
    assert "HMAC_SECRET_KEY" in errors
    assert "PUSH_API_TOKEN" in errors
    assert "PUSH_ADMIN_TOKEN" in errors
    assert "PUSH_INTERNAL_TOKEN" in errors


def test_production_rejects_mock_vectors_and_placeholder_redis(monkeypatch):
    monkeypatch.setenv("PUSH_ENV", "production")
    monkeypatch.setenv("HMAC_SECRET_KEY", "real-hmac-secret")
    monkeypatch.setenv("PUSH_API_TOKEN", "api")
    monkeypatch.setenv("PUSH_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("PUSH_INTERNAL_TOKEN", "internal")
    monkeypatch.setenv("REDIS_PASSWORD", "dev-only-change-me")
    monkeypatch.setenv("PUSH_USE_MOCK_VECTORS", "1")
    errors = production_config_errors()
    assert "REDIS_PASSWORD" in errors
    assert "PUSH_USE_MOCK_VECTORS" in errors


def test_production_accepts_complete_config(monkeypatch):
    monkeypatch.setenv("PUSH_ENV", "production")
    monkeypatch.setenv("HMAC_SECRET_KEY", "real-hmac-secret")
    monkeypatch.setenv("PUSH_API_TOKEN", "api")
    monkeypatch.setenv("PUSH_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("PUSH_INTERNAL_TOKEN", "internal")
    monkeypatch.setenv("REDIS_PASSWORD", "real-redis-secret")
    monkeypatch.setenv("PUSH_USE_MOCK_VECTORS", "0")
    assert production_config_errors() == []


def test_session_save_sets_ttl(monkeypatch):
    monkeypatch.setenv("PUSH_SESSION_TTL_SEC", "604800")
    from server.push_session_store import RedisStateBackend

    class FakeRedis:
        def __init__(self):
            self.saved = None

        def set(self, key, value, ex=None):
            self.saved = (key, value, ex)

    fake = FakeRedis()
    RedisStateBackend(fake, "push:session:abc").save({"urge": 0.5})
    assert fake.saved[0] == "push:session:abc"
    assert fake.saved[2] == 604800


def test_development_skips_production_gate(monkeypatch):
    monkeypatch.setenv("PUSH_ENV", "development")
    monkeypatch.setenv("HMAC_SECRET_KEY", "change-me-in-production")
    assert production_config_errors() == []


def test_should_run_respects_cooldown():
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    last = now - timedelta(hours=1)
    ok, reason = should_run(now, last, 100, min_new_samples=50, cooldown_hours=24)
    assert ok is False
    assert reason == "cooldown"


def test_cron_matches_daily_hour():
    assert cron_matches("0 3 * * *", datetime(2026, 1, 2, 3, 0))
    assert not cron_matches("0 3 * * *", datetime(2026, 1, 2, 4, 0))
