"""Token checks and production startup gates."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException

PLACEHOLDER_SECRETS = {
    "",
    "change-me-in-production",
    "change-this-in-production",
    "dev-only-change-me",
}


def token_matches(expected: str, provided: str) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def check_bearer(authorization: str | None, expected: str) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    provided = authorization.removeprefix("Bearer ").strip()
    if not token_matches(expected, provided):
        raise HTTPException(status_code=403, detail="Invalid token")


def production_config_errors() -> list[str]:
    if os.getenv("PUSH_ENV", "development").strip().lower() != "production":
        return []
    errors: list[str] = []
    secret = os.getenv("HMAC_SECRET_KEY", "change-me-in-production")
    if secret.strip() in PLACEHOLDER_SECRETS:
        errors.append("HMAC_SECRET_KEY")
    if not os.getenv("PUSH_API_TOKEN", "").strip():
        errors.append("PUSH_API_TOKEN")
    if not os.getenv("PUSH_ADMIN_TOKEN", "").strip():
        errors.append("PUSH_ADMIN_TOKEN")
    if not os.getenv("PUSH_INTERNAL_TOKEN", "").strip():
        errors.append("PUSH_INTERNAL_TOKEN")
    if os.getenv("REDIS_PASSWORD", "").strip() in PLACEHOLDER_SECRETS:
        errors.append("REDIS_PASSWORD")
    if os.getenv("PUSH_USE_MOCK_VECTORS", "0").strip().lower() in ("1", "true", "yes"):
        errors.append("PUSH_USE_MOCK_VECTORS")
    return errors


def enforce_production_config() -> None:
    errors = production_config_errors()
    if errors:
        raise RuntimeError(
            "refusing to start with PUSH_ENV=production; fix: " + ", ".join(errors)
        )
