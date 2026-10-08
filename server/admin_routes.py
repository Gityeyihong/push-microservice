"""Push admin API routes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .admin_audit import append_audit, read_audit
from .ab_resolver import (
    decision_config_path,
    get_ab_status,
    load_ab_experiment,
    load_profiles,
    save_ab_experiment,
    save_profiles,
)
from .models import DecisionConfigModel

router = APIRouter(prefix="/admin", tags=["push-admin"])


def require_push_admin(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("PUSH_ADMIN_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="PUSH_ADMIN_TOKEN not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != token:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _sessions_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sessions"


def _touch_config_mtime() -> None:
    """Ensure mtime changes so _current_decision_config reloads."""
    path = decision_config_path()
    if path.exists():
        path.touch()


@router.get("/status")
async def admin_status(_: None = Depends(require_push_admin)):
    from .main import _current_decision_config, cache_service, vector_service

    enc = vector_service.encoder_status() if vector_service.index is not None else {}
    cfg = _current_decision_config()
    return {
        "status": "ok",
        "redis": cache_service.available,
        "encoder": enc,
        "decision_config": cfg.model_dump(),
        **get_ab_status(),
    }


@router.get("/decision-config")
async def get_decision_config(_: None = Depends(require_push_admin)):
    path = decision_config_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    content = path.read_text(encoding="utf-8")
    return {"path": str(path), "content": content}


@router.put("/decision-config")
async def put_decision_config(
    request: Request,
    _: None = Depends(require_push_admin),
):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    try:
        validated = DecisionConfigModel(**body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid decision config: {exc}") from exc

    path = decision_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(validated.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _touch_config_mtime()
    from .main import _current_decision_config

    cfg = _current_decision_config()
    append_audit("decision_config.update", detail={"path": str(path)})
    return {"ok": True, "decision_config": cfg.model_dump()}


@router.get("/profiles")
async def get_profiles(_: None = Depends(require_push_admin)):
    return load_profiles()


@router.put("/profiles")
async def put_profiles(body: dict, _: None = Depends(require_push_admin)):
    if not isinstance(body, dict) or "profiles" not in body:
        raise HTTPException(status_code=400, detail="Body must contain 'profiles' object")
    if not isinstance(body["profiles"], dict):
        raise HTTPException(status_code=400, detail="'profiles' must be an object")
    save_profiles(body)
    append_audit("profiles.update", detail={"count": len(body["profiles"])})
    return body


class AbExperimentBody(BaseModel):
    active: bool = False
    name: str = ""
    variant_a: str = ""
    variant_b: str = ""
    split_b_percent: int = Field(50, ge=0, le=100)


@router.get("/ab-experiments")
async def get_ab_experiments(_: None = Depends(require_push_admin)):
    return get_ab_status()


@router.post("/ab-experiments")
async def post_ab_experiment(
    body: AbExperimentBody,
    _: None = Depends(require_push_admin),
):
    data = body.model_dump()
    profiles = load_profiles().get("profiles", {})
    for key in ("variant_a", "variant_b"):
        name = data.get(key, "")
        if name and name not in profiles:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {name}")
    save_ab_experiment(data)
    append_audit("ab_experiment.update", detail=data)
    return data


@router.delete("/ab-experiments")
async def delete_ab_experiment(_: None = Depends(require_push_admin)):
    data = {"active": False, "name": "", "variant_a": "", "variant_b": "", "split_b_percent": 50}
    save_ab_experiment(data)
    append_audit("ab_experiment.stop", detail={})
    return data


@router.get("/sessions")
async def list_sessions(_: None = Depends(require_push_admin)):
    sessions_dir = _sessions_dir()
    if not sessions_dir.is_dir():
        return {"sessions": []}
    sessions = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = {}
        sessions.append(
            {
                "session_id": path.stem,
                "path": str(path),
                "state": raw,
            }
        )
    return {"sessions": sessions, "count": len(sessions)}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _: None = Depends(require_push_admin)):
    sessions_dir = _sessions_dir()
    safe = session_id.replace("/", "_").replace("\\", "_")[:64]
    path = sessions_dir / f"{safe}.json"
    if path.exists():
        path.unlink()
    from .push_session_store import get_session_state

    get_session_state.cache_clear()
    append_audit("session.reset", detail={"session_id": session_id})
    return {"ok": True, "session_id": session_id}


@router.post("/videos/reload")
async def reload_videos(_: None = Depends(require_push_admin)):
    """Reload video metadata + FAISS from /app/data without rebuilding the image."""
    from .main import cache_service, reload_advisor_from_vectors, vector_service

    try:
        enc = vector_service.reload_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    reload_advisor_from_vectors()

    flushed = False
    if cache_service.available:
        try:
            await cache_service.flush_all()
            flushed = True
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    append_audit("videos.reload", detail={"encoder": enc, "cache_flushed": flushed})
    return {"ok": True, "encoder": enc, "cache_flushed": flushed}


@router.post("/cache/flush")
async def flush_cache(_: None = Depends(require_push_admin)):
    from .main import cache_service

    if not cache_service.available:
        return {"ok": False, "flushed": 0, "reason": "redis_unavailable"}
    try:
        await cache_service.flush_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    append_audit("cache.flush", detail={})
    return {"ok": True, "flushed": True}


@router.get("/audit")
async def admin_audit(limit: int = 50, offset: int = 0, _: None = Depends(require_push_admin)):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    return read_audit(limit=limit, offset=offset)
