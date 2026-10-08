"""Internal Push routes for admin gateway (read-only export + safe video sync)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from .config import VIDEO_META_PATH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["push-internal"])

PUSH_ROOT = Path(__file__).resolve().parent.parent


def require_push_internal(x_internal_token: str | None = Header(default=None)) -> None:
    token = os.environ.get("PUSH_INTERNAL_TOKEN", os.environ.get("PUSH_ADMIN_TOKEN", "")).strip()
    if not token:
        raise HTTPException(status_code=503, detail="PUSH_INTERNAL_TOKEN not configured")
    if not x_internal_token or x_internal_token.strip() != token:
        raise HTTPException(status_code=403, detail="Invalid internal token")


class VideoSyncRequest(BaseModel):
    videos: list[dict] = Field(default_factory=list)


def _validate_videos(videos: list[dict]) -> list[dict]:
    if not videos:
        raise HTTPException(status_code=400, detail="videos must be a non-empty array")
    for idx, item in enumerate(videos):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"videos[{idx}] must be an object")
        if not str(item.get("video_id", "")).strip():
            raise HTTPException(status_code=400, detail=f"videos[{idx}].video_id is required")
        if not str(item.get("title", "")).strip():
            raise HTTPException(status_code=400, detail=f"videos[{idx}].title is required")
    return videos


@router.get("/videos/export")
async def export_videos(_: None = Depends(require_push_internal)):
    path = Path(VIDEO_META_PATH)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"video catalog not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="video catalog must be a JSON array")
    return data


@router.post("/videos/sync")
async def sync_videos(body: VideoSyncRequest, _: None = Depends(require_push_internal)):
    videos = _validate_videos(body.videos)
    path = Path(VIDEO_META_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")

    if path.is_file():
        shutil.copy2(path, backup)

    try:
        path.write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = _build_generate_data_cmd(path)
        subprocess.run(cmd, cwd=str(PUSH_ROOT), check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as exc:
        logger.error("generate_data failed during sync: %s", exc)
        if backup.is_file():
            shutil.copy2(backup, path)
        raise HTTPException(status_code=422, detail="generate_data failed; index unchanged") from exc
    except Exception as exc:
        logger.error("video sync failed: %s", exc)
        if backup.is_file():
            shutil.copy2(backup, path)
        raise HTTPException(status_code=500, detail="video sync failed; index unchanged") from exc

    from .main import cache_service, reload_advisor_from_vectors, vector_service

    try:
        enc = vector_service.reload_data()
        flushed = await cache_service.flush_all()
        reload_advisor_from_vectors()
    except Exception as exc:
        logger.error("reload after sync failed: %s", exc)
        if backup.is_file():
            shutil.copy2(backup, path)
        raise HTTPException(status_code=500, detail="reload failed; restored backup JSON") from exc

    return {"ok": True, "video_count": len(videos), "encoder": enc, "cache_flushed": flushed}


def _build_generate_data_cmd(path: Path) -> list[str]:
    cmd = [sys.executable, str(PUSH_ROOT / "data" / "generate_data.py")]
    if os.environ.get("PUSH_USE_MOCK_VECTORS", "").strip().lower() in ("1", "true", "yes"):
        cmd.append("--mock")
    data_dir = PUSH_ROOT / "data"
    try:
        rel = path.resolve().relative_to(data_dir.resolve())
        cmd.append(f"--videos={rel.as_posix()}")
    except ValueError:
        cmd.append(f"--videos={path.name}")
    return cmd
