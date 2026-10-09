"""Internal routes (read-only catalog export + safe item sync)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from .config import catalog_path
from .security import token_matches

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["push-internal"])

PUSH_ROOT = Path(__file__).resolve().parent.parent
MAX_SYNC_ITEMS = 5000
_VECTOR_FILES = (
    "item_embeddings.npy",
    "item_topic_vecs.npy",
    "topic_vecs.npy",
    "faiss_index.bin",
    "vector_manifest.json",
    "help_topic_vec.npy",
)


def require_push_internal(x_internal_token: str | None = Header(default=None)) -> None:
    if os.getenv("PUSH_ENV", "development").strip().lower() == "production":
        token = os.environ.get("PUSH_INTERNAL_TOKEN", "").strip()
    else:
        token = os.environ.get("PUSH_INTERNAL_TOKEN", os.environ.get("PUSH_ADMIN_TOKEN", "")).strip()
    if not token:
        raise HTTPException(status_code=503, detail="PUSH_INTERNAL_TOKEN not configured")
    if not x_internal_token or not token_matches(token, x_internal_token.strip()):
        raise HTTPException(status_code=403, detail="Invalid internal token")


class ItemSyncRequest(BaseModel):
    items: list[dict] = Field(default_factory=list)


def _data_dir() -> Path:
    return PUSH_ROOT / "data"


def _backup_vectors(backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in _VECTOR_FILES:
        source = _data_dir() / name
        if source.is_file():
            shutil.copy2(source, backup_dir / name)


def _restore_vectors(backup_dir: Path) -> None:
    if not backup_dir.is_dir():
        return
    for name in _VECTOR_FILES:
        saved = backup_dir / name
        target = _data_dir() / name
        if saved.is_file():
            shutil.copy2(saved, target)


def _validate_items(items: list[dict]) -> list[dict]:
    if not items:
        raise HTTPException(status_code=400, detail="items must be a non-empty array")
    if len(items) > MAX_SYNC_ITEMS:
        raise HTTPException(status_code=400, detail=f"items exceed {MAX_SYNC_ITEMS}")
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"items[{idx}] must be an object")
        if not str(item.get("item_id", "")).strip():
            raise HTTPException(status_code=400, detail=f"items[{idx}].item_id is required")
        if not str(item.get("title", "")).strip():
            raise HTTPException(status_code=400, detail=f"items[{idx}].title is required")
    return items


@router.get("/items/export")
async def export_items(_: None = Depends(require_push_internal)):
    path = Path(catalog_path())
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"catalog not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="catalog must be a JSON array")
    return data


@router.post("/items/sync")
async def sync_items(body: ItemSyncRequest, _: None = Depends(require_push_internal)):
    items = _validate_items(body.items)
    path = Path(catalog_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")
    vector_backup = _data_dir() / ".sync-backup"
    if path.is_file():
        shutil.copy2(path, backup)
    _backup_vectors(vector_backup)

    def _restore_all() -> None:
        if backup.is_file():
            shutil.copy2(backup, path)
        _restore_vectors(vector_backup)

    try:
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = _build_generate_data_cmd(path)
        await asyncio.to_thread(subprocess.run, cmd, cwd=str(PUSH_ROOT), check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as exc:
        logger.error("generate_data failed during sync: %s", exc)
        _restore_all()
        raise HTTPException(
            status_code=422, detail="generate_data failed; catalog and vector files restored"
        ) from exc
    except Exception as exc:
        logger.error("item sync failed: %s", exc)
        _restore_all()
        raise HTTPException(
            status_code=500, detail="item sync failed; catalog and vector files restored"
        ) from exc

    from .main import cache_service, reload_handoff_from_vectors, vector_service

    try:
        enc = vector_service.reload_data()
        flushed = await cache_service.flush_all()
        reload_handoff_from_vectors()
    except Exception as exc:
        logger.error("reload after sync failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="files were written; this worker failed to reload, other workers pick up the new mtime",
        ) from exc

    shutil.rmtree(vector_backup, ignore_errors=True)
    return {"ok": True, "item_count": len(items), "encoder": enc, "cache_flushed": flushed}


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
