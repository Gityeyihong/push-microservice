"""Async training store (SQLAlchemy Core) shared by SQLite and Postgres backends."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

metadata = MetaData()

push_events = Table(
    "push_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float),
    Column("session_id", String(128)),
    Column("anchor", Text),
    Column("policy_version", String(32)),
)

push_items = Table(
    "push_items", metadata,
    Column("push_id", Integer, primary_key=True),
    Column("item_id", String(128), primary_key=True),
    Column("rank", Integer),
    Column("score", Float),
    Column("shown", Integer),
)

feedback_events = Table(
    "feedback_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float),
    Column("session_id", String(128)),
    Column("item_id", String(128)),
    Column("event", String(64)),
    Column("watch_ratio", Float),
    Column("reward", Float),
    Column("label", String(32)),
    Column("propensity", Float),
)

samples = Table(
    "samples", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Float),
    Column("anchor", Text),
    Column("positive_id", String(128)),
    Column("negatives", Text),
    Column("reward", Float),
    Column("propensity", Float),
    Column("label", String(32)),
    Column("review_status", String(32)),
    Column("dedup_hash", String(64), unique=True),
)

heldout = Table(
    "heldout", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("anchor", Text),
    Column("candidates", Text),
    Column("labels", Text),
    Column("frozen_at", Float),
)

watermarks = Table(
    "watermarks", metadata,
    Column("name", String(64), primary_key=True),
    Column("value", String(128)),
)


class TrainingStore(ABC):
    @abstractmethod
    async def init_schema(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    # ingestion
    @abstractmethod
    async def record_push(self, *, session_id: str, anchor: str, items: list[dict], policy_version: str = "") -> int: ...

    @abstractmethod
    async def record_feedback(self, *, session_id: str, item_id: str, event: str, watch_ratio: float, reward: float, label: str, propensity: float) -> None: ...

    # incremental extraction
    @abstractmethod
    async def get_watermark(self, name: str) -> int: ...

    @abstractmethod
    async def set_watermark(self, name: str, value: int) -> None: ...

    @abstractmethod
    async def get_new_exposures(self, since_id: int, limit: int) -> list[dict]: ...

    @abstractmethod
    async def get_feedback_for_sessions(self, session_ids: list[str]) -> dict[str, list[dict]]: ...

    # samples
    @abstractmethod
    async def upsert_samples(self, rows: list[dict]) -> int: ...

    @abstractmethod
    async def iter_samples(self, *, limit: int, offset: int = 0, label: str | None = None) -> list[dict]: ...

    @abstractmethod
    async def count_samples(self, label: str | None = None) -> int: ...

    # held-out
    @abstractmethod
    async def save_heldout(self, rows: list[dict]) -> int: ...

    @abstractmethod
    async def load_heldout(self) -> list[dict]: ...

    @abstractmethod
    async def has_heldout(self) -> bool: ...

    # funnel / metrics
    @abstractmethod
    async def funnel_counts(self) -> dict[str, int]: ...

    @abstractmethod
    async def compact(self) -> dict: ...


class SQLStore(TrainingStore):
    def __init__(self, url: str):
        self.url = url
        self.engine: AsyncEngine = create_async_engine(url, future=True)
        self._initialized = False

    async def init_schema(self) -> None:
        if self._initialized:
            return
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        self._initialized = True

    async def close(self) -> None:
        await self.engine.dispose()

    async def record_push(self, *, session_id: str, anchor: str, items: list[dict], policy_version: str = "") -> int:
        await self.init_schema()
        async with self.engine.begin() as conn:
            result = await conn.execute(
                insert(push_events).values(ts=time.time(), session_id=session_id, anchor=anchor, policy_version=policy_version)
            )
            push_id = int(result.inserted_primary_key[0])
            for idx, item in enumerate(items):
                await conn.execute(
                    insert(push_items).values(
                        push_id=push_id,
                        item_id=str(item.get("item_id")),
                        rank=idx,
                        score=float(item.get("score", 0.0)),
                        shown=1 if item.get("shown", True) else 0,
                    )
                )
            return push_id

    async def record_feedback(self, *, session_id: str, item_id: str, event: str, watch_ratio: float, reward: float, label: str, propensity: float) -> None:
        await self.init_schema()
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(feedback_events).values(
                    ts=time.time(), session_id=session_id, item_id=item_id, event=event,
                    watch_ratio=watch_ratio, reward=reward, label=label, propensity=propensity,
                )
            )

    async def get_watermark(self, name: str) -> int:
        await self.init_schema()
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(watermarks.c.value).where(watermarks.c.name == name))).first()
        return int(row[0]) if row and row[0] is not None else 0

    async def set_watermark(self, name: str, value: int) -> None:
        await self.init_schema()
        async with self.engine.begin() as conn:
            exists = (await conn.execute(select(watermarks.c.value).where(watermarks.c.name == name))).first()
            if exists:
                await conn.execute(update(watermarks).where(watermarks.c.name == name).values(value=str(value)))
            else:
                await conn.execute(insert(watermarks).values(name=name, value=str(value)))

    async def get_new_exposures(self, since_id: int, limit: int) -> list[dict]:
        await self.init_schema()
        async with self.engine.connect() as conn:
            push_rows = (
                await conn.execute(
                    select(push_events).where(push_events.c.id > since_id).order_by(push_events.c.id).limit(limit)
                )
            ).mappings().all()
            out = []
            for push in push_rows:
                item_rows = (
                    await conn.execute(
                        select(push_items).where(push_items.c.push_id == push["id"]).order_by(push_items.c.rank)
                    )
                ).mappings().all()
                out.append(
                    {
                        "id": push["id"],
                        "session_id": push["session_id"],
                        "anchor": push["anchor"],
                        "items": [dict(r) for r in item_rows],
                    }
                )
            return out

    async def get_feedback_for_sessions(self, session_ids: list[str]) -> dict[str, list[dict]]:
        if not session_ids:
            return {}
        await self.init_schema()
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(select(feedback_events).where(feedback_events.c.session_id.in_(session_ids)))
            ).mappings().all()
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["session_id"], []).append(dict(r))
        return out

    async def upsert_samples(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        await self.init_schema()
        inserted = 0
        async with self.engine.begin() as conn:
            for row in rows:
                if not row.get("dedup_hash"):
                    continue
                exists = (
                    await conn.execute(select(samples.c.id).where(samples.c.dedup_hash == row["dedup_hash"]))
                ).first()
                if exists:
                    continue
                await conn.execute(
                    insert(samples).values(
                        created_at=time.time(),
                        anchor=row.get("anchor", ""),
                        positive_id=row.get("positive_id", ""),
                        negatives=json.dumps(row.get("negatives", []), ensure_ascii=False),
                        reward=float(row.get("reward", 0.0)),
                        propensity=float(row.get("propensity", 1.0)),
                        label=row.get("label", "unknown"),
                        review_status=row.get("review_status", "none"),
                        dedup_hash=row["dedup_hash"],
                    )
                )
                inserted += 1
        return inserted

    async def iter_samples(self, *, limit: int, offset: int = 0, label: str | None = None) -> list[dict]:
        await self.init_schema()
        stmt = select(samples).order_by(samples.c.id)
        if label:
            stmt = stmt.where(samples.c.label == label)
        stmt = stmt.limit(limit).offset(offset)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["negatives"] = json.loads(d.get("negatives") or "[]")
            except json.JSONDecodeError:
                d["negatives"] = []
            out.append(d)
        return out

    async def count_samples(self, label: str | None = None) -> int:
        await self.init_schema()
        stmt = select(func.count()).select_from(samples)
        if label:
            stmt = stmt.where(samples.c.label == label)
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar() or 0)

    async def save_heldout(self, rows: list[dict]) -> int:
        await self.init_schema()
        async with self.engine.begin() as conn:
            await conn.execute(delete(heldout))
            for row in rows:
                await conn.execute(
                    insert(heldout).values(
                        anchor=row.get("anchor", ""),
                        candidates=json.dumps(row.get("candidates", []), ensure_ascii=False),
                        labels=json.dumps(row.get("labels", {}), ensure_ascii=False),
                        frozen_at=time.time(),
                    )
                )
        return len(rows)

    async def load_heldout(self) -> list[dict]:
        await self.init_schema()
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(heldout))).mappings().all()
        out = []
        for r in rows:
            out.append(
                {
                    "anchor": r["anchor"],
                    "candidates": json.loads(r["candidates"] or "[]"),
                    "labels": json.loads(r["labels"] or "{}"),
                }
            )
        return out

    async def has_heldout(self) -> bool:
        await self.init_schema()
        async with self.engine.connect() as conn:
            return (await conn.execute(select(func.count()).select_from(heldout))).scalar() > 0

    async def funnel_counts(self) -> dict[str, int]:
        await self.init_schema()
        async with self.engine.connect() as conn:
            exposures = int((await conn.execute(select(func.count()).select_from(push_items).where(push_items.c.shown == 1))).scalar() or 0)
            clicks = int((await conn.execute(select(func.count()).select_from(feedback_events).where(feedback_events.c.event == "click"))).scalar() or 0)
            successes = int((await conn.execute(select(func.count()).select_from(feedback_events).where(feedback_events.c.label == "success"))).scalar() or 0)
            failures = int((await conn.execute(select(func.count()).select_from(feedback_events).where(feedback_events.c.label == "failure"))).scalar() or 0)
        return {"exposures": exposures, "clicks": clicks, "successes": successes, "failures": failures}

    async def compact(self) -> dict:
        return {"ok": True, "backend": self.url.split(":", 1)[0]}
