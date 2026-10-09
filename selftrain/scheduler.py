"""Standalone training scheduler. Do not import this from the serving process.

Run: python -m selftrain.scheduler
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .constants import load_training_config, training_dir

logger = logging.getLogger(__name__)

_STATE_NAME = "scheduler_state.json"


def _field_matches(field: str, value: int) -> bool:
    field = field.strip()
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    return int(field) == value


def cron_matches(expr: str, moment: datetime) -> bool:
    """Match a 5-field cron expression. Day-of-week uses 0 or 7 for Sunday."""
    parts = (expr or "").split()
    if len(parts) != 5:
        raise ValueError(f"cron must have 5 fields: {expr}")
    minute, hour, day, month, dow = parts
    cron_dow = (moment.weekday() + 1) % 7
    dow_ok = _field_matches(dow, cron_dow) or (dow.strip() == "7" and cron_dow == 0)
    return (
        _field_matches(minute, moment.minute)
        and _field_matches(hour, moment.hour)
        and _field_matches(day, moment.day)
        and _field_matches(month, moment.month)
        and dow_ok
    )


def should_run(
    now: datetime,
    last_run: datetime | None,
    new_samples: int,
    *,
    min_new_samples: int,
    cooldown_hours: float,
) -> tuple[bool, str]:
    if last_run is not None:
        elapsed = (now - last_run).total_seconds()
        if elapsed < float(cooldown_hours) * 3600:
            return False, "cooldown"
    if new_samples < int(min_new_samples):
        return False, "too_few_samples"
    return True, "due"


def _state_path() -> Path:
    return training_dir() / _STATE_NAME


def load_last_run() -> datetime | None:
    path = _state_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        text = raw.get("last_run")
        if not text:
            return None
        return datetime.fromisoformat(text)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_last_run(moment: datetime) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_run": moment.astimezone(timezone.utc).isoformat()}),
        encoding="utf-8",
    )


async def count_new_exposures(min_needed: int) -> int:
    from .store import create_store

    store = create_store(load_training_config())
    try:
        watermark = await store.get_watermark("push_id")
        rows = await store.get_new_exposures(watermark, max(int(min_needed), 1))
        return len(rows)
    finally:
        await store.close()


async def tick(config: dict | None = None) -> dict:
    cfg = config or load_training_config()
    schedule = cfg.get("schedule") or {}
    if not schedule.get("enabled"):
        return {"ok": False, "reason": "schedule_disabled"}

    now = datetime.now(timezone.utc)
    new_samples = await count_new_exposures(int(schedule.get("min_new_samples", 50)))
    ok, reason = should_run(
        now,
        load_last_run(),
        new_samples,
        min_new_samples=int(schedule.get("min_new_samples", 50)),
        cooldown_hours=float(schedule.get("cooldown_hours", 24)),
    )
    if not ok:
        logger.info("scheduler skip: %s (new_samples=%s)", reason, new_samples)
        return {"ok": False, "reason": reason, "new_samples": new_samples}

    from .store.redis import RedisHelper

    helper = RedisHelper()
    locked = False
    if not helper.url:
        logger.warning("REDIS_URL unset; scheduler lock is process-local")
    else:
        connected = await helper.connect()
        if not connected:
            logger.warning("REDIS_URL set but Redis is down; skipping this cycle")
            return {"ok": False, "reason": "lock_unavailable"}
        locked = await helper.acquire_lock("training-cycle", ttl_sec=3600)
        if not locked:
            logger.info("scheduler skip: lock held")
            return {"ok": False, "reason": "lock_held"}

    try:
        from .loop import run_cycle

        result = await run_cycle(cfg, force=False)
        save_last_run(datetime.now(timezone.utc))
        return result
    finally:
        if locked:
            await helper.release_lock("training-cycle")


def _schedule_enabled(cfg: dict) -> bool:
    return bool((cfg.get("schedule") or {}).get("enabled"))


def run_forever(poll_seconds: float = 20) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("training scheduler started (separate from the push server)")
    last_slot = ""
    while True:
        cfg = load_training_config()
        schedule = cfg.get("schedule") or {}
        now = datetime.now()
        slot = now.strftime("%Y%m%d%H%M")
        if _schedule_enabled(cfg) and slot != last_slot:
            try:
                due = cron_matches(str(schedule.get("cron") or "0 3 * * *"), now)
            except ValueError as exc:
                logger.warning("bad cron: %s", exc)
                due = False
            if due:
                last_slot = slot
                try:
                    asyncio.run(tick(cfg))
                except Exception:
                    logger.exception("scheduler tick failed")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_forever()
