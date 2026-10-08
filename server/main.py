import logging
import time
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware

from .config import SERVER_HOST, SERVER_PORT
from .models import PushRequest, PushResponse, FeedbackEvent, FeedbackResponse, PolicyConfigModel
from .vector_service import VectorService
from .cache_service import CacheService
from .ranking import apply_ranking, extract_last_user_message
from .push_decision_service import run_push_decision
from .push_session_store import get_session_state
from .admin_routes import router as admin_router
from .internal_routes import router as internal_router
from .training_routes import router as training_router
from handoff import load_handoff
from policy import BuiltinThresholdPolicy, classify_feedback, load_policy_config, reload_policy_config
from selftrain.collector import TrainingCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

vector_service = VectorService()
cache_service = CacheService()
collector = TrainingCollector()
handoff_service = None
_policy_config_mtime: float = 0.0


def _current_policy_config() -> PolicyConfigModel:
    global _policy_config_mtime
    from pathlib import Path

    cfg_path = os.environ.get("POLICY_CONFIG_PATH", "")
    if not cfg_path:
        cfg_path = str(Path(__file__).resolve().parent.parent / "config" / "policy.json")
    try:
        mtime = os.path.getmtime(cfg_path)
        if mtime != _policy_config_mtime:
            reload_policy_config()
            _policy_config_mtime = mtime
    except OSError:
        pass
    return PolicyConfigModel(**load_policy_config().to_dict())


def _policy_config_for_request(session_id: str | None) -> PolicyConfigModel:
    _current_policy_config()
    from .ab_resolver import resolve_config_for_session

    return PolicyConfigModel(**resolve_config_for_session(session_id))


def reload_handoff_from_vectors() -> None:
    global handoff_service
    handoff_service = load_handoff(help_topic_vec=vector_service.help_topic_vec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Push Microservice...")
    vector_service.load()
    reload_handoff_from_vectors()
    await cache_service.connect()
    if cache_service.available:
        logger.info("Redis cache connected.")
    else:
        logger.warning("Redis not available, running without cache.")
    yield
    await cache_service.disconnect()
    logger.info("Server shutdown complete.")


app = FastAPI(title="Push Microservice", version="2.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.include_router(admin_router)
app.include_router(internal_router)
app.include_router(training_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info("%s %s -> %d (%.3fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.get("/health")
async def health():
    enc = vector_service.encoder_status() if vector_service.index is not None else {}
    return {
        "status": "ok",
        "redis": cache_service.available,
        "push_use_mock_vectors": os.getenv("PUSH_USE_MOCK_VECTORS", "0"),
        **enc,
    }


def _extract_user_message(context: str) -> str:
    last = extract_last_user_message(context) or ""
    if last:
        return last
    for line in context.strip().split("\n"):
        if line.startswith("用户:"):
            return line.removeprefix("用户:").strip()
        if line.lower().startswith("user:"):
            return line[5:].strip()
    return ""


async def _build_push_response(
    request: PushRequest,
    candidates,
    handoff_suggested: bool,
    policy_config: PolicyConfigModel,
) -> PushResponse:
    user_message = _extract_user_message(request.context)
    push_events, state_summary = run_push_decision(
        session_id=request.session_id,
        user_message=user_message,
        candidates=candidates,
        positive_rate=request.positive_rate,
        handoff_suggested=handoff_suggested,
        signals=request.signals,
        state=request.state,
        policy_config=policy_config.model_dump(),
    )
    try:
        pushed_ids = [
            e.data.get("item_id")
            for e in push_events
            if e.type == "item_push" and e.data.get("item_id")
        ]
        if pushed_ids:
            await collector.record_push(
                session_id=request.session_id,
                context=request.context,
                item_ids=pushed_ids,
                intent_query=getattr(request.intent, "query", "") or "",
            )
    except Exception as exc:
        logger.debug("collector.record_push skipped: %s", exc)

    return PushResponse(
        candidates=candidates,
        handoff_suggested=handoff_suggested,
        generated_at=datetime.now(timezone.utc).isoformat(),
        policy_config=policy_config,
        push_events=push_events,
        state_summary=state_summary,
    )


@app.post("/api/v1/push", response_model=PushResponse)
async def push(request: PushRequest):
    if not request.context.strip():
        raise HTTPException(status_code=400, detail="context cannot be empty")

    policy_config = _policy_config_for_request(request.session_id)
    intent = request.intent
    skip_cache = intent is not None and getattr(intent, "confidence", 0) >= 0.5

    cached = None if skip_cache else await cache_service.get(request.context)
    if cached:
        logger.info("Cache hit for context hash")
        raw_candidates = cached.get("candidates", [])
        guarded = apply_ranking(
            request.context,
            raw_candidates,
            vector_service.item_metadata,
            lexical_primary=vector_service._use_mock,
            intent=intent,
        )
        handoff_suggested = bool(cached.get("handoff_suggested")) or bool(
            request.signals.get("risk_alert")
        )
        return await _build_push_response(request, guarded, handoff_suggested, policy_config)

    candidates, ctx_vec, _ = vector_service.process_request(request.context, intent=intent)
    candidates = apply_ranking(
        request.context,
        candidates,
        vector_service.item_metadata,
        lexical_primary=vector_service._use_mock,
        intent=intent,
    )

    effective_handoff = bool(request.signals.get("risk_alert")) or handoff_service.should_handoff(
        context=request.context,
        signals=request.signals,
        state=request.state,
        positive_rate=request.positive_rate,
        context_vector=ctx_vec,
    )

    cache_payload = {
        "candidates": [c.model_dump() if hasattr(c, "model_dump") else c for c in candidates],
        "handoff_suggested": effective_handoff,
    }
    await cache_service.set(request.context, cache_payload)

    return await _build_push_response(request, candidates, effective_handoff, policy_config)


FEEDBACK_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reports", "feedback_events.jsonl"
)


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def feedback(event: FeedbackEvent):
    consecutive_negative = 0
    if event.session_id:
        state = get_session_state(event.session_id)
        consecutive_negative = state.get("consecutive_negative") or 0

    feedback_type, reason = classify_feedback(
        event=event.event,
        watch_duration_sec=event.watch_duration_sec,
        item_duration_sec=event.item_duration_sec,
        consecutive_negative=consecutive_negative,
        was_clicked=event.event in ("watch_end", "dismiss", "share", "favorite"),
    )

    if event.session_id:
        engine = BuiltinThresholdPolicy(state=get_session_state(event.session_id))
        engine.record_engagement(
            item_id=event.item_id,
            event=event.event,
            watch_duration_sec=event.watch_duration_sec,
            item_duration_sec=event.item_duration_sec,
        )

    log_entry = {
        "item_id": event.item_id,
        "event": event.event,
        "watch_duration_sec": event.watch_duration_sec,
        "item_duration_sec": event.item_duration_sec,
        "session_id": event.session_id,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
        "classified_feedback": feedback_type,
        "reason": reason,
    }
    logger.info("feedback_event %s", json.dumps(log_entry, ensure_ascii=False))
    try:
        await collector.record_feedback(
            session_id=event.session_id,
            item_id=event.item_id,
            event=event.event,
            watch_duration_sec=event.watch_duration_sec,
            item_duration_sec=event.item_duration_sec,
            classified=feedback_type,
        )
    except Exception as exc:
        logger.debug("collector.record_feedback skipped: %s", exc)

    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return FeedbackResponse(accepted=True, classified_feedback=feedback_type, reason=reason)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=SERVER_HOST, port=SERVER_PORT, workers=4, log_level="info")
