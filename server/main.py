import logging
import time
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import SERVER_HOST, SERVER_PORT
from .models import PushRequest, PushResponse, EngagementEvent, EngagementResponse, DecisionConfigModel
from .vector_service import VectorService
from .cache_service import CacheService
from .advisor_service import AdvisorService
from .relevance_guard import apply_relevance_guard, extract_last_user_message
from .push_decision_service import run_push_decision
from .push_session_store import get_session_state
from .admin_routes import router as admin_router
from .internal_routes import router as internal_router
from .training_routes import router as training_router
from mobile.decision_config import load_decision_config, reload_decision_config
from mobile.decision_engine import DecisionEngine
from mobile.engagement import classify_engagement
from selftrain.collector import TrainingCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

vector_service = VectorService()
cache_service = CacheService()
collector = TrainingCollector()
advisor_service = None
_decision_config_mtime: float = 0.0


def _current_decision_config() -> DecisionConfigModel:
    global _decision_config_mtime
    from pathlib import Path
    import os

    cfg_path = os.environ.get("DECISION_CONFIG_PATH", "")
    if not cfg_path:
        cfg_path = str(Path(__file__).resolve().parent.parent / "config" / "decision_config.json")
    try:
        mtime = os.path.getmtime(cfg_path)
        if mtime != _decision_config_mtime:
            reload_decision_config()
            _decision_config_mtime = mtime
    except OSError:
        pass
    return DecisionConfigModel(**load_decision_config().to_dict())


def _decision_config_for_request(session_id: str | None) -> DecisionConfigModel:
    _current_decision_config()
    from .ab_resolver import resolve_config_for_session

    return DecisionConfigModel(**resolve_config_for_session(session_id))


def reload_advisor_from_vectors() -> None:
    global advisor_service
    advisor_service = AdvisorService(vector_service.help_topic_vec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Dialogue Push System server...")

    vector_service.load()
    logger.info("Vector service loaded.")

    global advisor_service
    advisor_service = AdvisorService(vector_service.help_topic_vec)
    logger.info("Advisor service initialized.")

    await cache_service.connect()
    if cache_service.available:
        logger.info("Redis cache connected.")
    else:
        logger.warning("Redis not available, running without cache.")

    yield

    await cache_service.disconnect()
    logger.info("Server shutdown complete.")


app = FastAPI(
    title="Push Microservice",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.include_router(admin_router)
app.include_router(internal_router)
app.include_router(training_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info("%s %s → %d (%.3fs)", request.method, request.url.path, response.status_code, elapsed)
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
    last_user = extract_last_user_message(context) or ""
    if last_user:
        return last_user
    for line in context.strip().split("\n"):
        if line.startswith("用户:"):
            return line.removeprefix("用户:").strip()
    return ""


def _build_push_response(
    request: PushRequest,
    candidates,
    advisor_suggestion: bool,
    decision_config: DecisionConfigModel,
) -> PushResponse:
    user_message = _extract_user_message(request.context)
    push_events, state_summary = run_push_decision(
        session_id=request.session_id,
        user_message=user_message,
        candidates=candidates,
        positive_rate=request.positive_rate,
        advisor_suggestion=advisor_suggestion,
        risk_alert=request.risk_alert,
        conversation_state=request.conversation_state,
        interaction_mode=request.interaction_mode,
        decision_config=decision_config.model_dump(),
    )

    try:
        pushed_ids = [
            e.data.get("video_id")
            for e in push_events
            if e.type == "video_push" and e.data.get("video_id")
        ]
        if pushed_ids:
            collector.record_push(
                session_id=request.session_id,
                context=request.context,
                video_ids=pushed_ids,
                intent_query=getattr(request.push_intent, "intent_query", "") or "",
                interaction_mode=request.interaction_mode,
            )
    except Exception as exc:
        logger.debug("collector.record_push skipped: %s", exc)

    return PushResponse(
        candidates=candidates,
        advisor_suggestion=advisor_suggestion,
        generated_at=datetime.now(timezone.utc).isoformat(),
        decision_config=decision_config,
        push_events=push_events,
        state_summary=state_summary,
    )


@app.post("/api/v1/push", response_model=PushResponse)
async def push(request: PushRequest):
    if not request.context.strip():
        raise HTTPException(status_code=400, detail="context cannot be empty")

    decision_config = _decision_config_for_request(request.session_id)

    push_intent = request.push_intent
    skip_cache = (
        push_intent is not None
        and getattr(push_intent, "confidence", 0) >= 0.5
    )

    cached = None if skip_cache else await cache_service.get(request.context)
    if cached:
        logger.info("Cache hit for context hash")
        raw_candidates = cached.get("candidates", [])
        guarded = apply_relevance_guard(
            request.context,
            raw_candidates,
            vector_service.video_metadata,
            lexical_primary=vector_service._use_mock,
            push_intent=push_intent,
        )
        advisor_suggestion = bool(cached.get("advisor_suggestion"))
        return _build_push_response(
            request,
            guarded,
            advisor_suggestion or request.risk_alert,
            decision_config,
        )

    candidates, ctx_vec, _ = vector_service.process_request(
        request.context, push_intent=push_intent
    )
    candidates = apply_relevance_guard(
        request.context,
        candidates,
        vector_service.video_metadata,
        lexical_primary=vector_service._use_mock,
        push_intent=push_intent,
    )

    advisor_suggestion = advisor_service.check_advisor_suggestion(
        request.context, ctx_vec, request.positive_rate,
        risk_alert=request.risk_alert,
    )
    effective_advisor = advisor_suggestion or request.risk_alert

    cache_payload = {
        "candidates": [c.model_dump() if hasattr(c, "model_dump") else c for c in candidates],
        "advisor_suggestion": effective_advisor,
    }
    await cache_service.set(request.context, cache_payload)

    return _build_push_response(
        request,
        candidates,
        effective_advisor,
        decision_config,
    )


ENGAGEMENT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reports", "engagement_events.jsonl"
)


@app.post("/api/v1/engagement", response_model=EngagementResponse)
async def engagement(event: EngagementEvent):
    consecutive_negative = 0
    if event.session_id:
        state = get_session_state(event.session_id)
        consecutive_negative = state.get("consecutive_negative") or 0

    feedback_type, reason = classify_engagement(
        event=event.event,
        watch_duration_sec=event.watch_duration_sec,
        video_duration_sec=event.video_duration_sec,
        consecutive_negative=consecutive_negative,
        was_clicked=event.event in ("watch_end", "dismiss", "share", "favorite"),
    )

    if event.session_id:
        engine = DecisionEngine(state=get_session_state(event.session_id))
        engine.record_video_engagement(
            video_id=event.video_id,
            event=event.event,
            watch_duration_sec=event.watch_duration_sec,
            video_duration_sec=event.video_duration_sec,
        )

    log_entry = {
        "video_id": event.video_id,
        "event": event.event,
        "watch_duration_sec": event.watch_duration_sec,
        "video_duration_sec": event.video_duration_sec,
        "session_id": event.session_id,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
        "classified_feedback": feedback_type,
        "reason": reason,
    }
    logger.info("engagement_event %s", json.dumps(log_entry, ensure_ascii=False))

    try:
        collector.record_engagement(
            session_id=event.session_id,
            video_id=event.video_id,
            event=event.event,
            watch_duration_sec=event.watch_duration_sec,
            video_duration_sec=event.video_duration_sec,
            classified=feedback_type,
        )
    except Exception as exc:
        logger.debug("collector.record_engagement skipped: %s", exc)

    os.makedirs(os.path.dirname(ENGAGEMENT_LOG_PATH), exist_ok=True)
    with open(ENGAGEMENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return EngagementResponse(
        accepted=True,
        classified_feedback=feedback_type,
        reason=reason,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=4,
        log_level="info",
    )
