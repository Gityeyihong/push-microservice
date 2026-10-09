import asyncio
import logging
import time
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import SERVER_HOST, SERVER_PORT
from .models import PushRequest, PushResponse, FeedbackEvent, FeedbackResponse, PolicyConfigModel
from .vector_service import VectorService
from .cache_service import CacheService
from .ranking import extract_last_user_message, load_ranker
from .push_decision_service import run_push_decision
from .push_session_store import get_session_state
from .admin_routes import router as admin_router
from .internal_routes import router as internal_router
from .training_routes import router as training_router
from handoff import load_handoff
from policy import classify_feedback, load_policy, load_policy_config, reload_policy_config
from policy.config import PolicyConfig
from policy.keywords import signal_forces_handoff
from policy.state import FileStateBackend
from selftrain.collector import TrainingCollector

from .metrics import hit
from .ratelimit import allow, client_key, max_body_bytes
from .security import PLACEHOLDER_SECRETS, check_bearer, enforce_production_config

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


def _policy_config_for_request(session_id: str | None, experiment_id: str | None = None) -> PolicyConfigModel:
    _current_policy_config()
    from .ab_resolver import resolve_config_for_session

    try:
        resolved = resolve_config_for_session(session_id, experiment_id=experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PolicyConfigModel(**resolved)


def _decision_signals(request: PushRequest) -> dict:
    signals = dict(request.signals or {})
    signals["trigger_mode"] = request.trigger_mode
    if request.intent is not None:
        signals["content_need"] = request.intent.content_need
    return signals


def reload_handoff_from_vectors() -> None:
    global handoff_service
    handoff_service = load_handoff(help_topic_vec=vector_service.help_topic_vec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Push Microservice...")
    enforce_production_config()
    from .config import HMAC_SECRET_KEY

    if HMAC_SECRET_KEY.strip() in PLACEHOLDER_SECRETS:
        logger.warning("HMAC_SECRET_KEY is a placeholder; set a unique secret before production.")
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


_is_production = os.getenv("PUSH_ENV", "development").strip().lower() == "production"
app = FastAPI(
    title="Push Microservice",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.include_router(admin_router)
app.include_router(internal_router)
app.include_router(training_router)


@app.middleware("http")
async def limit_public_api(request: Request, call_next):
    if request.url.path.startswith("/api/v1"):
        body = await request.body()
        if len(body) > max_body_bytes():
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        key = client_key(
            request.headers.get("x-forwarded-for"),
            request.client.host if request.client else None,
        )
        if not await allow(key, cache_service):
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    return await call_next(request)


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


@app.get("/ready")
async def ready():
    if vector_service.index is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "redis": cache_service.available, "index": False},
        )
    try:
        vector_service.maybe_reload()
    except Exception as exc:
        logger.warning("vector reload failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "redis": cache_service.available, "index": False},
        )
    if not cache_service.available:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "redis": False, "index": True},
        )
    return {"status": "ok", "redis": True, "index": True}


def _cache_material(request: PushRequest) -> str:
    intent = request.intent.model_dump() if request.intent is not None else None
    return json.dumps(
        {
            "context": request.context,
            "item_type": request.item_type,
            "positive_rate": request.positive_rate,
            "signals": request.signals,
            "state": request.state,
            "intent": intent,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _extract_user_message(context: str) -> str:
    return extract_last_user_message(context) or ""


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("PUSH_API_TOKEN", "").strip()
    if not token:
        return
    check_bearer(authorization, token)


def _rank_candidates(context, candidates, metadata, *, lexical_primary, intent, item_type=None):
    ranker = load_ranker()
    return ranker.rank(
        context,
        candidates,
        metadata,
        lexical_primary=lexical_primary,
        intent=intent,
        item_type=item_type,
    )


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
        signals=_decision_signals(request),
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
async def push(request: PushRequest, _: None = Depends(require_api_token)):
    if not request.context.strip():
        raise HTTPException(status_code=400, detail="context cannot be empty")

    policy_config = _policy_config_for_request(request.session_id, request.experiment_id)
    intent = request.intent
    try:
        vector_service.maybe_reload()
    except Exception as exc:
        logger.warning("vector reload before push failed: %s", exc)
        raise HTTPException(status_code=503, detail="vector index is not ready") from exc
    cache_key = _cache_material(request)

    cached = await cache_service.get(cache_key)
    if cached:
        await hit("cache_hit", cache_service)
        logger.info("Cache hit for context hash")
        raw_candidates = cached.get("candidates", [])
        guarded = _rank_candidates(
            request.context,
            raw_candidates,
            vector_service.item_metadata,
            lexical_primary=vector_service._use_mock,
            intent=intent,
            item_type=request.item_type,
        )
        handoff_suggested = bool(cached.get("handoff_suggested")) or signal_forces_handoff(
            request.signals
        )
        await hit("push", cache_service)
        if handoff_suggested:
            await hit("handoff", cache_service)
        return await _build_push_response(request, guarded, handoff_suggested, policy_config)

    await hit("cache_miss", cache_service)
    candidates, ctx_vec, _ = vector_service.process_request(request.context, intent=intent)
    candidates = _rank_candidates(
        request.context,
        candidates,
        vector_service.item_metadata,
        lexical_primary=vector_service._use_mock,
        intent=intent,
        item_type=request.item_type,
    )

    effective_handoff = signal_forces_handoff(request.signals) or handoff_service.should_handoff(
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
    await cache_service.set(cache_key, cache_payload)

    await hit("push", cache_service)
    if effective_handoff:
        await hit("handoff", cache_service)
    return await _build_push_response(request, candidates, effective_handoff, policy_config)


def _append_feedback_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backend = FileStateBackend(path)
    with backend.exclusive():
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def feedback_log_path(moment: datetime | None = None) -> str:
    day = (moment or datetime.now(timezone.utc)).strftime("%Y%m%d")
    reports = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
    return os.path.join(reports, f"feedback-{day}.jsonl")


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def feedback(event: FeedbackEvent, _: None = Depends(require_api_token)):
    policy_model = _policy_config_for_request(event.session_id)
    if event.session_id:
        engine = load_policy(
            state=get_session_state(event.session_id),
            config=PolicyConfig.from_dict(policy_model.model_dump()),
        )
        recorded = engine.record_engagement(
            item_id=event.item_id,
            event=event.event,
            watch_duration_sec=event.watch_duration_sec,
            item_duration_sec=event.item_duration_sec,
        )
        feedback_type = recorded.get("feedback_type")
        reason = recorded.get("reason")
    else:
        feedback_type, reason = classify_feedback(
            event=event.event,
            watch_duration_sec=event.watch_duration_sec,
            item_duration_sec=event.item_duration_sec,
            consecutive_negative=0,
            was_clicked=False,
            positive_ratio=policy_model.positive_dwell_ratio,
            short_ratio=policy_model.short_dwell_ratio,
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

    log_path = feedback_log_path()
    line = json.dumps(log_entry, ensure_ascii=False) + "\n"
    await asyncio.to_thread(_append_feedback_line, log_path, line)

    metric = f"feedback_{feedback_type}" if feedback_type in ("positive", "negative", "strong_negative") else "feedback_other"
    await hit(metric, cache_service)

    return FeedbackResponse(accepted=True, classified_feedback=feedback_type, reason=reason)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=SERVER_HOST, port=SERVER_PORT, workers=4, log_level="info")
