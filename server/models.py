"""Generic request/response schemas for the push microservice."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class IntentModel(BaseModel):
    primary_topic: Optional[str] = Field(None, description="Boost candidates whose tags or text contain this topic")
    query: str = ""
    exclude_topics: List[str] = Field(default_factory=list, description="Drop candidates whose tags or text contain any of these")
    is_meta: bool = False
    content_need: Literal["item", "handoff", "none"] = Field(
        "none",
        description="none is no preference. item skips handoff unless a force-handoff signal is set. handoff prefers a handoff.",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PushEventModel(BaseModel):
    type: Literal["item_push", "handoff_push", "event"]
    data: Dict[str, Any]


class CandidateItem(BaseModel):
    item_id: str
    item_type: str = "item"
    similarity: float
    prob_base: float
    title: str
    media_url: str = ""
    duration_sec: float = 60
    item_ref: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class AWeightsModel(BaseModel):
    I: float = 0.18
    L: float = 0.22
    F: float = 0.32
    S: float = 0.10
    U: float = 0.18


class PolicyConfigModel(BaseModel):
    version: str = "2"
    target_push_rate_per_10: float = 2.5
    min_push_rate_per_10: float = 0.6
    max_push_rate_per_10: float = 2.5
    tau_init: float = 0.55
    tau_min: float = 0.3
    tau_max: float = 2.0
    tau_positive_mult: float = 0.92
    tau_negative_mult: float = 1.08
    a_threshold_0: Optional[float] = None
    a_threshold_1: float = 0.75
    a_threshold_2: float = 0.88
    a_weights: AWeightsModel = Field(default_factory=AWeightsModel)
    f_default_new_user: float = 0.6
    urge_initial: float = 0.5
    urge_delta: float = 0.07
    urge_active_item_decay: float = 0.7
    cooldown_ms: int = 90000
    cooldown_ms_max: int = 90000
    state_modulation: Dict[str, float] = Field(
        default_factory=lambda: {
            "start": -0.04,
            "explore": -0.02,
            "focus": 0.0,
            "close": 0.06,
            "INIT": -0.04,
            "GATHERING": -0.02,
            "DEEPENING": 0.0,
            "TAILORING": 0.06,
        }
    )
    recent_push_window_ms: int = 300000
    recent_push_load_divisor: int = 3
    session_msg_cap: int = 30
    auto_push_min_session_msgs: int = 2
    auto_item_target_rate: Optional[float] = None
    auto_item_quota_min_similarity: float = 0.40
    auto_item_quota_cooldown_msgs: int = 3
    auto_handoff_target_rate: Optional[float] = None
    auto_handoff_effective_rate: Optional[float] = None
    auto_handoff_quota_cooldown_msgs: int = 3
    auto_handoff_quota_min_session_msgs: int = 2
    handoff_push_cooldown_ms: int = 300000
    handoff_session_push_cap: int = 3
    provider: str = "builtin:default"
    positive_dwell_ratio: float = 0.50
    short_dwell_ratio: float = 0.10


class FeedbackEvent(BaseModel):
    item_id: str
    event: Literal[
        "exposed", "click", "watch_end", "dwell_end", "dismiss", "no_click",
        "share", "favorite", "not_interested",
    ]
    watch_duration_sec: float = 0
    item_duration_sec: float = 60
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class FeedbackResponse(BaseModel):
    accepted: bool
    classified_feedback: Optional[Literal["positive", "negative", "strong_negative"]] = None
    reason: Optional[str] = None


class PushResponse(BaseModel):
    candidates: List[CandidateItem]
    handoff_suggested: bool
    generated_at: str
    policy_config: PolicyConfigModel
    push_events: List[PushEventModel] = Field(default_factory=list)
    state_summary: Optional[Dict[str, Any]] = None


class PushRequest(BaseModel):
    context: str = Field(..., description="Recent conversation text")
    positive_rate: float = Field(0.6, ge=0.0, le=1.0, description="Positive feedback rate")
    trigger_mode: Literal["auto", "active"] = Field(
        "auto",
        description="active forces an item push when candidates exist; auto uses keyword detection",
    )
    item_type: Optional[str] = Field(None, description="Keep only candidates with this item_type")
    experiment_id: Optional[str] = Field(
        None,
        description="Force a named policy profile, or pass the active experiment name to keep session split",
    )
    signals: Dict[str, Any] = Field(default_factory=dict, description="Extensible signals (e.g. risk_alert)")
    state: str = Field("INIT", description="Generic conversation state label")
    session_id: Optional[str] = Field(None, description="Session id for server-side state")
    intent: Optional[IntentModel] = Field(None, description="Upstream pre-analyzed intent (optional)")
