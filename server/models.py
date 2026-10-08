from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


class PushIntentModel(BaseModel):
    primary_topic: Optional[str] = None
    intent_query: str = ""
    exclude_topics: List[str] = Field(default_factory=list)
    is_meta_request: bool = False
    content_need: Literal["video", "none"] = "none"
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PushEventModel(BaseModel):
    type: Literal["video_push", "advisor_push"]
    data: Dict[str, Any]


class CandidateVideo(BaseModel):
    video_id: str
    type: str = "video"
    similarity: float
    prob_base: float
    title: str
    cover_url: str
    duration_sec: float = 60
    vod_video_id: Optional[str] = None


class AWeightsModel(BaseModel):
    I: float = 0.18
    L: float = 0.22
    F: float = 0.32
    S: float = 0.10
    U: float = 0.18


class DecisionConfigModel(BaseModel):
    version: str = "1"
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
    urge_active_video_decay: float = 0.7
    cooldown_ms: int = 90000
    cooldown_ms_max: int = 90000
    state_modulation: Dict[str, float] = Field(
        default_factory=lambda: {
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
    auto_video_target_rate: Optional[float] = None
    auto_video_quota_min_similarity: float = 0.40
    auto_video_quota_cooldown_msgs: int = 3
    auto_advisor_target_rate: Optional[float] = None
    auto_advisor_effective_rate: Optional[float] = None
    auto_advisor_quota_cooldown_msgs: int = 3
    auto_advisor_quota_min_session_msgs: int = 2
    advisor_push_cooldown_ms: int = 300000
    advisor_session_push_cap: int = 3
    mentor_a_boost: float = 0.05
    mentor_cooldown_ms: int = 55000
    mentor_urge_delta_bonus: float = 0.02


class EngagementEvent(BaseModel):
    video_id: str
    event: Literal[
        "exposed", "click", "watch_end", "dismiss", "no_click",
        "share", "favorite", "not_interested",
    ]
    watch_duration_sec: float = 0
    video_duration_sec: float = 60
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class EngagementResponse(BaseModel):
    accepted: bool
    classified_feedback: Optional[Literal["positive", "negative", "strong_negative"]] = None
    reason: Optional[str] = None


class PushResponse(BaseModel):
    candidates: List[CandidateVideo]
    advisor_suggestion: bool
    generated_at: str
    decision_config: DecisionConfigModel
    push_events: List[PushEventModel] = Field(default_factory=list)
    state_summary: Optional[Dict[str, Any]] = None


class PushRequest(BaseModel):
    context: str = Field(..., description="最近对话文本")
    load_factor: float = Field(0.8, ge=0.0, le=1.0, description="负荷因子")
    positive_rate: float = Field(0.6, ge=0.0, le=1.0, description="正反馈率")
    intimacy: float = Field(0.4, ge=0.0, le=1.0, description="亲密度")
    urge: float = Field(0.7, ge=0.0, le=1.0, description="推送冲动值U")
    trigger_mode: Literal["auto", "active"] = "auto"
    resource_type: Literal["video", "advisor", "all"] = "video"
    experiment_id: Optional[str] = None
    risk_alert: bool = Field(False, description="P0: 上游风险监控信号")
    conversation_state: Literal["INIT", "GATHERING", "DEEPENING", "TAILORING"] = Field("INIT", description="P1: 对话阶段")
    session_id: Optional[str] = Field(None, description="会话 ID，用于服务端推送状态")
    interaction_mode: Literal["auto", "mentor", "proxy_chat"] = Field(
        "auto", description="交互模式（auto/mentor/proxy_chat）"
    )
    push_intent: Optional[PushIntentModel] = Field(
        None, description="上游预分析意图（可选）"
    )
