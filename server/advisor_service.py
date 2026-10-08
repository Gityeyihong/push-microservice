import numpy as np

from .config import HELP_SIMILARITY_THRESHOLD, POSITIVE_RATE_THRESHOLD
from data.topics import ADVISOR_KEYWORDS_URGENT


class AdvisorService:
    def __init__(self, help_topic_vec: np.ndarray):
        self.help_topic_vec = help_topic_vec

    def check_advisor_suggestion(
        self, context: str, ctx_vec: np.ndarray, positive_rate: float,
        risk_alert: bool = False,
    ) -> bool:
        if risk_alert:
            return True

        if any(kw in context for kw in ADVISOR_KEYWORDS_URGENT):
            return True

        ctx_vec_norm = ctx_vec / (np.linalg.norm(ctx_vec) + 1e-8)
        help_vec_norm = self.help_topic_vec / (np.linalg.norm(self.help_topic_vec) + 1e-8)
        similarity = float(np.dot(ctx_vec_norm, help_vec_norm))

        if similarity > HELP_SIMILARITY_THRESHOLD and positive_rate < POSITIVE_RATE_THRESHOLD:
            return True

        return False
