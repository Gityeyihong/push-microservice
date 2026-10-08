"""Explicit user-request keywords for active push triggers (sync with Android PushRequestKeywords)."""

import re

VIDEO_REQUEST_KEYWORDS = [
    "推荐视频",
    "来个视频",
    "看个技巧",
    "看看视频",
    "有视频吗",
    "视频教程",
    "教程推荐",
    "视频课程",
    "课程推荐",
    "有什么视频",
    "推荐一下视频",
    "想看视频",
    "看教程",
    "视频推荐",
    "教学视频",
    "有教学",
    "教学课",
    "有视频",
    "要课程",
    "要视频",
    "要教程",
    "推荐课程",
    "推荐个视频",
    "推荐个课",
    "有没有课",
    "有没有课程",
    "有没有视频",
    "有课吗",
    "有课程吗",
    "推个视频",
    "推个课",
    "来个课",
    "看看课程",
    "有什么课",
    "有什么课程",
]

VIDEO_REQUEST_TOPICS = ("视频", "课程", "教程", "教学课", "干货")
VIDEO_REQUEST_VERBS = ("要", "推荐", "看看", "有没有", "有什么", "推个", "来个", "想学", "教我", "有吗")
# Whole-message triggers — user explicitly asks for video content.
VIDEO_SHORT_TRIGGERS = ("视频", "课程", "教程", "干货", "推视频")

ADVISOR_REQUEST_KEYWORDS = [
    "叫导师",
    "真人老师",
    "找老师",
    "找导师",
    "有导师",
    "有导师吗",
    "有老师吗",
    "有没有导师",
    "有没有老师",
    "导师吗",
    "老师吗",
    "真人导师",
    "情感导师",
    "真人服务",
    "有真人",
    "真人指导",
    "真人情感",
    "要导师",
    "要老师",
]

_FIND_MENTOR_PHRASES = (
    "找老师",
    "找导师",
    "要找老师",
    "要找导师",
    "想找老师",
    "想找导师",
)

ADVISOR_REQUEST_TOPICS = ("导师", "真人", "老师", "情感导师")
# Whole-message triggers — user explicitly asks for a human mentor (not e.g. "女朋友是老师").
ADVISOR_SHORT_TRIGGERS = ("老师", "导师", "真人", "情感导师", "真人导师")
ADVISOR_REQUEST_VERBS = (
    "有吗",
    "有没有",
    "找",
    "要",
    "叫",
    "连线",
    "请",
    "推荐",
    "需要",
    "来",
)

# Platform mentor/coach service — not romantic pursuit of someone who happens to be a teacher.
_MENTOR_SERVICE_MARKERS = (
    "真人导师",
    "情感导师",
    "真人老师",
    "真人指导",
    "真人服务",
    "咨询情感导师",
)

_ROMANTIC_PURSUIT_MARKERS = (
    "女朋友",
    "男朋友",
    "谈恋爱",
    "在一起",
    "暗恋",
    "心仪",
    "表白",
    "当老婆",
    "当老公",
    "追我",
    "追她",
    "追他",
)


def _normalize_advisor_message(user_message: str) -> str:
    """Common input confusions (e.g. 道师/道士 vs 导师)."""
    msg = (user_message or "").strip()
    if not msg:
        return msg
    return msg.replace("道师", "导师").replace("道士", "导师")


def _has_you_ma_question(msg: str) -> bool:
    """Matches 有X吗 / 有没有X (X may sit between 有 and 吗)."""
    return bool(re.search(r"有(?:没有)?[^吗]{0,12}吗", msg))


def _is_garbled_input(msg: str) -> bool:
    """Keyboard noise / pasted garbage — not an intentional mentor request."""
    if re.search(r"\*{2,}", msg):
        return True
    if "//" in msg or "\\\\" in msg:
        return True
    if re.search(r"[a-zA-Z]{5,}", msg):
        return True
    return False


def _matches_explicit_find_mentor(msg: str) -> bool:
    bare = msg.strip("？?！!。., ")
    if bare in _FIND_MENTOR_PHRASES:
        return True
    return bool(
        re.search(
            r"(?:怎么|如何|想|要|请)?找(?:个)?(?:真人|情感)?(?:导师|老师)(?:吗|么|呢|吧)?",
            msg,
        )
    )


def _is_romantic_teacher_pursuit(msg: str) -> bool:
    """User pursues a teacher as love interest — not requesting platform mentor."""
    if "老师" not in msg and "导师" not in msg:
        return False
    if any(marker in msg for marker in _MENTOR_SERVICE_MARKERS):
        return False
    if re.search(r"找(?:个)?老师(?:去)?当", msg):
        return True
    if "追老师" in msg or re.search(r"追(?:上)?(?:这位|这个|那个)?老师", msg):
        return True
    if any(marker in msg for marker in _ROMANTIC_PURSUIT_MARKERS):
        return True
    if "喜欢" in msg and ("老师" in msg or "导师" in msg):
        return True
    return False


def is_video_request(user_message: str) -> bool:
    """True when the user explicitly asks for video/course content."""
    msg = (user_message or "").strip()
    if not msg:
        return False
    bare = msg.strip("？?！!。., ")
    if bare in VIDEO_SHORT_TRIGGERS:
        return True
    if any(kw in msg for kw in VIDEO_REQUEST_KEYWORDS):
        return True
    has_topic = any(t in msg for t in VIDEO_REQUEST_TOPICS)
    has_verb = any(v in msg for v in VIDEO_REQUEST_VERBS)
    return has_topic and has_verb


def is_advisor_request(user_message: str) -> bool:
    msg = _normalize_advisor_message(user_message)
    if not msg:
        return False
    if _is_garbled_input(msg):
        return False
    if _is_romantic_teacher_pursuit(msg):
        return False
    bare = msg.strip("？?！!。.")
    if bare in ADVISOR_SHORT_TRIGGERS:
        return True
    if _matches_explicit_find_mentor(msg):
        return True
    if any(kw in msg for kw in ADVISOR_REQUEST_KEYWORDS):
        return True
    has_topic = any(t in msg for t in ADVISOR_REQUEST_TOPICS)
    if has_topic and _has_you_ma_question(msg):
        return True
    has_verb = any(v in msg for v in ADVISOR_REQUEST_VERBS)
    return has_topic and has_verb
