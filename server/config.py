import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_root_env() -> None:
    """Load project-root .env without overriding existing process env."""
    root_env = os.path.normpath(os.path.join(DATA_DIR, "..", ".env"))
    if not os.path.isfile(root_env):
        return
    with open(root_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


_load_root_env()

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384

NUM_VIDEOS = 100
NUM_TOPICS = 12
FAISS_TOPK = 30
MMR_LAMBDA = 0.7
MMR_RESULT_N = 10

CONV_KERNEL = [0.2, 0.6, 0.2]

TOPIC_SOFTMAX_TEMP = 0.25
SERVICE_SOFTMAX_TEMP = 0.35

SIMILARITY_TEXT_WEIGHT = 0.75
SIMILARITY_TOPIC_WEIGHT = 0.25

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL = 60

HMAC_SECRET_KEY = os.getenv("HMAC_SECRET_KEY", "change-me-in-production")

HELP_SIMILARITY_THRESHOLD = 0.6
POSITIVE_RATE_THRESHOLD = 0.5

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
SERVER_WORKERS = int(os.getenv("SERVER_WORKERS", 4))

PUSH_USE_MOCK_VECTORS = os.getenv("PUSH_USE_MOCK_VECTORS", "0").strip().lower() in ("1", "true", "yes")
VECTOR_MANIFEST_PATH = os.path.join(DATA_DIR, "..", "data", "vector_manifest.json")

_push_videos_file = os.getenv("PUSH_VIDEOS", "catalog.json").strip() or "catalog.json"
DATA_ROOT = os.path.join(DATA_DIR, "..", "data")
EMBEDDINGS_PATH = os.path.join(DATA_ROOT, "video_embeddings.npy")
TOPIC_VECS_PATH = os.path.join(DATA_ROOT, "topic_vecs.npy")
TOPIC_VIDEO_VECS_PATH = os.path.join(DATA_ROOT, "video_topic_vecs.npy")
FAISS_INDEX_PATH = os.path.join(DATA_ROOT, "faiss_index.bin")
VIDEO_META_PATH = os.path.join(DATA_ROOT, _push_videos_file)
