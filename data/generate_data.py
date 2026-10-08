"""生成视频向量、话题向量、Faiss索引等离线数据。

用法:
    python -m data.generate_data          # 使用真实 sentence-transformers 模型
    python -m data.generate_data --mock   # 使用模拟随机向量（无需联网）
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.topics import TOPIC_WORDS, HELP_TOPIC_TEXT
from server.embedding_model import load_sentence_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_META = "catalog.json"
VIDEO_META_PATH = os.path.join(BASE_DIR, DEFAULT_VIDEO_META)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "video_embeddings.npy")
TOPIC_VECS_PATH = os.path.join(BASE_DIR, "topic_vecs.npy")
TOPIC_VIDEO_VECS_PATH = os.path.join(BASE_DIR, "video_topic_vecs.npy")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
MANIFEST_PATH = os.path.join(BASE_DIR, "vector_manifest.json")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384
MOCK_SEED = 42


def load_real_model():
    model, path, source = load_sentence_transformer()
    logger.info("Loaded embedding model (%s): %s", source, path)
    return model


class MockEncoder:
    """模拟编码器，使用确定性随机向量替代真实模型（用于无网络环境测试）。"""

    def __init__(self, dim: int = VECTOR_DIM, seed: int = MOCK_SEED):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        vecs = self.rng.randn(n, self.dim).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs = vecs / norms
        return vecs


def build_video_text(video: dict) -> str:
    return (
        f"标题：{video['title']}。 "
        f"文案：{video['description']}。 "
        f"简述：{video['summary']}。 "
        f"标签：{' '.join(video['tags'])}"
    )


def generate_topic_vectors(encoder) -> np.ndarray:
    logger.info("Encoding %d topics...", len(TOPIC_WORDS))
    encodings = encoder.encode(TOPIC_WORDS, normalize_embeddings=True)
    return encodings.astype(np.float32)


def generate_video_embeddings(encoder, videos: list) -> np.ndarray:
    logger.info("Encoding %d videos...", len(videos))
    texts = [build_video_text(v) for v in videos]
    encodings = encoder.encode(texts, normalize_embeddings=True)
    return encodings.astype(np.float32)


def generate_video_topic_vecs(videos: list, num_topics: int) -> np.ndarray:
    logger.info("Generating video-topic one-hot vectors...")
    topic_index = {t: i for i, t in enumerate(TOPIC_WORDS)}

    topic_vecs = np.zeros((len(videos), num_topics), dtype=np.float32)

    for i, video in enumerate(videos):
        for tag in video.get("tags", []):
            if tag in topic_index:
                topic_vecs[i, topic_index[tag]] = 1.0

        norm = np.linalg.norm(topic_vecs[i])
        if norm > 0:
            topic_vecs[i] /= norm

    return topic_vecs


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    logger.info("Building Faiss HNSW index (dim=%d)...", VECTOR_DIM)

    index = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 16
    index.add(embeddings)

    return index


def main():
    use_mock = "--mock" in sys.argv
    strict_real = "--strict" in sys.argv or os.getenv("PUSH_USE_MOCK_VECTORS", "0").strip().lower() not in (
        "1", "true", "yes",
    )

    video_meta = VIDEO_META_PATH
    push_videos = os.getenv("PUSH_VIDEOS", "").strip()
    if push_videos:
        video_meta = os.path.join(BASE_DIR, push_videos)
    for a in sys.argv:
        if a.startswith("--videos="):
            video_meta = os.path.join(BASE_DIR, a.split("=", 1)[1])

    logger.info("=== Generate Push Microservice Data ===")
    if use_mock:
        logger.info("Mode: MOCK (random vectors, no network required)")
        encoder = MockEncoder()
        mode = "mock"
    else:
        logger.info("Mode: REAL (sentence-transformers: %s)", MODEL_NAME)
        try:
            encoder = load_real_model()
            mode = "real"
        except Exception as e:
            if strict_real:
                logger.error("Failed to load model (prod mode): %s", e)
                sys.exit(1)
            logger.error("Failed to load model: %s", e)
            logger.info("Falling back to mock. Set PUSH_USE_MOCK_VECTORS=1 to allow mock explicitly.")
            encoder = MockEncoder()
            mode = "mock"

    with open(video_meta, "r", encoding="utf-8") as f:
        videos = json.load(f)
    logger.info("Loaded %d video metadata entries", len(videos))

    topic_vecs = generate_topic_vectors(encoder)
    np.save(TOPIC_VECS_PATH, topic_vecs)
    logger.info("Saved topic vectors → %s", TOPIC_VECS_PATH)

    video_embs = generate_video_embeddings(encoder, videos)
    np.save(EMBEDDINGS_PATH, video_embs)
    logger.info("Saved video embeddings → %s", EMBEDDINGS_PATH)

    video_topic_vecs = generate_video_topic_vecs(videos, len(TOPIC_WORDS))
    np.save(TOPIC_VIDEO_VECS_PATH, video_topic_vecs)
    logger.info("Saved video-topic vectors → %s", TOPIC_VIDEO_VECS_PATH)

    index = build_faiss_index(video_embs)
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info("Saved Faiss index → %s (total=%d)", FAISS_INDEX_PATH, index.ntotal)

    help_vec = encoder.encode([HELP_TOPIC_TEXT], normalize_embeddings=True)[0]
    help_path = os.path.join(BASE_DIR, "help_topic_vec.npy")
    np.save(help_path, help_vec.astype(np.float32))
    logger.info("Saved help topic vector → %s", help_path)

    manifest = {
        "mode": mode,
        "model": MODEL_NAME if mode == "real" else "mock",
        "video_count": len(videos),
        "video_file": os.path.basename(video_meta),
        "vector_dim": VECTOR_DIM,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Saved vector manifest → %s", MANIFEST_PATH)

    logger.info("=== Data generation complete! ===")
    logger.info(
        "Generated: %d topics, %d videos, %d-dim vectors (mode=%s)",
        len(TOPIC_WORDS),
        len(videos),
        VECTOR_DIM,
        mode,
    )


if __name__ == "__main__":
    main()
