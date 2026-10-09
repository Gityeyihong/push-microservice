"""生成条目向量、话题向量、Faiss 索引等离线数据。

用法:
    python -m data.generate_data          # 使用真实 sentence-transformers 模型
    python -m data.generate_data --mock   # 使用模拟随机向量（无需联网）
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.topics import TOPIC_WORDS, HELP_TOPIC_TEXT
from server.config import FAISS_EF_SEARCH
from server.embedding_model import load_sentence_transformer
from server.vector_service import _mock_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATALOG = "catalog.json"
CATALOG_PATH = os.path.join(BASE_DIR, DEFAULT_CATALOG)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "item_embeddings.npy")
TOPIC_VECS_PATH = os.path.join(BASE_DIR, "topic_vecs.npy")
ITEM_TOPIC_VECS_PATH = os.path.join(BASE_DIR, "item_topic_vecs.npy")
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
    """Text-stable random encoder. The same string always maps to the same vector."""

    def __init__(self, dim: int = VECTOR_DIM, seed: int = MOCK_SEED):
        self.dim = dim
        self.seed = seed

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.stack([_mock_vector(text, self.dim, self.seed) for text in texts])
        if normalize_embeddings:
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs.astype(np.float32)


def build_item_text(item: dict) -> str:
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return (
        f"title: {item.get('title', '')}. "
        f"description: {item.get('description', '')}. "
        f"summary: {item.get('summary', '')}. "
        f"tags: {' '.join(tags)}"
    )


def build_video_text(video: dict) -> str:
    return build_item_text(video)


def generate_topic_vectors(encoder) -> np.ndarray:
    logger.info("Encoding %d topics...", len(TOPIC_WORDS))
    encodings = encoder.encode(TOPIC_WORDS, normalize_embeddings=True)
    return encodings.astype(np.float32)


def generate_item_embeddings(encoder, items: list) -> np.ndarray:
    logger.info("Encoding %d items...", len(items))
    texts = [build_item_text(v) for v in items]
    encodings = encoder.encode(texts, normalize_embeddings=True)
    return encodings.astype(np.float32)


def generate_video_embeddings(encoder, videos: list) -> np.ndarray:
    return generate_item_embeddings(encoder, videos)


def generate_item_topic_vecs(items: list, num_topics: int) -> np.ndarray:
    logger.info("Generating item-topic one-hot vectors...")
    topic_index = {t: i for i, t in enumerate(TOPIC_WORDS)}

    topic_vecs = np.zeros((len(items), num_topics), dtype=np.float32)

    for i, video in enumerate(items):
        for tag in video.get("tags", []):
            if tag in topic_index:
                topic_vecs[i, topic_index[tag]] = 1.0

        norm = np.linalg.norm(topic_vecs[i])
        if norm > 0:
            topic_vecs[i] /= norm

    return topic_vecs


def generate_video_topic_vecs(videos: list, num_topics: int) -> np.ndarray:
    return generate_item_topic_vecs(videos, num_topics)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    logger.info("Building Faiss HNSW index (dim=%d)...", VECTOR_DIM)

    index = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(int(FAISS_EF_SEARCH), 30)
    index.add(embeddings)

    return index


def main():
    use_mock = "--mock" in sys.argv
    strict_real = "--strict" in sys.argv or os.getenv("PUSH_USE_MOCK_VECTORS", "0").strip().lower() not in (
        "1", "true", "yes",
    )

    catalog_name = os.getenv("PUSH_CATALOG", "").strip() or os.getenv("PUSH_VIDEOS", "").strip() or DEFAULT_CATALOG
    for a in sys.argv:
        if a.startswith("--catalog=") or a.startswith("--videos="):
            catalog_name = a.split("=", 1)[1]
    video_meta = os.path.join(BASE_DIR, catalog_name)

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
    logger.info("Loaded %d catalog entries", len(videos))

    topic_vecs = generate_topic_vectors(encoder)
    np.save(TOPIC_VECS_PATH, topic_vecs)
    logger.info("Saved topic vectors → %s", TOPIC_VECS_PATH)

    video_embs = generate_item_embeddings(encoder, videos)
    np.save(EMBEDDINGS_PATH, video_embs)
    logger.info("Saved item embeddings → %s", EMBEDDINGS_PATH)

    video_topic_vecs = generate_item_topic_vecs(videos, len(TOPIC_WORDS))
    np.save(ITEM_TOPIC_VECS_PATH, video_topic_vecs)
    logger.info("Saved item-topic vectors → %s", ITEM_TOPIC_VECS_PATH)

    index = build_faiss_index(video_embs)
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info("Saved Faiss index → %s (total=%d)", FAISS_INDEX_PATH, index.ntotal)

    help_vec = encoder.encode([HELP_TOPIC_TEXT], normalize_embeddings=True)[0]
    help_path = os.path.join(BASE_DIR, "help_topic_vec.npy")
    np.save(help_path, help_vec.astype(np.float32))
    logger.info("Saved help topic vector → %s", help_path)

    catalog_hash = hashlib.sha256()
    with open(video_meta, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            catalog_hash.update(chunk)
    manifest = {
        "mode": mode,
        "model": MODEL_NAME if mode == "real" else "mock",
        "item_count": len(videos),
        "catalog_file": os.path.basename(video_meta),
        "catalog_sha256": catalog_hash.hexdigest(),
        "vector_dim": VECTOR_DIM,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Saved vector manifest → %s", MANIFEST_PATH)

    logger.info("=== Data generation complete! ===")
    logger.info(
        "Generated: %d topics, %d items, %d-dim vectors (mode=%s)",
        len(TOPIC_WORDS),
        len(videos),
        VECTOR_DIM,
        mode,
    )


if __name__ == "__main__":
    main()
