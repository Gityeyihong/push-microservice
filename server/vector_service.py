import json
import logging
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss

from .config import (
    MODEL_NAME, VECTOR_DIM,
    FAISS_TOPK, MMR_LAMBDA, MMR_RESULT_N,
    CONV_KERNEL,
    TOPIC_SOFTMAX_TEMP, SERVICE_SOFTMAX_TEMP,
    SIMILARITY_TEXT_WEIGHT, SIMILARITY_TOPIC_WEIGHT,
    EMBEDDINGS_PATH, TOPIC_VECS_PATH, TOPIC_VIDEO_VECS_PATH,
    FAISS_INDEX_PATH, VIDEO_META_PATH,
)
from .config import PUSH_USE_MOCK_VECTORS, VECTOR_MANIFEST_PATH
from .embedding_model import load_sentence_transformer, resolve_model_path, model_source_label
from .models import CandidateItem
from .ranking import (
    extract_last_user_message,
    extract_user_messages,
    build_intent_query,
    resolve_intent_query,
    resolve_is_meta_request,
    is_meta_content_request,
    compute_lexical_score,
)
from data.topics import HELP_TOPIC_TEXT

logger = logging.getLogger(__name__)


class MockEncoder:
    """Deterministic random encoder matching data/generate_data.py MockEncoder."""

    def __init__(self, dim: int = VECTOR_DIM, seed: int = 42):
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


class VectorService:
    def __init__(self):
        self.model = None
        self.index: Optional[faiss.Index] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_metadata: List[Dict] = []
        self.topic_vectors: Optional[np.ndarray] = None
        self.item_topic_vecs: Optional[np.ndarray] = None
        self.help_topic_vec: Optional[np.ndarray] = None
        self._use_mock = False
        self._vector_manifest: dict = {}
        self._model_path = ""
        self._model_source = ""

    def load(self):
        if not self._data_files_exist():
            raise FileNotFoundError("Vector data files not found. Run data/generate_data.py first.")
        self._load_data()
        self._load_manifest()
        self._load_encoder()
        self._validate_manifest_encoder()

    def reload_data(self) -> dict:
        if not self._data_files_exist():
            raise FileNotFoundError("Vector data files not found. Generate data first.")
        self._load_data()
        self._load_manifest()
        if self.model is None:
            self._load_encoder()
        self._validate_manifest_encoder()
        self.help_topic_vec = self._encode_text(HELP_TOPIC_TEXT)
        logger.info("Vector data reloaded: %d items", len(self.item_metadata))
        return self.encoder_status()

    def _data_files_exist(self) -> bool:
        return all(
            os.path.exists(p)
            for p in [EMBEDDINGS_PATH, TOPIC_VECS_PATH, TOPIC_VIDEO_VECS_PATH, FAISS_INDEX_PATH, VIDEO_META_PATH]
        )

    def _load_data(self):
        self.item_embeddings = np.load(EMBEDDINGS_PATH)
        self.topic_vectors = np.load(TOPIC_VECS_PATH)
        self.item_topic_vecs = np.load(TOPIC_VIDEO_VECS_PATH)
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(VIDEO_META_PATH, "r", encoding="utf-8") as f:
            self.item_metadata = json.load(f)
        logger.info(
            "Loaded %d items, %d topics, index size=%d",
            len(self.item_metadata),
            self.topic_vectors.shape[0],
            self.index.ntotal,
        )

    def _load_manifest(self):
        self._vector_manifest = {}
        if os.path.isfile(VECTOR_MANIFEST_PATH):
            try:
                with open(VECTOR_MANIFEST_PATH, "r", encoding="utf-8") as f:
                    self._vector_manifest = json.load(f)
            except Exception as e:
                logger.warning("Failed to read vector manifest: %s", e)

    def _validate_manifest_encoder(self):
        data_mode = self._vector_manifest.get("mode")
        runtime_mode = "mock" if self._use_mock else "real"
        if data_mode and data_mode != runtime_mode:
            msg = f"Vector data mode={data_mode} but runtime encoder={runtime_mode}."
            if PUSH_USE_MOCK_VECTORS:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        if self.item_embeddings is not None and len(self.item_metadata) != len(self.item_embeddings):
            raise RuntimeError(
                f"Embedding rows ({len(self.item_embeddings)}) != metadata ({len(self.item_metadata)})."
            )

    def _load_encoder(self):
        if PUSH_USE_MOCK_VECTORS:
            logger.info("PUSH_USE_MOCK_VECTORS=1 — using mock encoder.")
            self.model = MockEncoder()
            self._use_mock = True
            self.help_topic_vec = self._encode_text(HELP_TOPIC_TEXT)
            return
        try:
            self.model, self._model_path, self._model_source = load_sentence_transformer()
            self._use_mock = False
        except Exception as e:
            raise RuntimeError(
                f"Real vectors require a local/hub embedding model: {e}. "
                "Run scripts/download_embedding_model.py or set PUSH_USE_MOCK_VECTORS=1."
            ) from e
        self.help_topic_vec = self._encode_text(HELP_TOPIC_TEXT)

    def encoder_status(self) -> dict:
        return {
            "encoder": "mock" if self._use_mock else "real",
            "model": MODEL_NAME if not self._use_mock else "mock",
            "model_path": self._model_path or resolve_model_path(),
            "model_source": self._model_source or ("mock" if self._use_mock else "unknown"),
            "vector_mode": self._vector_manifest.get("mode", "unknown"),
            "item_count": len(self.item_metadata),
        }

    def _encode_text(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def compute_topic_weights(self, ctx_vec: np.ndarray) -> np.ndarray:
        ctx_vec = ctx_vec / (np.linalg.norm(ctx_vec) + 1e-8)
        topic_norms = self.topic_vectors / (np.linalg.norm(self.topic_vectors, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(topic_norms, ctx_vec)
        return self._softmax(similarities / TOPIC_SOFTMAX_TEMP)

    def search_and_score(self, ctx_vec: np.ndarray, topic_weights: np.ndarray) -> List[Dict[str, Any]]:
        ctx_vec = ctx_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(ctx_vec, FAISS_TOPK)
        distances, indices = distances[0], indices[0]
        candidates = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.item_metadata):
                continue
            s_text = float(dist)
            t_item = self.item_topic_vecs[idx]
            match_topic = float(np.dot(topic_weights, t_item))
            s_final = SIMILARITY_TEXT_WEIGHT * s_text + SIMILARITY_TOPIC_WEIGHT * match_topic
            candidates.append(
                {
                    "idx": int(idx),
                    "meta": self.item_metadata[idx],
                    "vector": self.item_embeddings[idx].copy(),
                    "s_text": s_text,
                    "match_topic": match_topic,
                    "s_final": s_final,
                }
            )
        candidates.sort(key=lambda x: x["s_final"], reverse=True)
        return candidates

    def mmr_rerank(self, candidates: List[Dict], n: int = MMR_RESULT_N, lam: float = MMR_LAMBDA) -> List[Dict]:
        if len(candidates) <= n:
            return candidates
        result = []
        pool = list(candidates)
        while len(result) < n and pool:
            best_score = -float("inf")
            best_idx = -1
            for i, cand in enumerate(pool):
                relevance = cand["s_final"]
                max_sim = 0.0
                if result:
                    v_cand = cand["vector"] / (np.linalg.norm(cand["vector"]) + 1e-8)
                    max_sim = max(
                        float(np.dot(v_cand, r["vector"] / (np.linalg.norm(r["vector"]) + 1e-8)))
                        for r in result
                    )
                mmr_score = lam * relevance - (1 - lam) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            result.append(pool.pop(best_idx))
        return result

    def conv_smooth(self, scores: np.ndarray) -> np.ndarray:
        kernel = np.array(CONV_KERNEL, dtype=np.float32)
        n = len(scores)
        if n < 3:
            return scores.copy()
        smoothed = np.convolve(scores, kernel, mode="same")
        smoothed[0] = 0.8 * scores[0] + 0.2 * scores[1]
        smoothed[-1] = 0.8 * scores[-1] + 0.2 * scores[-2]
        return smoothed

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-8)

    def _lexical_catalog_search(self, user_text: str, topic_weights: np.ndarray, topk: int = FAISS_TOPK) -> List[Dict[str, Any]]:
        if not user_text.strip():
            return []
        scored: list[tuple[float, int, dict]] = []
        n_emb = len(self.item_embeddings) if self.item_embeddings is not None else 0
        for idx, meta in enumerate(self.item_metadata):
            if idx >= n_emb:
                break
            lex = compute_lexical_score(user_text, meta)
            if lex > 0.05:
                scored.append((lex, idx, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []
        candidates = []
        for lex, idx, meta in scored[:topk]:
            t_item = self.item_topic_vecs[idx]
            match_topic = float(np.dot(topic_weights, t_item))
            candidates.append(
                {
                    "idx": idx,
                    "meta": meta,
                    "vector": self.item_embeddings[idx].copy(),
                    "s_text": lex,
                    "match_topic": match_topic,
                    "s_final": lex,
                }
            )
        return candidates

    def process_request(self, context: str, intent=None) -> Tuple[List[CandidateItem], np.ndarray, float]:
        last_user = extract_last_user_message(context)
        intent_query = resolve_intent_query(context, intent) or last_user or context.strip()
        full_ctx_vec = self._encode_text(context)
        intent_vec = self._encode_text(intent_query)

        meta_request = resolve_is_meta_request(last_user or "", intent)
        if meta_request and intent_query != last_user:
            ctx_vec = 0.78 * intent_vec + 0.22 * full_ctx_vec
        elif len(extract_user_messages(context)) > 1:
            user_vec = self._encode_text(last_user) if last_user else intent_vec
            ctx_vec = 0.55 * intent_vec + 0.25 * user_vec + 0.20 * full_ctx_vec
        else:
            ctx_vec = intent_vec
        ctx_vec = (ctx_vec / (np.linalg.norm(ctx_vec) + 1e-8)).astype(np.float32)

        topic_weights = self.compute_topic_weights(ctx_vec)
        query_text = intent_query or last_user or context.strip()
        if self._use_mock and query_text:
            candidates = self._lexical_catalog_search(query_text, topic_weights)
            if not candidates:
                candidates = self.search_and_score(ctx_vec, topic_weights)
        else:
            candidates = self.search_and_score(ctx_vec, topic_weights)

        ranked = self.mmr_rerank(candidates)
        scores = np.array([c["s_final"] for c in ranked], dtype=np.float32)
        smoothed = self.conv_smooth(scores)
        probs = self._softmax(smoothed / SERVICE_SOFTMAX_TEMP)

        result = [
            CandidateItem(
                item_id=c["meta"]["item_id"],
                item_type=c["meta"].get("item_type", "item"),
                similarity=round(c["s_final"], 4),
                prob_base=round(float(probs[i]), 4),
                title=c["meta"]["title"],
                media_url=c["meta"].get("media_url") or c["meta"].get("cover_url") or "",
                duration_sec=c["meta"].get("duration_sec", 60),
                item_ref=c["meta"].get("item_ref"),
            )
            for i, c in enumerate(ranked)
        ]
        top1_similarity = result[0].similarity if result else 0.0
        return result, ctx_vec, top1_similarity
