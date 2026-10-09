"""FAISS L2 distances must rank nearer neighbors first."""

import numpy as np

from data.generate_data import MockEncoder as OfflineEncoder
from server.vector_service import MockEncoder as OnlineEncoder
from server.vector_service import VectorService


def test_hnsw_l2_prefers_near_neighbor():
    import faiss

    near = np.array([1, 0, 0, 0], dtype=np.float32)
    far = np.array([0, 1, 0, 0], dtype=np.float32)
    index = faiss.IndexHNSWFlat(4, 4)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 1
    index.add(np.stack([far, near]))

    service = VectorService()
    service.index = index
    service.item_metadata = [
        {"item_id": "far", "title": "far"},
        {"item_id": "near", "title": "near"},
    ]
    service.item_embeddings = np.stack([far, near])
    service.item_topic_vecs = np.zeros((2, 2), dtype=np.float32)

    hits = service.search_and_score(near, np.zeros(2, dtype=np.float32))
    assert [hit["meta"]["item_id"] for hit in hits] == ["near", "far"]
    assert hits[0]["s_text"] > 0.99
    assert hits[1]["s_text"] < 0.01
    assert index.hnsw.efSearch >= 30


def test_mock_encoder_is_stable_across_processes():
    offline = OfflineEncoder().encode(["billing invoice", "other"])
    online = OnlineEncoder().encode(["billing invoice", "other"])
    again = OnlineEncoder().encode(["billing invoice"])
    assert np.allclose(offline, online)
    assert np.allclose(online[0], again[0])
    assert not np.allclose(online[0], online[1])
