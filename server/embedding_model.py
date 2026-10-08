"""Load sentence-transformers from project-local models/ or HuggingFace hub."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PUSH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DIR = PUSH_ROOT / "models" / MODEL_NAME


def local_model_dir() -> Path:
    return DEFAULT_LOCAL_DIR


def is_valid_model_dir(path: Path | str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    has_config = (p / "config.json").is_file()
    has_weights = (p / "model.safetensors").is_file() or (p / "pytorch_model.bin").is_file()
    return has_config and has_weights


def _active_trained_dir():
    try:
        from selftrain.registry import get_active

        active = get_active()
        if active and is_valid_model_dir(active):
            return active
    except Exception:
        pass
    return None


def resolve_model_path() -> str:
    """Prefer trained active version → EMBEDDING_MODEL_PATH → local models/<name> → HF hub id."""
    trained = _active_trained_dir()
    if trained is not None:
        return str(trained)

    override = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if override and is_valid_model_dir(override):
        return str(Path(override).resolve())

    local = DEFAULT_LOCAL_DIR
    if is_valid_model_dir(local):
        return str(local)

    return MODEL_NAME


def model_source_label(resolved: str) -> str:
    if resolved == MODEL_NAME:
        return "hub"
    if Path(resolved).resolve() == DEFAULT_LOCAL_DIR.resolve():
        return "local"
    return "custom"


def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    path = resolve_model_path()
    source = model_source_label(path)
    logger.info("Loading embedding model (%s): %s", source, path)
    return SentenceTransformer(path), path, source
