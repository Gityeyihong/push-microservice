"""Self-training for the retrieval embedding model (Plasmid-inspired).

Modules:
    constants   default configuration
    events      training event stream
    reviewer    external review gate (API + manual, self-review disabled)
    collector   push/engagement data capture
    datagen     contrastive sample construction
    gate        sample gating (positive/correction/negative/drop)
    safety      training safety net (diversity, held-out regression, rollback)
    trainer     sentence-transformers fine-tuning
    registry    versioned model registry + active pointer + rollback
    evaluator   held-out retrieval metrics
    loop        orchestration
"""

from .constants import load_training_config, DEFAULT_CONFIG  # noqa: F401

__all__ = ["load_training_config", "DEFAULT_CONFIG"]
