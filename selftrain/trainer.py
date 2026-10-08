"""Embedding fine-tuning with sentence-transformers (MultipleNegativesRankingLoss).

Training is intended to run in an isolated subprocess (see `run_isolated`).
`python -m selftrain.trainer --base ... --samples ... --out ...` is the CLI used
by that subprocess, so heavy deps (torch/datasets) are imported lazily only there.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _resolve_base(base_model_path: str | None) -> str:
    if base_model_path:
        p = Path(base_model_path)
        if p.is_dir():
            return str(p)
        return base_model_path
    from server.embedding_model import resolve_model_path

    return resolve_model_path()


def train(
    base_model_path: str | None,
    samples: list[dict],
    output_dir: str | Path,
    params: dict | None = None,
) -> str:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    params = params or {}
    if not samples:
        raise ValueError("empty training set")

    loss_name = str(params.get("loss", "mnrl_weighted"))
    model = SentenceTransformer(_resolve_base(base_model_path))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(params.get("epochs", 3))
    batch_size = int(params.get("batch_size", 32))
    warmup = float(params.get("warmup_ratio", 0.1))
    max_steps = int(params.get("max_steps", -1))

    if loss_name == "soft_cosine":
        examples = [
            InputExample(texts=[s.get("anchor", ""), s.get("positive", "")], label=float(s.get("reward", 1.0)))
            for s in samples
            if s.get("anchor") and s.get("positive")
        ]
        if not examples:
            raise ValueError("no usable training examples")
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.CosineSimilarityLoss(model)
        steps = max(1, int(len(loader) * epochs * warmup))
        model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=steps,
                  output_path=str(output_dir), show_progress_bar=False)
        return str(output_dir)

    examples = []
    weights = []
    for s in samples:
        texts = [s.get("anchor", ""), s.get("positive", "")]
        texts.extend(n for n in (s.get("negatives") or []) if n)
        texts = [t for t in texts if t]
        if len(texts) >= 2:
            examples.append(InputExample(texts=texts))
            weights.append(max(0.05, float(s.get("reward", 1.0) or 1.0)))
    if not examples:
        raise ValueError("no usable training examples")

    sampler = None
    try:
        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    except Exception:
        sampler = None

    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = max_steps if max_steps > 0 else max(1, int(len(loader) * epochs * warmup))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=steps,
        output_path=str(output_dir),
        show_progress_bar=False,
    )
    return str(output_dir)


def run_isolated(
    base_model_path: str | None,
    samples: list[dict],
    output_dir: str | Path,
    params: dict | None = None,
    *,
    timeout: int = 3600,
) -> str:
    """Run training in a separate process; raises on failure/timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(samples, tmp, ensure_ascii=False)
        samples_path = tmp.name

    cmd = [
        sys.executable,
        "-m",
        "selftrain.trainer",
        "--base",
        base_model_path or "",
        "--samples",
        samples_path,
        "--out",
        str(output_dir),
        "--params",
        json.dumps(params or {}, ensure_ascii=False),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"isolated training failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
        )
    return str(output_dir)


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune embedding model")
    parser.add_argument("--base", default="")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    with open(args.samples, "r", encoding="utf-8") as f:
        samples = json.load(f)
    params = json.loads(args.params or "{}")
    path = train(args.base or None, samples, args.out, params)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
