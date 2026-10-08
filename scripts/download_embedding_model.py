#!/usr/bin/env python3
"""Download embedding model into push-system/models/ for fully offline inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PUSH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUSH_ROOT))

from server.embedding_model import DEFAULT_LOCAL_DIR, MODEL_NAME, is_valid_model_dir  # noqa: E402

REPO_ID = f"sentence-transformers/{MODEL_NAME}"


def download_via_hub(target: Path) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] huggingface_hub → {target}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )


def download_via_git(target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Target not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO_ID}"
    print(f"[download] git clone → {target}")
    subprocess.run(
        ["git", "clone", url, str(target)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download embedding model to push-system/models/")
    parser.add_argument(
        "--method",
        choices=("hub", "git"),
        default="hub",
        help="hub=snapshot_download (recommended); git=git clone HF repo",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if local model exists",
    )
    args = parser.parse_args()

    target = DEFAULT_LOCAL_DIR
    if is_valid_model_dir(target) and not args.force:
        print(f"[skip] Local model already present: {target}")
        print("       Use --force to re-download.")
        return

    if args.force and target.exists():
        import shutil
        shutil.rmtree(target)

    if args.method == "git":
        download_via_git(target)
    else:
        download_via_hub(target)

    if not is_valid_model_dir(target):
        raise SystemExit(f"Download finished but model invalid: {target}")

    print(f"[ok] Model ready: {target}")
    print("     Push service will load this path automatically (no HF network at runtime).")


if __name__ == "__main__":
    main()
