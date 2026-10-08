#!/usr/bin/env bash
# 启动 Push Microservice（Linux/macOS 开发用）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f data/faiss_index.bin ]]; then
  echo "[push] 缺少向量索引，生成中..."
  if [[ "${PUSH_USE_MOCK_VECTORS:-0}" =~ ^(1|true|yes)$ ]]; then
    python data/generate_data.py --mock
  else
    python data/generate_data.py
  fi
fi

exec python -m uvicorn server.main:app --host "${SERVER_HOST:-0.0.0.0}" --port "${SERVER_PORT:-8000}"
