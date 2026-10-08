#!/usr/bin/env bash
# 用法（在项目根目录）:
#   bash scripts/update_push_videos.sh
# 或
#   bash scripts/vod.sh
#
# 自动: 读 .env → 启动服务 → VOD 导出 → 重建向量 → 热重载（失败则 restart）
set -euo pipefail

ROOT="${PUSH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${1:-from-vod}"
VIDEOS_FILE="${PUSH_VIDEOS:-sample_videos.json}"

# 补全 ALIYUN_* / PUSH_ADMIN_TOKEN 的 .env 候选
ENV_CANDIDATES=(
  "$ROOT/.env"
)

resolve_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
    return 0
  fi
  echo "[ERROR] 未找到 docker compose"
  exit 1
}

load_env_chain() {
  local f key val
  for f in "${ENV_CANDIDATES[@]}"; do
    [[ -f "$f" ]] || continue
    echo "[env] $f"
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line//$'\r'/}"
      [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
      [[ "$line" != *"="* ]] && continue
      key="${line%%=*}"
      key="${key#"${key%%[![:space:]]*}"}"
      key="${key%"${key##*[![:space:]]}"}"
      val="${line#*=}"
      if [[ -z "${!key:-}" ]]; then
        export "$key=$val"
      fi
    done < "$f"
  done
  export PUSH_VIDEOS="${PUSH_VIDEOS:-sample_videos.json}"
  VIDEOS_FILE="$PUSH_VIDEOS"
}

require_aliyun() {
  if [[ -n "${ALIYUN_ACCESS_KEY_ID:-}" && -n "${ALIYUN_ACCESS_KEY_SECRET:-}" ]]; then
    export ALIYUN_VOD_REGION="${ALIYUN_VOD_REGION:-cn-shanghai}"
    return 0
  fi
  cat >&2 <<EOF
[ERROR] 缺少阿里云 VOD 密钥。

请在项目的 .env 中配置后重试:
  $ROOT/.env

需要:
  ALIYUN_ACCESS_KEY_ID=...
  ALIYUN_ACCESS_KEY_SECRET=...
  ALIYUN_VOD_REGION=cn-shanghai
EOF
  exit 1
}

ensure_stack_up() {
  echo "[up] docker compose up -d"
  "${COMPOSE[@]}" up -d
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "[ERROR] Push 未就绪，查看: ${COMPOSE[*]} logs push"
  exit 1
}

run_generate_data() {
  echo "[gen] 重建向量 ..."
  "${COMPOSE[@]}" exec -T push python "data/generate_data.py" "--videos=$VIDEOS_FILE"
}

run_vod_export_in_container() {
  local export_py="/app/scripts/export_aliyun_vod_videos.py"
  echo "[vod] 容器内导出 VOD ..."
  "${COMPOSE[@]}" exec -T push python "$export_py" \
    --output "/app/data/aliyun_videos.json" \
    --csv "/app/data/aliyun_videos.csv" \
    --raw "/app/data/aliyun_videos_raw.json"
  "${COMPOSE[@]}" exec -T push cp -f "/app/data/aliyun_videos.json" "/app/data/$VIDEOS_FILE"
}

run_vod_export_on_host() {
  local export_py="$ROOT/scripts/export_aliyun_vod_videos.py"
  [[ -f "$export_py" ]] || export_py="$ROOT/../scripts/export_aliyun_vod_videos.py"
  [[ -f "$export_py" ]] || { echo "[ERROR] 找不到 export_aliyun_vod_videos.py"; exit 1; }
  if ! python3 -c "import aliyunsdkvod" 2>/dev/null; then
    echo "[pip] 安装 VOD 依赖 ..."
    python3 -m pip install -q -r "$ROOT/requirements-vod-export.txt"
  fi
  echo "[vod] 宿主机导出 VOD（镜像未含导出脚本时的兜底）..."
  python3 "$export_py" \
    --output "$ROOT/data/aliyun_videos.json" \
    --csv "$ROOT/data/aliyun_videos.csv" \
    --raw "$ROOT/data/aliyun_videos_raw.json"
  cp -f "$ROOT/data/aliyun_videos.json" "$ROOT/data/$VIDEOS_FILE"
}

run_vod_export() {
  require_aliyun
  if "${COMPOSE[@]}" exec -T push test -f /app/scripts/export_aliyun_vod_videos.py 2>/dev/null; then
    if run_vod_export_in_container; then
      return 0
    fi
    echo "[WARN] 容器内导出失败，尝试宿主机 ..."
  fi
  run_vod_export_on_host
}

apply_reload() {
  local token="${PUSH_ADMIN_TOKEN:-}"
  if [[ -n "$token" ]]; then
    echo "[reload] 热重载 ..."
    if curl -sf -X POST "http://127.0.0.1:8000/admin/videos/reload" \
      -H "Authorization: Bearer $token" >/dev/null; then
      curl -s "http://127.0.0.1:8000/health"
      echo
      return 0
    fi
    echo "[WARN] 热重载失败（需新镜像含 /admin/videos/reload），改为 restart"
  fi
  echo "[restart] push ..."
  "${COMPOSE[@]}" restart push
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      curl -s "http://127.0.0.1:8000/health"
      echo
      return 0
    fi
    sleep 2
  done
  echo "[WARN] 健康检查超时"
  return 1
}

[[ -d "$ROOT" ]] || { echo "[ERROR] 目录不存在: $ROOT"; exit 1; }
resolve_compose
cd "$ROOT"
load_env_chain
[[ -f docker-compose.yml ]] || { echo "[ERROR] 缺少 docker-compose.yml"; exit 1; }
mkdir -p "$ROOT/data"

case "$MODE" in
  from-vod|"")
    ensure_stack_up
    run_vod_export
    run_generate_data
    apply_reload
    echo "[done] 视频库已从 VOD 更新"
    ;;
  rebuild)
    ensure_stack_up
    run_generate_data
    apply_reload
    ;;
  reload)
    ensure_stack_up
    apply_reload
    ;;
  restart)
    "${COMPOSE[@]}" restart push
    apply_reload
    ;;
  *)
    echo "用法: bash update_push_videos.sh   # 默认 = VOD 全量更新"
    echo "      bash update_push_videos.sh {rebuild|reload|restart}"
    exit 1
    ;;
esac
