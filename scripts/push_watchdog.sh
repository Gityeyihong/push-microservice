#!/usr/bin/env bash
# Push 进程存活：/health 失败则重启容器。依赖是否就绪看 /ready，不要拿它触发重启。
# 用法：cron/systemd 每小时执行；或手动 bash push_watchdog.sh
set -euo pipefail

PUSH_DIR="${PUSH_DIR:-/opt/push-microservice}"
PUSH_HEALTH_URL="${PUSH_HEALTH_URL:-http://127.0.0.1:8000/health}"
PUSH_SERVICE="${PUSH_SERVICE:-push}"
COMPOSE_FILE="${PUSH_COMPOSE_FILE:-}"
LOG_TAG="${PUSH_WATCHDOG_LOG_TAG:-push-watchdog}"

log() {
  logger -t "$LOG_TAG" "$*" 2>/dev/null || echo "[$LOG_TAG] $*"
}

resolve_compose() {
  if [[ -n "$COMPOSE_FILE" && -f "$PUSH_DIR/$COMPOSE_FILE" ]]; then
    :
  elif [[ -f "$PUSH_DIR/docker-compose.yml" ]]; then
    COMPOSE_FILE="docker-compose.yml"
  elif [[ -f "$PUSH_DIR/docker-compose.cloud.yml" ]]; then
    COMPOSE_FILE="docker-compose.cloud.yml"
  else
    log "skip: no compose file under $PUSH_DIR"
    exit 0
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose -f "$PUSH_DIR/$COMPOSE_FILE")
  elif docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose -f "$PUSH_DIR/$COMPOSE_FILE")
  else
    log "skip: docker compose not found"
    exit 0
  fi
}

health_ok() {
  curl -sf --max-time "${PUSH_HEALTH_TIMEOUT:-5}" "$PUSH_HEALTH_URL" >/dev/null 2>&1
}

restart_push() {
  local cid
  cid="$("${COMPOSE[@]}" ps -q "$PUSH_SERVICE" 2>/dev/null | head -1 || true)"
  if [[ -z "$cid" ]] || ! docker ps --format '{{.ID}}' | grep -qx "$cid"; then
    log "push down/missing — compose up -d $PUSH_SERVICE"
    (cd "$PUSH_DIR" && "${COMPOSE[@]}" up -d "$PUSH_SERVICE")
  else
    log "push unhealthy — docker restart $cid"
    docker restart "$cid" >/dev/null
  fi
}

main() {
  [[ -d "$PUSH_DIR" ]] || { log "skip: PUSH_DIR missing ($PUSH_DIR)"; exit 0; }
  resolve_compose

  if health_ok; then
    exit 0
  fi

  log "health check failed ($PUSH_HEALTH_URL)"
  restart_push

  sleep "${PUSH_WATCHDOG_WAIT_SEC:-8}"
  if health_ok; then
    log "recovered"
    exit 0
  fi

  log "still unhealthy after restart"
  exit 1
}

main "$@"
