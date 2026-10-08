#!/usr/bin/env bash
# 安装 Push 死后自动重启：systemd timer 每小时探测 /health
set -euo pipefail

PUSH_DIR="${PUSH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WATCHDOG_SRC="${WATCHDOG_SRC:-$PUSH_DIR/scripts/push_watchdog.sh}"
SERVICE_SRC="${SERVICE_SRC:-$PUSH_DIR/deploy/push-watchdog.service}"
TIMER_SRC="${TIMER_SRC:-$PUSH_DIR/deploy/push-watchdog.timer}"

if [[ ! -f "$WATCHDOG_SRC" ]]; then
  echo "[ERROR] missing $WATCHDOG_SRC"
  exit 1
fi
if [[ ! -f "$SERVICE_SRC" || ! -f "$TIMER_SRC" ]]; then
  echo "[ERROR] missing $SERVICE_SRC or $TIMER_SRC"
  echo "       copy deploy/push-watchdog.* into $PUSH_DIR/deploy/ first"
  exit 1
fi

mkdir -p "$PUSH_DIR/scripts" "$PUSH_DIR/deploy"
dest_watchdog="$PUSH_DIR/scripts/push_watchdog.sh"
if [[ "$WATCHDOG_SRC" -ef "$dest_watchdog" ]]; then
  chmod 0755 "$dest_watchdog"
else
  install -m 0755 "$WATCHDOG_SRC" "$dest_watchdog"
fi

install -m 0644 "$SERVICE_SRC" /etc/systemd/system/push-watchdog.service
install -m 0644 "$TIMER_SRC" /etc/systemd/system/push-watchdog.timer

systemctl daemon-reload
systemctl enable --now push-watchdog.timer

echo "OK: push-watchdog.timer enabled (every 1h)"
systemctl status push-watchdog.timer --no-pager || true
echo "Manual probe: bash $PUSH_DIR/scripts/push_watchdog.sh"
