#!/bin/sh
# Start as root only long enough to make bind-mounted data writable, then drop to uid 10001.
set -eu
if [ "$(id -u)" = "0" ]; then
  mkdir -p /app/data /app/reports /app/config
  chown -R push:push /app/data /app/reports /app/config
  exec runuser -u push -- "$@"
fi
exec "$@"
