#!/usr/bin/env bash
# Start the public deploy: signal server, Caddy, Cloudflare tunnel. systemd
# supervises the processes; this script drives the three `chattorrent-*` user
# units. start and restart rebuild dist/ first, so the site serves current
# code. See `./start.sh help`, and docs/deploying.md for the units and
# .env.

set -euo pipefail

UNITS=(chattorrent-signal chattorrent-caddy chattorrent-cloudflared)

usage() {
    cat <<EOF
Usage: ./start.sh [start|stop|restart|status|logs|dev]

  start     enable + start all three services (default)
  stop      stop all three (they stay enabled, so the next boot starts them again)
  restart   restart all three
  status    one-line state per service
  logs      follow all three logs together (Ctrl-C to stop watching)
  dev       run the dev site (Caddyfile.dev) and its own signaling server in
            the foreground, at http://localhost:8001
EOF
}

build() { node "$(dirname "$0")/tools/build-site.mjs" > /dev/null; echo "built dist/"; }

start() {
    build
    systemctl --user daemon-reload
    systemctl --user enable --now "${UNITS[@]}"
    echo
    status
    echo
    echo "https://chattorrent.net/chat"
}

stop()    { systemctl --user stop "${UNITS[@]}"; }
restart() { build; systemctl --user daemon-reload; systemctl --user restart "${UNITS[@]}"; status; }

status() {
    for u in "${UNITS[@]}"; do
        printf '%-28s %s\n' "$u" "$(systemctl --user is-active "$u" 2>&1)"
    done
}

logs() { journalctl --user -u chattorrent-signal -u chattorrent-caddy -u chattorrent-cloudflared -f; }

# Values are read literally, not sourced: a bcrypt hash contains `$`.
load_env() {
    local k v
    while IFS='=' read -r k v; do
        [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && export "$k=$v"
    done < "$HOME/.config/chattorrent/env"
}

# The dev signaling server is separate from the deploy's, so dev tabs do not
# join the public mesh.
dev() {
    cd "$(dirname "$0")"
    load_env
    PORT=8081 node signal-server.js &
    local sig=$!
    trap 'kill $sig 2>/dev/null' EXIT
    echo "dev: http://localhost:8001/web/peer.html?direct=1"
    caddy run --config Caddyfile.dev
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs ;;
    dev)     dev ;;
    -h|--help|help) usage ;;
    *) echo "unknown command: $1" >&2; usage >&2; exit 1 ;;
esac
