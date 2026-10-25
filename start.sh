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
Usage: ./start.sh [start|stop|restart|status|logs|dev|deploy]

  start     enable + start all three services (default)
  stop      stop all three (they stay enabled, so the next boot starts them again)
  restart   restart all three
  status    one-line state per service
  logs      follow all three logs together (Ctrl-C to stop watching)
  dev       run the dev site (Caddyfile.dev) and its own signaling server in
            the foreground, at http://localhost:8001
  deploy    copy the tracked files to DEPLOY_HOST (set in the env file) and restart
            the services there
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
    echo "dev: http://localhost:8001/ (the chat: http://localhost:8001/chat?direct=1)"
    caddy run --config Caddyfile.dev
}

# Sends the working tree's tracked files, committed or not. DEPLOY_HOST
# (user@host) comes from the environment or the env file.
deploy() {
    cd "$(dirname "$0")"
    [[ -n ${DEPLOY_HOST:-} ]] || load_env
    [[ -n ${DEPLOY_HOST:-} ]] || { echo "set DEPLOY_HOST=user@host in ~/.config/chattorrent/env" >&2; exit 1; }
    git ls-files -z | rsync -a --from0 --files-from=- ./ "$DEPLOY_HOST:Code/ChatTorrent/"
    ssh "$DEPLOY_HOST" 'cd Code/ChatTorrent && npm ci --omit=dev --no-audit --no-fund > /dev/null && XDG_RUNTIME_DIR=/run/user/$(id -u) ./start.sh restart'
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs ;;
    dev)     dev ;;
    deploy)  deploy ;;
    -h|--help|help) usage ;;
    *) echo "unknown command: $1" >&2; usage >&2; exit 1 ;;
esac
