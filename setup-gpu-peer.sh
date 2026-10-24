#!/usr/bin/env bash
# Run a ChatTorrent GPU peer on a rented, headless GPU machine. This file is
# self-contained: copy it over with scp, then
#
#   ./setup-gpu-peer.sh install      once: Chrome, the Vulkan loader, an NVIDIA ICD
#   ./setup-gpu-peer.sh start        headless Chrome on the peer page, in the background
#
# If the site is behind basic auth, start asks for the password once (or reads
# CHAT_PASS from the environment) and keeps it for restarts.
#
# The page opens on the welcome popup (the agreement and the GPU question), and
# the GPU is not used until it is answered. Answer it from your own machine
# through an SSH tunnel to Chrome's DevTools port:
#
#   ssh -L 9222:localhost:9222 you@gpubox       # keep this open while you click
#   local Chrome → chrome://inspect → Configure… → add localhost:9222 → inspect
#
# The DevTools port listens only on the GPU machine's loopback, so the tunnel
# is the only access. After you press Contribute the tunnel can be closed, and
# the tab keeps serving until `./setup-gpu-peer.sh stop`.

set -euo pipefail

# ?direct=1 lets this peer link directly to other direct peers (GPU to GPU),
# falling back to the TURN relay when they cannot reach each other. It shows
# this machine's address to the tabs it links to; visitors stay relay-only.
URL_DEFAULT='https://chattorrent.net/chat?direct=1'
PORT=${PORT:-9222}
STATE=${STATE:-$HOME/.chattorrent-peer}
PROFILE=$STATE/chrome-profile
LOG=$STATE/chrome.log
PIDFILE=$STATE/chrome.pid
AUTHFILE=$STATE/auth

usage() {
    cat <<EOF
Usage: ./setup-gpu-peer.sh [install|start [URL]|stop|restart [URL]|status|logs|forget]

  install   install google-chrome and the Vulkan loader (apt), and add an NVIDIA
            Vulkan ICD file if the driver has none (common in rented containers)
  start     launch headless Chrome on URL (default $URL_DEFAULT)
            in the background, DevTools on localhost:$PORT. If the site is behind
            its password gate, give the login once (see below); it is kept in
            \$STATE/auth (mode 600) so restart needs nothing
  forget    delete the saved login
  stop      stop that Chrome
  restart   stop, then start
  status    running or not, and which GPU the page got
  logs      follow the page's console output (Ctrl-C to stop watching)

Environment:
  CHAT_USER, CHAT_PASS   the site's login (user defaults to chatbot). Without
                         CHAT_PASS, start asks for it when run in a terminal;
                         CHAT_PASS= (set but empty) means the site has no gate
  PORT                   DevTools port, default 9222
  STATE                  where the profile, log and login live, default ~/.chattorrent-peer

  CHAT_PASS=secret ./setup-gpu-peer.sh start      # non-interactive
EOF
}

SUDO=
[ "$(id -u)" -eq 0 ] || SUDO=sudo

chrome_bin() {
    command -v google-chrome || command -v google-chrome-stable || true
}

running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

install() {
    if ! command -v apt-get > /dev/null; then
        echo "no apt-get here: install google-chrome and the Vulkan loader by hand" >&2
        exit 1
    fi
    $SUDO apt-get update -q
    $SUDO apt-get install -yq wget ca-certificates libvulkan1 vulkan-tools
    if [ -z "$(chrome_bin)" ]; then
        local deb
        deb=$(mktemp --suffix=.deb)
        wget -qO "$deb" https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
        $SUDO apt-get install -yq "$deb"
        rm -f "$deb"
    fi

    # Rented containers often have the NVIDIA driver's GL/Vulkan library but no
    # ICD file pointing at it. The Vulkan loader then does not find the GPU and
    # Chrome falls back to SwiftShader, a slow CPU rasterizer.
    local icds=(/usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json)
    if [ ! -e "${icds[0]}" ] && [ ! -e "${icds[1]}" ] \
        && ldconfig -p | grep -q libGLX_nvidia.so.0; then
        $SUDO mkdir -p /etc/vulkan/icd.d
        echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
            | $SUDO tee /etc/vulkan/icd.d/nvidia_icd.json > /dev/null
        echo "wrote /etc/vulkan/icd.d/nvidia_icd.json"
    fi

    echo
    "$(chrome_bin)" --version
    check_vulkan
}

check_vulkan() {
    if ! command -v vulkaninfo > /dev/null; then return; fi
    local gpus
    gpus=$(vulkaninfo --summary 2>/dev/null | grep -i 'deviceName' || true)
    echo "Vulkan devices:"
    echo "${gpus:-  (none)}"
    if [ -z "$gpus" ] || ! grep -qiv 'llvmpipe\|swiftshader' <<< "$gpus"; then
        echo "WARNING: no hardware Vulkan device. Chrome will fall back to a CPU renderer." >&2
        echo "  On a container, the host must expose the NVIDIA 'graphics' capability" >&2
        echo "  (NVIDIA_DRIVER_CAPABILITIES=all), not only 'compute,utility'." >&2
    fi
}

# Percent-encode a string for the userinfo part of a URL, so a password
# containing : @ / # or spaces is preserved.
urlencode() {
    local s=$1 out= c i
    for ((i = 0; i < ${#s}; i++)); do
        c=${s:i:1}
        case $c in
            [a-zA-Z0-9.~_-]) out+=$c ;;
            *) printf -v c '%%%02X' "'$c"; out+=$c ;;
        esac
    done
    printf '%s' "$out"
}

# The login: from the environment, then the saved file, then a prompt. Sets
# LOGIN_USER / LOGIN_PASS. An empty LOGIN_PASS means no basic auth.
login() {
    LOGIN_USER=${CHAT_USER:-chatbot}
    if [ -n "${CHAT_PASS+set}" ]; then
        LOGIN_PASS=$CHAT_PASS
    elif [ -f "$AUTHFILE" ]; then
        { IFS= read -r LOGIN_USER; IFS= read -r LOGIN_PASS; } < "$AUTHFILE"
        LOGIN_USER=${CHAT_USER:-$LOGIN_USER}
        return
    elif [ -t 0 ]; then
        IFS= read -rsp "Password for $LOGIN_USER (Enter if the site has no gate): " LOGIN_PASS
        echo
    else
        LOGIN_PASS=
    fi
    if [ -n "$LOGIN_PASS" ]; then
        (umask 077; printf '%s\n%s\n' "$LOGIN_USER" "$LOGIN_PASS" > "$AUTHFILE")
    fi
}

# DevTools' HTTP endpoints: list the open tabs' ids, and open a tab on a URL,
# printing its id. The URL goes in the query, percent-encoded whole, so a
# #fragment is preserved.
tab_ids() {
    curl -fs "http://127.0.0.1:$PORT/json/list" | grep -o '"id": *"[^"]*"' | cut -d'"' -f4
}
open_tab() {
    curl -fs -X PUT "http://127.0.0.1:$PORT/json/new?$(urlencode "$1")" \
        | grep -o '"id": *"[^"]*"' | cut -d'"' -f4
}

forget() { rm -f "$AUTHFILE"; echo "saved login deleted"; }

start() {
    local url=${1:-$URL_DEFAULT}
    local chrome
    chrome=$(chrome_bin)
    if [ -z "$chrome" ]; then
        echo "google-chrome not found: run ./setup-gpu-peer.sh install first" >&2
        exit 1
    fi
    if running; then
        echo "already running (pid $(cat "$PIDFILE")); run ./setup-gpu-peer.sh restart to relaunch" >&2
        exit 1
    fi
    mkdir -p "$PROFILE"
    chmod 700 "$STATE"
    login

    # Headless WebGPU on Linux: Vulkan through ANGLE, GPU sandbox off. The
    # three background flags stop Chrome throttling a tab that is not visible,
    # which would make a peer waiting on a job look hung. mDNS candidates do
    # not resolve reliably on rented hosts, so real addresses are used.
    # --no-sandbox because these machines usually run as root.
    setsid "$chrome" \
        --headless=new \
        --no-sandbox \
        --disable-gpu-sandbox \
        --enable-unsafe-webgpu \
        --enable-features=Vulkan \
        --use-angle=vulkan \
        --ignore-gpu-blocklist \
        --disable-backgrounding-occluded-windows \
        --disable-background-timer-throttling \
        --disable-renderer-backgrounding \
        --disable-features=WebRtcHideLocalIpsWithMdns \
        --no-first-run \
        --no-default-browser-check \
        --remote-debugging-address=127.0.0.1 \
        --remote-debugging-port="$PORT" \
        --user-data-dir="$PROFILE" \
        --enable-logging=stderr --v=0 \
        about:blank \
        > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"

    # Chrome is up once the DevTools endpoint answers.
    for _ in $(seq 30); do
        if curl -fs "http://127.0.0.1:$PORT/json/version" > /dev/null 2>&1; then break; fi
        if ! running; then
            echo "Chrome exited at startup. Last lines of $LOG:" >&2
            tail -20 "$LOG" >&2
            exit 1
        fi
        sleep 1
    done

    # The page is opened through DevTools, not on the command line, so the
    # login is not in Chrome's argv, where `ps` would show it. The login uses a
    # temporary tab on the site's root: Chrome answers the basic-auth challenge
    # from the URL's userinfo and caches it for the origin. The peer page is
    # then opened without userinfo, because a page whose URL has credentials
    # cannot fetch() relative URLs. Every request it makes (scripts, fetches,
    # the signaling WebSocket) uses the cached login.
    local blank login_tab=
    blank=$(tab_ids | head -1)
    if [ -n "$LOGIN_PASS" ]; then
        local scheme=${url%%://*} host=${url#*://}
        host=${host%%[/?#]*}
        login_tab=$(open_tab "$scheme://$(urlencode "$LOGIN_USER"):$(urlencode "$LOGIN_PASS")@$host/")
        sleep 3
    fi
    open_tab "$url" > /dev/null
    for id in $blank $login_tab; do
        curl -fs "http://127.0.0.1:$PORT/json/close/$id" > /dev/null || true
    done

    local host
    host=$(hostname -f 2>/dev/null || hostname)
    cat <<EOF
Chrome is running (pid $(cat "$PIDFILE")) on $url
It is waiting on the welcome popup. From your own machine:

  ssh -L $PORT:localhost:$PORT <you>@$host     (add -p <port> if your provider gives one)

With several machines, give each tunnel its own local port (the left-hand
number), e.g. -L 9223:localhost:$PORT for the second, -L 9224:… for the third.

then in your local Chrome open chrome://inspect, click "Configure…", add
localhost:$PORT (and each other local port), and press "inspect" under the
ChatTorrent tab listed for that port. Tick both
boxes, pick an amount, press Contribute. Check it got the real GPU with:

  ./setup-gpu-peer.sh status
EOF
}

stop() {
    if ! running; then echo "not running"; rm -f "$PIDFILE"; return; fi
    local pid
    pid=$(cat "$PIDFILE")
    # setsid made Chrome a process-group leader, so one signal reaches every child.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 10); do running || break; sleep 1; done
    running && kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "stopped"
}

status() {
    if running; then echo "running (pid $(cat "$PIDFILE")), DevTools on localhost:$PORT"
    else echo "not running"; return; fi
    # peer.js logs the adapter once the GPU is opened, which is after Contribute.
    local adapter
    adapter=$(grep -o 'WebGPU adapter: [^"]*' "$LOG" | tail -1 || true)
    if [ -z "$adapter" ]; then
        echo "no GPU opened yet: the welcome popup is probably still unanswered"
    else
        echo "$adapter"
        if grep -qi 'swiftshader\|llvmpipe' <<< "$adapter"; then
            echo "WARNING: that is a CPU renderer, not the GPU. See ./setup-gpu-peer.sh install" >&2
        fi
    fi
}

logs() {
    # Chrome logs a lot. Only the page's console lines are shown.
    tail -n +1 -F "$LOG" | grep --line-buffered 'CONSOLE'
}

case "${1:-help}" in
    install)  install ;;
    start)    start "${2:-}" ;;
    stop)     stop ;;
    restart)  stop; start "${2:-}" ;;
    status)   status ;;
    logs)     logs ;;
    forget)   forget ;;
    help|-h|--help) usage ;;
    *)        usage; exit 1 ;;
esac
