# Deploying

A ClusterChat deploy is three things:

1. **The pages**: static files (`web/`, plus `site/` in production).
2. **The signaling server**: `signal-server.js`, a Node WebSocket server the
   pages reach at `/ws` on their own origin.
3. **The weights**: a converted checkpoint, fetched by every peer from a URL
   named in `checkpoint.json`.

The repo uses [Caddy](https://caddyserver.com/) to serve the pages and proxy
`/ws` to the signaling server, but any web server that can do both works.

Pages need a *secure context*, or WebGPU and the private-mesh cryptography are
unavailable: `http://localhost` counts, anything else must be HTTPS.

## 1. Install

Prerequisites: Node.js, Python 3 (for conversion only), Caddy, and a browser
with WebGPU.

```bash
npm install                          # the signaling server's one dependency, ws
pip install -r tools/requirements.txt   # torch, transformers, numpy
```

## 2. Convert a checkpoint

`tools/convert_hf.py` takes a HuggingFace model id or local path and writes
`weights/<name>/`:

```bash
python tools/convert_hf.py --model HuggingFaceTB/SmolLM2-135M      # small, good for a first run
python tools/convert_hf.py --model Qwen/Qwen3.8-27B --dtype q8
python tools/convert_hf.py --model <id> --out weights/<name> --dtype f16
```

The output is `manifest.json` (every dimension, the dtype, the chat format),
`client.bin` (embedding and final norm), one `block_NN.bin` per block,
`lm_head.bin` when the head is untied, and `tokenizer/`. Options: `--dtype
f32|f16|q8` (default f32), `--out`, `--force` to overwrite, `--no-verify` to
skip the round-trip check. A gated HuggingFace model needs you to be logged in
(`huggingface-cli login`).

Then point `checkpoint.json` at it. For a local run, same-origin is simplest:

```json
{ "url": "/weights/smollm2-135m" }
```

Every page and the signaling server read this one file, so the whole mesh runs
one checkpoint. Restart the signaling server after changing it.

## 3. Run it locally

`Caddyfile.dev` serves the whole repo (including `weights/`) on
`http://localhost:8001` and proxies `/ws` to a signaling server on `:8081`.
Requests not addressed to localhost need a basic-auth password (user
`chatbot`), and Caddy won't start without its hash, so set one even for a local
run:

```bash
export CHAT_PASS_HASH="$(caddy hash-password --plaintext <password>)"
PORT=8081 node signal-server.js &
caddy run --config Caddyfile.dev
```

`./start.sh dev` does the same in one step, reading `CHAT_PASS_HASH` (and
anything else) from `~/.config/chattorrent/env` (see *Configuration*).

Then open:

- `http://localhost:8001/web/peer.html?direct=1`: the mesh. Open it in
  several tabs, or several browsers; each asks whether to contribute its GPU
  and how much, then joins. The header shows what each tab holds and the
  network's redundancy `r` (0: the model can't run yet; 1: it runs, but one
  departure breaks it; 2+: any one peer can leave). `?direct=1` lets tabs
  link directly, since a local run has no TURN relay.
- `http://localhost:8001/web/chat.html`: the whole model in one tab, no mesh.
  Useful for a model that fits on one GPU.

A single machine can hold the whole mesh: every tab gets its own share of the
same GPU.

## 4. Share it on a LAN

For friends on the same network, the dev setup is enough, with three changes:

1. **Listen on the LAN.** In `Caddyfile.dev`, change `bind 127.0.0.1 [::1]` to
   `bind 0.0.0.0` (or the machine's LAN address). The signaling server can
   stay on loopback: pages reach it through Caddy's `/ws`.
2. **Allow the page's origin.** The signaling server refuses sockets from
   pages it doesn't know (any `localhost` port is always allowed). Start it
   with the address the others will use:
   `PORT=8081 ALLOWED_ORIGINS=https://<host>:8001 node signal-server.js`.
3. **Give it HTTPS.** `http://<lan-ip>:8001` is not a secure context, so on
   the other machines WebGPU (and private meshes) won't work. Either serve it
   over HTTPS (in `Caddyfile.dev`, remove `auto_https off` and use a site
   address such as `https://<host>:8001` with `tls internal`, then trust
   Caddy's local root certificate on each machine), or, for a quick test in
   Chromium, add the address under
   `chrome://flags/#unsafely-treat-insecure-origin-as-secure` on each machine
   (and use that `http://` origin in `ALLOWED_ORIGINS`).

Everyone opens `peer.html?direct=1` at that address and enters the password.
Direct links work on one LAN without TURN. Together the tabs need enough GPU
memory to hold the whole model, at least once.

## 5. Put it on a domain

The production setup serves a locked-down build of the pages behind TLS, with
a TURN relay so peers on different networks can link.

**Build the site.** `npm run build` (`tools/build-site.mjs`) assembles
`dist/` from an allowlist: `web/peer.html` as `chat.html`, the modules it
imports, `checkpoint.json`, and everything in `site/`. Nothing else in the repo
(dotfiles, `web/dev.html`, `web/chat.html`, `tools/`, `oracle/`, `weights/`)
is reachable. Rebuild after every change.

**Serve it.** `Caddyfile` serves `dist/` on `127.0.0.1:8000` with clean URLs,
security headers, and `/ws` proxied to the signaling server on `:8080`:

| URL | |
|---|---|
| `/` | redirects to `/chat` until `site/index.html` exists |
| `/chat` | the global mesh; `/chat#mesh=<code>` a private one |
| `/ws` | the signaling server |
| `/<name>` | `site/<name>.html`: a page added to `site/` needs no config |

It sets `auto_https off` and expects something in front of it to terminate
TLS. Two ways to do that:

- **A Cloudflare Tunnel** (what the demo uses): `cloudflared tunnel run` with
  an ingress rule sending your hostname to `http://localhost:8000`. Nothing
  listens on a public port. In the Cloudflare dashboard, WebSockets must be on
  and Rocket Loader off for the zone.
- **Caddy itself**: replace `:8000` with your domain, delete
  `auto_https off` and the `bind` line, and Caddy gets a certificate
  automatically (ports 80 and 443 must reach it).

Any other reverse proxy works too, as long as it passes WebSocket upgrades on
`/ws`.

**Run the signaling server** on `127.0.0.1:8080` (`node signal-server.js`)
with at least `ALLOWED_ORIGINS=https://<your-domain>`. Its default is
`https://chattorrent.net`, which your pages won't match.

**Set up TURN.** By default `/chat` links tabs only through a TURN relay, and
the signaling server hands each peer the credentials in its welcome. Without
TURN, only tabs opened with `?direct=1` can link, and those see each other's
IP addresses. Two kinds are supported:

- **Your own TURN server** (recommended): [coturn](https://github.com/coturn/coturn)
  on a small VPS, with `use-auth-secret` and `static-auth-secret=<secret>` in
  its config. Set `TURN_SECRET` to the same secret and `TURN_URLS` to its
  URLs. The signaling server mints each peer its own credential, valid for
  24 hours, with no round trip. A VPS with a fixed price and included traffic,
  plus an outgoing bandwidth cap on it (`tc qdisc … tbf rate 50mbit`), limits
  what a stolen credential can cost you. Test it with the
  [Trickle ICE page](https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/):
  a `relay` candidate means it works.
- **[Cloudflare Realtime TURN](https://developers.cloudflare.com/realtime/turn/)**
  (`TURN_KEY_ID`, `TURN_API_TOKEN`): credentials minted through Cloudflare's
  API, shared by every peer and re-minted at half their 24-hour TTL. It is
  billed per GB relayed past a free allowance, with no spending cap, and
  anyone who opens the page can copy the credential.

When both are configured, your own server wins.

**Replace the demo's pages.** `site/privacy.html`, `site/terms.html`, the
welcome popup in `web/peer.html` and the credit ribbon describe the demo and
its operator; rewrite them for your deploy.

**Keep it running.** `./start.sh` drives three systemd user units in
`~/.config/systemd/user/` (`start | stop | restart | status | logs`; `start`
and `restart` rebuild `dist/` first). You write the units yourself:

| unit | runs |
|---|---|
| `chattorrent-signal` | `node signal-server.js`, `WorkingDirectory=` the repo |
| `chattorrent-caddy` | `caddy run`, `WorkingDirectory=` the repo (picks up `Caddyfile`) |
| `chattorrent-cloudflared` | `cloudflared tunnel --config ~/.cloudflared/config.yml run` |

The first two load `EnvironmentFile=%h/.config/chattorrent/env`.

## 6. Host the weights

Every peer downloads the positions it holds straight from the weights host, so
it needs to be fast, cheap to serve from, and reachable from your pages'
origin. The production `Caddyfile` doesn't serve `weights/` (a same-origin
`/weights/<name>` works only under `Caddyfile.dev`), so host them elsewhere:
object storage behind a CDN, or any static file server.

The host must:

- serve the whole `weights/<name>/` directory as is (`manifest.json`,
  `client.bin`, `block_NN.bin`, `lm_head.bin`, `tokenizer/`), plus the
  model's `LICENSE`/`NOTICE` if its license asks for them;
- allow CORS from your site's origin for `GET` and `HEAD`, with the `Range`
  header allowed. Files are fetched whole first so the browser and CDN can
  cache them; after a failure they are fetched again in 8 MiB ranges with
  retries;
- ideally send long cache lifetimes: a converted file never changes, so
  publish a changed checkpoint under a new name.

Some browsers can't fetch a single body over about 2 GiB, which is why an
untied LM head ships as its own file; a q8 27B checkpoint stays under that.

The demo uses Cloudflare R2 behind a custom domain. With an rclone remote for
R2:

```bash
rclone copy weights/<name> r2:<bucket>/<name> --transfers 4 --checksum --progress
rclone check weights/<name> r2:<bucket>/<name> --checksum
curl -sI https://<weights-host>/<name>/manifest.json
```

Then set `checkpoint.json` to `{ "url": "https://<weights-host>/<name>" }`,
set `WEIGHTS_ORIGIN=https://<weights-host>` in the env file (for the
Content-Security-Policy), and `./start.sh restart`: the signaling server reads
the checkpoint once at boot, and fetches its manifest for the model's depth.
Caddy revalidates `checkpoint.json` on every load, so browsers pick it up
without a hard reload.

## Configuration

The deploy reads `~/.config/chattorrent/env` (mode 600, outside the repo),
plain `KEY=value` lines:

```bash
CHAT_PASS_HASH=<caddy hash-password --plaintext …>   # Caddyfile.dev only
ALLOWED_ORIGINS=https://<your-domain>
WEIGHTS_ORIGIN=https://<weights-host>
TURN_SECRET=<coturn's static-auth-secret>
TURN_URLS=turn:turn.<your-domain>:3478?transport=udp,turn:turn.<your-domain>:3478?transport=tcp
# or, instead of those two:
# TURN_KEY_ID=<Cloudflare Realtime TURN key id>
# TURN_API_TOKEN=<its API token>
```

Signaling server environment:

| variable | default | |
|---|---|---|
| `PORT` | 8080 | |
| `HOST` | 127.0.0.1 | `0.0.0.0` to accept sockets directly rather than only through the proxy |
| `ALLOWED_ORIGINS` | `https://chattorrent.net` | comma-separated page origins that may open a socket; any `localhost` port is always allowed |
| `MAX_CONN_PER_IP` | 16 | sockets per client address, by Cloudflare's `CF-Connecting-IP` (uncapped without it) |
| `WEIGHTS_URL` | from `checkpoint.json` | override the checkpoint URL |
| `MANIFEST` | `./<url>/manifest.json` for a same-origin URL | the server's own copy of the manifest |
| `N_LAYER`, `MIN_BLOCKS` | from the manifest | override the model depth / minimum range |
| `TURN_SECRET`, `TURN_URLS` | unset | your own TURN server: coturn's `static-auth-secret`, and its comma-separated `turn:` URLs. Each peer gets its own 24-hour credential |
| `TURN_KEY_ID`, `TURN_API_TOKEN` | unset | Cloudflare TURN, used when the two above are unset. Without either kind, the default relay-only links can't open and only `?direct=1` tabs can link |
| `MAX_ROOMS`, `MAX_PEERS_PER_ROOM` | 1000, 64 | caps on private meshes, and on peers in any one mesh |
| `MAX_CHATS` | unset (no limit) | replies that may generate at once in the global mesh |
| `MAX_PRIVATE_CHATS` | unset (no limit) | cap on a private mesh's own limit, and its limit when the creator set none |
| `CHAT_LEASE_MS` | 600000 | a seat whose `chat-end` never arrives is freed after this long |
| `MESH_SESSIONS` | 8 | chats each peer keeps in memory in the global mesh, and in a private mesh whose creator set none. Peers size their KV pool to match, about 335 MiB per chat across Qwen3.8-27B at 16K |
| `MAX_PRIVATE_SESSIONS` | 32 | cap on a private mesh creator's "Chats in memory" |
| `MESH_CTX` | 16384 | context window in tokens in the global mesh, and in a private mesh whose creator set none. Rounded down to a length in `split.js`'s `CONTEXTS` and to the model's `n_pos` |
| `MAX_PRIVATE_CTX` | 32768 | cap on a private mesh creator's "Chat length" |

Caddy environment: `WEIGHTS_ORIGIN` (default
`https://weights.chattorrent.net`), and for `Caddyfile.dev`, `CHAT_PASS_HASH`
and `SIGNAL_PORT` (default 8081).

## Troubleshooting

- **Links do not open between networks**: TURN is not configured, or its
  credentials failed to mint (the signaling server logs which at boot). With
  coturn, check that `TURN_SECRET` matches its `static-auth-secret` and that
  the relay ports (UDP 49152–65535) are open; `journalctl -u coturn` shows
  refused logins. In
  `chrome://webrtc-internals`, a working relay shows a
  `candidateType=relay` candidate pair.
- **WebSocket refused with 403**: the page's origin is not in
  `ALLOWED_ORIGINS`. **429**: one address has `MAX_CONN_PER_IP` sockets open.
- **Cloudflare 5xx on `/ws`**: WebSockets must be enabled for the zone.
- **A page is blank**: Rocket Loader must be off for the zone (it rewrites
  module scripts); the console shows any CSP violation. The
  Content-Security-Policy is report-only until promoted in `Caddyfile`.
- **No WebGPU on another machine**: the page isn't a secure context (plain
  `http://` to anything but localhost).
- **A peer stays a spectator**: its GPU failed the kernel or smoke self-checks
  (the reason is shown in the page), it has no WebGPU, or its visitor chose
  not to contribute.
- **The weights don't load**: check the manifest URL with `curl -sI`, and the
  CORS headers on the weights host (the browser console names the file).

