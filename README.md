# ChatTorrent

A chatbot, with a large language model split across its users, entirely in the browser.

Every visitor can choose to hold a slice of the model on their own GPU, and together
the tabs answer everyone's chats. It's modelled after a torrenting network:
rather than one central inference server, it is decentralized across its users.
There is still one signaling server which introduces tabs to each other.
The model runs in browsers on WebGPU, with hand-written WGSL kernels and no ML
libraries. Browsers talk to each other through WebRTC.

I built this as an experiment to see how open models could be democratized and
made more accessible to everyday people with consumer hardware.

A demo may be running at [chattorrent.net](https://chattorrent.net), serving
Qwen3.8-27B at q8 (about 27 GiB, split over its peers).

## How it works

The model is cut into positions: the embedding, each transformer block, and
the head. A tab (a *peer*) holds a contiguous range of positions and downloads
only those weights. Ranges can overlap, so a peer leaving does not have to
break the model.

To generate a token, the residual stream is passed from peer to peer over
WebRTC datachannels. Each peer runs its range on its own GPU, and the peer
holding the head samples the token and feeds it back in for the next one.

The signaling server gives each tab an id, decides which range a new tab
should hold, broadcasts the roster, and relays WebRTC offers and answers. It
never sees a prompt, a reply or an activation.

A new tab starts with a small range, serves as soon as its first position has
downloaded, and grows its share in the background up to the VRAM limit its
visitor set. A tab with no GPU can still chat. When a peer leaves mid-reply,
the peer before it replays the lost work into another holder. If that fails,
the asking tab rebuilds the chat from its token history.

Also:

- Private meshes. An invite link (`/chat#mesh=<code>`) keeps its code in the
  URL fragment, and every WebRTC handshake is signed with a key derived from
  it, so only tabs holding the invite can link. The server sees who is in the
  room but cannot join it or read anything in it.
- Relay-only links by default, so peers do not learn each other's IP
  addresses.
- Chats are saved in the browser (IndexedDB) and can be resumed later.

## What it can run

`tools/convert_hf.py` converts a HuggingFace checkpoint and rejects any
architecture the kernels do not implement:

- Llama-style decoder blocks (RMSNorm, GQA/MHA, RoPE including Llama 3.1
  scaling, SwiGLU, tied or untied LM head): SmolLM2 and Llama 3.x.
- The Qwen3.8 hybrid (`qwen3_5`): Gated DeltaNet linear-attention layers mixed
  with full-attention layers that have partial RoPE, QK-norm and an attention
  output gate. Text only.

Weights are stored as f32, f16 or q8. The tokenizer must be byte-level BPE.
A peer needs a browser with WebGPU: desktop Chrome or Edge, Safari 26+, or
Firefox 141+ on Windows.

## Quick start

You need Node.js, Python 3, [Caddy](https://caddyserver.com/) and a browser
with WebGPU.

```bash
npm install
pip install -r tools/requirements.txt

# Convert a small model and point the mesh at it.
python tools/convert_hf.py --model HuggingFaceTB/SmolLM2-135M
echo '{ "url": "/weights/smollm2-135m" }' > checkpoint.json

# Serve the repo and the signaling server on http://localhost:8001.
export CHAT_PASS_HASH="$(caddy hash-password --plaintext <password>)"
PORT=8081 node signal-server.js &
caddy run --config Caddyfile.dev
```

Open `http://localhost:8001/web/peer.html?direct=1` in two or more tabs. Each
asks whether to contribute its GPU, then joins the mesh. One machine can hold
the whole mesh, since every tab gets a share of the same GPU.

## Tests

```bash
npm test            # Node suites, no GPU
npm run test:gpu    # real peers in headless Chrome, needs a GPU
```
## Documentation

- [`docs/deploying.md`](docs/deploying.md): converting checkpoints, running on
  a LAN or a domain, TURN, hosting the weights, configuration,
  troubleshooting.
- [`docs/architecture.md`](docs/architecture.md): positions, placement and
  growth, links, routing, the wire format, sessions, the KV cache, failure
  recovery, known gaps.
- [`docs/model.md`](docs/model.md): the forward pass and what each kernel
  computes.
- [`docs/development.md`](docs/development.md): page parameters, dev pages,
  tests, repo layout.

## License

GNU Affero General Public License v3.0. See [`LICENSE`](LICENSE). Under
[`NOTICE`](NOTICE), any deployment must keep a visible credit to ChatTorrent
in its interface.

The demo's privacy notice, terms and code of conduct (`site/privacy.html`,
`site/terms.html`) are written for that deploy and its
operator. Replace them if you run your own.
