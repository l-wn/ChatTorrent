<div align="center">

<img src="site/favicon.svg" alt="" width="72" height="72" />

# ClusterChat

**Run a 27B open model on the graphics cards your group already has, in the browser.**<br />
Everyone lends part of their GPU. Everyone gets their own chat.

[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-7022d6)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-hand--written%20kernels-0b0b0c)](docs/model.md)
[![WebRTC](https://img.shields.io/badge/WebRTC-relay%20only-0b0b0c)](docs/architecture.md)
[![Model](https://img.shields.io/badge/model-Qwen3.8--27B%20q8-0b0b0c)](#what-it-can-run)

[**Start a private cluster**](https://clusterchat.io/create) ·
[Try the public demo](https://clusterchat.io/chat) ·
[How it works](#how-it-works) ·
[Docs](#documentation)

<br />

<img src="docs/img/cluster.png" alt="Four computers in one cluster. Each lends a different amount of GPU memory and each person has their own chat, three of them writing replies at the same time." width="900" />

</div>

## Highlights

- **Your own chats.** Each person's conversations are separate and saved in
  their own browser. A cluster runs several replies at once.
- **Private clusters.** Start one and share the invite link. Only browsers
  holding it can connect. The signaling server can't join or read anything.
- **Nobody has to stay.** Each piece of the model can be held by more than one
  browser, so when someone leaves, the others carry on.
- **Nothing to install.** WebGPU runs the model with custom kernels.
  WebRTC links the browsers through a relay, so nobody learns anyone else's IP
  address.
- **Lending is optional.** Any modern browser can join just to chat.

> [!NOTE]
> **Your chat is separate, not secret.** Every computer in your cluster
> processes it. Invite people you trust. The public demo is open to anyone, so
> don't send it anything personal.

## At a glance

| | |
|---|---|
| **Model** | Qwen3.8-27B, 8-bit, about 27 GiB across the cluster |
| **Speed** | 6 tokens/s at default settings, up to 12 on repetitive text, on two RTX 3090s in different machines ([details](#performance)) |
| **Replies at once** | set by whoever starts the cluster |
| **Chats each browser keeps in memory** | 8 by default, up to 32 in a private cluster |
| **People per cluster** | up to 64 |
| **Install** | none: a browser tab |

## Performance

Qwen3.8-27B q8 on two RTX 3090s in separate machines, 81 ms round trip
between them, one reply at a time, 128-token replies. Tokens per second, total
tokens over total decode time:

| Prompts | Sampling | Plain | Prompt lookup | Prompt lookup + MTP |
|---|---|--:|--:|--:|
| Chat questions | page defaults | 4.0 | 4.1 | **6.0** |
| Repetitive (lists, code, tables) | page defaults | 4.2 | 4.2 | **8.1** |
| Chat questions | greedy | 4.4 | 6.2 | **9.1** |
| Repetitive (lists, code, tables) | greedy | 4.2 | 6.3 | **11.7** |

Every token is a full lap across both GPUs and the link between them, so
plain decoding is capped by the lap time. Speculative decoding breaks that
cap: the head drafts up to 7 tokens, from earlier text and from the model's
own multi-token prediction (MTP) head, and one lap checks them all. The text
is the same as plain decoding gives. With greedy sampling, every reply matched
its plain counterpart.

The numbers were measured with [`tools/load.mjs`](tools/load.mjs) on
2026-10-24, before sampling moved onto the GPU. The MTP head is opt-in for
now: the browser holding the head loads it with `?mtp=1`. The method is in
[Speculative decoding](docs/architecture.md#speculative-decoding).

```bash
# One visitor, six turns. Add --greedy and --prompts copy for the other rows.
node tools/load.mjs --url https://clusterchat.io/chat --users 1 --turns 6 --tokens 128 --spec
```

## How it works

<img src="docs/img/how-it-works.svg" alt="Ana's laptop and Lab PC 3 each hold pieces 1 to 6 of the model. Ben's gaming PC and Mia's desktop each hold pieces 7 to 12. Ben's message runs on Ana's laptop, then his own PC. Mia's runs on Lab PC 3, then her desktop. All four connect to a signaling server and a relay." width="100%" />

The model is cut into positions: the embedding, each transformer block, and
the head. A browser (a *peer*) holds a contiguous range of positions and
downloads only those weights. Ranges can overlap, so a peer leaving does not
have to break the model.

To generate a token, the residual stream is passed from peer to peer over
WebRTC datachannels. Each peer runs its range on its own GPU, and the peer
holding the head samples the token and feeds it back in for the next one.
Every chat is its own session with its own KV cache on each peer, and routing
balances sessions across peers that hold the same positions.

The signaling server gives each browser an id, decides which range a new
browser should hold, broadcasts the roster, and relays WebRTC offers and
answers. It never sees a prompt, a reply or an activation.

A new browser starts with a small range, serves as soon as its first position
has downloaded, and grows its share in the background up to the VRAM limit its
owner set. A browser with no GPU can still chat. When a peer leaves mid-reply,
the peer before it replays the lost work into another holder. If that fails,
the asking browser rebuilds the chat from its token history.

## Private clusters

A private cluster is for a group: friends, a lab, a club. Start one at
[clusterchat.io/create](https://clusterchat.io/create). You choose how much
GPU to lend and how many replies can run at once, then share the invite link.

- **The invite code never reaches the server.** It lives in the link's
  fragment (`/chat#mesh=<code>`), which browsers don't send.
- **Handshakes are signed.** Every WebRTC handshake is signed with a key
  derived from the code, so only browsers holding the invite can link.
- **The servers see who, not what.** The signaling server sees who is in the
  cluster but cannot join it or read anything in it.
- **Relay-only links** by default, so peers don't learn each other's IP
  addresses.
- **Chats stay in your browser** (IndexedDB) and can be resumed or exported
  later.

## What it doesn't do yet

- **One model per cluster.**
- **Chat only.** There is no API and no tool use yet. An OpenAI-compatible API
  is planned.
- **Peers are trusted.** Nothing checks that a browser ran its piece honestly.
- **First come, first served.** Each browser runs one job queue, so a long
  prompt can briefly slow other chats on the same peers.
- **Metadata is visible to the servers.** They see who is in a cluster, but not
  what anyone says.

The full list is under Known gaps in
[`docs/architecture.md`](docs/architecture.md#failures).

## Browser support

| Browser | Can lend its GPU on |
|---|---|
| Chrome, Edge 113+ | Windows, macOS, ChromeOS, Android. Linux may need WebGPU switched on in `chrome://flags`. |
| Safari 26+ | macOS, iPhone, iPad |
| Firefox | Windows (141+), Apple silicon Macs (145+) |

Any modern browser can join to chat without lending.

## What it can run

`tools/convert_hf.py` converts a Hugging Face checkpoint and rejects any
architecture the kernels do not implement:

- Llama-style decoder blocks (RMSNorm, GQA/MHA, RoPE including Llama 3.1
  scaling, SwiGLU, tied or untied LM head): SmolLM2 and Llama 3.x.
- The Qwen3.8 hybrid (`qwen3_5`): Gated DeltaNet linear-attention layers mixed
  with full-attention layers that have partial RoPE, QK-norm and an attention
  output gate, plus its MTP draft head. Text only.

Weights are stored as f32, f16 or q8. The tokenizer must be byte-level BPE.

## Quick start

You need Node.js, Python 3, [Caddy](https://caddyserver.com/) and a browser
with WebGPU.

```bash
git clone https://github.com/l-wn/ClusterChat && cd ClusterChat
npm install
pip install -r tools/requirements.txt

# Convert a small model and point the cluster at it.
python tools/convert_hf.py --model HuggingFaceTB/SmolLM2-135M
echo '{ "url": "/weights/smollm2-135m" }' > checkpoint.json

# Serve the repo and the signaling server on http://localhost:8001.
export CHAT_PASS_HASH="$(caddy hash-password --plaintext <password>)"
PORT=8081 node signal-server.js &
caddy run --config Caddyfile.dev
```

Open `http://localhost:8001/chat?direct=1` in two or more tabs. Each asks
whether to lend its GPU, then joins the cluster. One machine can hold the
whole cluster, since every tab gets a share of the same GPU.

## Tests

```bash
npm test            # Node suites, no GPU
npm run test:gpu    # real peers in headless Chrome, needs a GPU
```

## Documentation

| | |
|---|---|
| [`docs/deploying.md`](docs/deploying.md) | Converting checkpoints, running on a LAN or a domain, TURN, hosting the weights, configuration, troubleshooting |
| [`docs/architecture.md`](docs/architecture.md) | Positions, placement and growth, links, routing, the wire format, sessions, the KV cache, speculative decoding, failure recovery, known gaps |
| [`docs/model.md`](docs/model.md) | The forward pass and what each kernel computes |
| [`docs/development.md`](docs/development.md) | Page parameters, dev pages, tests, repo layout |

## License

GNU Affero General Public License v3.0. See [`LICENSE`](LICENSE). Under
[`NOTICE`](NOTICE), any deployment must keep a visible credit to ClusterChat
in its interface.

The demo's privacy notice, terms and code of conduct (`site/privacy.html`,
`site/terms.html`) are written for that deploy and its operator. Replace them
if you run your own.
