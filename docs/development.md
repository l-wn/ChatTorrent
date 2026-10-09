# Development

## Page parameters

`peer.html` (`/chat` in production) takes URL parameters that pin what a tab
holds, which turns background growth off:

```
?blocks=N    hold exactly N positions (0 = a client that holds nothing)
?gb=N        hold as many positions as N GiB buys
?at=lo-hi    take exactly this range
?slow=ms     pause between downloaded positions (to watch growth and filling)
?direct=1    link directly instead of only through TURN
```

Its settings panel has the sampler (temperature, top-k, repetition,
frequency and presence penalties, no-repeat n-gram, max tokens), the Qwen
thinking mode, **Decoding** (the closed loop, or a round trip through the
asking tab for every token), and **Profile**, which times each hop's GPU pass
(level 1) or every kernel (level 2) and shows the breakdown under the reply.

## Dev pages

Served only by `Caddyfile.dev`, never by the production build:

- `web/chat.html`: the whole model in one tab.
- `web/dev.html`: kernel tests, the smoke check, the tokenizer self-test, a
  full diff against a HuggingFace oracle, and a generation run. Parameters:
  `?weights=<url>`, `?oracle=<url>`, `?stream=1` (verify a block at a time,
  for a model larger than this GPU), `?decodeSteps=N`, and `?bench=1` (kernel
  and forward-pass microbenchmarks at the checkpoint's dimensions) or
  `?bench=<hidden>x<n_layer>x<n_heads>x<n_kv>x<head_dim>x<ffn>x<vocab>`.
- `web/dev/history.html`: the saved-chat checks against a real IndexedDB.

The oracle dumps in `oracle/` come from `tools/dump_*_oracle.py`, which run the
HuggingFace model and save reference tensors.

## Tests

```bash
npm test            # the Node suites in tools/*.test.mjs, no GPU
npm run test:gpu    # builds a tiny model, then runs real peers in headless Chrome
```

`npm test` covers placement, invites and their cryptography, saved chats, the
signaling server over real sockets, a chain of Runners over a stub GPU, and
`mesh.test.mjs`: N routers on a simulated wire, including a seeded soak that
kills peers at random (`--seed N`) and a fuzzer (`--fuzz N --fuzz-seed S`).

`npm run test:gpu` needs `google-chrome` and a real GPU (it aborts on a
software adapter). It builds `weights/tiny-test` (`tools/tinymodel.mjs`, the
real architecture at 1.3 MB) and runs `tools/mesh.gpu.test.mjs`: several people
prompting at once across many real tabs, the real signaling server and real
datachannels, with greedy decoding diffed against a single peer. Options:
`--only NAME`, `--wire bc` (BroadcastChannel instead of WebRTC, for fast
iteration), `--prompts N`, `--round-trip`, `--fuzz N --seed S`, `--heavy`,
`--show`.

What each suite checks is in [`architecture.md`](architecture.md), *Testing*.

## Layout

```
checkpoint.json      which checkpoint the mesh runs
signal-server.js     signaling: the join, rooms, TURN, origin and per-IP limits
signal-room.js       one mesh on the server: ids, placement, roster, SDP relay
Caddyfile, start.sh  the production deploy; Caddyfile.dev for local development
site/                public static pages and assets (about, 404, legal pages, icon, robots.txt, site.css)
dist/                built by tools/build-site.mjs; the only thing production serves (not committed)
web/                 the browser code (architecture.md, Components); peer.html is /chat
web/model/           kernels, forward pass, loader, tokenizer, sampler
web/dev/             kernel tests, oracle verification, benchmarks, test harness page
tools/build-site.mjs web/ + site/ → dist/
tools/convert_hf.py  HuggingFace checkpoint → weights/<name>/
tools/dump_*.py      HuggingFace oracle dumps → oracle/
tools/*.test.mjs     Node test suites; mesh.gpu.test.mjs drives headless Chrome
tools/harness/       static server and Chrome DevTools client for the GPU tests
oracle/              reference tensors for verification, written by tools/dump_*.py
weights/             converted checkpoints (not committed)
docs/                architecture, model, deploying and development docs
LICENSE, NOTICE      AGPL-3.0 and the attribution term
```
