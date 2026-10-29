# The model math

What the WGSL kernels in `web/model/kernels.js` compute, and how
`web/model/model.js` strings them into a forward pass. Two block kinds are
supported: the Llama-style full-attention block (SmolLM2, Llama 3.x, and
Qwen3.8-27B's full-attention layers) and Gated DeltaNet (Qwen3.8-27B's
linear-attention layers).

Every dimension comes from the checkpoint's manifest (`dimsFromManifest()` in
`web/model/loader.js`). `DIMS` in `kernels.js` holds SmolLM2-135M's values and
is used only by the tests and the smoke check.

## Conventions

- **Weights are `[out, in]`** (`nn.Linear`'s own layout), so nothing is
  transposed on export. Every projection is `Y[M,N] = X[M,K] @ W[N,K]^T`
  (`matmulNobias`), and so is the LM head, which reads `wte` directly when tied.
- **No biases** anywhere.
- **Activations and every accumulator are f32.** Stored weights are f16 or
  q8, and the KV cache is int8 with f32 scales (see Paged KV cache). The
  shaders widen both on read.
- `T_new` is the number of tokens in this job, `T_past` how many the session
  already has cached, and `T_act = T_past + T_new`.

## Full-attention block

Pre-norm residual. `x` is `[T_new, H]`.

```
h      = rmsnorm(x, input_layernorm)
q      = h @ q_proj^T                      # [T, n_heads*D]
k, v   = h @ k_proj^T, h @ v_proj^T        # [T, n_kv*D] each
(q, k  = rmsnorm per head (q_norm, k_norm))               # Qwen only
Qh     = reshape_heads(q)                  # [n_heads, T, D]
store_kv(rope(k) → K cache, v → V cache) at T_past       # int8, paged, head-major
rope(Qh)                                                  # V is never rotated
scores = Qh @ K^T / sqrt(D)                # [n_heads, T, T_act], GQA
causal_softmax(scores)                     # mask j > T_past + i
ctx    = scores @ V                        # [n_heads, T, D], GQA
merged = merge_heads(ctx)                  # [T, n_heads*D]
(merged = sigmoid(h @ attn_gate^T) * merged)              # Qwen only
x     += merged @ o_proj^T

h      = rmsnorm(x, post_attention_layernorm)
x     += (silu(h @ gate_proj^T) * (h @ up_proj^T)) @ down_proj^T    # SwiGLU
```

Then, after the last block, `rmsnorm(x, norm)` and the LM head
(`lm_head`, or `wte` when tied) give `[T, vocab]` logits. Only the last row is
read back and sampled.

### RMSNorm

```
y[t,h] = x[t,h] / sqrt(mean_h(x[t,h]^2) + eps) * weight[h]
```

No mean subtraction, no bias, ε inside the sqrt, as in HF. ε varies by
checkpoint and is baked into the shader source.

Qwen's RMSNorm computes `norm(x) * (1 + weight)`. `tools/convert_hf.py` adds
the 1 at conversion time, so the same kernel serves both. The Gated DeltaNet
output norm (`gated_norm`) uses a plain weight and gets no +1.

### RoPE

Split-half pairing: channel `i` pairs with `i + rot_dim/2`, **not** the
interleaved `(2i, 2i+1)` of GPT-J. The wrong pairing still produces fluent
text, but not this model's output.

```
inv_freq[i] = theta ^ (-2i / rot_dim)            i in [0, rot_dim/2)
angle       = (T_past + t) * inv_freq[i]
out[i]          = x[i] * cos - x[i + half] * sin
out[i + half]   = x[i + half] * cos + x[i] * sin
```

- `rot_dim` is `head_dim` except on a partial-rotary checkpoint (Qwen3.8-27B:
  `head_dim / 4`), where channels `[rot_dim, head_dim)` pass through untouched.
- Llama 3.1's `rope_scaling` rescales `inv_freq` (NTK-aware: unchanged at
  short wavelengths, divided by `factor` at long ones, blended between).
- θ comes from the manifest. SmolLM2 uses 100000, not the Llama default of
  10000.
- cos/sin come from a table built on the CPU in f64 (`ropeTable()`), not from
  shader trig. GPU transcendental accuracy varies by vendor by more than the
  tolerance allows, and peers splitting a model must agree.

### GQA

Query head `h` reads KV head `h / group`, with `group = n_heads / n_kv_heads`.
Scores and outputs keep `n_heads` heads; the KV cache is only `n_kv_heads`
deep. The three attention kernels run over chunks of the job's rows
(`attnRows()` in model.js), so the scores buffer stays near 32 MiB at long
contexts. Every output element has the same bits in any chunking.

### Paged KV cache

A session's K and V live in fixed-size pages (`PAGE_TOKENS`, 64 tokens) drawn
from a pool that holds one buffer per full-attention layer per K/V. A
per-session page table maps a logical token position to a physical page:

```
row = (pages[t / page] * n_kv + head) * page + (t % page)
```

A row is one token of one KV head, `RW = D/4 + D/QB` u32 words: `D` int8
values packed four to a word, then one f32 scale per `QB = min(32, D)` values
(`scale = max|x| / 127` over the block, values rounded to nearest and clamped
to ±127, as llama.cpp's q8_0). On Qwen3.8-27B a token of K and V on one
full-attention layer is 2.25 KiB, against 8 KiB in f32. `split.js`'s
`kvRowBytes()` prices it, and the server and the peers use the same number.

`store_kv` applies RoPE to K before quantizing, so a stored row is never
rotated again. Scales never span tokens, so a row's bits depend only on that
row's input: a rewind, a speculative overwrite, a chunked prefill and a
different split all write the same bytes.

`store_kv`, `qk_scores_gqa` and `attn_av_gqa` all read the table. A
contiguous cache (`makeCache()`, used by the single-tab page and the dev tools)
is the one-page case: `page = N_max`, `pages = [0]`.

## Gated DeltaNet block (Qwen3.8-27B linear attention)

No KV cache: each session carries a fixed-size recurrent state per layer,
`S[heads, Dk, Dv]`, plus a short history ring for the causal convolution. Both
persist across jobs, so a job needs no `T_past`.

```
h    = rmsnorm(x, input_layernorm)
q, k = h @ in_proj_q^T, h @ in_proj_k^T      # [T, nK*Dk]
v, z = h @ in_proj_v^T, h @ in_proj_z^T      # [T, nV*Dv]
b, a = h @ in_proj_b^T, h @ in_proj_a^T      # [T, nV]

q, k, v = silu(causal_conv1d(·))             # depthwise, width K, reading the
                                             # history ring for rows before
                                             # this job
q, k = l2norm per head, repeated nV/nK times
out  = gated_delta_net_step(q, k, v, a, b)   # recurrence below, updates S
out  = rmsnorm per head (gated_norm) * silu(z)
x   += out @ out_proj^T
update the conv history rings from this job's pre-conv q, k, v

h    = rmsnorm(x, post_attention_layernorm)
x   += SwiGLU(h)                             # same MLP as full attention
```

The recurrence, per head and token (`torch_recurrent_gated_delta_rule`):

```
g      = exp(-exp(A_log) * softplus(a + dt_bias))
beta   = sigmoid(b)
S      = g * S
kv[v]  = Σ_k S[k][v] * k[k]
delta  = (v_in - kv) * beta
S     += k ⊗ delta
out[v] = (1/sqrt(Dk)) * Σ_k S[k][v] * q[k]
```

`gated_delta_net_step` runs a whole chunk of tokens in one dispatch, with each
thread's slice of `S` held in registers. `A_log` and `dt_bias` are packed into
one `gate_params` buffer so the kernel fits WebGPU's minimum of 8 storage
bindings per stage. The converter splits the fused `in_proj_qkv` and
`conv1d` weights into q/k/v parts and keeps `conv1d`, `gate_params` and
`gated_norm` in f32.

A reused session frame must have its recurrent state cleared, because the
state is accumulated, not overwritten (`Runner.#cacheFor` does it).

### Rollback after a speculative verify

A verify job of n rows (architecture.md, Speculative decoding) is followed by
a job that keeps k <= n of them. `linearBlock` takes the session's replay data
for the layer (`makeReplayLayer`):

```
verify job:  copy S, hist{Q,K,V} -> replay          (copy_f32, before the block)
             in_proj_{q,k,v,a,b} write into replay.lin{Q,K,V,A,B}
             the block runs on those buffers as usual
next job, k < n:
             copy replay S, hist -> S, hist
             causal_conv1d, l2norm, gated_delta_net_step, update_conv_history
             over replay.lin* rows [0, k)          (linearCore, as in the forward)
             then the job's own block
```

The restore runs the forward pass's kernels on the forward pass's inputs, so
the state it leaves has the bits k one-token jobs leave. k = 0 (a verify
refused further down the line) only copies the state back.

## Matching plain decoding row by row

A verify row must produce the logits a one-token job produces for the same
position, or a verified token could differ from plain decoding's. Each kernel
on the verify path gives row i the same arithmetic in the same order as a
one-row call:

- the batched GEMV (2 to 32 rows) reads each weight once and accumulates every
  row with the GEMV's lane split, expression and reduction tree. Past 8 rows,
  q8 weights use `matmulRowsQ8Src`, which stages each row's slice of X in
  workgroup memory instead of having every thread re-read it from memory
  (on a 1070 Ti at Qwen3.8-27B's shapes, 16 rows then cost what 8 do and 32
  rows twice that). Which thread does the work changes, the arithmetic per
  row and column does not;
- attention: a score is one dot product, softmax visits only unmasked columns
  with a fixed thread-to-column mapping, and the value sum adds exact zeros for
  masked columns;
- rmsnorm, RoPE and the elementwise kernels work per row, and
  `gated_delta_net_step` runs tokens in order with the state in registers.

`kernels.test.js` reports whether a GPU matches to the bit (it needs only the
usual tolerance to serve); `spec.test.js` requires it.

## Batching several sessions in one pass

A Runner can run several sessions' jobs as one pass (architecture.md,
Batching). `blockSegs` takes a list of segments, each one session's `n` rows
at row `rowOff` of the scratch, with its cache, `T_past` and speculative
state. Rows are packed in order from row 0.

```
every row at once:   rmsnorm, every projection, q/k norm, attn_gate,
                     o_proj, out_proj, gated norm, the MLP, round_f16
per segment:         copy its q/k/v (or DeltaNet projections) to row 0 of
                     the segment scratch (makeSegScratch); store_kv, rope,
                     attention over its own pages (or the restore, snapshot
                     and recurrence on its own state); copy the output back
```

With at most 32 rows the projections use the batched GEMV above, so each row
gets the bits a one-row call gives, and a batched session's rows, KV and
recurrent state are the ones it gets alone. `batch.test.js` checks this to
the bit for both block kinds, f32 and q8 weights, with a verify and a
rollback inside the batch. One segment runs exactly the dispatches of
`block()`, with no copies.

## Weight formats

| dtype | stored as | notes |
|---|---|---|
| f32 | 4 bytes | |
| f16 | 2 bytes | needs the `shader-f16` device feature |
| q8 | int8, four per u32, plus one f32 scale per 64 values along K | 1.0625 bytes per weight; no optional feature needed |

In a q8 build, 1-D tensors (norm weights) and the tensors named above stay f32;
each manifest entry carries its own dtype. The matmul, embedding gather and
rmsnorm kernels are the only ones that read weights, and each is built from one
source template per dtype.

## Kernels

All in `web/model/kernels.js`, each documented there with its bindings and
dispatch shape:

| kernel | does |
|---|---|
| `matmul` (GEMV, rows, tiled; f32/f16/q8) | `Y = X @ W^T`; GEMV for one row, the batched GEMV for 2 to 32 rows (q8 past 8: X staged in workgroup memory), 16×16 tiles otherwise |
| `gather_embed_nopos` | `out[t] = wte[ids[t]]` |
| `rmsnorm` | one workgroup per row |
| `reshape_heads`, `merge_heads` | row-major ↔ head-major |
| `store_kv` | projection → int8 paged cache, K rotated first |
| `rope` | in-place rotation of Q from the CPU table |
| `qk_scores_gqa`, `causal_softmax`, `attn_av_gqa` | attention |
| `silu_mul`, `sigmoid_mul`, `add_inplace` | elementwise |
| `causal_conv1d`, `update_conv_history` | Gated DeltaNet convolution |
| `l2norm_heads`, `gated_delta_net_step` | Gated DeltaNet recurrence |
| `copy_f32` | a buffer copy inside a compute pass (speculative snapshots, the MTP head's inputs) |
| `argmax_rows` | each row's argmax as an f32 token id (the MTP draft chain) |
| `topk_stage` | per-row top M (value, index), bitonic sort of 1024 per workgroup, staged (GPU sampling) |

## Verification

- `web/dev/kernels.test.js` checks every kernel against an independent CPU
  reference, with dimensions that are not multiples of the workgroup sizes. Every peer runs it before
  serving.
- `web/dev/verify.js` diffs a whole forward pass against a HuggingFace oracle
  dump in dependency order: embedding, block 0's sub-steps (norm, q/k/v, RoPE,
  attention, MLP), every block, final norm, logits, argmax, and a 20-step greedy
  decode that must match token for token.
- `web/dev/verify_qwen_spike.js` and `verify_qwen_dispatch.js` check one layer of
  each Qwen kind against `oracle/qwen3.8-27b-spike/`.

## Gotchas

- RoPE is split-half, applied to Q and K after projection and before K is
  cached. V is never rotated.
- GQA divides only the K/V head index by `group`.
- SwiGLU needs both the gate and up projections live at once.
- Qwen's full-attention `q_proj` interleaves query and gate rows per head;
  the converter splits them into `q_proj` and `attn_gate`.
- Qwen's full-attention layers have `n_heads * head_dim` (6144) ≠ `hidden`
  (5120), so `o_proj` is rectangular.
