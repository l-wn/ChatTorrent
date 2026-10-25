// The forward pass for full-attention (Llama-style) and Gated DeltaNet blocks,
// built from kernels.js. The math is in docs/model.md. Weight loading is
// separate: everything here takes a weights object of GPUBuffers.
//
// Every buffer is allocated by the caller and reused across steps.
// makeScratch() and makeCache() are called once, and forward() allocates
// nothing.

import {
    F32, makeStorage, makeReadback, uploadF32, uploadU32,
    matmulNobias, gatherEmbedNoPos, rmsnorm, reshapeHeads, storeKV, rope,
    qkScoresGQA, causalSoftmax, attnAVGQA, mergeHeads, addInplace, siluMul,
    sigmoidMul, causalConv1d, updateConvHistory, l2NormalizeHeads, gatedDeltaNetStep,
    copyF32, argmaxRows, roundF16,
} from './kernels.js';
import { recurrentStateBytes, replayLayerBytes, kvRowBytes, kvQB, SPEC_COLS } from '../split.js';

// --- Shapes ---

// Checks the invariants the kernels assume: the head dims tile the hidden dim,
// and `group` is the GQA sharing factor. Returns a list of problems.
export function checkDims(d) {
    const problems = [];
    // Hybrid full-attention layers have n_heads*head_dim != hidden
    // (Qwen3.8-27B: 24*256 = 6144 vs 5120). o_proj maps qDim back to hidden.
    if (!d.hybrid && d.n_heads * d.head_dim !== d.hidden)
        problems.push(`n_heads*head_dim (${d.n_heads * d.head_dim}) != hidden (${d.hidden})`);
    if (d.hybrid && d.linear.nV % d.linear.nK !== 0)
        problems.push(`linear.nV (${d.linear.nV}) not divisible by linear.nK (${d.linear.nK})`);
    if (d.n_heads % d.n_kv_heads !== 0)
        problems.push(`n_heads (${d.n_heads}) not divisible by n_kv_heads (${d.n_kv_heads})`);
    if (d.n_heads / d.n_kv_heads !== d.group)
        problems.push(`group (${d.group}) != n_heads/n_kv_heads (${d.n_heads / d.n_kv_heads})`);
    if (d.head_dim % 2 !== 0)
        problems.push(`head_dim (${d.head_dim}) must be even for RoPE's split-half pairing`);
    // The int8 KV cache packs four values to a word, in blocks of kvQB().
    if (d.head_dim % 4 !== 0 || d.head_dim % kvQB(d) !== 0)
        problems.push(`head_dim (${d.head_dim}) must be a multiple of 4 and of ${kvQB(d)} for the KV cache`);
    return problems;
}

// Gated DeltaNet's per-session state for one linear_attention layer. `S` is
// the recurrence's state (fixed size, independent of context length), and
// `hist{Q,K,V}` are causal_conv1d's history rings. WebGPU zero-fills new
// buffers, so a new state starts at zero. stage.js clears a reused one.
export function makeRecurrentState(device, d) {
    const { nV, Dk, Dv, K4, keyDim, valueDim } = d.linear;
    const Kh = K4 - 1;
    return {
        S:     makeStorage(device, nV * Dk * Dv * F32),
        histQ: makeStorage(device, Kh * keyDim * F32),
        histK: makeStorage(device, Kh * keyDim * F32),
        histV: makeStorage(device, Kh * valueDim * F32),
    };
}

// Defined in split.js, because blockBytes() charges for it.
export { recurrentStateBytes, replayLayerBytes };

// One linear_attention layer's replay data for a speculating session. A
// verify job of n <= cols tokens records:
//   S, hist{Q,K,V}   the recurrent state and conv rings before the job
//   lin{Q,K,V,A,B}   the job's per-token projections (pre-conv q/k/v, a, b)
// A rejection keeping k < n tokens puts the state back and re-runs the same
// kernels over the first k rows (linearBlock), so the restored state has the
// bits a k-token job would have left.
export function makeReplayLayer(device, d, cols = SPEC_COLS) {
    const { nV, Dk, Dv, K4, keyDim, valueDim } = d.linear;
    const Kh = K4 - 1;
    return {
        cols,
        S:     makeStorage(device, nV * Dk * Dv * F32),
        histQ: makeStorage(device, Kh * keyDim * F32),
        histK: makeStorage(device, Kh * keyDim * F32),
        histV: makeStorage(device, Kh * valueDim * F32),
        linQ:  makeStorage(device, cols * keyDim * F32),
        linK:  makeStorage(device, cols * keyDim * F32),
        linV:  makeStorage(device, cols * valueDim * F32),
        linA:  makeStorage(device, cols * nV * F32),
        linB:  makeStorage(device, cols * nV * F32),
    };
}

// KV cache: one [n_kv, N_max] buffer of int8 rows per layer for K and one for
// V (the row format is kernels.js's store_kv).
// On a hybrid checkpoint, a linear_attention layer has a recurrent-state entry
// instead (K[i]=V[i]=null there, and recurrent[i]=null at every full_attention
// layer). `n_layer` defaults to the whole model. A peer holding part of the
// stack passes its own layer count: block() indexes the cache with the same
// local index it uses for w.layers.
//
// `layerLo` is the global index of local layer 0. Caches are allocated before
// any block's weights are loaded, so layer kinds come from `d.layer_types` by
// global index, not from the weights' `.kind`.
export function makeCache(device, d, N_max, n_layer = d.n_layer, layerLo = 0) {
    const bytes = kvRowBytes(d) * N_max;
    const K = [], V = [], recurrent = [];
    let total = 0;
    for (let i = 0; i < n_layer; i++) {
        const kind = d.layer_types?.[layerLo + i] ?? 'full_attention';
        if (kind === 'linear_attention') {
            K.push(null);
            V.push(null);
            recurrent.push(makeRecurrentState(device, d));
            total += recurrentStateBytes(d);
        } else {
            K.push(makeStorage(device, bytes));
            V.push(makeStorage(device, bytes));
            recurrent.push(null);
            // makeStorage allocates at least 16 bytes, which a one-token
            // cache of small heads can be under.
            total += Math.max(bytes, 16) * 2;
        }
    }
    return { N_max, K, V, recurrent, bytes: total };
}

// --- Paged KV ---
//
// makeCache() reserves a contiguous `[n_kv, N_max]` run of rows per layer per
// session. The pool allocates by tokens in use: one buffer per layer, divided
// into fixed-size pages, plus a per-session table mapping logical position to
// physical page. Smaller pages waste less at the tail of each session and cost
// more `pages` reads in attn_av_gqa.
//
// Only full_attention layers are paged. A linear_attention layer's recurrent
// state has no token axis, so it is a fixed cost per session and is stored on
// the session frame.

// `nPages` pages of `pageTokens` each, per full_attention layer in [layerLo,
// layerLo + n_layer). `free` is a stack of free physical page indices.
export function makePagePool(device, d, { pageTokens, nPages, n_layer = d.n_layer,
                                          layerLo = 0 } = {}) {
    const perPage = kvRowBytes(d) * pageTokens;
    const K = [], V = [];
    let bytes = 0;
    for (let i = 0; i < n_layer; i++) {
        const kind = d.layer_types?.[layerLo + i] ?? 'full_attention';
        if (kind === 'linear_attention') { K.push(null); V.push(null); continue; }
        K.push(makeStorage(device, perPage * nPages));
        V.push(makeStorage(device, perPage * nPages));
        bytes += perPage * nPages * 2;
    }
    const free = [];
    for (let i = nPages - 1; i >= 0; i--) free.push(i);   // pops lowest index first
    return { pageTokens, nPages, K, V, free, bytes };
}

// One session's view of the pool. It has the fields of a makeCache() result
// (`K`/`V`/`recurrent`/`N_max`) plus `pages` and `pageTokens`. `K`/`V` are the
// pool's shared buffers, and `pages` limits a session to the pages it owns.
//
// Frames are preallocated and reused, so an arriving job causes no GPU
// allocation.
export function makeSessionFrame(device, d, pool, maxPages, n_layer = d.n_layer,
                                 layerLo = 0) {
    const recurrent = [];
    let bytes = 0;
    for (let i = 0; i < n_layer; i++) {
        const kind = d.layer_types?.[layerLo + i] ?? 'full_attention';
        if (kind !== 'linear_attention') { recurrent.push(null); continue; }
        recurrent.push(makeRecurrentState(device, d));
        bytes += recurrentStateBytes(d);
    }
    return {
        N_max: maxPages * pool.pageTokens,
        pageTokens: pool.pageTokens,
        K: pool.K,
        V: pool.V,
        recurrent,
        // The table the kernels bind: logical page -> physical page. Sized for
        // the largest context a session can reach. One entry is written each
        // time a page is taken.
        pages: uploadU32(device, new Uint32Array(maxPages)),
        // Physical pages this frame holds, in logical order. The JS copy of
        // the table, used to return the pages.
        owned: [],
        maxPages,
        // Bytes this frame allocates itself: the recurrent state and the page
        // table. The pages belong to the pool.
        bytes: bytes + Math.max(maxPages * 4, 16),
    };
}

// Ensure this frame owns enough pages for `tokens` logical positions, taking
// more from the pool as needed. Returns false when the pool is empty; the
// caller then evicts or refuses.
//
// Table entries are written with queue.writeBuffer as pages are taken, so they
// are applied before any later dispatch reads them.
export function ensurePages(device, pool, frame, tokens) {
    const need = Math.ceil(tokens / pool.pageTokens);
    if (need > frame.maxPages) return false;
    while (frame.owned.length < need) {
        const phys = pool.free.pop();
        if (phys === undefined) return false;
        const logical = frame.owned.length;
        frame.owned.push(phys);
        device.queue.writeBuffer(frame.pages, logical * 4, new Uint32Array([phys]));
    }
    return true;
}

// Return every page to the pool. The table is not cleared: entries are only
// read for positions the session has written, and the next session to use the
// frame overwrites each entry before reading it.
export function freePages(pool, frame) {
    const n = frame.owned.length;
    while (frame.owned.length) pool.free.push(frame.owned.pop());
    return n;
}

const newStorage = makeStorage, newReadback = makeReadback;

// The bytes makeScratch() would allocate, computed by running it on a stub
// device that allocates nothing, so the two always agree.
export const scratchBytes = (d, T_max, N_max) =>
    makeScratch({ createBuffer: () => ({}) }, d, T_max, N_max).bytes;

// The most bytes of attention scores one dispatch writes. fullBlock runs
// attention over the job's rows in chunks of attnRows(), so the scores buffer
// does not grow with T_max times the context window.
const SCORES_BYTES = 32 * 2 ** 20;
export const attnRows = (d, T_max, N_max) =>
    Math.max(1, Math.min(T_max, Math.floor(SCORES_BYTES / (d.n_heads * N_max * F32))));

// Intermediate buffers, sized for the widest step (T_max rows). Data is packed
// at the current T, not T_max, so every kernel takes the live T_new/T_act.
// T_max only sets the allocation size.
export function makeScratch(device, d, T_max, N_max) {
    const rows = attnRows(d, T_max, N_max);
    const H = d.hidden, D = d.head_dim, nH = d.n_heads;
    const kvDim = d.n_kv_heads * D;
    // n_heads*head_dim. Equal to H except on hybrid full-attention layers, so
    // every merged-heads buffer is sized from this, not H.
    const qDim = nH * D;
    // Summed as each buffer is allocated, so `bytes` matches the allocations.
    let bytes = 0;
    const size = (n) => { bytes += Math.max(n, 16); return n; };
    const makeStorage = (dev, n) => newStorage(dev, size(n));
    const makeReadback = (dev, n) => newReadback(dev, size(n));
    const s = {
        x:       makeStorage(device, T_max * H * F32),          // residual stream [T, H]
        h:       makeStorage(device, T_max * H * F32),          // normed input    [T, H]
        q:       makeStorage(device, T_max * qDim * F32),       // [T, n_heads*D]
        k:       makeStorage(device, T_max * kvDim * F32),      // [T, n_kv*D]
        v:       makeStorage(device, T_max * kvDim * F32),      // [T, n_kv*D]
        qh:      makeStorage(device, nH * T_max * D * F32),     // [n_heads, T, D]
        scores:  makeStorage(device, nH * rows * N_max * F32),  // [n_heads, rows, T_act]
        ctx:     makeStorage(device, nH * T_max * D * F32),     // [n_heads, T, D]
        merged:  makeStorage(device, T_max * qDim * F32),       // [T, qDim]
        // Scratch for attn_gate/q_norm/k_norm, used only by blocks that have
        // those tensors. Separate from q/k/merged so no kernel reads and
        // writes the same buffer.
        gateRaw: makeStorage(device, T_max * qDim * F32),
        gated:   makeStorage(device, T_max * qDim * F32),
        qNormed: makeStorage(device, T_max * qDim * F32),
        kNormed: makeStorage(device, T_max * kvDim * F32),
        attn:    makeStorage(device, T_max * H * F32),
        gate:    makeStorage(device, T_max * d.ffn * F32),
        up:      makeStorage(device, T_max * d.ffn * F32),
        act:     makeStorage(device, T_max * d.ffn * F32),
        down:    makeStorage(device, T_max * H * F32),
        logits:  makeStorage(device, T_max * d.vocab * F32),    // [T, vocab]

        // Token ids for the embedding gather, written each step. A new buffer
        // per token would leak and would miss the bind-group cache, which keys
        // on buffer identity.
        ids:    makeStorage(device, T_max * F32),

        // One mappable buffer for each step's readback: one logits row or T
        // rows of hidden state, whichever is larger.
        readback: makeReadback(device, Math.max(d.vocab, T_max * H) * F32),
    };

    // Gated DeltaNet scratch, shared by every linear_attention layer.
    // Allocated only when `d.linear` is set.
    if (d.linear) {
        const { keyDim, valueDim, nV, Dk, Dv } = d.linear;
        Object.assign(s, {
            linQ: makeStorage(device, T_max * keyDim * F32),
            linK: makeStorage(device, T_max * keyDim * F32),
            linV: makeStorage(device, T_max * valueDim * F32),
            linZ: makeStorage(device, T_max * valueDim * F32),
            linA: makeStorage(device, T_max * nV * F32),
            linB: makeStorage(device, T_max * nV * F32),
            convQ: makeStorage(device, T_max * keyDim * F32),
            convK: makeStorage(device, T_max * keyDim * F32),
            convV: makeStorage(device, T_max * valueDim * F32),
            qL2: makeStorage(device, T_max * nV * Dk * F32),
            kL2: makeStorage(device, T_max * nV * Dk * F32),
            coreOut: makeStorage(device, T_max * nV * Dv * F32),
            gatedNorm: makeStorage(device, T_max * nV * Dv * F32),
            gatedFinal: makeStorage(device, T_max * nV * Dv * F32),
            linAttnOut: makeStorage(device, T_max * H * F32),
        });
    }

    s.attnRows = rows;
    s.bytes = bytes;
    return s;
}

// --- Forward pass ---

// The SwiGLU MLP. The same in every block, for both attention kinds.
function ffnBlock(device, pass, d, L, s, T_new) {
    const H = d.hidden;
    rmsnorm(device, pass, s.x, L.post_attention_layernorm, s.h, T_new, H, d.rms_eps);

    matmulNobias(device, pass, s.h, L.gate_proj, s.gate, T_new, H, d.ffn);
    matmulNobias(device, pass, s.h, L.up_proj,   s.up,   T_new, H, d.ffn);
    siluMul(device, pass, s.gate, s.up, s.act, T_new * d.ffn);
    matmulNobias(device, pass, s.act, L.down_proj, s.down, T_new, d.ffn, H);
    addInplace(device, pass, s.x, s.down, T_new * H);
}

// One pre-norm transformer block: attention with GQA and a KV cache, then
// SwiGLU. Updates s.x in place (both residual adds) and extends the cache by
// T_new rows.
//
// Also runs Qwen3.8-27B's hybrid full-attention layers, when the loaded block
// has the extra tensors (see loader.js's fullAttnBlockShapes): an attn_gate
// sigmoid gate on the merged attention output before o_proj, q_norm/k_norm on
// Q/K after projection, and partial rotary via d.rot_dim. o_proj's input width
// is qDim (n_heads*head_dim), which is larger than H on those layers.
export function fullBlock(device, pass, d, w, layer, cache, s, T_new, T_past) {
    const H = d.hidden, D = d.head_dim, nH = d.n_heads, nKV = d.n_kv_heads;
    const kvDim = nKV * D, qDim = nH * D;
    const T_act = T_past + T_new;
    const L = w.layers[layer];
    const K = cache.K[layer], V = cache.V[layer];
    // A contiguous cache is the one-page case: `pageTokens` is its whole
    // window and `pages` is null (see scatter_kv in kernels.js).
    const P = cache.pageTokens ?? cache.N_max;
    const pages = cache.pages ?? null;

    // --- attention ---
    rmsnorm(device, pass, s.x, L.input_layernorm, s.h, T_new, H, d.rms_eps);

    matmulNobias(device, pass, s.h, L.q_proj, s.q, T_new, H, qDim);
    matmulNobias(device, pass, s.h, L.k_proj, s.k, T_new, H, kvDim);
    matmulNobias(device, pass, s.h, L.v_proj, s.v, T_new, H, kvDim);
    if (L.attn_gate) matmulNobias(device, pass, s.h, L.attn_gate, s.gateRaw, T_new, H, qDim);

    // q_norm/k_norm normalize over head_dim only. [T,heads,D] has the same
    // layout as [T*heads,D], so this is the rmsnorm kernel with T*heads rows.
    // The output goes to separate buffers, so no kernel reads and writes the
    // same buffer.
    let qBuf = s.q, kBuf = s.k;
    if (L.q_norm) { rmsnorm(device, pass, s.q, L.q_norm, s.qNormed, T_new * nH,  D, d.rms_eps); qBuf = s.qNormed; }
    if (L.k_norm) { rmsnorm(device, pass, s.k, L.k_norm, s.kNormed, T_new * nKV, D, d.rms_eps); kBuf = s.kNormed; }

    // Q is reshaped head-major. K/V are quantized into the cache at T_past,
    // K rotated first. V is not rotated.
    reshapeHeads(device, pass, qBuf, s.qh, T_new, nH, D);
    storeKV(device, pass, kBuf, K, T_new, T_past, P, nKV, D,
            { theta: d.rope_theta, scaling: d.rope_scaling, rotDim: d.rot_dim ?? D }, pages);
    storeKV(device, pass, s.v, V, T_new, T_past, P, nKV, D, null, pages);

    // RoPE on Q after projection, before the scores. Q is dense: one page of
    // T_new rows starting at 0.
    rope(device, pass, s.qh, nH, T_new, D, T_past, T_new, 0, d.rope_theta, d.rope_scaling, d.rot_dim);

    // Attention in chunks of s.attnRows query rows, so the scores buffer is
    // bounded. Each output element is computed the same way in any chunking.
    for (let r0 = 0; r0 < T_new; r0 += s.attnRows) {
        const R = Math.min(s.attnRows, T_new - r0);
        qkScoresGQA(device, pass, s.qh, K, s.scores, R, T_act, P, D, nH, d.group, pages, r0, T_new);
        causalSoftmax(device, pass, s.scores, R, T_act, T_past + r0, nH);
        attnAVGQA(device, pass, s.scores, V, s.ctx, R, T_act, P, D, nH, d.group, pages, r0, T_new);
    }

    mergeHeads(device, pass, s.ctx, s.merged, T_new, nH, D);
    let attnIn = s.merged;
    if (L.attn_gate) { sigmoidMul(device, pass, s.gateRaw, s.merged, s.gated, T_new * qDim); attnIn = s.gated; }

    matmulNobias(device, pass, attnIn, L.o_proj, s.attn, T_new, qDim, H);
    addInplace(device, pass, s.x, s.attn, T_new * H);

    ffnBlock(device, pass, d, L, s, T_new);
}

// One Gated DeltaNet (linear_attention) block, Qwen3.8-27B's other layer kind.
//
// Takes no T_past: the recurrent state (`rc.S`) and the causal conv's history
// (`rc.hist{Q,K,V}`) hold the effect of all earlier jobs.
// The recurrence and conv-ring update for `T` rows of projections, shared by
// the forward pass and a speculative restore so both run the same kernels in
// the same order. Writes the core output to s.coreOut.
function linearCore(device, pass, d, L, rc, s, lin, T) {
    const { nK, nV, Dk, Dv, K4, keyDim, valueDim, group } = d.linear;

    // Rows before this job come from the history ring, so T_past is 0 here.
    causalConv1d(device, pass, lin.q, L.conv1d_q, s.convQ, T, keyDim,   K4, 0, rc.histQ);
    causalConv1d(device, pass, lin.k, L.conv1d_k, s.convK, T, keyDim,   K4, 0, rc.histK);
    causalConv1d(device, pass, lin.v, L.conv1d_v, s.convV, T, valueDim, K4, 0, rc.histV);

    // `group` applies Gated DeltaNet's repeat_interleave (query/key have fewer
    // heads than value/gate) in the same pass.
    l2NormalizeHeads(device, pass, s.convQ, s.qL2, T, nK, Dk, group);
    l2NormalizeHeads(device, pass, s.convK, s.kL2, T, nK, Dk, group);

    // One dispatch runs the recurrence through all T tokens on the persistent
    // state `rc.S`, keeping each head's state in registers for the whole chunk
    // (see gated_delta_net_step in kernels.js). Decode is the T = 1 case.
    gatedDeltaNetStep(device, pass, s.qL2, s.kL2, s.convV, lin.a, lin.b, L.gate_params,
        rc.S, s.coreOut, 0, nV, Dk, Dv, T);

    // Advance the ring with the raw (pre-conv) projections. This must come
    // after causalConv1d above, which reads the ring's previous contents.
    updateConvHistory(device, pass, rc.histQ, lin.q, T, keyDim,   K4);
    updateConvHistory(device, pass, rc.histK, lin.k, T, keyDim,   K4);
    updateConvHistory(device, pass, rc.histV, lin.v, T, valueDim, K4);
}

// `spec` (optional) is the session's speculative state at this Runner:
//   replay[layer]  its replay data (makeReplayLayer), or none
//   keep           not null: the last verify job left too many tokens; put
//                  the state back and re-run its first `keep` rows (0: none)
//   snap           this job is a verify: record what a later `keep` needs
function linearBlock(device, pass, d, w, layer, cache, s, T_new, spec = null) {
    const H = d.hidden;
    const { nV, Dv, keyDim, valueDim } = d.linear;
    const L = w.layers[layer];
    const rc = cache.recurrent[layer];
    const rp = spec?.replay?.[layer] ?? null;

    if (rp && spec.keep != null) {
        copyF32(device, pass, rp.S, rc.S, rc.S.size / F32);
        copyF32(device, pass, rp.histQ, rc.histQ, rc.histQ.size / F32);
        copyF32(device, pass, rp.histK, rc.histK, rc.histK.size / F32);
        copyF32(device, pass, rp.histV, rc.histV, rc.histV.size / F32);
        if (spec.keep > 0) {
            linearCore(device, pass, d, L, rc, s,
                { q: rp.linQ, k: rp.linK, v: rp.linV, a: rp.linA, b: rp.linB }, spec.keep);
        }
    }
    const snap = rp && spec.snap;
    if (snap) {
        copyF32(device, pass, rc.S, rp.S, rc.S.size / F32);
        copyF32(device, pass, rc.histQ, rp.histQ, rc.histQ.size / F32);
        copyF32(device, pass, rc.histK, rp.histK, rc.histK.size / F32);
        copyF32(device, pass, rc.histV, rp.histV, rc.histV.size / F32);
    }
    // A verify job's projections go straight into the replay buffers, which
    // then serve as this job's inputs too.
    const lin = snap
        ? { q: rp.linQ, k: rp.linK, v: rp.linV, a: rp.linA, b: rp.linB }
        : { q: s.linQ, k: s.linK, v: s.linV, a: s.linA, b: s.linB };

    rmsnorm(device, pass, s.x, L.input_layernorm, s.h, T_new, H, d.rms_eps);

    matmulNobias(device, pass, s.h, L.in_proj_q, lin.q, T_new, H, keyDim);
    matmulNobias(device, pass, s.h, L.in_proj_k, lin.k, T_new, H, keyDim);
    matmulNobias(device, pass, s.h, L.in_proj_v, lin.v, T_new, H, valueDim);
    matmulNobias(device, pass, s.h, L.in_proj_z, s.linZ, T_new, H, valueDim);
    matmulNobias(device, pass, s.h, L.in_proj_b, lin.b, T_new, H, nV);
    matmulNobias(device, pass, s.h, L.in_proj_a, lin.a, T_new, H, nV);

    linearCore(device, pass, d, L, rc, s, lin, T_new);


    rmsnorm(device, pass, s.coreOut, L.gated_norm, s.gatedNorm, T_new * nV, Dv, d.rms_eps);
    siluMul(device, pass, s.linZ, s.gatedNorm, s.gatedFinal, T_new * nV * Dv);
    matmulNobias(device, pass, s.gatedFinal, L.out_proj, s.linAttnOut, T_new, valueDim, H);
    addInplace(device, pass, s.x, s.linAttnOut, T_new * H);

    ffnBlock(device, pass, d, L, s, T_new);
}

// --- The MTP draft head (Qwen3.8 multi-token prediction) ---
//
// The head drafts the token after next from the trunk's last hidden state h
// (before the final norm) and the embedding e of the next token:
//
//     x      = fc_e · enorm(e) + fc_h · hnorm(h)
//     x      = one full-attention layer (x)            its own KV, per session
//     logits = head_draft · mtp_norm(x)                 the first draftVocab rows
//
// `m` is loader.js loadMtp()'s weights, `mc` the session's cache view for the
// layer (its K/V pool buffers and the frame's page table), `ms` its scratch
// (makeMtpScratch). Row r pairs hIn[r] (the hidden state of position past + r
// - 1) with ids[r] (the token at past + r) and writes the KV row at past + r.
// The layer's output is left in ms.x.
export function mtpBlock(device, pass, d, m, mc, ms, hIn, ids, T, past) {
    const H = d.hidden;
    gatherEmbedNoPos(device, pass, ids, m.embed_draft, ms.mtpEmb, T, H);
    rmsnorm(device, pass, ms.mtpEmb, m.enorm, ms.mtpEn, T, H, d.rms_eps);
    rmsnorm(device, pass, hIn, m.hnorm, ms.mtpHn, T, H, d.rms_eps);
    matmulNobias(device, pass, ms.mtpEn, m.fc_e, ms.x, T, H, H);
    matmulNobias(device, pass, ms.mtpHn, m.fc_h, ms.mtpFc, T, H, H);
    addInplace(device, pass, ms.x, ms.mtpFc, T * H);
    fullBlock(device, pass, d, { layers: [m] }, 0, mc, ms, T, past);
}

// The draft for each of T rows of ms.x: mtp_norm, the draft head, argmax. The
// ids are written as f32 into `ids` from `idOff`.
export function mtpDraft(device, pass, d, m, ms, ids, T, idOff) {
    const H = d.hidden, V = m.draftVocab;
    rmsnorm(device, pass, ms.x, m.mtp_norm, ms.h, T, H, d.rms_eps);
    matmulNobias(device, pass, ms.h, m.head_draft, ms.logits, T, H, V);
    argmaxRows(device, pass, ms.logits, ids, T, V, idOff);
}

// Chain K drafts after token `ids[0]` at position `pos`, from `h`, the trunk's
// hidden state at pos - 1 ([1, H]). Step k feeds the previous step's output
// and draft back in. The drafts land in ids[1..K]; the layer's KV rows
// pos..pos+K-1 are speculative and are rewritten by the next job's fill.
export function mtpChain(device, pass, d, m, mc, ms, h, ids, pos, K) {
    const H = d.hidden;
    copyF32(device, pass, h, ms.mtpH, H);
    for (let k = 0; k < K; k++) {
        copyF32(device, pass, ids, ms.mtpId, 1, k, 0);
        mtpBlock(device, pass, d, m, mc, ms, ms.mtpH, ms.mtpId, 1, pos + k);
        mtpDraft(device, pass, d, m, ms, ids, 1, k + 1);
        if (k + 1 < K) copyF32(device, pass, ms.x, ms.mtpH, H);
    }
}

// The MTP head's scratch: a block's (for `fullBlock`), with the logits sized
// for the draft vocabulary, plus its own inputs.
export function makeMtpScratch(device, d, draftVocab, T_max, N_max) {
    const s = makeScratch(device, { ...d, linear: undefined, vocab: draftVocab }, T_max, N_max);
    const H = d.hidden;
    const extra = {
        mtpEmb: T_max * H, mtpEn: T_max * H, mtpHn: T_max * H, mtpFc: T_max * H,
        mtpIn: T_max * H, mtpH: H, mtpId: 4, mtpIds: T_max + 8,
    };
    for (const [name, n] of Object.entries(extra)) {
        s[name] = makeStorage(device, n * F32);
        s.bytes += Math.max(n * F32, 16);
    }
    return s;
}

// Dispatches on the block's `kind`, set by loader.js's loadBlock()
// (randomWeights() sets 'full_attention').
// `spec` is the session's speculative state, read by linear_attention blocks
// only: a full-attention block's KV is overwritten by position, so a rejected
// row needs no undo.
//
// The residual stream leaves every block rounded to f16 values (round_f16 in
// kernels.js). Peers hand rows on as f16 (net.js), and are only ever cut
// between blocks, so with the rounding in the same places on every peer a
// reply is the same however the model is split.
export function block(device, pass, d, w, layer, cache, s, T_new, T_past, spec = null) {
    const kind = w.layers[layer].kind ?? 'full_attention';
    if (kind === 'linear_attention') linearBlock(device, pass, d, w, layer, cache, s, T_new, spec);
    else fullBlock(device, pass, d, w, layer, cache, s, T_new, T_past);
    roundF16(device, pass, s.x, T_new * d.hidden);
}

// Records the whole stack on `pass`, an open compute pass. `ids` is an f32
// buffer of T_new token ids (gather_embed_nopos casts them). Returns the
// logits buffer, [T_new, vocab].
//
// T_past = 0 with T_new = prompt length is a prefill. T_past = n with
// T_new = 1 is a decode step. The caller advances T_past by T_new afterwards.
export function forward(device, pass, d, w, cache, s, ids, T_new, T_past) {
    gatherEmbedNoPos(device, pass, ids, w.wte, s.x, T_new, d.hidden);
    roundF16(device, pass, s.x, T_new * d.hidden);          // as block() leaves it

    for (let layer = 0; layer < d.n_layer; layer++) {
        block(device, pass, d, w, layer, cache, s, T_new, T_past);
    }

    rmsnorm(device, pass, s.x, w.norm, s.h, T_new, d.hidden, d.rms_eps);
    // LM head: [vocab, hidden] read as [N, K]. Usually the embedding table
    // (tied). A checkpoint with a separate lm_head.weight has it in w.lm_head,
    // which takes precedence.
    matmulNobias(device, pass, s.h, w.lm_head ?? w.wte, s.logits, T_new, d.hidden, d.vocab);

    return s.logits;
}

// --- Placeholder weights ---

// Random full-attention weights with real shapes, for the smoke check. Names
// match the HF state dict, as the loader's do.
//
// Projections are small uniform values and norm weights are 1.0, which keeps
// the residual stream finite so the smoke check's finiteness test is
// meaningful.
export function randomWeights(device, d, seed = 1234) {
    let st = seed >>> 0;
    const rand = () => { st = (Math.imul(st, 1664525) + 1013904223) >>> 0; return st / 4294967296; };
    const mat = (rows, cols, scale = 0.05) => {
        const a = new Float32Array(rows * cols);
        for (let i = 0; i < a.length; i++) a[i] = (rand() * 2 - 1) * scale;
        return uploadF32(device, a);
    };
    const ones = (n) => uploadF32(device, new Float32Array(n).fill(1));

    const H = d.hidden, D = d.head_dim;
    const kvDim = d.n_kv_heads * D;

    return {
        // Weight layout is [out, in] everywhere, as exported.
        wte:  mat(d.vocab, H, 0.02),   // model.embed_tokens.weight, tied to lm_head
        norm: ones(H),                 // model.norm.weight
        layers: Array.from({ length: d.n_layer }, () => ({
            input_layernorm:          ones(H),
            q_proj:                   mat(d.n_heads * D, H),
            k_proj:                   mat(kvDim, H),
            v_proj:                   mat(kvDim, H),
            o_proj:                   mat(H, H),
            post_attention_layernorm: ones(H),
            gate_proj:                mat(d.ffn, H),
            up_proj:                  mat(d.ffn, H),
            down_proj:                mat(H, d.ffn),
            kind: 'full_attention',
        })),
    };
}

// Parameter count and stored byte size of a full-attention config, without
// allocating anything.
export function weightBudget(d) {
    const H = d.hidden, D = d.head_dim, kvDim = d.n_kv_heads * D;
    const perLayer = 2 * H                       // the two RMSNorm weights
        + d.n_heads * D * H + 2 * kvDim * H + H * H   // q, k, v, o
        + 3 * d.ffn * H;                              // gate, up, down
    // The embedding table counts twice when the LM head is not tied to it.
    const embeddings = d.vocab * H * (d.tied_lm_head === false ? 2 : 1);
    const params = embeddings + H + d.n_layer * perLayer;
    // `d.bytes` is the stored size per weight, from loader.js: 4 for f32, 2
    // for f16, 1.0625 for q8 (an int8 plus its share of one scale per 64).
    return { params, bytes: params * (d.bytes ?? F32) };
}
