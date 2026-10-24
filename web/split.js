// Assignment: who holds which positions.
//
// An `L`-block model has `L + 2` positions: the embedding, the blocks, and the
// head. A peer holds a half-open range `[lo, hi)` of them. `depth(x)` is how
// many peers hold position `x`, and `r` is the minimum depth.
//
// Everything here is pure: no WebGPU, WebRTC, DOM or `location`. The signaling
// server imports it to place peers, and the browsers import the same functions
// to read the roster.
//
// See docs/architecture.md (Positions, Claims, holdings and growth).

// The version of the messages peers exchange. The signaling server refuses a
// page with another, so one mesh never mixes two.
export const PROTOCOL = 2;

// --- The position line ---

// Only these helpers encode the meaning of a position.
export const positions = (L) => L + 2;

export const EMBED = 0;               // a prompt enters the mesh at position 0
export const headPos = (L) => L + 1;

// A claim `[lo, hi)` in positions, converted to `[lo, hi)` block indices.
// loader.js fetches by these, and Runner indexes layers and KV rows by them.
// Empty for a peer holding only the embedding or the head.
export const blockRange = (lo, hi, L) =>
    [Math.max(lo, 1) - 1, Math.max(Math.min(hi, L + 1) - 1, Math.max(lo, 1) - 1)];

// A range of positions as readable text, e.g. "the embedding + blocks 0–5".
export function describeRange(lo, hi, L) {
    const parts = [];
    if (lo === 0) parts.push('the embedding');
    const bLo = Math.max(lo, 1), bHi = Math.min(hi, L + 1);
    if (bHi > bLo) parts.push(bHi - bLo === 1 ? `block ${bLo - 1}` : `blocks ${bLo - 1}–${bHi - 2}`);
    if (hi === positions(L)) parts.push('the head');
    return parts.join(' + ');
}

// --- What a block costs, and the minimum range ---

// Defined here because the minimum range is computed from them, and the server
// and the peers must agree on it.

// Context window for a session. blockBytes() charges KV for it against every
// peer's claim, so a bigger window means fewer blocks per tab. On a hybrid
// checkpoint only the full-attention layers keep KV (see kvShare()).
export const N_MAX = 4096;

// Full-context KV caches a runner's page pool is sized for at the default
// SESSIONS: SLOTS * N_MAX tokens, shared by every session on the runner, not
// reserved per chat. A runner with another session count uses poolSlots().
export const SLOTS = 2;

// Tokens in one KV page (web/model/model.js's makePagePool). A session's KV is
// a list of pages, so a chat's cost follows the context it holds. A larger
// page wastes more at the tail (half a page per session on average); a smaller
// one means more page-table reads in attn_av_gqa.
export const PAGE_TOKENS = 64;

// Concurrent sessions a runner keeps frames for. A frame is a page table plus,
// on a hybrid checkpoint, the recurrent state, which has no token axis and is
// not paged. Independent of SLOTS: the sessions share the pool, and an
// arriving session is refused `busy` only when their pages together exceed it.
//
// Recurrent state is per session, so blockBytes() charges it SESSIONS times:
// ~3.3 MB per linear_attention layer per session on Qwen3.8-27B.
//
// The default. A mesh's creator can choose another number, which the welcome
// brings (signal-server.js's MESH_SESSIONS for the global mesh).
export const SESSIONS = 8;

// The most sessions a page will size a runner for, whatever the welcome says.
export const MAX_SESSIONS = 64;

// The KV pool keeps one full context per SESSIONS_PER_SLOT sessions, so a
// mesh that keeps more chats also has room for their history. poolSlots(SESSIONS)
// is SLOTS.
export const SESSIONS_PER_SLOT = SESSIONS / SLOTS;
export const poolSlots = (sessions) => Math.max(1, Math.ceil(sessions / SESSIONS_PER_SLOT));

// The sessions a runner is sized for: the caller's number (the dev harness's
// ?sessions=), else the mesh's from the welcome, else SESSIONS, clamped to
// [1, MAX_SESSIONS]. Returns the pool's slots with it.
export function runnerSessions(asked, mesh) {
    const n = Math.floor(asked ?? mesh ?? SESSIONS);
    const sessions = Number.isFinite(n) && n >= 1 ? Math.min(n, MAX_SESSIONS) : SESSIONS;
    return { sessions, slots: poolSlots(sessions) };
}

// How long a session must be unused before its frame or pages may be taken. A
// chat mid-reply is never evicted; the arriving session waits (WAIT_MS). An
// abandoned chat, including one whose tab closed without a `free`, is
// reclaimed after this long.
export const IDLE_MS = 2000;

// How long a new session waits at a peer with no frame for it before it is
// refused with `busy`. Waiting is first come, first served. If sessions were
// refused immediately and retried on their own timers, one could lose every
// time. Much shorter than the enqueuer's five-minute job timeout. 0 refuses
// immediately.
export const WAIT_MS = 30000;

// Stored bytes per weight, by manifest dtype. Defined here so the signaling
// server prices a block the same way the browsers do. q8 is int8 plus one f32
// scale per 64 values. Its 1-D norms stay f32, which is negligible.
export const DTYPE_BYTES = { f32: 4, f16: 2, q8: 1.0625 };

// Dimensions under both spellings: the manifest's (`n_head`, `n_kv_head`) and
// loader.js's (`n_heads`, `n_kv_heads`). `bytes` is the size of one stored
// weight. f16 or q8 shrinks a block's weights but not its KV, which is always
// f32 because the kernels read and write f32.
export const costDims = (m) => ({
    hidden:     m.hidden,
    n_layer:    m.n_layer,
    n_heads:    m.n_heads ?? m.n_head,
    n_kv_heads: m.n_kv_heads ?? m.n_kv_head,
    head_dim:   m.head_dim,
    ffn:        m.ffn,
    bytes:      m.bytes ?? DTYPE_BYTES[m.dtype ?? 'f32'] ?? 4,
    // Kept so this object and loader.js's dims give the same block cost.
    // Absent on non-hybrid checkpoints, where kvShare() is 1.
    layer_types: m.layer_types,
    // Gated DeltaNet's widths, for recurrentStateBytes(): loader.js's `linear`
    // or the manifest's `linear_*` fields. Absent when not hybrid.
    linear: m.linear ?? (m.linear_num_value_heads ? {
        nV:       m.linear_num_value_heads,
        Dk:       m.linear_key_head_dim,
        Dv:       m.linear_value_head_dim,
        K4:       m.linear_conv_kernel_dim,
        keyDim:   m.linear_num_key_heads * m.linear_key_head_dim,
        valueDim: m.linear_num_value_heads * m.linear_value_head_dim,
    } : undefined),
    // For endBytes(): the embedding and LM head tables.
    vocab:        m.vocab,
    tied_lm_head: m.tied_lm_head,
    // Actual file sizes, when the manifest lists its tensors.
    stored: m.stored ?? storedBytes(m),
});

// Sums tensor entries, including a q8 tensor's scales.
const sumEntries = (entries) =>
    (entries ?? []).reduce((n, e) => n + e.size + (e.scales_size ?? 0), 0);

// Stored bytes of one file, from the manifest's tensor entries. Undefined when
// the manifest does not list the file, so a caller can tell "not listed" from
// "empty". loader.js uses it to pick between a cacheable whole-file GET and the
// resumable ranged path.
export function fileBytes(m, fn) {
    const entries = m?.files?.[fn];
    return entries ? sumEntries(entries) : undefined;
}

// Stored bytes of every block and of the embedding and head tables, summed
// from the manifest's tensor entries (including a q8 tensor's scales). This is
// what loader.js uploads, so a claim is charged its real size. Undefined for a
// manifest with no file listing; the formulas below are used then.
export function storedBytes(m) {
    if (!m?.files || !(m.n_layer > 0)) return undefined;
    const tensor = (fn, name) =>
        sumEntries((m.files[fn] ?? []).filter((e) => e.name === name));
    const blocks = Array.from({ length: m.n_layer },
        (_, i) => fileBytes(m, `block_${String(i).padStart(2, '0')}.bin`));
    if (blocks.some((b) => !b)) return undefined;
    return {
        blocks,
        wte:     tensor('client.bin', 'wte'),
        norm:    tensor('client.bin', 'norm'),
        // 0, not undefined: a tied checkpoint has no lm_head.bin.
        lm_head: fileBytes(m, 'lm_head.bin') ?? 0,
        // The MTP draft head's bundle, 0 without one.
        mtp: m.mtp ? fileBytes(m, m.mtp.file) ?? 0 : 0,
    };
}

// What one held block's weights cost on the GPU, by kind:
//
//   full_attention
//     input_layernorm   H                    q_proj   qDim × H
//     k_proj, v_proj    kv × H each          o_proj   H × qDim
//     post_attn_norm    H                    gate, up, down   ffn × H each
//     attn_gate         qDim × H             (hybrid checkpoints only)
//
//   linear_attention (Gated DeltaNet)
//     in_proj_q, _k     keyDim × H each      in_proj_v, _z   valueDim × H each
//     in_proj_a, _b     nV × H each          out_proj        H × valueDim
//     conv1d_q/k/v      (2·keyDim + valueDim) × K4           gate, up, down
//
// `qDim` is n_heads*head_dim, which equals H except on a hybrid checkpoint's
// full-attention layers (Qwen3.8-27B: 6144 vs 5120).
//
// A fallback: blockWeightBytes() prefers the file sizes from storedBytes()
// when the manifest lists them. The embedding table is a position of its own
// (endBytes()) and is not counted here.
function formulaWeightBytes(d, kind) {
    const H = d.hidden, kv = d.n_kv_heads * d.head_dim, qDim = d.n_heads * d.head_dim;
    const ffn = d.ffn * H * 3 + H * 2;
    const hybrid = new Set(d.layer_types ?? []).size > 1;
    if (kind === 'linear_attention' && d.linear) {
        const { nV, K4, keyDim, valueDim } = d.linear;
        return (keyDim * H * 2 + valueDim * H * 2 + nV * H * 2 + H * valueDim
            + (keyDim * 2 + valueDim) * K4 + ffn) * (d.bytes ?? 4);
    }
    return (qDim * H * (hybrid ? 3 : 2) + kv * H * 2 + ffn) * (d.bytes ?? 4);
}

// One block's weights. `index` selects a block, `kind` a kind of block. With
// neither, returns the mean over the model, used to price a claim before its
// blocks are known. `d.bytes` applies only to the weights, so f16 saves less
// on wide-KV models than the parameter count suggests.
export function blockWeightBytes(d, { index, kind } = {}) {
    const kinds = d.layer_types;
    const kindOf = (i) => kinds?.[i] ?? 'full_attention';
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const stored = d.stored?.blocks;
    if (stored) {
        if (index != null) return stored[index];
        const of = kind ? stored.filter((_, i) => kindOf(i) === kind) : stored;
        if (of.length) return mean(of);
    }
    if (index != null) return formulaWeightBytes(d, kindOf(index));
    if (kind || !kinds?.length) return formulaWeightBytes(d, kind ?? 'full_attention');
    return mean(kinds.map((k) => formulaWeightBytes(d, k)));
}

// The fraction of blocks that keep a KV cache. 1 unless the checkpoint is
// hybrid: Qwen3.8-27B runs Gated DeltaNet on 48 of its 64 layers, which hold a
// fixed-size recurrent state instead. Averaged over the model, because a claim
// is priced before its blocks are known.
export const kvShare = (d) => {
    const kinds = d.layer_types;
    if (!kinds?.length) return 1;
    return kinds.filter((k) => k === 'full_attention').length / kinds.length;
};

// One session's recurrent state on one linear_attention layer (model.js's
// makeRecurrentState): the DeltaNet state plus causal_conv1d's three history
// rings, all f32. 0 on a checkpoint with no linear layers.
export function recurrentStateBytes(d) {
    if (!d.linear) return 0;
    const { nV, Dk, Dv, K4, keyDim, valueDim } = d.linear;
    const Kh = K4 - 1;
    return (nV * Dk * Dv + Kh * keyDim * 2 + Kh * valueDim) * 4;
}

// The widest speculative verify job, in tokens: the last sampled token plus
// up to SPEC_COLS - 1 drafts. Matches kernels.js's GEMV_ROWS_MAX, so every
// verify row goes through the batched GEMV and gets plain decoding's bits.
export const SPEC_COLS = 8;

// One linear_attention block's replay data for one speculating session
// (model.js makeReplayLayer): the recurrent state and conv rings as they were
// before a verify job, and that job's per-token inputs, enough to re-run its
// first k tokens after a rejection.
export function replayLayerBytes(d, cols = SPEC_COLS) {
    if (!d.linear) return 0;
    const { nV, keyDim, valueDim } = d.linear;
    return recurrentStateBytes(d) + cols * (keyDim * 2 + valueDim + nV * 2) * 4;
}

// Everything one held block costs on the GPU: its weights, plus its share of
// the KV pool (full_attention) or a recurrent state in every session frame
// (linear_attention). `index` selects a block and is exact, `kind` a kind of
// block. With neither, returns the mean weighted by kvShare().
//
// A linear_attention block also reserves `spec` speculative replay slots
// (stage.js, one per session speculating through it at once), `slots` by
// default, as the Runner does.
export function blockBytes(d, { nMax = N_MAX, slots = SLOTS, sessions = SESSIONS, spec = slots,
                                kind, index } = {}) {
    const kv = d.n_kv_heads * d.head_dim;
    if (index != null) kind = d.layer_types?.[index] ?? 'full_attention';
    const share = kind ? (kind === 'linear_attention' ? 0 : 1) : kvShare(d);
    return blockWeightBytes(d, { index, kind }) + kv * nMax * 4 * 2 * slots * share
        + (recurrentStateBytes(d) * sessions + replayLayerBytes(d) * spec) * (1 - share);
}

// What holding the MTP draft head costs, apart from its scratch: its file,
// and its layer's K and V laid out as one more KV pool layer.
export function mtpBytes(d, { nMax = N_MAX, slots = SLOTS } = {}) {
    if (!d.mtp) return 0;
    return (d.stored?.mtp ?? 0) + d.n_kv_heads * d.head_dim * nMax * 4 * 2 * slots;
}

// What holding an end position costs. The embedding is `wte` (client.bin,
// which also holds the final norm). The head is `lm_head`, or `wte` on a tied
// checkpoint, where a peer holding both ends stores it once.
export function endBytes(d, end) {
    const table = (d.vocab ?? 0) * d.hidden * (d.bytes ?? 4);
    const s = d.stored;
    const client = s ? s.wte + s.norm : table;
    if (end === 'head' && d.tied_lm_head === false) return s ? s.lm_head + s.norm : table;
    return client;
}

// The smallest range the server hands out, in bytes of weights. In bytes so
// it works across model sizes. Weights only: the minimum balances fetch cost
// and extra hops against depth, and KV is local VRAM, which the engine's
// capacity budget counts. A default; place() does not depend on it.
export const MIN_BYTES = 135 * 2 ** 20;

// What a tab offers to hold by default. Too low costs depth. Too high makes a
// tab download its whole claim before allocation fails, so this is set low.
export const DEFAULT_BYTES = 1.5 * 2 ** 30;

// How many blocks fit in `bytes`, rounded down. 0 when one block does not fit.
export function budgetBlocks(d, L, bytes = DEFAULT_BYTES, opts) {
    return Math.min(L, Math.floor(bytes / blockBytes(d, opts)));
}

// The minimum range in blocks, rounded up. At least one: if every block of a
// model exceeds MIN_BYTES, refusing single blocks would make it uncoverable.
export function floorBlocks(d, L) {
    return Math.min(L, Math.max(1, Math.ceil(MIN_BYTES / blockWeightBytes(d))));
}

// A holding is anything with numeric `lo`/`hi`, or null `lo` for a peer
// holding nothing. Roster entries have this shape.
export const holding = (p) => p?.lo != null && p.hi > p.lo;

export const holds = (p, x) => holding(p) && p.lo <= x && x < p.hi;

// --- The claim and the holding ---

// A peer serves each position as soon as it has downloaded it, so a roster
// entry carries two ranges:
//
//   [lo, hi)     the claim     reserved, not necessarily downloaded
//   [lo, have)   the holding   downloaded, warmed up, servable
//
// `have` starts at `lo` and only rises.
//
// `served()` returns the entry with its holding as the range. The browsers use
// it for routing, coverage, `r` and the block map, which all concern what can
// run now. The server places on the claims, so two peers downloading at once
// are not both sent to the same hole.
//
// A holding only grows, so a stale roster can only understate it, and a route
// computed from one is still valid.
export const served = (p) => (p?.have == null ? p : { ...p, hi: p.have });

// Whether a peer is still downloading part of its claim. Not used for routing.
// The block map uses it to show a position as loading, not unheld.
export const filling = (p) => holding(p) && (p.have ?? p.hi) < p.hi;

// Sorted, so every browser picks the same lowest id.
export const holders = (peers, x) =>
    peers.filter((p) => holds(p, x)).map((p) => p.id).sort((a, b) => a - b);

// depth[x] for every position, by a difference array and a prefix sum.
// O(n + L).
export function depths(peers, L) {
    const diff = new Int32Array(L + 1);
    for (const p of peers) {
        if (!holding(p)) continue;
        diff[Math.max(0, p.lo)]++;
        diff[Math.min(L, p.hi)]--;
    }
    const d = new Int32Array(L);
    let run = 0;
    for (let x = 0; x < L; x++) { run += diff[x]; d[x] = run; }
    return d;
}

// `r`: 0 means the model cannot run, 1 that any departure at the thinnest
// position breaks it, 2 that any single peer can leave mid-chat.
export function minDepth(peers, L) {
    const d = depths(peers, L);
    let r = Infinity;
    for (let x = 0; x < L; x++) r = Math.min(r, d[x]);
    return L === 0 ? 0 : r;
}

// A path from 0 to L exists exactly when the ranges cover every position.
export const covers = (peers, L) => minDepth(peers, L) >= 1;

// Uncovered runs of positions, for reporting.
export function gaps(peers, L) {
    const d = depths(peers, L);
    const out = [];
    for (let x = 0; x < L; x++) {
        if (d[x]) continue;
        const lo = x;
        while (x < L && !d[x]) x++;
        out.push([lo, x]);
    }
    return out;
}

// Coverage stats for the UI, in one pass over the roster.
export function coverage(peers, L) {
    const d = depths(peers, L);
    let covered = 0, r = L > 0 ? Infinity : 0;
    const holes = [];
    for (let x = 0; x < L; x++) {
        r = Math.min(r, d[x]);
        if (d[x] > 0) { covered++; continue; }
        const lo = x;
        while (x + 1 < L && !d[x + 1]) x++;
        holes.push([lo, x + 1]);
    }
    return { depths: d, covered, total: L, r: r === Infinity ? 0 : r, holes };
}

// How many blocks a peer claiming `cap` gets. Below `floor` it gets none: a
// range that small costs its holder a download and adds two hops per token for
// little depth.
//
// `floor` is a parameter because it depends on the model. The server computes
// it once and sends it in the welcome, so both sides use the same number.
export function size(cap, L, floor) {
    if (!(L > 0)) return 0;
    const f = Math.min(Math.max(1, Math.floor(floor) || 1), L);
    if (!Number.isFinite(cap) || cap < f) return 0;
    return Math.min(Math.floor(cap), L);
}

// --- Placement ---

// Where to put a new range of `n` positions: the window whose sorted depth
// profile is lexicographically lowest. In order of priority, that:
//
//   1. Fills holes. A window with three uncovered positions beats one with
//      two, whatever else is in either.
//   2. Raises `r`. Once nothing is uncovered, the window over the thinnest
//      positions wins.
//
// Sorted profiles are used instead of the sum of depths because a sum would
// trade one uncovered position for three shallow ones, and a network with one
// hole cannot run.
//
// Ties go to the lowest `lo`, so equal peers joining an empty line are placed
// edge to edge from position 0. A one-wide claim is the exception: it goes in
// the middle of the widest run of tied positions, so peers that join close
// together do not grow into the same stretch (engine.js's #growClaim() then
// grows each peer toward its less covered neighbour). Only width one bisects.
// Splitting a gap with a wider claim can leave remainders that need one more
// peer to cover than filling from an edge would (tools/split.test.mjs checks
// this exhaustively).
//
// O(L · n log n).
export function place(peers, n, L) {
    n = Math.max(1, Math.min(n, L));
    const d = depths(peers, L);

    let bestProfile = null;
    const ties = [];
    for (let lo = 0; lo + n <= L; lo++) {
        const profile = Array.from(d.subarray(lo, lo + n)).sort((a, b) => a - b);
        if (bestProfile === null || lexLess(profile, bestProfile)) {
            bestProfile = profile;
            ties.length = 0;
            ties.push(lo);
        } else if (!lexLess(bestProfile, profile)) {
            ties.push(lo);
        }
    }
    if (ties.length === 1 || n > 1) return ties[0];

    let bestStart = ties[0], bestLen = 1, runStart = ties[0], runLen = 1;
    for (let i = 1; i < ties.length; i++) {
        if (ties[i] === ties[i - 1] + 1) {
            runLen++;
        } else {
            if (runLen > bestLen) { bestLen = runLen; bestStart = runStart; }
            runStart = ties[i];
            runLen = 1;
        }
    }
    if (runLen > bestLen) { bestLen = runLen; bestStart = runStart; }
    return bestStart + Math.floor((bestLen - 1) / 2);
}

const lexLess = (a, b) => {
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i];
    return false;
};

// `peers` must not include the peer being assigned. Otherwise a peer being
// re-placed would avoid its own range.
export function assign(peers, cap, L, floor) {
    const n = size(cap, L, floor);
    if (!n) return null;
    const lo = place(peers, n, L);
    return { lo, hi: lo + n };
}
