// Wall-clock microbenchmarks for every kernel, plus the whole forward pass.
//
// Wall-clock, not timestamp-query, which is an optional feature: the bench
// must run on every GPU that a kernel change is compared on. The fixture is
// full-attention only.
//
// Method: record `iters` consecutive dispatches of one kernel on one encoder,
// submit once, and await queue.onSubmittedWorkDone(). One submit spreads the
// driver round trip over every iteration, leaving the kernel plus its
// per-dispatch launch cost, which is part of what the bench measures.
//
// Two numbers per kernel:
//   recordMs   CPU time to build the commands (uniform + bind group + dispatch)
//   gpuMs      submit → work done, divided by iters
// When recordMs is close to gpuMs, the kernel is limited by launch cost, and
// fusing it with another kernel helps more than optimizing its inner loop.
//
// Buffers are left zero-filled. WebGPU zero-initializes every buffer, so the
// arithmetic is defined and has no denormals, and timing does not depend on
// the values. Correctness is tested in kernels.test.js.

import * as K from '../model/kernels.js';
import { makeCache, makeScratch, forward } from '../model/model.js';

const F32 = 4;

// --- Timing ---

async function once(device, record, iters) {
    const enc = device.createCommandEncoder();
    const t0 = performance.now();
    K.withPass(enc, (pass) => { for (let i = 0; i < iters; i++) record(pass); });
    const cmd = enc.finish();
    const t1 = performance.now();
    device.queue.submit([cmd]);
    await device.queue.onSubmittedWorkDone();
    const t2 = performance.now();
    return { recordMs: (t1 - t0) / iters, gpuMs: (t2 - t1) / iters };
}

// Best of `reps`, after a warm-up submit. Best, not mean, because noise only
// makes a run slower (the compositor or another process using the GPU).
export async function timeKernel(device, name, record, { bytes = 0, iters = 32, reps = 3 } = {}) {
    try {
        await once(device, record, 2);                       // warm-up: compile, allocate, page in
        let best = null;
        for (let r = 0; r < reps; r++) {
            const t = await once(device, record, iters);
            if (!best || t.gpuMs < best.gpuMs) best = t;
        }
        const gbps = bytes ? (bytes / (best.gpuMs / 1000)) / 1e9 : 0;
        // `calls` is how many times `record` ran in total, warm-up included.
        // A caller counting dispatches divides by it to get a per-pass count.
        return { name, ok: true, ...best, bytes, gbps, calls: 2 + reps * iters };
    } catch (e) {
        return { name, ok: false, recordMs: 0, gpuMs: 0, bytes, gbps: 0, calls: 0, error: String(e) };
    }
}

// --- Fixture: one layer's weights at the model's dims ---

// The bench needs buffers of production shape, not a whole model. Every kernel
// here uses at most one layer, so one layer's weights plus the embedding table
// are enough. That is ~330 MB at SmolLM2-1.7B, against 3.2 GB for the real
// checkpoint, so 1.7B-shaped kernels can be measured on a laptop without
// downloading it.
//
// The weights are f16 when the device has shader-f16 and f32 otherwise. The
// caller decides and the header reports it, because f32 doubles the weight
// bytes read, which is what limits every matmul here. An f16 pipeline on a
// device without the feature fails to compile, every dispatch on it is
// dropped, and the kernel would measure as taking no time.
function fixture(device, d, T_new, N_max, f16) {
    const H = d.hidden, D = d.head_dim, nH = d.n_heads, nKV = d.n_kv_heads;
    const kvDim = nKV * D;
    const wbuf = f16
        ? (n) => K.makeStorageF16(device, n)
        : (n) => K.makeStorage(device, n * F32);

    return {
        wbytes: f16 ? 2 : 4,
        s: makeScratch(device, d, T_new, N_max),
        cache: makeCache(device, d, N_max, 1),
        ids: K.makeStorage(device, Math.max(T_new * F32, 16)),
        w: {
            norm:  wbuf(H),
            q:     wbuf(nH * D * H),
            k:     wbuf(kvDim * H),
            v:     wbuf(kvDim * H),
            o:     wbuf(H * H),
            gate:  wbuf(d.ffn * H),
            up:    wbuf(d.ffn * H),
            down:  wbuf(H * d.ffn),
            wte:   wbuf(d.vocab * H),
        },
    };
}

function destroyFixture(f) {
    for (const b of Object.values(f.w)) b.destroy();
    for (const b of Object.values(f.s)) if (b?.destroy) b.destroy();
    for (const b of [...f.cache.K, ...f.cache.V]) b.destroy();
    f.ids.destroy();
}

// --- The kernel suite ---

// Bytes per dispatch, counted as every input read plus every output written,
// ignoring cache reuse. For the matmuls W dominates, and the count is accurate
// at M = 1. For the tiled build at M = 64 the real traffic is lower, because a
// staged tile is reused, so prefill GB/s is an upper bound. Both are stable
// across a code change, which is enough for a before/after comparison.
export function kernelSuite(device, d, { T_new, N_max, T_past,
                                        f16 = device.features?.has('shader-f16') ?? false }) {
    const H = d.hidden, D = d.head_dim, nH = d.n_heads, nKV = d.n_kv_heads;
    const kvDim = nKV * D, T_act = T_past + T_new, W = f16 ? 2 : 4;
    const f = fixture(device, d, T_new, N_max, f16);
    const { s, cache, w, ids } = f;
    const Kc = cache.K[0], Vc = cache.V[0];

    const mm = (X, Wt, Y, M, Kd, N) => ({
        record: (e) => K.matmulNobias(device, e, X, Wt, Y, M, Kd, N),
        bytes: M * Kd * F32 + N * Kd * W + M * N * F32,
    });

    const items = [
        ['gather_embed',      { record: (e) => K.gatherEmbedNoPos(device, e, ids, w.wte, s.x, T_new, H),
                                bytes: T_new * H * (W + F32) }],
        ['rmsnorm',           { record: (e) => K.rmsnorm(device, e, s.x, w.norm, s.h, T_new, H, d.rms_eps),
                                bytes: 2 * T_new * H * F32 + H * W }],
        ['matmul q_proj',     mm(s.h, w.q, s.q, T_new, H, nH * D)],
        ['matmul k_proj',     mm(s.h, w.k, s.k, T_new, H, kvDim)],
        ['matmul v_proj',     mm(s.h, w.v, s.v, T_new, H, kvDim)],
        ['reshape_heads',     { record: (e) => K.reshapeHeads(device, e, s.q, s.qh, T_new, nH, D),
                                bytes: 2 * T_new * nH * D * F32 }],
        ['store_kv K',        { record: (e) => K.storeKV(device, e, s.k, Kc, T_new, T_past, N_max, nKV, D,
                                    { theta: d.rope_theta, scaling: d.rope_scaling, rotDim: d.rot_dim ?? D }),
                                bytes: T_new * kvDim * F32 + T_new * nKV * K.kvRowWords(D) * 4 }],
        ['rope Q',            { record: (e) => K.rope(device, e, s.qh, nH, T_new, D, T_past, T_new, 0, d.rope_theta),
                                bytes: 2 * T_new * nH * D * F32 }],
        ['qk_scores_gqa',     { record: (e) => K.qkScoresGQA(device, e, s.qh, Kc, s.scores, T_new, T_act, N_max, D, nH, d.group),
                                bytes: nH * T_new * D * F32 + nKV * T_act * K.kvRowWords(D) * 4 + nH * T_new * T_act * F32 }],
        ['causal_softmax',    { record: (e) => K.causalSoftmax(device, e, s.scores, T_new, T_act, T_past, nH),
                                bytes: 3 * nH * T_new * T_act * F32 }],
        ['attn_av_gqa',       { record: (e) => K.attnAVGQA(device, e, s.scores, Vc, s.ctx, T_new, T_act, N_max, D, nH, d.group),
                                bytes: nH * T_new * T_act * F32 + nKV * T_act * K.kvRowWords(D) * 4 + nH * T_new * D * F32 }],
        ['merge_heads',       { record: (e) => K.mergeHeads(device, e, s.ctx, s.merged, T_new, nH, D),
                                bytes: 2 * T_new * nH * D * F32 }],
        ['matmul o_proj',     mm(s.merged, w.o, s.attn, T_new, H, H)],
        ['add_inplace',       { record: (e) => K.addInplace(device, e, s.x, s.attn, T_new * H),
                                bytes: 3 * T_new * H * F32 }],
        ['matmul gate_proj',  mm(s.h, w.gate, s.gate, T_new, H, d.ffn)],
        ['matmul up_proj',    mm(s.h, w.up, s.up, T_new, H, d.ffn)],
        ['silu_mul',          { record: (e) => K.siluMul(device, e, s.gate, s.up, s.act, T_new * d.ffn),
                                bytes: 3 * T_new * d.ffn * F32 }],
        ['matmul down_proj',  mm(s.act, w.down, s.down, T_new, d.ffn, H)],
        ['matmul lm_head',    mm(s.h, w.wte, s.logits, T_new, H, d.vocab)],
    ];

    return { items, free: () => destroyFixture(f) };
}

// Run the suite for one shape (decode or prefill).
export async function benchKernels(device, d, opts) {
    const { items, free } = kernelSuite(device, d, opts);
    const rows = [];
    try {
        for (const [name, spec] of items) {
            rows.push(await timeKernel(device, name, spec.record, { bytes: spec.bytes }));
        }
    } finally {
        free();
    }
    return rows;
}

// --- Whole forward pass, on real weights ---

// Records the full stack as generate.js does (one encoder, one submit) but
// with no readback, so it reports GPU time for the pass without the map wait.
// The difference between this and the tok/s runGeneration() prints is the cost
// of the readback.
export async function benchForward(device, d, w, { T_new = 1, T_past = 0, N_max = 1024, iters = 8 } = {}) {
    const cache = makeCache(device, d, N_max);
    const s = makeScratch(device, d, T_new, N_max);
    const ids = K.makeStorage(device, Math.max(T_new * F32, 16));
    try {
        K.resetCounters();
        const record = (e) => forward(device, e, d, w, cache, s, ids, T_new, T_past);
        const t = await timeKernel(device, 'forward', record, { iters, reps: 3 });
        const c = K.counters();
        return {
            ...t,
            T_new, T_past,
            dispatches: t.calls ? Math.round(c.dispatches / t.calls) : 0,
            bindGroupsBuilt: c.bindGroupsBuilt,
            tokPerSec: t.gpuMs > 0 ? (1000 / t.gpuMs) * T_new : 0,
        };
    } finally {
        for (const b of Object.values(s)) if (b?.destroy) b.destroy();
        for (const b of [...cache.K, ...cache.V]) b.destroy();
        ids.destroy();
    }
}

// --- Rows: one q8 matmul at 1 to 64 rows ---
//
// What batching and speculative decoding cost on the projections: one q8
// matmul (group 32, random weights) at Qwen3.8-27B's shapes, for each row
// count. Up to GEMV_ROWS_MAX rows each row is bit-identical to a one-row call
// (kernels.test.js checks it); past it the tiled kernel runs.
const ROWS_SHAPES = [['gate/up', 5120, 17408], ['down', 17408, 5120], ['q/z', 5120, 6144]];
const ROWS_COUNTS = [1, 2, 4, 8, 12, 16, 24, 32, 64];

export async function benchRows(device) {
    const out = [];
    for (const [name, Kd, N] of ROWS_SHAPES) {
        const data = new Uint8Array(N * Kd);
        for (let o = 0; o < data.length; o += 65536) {
            crypto.getRandomValues(data.subarray(o, Math.min(o + 65536, data.length)));
        }
        const W = K.uploadQ8(device, data, new Float32Array(N * Math.ceil(Kd / 32)).fill(1e-3), 32);
        const X = K.makeStorage(device, 64 * Kd * F32), Y = K.makeStorage(device, 64 * N * F32);
        const row = [];
        for (const M of ROWS_COUNTS) {
            const t = await timeKernel(device, `${name} ${M}`,
                (e) => K.matmulNobias(device, e, X, W, Y, M, Kd, N), { iters: 16 });
            row.push({ M, ms: t.ok ? t.gpuMs : NaN });
        }
        out.push({ name, Kd, N, row });
        W.destroy(); X.destroy(); Y.destroy();
    }
    return out;
}

export function formatRowsBench(out) {
    const lines = ['\n--- q8 matmul by rows (ms; per row in brackets) ---',
        `  ${'shape'.padEnd(20)} ${ROWS_COUNTS.map((M) => pad(M, 13)).join('')}`];
    for (const { name, Kd, N, row } of out) {
        lines.push(`  ${`${name} ${Kd}→${N}`.padEnd(20)} ` + row.map(({ M, ms }) =>
            pad(`${ms.toFixed(2)} (${(ms / M).toFixed(2)})`, 13)).join(''));
    }
    return lines.join('\n');
}

// --- Reporting ---

const pad = (v, n) => String(v).padStart(n);

export function formatRows(title, rows) {
    const out = [`\n--- ${title} ---`,
        `  ${'kernel'.padEnd(20)} ${pad('gpu µs', 9)} ${pad('rec µs', 8)} ${pad('GB/s', 8)}`];
    let total = 0;
    for (const r of rows) {
        if (!r.ok) { out.push(`  ${r.name.padEnd(20)} ${pad('FAIL', 9)}  ${r.error}`); continue; }
        total += r.gpuMs;
        out.push(`  ${r.name.padEnd(20)} ${pad((r.gpuMs * 1000).toFixed(1), 9)}`
            + ` ${pad((r.recordMs * 1000).toFixed(1), 8)}`
            + ` ${pad(r.gbps ? r.gbps.toFixed(1) : '–', 8)}`);
    }
    out.push(`  ${'sum of one block'.padEnd(20)} ${pad((total * 1000).toFixed(1), 9)} µs`);
    return out.join('\n');
}

export function formatForward(r) {
    if (!r.ok) return `\n--- forward (T_new=${r.T_new}): FAIL ${r.error}`;
    return `\n--- forward pass (T_new=${r.T_new}, T_past=${r.T_past}) ---\n`
        + `  ${r.gpuMs.toFixed(2)} ms/pass  ·  ${r.tokPerSec.toFixed(1)} tok/s (GPU only, no readback)\n`
        + `  ${r.dispatches} dispatches/pass  ·  ${r.bindGroupsBuilt} bind groups built over the run\n`
        + `  ${(r.recordMs * 1000).toFixed(0)} µs CPU to record`;
}
