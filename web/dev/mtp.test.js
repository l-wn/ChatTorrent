// The MTP draft head's plumbing on random weights (model.js mtpBlock,
// mtpDraft, mtpChain):
//
// - argmax_rows picks the CPU's argmax, the lowest index on a tie;
// - mtpChain (one submit) drafts what the steps run one at a time draft;
// - a batched fill of the chain's inputs (what the next job's fill does)
//   predicts, row by row, the drafts the chain chose.
//
// Whether the head matches Qwen3.8's is checked end to end by its acceptance
// rate. Run with runMtpTests(device), which returns { ok, rows }.

import { F32, uploadF32, makeStorage, readF32, withPass, argmaxRows } from '../model/kernels.js';
import { mtpBlock, mtpDraft, mtpChain, makeMtpScratch } from '../model/model.js';

function makeRng(seed) {
    let s = seed >>> 0;
    return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}
const rand = (n, rng, scale = 1, offset = 0) =>
    Float32Array.from({ length: n }, () => offset + (rng() * 2 - 1) * scale);

const d = {
    hidden: 64, n_heads: 4, n_kv_heads: 2, head_dim: 16, group: 2, rot_dim: 8, ffn: 48,
    vocab: 300, rms_eps: 1e-6, rope_theta: 10000, hybrid: true, n_layer: 1,
};
const V = 50, NMAX = 64;

function mtpWeights(device, rng) {
    const H = d.hidden, q = d.n_heads * d.head_dim, kv = d.n_kv_heads * d.head_dim;
    const up = (a) => uploadF32(device, a);
    return {
        kind: 'full_attention', draftVocab: V,
        input_layernorm: up(rand(H, rng, 0.1, 1)), post_attention_layernorm: up(rand(H, rng, 0.1, 1)),
        q_proj: up(rand(q * H, rng, 0.3)), k_proj: up(rand(kv * H, rng, 0.3)),
        v_proj: up(rand(kv * H, rng, 0.3)), o_proj: up(rand(H * q, rng, 0.2)),
        attn_gate: up(rand(q * H, rng, 0.3)),
        q_norm: up(rand(d.head_dim, rng, 0.1, 1)), k_norm: up(rand(d.head_dim, rng, 0.1, 1)),
        gate_proj: up(rand(d.ffn * H, rng, 0.2)), up_proj: up(rand(d.ffn * H, rng, 0.2)),
        down_proj: up(rand(H * d.ffn, rng, 0.2)),
        enorm: up(rand(H, rng, 0.1, 1)), hnorm: up(rand(H, rng, 0.1, 1)),
        fc_e: up(rand(H * H, rng, 0.2)), fc_h: up(rand(H * H, rng, 0.2)),
        mtp_norm: up(rand(H, rng, 0.1, 1)),
        embed_draft: up(rand(V * H, rng, 1)), head_draft: up(rand(V * H, rng, 1)),
    };
}

// A contiguous cache view (the one-page case of the paged layout).
const cacheView = (device) => {
    const n = d.n_kv_heads * NMAX * d.head_dim * F32;
    return { K: [makeStorage(device, n)], V: [makeStorage(device, n)], pages: null,
             pageTokens: NMAX, N_max: NMAX };
};

export async function runMtpTests(device) {
    const rows = [];
    const assert = (name, ok, detail = '') => rows.push({ name, ok, detail });
    const rng = makeRng(9);
    const H = d.hidden;
    const run = (record) => {
        const enc = device.createCommandEncoder();
        withPass(enc, record);
        device.queue.submit([enc.finish()]);
    };

    // --- argmax_rows ---
    {
        const T = 5, N = 1037;
        const x = rand(T * N, rng, 3);
        x[1 * N + 17] = 9; x[1 * N + 900] = 9;          // a tie: the lower index wins
        const xb = uploadF32(device, x), ib = makeStorage(device, 16 * F32);
        run((pass) => argmaxRows(device, pass, xb, ib, T, N, 3));
        const got = Array.from((await readF32(device, ib, 16 * F32)).slice(3, 3 + T));
        const want = [];
        for (let r = 0; r < T; r++) {
            let bi = 0;
            for (let i = 1; i < N; i++) if (x[r * N + i] > x[r * N + bi]) bi = i;
            want.push(bi);
        }
        assert('argmax_rows matches the CPU, lowest index on a tie',
            JSON.stringify(got) === JSON.stringify(want), `${got} vs ${want}`);
    }

    const m = mtpWeights(device, rng);
    const prefix = 6, K = 4;
    const hPrefix = rand(prefix * H, rng), idsPrefix = Float32Array.from({ length: prefix }, () => Math.floor(rng() * V));
    const h0 = rand(H, rng), t0 = 7;

    // A cache holding the prompt's MTP rows, as the fill leaves them.
    async function primed() {
        const mc = cacheView(device), ms = makeMtpScratch(device, d, V, 16, NMAX);
        device.queue.writeBuffer(ms.mtpIn, 0, hPrefix);
        device.queue.writeBuffer(ms.mtpIds, 0, idsPrefix);
        run((pass) => mtpBlock(device, pass, d, m, mc, ms, ms.mtpIn, ms.mtpIds, prefix, 0));
        return { mc, ms };
    }

    // --- mtpChain against the steps one at a time ---
    let chain;
    {
        const { mc, ms } = await primed();
        const hb = uploadF32(device, h0);
        device.queue.writeBuffer(ms.mtpIds, 0, new Float32Array([t0]));
        run((pass) => mtpChain(device, pass, d, m, mc, ms, hb, ms.mtpIds, prefix, K));
        chain = Array.from((await readF32(device, ms.mtpIds, (K + 1) * F32)).slice(1), Math.round);
    }
    const xs = [];
    let steps = [];
    {
        const { mc, ms } = await primed();
        let h = h0, t = t0;
        for (let k = 0; k < K; k++) {
            device.queue.writeBuffer(ms.mtpIn, 0, h);
            device.queue.writeBuffer(ms.mtpId, 0, new Float32Array([t]));
            run((pass) => {
                mtpBlock(device, pass, d, m, mc, ms, ms.mtpIn, ms.mtpId, 1, prefix + k);
                mtpDraft(device, pass, d, m, ms, ms.mtpIds, 1, 0);
            });
            h = await readF32(device, ms.x, H * F32);
            xs.push(h);
            t = Math.round((await readF32(device, ms.mtpIds, F32))[0]);
            steps.push(t);
        }
    }
    assert(`mtpChain drafts what ${K} separate steps draft`,
        JSON.stringify(chain) === JSON.stringify(steps), `${chain} vs ${steps}`);
    assert('and the drafts are not all the same', new Set(steps).size > 1, `${steps}`);

    // --- a batched fill of the chain's inputs predicts the chain's drafts ---
    {
        const { mc, ms } = await primed();
        const hIn = new Float32Array(K * H);
        hIn.set(h0, 0);
        for (let k = 1; k < K; k++) hIn.set(xs[k - 1], k * H);
        device.queue.writeBuffer(ms.mtpIn, 0, hIn);
        device.queue.writeBuffer(ms.mtpIds, 0, Float32Array.from([t0, ...steps.slice(0, K - 1)]));
        run((pass) => {
            mtpBlock(device, pass, d, m, mc, ms, ms.mtpIn, ms.mtpIds, K, prefix);
            mtpDraft(device, pass, d, m, ms, ms.mtpIds, K, 8);
        });
        const got = Array.from((await readF32(device, ms.mtpIds, (8 + K) * F32)).slice(8), Math.round);
        assert('a batched fill of the chain\'s inputs predicts the chain\'s drafts',
            JSON.stringify(got) === JSON.stringify(steps), `${got} vs ${steps}`);
    }

    const failed = rows.filter((r) => !r.ok);
    console.log(`mtp tests: ${rows.length - failed.length}/${rows.length} passed`);
    return { ok: failed.length === 0, rows };
}
