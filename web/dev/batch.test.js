// Batching's GPU invariant (model.js blockSegs), on random weights: a pass
// that runs several sessions' rows together gives every session exactly the
// output rows, KV rows and recurrent state it gets when its job runs alone.
//
// Two worlds with the same weights and the same history. In one, each step
// runs the sessions' jobs as one batch. In the other, the same jobs run one at
// a time. The batches mix plain steps, a speculative verify (snap) and the job
// after one (restore), over a Gated DeltaNet block and a hybrid full-attention
// block (q/k norm, attn_gate, partial rotary), with f32 and q8 weights.
//
// Bitwise, not to a tolerance: a batched chat must sample the tokens it would
// sample alone. Run with runBatchTests(device), which returns { ok, rows }.

import { F32, uploadF32, uploadQ8, makeStorage, readF32, withPass } from '../model/kernels.js';
import { blockSegs, makeScratch, makeSegScratch, makePagePool, makeSessionFrame, ensurePages,
         makeReplayLayer } from '../model/model.js';

function makeRng(seed) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

const rand = (n, rng, scale = 1, offset = 0) =>
    Float32Array.from({ length: n }, () => offset + (rng() * 2 - 1) * scale);

// Two blocks: Gated DeltaNet, then a hybrid full-attention block. Small dims
// that are not multiples of the workgroup sizes, and Qwen3.8-27B's own
// DeltaNet heads on a small hidden width.
function dims({ nK = 2, nV = 4, Dk = 32, Dv = 32, hidden = 72, ffn = 100,
                n_heads = 2, n_kv_heads = 1, head_dim = 32, rot_dim = 8 } = {}) {
    return {
        hidden, n_heads, n_kv_heads, head_dim, group: n_heads / n_kv_heads, rot_dim,
        ffn, vocab: 33, rms_eps: 1e-6, n_layer: 2, rope_theta: 10000, hybrid: true,
        layer_types: ['linear_attention', 'full_attention'],
        linear: { nK, nV, Dk, Dv, K4: 4, keyDim: nK * Dk, valueDim: nV * Dv, group: nV / nK },
    };
}
const SHAPES = [
    ['small', dims()],
    ['qwen heads', dims({ nK: 16, nV: 48, Dk: 128, Dv: 128, hidden: 136, ffn: 40,
                          n_heads: 6, n_kv_heads: 2, head_dim: 64, rot_dim: 16 })],
];

// A [N, K] projection, as f32 or q8 (groups of 32 along K, the last partial).
function projection(device, rng, N, K, q8) {
    if (!q8) return uploadF32(device, rand(N * K, rng, 0.2));
    const group = 32;
    const data = new Int8Array(N * K);
    for (let i = 0; i < data.length; i++) data[i] = Math.floor(rng() * 255) - 127;
    const scales = rand(N * Math.ceil(K / group), rng, 0.002, 0.003);
    return uploadQ8(device, new Uint8Array(data.buffer), scales, group);
}

function weights(device, d, rng, q8) {
    const H = d.hidden, D = d.head_dim, { keyDim, valueDim, nV, Dv, K4 } = d.linear;
    const qDim = d.n_heads * D, kvDim = d.n_kv_heads * D;
    const up = (a) => uploadF32(device, a);
    const mat = (N, K) => projection(device, rng, N, K, q8);
    const ffn = () => ({
        post_attention_layernorm: up(rand(H, rng, 0.1, 1)),
        gate_proj: mat(d.ffn, H), up_proj: mat(d.ffn, H), down_proj: mat(H, d.ffn),
    });
    return { layers: [
        {
            kind: 'linear_attention',
            input_layernorm: up(rand(H, rng, 0.1, 1)),
            in_proj_q: mat(keyDim, H), in_proj_k: mat(keyDim, H), in_proj_v: mat(valueDim, H),
            in_proj_z: mat(valueDim, H), in_proj_b: mat(nV, H), in_proj_a: mat(nV, H),
            conv1d_q: up(rand(keyDim * K4, rng, 0.5)),
            conv1d_k: up(rand(keyDim * K4, rng, 0.5)),
            conv1d_v: up(rand(valueDim * K4, rng, 0.5)),
            gate_params: up(rand(2 * nV, rng, 0.5)),
            gated_norm: up(rand(Dv, rng, 0.1, 1)),
            out_proj: mat(H, valueDim),
            ...ffn(),
        },
        {
            kind: 'full_attention',
            input_layernorm: up(rand(H, rng, 0.1, 1)),
            q_proj: mat(qDim, H), k_proj: mat(kvDim, H), v_proj: mat(kvDim, H),
            attn_gate: mat(qDim, H), o_proj: mat(H, qDim),
            q_norm: up(rand(D, rng, 0.1, 1)), k_norm: up(rand(D, rng, 0.1, 1)),
            ...ffn(),
        },
    ] };
}

export async function runBatchTests(device) {
    const rows = [];
    for (const [label, d] of SHAPES) {
        for (const q8 of [false, true]) await runShape(device, d, `${label} ${q8 ? 'q8' : 'f32'}`, q8, rows);
    }
    const failed = rows.filter((r) => !r.ok);
    console.log(`batch tests: ${rows.length - failed.length}/${rows.length} passed`);
    return { ok: failed.length === 0, rows };
}

async function runShape(device, d, label, q8, rows) {
    const assert = (name, ok, detail = '') => rows.push({ name: `${label}: ${name}`, ok, detail });
    const H = d.hidden;
    const rng = makeRng(11);
    const w = weights(device, d, rng, q8);
    const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

    // One world: a page pool, a frame and a replay slot per session, its own
    // scratch. The worlds share only the weights.
    const SESSIONS = ['A', 'B', 'C'];
    function world() {
        const pool = makePagePool(device, d, { pageTokens: 16, nPages: 12, n_layer: 2, layerLo: 0 });
        const s = makeScratch(device, d, 32, 64);
        const g = makeSegScratch(device, d, 32, s);
        const out = makeStorage(device, 32 * H * F32);
        const sess = {};
        for (const id of SESSIONS) {
            sess[id] = { frame: makeSessionFrame(device, d, pool, 4, 2, 0), past: 0,
                         replay: [makeReplayLayer(device, d), null] };
        }
        return { pool, s, g, out, sess };
    }

    // Run `jobs` ({ id, X, spec? }) as one pass, rows packed in order. Pages
    // are taken session by session in job order, the same in both worlds.
    async function pass(W, jobs) {
        const segs = [];
        let off = 0;
        for (const { id, X, spec } of jobs) {
            const n = X.length / H, ss = W.sess[id];
            ensurePages(device, W.pool, ss.frame, ss.past + n);
            device.queue.writeBuffer(W.s.x, off * H * F32, X);
            segs.push({ cache: ss.frame, T_past: ss.past, n, rowOff: off,
                        spec: spec ? { replay: ss.replay, ...spec } : null });
            off += n;
        }
        const enc = device.createCommandEncoder();
        withPass(enc, (p) => {
            for (let layer = 0; layer < 2; layer++) blockSegs(device, p, d, w, layer, segs, W.s, W.g);
        });
        enc.copyBufferToBuffer(W.s.x, 0, W.out, 0, off * H * F32);
        device.queue.submit([enc.finish()]);
        const all = await readF32(device, W.out, off * H * F32);
        return segs.map((seg, i) => {
            W.sess[jobs[i].id].past += jobs[i].advance ?? seg.n;
            return all.slice(seg.rowOff * H, (seg.rowOff + seg.n) * H);
        });
    }
    // The same jobs one at a time.
    async function alone(W, jobs) {
        const outs = [];
        for (const job of jobs) outs.push(...await pass(W, [job]));
        return outs;
    }
    const state = async (W, id) => {
        const rc = W.sess[id].frame.recurrent[0];
        return [await readF32(device, rc.S, rc.S.size), await readF32(device, rc.histQ, rc.histQ.size),
                await readF32(device, rc.histK, rc.histK.size), await readF32(device, rc.histV, rc.histV.size)];
    };
    const kv = async (W) => [await readF32(device, W.pool.K[1], W.pool.K[1].size),
                             await readF32(device, W.pool.V[1], W.pool.V[1].size)];

    const B1 = world(), S1 = world();
    const X = (n) => rand(n * H, rng);

    // History, run alone in both: three prompts, then B's verify of 4 rows.
    const prompts = { A: X(3), B: X(5), C: X(2) };
    for (const W of [B1, S1]) {
        for (const id of SESSIONS) await pass(W, [{ id, X: prompts[id] }]);
    }
    const bVerify = X(4);
    for (const W of [B1, S1]) await pass(W, [{ id: 'B', X: bVerify, spec: { snap: true }, advance: 2 }]);
    // B's verify kept 2 of its 4 rows: B is now at 5 + 2.

    // Step 1: A plain, B rolls back to 2 kept rows and steps, C verifies 3.
    const step1 = [
        { id: 'A', X: X(1) },
        { id: 'B', X: X(1), spec: { keep: 2 } },
        { id: 'C', X: X(3), spec: { snap: true }, advance: 1 },
    ];
    const b1 = await pass(B1, step1);
    const s1 = await alone(S1, step1);
    step1.forEach(({ id }, i) => assert(`step 1, ${id}: batched rows bitwise == alone`, same(b1[i], s1[i])));

    // Step 2: C rolls back to 1 kept row, A sends 2 rows, B plain.
    const step2 = [
        { id: 'C', X: X(1), spec: { keep: 1 } },
        { id: 'A', X: X(2) },
        { id: 'B', X: X(1) },
    ];
    const b2 = await pass(B1, step2);
    const s2 = await alone(S1, step2);
    step2.forEach(({ id }, i) => assert(`step 2, ${id}: batched rows bitwise == alone`, same(b2[i], s2[i])));

    // Past 8 rows (q8 weights take kernels.js matmulRowsQ8Src there). Step 3:
    // 28 rows, B verifies 7 and keeps 3. Step 4: 25 rows, B rolls back.
    const step3 = [
        { id: 'A', X: X(12) },
        { id: 'B', X: X(7), spec: { snap: true }, advance: 3 },
        { id: 'C', X: X(9) },
    ];
    const b3 = await pass(B1, step3);
    const s3 = await alone(S1, step3);
    step3.forEach(({ id }, i) => assert(`step 3 (28 rows), ${id}: batched rows bitwise == alone`, same(b3[i], s3[i])));
    const step4 = [
        { id: 'B', X: X(1), spec: { keep: 3 } },
        { id: 'A', X: X(20) },
        { id: 'C', X: X(4) },
    ];
    const b4 = await pass(B1, step4);
    const s4 = await alone(S1, step4);
    step4.forEach(({ id }, i) => assert(`step 4 (25 rows), ${id}: batched rows bitwise == alone`, same(b4[i], s4[i])));

    for (const id of SESSIONS) {
        const [a, b] = [await state(B1, id), await state(S1, id)];
        assert(`${id}: recurrent state bitwise == alone`, a.every((x, i) => same(x, b[i])));
    }
    const [kb, ks] = [await kv(B1), await kv(S1)];
    assert('KV pool bitwise == alone', same(kb[0], ks[0]) && same(kb[1], ks[1]));

    // The control: the same batch with two sessions' caches swapped gives
    // different rows, so the checks above can fail.
    const B2 = world(), S2 = world();
    for (const W of [B2, S2]) for (const id of SESSIONS) await pass(W, [{ id, X: prompts[id] }]);
    const ctl = [{ id: 'A', X: X(1) }, { id: 'B', X: X(1) }];
    const sc = await alone(S2, ctl);
    const bc = await pass(B2, [{ ...ctl[0], id: 'B' }, { ...ctl[1], id: 'A' }]);
    assert('control: swapped caches give different rows', !same(bc[0], sc[0]) && !same(bc[1], sc[1]));
}
