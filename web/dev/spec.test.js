// Speculative decoding's GPU invariants, on random weights:
//
// - a Gated DeltaNet block that ran a verify job of n tokens and then keeps
//   only k of them (the rest were rejected) must hold exactly the state, and
//   give the next job exactly the output, that k plain one-token jobs give;
// - with every token kept, no restore runs and the same holds.
//
// Bitwise, not to a tolerance: a verified token must be the token plain
// decoding samples. Run with runSpecTests(device), which returns { ok, rows }.

import { F32, uploadF32, makeStorage, readF32, withPass } from '../model/kernels.js';
import { block, makeScratch, makeRecurrentState, makeReplayLayer } from '../model/model.js';

function makeRng(seed) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

const rand = (n, rng, scale = 1, offset = 0) =>
    Float32Array.from({ length: n }, () => offset + (rng() * 2 - 1) * scale);

// Small dims that are not multiples of the workgroup sizes, with nV/nK = 2,
// and Qwen3.8-27B's own DeltaNet heads (16 key heads, 48 value heads, 128
// wide), where the recurrence keeps its state in registers, on a small hidden
// width so random weights stay small.
function dims({ nK = 2, nV = 4, Dk = 32, Dv = 32, hidden = 72, ffn = 100 } = {}) {
    const K4 = 4;
    return {
        hidden, n_heads: 2, n_kv_heads: 1, head_dim: 36, group: 2, rot_dim: 36,
        ffn, vocab: 33, rms_eps: 1e-6, n_layer: 1,
        layer_types: ['linear_attention'], rope_theta: 10000, hybrid: true,
        linear: { nK, nV, Dk, Dv, K4, keyDim: nK * Dk, valueDim: nV * Dv, group: nV / nK },
    };
}
const SHAPES = [
    ['small', dims()],
    ['qwen heads', dims({ nK: 16, nV: 48, Dk: 128, Dv: 128, hidden: 136, ffn: 40 })],
];

function linearWeights(device, d, rng) {
    const H = d.hidden, { keyDim, valueDim, nV, Dv, K4 } = d.linear;
    const up = (a) => uploadF32(device, a);
    return {
        kind: 'linear_attention',
        input_layernorm: up(rand(H, rng, 0.1, 1)),
        post_attention_layernorm: up(rand(H, rng, 0.1, 1)),
        in_proj_q: up(rand(keyDim * H, rng, 0.2)),
        in_proj_k: up(rand(keyDim * H, rng, 0.2)),
        in_proj_v: up(rand(valueDim * H, rng, 0.2)),
        in_proj_z: up(rand(valueDim * H, rng, 0.2)),
        in_proj_b: up(rand(nV * H, rng, 0.2)),
        in_proj_a: up(rand(nV * H, rng, 0.2)),
        conv1d_q: up(rand(keyDim * K4, rng, 0.5)),
        conv1d_k: up(rand(keyDim * K4, rng, 0.5)),
        conv1d_v: up(rand(valueDim * K4, rng, 0.5)),
        gate_params: up(rand(2 * nV, rng, 0.5)),
        gated_norm: up(rand(Dv, rng, 0.1, 1)),
        out_proj: up(rand(H * valueDim, rng, 0.1)),
        gate_proj: up(rand(d.ffn * H, rng, 0.1)),
        up_proj: up(rand(d.ffn * H, rng, 0.1)),
        down_proj: up(rand(H * d.ffn, rng, 0.1)),
    };
}

export async function runSpecTests(device) {
    const rows = [];
    for (const [label, d] of SHAPES) await runShape(device, d, label, rows);
    const failed = rows.filter((r) => !r.ok);
    console.log(`spec tests: ${rows.length - failed.length}/${rows.length} passed`);
    return { ok: failed.length === 0, rows };
}

async function runShape(device, d, label, rows) {
    const assert = (name, ok, detail = '') => rows.push({ name: `${label}: ${name}`, ok, detail });

    const H = d.hidden;
    const rng = makeRng(5);
    const w = { layers: [linearWeights(device, d, rng)] };
    const s = makeScratch(device, d, 16, 64);
    const out = makeStorage(device, 16 * H * F32);

    // One job: write the rows into s.x, run the block, keep the output.
    async function job(cache, X, T, spec = null) {
        device.queue.writeBuffer(s.x, 0, X);
        const enc = device.createCommandEncoder();
        withPass(enc, (pass) => block(device, pass, d, w, 0, cache, s, T, 0, spec));
        enc.copyBufferToBuffer(s.x, 0, out, 0, T * H * F32);
        device.queue.submit([enc.finish()]);
        return readF32(device, out, T * H * F32);
    }
    const state = async (rc) => [
        await readF32(device, rc.S, rc.S.size), await readF32(device, rc.histQ, rc.histQ.size),
        await readF32(device, rc.histK, rc.histK.size), await readF32(device, rc.histV, rc.histV.size),
    ];
    const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
    const sameAll = (as, bs) => as.every((a, i) => same(a, bs[i]));

    const n = 6;
    const prefix = rand(3 * H, rng);           // a short prompt, so the state is not zero
    const verify = rand(n * H, rng);           // the verify job's n rows
    const next = rand(H, rng);                 // the job after it

    for (let k = 1; k <= n; k++) {
        // Speculative: prompt, a verify job of n rows (snap), then the next
        // job, which keeps k rows of the verify (restore when k < n).
        const cA = { recurrent: [makeRecurrentState(device, d)] };
        const replay = [makeReplayLayer(device, d)];
        await job(cA, prefix, 3);
        await job(cA, verify, n, { replay, snap: true });
        const outA = await job(cA, next, 1, { replay, keep: k < n ? k : null });

        // Plain: prompt, the first k verify rows one at a time, the next job.
        const cB = { recurrent: [makeRecurrentState(device, d)] };
        await job(cB, prefix, 3);
        for (let i = 0; i < k; i++) await job(cB, verify.slice(i * H, (i + 1) * H), 1);
        const outB = await job(cB, next, 1);

        const st = sameAll(await state(cA.recurrent[0]), await state(cB.recurrent[0]));
        assert(`deltanet verify ${n}, keep ${k}: state bitwise == ${k} plain steps`, st);
        assert(`deltanet verify ${n}, keep ${k}: next output bitwise == plain`, same(outA, outB));
    }

    // keep 0: a verify refused further down the line is re-sent as a plain
    // job from the same `past`, so the block puts back the state from before
    // the verify and keeps none of its rows.
    {
        const cA = { recurrent: [makeRecurrentState(device, d)] };
        const replay = [makeReplayLayer(device, d)];
        await job(cA, prefix, 3);
        await job(cA, verify, n, { replay, snap: true });
        const outA = await job(cA, next, 1, { replay, keep: 0 });
        const cB = { recurrent: [makeRecurrentState(device, d)] };
        await job(cB, prefix, 3);
        const outB = await job(cB, next, 1);
        assert(`deltanet verify ${n}, keep 0: next output and state bitwise == no verify`,
            same(outA, outB) && sameAll(await state(cA.recurrent[0]), await state(cB.recurrent[0])));
    }

    // The control: without the restore, the rejected rows stay folded into the
    // state, and the next output differs. Shows the checks above can fail.
    {
        const cA = { recurrent: [makeRecurrentState(device, d)] };
        const replay = [makeReplayLayer(device, d)];
        await job(cA, prefix, 3);
        await job(cA, verify, n, { replay, snap: true });
        const outA = await job(cA, next, 1);
        const cB = { recurrent: [makeRecurrentState(device, d)] };
        await job(cB, prefix, 3);
        for (let i = 0; i < 2; i++) await job(cB, verify.slice(i * H, (i + 1) * H), 1);
        const outB = await job(cB, next, 1);
        assert('control: skipping the restore gives a different output', !same(outA, outB));
    }

    // A verify job's own rows: row i must equal plain decoding's output for it.
    {
        const cA = { recurrent: [makeRecurrentState(device, d)] };
        const replay = [makeReplayLayer(device, d)];
        await job(cA, prefix, 3);
        const outV = await job(cA, verify, n, { replay, snap: true });
        const cB = { recurrent: [makeRecurrentState(device, d)] };
        await job(cB, prefix, 3);
        let ok = true;
        for (let i = 0; i < n; i++) {
            const o = await job(cB, verify.slice(i * H, (i + 1) * H), 1);
            if (!same(o, outV.slice(i * H, (i + 1) * H))) ok = false;
        }
        assert(`deltanet verify ${n}: every row bitwise == plain step`, ok);
    }

    // Sessions interleave on a Runner: A's verify, then B's verify, then A's
    // restore must see only A's replay data.
    {
        const mk = () => ({ c: { recurrent: [makeRecurrentState(device, d)] }, r: [makeReplayLayer(device, d)] });
        const A = mk(), B = mk();
        const otherVerify = rand(n * H, rng);
        await job(A.c, prefix, 3);
        await job(B.c, prefix.slice().reverse(), 3);
        await job(A.c, verify, n, { replay: A.r, snap: true });
        await job(B.c, otherVerify, n, { replay: B.r, snap: true });
        const outA = await job(A.c, next, 1, { replay: A.r, keep: 2 });

        const cP = { recurrent: [makeRecurrentState(device, d)] };
        await job(cP, prefix, 3);
        for (let i = 0; i < 2; i++) await job(cP, verify.slice(i * H, (i + 1) * H), 1);
        const outP = await job(cP, next, 1);
        assert('deltanet interleaved sessions: each restores only its own state',
            same(outA, outP) && sameAll(await state(A.c.recurrent[0]), await state(cP.recurrent[0])));
    }

}
