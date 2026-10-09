// Checks that the paged KV cache computes the same result as a contiguous one
// on a hybrid range, with real weights.
//
// tools/mesh.gpu.test.mjs checks that paging does not change the output on a
// non-hybrid model, where `pool.K/V` has no nulls and a frame has no recurrent
// state. On Qwen3.8-27B three layers in four are linear_attention. This file
// checks that case:
//
//   1. Index alignment. `w.layers[i]`, `pool.K[i]` and `frame.recurrent[i]`
//      are indexed by the same local layer number, and on a hybrid each has
//      nulls at different indices. A misalignment reads another layer's cache.
//   2. A reused frame must not keep the previous session's DeltaNet state. KV
//      rows are overwritten before they are read, but recurrent state is
//      accumulated, so a reused frame that was not cleared continues another
//      chat. The check shows that an uncleared reuse gives a different result,
//      then that `#cacheFor`'s clear fixes it.
//
// Needs a converted hybrid checkpoint on disk. It loads two real blocks, one
// of each kind, and no client.bin (the input is a synthetic residual stream).
// Two q8 blocks are ~0.8 GB.
//
//   runHybridPool(device, '/weights/qwen3.8-27b-q8')

import { readF32, withPass } from '../model/kernels.js';
import { Runner } from '../model/stage.js';
import { loadManifest, dimsFromManifest, loadBlock } from '../model/loader.js';
import { makeCache, makeScratch, makePagePool, makeSessionFrame, ensurePages,
         block } from '../model/model.js';
import { makeReporter } from './verify.js';

// A fixed pseudo-random residual stream. The input is the same every run, so
// any difference below comes from the cache.
function residual(T, H) {
    const x = new Float32Array(T * H);
    let seed = 12345;
    for (let i = 0; i < x.length; i++) {
        seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
        x[i] = ((seed >>> 8) / 8388608 - 1) * 0.5;
    }
    return x;
}

const identical = (p, q) => p.length === q.length && p.every((v, i) => v === q[i]);

export async function runHybridPool(device, baseUrl = '/weights/qwen3.8-27b-q8') {
    const r = makeReporter();
    const manifest = await loadManifest(baseUrl);
    const d = dimsFromManifest(manifest);

    if (!d.hybrid) {
        r.equal('the checkpoint is hybrid', false,
            `${manifest.model} has one layer kind, so there is nothing here to check`);
        return { ok: false, rows: r.rows, model: manifest.model };
    }

    // The first index where the two kinds are adjacent, so one range covers
    // both and the arrays it indexes have nulls at different indices.
    const lo = d.layer_types.findIndex((k, i) =>
        k !== d.layer_types[i + 1] && d.layer_types[i + 1] !== undefined);
    const nLayer = 2;

    const layers = [];
    for (let i = lo; i < lo + nLayer; i++) {
        layers.push(await loadBlock(device, { baseUrl, manifest, d, index: i }));
    }
    const w = { layers };
    r.equal('a range spanning both layer kinds loaded',
        new Set(layers.map((l) => l.kind)).size === 2,
        layers.map((l) => l.kind).join(' + '));

    const T = 8, N = 64;
    const s = makeScratch(device, d, T, N);
    const x = residual(T, d.hidden);

    const run = async (cache) => {
        device.queue.writeBuffer(s.x, 0, x);
        const enc = device.createCommandEncoder({ label: 'hybrid pool A/B' });
        withPass(enc, (pass) => {
            for (let i = 0; i < nLayer; i++) block(device, pass, d, w, i, cache, s, T, 0);
        });
        device.queue.submit([enc.finish()]);
        return readF32(device, s.x, T * d.hidden);
    };

    // --- contiguous, the reference ---
    const gold = await run(makeCache(device, d, N, nLayer, lo));

    // --- paged ---
    const pool = makePagePool(device, d, { pageTokens: 16, nPages: 8,
                                           n_layer: nLayer, layerLo: lo });
    const frame = makeSessionFrame(device, d, pool, 4, nLayer, lo);

    const kvAt = pool.K.map((k, i) => (k ? i : null)).filter((i) => i !== null);
    const recAt = frame.recurrent.map((x2, i) => (x2 ? i : null)).filter((i) => i !== null);
    r.equal('the pool holds KV for exactly the full-attention layers',
        kvAt.length + recAt.length === nLayer && !kvAt.some((i) => recAt.includes(i)),
        `KV at [${kvAt}], recurrent at [${recAt}]`);

    ensurePages(device, pool, frame, T);
    const paged = await run(frame);
    r.equal('a paged cache computes what a contiguous one computes',
        identical(gold, paged), 'bit-identical over both layer kinds');

    // --- a reused frame ---
    // First check the problem exists: the same input through the same frame a
    // second time must give a different result, because the recurrent state
    // carried over. Otherwise the check below proves nothing.
    const dirty = await run(frame);
    r.equal('a second run on the same frame differs, so the hazard is real',
        !identical(paged, dirty), 'recurrent state accumulated across sessions');

    const enc = device.createCommandEncoder({ label: 'clear recurrent state' });
    for (const rc of frame.recurrent) {
        if (!rc) continue;
        for (const buf of [rc.S, rc.histQ, rc.histK, rc.histV]) enc.clearBuffer(buf);
    }
    device.queue.submit([enc.finish()]);
    r.equal('and clearing it makes the reused frame a fresh one again',
        identical(paged, await run(frame)),
        'what stage.js #cacheFor does when it hands a used frame to a new session');

    return { ok: r.rows.every((x2) => x2.ok), rows: r.rows, model: manifest.model,
             layers: layers.map((l) => l.kind) };
}

// --- Growth, over a linear_attention position, with the pool ---
//
// Growing a claim by one position must update `w.layers`, `pool.K`, `pool.V`
// and every frame's `recurrent` together. On a hybrid each has nulls at
// different indices. `block()` indexes all of them with the same local layer
// number, so a misalignment reads another layer's cache.
//
// The check: a Runner grown to `[lo, hi)` must compute bit-identical output to
// a Runner created at `[lo, hi)`, from the same input. This needs no reference
// data.
//
// Both edges are tested, each growing onto a linear_attention block with a
// full_attention block already in the range. Qwen3.8-27B has three linear
// layers per full one, so blocks 2,3,4 are linear/full/linear, at positions
// 3,4,5.
export async function runHybridGrowth(device, baseUrl = '/weights/qwen3.8-27b-q8') {
    const r = makeReporter();
    const manifest = await loadManifest(baseUrl);
    const d = dimsFromManifest(manifest);
    if (!d.hybrid) {
        r.equal('the checkpoint is hybrid', false, `${manifest.model} has one layer kind`);
        return { ok: false, rows: r.rows, model: manifest.model };
    }

    // The first three consecutive blocks of kinds A, B, A, so growing onto
    // either end adds a block of the kind with no KV.
    const at = d.layer_types.findIndex((k, i) =>
        d.layer_types[i + 1] && d.layer_types[i + 1] !== k && d.layer_types[i + 2] === k);
    if (at < 0) {
        r.equal('the checkpoint alternates layer kinds', false, 'no A,B,A run found');
        return { ok: false, rows: r.rows, model: manifest.model };
    }

    const blocks = [];
    for (let i = at; i < at + 3; i++) {
        blocks.push(await loadBlock(device, { baseUrl, manifest, d, index: i }));
    }
    r.equal('three blocks spanning both kinds loaded',
        blocks[0].kind === blocks[2].kind && blocks[1].kind !== blocks[0].kind,
        blocks.map((b) => b.kind).join(', '));

    // Position p holds block p-1, so the three blocks are positions at+1 ..
    // at+3.
    const lo = at + 1, hi = at + 4;
    const T = 8, N = 64;
    const cfg = { nMax: N, tMax: T, slots: 2, pageTokens: 16, idleMs: 0 };
    const x = residual(T, d.hidden);

    // A job entering at the low edge and running the whole range. There is no
    // embedding or head, so the input and output are both hidden states.
    const feed = (runner, session) => runner.run(
        { k: 'job', s: session, pos: runner.lo, past: 0, n: T }, x);

    const born = new Runner(device, d, { layers: blocks.slice() },
        { lo, hi, have: hi, ...cfg });
    const gold = (await feed(born, 'born')).hidden;
    r.equal('a Runner born wide produced hidden state',
        gold?.length === T * d.hidden, `${gold?.length} floats`);

    // --- growing right, onto a linear_attention position ---
    {
        const grown = new Runner(device, d, { layers: blocks.slice(0, 2) },
            { lo, hi: hi - 1, have: hi - 1, ...cfg });
        grown.growRight();
        grown.append(blocks[2]);
        r.equal(`growRight onto ${blocks[2].kind} keeps the arrays in step`,
            grown.w.layers.length === 3 && grown.pool.K.length === 3
                && grown.free.every((f) => f.recurrent.length === 3),
            `layers ${grown.w.layers.length}, pool ${grown.pool.K.length}, ` +
            `recurrent ${grown.free[0].recurrent.length}`);
        r.equal('and a grown Runner computes what a wide one computes',
            identical(gold, (await feed(grown, 'right')).hidden),
            'bit-identical after growing the high edge');
    }

    // --- growing left, onto a linear_attention position ---
    {
        const grown = new Runner(device, d, { layers: blocks.slice(1) },
            { lo: lo + 1, hi, have: hi, ...cfg });
        const reserved = grown.growLeft();
        r.equal(`growLeft reserved for ${blocks[0].kind}`,
            reserved?.kind === blocks[0].kind && Array.isArray(reserved.frames),
            `kind ${reserved?.kind}, ${reserved?.frames?.length} frames`);
        grown.prepend(blocks[0], reserved);
        r.equal('prepend keeps the arrays in step',
            grown.w.layers.length === 3 && grown.pool.K.length === 3
                && grown.free.every((f) => f.recurrent.length === 3),
            `layers ${grown.w.layers.length}, pool ${grown.pool.K.length}, ` +
            `recurrent ${grown.free[0].recurrent.length}, blockLo ${grown.blockLo}`);
        r.equal('and a grown Runner computes what a wide one computes',
            identical(gold, (await feed(grown, 'left')).hidden),
            'bit-identical after growing the low edge');
    }

    return { ok: r.rows.every((x2) => x2.ok), rows: r.rows, model: manifest.model,
             grew: [blocks[0].kind, blocks[2].kind], positions: [lo, hi] };
}
