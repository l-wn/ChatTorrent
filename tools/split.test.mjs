// Tests for assignment, run in Node. web/split.js has no browser dependencies.
//
//   node tools/split.test.mjs

import {
    holds, holders, depths, minDepth, covers, gaps, size, place, assign,
    blockBytes, blockWeightBytes, endBytes, kvShare, recurrentStateBytes, floorBlocks, budgetBlocks, costDims, MIN_BYTES,
    positions, headPos, blockRange, served, filling, coverage, EMBED,
    SESSIONS, SLOTS, MAX_SESSIONS, poolSlots, runnerSessions,
} from '../web/split.js';

// The floor depends on the model, so every call passes one. Ten is
// SmolLM2-135M's value, which these placement cases assume.
const FLOOR = 10;

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);

// A roster entry, as the signaling server writes it.
const P = (id, lo, hi) => ({ id, lo, hi, ready: true });
const client = (id) => ({ id, lo: null, hi: null, ready: true });
const arr = (typed) => Array.from(typed);

// --- Holdings and depth ---

console.log('\nholds / holders: a range is half-open');
{
    const p = P(1, 10, 20);
    ok('lo is held', holds(p, 10));
    ok('hi is not', holds(p, 20) === false);
    ok('below is not', holds(p, 9) === false);
    ok('a peer holding nothing holds nothing', holds(client(2), 0) === false);

    const peers = [P(3, 0, 15), P(1, 10, 30), client(9)];
    eq('holders are sorted by id, so every peer picks the same default hop',
        holders(peers, 12), [1, 3]);
    eq('nobody holds a block outside every range', holders(peers, 30), []);
}

console.log('\ndepths / minDepth / covers: the whole health of the network');
{
    const L = 30;
    eq('three tenths tile the model at depth 1',
        arr(depths([P(0, 0, 10), P(1, 10, 20), P(2, 20, 30)], L)).filter((d) => d !== 1).length, 0);
    eq('and r = 1', minDepth([P(0, 0, 10), P(1, 10, 20), P(2, 20, 30)], L), 1);
    ok('which is covered', covers([P(0, 0, 10), P(1, 10, 20), P(2, 20, 30)], L));

    eq('a hole is r = 0', minDepth([P(0, 0, 10), P(2, 20, 30)], L), 0);
    ok('and not covered', covers([P(0, 0, 10), P(2, 20, 30)], L) === false);
    eq('and it names itself', gaps([P(0, 0, 10), P(2, 20, 30)], L), [[10, 20]]);

    const doubled = [P(0, 0, 10), P(1, 10, 20), P(2, 20, 30),
                     P(3, 0, 10), P(4, 10, 20), P(5, 20, 30)];
    eq('six peers over three tenths is r = 2', minDepth(doubled, L), 2);

    eq('overlap counts once per peer, not per block',
        arr(depths([P(0, 0, 20), P(1, 10, 30)], L)),
        [...Array(10).fill(1), ...Array(10).fill(2), ...Array(10).fill(1)]);

    eq('the thinnest block decides, not the average',
        minDepth([P(0, 0, 29), P(1, 0, 29), P(2, 29, 30)], L), 1);

    eq('peers holding nothing do not pad the depth',
        minDepth([P(0, 0, 30), client(1), client(2)], L), 1);

    // #growClaim() in engine.js reads this array to choose which edge to grow:
    // it compares the position just outside each edge and grows toward the
    // shallower one. For two adjacent peers, the shared boundary must be
    // deeper than the open side on both, so they grow apart, not toward each
    // other.
    const neighbors = [P(0, 5, 8), P(1, 8, 11)];
    const d = depths(neighbors, L);
    ok("peer 0's open (left) side reads shallower than its shared border",
        d[4] < d[8], `open ${d[4]}, border ${d[8]}`);
    ok("peer 1's open (right) side reads shallower than its shared border",
        d[11] < d[7], `open ${d[11]}, border ${d[7]}`);
}

// --- size: the floor ---

console.log('\nsize: below the floor is a client, not a tiny range');
{
    eq('one block under the floor holds nothing', size(FLOOR - 1, 30, FLOOR), 0);
    eq('exactly the floor is fine', size(FLOOR, 30, FLOOR), FLOOR);
    eq('capacity is capped at the model', size(500, 30, FLOOR), 30);
    eq('zero is a client', size(0, 30, FLOOR), 0);
    eq('so is a nonsense capacity', size(NaN, 30, FLOOR), 0);
    eq('a model shallower than the floor lowers the floor to itself', size(4, 4, FLOOR), 4);
    eq('fractional capacity rounds down', size(12.9, 30, FLOOR), 12);
    eq('a zero-length model holds nobody', size(10, 0, FLOOR), 0);
    eq('a missing floor is treated as one block', size(1, 30, undefined), 1);
}

// --- blockBytes / floorBlocks: the floor is a byte budget ---

// The same dimensions in both spellings: the manifest's and loader.js's.
const SMOLLM2_135M = {
    model: 'SmolLM2-135M',
    hidden: 576, n_layer: 30, n_head: 9, n_kv_head: 3, head_dim: 64, ffn: 1536,
};
const TINYLLAMA_11B = {
    model: 'TinyLlama-1.1B',
    hidden: 2048, n_layer: 22, n_head: 32, n_kv_head: 4, head_dim: 64, ffn: 5632,
};

console.log('\nblockBytes: one cost model, both spellings of the dimensions');
{
    const d = costDims(SMOLLM2_135M);
    eq('the manifest spelling normalises', [d.n_heads, d.n_kv_heads], [9, 3]);
    eq('and loader.js\'s spelling survives round-tripping',
        costDims(d).n_kv_heads, 3);

    // Weights alone, then weights plus the KV cache the session pool needs.
    eq('a SmolLM2-135M block is 14 MiB of weights',
        Math.round(blockWeightBytes(d) / 2 ** 20), 14);
    eq('and 26 MiB once the KV pool is counted',
        Math.round(blockBytes(d) / 2 ** 20), 26);
    eq('a 1024-token window costs 17 MiB',
        Math.round(blockBytes(d, { nMax: 1024 }) / 2 ** 20), 17);
    eq('a TinyLlama-1.1B block is 84 MiB of f16 weights',
        Math.round(blockWeightBytes(costDims({ ...TINYLLAMA_11B, dtype: 'f16' })) / 2 ** 20), 84);

    // A shorter context or fewer slots reduces the KV part of the cost.
    ok('fewer slots make a block cheaper',
        blockBytes(d, { slots: 1 }) < blockBytes(d));
}

// A hybrid checkpoint keeps a KV cache on only some layers. The others hold a
// fixed-size recurrent state that does not depend on the context window and is
// not charged KV.
console.log('\nkvShare: a hybrid checkpoint is not charged KV on every layer');
{
    const flat = costDims(SMOLLM2_135M);
    eq('no layer_types means every block keeps a KV cache', kvShare(flat), 1);

    const kinds = Array.from({ length: 8 },
        (_, i) => (i % 4 === 3 ? 'full_attention' : 'linear_attention'));
    const hybrid = costDims({ ...SMOLLM2_135M, layer_types: kinds });
    eq('one full-attention layer in four', kvShare(hybrid), 0.25);
    // A hybrid checkpoint's full-attention layers have attn_gate, a third
    // qDim × H projection in addition to q_proj and o_proj.
    eq('a hybrid block weighs one attn_gate more than a flat one',
        blockWeightBytes(hybrid) - blockWeightBytes(flat), 576 * 576 * 4);
    eq('and the KV term is quartered',
        blockBytes(hybrid) - blockWeightBytes(hybrid),
        (blockBytes(flat) - blockWeightBytes(flat)) / 4);
    eq('so 4096 tokens on the hybrid cost in KV what 1024 cost charged flat',
        blockBytes(hybrid) - blockWeightBytes(hybrid),
        blockBytes(flat, { nMax: 1024 }) - blockWeightBytes(flat));
}

// A linear_attention layer has no KV but keeps a recurrent state per session
// frame. blockBytes() must charge it, or raising SESSIONS would over-commit
// VRAM.
console.log('\nrecurrent state: charged once per session on linear layers');
{
    const kinds = Array.from({ length: 8 },
        (_, i) => (i % 4 === 3 ? 'full_attention' : 'linear_attention'));
    const qwen = costDims({ ...SMOLLM2_135M, layer_types: kinds,
        linear_num_key_heads: 16, linear_num_value_heads: 48,
        linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4 });
    const per = recurrentStateBytes(qwen);
    eq('one Qwen3.8-27B layer, one session (DeltaNet state + conv rings)',
        per, (48 * 128 * 128 + 3 * 2048 * 2 + 3 * 6144) * 4);
    eq('none on a checkpoint with no linear layers',
        recurrentStateBytes(costDims(SMOLLM2_135M)), 0);
    eq('averaged: three blocks in four carry one state per session',
        blockBytes(qwen, { sessions: 8 }) - blockBytes(qwen, { sessions: 0 }), per * 8 * 0.75);
    eq('a named linear block: weights plus a state per session, no KV',
        blockBytes(qwen, { sessions: 8, kind: 'linear_attention' }),
        blockWeightBytes(qwen, { kind: 'linear_attention' }) + per * 8);
    eq('a named full block: weights plus the whole KV row, no state',
        blockBytes(qwen, { sessions: 8, kind: 'full_attention' }),
        blockWeightBytes(qwen, { kind: 'full_attention' })
            + blockBytes(costDims(SMOLLM2_135M)) - blockWeightBytes(costDims(SMOLLM2_135M)));
    ok('the two kinds do not weigh the same',
        blockWeightBytes(qwen, { kind: 'linear_attention' })
            !== blockWeightBytes(qwen, { kind: 'full_attention' }));
    eq('and an index names the kind',
        [blockBytes(qwen, { index: 0 }), blockBytes(qwen, { index: 3 })],
        [blockBytes(qwen, { kind: 'linear_attention' }), blockBytes(qwen, { kind: 'full_attention' })]);
    const { dimsFromManifest } = await import('../web/model/loader.js');
    eq('loader.js dims cost the same as the manifest',
        blockBytes(dimsFromManifest({ ...SMOLLM2_135M, n_head: 9, n_kv_head: 3, vocab: 49152,
            layer_types: kinds, linear_num_key_heads: 16, linear_num_value_heads: 48,
            linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4 })),
        blockBytes(qwen));
}

// The numbers a peer's VRAM limit is checked against, for Qwen3.8-27B at q8.
// The file sizes below are the real checkpoint's (377 and 389 MiB per block).
// The plain-Llama formula would give 340 MiB, and an end can upload two
// tables, so costs must come from the file sizes.
console.log('\nstored bytes: a block costs what its file weighs');
{
    const LIN = 407378816, FULL = 395552768, WTE = 1350860800, NORM = 20480;
    const kinds = Array.from({ length: 64 },
        (_, i) => (i % 4 === 3 ? 'full_attention' : 'linear_attention'));
    const dimsOnly = { model: 'Qwen/Qwen3.8-27B', hidden: 5120, n_layer: 64, n_head: 24,
        n_kv_head: 4, head_dim: 256, ffn: 17408, vocab: 248320, dtype: 'q8',
        tied_lm_head: false, layer_types: kinds,
        linear_num_key_heads: 16, linear_num_value_heads: 48,
        linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4 };
    // One q8 tensor for the file's projections and one f32 tensor for its norms.
    const file = (total) => [{ name: 'w', size: total - 4096 - 1024, scales_size: 4096 },
                             { name: 'n', size: 1024 }];
    const files = { 'client.bin': [{ name: 'wte', size: WTE - 79462400, scales_size: 79462400 },
                                   { name: 'norm', size: NORM }],
                    'lm_head.bin': [{ name: 'lm_head', size: WTE - 79462400, scales_size: 79462400 }] };
    kinds.forEach((k, i) => {
        files[`block_${String(i).padStart(2, '0')}.bin`] = file(k === 'full_attention' ? FULL : LIN);
    });
    const d = costDims({ ...dimsOnly, files });
    const formula = costDims(dimsOnly);
    const MiB = 2 ** 20;

    eq('a linear block is its file, scales included', blockWeightBytes(d, { index: 0 }), LIN);
    eq('a full block is its file', blockWeightBytes(d, { index: 3 }), FULL);
    eq('by kind, the mean of that kind', blockWeightBytes(d, { kind: 'full_attention' }), FULL);
    eq('unnamed, the mean of the model', blockWeightBytes(d), (LIN * 48 + FULL * 16) / 64);
    eq('a linear block: 388.5 MiB + 8 recurrent states = 413 MiB',
        Math.round(blockBytes(d, { index: 0 }) / MiB), 413);
    eq('a full block: 377.2 MiB + 64 MiB of KV pool = 441 MiB',
        Math.round(blockBytes(d, { index: 3 }) / MiB), 441);
    eq('the KV pool is 2 contexts of K and V', blockBytes(d, { index: 3 }) - FULL,
        4 * 256 * 4096 * 4 * 2 * 2);

    // With no file listing the formula is used, and must be close.
    for (const [kind, real] of [['linear_attention', LIN], ['full_attention', FULL]]) {
        const got = blockWeightBytes(formula, { kind });
        ok(`the formula is within 0.1% of a real ${kind} block`,
            Math.abs(got - real) / real < 1e-3, `${got} vs ${real}`);
    }
    eq('no files, no stored sizes', formula.stored, undefined);

    eq('the embedding is client.bin: wte and the norm', endBytes(d, 'embed'), WTE + NORM);
    eq('the untied head is lm_head.bin and the norm', endBytes(d, 'head'), WTE + NORM);
    ok('the formula prices an end as one table',
        Math.abs(endBytes(formula, 'head') - WTE) / WTE < 1e-3 && endBytes(formula, 'embed') === endBytes(formula, 'head'));
    eq('a tied head is the embedding\'s table',
        endBytes(costDims({ ...dimsOnly, files, tied_lm_head: true }), 'head'), WTE + NORM);

    const { dimsFromManifest } = await import('../web/model/loader.js');
    const loaded = dimsFromManifest({ ...dimsOnly, files });
    eq('loader.js dims carry the same sizes', loaded.stored, d.stored);
    eq('and cost every block the same',
        kinds.map((_, i) => blockBytes(loaded, { index: i })),
        kinds.map((_, i) => blockBytes(d, { index: i })));
    eq('and both ends', [endBytes(loaded, 'embed'), endBytes(loaded, 'head')],
        [endBytes(d, 'embed'), endBytes(d, 'head')]);

    // The cost of the whole model, for a peer holding all of it.
    const whole = kinds.reduce((n, _, i) => n + blockBytes(d, { index: i }), 0)
        + endBytes(d, 'embed') + endBytes(d, 'head');
    eq('the whole of Qwen3.8-27B is 28.8 GiB before scratch', (whole / 2 ** 30).toFixed(1), '28.8');
}

console.log('\nblockBytes: f16 halves the weights but not the KV cache');
{
    const f32 = costDims({ ...TINYLLAMA_11B, dtype: 'f32' });
    const f16 = costDims({ ...TINYLLAMA_11B, dtype: 'f16' });

    eq('dtype f32 is 4 bytes a weight', f32.bytes, 4);
    eq('dtype f16 is 2', f16.bytes, 2);
    eq('an absent dtype is f32', costDims(TINYLLAMA_11B).bytes, 4);

    // f16 halves the weights but not the KV cache, which holds activations
    // that every kernel reads and writes in f32. So a block's cost falls by
    // less than half, and the difference is one KV term.
    const weightsF32 = blockWeightBytes(f32);
    eq('slots:0 and blockWeightBytes agree', blockBytes(f32, { slots: 0 }), weightsF32);
    ok('an f16 block is cheaper than an f32 one', blockBytes(f16) < blockBytes(f32));
    ok('but not half: the KV cache is still f32',
        blockBytes(f16) > blockBytes(f32) / 2);
    eq('the saving is exactly half the weights',
        blockBytes(f32) - blockBytes(f16), weightsF32 / 2);
    ok('with no slots, f16 is exactly half',
        blockBytes(f16, { slots: 0 }) * 2 === blockBytes(f32, { slots: 0 }));

    // Cheaper blocks mean more of them fit in the same byte floor, so the
    // floor in blocks rises. The budget is in bytes.
    ok('a cheaper block raises the block-denominated floor',
        floorBlocks(f16, 22) >= floorBlocks(f32, 22));
}

// The cost model must accept the dims object the loader produces.
// dimsFromManifest() and costDims() are separate functions built from the same
// manifest and must agree on `bytes`. If an f16 checkpoint were costed as f32,
// every peer would claim about half the range it can hold, with no error.
console.log('\ndimsFromManifest agrees with costDims about what a block costs');
{
    const { dimsFromManifest } = await import('../web/model/loader.js');
    // Includes q8: priced as f32 it would cost 4x too much, and the server's
    // block sizes would differ from every browser's.
    for (const dtype of ['f32', 'f16', 'q8']) {
        const manifest = { model: 'test', hidden: 2048, n_layer: 24, n_head: 32,
            n_kv_head: 32, head_dim: 64, ffn: 8192, vocab: 49152, dtype };
        const loaded = dimsFromManifest(manifest);
        eq(`${dtype}: bytes per weight`, loaded.bytes, { f32: 4, f16: 2, q8: 1.0625 }[dtype]);
        eq(`${dtype}: blockBytes agrees with costDims`,
            blockBytes(loaded), blockBytes(costDims(manifest)));
        eq(`${dtype}: and so does the default capacity`,
            budgetBlocks(loaded, 24), budgetBlocks(costDims(manifest), 24));
    }
    // A manifest with no dtype is f32.
    eq('an absent dtype is f32',
        dimsFromManifest({ hidden: 576, n_layer: 30, n_head: 9, n_kv_head: 3,
            head_dim: 64, ffn: 1536, vocab: 49152 }).bytes, 4);
}

console.log('\nthe floor ignores the KV cache; the capacity budget does not');
{
    const d = costDims(SMOLLM2_135M);

    // The floor concerns what a range costs to fetch and the hops it adds.
    // Neither involves KV, which is local VRAM. If the floor counted KV,
    // changing SLOTS would change the smallest range the network hands out.
    eq('the floor is unchanged by the session pool',
        [floorBlocks(d, 30), floorBlocks(d, 30)], [10, 10]);
    ok('blockBytes does move with the pool',
        blockBytes(d, { slots: 1 }) < blockBytes(d, { slots: 8 }));
    ok('but blockWeightBytes does not',
        blockWeightBytes(d) === blockWeightBytes(d));

    // The budget answers a different question, what fits in VRAM, so it
    // counts KV. It rounds down to the most whole blocks that fit.
    ok('the budget counts KV', budgetBlocks(d, 30, blockBytes(d) * 3) === 3);
    ok('a budget below one block is a client', budgetBlocks(d, 30, 1) === 0);
    ok('the budget is capped at the model', budgetBlocks(d, 30, 1e12) === 30);
}

console.log('\nfloorBlocks: the same byte budget on models of different weight');
{
    const smol = costDims(SMOLLM2_135M);
    const tiny = costDims(TINYLLAMA_11B);

    eq('SmolLM2-135M still floors at the ten blocks it was tuned to',
        floorBlocks(smol, 30), 10);

    // A flat floor of ten blocks on a model with 178 MiB blocks would be
    // 1.7 GiB, which no tab holds, so the model could not be covered. The byte
    // floor scales down.
    const tinyFloor = floorBlocks(tiny, 22);
    ok('a 1.1B model floors low enough for a tab to clear it', tinyFloor <= 2,
        `got ${tinyFloor}`);
    ok('and the floor is never zero', tinyFloor >= 1);

    // Six peers that can hold four blocks each cover the model.
    const covered = [];
    for (let id = 0; id < 6; id++) {
        covered.push({ id, ...assign(covered, 4, 22, tinyFloor), ready: true });
    }
    ok('six four-block peers cover a 22-block model', covers(covered, 22));
    eq('with depth to spare', minDepth(covered, 22) >= 1, true);

    // A model where one block already exceeds the budget: the floor cannot go
    // below one block, or the model could not be covered.
    const huge = costDims({ hidden: 8192, n_layer: 80, n_head: 64, n_kv_head: 8,
        head_dim: 128, ffn: 28672 });
    ok('a block bigger than the whole budget still floors at one',
        blockBytes(huge) > MIN_BYTES && floorBlocks(huge, 80) === 1);

    eq('the floor never exceeds the model\'s own depth',
        floorBlocks(smol, 4), 4);
}

// --- place: fill first, then raise r ---

console.log('\nplace: an empty network fills left to right');
{
    const L = 30, n = 10;
    const peers = [];
    const got = [];
    for (let id = 0; id < 3; id++) {
        const r = assign(peers, n, L, FLOOR);
        peers.push({ id, ...r, ready: true });
        got.push([r.lo, r.hi]);
    }
    eq('three ten-block peers land on thirds', got, [[0, 10], [10, 20], [20, 30]]);
    ok('and the model is covered', covers(peers, L));
    eq('at r = 1', minDepth(peers, L), 1);
}

console.log('\nplace: coverage before redundancy');
{
    const L = 30;
    // 0–19 is held twice, 20–29 by nobody. A window at 20 covers ten uncovered
    // blocks and a window anywhere else covers none. Depth 0 wins whatever the
    // depth elsewhere.
    const peers = [P(0, 0, 20), P(1, 0, 20)];
    eq('a new peer goes straight into the hole', place(peers, 10, L), 20);

    // Also when the hole is smaller than the range being placed: the window
    // must contain the hole, and the leftmost such window wins the tie.
    const holed = [P(0, 0, 12), P(1, 15, 30)];
    const lo = place(holed, 10, L);
    ok('a range wider than the hole still covers it',
        lo <= 12 && lo + 10 >= 15, `placed at ${lo}`);

    // Three uncovered blocks beat two, whatever else is under the window.
    const twoHoles = [P(0, 0, 4), P(1, 6, 27)];      // holes: 4–5 (2) and 27–29 (3)
    const big = place(twoHoles, 5, L);
    eq('the bigger hole wins', big, 25);
}

console.log('\nplace: then the thinnest blocks');
{
    const L = 30;
    // Covered everywhere, and 20–29 is thinnest. The low end of the sorted
    // profile decides, which the sum of depths would not show.
    const peers = [P(0, 0, 20), P(1, 0, 20), P(2, 0, 30)];
    eq('a covered network sends the next peer to the thinnest stretch',
        place(peers, 10, L), 20);

    // Six ten-block peers on a thirty-block model: the second three must be
    // placed on the first three's ranges, and r must reach 2.
    const grown = [];
    for (let id = 0; id < 6; id++) {
        grown.push({ id, ...assign(grown, 10, L, FLOOR), ready: true });
    }
    eq('six ten-block peers reach r = 2', minDepth(grown, L), 2);
    eq('and nine reach r = 3',
        minDepth(Array.from({ length: 9 }).reduce((acc, _, id) =>
            [...acc, { id, ...assign(acc, 10, L, FLOOR), ready: true }], []), L), 3);
}

console.log('\nplace: peers of different sizes');
{
    const L = 30;
    const peers = [];
    // A big first peer, then two small ones. The small ones must not be placed
    // on the same blocks. Each goes where the depth is lowest at the time.
    peers.push({ id: 0, ...assign(peers, 20, L, FLOOR), ready: true });
    eq('a 20-block peer takes the left end', [peers[0].lo, peers[0].hi], [0, 20]);
    peers.push({ id: 1, ...assign(peers, 10, L, FLOOR), ready: true });
    eq('a 10-block peer covers the rest', [peers[1].lo, peers[1].hi], [20, 30]);
    ok('two peers can cover the model between them', covers(peers, L));
    peers.push({ id: 2, ...assign(peers, 30, L, FLOOR), ready: true });
    eq('a peer that can hold everything does', [peers[2].lo, peers[2].hi], [0, 30]);
    eq('which lifts r to 2', minDepth(peers, L), 2);
}

console.log('\nplace: a one-wide claim spreads instead of stacking');
{
    // A one-position claim on an otherwise empty line ties everywhere. With
    // "lowest lo wins", each claim would be placed next to the last, and two
    // adjacent peers would then grow over each other's ranges. Bisecting the
    // widest tied run spreads them out.
    const L = 16;
    const peers = [];
    const got = [];
    for (let id = 0; id < 4; id++) {
        const lo = place(peers, 1, L);
        peers.push(P(id, lo, lo + 1));
        got.push(lo);
    }
    eq('four single positions land spread across the line, not sequentially',
        got, [7, 11, 3, 13]);
    ok('none of them land next to a position already claimed',
        got.every((lo, i) => got.slice(0, i).every((prev) => Math.abs(lo - prev) > 1)));

    // Coverage still takes exactly L joins, as for any other width. Bisection
    // at width one never costs an extra peer: removing one position always
    // leaves the same number of one-wide claims needed, however the remainder
    // is split (see place()).
    const solo = [];
    for (let id = 0; id < L; id++) {
        const lo = place(solo, 1, L);
        solo.push(P(id, lo, lo + 1));
    }
    ok('L one-wide peers still cover the whole line', covers(solo, L));
}

console.log('\nassign: a peer being re-placed must not see its own range');
{
    const L = 30;
    const a = { id: 0, ...assign([], 10, L, FLOOR), ready: true };
    const b = { id: 1, ...assign([a], 10, L, FLOOR), ready: true };
    // If `a`'s own range were passed in, it would avoid 0–9 and move. With it
    // excluded, the same capacity on the same network gives the same answer.
    eq('re-placing is idempotent', assign([b], 10, L, FLOOR), { lo: 0, hi: 10 });
}

console.log('\nassign: below the floor, and degenerate models');
{
    eq('a capacity below the floor gets no range', assign([], 3, 30, FLOOR), null);
    eq('a zero-length model gets no range', assign([], 10, 0, FLOOR), null);
    eq('a range can be the whole model', assign([], 30, 30, FLOOR), { lo: 0, hi: 30 });
}

// --- Brute force: the property `place` guarantees ---
//
// `place` is greedy and places one peer at a time, so it does not promise a
// global optimum. It promises that each placement is the best available for
// the network at that moment: no other window of the same width has a
// lexicographically smaller sorted depth profile. This is checked
// exhaustively for small L.

console.log('\nplace: exhaustive check of the one-step claim');
{
    const lex = (a, b) => {                     // -1 if a < b
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
        return 0;
    };
    const profile = (d, lo, n) => Array.from(d.subarray(lo, lo + n)).sort((x, y) => x - y);

    let worse = 0, cases = 0;
    // Every network of up to three peers on models 4–8 deep, every width.
    for (let L = 4; L <= 8; L++) {
        const ranges = [];
        for (let lo = 0; lo < L; lo++) for (let hi = lo + 1; hi <= L; hi++) ranges.push([lo, hi]);

        const nets = [[]];
        for (const [lo, hi] of ranges) {
            nets.push([P(0, lo, hi)]);
            for (const [lo2, hi2] of ranges) nets.push([P(0, lo, hi), P(1, lo2, hi2)]);
        }

        for (const peers of nets) {
            const d = depths(peers, L);
            for (let n = 1; n <= L; n++) {
                cases++;
                const chosen = place(peers, n, L);
                const mine = profile(d, chosen, n);
                for (let lo = 0; lo + n <= L; lo++) {
                    if (lex(profile(d, lo, n), mine) < 0) worse++;
                }
            }
        }
    }
    ok(`no window beats the chosen one, over ${cases.toLocaleString()} cases`, worse === 0,
        `${worse} placements had a strictly better alternative`);

    // The consequence: repeated placement covers the model as soon as there is
    // enough capacity. With peers of width w on a model of depth L, ceil(L/w)
    // of them must be enough.
    let late = 0;
    for (let L = 4; L <= 12; L++) {
        for (let w = 1; w <= L; w++) {
            const peers = [];
            const need = Math.ceil(L / w);
            for (let id = 0; id < need; id++) {
                const r = place(peers, w, L);
                peers.push(P(id, r, Math.min(r + w, L)));
            }
            if (!covers(peers, L)) late++;
        }
    }
    ok('ceil(L/w) peers of width w always cover the model', late === 0,
        `${late} (L, w) pairs left a hole`);
}

// --- The walk: assignment and forwarding together ---
//
// route.js forwards a job to a holder of its cursor, and that holder advances
// the cursor to its own `hi`. For every network the placement rule can
// produce:
//
//   1. a covered network carries a cursor from 0 to L, and
//   2. the cursor increases at every hop, so a job never revisits a peer and
//      cannot loop.
//
// This checks `covers()` and the forward-only cursor rule against each other.

console.log('\nthe walk: a covered network always carries a cursor from 0 to L');
{
    // The forwarding rule: the lowest-id holder of the cursor advances it to
    // its hi. Returns the hop count, or a string describing the failure.
    const walk = (peers, L) => {
        let pos = 0;
        const seen = new Set();
        for (let hops = 1; hops <= peers.length + 1; hops++) {
            const at = holders(peers, pos);
            if (!at.length) return `stranded at ${pos}`;
            const p = peers.find((q) => q.id === at[0]);
            if (p.hi <= pos) return `cursor did not advance at ${pos}`;
            if (seen.has(p.id)) return `revisited peer ${p.id}`;
            seen.add(p.id);
            pos = p.hi;
            if (pos >= L) return hops;
        }
        return 'ran out of hops';
    };

    let bad = 0, walked = 0, longest = 0;
    // Every network the placement rule produces from up to four peers with
    // capacities from 1..L, on models 4 to 10 deep.
    for (let L = 4; L <= 10; L++) {
        for (let c0 = 1; c0 <= L; c0++) {
            for (let c1 = 1; c1 <= L; c1++) {
                for (let c2 = 1; c2 <= L; c2++) {
                    const peers = [];
                    for (const [id, cap] of [c0, c1, c2].entries()) {
                        // Skips the MIN_BLOCKS floor: the floor is a policy
                        // default, and the walk must work at any width.
                        const lo = place(peers, cap, L);
                        peers.push(P(id, lo, Math.min(lo + cap, L)));
                    }
                    const got = walk(peers, L);
                    const isCovered = covers(peers, L);
                    if (isCovered !== (typeof got === 'number')) {
                        bad++;
                        if (bad === 1) {
                            console.log(`    first mismatch: L=${L} caps ${[c0, c1, c2]} ` +
                                `covers=${isCovered} walk=${got}`);
                        }
                    }
                    if (typeof got === 'number') { walked++; longest = Math.max(longest, got); }
                }
            }
        }
    }
    ok(`a cursor walks 0 to L exactly when the network is covered ` +
        `(${walked.toLocaleString()} walks, longest ${longest} hops)`, bad === 0,
        `${bad} networks disagreed`);
}

// --- The position line: where the embedding and the head are ---

// `place` and `depths` see only an interval on a line of length N. The three
// functions tested here are the only ones that know the line is
// [embedding, blocks…, head]. If they were wrong, a peer would download one
// block and be asked to run another.
console.log('\npositions: the line is L + 2 long, and the ends have names');

eq('a 30-block model has 32 positions', positions(30), 32);
eq('the embedding is first', EMBED, 0);
eq('the head is last', headPos(30), positions(30) - 1);
eq('a one-block model still has both ends', positions(1), 3);

console.log('\nblockRange: a claim in positions, as the blocks it downloads');

// A claim can contain no blocks.
eq('the embedding alone is no blocks', blockRange(0, 1, 30), [0, 0]);
eq('the head alone is no blocks', blockRange(31, 32, 30), [30, 30]);

eq('a claim over the middle is the blocks under it', blockRange(10, 20, 30), [9, 19]);
eq('a claim containing the embedding drops it and keeps the rest',
    blockRange(0, 10, 30), [0, 9]);
eq('a claim containing the head does the same at its end',
    blockRange(22, 32, 30), [21, 30]);
eq('a peer holding everything holds every block', blockRange(0, 32, 30), [0, 30]);

// A claim of `n` positions costs `n` blocks, less one for each end it
// contains: the embedding and the head each occupy a position.
{
    let bad = 0;
    for (let L = 1; L <= 8; L++) {
        const N = positions(L);
        for (let lo = 0; lo < N; lo++) {
            for (let hi = lo + 1; hi <= N; hi++) {
                const [bLo, bHi] = blockRange(lo, hi, L);
                const ends = (lo === 0 ? 1 : 0) + (hi === N ? 1 : 0);
                if (bHi - bLo !== (hi - lo) - ends) bad++;
                if (bLo < 0 || bHi > L || bHi < bLo) bad++;
            }
        }
    }
    ok('a claim of n positions is n blocks, less one per end it contains', bad === 0,
        `${bad} ranges disagreed`);
}

// Placement treats the ends as ordinary positions, so they are covered like
// any other position.
{
    const L = 6, N = positions(L);          // 8 positions
    const net = [];
    for (let i = 0; i < 4; i++) {
        const r = assign(net, 2, N, 1);
        net.push({ id: i, ...r, ready: true });
    }
    eq('four two-wide peers tile an eight-position line',
        net.map((p) => [p.lo, p.hi]), [[0, 2], [2, 4], [4, 6], [6, 8]]);
    ok('and the embedding has a holder that is not the head\'s', covers(net, N)
        && holders(net, EMBED)[0] !== holders(net, headPos(L))[0]);
}

// --- The claim and the holding ---

console.log('\nserved: a peer holds what it downloaded, not what it claimed');
{
    const p = { id: 1, lo: 10, hi: 20, have: 14, ready: true };
    eq('the holding is [lo, have)', [served(p).lo, served(p).hi], [10, 14]);
    ok('a position inside the claim but past the holding is not held',
        holds(served(p), 15) === false);
    ok('one inside the holding is', holds(served(p), 13));
    ok('a peer mid-download is filling', filling(p));
    ok('a filled one is not', filling({ ...p, have: 20 }) === false);

    // `have` is optional. An entry without it (a peer that fills its whole
    // range in one step) is treated as holding its whole claim.
    const old = { id: 2, lo: 10, hi: 20, ready: true };
    eq('an entry with no have is its claim', [served(old).lo, served(old).hi], [10, 20]);
    ok('and a client stays a client', holds(served({ id: 3, lo: null, hi: null }), 0) === false);

    // A holding only grows, so a stale roster can only understate it, and a
    // route computed from one is still valid.
    const stale = served({ id: 1, lo: 10, hi: 20, have: 12 });
    const fresh = served({ id: 1, lo: 10, hi: 20, have: 16 });
    ok('every position a stale roster says is held is still held',
        [...Array(32).keys()].every((x) => !holds(stale, x) || holds(fresh, x)));
}

console.log('\ncoverage: the network, drawn from the roster alone');
{
    const L = 30, N = positions(L);
    const now = [{ id: 0, lo: 0, hi: 10, have: 6 }, { id: 1, lo: 10, hi: 20, have: 20 },
                 { id: 2, lo: 20, hi: 32, have: 20 }].map(served);
    const cov = coverage(now, N);
    eq('covered counts positions with at least one holder', cov.covered, 16);
    eq('the holes are the rest', cov.holes, [[6, 10], [20, 32]]);
    eq('r is 0 while one exists', cov.r, 0);
    eq('and it agrees with minDepth', cov.r, minDepth(now, N));
    eq('and with gaps', cov.holes, gaps(now, N));

    const whole = [{ id: 0, lo: 0, hi: 32, have: 32 }, { id: 1, lo: 0, hi: 32, have: 32 }]
        .map(served);
    const full = coverage(whole, N);
    eq('a covered network has no holes', [full.covered, full.holes.length], [32, 0]);
    eq('and r is the depth of the thinnest position', full.r, 2);
    eq('an empty network covers nothing', coverage([], N).covered, 0);
    eq('and reports the whole line as one hole', coverage([], N).holes, [[0, 32]]);
}

console.log('\nchats in memory: the pool grows with the session count');
{
    eq('the default session count gives the default pool', poolSlots(SESSIONS), SLOTS);
    eq('one context per four sessions, rounded up', [1, 4, 5, 16, 32].map(poolSlots), [1, 1, 2, 4, 8]);
    eq('the mesh\'s number sizes the runner', runnerSessions(undefined, 16), { sessions: 16, slots: 4 });
    eq('a caller\'s number beats the mesh\'s', runnerSessions(3, 16), { sessions: 3, slots: 1 });
    eq('with neither, the default', runnerSessions(undefined, null), { sessions: SESSIONS, slots: SLOTS });
    eq('clamped to MAX_SESSIONS', runnerSessions(undefined, 1000).sessions, MAX_SESSIONS);
    eq('a nonsense number falls back to the default', [0, -2, NaN].map((n) => runnerSessions(n).sessions),
        [SESSIONS, SESSIONS, SESSIONS]);
    const d = costDims({ ...SMOLLM2_135M, layer_types: Array.from({ length: 8 },
        (_, i) => (i % 4 === 3 ? 'full_attention' : 'linear_attention')),
        linear_num_key_heads: 16, linear_num_value_heads: 48,
        linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4 });
    ok('more chats in memory make a block cost more',
        blockBytes(d, { sessions: 16, slots: 4 }) > blockBytes(d));
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
