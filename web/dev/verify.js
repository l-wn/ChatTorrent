// Diffs the real forward pass against the HuggingFace oracle, stage by stage
// in dependency order: embedding, block 0's sub-steps, every block, final
// norm, logits, argmax, then a greedy decode.
//
// The stack is run one layer at a time, with a submit and readback between
// layers, and the scratch buffers are read directly. After block 0 returns,
// `s` holds that block's intermediates (s.q, s.qh, s.attn, s.gate, …), so one
// block call gives ten comparisons without instrumenting model.js.
//
// Two intermediates are not in a buffer at the end of the block and are
// computed separately:
//   ln1               recomputed with a standalone rmsnorm before the block
//   after_attn_resid  x - down, since the MLP's residual add is x += down
//
// Tolerances are per stage. f32 matmul error grows with K and with layer
// count, so per-block tolerances are loose. The final checks (argmax and
// greedy decode) are exact.

import { F32, readF32, uploadF32, rmsnorm, matmulNobias, gatherEmbedNoPos, withPass, isQ8, q8Of } from '../model/kernels.js';
import { block, makeCache, makeScratch } from '../model/model.js';
import { loadNpy } from './npy.js';
import { generate } from '../model/generate.js';
import { decode } from '../model/tokenizer.js';
import { loadClient, loadBlock } from '../model/loader.js';

export const ORACLE_URL = '/oracle/smollm2';

// --- Diffing ---

export function diff(got, want) {
    if (got.length !== want.length) return { bad: `length ${got.length}, want ${want.length}` };
    let maxAbs = 0, at = -1, sum = 0, scale = 0;
    for (let i = 0; i < got.length; i++) {
        if (!Number.isFinite(got[i])) return { bad: `non-finite ${got[i]} at ${i}` };
        const e = Math.abs(got[i] - want[i]);
        sum += e;
        if (e > maxAbs) { maxAbs = e; at = i; }
        if (Math.abs(want[i]) > scale) scale = Math.abs(want[i]);
    }
    return { maxAbs, mean: sum / got.length, at, scale };
}

export function makeReporter() {
    const rows = [];
    return {
        rows,
        // Passes when the max error is within atol + rtol * max|want|, which
        // scales with the tensor's magnitude. An absolute tolerance alone does
        // not work on Llama's residual stream: channel 507 reaches ~19000 by
        // block 11 (the known massive-activation dimension), where a drift of
        // 2e-2 is 1e-6 relative and exact for f32. The same 2e-2 on the ±20
        // stream at block 0 would be a bug.
        close(name, got, want, { atol, rtol = 0 }) {
            const d = diff(got, want);
            if (d.bad) { rows.push({ name, ok: false, detail: d.bad }); return false; }
            const limit = atol + rtol * d.scale;
            const ok = d.maxAbs <= limit;
            rows.push({
                name, ok,
                detail: `max ${d.maxAbs.toExponential(2)} of ${limit.toExponential(1)}` +
                    ` (|x|<=${d.scale.toExponential(1)})  mean ${d.mean.toExponential(2)}` +
                    (ok ? '' : `  FIRST WORST at index ${d.at}`),
            });
            return ok;
        },
        equal(name, ok, detail) { rows.push({ name, ok, detail }); return ok; },
    };
}

// --- Oracle ---

async function loadJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}. Run the oracle dump script for this checkpoint.`);
    return r.json();
}

// Fetches everything the stages below need (logits is the largest).
//
// A hybrid (Qwen-style) manifest skips the block00_* fixtures, which assume a
// full-attention block 0. Its dump has only per-block outputs. Per-kernel
// hybrid checks are in verify_qwen_spike.js and verify_qwen_dispatch.js,
// against oracle/qwen3.8-27b-spike/.
export async function loadOracle(d, base = ORACLE_URL) {
    const names = d.hybrid
        ? ['embeddings', 'final_norm', 'logits']
        : [
            'embeddings', 'block00_ln1', 'block00_q', 'block00_k', 'block00_v',
            'block00_q_rope', 'block00_k_rope', 'block00_attn_out',
            'block00_after_attn_resid', 'block00_ln2', 'block00_gate', 'block00_up',
            'block00_mlp_out', 'final_norm', 'logits',
        ];
    for (let i = 0; i < d.n_layer; i++) names.push(`block_${String(i).padStart(2, '0')}_out`);

    const arrays = {};
    await Promise.all(names.map(async (n) => { arrays[n] = await loadNpy(`${base}/${n}.npy`); }));

    return {
        ...arrays,
        tokens: await loadJson(`${base}/tokens.json`),
        nextToken: await loadJson(`${base}/next_token.json`),
        greedy: await loadJson(`${base}/decode_greedy.json`),
    };
}

// --- Stages ---

// { atol, rtol } per stage. makeReporter().close explains why both are needed.
const TOL_F32 = {
    embed:  { atol: 1e-6 },                // a gather: exact up to the f32 round trip
    rope:   { atol: 1e-4, rtol: 1e-5 },
    block0: { atol: 1e-3, rtol: 1e-5 },    // one block of f32 matmul drift
    // A looser rtol than block0: drift accumulates over many layers, and with
    // the massive-activation channel the rtol term dominates. Real bugs give
    // errors about 100× higher.
    block:  { atol: 1e-3, rtol: 5e-5 },
    norm:   { atol: 1e-3, rtol: 1e-5 },    // ln_f renormalizes, which removes most residual drift
    logits: { atol: 5e-3, rtol: 1e-5 },    // a 576-term reduction × 49152 outputs
};

// q8 has its own tolerances. int8 weights have ~0.4% error by construction,
// and it compounds through the layers. On SmolLM2-135M the relative error is
// under 1% through block 27, ~7% by block 29, ~23% after the final RMSNorm and
// ~31% at the logits. Each tolerance is about 1.5x its measured value.
//
// The pass/fail checks for a q8 build are the two exact ones at the end,
// argmax and the 20-step greedy decode, which must match HuggingFace for any
// dtype. These tolerances only show where a difference started.
const TOL_Q8 = {
    embed:  { atol: 1e-5, rtol: 1e-2 },
    rope:   { atol: 1e-4, rtol: 1e-2 },
    block0: { atol: 1e-3, rtol: 1e-2 },
    block:  { atol: 1e-3, rtol: 1.2e-1 },
    norm:   { atol: 1e-3, rtol: 3.5e-1 },
    logits: { atol: 5e-3, rtol: 4.5e-1 },
};

// Runs every stage and returns { rows, ok }. `w` must be the real loaded weights.
export async function verifyAgainstOracle(device, d, w, oracle,
                                          { maxDecodeSteps = 20, dtype = 'f32' } = {}) {
    const r = makeReporter();
    const TOL = dtype === 'q8' ? TOL_Q8 : TOL_F32;
    const H = d.hidden, D = d.head_dim;

    const ids = oracle.tokens.input_ids;
    const T = ids.length;
    const N_max = T + maxDecodeSteps;

    const cache = makeCache(device, d, N_max);
    const s = makeScratch(device, d, T, N_max);
    const idsBuf = uploadF32(device, Float32Array.from(ids));

    const run = (label, record) => {
        const enc = device.createCommandEncoder({ label });
        withPass(enc, record, label);
        device.queue.submit([enc.finish()]);
    };
    const read = (buf, floats) => readF32(device, buf, floats * F32);

    // --- 1. embedding ---
    run('embed', (pass) => gatherEmbedNoPos(device, pass, idsBuf, w.wte, s.x, T, H));
    r.close('embeddings', await read(s.x, T * H), oracle.embeddings.data, TOL.embed);

    // --- 2. block 0 ---
    // A hybrid (Qwen-style) manifest skips the sub-stage diffs below, which
    // assume a full-attention block 0 (see loadOracle), and compares only
    // block_00's output.
    if (!d.hybrid) {
        run('ln1', (pass) => rmsnorm(device, pass, s.x, w.layers[0].input_layernorm, s.h, T, H));
        r.close('block00 rmsnorm (input_layernorm)', await read(s.h, T * H),
            oracle.block00_ln1.data, TOL.block0);
    }

    run('block00', (pass) => block(device, pass, d, w, 0, cache, s, T, 0));

    if (!d.hybrid) {
        const kvDim = d.n_kv_heads * D;
        r.close('block00 q_proj', await read(s.q, T * d.n_heads * D), oracle.block00_q.data, TOL.block0);
        r.close('block00 k_proj', await read(s.k, T * kvDim), oracle.block00_k.data, TOL.block0);
        r.close('block00 v_proj', await read(s.v, T * kvDim), oracle.block00_v.data, TOL.block0);

        r.close('block00 rope(Q)', await read(s.qh, d.n_heads * T * D),
            oracle.block00_q_rope.data, TOL.rope);
        // The K cache is [n_kv, N_max, D] with only rows 0..T written, so it
        // is compacted to [n_kv, T, D] before the diff with the oracle's dense
        // array.
        const kCache = await read(cache.K[0], d.n_kv_heads * N_max * D);
        const kDense = new Float32Array(d.n_kv_heads * T * D);
        for (let h = 0; h < d.n_kv_heads; h++)
            kDense.set(kCache.subarray(h * N_max * D, h * N_max * D + T * D), h * T * D);
        r.close('block00 rope(K cache)', kDense, oracle.block00_k_rope.data, TOL.rope);

        r.close('block00 attn out (o_proj)', await read(s.attn, T * H),
            oracle.block00_attn_out.data, TOL.block0);
        r.close('block00 rmsnorm (post_attention)', await read(s.h, T * H),
            oracle.block00_ln2.data, TOL.block0);
        r.close('block00 gate_proj', await read(s.gate, T * d.ffn), oracle.block00_gate.data, TOL.block0);
        r.close('block00 up_proj', await read(s.up, T * d.ffn), oracle.block00_up.data, TOL.block0);

        const down = await read(s.down, T * H);
        r.close('block00 mlp out (down_proj)', down, oracle.block00_mlp_out.data, TOL.block0);

        const x0 = await read(s.x, T * H);
        // x includes both residual adds. Subtracting the MLP output gives the
        // input of ln_2, which HF's pre-hook captured.
        r.close('block00 residual after attn', x0.map((v, i) => v - down[i]),
            oracle.block00_after_attn_resid.data, TOL.block0);
        r.close('block_00 out', x0, oracle.block_00_out.data, TOL.block0);
    } else {
        r.close('block_00 out', await read(s.x, T * H), oracle.block_00_out.data, TOL.block0);
    }

    // --- 3. the remaining blocks ---
    for (let i = 1; i < d.n_layer; i++) {
        run(`block${i}`, (pass) => block(device, pass, d, w, i, cache, s, T, 0));
        const name = `block_${String(i).padStart(2, '0')}`;
        r.close(`${name} out`, await read(s.x, T * H), oracle[`${name}_out`].data, TOL.block);
    }

    // --- 4. final norm + LM head ---
    run('ln_f', (pass) => rmsnorm(device, pass, s.x, w.norm, s.h, T, H));
    r.close('final_norm', await read(s.h, T * H), oracle.final_norm.data, TOL.norm);

    // Usually the embedding table (tied). An untied checkpoint's own lm_head
    // takes precedence, as in model.js's forward().
    run('lm_head', (pass) => matmulNobias(device, pass, s.h, w.lm_head ?? w.wte, s.logits, T, H, d.vocab));
    const logits = await read(s.logits, T * d.vocab);
    r.close('logits', logits, oracle.logits.data, TOL.logits);

    // --- 5. argmax at the last position ---
    const lastRow = logits.subarray((T - 1) * d.vocab, T * d.vocab);
    let argmax = 0;
    for (let i = 1; i < d.vocab; i++) if (lastRow[i] > lastRow[argmax]) argmax = i;
    r.equal('next token argmax', argmax === oracle.nextToken.next_id,
        `got ${argmax} ${JSON.stringify(await decode([argmax]))},` +
        ` want ${oracle.nextToken.next_id} ${JSON.stringify(oracle.nextToken.decoded)}`);

    // --- 6. greedy decode: the pass/fail check ---
    // Uses a new cache and scratch: generate() starts from an empty cache, and
    // the one above holds the prefill.
    const want = oracle.greedy.generated_ids.slice(0, maxDecodeSteps);
    const gen = await generate(device, d, w, makeCache(device, d, N_max),
        makeScratch(device, d, T, N_max), ids, { maxNewTokens: want.length });
    const same = gen.ids.length === want.length && gen.ids.every((v, i) => v === want[i]);
    r.equal(`greedy decode ${want.length} steps`, same,
        same
            ? JSON.stringify(await decode(gen.ids))
            : `got [${gen.ids}]\n       want [${want}]\n       got  ${JSON.stringify(await decode(gen.ids))}`);

    return { rows: r.rows, ok: r.rows.every((x) => x.ok) };
}

// --- Streaming verification, for a checkpoint too big for one GPU ---

// A block's (or client's) tensors are plain GPUBuffers. A q8 tensor also has a
// scales buffer, held in kernels.js's WeakMap, which must be destroyed with
// it.
//
// loadBlock() also sets a `kind` string on its result, which is skipped.
function destroyBuffers(obj) {
    for (const buf of Object.values(obj)) {
        if (!buf || typeof buf.destroy !== 'function') continue;
        if (isQ8(buf)) q8Of(buf).scales.destroy();
        buf.destroy();
    }
}

// The same diffs as verifyAgainstOracle, for a checkpoint whose weights do not
// fit in VRAM at once: load a window of blocks, run, diff, destroy, repeat.
// Peak VRAM is the window plus client.bin.
//
// The KV cache for every layer stays allocated (it is small at these context
// lengths), so the 20-step greedy decode works: each step re-streams every
// block.
export async function verifyAgainstOracleStreaming(device, { baseUrl, manifest, d }, oracle,
                                                    { maxDecodeSteps = 20, dtype = 'f32',
                                                      blockWindow = 8, onProgress } = {}) {
    const r = makeReporter();
    const TOL = dtype === 'q8' ? TOL_Q8 : TOL_F32;
    const H = d.hidden, D = d.head_dim;
    const progress = (msg) => onProgress?.(msg);

    const ids = oracle.tokens.input_ids;
    const T = ids.length;
    const N_max = T + maxDecodeSteps;

    progress('loading client.bin…');
    const client = await loadClient(device, { baseUrl, manifest, d });
    const w = { wte: client.wte, norm: client.norm, lm_head: client.lm_head,
                layers: new Array(d.n_layer) };
    const cache = makeCache(device, d, N_max);   // all layers; small, kept allocated
    const s = makeScratch(device, d, T, N_max);
    const idsBuf = uploadF32(device, Float32Array.from(ids));

    const run = (label, record) => {
        const enc = device.createCommandEncoder({ label });
        withPass(enc, record, label);
        device.queue.submit([enc.finish()]);
    };
    const read = (buf, floats) => readF32(device, buf, floats * F32);

    // Runs every block once, loading each window's weights and destroying
    // them afterwards. The caller chooses T_new/T_past; this is used for the
    // single-token decode steps. The prefill below is written out separately,
    // for the block00 diffs and the per-block oracle comparison.
    //
    // `blockWindow` blocks are fetched concurrently, since fetch latency
    // dominates the decode check. Compute runs in layer order within a
    // window.
    //
    // onSubmittedWorkDone() is awaited before destroying: run() only enqueues
    // a window's passes, and a buffer must not be destroyed while the GPU may
    // read it.
    const streamLayers = async (T_new, T_past, label) => {
        for (let start = 0; start < d.n_layer; start += blockWindow) {
            const end = Math.min(start + blockWindow, d.n_layer);
            const idxs = Array.from({ length: end - start }, (_, k) => start + k);
            const blocks = await Promise.all(
                idxs.map((i) => loadBlock(device, { baseUrl, manifest, d, index: i })));
            for (let k = 0; k < idxs.length; k++) {
                const i = idxs[k];
                w.layers[i] = blocks[k];
                run(`block${i}`, (pass) => block(device, pass, d, w, i, cache, s, T_new, T_past));
            }
            await device.queue.onSubmittedWorkDone();
            for (let k = 0; k < idxs.length; k++) {
                destroyBuffers(blocks[k]);
                w.layers[idxs[k]] = null;
            }
            progress(`${label} block ${end}/${d.n_layer}`);
        }
    };

    // --- 1. embedding ---
    progress('embedding…');
    run('embed', (pass) => gatherEmbedNoPos(device, pass, idsBuf, w.wte, s.x, T, H));
    r.close('embeddings', await read(s.x, T * H), oracle.embeddings.data, TOL.embed);

    // --- 2. block 0 ---
    // Hybrid (Qwen-style) manifests skip the sub-stage diffs, as in
    // verifyAgainstOracle (see loadOracle).
    progress(`prefill block 1/${d.n_layer}`);
    const b0 = await loadBlock(device, { baseUrl, manifest, d, index: 0 });
    w.layers[0] = b0;

    if (!d.hybrid) {
        run('ln1', (pass) => rmsnorm(device, pass, s.x, b0.input_layernorm, s.h, T, H));
        r.close('block00 rmsnorm (input_layernorm)', await read(s.h, T * H),
            oracle.block00_ln1.data, TOL.block0);
    }

    run('block00', (pass) => block(device, pass, d, w, 0, cache, s, T, 0));

    if (!d.hybrid) {
        const kvDim = d.n_kv_heads * D;
        r.close('block00 q_proj', await read(s.q, T * d.n_heads * D), oracle.block00_q.data, TOL.block0);
        r.close('block00 k_proj', await read(s.k, T * kvDim), oracle.block00_k.data, TOL.block0);
        r.close('block00 v_proj', await read(s.v, T * kvDim), oracle.block00_v.data, TOL.block0);

        r.close('block00 rope(Q)', await read(s.qh, d.n_heads * T * D),
            oracle.block00_q_rope.data, TOL.rope);
        const kCache = await read(cache.K[0], d.n_kv_heads * N_max * D);
        const kDense = new Float32Array(d.n_kv_heads * T * D);
        for (let h = 0; h < d.n_kv_heads; h++)
            kDense.set(kCache.subarray(h * N_max * D, h * N_max * D + T * D), h * T * D);
        r.close('block00 rope(K cache)', kDense, oracle.block00_k_rope.data, TOL.rope);

        r.close('block00 attn out (o_proj)', await read(s.attn, T * H),
            oracle.block00_attn_out.data, TOL.block0);
        r.close('block00 rmsnorm (post_attention)', await read(s.h, T * H),
            oracle.block00_ln2.data, TOL.block0);
        r.close('block00 gate_proj', await read(s.gate, T * d.ffn), oracle.block00_gate.data, TOL.block0);
        r.close('block00 up_proj', await read(s.up, T * d.ffn), oracle.block00_up.data, TOL.block0);

        const down = await read(s.down, T * H);
        r.close('block00 mlp out (down_proj)', down, oracle.block00_mlp_out.data, TOL.block0);

        const x0 = await read(s.x, T * H);
        r.close('block00 residual after attn', x0.map((v, i) => v - down[i]),
            oracle.block00_after_attn_resid.data, TOL.block0);
        r.close('block_00 out', x0, oracle.block_00_out.data, TOL.block0);
    } else {
        r.close('block_00 out', await read(s.x, T * H), oracle.block_00_out.data, TOL.block0);
    }

    destroyBuffers(b0);
    w.layers[0] = null;

    // --- 3. the remaining n_layer-1 blocks ---
    // Diffed one at a time, since s.x must be read back before the next block
    // overwrites it. The fetch of block i+1 starts as soon as block i's
    // weights arrive, so it downloads during block i's compute and readback.
    let nextBlock = loadBlock(device, { baseUrl, manifest, d, index: 1 });
    for (let i = 1; i < d.n_layer; i++) {
        progress(`prefill block ${i + 1}/${d.n_layer}`);
        const blk = await nextBlock;
        if (i + 1 < d.n_layer) nextBlock = loadBlock(device, { baseUrl, manifest, d, index: i + 1 });
        w.layers[i] = blk;
        run(`block${i}`, (pass) => block(device, pass, d, w, i, cache, s, T, 0));
        const name = `block_${String(i).padStart(2, '0')}`;
        r.close(`${name} out`, await read(s.x, T * H), oracle[`${name}_out`].data, TOL.block);
        destroyBuffers(blk);
        w.layers[i] = null;
    }

    // --- 4. final norm + LM head (client.bin is still loaded) ---
    run('ln_f', (pass) => rmsnorm(device, pass, s.x, w.norm, s.h, T, H));
    r.close('final_norm', await read(s.h, T * H), oracle.final_norm.data, TOL.norm);

    run('lm_head', (pass) => matmulNobias(device, pass, s.h, w.lm_head ?? w.wte, s.logits, T, H, d.vocab));
    const logits = await read(s.logits, T * d.vocab);
    r.close('logits', logits, oracle.logits.data, TOL.logits);

    // --- 5. argmax at the last position ---
    const lastRow = logits.subarray((T - 1) * d.vocab, T * d.vocab);
    let argmax = 0;
    for (let i = 1; i < d.vocab; i++) if (lastRow[i] > lastRow[argmax]) argmax = i;
    r.equal('next token argmax', argmax === oracle.nextToken.next_id,
        `got ${argmax} ${JSON.stringify(await decode([argmax]))},` +
        ` want ${oracle.nextToken.next_id} ${JSON.stringify(oracle.nextToken.decoded)}`);

    // --- 6. greedy decode: the pass/fail check ---
    // Continues from the state the prefill above left in `cache` (every
    // layer's K/V through position T). This equals generate()'s first step and
    // avoids streaming all n_layer blocks again for a second prefill.
    const want = oracle.greedy.generated_ids.slice(0, maxDecodeSteps);
    const gen = [argmax];
    let T_past = T;
    for (let step = 1; step < want.length; step++) {
        progress(`decode step ${step}/${want.length - 1}`);
        const tokBuf = uploadF32(device, Float32Array.from([gen[gen.length - 1]]));
        run('embed-step', (pass) => gatherEmbedNoPos(device, pass, tokBuf, w.wte, s.x, 1, H));
        await streamLayers(1, T_past, `decode step ${step}/${want.length - 1}`);
        run('ln_f-step', (pass) => rmsnorm(device, pass, s.x, w.norm, s.h, 1, H));
        run('lm_head-step', (pass) =>
            matmulNobias(device, pass, s.h, w.lm_head ?? w.wte, s.logits, 1, H, d.vocab));
        const stepLogits = await read(s.logits, d.vocab);
        let next = 0;
        for (let i = 1; i < d.vocab; i++) if (stepLogits[i] > stepLogits[next]) next = i;
        gen.push(next);
        T_past++;
        tokBuf.destroy();
    }
    const same = gen.length === want.length && gen.every((v, i) => v === want[i]);
    r.equal(`greedy decode ${want.length} steps`, same,
        same
            ? JSON.stringify(await decode(gen))
            : `got [${gen}]\n       want [${want}]\n       got  ${JSON.stringify(await decode(gen))}`);

    destroyBuffers(client);

    return { rows: r.rows, ok: r.rows.every((x) => x.ok) };
}
