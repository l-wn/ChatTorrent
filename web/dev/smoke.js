// Architecture smoke test.
//
// A small config with the same structure as SmolLM2: GQA with a sharing factor
// above 1, even head_dim, SwiGLU, tied LM head. A prefill and a decode step
// run almost instantly. It checks the wiring; numerics are checked in
// kernels.test.js and verify.js.
//
// The kernel tests check each kernel alone against a CPU reference, and the
// oracle verification checks the real weights against HuggingFace. This is the
// only check that runs the whole stack without downloading weights. The dev
// page and every peer run it before loading real weights.
//
// Returns { ok, rows }, the same shape as runAllTests.

import { F32, uploadF32, readF32, cacheStats, withPass } from '../model/kernels.js';
import { checkDims, makeCache, makeScratch, forward, randomWeights } from '../model/model.js';

const TINY = {
    hidden:     24,
    n_layer:     3,
    n_heads:     4,
    n_kv_heads:  2,
    head_dim:    6,
    ffn:        40,
    vocab:      32,
    n_pos:      16,
    group:       2,
    rms_eps:  1e-5,
    rope_theta: 10000.0,
};

// Runs a prefill and then one decode step, so the KV cache is used with
// T_past = 0 and with T_past > 0.
export async function smokeForward(device) {
    const d = TINY;

    const problems = checkDims(d);
    if (problems.length) {
        return { ok: false, rows: problems.map((p) => ({ name: 'TINY config', ok: false, detail: p })) };
    }

    const N_max = d.n_pos;
    const promptIds = new Float32Array([3, 14, 1, 5, 9]);
    const T_prompt = promptIds.length;

    const w = randomWeights(device, d);
    const cache = makeCache(device, d, N_max);
    const s = makeScratch(device, d, T_prompt, N_max);

    // 21 dispatches per block, plus 3 outside the blocks.
    const perPass = d.n_layer * 21 + 3;
    const before = cacheStats(device);

    // --- prefill: the whole prompt at once, empty cache ---
    const idsBuf = uploadF32(device, promptIds);
    let enc = device.createCommandEncoder({ label: 'prefill' });
    withPass(enc, (pass) => forward(device, pass, d, w, cache, s, idsBuf, T_prompt, 0));
    device.queue.submit([enc.finish()]);

    const prefill = await readF32(device, s.logits, T_prompt * d.vocab * F32);

    // --- decode: one token, cache holds the prompt ---
    const nextBuf = uploadF32(device, new Float32Array([7]));
    enc = device.createCommandEncoder({ label: 'decode' });
    withPass(enc, (pass) => forward(device, pass, d, w, cache, s, nextBuf, 1, T_prompt));
    device.queue.submit([enc.finish()]);

    const decode = await readF32(device, s.logits, 1 * d.vocab * F32);

    // --- checks ---
    // With random weights only structural properties can be checked: shapes
    // match, nothing is NaN/Inf after 3 blocks, and the output depends on its
    // inputs.
    const rows = [];
    const finite = (a) => a.every(Number.isFinite);
    const add = (name, ok, detail = '') => rows.push({ name, ok, detail });

    add('prefill logits are finite', finite(prefill));
    add('decode logits are finite', finite(decode));

    const rowsDiffer = (() => {
        for (let v = 0; v < d.vocab; v++)
            if (Math.abs(prefill[v] - prefill[(T_prompt - 1) * d.vocab + v]) > 1e-6) return true;
        return false;
    })();
    add('prefill rows differ by position (RoPE + causal mask are live)', rowsDiffer);

    const row = decode.slice(0, d.vocab);
    add('decode logits are not constant across vocab', Math.max(...row) - Math.min(...row) > 1e-6);

    const after = cacheStats(device);
    add('caches', true,
        `${2 * perPass} dispatches → ${after.uniforms - before.uniforms} uniforms,` +
        ` ${after.bindGroups - before.bindGroups} bind groups`);

    return { ok: rows.every((x) => x.ok), rows };
}
