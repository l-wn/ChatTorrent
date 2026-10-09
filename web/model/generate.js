// Prefill and decode loop on top of model.js's forward().
//
// One prefill runs the whole prompt (T_new = T, T_past = 0). Each decode step
// then runs one token (T_new = 1, T_past = n) on the KV cache the earlier
// steps filled. forward() is the same for both.
//
// Sampling is on the CPU, from the last logits row (sampling.js).

import { F32, withPass } from './kernels.js';
import { forward } from './model.js';
import { penaltyState, sampleFromLogitsRow } from './sampling.js';

// Re-exported for stage.js.
export { penaltyState, sampleFromLogitsRow };

// Records one forward pass, submits once, and returns the last row of logits.
// `ids` is a plain array of token ids. The gather kernel reads them as f32
// (see KERNEL.gather_embed_nopos).
//
// The ids are written to the scratch's reused buffer (a new buffer per token
// would miss the bind-group cache). The readback copies only the last logits
// row, on the same encoder as the compute.
export async function forwardIds(device, d, w, cache, s, ids, T_past) {
    const T_new = ids.length;
    device.queue.writeBuffer(s.ids, 0, Float32Array.from(ids));

    const enc = device.createCommandEncoder({ label: T_past ? 'decode' : 'prefill' });
    withPass(enc, (pass) => forward(device, pass, d, w, cache, s, s.ids, T_new, T_past));

    // The copy is outside the pass, on the same encoder, so it runs after
    // every dispatch above and one submit covers both.
    const rowBytes = d.vocab * F32;
    enc.copyBufferToBuffer(s.logits, (T_new - 1) * rowBytes, s.readback, 0, rowBytes);
    device.queue.submit([enc.finish()]);

    await s.readback.mapAsync(GPUMapMode.READ);
    const row = new Float32Array(s.readback.getMappedRange().slice(0));
    s.readback.unmap();
    return row;
}

// Greedy or sampled generation. The cache must be empty, `s` sized for at
// least the prompt length, and N_max at least promptIds.length +
// maxNewTokens.
//
// Returns { ids, tokenMs }. `ids` holds only the generated ids.
export async function generate(device, d, w, cache, s, promptIds,
                               { maxNewTokens = 20, eosId = null, onToken,
                                 ngram = 0, ...sampleOpts } = {}) {
    if (promptIds.length + maxNewTokens > cache.N_max) {
        throw new Error(`context ${promptIds.length + maxNewTokens} exceeds cache N_max ${cache.N_max}`);
    }

    const all = promptIds.slice();
    const out = [];
    const tokenMs = [];

    let t0 = performance.now();
    let row = await forwardIds(device, d, w, cache, s, promptIds, 0);
    let T_past = promptIds.length;
    tokenMs.push(performance.now() - t0);

    for (let i = 0; i < maxNewTokens; i++) {
        const id = sampleFromLogitsRow(row, {
            ...sampleOpts,
            ...penaltyState(all, { ngram, skip: eosId == null ? [] : [eosId] }),
        });
        all.push(id);
        out.push(id);
        onToken?.(id, out.length);
        if (id === eosId) break;

        if (i === maxNewTokens - 1) break;   // the last forward's output would be unused
        t0 = performance.now();
        row = await forwardIds(device, d, w, cache, s, [id], T_past);
        T_past += 1;
        tokenMs.push(performance.now() - t0);
    }

    return { ids: out, tokenMs };
}
