// Diffs one Qwen3.8-27B layer of each kind against oracle/qwen3.8-27b-spike/
// (written by tools/dump_qwen_oracle.py): layer 0 (linear_attention) from
// `embeddings`, and layer 3 (full_attention) from `block_02_out`, the output
// of the block before it. Uses no Runner and no model.js. Every kernel call
// and scratch buffer is set up here, with kernels.js's production kernels and
// verify.js's diff()/makeReporter().
//
// Console only: import this module and call runQwenSpike(device).

import {
    uploadF32, makeStorage, readF32, withPass,
    matmulNobias, rmsnorm, siluMul, addInplace, rope,
    reshapeHeads, scatterKV, qkScoresGQA, causalSoftmax, attnAVGQA, mergeHeads,
    causalConv1d, l2NormalizeHeads, gatedDeltaNetStep, sigmoidMul,
} from '../model/kernels.js';
import { loadNpy, loadJson } from './npy.js';
import { diff, makeReporter } from './verify.js';

export const ORACLE_URL = '/oracle/qwen3.8-27b-spike';

// Qwen3.8-27B's rms_norm_eps, from the checkpoint's config.json. kernels.js's
// rmsnorm() defaults to DIMS.rms_eps (1e-5, SmolLM2's value), which is 10x too
// large here. It visibly changes the output of the RMSNormGated norm, which is
// only head_v_dim=16 wide.
export const RMS_EPS = 1e-6;

// --- Oracle + weight loading ---

export async function loadAll(names, base) {
    const arrays = {};
    await Promise.all(names.map(async (n) => { arrays[n] = await loadNpy(`${base}/${n}.npy`); }));
    return arrays;
}

// Splits a [rows, K] weight matrix into named contiguous row ranges. Each
// output can be passed to matmulNobias as W[N,K]. The split is exact: the
// weight rows of nn.Linear(hidden, A+B+C) are the concatenation of the rows of
// three separate nn.Linear layers, and each output row's dot product is
// independent of the others.
export function splitRows(mat, K, ranges) {
    const out = {};
    for (const [name, [r0, r1]] of Object.entries(ranges)) {
        out[name] = mat.slice(r0 * K, r1 * K);
    }
    return out;
}

// q_proj's rows interleave query and gate per head ([q0..q63, g0..g63,
// q64..], not [q0..q383, g0..g383]). torch.chunk(view(*, heads, 2*D), 2,
// dim=-1) splits the last axis, which after the [heads,2D] reshape is two
// halves within each head's 128-row block. This is a gather, not a contiguous
// split. It is done here on the CPU at load time (nHeads*headDim*hidden
// floats), so no GPU kernel is needed.
export function splitInterleavedGate(mat, hidden, nHeads, headDim) {
    const q = new Float32Array(nHeads * headDim * hidden);
    const g = new Float32Array(nHeads * headDim * hidden);
    for (let h = 0; h < nHeads; h++) {
        const src2h = h * 2 * headDim;
        q.set(mat.subarray(src2h * hidden, (src2h + headDim) * hidden), h * headDim * hidden);
        g.set(mat.subarray((src2h + headDim) * hidden, (src2h + 2 * headDim) * hidden), h * headDim * hidden);
    }
    return { q, g };
}

// Qwen3_5RMSNorm computes norm(x)*(1+weight) (modeling_qwen3_5.py), and
// kernels.js's rmsnorm computes norm(x)*weight. Adding 1 to the weight at load
// time gives the same result with the existing kernel (tools/convert_hf.py
// does the same).
// Not applied to RMSNormGated's weight (linear_attn.norm). That class uses the
// plain weight with no +1, followed by siluMul (see runLayer0).
export function plusOne(arr) {
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] + 1;
    return out;
}

// --- Layer 0 (linear_attention) ---

export async function runLayer0(device, cfg, oracleBase = ORACLE_URL) {
    const r = makeReporter();
    const T = cfg.T, H = cfg.hidden_size;
    const convDim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const keyDim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const valueDim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const nK = cfg.linear_num_key_heads, nV = cfg.linear_num_value_heads;
    const Dk = cfg.linear_key_head_dim, Dv = cfg.linear_value_head_dim;
    const group = nV / nK;
    const K4 = cfg.linear_conv_kernel_dim;

    const o = await loadAll([
        'embeddings', 'layer00_ln1', 'layer00_qkv_raw', 'layer00_z_raw',
        'layer00_b_raw', 'layer00_a_raw', 'layer00_conv_pre', 'layer00_conv_post',
        'layer00_query_l2norm', 'layer00_key_l2norm', 'layer00_recurrent_core_out_seq',
        'layer00_gated_norm_out', 'layer00_linear_attn_out', 'layer00_after_attn_resid',
        'layer00_ln2', 'layer00_gate', 'layer00_up', 'layer00_mlp_out', 'block_00_out',
    ], oracleBase);

    const w = await loadAll([
        'weight_layer00_input_layernorm', 'weight_layer00_in_proj_qkv',
        'weight_layer00_in_proj_z', 'weight_layer00_in_proj_b', 'weight_layer00_in_proj_a',
        'weight_layer00_conv1d', 'weight_layer00_A_log', 'weight_layer00_dt_bias',
        'weight_layer00_gated_norm', 'weight_layer00_out_proj',
        'weight_layer00_post_attention_layernorm', 'weight_layer00_gate_proj',
        'weight_layer00_up_proj', 'weight_layer00_down_proj',
    ], oracleBase);

    // in_proj_qkv rows: [0,keyDim) query, [keyDim,2*keyDim) key, [2*keyDim,convDim) value.
    const qkv = splitRows(w.weight_layer00_in_proj_qkv.data, H, {
        q: [0, keyDim], k: [keyDim, 2 * keyDim], v: [2 * keyDim, convDim],
    });
    // conv1d weight rows are in the same channel order as in_proj_qkv's output.
    const convW = splitRows(w.weight_layer00_conv1d.data, K4, {
        q: [0, keyDim], k: [keyDim, 2 * keyDim], v: [2 * keyDim, convDim],
    });

    const g = (data) => uploadF32(device, data instanceof Float32Array ? data : new Float32Array(data));
    const Wq = g(qkv.q), Wk = g(qkv.k), Wv = g(qkv.v);
    const convWq = g(convW.q), convWk = g(convW.k), convWv = g(convW.v);
    const Wz = g(w.weight_layer00_in_proj_z.data), Wb = g(w.weight_layer00_in_proj_b.data);
    const Wa = g(w.weight_layer00_in_proj_a.data);
    // Packed [A_log(nV), dt_bias(nV)]. The gated_delta_net_step comment in
    // kernels.js explains why they share one storage binding.
    const gateParams = g(Float32Array.of(...w.weight_layer00_A_log.data, ...w.weight_layer00_dt_bias.data));
    const gatedNormW = g(w.weight_layer00_gated_norm.data);
    const outProjW = g(w.weight_layer00_out_proj.data);
    const ln1W = g(plusOne(w.weight_layer00_input_layernorm.data));
    const ln2W = g(plusOne(w.weight_layer00_post_attention_layernorm.data));
    const gateProjW = g(w.weight_layer00_gate_proj.data), upProjW = g(w.weight_layer00_up_proj.data);
    const downProjW = g(w.weight_layer00_down_proj.data);
    const ffn = w.weight_layer00_gate_proj.shape[0]; // intermediate_size

    const x = uploadF32(device, o.embeddings.data);
    const ln1 = makeStorage(device, T * H * 4);
    const qPre = makeStorage(device, T * keyDim * 4), kPre = makeStorage(device, T * keyDim * 4);
    const vPre = makeStorage(device, T * valueDim * 4);
    const qConv = makeStorage(device, T * keyDim * 4), kConv = makeStorage(device, T * keyDim * 4);
    const vConv = makeStorage(device, T * valueDim * 4);
    const zRaw = makeStorage(device, T * valueDim * 4);
    const bRaw = makeStorage(device, T * nV * 4), aRaw = makeStorage(device, T * nV * 4);
    const qL2 = makeStorage(device, T * nV * Dk * 4), kL2 = makeStorage(device, T * nV * Dk * 4);
    const S = makeStorage(device, nV * Dk * Dv * 4);
    const coreOut = makeStorage(device, T * nV * Dv * 4);
    const gatedNorm = makeStorage(device, T * nV * Dv * 4);
    const gatedFinal = makeStorage(device, T * nV * Dv * 4);
    const linAttnOut = makeStorage(device, T * H * 4);
    const ln2 = makeStorage(device, T * H * 4);
    const gateFfn = makeStorage(device, T * ffn * 4), upFfn = makeStorage(device, T * ffn * 4);
    const actFfn = makeStorage(device, T * ffn * 4), downFfn = makeStorage(device, T * H * 4);

    const enc0 = device.createCommandEncoder();
    withPass(enc0, (pass) => {
        rmsnorm(device, pass, x, ln1W, ln1, T, H, RMS_EPS);
        matmulNobias(device, pass, ln1, Wq, qPre, T, H, keyDim);
        matmulNobias(device, pass, ln1, Wk, kPre, T, H, keyDim);
        matmulNobias(device, pass, ln1, Wv, vPre, T, H, valueDim);
        matmulNobias(device, pass, ln1, Wz, zRaw, T, H, valueDim);
        matmulNobias(device, pass, ln1, Wb, bRaw, T, H, nV);
        matmulNobias(device, pass, ln1, Wa, aRaw, T, H, nV);
        causalConv1d(device, pass, qPre, convWq, qConv, T, keyDim, K4);
        causalConv1d(device, pass, kPre, convWk, kConv, T, keyDim, K4);
        causalConv1d(device, pass, vPre, convWv, vConv, T, valueDim, K4);
        l2NormalizeHeads(device, pass, qConv, qL2, T, nK, Dk, group);
        l2NormalizeHeads(device, pass, kConv, kL2, T, nK, Dk, group);
        gatedDeltaNetStep(device, pass, qL2, kL2, vConv, aRaw, bRaw, gateParams,
            S, coreOut, 0, nV, Dk, Dv, T);
        rmsnorm(device, pass, coreOut, gatedNormW, gatedNorm, T * nV, Dv, RMS_EPS);
        // zRaw is [T,valueDim] = [T,nV,Dv] flat, the same layout as
        // gatedNorm's [T*nV,Dv] rows, so the two match element for element.
        siluMul(device, pass, zRaw, gatedNorm, gatedFinal, T * nV * Dv);
        matmulNobias(device, pass, gatedFinal, outProjW, linAttnOut, T, valueDim, H);
        addInplace(device, pass, x, linAttnOut, T * H);
        rmsnorm(device, pass, x, ln2W, ln2, T, H, RMS_EPS);
        matmulNobias(device, pass, ln2, gateProjW, gateFfn, T, H, ffn);
        matmulNobias(device, pass, ln2, upProjW, upFfn, T, H, ffn);
        siluMul(device, pass, gateFfn, upFfn, actFfn, T * ffn);
        matmulNobias(device, pass, actFfn, downProjW, downFfn, T, ffn, H);
        addInplace(device, pass, x, downFfn, T * H);
    }, 'qwen-spike-layer0');
    device.queue.submit([enc0.finish()]);

    const read = (buf, n) => readF32(device, buf, n * 4);
    r.close('layer00 ln1', await read(ln1, T * H), o.layer00_ln1.data, { atol: 1e-4, rtol: 1e-5 });
    r.close('layer00 conv_post (q|k|v channels concatenated)',
        [...await read(qConv, T * keyDim), ...await read(kConv, T * keyDim), ...await read(vConv, T * valueDim)],
        // The oracle's conv_post has channels in [q,k,v] order per row, with
        // rows T-major. Reordered to the [q-block, k-block, v-block] layout
        // read back above.
        reorderConvOracle(o.layer00_conv_post.data, T, keyDim, keyDim, valueDim),
        { atol: 1e-4, rtol: 1e-4 });
    r.close('layer00 query l2norm', await read(qL2, T * nV * Dk), o.layer00_query_l2norm.data,
        { atol: 1e-4, rtol: 1e-4 });
    r.close('layer00 key l2norm', await read(kL2, T * nV * Dk), o.layer00_key_l2norm.data,
        { atol: 1e-4, rtol: 1e-4 });
    r.close('layer00 recurrent core_out', await read(coreOut, T * nV * Dv),
        o.layer00_recurrent_core_out_seq.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('layer00 gated_norm_out', await read(gatedFinal, T * nV * Dv), o.layer00_gated_norm_out.data,
        { atol: 1e-3, rtol: 1e-3 });
    r.close('layer00 linear_attn_out', await read(linAttnOut, T * H), o.layer00_linear_attn_out.data,
        { atol: 1e-3, rtol: 1e-3 });
    r.close('layer00 ln2', await read(ln2, T * H), o.layer00_ln2.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('layer00 mlp gate', await read(gateFfn, T * ffn), o.layer00_gate.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('layer00 mlp up', await read(upFfn, T * ffn), o.layer00_up.data, { atol: 1e-3, rtol: 1e-3 });
    const down = await read(downFfn, T * H), blockOut = await read(x, T * H);
    // after_attn_resid is not in a buffer when the pass finishes, because x
    // also has the FFN's residual (down) added. It is computed as x - down,
    // since the MLP's residual add is x += down (as in verify.js).
    r.close('layer00 after_attn_resid', sub(blockOut, down), o.layer00_after_attn_resid.data,
        { atol: 1e-3, rtol: 1e-3 });
    r.close('layer00 mlp out', down, o.layer00_mlp_out.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('block_00 out', blockOut, o.block_00_out.data, { atol: 1e-3, rtol: 1e-3 });

    return { rows: r.rows, ok: r.rows.every((x) => x.ok) };
}

function sub(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
}

// The oracle's layer00_conv_post.npy is [T, convDim], with convDim columns in
// [q(64) | k(64) | v(192)] order. This re-slices it into the
// [q-block, k-block, v-block] concatenation the GPU readback above produces,
// so one diff() call covers it.
function reorderConvOracle(flat, T, keyDim, keyDim2, valueDim) {
    const convDim = keyDim + keyDim2 + valueDim;
    const out = new Float32Array(T * convDim);
    let o = 0;
    for (const [off, len] of [[0, keyDim], [keyDim, keyDim2], [keyDim + keyDim2, valueDim]]) {
        for (let t = 0; t < T; t++) {
            for (let d = 0; d < len; d++) out[o++] = flat[t * convDim + off + d];
        }
    }
    return out;
}

// --- Layer 3 (full_attention) ---

export async function runLayer3(device, cfg, oracleBase = ORACLE_URL) {
    const r = makeReporter();
    const T = cfg.T, H = cfg.hidden_size, D = cfg.head_dim, nH = cfg.num_attention_heads;
    const nKV = cfg.num_key_value_heads, rotDim = cfg.rot_dim, theta = cfg.rope_theta;
    const groupGQA = nH / nKV;

    const o = await loadAll([
        'block_02_out', 'layer03_ln1', 'layer03_q_proj_raw', 'layer03_k_proj_raw',
        'layer03_v', 'layer03_q_norm_out', 'layer03_k_norm_out', 'layer03_gate_raw',
        'layer03_q_rope', 'layer03_k_rope', 'layer03_attn_out', 'layer03_after_attn_resid',
        'layer03_ln2', 'layer03_gate', 'layer03_up', 'layer03_mlp_out', 'block_03_out',
    ], oracleBase);

    const w = await loadAll([
        'weight_layer03_input_layernorm', 'weight_layer03_q_proj', 'weight_layer03_k_proj',
        'weight_layer03_v_proj', 'weight_layer03_q_norm', 'weight_layer03_k_norm',
        'weight_layer03_o_proj', 'weight_layer03_post_attention_layernorm',
        'weight_layer03_gate_proj', 'weight_layer03_up_proj', 'weight_layer03_down_proj',
    ], oracleBase);

    const { q: qRows, g: gateRows } = splitInterleavedGate(w.weight_layer03_q_proj.data, H, nH, D);

    const g = (data) => uploadF32(device, data instanceof Float32Array ? data : new Float32Array(data));
    const Wq = g(qRows), Wgate = g(gateRows);
    const Wk = g(w.weight_layer03_k_proj.data), Wv = g(w.weight_layer03_v_proj.data);
    const qNormW = g(plusOne(w.weight_layer03_q_norm.data));
    const kNormW = g(plusOne(w.weight_layer03_k_norm.data));
    const oProjW = g(w.weight_layer03_o_proj.data);
    const ln1W = g(plusOne(w.weight_layer03_input_layernorm.data));
    const ln2W = g(plusOne(w.weight_layer03_post_attention_layernorm.data));
    const gateProjW = g(w.weight_layer03_gate_proj.data), upProjW = g(w.weight_layer03_up_proj.data);
    const downProjW = g(w.weight_layer03_down_proj.data);
    const ffn = w.weight_layer03_gate_proj.shape[0];

    const x = uploadF32(device, o.block_02_out.data);
    const ln1 = makeStorage(device, T * H * 4);
    const query = makeStorage(device, T * nH * D * 4), gateRaw = makeStorage(device, T * nH * D * 4);
    const kRaw = makeStorage(device, T * nKV * D * 4), vRaw = makeStorage(device, T * nKV * D * 4);
    const queryNormed = makeStorage(device, T * nH * D * 4), kNormed = makeStorage(device, T * nKV * D * 4);
    const qh = makeStorage(device, nH * T * D * 4);
    const N_max = T;
    const Kcache = makeStorage(device, nKV * N_max * D * 4), Vcache = makeStorage(device, nKV * N_max * D * 4);
    const scores = makeStorage(device, nH * T * N_max * 4);
    const ctx = makeStorage(device, nH * T * D * 4), merged = makeStorage(device, T * nH * D * 4);
    const gated = makeStorage(device, T * nH * D * 4), attnOut = makeStorage(device, T * H * 4);
    const ln2 = makeStorage(device, T * H * 4);
    const gateFfn = makeStorage(device, T * ffn * 4), upFfn = makeStorage(device, T * ffn * 4);
    const actFfn = makeStorage(device, T * ffn * 4), downFfn = makeStorage(device, T * H * 4);

    const enc3 = device.createCommandEncoder();
    withPass(enc3, (pass) => {
        rmsnorm(device, pass, x, ln1W, ln1, T, H, RMS_EPS);
        matmulNobias(device, pass, ln1, Wq, query, T, H, nH * D);
        matmulNobias(device, pass, ln1, Wgate, gateRaw, T, H, nH * D);
        matmulNobias(device, pass, ln1, Wk, kRaw, T, H, nKV * D);
        matmulNobias(device, pass, ln1, Wv, vRaw, T, H, nKV * D);
        // q_norm/k_norm normalize over head_dim only. [T,heads,D] has the
        // same layout as [T*heads,D], so this is the rmsnorm kernel with
        // T*heads rows.
        rmsnorm(device, pass, query, qNormW, queryNormed, T * nH, D, RMS_EPS);
        rmsnorm(device, pass, kRaw, kNormW, kNormed, T * nKV, D, RMS_EPS);
        reshapeHeads(device, pass, queryNormed, qh, T, nH, D);
        scatterKV(device, pass, kNormed, Kcache, T, 0, N_max, nKV, D);
        scatterKV(device, pass, vRaw, Vcache, T, 0, N_max, nKV, D);
        rope(device, pass, qh, nH, T, D, 0, T, 0, theta, null, rotDim);
        rope(device, pass, Kcache, nKV, T, D, 0, N_max, 0, theta, null, rotDim);
        qkScoresGQA(device, pass, qh, Kcache, scores, T, T, N_max, D, nH, groupGQA);
        causalSoftmax(device, pass, scores, T, T, 0, nH);
        attnAVGQA(device, pass, scores, Vcache, ctx, T, T, N_max, D, nH, groupGQA);
        mergeHeads(device, pass, ctx, merged, T, nH, D);
        sigmoidMul(device, pass, gateRaw, merged, gated, T * nH * D);
        matmulNobias(device, pass, gated, oProjW, attnOut, T, H, H);
        addInplace(device, pass, x, attnOut, T * H);
        rmsnorm(device, pass, x, ln2W, ln2, T, H, RMS_EPS);
        matmulNobias(device, pass, ln2, gateProjW, gateFfn, T, H, ffn);
        matmulNobias(device, pass, ln2, upProjW, upFfn, T, H, ffn);
        siluMul(device, pass, gateFfn, upFfn, actFfn, T * ffn);
        matmulNobias(device, pass, actFfn, downProjW, downFfn, T, ffn, H);
        addInplace(device, pass, x, downFfn, T * H);
    }, 'qwen-spike-layer3');
    device.queue.submit([enc3.finish()]);

    const read = (buf, n) => readF32(device, buf, n * 4);
    r.close('layer03 ln1', await read(ln1, T * H), o.layer03_ln1.data, { atol: 1e-4, rtol: 1e-5 });
    r.close('layer03 q_norm_out', await read(queryNormed, T * nH * D), o.layer03_q_norm_out.data,
        { atol: 1e-4, rtol: 1e-4 });
    r.close('layer03 k_norm_out', await read(kNormed, T * nKV * D), o.layer03_k_norm_out.data,
        { atol: 1e-4, rtol: 1e-4 });
    r.close('layer03 gate_raw', await read(gateRaw, T * nH * D), o.layer03_gate_raw.data,
        { atol: 1e-4, rtol: 1e-5 });
    r.close('layer03 q_rope (head-major)', await read(qh, nH * T * D),
        toHeadMajor(o.layer03_q_rope.data, T, nH, D), { atol: 1e-3, rtol: 1e-3 });
    r.close('layer03 k_rope (head-major)', await read(Kcache, nKV * N_max * D),
        toHeadMajor(o.layer03_k_rope.data, T, nKV, D), { atol: 1e-3, rtol: 1e-3 });
    r.close('layer03 attn_out', await read(attnOut, T * H), o.layer03_attn_out.data,
        { atol: 1e-3, rtol: 1e-3 });
    r.close('layer03 ln2', await read(ln2, T * H), o.layer03_ln2.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('layer03 mlp gate', await read(gateFfn, T * ffn), o.layer03_gate.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('layer03 mlp up', await read(upFfn, T * ffn), o.layer03_up.data, { atol: 1e-3, rtol: 1e-3 });
    const down3 = await read(downFfn, T * H), blockOut3 = await read(x, T * H);
    // Computed as x - down, as for layer 0's after_attn_resid.
    r.close('layer03 after_attn_resid', sub(blockOut3, down3), o.layer03_after_attn_resid.data,
        { atol: 1e-3, rtol: 1e-3 });
    r.close('layer03 mlp out', down3, o.layer03_mlp_out.data, { atol: 1e-3, rtol: 1e-3 });
    r.close('block_03 out', blockOut3, o.block_03_out.data, { atol: 1e-3, rtol: 1e-3 });

    return { rows: r.rows, ok: r.rows.every((x) => x.ok) };
}

// The oracle's *_rope.npy is [T,heads,D] (T-major). The GPU pipeline keeps Q/K
// head-major ([heads,T,D]) after reshapeHeads/scatterKV, as in production. The
// oracle array is re-laid out here so one diff() call covers it.
function toHeadMajor(flat, T, heads, D) {
    const out = new Float32Array(T * heads * D);
    for (let t = 0; t < T; t++) {
        for (let h = 0; h < heads; h++) {
            for (let d = 0; d < D; d++) {
                out[(h * T + t) * D + d] = flat[(t * heads + h) * D + d];
            }
        }
    }
    return out;
}

// --- Entry point ---

export async function runQwenSpike(device, oracleBase = ORACLE_URL) {
    const cfg = await loadJson(`${oracleBase}/spike_config.json`);
    const l0 = await runLayer0(device, cfg, oracleBase);
    const l3 = await runLayer3(device, cfg, oracleBase);
    return {
        layer0: l0, layer3: l3,
        ok: l0.ok && l3.ok,
        rows: [...l0.rows.map((x) => ({ ...x, name: `layer0: ${x.name}` })),
               ...l3.rows.map((x) => ({ ...x, name: `layer3: ${x.name}` }))],
    };
}
