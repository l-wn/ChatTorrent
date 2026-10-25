// Runs the same two oracle layers as verify_qwen_spike.js (layer 0 =
// linear_attention, layer 3 = full_attention, from oracle/qwen3.8-27b-spike/)
// through the production per-kind path: a manifest-shaped object through
// loader.js's dimsFromManifest()/checkManifest(), model.js's
// makeCache()/makeScratch(), and block(), which dispatches on
// w.layers[layer].kind as loadBlock() sets it.
//
// The raw HF-shaped oracle tensors are converted to the named tensors of a
// converted block with verify_qwen_spike.js's
// splitRows/splitInterleavedGate/plusOne, as tools/convert_hf.py does.
//
// Console only: import this module and call runQwenDispatch(device).

import { uploadF32, readF32, withPass } from '../model/kernels.js';
import { dimsFromManifest, checkManifest } from '../model/loader.js';
import { checkDims, makeCache, makeScratch, block } from '../model/model.js';
import { loadJson } from './npy.js';
import { loadAll, splitRows, splitInterleavedGate, plusOne, RMS_EPS, KV_TOL } from './verify_qwen_spike.js';
import { makeReporter } from './verify.js';

export const ORACLE_URL = '/oracle/qwen3.8-27b-spike';

// spike_config.json is HF-shaped (hidden_size, num_attention_heads, …). Real
// checkpoints go through convert_hf.py, which produces the manifest fields
// dimsFromManifest() reads. This function builds the same fields, so the test
// uses loader.js's dims-building and validation.
function manifestFromCfg(cfg) {
    return {
        hidden: cfg.hidden_size,
        n_layer: cfg.num_hidden_layers,
        n_head: cfg.num_attention_heads,
        n_kv_head: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        ffn: cfg.intermediate_size,
        vocab: cfg.vocab_size,
        rms_eps: RMS_EPS,
        rope_theta: cfg.rope_theta,
        weight_layout: 'out_in',
        norm: 'rmsnorm',
        act: 'silu',
        dtype: 'f32',
        tied_lm_head: true,
        layer_types: cfg.layer_types,
        rot_dim: cfg.rot_dim,
        linear_num_key_heads: cfg.linear_num_key_heads,
        linear_num_value_heads: cfg.linear_num_value_heads,
        linear_key_head_dim: cfg.linear_key_head_dim,
        linear_value_head_dim: cfg.linear_value_head_dim,
        linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
    };
}

// Builds `d` as loadWeights() does (dimsFromManifest(), then
// checkManifest()/checkDims()) and throws on the first problem, as a real boot
// does before fetching anything.
function dimsChecked(cfg) {
    const manifest = manifestFromCfg(cfg);
    const d = dimsFromManifest(manifest);
    const problems = [...checkManifest(manifest, d), ...checkDims(d)];
    if (problems.length) throw new Error(`qwen dispatch check: ${problems.join('; ')}`);
    return d;
}

const g = (device, data) => uploadF32(device, data instanceof Float32Array ? data : new Float32Array(data));

// Layer 0's tensors, with the names and shapes that loader.js's
// linearAttnBlockShapes() and tools/convert_hf.py's
// linear_attn_layer_tensors() use for a converted block.
async function buildLinearLayerWeights(device, d, oracleBase) {
    const { keyDim, valueDim, K4 } = d.linear;
    const convDim = 2 * keyDim + valueDim;
    const o = await loadAll([
        'weight_layer00_input_layernorm', 'weight_layer00_in_proj_qkv',
        'weight_layer00_in_proj_z', 'weight_layer00_in_proj_b', 'weight_layer00_in_proj_a',
        'weight_layer00_conv1d', 'weight_layer00_A_log', 'weight_layer00_dt_bias',
        'weight_layer00_gated_norm', 'weight_layer00_out_proj',
        'weight_layer00_post_attention_layernorm', 'weight_layer00_gate_proj',
        'weight_layer00_up_proj', 'weight_layer00_down_proj',
    ], oracleBase);

    const ranges = { q: [0, keyDim], k: [keyDim, 2 * keyDim], v: [2 * keyDim, convDim] };
    const qkv = splitRows(o.weight_layer00_in_proj_qkv.data, d.hidden, ranges);
    const convW = splitRows(o.weight_layer00_conv1d.data, K4, ranges);
    const gateParams = Float32Array.of(...o.weight_layer00_A_log.data, ...o.weight_layer00_dt_bias.data);

    return {
        kind: 'linear_attention',
        input_layernorm: g(device, plusOne(o.weight_layer00_input_layernorm.data)),
        post_attention_layernorm: g(device, plusOne(o.weight_layer00_post_attention_layernorm.data)),
        in_proj_q: g(device, qkv.q), in_proj_k: g(device, qkv.k), in_proj_v: g(device, qkv.v),
        in_proj_z: g(device, o.weight_layer00_in_proj_z.data),
        in_proj_b: g(device, o.weight_layer00_in_proj_b.data),
        in_proj_a: g(device, o.weight_layer00_in_proj_a.data),
        conv1d_q: g(device, convW.q), conv1d_k: g(device, convW.k), conv1d_v: g(device, convW.v),
        gate_params: g(device, gateParams),
        gated_norm: g(device, o.weight_layer00_gated_norm.data),
        out_proj: g(device, o.weight_layer00_out_proj.data),
        gate_proj: g(device, o.weight_layer00_gate_proj.data),
        up_proj: g(device, o.weight_layer00_up_proj.data),
        down_proj: g(device, o.weight_layer00_down_proj.data),
    };
}

// Layer 3's tensors, with the names and shapes that loader.js's
// fullAttnBlockShapes() and tools/convert_hf.py's full_attn_layer_tensors()
// use.
async function buildFullLayerWeights(device, d, oracleBase) {
    const o = await loadAll([
        'weight_layer03_input_layernorm', 'weight_layer03_q_proj', 'weight_layer03_k_proj',
        'weight_layer03_v_proj', 'weight_layer03_q_norm', 'weight_layer03_k_norm',
        'weight_layer03_o_proj', 'weight_layer03_post_attention_layernorm',
        'weight_layer03_gate_proj', 'weight_layer03_up_proj', 'weight_layer03_down_proj',
    ], oracleBase);

    const { q: qRows, g: gateRows } =
        splitInterleavedGate(o.weight_layer03_q_proj.data, d.hidden, d.n_heads, d.head_dim);

    return {
        kind: 'full_attention',
        input_layernorm: g(device, plusOne(o.weight_layer03_input_layernorm.data)),
        q_proj: g(device, qRows),
        attn_gate: g(device, gateRows),
        k_proj: g(device, o.weight_layer03_k_proj.data),
        v_proj: g(device, o.weight_layer03_v_proj.data),
        q_norm: g(device, plusOne(o.weight_layer03_q_norm.data)),
        k_norm: g(device, plusOne(o.weight_layer03_k_norm.data)),
        o_proj: g(device, o.weight_layer03_o_proj.data),
        post_attention_layernorm: g(device, plusOne(o.weight_layer03_post_attention_layernorm.data)),
        gate_proj: g(device, o.weight_layer03_gate_proj.data),
        up_proj: g(device, o.weight_layer03_up_proj.data),
        down_proj: g(device, o.weight_layer03_down_proj.data),
    };
}

// Runs one layer alone through makeCache()/makeScratch()/block(). `layerLo` is
// the layer's global index, which makeCache needs to look up its kind in
// d.layer_types. block() gets local index 0 into a one-element w.layers array,
// as it does from a Runner holding only this position (see stage.js's #run()).
async function runLayerDispatch(device, d, T, layerLo, buildWeights, xName, outName, r, name, oracleBase,
                                tol = { atol: 1e-3, rtol: 1e-3 }) {
    const o = await loadAll([xName, outName], oracleBase);
    const layerW = await buildWeights(device, d, oracleBase);
    const w = { layers: [layerW] };

    const cache = makeCache(device, d, T, 1, layerLo);
    const s = makeScratch(device, d, T, T);
    device.queue.writeBuffer(s.x, 0, o[xName].data);

    const enc = device.createCommandEncoder();
    withPass(enc, (pass) => block(device, pass, d, w, 0, cache, s, T, 0), `qwen-dispatch-${name}`);
    device.queue.submit([enc.finish()]);

    r.close(`${name} out (via real block() dispatch)`,
        await readF32(device, s.x, T * d.hidden * 4), o[outName].data, tol);
}

export async function runLayer0Dispatch(device, cfg, oracleBase = ORACLE_URL) {
    const r = makeReporter();
    const d = dimsChecked(cfg);
    await runLayerDispatch(device, d, cfg.T, 0, buildLinearLayerWeights,
        'embeddings', 'block_00_out', r, 'layer00', oracleBase);
    return { rows: r.rows, ok: r.rows.every((x) => x.ok) };
}

export async function runLayer3Dispatch(device, cfg, oracleBase = ORACLE_URL) {
    const r = makeReporter();
    const d = dimsChecked(cfg);
    await runLayerDispatch(device, d, cfg.T, 3, buildFullLayerWeights,
        'block_02_out', 'block_03_out', r, 'layer03', oracleBase, KV_TOL);
    return { rows: r.rows, ok: r.rows.every((x) => x.ok) };
}

export async function runQwenDispatch(device, oracleBase = ORACLE_URL) {
    const cfg = await loadJson(`${oracleBase}/spike_config.json`);
    const l0 = await runLayer0Dispatch(device, cfg, oracleBase);
    const l3 = await runLayer3Dispatch(device, cfg, oracleBase);
    return {
        layer0: l0, layer3: l3,
        ok: l0.ok && l3.ok,
        rows: [...l0.rows.map((x) => ({ ...x, name: `layer0: ${x.name}` })),
               ...l3.rows.map((x) => ({ ...x, name: `layer3: ${x.name}` }))],
    };
}
