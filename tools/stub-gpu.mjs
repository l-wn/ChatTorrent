// A stub GPU that does no arithmetic and records what it was asked to do.
//
// Every weight tensor is a tagged marker, not a real buffer, and every bind
// group that references one appends its tag to a trace. A run through a Runner
// therefore produces the ordered list of weights it used. The tests use it to
// check the property the split depends on:
//
//   A chain of peers covering the position line dispatches the same work as a
//   single tab, in the same order.
//
// The kernels, the block code and the sampler are shared with the single-tab
// path and are checked against the oracle. What remains to test is which
// weights are bound at which cursor, which is index arithmetic and needs no
// GPU.
//
// Shared by tools/stage.test.mjs, tools/mesh.test.mjs and
// tools/engine.test.mjs, which all use the same stub and the same tiny model.

import { positions, blockRange } from '../web/split.js';

globalThis.GPUMapMode = { READ: 1 };
globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8, UNIFORM: 16,
                              QUERY_RESOLVE: 512 };

export const collapse = (xs) => xs.filter((x, i) => x !== xs[i - 1]);

// getMappedRange respects (offset, size), like the real API. The readback
// buffer is pooled and sized for the largest readback, so a caller mapping a
// smaller range must get exactly that range. Returning the whole buffer would
// let a wrong-length payload pass here and fail on real hardware.
const mappable = (size) => ({
    size,
    mapAsync: async () => {},
    getMappedRange(offset = 0, length) { return new ArrayBuffer(length ?? size - offset); },
    unmap() {}, destroy() {},
});

export const marker = (tag) => ({ __tag: tag, ...mappable(1 << 16) });

// Makes one stub GPU with its own trace. Each bind group that references a
// tagged weight appends an entry. Consecutive duplicates are collapsed on
// read: one block is 21 dispatches that all bind that block's tensors, and the
// tests care about the order of weights used.
//
// A factory, not a singleton, because the trace is per device and peers do not
// share a GPU. tools/mesh.test.mjs runs several peers in one process. With a
// shared trace, a replay into one peer while another is working would be
// recorded as one run, and assertions about which blocks a peer dispatched
// would include the other peer's work.
//
// `timestamps` grants `timestamp-query`, for model/gpuprof.js. The stub times
// nothing (a resolved query reads back as zeros), but it counts the passes
// opened and the passes with timestamp writes, which is enough to check for
// one timed pass per dispatch.
//
// `fallback(bytes)` fills a readback when no one-shot fill is set (null:
// zeros).
export function makeGpu({ timestamps = false, fallback = null } = {}) {
    let trace = [];
    const passes = { opened: 0, timed: 0, resolved: 0, cleared: 0 };
    // The contents of the next readback. The stub does no arithmetic, so a
    // caller that wants specific values from a run sets them here, and #run
    // reads them as the GPU's output. Cleared after one readback; with nothing
    // set, the readback is zeros.
    //
    // This is a hook, not a patch applied afterwards, because Runner.#run
    // writes its boundary log from the same buffer during the run. Patching
    // afterwards would leave zeros in the log for a microtask, and a repair
    // reading it then would replay zeros.
    let fill = null;
    const setFill = (fn) => { fill = fn; };
    const readback = (size) => ({
        ...mappable(size),
        getMappedRange(offset = 0, length) {
            const n = length ?? size - offset;
            const f = fill ? fill(n) : fallback?.(n);
            fill = null;
            return f ? f.buffer : new ArrayBuffer(n);
        },
    });
    // Bytes in buffers created and not yet destroyed, which is what a real GPU
    // would hold for this device. Used to check Runner.vramBytes.
    let allocated = 0;
    const tracked = (size) => {
        allocated += size;
        let live = true;
        return { ...readback(size), destroy() { if (live) allocated -= size; live = false; } };
    };
    const device = {
        features: new Set(timestamps ? ['timestamp-query'] : []),
        limits: {},
        // Buffers have their real size, so readF32() returns an array of the
        // length the next peer in the chain checks.
        createBuffer: ({ size }) => tracked(size),
        createShaderModule: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createBindGroup: ({ entries = [] }) => ({
            __tags: entries.map((e) => e.resource?.buffer?.__tag).filter(Boolean),
        }),
        createQuerySet: ({ count }) => ({ count, destroy() {} }),
        // A growth step wraps its allocation in error scopes (engine.js's
        // #growClaim). No error is reported, so growth proceeds as it does on a
        // GPU with VRAM to spare.
        pushErrorScope() {},
        popErrorScope: async () => null,
        createCommandEncoder: () => ({
            beginComputePass: (desc) => {
                passes.opened++;
                if (desc?.timestampWrites) passes.timed++;
                return {
                    setPipeline() {}, dispatchWorkgroups() {}, end() {},
                    setBindGroup(_, g) { for (const t of g?.__tags ?? []) trace.push(t); },
                };
            },
            resolveQuerySet() { passes.resolved++; },
            // Clearing a reused session's recurrent state (stage.js's
            // #cacheFor, or a restart from past 0 in #run). The stub does not
            // read buffer contents, so it only counts the call.
            clearBuffer() { passes.cleared++; },
            copyBufferToBuffer() {}, finish: () => ({}),
        }),
        queue: { writeBuffer() {}, submit() {}, onSubmittedWorkDone: async () => {} },
    };
    return {
        device,
        setFill,
        resetTrace: () => { trace = []; },
        getTrace: () => trace,
        allocated: () => allocated,
        // Everything the run used, collapsed. The usual way to read a trace.
        takeTrace: () => collapse(trace),
        passes,
    };
}

// What kernels.js topkRows reads back for an all-zero logits row: M (value,
// index) pairs, value 0, ids 0..M-1 (ties go to the lower index). For a
// fallback that answers a head's candidate readback (8 bytes a pair) as the
// GPU would.
export const zeroTopk = (bytes) => {
    if (bytes % 8 !== 0) return null;
    const out = new Float32Array(bytes / 4);
    const ids = new Uint32Array(out.buffer);
    for (let m = 0; m < bytes / 8; m++) ids[m * 2 + 1] = m;
    return out;
};

// A shared instance, for suites that drive one Runner at a time.
export const { device, setFill, resetTrace, getTrace, takeTrace } = makeGpu();

// A model small enough to enumerate every way of cutting it, with a shape that
// uses every path in model.js: GQA (n_heads > n_kv_heads), even head_dim for
// RoPE's split-half pairing, and an untied head so the LM head's weight is
// distinguishable from the embedding's.
//
// `vocab` is not a multiple of `hidden`, so a readback of one logits row is
// never the same size as a readback of T rows of the residual stream.
// mesh.test.mjs's toy model tells the two apart by size. With vocab 32, a
// four-token job would be read as a logits row.
export const D = { hidden: 8, n_layer: 6, n_heads: 2, n_kv_heads: 1, head_dim: 4, ffn: 16,
                   vocab: 33, n_pos: 64, group: 2, rms_eps: 1e-5, rope_theta: 1e4,
                   tied_lm_head: false };

export const L = D.n_layer;
export const N = positions(L);          // 8

export const BLOCK_TENSORS = ['input_layernorm', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                              'post_attention_layernorm', 'gate_proj', 'up_proj', 'down_proj'];

// What one tab dispatches: the embedding, every block in order, the final norm
// and the LM head. Every arrangement of peers must reproduce this list.
export const SINGLE = ['wte', ...Array.from({ length: L }, (_, i) => `L${i}`), 'norm', 'head'];

// The weights a peer holding `[lo, hi)` has loaded: the blocks in that range,
// indexed from 0, plus client.bin's tensors when it holds an end.
export function weightsFor(lo, hi) {
    const [bLo, bHi] = blockRange(lo, hi, L);
    const w = { layers: [] };
    for (let b = bLo; b < bHi; b++) {
        w.layers.push(Object.fromEntries(BLOCK_TENSORS.map((k) => [k, marker(`L${b}`)])));
    }
    if (lo === 0) w.wte = marker('wte');
    if (hi === N) { w.norm = marker('norm'); w.lm_head = marker('head'); }
    return w;
}

// What one position's download passes to Runner.append(): client.bin's tensors
// at either end, and one block's tensors in between.
export const pieceFor = (p) => p === 0 ? { wte: marker('wte') }
    : p === N - 1 ? { norm: marker('norm'), lm_head: marker('head') }
    : Object.fromEntries(BLOCK_TENSORS.map((k) => [k, marker(`L${p - 1}`)]));

// The same model as a hybrid checkpoint, shaped like Qwen3.8-27B: every other
// block is Gated DeltaNet (linear_attention), whose recurrent state has no
// position axis. Its tensors are tagged `R<b>`, not `L<b>`, so a trace and
// mesh.test.mjs's toy oracle can tell the two kinds of state apart.
export const LAYER_TYPES = Array.from({ length: L },
    (_, i) => i % 2 ? 'full_attention' : 'linear_attention');
export const DH = { ...D, hybrid: true, layer_types: LAYER_TYPES,
                    linear: { nK: 1, nV: 2, Dk: 4, Dv: 4, K4: 4, keyDim: 4, valueDim: 8, group: 2 } };

const LINEAR_TENSORS = ['input_layernorm', 'post_attention_layernorm', 'in_proj_q', 'in_proj_k',
                        'in_proj_v', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'conv1d_q', 'conv1d_k',
                        'conv1d_v', 'gate_params', 'gated_norm', 'out_proj', 'gate_proj', 'up_proj',
                        'down_proj'];

const hybridBlock = (b) => LAYER_TYPES[b] === 'linear_attention'
    ? { kind: 'linear_attention', ...Object.fromEntries(LINEAR_TENSORS.map((k) => [k, marker(`R${b}`)])) }
    : { kind: 'full_attention', ...Object.fromEntries(BLOCK_TENSORS.map((k) => [k, marker(`L${b}`)])) };

// weightsFor() / pieceFor() for DH.
export function hybridWeightsFor(lo, hi) {
    const [bLo, bHi] = blockRange(lo, hi, L);
    const w = { layers: [] };
    for (let b = bLo; b < bHi; b++) w.layers.push(hybridBlock(b));
    if (lo === 0) w.wte = marker('wte');
    if (hi === N) { w.norm = marker('norm'); w.lm_head = marker('head'); }
    return w;
}

export const hybridPieceFor = (p) => p === 0 || p === N - 1 ? pieceFor(p) : hybridBlock(p - 1);
