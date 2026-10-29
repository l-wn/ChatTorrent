// Every WGSL kernel and its JS wrapper.
//
// Layout of this file:
//   1. config       default model dims
//   2. gpu helpers  pipeline cache, bind groups, dispatch, buffer upload/read
//   3. KERNEL       every WGSL shader, keyed by name
//   4. ops          one JS wrapper per kernel (uniform + bind + dispatch)
//
// The forward pass is in model.js, the tests in web/dev/kernels.test.js, and
// the math each kernel implements in docs/model.md.
//
// Three caches, all keyed on content: pipelines by WGSL source, uniform buffers
// by byte payload, and bind groups by (pipeline, buffers). A pass reuses the
// objects that are the same in every layer.

// --- 1. Config ---

export const F32 = 4;

// SmolLM2-135M: the defaults for tests and the smoke check. Real checkpoints
// take their dims from the manifest (loader.js's dimsFromManifest). `group` =
// n_heads / n_kv_heads is the GQA sharing factor.
export const DIMS = {
    hidden:     576,
    n_layer:     30,
    n_heads:      9,
    n_kv_heads:   3,
    head_dim:    64,
    ffn:       1536,
    vocab:    49152,
    n_pos:     8192,
    group:        3,
    rms_eps:   1e-5,
    // SmolLM2's config.json sets rope_theta = 100000, not the Llama default of
    // 10000. A wrong base gives wrong but fluent text.
    rope_theta: 100000.0,
};

// --- 2. GPU helpers ---

// GPUBufferUsage bits as literals, so this module loads on insecure origins,
// where the `GPUBufferUsage` global is undefined.
const STORAGE   = 0x80;
const COPY_DST  = 0x08;
const COPY_SRC  = 0x04;
const UNIFORM   = 0x40;
const MAP_READ  = 0x01;

// Uniform payloads. WGSL rounds uniform struct sizes up to a multiple of 16
// bytes, so uniforms are built as 4 or 8 u32 slots, unused slots zero.
export const u4 = (a = 0, b = 0, c = 0, d = 0) => new Uint32Array([a, b, c, d]);
export const u8 = (a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0) =>
    new Uint32Array([a, b, c, d, e, f, g, h]);

// The GPU WebGPU chose, as a printable string. On a laptop the compositor and
// WebGPU can use different GPUs.
//
// `adapter.info` is the current API and `requestAdapterInfo()` the older async
// one. Either may return blank fields.
export async function adapterLabel(adapter) {
    const info = adapter.info
        ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
    const parts = [info?.description, info?.device, info?.vendor, info?.architecture]
        .filter((v) => v && String(v).trim());
    const seen = [...new Set(parts.map(String))];
    return seen.length ? seen.join(' · ') : 'unidentified GPU (the browser reports nothing)';
}

export function uploadF32(device, data, extraUsage = 0) {
    const buf = device.createBuffer({
        size: Math.max(data.byteLength, 16),
        usage: STORAGE | COPY_DST | COPY_SRC | extraUsage,
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
}

// f16 weights. The bytes are uploaded as-is (WebGPU buffers are untyped). A
// WeakSet records which buffers are f16, because the shader source depends on
// it and a GPUBuffer has no type.
//
// Only weights are f16. Activations, the KV cache and every accumulator are
// f32.
const f16Buffers = new WeakSet();

export const isF16 = (buf) => f16Buffers.has(buf);

export function uploadF16(device, data, extraUsage = 0) {
    // writeBuffer needs a multiple of 4 bytes, so an odd-length f16 tensor is
    // padded by one element. The pad is never read: every shader bounds its
    // indexing by the uniform dims.
    const padded = data.byteLength % 4 === 0 ? data : (() => {
        const out = new Uint16Array(data.length + 1);
        out.set(data);
        return out;
    })();
    const buf = uploadF32(device, padded, extraUsage);   // uploads raw bytes
    f16Buffers.add(buf);
    return buf;
}

// A zero-filled f16 weight buffer of `n` elements, registered for isF16().
// For the bench harness.
export function makeStorageF16(device, n, extraUsage = 0) {
    const buf = makeStorage(device, n * 2, extraUsage);
    f16Buffers.add(buf);
    return buf;
}

// q8 weights: int8 values packed four to a u32, plus one f32 scale per group
// of `group` values along K. A WeakMap records which buffers are quantized and
// where their scales are.
//
// q8 needs no optional device feature (f16 needs `shader-f16`, which some GPUs
// lack), so it runs anywhere, at a quarter of f32's size.
const q8Meta = new WeakMap();

export const isQ8 = (buf) => q8Meta.has(buf);
export const q8Of = (buf) => q8Meta.get(buf);

// `data` is the raw int8 block as bytes. `scales` is [N, ceil(K/group)] f32.
export function uploadQ8(device, data, scales, group) {
    const buf = uploadF32(device, data);          // uploads raw bytes
    q8Meta.set(buf, { scales: uploadF32(device, scales), group });
    return buf;
}

export function uploadU32(device, data) {
    const buf = device.createBuffer({
        size: Math.max(data.byteLength, 16),
        usage: STORAGE | COPY_DST | COPY_SRC,
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
}

export function makeStorage(device, byteLength, extraUsage = 0) {
    return device.createBuffer({
        size: Math.max(byteLength, 16),
        usage: STORAGE | COPY_DST | COPY_SRC | extraUsage,
    });
}

// --- Uniform / bind group caches ---
//
// Uniforms and bind groups depend only on their inputs, and every layer
// dispatches with the same dims, so a pass reuses a small set of objects.
// Uniforms are keyed on their byte payload.
//
// One uniform buffer per kernel, rewritten before each dispatch, would not
// work: queue.writeBuffer is applied before the command buffer executes, so
// every layer would see the last write. Each payload needs its own buffer.
//
// Entries that include T_past go stale every decode step, so both caches are
// bounded LRUs. Evicted entries are dropped, not destroyed, because a cached
// bind group may still reference an evicted buffer. For the same reason, do
// not destroy() a buffer that has been passed to an op while the device is
// still in use.
//
// Uniform payloads are shared across layers, so few are live at once. Bind
// groups reference each layer's own weight and KV buffers, so there are about
// n_layer × 14 stable ones. The limit is above that.
const UNIFORM_CACHE_LIMIT    = 256;
const BINDGROUP_CACHE_LIMIT  = 4096;

const uniformCache   = new WeakMap();   // device → Map<payload, GPUBuffer>
const bindGroupCache = new WeakMap();   // device → Map<ids, GPUBindGroup>

function perDevice(cache, device) {
    let m = cache.get(device);
    if (!m) { m = new Map(); cache.set(device, m); }
    return m;
}

function lruGet(map, key) {
    if (!map.has(key)) return undefined;
    const v = map.get(key);
    map.delete(key);            // reinsert as the most recent entry
    map.set(key, v);
    return v;
}

function lruSet(map, key, value, limit) {
    map.set(key, value);
    while (map.size > limit) map.delete(map.keys().next().value);
    return value;
}

// Counters, for the bench harness. `bindGroupsBuilt` counts cache misses: with
// all-static uniforms it stops rising after the first token. `dispatches`
// counts recorded dispatches, a measure of launch overhead.
let bindGroupsBuilt = 0;
let dispatches = 0;
export const counters = () => ({ bindGroupsBuilt, dispatches });
export const resetCounters = () => { bindGroupsBuilt = 0; dispatches = 0; };

// Stable ids for GPU objects, so a bind group can be keyed by identity.
let nextObjId = 1;
const objIds = new WeakMap();
function idOf(o) {
    let i = objIds.get(o);
    if (i === undefined) { i = nextObjId++; objIds.set(o, i); }
    return i;
}

export function uniform(device, data) {
    const cache = perDevice(uniformCache, device);
    const key = data.join(',');
    const hit = lruGet(cache, key);
    if (hit) return hit;

    const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const size = Math.max(16, Math.ceil(view.byteLength / 16) * 16);
    const buf = device.createBuffer({ size, usage: UNIFORM | COPY_DST });
    device.queue.writeBuffer(buf, 0, view);
    return lruSet(cache, key, buf, UNIFORM_CACHE_LIMIT);
}

// Cache occupancy, for checking the hit rate.
export function cacheStats(device) {
    return {
        uniforms:        perDevice(uniformCache, device).size,
        bindGroups:      perDevice(bindGroupCache, device).size,
        uniformLimit:    UNIFORM_CACHE_LIMIT,
        bindGroupLimit:  BINDGROUP_CACHE_LIMIT,
    };
}

// A mappable destination for readbacks, allocated once and reused. A per-token
// caller records the copy on its own encoder and maps this after its one
// submit, which avoids readF32()'s extra command buffer.
export function makeReadback(device, byteLength) {
    return device.createBuffer({
        size: Math.max(byteLength, 16),
        usage: COPY_DST | MAP_READ,
    });
}

// Submits its own encoder. For one-off reads, not inside a forward pass.
export async function readF32(device, src, byteLength) {
    const dst = device.createBuffer({ size: byteLength, usage: COPY_DST | MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, byteLength);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(dst.getMappedRange().slice(0));
    dst.unmap();
    dst.destroy();
    return out;
}

// device → Map<wgsl source, pipeline>. Keyed by source, so kernels with
// identical code share a pipeline and each source compiles once.
const pipelineCache = new WeakMap();

export function getPipeline(device, code, entry = 'main') {
    let perDevice = pipelineCache.get(device);
    if (!perDevice) { perDevice = new Map(); pipelineCache.set(device, perDevice); }

    let pipe = perDevice.get(code);
    if (pipe) return pipe;

    const module = device.createShaderModule({ code });
    pipe = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: entry },
    });
    perDevice.set(code, pipe);
    return pipe;
}

// Buffers are bound at @binding(0), @binding(1), ... in array order. Cached
// on (pipeline, buffer identities) in the bind group LRU.
export function bind(device, pipeline, buffers) {
    const cache = perDevice(bindGroupCache, device);
    const key = `${idOf(pipeline)}:${buffers.map(idOf).join(',')}`;
    const hit = lruGet(cache, key);
    if (hit) return hit;

    bindGroupsBuilt++;
    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    return lruSet(cache, key, bg, BINDGROUP_CACHE_LIMIT);
}

// Record one dispatch on a pass the caller owns and keeps open. Beginning a
// pass has a cost, so a whole forward pass shares one.
//
// The spec guarantees that dispatches in one compute pass execute in order and
// that each sees the writes of the one before.
//
// `name` labels this dispatch in a profile (model/gpuprof.js). It is used only
// when `pass.prof` is set.
export function dispatch(pass, pipeline, bindings, x, y = 1, z = 1, name) {
    dispatches++;
    pass.prof?.mark(name);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(x, y, z);
}

// Open one compute pass, pass it to `record`, and always end it.
//
// `prof`, from model/gpuprof.js, substitutes a pass with timestamp writes, or
// at kernel level an object that opens one timed pass per dispatch.
export function withPass(encoder, record, label, prof = null) {
    const pass = prof ? prof.pass(encoder, label)
        : encoder.beginComputePass(label ? { label } : undefined);
    try {
        return record(pass);
    } finally {
        pass.end();
    }
}

// --- 3. Kernels ---
//
// Each entry documents its binding order and dispatch shape. The wrapper in
// section 4 must match both.

export const KERNEL = {};

// RMSNorm's eps is a `const` in the WGSL source, not a uniform, so the
// compiler can fold it. Each distinct value compiles one variant, cached by
// source in getPipeline().
//
// Source strings are memoized by key, since wrappers request them on every
// dispatch.
const srcMemo = new Map();
const memoSrc = (key, build) => {
    let src = srcMemo.get(key);
    if (src === undefined) { src = build(); srcMemo.set(key, src); }
    return src;
};

// An integral Number prints as `100000`, which WGSL reads as an abstract-int
// literal. It would convert to f32, but `100000.0` is written to make the type
// explicit.
const f32lit = (v) => (Number.isInteger(v) ? v.toFixed(1) : String(v));

// Weight storage is f16, q8 or f32. Everything else, including every
// accumulator, is f32. Each kernel that reads f16 or f32 weights is one source
// template with the binding's element type substituted. `enable f16` must come
// before every declaration, so it starts the template. Reads are always
// wrapped in `f32(...)`, which is a no-op in the f32 build.
const wEnable = (f16) => (f16 ? 'enable f16;\n' : '');
const wType   = (f16) => (f16 ? 'f16' : 'f32');

// --- matmul ---
// Y[M,N] = X[M,K] @ W[N,K]^T. Weights are stored [out, in] = [N, K], so row
// `col` of W holds every weight for output neuron `col`.
//
// Two builds for two shapes:
//
//   decode   M = 1      a matrix-vector product. Reading W dominates.
//   prefill  M = T_MAX  a full GEMM. Every row uses the same W, so W is
//                       staged once per tile and reused.

// --- decode (GEMV) ---
//
// LANES threads cooperate on one output column, and COLS columns share a
// workgroup. Each lane loads a vec4 at a time, so adjacent lanes read adjacent
// addresses of W. LANES = 32 matches the subgroup width on most hardware,
// which keeps the workgroup-memory reduction cheap.
//
// Every thread reaches every barrier. Threads outside the matrix contribute 0
// and do not return early, because a barrier that only some invocations reach
// is undefined behaviour.
//
// Accumulation is f32 even with f16 weights: `vec4<f32>(w)` widens before the
// dot product. kernels.test.js holds the f16 build to the f32 tolerance, which
// catches an f16 accumulator.
const GEMV_LANES = 32;
const GEMV_COLS  = 8;
const GEMV_WG    = GEMV_LANES * GEMV_COLS;

// vec4 needs K % 4 == 0. The scalar build handles any other K (the unit tests
// use K = 7 and K = 2053).
export const matmulGemvSrc = (f16 = false, vec4 = false) =>
    memoSrc(`matmul_gemv:${f16}:${vec4}`, () => /* wgsl */ `${wEnable(f16)}
struct U { M: u32, K: u32, N: u32, _p: u32 }
${vec4 ? `
@group(0) @binding(0) var<storage, read>       X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W: array<vec4<${wType(f16)}>>;
` : `
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<${wType(f16)}>;
`}
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;

var<workgroup> red: array<f32, ${GEMV_WG}u>;

@compute @workgroup_size(${GEMV_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid  = lid.x;
    let lane = tid % ${GEMV_LANES}u;          // lane within this column
    let col  = wg.x * ${GEMV_COLS}u + tid / ${GEMV_LANES}u;
    let row  = wg.y;
    let live = col < u.N && row < u.M;

    // Out-of-range threads run the reduction and reach every barrier. They
    // contribute 0 and do not store. A barrier that only some invocations in a
    // workgroup reach is undefined behaviour.
    var acc = 0.0;
    if (live) {
${vec4 ? `
        let K4 = u.K / 4u;
        let xBase = row * K4;
        let wBase = col * K4;
        for (var i = lane; i < K4; i += ${GEMV_LANES}u) {
            acc += dot(X[xBase + i], vec4<f32>(W[wBase + i]));
        }
` : `
        let xBase = row * u.K;
        let wBase = col * u.K;
        for (var i = lane; i < u.K; i += ${GEMV_LANES}u) {
            acc += X[xBase + i] * f32(W[wBase + i]);
        }
`}
    }

    // Reduce within each column's own LANES-wide segment. s < LANES and the
    // segment is LANES-aligned, so tid + s stays inside the column and the
    // columns in this workgroup reduce independently.
    red[tid] = acc;
    workgroupBarrier();
    for (var s = ${GEMV_LANES}u / 2u; s > 0u; s >>= 1u) {
        if (lane < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }

    if (lane == 0u && live) { Y[row * u.N + col] = red[tid]; }
}
`);

// --- decode, q8 ---
//
// The lane layout follows the quantization group: `group` weights are
// `group/4` u32 words, so that many lanes cover one group and all read the
// same scale. The words the lanes read are consecutive, so the weight load is
// coalesced.
//
// Accumulation is f32. The scale multiplies each group's dot product, not
// each weight.
export const matmulGemvQ8Src = (group) =>
    memoSrc(`matmul_gemv_q8:${group}`, () => {
        const LANES = group / 4;                       // u32 words in one group
        const COLS  = Math.max(1, Math.floor(256 / LANES));
        const WG    = LANES * COLS;
        return /* wgsl */ `
struct U { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<storage, read>       X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;
@group(0) @binding(4) var<storage, read>       S: array<f32>;

var<workgroup> red: array<f32, ${WG}u>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid  = lid.x;
    let lane = tid % ${LANES}u;
    let col  = wg.x * ${COLS}u + tid / ${LANES}u;
    let row  = wg.y;
    let live = col < u.N && row < u.M;

    let K4     = u.K / 4u;                             // u32 words per row
    let nGroup = (u.K + ${group}u - 1u) / ${group}u;   // scales per row

    var acc = 0.0;
    if (live) {
        let xBase = row * K4;
        let wBase = col * K4;
        let sBase = col * nGroup;
        for (var g = 0u; g < nGroup; g++) {
            let idx = g * ${LANES}u + lane;
            // The last group is partial when K is not a multiple of the group.
            if (idx < K4) {
                let q = vec4<f32>(unpack4xI8(W[wBase + idx]));
                acc += S[sBase + g] * dot(X[xBase + idx], q);
            }
        }
    }

    red[tid] = acc;
    workgroupBarrier();
    for (var s = ${LANES}u / 2u; s > 0u; s >>= 1u) {
        if (lane < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }

    if (lane == 0u && live) { Y[row * u.N + col] = red[tid]; }
}
`;
    });

// --- a few rows (batched GEMV) ---
//
// For 2 <= M <= GEMV_ROWS_MAX: a speculative verify job, a batch of several
// chats' jobs, or a short prefill tail. The GEMV above with R rows per
// workgroup: each weight is read once and used for every row. Each row's
// arithmetic is the GEMV's (the same lane split, the same accumulation
// expression, the same reduction tree), so every row is bit-identical to a
// one-row call. That is what makes a verified token the token plain decoding
// would have produced, and a batched chat's token the one it gets alone.
//
// The reduction runs RED_ROWS rows at a time through one shared buffer, so
// the workgroup memory stays within WebGPU's default 16 KiB at any R. Up to
// RED_ROWS rows R is M. Past it R is M rounded up to a multiple of RED_ROWS,
// so few variants are compiled, and rows from M on are not stored. Past
// RED_ROWS q8 weights use matmulRowsQ8Src below, which is faster there.
export const GEMV_ROWS_MAX = 32;
const RED_ROWS = 8;
export const gemvRows = (M) => (M <= RED_ROWS ? M : Math.ceil(M / RED_ROWS) * RED_ROWS);

export const matmulGemvRowsSrc = (f16, vec4, R) =>
    memoSrc(`matmul_gemv_rows:${f16}:${vec4}:${R}`, () => /* wgsl */ `${wEnable(f16)}
struct U { M: u32, K: u32, N: u32, _p: u32 }
${vec4 ? `
@group(0) @binding(0) var<storage, read>       X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W: array<vec4<${wType(f16)}>>;
` : `
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<${wType(f16)}>;
`}
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;

const R = ${R}u;
const RED = ${Math.min(R, 8)}u;
var<workgroup> red: array<f32, ${GEMV_WG * Math.min(R, 8)}u>;

@compute @workgroup_size(${GEMV_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid  = lid.x;
    let lane = tid % ${GEMV_LANES}u;
    let col  = wg.x * ${GEMV_COLS}u + tid / ${GEMV_LANES}u;
    let liveCol = col < u.N;

    var acc: array<f32, ${R}>;
    for (var r = 0u; r < R; r++) { acc[r] = 0.0; }
    if (liveCol) {
${vec4 ? `
        let K4 = u.K / 4u;
        let wBase = col * K4;
        for (var i = lane; i < K4; i += ${GEMV_LANES}u) {
            let w = vec4<f32>(W[wBase + i]);
            for (var r = 0u; r < R; r++) { acc[r] += dot(X[r * K4 + i], w); }
        }
` : `
        let wBase = col * u.K;
        for (var i = lane; i < u.K; i += ${GEMV_LANES}u) {
            let w = f32(W[wBase + i]);
            for (var r = 0u; r < R; r++) { acc[r] += X[r * u.K + i] * w; }
        }
`}
    }

    for (var r0 = 0u; r0 < R; r0 += RED) {
        for (var r = 0u; r < RED; r++) { red[r * ${GEMV_WG}u + tid] = acc[r0 + r]; }
        workgroupBarrier();
        for (var s = ${GEMV_LANES}u / 2u; s > 0u; s >>= 1u) {
            if (lane < s) {
                for (var r = 0u; r < RED; r++) {
                    red[r * ${GEMV_WG}u + tid] += red[r * ${GEMV_WG}u + tid + s];
                }
            }
            workgroupBarrier();
        }
        if (lane == 0u && liveCol) {
            for (var r = 0u; r < RED; r++) {
                if (r0 + r < u.M) { Y[(r0 + r) * u.N + col] = red[r * ${GEMV_WG}u + tid]; }
            }
        }
        workgroupBarrier();
    }
}
`);

// The q8 GEMV with R rows per workgroup, on the same terms as above.
export const matmulGemvRowsQ8Src = (group, R) =>
    memoSrc(`matmul_gemv_rows_q8:${group}:${R}`, () => {
        const LANES = group / 4;
        const COLS  = Math.max(1, Math.floor(256 / LANES));
        const WG    = LANES * COLS;
        return /* wgsl */ `
struct U { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<storage, read>       X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;
@group(0) @binding(4) var<storage, read>       S: array<f32>;

const R = ${R}u;
const RED = ${Math.min(R, 8)}u;
var<workgroup> red: array<f32, ${WG * Math.min(R, 8)}u>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid  = lid.x;
    let lane = tid % ${LANES}u;
    let col  = wg.x * ${COLS}u + tid / ${LANES}u;
    let liveCol = col < u.N;

    let K4     = u.K / 4u;
    let nGroup = (u.K + ${group}u - 1u) / ${group}u;

    var acc: array<f32, ${R}>;
    for (var r = 0u; r < R; r++) { acc[r] = 0.0; }
    if (liveCol) {
        let wBase = col * K4;
        let sBase = col * nGroup;
        for (var g = 0u; g < nGroup; g++) {
            let idx = g * ${LANES}u + lane;
            if (idx < K4) {
                let q  = vec4<f32>(unpack4xI8(W[wBase + idx]));
                let sc = S[sBase + g];
                for (var r = 0u; r < R; r++) { acc[r] += sc * dot(X[r * K4 + idx], q); }
            }
        }
    }

    for (var r0 = 0u; r0 < R; r0 += RED) {
        for (var r = 0u; r < RED; r++) { red[r * ${WG}u + tid] = acc[r0 + r]; }
        workgroupBarrier();
        for (var s = ${LANES}u / 2u; s > 0u; s >>= 1u) {
            if (lane < s) {
                for (var r = 0u; r < RED; r++) {
                    red[r * ${WG}u + tid] += red[r * ${WG}u + tid + s];
                }
            }
            workgroupBarrier();
        }
        if (lane == 0u && liveCol) {
            for (var r = 0u; r < RED; r++) {
                if (r0 + r < u.M) { Y[(r0 + r) * u.N + col] = red[r * ${WG}u + tid]; }
            }
        }
        workgroupBarrier();
    }
}
`;
    });

// --- many rows, q8 (batched decode) ---
//
// The q8 GEMV for 9 to GEMV_ROWS_MAX rows, each row bit-identical to a
// one-row call. matmulGemvRowsQ8Src has every thread read its rows of X from
// memory for every weight word, which past ~8 rows costs more than reading
// the weights. Here the workgroup stages ROWS_KCW words of every row of X in
// workgroup memory at a time, and threads read them from there. On a 1070 Ti
// at Qwen3.8-27B's gate shape, 16 rows then take what 8 do and 32 rows twice
// that, against 5x for the plain rows kernel.
//
// Per (row, column) the arithmetic is the GEMV's: lane `l` of LANES takes
// words g * LANES + l for g in increasing order, accumulates
// `acc += scale * dot(x, q)`, and the lanes are summed by the same tree, 8
// rows at a time. Only where X is read from changes.
//
// R is a multiple of 8 (gemvRows). Rows from M on are computed on whatever X
// holds there and not stored.
const ROWS_WORDS = 16;                       // words of X staged per row
const rowsKcw = (group) => Math.max(group / 4, ROWS_WORDS);
// Whether the staged X and the reduction fit WebGPU's default 16 KiB of
// workgroup memory. If not, matmulNobias uses matmulGemvRowsQ8Src.
export const rowsQ8Fits = (group, R) => R * rowsKcw(group) * 16 + 256 * 8 * 4 <= 16384;

export const matmulRowsQ8Src = (group, R) =>
    memoSrc(`matmul_rows_q8:${group}:${R}`, () => {
        const LANES = group / 4;
        const COLS  = Math.floor(256 / LANES);
        const WG    = LANES * COLS;
        const KCW   = rowsKcw(group);
        return /* wgsl */ `
struct U { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<storage, read>       X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;
@group(0) @binding(4) var<storage, read>       S: array<f32>;

const R = ${R}u;
var<workgroup> red: array<f32, ${WG * 8}u>;
var<workgroup> xs: array<vec4<f32>, ${R * KCW}u>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid  = lid.x;
    let lane = tid % ${LANES}u;
    let col  = wg.x * ${COLS}u + tid / ${LANES}u;
    let liveCol = col < u.N;

    let K4     = u.K / 4u;
    let nGroup = (u.K + ${group}u - 1u) / ${group}u;
    let wBase  = col * K4;
    let sBase  = col * nGroup;

    var acc: array<f32, ${R}>;
    for (var r = 0u; r < R; r++) { acc[r] = 0.0; }

    for (var k0 = 0u; k0 < K4; k0 += ${KCW}u) {
        for (var i = tid; i < ${R * KCW}u; i += ${WG}u) {
            let w = k0 + i % ${KCW}u;
            xs[i] = select(vec4<f32>(0.0), X[(i / ${KCW}u) * K4 + w], w < K4);
        }
        workgroupBarrier();
        if (liveCol) {
            for (var j = 0u; j < ${KCW / LANES}u; j++) {
                let idx = k0 + j * ${LANES}u + lane;
                if (idx < K4) {
                    let q  = vec4<f32>(unpack4xI8(W[wBase + idx]));
                    let sc = S[sBase + idx / ${LANES}u];
                    for (var r = 0u; r < R; r++) {
                        acc[r] += sc * dot(xs[r * ${KCW}u + j * ${LANES}u + lane], q);
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var r0 = 0u; r0 < R; r0 += 8u) {
        for (var r = 0u; r < 8u; r++) { red[r * ${WG}u + tid] = acc[r0 + r]; }
        workgroupBarrier();
        for (var s = ${LANES}u / 2u; s > 0u; s >>= 1u) {
            if (lane < s) {
                for (var r = 0u; r < 8u; r++) {
                    red[r * ${WG}u + tid] += red[r * ${WG}u + tid + s];
                }
            }
            workgroupBarrier();
        }
        if (lane == 0u && liveCol) {
            for (var r = 0u; r < 8u; r++) {
                if (r0 + r < u.M) { Y[(r0 + r) * u.N + col] = red[r * ${WG}u + tid]; }
            }
        }
        workgroupBarrier();
    }
}
`;
    });

// --- prefill, q8 ---
// The tiled kernel with dequantization in the staging step. Each staged weight
// is used TILE times, so the dequantization cost is amortized. Only the read
// of W differs from the f32 build.
export const matmulTiledQ8Src = (group) =>
    memoSrc(`matmul_tiled_q8:${group}`, () => /* wgsl */ `
struct U { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;
@group(0) @binding(4) var<storage, read>       S: array<f32>;

var<workgroup> Xs: array<f32, ${16 * 16}u>;
var<workgroup> Ws: array<f32, ${16 * 16}u>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.y * 16u + lid.y;
    let col = wg.x * 16u + lid.x;
    let nGroup = (u.K + ${group}u - 1u) / ${group}u;

    var acc = 0.0;
    let tiles = (u.K + 15u) / 16u;

    for (var t = 0u; t < tiles; t++) {
        let k = t * 16u + lid.x;

        var xv = 0.0;
        if (row < u.M && k < u.K) { xv = X[row * u.K + k]; }
        Xs[lid.y * 16u + lid.x] = xv;

        let wn = wg.x * 16u + lid.y;
        var wv = 0.0;
        if (wn < u.N && k < u.K) {
            let flat = wn * u.K + k;
            var q4 = unpack4xI8(W[flat / 4u]);
            wv = f32(q4[flat % 4u]) * S[wn * nGroup + k / ${group}u];
        }
        Ws[lid.x * 16u + lid.y] = wv;

        workgroupBarrier();
        for (var j = 0u; j < 16u; j++) {
            acc += Xs[lid.y * 16u + j] * Ws[j * 16u + lid.x];
        }
        workgroupBarrier();
    }

    if (row < u.M && col < u.N) { Y[row * u.N + col] = acc; }
}
`);

// --- prefill: square tiles staged through workgroup memory ---
//
// A 16x16 tile of Y per workgroup. Each iteration stages a 16x16 tile of X and
// of W into workgroup memory, so every value fetched from global memory is
// used 16 times. bindings: X, W, Y, u{M,K,N}   dispatch: ceil(N/16), ceil(M/16)
//
// In both staging reads lid.x, the fastest-varying index, walks K. Each load
// is then a contiguous run across threads for both operands, and the store to
// Y is contiguous across `col`.
const TILE = 16;

export const matmulTiledSrc = (f16 = false) =>
    memoSrc(`matmul_tiled:${f16}`, () => /* wgsl */ `${wEnable(f16)}
struct U { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<${wType(f16)}>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             u: U;

// Xs[m][k] and Ws[k][n]. Ws is stored transposed relative to how W is read, so
// the inner loop walks both operands contiguously.
var<workgroup> Xs: array<f32, ${TILE * TILE}u>;
var<workgroup> Ws: array<f32, ${TILE * TILE}u>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.y * ${TILE}u + lid.y;
    let col = wg.x * ${TILE}u + lid.x;

    var acc = 0.0;
    let tiles = (u.K + ${TILE}u - 1u) / ${TILE}u;

    for (var t = 0u; t < tiles; t++) {
        let k = t * ${TILE}u + lid.x;      // lid.x walks K for both loads

        // X[row, k]
        var xv = 0.0;
        if (row < u.M && k < u.K) { xv = X[row * u.K + k]; }
        Xs[lid.y * ${TILE}u + lid.x] = xv;

        // W[wg.x*TILE + lid.y, k] -> Ws[k][lid.y]
        let wn = wg.x * ${TILE}u + lid.y;
        var wv = 0.0;
        if (wn < u.N && k < u.K) { wv = f32(W[wn * u.K + k]); }
        Ws[lid.x * ${TILE}u + lid.y] = wv;

        workgroupBarrier();

        for (var j = 0u; j < ${TILE}u; j++) {
            acc += Xs[lid.y * ${TILE}u + j] * Ws[j * ${TILE}u + lid.x];
        }
        workgroupBarrier();
    }

    if (row < u.M && col < u.N) { Y[row * u.N + col] = acc; }
}
`);

// --- gather_embed_nopos ---
// out[T,H] = wte[ids[t]]. No wpe: RoPE carries position.
// `ids` is an f32 buffer holding integral values (uploaded with uploadF32),
// and the kernel casts them to u32.
// bindings: ids, wte, out, u{T,H}   dispatch: ceil(H/64), T
export const gatherEmbedNoPosSrc = (f16 = false) =>
    memoSrc(`gather_embed_nopos:${f16}`, () => /* wgsl */ `${wEnable(f16)}
struct U { T: u32, H: u32, _p: u32}
@group(0) @binding(0) var<storage, read>       ids : array<f32>;
@group(0) @binding(1) var<storage, read>       wte : array<${wType(f16)}>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>             u   : U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y;
    let h = gid.x;
    if (t >= u.T || h >= u.H) { return; }

   let v = u32(ids[t]);
   out[t * u.H + h] = f32(wte[v * u.H + h]);
}
`);
KERNEL.gather_embed_nopos = gatherEmbedNoPosSrc(false);

// The q8 build. wte is the largest single tensor, so it is quantized too.
//
// Unlike the matmuls, this has no alignment constraint: `flat` is the element
// index into the whole [vocab*H] int8 block, and the word and byte are derived
// from it for any H.
export const gatherEmbedNoPosQ8Src = (group) =>
    memoSrc(`gather_embed_nopos_q8:${group}`, () => /* wgsl */ `
struct U { T: u32, H: u32, _p: u32}
@group(0) @binding(0) var<storage, read>       ids : array<f32>;
@group(0) @binding(1) var<storage, read>       wte : array<u32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>             u   : U;
@group(0) @binding(4) var<storage, read>       S   : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y;
    let h = gid.x;
    if (t >= u.T || h >= u.H) { return; }

    let v      = u32(ids[t]);
    let nGroup = (u.H + ${group}u - 1u) / ${group}u;
    let flat   = v * u.H + h;

    var q4 = unpack4xI8(wte[flat / 4u]);
    out[t * u.H + h] = f32(q4[flat % 4u]) * S[v * nGroup + h / ${group}u];
}
`);


// --- rmsnorm ---
// y[t,h] = x[t,h] / sqrt(mean_h(x^2) + eps) * weight[h]. No mean subtraction,
// no bias. One workgroup per row, one cooperative reduction over Σx².
// bindings: x, weight, y, u{T,H}    dispatch: T
// eps varies by checkpoint (config.json rms_norm_eps), so the source is a
// function of it. KERNEL.rmsnorm is the build for DIMS.
//
// Structured like the GEMV: threads stride across H, partials reduce in
// workgroup memory, and every thread reaches every barrier. A row past T runs
// the reduction and does not store.
const RMS_WG = 256;

export const rmsnormSrc = (eps, f16 = false) =>
    memoSrc(`rmsnorm:${eps}:${f16}`, () => /* wgsl */ `${wEnable(f16)}
struct U { T: u32, H: u32 }
@group(0) @binding(0) var<storage, read>       x      : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<${wType(f16)}>;
@group(0) @binding(2) var<storage, read_write> y      : array<f32>;
@group(0) @binding(3) var<uniform>             u      : U;

var<workgroup> partial: array<f32, ${RMS_WG}u>;

@compute @workgroup_size(${RMS_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let t    = wg.x;
    let live = t < u.T;
    let row  = t * u.H;

    var ss = 0.0;
    if (live) {
        for (var h = lid.x; h < u.H; h += ${RMS_WG}u) {
            let v = x[row + h];
            ss += v*v;
        }
    }
    partial[lid.x] = ss;
    workgroupBarrier();

    for (var s = ${RMS_WG}u / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) { partial[lid.x] += partial[lid.x + s]; }
        workgroupBarrier();
    }

    // No further barrier is needed: the loop's last iteration ends in one, so
    // partial[0] is final and visible to every thread, and nothing below
    // writes to partial.
    let scale = 1.0 / sqrt(partial[0] / f32(u.H) + ${f32lit(eps)});

    if (live) {
        for (var h = lid.x; h < u.H; h += ${RMS_WG}u) {
            y[row + h] = x[row + h] * scale * f32(weight[h]);
        }
    }
}
`);
KERNEL.rmsnorm = rmsnormSrc(DIMS.rms_eps);

// --- reshape_heads ---
// q[t, h*D+d] → Qh[h, t, d]. Row-major to head-major, Q only.
// bindings: src, dst, u{T,n_heads,D}          dispatch: ceil(D/64), T, n_heads
KERNEL.reshape_heads = /* wgsl */ `
struct U { T: u32, n_heads: u32, D: u32 }
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             u   : U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y;
    let h = gid.z;
    let d = gid.x;

    if (t >= u.T || h >= u.n_heads || d >= u.D ) { return; }

    dst[h*u.T*u.D + t*u.D + d] = src[t*(u.n_heads*u.D) + h*u.D +d];
}
`;

// --- store_kv ---
// [T_new, n_kv*D] projection → the int8 KV cache at time offset T_past.
// Called once for K (with RoPE) and once for V (rot = 0).
//
// Format: the cache is u32 words in rows of one (page, kv head, slot). A row
// is D/4 words of packed int8 (four values per word, low byte first), then
// D/QB f32 scales, one per QB values: x ≈ q * scale, with scale = max|x|/127
// over the block. QB = min(32, D), split.js's kvQB(). Row stride
// RW = D/4 + D/QB words.
//
// Paging: a session's KV is a list of fixed-size pages from a pool shared by
// the whole peer. `pages` maps a logical token position to its physical page,
// so memory follows the tokens in use (see model.js's makePagePool()). Every
// kernel that touches the cache reads that table. Row r of page p, head n is
// at ((p*n_kv + n)*page + r) * RW.
//
// A contiguous cache is the one-page case: with `page` = N_max and
// `pages = [0]`, the row reduces to n*N_max + t.
//
// RoPE is applied here, before quantizing, with the rope kernel's arithmetic
// and table (split-half pairing; channels [rot, D) pass through). A row's
// bytes depend only on that row's input, so a rewind, a speculative overwrite
// or a chunked prefill writes the same bits as one job would.
// bindings: src, cache, u{T_new,T_past,page,n_kv,D,rot}, pages, tbl
// dispatch: ceil(T_new*n_kv*(D/QB) / 64)
KERNEL.store_kv = /* wgsl */ `
struct U { T_new: u32, T_past: u32, page: u32, n_kv: u32, D: u32, rot: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read>       src   : array<f32>;
@group(0) @binding(1) var<storage, read_write> cache : array<u32>;
@group(0) @binding(2) var<uniform>             u     : U;
@group(0) @binding(3) var<storage, read>       pages : array<u32>;
@group(0) @binding(4) var<storage, read>       tbl   : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let QB = min(32u, u.D);
    let nb = u.D / QB;
    let id = gid.x;
    if (id >= u.T_new * u.n_kv * nb) { return; }
    let b   = id % nb;
    let n   = (id / nb) % u.n_kv;
    let t_n = id / (nb * u.n_kv);
    let t   = u.T_past + t_n;
    let base = t_n * u.n_kv * u.D + n * u.D;
    let half = u.rot / 2u;

    var x: array<f32, 32>;
    var amax = 0.0;
    for (var j = 0u; j < QB; j++) {
        let c = b * QB + j;
        var v = src[base + c];
        if (c < u.rot) {
            let i = c % half;
            let k = (t * half + i) * 2u;
            let cs = tbl[k];
            let sn = tbl[k + 1u];
            let a = src[base + i];
            let o = src[base + i + half];
            if (c < half) { v = a*cs - o*sn; } else { v = o*cs + a*sn; }
        }
        x[j] = v;
        amax = max(amax, abs(v));
    }
    let scale = amax / 127.0;
    let inv = select(0.0, 1.0 / scale, scale > 0.0);

    let RW = u.D / 4u + nb;
    let at = ((pages[t / u.page] * u.n_kv + n) * u.page + t % u.page) * RW;
    for (var w = 0u; w < QB / 4u; w++) {
        let q = round(vec4<f32>(x[4u*w], x[4u*w + 1u], x[4u*w + 2u], x[4u*w + 3u]) * inv);
        cache[at + b * (QB / 4u) + w] = pack4xI8(vec4<i32>(clamp(q, vec4<f32>(-127.0), vec4<f32>(127.0))));
    }
    cache[at + u.D / 4u + b] = bitcast<u32>(scale);
}
`;

// --- rope ---
// In-place rotation of an f32 [heads, rows, D] buffer: Q in the forward pass.
// K is rotated by store_kv as it is quantized, with the same arithmetic and
// table. Llama split-half pairing: channel i pairs with i+rot_dim/2, not
// (2i, 2i+1).
//
// rot_dim (rotation width) and head_dim (per-head storage stride) are separate
// so a partial-rotary checkpoint (Qwen3.8-27B: rot_dim = head_dim/4) leaves
// channels [rot_dim, head_dim) unchanged, matching HF's apply_rotary_pos_emb.
// `page` is the buffer's rows per head and row0 the first row to rotate. A
// paged buffer passes its page table; a dense one is the one-page case with
// `page` = T and `pages = [0]`, the wrapper's default.
//
// The angles come from a table built on the CPU in f64, not from sin/cos in
// the shader. GPU trig accuracy varies by vendor by more than rope's
// tolerance, and peers splitting a model must compute identical values.
// bindings: buf, u{n_heads,T,rot_dim,head_dim,T_past,page,row0}, tbl, pages
// dispatch: ceil((rot_dim/2)/64), T, n_heads
KERNEL.rope = /* wgsl */ `
struct U { n_heads: u32, T: u32, rot_dim: u32, head_dim: u32, T_past: u32, page: u32, row0: u32, _p: u32 }
@group(0) @binding(0) var<storage, read_write> buf   : array<f32>;
@group(0) @binding(1) var<uniform>             u     : U;
@group(0) @binding(2) var<storage, read>       tbl   : array<f32>;
@group(0) @binding(3) var<storage, read>       pages : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;   // pair index, 0..rot_dim/2. This thread owns channels i and i+half.
    let t = gid.y;
    let h = gid.z;

    let half = u.rot_dim / 2u;
    if (i >= half || t >= u.T || h >= u.n_heads) { return; }

    // The angle depends on absolute position, so it is the same for Q and for
    // the cache. Only the row holding the data differs. tbl is [pos][pair]
    // with cos and sin interleaved.
    let k = ((u.T_past + t) * half + i) * 2u;
    let c = tbl[k];
    let s = tbl[k + 1u];

    // The stride is head_dim, not rot_dim: channels [rot_dim, head_dim) are
    // not rotated but are stored in the row.
    let row = u.row0 + t;
    let phys = pages[row / u.page];
    let base = (phys*u.n_heads + h)*u.page*u.head_dim + (row % u.page)*u.head_dim;

    // Llama split-half: i pairs with i+half. Both reads come before either
    // write, because the second output needs the first input's old value.
    let a = buf[base + i];
    let b = buf[base + i + half];
    buf[base + i]        = a*c - b*s;
    buf[base + i + half] = b*c + a*s;
}
`;

// --- qk_scores_gqa ---
// scores[n_heads, T_new, T_act] = Q @ K_cache^T * (1/sqrt(D)), for the T_new
// query rows starting at r0 of Q's T_q. GQA: query head h reads KV head
// h / group. K is the int8 cache (see store_kv): each block's int8 dot
// product is scaled once.
//
// One `pages` lookup per thread, since each thread has one `t_a`.
// bindings: Q, K_cache, scores, u{T_new,T_act,page,D,n_heads,group,r0,T_q}, pages
// dispatch: ceil(T_new/8), ceil(T_act/8), n_heads
KERNEL.qk_scores_gqa = /* wgsl */ `
struct U { T_new: u32, T_act: u32, page: u32, D: u32, n_heads: u32, group: u32, r0: u32, T_q: u32 }
@group(0) @binding(0) var<storage, read>       Q       : array<f32>;
@group(0) @binding(1) var<storage, read>       K_cache : array<u32>;
@group(0) @binding(2) var<storage, read_write> scores  : array<f32>;
@group(0) @binding(3) var<uniform>             u       : U;
@group(0) @binding(4) var<storage, read>       pages   : array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t_n  = gid.x;
    let t_a  = gid.y;
    let h    = gid.z;

    if (t_n >= u.T_new || t_a >= u.T_act || h >= u.n_heads) { return; }

    let QB = min(32u, u.D);
    let nb = u.D / QB;
    let RW = u.D / 4u + nb;
    let kv_h = h / u.group;
    let n_kv = u.n_heads / u.group;
    let Q_base = h*u.T_q*u.D + (u.r0 + t_n)*u.D;
    let at = ((pages[t_a / u.page]*n_kv + kv_h)*u.page + t_a % u.page) * RW;

    var sum = 0.0;
    for (var b = 0u; b < nb; b++) {
        var s = 0.0;
        for (var w = 0u; w < QB / 4u; w++) {
            let k4 = vec4<f32>(unpack4xI8(K_cache[at + b*(QB/4u) + w]));
            let i  = Q_base + b*QB + 4u*w;
            s += dot(vec4<f32>(Q[i], Q[i + 1u], Q[i + 2u], Q[i + 3u]), k4);
        }
        sum += s * bitcast<f32>(K_cache[at + u.D/4u + b]);
    }

    let scale = 1.0 / sqrt(f32(u.D));
    scores[h*u.T_new*u.T_act + t_n*u.T_act + t_a] = sum * scale;
}
`;

// --- causal_softmax ---
// Row-wise softmax over scores[n_heads, T_new, T_act], masking j > T_past + i.
// One workgroup per (row, head), three cooperative passes: max, exp+sum, scale.
// bindings: scores, u{T_new,T_act,T_past,n_heads}   dispatch: T_new, n_heads
//
// Structured like rmsnorm. `red` is used for both reductions, so there is a
// barrier after reading row_max and before the sum overwrites it. Without it
// one thread could overwrite red[0] while another is still reading the max.
const SOFTMAX_WG = 256;

KERNEL.causal_softmax = /* wgsl */ `
struct U { T_new: u32, T_act: u32, T_past: u32, n_heads: u32 }
@group(0) @binding(0) var<storage, read_write> scores : array<f32>;
@group(0) @binding(1) var<uniform>             u     : U;

var<workgroup> red: array<f32, ${SOFTMAX_WG}u>;

@compute @workgroup_size(${SOFTMAX_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i    = wg.x;   // row within T_new
    let j    = wg.y;   // head
    let live = i < u.T_new && j < u.n_heads;
    let base = j*u.T_new*u.T_act + i*u.T_act;

    // Inclusive: column k is visible to row i iff k <= T_past + i. k = 0
    // always qualifies, so a live row has at least one term and the sum below
    // is never zero.
    let limit = u.T_past + i;

    // --- pass 1: row max ---
    var m = -1e30;
    if (live) {
        for (var k = lid.x; k < u.T_act; k += ${SOFTMAX_WG}u) {
            if (k <= limit) { m = max(m, scores[base + k]); }
        }
    }
    red[lid.x] = m;
    workgroupBarrier();
    for (var s = ${SOFTMAX_WG}u / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
        workgroupBarrier();
    }
    let row_max = red[0];
    workgroupBarrier();          // every thread has read red[0], so it can be reused

    // --- pass 2: exp in place, and its sum ---
    var acc = 0.0;
    if (live) {
        for (var k = lid.x; k < u.T_act; k += ${SOFTMAX_WG}u) {
            if (k <= limit) {
                let e = exp(scores[base + k] - row_max);
                scores[base + k] = e;
                acc += e;
            }
        }
    }
    red[lid.x] = acc;
    workgroupBarrier();
    for (var s = ${SOFTMAX_WG}u / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) { red[lid.x] += red[lid.x + s]; }
        workgroupBarrier();
    }
    let total = red[0];

    // --- pass 3: normalize, and zero the masked columns ---
    if (live) {
        for (var k = lid.x; k < u.T_act; k += ${SOFTMAX_WG}u) {
            if (k > limit) { scores[base + k] = 0.0; }
            else           { scores[base + k] = scores[base + k] / total; }
        }
    }
}
`;

// --- attn_av_gqa ---
// out[n_heads, T_q, D] rows r0 .. r0+T_new-1 = scores @ V_cache. Same GQA
// mapping as qk_scores. V is the int8 cache (see store_kv).
//
// Each thread walks the whole time axis, so the outer loop is over pages and
// the inner loop over slots: one `pages` lookup per page, not per token. With
// a contiguous cache (`page` >= T_act) the outer loop runs once.
// bindings: scores, V_cache, out, u{T_new,T_act,page,D,n_heads,group,r0,T_q}, pages
// dispatch: ceil(T_new/8), ceil(D/8), n_heads
KERNEL.attn_av_gqa = /* wgsl */ `
struct U { T_new: u32, T_act: u32, page: u32, D: u32, n_heads: u32, group: u32, r0: u32, T_q: u32 }
@group(0) @binding(0) var<storage, read>       scores  : array<f32>;
@group(0) @binding(1) var<storage, read>       V_cache : array<u32>;
@group(0) @binding(2) var<storage, read_write> out     : array<f32>;
@group(0) @binding(3) var<uniform>             u       : U;
@group(0) @binding(4) var<storage, read>       pages   : array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t_n = gid.x;
    let d   = gid.y;
    let h   = gid.z;

    if (t_n >= u.T_new || d >= u.D || h >= u.n_heads) { return; }

    let QB = min(32u, u.D);
    let RW = u.D / 4u + u.D / QB;
    let h_kv = h / u.group;
    let n_kv = u.n_heads / u.group;
    let scores_base = h*u.T_new*u.T_act + t_n*u.T_act;
    // This thread's byte and scale within a row.
    let word  = d / 4u;
    let shift = (d % 4u) * 8u;
    let sc    = u.D / 4u + d / QB;

    var sum = 0.0;
    var i = 0u;
    loop {
        if (i >= u.T_act) { break; }
        let phys = pages[i / u.page];
        let slot = i % u.page;
        // Up to the end of this page or the end of the context, whichever is
        // first.
        let n = min(u.page - slot, u.T_act - i);
        let row = ((phys*n_kv + h_kv)*u.page + slot) * RW;
        for (var j = 0u; j < n; j++) {
            let r = row + j*RW;
            let q = f32(extractBits(bitcast<i32>(V_cache[r + word]), shift, 8u));
            sum += scores[scores_base + i + j] * (q * bitcast<f32>(V_cache[r + sc]));
        }
        i += n;
    }

    out[h*u.T_q*u.D + (u.r0 + t_n)*u.D + d] = sum;
}
`;

// --- merge_heads ---
// [n_heads, T, D] → [T, n_heads*D]. Inverse of reshape_heads.
// bindings: heads, out, u{T,n_heads,D}         dispatch: ceil(D/64), T, n_heads
KERNEL.merge_heads = /* wgsl */ `
struct U { T: u32, n_heads: u32, D: u32, _p: u32 }
@group(0) @binding(0) var<storage, read>       heads: array<f32>;
@group(0) @binding(1) var<storage, read_write> out  : array<f32>;
@group(0) @binding(2) var<uniform>             u    : U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let t = gid.y;
  let h = gid.z;
  if (d >= u.D) { return; }
  out[t * u.n_heads * u.D + h * u.D + d] = heads[h * u.T * u.D + t * u.D + d];
}
`;

// --- add_inplace ---
// a[i] += b[i]. Residual connections.
// bindings: a, b, u{N}                         dispatch: ceil(N/256)
KERNEL.add_inplace = /* wgsl */ `
struct U { N: u32 }
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<uniform>             u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.N) { return; }
  a[i] = a[i] + b[i];
}
`;

// --- round_f16 ---
// a[i] = a[i] rounded to the nearest f16 value, ties to even, kept as f32.
// Values past ±65504 clamp to it; NaN stays NaN. The same rounding as
// web/f16.js, bit for bit: it is run on the residual stream after every block
// (model.js block()), so rows handed to another peer are already halves and
// a reply is the same however the model is split. Integer ops only, so every
// GPU rounds alike.
// bindings: a, u{N}   dispatch: ceil(N/256)
KERNEL.round_f16 = /* wgsl */ `
struct U { N: u32 }
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<uniform>             u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.N) { return; }
  let b = bitcast<u32>(a[i]);
  let sign = b & 0x80000000u;
  let m = b & 0x7fffffffu;
  if (m > 0x7f800000u) { return; }                      // NaN
  if (m > 0x477fe000u) { a[i] = bitcast<f32>(sign | 0x477fe000u); return; }   // past 65504
  if (m < 0x38800000u) {
    // Below 2^-14 the halves are multiples of 2^-24. Scaling by a power of
    // two is exact, and round() is to nearest even.
    let q = round(bitcast<f32>(m) * 16777216.0) / 16777216.0;
    a[i] = bitcast<f32>(sign | bitcast<u32>(q));
    return;
  }
  // Keep 10 of the 23 mantissa bits. A carry into the exponent is right.
  let rest = m & 0x1fffu;
  var r = m & ~0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (r & 0x2000u) != 0u)) { r += 0x2000u; }
  a[i] = bitcast<f32>(sign | r);
}
`;

// --- copy_f32 ---
// dst[dstOff + i] = src[srcOff + i] for i < N. A copy recorded inside an open
// compute pass, where copyBufferToBuffer is not allowed (speculative-decoding
// snapshots, the MTP head's inputs). Two-dimensional, so N may exceed 65535
// workgroups' worth of 256.
// bindings: src, dst, u{N, rowLen, srcOff, dstOff}   dispatch: min(ceil(N/256), 4096), rows
KERNEL.copy_f32 = /* wgsl */ `
struct U { N: u32, rowLen: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * u.rowLen + gid.x;
  if (gid.x >= u.rowLen || i >= u.N) { return; }
  dst[u.dstOff + i] = src[u.srcOff + i];
}
`;

// --- argmax_rows ---
// ids[idOff + r] = f32(argmax of row r of x[T, N]), the lowest index on a tie
// (as sampling.js's greedy pick). One workgroup per row. Written as f32 so
// gather_embed_nopos can read it as the next step's token: the MTP draft chain
// runs on the GPU without a readback between steps.
// bindings: x, ids, u{N, idOff}                 dispatch: T
const ARGMAX_WG = 256;
KERNEL.argmax_rows = /* wgsl */ `
struct U { N: u32, idOff: u32 }
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> ids: array<f32>;
@group(0) @binding(2) var<uniform>             u: U;

var<workgroup> bv: array<f32, ${ARGMAX_WG}u>;
var<workgroup> bi: array<u32, ${ARGMAX_WG}u>;

@compute @workgroup_size(${ARGMAX_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wg.x * u.N;
  var best = -3.0e38;
  var at = 0xffffffffu;
  for (var i = lid.x; i < u.N; i += ${ARGMAX_WG}u) {
    let v = x[base + i];
    if (v > best) { best = v; at = i; }
  }
  bv[lid.x] = best;
  bi[lid.x] = at;
  workgroupBarrier();
  for (var s = ${ARGMAX_WG}u / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      let v = bv[lid.x + s];
      let j = bi[lid.x + s];
      if (v > bv[lid.x] || (v == bv[lid.x] && j < bi[lid.x])) { bv[lid.x] = v; bi[lid.x] = j; }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) { ids[u.idOff + wg.x] = f32(bi[0]); }
}
`;

// --- topk_stage ---
// One stage of the per-row top-M selection (topkRows). Each workgroup loads
// 1024 (value, index) pairs of one row, bitonic-sorts them best first (larger
// value, then lower index) and writes its top M. The first stage reads a
// logits row (index = position); later stages read the pairs the stage before
// wrote. Pairs are vec2<f32> with the index bitcast. Rows are on dispatch y,
// from source row `row0`.
// bindings: src, dst, u{n, M, srcStride, dstStride, row0}   dispatch: ceil(n/1024), rows
export const TOPK_CHUNK = 1024;
export const topkStageSrc = (first) => memoSrc(`topk_stage:${first}`, () => /* wgsl */ `
struct U { n: u32, M: u32, srcStride: u32, dstStride: u32, row0: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read>       src: array<${first ? 'f32' : 'vec2<f32>'}>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             u: U;

var<workgroup> sv: array<f32, 1024>;
var<workgroup> si: array<u32, 1024>;

fn better(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
  return av > bv || (av == bv && ai < bi);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.y;
  let base = wg.x * 1024u;
  for (var t = 0u; t < 4u; t++) {
    let k = lid.x + t * 256u;
    let i = base + k;
    if (i < u.n) {
${first ? `      sv[k] = src[(u.row0 + row) * u.srcStride + i];
      si[k] = i;` : `      let p = src[row * u.srcStride + i];
      sv[k] = p.x;
      si[k] = bitcast<u32>(p.y);`}
    } else {
      // -inf, below every logit. The u.n & 0u term keeps it out of
      // constant evaluation, which rejects an infinite f32.
      sv[k] = bitcast<f32>(0xff800000u | (u.n & 0u));
      si[k] = 0xffffffffu;
    }
  }
  workgroupBarrier();
  for (var size = 2u; size <= 1024u; size <<= 1u) {
    for (var stride = size >> 1u; stride > 0u; stride >>= 1u) {
      for (var t = 0u; t < 4u; t++) {
        let i = lid.x + t * 256u;
        let l = i ^ stride;
        if (l > i) {
          let up = (i & size) == 0u;
          if (up == better(sv[l], si[l], sv[i], si[i])) {
            let tv = sv[i]; sv[i] = sv[l]; sv[l] = tv;
            let ti = si[i]; si[i] = si[l]; si[l] = ti;
          }
        }
      }
      workgroupBarrier();
    }
  }
  for (var m = lid.x; m < u.M; m += 256u) {
    dst[row * u.dstStride + wg.x * u.M + m] = vec2<f32>(sv[m], bitcast<f32>(si[m]));
  }
}
`);

// --- silu_mul ---
// out[i] = silu(gate[i]) * up[i], silu(z) = z / (1 + exp(-z)).
// Two input buffers and a separate output.
// bindings: gate, up, out, u{N}                dispatch: ceil(N/256)
KERNEL.silu_mul = /* wgsl */ `
struct U { N: u32 }
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up  : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>             u   : U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;

    if (n >= u.N) { return; }

    let scale = 1.0 / (1.0 + exp(-gate[n]));
    out[n] = gate[n] * scale * up[n];
}
`;

// --- Gated DeltaNet (Qwen3.8-27B's linear_attention layers) ---
//
// All f32: none of these kernels reads a quantized weight. The matmul-shaped
// tensors (in_proj_*, out_proj) go through matmulNobias. The converter keeps
// the conv1d weight, gate_params (packed A_log/dt_bias) and gated_norm as f32.
// RMSNormGated is rmsnorm followed by silu_mul.

// --- causal_conv1d ---
// out[t,c] = silu(sum_{k=0}^{K-1} weight[c,k] * x[T_past+t-(K-1)+k, c]).
// Out-of-range reads (possible only near absolute position 0) count as 0.
// Depthwise (one filter per channel), with SiLU fused. `x` holds this job's
// rows, indexed from T_past.
//
// The `hist` build also reads a [K-1,C] ring of the rows just before `x`'s
// row 0 (see update_conv_history). A window reaching before this job reads
// `hist[tt - T_past + (K-1)]`. Without `hist`, T_past = 0 and the whole
// sequence is in `x`.
// bindings: x, weight, y, u{T,C,K,T_past}[, hist]   dispatch: ceil(C/64), T, 1
export const causalConv1dSrc = (hasHist = false) =>
    memoSrc(`causal_conv1d:${hasHist}`, () => /* wgsl */ `
struct U { T: u32, C: u32, K: u32, T_past: u32 }
@group(0) @binding(0) var<storage, read>       x      : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> y      : array<f32>;
@group(0) @binding(3) var<uniform>             u      : U;
${hasHist ? '@group(0) @binding(4) var<storage, read> hist : array<f32>;' : ''}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let t = gid.y;
    if (c >= u.C || t >= u.T) { return; }

    var acc = 0.0;
    for (var k = 0u; k < u.K; k = k + 1u) {
        let tt = i32(u.T_past + t) - i32(u.K - 1u) + i32(k);
        if (tt >= 0) {
            acc = acc + weight[c * u.K + k] * x[u32(tt) * u.C + c];
        }
${hasHist ? `        else {
            // tt < 0 and tt >= T_past - (K-1) by the loop's bounds, so this
            // index is in [0, K-2].
            let hi = tt - i32(u.T_past) + i32(u.K - 1u);
            acc = acc + weight[c * u.K + k] * hist[u32(hi) * u.C + c];
        }` : ''}
    }
    let s = 1.0 / (1.0 + exp(-acc));
    y[t * u.C + c] = acc * s;
}
`);
KERNEL.causal_conv1d = causalConv1dSrc(false);

// --- update_conv_history ---
// Advances causal_conv1d's [K-1,C] history ring in place past a chunk of
// `T_new` rows (the buffer causal_conv1d just read as `x`), so the next call's
// window includes this chunk's tail. One invocation per channel, writing only
// its own column, so no barrier is needed.
//
// new_hist[r] (r in [0,K-2]) is row (T_new-(K-1)+r) of [old_hist ++ newRows].
// Negative rows come from the old ring, the rest from this chunk. The rows are
// staged in a private array because the ring's read and write ranges overlap.
//
// The array length must be a compile-time constant. CONV_HIST_MAX bounds K-1
// (Qwen3.8-27B's K is 4).
// bindings: hist[(K-1)*C], newRows[T_new*C], u{T_new,C,Kh}
// dispatch: ceil(C/64), 1, 1
const CONV_HIST_MAX = 63;

KERNEL.update_conv_history = /* wgsl */ `
struct U { T_new: u32, C: u32, Kh: u32 }
@group(0) @binding(0) var<storage, read_write> hist    : array<f32>;
@group(0) @binding(1) var<storage, read>       newRows : array<f32>;
@group(0) @binding(2) var<uniform>             u       : U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= u.C) { return; }

    var tmp: array<f32, ${CONV_HIST_MAX}u>;
    for (var r = 0u; r < u.Kh; r = r + 1u) {
        let logical = i32(u.T_new) - i32(u.Kh) + i32(r);
        if (logical < 0) {
            // In the old ring at row i = T_new + r, from solving
            // i - Kh = logical.
            tmp[r] = hist[u32(i32(u.T_new) + i32(r)) * u.C + c];
        } else {
            tmp[r] = newRows[u32(logical) * u.C + c];
        }
    }
    for (var r = 0u; r < u.Kh; r = r + 1u) {
        hist[r * u.C + c] = tmp[r];
    }
}
`;

// --- l2norm_heads ---
// y[t, srcHead*group+g, :] = x[t,srcHead,:] * rsqrt(sum(x^2) + 1e-6), for g in
// 0..group. This is a sum, not a mean, so it differs from rmsnorm. `group`
// applies Gated DeltaNet's repeat_interleave (query/key have fewer heads than
// value) in the same pass. group=1 is a plain per-head L2 norm.
// bindings: x, y, u{T,srcHeads,D,group}        dispatch: T*srcHeads, 1, 1
const L2_WG = 256;

KERNEL.l2norm_heads = /* wgsl */ `
struct U { T: u32, srcHeads: u32, D: u32, group: u32 }
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read_write> y : array<f32>;
@group(0) @binding(2) var<uniform>             u : U;

var<workgroup> partial: array<f32, ${L2_WG}u>;

@compute @workgroup_size(${L2_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row  = wg.x;
    let rows = u.T * u.srcHeads;
    let live = row < rows;
    let t    = row / u.srcHeads;
    let sh   = row % u.srcHeads;
    let base = row * u.D;

    var ss = 0.0;
    if (live) {
        for (var d = lid.x; d < u.D; d += ${L2_WG}u) {
            let v = x[base + d];
            ss += v * v;
        }
    }
    partial[lid.x] = ss;
    workgroupBarrier();

    for (var s = ${L2_WG}u / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) { partial[lid.x] += partial[lid.x + s]; }
        workgroupBarrier();
    }

    let scale = 1.0 / sqrt(partial[0] + 1e-6);
    if (live) {
        let dstHeads = u.srcHeads * u.group;
        for (var g = 0u; g < u.group; g = g + 1u) {
            let dstBase = (t * dstHeads + sh * u.group + g) * u.D;
            for (var d = lid.x; d < u.D; d += ${L2_WG}u) {
                y[dstBase + d] = x[base + d] * scale;
            }
        }
    }
}
`;

// --- gated_delta_net_step ---
// Gated DeltaNet's recurrence, on a persistent state buffer S[heads,Dk,Dv]
// (zero for a new session). One dispatch advances the state through nT
// consecutive tokens [t0, t0+nT): decode passes nT=1, prefill the whole chunk.
//
// Per head and token the reference (torch_recurrent_gated_delta_rule) is
//     S      = g·S
//     kv[v]  = Σ_k S[k][v]·k[k]
//     delta  = (v_in[v] − kv[v])·beta
//     S[k][v] += k[k]·delta[v]
//     out[v] = qScale·Σ_k S_new[k][v]·q[k]
// Since S_new = g·S + k⊗delta, the readout is
//     out[v] = qScale·( Σ_k (g·S)[k][v]·q[k] + delta[v]·(k·q) ),
// so one pass over the old state per token gives everything: per column, two
// dot products over k (kv, qo), plus the per-head scalar k·q. The update is
// then written from values already in registers.
//
// Work split: one workgroup per (head, tile of TV value-columns), with KS
// k-groups × TV columns = 256 threads. Each thread owns one column and ROWS =
// ceil(Dk/KS) rows of it, held in a private array for the whole dispatch, so
// the state is read from VRAM once and written once per dispatch. Loads are
// coalesced: a k-group's TV threads read TV consecutive vv for the same kk.
// The per-column dot products are reduced across the KS k-groups through
// workgroup memory. That memory is double-buffered on token parity, so one
// barrier per token is enough: a writer of buffer b at token t+2 has passed
// the t+1 barrier, which no reader of buffer b from token t reaches before
// finishing its reads. Each workgroup owns a disjoint slice of S.
//
// Every thread reaches every barrier. Out-of-range lanes (vv ≥ Dv, kk ≥ Dk,
// which occur only when Dk/Dv are not multiples of the tile, e.g. the unit
// test's 5×7) compute on zeros and skip their writes.
//
// The gates (g, beta) are computed inline on every call from the raw
// projections and per-head params, as in the reference forward. A_log and
// dt_bias are packed into one params[2*heads] buffer (A_log at [0,heads),
// dt_bias at [heads,2*heads)), so the kernel needs 8 storage bindings, the
// per-stage minimum WebGPU guarantees. An adapter limited to 8 fails a
// pipeline with 9 without an error and corrupts the submission.
//
// Dk/Dv are constants in the source (they are fixed per checkpoint), so ROWS
// is a compile-time bound and the row array stays in registers. The tile
// shape is tunable: {kSplit 8, tileV 32} gives 192 workgroups × 16 registers
// at 48×128×128; {4, 64} would give 96 × 32.
// bindings: q,k[T*heads*Dk], v[T*heads*Dv], a_raw,b_raw[T*heads],
// params[2*heads] (A_log then dt_bias), S[heads*Dk*Dv] (persistent,
// read_write), out[T*heads*Dv], u{t0,nT,heads,_}
// dispatch: ceil(Dv/tileV), heads, 1
export const GDN_TILE = { kSplit: 8, tileV: 32 };

export const gdnStepSrc = (Dk, Dv, { kSplit = GDN_TILE.kSplit, tileV = GDN_TILE.tileV } = {}) =>
    memoSrc(`gdn_step:${Dk}:${Dv}:${kSplit}:${tileV}`, () => {
        const ROWS = Math.ceil(Dk / kSplit);
        const WG   = kSplit * tileV;
        return /* wgsl */ `
struct U { t0: u32, nT: u32, heads: u32, _pad: u32 }
@group(0) @binding(0)  var<storage, read>       q      : array<f32>;
@group(0) @binding(1)  var<storage, read>       k      : array<f32>;
@group(0) @binding(2)  var<storage, read>       v      : array<f32>;
@group(0) @binding(3)  var<storage, read>       a_raw  : array<f32>;
@group(0) @binding(4)  var<storage, read>       b_raw  : array<f32>;
@group(0) @binding(5)  var<storage, read>       params : array<f32>;
@group(0) @binding(6)  var<storage, read_write> S      : array<f32>;
@group(0) @binding(7)  var<storage, read_write> out    : array<f32>;
@group(0) @binding(8)  var<uniform>             u      : U;

const DK   = ${Dk}u;
const DV   = ${Dv}u;
const KS   = ${kSplit}u;
const TV   = ${tileV}u;
const ROWS = ${ROWS}u;

// Partial dot products per (k-group, column), in two copies indexed by token
// parity. pkq is per k-group only, since k·q is the same for every column.
var<workgroup> pkv: array<f32, ${2 * kSplit * tileV}u>;
var<workgroup> pqo: array<f32, ${2 * kSplit * tileV}u>;
var<workgroup> pkq: array<f32, ${2 * kSplit}u>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h    = wg.y;
    let col  = lid.x % TV;
    let kg   = lid.x / TV;
    let vv   = wg.x * TV + col;
    let live = vv < DV;
    let vvc  = min(vv, DV - 1u);           // clamped index for out-of-range lanes' reads
    let k0   = kg * ROWS;

    // torch_recurrent_gated_delta_rule scales Q by 1/sqrt(Dk) after L2-norm.
    // Q is used only in the readout, so the scale is applied there.
    let qScale = inverseSqrt(f32(DK));
    let sBase  = h * DK * DV;

    var s: array<f32, ROWS>;
    for (var i = 0u; i < ROWS; i = i + 1u) {
        let kk = k0 + i;
        // select() evaluates both arms, so out-of-range lanes read a clamped
        // in-range cell and discard it.
        s[i] = select(0.0, S[sBase + min(kk, DK - 1u) * DV + vvc], live && kk < DK);
    }

    for (var t = u.t0; t < u.t0 + u.nT; t = t + 1u) {
        let a = a_raw[t * u.heads + h] + params[u.heads + h];
        // softplus(a) = log(1+exp(a)). The a>20 branch avoids overflow in exp(a).
        let sp   = select(log(1.0 + exp(a)), a + log(1.0 + exp(-a)), a > 20.0);
        let g    = exp(-exp(params[h]) * sp);
        let beta = 1.0 / (1.0 + exp(-b_raw[t * u.heads + h]));

        let qBase = (t * u.heads + h) * DK;
        let vBase = (t * u.heads + h) * DV;
        let buf   = ((t - u.t0) & 1u) * KS;

        // Decay this thread's rows, then add its share of the three dot products.
        var kv = 0.0;
        var qo = 0.0;
        var kq = 0.0;
        for (var i = 0u; i < ROWS; i = i + 1u) {
            let kk = k0 + i;
            if (kk < DK) {
                let ki = k[qBase + kk];
                let qi = q[qBase + kk];
                s[i] = s[i] * g;
                kv = kv + s[i] * ki;
                qo = qo + s[i] * qi;
                kq = kq + ki * qi;
            }
        }
        pkv[(buf + kg) * TV + col] = kv;
        pqo[(buf + kg) * TV + col] = qo;
        if (col == 0u) { pkq[buf + kg] = kq; }
        workgroupBarrier();

        var kvT = 0.0;
        var qoT = 0.0;
        var kqT = 0.0;
        for (var j = 0u; j < KS; j = j + 1u) {
            kvT = kvT + pkv[(buf + j) * TV + col];
            qoT = qoT + pqo[(buf + j) * TV + col];
            kqT = kqT + pkq[buf + j];
        }
        let delta = (v[vBase + vvc] - kvT) * beta;

        for (var i = 0u; i < ROWS; i = i + 1u) {
            let kk = k0 + i;
            if (kk < DK) { s[i] = s[i] + k[qBase + kk] * delta; }
        }
        if (kg == 0u && live) {
            out[vBase + vv] = (qoT + delta * kqT) * qScale;
        }
    }

    for (var i = 0u; i < ROWS; i = i + 1u) {
        let kk = k0 + i;
        if (live && kk < DK) { S[sBase + kk * DV + vv] = s[i]; }
    }
}
`;
    });
KERNEL.gated_delta_net_step = gdnStepSrc(128, 128);

// --- sigmoid_mul ---
// out[i] = sigmoid(gate[i]) * a[i]: Qwen's attention output gate
// (`attn_output * torch.sigmoid(gate)`, before o_proj). The buffers are
// flattened.
// bindings: gate, a, out, u{N}                 dispatch: ceil(N/256)
KERNEL.sigmoid_mul = /* wgsl */ `
struct U { N: u32 }
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       a   : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>             u   : U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= u.N) { return; }
    let s = 1.0 / (1.0 + exp(-gate[n]));
    out[n] = s * a[n];
}
`;

// --- 4. Ops: one wrapper per kernel ---
//
// Every wrapper takes (device, pass, ...buffers, ...dims), records a dispatch
// on `pass`, and returns nothing. The caller allocates outputs with
// makeStorage().

// Y[M,N] = X[M,K] @ W[N,K]^T
//
// Picks the build from M: GEMV for one row, the batched GEMV for up to
// GEMV_ROWS_MAX rows, tiled otherwise. The GEMV builds give each row the same
// bits; the tiled build agrees with them to f32 rounding. kernels.test.js
// checks all three against the same CPU reference.
export function matmulNobias(device, pass, X, W, Y, M, K, N) {
    const gemv = M === 1;
    const rows = M > 1 && M <= GEMV_ROWS_MAX;
    const q8 = q8Of(W);
    // The profile name uses K→N, since the kernel does not know which
    // projection it is. The shape distinguishes gate/up, down, the attention
    // projections and the LM head.
    const name = pass.prof && `matmul ${q8 ? 'q8' : isF16(W) ? 'f16' : 'f32'}`
        + `${gemv ? '' : rows ? ` rows${M}` : ' tiled'} ${K}→${N}`;

    // Three weight formats and three shapes. Each combination is a separate
    // compiled source, not a branch in the shader.
    if (q8) {
        // Past 8 rows, the kernel that stages X in workgroup memory.
        const R = gemvRows(M);
        const many = rows && M > 8 && rowsQ8Fits(q8.group, R);
        const p  = getPipeline(device, gemv ? matmulGemvQ8Src(q8.group)
                                     : many ? matmulRowsQ8Src(q8.group, R)
                                     : rows ? matmulGemvRowsQ8Src(q8.group, R)
                                            : matmulTiledQ8Src(q8.group));
        const u  = uniform(device, u4(M, K, N));
        const bg = bind(device, p, [X, W, Y, u, q8.scales]);
        // The q8 GEMV covers one quantization group with one lane-group, so
        // its columns per workgroup depend on the group size.
        const cols = Math.max(1, Math.floor(256 / (q8.group / 4)));
        if (gemv) dispatch(pass, p, bg, Math.ceil(N / cols), M, 1, name);
        else if (rows) dispatch(pass, p, bg, Math.ceil(N / cols), 1, 1, name);
        else dispatch(pass, p, bg, Math.ceil(N / 16), Math.ceil(M / 16), 1, name);
        return;
    }

    // vec4 whenever K is a multiple of 4, which holds for every real checkpoint.
    const p  = getPipeline(device, gemv ? matmulGemvSrc(isF16(W), K % 4 === 0)
                                 : rows ? matmulGemvRowsSrc(isF16(W), K % 4 === 0, gemvRows(M))
                                        : matmulTiledSrc(isF16(W)));
    const u  = uniform(device, u4(M, K, N));
    const bg = bind(device, p, [X, W, Y, u]);
    if (gemv) dispatch(pass, p, bg, Math.ceil(N / GEMV_COLS), M, 1, name);
    else if (rows) dispatch(pass, p, bg, Math.ceil(N / GEMV_COLS), 1, 1, name);
    else dispatch(pass, p, bg, Math.ceil(N / 16), Math.ceil(M / 16), 1, name);
}

// out[T,H] = wte[ids[t]]
export function gatherEmbedNoPos(device, pass, ids, wte, out, T, H) {
    const q8 = q8Of(wte);
    const p  = getPipeline(device, q8 ? gatherEmbedNoPosQ8Src(q8.group)
                                      : gatherEmbedNoPosSrc(isF16(wte)));
    const u  = uniform(device, u4(T, H));
    const bg = bind(device, p, q8 ? [ids, wte, out, u, q8.scales] : [ids, wte, out, u]);
    dispatch(pass, p, bg, Math.ceil(H / 64), T, 1, 'gather_embed');
}

// y[T,H] = rmsnorm(x[T,H]) * weight[H]
export function rmsnorm(device, pass, x, weight, y, T, H, eps = DIMS.rms_eps) {
    const p  = getPipeline(device, rmsnormSrc(eps, isF16(weight)));
    const u  = uniform(device, u4(T, H));
    const bg = bind(device, p, [x, weight, y, u]);
    dispatch(pass, p, bg, T, 1, 1, pass.prof && `rmsnorm ${H}`);
}

// src[T, nHeads*D] → dst[nHeads, T, D]
export function reshapeHeads(device, pass, src, dst, T, nHeads, D) {
    const p  = getPipeline(device, KERNEL.reshape_heads);
    const u  = uniform(device, u4(T, nHeads, D));
    const bg = bind(device, p, [src, dst, u]);
    dispatch(pass, p, bg, Math.ceil(D / 64), T, nHeads, 'reshape_heads');
}

// The page table for a non-paged buffer: one page, physical page 0, so every
// address in the cache kernels reduces to the contiguous one. One shared
// buffer per device, so bind groups stay cacheable.
//
// `pages = null` means the buffer is one dense run of rows: Q, and every cache
// from makeCache().
const onePageCache = new WeakMap();
export function onePage(device) {
    let buf = onePageCache.get(device);
    if (!buf) onePageCache.set(device, buf = uploadU32(device, new Uint32Array([0])));
    return buf;
}

// u32 words in one KV row (one token, one KV head): D/4 of int8, then one
// f32 scale per min(32, D) values. See store_kv; split.js's kvRowBytes() is
// n_kv rows of this, in bytes.
export const kvRowWords = (D) => D / 4 + D / Math.min(32, D);

// src[T_new, nKV*D] → the int8 cache at time offset T_past. `rope` is
// { theta, scaling, rotDim } for K and null for V.
export function storeKV(device, pass, src, cache, T_new, T_past, pageTokens, nKV, D,
                        rope = null, pages = null) {
    const p   = getPipeline(device, KERNEL.store_kv);
    const rot = rope ? (rope.rotDim ?? D) : 0;
    const u   = uniform(device, u8(T_new, T_past, pageTokens, nKV, D, rot));
    // V binds the (smallest) table too, so the layout is one bind group shape.
    const tbl = ropeTable(device, rope ? rot : 2, rope?.theta ?? DIMS.rope_theta,
                          rope ? T_past + T_new : 1, rope?.scaling);
    const bg  = bind(device, p, [src, cache, u, pages ?? onePage(device), tbl]);
    const n   = T_new * nKV * (D / Math.min(32, D));
    dispatch(pass, p, bg, Math.ceil(n / 64), 1, 1, 'store_kv');
}

// Rows of an int8 cache read back as f32 [rows, D], for checks. `words` is
// the buffer's contents as a Uint32Array, `row0` the first row.
export function dequantKV(words, D, rows, row0 = 0) {
    const RW = kvRowWords(D), QB = Math.min(32, D);
    const f = new Float32Array(words.buffer, words.byteOffset, words.length);
    const b = new Int8Array(words.buffer, words.byteOffset, words.byteLength);
    const out = new Float32Array(rows * D);
    for (let r = 0; r < rows; r++) {
        const at = (row0 + r) * RW;
        for (let d = 0; d < D; d++) {
            out[r * D + d] = b[at * 4 + d] * f[at + D / 4 + Math.floor(d / QB)];
        }
    }
    return out;
}

// device → Map<"D:theta[:scaling]", { cap, buf }>. cos/sin for absolute
// positions 0..cap-1, laid out [pos][pair] with the two interleaved.
//
// Grown by doubling on demand, never shrunk. A replaced buffer is dropped, not
// destroy()ed, because a cached bind group may still reference it.
const ropeTableCache = new WeakMap();

// Llama 3.1's NTK-aware rescale of the plain frequency `f0 = θ^(-2i/D)`:
// unchanged at short wavelengths, divided by `factor` at long ones, and
// linearly blended in between. `scaling` is the manifest's `rope_scaling`
// entry, which has the same four fields as HF config.json's `rope_scaling`
// block. Reference: transformers'
// `modeling_rope_utils.py::_compute_llama3_parameters`.
function llama3Scale(f0, scaling) {
    const { factor, low_freq_factor, high_freq_factor,
            original_max_position_embeddings: oldLen } = scaling;
    const wavelen     = (2 * Math.PI) / f0;
    const lowWavelen  = oldLen / low_freq_factor;
    const highWavelen = oldLen / high_freq_factor;
    if (wavelen < highWavelen) return f0;
    if (wavelen > lowWavelen) return f0 / factor;
    const smooth = (oldLen / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
    return smooth * f0 + (1 - smooth) * (f0 / factor);
}

export function ropeTable(device, D, theta, needPos, scaling) {
    const cache = perDevice(ropeTableCache, device);
    const key   = scaling
        ? `${D}:${theta}:${scaling.factor}:${scaling.low_freq_factor}:` +
          `${scaling.high_freq_factor}:${scaling.original_max_position_embeddings}`
        : `${D}:${theta}`;
    const hit   = cache.get(key);
    if (hit && hit.cap >= needPos) return hit.buf;

    const cap  = Math.max(needPos, hit ? hit.cap * 2 : 256);
    const half = D / 2;
    const data = new Float32Array(cap * half * 2);

    // Computed in f64 and rounded to f32 once on store, like the CPU reference
    // in kernels.test.js.
    for (let i = 0; i < half; i++) {
        let invFreq = Math.pow(theta, -(2 * i) / D);
        if (scaling) invFreq = llama3Scale(invFreq, scaling);
        for (let pos = 0; pos < cap; pos++) {
            const ang = pos * invFreq;
            const at  = (pos * half + i) * 2;
            data[at]     = Math.cos(ang);
            data[at + 1] = Math.sin(ang);
        }
    }

    const buf = uploadF32(device, data);
    cache.set(key, { cap, buf });
    return buf;
}

// In-place RoPE on a dense f32 buffer: rope(dev, pass, Qh, nHeads, T, D, T_past, T, 0).
// rotDim < D is a partial-rotary checkpoint (Qwen3.8-27B: rotDim = D/4).
// Channels [rotDim, D) are left unchanged.
export function rope(device, pass, buf, nHeads, T, D, T_past, timeStride, row0 = 0,
                     theta = DIMS.rope_theta, scaling = DIMS.rope_scaling, rotDim = D,
                     pages = null) {
    const p   = getPipeline(device, KERNEL.rope);
    const u   = uniform(device, u8(nHeads, T, rotDim, D, T_past, timeStride, row0));
    const tbl = ropeTable(device, rotDim, theta, T_past + T, scaling);
    const bg  = bind(device, p, [buf, u, tbl, pages ?? onePage(device)]);
    dispatch(pass, p, bg, Math.ceil((rotDim / 2) / 64), T, nHeads, 'rope');
}

// scores[nHeads, T_new, T_act] = Q @ K_cache^T / sqrt(D), for Q's rows
// r0 .. r0+T_new-1 of T_q (T_q = T_new, r0 = 0: every row).
export function qkScoresGQA(device, pass, Q, Kcache, scores,
                            T_new, T_act, pageTokens, D, nHeads, group, pages = null,
                            r0 = 0, T_q = T_new) {
    const p  = getPipeline(device, KERNEL.qk_scores_gqa);
    const u  = uniform(device, u8(T_new, T_act, pageTokens, D, nHeads, group, r0, T_q));
    const bg = bind(device, p, [Q, Kcache, scores, u, pages ?? onePage(device)]);
    dispatch(pass, p, bg, Math.ceil(T_new / 8), Math.ceil(T_act / 8), nHeads, 'qk_scores');
}

// In-place causal softmax over scores[nHeads, T_new, T_act]
export function causalSoftmax(device, pass, scores, T_new, T_act, T_past, nHeads) {
    const p  = getPipeline(device, KERNEL.causal_softmax);
    const u  = uniform(device, u4(T_new, T_act, T_past, nHeads));
    const bg = bind(device, p, [scores, u]);
    dispatch(pass, p, bg, T_new, nHeads, 1, 'causal_softmax');
}

// out[nHeads, T_q, D] rows r0 .. r0+T_new-1 = scores @ V_cache
export function attnAVGQA(device, pass, scores, Vcache, out,
                          T_new, T_act, pageTokens, D, nHeads, group, pages = null,
                          r0 = 0, T_q = T_new) {
    const p  = getPipeline(device, KERNEL.attn_av_gqa);
    const u  = uniform(device, u8(T_new, T_act, pageTokens, D, nHeads, group, r0, T_q));
    const bg = bind(device, p, [scores, Vcache, out, u, pages ?? onePage(device)]);
    dispatch(pass, p, bg, Math.ceil(T_new / 8), Math.ceil(D / 8), nHeads, 'attn_av');
}

// heads[nHeads, T, D] → out[T, nHeads*D]
export function mergeHeads(device, pass, heads, out, T, nHeads, D) {
    const p  = getPipeline(device, KERNEL.merge_heads);
    const u  = uniform(device, u4(T, nHeads, D));
    const bg = bind(device, p, [heads, out, u]);
    dispatch(pass, p, bg, Math.ceil(D / 64), T, nHeads, 'merge_heads');
}

// a[i] += b[i]
export function addInplace(device, pass, a, b, N) {
    const p  = getPipeline(device, KERNEL.add_inplace);
    const u  = uniform(device, u4(N));
    const bg = bind(device, p, [a, b, u]);
    dispatch(pass, p, bg, Math.ceil(N / 256), 1, 1, 'add_inplace');
}

// a[0, N) rounded in place to f16 values (see round_f16).
export function roundF16(device, pass, a, N) {
    const p  = getPipeline(device, KERNEL.round_f16);
    const u  = uniform(device, u4(N));
    const bg = bind(device, p, [a, u]);
    dispatch(pass, p, bg, Math.ceil(N / 256), 1, 1, 'round_f16');
}

// dst[dstOff, dstOff + N) = src[srcOff, srcOff + N), inside a compute pass.
// Offsets and N are in floats.
export function copyF32(device, pass, src, dst, N, srcOff = 0, dstOff = 0) {
    const p  = getPipeline(device, KERNEL.copy_f32);
    const wgX = Math.min(Math.ceil(N / 256), 4096);
    const rowLen = wgX * 256;
    const u  = uniform(device, u4(N, rowLen, srcOff, dstOff));
    const bg = bind(device, p, [src, dst, u]);
    dispatch(pass, p, bg, wgX, Math.ceil(N / rowLen), 1, 'copy_f32');
}

// The top M (value, index) pairs of each of T rows of `logits` [., V] from row
// `row0`, best first, ties to the lower index. Runs topk_stage until one workgroup's
// output is left, ping-ponging between `a` and `b`, each at least
// T * ceil(V / 1024) * M pairs. Returns the buffer holding the result, M
// pairs per row.
export function topkRows(device, pass, logits, T, V, M, a, b, row0 = 0) {
    // Each stage keeps M of every 1024, so M must be well below 1024 for the
    // stages to converge.
    if (!(M >= 1 && M <= TOPK_CHUNK / 4)) throw new Error(`topkRows: M ${M} is outside 1–${TOPK_CHUNK / 4}`);
    let src = logits, n = V, first = true, out = a, other = b;
    for (;;) {
        const groups = Math.ceil(n / TOPK_CHUNK);
        const p  = getPipeline(device, topkStageSrc(first));
        const u  = uniform(device, new Uint32Array([n, M, first ? V : n, groups * M, first ? row0 : 0, 0, 0, 0]));
        const bg = bind(device, p, [src, out, u]);
        dispatch(pass, p, bg, groups, T, 1, 'topk_stage');
        if (groups === 1) return out;
        src = out; n = groups * M; first = false;
        [out, other] = [other, out];
    }
}

// Pairs topkRows() needs in each of its two buffers.
export const topkPairs = (T, V, M) => T * Math.ceil(V / TOPK_CHUNK) * M;

// ids[idOff + r] = argmax of row r of x[T, N], as an f32 token id.
export function argmaxRows(device, pass, x, ids, T, N, idOff = 0) {
    const p  = getPipeline(device, KERNEL.argmax_rows);
    const u  = uniform(device, u4(N, idOff));
    const bg = bind(device, p, [x, ids, u]);
    dispatch(pass, p, bg, T, 1, 1, 'argmax_rows');
}

// out[i] = silu(gate[i]) * up[i]
export function siluMul(device, pass, gate, up, out, N) {
    const p  = getPipeline(device, KERNEL.silu_mul);
    const u  = uniform(device, u4(N));
    const bg = bind(device, p, [gate, up, out, u]);
    dispatch(pass, p, bg, Math.ceil(N / 256), 1, 1, 'silu_mul');
}

// out[t,c] = silu of a depthwise causal conv of x over the last K positions,
// ending at absolute position T_past+t. `hist`, when given, is a [K-1,C] ring
// holding the K-1 rows just before x's row 0.
export function causalConv1d(device, pass, x, weight, y, T, C, K, T_past = 0, hist = null) {
    const p  = getPipeline(device, causalConv1dSrc(!!hist));
    const u  = uniform(device, u4(T, C, K, T_past));
    const bg = bind(device, p, hist ? [x, weight, y, u, hist] : [x, weight, y, u]);
    dispatch(pass, p, bg, Math.ceil(C / 64), T, 1, 'causal_conv1d');
}

// Advance causal_conv1d's [K-1,C] history ring in place past `newRows` (the
// buffer just passed to causalConv1d as `x`).
export function updateConvHistory(device, pass, hist, newRows, T_new, C, K) {
    const p  = getPipeline(device, KERNEL.update_conv_history);
    const u  = uniform(device, u4(T_new, C, K - 1));
    const bg = bind(device, p, [hist, newRows, u]);
    dispatch(pass, p, bg, Math.ceil(C / 64), 1, 1, 'update_conv_history');
}

// y[t, srcHead*group+g, :] = L2-normalize(x[t,srcHead,:]), duplicated `group` times
export function l2NormalizeHeads(device, pass, x, y, T, srcHeads, D, group = 1) {
    const p  = getPipeline(device, KERNEL.l2norm_heads);
    const u  = uniform(device, u4(T, srcHeads, D, group));
    const bg = bind(device, p, [x, y, u]);
    dispatch(pass, p, bg, T * srcHeads, 1, 1, 'l2norm_heads');
}

// Advance the Gated DeltaNet recurrence through tokens [t0, t0+nT) on
// persistent state S, writing out[t] for each. Decode calls this with nT=1.
// Prefill passes its whole chunk, so the state stays in registers throughout.
// `params` is A_log and dt_bias concatenated ([A_log(heads), dt_bias(heads)]).
export function gatedDeltaNetStep(device, pass, q, k, v, aRaw, bRaw, params,
                                   S, out, t0, heads, Dk, Dv, nT = 1) {
    const p  = getPipeline(device, gdnStepSrc(Dk, Dv));
    const u  = uniform(device, u4(t0, nT, heads, 0));
    const bg = bind(device, p, [q, k, v, aRaw, bRaw, params, S, out, u]);
    dispatch(pass, p, bg, Math.ceil(Dv / GDN_TILE.tileV), heads, 1, 'gdn_step');
}

// out[i] = sigmoid(gate[i]) * a[i]
export function sigmoidMul(device, pass, gate, a, out, N) {
    const p  = getPipeline(device, KERNEL.sigmoid_mul);
    const u  = uniform(device, u4(N));
    const bg = bind(device, p, [gate, a, out, u]);
    dispatch(pass, p, bg, Math.ceil(N / 256), 1, 1, 'sigmoid_mul');
}
