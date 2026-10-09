// Fetches a converted checkpoint's manifest.json and .bin files and uploads
// each tensor to a GPUBuffer. Returns the weights object model.js takes:
//
//   { wte, norm, lm_head?, layers: [ { kind, input_layernorm, q_proj, … }, ... ] }
//
// Each layer has its `kind` ('full_attention' or 'linear_attention') and that
// kind's tensors (fullAttnBlockShapes / linearAttnBlockShapes below). The
// manifest uses the same tensor names (tools/convert_hf.py), so a missing name
// means the converter and loader disagree, and it throws.
//
// The manifest holds the model's dimensions, and every expected tensor shape
// is derived from them. `DIMS` is a fallback for callers that pass no
// manifest.
//
// Files are fetched one at a time, so onProgress can drive a UI and peak host
// memory is one file plus the embedding table.

import { uploadF32, uploadF16, uploadQ8, isQ8, q8Of, DIMS } from './kernels.js';
import { DTYPE_BYTES, storedBytes } from '../split.js';

// Weight formats the kernels can read. f16 and q8 reduce the download and the
// VRAM a range costs. The shaders convert to f32 on read, so all accumulation
// is f32.
const DTYPES = {
    f32: { bytes: DTYPE_BYTES.f32, array: Float32Array, upload: uploadF32 },
    f16: { bytes: DTYPE_BYTES.f16, array: Uint16Array,  upload: uploadF16 },
    // q8 describes a file, not every tensor in it: a q8 build stores its 2-D
    // projections as int8 plus per-group scales, and its 1-D norm weights as
    // f32. Each entry has its own dtype. `bytes` is the effective cost
    // including the scales (one f32 per 64 values); split.js sizes peer claims
    // from it.
    q8:  { bytes: DTYPE_BYTES.q8, array: Uint8Array, upload: null },
};

// The checkpoint this deploy runs, from /checkpoint.json: `{ "url": … }`. The
// URL is a same-origin `/weights/<dir>` (Caddy serves the repo root) or an
// absolute `https://` prefix. The signaling server reads the same file and
// sends the URL to every peer in its welcome. chat.html and dev.html fetch the
// file directly.
//
// peer.html has no ?weights override. A peer's output feeds every peer
// downstream, so a URL that changed its weights could corrupt other people's
// tokens.
export const CHECKPOINT_URL = '/checkpoint.json';

// Memoized as a promise, so concurrent callers share one fetch.
let active = null;

export function activeWeights() {
    active ??= (async () => {
        let r;
        try {
            r = await fetch(CHECKPOINT_URL);
        } catch (e) {
            active = null;    // may be transient: let the next caller retry
            throw new Error(`${CHECKPOINT_URL} could not be fetched (${e.message}); ` +
                `it names the checkpoint this deploy runs`);
        }
        if (!r.ok) {
            throw new Error(`${CHECKPOINT_URL}: ${r.status}. This file names the ` +
                `checkpoint every page loads. It is at the repo root, next to ` +
                `signal-server.js, and is committed (weights/ is not).`);
        }
        const cfg = await r.json();
        if (typeof cfg.url !== 'string' || !cfg.url) {
            throw new Error(`${CHECKPOINT_URL} has no "url". Expected ` +
                `{ "url": "/weights/<dir>" }`);
        }
        const url = cfg.url.replace(/\/$/, '');
        // A bare host is a valid relative URL and would be fetched from this
        // page's origin, so it is rejected.
        if (!url.startsWith('/') && !/^https?:\/\//.test(url)) {
            throw new Error(`${CHECKPOINT_URL}: "${url}" has no scheme and is not a ` +
                `rooted path, so it would be resolved against this page rather than ` +
                `fetched from that host. Use "/weights/<dir>" or "https://<host>/<prefix>".`);
        }
        return url;
    })();
    return active;
}

// Retries a thrown fetch or a 5xx with backoff. A 4xx means the request is
// wrong and is not retried.
export async function fetchRetry(url, { retries = 3, delayMs = 500 } = {}) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const r = await fetch(url);
            if (r.ok || r.status < 500) return r;
            lastErr = new Error(`${url}: ${r.status}`);
        } catch (err) {
            lastErr = err;
        }
        if (attempt < retries) await new Promise((res) => setTimeout(res, delayMs * 2 ** attempt));
    }
    throw lastErr;
}

export async function loadManifest(baseUrl) {
    baseUrl ??= await activeWeights();
    const r = await fetchRetry(`${baseUrl}/manifest.json`);
    if (!r.ok) {
        throw new Error(
            `${baseUrl}/manifest.json: ${r.status}. Run:  python tools/convert_hf.py`);
    }
    return r.json();
}

// The manifest is the source of truth for a loaded model, not the DIMS
// constant in kernels.js. This maps its field names to the ones model.js and
// kernels.js use. Callers pass the result on and do not import DIMS.
export function dimsFromManifest(manifest) {
    const need = ['hidden', 'n_layer', 'n_head', 'n_kv_head', 'head_dim', 'ffn', 'vocab'];
    const missing = need.filter((k) => typeof manifest[k] !== 'number');
    if (missing.length) throw new Error(`manifest is missing ${missing.join(', ')}`);

    // `layer_types` gives each block's kind. `hybrid` means more than one
    // distinct kind, as in tools/convert_hf.py's check_supported(). A
    // checkpoint without `layer_types` has neither.
    const layer_types = manifest.layer_types ?? undefined;
    const hybrid = !!layer_types && new Set(layer_types).size > 1;

    return {
        hidden:     manifest.hidden,
        n_layer:    manifest.n_layer,
        n_heads:    manifest.n_head,
        n_kv_heads: manifest.n_kv_head,
        head_dim:   manifest.head_dim,
        ffn:        manifest.ffn,
        vocab:      manifest.vocab,
        n_pos:      manifest.n_pos ?? 2048,
        group:      manifest.group ?? manifest.n_head / manifest.n_kv_head,
        rms_eps:    manifest.rms_eps ?? 1e-5,
        rope_theta: manifest.rope_theta ?? 10000.0,
        // Set only for Llama 3.1's NTK-aware variant, which rescales the
        // frequency table (kernels.js's ropeTable()).
        rope_scaling: manifest.rope_scaling ?? undefined,
        tied_lm_head: manifest.tied_lm_head !== false,
        model:      manifest.model ?? 'unknown',
        // Stored bytes per weight, for split.js's blockBytes().
        dtype:      manifest.dtype ?? 'f32',
        bytes:      DTYPES[manifest.dtype ?? 'f32']?.bytes ?? 4,
        layer_types,
        hybrid,
        // Actual file sizes, for split.js's blockWeightBytes() and endBytes().
        // Undefined when the manifest lists no files.
        stored:     storedBytes(manifest),
        // Partial-rotary width (Qwen3.8-27B: head_dim/4). Defaults to head_dim.
        rot_dim:    manifest.rot_dim ?? manifest.head_dim,
        // Gated DeltaNet's head and width config, separate from
        // full-attention's n_heads/head_dim above. `group` here is the
        // value-head repeat factor (nV/nK), not full-attention's GQA `group`.
        linear: hybrid ? {
            nK:  manifest.linear_num_key_heads,
            nV:  manifest.linear_num_value_heads,
            Dk:  manifest.linear_key_head_dim,
            Dv:  manifest.linear_value_head_dim,
            K4:  manifest.linear_conv_kernel_dim,
            keyDim:   manifest.linear_num_key_heads * manifest.linear_key_head_dim,
            valueDim: manifest.linear_num_value_heads * manifest.linear_value_head_dim,
            group:    manifest.linear_num_value_heads / manifest.linear_num_key_heads,
        } : undefined,
    };
}

// Lists what these kernels cannot run, checked before anything is fetched. The
// converter rejects the same cases, so a problem here means a hand-edited
// manifest or one from a newer converter.
export function checkManifest(manifest, d = dimsFromManifest(manifest)) {
    const problems = [];

    if (manifest.weight_layout !== 'out_in')
        problems.push(`weight_layout ${manifest.weight_layout}, expected out_in ` +
            `(matmul_nobias computes X @ W^T and consumes [out, in] directly)`);
    if (manifest.norm && manifest.norm !== 'rmsnorm')
        problems.push(`norm ${manifest.norm}, but only rmsnorm is implemented`);
    if (manifest.act && manifest.act !== 'silu')
        problems.push(`act ${manifest.act}, but only silu/SwiGLU is implemented`);
    if (manifest.dtype && !(manifest.dtype in DTYPES))
        problems.push(`dtype ${manifest.dtype}, but only ${Object.keys(DTYPES).join('/')} `
            + `are implemented`);

    // Non-hybrid checkpoints only: a hybrid full-attention layer's q width
    // differs from hidden (Qwen3.8-27B: 6144 vs 5120).
    if (!d.hybrid && d.n_heads * d.head_dim !== d.hidden)
        problems.push(`n_head*head_dim (${d.n_heads * d.head_dim}) != hidden (${d.hidden})`);
    if (d.hybrid && d.linear.nV % d.linear.nK !== 0)
        problems.push(`linear_num_value_heads (${d.linear.nV}) not divisible by `
            + `linear_num_key_heads (${d.linear.nK}): l2NormalizeHeads' integer `
            + `group factor needs this`);
    if (d.n_heads % d.n_kv_heads !== 0)
        problems.push(`n_head (${d.n_heads}) not divisible by n_kv_head (${d.n_kv_heads})`);
    if (d.head_dim % 2 !== 0)
        problems.push(`head_dim (${d.head_dim}) must be even for RoPE's split-half pairing`);
    if (manifest.rope_scaling) {
        const need = ['factor', 'low_freq_factor', 'high_freq_factor',
            'original_max_position_embeddings'];
        const missing = need.filter((k) => typeof manifest.rope_scaling[k] !== 'number');
        if (missing.length)
            problems.push(`rope_scaling is missing ${missing.join(', ')}`);
    }

    return problems;
}

// Fetching a whole file, with retries per chunk.
//
// A large download usually fails in the streamed body, not the request, so the
// body read is inside the retry. After a failed whole-file GET the file is
// fetched in 8 MiB ranges, so a failure costs one chunk.
const CHUNK = 8 * 1024 * 1024;

// Fetch one range, request and body, retried as a unit.
async function fetchRange(url, from, to, retries = 4) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        // Set inside the try and checked after it, so the catch below does not
        // retry a 4xx.
        let permanent = false;
        try {
            const r = await fetch(url, { headers: { Range: `bytes=${from}-${to}` } });
            if (r.ok) {
                return { status: r.status, buf: await r.arrayBuffer(),
                         range: r.headers.get('content-range') };
            }
            lastErr = new Error(`${url}: ${r.status}`);
            // A 4xx (missing file, bad range, expired auth) will not change on
            // retry.
            permanent = r.status < 500;
        } catch (err) {
            lastErr = err;
        }
        if (permanent) throw lastErr;
        // A longer backoff than fetchRetry's 500ms base: these failures are
        // usually a saturated uplink.
        if (attempt < retries) {
            await new Promise((res) => setTimeout(res, 750 * 2 ** attempt));
        }
    }
    throw lastErr;
}

// One plain GET with no Range header, body included. Tried first because the
// browser cache and Cloudflare's edge store a 200, and do not cache 206
// responses the same way.
async function fetchWhole(url) {
    const r = await fetch(url);
    if (r.ok) return r.arrayBuffer();
    const err = new Error(`${url}: ${r.status}`);
    err.permanent = r.status < 500;
    throw err;
}

export async function loadFile(baseUrl, filename, onBytes) {
    const url = `${baseUrl}/${filename}`;

    // Whole file first, so a cached copy is used and a new one can be cached.
    // Ranges are used only after a failure.
    try {
        return await fetchWhole(url);
    } catch (err) {
        if (err.permanent) throw err;
        // Continue with the ranged path, which retries and resumes.
    }

    // The first range is also the probe. A server that ignores Range answers
    // 200 with the whole body, which is returned as-is.
    const first = await fetchRange(url, 0, CHUNK - 1);
    if (first.status !== 206) return first.buf;

    const total = Number(first.range?.match(/\/(\d+)$/)?.[1]);
    if (!Number.isFinite(total) || total <= 0) {
        throw new Error(`${filename}: 206 without a usable Content-Range `
            + `(${first.range ?? 'absent'})`);
    }
    if (total <= first.buf.byteLength) return first.buf;

    const out = new Uint8Array(total);
    out.set(new Uint8Array(first.buf), 0);
    let done = first.buf.byteLength;
    onBytes?.(done, total);

    while (done < total) {
        const to = Math.min(done + CHUNK, total) - 1;
        const { buf } = await fetchRange(url, done, to);
        if (!buf.byteLength) throw new Error(`${filename}: empty range at ${done}`);
        out.set(new Uint8Array(buf), done);
        done += buf.byteLength;
        onBytes?.(done, total);
    }
    return out.buffer;
}

// One .bin → { name: GPUBuffer }, checking each tensor's shape against
// `expected`, which maps name → shape.
//
// f16 and q8 bytes are uploaded unchanged, and the shaders reinterpret them.
//
// Wrapped in error scopes, because a failed createBuffer/writeBuffer reports
// nothing: WebGPU returns an invalid buffer, every later use is a no-op, and
// the peer would compute with missing weights. The error is thrown, so
// engine.js's #fill() can stop and give back the rest of the claim.
//
// The result has `bytes`, the GPU memory the upload uses (including a q8
// tensor's scales). It is non-enumerable, so iterating the object yields only
// name → GPUBuffer. `strict: false` uploads only the `expected` tensors from a
// file that holds more.
async function uploadTensors(device, blob, entries, filename, expected, dtype = 'f32',
                             { strict = true } = {}) {
    const byName = new Map(entries.map((e) => [e.name, e]));
    const out = {};
    let total = 0;
    // Two scopes: a failed createBuffer is 'out-of-memory', and the
    // writeBuffer that follows on the invalid buffer is a 'validation' error.
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let caught;
    try {
        for (const [name, shape] of Object.entries(expected)) {
            const e = byName.get(name);
            if (!e) throw new Error(`${filename}: no tensor named ${name}`);
            if (e.shape.length !== shape.length || e.shape.some((v, i) => v !== shape[i]))
                throw new Error(`${filename}:${name} shape [${e.shape}], expected [${shape}]`);

            // The entry's dtype overrides the file's: a q8 build mixes
            // quantized projections with f32 norm weights.
            const per = e.dtype ?? dtype;

            if (per === 'q8') {
                if (e.scales_offset === undefined || !e.group)
                    throw new Error(`${filename}:${name} is q8 but carries no scales/group`);
                // Both views need 4-byte alignment. int8 blocks are N*K bytes
                // with K divisible by 4, so every offset in the file is
                // aligned.
                if (e.offset % 4 || e.scales_offset % 4)
                    throw new Error(`${filename}:${name} q8 offsets ${e.offset}/`
                        + `${e.scales_offset} are not 4-byte aligned`);
                const ngroup = Math.ceil(shape[1] / e.group);
                if (e.scales_size !== shape[0] * ngroup * 4)
                    throw new Error(`${filename}:${name} scales are ${e.scales_size} bytes, `
                        + `expected ${shape[0] * ngroup * 4} for [${shape}] at group ${e.group}`);
                out[name] = uploadQ8(device,
                    new Uint8Array(blob, e.offset, e.size),
                    new Float32Array(blob, e.scales_offset, e.scales_size / 4),
                    e.group);
                total += e.size + e.scales_size;
                continue;
            }

            const { bytes, array: Arr, upload } = DTYPES[per];
            // A .bin is concatenated tensors, and every offset is aligned to
            // its tensor's element size.
            if (e.offset % bytes || e.size % bytes)
                throw new Error(`${filename}:${name} offset ${e.offset}/size ${e.size} `
                    + `not aligned to ${per}`);
            out[name] = upload(device, new Arr(blob, e.offset, e.size / bytes));
            total += e.size;
        }
        for (const e of entries) {
            if (strict && !(e.name in expected)) throw new Error(`${filename}: unexpected tensor ${e.name}`);
        }
    } catch (err) {
        caught = err;
    }
    // Popped in the reverse of the push order.
    const validation = await device.popErrorScope();
    const oom = await device.popErrorScope();
    // On any failure, destroy what was uploaded. The caller never receives
    // these buffers, and the GPU may be out of memory.
    if (caught || oom || validation) destroyTensors(...Object.values(out));
    if (caught) throw caught;
    if (oom || validation) {
        throw new Error(`${filename}: GPU ran out of memory uploading its tensors `
            + `(${(oom ?? validation).message || 'out of memory'}). This GPU cannot hold `
            + `as many blocks as were claimed for it.`);
    }
    return Object.defineProperty(out, 'bytes', { value: total, writable: true });
}

// Destroy tensors made by uploadTensors(), including q8 scales. Dropping the
// last reference is not enough: a GPUBuffer keeps its memory until garbage
// collection, which may never run for a 1 GiB table.
export function destroyTensors(...buffers) {
    for (const buf of buffers) {
        if (!buf || typeof buf.destroy !== 'function') continue;
        if (isQ8(buf)) q8Of(buf).scales.destroy();
        buf.destroy();
    }
}

// The tensors of a full_attention block, derived from the model's dimensions.
// `qDim` (n_heads*head_dim) equals `H` except on a hybrid checkpoint's
// full-attention layers (Qwen3.8-27B: 6144 vs 5120). Those layers also have
// attn_gate (split from a double-width q_proj at conversion) and
// q_norm/k_norm.
function fullAttnBlockShapes(d) {
    const H = d.hidden, kvDim = d.n_kv_heads * d.head_dim, qDim = d.n_heads * d.head_dim;
    const shapes = {
        input_layernorm:          [H],
        q_proj:                   [qDim, H],
        k_proj:                   [kvDim, H],
        v_proj:                   [kvDim, H],
        o_proj:                   [H, qDim],
        post_attention_layernorm: [H],
        gate_proj:                [d.ffn, H],
        up_proj:                  [d.ffn, H],
        down_proj:                [H, d.ffn],
    };
    if (d.hybrid) {
        shapes.attn_gate = [qDim, H];
        shapes.q_norm = [d.head_dim];
        shapes.k_norm = [d.head_dim];
    }
    return shapes;
}

// The tensors of a linear_attention (Gated DeltaNet) block. Called only for a
// hybrid checkpoint's linear layers, so `d.linear` is set. Names and shapes
// match tools/convert_hf.py's linear_attn_layer_tensors().
function linearAttnBlockShapes(d) {
    const H = d.hidden;
    const { keyDim, valueDim, nV, Dv, K4 } = d.linear;
    return {
        input_layernorm:          [H],
        post_attention_layernorm: [H],
        in_proj_q: [keyDim, H],
        in_proj_k: [keyDim, H],
        in_proj_v: [valueDim, H],
        in_proj_z: [valueDim, H],
        in_proj_b: [nV, H],
        in_proj_a: [nV, H],
        conv1d_q:  [keyDim, K4],
        conv1d_k:  [keyDim, K4],
        conv1d_v:  [valueDim, K4],
        gate_params: [2 * nV],
        gated_norm:  [Dv],
        out_proj:    [H, valueDim],
        gate_proj:   [d.ffn, H],
        up_proj:     [d.ffn, H],
        down_proj:   [H, d.ffn],
    };
}

function blockShapes(d, kind = 'full_attention') {
    return kind === 'linear_attention' ? linearAttnBlockShapes(d) : fullAttnBlockShapes(d);
}

// f16 weights need a device created with `shader-f16`. Checked before the
// download, so it does not fail later at shader compile. q8 needs no optional
// feature.
function dtypeOf(device, manifest, baseUrl) {
    const dtype = manifest?.dtype ?? 'f32';
    if (dtype === 'f16' && !device.features?.has('shader-f16')) {
        throw new Error(
            `${manifest?.model ?? baseUrl} ships f16 weights but this GPU's WebGPU `
            + `device has no shader-f16 support, so the kernels cannot read them.`);
    }
    return dtype;
}

// Load and upload one block, the unit a peer downloads and starts serving.
// `index` is the block's index in the model, not in the caller's range.
export async function loadBlock(device, { baseUrl, manifest, d, index,
                                          onBytes } = {}) {
    baseUrl ??= await activeWeights();
    const fn = `block_${String(index).padStart(2, '0')}.bin`;
    const blob = await loadFile(baseUrl, fn, onBytes);
    // The kind comes from the manifest's layer_types ('full_attention' when
    // absent). It is attached to the weights, and model.js's block()
    // dispatches on it.
    const kind = d.layer_types?.[index] ?? 'full_attention';
    const uploaded = await uploadTensors(device, blob, manifest.files[fn], fn, blockShapes(d, kind),
        dtypeOf(device, manifest, baseUrl));
    return Object.assign(uploaded, { kind });
}

// client.bin: `wte` and the final norm. An untied checkpoint's LM head is in a
// separate lm_head.bin, because wte and lm_head together can exceed the ~2 GiB
// several browsers allow for one fetched body. A tied checkpoint has no
// lm_head.bin, and forward() uses wte.
export async function loadClient(device, { baseUrl, manifest, d,
                                           onBytes } = {}) {
    baseUrl ??= await activeWeights();
    const dtype = dtypeOf(device, manifest, baseUrl);

    const blob = await loadFile(baseUrl, 'client.bin', onBytes);
    const out = await uploadTensors(device, blob, manifest.files['client.bin'], 'client.bin',
        { wte: [d.vocab, d.hidden], norm: [d.hidden] }, dtype);   // read as [N, K]

    if (!d.tied_lm_head) {
        // Uses the same onBytes as client.bin: loadWeights() counts both
        // files as one unit of progress.
        const lmBlob = await loadFile(baseUrl, 'lm_head.bin', onBytes);
        const head = await uploadTensors(device, lmBlob, manifest.files['lm_head.bin'],
            'lm_head.bin', { lm_head: [d.vocab, d.hidden] }, dtype);
        out.lm_head = head.lm_head;
        out.bytes += head.bytes;
    }
    return out;
}

// What position 0 needs: `wte`, and the final norm stored in the same file.
// Not the LM head: on an untied checkpoint that is a second table of the same
// size, and only the peer holding the last position reads it.
//
// `have` is the weights a Runner already holds. A norm it has is not uploaded
// again.
export async function loadEmbed(device, { baseUrl, manifest, d, onBytes, have } = {}) {
    baseUrl ??= await activeWeights();
    const dtype = dtypeOf(device, manifest, baseUrl);
    // Tied, and the head is already loaded: its table is the embedding.
    if (d.tied_lm_head && have?.lm_head) {
        return Object.defineProperty({ wte: have.lm_head, norm: have.norm }, 'bytes',
            { value: 0, writable: true });
    }
    const blob = await loadFile(baseUrl, 'client.bin', onBytes);
    const out = await uploadTensors(device, blob, manifest.files['client.bin'], 'client.bin',
        { wte: [d.vocab, d.hidden], ...(have?.norm ? {} : { norm: [d.hidden] }) }, dtype,
        { strict: false });
    return out;
}

// What the last position needs: the final norm and the LM head. On a tied
// checkpoint the head is `wte`, so this loads client.bin, or nothing for a
// peer that already holds the embedding. On an untied checkpoint it loads
// lm_head.bin plus the norm. The norm is a few KB of client.bin and is fetched
// as a range, to avoid downloading `wte`, which this position does not read.
export async function loadHead(device, { baseUrl, manifest, d, onBytes, have } = {}) {
    baseUrl ??= await activeWeights();
    const dtype = dtypeOf(device, manifest, baseUrl);
    const none = () => Object.defineProperty({}, 'bytes', { value: 0, writable: true });

    if (d.tied_lm_head) {
        if (have?.wte && have?.norm) return Object.assign(none(), { norm: have.norm, wte: have.wte });
        const out = await loadEmbed(device, { baseUrl, manifest, d, onBytes, have });
        out.norm ??= have?.norm;
        return out;
    }

    const out = none();
    if (have?.norm) {
        out.norm = have.norm;
    } else {
        const norm = await loadNorm(device, { baseUrl, manifest, d, dtype });
        out.norm = norm.norm;
        out.bytes += norm.bytes;
    }
    const blob = await loadFile(baseUrl, 'lm_head.bin', onBytes);
    const head = await uploadTensors(device, blob, manifest.files['lm_head.bin'],
        'lm_head.bin', { lm_head: [d.vocab, d.hidden] }, dtype);
    out.lm_head = head.lm_head;
    out.bytes += head.bytes;
    return out;
}

// client.bin's `norm` only, with one ranged request for its bytes. A server
// that ignores Range answers with the whole file, which is used. On any other
// failure the whole file is fetched with loadFile().
async function loadNorm(device, { baseUrl, manifest, d, dtype }) {
    const entries = manifest.files['client.bin'];
    const e = entries.find((x) => x.name === 'norm');
    const expected = { norm: [d.hidden] };
    if (!e) throw new Error('client.bin: no tensor named norm');
    try {
        const { status, buf } = await fetchRange(`${baseUrl}/client.bin`, e.offset,
            e.offset + e.size - 1);
        if (status === 206 && buf.byteLength === e.size) {
            return await uploadTensors(device, buf, [{ ...e, offset: 0 }], 'client.bin', expected, dtype);
        }
        if (status === 200 && buf.byteLength >= e.offset + e.size) {
            return await uploadTensors(device, buf, entries, 'client.bin', expected, dtype,
                { strict: false });
        }
    } catch (err) {
        console.warn(`client.bin: could not fetch norm as a range (${err.message}); `
            + `fetching the whole file`);
    }
    const blob = await loadFile(baseUrl, 'client.bin');
    return uploadTensors(device, blob, entries, 'client.bin', expected, dtype, { strict: false });
}

// Returns the weights object plus the dims it was loaded with. Pass those dims
// to forward()/makeCache()/makeScratch().
//
// `blocks` is a half-open range [lo, hi) for part of the stack. The returned
// `layers` is indexed from 0 either way. `client` fetches client.bin (and
// lm_head.bin), needed only for the embedding and head.
export async function loadWeights(device, { baseUrl, manifest, d,
                                            onProgress, blocks, client = true } = {}) {
    baseUrl ??= await activeWeights();
    manifest ??= await loadManifest(baseUrl);
    d ??= manifest ? dimsFromManifest(manifest) : DIMS;

    const problems = checkManifest(manifest, d);
    if (problems.length) {
        throw new Error(`${manifest.model ?? baseUrl} cannot run on these kernels:\n  ` +
            problems.join('\n  '));
    }

    dtypeOf(device, manifest, baseUrl);        // fail before downloading anything

    const [lo, hi] = blocks ?? [0, d.n_layer];
    const total = (hi - lo) + (client ? 1 : 0);
    let done = 0;
    const log = (msg) => onProgress?.(done, total, msg);
    const mib = (name) => (b, t) =>
        log(`${name} ${(b / 2 ** 20).toFixed(0)}/${(t / 2 ** 20).toFixed(0)} MiB`);

    let shared = null;
    if (client) {
        log('fetching client.bin…');
        shared = await loadClient(device, { baseUrl, manifest, d, onBytes: mib('client.bin') });
        done++;
        log('client.bin uploaded');
    }

    const layers = [];
    for (let i = lo; i < hi; i++) {
        const fn = `block_${String(i).padStart(2, '0')}.bin`;
        log(`fetching ${fn}…`);
        layers.push(await loadBlock(device, { baseUrl, manifest, d, index: i,
                                              onBytes: mib(fn) }));
        done++;
        log(`${fn} uploaded`);
    }

    return { wte: shared?.wte, norm: shared?.norm, lm_head: shared?.lm_head, layers, manifest, d };
}
