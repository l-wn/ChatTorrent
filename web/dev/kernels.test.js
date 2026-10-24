// Correctness tests for every kernel in kernels.js.
//
// Each test builds random inputs, runs the GPU kernel, and compares against a
// CPU reference. The references are written from the spec in docs/model.md,
// not from the WGSL, so a mistake in a shader is not copied into its
// reference.
//
// Every dim in a test is a different number, and none is a multiple of its
// kernel's workgroup size. This exposes swapped strides (T_act vs N_max vs D)
// and missing bounds checks on the partial last workgroup, which round or
// equal dims would hide.
//
// Run with runAllTests(device), which returns { ok, rows }. The rows name each
// kernel that failed. engine.js refuses to serve on a failure and reports the
// names.

import {
    F32, DIMS, uploadF32, uploadF16, uploadQ8, makeStorage, readF32,
    matmulNobias, GEMV_ROWS_MAX, gatherEmbedNoPos, rmsnorm, reshapeHeads, scatterKV, rope,
    qkScoresGQA, causalSoftmax, attnAVGQA, mergeHeads, addInplace, siluMul, withPass,
    sigmoidMul, causalConv1d, l2NormalizeHeads, gatedDeltaNetStep, updateConvHistory,
} from '../model/kernels.js';

// --- Harness ---

// Deterministic LCG, so a failing test reproduces on the next run.
function makeRng(seed) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

// Values in [-1, 1), rounded to f32 so the CPU reference starts from the same
// bits the GPU sees.
function randF32(n, rng) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = rng() * 2 - 1;
    return a;
}

function makeReporter() {
    const rows = [];
    return {
        rows,
        // Elementwise comparison against a reference array.
        close(name, got, want, atol = 1e-5) {
            if (got.length !== want.length) {
                rows.push({ name, ok: false, detail: `length ${got.length}, want ${want.length}` });
                return false;
            }
            let maxErr = 0, at = -1;
            for (let i = 0; i < got.length; i++) {
                if (!Number.isFinite(got[i])) {
                    rows.push({ name, ok: false, detail: `non-finite ${got[i]} at index ${i}` });
                    return false;
                }
                const e = Math.abs(got[i] - want[i]);
                if (e > maxErr) { maxErr = e; at = i; }
            }
            const ok = maxErr <= atol;
            rows.push({
                name, ok,
                detail: ok
                    ? `max err ${maxErr.toExponential(2)}`
                    : `max err ${maxErr.toExponential(2)} at [${at}]: got ${got[at]}, want ${want[at]}, atol ${atol}`,
            });
            return ok;
        },
        // A named boolean check.
        assert(name, ok, detail = '') {
            rows.push({ name, ok: !!ok, detail });
            return !!ok;
        },
    };
}

// Record one kernel call and submit it. `record` receives an open compute pass.
function submit(device, label, record) {
    const enc = device.createCommandEncoder({ label });
    withPass(enc, record, label);
    device.queue.submit([enc.finish()]);
}

// --- CPU references, from the spec (docs/model.md) ---

function matmulNobiasCPU(X, W, M, K, N) {
    const Y = new Float32Array(M * N);
    for (let row = 0; row < M; row++)
        for (let col = 0; col < N; col++) {
            let acc = 0;
            for (let i = 0; i < K; i++) acc += X[row * K + i] * W[col * K + i];
            Y[row * N + col] = acc;
        }
    return Y;
}

function gatherEmbedCPU(ids, wte, T, H) {
    const out = new Float32Array(T * H);
    for (let t = 0; t < T; t++)
        for (let h = 0; h < H; h++) out[t * H + h] = wte[ids[t] * H + h];
    return out;
}

function rmsnormCPU(x, weight, T, H, eps = DIMS.rms_eps) {
    const y = new Float32Array(T * H);
    for (let t = 0; t < T; t++) {
        let ms = 0;
        for (let h = 0; h < H; h++) { const v = x[t * H + h]; ms += v * v; }
        const scale = 1 / Math.sqrt(ms / H + eps);
        for (let h = 0; h < H; h++) y[t * H + h] = x[t * H + h] * scale * weight[h];
    }
    return y;
}

function reshapeHeadsCPU(src, T, nHeads, D) {
    const dst = new Float32Array(nHeads * T * D);
    for (let h = 0; h < nHeads; h++)
        for (let t = 0; t < T; t++)
            for (let d = 0; d < D; d++)
                dst[h * T * D + t * D + d] = src[t * nHeads * D + h * D + d];
    return dst;
}

function mergeHeadsCPU(heads, T, nHeads, D) {
    const out = new Float32Array(T * nHeads * D);
    for (let h = 0; h < nHeads; h++)
        for (let t = 0; t < T; t++)
            for (let d = 0; d < D; d++)
                out[t * nHeads * D + h * D + d] = heads[h * T * D + t * D + d];
    return out;
}

function scatterKVCPU(src, cache0, T_new, T_past, N_max, nKV, D) {
    const cache = Float32Array.from(cache0);
    for (let n = 0; n < nKV; n++)
        for (let t = 0; t < T_new; t++)
            for (let d = 0; d < D; d++)
                cache[n * N_max * D + (T_past + t) * D + d] = src[t * nKV * D + n * D + d];
    return cache;
}

// Llama 3.1's "llama3" RoPE scaling, written from HF's public spec
// (transformers/modeling_rope_utils.py::_compute_llama3_parameters), not from
// kernels.js: unchanged at short wavelengths, divided by `factor` at long
// ones, and linearly blended in between.
function llama3InvFreqCPU(f0, { factor, low_freq_factor, high_freq_factor,
                                 original_max_position_embeddings: oldLen }) {
    const wavelen = (2 * Math.PI) / f0;
    const lowWavelen = oldLen / low_freq_factor, highWavelen = oldLen / high_freq_factor;
    if (wavelen < highWavelen) return f0;
    if (wavelen > lowWavelen) return f0 / factor;
    const smooth = (oldLen / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
    return smooth * f0 + (1 - smooth) * (f0 / factor);
}

// `interleaved` produces the GPT-J pairing, not Llama's split-half. Used only
// to check that the test can tell the two apart.
function ropeCPU(buf, nHeads, T, D, T_past, timeStride, row0,
                 { interleaved = false, theta = DIMS.rope_theta, scaling = null, rotDim = D } = {}) {
    const out = Float32Array.from(buf);
    const half = rotDim / 2;
    for (let h = 0; h < nHeads; h++)
        for (let t = 0; t < T; t++) {
            const base = h * timeStride * D + (row0 + t) * D;
            const absPos = T_past + t;
            for (let i = 0; i < half; i++) {
                let invFreq = Math.pow(theta, -(2 * i) / rotDim);
                if (scaling) invFreq = llama3InvFreqCPU(invFreq, scaling);
                const ang = absPos * invFreq;
                const c = Math.cos(ang), s = Math.sin(ang);
                const ia = interleaved ? 2 * i : i;
                const ib = interleaved ? 2 * i + 1 : i + half;
                const a = buf[base + ia], b = buf[base + ib];
                out[base + ia] = a * c - b * s;
                out[base + ib] = b * c + a * s;
            }
        }
    return out;
}

function qkScoresGQACPU(Q, K, T_new, T_act, N_max, D, nHeads, group) {
    const out = new Float32Array(nHeads * T_new * T_act);
    const scale = 1 / Math.sqrt(D);
    for (let h = 0; h < nHeads; h++) {
        const kvh = Math.floor(h / group);
        for (let i = 0; i < T_new; i++)
            for (let j = 0; j < T_act; j++) {
                let acc = 0;
                for (let d = 0; d < D; d++)
                    acc += Q[h * T_new * D + i * D + d] * K[kvh * N_max * D + j * D + d];
                out[h * T_new * T_act + i * T_act + j] = acc * scale;
            }
    }
    return out;
}

function causalSoftmaxCPU(scores, T_new, T_act, T_past, nHeads) {
    const out = Float32Array.from(scores);
    for (let h = 0; h < nHeads; h++)
        for (let i = 0; i < T_new; i++) {
            const base = h * T_new * T_act + i * T_act;
            const lim = T_past + i;                 // last visible column
            let max = -Infinity;
            for (let k = 0; k < T_act && k <= lim; k++) max = Math.max(max, scores[base + k]);
            let sum = 0;
            for (let k = 0; k < T_act; k++) {
                if (k <= lim) { const e = Math.exp(scores[base + k] - max); out[base + k] = e; sum += e; }
                else out[base + k] = 0;
            }
            for (let k = 0; k < T_act && k <= lim; k++) out[base + k] /= sum;
        }
    return out;
}

function attnAVGQACPU(scores, V, T_new, T_act, N_max, D, nHeads, group) {
    const out = new Float32Array(nHeads * T_new * D);
    for (let h = 0; h < nHeads; h++) {
        const kvh = Math.floor(h / group);
        for (let i = 0; i < T_new; i++)
            for (let d = 0; d < D; d++) {
                let acc = 0;
                for (let j = 0; j < T_act; j++)
                    acc += scores[h * T_new * T_act + i * T_act + j] * V[kvh * N_max * D + j * D + d];
                out[h * T_new * D + i * D + d] = acc;
            }
    }
    return out;
}

const addInplaceCPU = (a, b) => a.map((v, i) => v + b[i]);
const siluMulCPU = (g, up) => g.map((v, i) => (v / (1 + Math.exp(-v))) * up[i]);
const sigmoidMulCPU = (gate, a) => gate.map((v, i) => (1 / (1 + Math.exp(-v))) * a[i]);

// out[t,c] = silu of a depthwise causal conv of x over the last K positions,
// ending at absolute position T_past+t. x is indexed by absolute position.
// T_past=0 means the whole sequence is in x.
function causalConv1dCPU(x, weight, T, C, K, T_past = 0) {
    const out = new Float32Array(T * C);
    for (let t = 0; t < T; t++)
        for (let c = 0; c < C; c++) {
            let acc = 0;
            for (let k = 0; k < K; k++) {
                const tt = T_past + t - (K - 1) + k;
                if (tt >= 0) acc += weight[c * K + k] * x[tt * C + c];
            }
            out[t * C + c] = acc / (1 + Math.exp(-acc));
        }
    return out;
}

// The same conv, with reads before row 0 taken from a persistent [K-1,C] ring
// and not zero-padded. This is decode across jobs, where `x` holds only the
// current job's rows (T_past is 0; see causalConv1dSrc).
function causalConv1dHistCPU(x, weight, hist, T, C, K) {
    const out = new Float32Array(T * C);
    const Kh = K - 1;
    for (let t = 0; t < T; t++)
        for (let c = 0; c < C; c++) {
            let acc = 0;
            for (let k = 0; k < K; k++) {
                const tt = t - Kh + k;
                const v = tt >= 0 ? x[tt * C + c] : hist[(tt + Kh) * C + c];
                acc += weight[c * K + k] * v;
            }
            out[t * C + c] = acc / (1 + Math.exp(-acc));
        }
    return out;
}

// new_hist[r] = row (T_new-(K-1)+r) of the logical [hist ++ newRows] sequence.
// A negative row is in the old ring, a non-negative row in this chunk.
function updateConvHistoryCPU(hist, newRows, T_new, C, K) {
    const Kh = K - 1;
    const out = new Float32Array(Kh * C);
    for (let r = 0; r < Kh; r++) {
        const logical = T_new - Kh + r;
        for (let c = 0; c < C; c++) {
            out[r * C + c] = logical < 0 ? hist[(logical + Kh) * C + c] : newRows[logical * C + c];
        }
    }
    return out;
}

// y[t, srcHead*group+g, :] = L2-normalize(x[t,srcHead,:]), duplicated `group` times
function l2NormalizeHeadsCPU(x, T, srcHeads, D, group = 1) {
    const dstHeads = srcHeads * group;
    const out = new Float32Array(T * dstHeads * D);
    for (let t = 0; t < T; t++)
        for (let sh = 0; sh < srcHeads; sh++) {
            const base = (t * srcHeads + sh) * D;
            let ss = 0;
            for (let d = 0; d < D; d++) ss += x[base + d] * x[base + d];
            const scale = 1 / Math.sqrt(ss + 1e-6);
            for (let g = 0; g < group; g++) {
                const dstBase = (t * dstHeads + sh * group + g) * D;
                for (let d = 0; d < D; d++) out[dstBase + d] = x[base + d] * scale;
            }
        }
    return out;
}

// The Gated DeltaNet recurrence as one JS loop over T tokens. Checks that
// gatedDeltaNetStep gives the same result whether it runs the chunk in one
// dispatch or one dispatch per token with persistent state. The state must
// finish decaying before any column's kv_mem is read, so the loop has two
// phases.
function gatedDeltaNetCPU(q, k, v, aRaw, bRaw, ALog, dtBias, T, heads, Dk, Dv) {
    const S = new Float64Array(heads * Dk * Dv);
    const out = new Float32Array(T * heads * Dv);
    const qScale = 1 / Math.sqrt(Dk);
    for (let t = 0; t < T; t++)
        for (let h = 0; h < heads; h++) {
            const a = aRaw[t * heads + h] + dtBias[h];
            const softplus = a > 20 ? a + Math.log(1 + Math.exp(-a)) : Math.log(1 + Math.exp(a));
            const g = Math.exp(-Math.exp(ALog[h]) * softplus);
            const beta = 1 / (1 + Math.exp(-bRaw[t * heads + h]));
            const sBase = h * Dk * Dv;
            const qBase = (t * heads + h) * Dk, kBase = qBase, vBase = (t * heads + h) * Dv;

            for (let i = 0; i < Dk * Dv; i++) S[sBase + i] *= g;

            for (let vv = 0; vv < Dv; vv++) {
                let kv = 0;
                for (let kk = 0; kk < Dk; kk++) kv += S[sBase + kk * Dv + vv] * k[kBase + kk];
                const delta = (v[vBase + vv] - kv) * beta;

                let o = 0;
                for (let kk = 0; kk < Dk; kk++) {
                    const idx = sBase + kk * Dv + vv;
                    S[idx] += k[kBase + kk] * delta;
                    o += S[idx] * q[qBase + kk];
                }
                out[vBase + vv] = o * qScale;
            }
        }
    return out;
}

// --- Reference self-check ---
//
// Every GPU test is compared against the references above, so a bug in a
// reference would go unnoticed if the GPU had the same bug. These tests check
// the references against properties that hold by construction. They need no
// GPU and run first.

export function checkReferences(r = makeReporter()) {
    const rng = makeRng(99);

    // matmul on a hand-computed case with simple W rows. Checks the [out, in]
    // weight convention.
    {
        const X = new Float32Array([1, 2, 3, 4, 5, 6]);       // [2, 3]
        const W = new Float32Array([1, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 1]); // [4, 3]
        const want = new Float32Array([1, 2, 6, 5, 4, 5, 15, 14]);
        r.close('ref  matmul_nobias vs hand-computed', matmulNobiasCPU(X, W, 2, 3, 4), want, 0);
    }

    // reshape and merge must be exact inverses.
    {
        const T = 3, nH = 4, D = 6;
        const src = randF32(T * nH * D, rng);
        const back = mergeHeadsCPU(reshapeHeadsCPU(src, T, nH, D), T, nH, D);
        r.close('ref  merge(reshape(x)) == x', back, src, 0);
    }

    // rmsnorm with weight=1 gives a mean of squares of 1 (apart from eps).
    {
        const T = 2, H = 40;
        const x = randF32(T * H, rng);
        const y = rmsnormCPU(x, new Float32Array(H).fill(1), T, H);
        let worst = 0;
        for (let t = 0; t < T; t++) {
            let ms = 0;
            for (let h = 0; h < H; h++) ms += y[t * H + h] ** 2;
            worst = Math.max(worst, Math.abs(ms / H - 1));
        }
        r.assert('ref  rmsnorm normalises to unit mean-square', worst < 1e-4,
            `worst |ms-1| ${worst.toExponential(2)}`);
    }

    // RoPE is a rotation: pair magnitudes are preserved, and position 0 is the
    // identity. Both hold for split-half and interleaved, so the test also
    // asserts that the two pairings give different results.
    {
        const nH = 2, T = 3, D = 8, T_past = 4, half = D / 2;
        const q = randF32(nH * T * D, rng);
        const rot = ropeCPU(q, nH, T, D, T_past, T, 0);
        let worstNorm = 0;
        for (let h = 0; h < nH; h++)
            for (let t = 0; t < T; t++) {
                const base = h * T * D + t * D;
                for (let i = 0; i < half; i++)
                    worstNorm = Math.max(worstNorm, Math.abs(
                        Math.hypot(rot[base + i], rot[base + i + half]) -
                        Math.hypot(q[base + i], q[base + i + half])));
            }
        r.assert('ref  rope preserves pair magnitude', worstNorm < 1e-6,
            `worst |Δ| ${worstNorm.toExponential(2)}`);

        const atZero = ropeCPU(q, nH, T, D, 0, T, 0);
        let z = 0;
        for (let h = 0; h < nH; h++)
            for (let d = 0; d < D; d++)
                z = Math.max(z, Math.abs(atZero[h * T * D + d] - q[h * T * D + d]));
        r.assert('ref  rope is identity at position 0', z === 0, `max |Δ| ${z}`);

        const inter = ropeCPU(q, nH, T, D, T_past, T, 0, { interleaved: true });
        let diff = 0;
        for (let i = 0; i < rot.length; i++) diff = Math.max(diff, Math.abs(rot[i] - inter[i]));
        r.assert('ref  split-half != interleaved', diff > 1e-3,
            `max diff ${diff.toExponential(2)}`);
    }

    // Softmax rows sum to 1, and masked columns are exactly zero.
    {
        const T_new = 3, T_past = 2, T_act = 5, nH = 2;
        const s = randF32(nH * T_new * T_act, rng);
        const p = causalSoftmaxCPU(s, T_new, T_act, T_past, nH);
        let worstSum = 0, leaked = 0;
        for (let h = 0; h < nH; h++)
            for (let i = 0; i < T_new; i++) {
                const base = h * T_new * T_act + i * T_act;
                let sum = 0;
                for (let k = 0; k < T_act; k++) {
                    sum += p[base + k];
                    if (k > T_past + i && p[base + k] !== 0) leaked++;
                }
                worstSum = Math.max(worstSum, Math.abs(sum - 1));
            }
        r.assert('ref  causal_softmax rows sum to 1', worstSum < 1e-6,
            `worst |Σ-1| ${worstSum.toExponential(2)}`);
        r.assert('ref  causal_softmax masks the future', leaked === 0, `${leaked} leaks`);
    }

    // With a one-hot scores row, attn_av must return that V row exactly. This
    // checks the GQA head mapping and the N_max cache stride.
    {
        const T_new = 1, T_act = 4, N_max = 9, D = 3, nH = 6, group = 3, nKV = nH / group;
        const V = randF32(nKV * N_max * D, rng);
        const scores = new Float32Array(nH * T_new * T_act);
        for (let h = 0; h < nH; h++) scores[h * T_act + 2] = 1;   // select key j=2
        const out = attnAVGQACPU(scores, V, T_new, T_act, N_max, D, nH, group);
        let worst = 0;
        for (let h = 0; h < nH; h++) {
            const kvh = Math.floor(h / group);
            for (let d = 0; d < D; d++)
                worst = Math.max(worst, Math.abs(out[h * D + d] - V[kvh * N_max * D + 2 * D + d]));
        }
        r.assert('ref  attn_av with one-hot scores selects that V row', worst === 0,
            `max |Δ| ${worst}`);
    }

    return r;
}

// --- Tests ---

async function testMatmulNobias(device, r) {
    const M = 5, K = 7, N = 11;                    // none a multiple of 8
    const rng = makeRng(1);
    const X = randF32(M * K, rng), W = randF32(N * K, rng);

    const xb = uploadF32(device, X), wb = uploadF32(device, W);
    const yb = makeStorage(device, M * N * F32);
    submit(device, 'matmul_nobias', (pass) => matmulNobias(device, pass, xb, wb, yb, M, K, N));

    r.close('matmul_nobias', await readF32(device, yb, M * N * F32),
        matmulNobiasCPU(X, W, M, K, N));
}

// --- f16 weights ---
//
// The shader reads an f16 weight and must widen it to f32 before accumulating.
// The test starts from chosen f16 bit patterns, decodes them to exact f32 in
// JS for the reference, and uploads the raw bytes. No f32 -> f16 encoding is
// involved, so no rounding routine needs testing.
//
// f16ToF32 is exact for every normal f16 value.
function f16ToF32(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >>> 10) & 0x1f;
    const man = h & 0x3ff;
    if (exp === 0) return sign * (2 ** -14) * (man / 1024);      // subnormal
    if (exp === 31) return man ? NaN : sign * Infinity;
    return sign * (2 ** (exp - 15)) * (1 + man / 1024);
}

// Random f16 values with exponents 11..18 (about 2^-4 to 2^3), the range of
// real weights, away from subnormals and the 65504 maximum.
function randF16Bits(n, rng) {
    const bits = new Uint16Array(n);
    for (let i = 0; i < n; i++) {
        const sign = rng() < 0.5 ? 0x8000 : 0;
        const exp = 11 + Math.floor(rng() * 8);
        bits[i] = sign | (exp << 10) | (Math.floor(rng() * 1024) & 0x3ff);
    }
    return bits;
}

async function testMatmulNobiasF16(device, r) {
    // A device without shader-f16 can still serve an f32 checkpoint, so the
    // test is skipped, not failed. loader.js refuses an f16 checkpoint on such
    // a device.
    if (!device.features?.has('shader-f16')) {
        r.assert('matmul_nobias f16 (skipped: no shader-f16 on this device)', true);
        return;
    }

    const M = 5, K = 7, N = 11;                    // none a multiple of 8
    const rng = makeRng(11);
    const X = randF32(M * K, rng);
    const wBits = randF16Bits(N * K, rng);
    const W = Float32Array.from(wBits, f16ToF32);  // the values the GPU sees

    const xb = uploadF32(device, X), wb = uploadF16(device, wBits);
    const yb = makeStorage(device, M * N * F32);
    submit(device, 'matmul_nobias f16',
        (pass) => matmulNobias(device, pass, xb, wb, yb, M, K, N));

    // The same tolerance as the f32 build: the weights are exact f16 values
    // and the accumulator is f32, so widening is the only extra step and it is
    // lossless. A looser tolerance would not catch an f16 accumulator.
    r.close('matmul_nobias f16 weights, f32 accumulate',
        await readF32(device, yb, M * N * F32), matmulNobiasCPU(X, W, M, K, N));
}

// --- the decode path ---
//
// M = 1 uses a different kernel from M > 1: a cooperative GEMV, not a tile.
// Every decoded token has this shape. Both builds are checked against the
// same CPU reference.
async function testMatmulNobiasGemv(device, r) {
    const rng = makeRng(21);

    // K much larger than the 256-thread workgroup, so the strided accumulation
    // loop wraps many times. K and N have no common factors with the workgroup
    // sizes.
    for (const [K, N, label] of [[7, 11, 'tiny'], [2053, 577, 'K >> workgroup']]) {
        const X = randF32(K, rng), W = randF32(N * K, rng);
        const xb = uploadF32(device, X), wb = uploadF32(device, W);
        const yb = makeStorage(device, N * F32);
        submit(device, 'matmul gemv', (pass) => matmulNobias(device, pass, xb, wb, yb, 1, K, N));
        // The tolerance scales with K: the reduction sums K products, and the
        // GPU's tree order differs from the reference's sequential order.
        r.close(`matmul_nobias M=1 (${label}, K=${K} N=${N})`,
            await readF32(device, yb, N * F32), matmulNobiasCPU(X, W, 1, K, N),
            1e-5 * Math.max(1, K / 64));
    }

    // The same shape with f16 weights, since decode is where f16 is used.
    // Exact f16 inputs and an f32 accumulator, so the f32 tolerance applies.
    if (device.features?.has('shader-f16')) {
        const K = 1031, N = 129;
        const X = randF32(K, rng);
        const wBits = randF16Bits(N * K, rng);
        const W = Float32Array.from(wBits, f16ToF32);
        const xb = uploadF32(device, X), wb = uploadF16(device, wBits);
        const yb = makeStorage(device, N * F32);
        submit(device, 'matmul gemv f16',
            (pass) => matmulNobias(device, pass, xb, wb, yb, 1, K, N));
        r.close('matmul_nobias M=1 with f16 weights',
            await readF32(device, yb, N * F32), matmulNobiasCPU(X, W, 1, K, N),
            1e-5 * Math.max(1, K / 64));
    } else {
        r.assert('matmul_nobias M=1 f16 (skipped: no shader-f16)', true);
    }

    // The kernel is selected by M, so a 1-row call and the first row of a
    // tiled call go through different code and must give the same numbers.
    {
        const K = 300, N = 40, M = GEMV_ROWS_MAX + 9;
        const X = randF32(M * K, rng), W = randF32(N * K, rng);
        const xb = uploadF32(device, X), wb = uploadF32(device, W);
        const y1 = makeStorage(device, N * F32), y2 = makeStorage(device, M * N * F32);
        submit(device, 'gemv row', (pass) => matmulNobias(device, pass, xb, wb, y1, 1, K, N));
        submit(device, 'tiled rows', (pass) => matmulNobias(device, pass, xb, wb, y2, M, K, N));
        const a = await readF32(device, y1, N * F32);
        const b = (await readF32(device, y2, M * N * F32)).slice(0, N);
        r.close('the M=1 and tiled kernels agree on the same row', a, b, 1e-4);
    }
}

// --- a few rows (a speculative verify) ---
//
// For 2 <= M <= GEMV_ROWS_MAX every row must match a one-row call. The kernels
// are written to give the same bits, so a verified token is the one plain
// decoding samples; the detail says whether this GPU does. Bit-exactness is
// not required to serve (the output is right either way), so the check is to
// the GEMV tolerance. web/dev/spec.test.js holds it to the bit.
async function testMatmulRowsBitwise(device, r) {
    const rng = makeRng(31);
    const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
    const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

    async function check(label, K, N, upload) {
        const wb = upload(K, N);
        let bitwise = true, worst = 0, scale = 0;
        for (let M = 2; M <= GEMV_ROWS_MAX; M++) {
            const X = randF32(M * K, rng);
            const xb = uploadF32(device, X);
            const ym = makeStorage(device, M * N * F32);
            submit(device, `rows ${M}`, (pass) => matmulNobias(device, pass, xb, wb, ym, M, K, N));
            const rows = await readF32(device, ym, M * N * F32);
            for (let i = 0; i < M; i++) {
                const x1 = uploadF32(device, X.slice(i * K, (i + 1) * K));
                const y1 = makeStorage(device, N * F32);
                submit(device, 'row 1', (pass) => matmulNobias(device, pass, x1, wb, y1, 1, K, N));
                const one = await readF32(device, y1, N * F32);
                const row = rows.slice(i * N, (i + 1) * N);
                if (!same(one, row)) bitwise = false;
                worst = Math.max(worst, maxDiff(one, row));
                scale = Math.max(scale, ...one.map(Math.abs));
            }
        }
        r.assert(`matmul rows ${label} M=2..${GEMV_ROWS_MAX} match M=1`,
            worst <= 1e-5 * Math.max(1, K / 64) * Math.max(1, scale),
            bitwise ? 'bitwise' : `not bitwise, max diff ${worst.toExponential(2)}`);
    }

    // vec4 (K % 4 == 0), the scalar build, f16, and q8 with a partial group.
    await check('f32 vec4', 516, 77, (K, N) => uploadF32(device, randF32(N * K, rng)));
    await check('f32 scalar', 301, 45, (K, N) => uploadF32(device, randF32(N * K, rng)));
    if (device.features?.has('shader-f16')) {
        await check('f16', 516, 77, (K, N) => uploadF16(device, randF16Bits(N * K, rng)));
    } else {
        r.assert('matmul rows f16 (skipped: no shader-f16)', true);
    }
    const group = 64;
    await check('q8', 6 * group + 32, 70, (K, N) => {
        const data = new Int8Array(N * K);
        for (let i = 0; i < data.length; i++) data[i] = Math.floor(rng() * 255) - 127;
        const scales = randF32(N * Math.ceil(K / group), rng);
        return uploadQ8(device, new Uint8Array(data.buffer), scales, group);
    });

    // And against the CPU reference, so the rows are right as well as alike.
    {
        const M = 5, K = 516, N = 77;
        const X = randF32(M * K, rng), W = randF32(N * K, rng);
        const xb = uploadF32(device, X), wb = uploadF32(device, W);
        const yb = makeStorage(device, M * N * F32);
        submit(device, 'rows ref', (pass) => matmulNobias(device, pass, xb, wb, yb, M, K, N));
        r.close('matmul rows M=5 vs reference', await readF32(device, yb, M * N * F32),
            matmulNobiasCPU(X, W, M, K, N), 1e-5 * Math.max(1, K / 64));
    }
}

async function testGatherEmbed(device, r) {
    const T = 3, H = 70, vocab = 13;               // H not a multiple of 64
    const rng = makeRng(2);
    const wte = randF32(vocab * H, rng);
    const ids = new Float32Array([7, 0, 12]);      // kernel casts f32 -> u32

    const idb = uploadF32(device, ids), wb = uploadF32(device, wte);
    const ob = makeStorage(device, T * H * F32);
    submit(device, 'gather_embed', (pass) => gatherEmbedNoPos(device, pass, idb, wb, ob, T, H));

    r.close('gather_embed_nopos', await readF32(device, ob, T * H * F32),
        gatherEmbedCPU(ids, wte, T, H));
}

async function testRmsnorm(device, r) {
    const T = 3, H = 70;
    const rng = makeRng(3);
    const x = randF32(T * H, rng), w = randF32(H, rng);

    const xb = uploadF32(device, x), wb = uploadF32(device, w);
    const yb = makeStorage(device, T * H * F32);
    submit(device, 'rmsnorm', (pass) => rmsnorm(device, pass, xb, wb, yb, T, H));

    const got = await readF32(device, yb, T * H * F32);
    r.close('rmsnorm', got, rmsnormCPU(x, w, T, H));

    // With weight = 1 the normalized row must have a mean of squares of 1.
    // This catches a LayerNorm-style mean subtraction.
    const ones = new Float32Array(H).fill(1);
    const ob = uploadF32(device, ones);
    const yb2 = makeStorage(device, T * H * F32);
    submit(device, 'rmsnorm unit', (pass) => rmsnorm(device, pass, xb, ob, yb2, T, H));
    const g2 = await readF32(device, yb2, T * H * F32);

    let worst = 0;
    for (let t = 0; t < T; t++) {
        let ms = 0;
        for (let h = 0; h < H; h++) ms += g2[t * H + h] ** 2;
        worst = Math.max(worst, Math.abs(ms / H - 1));
    }
    r.assert('rmsnorm  mean-square == 1 (weight=1)', worst < 1e-4, `worst |ms-1| ${worst.toExponential(2)}`);
}

async function testReshapeMergeHeads(device, r) {
    const T = 3, nHeads = 4, D = 6;
    const rng = makeRng(4);
    const src = randF32(T * nHeads * D, rng);

    const sb = uploadF32(device, src);
    const hb = makeStorage(device, nHeads * T * D * F32);
    submit(device, 'reshape_heads', (pass) => reshapeHeads(device, pass, sb, hb, T, nHeads, D));
    const heads = await readF32(device, hb, nHeads * T * D * F32);
    r.close('reshape_heads', heads, reshapeHeadsCPU(src, T, nHeads, D));

    const mb = makeStorage(device, T * nHeads * D * F32);
    submit(device, 'merge_heads', (pass) => mergeHeads(device, pass, hb, mb, T, nHeads, D));
    const merged = await readF32(device, mb, T * nHeads * D * F32);
    r.close('merge_heads', merged, mergeHeadsCPU(heads, T, nHeads, D));

    // merge(reshape(x)) == x. Catches a pair of transposed strides that the
    // two references would each accept.
    r.close('reshape→merge round trip', merged, src, 0);
}

async function testScatterKV(device, r) {
    const T_new = 3, T_past = 2, N_max = 11, nKV = 3, D = 6;   // N_max > T_past+T_new
    const rng = makeRng(5);
    const src = randF32(T_new * nKV * D, rng);
    const SENTINEL = 999;
    const cache0 = new Float32Array(nKV * N_max * D).fill(SENTINEL);

    const sb = uploadF32(device, src), cb = uploadF32(device, cache0);
    submit(device, 'scatter_kv', (pass) =>
        scatterKV(device, pass, sb, cb, T_new, T_past, N_max, nKV, D));

    const got = await readF32(device, cb, nKV * N_max * D * F32);
    r.close('scatter_kv', got, scatterKVCPU(src, cache0, T_new, T_past, N_max, nKV, D));

    // Everything outside [T_past, T_past+T_new) must be unchanged. A wrong
    // head stride would write into a neighbouring head's rows.
    let clobbered = 0;
    for (let n = 0; n < nKV; n++)
        for (let t = 0; t < N_max; t++)
            if (t < T_past || t >= T_past + T_new)
                for (let d = 0; d < D; d++)
                    if (got[n * N_max * D + t * D + d] !== SENTINEL) clobbered++;
    r.assert('scatter_kv  leaves the rest of the cache alone', clobbered === 0,
        `${clobbered} clobbered slots`);
}

async function testRope(device, r) {
    const nHeads = 3, T = 4, D = 8, T_past = 5;
    const rng = makeRng(6);

    // --- Q call site: dense [nHeads, T, D], time stride T, rows start at 0 ---
    const q = randF32(nHeads * T * D, rng);
    const qb = uploadF32(device, q);
    submit(device, 'rope Q', (pass) => rope(device, pass, qb, nHeads, T, D, T_past, T, 0));
    const gotQ = await readF32(device, qb, nHeads * T * D * F32);

    r.close('rope  Q (split-half)', gotQ, ropeCPU(q, nHeads, T, D, T_past, T, 0));

    // If the interleaved pairing matched, the split-half check above would
    // prove nothing.
    const inter = ropeCPU(q, nHeads, T, D, T_past, T, 0, { interleaved: true });
    let maxVsInterleaved = 0;
    for (let i = 0; i < gotQ.length; i++)
        maxVsInterleaved = Math.max(maxVsInterleaved, Math.abs(gotQ[i] - inter[i]));
    r.assert('rope  differs from interleaved (GPT-J) pairing', maxVsInterleaved > 1e-3,
        `max diff vs interleaved ${maxVsInterleaved.toExponential(2)}`);

    // A rotation preserves the length of each (i, i+half) pair for any angle.
    // This does not use the reference, so it catches a wrong sign or a dropped
    // term that the reference might share.
    const half = D / 2;
    let worstNorm = 0;
    for (let h = 0; h < nHeads; h++)
        for (let t = 0; t < T; t++) {
            const base = h * T * D + t * D;
            for (let i = 0; i < half; i++) {
                const before = Math.hypot(q[base + i], q[base + i + half]);
                const after = Math.hypot(gotQ[base + i], gotQ[base + i + half]);
                worstNorm = Math.max(worstNorm, Math.abs(after - before));
            }
        }
    r.assert('rope  preserves each pair\'s magnitude', worstNorm < 1e-5,
        `worst |Δ| ${worstNorm.toExponential(2)}`);

    // Position 0 has angle 0 for every channel, so with T_past = 0, row 0 must
    // be bitwise unchanged.
    const q2 = randF32(nHeads * T * D, rng);
    const q2b = uploadF32(device, q2);
    submit(device, 'rope pos0', (pass) => rope(device, pass, q2b, nHeads, T, D, 0, T, 0));
    const gotQ2 = await readF32(device, q2b, nHeads * T * D * F32);
    let pos0Err = 0;
    for (let h = 0; h < nHeads; h++)
        for (let d = 0; d < D; d++)
            pos0Err = Math.max(pos0Err, Math.abs(gotQ2[h * T * D + d] - q2[h * T * D + d]));
    r.assert('rope  is identity at absolute position 0', pos0Err === 0,
        `max |Δ| ${pos0Err}`);

    // --- K-cache call site: [nKV, N_max, D], time stride N_max, rows at T_past ---
    // N_max differs from T, so a stride mix-up cannot pass by coincidence.
    const nKV = 3, N_max = 11;
    const k = randF32(nKV * N_max * D, rng);
    const kb = uploadF32(device, k);
    submit(device, 'rope K', (pass) => rope(device, pass, kb, nKV, T, D, T_past, N_max, T_past));
    const gotK = await readF32(device, kb, nKV * N_max * D * F32);

    r.close('rope  K cache (stride N_max, row0 T_past)', gotK,
        ropeCPU(k, nKV, T, D, T_past, N_max, T_past));

    // Rows outside the requested range must be bit-identical. Rotating a
    // cached row twice is a common KV-cache RoPE bug.
    let touched = 0;
    for (let n = 0; n < nKV; n++)
        for (let t = 0; t < N_max; t++)
            if (t < T_past || t >= T_past + T)
                for (let d = 0; d < D; d++) {
                    const idx = n * N_max * D + t * D + d;
                    if (gotK[idx] !== k[idx]) touched++;
                }
    r.assert('rope  touches only rows [T_past, T_past+T)', touched === 0,
        `${touched} rows outside the window changed`);

    // --- llama3 rope scaling (Llama 3.1's config; SmolLM2's DIMS has none) ---
    const scaling = { factor: 8, low_freq_factor: 1, high_freq_factor: 4,
                       original_max_position_embeddings: 8192 };
    const theta3 = 500000;
    const q3 = randF32(nHeads * T * D, rng);
    const q3b = uploadF32(device, q3);
    submit(device, 'rope llama3', (pass) =>
        rope(device, pass, q3b, nHeads, T, D, T_past, T, 0, theta3, scaling));
    const gotQ3 = await readF32(device, q3b, nHeads * T * D * F32);
    r.close('rope  llama3 scaling matches the reference', gotQ3,
        ropeCPU(q3, nHeads, T, D, T_past, T, 0, { theta: theta3, scaling }));

    // Scaling must change the result, or the assertion above would also pass
    // with scaling ignored.
    const unscaled = ropeCPU(q3, nHeads, T, D, T_past, T, 0, { theta: theta3 });
    let maxVsUnscaled = 0;
    for (let i = 0; i < gotQ3.length; i++)
        maxVsUnscaled = Math.max(maxVsUnscaled, Math.abs(gotQ3[i] - unscaled[i]));
    r.assert('rope  llama3 scaling differs from the plain form', maxVsUnscaled > 1e-3,
        `max diff vs unscaled ${maxVsUnscaled.toExponential(2)}`);

    // --- partial rotary (Qwen3.8-27B: rot_dim < head_dim) ---
    const rotDim = D / 2;                             // rotate only the first half
    const q4 = randF32(nHeads * T * D, rng);
    const q4b = uploadF32(device, q4);
    submit(device, 'rope partial', (pass) =>
        rope(device, pass, q4b, nHeads, T, D, T_past, T, 0, undefined, undefined, rotDim));
    const gotQ4 = await readF32(device, q4b, nHeads * T * D * F32);
    r.close('rope  partial rotary matches the reference', gotQ4,
        ropeCPU(q4, nHeads, T, D, T_past, T, 0, { rotDim }));

    // Channels [rotDim, D) must be bit-identical: the pass-through half of a
    // partial-rotary checkpoint (apply_rotary_pos_emb's q_rot/q_pass split).
    let untouched = 0;
    for (let h = 0; h < nHeads; h++)
        for (let t = 0; t < T; t++)
            for (let d = rotDim; d < D; d++) {
                const idx = h * T * D + t * D + d;
                if (gotQ4[idx] !== q4[idx]) untouched++;
            }
    r.assert('rope  partial rotary leaves [rotDim, D) untouched', untouched === 0,
        `${untouched} channels outside rot_dim changed`);
}

async function testAttention(device, r) {
    // Every dim distinct. N_max > T_act, so the cache stride is tested.
    const T_new = 3, T_past = 2, T_act = T_past + T_new, N_max = 11;
    const D = 6, nHeads = 9, group = DIMS.group, nKV = nHeads / group;
    const rng = makeRng(7);

    const Q = randF32(nHeads * T_new * D, rng);
    // Only the first T_act rows of each head are valid. The rest is random, so
    // a read past T_act shows up as a mismatch.
    const K = randF32(nKV * N_max * D, rng);
    const V = randF32(nKV * N_max * D, rng);

    const qb = uploadF32(device, Q), kb = uploadF32(device, K), vb = uploadF32(device, V);
    const sb = makeStorage(device, nHeads * T_new * T_act * F32);
    const ob = makeStorage(device, nHeads * T_new * D * F32);

    // --- qk_scores_gqa ---
    submit(device, 'qk_scores', (pass) =>
        qkScoresGQA(device, pass, qb, kb, sb, T_new, T_act, N_max, D, nHeads, group));
    const rawScores = await readF32(device, sb, nHeads * T_new * T_act * F32);
    const wantScores = qkScoresGQACPU(Q, K, T_new, T_act, N_max, D, nHeads, group);
    r.close('qk_scores_gqa', rawScores, wantScores);

    // Heads in one GQA group share a KV head, so with the same query they
    // would agree. Assert they differ, which shows Q is indexed by h.
    let spread = 0;
    for (let i = 0; i < T_new * T_act; i++)
        spread = Math.max(spread, Math.abs(rawScores[i] - rawScores[T_new * T_act + i]));
    r.assert('qk_scores_gqa  distinct heads within a GQA group', spread > 1e-3,
        `head0 vs head1 max diff ${spread.toExponential(2)}`);

    // --- causal_softmax ---
    submit(device, 'softmax', (pass) =>
        causalSoftmax(device, pass, sb, T_new, T_act, T_past, nHeads));
    const probs = await readF32(device, sb, nHeads * T_new * T_act * F32);
    r.close('causal_softmax', probs, causalSoftmaxCPU(wantScores, T_new, T_act, T_past, nHeads));

    let worstSum = 0, leaked = 0;
    for (let h = 0; h < nHeads; h++)
        for (let i = 0; i < T_new; i++) {
            const base = h * T_new * T_act + i * T_act;
            let sum = 0;
            for (let k = 0; k < T_act; k++) {
                sum += probs[base + k];
                if (k > T_past + i && probs[base + k] !== 0) leaked++;
            }
            worstSum = Math.max(worstSum, Math.abs(sum - 1));
        }
    r.assert('causal_softmax  rows sum to 1', worstSum < 1e-5, `worst |Σ-1| ${worstSum.toExponential(2)}`);
    r.assert('causal_softmax  masks the future exactly', leaked === 0, `${leaked} unmasked cells`);

    // --- attn_av_gqa ---
    submit(device, 'attn_av', (pass) =>
        attnAVGQA(device, pass, sb, vb, ob, T_new, T_act, N_max, D, nHeads, group));
    const ctx = await readF32(device, ob, nHeads * T_new * D * F32);
    r.close('attn_av_gqa', ctx, attnAVGQACPU(probs, V, T_new, T_act, N_max, D, nHeads, group));

    // All three kernels chained, from the original inputs. Catches a layout
    // disagreement between kernels that each match their own reference.
    const chain = attnAVGQACPU(
        causalSoftmaxCPU(qkScoresGQACPU(Q, K, T_new, T_act, N_max, D, nHeads, group),
            T_new, T_act, T_past, nHeads),
        V, T_new, T_act, N_max, D, nHeads, group);
    r.close('attention chain (qk→softmax→av)', ctx, chain, 2e-5);
}

// A speculative verify runs n query rows at once. Row i must match a one-row
// job at T_past + i; the detail says whether it does to the bit (see
// testMatmulRowsBitwise).
async function testAttentionRowsBitwise(device, r) {
    const T_past = 37, n = 6, N_max = 64;
    const D = 32, nHeads = 8, group = 4, nKV = nHeads / group;
    const rng = makeRng(17);
    const Q = randF32(nHeads * n * D, rng);
    const K = randF32(nKV * N_max * D, rng), V = randF32(nKV * N_max * D, rng);
    const kb = uploadF32(device, K), vb = uploadF32(device, V);

    async function run(Qrows, T_new, past) {
        const T_act = past + T_new;
        const qb = uploadF32(device, Qrows);
        const sb = makeStorage(device, nHeads * T_new * T_act * F32);
        const ob = makeStorage(device, nHeads * T_new * D * F32);
        submit(device, 'attn rows', (pass) => {
            qkScoresGQA(device, pass, qb, kb, sb, T_new, T_act, N_max, D, nHeads, group);
            causalSoftmax(device, pass, sb, T_new, T_act, past, nHeads);
            attnAVGQA(device, pass, sb, vb, ob, T_new, T_act, N_max, D, nHeads, group);
        });
        return readF32(device, ob, nHeads * T_new * D * F32);
    }

    const all = await run(Q, n, T_past);
    let bitwise = true, worst = 0;
    for (let i = 0; i < n; i++) {
        const q1 = new Float32Array(nHeads * D);
        for (let h = 0; h < nHeads; h++) q1.set(Q.subarray((h * n + i) * D, (h * n + i + 1) * D), h * D);
        const one = await run(q1, 1, T_past + i);
        for (let h = 0; h < nHeads; h++)
            for (let d = 0; d < D; d++) {
                const a = one[h * D + d], b = all[(h * n + i) * D + d];
                if (!Object.is(a, b)) bitwise = false;
                worst = Math.max(worst, Math.abs(a - b));
            }
    }
    r.assert(`attention row i of T_new=${n} matches T_new=1 at T_past+i`, worst <= 1e-5,
        bitwise ? 'bitwise' : `not bitwise, max diff ${worst.toExponential(2)}`);
}

async function testAddInplace(device, r) {
    const N = 1000;                                 // not a multiple of 256
    const rng = makeRng(8);
    const a = randF32(N, rng), b = randF32(N, rng);

    const ab = uploadF32(device, a), bb = uploadF32(device, b);
    submit(device, 'add_inplace', (pass) => addInplace(device, pass, ab, bb, N));

    r.close('add_inplace', await readF32(device, ab, N * F32), addInplaceCPU(a, b));
}

async function testSiluMul(device, r) {
    const N = 1000;
    const rng = makeRng(9);
    // Scaled to ±8 to cover the sigmoid's tails as well as its linear part.
    const g = randF32(N, rng).map((v) => v * 8);
    const up = randF32(N, rng);

    const gb = uploadF32(device, g), ub = uploadF32(device, up);
    const ob = makeStorage(device, N * F32);
    submit(device, 'silu_mul', (pass) => siluMul(device, pass, gb, ub, ob, N));

    r.close('silu_mul', await readF32(device, ob, N * F32), siluMulCPU(g, up));
}

async function testSigmoidMul(device, r) {
    const N = 777;
    const rng = makeRng(13);
    const gate = randF32(N, rng).map((v) => v * 8);   // covers the sigmoid's tails
    const a = randF32(N, rng);

    const gb = uploadF32(device, gate), ab = uploadF32(device, a);
    const ob = makeStorage(device, N * F32);
    submit(device, 'sigmoid_mul', (pass) => sigmoidMul(device, pass, gb, ab, ob, N));

    r.close('sigmoid_mul', await readF32(device, ob, N * F32), sigmoidMulCPU(gate, a));
}

async function testCausalConv1d(device, r) {
    const T = 7, C = 5, K = 4;                        // awkward dims, none a multiple of 64
    const rng = makeRng(10);
    const x = randF32(T * C, rng);
    const weight = randF32(C * K, rng);

    const xb = uploadF32(device, x), wb = uploadF32(device, weight);
    const yb = makeStorage(device, T * C * F32);
    submit(device, 'causal_conv1d', (pass) => causalConv1d(device, pass, xb, wb, yb, T, C, K));

    r.close('causal_conv1d', await readF32(device, yb, T * C * F32),
        causalConv1dCPU(x, weight, T, C, K));

    // T_past > 0 indexes x by absolute position. A buffer holding earlier rows
    // plus T_new new ones must match the same window read directly.
    const T_past = 3, T_new = 4;
    const full = randF32((T_past + T_new) * C, rng);
    const fb = uploadF32(device, full);
    const yb2 = makeStorage(device, T_new * C * F32);
    submit(device, 'causal_conv1d T_past', (pass) =>
        causalConv1d(device, pass, fb, wb, yb2, T_new, C, K, T_past));
    r.close('causal_conv1d  T_past > 0 (absolute-position window)',
        await readF32(device, yb2, T_new * C * F32),
        causalConv1dCPU(full, weight, T_new, C, K, T_past));

    // hist: a persistent [K-1,C] ring holding rows an earlier job computed.
    // This is the usual decode case, where `x` holds only this job's rows and
    // T_past is 0.
    const Kh = K - 1;
    const hist = randF32(Kh * C, rng);
    const histb = uploadF32(device, hist);
    const T3 = 3;
    const x3 = randF32(T3 * C, rng);
    const xb3 = uploadF32(device, x3);
    const yb3 = makeStorage(device, T3 * C * F32);
    submit(device, 'causal_conv1d hist', (pass) =>
        causalConv1d(device, pass, xb3, wb, yb3, T3, C, K, 0, histb));
    r.close('causal_conv1d  hist (decode-across-jobs window)',
        await readF32(device, yb3, T3 * C * F32),
        causalConv1dHistCPU(x3, weight, hist, T3, C, K));
}

async function testUpdateConvHistory(device, r) {
    const C = 5, K = 4, Kh = K - 1;               // awkward dims, none a multiple of 64
    const rng = makeRng(14);
    const hist = randF32(Kh * C, rng);

    // T_new < K-1: the new ring mixes shifted old history with new rows.
    const T1 = 2;
    const new1 = randF32(T1 * C, rng);
    const histb1 = uploadF32(device, hist);
    const newb1 = uploadF32(device, new1);
    submit(device, 'update_conv_history T_new<Kh', (pass) =>
        updateConvHistory(device, pass, histb1, newb1, T1, C, K));
    r.close('update_conv_history  T_new < K-1 (shifted mix)',
        await readF32(device, histb1, Kh * C * F32),
        updateConvHistoryCPU(hist, new1, T1, C, K));

    // T_new >= K-1: the new ring is the tail of the new rows.
    const T2 = 5;
    const new2 = randF32(T2 * C, rng);
    const histb2 = uploadF32(device, hist);
    const newb2 = uploadF32(device, new2);
    submit(device, 'update_conv_history T_new>=Kh', (pass) =>
        updateConvHistory(device, pass, histb2, newb2, T2, C, K));
    r.close('update_conv_history  T_new >= K-1 (own tail)',
        await readF32(device, histb2, Kh * C * F32),
        updateConvHistoryCPU(hist, new2, T2, C, K));
}

async function testL2NormalizeHeads(device, r) {
    const T = 5, srcHeads = 3, D = 7;
    const rng = makeRng(11);
    const x = randF32(T * srcHeads * D, rng);

    const xb = uploadF32(device, x);
    const yb = makeStorage(device, T * srcHeads * D * F32);
    submit(device, 'l2norm_heads', (pass) => l2NormalizeHeads(device, pass, xb, yb, T, srcHeads, D));
    r.close('l2norm_heads', await readF32(device, yb, T * srcHeads * D * F32),
        l2NormalizeHeadsCPU(x, T, srcHeads, D));

    // group > 1 applies a GQA-style repeat_interleave in the same pass.
    const group = 2;
    const yb2 = makeStorage(device, T * srcHeads * group * D * F32);
    submit(device, 'l2norm_heads group', (pass) =>
        l2NormalizeHeads(device, pass, xb, yb2, T, srcHeads, D, group));
    r.close('l2norm_heads  group>1 (repeat_interleave)',
        await readF32(device, yb2, T * srcHeads * group * D * F32),
        l2NormalizeHeadsCPU(x, T, srcHeads, D, group));
}

// One GDN run for (T, heads, Dk, Dv) on random inputs. Returns the GPU output
// and final state from running the kernel two ways, plus the CPU reference:
// the whole chunk in one dispatch (as prefill does) and one dispatch per token
// (as decode does, which also checks that the state persists between
// dispatches).
async function runGdn(device, T, heads, Dk, Dv, seed, bRawOverride = null) {
    const rng = makeRng(seed);
    const q = randF32(T * heads * Dk, rng), k = randF32(T * heads * Dk, rng);
    const v = randF32(T * heads * Dv, rng);
    const aRaw = randF32(T * heads, rng);
    const bRaw = bRawOverride ?? randF32(T * heads, rng);
    const ALog = randF32(heads, rng).map((x) => x - 1);   // A_log is typically negative
    const dtBias = randF32(heads, rng);

    const qb = uploadF32(device, q), kb = uploadF32(device, k), vb = uploadF32(device, v);
    const ab = uploadF32(device, aRaw), bb = uploadF32(device, bRaw);
    const params = uploadF32(device, Float32Array.of(...ALog, ...dtBias));
    const sBytes = heads * Dk * Dv * F32, oBytes = T * heads * Dv * F32;

    const run = async (label, record) => {
        const S = makeStorage(device, sBytes);   // zeroed by makeStorage
        const out = makeStorage(device, oBytes);
        const enc = device.createCommandEncoder();
        withPass(enc, (pass) => record(pass, S, out), label);
        device.queue.submit([enc.finish()]);
        return { out: await readF32(device, out, oBytes), S: await readF32(device, S, sBytes) };
    };
    const batched = await run('gated_delta_net_step batched', (pass, S, out) =>
        gatedDeltaNetStep(device, pass, qb, kb, vb, ab, bb, params, S, out, 0, heads, Dk, Dv, T));
    const perToken = await run('gated_delta_net_step per-token', (pass, S, out) => {
        for (let t = 0; t < T; t++) {
            gatedDeltaNetStep(device, pass, qb, kb, vb, ab, bb, params, S, out, t, heads, Dk, Dv);
        }
    });
    const want = gatedDeltaNetCPU(q, k, v, aRaw, bRaw, ALog, dtBias, T, heads, Dk, Dv);
    return { batched, perToken, want };
}

async function testGatedDeltaNetStep(device, r) {
    // Dk=5 gives one live row per k-group, and Dv=7 leaves most of the column
    // tile out of range. This tests the out-of-range guards.
    {
        const T = 5, heads = 3, Dk = 5, Dv = 7;
        const { batched, perToken, want } = await runGdn(device, T, heads, Dk, Dv, 12);
        r.close('gated_delta_net_step  chunk in one dispatch', batched.out, want, 1e-4);
        r.close('gated_delta_net_step  one dispatch per token', perToken.out, want, 1e-4);
        // Both ways must leave the same state: the arithmetic is the same and
        // only the span held in registers differs.
        r.close('gated_delta_net_step  state agrees batched vs per-token', batched.S, perToken.S, 1e-6);
    }

    // Production shape: every thread owns a full 16-row strip, with four
    // column tiles per head and 48 heads. No lanes are out of range.
    {
        const T = 8, heads = 48, Dk = 128, Dv = 128;
        const { batched, perToken, want } = await runGdn(device, T, heads, Dk, Dv, 13);
        r.close('gated_delta_net_step  48×128×128 chunk', batched.out, want, 1e-4);
        r.close('gated_delta_net_step  48×128×128 per-token', perToken.out, want, 1e-4);
        r.close('gated_delta_net_step  48×128×128 state agrees', batched.S, perToken.S, 1e-6);
    }

    // With beta ≈ 0 (b_raw very negative) and a zero initial state, the
    // readout must be near zero. This does not use the reference, and catches
    // beta failing to gate delta, which the comparison above could miss on
    // random inputs.
    {
        const T = 5, heads = 3, Dk = 5, Dv = 7;
        const bZero = new Float32Array(T * heads).fill(-30);
        const { batched } = await runGdn(device, T, heads, Dk, Dv, 12, bZero);
        let maxOut = 0;
        for (const val of batched.out) maxOut = Math.max(maxOut, Math.abs(val));
        r.assert('gated_delta_net_step  beta≈0 keeps the readout near zero',
            maxOut < 1e-3, `max |out| ${maxOut.toExponential(2)}`);
    }
}

// --- Runner ---

export async function runAllTests(device) {
    const r = makeReporter();

    // Check the references before using them.
    checkReferences(r);

    // In dependency order, cheapest first.
    await testMatmulNobias(device, r);
    await testMatmulNobiasF16(device, r);
    await testMatmulNobiasGemv(device, r);
    await testMatmulRowsBitwise(device, r);
    await testGatherEmbed(device, r);
    await testRmsnorm(device, r);
    await testSiluMul(device, r);
    await testAddInplace(device, r);
    await testReshapeMergeHeads(device, r);
    await testScatterKV(device, r);
    await testRope(device, r);
    await testAttention(device, r);
    await testAttentionRowsBitwise(device, r);
    await testSigmoidMul(device, r);
    await testCausalConv1d(device, r);
    await testUpdateConvHistory(device, r);
    await testL2NormalizeHeads(device, r);
    await testGatedDeltaNetStep(device, r);

    const failed = r.rows.filter((x) => !x.ok);
    console.group(`kernel tests: ${r.rows.length - failed.length}/${r.rows.length} passed`);
    for (const row of r.rows) {
        console.log(`${row.ok ? 'PASS' : 'FAIL'}  ${row.name.padEnd(42)} ${row.detail}`);
    }
    console.groupEnd();

    if (failed.length) console.error(`${failed.length} kernel test(s) failed:`, failed.map((f) => f.name));
    return { ok: failed.length === 0, rows: r.rows };
}
