// IEEE half precision for the residual stream a peer hands on (see net.js's
// pack()). The rows are rounded once, where they leave a Runner, and every
// later reader (the next peer, the boundary log, a replay) sees those bits.
//
// No WebGPU and no Float16Array, so node tests and every browser round the
// same way.

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

// The largest finite half. Larger values are clamped to it, not sent as
// infinity, so one outlier cannot poison the next peer's arithmetic.
const MAX = 0x7bff;

// One f32 to its half bits, rounding to nearest, ties to even.
function half(x) {
    f32[0] = x;
    const b = u32[0];
    const sign = (b >>> 16) & 0x8000;
    const exp = (b >>> 23) & 0xff;
    const man = b & 0x7fffff;
    if (exp === 0xff) return sign | (man ? 0x7e00 : MAX);    // NaN stays NaN; ±inf clamps
    const e = exp - 127 + 15;
    if (e >= 0x1f) return sign | MAX;
    if (e <= 0) {
        // Subnormal or zero: shift the implicit bit in, then round.
        if (e < -10) return sign;
        const m = man | 0x800000;
        const shift = 14 - e;
        let h = m >>> shift;
        const rest = m & ((1 << shift) - 1), mid = 1 << (shift - 1);
        if (rest > mid || (rest === mid && (h & 1))) h++;
        return sign | h;
    }
    let h = (e << 10) | (man >>> 13);
    const rest = man & 0x1fff;
    if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h++;  // a carry into the exponent is right
    return sign | Math.min(h, MAX);
}

// f32 values → half bits.
export function toF16(src) {
    const out = new Uint16Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = half(src[i]);
    return out;
}

// Every half's value as f32, built once.
let table = null;
function buildTable() {
    table = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
        const sign = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
        table[h] = e === 0 ? sign * m * 2 ** -24
            : e === 0x1f ? (m ? NaN : sign * Infinity)
            : sign * (1 + m / 1024) * 2 ** (e - 15);
    }
}

// Half bits → f32 values, exactly.
export function fromF16(bits) {
    if (!table) buildTable();
    const out = new Float32Array(bits.length);
    for (let i = 0; i < bits.length; i++) out[i] = table[bits[i]];
    return out;
}
