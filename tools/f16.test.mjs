// Tests web/f16.js: the half-precision rounding of the residual stream peers
// send each other (net.js's pack()).
//
//   node tools/f16.test.mjs

import { toF16, fromF16 } from '../web/f16.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};

console.log('f16: every half, both ways');
{
    const all = Uint16Array.from({ length: 65536 }, (_, i) => i);
    const v = fromF16(all), back = toF16(v);
    let bad = 0;
    for (let i = 0; i < 65536; i++) {
        if (((i >>> 10) & 0x1f) === 0x1f) continue;          // inf and NaN, below
        if (back[i] !== i) bad++;
    }
    ok('every finite half survives a round trip', bad === 0, `${bad} differ`);
    ok('the smallest subnormal is 2^-24', v[1] === 2 ** -24);
    ok('the largest finite half is 65504', v[0x7bff] === 65504);

    // Halfway between two neighbours goes to the one with an even mantissa,
    // for both signs and across the subnormal boundary.
    let ties = 0;
    for (const sign of [0, 0x8000]) {
        for (let i = 0; i < 0x7bff; i++) {
            const a = v[sign | i], b = v[sign | (i + 1)];
            const mid = Math.fround((a + b) / 2);
            if (mid === a || mid === b) continue;
            if (toF16([mid])[0] !== (sign | (i % 2 ? i + 1 : i))) ties++;
        }
    }
    ok('ties round to even', ties === 0, `${ties} wrong`);
}

console.log('\nf16: f32 values');
{
    let s = 1, worst = 0;
    const rand = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 32; };
    const x = Float32Array.from({ length: 100000 }, () => (rand() - 0.5) * 2 ** (rand() * 30 - 15));
    const y = fromF16(toF16(x));
    for (let i = 0; i < x.length; i++) {
        if (Math.abs(x[i]) >= 2 ** -14) worst = Math.max(worst, Math.abs(y[i] - x[i]) / Math.abs(x[i]));
    }
    ok('normal values are within half an f16 step', worst <= 2 ** -11, `worst ${worst}`);
    const big = fromF16(toF16([1e6, -1e6, Infinity, -Infinity, 65520]));
    ok('values past the range clamp to ±65504, not infinity',
        [...big].join() === '65504,-65504,65504,-65504,65504', [...big].join());
    ok('NaN stays NaN', Number.isNaN(fromF16(toF16([NaN]))[0]));
    ok('-0 keeps its sign', Object.is(fromF16(toF16([-0]))[0], -0));
    ok('a value below half the smallest subnormal is zero', fromF16(toF16([2 ** -26]))[0] === 0);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
