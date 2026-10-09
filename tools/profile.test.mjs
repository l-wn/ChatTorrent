// Tests web/profile.js: the arithmetic behind the timing breakdown under a
// reply.
//
// The traces are tested where they are produced (stage.test.mjs for a Runner's
// phases and GPU profile, mesh.test.mjs for a trace crossing the wire). This
// file tests what the enqueuer computes from them: a token's round trip is
// split into hops, links and a residual that is the exact remainder, and
// per-kernel rows are averaged per job, not summed per turn.
//
//   node tools/profile.test.mjs

import { summarize, formatText, formatLine, pct } from '../web/profile.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `\n        got  ${JSON.stringify(got)}\n        want ${JSON.stringify(want)}`);
const near = (a, b) => Math.abs(a - b) < 1e-9;

console.log('\npercentiles');
{
    eq('median of an odd list', pct([5, 1, 3], 50), 3);
    eq('nearest rank, not interpolated', pct([1, 2, 3, 4], 50), 2);
    eq('p90 of ten', pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90), 9);
    eq('ignores what is not a number', pct([null, undefined, NaN, 4], 50), 4);
    eq('null for nothing', pct([], 50), null);
}

// Two peers: 0 holds 0–3 and hands on, 1 holds 3–N and samples. The enqueuer
// is a spectator, so its own hand-off at position 0 is a link too.
const hop0 = (ms, rtt, extra = {}) => ({
    id: 0, from: 0, to: 3, ms, queue: 0, plan: 0.1, encode: 2, submit: 0.1,
    wait: ms - 3, copy: 0.3, log: 0.2, ui: 0.3, dispatches: 40, bindGroups: 6,
    ...(rtt != null && { link: { rtt, send: 0.5 } }), ...extra,
});
const hop1 = (ms, extra = {}) => ({
    id: 1, from: 3, to: 8, ms, queue: 1, plan: 0, encode: 3, submit: 0.1,
    wait: ms - 6, copy: 0.1, sample: 1.5, ui: 0.3, dispatches: 60, bindGroups: 0, ...extra,
});

console.log('\naccounting: a round trip is hops, links and a named residual');
{
    const samples = [
        { kind: 'prefill', n: 7, rt: 500, link: { rtt: 20 }, trace: [hop0(200, null), hop1(250)] },
        { kind: 'decode', n: 1, rt: 100, overhead: 4, link: { rtt: 10 }, trace: [hop0(40, null), hop1(30)] },
        { kind: 'decode', n: 1, rt: 110, overhead: 6, link: { rtt: 12 }, trace: [hop0(42, 8), hop1(31)] },
        { kind: 'decode', n: 1, rt: 120, link: { rtt: 14 }, trace: [hop0(44, 10), hop1(32)] },
    ];
    const s = summarize({ level: 0, samples });

    eq('decode and prefill are kept apart', [s.decode.jobs, s.prefill.jobs, s.prefill.tokens], [3, 1, 7]);
    // Token 1: 100 - (40 + 30) - 10/2 = 25. Token 2: 110 - 73 - (12 + 8)/2 = 27.
    // Token 3: 120 - 76 - (14 + 10)/2 = 32.
    eq('the residual is exactly what hops and links do not explain',
        [s.decode.residual.med, s.decode.residual.p90], [27, 32]);
    eq('links count once, outbound, as half their round trip', s.decode.transit.med, 10);
    eq('enqueuer overhead only from the jobs that had one', s.decode.overhead.med, 4);
    eq('one row per peer, in path order', s.decode.hops.map((h) => [h.id, h.from, h.to, h.jobs]),
        [[0, 0, 3, 3], [1, 3, 8, 3]]);
    eq('each phase summarised', [s.decode.hops[0].ms.med, s.decode.hops[1].sample.med,
                                  s.decode.hops[0].dispatches.med], [42, 1.5, 40]);
    eq('the outgoing link only where one was timed', [s.decode.hops[0].rtt.med, s.decode.hops[1].rtt],
        [8, undefined]);
    ok('the one-liner names every hop', /peer0 42ms → peer1 31ms/.test(formatLine(s)), formatLine(s));
}

console.log('\nkernels: merged per peer, per job');
{
    const gpu = (a, b, quantized) => ({
        level: 2, ts: true, gpuMs: a + b + 1, kernelMs: a + b, quantized,
        kernels: { 'full/matmul q8 5120→17408': { n: 2, gpuMs: a, cpuMs: 0.2 },
                   'full/rmsnorm 5120':         { n: 4, gpuMs: b, cpuMs: 0.1 } },
    });
    const samples = [
        { kind: 'decode', n: 1, rt: 50, trace: [hop0(20, null, { gpu: gpu(6, 2, true) })] },
        { kind: 'decode', n: 1, rt: 50, trace: [hop0(20, null, { gpu: gpu(8, 2, false), label: 'TITAN RTX' })] },
    ];
    const h = summarize({ level: 2, samples }).decode.hops[0];

    eq('sorted by GPU time', h.kernels.map((k) => k.name), ['full/matmul q8 5120→17408', 'full/rmsnorm 5120']);
    ok('averaged per job, not summed over the turn',
        near(h.kernels[0].gpuMs, 7) && near(h.kernels[0].n, 2) && near(h.kernels[1].cpuMs, 0.1));
    ok('shares add to one', near(h.kernels.reduce((a, k) => a + k.share, 0), 1));
    eq('quantized only if every job was', h.quantized, false);
    eq('the adapter name sticks to the row', h.label, 'TITAN RTX');
    eq('pass and kernel totals summarised', [h.gpu.med, h.kernelMs.med], [9, 8]);

    const noTs = summarize({ level: 2, samples: [{ kind: 'decode', n: 1, rt: 10, trace: [hop0(5, null, {
        gpu: { level: 2, ts: false, kernels: { 'full/add_inplace': { n: 3, gpuMs: null, cpuMs: 0.4 } } },
    })] }] });
    const k = noTs.decode.hops[0].kernels[0];
    ok('without timestamps a kernel keeps its CPU figure and has no share',
        k.gpuMs === null && k.share === null && near(k.cpuMs, 0.4));
    ok('and the text says why there is no GPU number',
        /no timestamp-query/.test(formatText(noTs)), formatText(noTs));
}

console.log('\ntext');
{
    const samples = [{ kind: 'decode', n: 1, rt: 80, overhead: 3, link: { rtt: 6 },
                       trace: [hop0(30, 4, { label: 'TITAN RTX',
                                             gpu: { level: 1, ts: true, gpuMs: 25, quantized: true } }),
                               hop1(40)] }];
    const text = formatText(summarize({ level: 1, samples }));
    ok('names the level', /^profile: pass/.test(text));
    ok('one line per hop with its GPU time', /peer0 TITAN RTX\s+pos 0–2\s+ms 30 .*\[gpu 25\]/.test(text), text);
    ok('flags quantized timestamps', /quantized to 100µs/.test(text));
    ok('shows the outgoing link', /link out: rtt 4\.0/.test(text));
    ok('and the residual as its own row', /return path \+ unaccounted\s+5\.0/.test(text), text);
    eq('an empty profile summarises to nothing', summarize(null), { level: 0, decode: null, prefill: null });
    eq('and has no one-liner', formatLine(summarize(null)), '');
}

console.log('\na closed-loop reply');
{
    const hop = (id, ms, from, to, rtt) => ({ id, ms, from, to, ...(rtt != null && { link: { rtt } }) });
    const samples = [
        { kind: 'prefill', n: 7, rt: 300, loop: true, trace: [hop(0, 90, 0, 4), hop(1, 80, 4, 10)] },
        { kind: 'decode', n: 1, rt: 60, loop: true, link: { rtt: 8 }, trace: [hop(0, 20, 0, 4, 6), hop(1, 30, 4, 10)] },
        { kind: 'decode', n: 1, rt: 62, loop: true, link: { rtt: 8 }, trace: [hop(0, 21, 0, 4, 6), hop(1, 31, 4, 10)] },
    ];
    const s = summarize({ level: 0, samples });
    ok('is recognised as one', s.decode.loop === true);
    eq('and its interval accounted the same way: hops, the head\'s hand-off, the hop between',
        s.decode.transit.med, 7);
    const text = formatText(s);
    ok('labelled as a closed loop, with an interval rather than a round trip',
        /decode \(closed loop\) · 2 jobs, 2 tok · interval 60/.test(text), `\n${text}`);
    ok('and no return path claimed', !/decode \(closed loop\)[\s\S]*return path/.test(text.split('prefill')[0]));
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
