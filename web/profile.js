// Where a turn's time went, computed by the enqueuer from the traces its jobs
// return. Pure: no DOM, no GPU, no clock.
//
// The input is Engine.lastProfile.samples, one per job:
//
//   { kind: 'prefill' | 'decode', n, rt, overhead?, link?, trace: [hop, …] }
//
// A token's round trip is split as
//
//     rt = Σ hop.ms  +  Σ link.rtt / 2  +  residual
//
// Each link is counted once, outbound. The residual is the remainder: the
// answer's return path, and time a message waited before its handler ran. For
// a closed-loop sample (`loop: true`), `rt` is the time since the previous
// answer. Every number is a difference taken on one machine's clock.

// The phases a hop can report, in the order a job goes through them.
export const PHASES = ['queue', 'plan', 'encode', 'submit', 'wait', 'copy', 'sample', 'log', 'ui'];

// Nearest-rank percentile of an unsorted list. Null for an empty list.
export function pct(xs, p) {
    const v = xs.filter((x) => typeof x === 'number' && Number.isFinite(x)).sort((a, b) => a - b);
    if (!v.length) return null;
    return v[Math.min(v.length - 1, Math.max(0, Math.ceil((p / 100) * v.length) - 1))];
}

const stat = (xs) => {
    const med = pct(xs, 50);
    return med == null ? null : { med, p90: pct(xs, 90) };
};

export function summarize(profile) {
    const samples = profile?.samples ?? [];
    return {
        level: profile?.level ?? 0,
        decode: group(samples.filter((s) => s.kind === 'decode')),
        prefill: group(samples.filter((s) => s.kind === 'prefill')),
    };
}

function group(samples) {
    if (!samples.length) return null;

    // One row per (peer, entry, exit). A decode path is pinned, so every token
    // of a turn uses the same rows. A repair mid-turn adds new rows.
    const rows = new Map();
    const rt = [], overhead = [], transit = [], residual = [];

    for (const s of samples) {
        let local = 0, net = s.link?.rtt != null ? s.link.rtt / 2 : 0;
        for (const hop of s.trace ?? []) {
            const key = `${hop.id}|${hop.from ?? '?'}|${hop.to ?? '?'}`;
            let row = rows.get(key);
            if (!row) {
                rows.set(key, row = { id: hop.id, from: hop.from, to: hop.to, label: null,
                                      jobs: 0, raw: {}, kernels: {}, profiled: 0,
                                      ts: null, quantized: null });
            }
            row.jobs++;
            if (hop.label) row.label = hop.label;
            const put = (name, v) => { if (typeof v === 'number') (row.raw[name] ??= []).push(v); };
            put('ms', hop.ms);
            for (const p of PHASES) put(p, hop[p]);
            put('dispatches', hop.dispatches);
            put('bindGroups', hop.bindGroups);
            // Jobs in the pass that ran this one (stage.js batching).
            put('batch', hop.batch ?? 1);
            put('rtt', hop.link?.rtt);
            put('send', hop.link?.send);
            if (hop.gpu) {
                row.ts = hop.gpu.ts;
                put('gpu', hop.gpu.gpuMs);
                put('kernelMs', hop.gpu.kernelMs);
                if (hop.gpu.quantized != null) row.quantized = (row.quantized ?? true) && hop.gpu.quantized;
                if (hop.gpu.kernels) {
                    row.profiled++;
                    for (const [name, k] of Object.entries(hop.gpu.kernels)) {
                        const acc = row.kernels[name] ??= { n: 0, gpuMs: k.gpuMs == null ? null : 0, cpuMs: 0 };
                        acc.n += k.n;
                        acc.cpuMs += k.cpuMs;
                        if (k.gpuMs != null) acc.gpuMs = (acc.gpuMs ?? 0) + k.gpuMs;
                    }
                }
            }
            local += hop.ms ?? 0;
            if (hop.link?.rtt != null) net += hop.link.rtt / 2;
        }
        rt.push(s.rt);
        transit.push(net);
        if (typeof s.rt === 'number') residual.push(s.rt - local - net);
        if (typeof s.overhead === 'number') overhead.push(s.overhead);
    }

    const hops = [...rows.values()].sort((a, b) => (a.from ?? 0) - (b.from ?? 0)).map((row) => {
        const out = { id: row.id, label: row.label, from: row.from, to: row.to, jobs: row.jobs,
                      ts: row.ts, quantized: row.quantized };
        for (const [name, xs] of Object.entries(row.raw)) out[name] = stat(xs);
        // Averaged per job, not summed over the turn, so a kernel row can be
        // compared with the hop's `ms`.
        if (row.profiled) {
            const per = row.profiled;
            const total = Object.values(row.kernels).reduce((a, k) => a + (k.gpuMs ?? 0), 0);
            out.kernels = Object.entries(row.kernels).map(([name, k]) => ({
                name,
                n: k.n / per,
                gpuMs: k.gpuMs == null ? null : k.gpuMs / per,
                cpuMs: k.cpuMs / per,
                share: total > 0 && k.gpuMs != null ? k.gpuMs / total : null,
            })).sort((a, b) => ((b.gpuMs ?? b.cpuMs) - (a.gpuMs ?? a.cpuMs)));
        }
        return out;
    });

    return {
        loop: samples.length > 0 && samples.every((s) => s.loop),
        jobs: samples.length,
        tokens: samples.reduce((a, s) => a + (s.n ?? 0), 0),
        rt: stat(rt),
        overhead: stat(overhead),
        transit: stat(transit),
        residual: stat(residual),
        hops,
    };
}

// --- Text ---

const f = (x) => x == null ? '–' : Math.abs(x) < 10 ? x.toFixed(1) : x.toFixed(0);
const med = (s) => f(s?.med);

// One line for the reply's meta: the median token, hop by hop.
export function formatLine(summary) {
    const g = summary?.decode ?? summary?.prefill;
    if (!g?.hops.length) return '';
    const hops = g.hops.map((h) => `peer${h.id} ${med(h.ms)}ms`).join(' → ');
    return `hops: ${hops} · links ${med(g.transit)}ms · rest ${med(g.residual)}ms`;
}

// The full breakdown: medians, with p90 for the main figures. Monospace, for a
// <pre> or the console.
export function formatText(summary, { top = 12 } = {}) {
    const out = [];
    const levels = ['off', 'pass', 'kernel'];
    out.push(`profile: ${levels[summary.level] ?? summary.level}`
        + (summary.level >= 2 ? '  (one pass per dispatch, so slower than a real token; read shares, not totals)' : ''));

    for (const [kind, g] of [['decode', summary.decode], ['prefill', summary.prefill]]) {
        if (!g) continue;
        out.push('');
        out.push(`${kind}${g.loop && kind === 'decode' ? ' (closed loop)' : ''} · `
            + `${g.jobs} job${g.jobs === 1 ? '' : 's'}, ${g.tokens} tok · `
            + `${g.loop && kind === 'decode' ? 'interval' : 'rt'} ${med(g.rt)} ms/job (p90 ${f(g.rt?.p90)})`
            + (g.overhead ? ` · enqueuer between jobs ${med(g.overhead)}` : ''));
        for (const h of g.hops) {
            const name = `peer${h.id}${h.label ? ` ${h.label}` : ''}`;
            const phases = PHASES.filter((p) => h[p] && h[p].med >= 0.05)
                .map((p) => `${p} ${med(h[p])}`).join(' · ');
            const gpu = h.gpu ? ` [gpu ${med(h.gpu)}${h.kernelMs ? `, kernels ${med(h.kernelMs)}` : ''}]`
                : summary.level >= 1 && h.ts === false ? ' [gpu n/a: no timestamp-query]' : '';
            out.push(`  ${name}  pos ${h.from ?? '?'}–${h.to != null ? h.to - 1 : '?'}  `
                + `ms ${med(h.ms)} (p90 ${f(h.ms?.p90)}) = ${phases}${gpu}`);
            const counts = [h.dispatches && `dispatches ${med(h.dispatches)}`,
                            h.bindGroups && `bind groups built ${med(h.bindGroups)}`,
                            h.batch?.p90 > 1 && `jobs per pass ${med(h.batch)} (p90 ${f(h.batch.p90)})`,
                            h.quantized && 'timestamps quantized to 100µs'].filter(Boolean);
            if (counts.length) out.push(`      ${counts.join(' · ')}`);
            if (h.rtt || h.send) {
                out.push(`      link out: rtt ${med(h.rtt)} (p90 ${f(h.rtt?.p90)}) · send ${med(h.send)}`);
            }
            if (h.kernels) {
                out.push(`      ${'kernel'.padEnd(40)} ${'gpu ms'.padStart(8)} ${'share'.padStart(6)} `
                    + `${'cpu ms'.padStart(8)} ${'n'.padStart(5)}`);
                for (const k of h.kernels.slice(0, top)) {
                    out.push(`      ${k.name.padEnd(40)} ${(k.gpuMs == null ? '–' : k.gpuMs.toFixed(2)).padStart(8)} `
                        + `${(k.share == null ? '–' : `${(k.share * 100).toFixed(0)}%`).padStart(6)} `
                        + `${k.cpuMs.toFixed(2).padStart(8)} ${String(Math.round(k.n)).padStart(5)}`);
                }
                if (h.kernels.length > top) out.push(`      … ${h.kernels.length - top} more in the JSON`);
            }
        }
        out.push(`  links, outbound (Σ rtt/2)  ${med(g.transit)}`);
        out.push(`  ${g.loop && kind === 'decode' ? 'unaccounted              ' : 'return path + unaccounted'}  `
            + `${med(g.residual)} (p90 ${f(g.residual?.p90)})`);
    }
    return out.join('\n');
}
