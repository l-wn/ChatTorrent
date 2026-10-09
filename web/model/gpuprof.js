// GPU timing for one job, using `timestamp-query`. Without that feature only
// CPU numbers are reported.
//
// The enqueuer picks a level and sends it in the job header as `prof`:
//
//   1  "pass"    one begin/end timestamp around the job's compute pass
//   2  "kernel"  one timed pass per dispatch, recorded by kernel name. The
//                extra passes add overhead, so use it for shares, not totals.
//
// Chrome quantizes timestamps to 100 µs unless
// chrome://flags/#enable-webgpu-developer-features is on. `quantized` in the
// result reports it.

const NS = 1e6;
const QUERIES = 4096;            // the WebGPU cap on one query set's count
const QUANTUM = 100_000n;        // Chrome's default timestamp granularity, in ns
const READ = () => globalThis.GPUMapMode.READ;

// device → the query sets and buffers a profiled job borrows. Jobs on one
// device do not interleave (Runner serializes them). A pool is used so that a
// job that threw between record and read only fails to return its chunks, and
// the next job gets clean ones.
const pools = new WeakMap();

function chunk(device) {
    let pool = pools.get(device);
    if (!pool) pools.set(device, pool = []);
    const U = globalThis.GPUBufferUsage;
    return pool.pop() ?? {
        set: device.createQuerySet({ type: 'timestamp', count: QUERIES }),
        resolve: device.createBuffer({ size: QUERIES * 8, usage: U.QUERY_RESOLVE | U.COPY_SRC }),
        map: device.createBuffer({ size: QUERIES * 8, usage: U.COPY_DST | U.MAP_READ }),
    };
}

// µs resolution: a small kernel summed over one peer's layers is well under a
// millisecond, and rounding to 0.01 ms would hide those rows.
const round = (x) => Math.round(x * 1000) / 1000;

export const hasTimestamps = (device) => !!device.features?.has?.('timestamp-query');

// Returns null for level 0. Otherwise one profiler for one job: pass() while
// recording, finish() before submit, map() in parallel with the readback's
// mapAsync, collect() once both have resolved.
export function makeGpuProf(device, level) {
    if (!(level >= 1)) return null;
    const ts = hasTimestamps(device);
    const now = () => performance.now();

    const chunks = [];            // borrowed, in query order
    let used = 0;                 // queries written so far
    let kind = '';                // the layer kind the next dispatch runs in
    const names = [];             // level 2: `${kind}/${name}` per dispatch
    const cpu = [];               // level 2: ms of CPU since the previous dispatch
    let last = 0;

    // Reserve two query indices, moving to a new query set every 4096. A hop
    // over 30 layers is a few hundred dispatches per job, but a peer holding
    // many layers can exceed one set on a T = 64 prefill.
    function writes() {
        const i = used;
        used += 2;
        const c = Math.floor(i / QUERIES);
        while (chunks.length <= c) chunks.push(chunk(device));
        return {
            querySet: chunks[c].set,
            beginningOfPassWriteIndex: i - c * QUERIES,
            endOfPassWriteIndex: i - c * QUERIES + 1,
        };
    }

    const self = {
        level,
        ts,

        // The layer kind of the following dispatches: 'embed', 'full',
        // 'linear' or 'head'.
        at(k) { kind = k; },

        // Called by dispatch() in kernels.js, before it records anything.
        mark(name) {
            const t = now();
            names.push(`${kind}/${name ?? '?'}`);
            cpu.push(t - last);
            last = t;
        },

        pass(encoder, label) {
            last = now();
            if (level < 2) {
                return encoder.beginComputePass(ts ? { label, timestampWrites: writes() }
                                                   : label ? { label } : undefined);
            }
            // No timestamps. The names and CPU split are still recorded, and
            // the pass is not split.
            if (!ts) {
                const real = encoder.beginComputePass(label ? { label } : undefined);
                return {
                    prof: self,
                    setPipeline: (p) => real.setPipeline(p),
                    setBindGroup: (i, g) => real.setBindGroup(i, g),
                    dispatchWorkgroups: (x, y, z) => real.dispatchWorkgroups(x, y, z),
                    end: () => real.end(),
                };
            }
            // One real pass per dispatch. dispatch() always calls setPipeline,
            // setBindGroup, dispatchWorkgroups in that order, so the first
            // opens a pass and the last closes it.
            let cur = null;
            return {
                prof: self,
                setPipeline(p) {
                    cur?.end();
                    cur = encoder.beginComputePass({ label, timestampWrites: writes() });
                    cur.setPipeline(p);
                },
                setBindGroup(i, g) { cur.setBindGroup(i, g); },
                dispatchWorkgroups(x, y, z) { cur.dispatchWorkgroups(x, y, z); cur.end(); cur = null; },
                end() { cur?.end(); cur = null; },
            };
        },

        // Recorded on the job's encoder, after the pass and before submit.
        finish(encoder) {
            for (let c = 0; c < chunks.length; c++) {
                const n = Math.min(QUERIES, used - c * QUERIES);
                encoder.resolveQuerySet(chunks[c].set, 0, n, chunks[c].resolve, 0);
                encoder.copyBufferToBuffer(chunks[c].resolve, 0, chunks[c].map, 0, n * 8);
            }
        },

        // Started in parallel with the readback's mapAsync, so profiling adds
        // no serial wait. Separate from collect() so the caller controls the
        // order the two mapped ranges are read in.
        map() {
            return Promise.all(chunks.map((c, i) =>
                c.map.mapAsync(READ(), 0, Math.min(QUERIES, used - i * QUERIES) * 8)));
        },

        collect() {
            const t = new BigInt64Array(used);
            for (let c = 0; c < chunks.length; c++) {
                const n = Math.min(QUERIES, used - c * QUERIES);
                t.set(new BigInt64Array(chunks[c].map.getMappedRange(0, n * 8).slice(0)), c * QUERIES);
                chunks[c].map.unmap();
            }
            const pool = pools.get(device);
            pool?.push(...chunks.splice(0));

            // A timestamp the implementation could not take reads as zero, and
            // an end before its begin is not a valid duration. Both count as
            // zero.
            const span = (a, b) => (t[b] > t[a] && t[a] > 0n ? t[b] - t[a] : 0n);
            let quantized = used > 0;
            let any = false;
            for (let i = 0; i + 1 < used; i += 2) {
                const d = span(i, i + 1);
                if (d === 0n) continue;
                any = true;
                if (d % QUANTUM !== 0n) quantized = false;
            }

            const out = { level, ts };
            if (ts && used) {
                // From the first pass's begin to the last pass's end: the
                // job's GPU wall time. At level 2 this includes the extra pass
                // boundaries, which the kernels' sum below does not.
                out.gpuMs = round(Number(span(0, used - 1)) / NS);
                out.quantized = any && quantized;
            }
            if (level >= 2) {
                const kernels = {};
                let kernelNs = 0n;
                for (let i = 0; i < names.length; i++) {
                    const k = kernels[names[i]] ??= { n: 0, gpuMs: ts ? 0 : null, cpuMs: 0 };
                    k.n++;
                    k.cpuMs += cpu[i];
                    if (ts && 2 * i + 1 < used) {
                        const d = span(2 * i, 2 * i + 1);
                        kernelNs += d;
                        k.gpuMs += Number(d) / NS;
                    }
                }
                for (const k of Object.values(kernels)) {
                    k.cpuMs = round(k.cpuMs);
                    if (k.gpuMs != null) k.gpuMs = round(k.gpuMs);
                }
                out.kernels = kernels;
                if (ts) out.kernelMs = round(Number(kernelNs) / NS);
                out.dispatches = names.length;
            }
            return out;
        },
    };
    return self;
}
