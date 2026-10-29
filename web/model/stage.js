// One peer's range of the model.
//
// A Runner owns the positions it loaded (`[lo, hi)` of the model's `L + 2`), a
// KV page pool with per-session frames, and one set of scratch buffers. The
// scratch is shared, so jobs run one at a time and run() queues.
//
//     0            1 .. L              L+1
//     embedding    blocks 0 .. L-1     final norm + LM head + sample
//
// A job enters at its cursor, anywhere in the range, and runs to the session's
// cut (#exit()). The Runner also keeps the residual stream it hands downstream
// (the boundary log), so route.js can replay it into another holder when the
// next peer leaves.
//
// See docs/architecture.md (KV cache and admission, Routing, Failures).

import { F32, makeStorage, makeReadback, readF32, gatherEmbedNoPos, rmsnorm, matmulNobias,
         withPass, counters, copyF32, topkRows, topkPairs, roundF16, GEMV_ROWS_MAX } from './kernels.js';
import { blockSegs, makeSegScratch, makeCache, makeScratch, makeRecurrentState, recurrentStateBytes,
         makePagePool, makeSessionFrame, ensurePages, freePages, makeReplayLayer,
         replayLayerBytes, mtpBlock, mtpChain, makeMtpScratch } from './model.js';
import { makeGpuProf } from './gpuprof.js';
import { sampleFromLogitsRow } from './generate.js';
import { sampleFromCandidates, candidatesFor, TOPK_MAX } from './sampling.js';
import { toF16, fromF16 } from '../f16.js';
import { positions, blockRange, kvRowBytes, PAGE_TOKENS, IDLE_MS, WAIT_MS, SPEC_COLS } from '../split.js';

// The session id a warm-up pass runs under. Peer-minted ids are
// `${peerId}-${base36}`, so it cannot collide with a chat.
const WARMUP = '\0warmup';

// Initial row capacity of a boundary log: four prefill chunks.
const LOG_CHUNK = 256;

// Sessions run() holds jobs queued for before it refuses another with `busy`,
// and jobs it holds for one session. A session has one decode job in flight,
// or a prompt's prefill chunks (route.js PREFILL_WINDOW) and its last chunk.
// So the first is several times `sessions` and the second a few more than a
// prompt sends. Together they bound the GPU time and payload memory a flood
// can take.
export const MAX_QUEUE = 32;
export const MAX_SESSION_QUEUE = 8;

// How long a capacity refusal is remembered for its session (see #cold()).
const REFUSED_MS = 60_000;

// How recently a session must have run here to count towards `load`. Covers
// the gap between a reply's tokens and a short pause between turns.
export const ACTIVE_MS = 5000;

// Keep-warm: while jobs are arriving, the GPU runs a small kernel whenever
// this Runner's queue is empty, so the driver does not drop the clocks
// between tokens. It stops KEEP_WARM_MS after the last job. WARM_DEPTH rounds
// are kept in flight, because a round's completion is reported about 2 ms
// after it ends: with one, the GPU is busy under 20% of the time and an
// RTX 3090 stays at P5. Three hold it at P0. A round of WARM_ITERS takes about
// 0.5 ms there, so a new job waits at most about 1.5 ms behind them.
export const KEEP_WARM_MS = 3000;
const WARM_DEPTH = 3;
const WARM_ITERS = 16384;

// One workgroup running a dependent fma chain: the GPU counts as busy while
// it runs, and it uses one SM and no memory bandwidth. b = [x0, a, c, out].
const WARM_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> b: array<f32, 4>;
@group(0) @binding(1) var<uniform> n: u32;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) {
    var x = b[0] + f32(i);
    let a = b[1];
    let c = b[2];
    for (var k = 0u; k < n; k++) { x = fma(x, a, c); }
    if (i == 0u) { b[3] = x; }
}`;

function makeWarmKernel(device) {
    const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array([0, 0.5, 0.25, 0]));
    const iters = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(iters, 0, new Uint32Array([WARM_ITERS]));
    const pipe = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: WARM_WGSL }), entryPoint: 'main' },
    });
    const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: { buffer: iters } },
    ] });
    const k = { pipe, bind, bytes: 32, lost: false };
    // On a lost device a round completes at once, so the lanes stop.
    device.lost?.then(() => { k.lost = true; });
    return k;
}

// A replay layer's buffers (model.js makeReplayLayer).
const SLOT_BUFS = ['S', 'histQ', 'histK', 'histV', 'linQ', 'linK', 'linV', 'linA', 'linB'];

// Thrown by #cacheFor for a job that must wait for a frame. The queue catches
// it and parks the job. Never leaves this file.
const WAIT = new Error('waiting for a session frame');

export class Runner {
    // `lo`/`hi` are the half-open position range this peer was assigned.
    // `w.layers` must hold the blocks inside it, indexed from 0. `have` is how
    // much of `[lo, hi)` is downloaded: `lo` for a Runner filled one position
    // at a time by append(), `hi` (the default) for one given a whole range.
    constructor(device, d, w, { lo, hi, have = hi, nMax, tMax, slots = 5,
                                sessions = slots, pageTokens = PAGE_TOKENS,
                                idleMs = IDLE_MS, waitMs = WAIT_MS, specSlots = slots,
                                keepWarmMs = 0, batchRows = GEMV_ROWS_MAX }) {
        const [blockLo, blockHi] = blockRange(lo, hi, d.n_layer);
        const held = blockRange(lo, have, d.n_layer)[1] - blockLo;
        if (held !== w.layers.length) {
            throw new Error(`positions ${lo}–${have - 1} contain ${held} blocks `
                + `but ${w.layers.length} loaded`);
        }
        this.device = device;
        this.d = d;
        this.w = w;
        this.lo = lo;
        this.hi = hi;
        // The end of what is downloaded and warmed up; `hi` is the end of the
        // claim. append() moves it. holds() checks `have`, not `hi`.
        this.have = have;
        // The model index of `w.layers[0]`. Converts a cursor to an index into
        // the loaded blocks and their KV rows.
        this.blockLo = blockLo;
        this.nMax = nMax;
        this.tMax = tMax;
        this.scratch = makeScratch(device, d, tMax, nMax);
        // Batching (see #batchWith): the most rows one pass carries for
        // several jobs. 1 turns it off. Capped where the projections stop
        // giving each row a one-row call's bits (kernels.js GEMV_ROWS_MAX).
        this.batchRows = Math.max(1, Math.min(batchRows, GEMV_ROWS_MAX, tMax));
        // Rows the head may sample in one pass: a verify job's, or a batch's.
        this.sampleRows = Math.max(SPEC_COLS, this.batchRows);
        this.seg = this.batchRows > 1 ? makeSegScratch(device, d, this.batchRows, this.scratch) : null;
        this.ids = makeStorage(device, tMax * F32);          // token ids, for the embedding
        // Bytes of weights uploaded for this Runner, as measured by loader.js
        // (`weights.bytes`). Used by vramBytes.
        this.weightBytes = w.layers.reduce((n, l) => n + (l.bytes ?? 0), 0) + (w.bytes ?? 0);
        // KV is allocated for the whole claim, so the VRAM is committed up
        // front and an appended block has a cache to index.
        this.blocks = blockHi - blockLo;

        // The KV pool holds `slots` full contexts' worth of pages, and
        // `sessions` frames draw from it. A chat uses pages for the tokens it
        // holds, not for `nMax`, so the pool fits more than `slots` short
        // chats. A page is at most the whole window.
        this.pageTokens = Math.min(pageTokens, nMax);
        // Pages per session. The pool has `slots` times that.
        this.maxPages = Math.ceil(nMax / this.pageTokens);
        // A layer's K pool buffer is bound whole, so it must fit the device's
        // binding limit.
        const poolBuffer = kvRowBytes(d) * this.pageTokens * slots * this.maxPages;
        const bindMax = device.limits?.maxStorageBufferBindingSize;
        if (bindMax && poolBuffer > bindMax) {
            throw new Error(`this GPU can bind at most ${(bindMax / 2 ** 20).toFixed(0)} MiB, and a ` +
                `${nMax}-token context for ${sessions} chats needs a ${(poolBuffer / 2 ** 20).toFixed(0)} ` +
                `MiB KV buffer per layer. A cluster with a shorter context or fewer chats in memory fits.`);
        }
        this.pool = makePagePool(device, d, {
            pageTokens: this.pageTokens,
            nPages: slots * this.maxPages,
            n_layer: this.blocks,
            layerLo: this.blockLo,
        });
        this.slots = slots;
        this.sessions = sessions;
        this.free = Array.from({ length: sessions },
            () => makeSessionFrame(device, d, this.pool, this.maxPages,
                                   this.blocks, this.blockLo));
        this.live = new Map();                               // session -> frame, LRU order
        // session -> (entry cursor -> exit position): the cut, fixed when the
        // session first runs here. See #exit().
        this.cuts = new Map();
        // session -> (exit cursor -> the residual stream handed over there):
        // the boundary log, replayed into another holder when the peer it was
        // handed to leaves. route.js keys that boundary by the same exit
        // cursor. See #log().
        this.logs = new Map();
        // Work waiting to run, one task at a time: jobs, and the functions
        // schedule() adds (releases, the waiting list's barrier). #next()
        // picks the task.
        this.tasks = [];
        this.draining = false;
        // session -> when a job of it was last refused for capacity here.
        this.refused = new Map();
        this.backlog = 0;                                   // jobs queued or running
        // Jobs run() has accepted and not yet settled, by session. A session
        // with any is not idle, however long its job waits in the queue.
        this.pending = new Map();
        // Milliseconds per position for one token, averaged. Seeded from the
        // warm-up passes, then measured on real decode tokens. Null until
        // either has run. route.js compares peers' speeds with it.
        this.msPerPos = null;
        this.warmups = 0;           // warm-up passes run; the first compiles shaders
        this.decoded = false;       // whether msPerPos comes from decode tokens
        // The part of a decode job that does not scale with positions:
        // reading back, sampling at the head, logging the hand-off. Averaged,
        // null until the first decode. With msPerPos it gives a job's cost.
        this.overheadMs = null;
        // How long jobs have waited in this runner's queue lately, averaged.
        this.queueAvgMs = 0;
        // When the last non-warm-up job ran. 0 until the first one.
        this.lastJobAt = 0;

        // How long a session must be idle before it can be evicted, counted
        // from the end of its last job (0: evictable as soon as none of its
        // jobs is queued or running). See split.js's IDLE_MS.
        this.idleMs = idleMs;
        // How long a new session waits for a frame before it is refused (0:
        // refused immediately). See split.js's WAIT_MS.
        this.waitMs = waitMs;
        // Jobs of sessions waiting for a frame, in arrival order. See #park().
        this.waiting = [];
        // The waiting session whose jobs are back on the queue to be admitted.
        this.turn = null;
        this.counts = { admitted: 0, evicted: 0, cold: 0, released: 0, refused: 0, waited: 0,
                        nospec: 0, batched: 0, batchedJobs: 0 };

        // Speculative decoding's replay slots (see #specSlot()): at most
        // `specSlots` sessions speculate through this Runner's linear_attention
        // blocks at once. A slot is made on first use and kept for reuse.
        // split.js's blockBytes() reserves room for all of them.
        this.specSlots = specSlots;
        this.specFree = [];
        this.specLive = new Map();                           // session -> slot
        this.specReadback = null;                            // the head's verify rows
        this.topk = null;                                    // GPU sampling's buffers

        // Keep-warm (see KEEP_WARM_MS). 0 turns it off.
        this.keepWarmMs = keepWarmMs;
        this.warmKernel = null;
        this.warming = false;
        this.activeAt = 0;                                   // when run() last took or settled a job
        this.onIdle = [];                                    // resolved when the queue drains
    }

    // `embed` needs `w.wte`. `head` needs `w.norm` and `w.lm_head` (or `w.wte`
    // on a tied checkpoint). Getters, because growth moves `lo`/`hi`.
    get embed() { return this.lo === 0; }
    get head() { return this.hi === positions(this.d.n_layer); }

    // Bytes this Runner holds on the GPU: weights, the KV page pool, every
    // session frame, the warm-up cache and the scratch. Summed from actual
    // allocations, so it can be checked against a VRAM limit.
    get vramBytes() {
        let bytes = this.weightBytes + this.pool.bytes + this.scratch.bytes
            + Math.max(this.tMax * F32, 16) + (this.seg?.bytes ?? 0);
        for (const frame of this.free) bytes += frame.bytes;
        for (const frame of this.live.values()) bytes += frame.bytes;
        if (this.warm) bytes += this.warm.bytes;
        for (const slot of [...this.specFree, ...this.specLive.values()]) bytes += slot.bytes;
        if (this.specReadback) bytes += this.specReadback.size;
        if (this.topk) bytes += this.topk.bytes;
        if (this.warmKernel) bytes += this.warmKernel.bytes;
        return bytes + (this.mtpBytes ?? 0);
    }

    // --- The MTP draft head ---
    //
    // A Runner holding the head can also hold the MTP head (loader.js
    // loadMtp()). For a job flagged `mtp` that reaches the head, the layer's
    // KV is filled from the job's rows (each row's token with the trunk's
    // hidden state one position earlier), in the same submit. After sampling,
    // a second submit keeps the hidden state the sample came from (per session,
    // `frame.mtpH`) and, for a loop job with speculation on, chains up to
    // SPEC_COLS - 1 drafts after the sampled token. They come back as `mtp`
    // and route.js may send them as the next verify job's drafts.
    //
    // The layer's K/V buffers are laid out as a pool layer, so each session's
    // page table addresses them too.
    attachMtp(m) {
        const { device, d } = this;
        const kv = this.#poolBufferBytes();
        this.mtp = m;
        this.mtpK = makeStorage(device, kv);
        this.mtpV = makeStorage(device, kv);
        this.mtpScratch = makeMtpScratch(device, d, m.draftVocab, this.tMax, this.nMax);
        this.mtpReadback = makeReadback(device, SPEC_COLS * F32);
        this.mtpBatchIds = makeStorage(device, this.sampleRows * F32);    // a batch's ids, by row
        const row = d.hidden * F32;
        const frames = [...this.free, ...this.live.values()];
        for (const frame of frames) frame.mtpH = makeStorage(device, row);
        this.weightBytes += m.bytes ?? 0;
        this.mtpBytes = 2 * kv + this.mtpScratch.bytes + (SPEC_COLS + this.sampleRows) * F32
            + frames.length * row;
    }

    #mtpCache(frame) {
        return { K: [this.mtpK], V: [this.mtpV], pages: frame.pages, pageTokens: frame.pageTokens,
                 N_max: frame.N_max };
    }

    // After sampling `t` from row `a` of a job: keep that row's hidden state
    // and, when asked, chain drafts after `t`. Returns the drafts, or null.
    // `rowOff` is the job's first row in the pass that ran it.
    async #mtpAfter(h, frame, a, t, rowOff = 0) {
        const { device, d } = this;
        const ms = this.mtpScratch, H = d.hidden;
        const pos = h.past + a + 1;
        const K = h.loop?.spec && t < this.mtp.draftVocab
            ? Math.max(0, Math.min(SPEC_COLS - 1, this.nMax - pos - 1)) : 0;
        device.queue.writeBuffer(ms.mtpIds, 0, new Float32Array([t]));
        const enc = device.createCommandEncoder({ label: 'mtp draft' });
        withPass(enc, (pass) => {
            copyF32(device, pass, this.scratch.x, frame.mtpH, H, (rowOff + a) * H, 0);
            if (K > 0) mtpChain(device, pass, d, this.mtp, this.#mtpCache(frame), ms, frame.mtpH,
                                ms.mtpIds, pos, K);
        }, 'mtp draft');
        if (K === 0) { device.queue.submit([enc.finish()]); return null; }
        enc.copyBufferToBuffer(ms.mtpIds, F32, this.mtpReadback, 0, K * F32);
        device.queue.submit([enc.finish()]);
        await this.mtpReadback.mapAsync(GPUMapMode.READ, 0, K * F32);
        const ids = Array.from(new Float32Array(this.mtpReadback.getMappedRange(0, K * F32)), Math.round);
        this.mtpReadback.unmap();
        return ids;
    }

    // --- Speculative decoding ---
    //
    // A verify job carries `drafts`: the job's ids are the last sampled token
    // and the drafts after it, and the head samples every row. The next job of
    // the session starts at `past + kept`, so each block tells from `past`
    // alone how many of the verify's tokens survived. Full-attention KV is
    // addressed by position and needs nothing. A linear_attention block's
    // recurrent state folded every row in, so it is put back from the
    // session's replay slot and the kept rows re-run (model.js restoreSpec).
    //
    // Every piece of this is per session: the markers live on the session's
    // frame and the replay data in its own slot, so chats interleaving on one
    // Runner never touch each other's.

    // The session's replay slot, leased on its first verify job here and held
    // until its frame goes (release, eviction). Throws `nospec` when all
    // `specSlots` are leased, or the GPU has no memory for a new one: nothing
    // has changed yet, so the sender re-sends the job as a plain one.
    async #specSlot(session) {
        const live = this.specLive.get(session);
        if (live) return live;
        const refuse = (why) => {
            this.counts.nospec++;
            return Object.assign(new Error(`no speculative replay slot here (${why})`), { nospec: true });
        };
        let slot = this.specFree.pop() ?? this.#specSteal();
        if (!slot) {
            if (this.specLive.size >= this.specSlots) throw refuse(`${this.specSlots} in use`);
            const { device } = this;
            device.pushErrorScope?.('out-of-memory');
            const layers = [];
            let bytes = 0;
            for (let i = 0; i < this.blocks; i++) {
                const kind = this.d.layer_types?.[this.blockLo + i] ?? 'full_attention';
                if (kind !== 'linear_attention') { layers.push(null); continue; }
                layers.push(makeReplayLayer(device, this.d));
                bytes += replayLayerBytes(this.d);
            }
            const oom = await device.popErrorScope?.();
            if (oom) {
                for (const rp of layers) if (rp) for (const k of SLOT_BUFS) rp[k].destroy();
                throw refuse('out of GPU memory');
            }
            slot = { layers, bytes };
        }
        this.specLive.set(session, slot);
        return slot;
    }

    // The slot of the least recently used session that has been idle for
    // `idleMs` with no job queued here, or null. That session loses its
    // rollback: if its next job starts inside its last verify, the block
    // refuses it as no longer cached, and the enqueuer rebuilds the chat.
    #specSteal() {
        const now = performance.now();
        for (const [id, frame] of this.live) {
            const slot = this.specLive.get(id);
            if (!slot || this.pending.has(id) || now - (frame.lastUse ?? 0) < this.idleMs) continue;
            this.specLive.delete(id);
            frame.specAt?.clear();
            frame.specLost = true;
            return slot;
        }
        return null;
    }

    // The session's frame is gone: its slot goes back for reuse.
    #specRelease(session) {
        const slot = this.specLive.get(session);
        if (!slot) return;
        this.specLive.delete(session);
        this.specFree.push(slot);
    }

    // Growth changes the block layout. Free slots are dropped (the next lease
    // makes one with the new layout). A live session never crosses a new block,
    // since its cut is pinned, so its slot only needs the index kept aligned.
    #specGrow(atStart) {
        for (const slot of this.specFree) {
            for (const rp of slot.layers) if (rp) for (const k of SLOT_BUFS) rp[k].destroy();
        }
        this.specFree = [];
        for (const slot of this.specLive.values()) {
            if (atStart) slot.layers.unshift(null); else slot.layers.push(null);
        }
    }

    // GPU sampling's buffers (kernels.js topkRows): the two it ping-pongs
    // between, and the readback of the candidates, sized for `sampleRows` rows
    // of TOPK_MAX. Made the first time this Runner samples.
    #topkBuffers() {
        if (this.topk) return this.topk;
        const pairs = topkPairs(this.sampleRows, this.d.vocab, TOPK_MAX) * 8;
        const readback = this.sampleRows * TOPK_MAX * 8;
        return this.topk = {
            a: makeStorage(this.device, pairs), b: makeStorage(this.device, pairs),
            readback: makeReadback(this.device, readback), bytes: 2 * pairs + readback,
        };
    }

    // The buffer the head reads whole logits rows back into, `sampleRows` of
    // them (a verify job's, or a batch's). Made the first time it is needed.
    #verifyReadback() {
        return this.specReadback ??= makeReadback(this.device, this.sampleRows * this.d.vocab * F32);
    }

    // The warm-up cache is sized for the blocks held, so growth replaces it.
    // The buffers are destroyed explicitly: an unreferenced GPUBuffer keeps
    // its memory until garbage collection.
    #dropWarm() {
        const warm = this.warm;
        this.warm = null;
        if (!warm) return;
        for (const buf of [...warm.K, ...warm.V]) buf?.destroy();
        for (const rc of warm.recurrent) {
            if (rc) for (const buf of [rc.S, rc.histQ, rc.histK, rc.histV]) buf.destroy();
        }
    }

    // Load the next position, `have`. The holding is always the contiguous
    // prefix `[lo, have)` of the claim.
    //
    // `weights` depends on the position: the embedding needs client.bin's
    // `wte`, the head needs its `norm` (and `lm_head` if untied), and every
    // other position is one block.
    append(weights) {
        const p = this.have;
        if (p >= this.hi) throw new Error(`nothing left to append: ${this.lo}–${this.hi - 1} is full`);
        if (p === 0) {
            if (!weights?.wte) throw new Error('position 0 is the embedding and needs wte');
            this.w.wte = weights.wte;
            // client.bin also has norm. Keep it so the head need not fetch it.
            this.w.norm ??= weights.norm;
        } else if (p === positions(this.d.n_layer) - 1) {
            if (!weights?.norm) throw new Error('the last position is the head and needs norm');
            this.w.norm = weights.norm;
            this.w.lm_head = weights.lm_head ?? weights.wte ?? this.w.wte;
        } else {
            // A full_attention block has q_proj, a linear_attention block has
            // in_proj_q. uploadTensors() has checked the exact names.
            if (!weights?.q_proj && !weights?.in_proj_q)
                throw new Error(`position ${p} is a block and needs its tensors`);
            this.w.layers.push(weights);
        }
        this.weightBytes += weights.bytes ?? 0;
        this.have = p + 1;
        return this.have;
    }

    // Load the position just below `lo`. Applies a left-growth step (the KV
    // that growLeft() reserved, the weights and the indices) synchronously,
    // with no `await`.
    //
    // `block()` converts a position to an index with `p - 1 - blockLo`, for
    // both `w.layers` and every cache's `K`/`V`, so `blockLo` and the arrays
    // must change together. growLeft() spans a network fetch and changes
    // neither. This method changes `blockLo`, `lo` and the arrays in one
    // synchronous step, so no running job sees them out of sync.
    prepend(weights, reserved) {
        const p = this.lo - 1;
        if (p < 0) throw new Error('nothing left to prepend: position 0 is already the lowest');
        if (p === 0) {
            if (!weights?.wte) throw new Error('position 0 is the embedding and needs wte');
            this.w.wte = weights.wte;
            this.w.norm ??= weights.norm;
        } else {
            if (!weights?.q_proj && !weights?.in_proj_q)
                throw new Error(`position ${p} is a block and needs its tensors`);
            this.w.layers.unshift(weights);
            if (reserved.kind === 'linear_attention') {
                this.pool.K.unshift(null);
                this.pool.V.unshift(null);
                for (const { frame, recurrent } of reserved.frames) {
                    frame.recurrent.unshift(recurrent);
                    frame.bytes += recurrentStateBytes(this.d);
                }
            } else {
                this.pool.K.unshift(reserved.K);
                this.pool.V.unshift(reserved.V);
                this.pool.bytes += reserved.bytes * 2;
                for (const frame of [...this.free, ...this.live.values()]) {
                    frame.recurrent.unshift(null);
                }
            }
            // The next warmup() recreates it at the new size.
            this.#dropWarm();
            this.#specGrow(true);
            this.blocks++;
            this.blockLo--;
        }
        this.weightBytes += weights.bytes ?? 0;
        this.lo = p;
        return this.lo;
    }

    // Reserve the KV (or, for a linear_attention block, the recurrent state)
    // that the position below `lo` needs, before it is downloaded. Returns the
    // buffers unattached; prepend() attaches them. Position 0, the embedding,
    // needs none.
    //
    // The new position is `this.lo - 1`, so its global block index (the key
    // into `d.layer_types`) is `this.lo - 2`.
    growLeft() {
        if (this.lo <= 0) throw new Error('nothing left to grow: already at position 0');
        if (this.lo <= 1) return null;
        const kind = this.d.layer_types?.[this.lo - 2] ?? 'full_attention';
        // Recurrent state is per session: one per frame. KV is one pair of
        // pool buffers per peer.
        if (kind === 'linear_attention') {
            return {
                kind,
                frames: [...this.free, ...this.live.values()].map((frame) => ({
                    frame, recurrent: makeRecurrentState(this.device, this.d),
                })),
            };
        }
        const bytes = this.#poolBufferBytes();
        return {
            kind,
            bytes,
            K: makeStorage(this.device, bytes),
            V: makeStorage(this.device, bytes),
        };
    }

    // Bytes of one layer's K (or V) pool buffer.
    #poolBufferBytes() {
        return kvRowBytes(this.d) * this.pool.pageTokens * this.pool.nPages;
    }

    // Reserve and attach the state for the position at `hi`, then raise `hi`.
    // Appending to the arrays changes no existing index, and the new position
    // is not served until append() advances `have`. The head position needs no
    // state.
    //
    // The new position is `this.hi` before the increment, so its global block
    // index is `this.hi - 1`.
    growRight() {
        const N = positions(this.d.n_layer);
        if (this.hi >= N) throw new Error('nothing left to grow: already at the last position');
        if (this.hi < N - 1) {
            const kind = this.d.layer_types?.[this.hi - 1] ?? 'full_attention';
            if (kind === 'linear_attention') {
                this.pool.K.push(null);
                this.pool.V.push(null);
                for (const frame of [...this.free, ...this.live.values()]) {
                    frame.recurrent.push(makeRecurrentState(this.device, this.d));
                    frame.bytes += recurrentStateBytes(this.d);
                }
            } else {
                const bytes = this.#poolBufferBytes();
                this.pool.K.push(makeStorage(this.device, bytes));
                this.pool.V.push(makeStorage(this.device, bytes));
                this.pool.bytes += bytes * 2;
                for (const frame of [...this.free, ...this.live.values()]) {
                    frame.recurrent.push(null);
                }
            }
            this.#dropWarm();
            this.#specGrow(false);
            this.blocks++;
        }
        this.hi++;
    }

    // Whether this Runner can run a job whose cursor is at `pos`. Several
    // peers may hold the same position; choosing between them is routing.
    holds(pos) { return pos >= this.lo && pos < this.have; }

    // Jobs run in arrival order. A failed job does not block the ones after
    // it. `want` is where the router would like this job to leave (route.js's
    // plan()). #exit() clamps it to what this peer holds, and ignores it for a
    // session that already has a cut here.
    //
    // `queuedAt` is taken on arrival, so time spent queued is reported
    // separately from GPU time.
    //
    // Refused with `busy` once MAX_QUEUE sessions have jobs queued, or its
    // session has MAX_SESSION_QUEUE. A job waiting for a frame counts.
    run(h, hidden, want) {
        const s = h.s;
        const mine = this.pending.get(s) ?? 0;
        if (mine >= MAX_SESSION_QUEUE || (!mine && this.pending.size >= MAX_QUEUE)) {
            return Promise.reject(this.#atCapacity('queue slot', s));
        }
        this.backlog++;
        this.pending.set(s, (this.pending.get(s) ?? 0) + 1);
        this.#keepWarm();
        return new Promise((resolve, reject) => {
            this.#submit({ h, hidden, want, queuedAt: performance.now(), resolve, reject });
        }).finally(() => {
            this.activeAt = performance.now();
            this.backlog--;
            const n = this.pending.get(s) - 1;
            if (n > 0) this.pending.set(s, n);
            else this.pending.delete(s);
            // Idle time counts from the end of the session's last job.
            const frame = this.live.get(s);
            if (frame) frame.lastUse = performance.now();
        });
    }

    // Queue one job. A job that finds no frame for its session is parked, so
    // the jobs behind it, which belong to other chats, are not delayed.
    #submit(job) {
        this.#enqueue({ s: job.h.s, fill: !!job.h.fill, job });
    }

    // Run `fn` on the queue, after the work queued ahead of it. Resolves with
    // its result.
    schedule(fn) {
        return new Promise((resolve, reject) => {
            this.#enqueue({ s: null, fill: false,
                            run: async () => { try { resolve(await fn()); } catch (e) { reject(e); } } });
        });
    }

    #enqueue(task) {
        this.tasks.push(task);
        if (this.draining) return;
        this.draining = true;
        // Started on a microtask, so run() returns before its job starts.
        queueMicrotask(() => this.#drain());
    }

    async #drain() {
        while (this.tasks.length) {
            const [task] = this.tasks.splice(this.#next(), 1);
            const run = task.job ? this.#runJobs([task.job, ...this.#batchWith(task.job)]) : task.run();
            try { await run; } catch { /* a task reports its own failure */ }
        }
        this.draining = false;
        for (const wake of this.onIdle.splice(0)) wake();
    }

    // Start the keep-warm loop, unless it is off or already running.
    #keepWarm() {
        this.activeAt = performance.now();
        if (!this.keepWarmMs || this.warming) return;
        this.warming = true;
        this.#warmLoop().catch(() => { /* a lost device: the next job reports it */ })
            .finally(() => { this.warming = false; });
    }

    // WARM_DEPTH lanes, each submitting one round at a time while the queue
    // is empty.
    async #warmLoop() {
        const { device } = this;
        const k = this.warmKernel ??= makeWarmKernel(device);
        const lane = async () => {
            while (!k.lost && performance.now() - this.activeAt < this.keepWarmMs) {
                if (this.draining) {
                    await new Promise((wake) => this.onIdle.push(wake));
                    continue;
                }
                const enc = device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(k.pipe);
                pass.setBindGroup(0, k.bind);
                pass.dispatchWorkgroups(1);
                pass.end();
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();
            }
        };
        await Promise.all(Array.from({ length: WARM_DEPTH }, lane));
    }

    // Which queued task runs next: the oldest. A task carries its session and
    // whether it is a prefill chunk (`fill`), so a policy that runs other
    // chats' decode jobs ahead of a long prefill can be added here. It must
    // keep each session's tasks, and every task with no session, in order.
    #next() {
        return 0;
    }

    // Batching: the queued jobs that run in one pass with `lead`, taken off
    // the queue. Decode is bound by reading the weights, which a pass does
    // once for all its rows, so several chats' tokens cost about what one
    // does. A job joins if it enters at the same cursor and fits in
    // `batchRows` with the others. Replays, prefill chunks, warm-ups and
    // profiled jobs run alone. Order is kept: a session's later jobs never
    // pass its earlier ones, and nothing passes a task with no session.
    // Nothing waits for a batch to fill: jobs that arrive while a pass runs
    // make the next one.
    #batchWith(lead) {
        if (this.batchRows < 2 || !this.#batchable(lead.h)) return [];
        const seen = new Set([lead.h.s]);
        const jobs = [];
        let rows = lead.h.n;
        for (let i = 0; i < this.tasks.length && rows < this.batchRows; i++) {
            const task = this.tasks[i];
            if (task.s === null) break;
            const fits = !seen.has(task.s) && this.#batchable(task.job.h)
                && task.job.h.pos === lead.h.pos && rows + task.job.h.n <= this.batchRows;
            seen.add(task.s);
            if (!fits) continue;
            jobs.push(task.job);
            rows += task.job.h.n;
            this.tasks.splice(i--, 1);
        }
        return jobs;
    }

    #batchable(h) {
        return h.s !== WARMUP && !h.replay && !h.fill && !h.prof
            && Number.isInteger(h.n) && h.n >= 1 && h.n <= this.batchRows;
    }

    // The waiting list: jobs whose session has no frame yet, in arrival order.
    // A session keeps the place of its first job. A job that was let through
    // and found the frame gone goes to the front, after any jobs of its own
    // session already there.
    #park(job) {
        if (job.deadline === undefined) {
            job.deadline = performance.now() + this.waitMs;
            this.counts.waited++;
            this.waiting.push(job);
        } else {
            let i = 0;
            while (i < this.waiting.length && this.waiting[i].h.s === job.h.s) i++;
            this.waiting.splice(i, 0, job);
        }
        // No event signals a session going idle, so the list is polled on a
        // timer while it is non-empty, at a quarter of the idle time.
        this.waitTimer ??= setInterval(() => this.#pump(),
            Math.max(5, Math.min(250, this.idleMs / 4)));
    }

    // Refuse the jobs that have waited past their deadline. Then, if a frame
    // is available, put every job of the first waiting session back on the
    // queue, in order. One session at a time: `turn` lets that session through
    // #cacheFor and is held until its jobs have run.
    #pump() {
        const now = performance.now();
        const late = this.waiting.filter((j) => now >= j.deadline);
        if (late.length) {
            this.waiting = this.waiting.filter((j) => now < j.deadline);
            for (const j of late) j.reject(this.#atCapacity('session', j.h.s));
        }
        if (!this.waiting.length) {
            clearInterval(this.waitTimer);
            this.waitTimer = undefined;
            return;
        }
        if (this.turn !== null) return;
        if (!this.free.length && this.#evictable() == null) return;

        const s = this.waiting[0].h.s;
        const jobs = this.waiting.filter((j) => j.h.s === s);
        this.waiting = this.waiting.filter((j) => j.h.s !== s);
        this.turn = s;
        for (const job of jobs) this.#submit(job);
        this.schedule(() => { this.turn = null; });
    }

    // Run one job through newly loaded positions before they are announced.
    // Catches a GPU that produces NaN on them (usually reduced-precision
    // accumulators) and compiles the shaders.
    //
    // Runs from `from` to `want` (default: everything held). A left-growth
    // step passes `want` to cover only the new position.
    //
    // Returns { blocks, ms, blocksPerSec }, including shader compilation. Run
    // it twice for a steady-state figure.
    async warmup(from = this.lo, want = this.have) {
        const s = WARMUP;
        const hidden = from === 0 ? null : new Float32Array(this.d.hidden);
        const t0 = performance.now();
        try {
            const r = await this.run(
                { k: 'job', s, pos: from, past: 0, n: 1, ids: [0], opts: { temp: 0, topK: 0 } },
                hidden, want);

            // Check the values, not the sampled id: argmax over an all-NaN row
            // returns id 0, which is also a valid token.
            const out = r.at === positions(this.d.n_layer)
                ? await readF32(this.device, this.scratch.logits, this.d.vocab * F32)
                : r.f32;
            if (!out.every(Number.isFinite)) {
                throw new Error(`a warm-up pass through positions ${from}–${r.at - 1} produced ` +
                    'non-finite output. This GPU cannot run them ' +
                    '(reduced-precision accumulators?)');
            }

            const ms = performance.now() - t0;
            const blocks = r.at - from;
            // A speed estimate for routing before any real token. The first
            // pass compiles the shaders, so it is skipped.
            if (this.warmups++ > 0 && !this.decoded && blocks > 0) {
                const per = ms / blocks;
                this.msPerPos = this.msPerPos == null ? per : 0.8 * this.msPerPos + 0.2 * per;
            }
            return { blocks, ms, blocksPerSec: ms > 0 ? (blocks * 1000) / ms : 0 };
        } finally {
            this.cuts.delete(s);
        }
    }

    // How far this session runs here. Fixed the first time the session enters
    // at `at` and never changed, because this peer holds the session's KV for
    // exactly those positions.
    //
    // Example: a peer holding 10-13 takes a session at 10, then grows to
    // 10-15. If the next job ran to 15, attention at 14 and 15 would read
    // cache rows that were never written, and the peer holding the real KV
    // from 14 would not get the job. So new positions serve only new sessions.
    // Keyed by entry cursor too, because overlapping ranges let one session
    // enter the same peer at two cursors.
    #exit(session, at, want) {
        let cuts = this.cuts.get(session);
        if (!cuts) this.cuts.set(session, cuts = new Map());
        const pinned = cuts.get(at);
        if (pinned !== undefined) return pinned;
        // Clamp to what is downloaded, and to at least one position. holds(at)
        // has already confirmed this peer can advance the cursor.
        const to = Math.min(Math.max(want ?? this.have, at + 1), this.have);
        cuts.set(at, to);
        return to;
    }

    // The session is over: return its frame.
    //
    // Runs on the queue. Releasing while a job of that session is running
    // would give its buffers to another chat mid-job, so queued jobs finish
    // first.
    release(session) {
        return this.schedule(() => this.#release(session));
    }

    // Idempotent: a `free` often arrives for a session this peer evicted or
    // never held. Drops the frame, cut and boundary log, as eviction does.
    #release(session) {
        const cache = this.live.get(session);
        if (cache) {
            this.live.delete(session);
            freePages(this.pool, cache);
            this.free.push(cache);
            this.counts.released++;
        }
        this.#specRelease(session);
        this.cuts.delete(session);
        this.logs.delete(session);
        this.refused.delete(session);
        // Also remove its jobs from the waiting list.
        const mine = this.waiting.filter((j) => j.h.s === session);
        if (mine.length) {
            this.waiting = this.waiting.filter((j) => j.h.s !== session);
            for (const j of mine) j.reject(new Error(`session ${session} ended while it waited for a frame`));
        }
        // A frame is free: admit the first waiting session without waiting for
        // the timer.
        if (this.waiting.length) this.#pump();
    }

    // This peer's active work: the positions it runs per token, summed over
    // sessions with a job here or one within ACTIVE_MS. route.js reports it
    // in acks, so the peers handing work here can balance new sessions.
    get load() { return this.#activity().w; }

    // How many sessions `load` counts: each costs one job's overhead per
    // token, whatever its width.
    get active() { return this.#activity().n; }

    // How long a job arriving now would wait, by recent jobs. Zero once the
    // runner has been quiet for ACTIVE_MS.
    get queueMs() { return performance.now() - this.lastJobAt < ACTIVE_MS ? this.queueAvgMs : 0; }

    // Where this runner's active sessions that entered at `pos` leave, and
    // how many leave at each: cut -> count. Sessions with the same entry and
    // cut can share a pass (#batchWith), so route.js puts a new session on
    // the cut most of them use.
    groups(pos) {
        const now = performance.now();
        const out = new Map();
        for (const [s, frame] of this.live) {
            if (!this.pending.has(s) && now - (frame.lastUse ?? 0) >= ACTIVE_MS) continue;
            const to = this.cuts.get(s)?.get(pos);
            if (to !== undefined) out.set(to, (out.get(to) ?? 0) + 1);
        }
        return out;
    }

    #activity() {
        const now = performance.now();
        let n = 0, w = 0;
        for (const [s, frame] of this.live) {
            if (!this.pending.has(s) && now - (frame.lastUse ?? 0) >= ACTIVE_MS) continue;
            n++;
            for (const [at, to] of this.cuts.get(s) ?? []) w += to - at;
        }
        return { n, w };
    }

    // The first live session idle for at least `idleMs`, or null if none is.
    // A session with a job queued or running here is never idle: under load a
    // job can wait in the queue longer than `idleMs`. `live` is in LRU order,
    // so the first match is the least recently used.
    #evictable() {
        const now = performance.now();
        for (const [id, frame] of this.live) {
            if (this.pending.has(id)) continue;
            if (now - (frame.lastUse ?? 0) >= this.idleMs) return id;
        }
        return null;
    }

    // A capacity refusal. route.js passes it back to the enqueuer without
    // repairing, since no hand-off is broken.
    #atCapacity(what, session = null) {
        const err = new Error(`this peer is serving ${this.live.size} chats and has no ` +
            `${what} free for another right now; try again in a moment`);
        err.busy = true;
        this.counts.refused++;
        if (session != null) {
            const now = performance.now();
            for (const [s, at] of this.refused) if (now - at >= REFUSED_MS) this.refused.delete(s);
            this.refused.set(session, now);
        }
        return err;
    }

    // A job that finds its session's cache behind it: cold, which route.js
    // repairs. But after a job of the session was refused here for capacity,
    // it is the next chunk of that prompt, sent before the refusal came back.
    // It is refused for capacity too. A repair would run a prompt, and open a
    // decode loop, that the enqueuer has already given up on.
    #cold(session, msg) {
        const at = this.refused.get(session);
        if (at !== undefined && performance.now() - at < REFUSED_MS) {
            const err = new Error(`an earlier job of this chat was refused here for capacity; ` +
                `try again in a moment`);
            err.busy = true;
            return err;
        }
        this.counts.cold++;
        return new Error(msg);
    }

    #cacheFor(session, past) {
        // Warm-up has its own one-row cache, so warming a new position while
        // serving never evicts a live chat.
        if (session === WARMUP) {
            return this.warm ??= makeCache(this.device, this.d, 1, this.blocks, this.blockLo);
        }

        const live = this.live.get(session);
        if (live) {                                  // move to the end: Map keeps insertion order
            this.live.delete(session);
            this.live.set(session, live);
            live.lastUse = performance.now();
            return live;
        }
        // The session already has a job waiting. This one waits behind it,
        // whatever its `past`, so the session's jobs run in arrival order.
        if (this.waiting.some((j) => j.h.s === session)) throw WAIT;
        if (past > 0) throw this.#cold(session, `session ${session} is no longer cached here`);
        // First come, first served: a new session does not take a frame while
        // other sessions are waiting for one.
        if (this.waiting.length && this.turn !== session) throw WAIT;

        let cache = this.free.pop();
        if (!cache) {
            const oldest = this.#evictable();
            if (oldest == null) throw this.waitMs > 0 ? WAIT : this.#atCapacity('session', session);
            this.counts.evicted++;
            cache = this.live.get(oldest);
            this.live.delete(oldest);
            // Its pages, cut and boundary log are dropped too. An evicted
            // session cannot resume (its next job throws). Dropping the logs
            // here bounds them to `sessions`.
            freePages(this.pool, cache);
            this.cuts.delete(oldest);
            this.logs.delete(oldest);
            this.#specRelease(oldest);
        }
        // A reused frame holds the previous session's state. For KV that is
        // harmless: scatter_kv overwrites every row before attention reads it.
        // Recurrent state is accumulated, not overwritten, so it is cleared
        // before a new session uses the frame.
        const stale = cache.recurrent.some(Boolean) && cache.used;
        if (stale) {
            const enc = this.device.createCommandEncoder({ label: 'clear recurrent state' });
            for (const rc of cache.recurrent) {
                if (!rc) continue;
                for (const buf of [rc.S, rc.histQ, rc.histK, rc.histV]) enc.clearBuffer(buf);
            }
            this.device.queue.submit([enc.finish()]);
        }
        cache.used = true;
        // Rows each block holds for this session, and the last verify job each
        // block ran if the session has not moved on since. See #checkFed().
        cache.fed = new Map();
        cache.specAt = new Map();
        cache.specLost = false;

        cache.lastUse = performance.now();
        this.live.set(session, cache);
        this.counts.admitted++;
        return cache;
    }

    // Check that this session's state at every block the job crosses matches
    // `past`.
    //
    // A full_attention block addresses its KV by position, so a job may start
    // at or below the rows it holds (a rewind overwrites the tail) but not
    // above, where no rows were written. A linear_attention block has no
    // position axis: its recurrent state and conv ring fold in everything fed
    // so far. A job must continue where the last one ended. It cannot be
    // rewound, only restarted: `past` 0 clears the state. The buffers to clear
    // are returned, and the caller encodes the clear before the pass.
    //
    // The one exception is a verify job (see Speculative decoding above). If
    // the block's last job was a verify of n rows from `base`, the next job
    // may start anywhere in (base, base + n]: it keeps `past - base` of those
    // rows, and `keep` says so. With `rb` (a verify refused further down the
    // line, re-sent as a plain job) it may also start at `base`, keeping none.
    //
    // Keyed by global block index, so growth on either side does not shift it.
    // A mismatch throws the same error as an evicted cache and is handled the
    // same way: passed back, repaired locally, or rebuilt by the enqueuer.
    //
    // A job the head fed (`auto`, route.js's closed loop) must continue every
    // block exactly, since only the enqueuer rewinds. If it is behind, it is a
    // duplicate from a repair re-sending a hand-off that had arrived. It
    // throws `stale`, which route.js drops without answering.
    //
    // Returns { reset, keep }: keep is null when nothing is rolled back.
    #checkFed(session, cache, at, last, past, exact = false, rb = false) {
        const reset = [];
        let keep = null;
        for (let p = Math.max(at, 1); p < last; p++) {
            const b = p - 1;
            const fed = cache.fed.get(b) ?? 0;
            const rc = cache.recurrent[b - this.blockLo];
            const sp = cache.specAt?.get(b);
            let k = null;
            if (sp && fed === sp.base + sp.n && past < fed
                && (past > sp.base || (rb && past === sp.base))) {
                k = past - sp.base;
            } else if (past === 0 && !exact) {
                if (rc && fed > 0) reset.push(rc);
                continue;
            } else if (rc || exact ? past === fed : past <= fed) {
                continue;
            } else if (exact && past < fed && !cache.specLost) {
                // A frame whose replay slot was taken (#specSteal) cannot tell
                // a duplicate from a job that needed the rollback, so it
                // refuses as cold instead, which reaches the enqueuer.
                throw Object.assign(new Error(`a duplicate of session ${session}'s job at ${past} ` +
                    `(block ${b} is already at ${fed})`), { stale: true });
            } else if (past > fed) {
                throw this.#cold(session, `session ${session} is no longer cached here ` +
                    `(block ${b} holds ${fed} rows, the job continues from ${past})`);
            } else {
                this.counts.cold++;
                throw new Error(`session ${session} is no longer cached here ` +
                    `(block ${b} holds ${fed} rows, the job continues from ${past})`);
            }
            // Every block a session crosses here ran the same jobs, so they
            // agree on what to keep.
            if (keep !== null && keep !== k) {
                this.counts.cold++;
                throw new Error(`session ${session}'s blocks disagree on a rollback ` +
                    `(${keep} and ${k} rows)`);
            }
            keep = k;
        }
        return { reset, keep };
    }

    // Get enough pages for `tokens` logical positions from the pool, evicting
    // idle sessions whole if needed. A session that is not idle is never
    // evicted; the job is refused.
    #pagesFor(session, frame, tokens) {
        if (ensurePages(this.device, this.pool, frame, tokens)) return;
        for (;;) {
            const id = this.#evictable();
            if (id == null || id === session) break;
            const victim = this.live.get(id);
            this.live.delete(id);
            this.free.push(victim);
            freePages(this.pool, victim);
            this.cuts.delete(id);
            this.logs.delete(id);
            this.#specRelease(id);
            this.counts.evicted++;
            if (ensurePages(this.device, this.pool, frame, tokens)) return;
        }
        // Nothing left to reclaim. If this context is larger than the pool, it
        // is a configuration error and waiting will not help. Otherwise every
        // live chat is active and the job is refused for capacity.
        if (Math.ceil(tokens / this.pool.pageTokens) > frame.maxPages
            || this.pool.nPages < Math.ceil(tokens / this.pool.pageTokens)) {
            throw new Error(`the KV pool cannot fit ${tokens} tokens for session ` +
                `${session} (${this.pool.nPages} pages of ${this.pool.pageTokens})`);
        }
        throw this.#atCapacity('KV page', session);
    }

    // The rows handed downstream at `at`, kept for a replay. Rows are stored
    // at their absolute position, so a job at a lower `past` (after a rewind)
    // overwrites the same rows it overwrites in the downstream KV caches, and
    // `len` is set to match.
    //
    // Allocated on first use and doubled from LOG_CHUNK up to `nMax`, so a
    // short chat uses little host RAM. Rows are half bits, as sent.
    #log(s, at, past, rows) {
        if (s === WARMUP) return;              // a warm-up hands its rows to nobody
        const H = this.d.hidden;
        let boundaries = this.logs.get(s);
        if (!boundaries) this.logs.set(s, boundaries = new Map());
        let log = boundaries.get(at);
        if (!log) {
            const cap = Math.min(LOG_CHUNK, this.nMax);
            boundaries.set(at, log = { rows: new Uint16Array(cap * H), len: 0 });
        }

        const need = past + rows.length / H;
        if (need * H > log.rows.length) {
            let cap = log.rows.length / H;
            while (cap < need) cap = Math.min(this.nMax, cap * 2);
            const grown = new Uint16Array(cap * H);
            grown.set(log.rows);               // rows are at absolute positions
            log.rows = grown;
        }
        log.rows.set(rows, past * H);
        log.len = need;
    }

    // Run queued jobs, as one pass where they can share one. Each is prepared
    // in turn. One that cannot run (no frame yet, a cold cache, a bad header)
    // leaves the batch the way it would fail a pass of its own, and the rest
    // run. Jobs that leave at different positions run as separate passes.
    async #runJobs(jobs) {
        let t0 = performance.now(), c0 = counters();
        const groups = new Map();
        for (const job of jobs) {
            let plan;
            try { plan = await this.#prepare(job.h, job.hidden, job.want); } catch (err) {
                this.#failed(job, err);
                continue;
            }
            const key = `${plan.to}:${plan.head}:${plan.filling}:${plan.spent}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push({ job, plan });
        }
        for (const group of groups.values()) {
            let results;
            try {
                results = await this.#pass(group.map((g) => g.plan), group.map((g) => g.job.queuedAt), t0, c0);
            } catch (err) {
                for (const { job } of group) this.#failed(job, err);
                continue;
            } finally {
                t0 = performance.now();
                c0 = counters();
            }
            group.forEach(({ job }, i) => {
                if ('err' in results[i]) this.#failed(job, results[i].err);
                else job.resolve(results[i].value);
            });
        }
    }

    #failed(job, err) {
        if (err === WAIT) this.#park(job);
        else job.reject(err);
    }

    // Check a job and get its session ready to run: the cut, the frame, the
    // rows each block holds, the replay slot, the KV pages. Throws before
    // anything changes if the job cannot run. Returns what #pass needs.
    async #prepare(h, hidden, want) {
        const { d } = this;
        const T = h.n, H = d.hidden;

        // When this peer last ran a non-warm-up job. engine.js's #growClaim()
        // backs off while jobs are arriving.
        if (h.s !== WARMUP) this.lastJobAt = performance.now();

        // Routing should not send a job to a peer that cannot advance it. This
        // check stops a stale route being answered from the wrong part of the
        // KV cache.
        const at = h.pos;
        if (!this.holds(at)) {
            throw new Error(`position ${at} is not held here ` +
                `(this peer has ${this.lo}–${this.have - 1})`);
        }
        // Where this job leaves. For a session that has run here before, this
        // is its existing cut, whatever this peer holds.
        const to = this.#exit(h.s, at, want);
        const end = to === positions(d.n_layer);
        // A replay only rewrites KV rows; its output is not used. At the end
        // of the line it runs no LM head, reads nothing back and samples
        // nothing. Before the end it is an ordinary job, handed on with the
        // flag set so it rebuilds the caches down the line.
        // A prefill chunk other than the prompt's last (`fill`) only fills
        // caches too. At the end of the line it runs no LM head and samples
        // nothing, but keeps the MTP head's state, and route.js answers it.
        const filling = end && !h.replay && !!h.fill;
        const head = end && !h.replay && !h.fill;
        const spent = end && !!h.replay;
        if (h.fill && h.drafts != null) throw new Error('a prefill chunk cannot carry drafts');
        // The header came from another peer. `n` sizes every scratch write and
        // `ids` goes into the embedding gather, so a job that does not fit the
        // buffers is refused before it reaches the GPU.
        if (!Number.isInteger(T) || T < 1 || T > this.tMax) {
            throw new Error(`a job of ${T} tokens is outside 1–${this.tMax}`);
        }
        if (!Number.isInteger(h.past) || h.past < 0) throw new Error(`past ${h.past} is not a position`);
        if (h.past + T > this.nMax) throw new Error(`context ${h.past + T} exceeds ${this.nMax}`);

        // Only a job entering at position 0 carries ids, since position 0 is
        // the embedding gather. A job entering anywhere else carries the
        // residual stream. That includes the embedding's holder, when an
        // overlapping peer upstream handed off inside this range.
        const gather = at === 0;
        // Only the count and type are checked. WebGPU clamps the gather's
        // reads, so an id past the vocabulary gives a wrong answer, not an
        // out-of-bounds read.
        if (gather && !(Array.isArray(h.ids) && h.ids.length === T &&
                        h.ids.every((id) => Number.isInteger(id) && id >= 0))) {
            throw new Error(`a job at position 0 needs ${T} token ids`);
        }
        if (!gather && hidden?.length !== T * H) {
            throw new Error(`payload is ${hidden?.length ?? 0} floats, expected ${T * H}`);
        }
        // Rows from another peer or a boundary log are half bits (f16.js).
        if (hidden instanceof Uint16Array) hidden = fromF16(hidden);
        // A verify job: its ids are the last sampled token and the drafts.
        const drafts = h.drafts ?? null;
        const verify = drafts !== null && h.s !== WARMUP && !h.replay;
        if (drafts !== null && !(Array.isArray(drafts) && drafts.length >= 1
                && drafts.length === T - 1 && T <= SPEC_COLS
                && drafts.every((id) => Number.isInteger(id) && id >= 0))) {
            throw new Error(`a verify job needs 1–${SPEC_COLS - 1} draft ids, one fewer than its ${T} tokens`);
        }
        const cache = this.#cacheFor(h.s, h.past);
        // The block positions this job runs: from the cursor (or position 1,
        // if the cursor is the embedding) up to the session's cut or the head,
        // whichever is first.
        const first = Math.max(at, 1), last = Math.min(to, d.n_layer + 1);
        const { reset, keep } = h.s === WARMUP ? { reset: [], keep: null }
            : this.#checkFed(h.s, cache, at, last, h.past, !!h.auto, !!h.rb);
        // The replay slot, when a linear_attention block here snapshots (a
        // verify) or rolls back (the job after one). Leased before anything
        // changes, so a `nospec` refusal leaves the session as it was.
        let linear = false;
        for (let p = first; p < last; p++) {
            if (cache.recurrent[p - 1 - this.blockLo]) { linear = true; break; }
        }
        let slot = null;
        if (linear && keep !== null) {
            slot = this.specLive.get(h.s);
            if (!slot) {
                this.counts.cold++;
                throw new Error(`session ${h.s} has no replay data here to roll back`);
            }
        }
        if (linear && verify) slot = await this.#specSlot(h.s);
        const spec = slot ? { replay: slot.layers, snap: verify, keep } : null;
        // Before any dispatch: scatter_kv writes rows h.past .. h.past+T-1, so
        // their pages must be allocated and in the table. The warm-up cache is
        // contiguous and not pooled.
        // The MTP head writes its own rows past the job's: the drafts it
        // chains after the sample.
        const mtp = (head || filling) && !!this.mtp && !!h.mtp && h.s !== WARMUP;
        if (h.s !== WARMUP) {
            this.#pagesFor(h.s, cache, mtp && head ? Math.min(this.nMax, h.past + T + SPEC_COLS) : h.past + T);
        }
        const fill = mtp && Array.isArray(h.ids) && h.ids.length === T;

        // The rows the head samples (the last, or every row of a verify) and
        // the sampler settings of each. The GPU selects each row's raw top M
        // logits (sampling.js candidatesFor), and only those are read back.
        // null M: some row needs the whole row (temp > 0 with no top-k).
        const sampled = head && !spent ? (verify ? T : 1) : 0;
        const optsOf = (i) => (i === 0 ? h.opts : (h.rowOpts?.[i - 1] ?? h.opts));
        let M = sampled ? 0 : null;
        for (let i = 0; i < sampled && M !== null; i++) {
            const m = candidatesFor(optsOf(i), d.vocab);
            M = m === null ? null : Math.max(M, m);
        }
        return { h, hidden, T, at, to, first, last, head, filling, spent, gather, verify, drafts,
                 cache, reset, spec, mtp, fill, sampled, optsOf, M };
    }

    // Encode, submit and read back one pass for prepared jobs that enter at
    // the same cursor and leave at the same cut. Their rows are packed in
    // order from row 0 (see model.js blockSegs). Returns one { value } or
    // { err } per job.
    async #pass(plans, queued, t0, c0) {
        const { device, d, scratch: s } = this;
        const H = d.hidden;
        const one = plans.length === 1;
        const { h: h0, at, to, first, last, head, filling, spent, gather } = plans[0];
        let T = 0, sampled = 0;
        for (const p of plans) {
            p.rowOff = T;
            p.sampledOff = sampled;
            T += p.T;
            sampled += p.sampled;
        }
        // Rows sampled in one pass share a top-M: the largest any row needs,
        // or the whole row if any needs it.
        let M = sampled ? 0 : null;
        for (const p of plans) {
            if (!p.sampled) continue;
            M = M === null || p.M === null ? null : Math.max(M, p.M);
        }
        const topk = M ? this.#topkBuffers() : null;
        let picked = null;
        const ms = this.mtpScratch;
        if (one && plans[0].fill) device.queue.writeBuffer(ms.mtpIds, 0, Float32Array.from(h0.ids));
        else if (plans.some((p) => p.fill)) {
            for (const p of plans) {
                if (p.fill) device.queue.writeBuffer(this.mtpBatchIds, p.rowOff * F32, Float32Array.from(p.h.ids));
            }
        }

        // writeBuffer is ordered on the queue, so these are applied before the
        // command buffer below runs.
        if (gather) device.queue.writeBuffer(this.ids, 0, Float32Array.from(plans.flatMap((p) => p.h.ids)));
        else for (const p of plans) device.queue.writeBuffer(s.x, p.rowOff * H * F32, p.hidden);

        // GPU timing requested by the enqueuer (gpuprof.js). Not for a replay
        // or a warm-up: neither produces a token. A job asking for it runs
        // alone.
        const prof = !one || h0.replay || h0.s === WARMUP ? null : makeGpuProf(device, h0.prof);

        const segs = plans.map((p) => ({ cache: p.cache, T_past: p.h.past, n: p.T, rowOff: p.rowOff,
                                         spec: p.spec }));
        const enc = device.createCommandEncoder({ label: `positions ${at}–${to - 1}` });
        // Clear recurrent state that holds rows when the job restarts from
        // `past` 0 (see #checkFed), before the pass that reads it.
        for (const p of plans) {
            for (const rc of p.reset) {
                for (const buf of [rc.S, rc.histQ, rc.histK, rc.histV]) enc.clearBuffer(buf);
            }
        }
        withPass(enc, (pass) => {
            prof?.at('embed');
            if (gather) {
                gatherEmbedNoPos(device, pass, this.ids, this.w.wte, s.x, T, H);
                // Rounded as block() rounds its output, so the embedding's
                // rows are the same f16 values whoever holds the next block.
                roundF16(device, pass, s.x, T * H);
            }

            // `w.layers` and the cache are both indexed from `blockLo`:
            // position `p` is block `p - 1`, which is layer `p - 1 - blockLo`.
            for (let p = first; p < last; p++) {
                const layer = p - 1 - this.blockLo;
                prof?.at(this.w.layers[layer].kind === 'linear_attention' ? 'linear' : 'full');
                blockSegs(device, pass, d, this.w, layer, segs, s, this.seg);
            }

            // Sample when the cut reaches the end of the line. A peer that
            // holds the head can stop before it for a session whose cut is
            // earlier.
            //
            // A verify samples every row. Any other job samples only its last,
            // so the head runs on that row alone. `s.down` is free once the
            // blocks have run. A batch gathers every job's sampled rows there.
            if (head) {
                prof?.at('head');
                let src = s.x;
                if (one) {
                    if (!plans[0].verify && T > 1) {
                        copyF32(device, pass, s.x, s.down, H, (T - 1) * H, 0);
                        src = s.down;
                    }
                } else {
                    for (const p of plans) {
                        copyF32(device, pass, s.x, s.down, p.sampled * H,
                                (p.rowOff + p.T - p.sampled) * H, p.sampledOff * H);
                    }
                    src = s.down;
                }
                rmsnorm(device, pass, src, this.w.norm, s.h, sampled, H, d.rms_eps);
                matmulNobias(device, pass, s.h, this.w.lm_head ?? this.w.wte,
                             s.logits, sampled, H, d.vocab);
                if (topk) picked = topkRows(device, pass, s.logits, sampled, d.vocab, M,
                                            topk.a, topk.b, 0);
            }
            // The MTP layer's KV for this job's positions. Row r pairs the
            // token at past + r with the trunk's hidden state at past + r - 1:
            // the session's kept row for r = 0, this job's rows after.
            for (const p of plans) {
                if (!p.fill) continue;
                const off = p.rowOff * H;
                if (!one) copyF32(device, pass, this.mtpBatchIds, ms.mtpIds, p.T, p.rowOff, 0);
                copyF32(device, pass, p.cache.mtpH, ms.mtpIn, H);
                if (p.T > 1) copyF32(device, pass, s.x, ms.mtpIn, (p.T - 1) * H, off, H);
                mtpBlock(device, pass, d, this.mtp, this.#mtpCache(p.cache), ms, ms.mtpIn,
                         ms.mtpIds, p.T, p.h.past);
                // A prefill chunk samples nothing, so #mtpAfter does not run.
                // Its last row is the next job's row 0.
                if (filling) copyF32(device, pass, s.x, p.cache.mtpH, H, off + (p.T - 1) * H, 0);
            }
        }, `positions ${at}–${to - 1}`, prof);

        // The readback copy is on the same encoder, outside the pass, in the
        // same submit. A job that ends at the head reads back the logits rows
        // it sampled (or their top M). A prefill chunk at the end reads back
        // nothing. Any other job reads back the hidden state.
        const rowBytes = d.vocab * F32;
        const bytes = topk ? sampled * M * 8 : head ? sampled * rowBytes : T * H * F32;
        const readback = topk ? topk.readback : head && sampled > 1 ? this.#verifyReadback() : s.readback;
        if (topk) {
            enc.copyBufferToBuffer(picked, 0, readback, 0, bytes);
        } else if (!spent && !filling) {
            enc.copyBufferToBuffer(head ? s.logits : s.x, 0, readback, 0, bytes);
        }
        prof?.finish(enc);
        const cmd = enc.finish();
        const t1 = performance.now();
        device.queue.submit([cmd]);
        const t2 = performance.now();
        if (!one) {
            this.counts.batched++;
            this.counts.batchedJobs += plans.length;
        }
        for (const { h, T: n, cache, verify } of plans) {
            if (h.s === WARMUP) continue;
            for (let p = first; p < last; p++) {
                cache.fed.set(p - 1, h.past + n);
                if (verify) cache.specAt.set(p - 1, { base: h.past, n });
                else cache.specAt.delete(p - 1);
            }
        }

        // A replay that reached the end returns neither a token nor a hidden
        // state. route.js uses that to tell the three cases apart.
        if (spent) return plans.map(() => ({ value: { at: to } }));

        // The timestamps are mapped in parallel with the readback, so
        // profiling adds no latency. The readback is read first. A prefill
        // chunk at the end waits for the GPU instead, so jobs still run one
        // at a time.
        await Promise.all([filling ? device.queue.onSubmittedWorkDone()
                                   : readback.mapAsync(GPUMapMode.READ, 0, bytes), prof?.map()]);
        const t3 = performance.now();
        let out = null;
        if (!filling) {
            out = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
            readback.unmap();
        }
        const t4 = performance.now();
        const gpu = prof?.collect();
        const c1 = counters();
        const pass = { out, M, topk, t0, t1, t2, t3, t4, c0, c1, gpu, batch: plans.length };
        const results = [];
        for (let i = 0; i < plans.length; i++) {
            try { results.push({ value: await this.#finish(plans[i], queued[i], pass) }); }
            catch (err) { results.push({ err }); }
        }
        return results;
    }

    // One job's part of a pass, after its readback: sample at the head, or
    // log and hand on the rows it leaves with.
    async #finish(p, queuedAt, { out, M, topk, t0, t1, t2, t3, t4, c0, c1, gpu, batch }) {
        const { d } = this;
        const H = d.hidden;
        const { h, T, at, to, head, filling, verify, drafts, cache, mtp, optsOf } = p;
        const tf = performance.now();

        // Where this job's time went, on this machine's clock. `encode` is the
        // CPU cost of launching the work (cache lookup, uploads, recording
        // dispatches). `wait` is from submit until the readback mapped. The
        // counters are dispatches recorded and bind-group cache misses. In a
        // batch these are the pass's, shared by its jobs.
        const ms = (a, b) => Math.round((b - a) * 100) / 100;
        const phases = {
            queue: ms(queuedAt, t0), encode: ms(t0, t1), submit: ms(t1, t2),
            wait: ms(t2, t3), copy: ms(t3, t4),
            dispatches: c1.dispatches - c0.dispatches,
            bindGroups: c1.bindGroupsBuilt - c0.bindGroupsBuilt,
            ...(batch > 1 && { batch }),
        };
        const timed = () => this.#timed(h, queuedAt, t0, t3, to - at, batch);
        if (filling) {
            timed();
            return { at: to, filled: true, phases, ...(gpu && { gpu }) };
        }

        // Sample row i of this job's sampled rows: from its candidates, or
        // from the whole logits row.
        const ids = topk ? new Uint32Array(out.buffer, out.byteOffset, out.length) : null;
        const sampleRow = (i) => {
            const r = p.sampledOff + i;
            if (!topk) return sampleFromLogitsRow(out.subarray(r * d.vocab, (r + 1) * d.vocab), optsOf(i));
            const c = [], v = [];
            for (let m = 0; m < M; m++) { v.push(out[(r * M + m) * 2]); c.push(ids[(r * M + m) * 2 + 1]); }
            return sampleFromCandidates(c, v, optsOf(i), d.vocab);
        };

        // `at` is the cursor's new position. The caller routes on it.
        if (head && verify) {
            // Row i predicts the token after ids[i]. Each is sampled exactly as
            // plain decoding would sample it: with the repetition summary for
            // the context up to it (`rowOpts[i - 1]`, built assuming the drafts
            // before it were accepted). The draft is accepted while the sample
            // matches it. The first mismatch is the next token, and so is the
            // sample after the last draft when every draft is accepted.
            const tokens = [];
            for (let i = 0; i < T; i++) {
                const t = sampleRow(i);
                tokens.push(t);
                if (i === T - 1 || t !== drafts[i]) break;
            }
            phases.sample = ms(tf, performance.now());
            const next = mtp ? await this.#mtpAfter(h, cache, tokens.length - 1, tokens.at(-1), p.rowOff) : null;
            if (mtp) phases.mtp = ms(tf, performance.now()) - phases.sample;
            timed();
            return { at: to, token: tokens[tokens.length - 1], tokens, phases, ...(gpu && { gpu }),
                     ...(next && { mtp: next }) };
        }
        if (head) {
            // h.opts already holds the repetition summary, computed from the
            // token history by the enqueuer or by the head in a closed loop
            // (route.js's jobOpts()).
            const token = sampleRow(0);
            phases.sample = ms(tf, performance.now());
            const next = mtp ? await this.#mtpAfter(h, cache, T - 1, token, p.rowOff) : null;
            if (mtp) phases.mtp = ms(tf, performance.now()) - phases.sample;
            timed();
            return { at: to, token, phases, ...(gpu && { gpu }), ...(next && { mtp: next }) };
        }
        // The rows leave as half bits, rounded once here: the next peer, the
        // boundary log and any replay of it all see the same values.
        const rows = batch > 1 ? out.subarray(p.rowOff * H, (p.rowOff + T) * H) : out;
        const bits = toF16(rows);
        this.#log(h.s, to, h.past, bits);
        phases.log = ms(tf, performance.now());
        timed();
        return { at: to, hidden: bits, phases, ...(gpu && { gpu }), ...(h.s === WARMUP && { f32: out }) };
    }

    // Update the timings route.js balances on, after a job. `t0` is when it
    // left the queue and `t3` when its readback mapped; what follows is the
    // part that does not scale with positions.
    // A batch's pass time is shared by its jobs, so only a pass of one job
    // updates the speed.
    #timed(h, queuedAt, t0, t3, positions, batch = 1) {
        if (h.s === WARMUP || h.replay) return;
        this.queueAvgMs = 0.8 * this.queueAvgMs + 0.2 * (t0 - queuedAt);
        if (h.n !== 1 || positions < 1 || batch > 1) return;
        const per = (t3 - t0) / positions;
        const over = performance.now() - t3;
        // The first decode token replaces the warm-up estimate.
        this.msPerPos = this.decoded ? 0.9 * this.msPerPos + 0.1 * per : per;
        this.overheadMs = this.overheadMs == null ? over : 0.9 * this.overheadMs + 0.1 * over;
        this.decoded = true;
    }
}
