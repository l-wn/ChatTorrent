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

import { F32, makeStorage, readF32, gatherEmbedNoPos, rmsnorm, matmulNobias, withPass, counters }
    from './kernels.js';
import { block, makeCache, makeScratch, makeRecurrentState, recurrentStateBytes,
         makePagePool, makeSessionFrame, ensurePages, freePages } from './model.js';
import { makeGpuProf } from './gpuprof.js';
import { sampleFromLogitsRow } from './generate.js';
import { positions, blockRange, PAGE_TOKENS, IDLE_MS, WAIT_MS } from '../split.js';

// The session id a warm-up pass runs under. Peer-minted ids are
// `${peerId}-${base36}`, so it cannot collide with a chat.
const WARMUP = '\0warmup';

// Initial row capacity of a boundary log: four prefill chunks.
const LOG_CHUNK = 256;

// Jobs run() holds queued before it refuses more with `busy`. Normal traffic
// is one job in flight per session, so this is several times `sessions`. The
// limit bounds the GPU time and payload memory a flood can take.
export const MAX_QUEUE = 32;

// How recently a session must have run here to count towards `load`. Covers
// the gap between a reply's tokens and a short pause between turns.
export const ACTIVE_MS = 5000;

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
                                idleMs = IDLE_MS, waitMs = WAIT_MS }) {
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
        this.pool = makePagePool(device, d, {
            pageTokens: this.pageTokens,
            nPages: slots * this.maxPages,
            n_layer: this.blocks,
            layerLo: this.blockLo,
        });
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
        this.queue = Promise.resolve();
        this.backlog = 0;                                   // jobs queued or running
        // Jobs run() has accepted and not yet settled, by session. A session
        // with any is not idle, however long its job waits in the queue.
        this.pending = new Map();
        // Milliseconds per position for one decode token, averaged. Null
        // until the first decode job. route.js compares peers' speeds with it.
        this.msPerPos = null;
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
        this.counts = { admitted: 0, evicted: 0, cold: 0, released: 0, refused: 0, waited: 0 };
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
            + Math.max(this.tMax * F32, 16);
        for (const frame of this.free) bytes += frame.bytes;
        for (const frame of this.live.values()) bytes += frame.bytes;
        if (this.warm) bytes += this.warm.bytes;
        return bytes;
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
        return this.d.n_kv_heads * this.pool.pageTokens * this.d.head_dim * F32
            * this.pool.nPages;
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
    // Refused with `busy` once MAX_QUEUE jobs are queued. A job waiting for a
    // frame counts towards the limit.
    run(h, hidden, want) {
        if (this.backlog >= MAX_QUEUE) return Promise.reject(this.#atCapacity('queue slot'));
        this.backlog++;
        const s = h.s;
        this.pending.set(s, (this.pending.get(s) ?? 0) + 1);
        return new Promise((resolve, reject) => {
            this.#submit({ h, hidden, want, queuedAt: performance.now(), resolve, reject });
        }).finally(() => {
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
        this.queue = this.queue.then(() =>
            this.#run(job.h, job.hidden, job.want, job.queuedAt).then(job.resolve, (err) => {
                if (err === WAIT) this.#park(job);
                else job.reject(err);
            }));
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
            for (const j of late) j.reject(this.#atCapacity('session'));
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
        this.queue = this.queue.then(() => { this.turn = null; });
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
                : r.hidden;
            if (!out.every(Number.isFinite)) {
                throw new Error(`a warm-up pass through positions ${from}–${r.at - 1} produced ` +
                    'non-finite output. This GPU cannot run them ' +
                    '(reduced-precision accumulators?)');
            }

            const ms = performance.now() - t0;
            const blocks = r.at - from;
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
        const done = this.queue.then(() => this.#release(session));
        this.queue = done.catch(() => {});
        return done;
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
        this.cuts.delete(session);
        this.logs.delete(session);
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
    get load() {
        const now = performance.now();
        let n = 0;
        for (const [s, frame] of this.live) {
            if (!this.pending.has(s) && now - (frame.lastUse ?? 0) >= ACTIVE_MS) continue;
            for (const [at, to] of this.cuts.get(s) ?? []) n += to - at;
        }
        return n;
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
    #atCapacity(what) {
        const err = new Error(`this peer is serving ${this.live.size} chats and has no ` +
            `${what} free for another right now; try again in a moment`);
        err.busy = true;
        this.counts.refused++;
        return err;
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
        if (past > 0) {
          this.counts.cold++;
          throw new Error(`session ${session} is no longer cached here`);
        }
        // First come, first served: a new session does not take a frame while
        // other sessions are waiting for one.
        if (this.waiting.length && this.turn !== session) throw WAIT;

        let cache = this.free.pop();
        if (!cache) {
            const oldest = this.#evictable();
            if (oldest == null) throw this.waitMs > 0 ? WAIT : this.#atCapacity('session');
            this.counts.evicted++;
            cache = this.live.get(oldest);
            this.live.delete(oldest);
            // Its pages, cut and boundary log are dropped too. An evicted
            // session cannot resume (its next job throws). Dropping the logs
            // here bounds them to `sessions`.
            freePages(this.pool, cache);
            this.cuts.delete(oldest);
            this.logs.delete(oldest);
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
        // Rows each block holds for this session. See #checkFed().
        cache.fed = new Map();

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
    // Keyed by global block index, so growth on either side does not shift it.
    // A mismatch throws the same error as an evicted cache and is handled the
    // same way: passed back, repaired locally, or rebuilt by the enqueuer.
    //
    // A job the head fed (`auto`, route.js's closed loop) must continue every
    // block exactly, since only the enqueuer rewinds. If it is behind, it is a
    // duplicate from a repair re-sending a hand-off that had arrived. It
    // throws `stale`, which route.js drops without answering.
    #checkFed(session, cache, at, last, past, exact = false) {
        const reset = [];
        for (let p = Math.max(at, 1); p < last; p++) {
            const b = p - 1;
            const fed = cache.fed.get(b) ?? 0;
            const rc = cache.recurrent[b - this.blockLo];
            if (past === 0 && !exact) { if (rc && fed > 0) reset.push(rc); continue; }
            if (rc || exact ? past === fed : past <= fed) continue;
            if (exact && past < fed) {
                throw Object.assign(new Error(`a duplicate of session ${session}'s job at ${past} ` +
                    `(block ${b} is already at ${fed})`), { stale: true });
            }
            this.counts.cold++;
            throw new Error(`session ${session} is no longer cached here ` +
                `(block ${b} holds ${fed} rows, the job continues from ${past})`);
        }
        return reset;
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
        throw this.#atCapacity('KV page');
    }

    // The rows handed downstream at `at`, kept for a replay. Rows are stored
    // at their absolute position, so a job at a lower `past` (after a rewind)
    // overwrites the same rows it overwrites in the downstream KV caches, and
    // `len` is set to match.
    //
    // Allocated on first use and doubled from LOG_CHUNK up to `nMax`, so a
    // short chat uses little host RAM.
    #log(s, at, past, rows) {
        if (s === WARMUP) return;              // a warm-up hands its rows to nobody
        const H = this.d.hidden;
        let boundaries = this.logs.get(s);
        if (!boundaries) this.logs.set(s, boundaries = new Map());
        let log = boundaries.get(at);
        if (!log) {
            const cap = Math.min(LOG_CHUNK, this.nMax);
            boundaries.set(at, log = { rows: new Float32Array(cap * H), len: 0 });
        }

        const need = past + rows.length / H;
        if (need * H > log.rows.length) {
            let cap = log.rows.length / H;
            while (cap < need) cap = Math.min(this.nMax, cap * 2);
            const grown = new Float32Array(cap * H);
            grown.set(log.rows);               // rows are at absolute positions
            log.rows = grown;
        }
        log.rows.set(rows, past * H);
        log.len = need;
    }

    async #run(h, hidden, want, queuedAt = performance.now()) {
        const { device, d, scratch: s } = this;
        const T = h.n, H = d.hidden;
        const t0 = performance.now();
        const c0 = counters();

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
        const head = end && !h.replay;
        const spent = end && !!h.replay;
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
        const cache = this.#cacheFor(h.s, h.past);
        // The block positions this job runs: from the cursor (or position 1,
        // if the cursor is the embedding) up to the session's cut or the head,
        // whichever is first.
        const first = Math.max(at, 1), last = Math.min(to, d.n_layer + 1);
        const reset = h.s === WARMUP ? [] : this.#checkFed(h.s, cache, at, last, h.past, !!h.auto);
        // Before any dispatch: scatter_kv writes rows h.past .. h.past+T-1, so
        // their pages must be allocated and in the table. The warm-up cache is
        // contiguous and not pooled.
        if (h.s !== WARMUP) this.#pagesFor(h.s, cache, h.past + T);

        // writeBuffer is ordered on the queue, so these are applied before the
        // command buffer below runs.
        if (gather) device.queue.writeBuffer(this.ids, 0, Float32Array.from(h.ids));
        else        device.queue.writeBuffer(s.x, 0, hidden);

        // GPU timing requested by the enqueuer (gpuprof.js). Not for a replay
        // or a warm-up: neither produces a token.
        const prof = h.replay || h.s === WARMUP ? null : makeGpuProf(device, h.prof);

        const enc = device.createCommandEncoder({ label: `positions ${at}–${to - 1}` });
        // Clear recurrent state that holds rows when the job restarts from
        // `past` 0 (see #checkFed), before the pass that reads it.
        for (const rc of reset) {
            for (const buf of [rc.S, rc.histQ, rc.histK, rc.histV]) enc.clearBuffer(buf);
        }
        withPass(enc, (pass) => {
            prof?.at('embed');
            if (gather) gatherEmbedNoPos(device, pass, this.ids, this.w.wte, s.x, T, H);

            // `w.layers` and the cache are both indexed from `blockLo`:
            // position `p` is block `p - 1`, which is layer `p - 1 - blockLo`.
            for (let p = first; p < last; p++) {
                const layer = p - 1 - this.blockLo;
                prof?.at(this.w.layers[layer].kind === 'linear_attention' ? 'linear' : 'full');
                block(device, pass, d, this.w, layer, cache, s, T, h.past);
            }

            // Sample when the cut reaches the end of the line. A peer that
            // holds the head can stop before it for a session whose cut is
            // earlier.
            if (head) {
                prof?.at('head');
                rmsnorm(device, pass, s.x, this.w.norm, s.h, T, H, d.rms_eps);
                matmulNobias(device, pass, s.h, this.w.lm_head ?? this.w.wte,
                             s.logits, T, H, d.vocab);
            }
        }, `positions ${at}–${to - 1}`, prof);

        // The readback copy is on the same encoder, outside the pass, in the
        // same submit. A job that ends at the head reads back only the last
        // logits row and samples it. Any other job reads back the hidden
        // state. Prefill chunks sample too; the enqueuer keeps only the last
        // chunk's token.
        const rowBytes = d.vocab * F32;
        const bytes = head ? rowBytes : T * H * F32;
        if (!spent) {
            enc.copyBufferToBuffer(head ? s.logits : s.x, head ? (T - 1) * rowBytes : 0,
                                   s.readback, 0, bytes);
        }
        prof?.finish(enc);
        const cmd = enc.finish();
        const t1 = performance.now();
        device.queue.submit([cmd]);
        const t2 = performance.now();
        if (h.s !== WARMUP) for (let p = first; p < last; p++) cache.fed.set(p - 1, h.past + T);

        // A replay that reached the end returns neither a token nor a hidden
        // state. route.js uses that to tell the three cases apart.
        if (spent) return { at: to };

        // The timestamps are mapped in parallel with the readback, so
        // profiling adds no latency. The readback is read first.
        await Promise.all([s.readback.mapAsync(GPUMapMode.READ, 0, bytes), prof?.map()]);
        const t3 = performance.now();
        const out = new Float32Array(s.readback.getMappedRange(0, bytes).slice(0));
        s.readback.unmap();
        const t4 = performance.now();
        const gpu = prof?.collect();
        if (T === 1 && !h.replay && h.s !== WARMUP) {
            const per = (t3 - t0) / (to - at);
            this.msPerPos = this.msPerPos == null ? per : 0.9 * this.msPerPos + 0.1 * per;
        }

        // Where this job's time went, on this machine's clock. `encode` is the
        // CPU cost of launching the work (cache lookup, uploads, recording
        // dispatches). `wait` is from submit until the readback mapped. The
        // counters are dispatches recorded and bind-group cache misses.
        const c1 = counters();
        const ms = (a, b) => Math.round((b - a) * 100) / 100;
        const phases = {
            queue: ms(queuedAt, t0), encode: ms(t0, t1), submit: ms(t1, t2),
            wait: ms(t2, t3), copy: ms(t3, t4),
            dispatches: c1.dispatches - c0.dispatches,
            bindGroups: c1.bindGroupsBuilt - c0.bindGroupsBuilt,
        };

        // `at` is the cursor's new position. The caller routes on it.
        if (head) {
            // h.opts already holds the repetition summary, computed from the
            // token history by the enqueuer or by the head in a closed loop
            // (route.js's jobOpts()).
            const token = sampleFromLogitsRow(out, h.opts);
            phases.sample = ms(t4, performance.now());
            return { at: to, token, phases, ...(gpu && { gpu }) };
        }
        this.#log(h.s, to, h.past, out);
        phases.log = ms(t4, performance.now());
        return { at: to, hidden: out, phases, ...(gpu && { gpu }) };
    }
}
