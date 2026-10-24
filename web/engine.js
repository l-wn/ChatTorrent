// The engine: joins the mesh, optionally serves and grows a claim, and runs
// generation. It sits between a caller's prompt and the wire protocol
// (net.js/route.js). It has no DOM access and does not know the chat template:
// the caller wraps its own turns and detects its own stop sequences. The
// engine deals in tokens, blocks and positions.
//
// Background events (growth, a repair, a lost GPU device) are reported through
// one `onEvent(type, detail)` callback.
//
// One Engine is one mesh identity and one active generation at a time.

import { adapterLabel } from './model/kernels.js';
import { checkDims, scratchBytes } from './model/model.js';
import { loadManifest, loadBlock, loadEmbed, loadHead, dimsFromManifest } from './model/loader.js';
import { Runner } from './model/stage.js';
import { encode, decode, source as tokenizerSource } from './model/tokenizer.js';
import { runAllTests } from './dev/kernels.test.js';
import { smokeForward } from './dev/smoke.js';
import { Mesh } from './net.js';
import { makeRouter, makeConversation, T_MAX } from './route.js';
import { blockBytes, endBytes, floorBlocks, positions, describeRange, N_MAX, SLOTS,
         SESSIONS, runnerSessions } from './split.js';

// How long a Runner must be idle before #growClaim() takes another step, so
// downloads and warm-ups do not compete with live chats.
const GROWTH_QUIET_MS = 2000;

// A growth step whose download fails is retried after this long, doubling up
// to the second value, while the tab keeps serving what it has. Only a GPU out
// of memory or a permanent failure (a 4xx) ends growth.
const GROW_RETRY_MS = [15_000, 300_000];

// Bytes formatted as the VRAM pill shows them (web/peer.js's refresh), so a
// status note and the pill print the same number the same way.
const gib = (b) => `${(b / 2 ** 30).toFixed(2)} GiB`;

export class ContextFullError extends Error {
    constructor(needed, room) {
        super(`this prompt needs ${needed} tokens and only ${room} are left in the ` +
            `${N_MAX}-token context`);
        this.needed = needed;
        this.room = room;
    }
}

export class Engine {
    #net = null;
    #router = null;
    #runner = null;
    #manifest = null;
    #baseUrl = null;
    #dims = null;
    #role = 'joining';       // 'joining' -> 'worker' | 'spectator'
    #canPrompt = null;
    #stopRequested = false;
    #generating = false;     // a generate() is running; chats cannot be switched
    #vramLimit = Infinity;
    #growWake = null;        // resolve() of the promise a paused #growClaim() awaits
    #growPausedUnder = null; // the limit the pause note was emitted for; null while growing
    #sp = 0;                 // blocks/sec measured by warmup() in #fill()
    #lastStopReason = null;  // 'length' | 'context' | 'eos' | 'stopped', set by generate()
    #lastPromptTokens = 0;
    #lastFed = 0;            // generated tokens the last generate() left in the context
    #stream = null;          // the reply being streamed, if any
    #lastProfile = null;     // { level, samples }: one sample per job of the last generate()
    #gpuLabel = null;        // this tab's adapter name, reported on a profiled hop
    #onEvent;
    #growRetryMs = GROW_RETRY_MS;
    #loadOverride = null;    // test-only replacement for #load

    constructor({ onEvent = () => {} } = {}) {
        this.#onEvent = onEvent;
    }

    // Test-only: build an Engine around an existing net/router/runner,
    // skipping connect()/serve(). Used by tools/engine.test.mjs and
    // tools/grow.test.mjs.
    //
    // `grow` starts the background growth loop that serve() starts once a
    // claim is downloaded, on `device`/`dims`. `vramLimit` is applied first, so
    // the limit decides the first step. `load` replaces the download of one
    // position and `growRetryMs` the retry delays, so a test can make a
    // growth download fail and then succeed.
    static forTest({ net, router, runner = null, dims = null, device = null,
                     vramLimit = Infinity, grow = false, onEvent = () => {},
                     load = null, growRetryMs = GROW_RETRY_MS } = {}) {
        const engine = new Engine({ onEvent });
        engine.#loadOverride = load;
        engine.#growRetryMs = growRetryMs;
        engine.#net = net;
        engine.#router = router;
        engine.#runner = runner;
        engine.#dims = dims;
        engine.#canPrompt = true;
        engine.#role = runner ? 'worker' : 'spectator';
        engine.#vramLimit = vramLimit;
        if (grow) engine.#growClaim(device, dims).catch((err) => console.error('growClaim:', err));
        return engine;
    }

    #emit(type, detail = {}) { this.#onEvent(type, detail); }

    // --- Joining ---

    // Join the mesh and load what every role needs. Resolves once a prompt
    // could be sent. Rejects if the server refuses this tab or the checkpoint
    // manifest cannot be read. A tokenizer failure does not reject, since
    // this tab can still serve blocks; check `canPrompt` on the resolved
    // value.
    //
    // `room` is the mesh to join (web/room.js's openRoom), the global one by
    // default.
    //
    // `relayOnly` (default true) sends every link through a TURN relay, so no
    // peer sees this tab's address (see web/net.js). `chats` is the limit on
    // replies at once, and `sessions` the chats each peer keeps in memory, to
    // ask for if this join creates a private mesh.
    async connect({ room, relayOnly = true, chats = null, sessions = null } = {}) {
        const net = new Mesh({
            room,
            chats,
            sessions,
            relayOnly,
            onMessage: (h, payload, raw, from) => this.#router.route(h, payload, raw, from),
            onChange: () => this.#emit('change'),
        });
        this.#net = net;
        this.#router = makeRouter({
            net,
            // The scope chats are stored and checked under. Not the invite.
            mesh: room?.historyTag ?? 'global',
            runner: () => this.#runner,
            onChange: () => this.#emit('change'),
            // `reason`: 'repair' (a peer left), 'resume' (a parked or stored
            // chat), 'rewind' (see rewind()). Only a repair has an error.
            onRepair: (err, tokens, reason) => {
                this.#emit('repair', { message: err?.message ?? null, tokens, reason });
                return () => this.#emit('repaired', { reason });
            },
            onFix: (h) => this.#emit('fix', { from: h.from, rows: h.rows }),
            onProgress: (sent, total) => this.#emit('rebuild-progress', { sent, total }),
            isCancelled: () => this.#stopRequested,
            gpu: () => this.#gpuLabel,
        });
        await net.ready;

        this.#baseUrl = net.weightsUrl ?? undefined;
        if (this.#baseUrl) tokenizerSource(this.#baseUrl);

        let manifest;
        try {
            manifest = await loadManifest(this.#baseUrl);
        } catch (err) {
            this.#role = 'spectator';
            this.#canPrompt = false;
            throw err;
        }
        this.#manifest = manifest;
        this.#router.eos = manifest.eos_token_id ?? 0;
        this.#router.hybrid = !!dimsFromManifest(manifest).hybrid;

        this.#canPrompt = await encode('warm').then(() => true, (err) => {
            console.warn('no tokenizer:', err);
            return false;
        });

        return { manifest, canPrompt: this.#canPrompt };
    }

    // --- Serving ---

    // Try to hold a range of the model. Unless the caller fixed a size or
    // placement, keep growing it in the background. Never rejects: on failure
    // this tab is a spectator or a partial worker, and the resolved value says
    // which.
    //
    // `blocks`/`vram`/`at` are plain values. The caller reads them from a form
    // field or URL parameter.
    async serve({ blocks = null, vram = null, at = null, slowMs = 0, floor = null,
                  sessions = undefined, pageTokens = undefined, waitMs = undefined } = {}) {
        const { net, manifest, baseUrl } = this;
        // Chats kept in memory: the caller's number (the dev harness), else the
        // mesh's from the welcome. Read once: a later welcome cannot resize a
        // Runner that is already allocated.
        let slots;
        ({ sessions, slots } = runnerSessions(sessions, net?.sessions));
        try {
            const d = dimsFromManifest(manifest);
            const problems = checkDims(d);
            if (problems.length) throw new Error(`${manifest.model}: ${problems.join('; ')}`);
            this.#dims = d;

            if (!net.blocks || !net.positions) {
                throw new Error('the signaling server did not say how the model is split. ' +
                    'Restart it (./start.sh restart).');
            }
            if (d.n_layer !== net.blocks) {
                throw new Error(`the signaling server is splitting a ${net.blocks}-block model but ` +
                    `${manifest.model} has ${d.n_layer}. Set N_LAYER=${d.n_layer} on the server.`);
            }

            this.#emit('progress', { text: 'checking this GPU…' });
            const device = await this.#openDevice();
            await this.#selfCheck(device);

            this.#emit('progress', { text: 'asking for a range of blocks…' });
            const cap = at ? at[1] - at[0]
                : this.#capacity(d, net, { blocks, vram, floor, sessions, slots });
            const range = await net.claim(cap, at);
            if (!range) {
                throw new Error(`this tab offered ${cap} blocks and the network's minimum is ` +
                    `${net.minBlocks}`);
            }
            const [lo, hi] = range;
            if (at && (lo !== at[0] || hi !== at[1])) {
                throw new Error(`asked for positions ${at[0]}–${at[1] - 1} but the server placed ` +
                    `this tab at ${lo}–${hi - 1}. The requested range does not fit this model.`);
            }
            const held = describeRange(lo, hi, d.n_layer);
            this.#emit('progress', { text: `downloading ${held} (you can chat meanwhile)` });

            await this.#fill(device, d, lo, hi, slowMs, sessions, slots, pageTokens, waitMs);

            const note = `Holding ${held} of ${manifest.model}: ${hi - lo} of ` +
                `${positions(d.n_layer)} positions. This tab runs that stretch of every job ` +
                `routed to it, and can send prompts of its own.`
                + (at ? ' Placed by hand (?at=), not by the server. A mesh that looks '
                      + 'broken may have a hole nobody was asked to fill.' : '');

            // `blocks`/`vram`/`at` fix the holding, so they turn growth off.
            // `floor` only sets the starting size, so growth stays on.
            const autoGrow = !at && blocks == null && vram == null;
            if (autoGrow) {
                this.#growClaim(device, d).catch((err) => console.error('growClaim:', err));
            }
            return { role: 'worker', held: [lo, hi], note, error: false };
        } catch (err) {
            console.warn('serve stopped:', err);
            const r = this.#runner;

            // A failure part way through leaves a working peer: everything
            // below `have` is downloaded, warmed up, announced and routable.
            if (r && r.have > r.lo) {
                this.#role = 'worker';
                const now = describeRange(r.lo, r.have, r.d.n_layer);
                net.releaseTail(r.have);
                return {
                    role: 'worker', held: [r.lo, r.have], error: false,
                    note: `Stopped growing at ${now} (${err.message}). This tab keeps serving ` +
                        `what it has; the rest of its claim has been given back to the network.`,
                };
            }
            this.#role = 'spectator';
            this.#runner = null;
            net.claim(0).catch(() => {});
            return {
                role: 'spectator', held: null, error: true,
                note: `Spectator: this tab holds no blocks and will never be routed a job ` +
                    `(${err.message}). It can still prompt the mesh like any other tab.`,
            };
        } finally {
            this.#emit('change');
        }
    }

    // The limit background growth pauses under. #growClaim() re-reads it on
    // every step, so a lower limit takes effect at the next one; a higher one
    // also wakes a loop parked at the limit, which grows on from the holding
    // already in VRAM. A tab pinned with ?at=/?blocks=/?gb= has no loop to
    // wake, so its holding stays fixed.
    setVramLimit(bytes) {
        this.#vramLimit = bytes;
        const wake = this.#growWake;
        this.#growWake = null;
        wake?.();
    }

    get manifest() { return this.#manifest; }
    get baseUrl() { return this.#baseUrl; }

    // How many positions to claim: an explicit block count, a VRAM budget in
    // bytes, or by default the network's minimum, which background growth then
    // extends.
    #capacity(d, net, { blocks, vram, floor = null, sessions = SESSIONS, slots = SLOTS }) {
        const N = net.positions || positions(d.n_layer);
        const clamp = (n) => Math.max(0, Math.min(Math.floor(n) || 0, N));
        if (blocks != null) return clamp(blocks);
        if (vram != null) {
            // The range is not known yet, so blocks are priced at their mean
            // cost and one end is assumed (a claim short of the whole model
            // holds at most one). The scratch is needed whatever is held.
            const fixed = scratchBytes(d, T_MAX, N_MAX)
                + Math.max(endBytes(d, 'embed'), endBytes(d, 'head'));
            return clamp((vram - fixed) / blockBytes(d, { sessions, slots }));
        }
        return clamp(floor ?? floorBlocks(d, N));
    }

    // Download a claim one position at a time, serving each as soon as it is
    // loaded. Ascending from `lo`, because a holding is the contiguous prefix
    // `[lo, have)`.
    async #fill(device, d, lo, hi, slowMs, sessions = SESSIONS, slots = SLOTS,
                pageTokens = undefined, waitMs = undefined) {
        const { net } = this;
        const L = d.n_layer, N = positions(L);

        device.pushErrorScope('out-of-memory');
        device.pushErrorScope('validation');
        const runner = new Runner(device, d, { layers: [] },
            { lo, hi, have: lo, nMax: N_MAX, tMax: T_MAX, slots, sessions,
              ...(pageTokens ? { pageTokens } : {}),
              ...(waitMs !== undefined ? { waitMs } : {}) });
        const validation = await device.popErrorScope();
        const oom = await device.popErrorScope();
        if (oom || validation) {
            throw new Error(`GPU ran out of memory reserving the KV cache for positions `
                + `${lo}-${hi - 1} (${(oom ?? validation).message || 'out of memory'}). This `
                + `GPU cannot hold as many blocks as were claimed for it.`);
        }
        this.#runner = runner;

        for (let p = lo; p < hi; p++) {
            if (slowMs && p > lo) {
                this.#emit('progress', { text: `?slow=${slowMs}: holding at ` +
                    `${describeRange(lo, p, L)} before fetching the next position…` });
                await new Promise((r) => setTimeout(r, slowMs));
            }
            const what = describeRange(p, p + 1, L);
            const step = (b, t) => {
                this.#emit('progress', {
                    text: t ? `${what}: ${(b / 2 ** 20).toFixed(0)}/${(t / 2 ** 20).toFixed(0)} MiB`
                            : `fetching ${what}…`,
                    fraction: (p - lo + (t ? b / t : 0)) / (hi - lo),
                });
            };
            step(0, 0);

            const weights = await this.#load(device, d, p, step);

            this.#runner.append(weights);
            const warm = await this.#runner.warmup(p);
            if (warm) this.#sp = Math.max(this.#sp, warm.blocksPerSec);

            net.announceHave(p + 1);
            if (p === lo) {
                this.#role = 'worker';
                net.announceReady();
            }
            this.#emit('progress', {
                text: p + 1 < hi ? `serving ${describeRange(lo, p + 1, L)}, still downloading the rest`
                                  : 'done',
            });
            this.#emit('change');
        }
    }

    // After the initial claim is downloaded, add one position at a time toward
    // the less-covered neighbour (alternating on a tie) whenever the runner
    // has been idle. Stops when an allocation fails, a download fails for good
    // (a 4xx), or the whole model is held. A download that fails for any
    // other reason is retried (#loadForGrowth).
    // At the VRAM limit it pauses instead, and setVramLimit() wakes it, so a
    // raised limit resumes growth from the holding already in VRAM.
    async #growClaim(device, d) {
        const { manifest, net } = this;
        const L = d.n_layer, N = positions(L);
        let side = 'left';
        for (;;) {
            const r = this.#runner;
            if (!r) return;
            if (r.lo === 0 && r.hi === N) {
                this.#emit('status', {
                    text: `Holding the whole of ${manifest?.model ?? 'the model'}. Nothing left to grow into.`,
                });
                return;
            }

            const idle = performance.now() - r.lastJobAt;
            if (idle < GROWTH_QUIET_MS) {
                await new Promise((resolve) => setTimeout(resolve, GROWTH_QUIET_MS - idle));
                continue;
            }

            const canLeft = r.lo > 0, canRight = r.hi < N;
            const depths = net.coverage.depths;
            const leftDepth = canLeft ? depths[r.lo - 1] : Infinity;
            const rightDepth = canRight ? depths[r.hi] : Infinity;
            const dir = !canLeft ? 'right'
                : !canRight ? 'left'
                : leftDepth < rightDepth ? 'left'
                : rightDepth < leftDepth ? 'right'
                : side;

            try {
                // The cost of the position being added: that block's weights
                // and its KV or recurrent state, or an end's table.
                const next = dir === 'left' ? r.lo - 1 : r.hi;
                const needed = r.vramBytes + this.#cost(d, r, next);
                if (needed > this.#vramLimit) {
                    // The loop parks until setVramLimit() wakes it, so a
                    // raised limit grows on from what is already in VRAM. One
                    // note per limit value, so a visitor trying several values
                    // gets one note each rather than one per step.
                    if (this.#growPausedUnder !== this.#vramLimit) {
                        this.#growPausedUnder = this.#vramLimit;
                        this.#emit('status', {
                            text: `Paused growing at ${describeRange(r.lo, r.have, L)}: the next ` +
                                `position would take this tab to ${gib(needed)}, past its ` +
                                `${gib(this.#vramLimit)} VRAM limit. Raise the limit, or clear it, ` +
                                `and growth carries on from here — nothing already held is ` +
                                `dropped or downloaded again.`,
                        });
                    }
                    await new Promise((resolve) => { this.#growWake = resolve; });
                    continue;
                }
                if (this.#growPausedUnder !== null) {
                    this.#growPausedUnder = null;
                    this.#emit('status', {
                        text: `The VRAM limit has room again: growing on from ` +
                            `${describeRange(r.lo, r.have, L)}.`,
                    });
                }

                const p = dir === 'left' ? r.lo - 1 : r.hi;
                device.pushErrorScope('out-of-memory');
                device.pushErrorScope('validation');
                const reserved = dir === 'left' ? r.growLeft() : (r.growRight(), null);
                const validation = await device.popErrorScope();
                const oom = await device.popErrorScope();
                if (oom || validation) {
                    throw new Error(`GPU ran out of memory extending the KV cache for position ` +
                        `${p} (${(oom ?? validation).message || 'out of memory'})`);
                }

                const weights = await this.#loadForGrowth(device, d, p, r);

                if (dir === 'left') {
                    r.prepend(weights, reserved);
                    await r.warmup(r.lo, r.lo + 1);
                } else {
                    r.append(weights);
                    await r.warmup(p);
                }

                // Announced only once the position is downloaded and warmed up.
                net.grow(r.lo, r.hi);
                this.#emit('grow', { lo: r.lo, hi: r.hi });
                // Flipped only by a step that happened, so the tie-break
                // alternates between growth steps. A paused iteration must not
                // consume it, or the direction after a resume would not follow
                // the last step taken.
                side = dir === 'left' ? 'right' : 'left';
            } catch (err) {
                const r2 = this.#runner;
                this.#emit('status', {
                    text: `Stopped growing at ${describeRange(r2.lo, r2.have, L)} (${err.message}). ` +
                        `This tab keeps serving what it has.`,
                });
                return;
            }
        }
    }

    // Download one position for a growth step. A failed download (a stall, a
    // dropped connection, a 5xx) is retried with backoff, keeping the KV
    // already reserved for it, while the tab serves what it has. A GPU out of
    // memory, a permanent failure, or a replaced runner ends the step.
    async #loadForGrowth(device, d, p, r) {
        const L = d.n_layer;
        let wait = this.#growRetryMs[0];
        for (;;) {
            try {
                return await this.#load(device, d, p);
            } catch (err) {
                if (err.permanent || /ran out of memory/.test(err.message) || this.#runner !== r) throw err;
                this.#emit('status', {
                    text: `Downloading ${describeRange(p, p + 1, L)} failed (${err.message}). ` +
                        `Trying again in ${Math.round(wait / 1000)}s; this tab keeps serving ` +
                        `${describeRange(r.lo, r.have, L)}.`,
                });
                await new Promise((resolve) => setTimeout(resolve, wait));
                if (this.#runner !== r) throw err;
                wait = Math.min(wait * 2, this.#growRetryMs[1]);
            }
        }
    }

    // Upload one position's weights. An end loads only what it needs: `wte`
    // for the embedding, the final norm and LM head for the head. It reuses
    // tensors the runner already holds, so a peer holding both ends stores a
    // tied table once.
    #load(device, d, p, onBytes) {
        if (this.#loadOverride) return this.#loadOverride(p);
        const { manifest, baseUrl } = this;
        const have = this.#runner?.w;
        if (p === 0) return loadEmbed(device, { baseUrl, manifest, d, onBytes, have });
        if (p === positions(d.n_layer) - 1) return loadHead(device, { baseUrl, manifest, d, onBytes, have });
        return loadBlock(device, { baseUrl, manifest, d, index: p - 1, onBytes });
    }

    // What position `p` will add to `r.vramBytes`, from the manifest's file
    // sizes (split.js). #growClaim() checks it against the VRAM limit before
    // fetching.
    #cost(d, r, p) {
        const tied = d.tied_lm_head !== false;
        if (p === 0) return tied && r.w.lm_head ? 0 : endBytes(d, 'embed');
        if (p === positions(d.n_layer) - 1) return tied && r.w.wte ? 0 : endBytes(d, 'head');
        return blockBytes(d, { nMax: N_MAX, slots: r.slots, sessions: r.sessions, index: p - 1 });
    }

    async #openDevice() {
        if (!navigator.gpu) throw new Error('This browser has no WebGPU, so this tab can chat but '
            + 'cannot hold any of the model. WebGPU needs a recent desktop Chrome or Edge, '
            + 'Safari 26 or later, or Firefox 141 or later on Windows.');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('WebGPU is present but no adapter was returned.');

        const gpu = await adapterLabel(adapter);
        this.#gpuLabel = gpu;
        console.log(`WebGPU adapter: ${gpu}  (maxBufferSize `
            + `${(adapter.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB)`);
        this.#emit('progress', { text: `GPU: ${gpu}` });

        // 2 GiB fits Qwen3.8-27B's ~1.27 GiB q8 wte and lm_head. Clamped to
        // the adapter's limit below.
        const want = 2 * (1 << 30);
        // `timestamp-query` is used only by profiled jobs (model/gpuprof.js).
        // Without it a profile has only CPU phases.
        const features = ['shader-f16', 'timestamp-query'].filter((f) => adapter.features.has(f));
        const device = await adapter.requestDevice({
            requiredFeatures: features,
            requiredLimits: {
                maxBufferSize: Math.min(want, adapter.limits.maxBufferSize),
                maxStorageBufferBindingSize: Math.min(want, adapter.limits.maxStorageBufferBindingSize),
            },
        });
        device.lost.then((info) => {
            if (info.reason === 'destroyed') return;
            this.#emit('status', {
                text: `WebGPU device lost: ${info.message}. Reload the page.`,
                error: true, pill: 'device lost',
            });
        });
        let gpuErrorNoted = false;
        device.addEventListener('uncapturederror', (ev) => {
            console.error('WebGPU error:', ev.error.message);
            if (gpuErrorNoted) return;
            gpuErrorNoted = true;
            this.#emit('status', {
                text: `WebGPU reported an error this tab wasn't watching for `
                    + `(${ev.error.message}). Replies from here on may be corrupted. Reload `
                    + `the page if they start looking wrong.`,
                error: true, pill: 'GPU error',
            });
        });
        return device;
    }

    async #selfCheck(device) {
        const checks = [
            { label: 'kernels', run: () => runAllTests(device) },
            { label: 'the architecture', run: () => smokeForward(device) },
        ];
        for (const [i, { label, run }] of checks.entries()) {
            this.#emit('progress', { text: `checking ${label}…`, fraction: i / checks.length });
            const { ok, rows } = await run();
            if (ok) continue;
            const bad = rows.filter((r) => !r.ok);
            throw new Error(
                `${bad.length} of ${rows.length} checks on ${label} failed on this GPU, so this ` +
                `tab will not join the mesh:\n  ` +
                bad.map((r) => `${r.name}: ${r.detail ?? ''}`).join('\n  '));
        }
    }

    // --- Generation ---

    // Stream a reply to `text`, which the caller has already wrapped in its
    // chat template. Yields one `{ text, tokenId, trace }` per generated
    // token, where `text` is the whole reply decoded so far. Ends on EOS, on
    // `maxTokens` ('length', default Infinity), when the context is full
    // ('context'), or on stop() ('stopped'). `lastStopReason` says which. To
    // end earlier (a stop-sequence match), stop pulling: breaking a
    // `for await` loop finalizes the generator.
    //
    // Every job's timing is kept in `lastProfile` for web/profile.js: `rt` and
    // `trace` from the router, plus `overhead`, the time this tab spent
    // between an answer and sending the next job. `prof` (0/1/2) also asks the
    // peers for GPU timings (see model/gpuprof.js).
    //
    // `loop` (default on) decodes with the closed loop (route.js's decode()):
    // the head feeds each token back in and this tab only receives, so there
    // is no round trip from this tab per token. stop() halts the loop. Tokens
    // already fed (usually one or two) still arrive and are part of the reply.
    // With `loop` off, this tab sends every token back itself.
    //
    // `lastFed` is how many generated tokens the context gained. A caller that
    // cuts the reply short rewinds from it. With the loop it can be more than
    // were yielded.
    async *generate(text, opts = {}) {
        this.#generating = true;
        const start = this.#router.ctxLen;
        this.#lastFed = 0;
        try {
            yield* (opts.loop === false ? this.#roundTrip(text, opts) : this.#closedLoop(text, opts));
        } finally {
            this.#stream = null;
            this.#generating = false;
            // A turn refused for a full context fed nothing, not even its
            // prompt.
            const fedPrompt = this.#router.ctxLen > start ? this.#lastPromptTokens : 0;
            this.#lastFed = Math.max(0, this.#router.ctxLen - start - fedPrompt);
        }
    }

    // Shared setup: the prompt's ids, the budget, and which limit applies.
    async #prepare(text, { temp, topK, rep, freq, pres, ngram, maxTokens = Infinity, prof = 0 }) {
        this.#stopRequested = false;
        this.#router.resetRepairs();
        // `rep`/`freq`/`pres`/`ngram` are passed through. The router turns
        // them into the summary the head samples with, since the router has
        // the token ids and the head has the logits row.
        const opts = { temp, topK, rep, freq, pres, ngram, prof };
        const samples = [];
        this.#lastProfile = { level: prof, samples };
        const promptIds = await encode(text);
        this.#lastPromptTokens = promptIds.length;
        const room = N_MAX - this.#router.ctxLen - promptIds.length;
        if (room <= 0) throw new ContextFullError(promptIds.length, N_MAX - this.#router.ctxLen);
        const budget = Math.min(maxTokens, room);
        // Which limit the budget comes from. They need different fixes from
        // the user: a higher cap, or a new chat.
        const ceiling = budget < maxTokens ? 'context' : 'length';
        return { opts, samples, promptIds, budget, ceiling };
    }

    async *#closedLoop(text, options) {
        const { opts, samples, promptIds, budget, ceiling } = await this.#prepare(text, options);
        const keep = (kind, n, a) => samples.push({ kind, n, rt: a.rt, loop: true, closed: a.closed,
            ...(a.link && { link: a.link }), trace: a.trace ?? [] });

        // Every chunk but the last is plain prefill. The last opens the loop.
        const last = Math.floor((promptIds.length - 1) / T_MAX) * T_MAX;
        for (let i = 0; i < last; i += T_MAX) {
            const chunk = promptIds.slice(i, i + T_MAX);
            const r = await this.#router.submit(chunk, opts);
            samples.push({ kind: 'prefill', n: chunk.length, rt: r.rt,
                           ...(r.link && { link: r.link }), trace: r.trace ?? [] });
            this.#emit('change');
        }

        const stream = this.#router.decode(promptIds.slice(last), opts, { budget });
        this.#stream = stream;
        if (this.#stopRequested) stream.halt();
        const genIds = [];
        try {
            let first = true;
            for await (const a of stream) {
                keep(first ? 'prefill' : 'decode', first ? promptIds.length - last : 1, a);
                first = false;
                this.#emit('change');
                if (!a.fed) continue;          // the answer that ended the stream; not in the reply
                genIds.push(a.token);
                // Decode the whole run each time: a UTF-8 sequence is often
                // split across two tokens.
                yield { text: await decode(genIds), tokenId: a.token, trace: a.trace };
            }
            this.#lastStopReason = stream.reason === 'eos' ? 'eos'
                : stream.reason === 'stopped' ? 'stopped' : ceiling;
        } finally {
            // On a consumer that stopped pulling (a stop sequence) or an
            // error: halt the loop and wait for the history to settle before
            // anyone rewinds it. An error during the drain is thrown, since
            // the context may not hold everything that was shown.
            stream.halt();
            await stream.drain();
            if (stream.error) throw stream.error;
        }
    }

    async *#roundTrip(text, options) {
        const { opts, samples, promptIds, budget, ceiling } = await this.#prepare(text, options);
        const keep = (kind, n, r) => {
            samples.push({ kind, n, rt: r.rt, ...(r.link && { link: r.link }), trace: r.trace ?? [] });
            return r;
        };
        // Prefill: T_MAX tokens per job. Every job returns a sampled token.
        // Only the last chunk's is the first token of the reply.
        let tok = null, eos = false, trace = null;
        for (let i = 0; i < promptIds.length; i += T_MAX) {
            const chunk = promptIds.slice(i, i + T_MAX);
            ({ token: tok, eos } = keep('prefill', chunk.length, await this.#router.submit(chunk, opts)));
            this.#emit('change');
        }
        let answered = performance.now();

        const genIds = [];
        this.#lastStopReason = ceiling;
        for (let n = 0; n < budget; n++) {
            if (this.#stopRequested) { this.#lastStopReason = 'stopped'; return; }
            // The head marks an end-of-text token. The budget is checked here.
            if (eos) { this.#lastStopReason = 'eos'; return; }

            genIds.push(tok);
            // Decode the whole run each time: a UTF-8 sequence is often split
            // across two tokens.
            const shown = await decode(genIds);
            yield { text: shown, tokenId: tok, trace };

            const prev = samples.at(-1);
            if (prev) prev.overhead = Math.round((performance.now() - answered) * 100) / 100;
            ({ token: tok, eos, trace } = keep('decode', 1, await this.#router.submit([tok], opts)));
            answered = performance.now();
            this.#emit('change');
        }
    }

    // End the current generate(): after the in-flight token, or with the
    // closed loop, once the tokens already fed have arrived.
    stop() {
        this.#stopRequested = true;
        this.#stream?.halt();
    }

    // Drop the last `n` tokens from the context, for a caller whose template
    // logic decides some generated tokens should not be kept. Later jobs
    // overwrite the KV rows past the new end. On a hybrid checkpoint the
    // recurrent state cannot be trimmed, so the next turn rebuilds the context
    // (route.js's rewind()).
    rewind(n) { this.#router.rewind(n); }

    // --- Chats ---
    //
    // A conversation (route.js's makeConversation) is one chat's token history
    // and the session the peers hold it under. One is active. The others are
    // parked, and their caches stay on the peers until idle eviction. None of
    // these methods may be called during a generate().

    #idle(what) {
        if (this.#generating) throw new Error(`a reply is still being generated. Stop it before ${what}.`);
    }

    // Make a conversation in this tab's mesh without activating it. It is
    // empty, or holds the token ids of a stored chat, which its first turn
    // rebuilds from.
    newConversation(history = []) {
        return makeConversation({ mesh: this.meshTag, history });
    }

    // Make `conv` the active chat. Throws for a chat of another mesh.
    switchTo(conv) { this.#idle('switching chats'); this.#router.switchTo(conv); }

    // Start a new chat and activate it. The previous one is parked.
    newChat() { this.#idle('starting a new chat'); return this.#router.newChat(); }

    // End a chat and free its caches on the peers immediately. If it was the
    // active chat, a new one replaces it.
    discard(conv = this.conversation) { this.#idle('deleting a chat'); this.#router.discard(conv); }

    // End the active chat and start a new one.
    reset() { this.discard(); }

    drop() { this.#net?.stop(); }

    // --- Read-only state, for the caller's rendering ---

    get net() { return this.#net; }
    get runner() { return this.#runner; }
    get role() { return this.#role; }
    get canPrompt() { return this.#canPrompt; }
    get dims() { return this.#dims; }
    get ctxLen() { return this.#router?.ctxLen ?? 0; }
    get conversation() { return this.#router?.conversation ?? null; }
    get meshTag() { return this.#router?.mesh ?? null; }
    // A copy of the active chat's token ids: what the peers were fed, and what
    // a stored chat is rebuilt from.
    get history() { return Uint32Array.from(this.#router?.history ?? []); }
    get ran() { return this.#router?.ran ?? 0; }
    get blocksPerSec() { return this.#sp; }
    get vramLimit() { return this.#vramLimit; }
    get lastStopReason() { return this.#lastStopReason; }
    get lastPromptTokens() { return this.#lastPromptTokens; }
    get lastFed() { return this.#lastFed; }
    get lastProfile() { return this.#lastProfile; }
    get vramBytes() { return this.#runner?.vramBytes ?? 0; }
}
