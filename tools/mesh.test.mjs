// Tests the protocol between several peers sharing one roster: N routers
// (web/route.js) over N Runners (web/model/stage.js), with the roster read
// through N real `Mesh` instances (web/net.js). Only the datachannel is faked.
//
//   node tools/mesh.test.mjs [--seed N] [--fuzz N --fuzz-seed S]
//
// The stub GPU does no arithmetic, so every sampled token would be 0. The
// harness therefore wraps each Runner in a proxy that supplies a toy model: a
// digest folded once per token position and once per block, with per-session
// state for each block. The digest does not depend on how the context was
// chunked or on which peer ran which block, and it does depend on the whole
// prefix. A repair or rebuild that drops or repeats part of the context gives
// different tokens.

import { Runner } from '../web/model/stage.js';
import { makeGpu, D, DH, L, N, SINGLE, weightsFor, pieceFor, hybridWeightsFor } from './stub-gpu.mjs';
import { SLOTS, coverage, served } from '../web/split.js';

const F32 = 4;

// net.js uses `location` and `WebSocket` in the Mesh constructor and nothing
// else until a message arrives, so stubs are enough. net.test.mjs installs the
// same two.
globalThis.location = { protocol: 'http:', host: 'localhost:8000' };
globalThis.WebSocket = class { addEventListener() {} send() {} close() {} readyState = 1; };
// route.js mints session keys with Web Crypto, which is a global in browsers
// and in Node 19+. Older Node versions have it in node:crypto.
globalThis.crypto ??= (await import('node:crypto')).webcrypto;

const { Mesh, pack, unpack } = await import('../web/net.js');
const { makeRouter, makeConversation, T_MAX } = await import('../web/route.js');

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `\n        got  ${JSON.stringify(got)}\n        want ${JSON.stringify(want)}`);

// --- The toy model ---

const H = D.hidden;
const SEED = 0x9e3779b9;

// A 32-bit mixing function. Values that go through the f32 payload are masked
// to 24 bits, because a float32 represents integers exactly only up to 2^24. A
// digest that was rounded in transit would look like corruption.
function mix(a, b) {
    let x = ((a ^ b) + 0x9e3779b9) >>> 0;
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
    return (x ^ (x >>> 16)) >>> 0;
}

const tagIds = new Map();
function tagId(t) {
    let id = tagIds.get(t);
    if (id === undefined) {
        id = 0;
        for (let i = 0; i < t.length; i++) id = mix(id, t.charCodeAt(i));
        tagIds.set(t, id);
    }
    return id;
}

// Only the blocks have KV. The embedding and the head are position-wise in the
// real model, so they have no accumulated state here either.
const HAS_KV = /^L\d+$/;

// In a hybrid world, linear_attention blocks (stub-gpu.mjs's DH) have
// recurrent state: one running fold per session per block, with no position
// axis, like Gated DeltaNet's `S`. Feeding a position twice folds it twice, so
// a rewind that reached the peers, or a replay into an uncleared frame, gives
// a different answer. The fold restarts only when the real Runner clears the
// buffers, so the toy follows what the Runner did.
const RECURRENT = /^R\d+$/;

// --- A peer ---

// The proxy Runner. Every guard under test is the real Runner's. The proxy
// adds the toy forward pass and one ledger assertion that the real Runner
// cannot make, because it never sees more than one job's context.
function peerRunner(world, id, lo, hi, have, slots, idleMs = 0) {
    // Each peer has its own GPU and trace. With a shared trace, a replay into
    // one peer while another is working would be recorded as one run.
    const { device, setFill, resetTrace, takeTrace, passes } = makeGpu();
    // nMax is well above T_MAX, so a context can be longer than one replay
    // chunk and the chunk loop in replay() is tested.
    // `idleMs: 0`: this suite's eviction tests need a second session to take
    // the first's slot immediately. At the default IDLE_MS it would be refused
    // (split.js's IDLE_MS). Capacity refusal is tested on real hardware by the
    // `busy` rows of tools/mesh.gpu.test.mjs.
    const real = new Runner(device, world.hybrid ? DH : D,
        (world.hybrid ? hybridWeightsFor : weightsFor)(lo, have),
        // `waitMs: 0`: no job waits for a frame. With `idleMs` 0 a frame can
        // always be taken, and the fuzz expects refusals to be immediate.
        // Waiting is tested in tools/stage.test.mjs.
        { lo, hi, have, nMax: 4 * T_MAX, tMax: T_MAX, slots, idleMs, waitMs: 0 });

    // `${session}|${tag}` -> the prefix fold, one entry per token position.
    // Stored by position, like the KV cache it models, so re-running a
    // position with the same input changes nothing and a replay does not look
    // like corruption.
    const kv = new Map();
    // `${session}|${tag}` -> the recurrent fold. See RECURRENT.
    const rec = new Map();
    // `${session}|${entry}` -> { len, at }: how far this cursor has been fed,
    // and the input digest at each position, so a rewrite can be checked. A
    // repair rewrites rows (a replay re-feeds the prefix, and the re-sent job
    // re-feeds its own rows), and each rewrite must store the same values.
    const fed = new Map();

    // The stub's trace is one array per device, so resetTrace/takeTrace must
    // bracket exactly one run. The real Runner serializes its work (it has one
    // set of scratch buffers), and the proxy serializes the measurement around
    // it. A replay sends a whole prefix at once, so a peer can have several
    // jobs queued.
    let queue = Promise.resolve();

    return {
        real,
        get lo() { return real.lo; },
        get hi() { return real.hi; },
        get have() { return real.have; },
        // Both are read by route.js's repair path: the boundary log it
        // replays, and the width of a row in it. The router only ever sees the
        // proxy, so the proxy exposes them.
        get d() { return real.d; },
        get logs() { return real.logs; },
        resetTrace, takeTrace,
        holds: (p) => real.holds(p),
        append: (w) => real.append(w),
        prepend: (w, reserved) => real.prepend(w, reserved),
        growLeft: () => real.growLeft(),
        growRight: () => real.growRight(),
        // Runs on the proxy's queue too. A warm-up is a run with its own
        // readback, and if queued ahead of a job it would consume the toy
        // readback that job set, leaving the job with zeros.
        warmup(from, want) {
            const done = queue.then(() => real.warmup(from, want));
            queue = done.catch(() => {});
            return done;
        },
        // Passed to the real Runner, which queues it with its jobs. The toy
        // state for the session is pruned by the `real.live` sweep after the
        // next run.
        release: (session) => real.release(session),

        run(h, hidden, want) {
            const done = queue.then(() => one(h, hidden, want));
            queue = done.catch(() => {});
            return done;
        },
    };

    // The toy forward pass. It is installed as the run's readback, so
    // Runner.#run reads the digests as the GPU's output and writes its
    // boundary log from them during the run. Patching the result afterwards
    // would leave zeros in the log for a microtask, and a repair reading it
    // then would replay zeros.
    //
    // `bytes` is the size of the run's readback. A [vocab] row means the run
    // will sample. null means it is a replay that reached the end of the line
    // and reads nothing. Neither produces rows to hand on.
    async function one(h, hidden, want) {
        resetTrace();
        let last = 0, filled = false;
        const cleared = passes.cleared;

        function fold(tags, bytes) {
            // The ledger. Every job for a session arriving at this peer at
            // this cursor must continue where the last one stopped, and a
            // rewrite (a replay, or a repair's re-sent job) must store the
            // same values as before. The real Runner cannot check either,
            // since it only receives `past`.
            const fk = `${h.s}|${h.pos}`;
            let seen = fed.get(fk);
            if (!seen) fed.set(fk, seen = { len: 0, at: new Uint32Array(real.nMax) });
            if (h.past > seen.len) {
                world.violation(`peer ${id}: session ${h.s} entered at ${h.pos} with ` +
                    `past=${h.past}, but only ${seen.len} positions have been fed there`);
            }
            // A rewind is a job from the enqueuer that arrives below what was
            // fed, with new input from some position on. The context was cut
            // back and the job writes a new tail (its first ids may match what
            // they overwrite). A replay, a repair's re-sent job, and a job the
            // head fed must store the same values as before.
            const mayRewind = !h.replay && !h.auto;

            const rows = bytes === null || bytes === D.vocab * F32
                ? null : new Float32Array(bytes / F32);
            for (let j = 0; j < h.n; j++) {
                const p = h.past + j;
                let d = (h.pos === 0 ? h.ids[j] : hidden[j * H]) >>> 0;
                if (p < seen.len && seen.at[p] !== d) {
                    if (mayRewind) seen.len = p;
                    else world.violation(`peer ${id}: session ${h.s} re-fed position ${p} ` +
                        `at ${h.pos} with a different residual`);
                }
                seen.at[p] = d;
                for (const t of tags) {
                    d = mix(d, tagId(t));
                    if (RECURRENT.test(t)) {
                        const k = `${h.s}|${t}`;
                        if (j === 0 && passes.cleared > cleared) rec.delete(k);
                        const st = mix(rec.get(k) ?? SEED, d);
                        rec.set(k, st);
                        d = mix(d, st);
                    }
                    if (HAS_KV.test(t)) {
                        const k = `${h.s}|${t}`;
                        let pref = kv.get(k);
                        if (!pref) kv.set(k, pref = new Uint32Array(real.nMax));
                        pref[p] = mix(p ? pref[p - 1] : SEED, d);
                        d = mix(d, pref[p]);
                    }
                    d &= 0xffffff;
                }
                if (rows) rows[j * H] = d;
                last = d;
            }
            seen.len = Math.max(seen.len, h.past + h.n);
            return rows;
        }


        setFill((bytes) => {
            filled = true;
            return fold(takeTrace(), bytes) ?? new Float32Array(D.vocab);
        });

        let r;
        try {
            r = await real.run(h, hidden, want);     // the real Runner's guards run first
        } finally {
            setFill(null);
        }
        // A replay that reached the end of the line reads nothing back, so the
        // fold has not run. It hands on no rows and keeps no log, so folding
        // it here is safe, and it must still be counted.
        if (!filled) fold(takeTrace(), null);

        // The toy follows the real Runner. A session the real Runner evicted
        // loses its toy state too. Otherwise the next job would be answered
        // from a prefix whose KV was evicted, and the eviction would go
        // unnoticed.
        for (const m of [kv, rec]) {
            for (const k of [...m.keys()]) {
                if (!real.live.has(k.slice(0, k.indexOf('|')))) m.delete(k);
            }
        }

        // Three result shapes: a token, a hidden state to hand on, or neither
        // (a replay that reached the end of the line). The other fields (phase
        // timings, a GPU profile) are the real Runner's and are passed
        // through.
        if (r.token !== undefined) return { ...r, token: last % D.vocab };
        return r;
    }
}

// --- The world ---

class World {
    // `maxRepairs` is set per World. It bounds how long a failing mesh keeps a
    // user waiting and is not needed for correctness. The soak checks that
    // every departure the mesh can absorb is absorbed, and a budget would make
    // it fail for reasons unrelated to routing.
    constructor({ maxRepairs, maxFixes, maxLocalReplay, hybrid = false, hybridRouter = hybrid,
                  idleMs = 0, ackRate = Infinity } = {}) {
        // Bytes per second a hop's ack wait allows for on top of ackTimeout.
        // Unlimited here, so the 15ms timeout in the other tests is exact.
        this.ackRate = ackRate;
        // How long a runner's session must be idle before another may take its
        // frame. 0 in most of this suite, whose eviction tests need it. With
        // the default IDLE_MS a new session is refused as busy.
        this.idleMs = idleMs;
        // Runners use stub-gpu.mjs's hybrid DH, and routers are told so.
        // `hybridRouter: false` makes a router that does not know, so only the
        // runners guard the recurrent state.
        this.hybrid = hybrid;
        this.hybridRouter = hybridRouter;
        this.maxRepairs = maxRepairs;
        // `maxFixes: 0` turns local repair off, so the error path and the
        // enqueuer's rebuild can be tested. With repair on they do not run.
        this.maxFixes = maxFixes;
        // When low, no context is short enough to repair locally, so a
        // departure goes to the enqueuer's rebuild.
        this.maxLocalReplay = maxLocalReplay;
        this.peers = [];          // the roster, shared by reference with every Mesh
        this.tabs = new Map();    // id -> { net, router, runner }
        this.zombies = new Set(); // peers whose links look open and drop everything
        this.violations = [];
        this.wire = [];           // every message sent: { from, to, k, addr }
        this.slowLinks = new Map();   // `${from}>${to}` -> a slow link; see slowLink()
        this.backedUp = new Set();    // `${from}>${to}` links above net.js's high-water mark
    }

    violation(msg) { this.violations.push(msg); }

    // A peer that claims `[lo, hi)` and has downloaded `[lo, have)`.
    // `lo === null` is a spectator: in the mesh, linked to the entry point,
    // never routed a job.
    join(id, lo = null, hi = null, { have = hi, slots = SLOTS, ready = true, closedLoop = true } = {}) {
        const net = new Mesh({});
        net.ready.catch(() => {});          // never welcomed (the WebSocket is a stub)
        Object.assign(net, {
            id, blocks: L, positions: N, minBlocks: 1,
            range: lo == null ? null : [lo, hi],
            peers: this.peers,
        });

        const runner = lo == null ? null : peerRunner(this, id, lo, hi, have, slots, this.idleMs);
        const tab = { id, net, runner, repairs: 0, fixes: [], sessions: [],
                      progress: [], cancelled: false };

        // The wire. `linked()` is the real method, reading the links this
        // world installs from the real `wants` rule, so a send fails exactly
        // when the production code would treat the route as broken.
        net.send = (to, header, payload) => {
            if (to == null || !net.linked(to)) return false;
            // Like the real send(): a backed-up link refuses a message with a
            // residual stream and lets a header-only message through.
            if (this.backedUp.has(`${id}>${to}`)) {
                const body = header instanceof ArrayBuffer ? unpack(header).payload : payload;
                if (body?.byteLength) return false;
            }
            this.deliver(id, to, header, payload);
            return true;
        };
        net.buffered = (to) => this.buffered(id, to);
        net.backedUp = (to) => net.linked(to) && this.backedUp.has(`${id}>${to}`);

        tab.router = makeRouter({
            net,
            runner: () => tab.runner,
            hybrid: this.hybridRouter,
            // false makes the head answer loop jobs without feeding.
            closedLoop,
            onRepair: (err, tokens, reason) => {
                tab.repairs++;
                tab.lastRepair = { err, tokens, reason };
            },
            onFix: (h) => { tab.fixes.push(h); },
            // A test can set `tab.onProgress`, e.g. to cancel a rebuild when
            // its first chunk arrives.
            onProgress: (sent, total) => {
                tab.progress.push({ sent, total });
                tab.onProgress?.(sent, total);
            },
            isCancelled: () => tab.cancelled,
            ackTimeout: 15,       // short, so zombie and slow-link tests run quickly
            ackRate: this.ackRate,
            jobTimeout: 500,      // the enqueuer's last-resort timeout, rarely reached
            ...(this.maxRepairs === undefined ? {} : { maxRepairs: this.maxRepairs }),
            ...(this.maxFixes === undefined ? {} : { maxFixes: this.maxFixes }),
            ...(this.maxLocalReplay === undefined ? {} : { maxLocalReplay: this.maxLocalReplay }),
        });
        tab.router.eos = -1;      // no toy token is end-of-text

        this.peers.push({ id, lo, hi, have: lo == null ? null : have, ready: lo != null && ready });
        this.tabs.set(id, tab);
        this.sync();
        return tab;
    }

    // Send a message. It is packed and unpacked, so the framing and the
    // zero-copy `raw` forward path are tested, and a sender cannot change a
    // header after sending.
    deliver(from, to, header, payload) {
        const buf = header instanceof ArrayBuffer ? header : pack(header, payload);
        // `to` is the hop the message took, and `addr` is the addressee in its
        // header. They differ when a message is relayed through the entry
        // point. An assertion about who an `err` is addressed to must read
        // `addr`.
        const { header: sent } = unpack(buf);
        this.wire.push({ from, to, k: sent.k, addr: sent.to, s: sent.s, pos: sent.pos,
                         replay: !!sent.replay, past: sent.past, n: sent.n, prof: sent.prof,
                         opts: sent.opts, loop: !!sent.loop, fed: sent.fed, auto: !!sent.auto });
        const slow = this.slowLinks.get(`${from}>${to}`);
        const later = (fn) => {
            if (!slow) return queueMicrotask(fn);
            // Without `drain`, the message arrives late and does not delay
            // later messages.
            if (!slow.drain) {
                const ms = slow.ms(sent);
                return ms ? setTimeout(fn, ms) : queueMicrotask(fn);
            }
            // With `drain`, each message takes `ms` and they stay in order: a
            // message never arrives before the one sent ahead of it.
            const now = performance.now();
            slow.until = Math.max(now + slow.ms(sent), slow.until);
            setTimeout(fn, slow.until - now);
        };
        later(() => {
            // Checked again at delivery: a peer can close its tab after the
            // send, and the real wire drops messages in flight.
            const tab = this.tabs.get(to);
            if (!tab || this.zombies.has(to)) return;
            const { header: h, payload: p } = unpack(buf);
            // The fourth argument is the sender's id, which the real Mesh
            // takes from the link the message arrived on (net.js's #attach).
            // The backward error path depends on it.
            tab.router.route(h, p, buf, from);
        });
    }

    // Make a link on which a message takes `ms(header)` to arrive, in order,
    // like a prefill's residual stream on a long-distance link. `drain` says
    // whether the sender sees its queue shrinking meanwhile (RTCDataChannel's
    // bufferedAmount, net.js's buffered()), as on a live link. Without it the
    // message arrives late.
    slowLink(from, to, ms, { drain = true } = {}) {
        this.slowLinks.set(`${from}>${to}`, { ms, drain, until: 0 });
    }

    // What the sender sees queued on a link: here the milliseconds of transfer
    // left, which only decrease.
    buffered(from, to) {
        const slow = this.slowLinks.get(`${from}>${to}`);
        return slow?.drain ? Math.max(0, slow.until - performance.now()) : 0;
    }

    // Installs links by the real rule. `#sync()` cannot be called, since it
    // needs an RTCPeerConnection, so this installs what it would build once
    // ICE finished. `wants` is the production symmetric closure of `#targets`,
    // so a departure closes the same links as in production.
    sync() {
        for (const tab of this.tabs.values()) {
            tab.net.links = new Map();
            for (const id of tab.net.wants) {
                if (this.tabs.has(id)) {
                    tab.net.links.set(id, { pc: {}, ch: { readyState: 'open' } });
                }
            }
        }
    }

    entry(id) { return this.peers.find((p) => p.id === id); }

    // Whether the model would still be covered without this peer. This is the
    // check `Mesh.staffed` makes, on the same view of the roster. The soak
    // uses it to tell a repair that must succeed from one that may fail.
    survivable(id) {
        const rest = this.peers.filter((p) => p.ready && p.id !== id
            && !this.zombies.has(p.id)).map(served);
        return coverage(rest, N).r >= 1;
    }

    covered() { return this.survivable(null); }

    // Download, announce and serve one more position.
    async grow(id, n = 1) {
        const tab = this.tabs.get(id), e = this.entry(id);
        for (let i = 0; i < n && tab.runner.have < tab.runner.hi; i++) {
            const p = tab.runner.have;
            tab.runner.append(pieceFor(p));
            await tab.runner.warmup(p);
            e.have = p + 1;
        }
        this.sync();
    }

    // Widen the claim by one position on an edge, past the original claim.
    // growLeft()/growRight() reserve new KV here. grow() above only fills KV
    // the claim reserved at construction. Follows #growClaim() in engine.js:
    // reserve, download, then announce, for the edge that moved.
    //
    // `gap`, when given, is awaited between growLeft() reserving the KV and
    // prepend() attaching it with the weights. It stands in for the network
    // fetch #growClaim() does there, so a test can run a job in that window.
    // See "a job for an existing position must not land mid left-growth
    // commit" below.
    async growEdge(id, dir, { gap } = {}) {
        const tab = this.tabs.get(id), e = this.entry(id), r = tab.runner;
        if (dir === 'left') {
            if (r.lo <= 0) return false;
            const p = r.lo - 1;
            const reserved = r.growLeft();
            if (gap) await gap();
            r.prepend(pieceFor(p), reserved);
            await r.warmup(r.lo, r.lo + 1);
            e.lo = r.lo;
        } else {
            if (r.hi >= N) return false;
            const p = r.hi;
            r.growRight();
            r.append(pieceFor(p));
            await r.warmup(p);
            e.hi = r.hi;
            e.have = r.have;
        }
        this.sync();
        return true;
    }

    // The tab closed. The datachannel fires close, every link to the peer
    // closes, and `send` returns false. This is the common kind of departure.
    kill(id) {
        this.tabs.delete(id);
        this.peers.splice(this.peers.findIndex((p) => p.id === id), 1);
        this.zombies.delete(id);
        this.sync();
    }

    // The laptop lid closed, the tab was throttled, or the GPU hung. The
    // channel still looks open, `send` returns true, and nothing is delivered.
    zombie(id) { this.zombies.add(id); }

    // Delay this peer's GPU work without delaying the ack it sends on
    // receipt: a slow hop, not a dead one like zombie(). The ack is sent in
    // deliver(), before runJob() calls runner().run(), so wrapping run()
    // gives a prompt ack and slow compute.
    slow(id, ms) {
        const tab = this.tabs.get(id);
        const run = tab.runner.run.bind(tab.runner);
        tab.runner.run = (h, hidden, want) =>
            new Promise((resolve) => setTimeout(resolve, ms)).then(() => run(h, hidden, want));
    }

    // The Stop button, pressed during a rebuild.
    cancel(id) { this.tabs.get(id).cancelled = true; }
}

// Prefill in T_MAX chunks, then decode, like respond() in peer.js but without
// the tokenizer, the streaming or the turn cut.
async function generate(tab, promptIds, budget, hooks = {}) {
    const opts = { temp: 0, topK: 0, ...hooks.opts };
    const out = [];
    let tok = null;
    for (let i = 0; i < promptIds.length; i += T_MAX) {
        ({ token: tok } = await tab.router.submit(promptIds.slice(i, i + T_MAX), opts));
    }
    // A session is minted on a chat's first job. Record it when it appears.
    if (tab.router.session !== tab.sessions.at(-1)) tab.sessions.push(tab.router.session);
    for (let n = 0; n < budget; n++) {
        out.push(tok);
        await hooks.between?.(n);
        ({ token: tok } = await tab.router.submit([tok], opts));
        if (tab.router.session !== tab.sessions.at(-1)) tab.sessions.push(tab.router.session);
    }
    return out;
}

const PROMPT = [11, 22, 33, 44, 55, 66, 77];
const BUDGET = 8;

// A world of peers cut at `cuts`, plus a spectator that prompts. The spectator
// holds nothing, so every job crosses the wire and the answer comes back
// through the entry point, which is the longest path in the protocol.
function chain(cuts, opts = {}, world = {}) {
    const w = new World(world);
    for (let i = 0; i < cuts.length - 1; i++) w.join(i, cuts[i], cuts[i + 1], opts);
    return { w, asker: w.join(100) };
}

// ---

console.log('\none peer holding everything: the reference');

const { w: w0, asker: a0 } = chain([0, N]);
const CONTROL = await generate(a0, PROMPT, BUDGET);
{
    ok('a whole generation completes', CONTROL.length === BUDGET);
    ok('and its tokens are not all the same value',
        new Set(CONTROL).size > 1, `got ${JSON.stringify(CONTROL)}`);
    eq('no ledger violations', w0.violations, []);
    // Every job of a session enters position 0 on the same peer, and its
    // answer returns through that peer as the entry point.
    ok('the spectator never ran anything', a0.runner === null);
}

console.log('\nevery way of cutting the line: the split must be invisible');
{
    let bad = [], arrangements = 0;
    for (let mask = 0; mask < 1 << (N - 1); mask++) {
        const cuts = [0];
        for (let x = 1; x < N; x++) if (mask & (1 << (x - 1))) cuts.push(x);
        cuts.push(N);
        const { w, asker } = chain(cuts);
        const got = await generate(asker, PROMPT, BUDGET);
        arrangements++;
        if (JSON.stringify(got) !== JSON.stringify(CONTROL)) bad.push(cuts.join('-'));
        if (w.violations.length) bad.push(`${cuts.join('-')}: ${w.violations[0]}`);
    }
    ok(`every chain produces the reference tokens (${arrangements} arrangements)`,
        bad.length === 0, `${bad.length} disagreed, e.g. ${bad[0]}`);
}

console.log('\nthe trace: the mesh dispatches what one tab dispatches');
{
    // The trace is per Runner, so it is read from the proxies. One decode step
    // across the whole mesh must equal the single-tab list, in order.
    const { w, asker } = chain([0, 2, 5, N]);
    const seen = [];
    for (const tab of w.tabs.values()) {
        if (!tab.runner) continue;
        const peer = tab.runner, real = peer.real, inner = real.run.bind(real);
        real.run = async (h, hidden, want) => {
            peer.resetTrace();
            const r = await inner(h, hidden, want);
            seen.push(...peer.takeTrace());
            return r;
        };
    }
    await generate(asker, [7], 1);
    // Two positions ran: the one-token prompt and the one decode step.
    eq('two token positions dispatch the single-tab list twice',
        seen, [...SINGLE, ...SINGLE]);
}

console.log('\ntiming: every hop reports where its time went');
{
    // A spectator asking a three-peer chain: every job crosses every link, so
    // every hop and every link reports timing.
    const { w, asker } = chain([0, 2, 5, N]);
    const opts = { temp: 0, topK: 0, prof: 1 };
    const seenProf = [];
    for (const tab of w.tabs.values()) {
        if (!tab.runner) continue;
        const real = tab.runner.real, inner = real.run.bind(real);
        real.run = (h, hidden, want) => { seenProf.push([tab.id, h.prof]); return inner(h, hidden, want); };
    }

    const first = await asker.router.submit([7], opts);
    const s0 = asker.router.session;       // minted by that first job
    const second = await asker.router.submit([first.token], opts);

    eq('one trace entry per hop, in path order', second.trace.map((h) => [h.id, h.from, h.to]),
        [[0, 0, 2], [1, 2, 5], [2, 5, N]]);
    const PH = ['queue', 'plan', 'encode', 'submit', 'wait', 'copy', 'ui'];
    ok('each carries its phases, none negative',
        second.trace.every((h) => PH.every((p) => typeof h[p] === 'number' && h[p] >= 0)
            && typeof h.dispatches === 'number' && h.dispatches > 0));
    ok('the head samples and the others log', second.trace.at(-1).sample >= 0
        && second.trace.slice(0, -1).every((h) => h.log >= 0 && h.sample === undefined));
    ok('the enqueuer times the round trip and its own hand-off',
        second.rt >= 0 && typeof second.link?.rtt === 'number');
    ok('a link has no timing on its first crossing: the ack lands after the trace left',
        first.trace.every((h) => h.link === undefined));
    ok('and has it from the second token on, on every hop that hands on',
        second.trace.slice(0, -1).every((h) => typeof h.link?.rtt === 'number'
            && typeof h.link?.send === 'number') && second.trace.at(-1).link === undefined);
    eq('the header asked every peer on the path', seenProf.filter(([, p]) => p === 1).map(([id]) => id),
        [0, 1, 2, 0, 1, 2]);
    ok('and without timestamp-query no GPU numbers are invented',
        second.trace.every((h) => h.gpu?.ts === false && h.gpu.gpuMs === undefined));

    const off = await asker.router.submit([second.token], { temp: 0, topK: 0 });
    ok('a job that did not ask carries no profile and no prof header',
        off.trace.every((h) => h.gpu === undefined)
        && w.wire.filter((m) => m.k === 'job').slice(-3).every((m) => m.prof === undefined));
    ok('the enqueuer had timed its own hand-off', asker.router.links.has(`${s0}|0`));
    asker.router.discard();
    eq('and a retired session forgets it', [...asker.router.links.keys()]
        .filter((k) => k.startsWith(`${s0}|`)), []);
}

console.log('\ntiming: a replay never carries the profile');
{
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);
    let mark = 0;
    const got = await generate(asker, PROMPT, BUDGET, {
        opts: { prof: 2 },
        between: async (n) => {
            if (n !== 4) return;
            mark = w.wire.length;
            w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });
    eq('a profiled generation still matches the control through a repair', got, CONTROL);
    const jobs = w.wire.slice(mark).filter((m) => m.k === 'job');
    ok('the repair replayed', jobs.some((m) => m.replay));
    ok('no replay asked for a profile', jobs.filter((m) => m.replay).every((m) => m.prof === undefined));
    ok('while the re-sent job kept it', jobs.filter((m) => !m.replay).every((m) => m.prof === 2));
}

console.log('\npinning: a session meets the same KV cache every time');
{
    // Two peers hold 4–N. Only one of them may receive this session.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);

    await generate(asker, PROMPT, BUDGET);
    const served = [1, 2].filter((id) => w.tabs.get(id).runner.real.live.size > 0);
    eq('exactly one of the two holders carried the session', served.length, 1);
    eq('and no ledger violations', w.violations, []);

    // A second session may pick either: the pin is per session.
    asker.router.newChat();
    await generate(asker, PROMPT, 2);
    ok('a new session is routed independently of the old one',
        w.tabs.get(served[0]).runner.real.live.size >= 1);
}

console.log('\ngrowth mid-session: new blocks only ever serve new sessions');
{
    const w = new World();
    w.join(0, 0, 4, { have: 4 });
    w.join(1, 4, N, { have: 6 });      // still downloading the last two positions
    w.join(2, 6, N, { have: N });      // covers what peer 1 has not loaded
    const asker = w.join(100);

    const cuts = () => [...w.tabs.get(1).runner.real.cuts.entries()]
        .map(([s, m]) => [s, [...m.entries()]]);

    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => { if (n === 3) await w.grow(1, 2); },
    });
    eq('a session that grew under it still produces the reference tokens', got, CONTROL);
    eq('and nothing was fed a gap', w.violations, []);
    ok('peer 1 finished its download mid-chat', w.tabs.get(1).runner.have === N);
    ok("but the live session's cut did not move",
        cuts().every(([, m]) => m.every(([at, to]) => to === 6 || at >= 6)),
        `${JSON.stringify(cuts())}`);
}

console.log('\ngrowth past the original claim: widening on both edges mid-session');
{
    // Peer 1 starts with only the middle of the line, and peers 0 and 2 cover
    // the rest. This tests growLeft()/growRight()/prepend() widening a claim
    // past its original range. grow() above uses only append(), where the KV
    // for the whole claim is reserved at construction.
    const w = new World();
    w.join(0, 0, 3, { have: 3 });
    w.join(1, 3, 5, { have: 5 });
    w.join(2, 5, N, { have: N });
    const asker = w.join(100);

    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n === 2) await w.growEdge(1, 'left');
            if (n === 4) await w.growEdge(1, 'right');
        },
    });
    eq('a claim grown past its original edges still produces the reference tokens', got, CONTROL);
    eq('and nothing was fed a gap', w.violations, []);
    eq('peer 1 grew left past its original lo', w.tabs.get(1).runner.lo, 2);
    ok('peer 1 grew right past its original hi, one position, and downloaded it',
        w.tabs.get(1).runner.hi === 6 && w.tabs.get(1).runner.have === 6);

    // After both growth steps, a new session entering at the new left edge
    // runs to the new right edge on peer 1 alone, which shows the widened
    // range is served.
    w.tabs.get(1).runner.resetTrace();
    const solo = await w.tabs.get(1).runner.run(
        { k: 'job', s: 'fresh-after-growth', pos: 2, past: 0, n: 1, ids: [1], opts: {} },
        new Float32Array(D.hidden));
    ok('and reaches the newly-grown right edge', solo.at === 6, `got ${solo.at}`);
}

console.log('\na job for an existing position must not land mid left-growth commit');
{
    // growLeft() and prepend() are separate steps. In #growClaim() (engine.js)
    // a network fetch happens between growLeft() reserving the KV and
    // prepend() attaching it with the weights. `block()` converts a position
    // to an index with `p - 1 - blockLo`, for both `w.layers` and the KV
    // cache, so `blockLo` and the arrays must change together. This test
    // holds the gap open with a delay and routes a job through position 3,
    // which the peer already held, while position 2 is being added.
    const w = new World();
    w.join(0, 0, 3, { have: 3 });
    w.join(1, 3, 5, { have: 5 });   // blocks 2, 3; growing left adds block 1
    w.join(2, 5, N, { have: N });
    w.join(100);

    const gap = () => new Promise((resolve) => setTimeout(resolve, 10));
    const growing = w.growEdge(1, 'left', { gap });

    const runner = w.tabs.get(1).runner;
    runner.resetTrace();
    const during = await runner.run(
        { k: 'job', s: 'during-growth', pos: 3, past: 0, n: 1, ids: [1], opts: {} },
        new Float32Array(D.hidden))
        .then(() => null, (e) => e.message);
    ok('a job for an existing position completes without error mid-growth',
        during === null, `got ${during}`);
    ok('and dispatched block 2: the block position 3 holds',
        runner.takeTrace().includes('L2'), `got ${JSON.stringify(runner.takeTrace())}`);

    await growing;
    eq('and nothing was fed a gap', w.violations, []);
}

console.log('\na clean kill mid-generation: local repair fixes it where it broke');
{
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);                   // the replacement
    const asker = w.join(100);

    // Kill whichever of peers 1 and 2 the session is pinned to.
    let mark = 0, ctxAtKill = 0;
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n !== 4) return;
            mark = w.wire.length;
            ctxAtKill = asker.router.ctxLen;
            const carrying = [1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0);
            w.kill(carrying);
        },
    });
    const after = w.wire.slice(mark);

    eq('the answer completes and matches the unkilled control', got, CONTROL);

    // Peer 0 held the log for the broken boundary, so it picked another
    // holder, replayed the context into it and re-sent the job. The enqueuer
    // did not rebuild.
    eq('the enqueuer spent no context rebuild at all', asker.repairs, 0);
    eq('and never had to retire its session', asker.sessions.length, 1);
    eq('the peer holding the log did the repair', w.tabs.get(0).router.repaired, 1);
    const replayed = after.filter((m) => m.k === 'job' && m.replay && m.from === 0);
    ok('and replayed into the replacement rather than merely routing to it',
        replayed.length > 0);
    // The replay is exactly the prefix. The failed job's own rows are the tail
    // of the log and the job carries them, so replaying them too would run
    // them twice. Replaying fewer would leave a gap in the replacement's
    // context.
    eq('replaying exactly the context the failed job had behind it',
        replayed.reduce((t, m) => t + m.n, 0), ctxAtKill);

    // A `fix` is a notification, not an answer. It reaches the enqueuer and
    // does not affect the job in flight.
    eq('the enqueuer is told, as news', after.filter((m) => m.k === 'fix')
        .map((m) => [m.from, m.addr]), [[0, 100]]);
    eq('naming the boundary that was repaired', asker.fixes.map((h) => [h.from, h.at]),
        [[0, 4]]);

    eq('nothing was answered from a cold cache', w.violations, []);
    ok('the surviving holder carried it',
        [1, 2].filter((id) => w.tabs.has(id) && w.tabs.get(id).runner.real.live.size > 0)
            .length === 1);
}

console.log('\nthe same kill with repair switched off: the enqueuer rebuilds');
{
    // maxFixes 0 forces the rebuild path, which must keep working: a failure
    // at a boundary nobody holds a log for (most often the enqueuer's own
    // first hop) is handled by it.
    const w = new World({ maxFixes: 0 });
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);

    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n !== 4) return;
            const carrying = [1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0);
            w.kill(carrying);
        },
    });

    eq('the answer still completes and matches the control', got, CONTROL);
    eq('at the cost of exactly one context rebuild', asker.repairs, 1);
    ok('under a new session id', asker.sessions.length === 2,
        `${JSON.stringify(asker.sessions)}`);
    eq('and the dead id keeps no pins', [...asker.router.routes.keys()]
        .filter((k) => k.startsWith(`${asker.sessions[0]}|`)), []);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\nan error walks back down the chain it came up');
{
    // Five holders in a line, so the failure is detected in the middle: peer 2
    // notices, its upstream is peer 1, and the entry point is peer 0. All
    // three are different peers. In a three-peer chain the upstream and the
    // entry point are the same tab, and an `err` sent to the wrong one could
    // not be told apart. Repair is off here. With it on, peer 2 would repair
    // locally and send no error.
    const w = new World({ maxFixes: 0 });
    w.join(0, 0, 2);
    w.join(1, 2, 4);
    w.join(2, 4, 6);
    w.join(3, 6, N);               // the tail this session pins
    w.join(4, 6, N);               // its replacement
    const asker = w.join(100);

    let mark = 0;
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n !== 4) return;
            mark = w.wire.length;
            w.kill([3, 4].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });

    const errs = w.wire.slice(mark).filter((m) => m.k === 'err');

    // Peer 2 could not hand the job on, so it sends an `err` to the peer that
    // handed it the job, which could repair from its boundary log. With repair
    // off, each peer passes it back one hop until it reaches the enqueuer.
    eq('the err is addressed backwards, one hop at a time, down to the enqueuer',
        errs.map((m) => [m.from, m.addr]), [[2, 1], [1, 0], [0, 100]]);

    // The enqueuer then rebuilds, triggered by the error and not by the job
    // timeout. With repair on, the first peer in this list would repair it.
    ok('and names the peer that actually detected it',
        /^peer 2: no peer that can run position 6 is reachable$/
            .test(asker.lastRepair?.err?.message ?? ''),
        `got ${asker.lastRepair?.err?.message}`);
    eq('the answer still completes and matches the unkilled control', got, CONTROL);
    eq('on exactly one context rebuild', asker.repairs, 1);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\nan error for a job submitted locally still answers its own tab');
{
    // A job this tab submitted went through route() with no channel, so there
    // is no upstream entry. fail() then uses `h.from`, which is this tab's own
    // id, and the error is delivered locally without being sent anywhere.
    const w = new World();
    w.join(0, 0, N);               // the only holder
    const asker = w.join(100);

    let err = null, mark = 0;
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => { if (n === 3) { mark = w.wire.length; w.kill(0); } },
    }).catch((e) => { err = e; return null; });

    ok('the generation fails rather than answering', got === null);
    ok('and the enqueuer is told by itself, naming the boundary that broke',
        /^peer 100: no peer that can run position 0 is reachable$/.test(err?.message ?? ''),
        `got ${err?.message}`);
    eq('the enqueuer holds no upstream for a session it submitted',
        [...asker.router.upstream.keys()], []);
    eq('and no err crossed the wire to say so', w.wire.slice(mark)
        .filter((m) => m.k === 'err'), []);
}

console.log('\na kill with nobody to replace it: fail loudly, do not guess');
{
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);                   // the only holder of the second half
    const asker = w.join(100);

    let err = null;
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => { if (n === 3) w.kill(1); },
    }).catch((e) => { err = e; return null; });

    ok('the generation fails rather than answering', got === null);
    // Position 4, not 0: the peer that could not advance the job reports the
    // error, so the message names the boundary that broke, not where the
    // enqueuer submitted the job.
    ok('and says which position nobody can run',
        /no peer that can run position 4 is reachable/.test(err?.message ?? ''),
        `got ${err?.message}`);
    eq('with no cold-cache answer anywhere', w.violations, []);
}

console.log('\na zombie peer: nothing can repair a peer that swallows');
{
    // There is no other holder of the second half, so repair finds no
    // candidate. The zombie never acks the hand-off, so the failure is
    // detected after about ackTimeout, not jobTimeout, and the error names the
    // peer.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);               // the only holder of the second half
    const asker = w.join(100);

    let err = null;
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => { if (n === 4) w.zombie(1); },
    }).catch((e) => { err = e; return null; });

    ok('the generation gives up rather than answering', got === null);
    ok('having learned from the ack timing out, naming the boundary',
        /unreachable|no peer that can run/.test(err?.message ?? ''), `got ${err?.message}`);
    eq('and no peer repaired anything, because no alternate exists',
        w.tabs.get(0).router.repaired, 0);
    // Every rebuild's first job is routed to the zombie and its ack times out.
    // A failed rebuild is retried, so the turn's whole repair budget is spent
    // before the failure is reported, and no more than the budget.
    eq('the whole repair budget went on the same peer, and no more', asker.repairs, 2);
}

console.log('\na slow but alive peer: legitimately long compute is not a departure');
{
    // An end-to-end timeout cannot tell a dead peer from one computing a long
    // prefill. The ack can: it is sent in deliver(), before runner().run() is
    // called, so a hop's compute time does not delay it.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);

    w.slow(1, 200);   // much longer than ackTimeout (15ms), shorter than jobTimeout (500ms)
    const got = await generate(asker, PROMPT, BUDGET);

    eq('the answer matches the reference', got, CONTROL);
    eq('with no rebuild', asker.repairs, 0);
    eq('and no local repair: nothing ever looked dead', w.tabs.get(0).router.repaired, 0);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\na genuine zombie, caught by the ack alone: no kill(), no send ever fails');
{
    // Two peers hold the second half, so a repair candidate exists.
    // World.deliver() drops messages to the zombie without calling its
    // route(), so it never acks. No send fails and the zombie sends no err, so
    // only the ack timeout can trigger the repair.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);

    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n !== 4) return;
            const carrying = [1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0);
            w.zombie(carrying);   // no w.kill(): the link stays open and send returns true
        },
    });

    eq('the answer matches the reference', got, CONTROL);
    eq('with no rebuild', asker.repairs, 0);
    eq('and one local repair, found by the ack timing out', w.tabs.get(0).router.repaired, 1);
    eq('naming the boundary the zombie held', asker.fixes.at(-1)?.at, 4);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\na zombie the roster has caught up with: an ordinary departure');
{
    // Once the signaling server drops the zombie, `send` fails, and the peer
    // that detects it repairs as for any departure.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);

    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n !== 4) return;
            const carrying = [1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0);
            w.zombie(carrying);
            w.kill(carrying);
        },
    });
    eq('the answer matches the control', got, CONTROL);
    eq('with no rebuild', asker.repairs, 0);
    eq('and one local repair', w.tabs.get(0).router.repaired, 1);
}

console.log('\neviction: a third chat must not answer the first from a cold cache');
{
    const w = new World();
    w.join(0, 0, N, { slots: 1 });     // one slot: a second session evicts the first
    const a = w.join(100), b = w.join(101);

    await generate(a, [5, 6], 1);
    await generate(b, [7, 8], 1);      // evicts a's cache

    // The next job arrives with past > 0 and no cache, so #cacheFor throws.
    // The enqueuer receives the error and rebuilds, so eviction costs a full
    // prefill and the chat continues.
    const got = await generate(a, [9], 1);
    ok('the evicted session is refused rather than answered cold',
        /no longer cached here/.test(a.lastRepair?.err?.message ?? ''),
        `got ${a.lastRepair?.err?.message}`);
    eq('and the enqueuer rebuilds through it', a.repairs, 1);
    eq('so the chat survives the eviction', got.length, 1);
    eq('with nothing silently continued', w.violations, []);
}

console.log('\na context longer than one chunk: the replay has to loop');
{
    // The only test where a repair replays more than one chunk. The others use
    // a 7-token prompt, so `past` stays below T_MAX and the loop in replay()
    // runs once. An off-by-one in that loop would truncate the prefix.
    const LONG = Array.from({ length: 2 * T_MAX + 5 }, (_, i) => (i * 7) % 29);

    const ctl = chain([0, 4, N]);
    const control = await generate(ctl.asker, LONG, 3);

    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);

    let mark = 0, ctxAtKill = 0;
    const got = await generate(asker, LONG, 3, {
        between: async (n) => {
            if (n !== 1) return;
            mark = w.wire.length;
            ctxAtKill = asker.router.ctxLen;
            w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });
    const replayed = w.wire.slice(mark)
        .filter((m) => m.k === 'job' && m.replay && m.from === 0);

    ok('the context really did outgrow one chunk', ctxAtKill > T_MAX,
        `${ctxAtKill}`);
    ok('so the replay went out in several', replayed.length > 1,
        `${replayed.length} chunk(s)`);
    eq('each carrying its own past, in order, with no gap',
        replayed.map((m) => m.past), replayed.reduce((a, m) => [...a, a.at(-1) + m.n], [0])
            .slice(0, -1));
    eq('and covering exactly the context behind the failed job',
        replayed.reduce((t, m) => t + m.n, 0), ctxAtKill);
    eq('the answer matches the unkilled control', got, control);
    eq('with no rebuild', asker.repairs, 0);
    eq('and nothing answered from a cold cache', w.violations, []);
}

console.log('\neviction behind a relay: the upstream refills it rather than reporting it');
{
    // Eviction on a peer downstream of a relay. #cacheFor throws, the error
    // goes back over the channel the job arrived on, and the peer that handed
    // the job over replays its boundary log. The enqueuer does not rebuild.
    // With this, SLOTS limits speed, not the number of chats.
    const w = new World();
    w.join(0, 0, 4);                   // the relay: holds the log for boundary 4
    w.join(1, 4, N, { slots: 1 });     // one slot, so a second chat evicts the first
    const a = w.join(100), b = w.join(101);

    await generate(a, PROMPT, 2);
    await generate(b, [7, 8], 1);      // evicts a's cache on peer 1

    let mark = w.wire.length;
    const got = await generate(a, [9], 1);
    const after = w.wire.slice(mark);

    eq('the evicted chat answers rather than dying', got.length, 1);
    eq('and the enqueuer never hears about it', a.repairs, 0);
    eq('the relay repaired the boundary it holds the log for',
        w.tabs.get(0).router.repaired, 1);
    ok('by replaying into the same peer, which is reachable and holds the position',
        after.some((m) => m.k === 'job' && m.replay && m.from === 0 && m.to === 1));
    eq('nothing was continued from a cold cache', w.violations, []);
}

console.log('\na replay is not evidence a boundary works');
{
    // A replay crossing a boundary is part of a repair. It must not reset that
    // boundary's repair budget, or two peers could repair each other
    // indefinitely. It must not become the job to re-send either: nobody waits
    // on a replay.
    const w = new World();
    w.join(0, 0, 2);
    w.join(1, 2, 4);
    w.join(2, 4, N);
    const asker = w.join(100);

    await generate(asker, PROMPT, 2);
    const s = asker.router.session;
    const one = w.tabs.get(1).router;
    const key = [...one.outbound.keys()].find((k) => k.startsWith(`${s}|`));

    const before = { ...one.outbound.get(key) };
    one.fixes.set(key, 1);            // the state a repair at this boundary leaves

    // A replay across the same boundary, as an upstream repair sends it.
    // Peer 0's log at boundary 2 holds the rows peer 0 hands over there, which
    // is what peer 1 receives.
    const log = w.tabs.get(0).runner.logs.get(s).get(2);
    w.deliver(0, 1, { k: 'job', s, from: 100, pos: 2, past: 0, n: 2, replay: true },
              log.rows.subarray(0, 2 * H));
    await new Promise((r) => setTimeout(r, 5));

    eq('a replay does not become the job to re-send',
        [one.outbound.get(key).past, one.outbound.get(key).n], [before.past, before.n]);
    eq('and does not clear the budget the repair is spending', one.fixes.get(key), 1);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\nthe fix budget: a boundary that keeps breaking stops being repaired');
{
    // The budget bounds repairs within one job and is cleared when a job
    // crosses the boundary normally. It runs out when a repair's replacement
    // fails too, not when a chat is repaired once per token.
    //
    // Three peers hold 4–6. Once the tail is killed, nobody holds 6–N. Peer 0
    // re-routes to another holder of position 4, which cannot hand on at 6.
    // The error comes back and peer 0 tries again. Without a bound it would
    // cycle through the roster indefinitely.
    const w = new World({ maxRepairs: 0 });
    w.join(0, 0, 4);
    for (const id of [1, 2, 3] ) w.join(id, 4, 6);
    w.join(4, 6, N);                   // the only tail
    const asker = w.join(100);

    let err = null;
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => { if (n === 2) w.kill(4); },
    }).catch((e) => { err = e; return null; });

    ok('the generation fails rather than looping', got === null);
    eq('after exactly the budgeted number of local repairs',
        w.tabs.get(0).router.repaired, 2);
    ok('and the failure names the position nobody can run',
        /no peer that can run position 6 is reachable/.test(err?.message ?? ''),
        `got ${err?.message}`);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\nthe rebuild budget: a mesh losing peers faster than it answers');
{
    // Repair is off, so every departure reaches the enqueuer and the rebuild
    // budget is tested.
    const w = new World({ maxFixes: 0 });
    w.join(0, 0, 4);
    for (const id of [1, 2, 3, 4]) w.join(id, 4, N);
    const asker = w.join(100);

    let err = null;
    await generate(asker, PROMPT, BUDGET, {
        between: async () => {
            const carrying = [1, 2, 3, 4].find(
                (id) => w.tabs.has(id) && w.tabs.get(id).runner.real.live.size > 0);
            if (carrying !== undefined) w.kill(carrying);
        },
    }).catch((e) => { err = e; });

    ok('the turn gives up rather than rebuilding forever', err !== null,
        `${asker.repairs} repairs`);
    ok('after exactly the budgeted number of rebuilds', asker.repairs === 2,
        `${asker.repairs}`);
}

console.log('\nan expensive local repair escalates instead of grinding through it');
{
    // Every hop on the way back sees the same session `past`. When it is above
    // maxLocalReplay, repair() declines before trying a candidate, so the
    // failure reaches the enqueuer. Uses `kill()`, not `zombie()`: the peer is
    // removed from the roster, so the rebuild's new session is not routed to
    // it. This test covers the size limit only.
    const w = new World({ maxLocalReplay: 2 });
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);   // an alternate holder exists; it is declined for cost
    const asker = w.join(100);

    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n !== 4) return;
            w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });

    eq('the answer matches the reference', got, CONTROL);
    eq('no hop attempted a local repair', w.tabs.get(0).router.repaired, 0);
    eq('the enqueuer rebuilt instead', asker.repairs, 1);
    eq('with nothing answered from a cold cache', w.violations, []);
}

console.log('\nrecover() reports progress as the rebuild proceeds');
{
    // A context longer than one T_MAX chunk, so the rebuild's loop runs more
    // than once and reports progress more than once. Local repair is off, so
    // the failure reaches recover().
    const LONG = Array.from({ length: T_MAX + 5 }, (_, i) => (i * 7) % 29);
    const w = new World({ maxFixes: 0 });
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);

    await generate(asker, LONG, 1, {
        between: async (n) => {
            if (n !== 0) return;
            w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });

    ok('more than one progress call, since the context outgrows one chunk',
        asker.progress.length > 1, `${asker.progress.length}`);
    eq('the final call reports the whole rebuilt context',
        asker.progress.at(-1), { sent: asker.router.ctxLen, total: asker.router.ctxLen });
    ok('every call is non-decreasing and bounded by the same total',
        asker.progress.every((p, i) => p.total === asker.progress[0].total
            && p.sent <= p.total && (i === 0 || p.sent > asker.progress[i - 1].sent)));
}

console.log('\ncancelling a rebuild stops it before it finishes');
{
    // The Stop button during a rebuild. Cancellation is checked between
    // chunks, so the test cancels on the first progress call, after one chunk
    // has been accepted.
    const LONG = Array.from({ length: T_MAX + 5 }, (_, i) => (i * 7) % 29);
    const w = new World({ maxFixes: 0 });
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);
    asker.onProgress = () => w.cancel(100);

    let err = null;
    await generate(asker, LONG, 1, {
        between: async (n) => {
            if (n !== 0) return;
            w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    }).catch((e) => { err = e; });

    ok('the rebuild is rejected as cancelled', /cancelled/.test(err?.message ?? ''),
        `got ${err?.message}`);
    eq('exactly one chunk had already landed', asker.progress.length, 1);
    // The partial session is dropped, not kept as a shorter chat. The chat
    // keeps everything it had before the turn (all of the context except the
    // token the failed job was feeding), and its next turn rebuilds it.
    eq('and the chat keeps its whole context, not the half that was rebuilt',
        asker.router.ctxLen, asker.progress[0].total - 1);
    eq('under no session, so nothing half-built is ever asked again', asker.router.session, null);
}

// --- The soak ---
//
// Repairs should not conflict: routing state is per session, and a repair
// starts at the peer that holds the log. This checks it by killing random
// peers mid-generation in random meshes.

const seedArg = process.argv.indexOf('--seed');
const SEED0 = seedArg > 0 ? Number(process.argv[seedArg + 1]) >>> 0 : (Date.now() >>> 0);

// xorshift32. A run is reproducible from the seed printed on failure.
function rng(seed) {
    let s = seed || 1;
    return () => {
        s ^= s << 13; s >>>= 0;
        s ^= s >>> 17;
        s ^= s << 5; s >>>= 0;
        return s / 0x100000000;
    };
}

// A covering set of ranges with random widths, overlap and download progress.
// Repair needs overlap, so the number of peers is chosen to keep the thinnest
// position at depth >= 2.
function randomMesh(rand, { deep = true } = {}) {
    const cuts = [0];
    for (let x = 1; x < N; x++) if (rand() < 0.45) cuts.push(x);
    cuts.push(N);
    const ranges = [];
    for (let i = 0; i < cuts.length - 1; i++) ranges.push([cuts[i], cuts[i + 1]]);
    if (!deep) return ranges;
    // A second, independently cut chain over the same line. Every position
    // then has at least two holders, so any single peer can leave.
    const cuts2 = [0];
    for (let x = 1; x < N; x++) if (rand() < 0.45) cuts2.push(x);
    cuts2.push(N);
    for (let i = 0; i < cuts2.length - 1; i++) ranges.push([cuts2[i], cuts2[i + 1]]);
    return ranges;
}

// Two modes with different assertions:
//
//   survivable   every kill leaves the line covered, so every repair or
//                rebuild must succeed and the answer must equal the
//                reference. A refusal is a failure.
//   murderous    kills ignore coverage. The answer may not arrive. If it does
//                it must be correct, and a failure must be one of the errors
//                the protocol produces.
async function soak(mode, iters, rand, gen = generate) {
    let runs = 0, matched = 0, refused = 0, absorbed = 0, fixed = 0;
    const broken = [];

    for (let iter = 0; iter < iters && broken.length < 3; iter++) {
        // A mesh that cannot answer before any kill tests nothing. Partial
        // downloads often leave holes, so the mesh is redrawn until it is
        // covered.
        let w = null, asker = null, ranges = null;
        for (let attempt = 0; attempt < 30 && !w; attempt++) {
            ranges = randomMesh(rand);
            const candidate = new World({ maxRepairs: 99 });
            ranges.forEach(([lo, hi], i) => {
                // Some peers join mid-download, so plan() must stop before the
                // end of a claim and hops() must prefer the peer that reaches
                // furthest.
                const have = rand() < 0.3 ? lo + 1 + Math.floor(rand() * (hi - lo)) : hi;
                candidate.join(i, lo, hi, { have: Math.min(have, hi) });
            });
            asker = candidate.join(100);
            if (candidate.covered()) w = candidate;
        }
        if (!w) continue;
        runs++;

        // A zombie is a tab that stopped answering. The signaling server
        // notices one step later and removes it from the roster. Without that
        // removal, the peer could never be routed around.
        let pendingKill = null;

        let err = null;
        const got = await gen(asker, PROMPT, BUDGET, {
            between: async () => {
                if (pendingKill !== null) { w.kill(pendingKill); pendingKill = null; }
                const r = rand();
                const live = [...w.tabs.keys()].filter((id) => id !== 100);
                if (!live.length) return;

                if (r < 0.30) {
                    const growable = live.filter((id) => w.tabs.get(id).runner.have
                        < w.tabs.get(id).runner.hi);
                    if (growable.length) {
                        await w.grow(growable[Math.floor(rand() * growable.length)], 1);
                    }
                    return;
                }
                if (r > 0.75) return;

                // Half the time, pick the peer this session is pinned to,
                // which forces a repair or rebuild. Otherwise pick any peer.
                const carrying = live.filter(
                    (id) => w.tabs.get(id).runner.real.live.size > 0);
                const pool = (rand() < 0.5 && carrying.length) ? carrying : live;
                const victim = pool[Math.floor(rand() * pool.length)];
                if (mode === 'survivable') {
                    // Clean departures only. Per-hop acks detect a zombie
                    // quickly (tested above), but a departure at hop 0 needs a
                    // full recover() rebuild, and a zombie stays in the roster
                    // until the next step's between() runs. hops() can then
                    // send every rebuild's first job to it until the turn's
                    // repair budget is spent. Zombies are therefore used only
                    // in murderous mode.
                    if (w.survivable(victim)) w.kill(victim);
                    return;
                }
                if (r < 0.68) w.kill(victim);
                else { w.zombie(victim); pendingKill = victim; }
            },
        }).catch((e) => { err = e; return null; });

        const where = `seed ${SEED0} ${mode} iter ${iter}`;
        if (w.violations.length) {
            broken.push(`${where}: ${w.violations[0]}`);
        } else if (got === null) {
            if (mode === 'survivable') {
                broken.push(`${where}: coverage never dropped, yet: ${err.message}\n` +
                    `        mesh ${JSON.stringify(ranges)}`);
            } else if (/is reachable|unreachable|no answer from the mesh|no longer cached/
                .test(err.message)) {
                refused++;
            } else {
                broken.push(`${where}: unexpected error: ${err.message}`);
            }
        } else if (JSON.stringify(got) === JSON.stringify(CONTROL)) {
            matched++;
            // The peer that detected the departure repaired it, and the
            // enqueuer did not rebuild.
            if (asker.repairs === 0) absorbed++;
            fixed += [...w.tabs.values()].reduce((n, t) => n + t.router.repaired, 0);
        } else {
            broken.push(`${where}: wrong tokens\n` +
                `        got  ${JSON.stringify(got)}\n` +
                `        want ${JSON.stringify(CONTROL)}\n` +
                `        mesh ${JSON.stringify(ranges)}`);
        }
    }
    return { runs, matched, refused, absorbed, fixed, broken };
}

console.log(`\nthe soak: random meshes, random growth, random kills  (--seed ${SEED0})`);
{
    const rand = rng(SEED0);

    const s = await soak('survivable', 120, rand);
    console.log(`        survivable: ${s.runs} runs, ${s.matched} completed, ` +
        `${s.absorbed} without a rebuild (${s.fixed} local repairs)`);
    ok('a departure the mesh could absorb never costs a token',
        s.broken.length === 0 && s.matched === s.runs,
        `\n        ${s.broken.join('\n        ') || `only ${s.matched}/${s.runs} completed`}`);
    // Checks that local repair runs. Without this, the suite would pass if
    // repair never ran.
    ok('and a large share are absorbed where they broke, costing the enqueuer nothing',
        s.fixed > 0 && s.absorbed > s.runs / 3,
        `${s.absorbed}/${s.runs} rebuild-free, ${s.fixed} repairs`);

    const m = await soak('murderous', 120, rand);
    console.log(`        murderous:  ${m.runs} runs, ${m.matched} completed, ` +
        `${m.refused} refused loudly (${m.fixed} local repairs)`);
    ok('and one it could not is refused rather than answered wrongly',
        m.broken.length === 0, `\n        ${m.broken.join('\n        ')}`);
    ok('with both outcomes actually exercised',
        m.matched > 0 && m.refused > 0, `${m.matched} / ${m.refused}`);
}

// Session ids are guessable, so a peer off the path could forge the answer to
// another chat, or end it. Each forgery is sent when the real job leaves, so
// it arrives before the real answer and would replace it if accepted.
console.log('\nforged answers: only the path can answer a chat');
{
    const w = new World();
    w.join(0, 0, 4);                 // the hub
    w.join(1, 4, N);                 // the tail, which samples
    const mallory = w.join(50);      // a spectator: linked to the hub, holds nothing
    const asker = w.join(100);
    const opts = { temp: 0, topK: 0 };
    const FAKE = 12345;

    // CONTROL's prompt, one decode step at a time, calling `forge` with every
    // submit.
    async function run(forge) {
        let tok = null;
        for (let i = 0; i < PROMPT.length; i += T_MAX) {
            ({ token: tok } = await asker.router.submit(PROMPT.slice(i, i + T_MAX), opts));
        }
        const out = [];
        for (let n = 0; n < BUDGET; n++) {
            out.push(tok);
            const s = asker.router.session;
            const answer = asker.router.submit([tok], opts);
            forge(s, asker.router.pending.get(s).key);
            ({ token: tok } = await answer);
        }
        return out;
    }

    // Even with the right key: a spectator cannot have sampled a token, so the
    // hub does not relay one from it.
    const relayed = await run((s, key) =>
        w.deliver(50, 0, { k: 'tok', s, key, to: 100, token: FAKE }));
    eq('a tok relayed from a peer that holds no last position never arrives', relayed, CONTROL);

    // The tail may answer, but not for a job whose key it was never given.
    asker.router.newChat();
    const keyless = await run((s) =>
        w.deliver(1, 0, { k: 'tok', s, key: 'guessed', to: 100, token: FAKE }));
    eq('a tok with the wrong key is dropped, and the real one still lands', keyless, CONTROL);

    // Delivered directly to the asker, where no such link exists. The asker
    // checks the link the `tok` arrived on.
    asker.router.newChat();
    const direct = await run((s, key) =>
        w.deliver(50, 100, { k: 'tok', s, key, to: 100, token: FAKE }));
    eq('a tok handed straight over by a peer that cannot answer is dropped', direct, CONTROL);

    asker.router.newChat();
    const errs = await run((s) =>
        w.deliver(1, 0, { k: 'err', s, key: 'guessed', to: 100, from: 1, msg: 'forged' }));
    eq('an err with the wrong key does not fail the job', errs, CONTROL);
    eq('or send the asker into a rebuild', asker.repairs, 0);

    // A `free` would drop a live chat's cache. The tail holds the session.
    const s = asker.router.session;
    ok('the tail holds the session', w.tabs.get(1).runner.real.live.has(s));
    w.deliver(50, 1, { k: 'free', s, to: 1 });
    await new Promise((res) => setTimeout(res, 10));
    ok('a free from a peer that never handed it the session is ignored',
        w.tabs.get(1).runner.real.live.has(s));
    asker.router.discard();
    await new Promise((res) => setTimeout(res, 10));
    ok('while the real one, walked down the pins, still frees it',
        !w.tabs.get(1).runner.real.live.has(s));
    eq('and no ledger violations', w.violations, []);
    void mallory;
}

// --- Conversations: several chats per tab, switching, resuming, rebuilding ---

// Two uninterrupted turns of one chat. Every switch, eviction and rebuild
// below must reproduce this answer. The toy model is chunk-invariant, so a
// chat rebuilt from its history gives the same tokens.
const TURN2 = [5, 9];
async function twoTurns(opts = {}) {
    const { asker } = chain([0, 4, N], opts);
    const one = await generate(asker, PROMPT, 3);
    const two = await generate(asker, TURN2, 3);
    return { one, two, history: [...asker.router.history] };
}
const admitted = (w) => [...w.tabs.values()]
    .reduce((n, t) => n + (t.runner?.real.counts.admitted ?? 0), 0);

console.log('\nswitching back to a parked chat whose caches are still there resumes it');
{
    const ctl = await twoTurns();
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);

    eq('the first turn matches', await generate(asker, PROMPT, 3), ctl.one);
    const a = asker.router.conversation, sA = a.session;
    asker.router.newChat();
    await generate(asker, [3, 1, 4], 2);
    ok('the other chat ran under its own session', asker.router.session !== sA);

    asker.router.switchTo(a);
    const before = admitted(w);
    eq('the resumed turn matches a chat that never left', await generate(asker, TURN2, 3), ctl.two);
    eq('under the same session', asker.router.session, sA);
    eq('with no rebuild', asker.repairs, 0);
    eq('and no peer admitting it afresh', admitted(w), before);
    eq('and no ledger violations', w.violations, []);
}

console.log('\nswitching back after the caches were evicted rebuilds, and says it is a resume');
{
    const ctl = await twoTurns();
    const w = new World();
    w.join(0, 0, 4, { slots: 1 });
    w.join(1, 4, N, { slots: 1 });        // one chat per peer: a second chat evicts the first
    const asker = w.join(100);

    await generate(asker, PROMPT, 3);
    const a = asker.router.conversation;
    asker.router.newChat();
    await generate(asker, [3, 1, 4], 2);
    asker.router.switchTo(a);

    eq('the resumed turn still matches', await generate(asker, TURN2, 3), ctl.two);
    eq('after one rebuild', asker.repairs, 1);
    eq('reported as a resume, not a departure', asker.lastRepair?.reason, 'resume');
    ok('found by the cold-cache refusal',
        /no longer cached here/.test(asker.lastRepair?.err?.message ?? ''),
        `got ${asker.lastRepair?.err?.message}`);
    eq('and no ledger violations', w.violations, []);
}

console.log('\nonly the tail evicted it: the boundary log repairs that without a rebuild');
{
    const ctl = await twoTurns();
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N, { slots: 1 });
    const asker = w.join(100);

    await generate(asker, PROMPT, 3);
    const a = asker.router.conversation;
    asker.router.newChat();
    await generate(asker, [3, 1, 4], 2);
    asker.router.switchTo(a);

    eq('the resumed turn matches', await generate(asker, TURN2, 3), ctl.two);
    eq('with no rebuild', asker.repairs, 0);
    ok('because peer 0 replayed its boundary into the tail', w.tabs.get(0).router.repaired >= 1);
    eq('and no ledger violations', w.violations, []);
}

console.log('\nrewind on a hybrid checkpoint: the cut tokens must leave the recurrent state too');
for (const [label, hybridRouter] of [['a router that knows', true],
                                     ['a router that does not: the runners refuse', false]]) {
    const opts = { hybrid: true, hybridRouter };
    const { asker } = chain([0, 3, N], {}, opts);
    await generate(asker, PROMPT, 5);
    asker.router.rewind(3);
    const kept = [...asker.router.history];
    const got = await generate(asker, TURN2, 3);

    // The same chat without the three tokens that were rewound.
    const ctl = chain([0, 3, N], {}, opts);
    const want = await generate(ctl.asker, [...kept, ...TURN2], 3);
    eq(`${label}: the next turn is conditioned on what was kept, and nothing else`, got, want);
    if (hybridRouter) {
        eq('  rebuilt as a rewind', asker.lastRepair?.reason, 'rewind');
        eq('  without spending the turn\'s repair budget', asker.router.repairs, 0);
    } else {
        ok('  found by the refusal instead',
            /no longer cached here/.test(asker.lastRepair?.err?.message ?? ''));
    }
}

console.log('\na replay into a peer still holding the session restarts its recurrent state');
{
    const opts = { hybrid: true };
    const ctl = chain([0, 2, 5, N], {}, opts);
    const control = await generate(ctl.asker, PROMPT, BUDGET);

    const w = new World(opts);
    w.join(0, 0, 2);
    w.join(1, 2, 5);
    w.join(2, 2, 5);
    w.join(3, 5, N);
    const asker = w.join(100);
    const got = await generate(asker, PROMPT, BUDGET, {
        between: async (n) => {
            if (n === 4) w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });
    eq('the answer matches the control', got, control);
    eq('with a local repair and no rebuild', [w.tabs.get(0).router.repaired, asker.repairs], [1, 0]);
    ok('whose replay went on into the tail, which still held the session',
        w.wire.some((m) => m.replay && m.to === 3));
    eq('and no ledger violations', w.violations, []);
}

console.log('\na chat belongs to its mesh, and a switch waits for the reply');
{
    const { asker } = chain([0, 4, N]);
    await generate(asker, PROMPT, 2);
    const a = asker.router.conversation, hist = [...a.history];

    let err = null;
    try { asker.router.switchTo(makeConversation({ mesh: 'a1b2c3d4', history: [1, 2, 3] })); }
    catch (e) { err = e; }
    ok('a chat from another mesh is refused', /another mesh/.test(err?.message ?? ''),
        `got ${err?.message}`);
    ok('leaving the active chat as it was',
        asker.router.conversation === a && JSON.stringify(a.history) === JSON.stringify(hist));

    const inFlight = asker.router.submit([7], { temp: 0, topK: 0 });
    err = null;
    try { asker.router.newChat(); } catch (e) { err = e; }
    ok('switching while a job is in flight is refused', /still being generated/.test(err?.message ?? ''));
    await inFlight;
    ok('and allowed once it has landed', asker.router.newChat() === asker.router.conversation);
}

console.log('\nswitching away parks a chat; deleting it frees its caches');
{
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    await generate(asker, PROMPT, 2);
    const a = asker.router.conversation, sA = a.session;
    asker.router.newChat();
    await new Promise((res) => setTimeout(res, 10));
    ok('a parked chat is still cached on every peer of its path',
        [0, 1].every((id) => w.tabs.get(id).runner.real.live.has(sA)));

    asker.router.discard(a);
    await new Promise((res) => setTimeout(res, 10));
    ok('and discarding it frees it along the path',
        [0, 1].every((id) => !w.tabs.get(id).runner.real.live.has(sA)));
    eq('without touching the active chat', asker.router.conversation.history.length, 0);
}

console.log('\na late error for a parked chat is not mistaken for anybody else\'s');
{
    const ctl = await twoTurns();
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    await generate(asker, PROMPT, 2);
    const a = asker.router.conversation;
    asker.router.newChat();

    // Carries the parked chat's key, so only the ownership check can stop it.
    w.deliver(0, 100, { k: 'err', s: a.session, key: a.key, to: 100, from: 0, msg: 'late' });
    eq('the active chat is unaffected', await generate(asker, PROMPT, 3), ctl.one);
    eq('and nothing was rebuilt', asker.repairs, 0);
}

console.log('\na chat loaded from storage (a history and no session) rebuilds before it asks');
{
    const ctl = await twoTurns();
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);

    // What peer.js passes the engine from IndexedDB: the history after turn one.
    const stored = ctl.history.slice(0, PROMPT.length + 3);
    asker.router.switchTo(makeConversation({ mesh: 'global', history: stored }));
    eq('the next turn matches the chat that never left', await generate(asker, TURN2, 3), ctl.two);
    eq('reported as a resume', asker.lastRepair?.reason, 'resume');
    eq('costing no repair budget', asker.router.repairs, 0);
    eq('and never sending a job that could only come back cold',
        w.wire.filter((m) => m.k === 'err').length, 0);
}

console.log('\na turn that failed must not leave answers behind for the next one');
{
    // A job that arrives after its hop gave up on it. The hop reports the
    // failure and the enqueuer rebuilds, but the job still runs and its answer
    // is sent back. The next turn must only accept answers to its own jobs.
    const ctl = chain([0, 4, N]);
    const want = await generate(ctl.asker, [9, 8, 7], 6);

    const w = new World({ maxFixes: 0 });
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    let late = 3;                                  // the turn and both of its rebuilds
    w.slowLink(0, 1, (h) => (h.k === 'job' && h.n > 1 && late-- > 0 ? 40 : 0), { drain: false });
    // Adds GPU time per job, so the next turn is still waiting on the mesh
    // when the late answers arrive, as on a real mesh.
    w.slow(1, 4);

    let err = null;
    try { await generate(asker, PROMPT, 3); } catch (e) { err = e; }
    ok('the turn whose hand-off was too late fails', /unreachable/.test(err?.message ?? ''),
        `got ${err?.message}`);
    // The next turn starts immediately. The late jobs arrive while it waits
    // on its own jobs, and their answers arrive first.
    const got = await generate(asker, [9, 8, 7], 6).catch((e) => e.message);
    eq('the next turn is answered by its own jobs, not the failed one\'s', got, want);
    eq('and no ledger violations', w.violations, []);
}

console.log('\na slow link that is still delivering is not a dead one');
{
    // A prefill over a slow, relayed link: it takes much longer than the ack
    // timeout, but the sender sees its queue draining throughout.
    const ctl = chain([0, 4, N]);
    const want = await generate(ctl.asker, PROMPT, 4);

    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    w.slowLink(0, 1, (h) => (h.k === 'job' && h.n > 1 ? 60 : 1));   // ackTimeout is 15

    eq('the answer matches', await generate(asker, PROMPT, 4), want);
    eq('with no repair and no rebuild', [w.tabs.get(0).router.repaired, asker.repairs], [0, 0]);
    ok('and the hand-off was never reported as broken', !w.wire.some((m) => m.k === 'err'));
}

console.log('\na big job crossing a link that shows no queue is given time to land');
{
    // A prefill over the TURN relay. The sender's queue empties into SCTP
    // immediately, so there is no draining to watch, and the bytes arrive well
    // after ackTimeout. The wait is sized from the payload (ackRate). With no
    // allowance the same link fails the turn, as the test above shows.
    const ctl = chain([0, 4, N]);
    const want = await generate(ctl.asker, PROMPT, 4);

    // PROMPT is 7 rows of D.hidden (8) f32s, ~224 bytes. At 2 bytes/ms the
    // allowance is over 100ms.
    const w = new World({ ackRate: 2000 });
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    w.slowLink(0, 1, (h) => (h.k === 'job' && h.n > 1 ? 40 : 0), { drain: false });   // ackTimeout is 15

    eq('the answer matches', await generate(asker, PROMPT, 4), want);
    eq('with no repair and no rebuild', [w.tabs.get(0).router.repaired, asker.repairs], [0, 0]);
    ok('and the hand-off was never reported as broken', !w.wire.some((m) => m.k === 'err'));
}

console.log('\na boundary log begun again is never replayed past its end');
{
    // A peer's session was released and then re-admitted by a replay of a
    // shorter context, so its log is shorter than the job it last handed on.
    // An error for that hand-off must not be repaired from rows the log does
    // not hold. Those rows are zeros and would be sent downstream as the
    // residual stream.
    const w = new World();
    w.join(0, 0, 2);
    w.join(1, 2, 5);
    w.join(2, 5, N);
    w.join(3, 5, N);
    const asker = w.join(100);
    await generate(asker, PROMPT, 8);
    const s = asker.router.session, key = asker.router.conversation.key;
    const mid = w.tabs.get(1);
    await mid.runner.release(s);
    // The replay peer 0 would send: the first five rows of its boundary log.
    const rows = w.tabs.get(0).runner.logs.get(s).get(2).rows.slice(0, 5 * D.hidden);
    w.deliver(0, 1, { k: 'job', s, key, from: 100, pos: 2, past: 0, n: 5, replay: true }, rows);
    await new Promise((res) => setTimeout(res, 10));
    ok('the log was begun again, shorter than the hand-off it sits behind',
        mid.runner.logs.get(s)?.get(5)?.len === 5);
    const before = mid.router.repaired;
    const downstream = [2, 3].find((id) => w.tabs.get(id).runner.real.live.has(s));
    w.deliver(downstream, 1, { k: 'err', s, key, to: 1, from: downstream, msg: 'gone' });
    await new Promise((res) => setTimeout(res, 10));
    eq('the error is not repaired from rows the log does not hold', mid.router.repaired, before);
    eq('and nothing was fed from them', w.violations, []);
}

// --- The closed decode loop ---
//
// The head feeds each token it samples back in, and the enqueuer only receives
// (route.js's decode()). Every test below checks that the tokens, and the
// context left afterwards, equal what the round trip produces: for every cut
// of the line, every position of the enqueuer, every way a reply can end, and
// every way a peer can fail mid-reply.

const GREEDY = { temp: 0, topK: 0 };

// Longer references, one per model kind, for replies longer than BUDGET.
const CONTROL_F = await generate(chain([0, N]).asker, PROMPT, 40);
const CONTROL_H = await generate(chain([0, N], {}, { hybrid: true }).asker, PROMPT, 40);
const CONTROL_OF = (hybrid) => (hybrid ? CONTROL_H : CONTROL_F);

// The closed-loop version of generate(). Every chunk of the prompt but the
// last is plain prefill, the last goes through decode(), and the fed answers
// are the reply. `between(n, stream)` runs after the n-th token arrives; the
// loop does not wait for it. `stopAfter` halts there. `breakAfter` stops
// pulling there (as a stop sequence does), which drains the rest into the
// history.
async function streamed(tab, promptIds, budget, hooks = {}) {
    const opts = { ...GREEDY, ...hooks.opts };
    const last = Math.floor((promptIds.length - 1) / T_MAX) * T_MAX;
    for (let i = 0; i < last; i += T_MAX) {
        await tab.router.submit(promptIds.slice(i, i + T_MAX), opts);
    }
    const stream = tab.router.decode(promptIds.slice(last), opts, { budget });
    tab.stream = stream;
    hooks.onStream?.(stream);
    const out = [];
    tab.lastEnd = null;
    try {
        for await (const a of stream) {
            if (!a.fed) { tab.lastEnd = a; continue; }
            out.push(a.token);
            if (tab.router.session !== tab.sessions.at(-1)) tab.sessions.push(tab.router.session);
            await hooks.between?.(out.length - 1, stream);
            if (hooks.stopAfter === out.length) stream.halt();
            if (hooks.breakAfter === out.length) break;
        }
    } finally {
        // An error ends the stream without return(), and a break has already
        // drained. An error during the drain is thrown as the turn's error, as
        // in engine.js.
        await stream.drain();
        if (stream.error) throw stream.error;
    }
    return out;
}

const jobsFrom = (w, id) => w.wire.filter((m) => m.k === 'job' && m.from === id && !m.replay);
const settleMs = (ms) => new Promise((res) => setTimeout(res, ms));

// Checks that no loop is running: no head feeds a token for `ms`. A repair may
// still be finishing (a hand-off whose ack was lost when a peer left is
// re-sent once), but it feeds nothing.
async function quiet(w, ms = 40) {
    const fed = () => w.wire.filter((m) => m.k === 'job' && m.auto && m.pos === 0).length;
    await settleMs(ms);
    const before = fed();
    await settleMs(ms);
    return fed() === before;
}

// Every arrangement of cuts, as in the round-trip sweep above.
function everyCut() {
    const all = [];
    for (let mask = 0; mask < 1 << (N - 1); mask++) {
        const cuts = [0];
        for (let x = 1; x < N; x++) if (mask & (1 << (x - 1))) cuts.push(x);
        cuts.push(N);
        all.push(cuts);
    }
    return all;
}

console.log('\nclosed loop: every way of cutting the line gives the round trip\'s reply');
for (const hybrid of [false, true]) {
    let bad = [], n = 0, closed = 0;
    const control = await generate(chain([0, N], {}, { hybrid }).asker, PROMPT, BUDGET);
    for (const cuts of everyCut()) {
        const { w, asker } = chain(cuts, {}, { hybrid });
        const got = await streamed(asker, PROMPT, BUDGET);
        n++;
        if (JSON.stringify(got) !== JSON.stringify(control)) bad.push(cuts.join('-'));
        else if (JSON.stringify(asker.router.history) !== JSON.stringify([...PROMPT, ...control])) {
            bad.push(`${cuts.join('-')}: history`);
        } else if (w.violations.length) bad.push(`${cuts.join('-')}: ${w.violations[0]}`);
        if (jobsFrom(w, 100).length === 1) closed++;
    }
    ok(`${hybrid ? 'hybrid' : 'full attention'}: every chain gives the same tokens and context (${n} arrangements)`,
        bad.length === 0, `${bad.length} wrong, e.g. ${bad[0]}`);
    ok('and on every one of them the enqueuer sent the prompt and nothing more',
        closed === n, `${closed}/${n}`);
}

console.log('\nclosed loop: budgets, long prompts, and where the enqueuer sits');
{
    const LONG = Array.from({ length: 2 * T_MAX + 5 }, (_, i) => (i * 7) % 29);
    for (const [prompt, budget] of [[PROMPT, 1], [PROMPT, 2], [PROMPT, 17], [LONG, 5], [[3], 9]]) {
        const control = await generate(chain([0, 3, 6, N]).asker, prompt, budget);
        const { w, asker } = chain([0, 3, 6, N]);
        const got = await streamed(asker, prompt, budget);
        eq(`prompt of ${prompt.length}, budget ${budget}: the round trip's tokens`, got, control);
        eq('  and its context, prompt plus every token of the reply', asker.router.history,
            [...prompt, ...control]);
        eq('  the answer that ended it was not fed', asker.lastEnd?.fed, false);
        eq('  one job per prompt chunk left the enqueuer, and no more',
            jobsFrom(w, 100).length, Math.ceil(prompt.length / T_MAX));
    }

    // An enqueuer holding position 0 is the entry. One holding the end of the
    // line is the head that feeds. One holding everything runs the whole loop
    // in its own tab.
    for (const [label, ranges, askerIdx] of [
        ['an enqueuer holding position 0 (the entry)', [[0, 4], [4, N]], 0],
        ['an enqueuer holding the head (it feeds its own loop)', [[0, 4], [4, N]], 1],
        ['an enqueuer holding the whole model (the loop never leaves it)', [[0, N]], 0],
        ['an enqueuer in the middle of a three-peer line', [[0, 3], [3, 6], [6, N]], 1],
    ]) {
        const w = new World();
        ranges.forEach(([lo, hi], i) => w.join(i, lo, hi));
        const asker = w.tabs.get(askerIdx);
        const got = await streamed(asker, PROMPT, BUDGET);
        eq(`${label}: the reference tokens`, got, CONTROL);
        eq('  and no ledger violations', w.violations, []);
    }
}

console.log('\nclosed loop: the enqueuer\'s distance stops costing a round trip per token');
{
    // A few milliseconds each way between the enqueuer and the mesh, and no
    // delay elsewhere. The round trip pays it on every token, the loop once.
    const timed = async (fn) => {
        const w = new World();
        w.join(0, 0, 4);
        w.join(1, 4, N);
        const asker = w.join(100);
        w.slowLink(100, 0, () => 5);
        w.slowLink(0, 100, () => 5);
        const t0 = performance.now();
        const got = await fn(asker);
        return { got, ms: performance.now() - t0 };
    };
    const rt = await timed((a) => generate(a, PROMPT, 16));
    const lp = await timed((a) => streamed(a, PROMPT, 16));
    eq('the same reply either way', lp.got, rt.got);
    ok(`16 tokens: round trip ${rt.ms.toFixed(0)} ms, closed loop ${lp.ms.toFixed(0)} ms`,
        lp.ms < rt.ms / 4, 'the loop should be several times faster');
}

console.log('\nclosed loop: end-of-text, and a stopped reply');
{
    // End-of-text: one token of the reference reply is made the end. The loop
    // stops there and does not feed it, like the round trip.
    const at = CONTROL.findIndex((t, i) => CONTROL.indexOf(t) === i && i >= 3);
    const { w, asker } = chain([0, 4, N]);
    for (const tab of w.tabs.values()) tab.router.eos = CONTROL[at];
    const got = await streamed(asker, PROMPT, BUDGET);
    eq('the reply ends before its end-of-text token', got, CONTROL.slice(0, at));
    eq('which was never fed', asker.router.history, [...PROMPT, ...CONTROL.slice(0, at)]);
    ok('and the stream says why', asker.lastEnd?.eos === true && asker.stream.reason === 'eos');

    // Halted after three tokens. The tokens already fed still arrive, they are
    // the reply, and they are what the context holds. The next turn continues
    // from that context.
    for (const hybrid of [false, true]) {
        const label = hybrid ? 'hybrid' : 'full attention';
        const s = chain([0, 3, N], {}, { hybrid });
        for (const id of [0, 1]) s.w.slow(id, 2);        // so the loop is still running at the halt
        const reply = await streamed(s.asker, PROMPT, 40, { stopAfter: 3 });
        const control = CONTROL_OF(hybrid);
        ok(`${label}: a halt ends the reply early`, reply.length >= 3 && reply.length < 40,
            `${reply.length} tokens`);
        eq(`${label}: with the round trip's tokens as far as it went`, reply, control.slice(0, reply.length));
        eq(`${label}: and exactly those in the context`, s.asker.router.history, [...PROMPT, ...reply]);
        eq(`${label}: said to be stopped`, s.asker.stream.reason, 'stopped');

        const next = await streamed(s.asker, TURN2, 5);
        const ctl = await generate(chain([0, N], {}, { hybrid }).asker, [...PROMPT, ...reply, ...TURN2], 5);
        eq(`${label}: the next turn carries on from exactly that context`, next, ctl);
        eq(`${label}: with no rebuild`, s.asker.repairs, 0);
        eq(`${label}: no ledger violations`, s.w.violations, []);
        ok(`${label}: and nothing left running`, await quiet(s.w));
    }

    // A halt must carry the loop's key, since session ids are guessable.
    {
        const s = chain([0, 4, N]);
        s.w.slow(1, 2);
        const got = await streamed(s.asker, PROMPT, 12, {
            between: (n, stream) => {
                if (n !== 1) return;
                s.w.deliver(0, 1, { k: 'halt', s: s.asker.router.session, key: 'guessed', to: 1 });
            },
        });
        eq('a halt with the wrong key is ignored, and the reply runs to its budget', got.length, 12);
    }
}

console.log('\nclosed loop: a reply cut at a stop sequence: drain, then rewind');
for (const hybrid of [false, true]) {
    const label = hybrid ? 'hybrid' : 'full attention';
    const s = chain([0, 3, N], {}, { hybrid });
    // The stop sequence starts inside the third token, so two are kept.
    const reply = await streamed(s.asker, PROMPT, 20, { breakAfter: 3 });
    const fed = s.asker.router.history.length - PROMPT.length;
    ok(`${label}: breaking off still leaves everything fed in the context`,
        fed >= 3, `${fed} fed, ${reply.length} seen`);
    eq(`${label}: and what was fed is the reference continuation`,
        s.asker.router.history.slice(PROMPT.length), CONTROL_OF(hybrid).slice(0, fed));
    s.asker.router.rewind(fed - 2);
    const next = await streamed(s.asker, TURN2, 4);
    const ctl = await generate(chain([0, N], {}, { hybrid }).asker,
        [...PROMPT, ...reply.slice(0, 2), ...TURN2], 4);
    eq(`${label}: after the rewind the next turn sees only what was kept`, next, ctl);
    if (hybrid) eq('  rebuilt, as a rewind through recurrent state must be', s.asker.lastRepair?.reason, 'rewind');
    eq(`${label}: no ledger violations`, s.w.violations, []);
}

console.log('\nclosed loop: the loop closes again on the turn after a rewind');
{
    // Full attention keeps its session through a rewind, so the next turn's
    // loop starts below where the previous loop's head fed to. It is a new
    // loop and must close as one, without falling back to the round trip.
    const s = chain([0, 3, N]);
    const reply = await streamed(s.asker, PROMPT, 12, { breakAfter: 4 });
    s.asker.router.rewind(s.asker.router.history.length - PROMPT.length - 2);
    const mark = jobsFrom(s.w, 100).length;
    const next = await streamed(s.asker, TURN2, 8);
    const ctl = await generate(chain([0, N]).asker, [...PROMPT, ...reply.slice(0, 2), ...TURN2], 8);
    eq('the turn after a rewind gives the right tokens', next, ctl);
    eq('under the same session, with no rebuild', s.asker.repairs, 0);
    eq('and its loop closed: the enqueuer sent the prompt and nothing more',
        jobsFrom(s.w, 100).length - mark, 1);
}

console.log('\nclosed loop: peers failing mid-reply');
{
    // A middle peer leaves. The peer before it replays its boundary log into
    // the other holder, and the loop continues.
    {
        const w = new World();
        w.join(0, 0, 2);
        w.join(1, 2, 5);
        w.join(2, 2, 5);
        w.join(3, 5, N);
        const asker = w.join(100);
        const got = await streamed(asker, PROMPT, BUDGET, {
            between: (n) => {
                if (n === 3) w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
            },
        });
        eq('the middle leaves: the reference tokens', got, CONTROL);
        eq('  repaired where it broke, no rebuild', [w.tabs.get(0).router.repaired > 0, asker.repairs], [true, 0]);
        eq('  no ledger violations', w.violations, []);
    }

    // The entry leaves. The head's hand-off back to position 0 breaks, and
    // only the enqueuer can repair it, because only it has the ids.
    {
        const w = new World();
        w.join(0, 0, 4);
        w.join(1, 0, 4);
        w.join(2, 4, N);
        const asker = w.join(100);
        let killed = null;
        const got = await streamed(asker, PROMPT, BUDGET, {
            between: (n) => {
                if (n !== 3) return;
                killed = [0, 1].find((id) => w.tabs.get(id).runner.real.live.size > 0);
                w.kill(killed);
            },
        });
        eq('the entry leaves: the reference tokens', got, CONTROL);
        eq('  after one rebuild, which reopened the loop', asker.repairs, 1);
        eq('  and the context is exact', asker.router.history, [...PROMPT, ...CONTROL]);
        eq('  no ledger violations', w.violations, []);
        ok('  and nothing left running', await quiet(w));
    }

    // The head leaves. The peer before it re-sends the job to the other head,
    // which continues the loop. With penalties on, the new head has no context
    // to compute them from, so it declines and the enqueuer opens a new loop.
    for (const opts of [{}, { freq: 0.5, ngram: 3 }]) {
        const label = Object.keys(opts).length ? 'with penalties on' : 'with penalties off';
        const w = new World();
        w.join(0, 0, 4);
        w.join(1, 4, N);
        w.join(2, 4, N);
        const asker = w.join(100);
        const got = await streamed(asker, PROMPT, BUDGET, {
            opts,
            between: (n) => {
                if (n === 3) w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
            },
        });
        eq(`the head leaves, ${label}: the reference tokens`, got, CONTROL);
        eq('  no rebuild', asker.repairs, 0);
        eq('  no ledger violations', w.violations, []);
    }

    // A zombie: in the roster and dropping every message, until the roster
    // is updated and it is removed.
    for (const at of ['middle', 'head']) {
        const w = new World();
        w.join(0, 0, 3);
        w.join(1, 3, 6);
        w.join(2, 3, 6);
        w.join(3, 6, N);
        w.join(4, 6, N);
        const asker = w.join(100);
        const pair = at === 'middle' ? [1, 2] : [3, 4];
        let z = null;
        const got = await streamed(asker, PROMPT, BUDGET, {
            between: (n) => {
                if (n === 2) { z = pair.find((id) => w.tabs.get(id).runner.real.live.size > 0); w.zombie(z); }
                if (n === 5 && z != null) w.kill(z);
            },
        });
        eq(`a zombie at the ${at}: the reference tokens`, got, CONTROL);
        eq('  no ledger violations', w.violations, []);
    }

    // Slow links, with and without a draining queue.
    {
        const w = new World();
        w.join(0, 0, 4);
        w.join(1, 4, N);
        const asker = w.join(100);
        w.slowLink(0, 1, (h) => (h.k === 'job' ? 3 : 0));
        w.slowLink(1, 0, (h) => (h.k === 'job' ? 3 : 0));
        eq('slow hand-offs both ways round the loop: the reference tokens',
            await streamed(asker, PROMPT, BUDGET), CONTROL);
        eq('  with nothing repaired', [w.tabs.get(0).router.repaired, w.tabs.get(1).router.repaired, asker.repairs], [0, 0, 0]);
    }
}

console.log('\na backed-up link is a refusal, not a departure');
{
    // Two peers behind a slow relay: the first peer's queue to the second goes
    // above the high-water mark. The hand-off is refused as `busy`, nothing is
    // replayed or rebuilt, and the chat continues once the queue has drained.
    // Treating it as "peer unreachable" would send every rebuild over the same
    // full link.
    const NEXT = [9, 8, 7];
    const ctl = chain([0, 4, N]);
    await generate(ctl.asker, PROMPT, 4);
    const want = await generate(ctl.asker, NEXT, 4);

    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    await generate(asker, PROMPT, 4);
    const before = [asker.router.session, [...asker.router.history]];

    w.backedUp.add('0>1');
    const err = await generate(asker, NEXT, 4).then(() => null, (e) => e);
    ok('the turn is refused as busy', err?.busy === true, `${err?.message}`);
    ok('naming the link', /the link to peer 1 is backed up/.test(err?.message ?? ''), `${err?.message}`);
    eq('nothing is repaired and nothing rebuilt', [w.tabs.get(0).router.repaired, asker.repairs], [0, 0]);
    ok('no replay goes down the link', !w.wire.some((m) => m.replay));
    eq('the chat keeps its session and its context', [asker.router.session, [...asker.router.history]], before);

    w.backedUp.clear();
    eq('asked again once the link has drained, the reference tokens', await generate(asker, NEXT, 4), want);
    eq('still with nothing rebuilt', [w.tabs.get(0).router.repaired, asker.repairs], [0, 0]);
    eq('and no ledger violations', w.violations, []);
}

console.log('\nclosed loop: a link that backs up mid-reply ends the reply, and nothing else');
{
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    let session = null;
    const got = [];
    const err = await streamed(asker, PROMPT, 40, {
        between: (n) => {
            got.push(n);
            if (n === 3) { session = asker.router.session; w.backedUp.add('0>1'); }
        },
    }).then(() => null, (e) => e);
    ok('the reply ends as busy', err?.busy === true, `${err?.message}`);
    ok('short of its budget', got.length >= 4 && got.length < 40, `${got.length} tokens`);
    eq('with nothing repaired and nothing rebuilt',
        [w.tabs.get(0).router.repaired, w.tabs.get(1).router.repaired, asker.repairs], [0, 0, 0]);
    ok('no replay goes down the link', !w.wire.some((m) => m.replay));
    eq('the chat keeps its session', asker.router.session, session);
    eq('and what it has is a prefix of the reference',
        asker.router.history.slice(PROMPT.length), CONTROL_F.slice(0, asker.router.history.length - PROMPT.length));
    ok('nothing is still feeding the loop', await quiet(w));
    eq('no ledger violations', w.violations, []);
}

console.log('\nclosed loop: a head with no open link to the entry yet');
{
    // The head links to every holder of position 0, but a link can still be
    // opening. Until it opens, the head declines and the enqueuer feeds each
    // token itself. Once it opens, the loop closes.
    // There are two holders of position 0, so answers still reach the
    // enqueuer through the other one while the head's link to this session's
    // entry (peer 0, the lower id) is down.
    const w = new World();
    w.join(0, 0, 4);
    w.join(2, 0, 4);
    w.join(1, 4, N);
    const asker = w.join(100);
    const head = w.tabs.get(1).net;
    const links = head.links;
    head.links = new Map([...links].filter(([id]) => id !== 0));
    const got = await streamed(asker, PROMPT, BUDGET, {
        between: (n) => { if (n === 3) head.links = links; },
    });
    eq('the reference tokens either way', got, CONTROL);
    ok('fed from the enqueuer while the link was down, closed at the head once it was up',
        jobsFrom(w, 100).length > 1 && jobsFrom(w, 100).length < BUDGET + 1,
        `${jobsFrom(w, 100).length} jobs from the enqueuer`);
    eq('with nothing failing on the way', [asker.repairs, w.wire.filter((m) => m.k === 'err').length], [0, 0]);
}

console.log('\nclosed loop: peers from before it');
{
    // A head with the loop off answers without feeding, and the enqueuer feeds
    // every token itself. Nothing else changes.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N, { closedLoop: false });
    const asker = w.join(100);
    const got = await streamed(asker, PROMPT, BUDGET);
    eq('an old head: the reference tokens', got, CONTROL);
    eq('  every token sent from the enqueuer, as before', jobsFrom(w, 100).length, BUDGET + 1);
    ok('  and no answer claims to be fed', w.wire.filter((m) => m.k === 'tok').every((m) => m.fed === undefined));

    // A looping and a non-looping head at the same position: the session uses
    // whichever it is pinned to.
    for (const oldFirst of [true, false]) {
        const m = new World();
        m.join(0, 0, 4);
        m.join(1, 4, N, { closedLoop: !oldFirst });
        m.join(2, 4, N, { closedLoop: oldFirst });
        const a = m.join(100);
        eq(`a mixed pair of heads (${oldFirst ? 'old' : 'new'} first): the reference tokens`,
            await streamed(a, PROMPT, BUDGET), CONTROL);
    }

    // A non-looping head leaves mid-reply and a looping one takes over, and
    // the reverse.
    for (const oldGoes of [true, false]) {
        const m = new World();
        m.join(0, 0, 4);
        m.join(1, 4, N, { closedLoop: !oldGoes });
        m.join(2, 4, N, { closedLoop: oldGoes });
        const a = m.join(100);
        const got = await streamed(a, PROMPT, BUDGET, {
            between: (n) => {
                if (n === 3) m.kill([1, 2].find((id) => m.tabs.get(id).runner.real.live.size > 0));
            },
        });
        eq(`the ${oldGoes ? 'old' : 'new'} head leaves for the other: the reference tokens`, got, CONTROL);
    }
}

console.log('\nclosed loop: a send that fails on the spot, from inside an answer');
{
    // The head does not loop, so the enqueuer sends every token itself. The
    // session's entry leaves mid-reply, so the next send fails synchronously:
    // its error comes back through deliver() while accept() is still handling
    // the answer, and the stream restarts under a new session. That answer
    // must not be counted as the new session's.
    const w = new World();
    w.join(0, 0, 4);
    w.join(2, 0, 4);
    w.join(1, 4, N, { closedLoop: false });
    const asker = w.join(100);
    const got = await streamed(asker, PROMPT, BUDGET, {
        between: (n) => {
            if (n === 2) w.kill([0, 2].find((id) => w.tabs.get(id).runner.real.live.has(asker.router.session)));
        },
    });
    eq('the reference tokens', got, CONTROL);
    eq('and the context is exact', asker.router.history, [...PROMPT, ...CONTROL]);
    eq('after one rebuild', asker.repairs, 1);
    eq('no ledger violations', w.violations, []);
}

console.log('\nclosed loop: every job\'s penalty summary is the one the enqueuer would have sent');
{
    // The head computes the repetition summaries. Every job's `opts` is read
    // from the wire, keyed by its `past`, and the loop's must equal the round
    // trip's.
    const PENS = { rep: 1.1, freq: 0.4, pres: 0.2, ngram: 2 };
    // A prompt with repeated tokens, so every summary is non-empty.
    const REP = [4, 5, 4, 5, 4, 7, 4, 5];
    const summaries = (w) => {
        const out = {};
        for (const m of w.wire) if (m.k === 'job' && m.pos === 0 && !m.replay) out[m.past] = JSON.stringify(m.opts);
        return out;
    };
    const a = chain([0, 4, N]);
    await generate(a.asker, REP, 10, { opts: PENS });
    const b = chain([0, 4, N]);
    await streamed(b.asker, REP, 10, { opts: PENS });
    const want = summaries(a.w), got = summaries(b.w);
    eq('every job, from the prompt to the last token, carries the same summary', got, want);
    ok('  and they are not empty', Object.values(got).some((o) => /"pen"|"ban"/.test(o)));

    // The same across a head change mid-reply. The new head has no context,
    // so the enqueuer must open a new loop.
    const w = new World();
    w.join(0, 0, 4);
    w.join(1, 4, N);
    w.join(2, 4, N);
    const asker = w.join(100);
    await streamed(asker, REP, 10, {
        opts: PENS,
        between: (n) => {
            if (n === 4) w.kill([1, 2].find((id) => w.tabs.get(id).runner.real.live.size > 0));
        },
    });
    eq('the same summaries across a head change mid-reply', summaries(w), want);
}

console.log('\nclosed loop: nothing runs on once nobody wants it');
{
    // The enqueuer's tab closes mid-reply. The head sees it is gone from the
    // roster and stops feeding.
    {
        const w = new World();
        w.join(0, 0, 4);
        w.join(1, 4, N);
        const asker = w.join(100);
        w.slow(1, 2);
        let ranAt = null;
        streamed(asker, PROMPT, 400, {
            between: (n) => {
                if (n !== 2) return;
                w.peers.splice(w.peers.findIndex((p) => p.id === 100), 1);
                w.sync();
                ranAt = w.tabs.get(1).router.ran;
            },
        }).catch(() => {});
        await settleMs(60);
        const ran = w.tabs.get(1).router.ran;
        ok('the enqueuer leaves: the loop stops within a token or two',
            ranAt !== null && ran - ranAt <= 3, `ran ${ran - (ranAt ?? 0)} more positions`);
    }

    // A hand-off that timed out but was only late. The enqueuer rebuilds, and
    // the old loop, whose job still arrives, must not run alongside the new
    // one.
    {
        const w = new World({ maxFixes: 0 });
        w.join(0, 0, 4);
        w.join(1, 4, N);
        const asker = w.join(100);
        let late = 1;
        w.slowLink(1, 0, (h) => (h.k === 'job' && h.past > PROMPT.length + 2 && late-- > 0 ? 40 : 0),
            { drain: false });
        const got = await streamed(asker, PROMPT, BUDGET);
        eq('a loop job that turned up late: the reference tokens', got, CONTROL);
        eq('  after a rebuild', asker.repairs, 1);
        await settleMs(80);
        const first = asker.sessions[0];
        const after = w.wire.filter((m) => m.k === 'job' && m.s === first).length;
        await settleMs(40);
        eq('  and the old session\'s loop ran no further',
            w.wire.filter((m) => m.k === 'job' && m.s === first).length, after);
        ok('  nothing left running at all', await quiet(w));
    }

    // After deleting an idle chat, and after a halted reply, nothing is
    // running.
    {
        const s = chain([0, 4, N]);
        await streamed(s.asker, PROMPT, 30, { stopAfter: 1 });
        ok('a halted reply leaves nothing running', await quiet(s.w));
        s.asker.router.discard();
        ok('nor does deleting its chat', await quiet(s.w));
    }
}

console.log('\nclosed loop: a duplicate of an old job never starts the loop again');
{
    // A repair re-sends a hand-off whose ack was lost, after the reply has
    // ended, so the job reaches the head again. It must feed nothing. The head
    // remembers how far it fed. If the job reaches the entry through another
    // head, the entry drops it as a stale duplicate and sends no answer.
    for (const hybrid of [false, true]) {
        const label = hybrid ? 'hybrid' : 'full attention';
        const w = new World({ hybrid });
        w.join(0, 0, 4);
        w.join(1, 4, N);
        const asker = w.join(100);
        let copy = null;
        const tap = w.tabs.get(1).router.route;
        const reply = await streamed(asker, PROMPT, 8, {
            onStream: () => {
                // Keep a copy of a loop job as it reaches the head.
                w.tabs.get(1).router.route = (h, p, raw, from) => {
                    if (h.k === 'job' && h.loop && h.past === PROMPT.length + 3 && !copy) {
                        copy = { h: { ...h }, p: p && new Float32Array(p) };
                    }
                    return tap(h, p, raw, from);
                };
            },
        });
        w.tabs.get(1).router.route = tap;
        const before = w.wire.filter((m) => m.k === 'job' && m.auto && m.pos === 0).length;
        w.deliver(0, 1, copy.h, copy.p);
        await settleMs(20);
        eq(`${label}: the reply was whole`, reply, CONTROL_OF(hybrid).slice(0, 8));
        eq(`${label}: the head fed nothing from the duplicate`,
            w.wire.filter((m) => m.k === 'job' && m.auto && m.pos === 0).length, before);

        // Sent directly to the entry: a head-fed job for a position the entry
        // is already past. It is dropped and no error is sent.
        const errs = w.wire.filter((m) => m.k === 'err').length;
        w.deliver(1, 0, { k: 'job', s: asker.router.session, key: asker.router.conversation.key, from: 100,
                          pos: 0, past: PROMPT.length + 2, n: 1, ids: [reply[2]], opts: GREEDY,
                          loop: { left: 3 }, auto: true, entry: 0 });
        await settleMs(20);
        eq(`${label}: a duplicate reaching the entry is dropped, and nobody answered`,
            w.wire.filter((m) => m.k === 'err').length - errs, 0);
        eq(`${label}: no ledger violations`, w.violations, []);
        const next = await streamed(asker, TURN2, 3);
        const ctl = await generate(chain([0, N], {}, { hybrid }).asker, [...PROMPT, ...reply, ...TURN2], 3);
        eq(`${label}: and the chat carries on untouched`, next, ctl);
    }
}

console.log('\nclosed loop: only the loop the enqueuer opened last may feed a session');
{
    // A job of a loop the enqueuer has abandoned, revived by a repair that
    // finished late. It reaches the entry with a `past` equal to the rows the
    // entry holds, so the row counts would accept it. The entry knows the
    // session's current loop id and drops it, and the chat is unaffected.
    for (const hybrid of [false, true]) {
        const label = hybrid ? 'hybrid' : 'full attention';
        const w = new World({ hybrid });
        w.join(0, 0, 4);
        w.join(1, 4, N);
        const asker = w.join(100);
        const LONG = Array.from({ length: T_MAX + 3 }, (_, i) => (i * 5) % 29);
        // A job of the first turn's loop, captured as it reached the entry.
        let old = null;
        const tap = w.tabs.get(0).router.route;
        w.tabs.get(0).router.route = (h, p, raw, from) => {
            if (h.k === 'job' && h.auto && !old) old = { ...h };
            return tap(h, p, raw, from);
        };
        const one = await streamed(asker, PROMPT, 4);
        w.tabs.get(0).router.route = tap;
        const two = await streamed(asker, LONG, 3);     // a new turn, two chunks long
        const ranBefore = w.tabs.get(0).router.ran;
        // The old loop's job, with `past` set to the rows the entry holds.
        w.deliver(1, 0, { ...old, past: asker.router.history.length, loop: { ...old.loop, left: 5 } });
        await settleMs(20);
        eq(`${label}: the entry ran nothing for a loop that is no longer the session's`,
            w.tabs.get(0).router.ran, ranBefore);
        const three = await streamed(asker, TURN2, 4);
        const ctl = await generate(chain([0, N], {}, { hybrid }).asker, [...PROMPT, ...one, ...LONG, ...two, ...TURN2], 4);
        eq(`${label}: and the chat carries on untouched`, three, ctl);
        eq(`${label}: no ledger violations`, w.violations, []);
    }
}

console.log('\nclosed loop: several people at once');
for (const slots of [3, 1]) {
    // Three enqueuers through the same three peers. With one slot each they
    // evict one another mid-reply repeatedly. Every eviction is detected and
    // rebuilt, and each reply matches its own chat.
    const prompts = [PROMPT, [9, 8, 7, 6], [21, 3]];
    const controls = [];
    for (const p of prompts) controls.push(await generate(chain([0, N]).asker, p, 10));
    const w = new World({ maxRepairs: 200 });
    w.join(0, 0, 3, { slots });
    w.join(1, 3, 6, { slots });
    w.join(2, 6, N, { slots });
    const askers = prompts.map((_, i) => w.join(100 + i));
    for (const id of [0, 1, 2]) w.slow(id, 1);
    const got = await Promise.all(askers.map((a, i) => streamed(a, prompts[i], 10)));
    eq(`${slots} slot${slots === 1 ? '' : 's'} per peer, three enqueuers: every reply is its own`, got, controls);
    if (slots === 1) ok('  and they really did evict one another', askers.some((a) => a.repairs > 0));
    eq('  no ledger violations', w.violations, []);
    ok('  nothing left running', await quiet(w));
}

console.log('\nclosed loop: conversations');
{
    // Three cases: parked and resumed with its caches intact, evicted while
    // parked, and loaded from storage with no session.
    const ctl = await twoTurns();
    for (const [label, slots, stored] of [['resumed warm', SLOTS, false], ['resumed after eviction', 1, false],
                                          ['loaded from storage', SLOTS, true]]) {
        const w = new World();
        w.join(0, 0, 4, { slots });
        w.join(1, 4, N, { slots });
        const asker = w.join(100);
        eq(`${label}: the first turn`, await streamed(asker, PROMPT, 3), ctl.one);
        let a = asker.router.conversation;
        asker.router.newChat();
        await streamed(asker, [3, 1, 4], 2);
        if (stored) a = makeConversation({ mesh: 'global', history: [...a.history] });
        asker.router.switchTo(a);
        eq(`${label}: the next turn continues it`, await streamed(asker, TURN2, 3), ctl.two);
        eq(`${label}: no ledger violations`, w.violations, []);
    }
}

// --- The fuzz: random meshes, random askers, random replies ---
//
// Combines everything above at random: random cuts and depth, a hybrid model
// or not, one to three askers (some spectators, some tabs holding part of the
// model), the loop or the round trip, some non-looping heads, several turns
// each with new chats and switches between them, replies halted, cut short and
// rewound, and peers leaving mid-reply wherever the mesh can survive it.
//
// The oracle does not predict the outcome. It checks what happened: every
// turn's context must be the previous context, plus its prompt, plus a
// continuation that a single peer using the round trip produces, token for
// token, from that context. Afterwards nothing may be running.
//
//   node tools/mesh.test.mjs --fuzz 2000 --fuzz-seed 12345

const fuzzArg = process.argv.indexOf('--fuzz');
const FUZZ = fuzzArg > 0 ? Number(process.argv[fuzzArg + 1]) : 60;
const fuzzSeedArg = process.argv.indexOf('--fuzz-seed');
const FUZZ_SEED = fuzzSeedArg > 0 ? Number(process.argv[fuzzSeedArg + 1]) >>> 0 : (Date.now() >>> 0) ^ 0x5bd1e995;

// The toy runners hold 4·T_MAX positions. Conversations stay within that.
const ROOM = 4 * T_MAX - 4;

// Whether a single peer using the round trip produces `continuation` after
// `context`. Returns the index of the first token that differs, or -1.
async function diverges(hybrid, context, continuation) {
    const { asker } = chain([0, N], {}, { hybrid });
    let tok = null;
    for (let i = 0; i < context.length; i += T_MAX) {
        ({ token: tok } = await asker.router.submit(context.slice(i, i + T_MAX), GREEDY));
    }
    for (let i = 0; i < continuation.length; i++) {
        if (continuation[i] !== tok) return i;
        ({ token: tok } = await asker.router.submit([continuation[i]], GREEDY));
    }
    return -1;
}

async function fuzzOne(seed) {
    const rand = rng(seed);
    const pick = (xs) => xs[Math.floor(rand() * xs.length)];
    const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
    const hybrid = rand() < 0.4;
    const slots = int(1, 3);
    // Usually the default admission rule, scaled to this suite's timing: an
    // active session is never evicted and a new one is refused as busy.
    // Sometimes the opposite: every session can be evicted at any moment, so
    // askers take one another's caches mid-reply and mid-rebuild, with a small
    // repair budget. Turns may then fail, but must never give wrong tokens.
    const thrash = rand() < 0.15;
    const w = new World({ hybrid, maxRepairs: thrash ? 3 : 200, idleMs: thrash ? 0 : 25 });
    const ranges = randomMesh(rand, { deep: rand() < 0.7 });
    ranges.forEach(([lo, hi], i) => w.join(i, lo, hi, { slots, closedLoop: rand() > 0.15 }));
    const workers = ranges.map((_, i) => i);

    // One to three askers: each a spectator or one of the workers.
    const askers = [];
    for (let j = 0, n = int(1, 3); j < n; j++) {
        const own = rand() < 0.3 ? workers.find((id) => !askers.includes(w.tabs.get(id))) : undefined;
        askers.push(own !== undefined ? w.tabs.get(own) : w.join(100 + j));
    }
    const eosId = rand() < 0.2 ? int(0, D.vocab - 1) : -1;
    for (const tab of w.tabs.values()) tab.router.eos = eosId;
    for (const id of workers) if (rand() < 0.3) w.slow(id, int(1, 3));
    const askerIds = new Set(askers.map((a) => a.id));

    const problems = [];
    const where = (a, t) => `fuzz seed ${seed}: asker ${a.id} turn ${t}`;

    async function person(a) {
        const chats = [a.router.conversation];
        for (let t = 0, turns = int(1, 4); t < turns; t++) {
            // Occasionally start a new chat or switch to an earlier one.
            const r = rand();
            if (r < 0.15) chats.push(a.router.newChat());
            else if (r < 0.25 && chats.length > 1) a.router.switchTo(pick(chats));

            const budget = int(1, 12);
            let room = ROOM - a.router.ctxLen - budget;
            if (room < 2) { chats.push(a.router.newChat()); room = ROOM - budget; }
            const prompt = Array.from({ length: int(1, Math.min(2 * T_MAX + 5, room)) },
                () => int(0, D.vocab - 1));
            const loop = rand() < 0.8;
            const action = pick(['none', 'none', 'stop', 'cut']);
            const k = int(1, budget);
            const killAt = rand() < 0.3 ? int(0, budget - 1) : -1;
            const before = [...a.router.history];

            const between = (n) => {
                if (n !== killAt) return;
                const victims = workers.filter((id) => w.tabs.has(id) && !askerIds.has(id) && w.survivable(id));
                if (victims.length) w.kill(pick(victims));
            };

            let reply, err = null;
            try {
                reply = loop
                    ? await streamed(a, prompt, budget, { between,
                        ...(action === 'stop' && { stopAfter: k }), ...(action === 'cut' && { breakAfter: k }) })
                    : await generate(a, prompt, action === 'none' ? budget : Math.min(k, budget), { between });
            } catch (e) { err = e; }

            const history = [...a.router.history];
            const head = history.slice(0, before.length);
            if (JSON.stringify(head) !== JSON.stringify(before)) {
                problems.push(`${where(a, t)}: the context before the turn changed`);
                continue;
            }
            const tail = history.slice(before.length);
            if (err) {
                // A capacity refusal is always allowed. In thrash mode a turn
                // that ran out of repairs is allowed too. In both cases, what
                // the context gained must be a prefix of what a single peer
                // produces.
                const allowed = err.busy
                    || (thrash && /no longer cached|is reachable|unreachable|no answer from the mesh/.test(err.message));
                if (!allowed) { problems.push(`${where(a, t)}: ${err.message}`); continue; }
                const part = tail.slice(0, prompt.length);
                if (JSON.stringify(part) !== JSON.stringify(prompt.slice(0, part.length))) {
                    problems.push(`${where(a, t)}: a failed turn left a context that is not its prompt`);
                } else if (tail.length > prompt.length
                           && await diverges(hybrid, [...before, ...prompt], tail.slice(prompt.length)) >= 0) {
                    problems.push(`${where(a, t)}: a failed turn left a reply that is not the reference's`);
                }
                continue;
            }
            if (JSON.stringify(tail.slice(0, prompt.length)) !== JSON.stringify(prompt)) {
                problems.push(`${where(a, t)}: the prompt is not what the context gained`);
                continue;
            }
            const fed = tail.slice(prompt.length);
            if (JSON.stringify(fed.slice(0, reply.length)) !== JSON.stringify(reply)) {
                problems.push(`${where(a, t)}: the reply shown is not what was fed ` +
                    `(${JSON.stringify(reply)} vs ${JSON.stringify(fed)})`);
                continue;
            }
            const bad = await diverges(hybrid, [...before, ...prompt], fed);
            if (bad >= 0) {
                problems.push(`${where(a, t)}: token ${bad} of the reply is not the reference's`);
                continue;
            }
            if (fed.length > budget) problems.push(`${where(a, t)}: ${fed.length} fed past a budget of ${budget}`);
            if (loop && action === 'none' && eosId < 0 && fed.length !== budget) {
                problems.push(`${where(a, t)}: ${fed.length} of a budget of ${budget}`);
            }
            // Only the loop is checked for end-of-text: the round-trip helper
            // in this file does not stop at it.
            if (loop && fed.includes(eosId)) problems.push(`${where(a, t)}: end-of-text was fed`);
            // A cut reply: rewind to what was kept, as turn.js does.
            if (loop && action === 'cut' && fed.length > k) a.router.rewind(fed.length - k);
        }
    }

    await Promise.all(askers.map(person));
    if (w.violations.length) problems.push(`fuzz seed ${seed}: ${w.violations[0]}`);
    if (!(await quiet(w, 30))) problems.push(`fuzz seed ${seed}: a loop still feeding after every reply ended`);
    return problems;
}

console.log(`\nclosed loop: the fuzz: ${FUZZ} random scenarios  (--fuzz-seed ${FUZZ_SEED})`);
{
    const seeds = rng(FUZZ_SEED);
    const problems = [];
    for (let i = 0; i < FUZZ && problems.length < 5; i++) {
        problems.push(...await fuzzOne(Math.floor(seeds() * 0xffffffff) >>> 0 || 1));
    }
    ok(`every random scenario kept every reply and every context exact`,
        problems.length === 0, `\n        ${problems.slice(0, 5).join('\n        ')}`);
}

console.log(`\nthe soak, closed loop: random meshes, random growth, random kills  (--seed ${SEED0})`);
{
    const rand = rng(SEED0 ^ 0x9e3779b9);
    const s = await soak('survivable', 80, rand, streamed);
    console.log(`        survivable: ${s.runs} runs, ${s.matched} completed, ${s.absorbed} without a rebuild`);
    ok('a departure the mesh could absorb never costs a token, looping',
        s.broken.length === 0 && s.matched === s.runs,
        `\n        ${s.broken.join('\n        ') || `only ${s.matched}/${s.runs} completed`}`);
    const m = await soak('murderous', 80, rand, streamed);
    console.log(`        murderous:  ${m.runs} runs, ${m.matched} completed, ${m.refused} refused loudly`);
    ok('and one it could not is refused rather than answered wrongly, looping',
        m.broken.length === 0, `\n        ${m.broken.join('\n        ')}`);
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail) console.log(`reproduce with:  node tools/mesh.test.mjs --seed ${SEED0} --fuzz-seed ${FUZZ_SEED}`);
process.exit(fail ? 1 : 0);
