// The router: one peer's share of the protocol. One router per peer, so a test
// can run several in one process over a shared roster (tools/mesh.test.mjs).
//
// A job carries a cursor (`pos`). A peer that holds the cursor runs the job to
// its cut and hands it on, and the peer holding the head samples a `tok` and
// sends it back. Two rules hold throughout: the cursor only moves forward, and
// the peer a session is pinned to at a boundary is never re-picked, because it
// holds that session's KV.
//
// The message headers, the answer checks (`key`, Mesh.mayAnswer), repairs, the
// closed decode loop and conversations are described in docs/architecture.md
// (Routing, Wire format, Sessions and generation, Chats, Failures).

import { penaltyState } from './model/sampling.js';

// Longest run of tokens in one job. Chosen for latency: net.js fragments a
// message too big for one datachannel send.
export const T_MAX = 64;

// A peer's load, as reported in its acks, is used for this long.
const LOAD_TTL_MS = 10_000;

// Positions just routed to a peer count towards its load for this long, or
// until its next ack, so a burst of new sessions does not all go one way.
const ASSIGN_TTL_MS = 2000;

const ROUTES = 128;        // (session, boundary) pairs a peer pins a hop for

// The enqueuer's timeout for one job, end to end. Per-hop acks detect most
// failures, so this is sized for the slowest valid job (a long prefill on a
// loaded mesh).
const JOB_TIMEOUT = 300_000;

// How long a peer waits for the next hop to ack a job before treating the hop
// as a failed send. The receiver acks before any GPU work, so this covers
// network RTT plus the time the job takes to cross.
const ACK_TIMEOUT = 1500;

// Slowest transfer rate, in bytes per second, that a hop's wait allows for on
// top of ACK_TIMEOUT. A prefill chunk is ~1.3 MB on the 27B model and can take
// several seconds to cross the TURN relay (policed at 1.25 MB/s per address).
// `bufferedAmount` empties once SCTP has the bytes, before they arrive, so the
// wait is sized from the payload. A decode job is a few KB and adds almost
// nothing.
const ACK_RATE = 256 * 1024;

// Consecutive local repairs one boundary may make before passing the error
// back. Reset when a job crosses the boundary. Bounds replays into a peer that
// answers every job with an error.
const MAX_FIXES = 2;

// Longest boundary log a local repair will replay. A replay costs as much as a
// prefill of the same length. Longer ones go to the enqueuer's recover(),
// which reports progress and can be cancelled.
const MAX_LOCAL_REPLAY = 256;

// Context rebuilds one turn may make before the failure reaches the user.
const MAX_REPAIRS = 2;

// Hop budget for an answer, on top of the length of the line. Bounds an answer
// whose addressee nobody links to.
const TTL = 8;

// How far back the repetition summary looks, and how many ids it carries. It
// adds a few hundred bytes of JSON to each job.
const PEN_WINDOW = 256;
const PEN_TOP = 32;

// Out-of-order answers a stream holds while waiting for an earlier one. They
// arrive out of order only when two routes home interleave (direct and via the
// hub).
const EARLY = 8;

// Monotonic, so two recover() calls in one millisecond get distinct session
// ids.
let minted = 0;

// Whether any repetition setting needs the token history.
export const penalized = (opts) =>
    (opts.rep ?? 1) !== 1 || opts.freq > 0 || opts.pres > 0 || (opts.ngram ?? 0) >= 2;

// A job's sampler settings: the user's, plus the repetition summary for the
// token it will sample. The summary is computed from `ctx`: every token fed so
// far, including this job's ids. The enqueuer builds it for the jobs it sends,
// the head for the jobs a loop feeds. With every penalty off there is no
// summary.
//
// `eos` is skipped: after the first turn the end-of-turn id is in the history,
// and penalizing it would make a long reply less likely to stop.
export function jobOpts(opts, ctx, eos) {
    const base = { temp: opts.temp, topK: opts.topK, rep: opts.rep, freq: opts.freq, pres: opts.pres };
    if (!penalized(opts)) return base;
    const scored = (opts.rep ?? 1) !== 1 || opts.freq > 0 || opts.pres > 0;
    return { ...base, ...penaltyState(ctx, {
        window: PEN_WINDOW, top: scored ? PEN_TOP : 0, ngram: opts.ngram ?? 0, skip: [eos],
    }) };
}

// One chat as the router sees it.
//   history   every token id the peers hold for `session`. A rebuild re-feeds
//             it. A chat loaded from storage has a history and no session.
//   session, key   minted on the first job.
//   parked    the chat was switched away from and back. A failure on the next
//             job is reported as a resume, not a departure.
//   stale     the peers' state cannot be continued from `history` (a rewind
//             through recurrent layers, or an edit in another tab). The next
//             job rebuilds.
export function makeConversation({ mesh = 'global', history = [] } = {}) {
    return { mesh, history: Array.from(history), session: null, key: null,
             parked: false, stale: false };
}

// `runner` is a getter: there is no Runner until the download starts, and
// never one for a spectator. `onRepair(err, tokens, reason)` may return a
// function, called when the rebuild finishes. `reason` is 'repair' (a peer
// failed, `err` says how), 'resume' (a parked or stored chat with no caches)
// or 'rewind' (see rewind()).
//
// `mesh` is the scope conversations are made in and checked against (room.js's
// historyTag). `hybrid` says the checkpoint has linear_attention blocks, whose
// state rewind() cannot trim. `hybrid` and `eos` come from the manifest and
// can be set after construction.
export function makeRouter({
    net,
    mesh = 'global',
    hybrid = false,
    runner = () => null,
    onChange = () => {},
    onRepair = () => {},
    onFix = () => {},
    onProgress = () => {},
    isCancelled = () => false,
    gpu = () => null,              // this tab's adapter name, for a profiled hop
    jobTimeout = JOB_TIMEOUT,
    ackTimeout = ACK_TIMEOUT,
    ackRate = ACK_RATE,
    maxRepairs = MAX_REPAIRS,
    maxFixes = MAX_FIXES,
    maxLocalReplay = MAX_LOCAL_REPLAY,
    // Whether this peer, as the head, runs the closed loop. When off, it
    // answers loop jobs without feeding. Tests use this.
    closedLoop = true,
} = {}) {
    const routes = new Map();      // `${session}|${cursor}` -> who picks it up there
    const upstream = new Map();    // `${session}|${cursor}` -> who handed it over
    const outbound = new Map();    // `${session}|${cursor}` -> the job in flight there
    const fixes = new Map();       // `${session}|${cursor}` -> repairs since a success
    const acks = new Map();        // `${session}|${cursor}` -> { timer, to }: the hop
                                    // that has not yet acked a job
    const pending = new Map();     // session -> { resolve, reject, timer }
    const links = new Map();       // `${session}|${cursor}` -> { rtt, send }: timing of
                                    // the last hand-off across that boundary
    const loads = new Map();       // peer -> { w, p, at }: its active work in positions,
                                    // its ms per position, when it last acked
    const assigned = new Map();    // peer -> { n, at }: positions routed to it since
                                    // its last ack
    const planned = new Map();     // session -> { n, at }: positions I chose to run for
                                    // it, until its cut exists here
    const senders = new Map();     // session -> Set of peers that handed me a job
                                    // of it. They may `free` it here.

    // The head's side of the closed loop, per session: the key its loop jobs
    // carry (a `halt` must match it), the context its penalties are computed
    // from, whether a halt has arrived, and how far it has fed. Kept after the
    // loop ends, so a duplicate job cannot restart it.
    const heading = new Map();     // session -> { key, id, ctx|null, halt, fedTo, ended }
    let loops = 0;                 // the id of the last loop this tab opened
    // The entry's side: the loop the enqueuer opened last, per session. Only
    // that loop may feed the session (runJob()).
    const current = new Map();     // session -> loop id
    // The enqueuer's side: a reply being streamed, per session (decode()).
    const streams = new Map();     // session -> stream state

    // The active conversation, and every session this tab has minted. Answers
    // and errors for a parked chat are also handled here.
    let conv = makeConversation({ mesh });
    const own = new Map();         // session -> conversation

    let repairs = 0;               // rebuilds spent on the turn in flight
    let repaired = 0;              // hand-offs this peer fixed for somebody else
    let ran = 0;                   // token positions this peer has run, all sessions
    let eos = 0;                   // end-of-text id, from the manifest

    // Mint a session and key for `c`. The id says who asked and roughly when,
    // so it is guessable. The key is the secret.
    const newSession = (c) => {
        minted = Math.max(minted + 1, Date.now());
        const bytes = crypto.getRandomValues(new Uint8Array(16));
        c.key = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
        c.session = `${net?.id ?? 'x'}-${minted.toString(36)}`;
        own.set(c.session, c);
    };

    // --- Routing ---

    // Also checks readiness: submitJob() calls route() locally, which skips
    // hops()'s ready filter, and a peer that is still downloading must not
    // take its own job.
    function iCanRun(pos) {
        return !!runner()?.holds(pos) && !!net.me?.ready;
    }

    // `raw` is the buffer a message arrived in, forwarded as-is. `from` is the
    // peer on the other end of the channel it arrived on, or null for a message
    // this tab produced. `from` comes from the connection, not the header, so
    // the reverse path is authenticated.
    function route(h, payload, raw = null, from = null) {
        const mine = h.k === 'job' ? iCanRun(h.pos) : h.to === net.id;
        if (mine) return deliver(h, payload, from);
        if (h.to != null && !net.has(h.to)) return;   // its addressee left

        // Relay a token only from a peer that could have sampled or carried it.
        if (h.k === 'tok' && from != null && !net.mayAnswer(from)) {
            console.warn(`dropped a tok for peer ${h.to} from peer ${from}, which cannot answer`);
            return;
        }

        if (h.k === 'job') return forward(h, payload, raw, from);

        // An answer can circle when its addressee is in the roster but nobody
        // has an open link to it yet. Answers are ~60 bytes, so repacking with
        // a decremented ttl is cheap.
        const ttl = (h.ttl ?? (TTL + (net.positions || 0))) - 1;
        if (ttl <= 0) {
            console.warn(`dropped a ${h.k} for peer ${h.to}: hop budget spent`);
            return;
        }
        h = { ...h, ttl };

        // Anything I cannot reach goes to the hub, which links to everyone.
        const to = net.linked(h.to) ? h.to : net.hub;
        if (!net.send(to, h, payload)) fail(h, to);
    }

    // Separate from route() because runJob() must not go back through iCanRun:
    // a peer stops early when another holder reaches further.
    function forward(h, payload, raw = null, from = null) {
        // Keyed by the cursor the job leaves at (runJob() already moved it), so
        // `upstream` and `routes` share keys and fail() can look it up.
        // Recorded before the send so it is set even when the send fails.
        if (from != null) remember(h.s, h.pos, from);

        // Try every candidate. A pinned boundary has at most one (see hops()).
        const key = `${h.s}|${h.pos}`;
        const cands = hops(h.s, h.pos);
        for (const to of cands) {
            if (!net.send(to, raw ?? h, payload)) continue;
            // `outbound` keeps the latest job, to re-send if this boundary
            // breaks. A successful crossing resets the repair budget. A replay
            // does neither: it is part of a repair, and counting it would let
            // two peers reset each other's budget indefinitely.
            if (!h.replay) { put(outbound, key, h); fixes.delete(key); armAck(key, h, payload, raw, to); }
            return pin(h.s, h.pos, to);
        }

        // A hop that is present but backed up refuses the job. The session's
        // caches are intact, and a repair would add a replay to the load.
        const full = cands.find((id) => net.backedUp?.(id));
        if (full !== undefined) return fail(h, full, true);

        // Only a pinned boundary is repaired. Unpinned, the loop above already
        // tried every candidate.
        if (routes.has(key) && repair(h, payload, raw, cands[0] ?? null)) return;
        return fail(h, cands[0] ?? null);
    }

    // Who this session's cursor may be handed to at `pos`, best first. A pinned
    // boundary returns only its pin, or nothing if that peer is not a ready
    // holder of `pos`. Nothing means an error to repair; the pin is not
    // re-picked here.
    //
    // Keyed by boundary and session: a tab holding 10-19 that also prompts
    // pins a hop at 0 for its own jobs and another at 20 for the work it ran.
    function hops(s, pos) {
        const pinned = routes.get(`${s}|${pos}`);
        if (pinned !== undefined) {
            return net.next(pos).includes(pinned) ? [pinned] : [];
        }

        return rankAt(pos);
    }

    // Ready holders of `pos`, best first: an open link (ICE costs a second),
    // then the least loaded, then the one that reaches furthest. With no load
    // reported anywhere, that is reach alone. Never me: I either declined
    // this job in route() or chose to stop here.
    function rankAt(pos) {
        const ids = net.next(pos).filter((id) => id !== net.id);
        const ls = new Map(ids.map((id) => [id, loadOf(id)]));
        const timed = ids.every((id) => ls.get(id).p != null);
        const rank = (id) => [net.linked(id) ? 0 : 1, cost(ls.get(id), 0, timed),
                              -net.reach(id, pos), id];
        return ids.sort((a, b) => {
            const [x, y] = [rank(a), rank(b)];
            return (x[0] - y[0]) || (x[1] - y[1]) || (x[2] - y[2]) || (x[3] - y[3]);
        });
    }

    // A peer's active work in positions (`w`) and its ms per position (`p`,
    // null if unknown). Mine from my runner; another's from its last ack,
    // plus what I routed to it since. A report older than LOAD_TTL_MS, or
    // none (an older page), counts as idle.
    function loadOf(id) {
        if (id === net.id) {
            const r = runner();
            // Sessions planned here but not yet run count too, so a burst of
            // new chats does not all see an idle peer.
            const now = performance.now();
            let w = r?.load ?? 0;
            for (const [s, x] of planned) {
                if (now - x.at >= ASSIGN_TTL_MS || r?.cuts?.has(s)) planned.delete(s);
                else w += x.n;
            }
            return { w, p: r?.msPerPos ?? null };
        }
        const now = performance.now();
        const l = loads.get(id);
        const fresh = l && now - l.at < LOAD_TTL_MS;
        const a = assigned.get(id);
        const extra = a && now - a.at < ASSIGN_TTL_MS ? a.n : 0;
        return { w: (fresh ? l.w : 0) + extra, p: fresh ? l.p ?? null : null };
    }

    // A peer's work per token round, in ms when every peer compared has a
    // measured speed, otherwise in positions.
    const cost = (l, more, timed) => (l.w + more) * (timed ? l.p : 1);

    // Count `n` positions as on their way to `id` until it next acks.
    function assign(id, n) {
        const now = performance.now();
        const a = assigned.get(id);
        const prior = a && now - a.at < ASSIGN_TTL_MS ? a.n : 0;
        assigned.set(id, { n: prior + n, at: now });
    }

    // Every per-boundary map is capped at ROUTES, oldest out first.
    function put(map, key, value) {
        if (!map.has(key) && map.size >= ROUTES) map.delete(map.keys().next().value);
        map.set(key, value);
    }

    // Pinned on send success only. A failed send allocated no KV cache.
    function pin(s, pos, to) {
        const key = `${s}|${pos}`;
        if (routes.has(key)) return;
        put(routes, key, to);
    }

    // Record who handed this session's cursor over at `pos`. The first sender
    // is kept: a session enters a peer from one place only.
    function remember(s, pos, id) {
        const key = `${s}|${pos}`;
        if (upstream.has(key)) return;
        put(upstream, key, id);
    }

    // Wait for the next hop to ack the job just sent. Expiry is handled like a
    // failed send. Not armed for a replay, which is never acked.
    //
    // The far end acks once the whole message has arrived, but send() returns
    // when it is queued. A prefill's residual stream is half a megabyte or
    // more and can take seconds to cross a relayed link. So if the link's send
    // queue is still shrinking at expiry, the wait is extended. A queue that
    // stops shrinking, or a delivered message that is never acked, fails.
    //
    // The first wait also allows for the message's size at ACK_RATE, since the
    // queue empties into SCTP before the bytes arrive.
    function armAck(key, h, payload, raw, to) {
        clearAck(key);   // a new send at this boundary replaces any earlier wait
        const wait = { to, sentAt: performance.now(), queued: net.buffered?.(to) ?? 0 };
        const bytes = raw?.byteLength ?? payload?.byteLength ?? 0;
        const first = ackTimeout + bytes / ackRate * 1000;
        const expire = () => {
            const queued = net.buffered?.(to) ?? 0;
            if (wait.queued > 0 && queued < wait.queued) {
                wait.queued = queued;
                wait.timer = setTimeout(expire, ackTimeout);
                return;
            }
            acks.delete(key);
            // Backed up and not draining: a busy refusal.
            if (net.backedUp?.(to)) return fail(h, to, true);
            if (routes.has(key) && repair(h, payload, raw, to)) return;
            fail(h, to);
        };
        wait.timer = setTimeout(expire, first);
        // `sentAt` is taken after the send returned, so the round trip
        // excludes packing, which the caller times separately.
        acks.set(key, wait);
    }

    function clearAck(key) {
        const w = acks.get(key);
        if (!w) return;
        clearTimeout(w.timer);
        acks.delete(key);
    }

    // An `err` header carries no cursor, so the previous hop is looked up by
    // session alone. A peer that is not the enqueuer sees a session enter at
    // one place, so there is at most one match.
    function backwards(s) {
        for (const [key, id] of upstream) if (key.startsWith(`${s}|`)) return id;
        return null;
    }

    // The boundary this peer hands `s` over at. Not called for the enqueuer's
    // own sessions.
    function outgoing(s) {
        for (const key of outbound.keys()) if (key.startsWith(`${s}|`)) return key;
        return null;
    }

    // Everything this tab remembers about a session's routing. Separate from
    // forget() because a `free` for another tab's session needs only this and
    // not `pending`, which belongs to the enqueuer.
    function drop(s) {
        for (const key of routes.keys()) {
            if (key.startsWith(`${s}|`)) routes.delete(key);
        }
        for (const key of upstream.keys()) {
            if (key.startsWith(`${s}|`)) upstream.delete(key);
        }
        for (const key of outbound.keys()) {
            if (key.startsWith(`${s}|`)) { outbound.delete(key); fixes.delete(key); }
        }
        // clearAck, not delete: the value holds a running timer.
        for (const key of acks.keys()) {
            if (key.startsWith(`${s}|`)) clearAck(key);
        }
        for (const key of links.keys()) {
            if (key.startsWith(`${s}|`)) links.delete(key);
        }
        senders.delete(s);
        heading.delete(s);
        current.delete(s);
    }

    // Tell every peer this tab handed `s` to that the session is over, so its
    // KV goes back to the pool. Each receiver passes it on along its own pins.
    //
    // Advisory: no ack, no retry. An evicted pin, a repair or a dropped message
    // can stop it early, and the runner's idle eviction reclaims the frame
    // anyway.
    //
    // It cannot loop: #exit() puts every cut past the entry cursor, so pins
    // only point further along the line.
    function freeSession(s) {
        for (const key of routes.keys()) {
            if (!key.startsWith(`${s}|`)) continue;
            route({ k: 'free', s, to: routes.get(key) });
        }
    }

    // A clean end of a session: release its cache here (a tab that holds blocks
    // and prompts runs its own first stretch), free it along the chain, then
    // forget it locally. The order matters: freeSession() reads the pins that
    // forget() deletes.
    //
    // Not used by recover(). A failed session's repair traffic (an `err`
    // travelling back, a replay travelling forward) may be in flight and needs
    // the caches and boundary logs. Those are left to idle eviction.
    function retire(s) {
        runner()?.release(s);
        freeSession(s);
        forget(s);
    }

    // Local only: no release, no message. Safe to call during a failure.
    function forget(s) {
        drop(s);
        own.delete(s);
        const job = pending.get(s);
        if (!job) return;
        clearTimeout(job.timer);
        pending.delete(s);
    }

    // --- Local repair: fix the hand-off at the peer that made it ---

    // Send a boundary's rows to `to`, chunked at T_MAX, each chunk with its own
    // `past`.
    //
    // Starts from row 0 and is forwarded by the ordinary rule. The replacement
    // picks its own cut, so later peers may take the session at different
    // cursors. Rewriting the whole path with the same inputs is idempotent.
    //
    // `upto` is the failed job's `past`, not the log's length: the log also
    // holds the job's own rows, which the re-sent job carries.
    //
    // Refuses when the log is shorter than `upto`. A log restarted after its
    // session was released or evicted here can be shorter than the job, and
    // replaying past its end would send rows of zeros as the residual stream.
    function replay(s, key, at, upto, to) {
        const log = runner()?.logs.get(s)?.get(at);
        if (!log || log.len < upto) return false;
        const H = runner().d.hidden;
        for (let i = 0; i < upto; i += T_MAX) {
            const n = Math.min(T_MAX, upto - i);
            const h = { k: 'job', s, key, from: net.id, pos: at, past: i, n, replay: true };
            if (!net.send(to, h, log.rows.subarray(i * H, (i + n) * H))) return false;
        }
        return true;
    }

    // The peer that detects a dead next hop holds the log for that boundary,
    // so it repairs locally: drop the pin, pick another holder, replay what
    // this boundary already handed over, and send the job again.
    //
    // `failed` is excluded only when the send failed, since that link is down.
    // A peer that answered with an `err` is reachable and may be the only
    // holder of the position. An evicted cache there needs the replay, so the
    // err path passes null.
    //
    // Links are ordered and Runner.run queues, so the chunks and the job after
    // them execute in order at every hop.
    function repair(h, payload, raw, failed) {
        if (h.past > maxLocalReplay) return false;

        const key = `${h.s}|${h.pos}`;
        if ((fixes.get(key) ?? 0) >= maxFixes) return false;

        const pinned = routes.get(key);
        routes.delete(key);                      // so hops() ranks every holder
        for (const to of hops(h.s, h.pos)) {
            if (to === failed) continue;
            if (!replay(h.s, h.key, h.pos, h.past, to)) continue;
            if (!net.send(to, raw ?? h, payload)) continue;
            put(fixes, key, (fixes.get(key) ?? 0) + 1);
            pin(h.s, h.pos, to);
            if (!h.replay) armAck(key, h, payload, raw, to);   // the replacement may be dead too
            repaired++;
            onChange();
            // Tell the enqueuer (`h.from`). This is not an answer.
            route({ k: 'fix', s: h.s, key: h.key, to: h.from, from: net.id, at: h.pos, rows: h.past });
            return true;
        }
        // Nothing took it. Restore the pin, so the boundary is not re-picked
        // onto a cold cache.
        if (pinned !== undefined) routes.set(key, pinned);
        return false;
    }

    // This message cannot be sent on. For a job, answer the sender with an
    // `err` now so it does not wait for the timeout. An undeliverable answer
    // is dropped.
    //
    // `busy` means the hop is present but its link is backed up. It is a
    // capacity refusal like a runner's and is not repaired or rebuilt, since
    // either would send more over the full link.
    function fail(h, to, busy = false) {
        const why = busy ? `the link to peer ${to} is backed up; try again in a moment`
            : to == null ? `no peer that can run position ${h.pos} is reachable`
            : `peer ${to} is unreachable`;
        if (h.k !== 'job') return console.warn('dropped a message:', why);

        // Back one hop, to the peer that handed the job over. `h.from` covers
        // the enqueuer's own job.
        const back = upstream.get(`${h.s}|${h.pos}`) ?? h.from;
        route({ k: 'err', s: h.s, key: h.key, to: back, from: net.id, msg: why,
                busy: busy || undefined });
    }

    function deliver(h, payload, from = null) {
        if (h.k === 'job') {
            // A set: a repair can hand the same session in from another peer,
            // and either may free it.
            if (from != null) put(senders, h.s, (senders.get(h.s) ?? new Set()).add(from));
            // Ack before any GPU work. Not for a job this tab handed itself
            // (`from == null`) or a replay; nobody waits on those.
            if (from != null && !h.replay) {
                const r = runner();
                net.send(from, { k: 'ack', s: h.s, to: from, at: h.pos,
                                 ...(Number.isFinite(r?.load) && { w: r.load }),
                                 ...(r?.msPerPos != null && { p: Math.round(r.msPerPos * 1000) / 1000 }) });
            }
            return runJob(h, payload, from);
        }

        if (h.k === 'ack') {
            // The receiver's load, which plan() and rankAt() balance on.
            if (from != null && Number.isFinite(h.w)) {
                loads.set(from, { w: h.w, p: Number.isFinite(h.p) ? h.p : null, at: performance.now() });
                assigned.delete(from);
            }
            const key = `${h.s}|${h.at}`;
            const w = acks.get(key);
            // Ignore an ack from a peer other than the one being waited on
            // (the wait can be re-armed for another peer after a repair).
            if (w && w.to === from) {
                // The receiver acks before any GPU work, so this is the hop's
                // network round trip. It is reported on the next job across
                // this boundary (runJob), since this job's trace has left.
                const rtt = Math.round((performance.now() - w.sentAt) * 100) / 100;
                put(links, key, { ...links.get(key), rtt });
                clearAck(key);
            }
            return;
        }

        // The session is over: release it here and pass the `free` along this
        // peer's own pins. Never acked or answered (see freeSession()).
        // Accepted only from a peer that handed the session over, so other
        // peers cannot drop a live chat's cache.
        if (h.k === 'free') {
            if (from != null && !senders.get(h.s)?.has(from)) {
                // No senders is normal for a tab that prompts and also holds
                // the end of the line: its own free comes back along the pins
                // after it has forgotten the session.
                if (senders.has(h.s)) {
                    console.warn(`ignored a free for ${h.s} from peer ${from}, which did not hand it over`);
                }
                return;
            }
            retire(h.s);
            return;
        }

        // Stop feeding a loop. Requires the key the loop's jobs carried, since
        // session ids are guessable.
        if (h.k === 'halt') {
            const loop = heading.get(h.s);
            if (loop && loop.key === h.key && !loop.ended) loop.halt = true;
            return;
        }

        // Another peer repaired a hand-off for this chat. Handled before the
        // lookup below because the job is still in flight.
        if (h.k === 'fix') {
            if (h.s === conv.session && h.key === conv.key) onFix(h);
            return;
        }

        if (h.k === 'tok' && from != null && !net.mayAnswer(from)) {
            return console.warn(`dropped a tok from peer ${from}, which cannot answer`);
        }

        // An answer in a reply this tab is streaming. Handled synchronously
        // because the next one may follow immediately.
        const stream = streams.get(h.s);
        if (stream && (h.k === 'tok' || h.k === 'err')) {
            if (h.key !== stream.key) {
                return console.warn(`dropped a ${h.k} for ${h.s} with the wrong key`);
            }
            return h.k === 'tok' ? accept(stream, h) : failed(stream, h);
        }

        const job = pending.get(h.s);
        // Drop an answer with the wrong key. The job stays pending.
        if (job && h.key !== job.key) {
            return console.warn(`dropped a ${h.k} for ${h.s} with the wrong key`);
        }
        // Drop a token sampled for an earlier job of this session (one a hop
        // gave up on that ran anyway). Accepting it would shift every later
        // token by one. A `tok` without `past` is accepted.
        if (job && h.k === 'tok' && h.past !== undefined && h.past !== job.past) {
            return console.warn(`dropped a late tok for ${h.s} (past ${h.past}, waiting on ${job.past})`);
        }
        if (job) {
            pending.delete(h.s);
            clearTimeout(job.timer);
            if (h.k === 'err') {
                const err = new Error(`peer ${h.from ?? ''}: ${h.msg}`);
                if (h.busy) err.busy = true;
                job.reject(err);
            }
            else job.resolve(h);
            return;
        }

        if (h.k !== 'err') return;                     // a stale tok, e.g. after New chat

        // An `err` about a hand-off this peer made downstream: a dead next hop
        // or an evicted cache. This peer holds the log for that boundary, so
        // it tries a local repair. `outbound` names the boundary and the job
        // to re-send. The job's payload is the log's tail (one job per session
        // is in flight).
        //
        // Skipped for this tab's own sessions: either the timeout fired and
        // recover() is rebuilding, or the chat is parked and left to idle
        // eviction. Skipped for `busy`: the peer has no room, and a replay
        // would take capacity from another chat. Both are passed back.
        const out = h.busy || own.has(h.s) ? null : outgoing(h.s);
        const sent = out && outbound.get(out);
        // The err must carry the key of the job this peer handed on.
        if (sent && h.key !== sent.key) {
            return console.warn(`dropped an err for ${h.s} with the wrong key`);
        }
        const log = sent && runner()?.logs.get(h.s)?.get(sent.pos);
        // The log must hold the job's own rows too (see replay()).
        if (log && log.len >= sent.past + sent.n) {
            const H = runner().d.hidden;
            const rows = log.rows.subarray(sent.past * H, (sent.past + sent.n) * H);
            if (repair(sent, rows, null, null)) return;
        }

        // Otherwise one more hop back, through route() so the ttl is spent.
        const back = backwards(h.s);
        if (back == null || back === net.id) return;   // stale
        route({ ...h, to: back });
    }

    // How far to run a job that entered at `pos`. This is a proposal:
    // Runner.#exit decides, since a session that ran here before keeps the
    // boundary it started on.
    //
    // Where to hand the session on. Decided on its first job here and pinned
    // (Runner.#exit), so a session keeps its path.
    //
    // On an idle mesh: the furthest position I hold that another peer can
    // pick up, because hop count dominates decode latency. Under load: the
    // exit that evens out my work and the next peer's, among the exits whose
    // next peer reaches as far as the furthest one's, so balancing never adds
    // a hop. Stopping short of my last position also routes around a hole at
    // my boundary.
    function plan(h) {
        const pos = h.pos;
        const r = runner();

        // The model ends inside this peer: run to the end and sample. Nobody
        // holds `positions`, so the loop below never returns it.
        if (r.have >= net.positions) return net.positions;

        // A session that ran here before keeps its cut.
        const cut = r.cuts?.get(h.s)?.get(pos);
        if (cut !== undefined) return cut;

        // Every position I could stop at and who can pick it up there, e.g.
        // `{ 11: [c], 12: [d], 13: [q, f], 15: [] }`. Walk back from the
        // furthest. Only exits with an open link are candidates: the choice
        // is pinned for the session, and an extra hop is better than a
        // failed send.
        const exits = net.exits(pos, r.have);
        const cands = [];              // { at, who }, furthest first
        let reachable = null;
        for (let at = r.have; at > pos; at--) {
            const who = (exits.get(at) ?? []).filter((id) => id !== net.id);
            if (!who.length) continue;
            if (who.some((id) => net.linked(id))) cands.push({ at, who: rankAt(at)[0] });
            reachable ??= at;
        }
        // Nobody reachable can take it. Run as far as I can; route() reports
        // the failure.
        if (!cands.length) return reachable ?? r.have;

        const far = cands[0];
        const end = (c) => c.at + net.reach(c.who, c.at);
        const ok = cands.filter((c) => end(c) >= end(far));
        const me = loadOf(net.id);
        const them = ok.map((c) => loadOf(c.who));
        const take = (c) => {
            planned.set(h.s, { n: c.at - pos, at: performance.now() });
            assign(c.who, net.reach(c.who, c.at));
            return c.at;
        };
        if (me.w === 0 && them.every((l) => l.w === 0)) return take(far);

        // The exit after which the busier of the two carries least. Ties go
        // to the further exit, which `ok` lists first.
        const timed = me.p != null && them.every((l) => l.p != null);
        let best = 0, score = Infinity;
        ok.forEach((c, i) => {
            const x = Math.max(cost(me, c.at - pos, timed),
                               cost(them[i], net.reach(c.who, c.at), timed));
            if (x < score) { best = i; score = x; }
        });
        return take(ok[best]);
    }

    async function runJob(h, hidden, from = null) {
        // At the entry, only the loop the enqueuer opened last may feed the
        // session. A head-fed job of any other loop is dropped. Such a job can
        // arrive when a late repair revives a loop after its reply ended or
        // after a new turn started, and it would write the same rows as the
        // live loop. Every turn, resend and rebuild opens a new loop. Any job
        // from the enqueuer ends the previous loop, including plain prefill
        // chunks, which carry no loop.
        if (h.pos === 0 && !h.replay) {
            if (!h.auto) put(current, h.s, h.loop?.id ?? null);
            else if (current.has(h.s) && current.get(h.s) !== h.loop?.id) {
                return console.warn(`dropped a job of loop ${h.loop?.id} of ${h.s}, which is no longer its loop`);
            }
        }
        try {
            const t0 = performance.now();
            const want = plan(h);
            const t1 = performance.now();
            const r = await runner().run(h, hidden, want);
            const t2 = performance.now();
            ran += h.n;            // token positions, so every peer counts the same work
            onChange();            // a DOM refresh on a real tab, timed as `ui`
            const t3 = performance.now();
            // Trace of where this hop's time went, read at the enqueuer by
            // web/profile.js. `ms` is the whole hop; the phases (stage.js)
            // break it down. `link` is the previous hand-off across the
            // boundary this job leaves at, because this job's ack arrives
            // after its trace has left.
            const round = (x) => Math.round(x * 100) / 100;
            const hop = { id: net.id, ms: round(t3 - t0), blocks: r.at - h.pos };
            if (r.phases) {
                Object.assign(hop, { from: h.pos, to: r.at, plan: round(t1 - t0),
                                     ...r.phases, ui: round(t3 - t2) });
            }
            if (r.gpu) hop.gpu = r.gpu;
            if (h.prof) hop.label = gpu() ?? undefined;
            const out = `${h.s}|${r.at}`;
            if (links.has(out)) hop.link = links.get(out);
            const trace = [...(h.trace ?? []), hop];
            if (r.token !== undefined) {
                // Send the next job first: it is on the critical path.
                const looping = h.loop && closedLoop;
                const fed = looping ? feed(h, r.token) : false;
                route({ k: 'tok', s: h.s, key: h.key, to: h.from, past: h.past, token: r.token,
                        eos: r.token === eos, trace,
                        ...(looping && { fed, head: net.id,
                                         ...(links.has(`${h.s}|0`) && { link: links.get(`${h.s}|0`) }) }) });
            } else if (r.hidden !== undefined) {
                // Leaves at `r.at`, the session's cut. ids exist only at
                // position 0. `replay` is copied by the spread, so a replay
                // rebuilds caches to the end of the line. The peer that ran
                // position 0 stamps `entry` for the head.
                forward({ ...h, pos: r.at, ids: undefined, trace, ...(h.pos === 0 && { entry: net.id }) },
                        r.hidden, null, from);
                // Packing and sending time on this tab, stored with the RTT.
                if (!h.replay) put(links, out, { ...links.get(out), send: round(performance.now() - t3) });
            }
            // Neither: a replay reached the end of the line. Nobody is waiting.
        } catch (err) {
            // A duplicate of a loop job the session is already past. Nobody is
            // waiting on it, so no `err` is sent.
            if (err.stale) return console.warn(`dropped ${err.message}`);
            // Back one hop like fail(), but to the channel the job arrived on:
            // the cursor has not moved, so there is no exit cursor to key by.
            // A refused or evicted cache (Runner.#cacheFor) ends up here.
            route({ k: 'err', s: h.s, key: h.key, to: from ?? h.from, from: net.id,
                    msg: err.message, busy: err.busy || undefined });
        }
    }

    // --- The head's side of the closed loop ---

    // Feed the token this job sampled back in as the session's next job, if
    // the loop should continue. Returns true if it did.
    //
    // The loop stops on end-of-text (which is not fed), when `left` runs out,
    // on a halt, and when the enqueuer has left the mesh.
    //
    // The next job goes to `entry`, the peer holding this session's KV at
    // position 0, through forward(), so it is acked like any hand-off. With no
    // open link to the entry, the head declines. There is no boundary log at
    // position 0 (it takes ids, which only the enqueuer keeps), so a broken
    // hand-off here sends an `err` to the enqueuer, which rebuilds.
    //
    // With penalties on, the next job's summary needs the whole context (the
    // n-gram ban scans all of it). The job that starts a loop carries it once
    // and the head keeps it up to date. A head without it (a repair moved the
    // end of the line to another peer) declines, and the enqueuer starts a new
    // loop.
    //
    // Each position is fed once per loop, and every loop is numbered
    // (`loop.id`). A job of an older loop than the one this head knows, or of
    // the same loop at a position already fed, is a duplicate from a repair
    // re-sending a hand-off. It feeds nothing, whether or not the loop has
    // ended. A newer loop starts fresh, even at a lower position (after a
    // rewind).
    function feed(h, token) {
        const { s, key, loop } = h;
        const at = h.past + h.n;
        const id = loop.id ?? 0;
        const known = heading.get(s)?.key === key ? heading.get(s) : null;
        if (known && (id < known.id || (id === known.id && at <= known.fedTo))) return false;
        const state = known?.id === id ? known : null;
        const end = () => { put(heading, s, { key, id, ctx: null, halt: false, fedTo: at, ended: true }); return false; };
        const owner = h.from === net.id || net.has(h.from);
        if (token === eos || state?.halt || !(loop.left > 0) || !owner) return end();
        // The head must have an open link to the entry. Every peer links to
        // every holder of position 0, but a link can still be opening (a new
        // peer's ICE). A failed hand-off here cannot be repaired, so decline.
        // The enqueuer feeds this token itself and opens a new loop.
        const entry = h.pos === 0 ? net.id : h.entry;
        if (entry == null || (entry !== net.id && !net.linked(entry))) return end();

        const settings = { ...h.opts, ngram: loop.ngram };
        let ctx = null;
        if (penalized(settings)) {
            if (loop.ctx) ctx = [...loop.ctx];
            else if (state && !state.ended && state.ctx?.length === at) ctx = state.ctx;
            else return end();
            ctx.push(token);
        }
        put(heading, s, { key, id, ctx, halt: false, fedTo: at, ended: false });

        const next = {
            k: 'job', s, key, from: h.from, pos: 0, past: h.past + h.n, n: 1, ids: [token],
            opts: jobOpts(settings, ctx ?? [], eos),
            loop: { id, left: loop.left - 1, ...(loop.ngram && { ngram: loop.ngram }) },
            auto: true,
            prof: h.prof,
        };
        if (entry === net.id) {
            runJob(next, null, null);
        } else {
            pin(s, 0, entry);
            forward(next, null);
        }
        return true;
    }

    // --- The enqueuer's session ---

    // The job header's sampler settings for `ids`, from the chat's history.
    const optsFor = (ids, opts, c = conv) => jobOpts(opts, [...c.history, ...ids], eos);

    // What a job asks of the loop: `left` more tokens and, when a penalty
    // needs them, the n-gram setting and the context the head computes its
    // summaries from. Each call opens a loop with the next id.
    function loopFor(ids, opts, left, c = conv) {
        const id = ++loops;
        if (!penalized(opts)) return { id, left };
        return { id, left, ...(opts.ngram && { ngram: opts.ngram }), ctx: [...c.history, ...ids] };
    }

    // One job per session is in flight at a time, so the session id matches
    // the answer to the request.
    //
    // The answer comes back with `rt`, the time from submit to answer on this
    // tab's clock, and `link`, the timing of this tab's own hand-off at
    // position 0. `prof` asks every peer on the path to time the job on its
    // GPU (model/gpuprof.js).
    function submitJob(ids, past, opts, loop = null) {
        const s = conv.session, k = conv.key;
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                pending.delete(s);
                reject(new Error(`no answer from the mesh after ${jobTimeout / 1000}s` +
                    '. A peer probably left.'));
            }, jobTimeout);
            const t0 = performance.now();
            const answer = (h) => resolve({
                ...h,
                rt: Math.round((performance.now() - t0) * 100) / 100,
                ...(links.has(`${s}|0`) && { link: links.get(`${s}|0`) }),
            });
            pending.set(s, { resolve: answer, reject, timer, key: k, past });
            route({
                k: 'job', s, key: k, from: net.id, pos: 0, past, n: ids.length, ids,
                opts: optsFor(ids, opts),
                prof: opts.prof || undefined,
                ...(loop && { loop }),
            });
        });
    }

    // ids are appended once the job comes back, so `history` holds only what
    // the runners have cached.
    //
    // A conversation with a history and no live session (loaded from storage,
    // or `stale`) is rebuilt first, since a job at its `past` would find no
    // cache.
    //
    // A job that fails for good abandons its session (abandon()). Jobs of it
    // may be running, so the session is not used again.
    async function submit(ids, opts) {
        const c = conv;
        if (c.history.length && (c.session == null || c.stale)) {
            return settle(c, rebuild(ids, opts, c.stale ? 'rewind' : 'resume'));
        }
        if (c.session == null) newSession(c);
        let r;
        try {
            r = await submitJob(ids, c.history.length, opts);
        } catch (err) {
            // A capacity refusal is thrown to the caller. The context is still
            // cached and a rebuild would need more capacity. The caller waits
            // and retries.
            if (err.busy) throw err;
            return settle(c, recover(ids, opts, err));
        }
        c.history.push(...ids);
        c.parked = false;
        return r;
    }

    // Await a rebuild. If it fails, abandon what it left behind.
    async function settle(c, rebuilding) {
        const before = [...c.history];
        try {
            return await rebuilding;
        } catch (err) {
            abandon(c, err.history ?? before);
            throw err;
        }
    }

    // Stop using this conversation's session. Its jobs may be in flight, and
    // their answers are dropped. Local only: the caches are left to idle
    // eviction because that traffic may need them. `history` is restored in
    // full (a failed rebuild had truncated it), so the next turn rebuilds all
    // of it under a new session.
    function abandon(c, history) {
        if (c.session != null) forget(c.session);
        c.session = c.key = null;
        c.history.length = 0;
        c.history.push(...history);
    }

    // A peer holding part of this session failed, or the caches of a parked
    // chat were evicted. Rebuild the context and answer the failed job. Each
    // attempt counts against the turn's repair budget, and a rebuild that
    // fails is retried while the budget lasts.
    async function recover(ids, opts, err) {
        const c = conv;
        let reason = c.parked ? 'resume' : 'repair';
        for (;;) {
            if (repairs >= maxRepairs) throw err;
            repairs++;
            try {
                return await rebuild(ids, opts, reason, err);
            } catch (e) {
                if (!retryable(e)) throw e;
                c.history.length = 0;
                c.history.push(...e.history);
                err = e;
                reason = 'repair';
            }
        }
    }

    // A failed rebuild is retried unless it was cancelled or refused for
    // capacity.
    const retryable = (e) => !e.busy && !e.cancelled && Array.isArray(e.history);

    // Re-feed the active conversation's whole history under a new session id.
    //
    // A new id means new KV caches, cuts and hops() over the current roster,
    // so nothing from the previous session is reused. After a failure (`err`)
    // the old caches are left to idle eviction, since repair traffic may need
    // them. Without a failure the old session is retired.
    //
    // `ids` go on the end of the history, so the last chunk samples the token
    // the failed job owed. This costs a full prefill, so onRepair and
    // onProgress report it.
    //
    // `finish`, from a stream, sends the last chunk itself as the job that
    // reopens the loop. It resolves once the stream has taken its answer and
    // added the chunk to the history.
    async function rebuild(ids, opts, reason, err = null, finish = null) {
        const c = conv;
        const ctx = [...c.history, ...ids];
        const done = onRepair(err, ctx.length, reason);

        // Retire the old id before the first new job goes out, so route()
        // cannot send a rebuild job to a pin of the old session.
        if (c.session != null) (err ? forget : retire)(c.session);
        newSession(c);
        c.parked = c.stale = false;
        // History is refilled chunk by chunk, so after a partial failure it
        // matches what the new session's runners hold.
        c.history.length = 0;

        let r = null;
        try {
            for (let i = 0; i < ctx.length; i += T_MAX) {
                // Checked between chunks only.
                if (isCancelled()) throw Object.assign(new Error('rebuild cancelled'), { cancelled: true });
                const chunk = ctx.slice(i, i + T_MAX);
                if (finish && i + T_MAX >= ctx.length) {
                    r = await finish(chunk);
                } else {
                    r = await submitJob(chunk, c.history.length, opts);
                    c.history.push(...chunk);
                }
                onProgress(c.history.length, ctx.length);
                onChange();
            }
        } catch (e) {
            // The chat's history before this turn, for abandon() to restore.
            e.history = ctx.slice(0, ctx.length - ids.length);
            throw e;
        }
        done?.();
        return r;
    }

    // --- Streaming a reply: the enqueuer's side of the closed loop ---
    //
    // decode() sends a prompt's last chunk with `loop` set and returns the
    // answers as an async iterator, one per token. Answers are handled
    // synchronously in deliver(). `history` gains ids as the peers confirm
    // caching them: the prompt with the first answer, and each fed token with
    // the answer after it.
    //
    // An answer is `{ token, eos, fed, closed, trace, rt, link }`. The fed
    // answers are the reply. The last answer ends the stream and is not fed
    // (end-of-text, the budget or a halt ended it); its token is not part of
    // the reply.
    //
    // On an `err` or the job timeout, the stream rebuilds the context under a
    // new session and reopens the loop with the remaining budget. The old loop
    // is sent a halt in case it is running. A head that does not close the
    // loop answers with `fed` unset, and the stream feeds that token itself.

    function decode(ids, opts, { budget }) {
        const st = {
            c: conv, opts, budget,
            s: null, key: null,
            sent: 0,              // generated tokens handed to the mesh to feed
            feeding: null,        // the ids of the job whose answer is awaited
            expect: 0,            // and its `past`
            early: new Map(),     // answers that overtook it, by `past`
            head: null,           // who is closing the loop, from its answers
            halted: false, done: false, reason: null, error: null, thrown: false,
            queue: [], waiters: [], timer: null,
            at: performance.now(),
            first: null,          // rebuild()'s wait on the chunk that reopens the loop
        };
        begin(st, ids);
        return streamOf(st);
    }

    // A chat with a history and no session is rebuilt first, as in submit().
    // Otherwise the job is sent directly.
    function begin(st, ids) {
        const c = st.c;
        if (c.history.length && (c.session == null || c.stale)) {
            return restart(st, ids, c.stale ? 'rewind' : 'resume', null);
        }
        if (c.session == null) newSession(c);
        send(st, ids);
    }

    // A failed rebuild is retried while the turn's repair budget lasts.
    // settle() has dropped the partial session and restored the history.
    function restart(st, ids, reason, err) {
        const reopen = (chunk) => new Promise((resolve, reject) => {
            st.first = { resolve, reject };
            send(st, chunk);
        });
        settle(st.c, rebuild(ids, st.opts, reason, err, reopen)).catch((e) => {
            if (st.done) return;
            if (!retryable(e) || repairs >= maxRepairs) return finish(st, null, e);
            repairs++;
            restart(st, ids, 'repair', e);
        });
    }

    // One job of the stream: the prompt's chunk, or a token the head did not
    // feed. `left` is how many tokens the head may feed after it.
    function send(st, ids) {
        const c = st.c;
        st.s = c.session;
        st.key = c.key;
        st.feeding = ids;
        st.expect = c.history.length;
        streams.set(st.s, st);
        arm(st);
        route({
            k: 'job', s: st.s, key: st.key, from: net.id, pos: 0, past: st.expect,
            n: ids.length, ids, opts: optsFor(ids, st.opts, c),
            prof: st.opts.prof || undefined,
            loop: loopFor(ids, st.opts, st.budget - st.sent, c),
        });
    }

    // Per-answer timeout. Acks and errors detect most failures first.
    function arm(st) {
        clearTimeout(st.timer);
        st.timer = setTimeout(() => failed(st, new Error(
            `no answer from the mesh after ${jobTimeout / 1000}s. A peer probably left.`)), jobTimeout);
    }

    function accept(st, h) {
        if (st.done || h.s !== st.s) return;
        if (h.past !== st.expect) {
            // Behind: an answer to a job this stream gave up on. Ahead: an
            // answer that arrived early, held until the awaited one arrives.
            if (h.past > st.expect && st.early.size < EARLY) st.early.set(h.past, h);
            return;
        }
        clearTimeout(st.timer);
        const c = st.c;
        c.history.push(...st.feeding);
        c.parked = false;

        const now = performance.now();
        // `closed`: the head fed the token, so this tab did not have to.
        const answer = {
            token: h.token, eos: !!h.eos, fed: !!h.fed, closed: !!h.fed, trace: h.trace ?? [],
            rt: Math.round((now - st.at) * 100) / 100,
            ...(h.link && { link: h.link }),
        };
        st.at = now;
        if (h.head != null) st.head = h.head;

        // Not fed and the reply is not over: the head does not close the loop
        // or declined, so this tab feeds the token and opens a new loop.
        const resend = !h.fed && !h.eos && !st.halted && st.sent < st.budget;
        if (h.fed) {
            // The head has sent it as the next job. The next answer confirms it.
            st.sent++;
            st.feeding = [h.token];
            st.expect = c.history.length;
            arm(st);
            // A halt requested before the head was known.
            if (st.halted) sendHalt(st);
        } else if (resend) {
            st.sent++;
            answer.fed = true;
        }
        st.queue.push(answer);
        if (st.first) { st.first.resolve(answer); st.first = null; }
        if (!h.fed && !resend) return finish(st, h.eos ? 'eos' : st.halted ? 'stopped' : 'budget');
        wake(st);

        // Must be the last statement. A send can fail synchronously, with the
        // error coming back through deliver() and restarting the stream under
        // a new session.
        if (resend) return send(st, [h.token]);
        const next = st.early.get(st.expect);
        if (next) { st.early.delete(st.expect); accept(st, next); }
    }

    function failed(st, h) {
        if (st.done) return;
        clearTimeout(st.timer);
        const err = h instanceof Error ? h : new Error(`peer ${h.from ?? ''}: ${h.msg}`);
        if (h.busy) err.busy = true;
        if (streams.get(st.s) === st) streams.delete(st.s);
        st.early.clear();

        // The chunk reopening the loop failed. rebuild() handles that.
        if (st.first) { st.first.reject(err); st.first = null; return; }
        // A capacity refusal is thrown to the caller, as in submit(), and the
        // session is kept. The job was refused, so no loop is running and the
        // context is still cached.
        if (err.busy) return finish(st, null, err);

        // Stop everything running under this session. The halt reaches the
        // head this tab knows of. A repair in progress can hand the loop to a
        // new head that never receives the halt, so the session is also freed
        // along its pins. That includes the entry, which every loop job passes
        // through and which then has no cache for it.
        // submit() only forgets a failed session, since its repair traffic may
        // need the caches. A stream that has given up on a session has no use
        // for that traffic.
        sendHalt(st);
        st.head = null;
        if (repairs >= maxRepairs) {
            // The session is freed, so the next turn rebuilds.
            if (st.s != null) retire(st.s);
            abandon(st.c, [...st.c.history]);
            return finish(st, null, err);
        }
        if (st.s != null) retire(st.s);
        repairs++;
        restart(st, st.feeding, st.c.parked ? 'resume' : 'repair', err);
    }

    function sendHalt(st) {
        if (st.head == null || st.s == null) return;
        if (st.head === net.id) {
            const loop = heading.get(st.s);
            if (loop?.key === st.key) loop.halt = true;
        } else {
            route({ k: 'halt', s: st.s, key: st.key, to: st.head });
        }
    }

    function finish(st, reason, error = null) {
        if (st.done) return;
        st.done = true;
        st.reason = reason;
        st.error = error;
        clearTimeout(st.timer);
        if (streams.get(st.s) === st) streams.delete(st.s);
        if (st.first) { st.first.reject(error ?? new Error('the reply ended')); st.first = null; }
        wake(st);
    }

    function wake(st) {
        const waiters = st.waiters;
        st.waiters = [];
        for (const w of waiters) w();
    }
    const woken = (st) => new Promise((res) => st.waiters.push(res));

    // The consumer's handle on a stream.
    function streamOf(st) {
        return {
            [Symbol.asyncIterator]() { return this; },
            async next() {
                for (;;) {
                    if (st.queue.length) return { value: st.queue.shift(), done: false };
                    if (st.error && !st.thrown) { st.thrown = true; throw st.error; }
                    if (st.done) return { value: undefined, done: true };
                    await woken(st);
                }
            },
            // A consumer that stops pulling (a `break`) halts the loop and
            // waits for every fed token, so the history matches the peers.
            async return() {
                this.halt();
                await this.drain();
                return { value: undefined, done: true };
            },
            // Ask the loop to stop. Tokens already fed arrive as part of the
            // reply.
            halt() {
                if (st.done || st.halted) return;
                st.halted = true;
                sendHalt(st);
            },
            // Resolves when the stream has ended, with or without an error.
            // `history` then holds everything the peers do.
            async drain() {
                while (!st.done) await woken(st);
            },
            get done() { return st.done; },
            get reason() { return st.reason; },
            get sent() { return st.sent; },
            // The error the stream ended on, if next() has not thrown it. That
            // happens when the consumer stopped pulling before the failure. A
            // caller that drains must check it.
            get error() { return st.thrown ? null : st.error; },
        };
    }

    // Whether a reply is being generated in this conversation.
    const busy = (c) => (c.session != null && pending.has(c.session))
        || [...streams.values()].some((st) => st.c === c && !st.done);

    // Make `c` the active conversation. Nothing is freed or forgotten. The
    // previous one keeps its pins, so switching back reaches the same caches
    // if they have not been evicted.
    function switchTo(c) {
        if (c?.mesh !== mesh) {
            throw new Error(`that chat belongs to another mesh (${c?.mesh ?? 'none'}), ` +
                `not this one (${mesh})`);
        }
        if (c === conv) return;
        if (busy(conv)) {
            throw new Error('a reply is still being generated. Stop it before switching chats.');
        }
        c.parked = c.session != null;
        if (c.session != null) own.set(c.session, c);
        conv = c;
    }

    return {
        route,
        submit,
        decode,
        switchTo,

        // Make a new conversation in this mesh and activate it. The previous
        // one is parked, not freed.
        newChat() {
            const c = makeConversation({ mesh });
            switchTo(c);
            return c;
        },

        // The clean end of a chat, and the only place a session is freed on
        // purpose. Nothing for it is in flight. Discarding the active
        // conversation starts a new one.
        discard(c = conv) {
            if (c === conv && busy(c)) {
                throw new Error('a reply is still being generated. Stop it first.');
            }
            if (c.session != null) retire(c.session);
            c.session = c.key = null;
            if (c === conv) conv = makeConversation({ mesh });
        },

        // Text cut from the display must leave the context too, or the next
        // turn is conditioned on text the user cannot see. Every job carries
        // T_past, so for KV a rewind drops the tail and later jobs overwrite
        // the stale rows. Recurrent (linear_attention) state has already
        // folded the cut tokens in and cannot be trimmed. The peers would
        // refuse to continue it (stage.js's #checkFed), so on a hybrid
        // checkpoint the next job rebuilds.
        rewind(n) {
            if (n <= 0) return;
            conv.history.length = Math.max(0, conv.history.length - n);
            if (hybrid) conv.stale = true;
        },

        resetRepairs() { repairs = 0; },   // per turn, not per session

        // Live maps, for tests and diagnostics.
        plan, hops, routes, upstream, outbound, fixes, acks, pending, links, heading, streams,

        get session() { return conv.session; },
        get conversation() { return conv; },
        get mesh() { return mesh; },
        get ctxLen() { return conv.history.length; },
        get history() { return conv.history; },
        get ran() { return ran; },
        // The last load each peer reported in its acks (see plan()).
        get loads() { return new Map(loads); },
        get repairs() { return repairs; },
        get repaired() { return repaired; },   // hand-offs fixed for other peers
        get eos() { return eos; },
        set eos(v) { eos = v; },
        get hybrid() { return hybrid; },
        set hybrid(v) { hybrid = !!v; },
    };
}
