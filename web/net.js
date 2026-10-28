// The mesh: the signaling socket, the links between peers, and message
// framing. route.js decides which holder of a position a message goes to.
//
// Every peer computes the same link set from the roster: a peer links to the
// holders of every position it could hand a job over at, and to every holder
// of position 0 (the hub). The lower id offers. Delivery is ordered and
// reliable, which the KV caches depend on.
//
// A Mesh belongs to one `room` (web/room.js): the global mesh or a private
// one. By default every link goes through a TURN relay, so peers do not see
// each other's addresses. `relayOnly: false` (peer.html's `?direct=1`) allows
// direct links.
//
// See docs/architecture.md (Joining, Private meshes, Links, Wire format).

import { holders, holding, covers, minDepth, served, coverage, PROTOCOL } from './split.js';
import { toF16 } from './f16.js';
import { GLOBAL } from './room.js';

// Fallback ICE list. The signaling server sends the real list, TURN included,
// in the welcome. GATHER_MS bounds gathering when a STUN or TURN server never
// answers. It must be longer than a TURN allocation to a distant relay takes:
// nothing is trickled, so a candidate that arrives later is never used.
const ICE = [{ urls: 'stun:stun.l.google.com:19302' }];
const GATHER_MS = 5000;

// How long one socket may wait for the signaling server's welcome before it is
// closed and another opened. The server answers within a round trip plus at
// most one TURN-credential mint, so this allows for a slow mobile network. A
// socket can stay in CONNECTING indefinitely with no error or close event.
export const READY_TIMEOUT_MS = 10000;

// The server pings every 30s. After two missed pings the socket is treated as
// dead. Through a tunnel a socket can die without a close event.
const PING_TIMEOUT_MS = 75000;

// Reconnect backoff. Datachannels are peer to peer and keep running while the
// signaling socket is down, but no new peer can be introduced, so the socket
// is retried indefinitely.
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 15000;

// A link whose ICE fails, or that has not opened LINK_OPEN_MS after it was
// made (an offer or answer lost while the socket was down), is offered again
// after a per-peer backoff: [first, max], doubling, reset when the link opens.
// The retry does not depend on a roster change. The backoff limits offers to a
// peer that can never be reached (no TURN, a blocked port).
export const LINK_RETRY_MS = [2000, 30000];
export const LINK_OPEN_MS = 30000;

// How long to wait for a range after stating a capacity. The server answers
// from memory, so this fires only if the socket died.
const ASSIGN_TIMEOUT_MS = 10000;

// Whether this ICE server list includes a TURN relay.
export const hasTurn = (ice) => (ice ?? []).some((s) =>
    [].concat(s?.urls ?? []).some((u) => /^turns?:/.test(u)));

// Returns the SDP with every non-relay ICE candidate removed. Host,
// server-reflexive and peer-reflexive candidates are this machine's own
// addresses. All other lines are kept.
export function relayOnlySdp(sdp) {
    return sdp.split(/\r\n/).filter((line) => !/^a=candidate:/.test(line)
        || / typ relay( |$)/.test(line)).join('\r\n');
}

// Close codes from signal-server.js that a reconnect cannot fix: a join the
// server did not accept (this page is older than the server), and a full mesh.
const REFUSED = { 4001: 'the signaling server did not accept this page. Reload it.',
                  4002: 'this cluster is full' };

// --- Framing ---

// A message is a JSON header plus an optional f16 payload (the hidden state):
//
//   [0..4)            u32  header byte length
//   [4..4+n)          UTF-8 JSON header
//   [.. pad to 4)     zeros, so the payload view is aligned
//   [..]              f16 payload
//
// The header is JSON so fields can be added. It is about 200 bytes next to a
// 10 KB payload (one Qwen3.8-27B row). The payload is half bits (f16.js): a
// Runner rounds its rows once as they leave it, so they arrive as a
// Uint16Array and pass through unchanged. A Float32Array is rounded here.
export function pack(header, payload = null) {
    const json = new TextEncoder().encode(JSON.stringify(header));
    const start = 4 + json.length + ((4 - (json.length % 4)) % 4);
    const bits = payload && (payload instanceof Uint16Array ? payload : toF16(payload));
    const buf = new ArrayBuffer(start + (bits?.byteLength ?? 0));
    new DataView(buf).setUint32(0, json.length, true);
    new Uint8Array(buf, 4, json.length).set(json);
    if (bits) new Uint16Array(buf, start).set(bits);
    return buf;
}

// The payload comes back as half bits, a Uint16Array.
export function unpack(buf) {
    const n = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    const start = 4 + n + ((4 - (n % 4)) % 4);
    const payload = buf.byteLength > start ? new Uint16Array(buf, start) : null;
    return { header, payload };
}

// --- Fragmentation ---

// SCTP has a per-message size limit (Chrome advertises
// `a=max-message-size:262144`), and `ch.send()` throws above it. A prefill
// chunk of T_MAX rows is larger, so a message too big for one send is split
// into several, each with a 16-byte prefix:
//
//   [0..4)     u32  FRAG_MARK       0xFFFFFFFF
//   [4..8)     u32  message id      per link, wraps at 2^32
//   [8..12)    u32  fragment index
//   [12..16)   u32  fragment count
//   [16..)          the slice
//
// An unfragmented message starts with a JSON header length of a few hundred,
// so it cannot be confused with the marker. A message that fits (every decode
// step) is sent as-is, with no prefix.
//
// The channel is ordered and reliable, and `send()` pushes every fragment
// before it returns, so fragments arrive in order and do not interleave. The
// id and index are checked anyway, so a bug here throws instead of corrupting
// the residual stream.
export const FRAG_MARK = 0xFFFFFFFF;
export const FRAG_HEAD = 16;

// Size limit for one `ch.send()` when the connection does not report one.
// This is Chrome's advertised value. It also caps the value read from `sctp`:
// too high throws, too low costs one more fragment.
export const WIRE_DEFAULT = 262144;

export function wireLimit(pc) {
    const n = pc?.sctp?.maxMessageSize;
    return Number.isFinite(n) && n > FRAG_HEAD ? Math.min(n, WIRE_DEFAULT) : WIRE_DEFAULT;
}

// Split a packed message into wire-sized pieces. A message that fits is
// returned as-is in a one-element array, with no allocation or copy.
export function fragment(buf, limit = WIRE_DEFAULT, id = 0) {
    if (buf.byteLength <= limit) return [buf];
    // Slice on 4-byte boundaries, so every slice holds whole f32s.
    const body = Math.max(4, (limit - FRAG_HEAD) & ~3);
    const count = Math.ceil(buf.byteLength / body);
    const src = new Uint8Array(buf);
    const out = [];
    for (let i = 0; i < count; i++) {
        const slice = src.subarray(i * body, Math.min((i + 1) * body, src.length));
        const frame = new ArrayBuffer(FRAG_HEAD + slice.length);
        const dv = new DataView(frame);
        dv.setUint32(0, FRAG_MARK, true);
        dv.setUint32(4, id, true);
        dv.setUint32(8, i, true);
        dv.setUint32(12, count, true);
        new Uint8Array(frame, FRAG_HEAD).set(slice);
        out.push(frame);
    }
    return out;
}

// The largest message reassemble() will build. The largest valid one is a
// T_MAX prefill chunk of T_MAX * hidden * 4 bytes, about 1.3 MB on the 27B
// model. The limit stops a peer claiming a huge fragment count to make this
// tab buffer everything that follows.
export const MAX_MESSAGE = 32 * 1024 * 1024;

// The receiving half. `state` is one plain object per link, owned by #attach.
// Returns the whole message, or null while fragments are still arriving. An
// unfragmented buffer is returned unchanged.
export function reassemble(state, buf) {
    if (buf.byteLength < FRAG_HEAD) return buf;
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== FRAG_MARK) return buf;

    const id = dv.getUint32(4, true);
    const i = dv.getUint32(8, true);
    const count = dv.getUint32(12, true);

    if (i === 0) {
        state.id = id;
        state.count = count;
        state.parts = [];
        state.bytes = 0;
    } else if (state.parts == null || state.id !== id || state.parts.length !== i) {
        // Cannot happen over an ordered channel. Throwing loses one message
        // and avoids passing on a corrupt hidden state.
        state.parts = null;
        throw new Error(`fragment ${i} of message ${id} arrived out of order`);
    }
    state.bytes += buf.byteLength - FRAG_HEAD;
    if (state.bytes > MAX_MESSAGE) {
        state.parts = null;
        throw new Error(`message ${id} is over ${MAX_MESSAGE / 1048576} MiB`);
    }
    state.parts.push(new Uint8Array(buf, FRAG_HEAD));
    if (state.parts.length < state.count) return null;

    const out = new Uint8Array(state.parts.reduce((n, s) => n + s.length, 0));
    let at = 0;
    for (const s of state.parts) { out.set(s, at); at += s.length; }
    state.parts = null;
    return out.buffer;
}

// How much a link's send buffer may hold before send() refuses a job. Chrome
// closes a datachannel when its buffer overruns (16 MiB), which would break
// every session sharing the link. A peer that has stopped draining will not
// return its jobs anyway.
export const SEND_HIGH_WATER = 8 * 1024 * 1024;

// SEND_HIGH_WATER applies to messages that carry a residual stream. Header-only
// messages (an ack, a token, an error) are tens of bytes and tell the far end
// that a job arrived, what it produced, or that it was refused. They must get
// through a backed-up link, so anything under SEND_SMALL is sent up to
// SEND_CEILING, which is below Chrome's limit.
export const SEND_SMALL = 4096;
export const SEND_CEILING = 12 * 1024 * 1024;

// --- Rate limiting ---

// What one link may deliver to this tab, as [per second, burst]. Anything over
// is dropped on arrival, before it is reassembled, parsed or routed, so a
// flood costs one counter update per frame.
//
// Sized for a link at its busiest: the hub's link to a peer carries every
// session routed through it. `bytes`: the burst fits many 256-row replays of
// the 27B model (about 5 MB each). `msgs`: decode traffic is a job, an ack and
// an answer per token, and a small model decodes at hundreds of tokens a
// second per chat (the tiny test model: ~280). The limit is about ten times
// that. Routing a header takes microseconds. The GPU is protected separately
// by Runner's MAX_QUEUE and MAX_SESSION_QUEUE.
//
// A dropped message is handled like any lost message: its ack never arrives
// and the sender repairs around this tab.
export const RATE = {
    bytes: [64 * 1024 * 1024, 128 * 1024 * 1024],
    msgs: [5000, 10000],
};

// A token bucket: `rate` per second, holding at most `burst`. Starts full.
export function bucket(rate, burst, now = performance.now()) {
    let level = burst, at = now;
    return (n = 1, t = performance.now()) => {
        level = Math.min(burst, level + ((t - at) / 1000) * rate);
        at = t;
        if (level < n) return false;
        level -= n;
        return true;
    };
}

// --- Mesh ---

const iceDone = (pc) => new Promise((res) => {
    if (pc.iceGatheringState === 'complete') return res();
    pc.addEventListener('icecandidate', (e) => { if (!e.candidate) res(); });
    setTimeout(res, GATHER_MS);
});

// How long a held announcement waits for its links before it is sent anyway
// (see Mesh's #flush()). ICE takes about a second, longer through TURN.
export const LINK_WAIT_MS = 5000;

// How long seat() waits for the signaling server to answer.
const SEAT_WAIT_MS = 5000;

export class Mesh {
    // onMessage(header, payload, raw, from). `raw` is the whole reassembled
    // message as bytes, so a forwarder can pass it to send(), which fragments
    // it again for the next link's limit.
    //
    // `from` is the peer on the other end of the channel the message arrived
    // on. It comes from the connection, not the header, so the reverse path is
    // authenticated. route.js uses it to send a failure back one hop.
    //
    // `room` is the mesh to join (web/room.js), the global one by default.
    // `chats` is the limit on replies at once this tab asks for when its join
    // creates a private mesh (the server ignores it otherwise).
    // `sessions` is how many chats each peer keeps in memory, and `ctx` the
    // context window in tokens, asked for the same way.
    // `relayOnly` routes every link through TURN, so no peer learns this tab's
    // address. `rate`, `linkWait`, `readyTimeout`, `linkRetry` and `linkOpen`
    // override RATE, LINK_WAIT_MS, READY_TIMEOUT_MS, LINK_RETRY_MS and
    // LINK_OPEN_MS, for tests.
    constructor({ onMessage, onChange, rate = RATE, room = GLOBAL, chats = null, sessions = null,
                  ctx = null, relayOnly = true,
                  linkWait = LINK_WAIT_MS, readyTimeout = READY_TIMEOUT_MS,
                  linkRetry = LINK_RETRY_MS, linkOpen = LINK_OPEN_MS } = {}) {
        this.id = null;
        this.blocks = 0;            // L, the model's depth, from the welcome
        this.positions = 0;         // L + 2: what peers are placed on
        this.weightsUrl = null;     // the checkpoint the mesh runs, from the welcome
        this.minBlocks = 0;         // the smallest range the server will hand out
        this.range = null;          // [lo, hi) once assigned, null for a client
        this.peers = [];
        this.onMessage = onMessage;
        this.onChange = onChange;
        this.rate = rate;
        this.ice = ICE;             // until the welcome brings the real list
        this.links = new Map();     // id -> { pc, ch, offerer }
        this.room = room;
        this.chats = chats;
        // Replies that may generate at once in this mesh, from the welcome.
        // Null for no limit.
        this.maxChats = null;
        this.askSessions = sessions;
        // Chats each peer keeps in memory in this mesh, from the welcome. Null
        // if the server sends none.
        this.sessions = null;
        this.askCtx = ctx;
        // The context window of this mesh's chats, from the welcome. Null if
        // the server sends none (split.js's runnerCtx() then picks the default).
        this.ctx = null;
        this.relayOnly = relayOnly;
        this.linkWait = linkWait;
        this.readyTimeout = readyTimeout;
        this.linkRetry = linkRetry;
        this.linkOpen = linkOpen;
        this.authFailures = 0;      // offers/answers that failed the room's seal
        this.refused = null;        // why the server refused this tab, if it did
        // Why the signaling server has not welcomed this tab yet, for the page
        // to show while retrying. Null once welcomed.
        this.trouble = null;

        // Resolves on the first welcome, from whichever socket brings it. A
        // socket that errors, closes or hangs before its welcome is retried
        // with backoff (#dropped). Rejects only when the server refuses this
        // tab (#stopFor).
        this.ready = new Promise((resolve, reject) => {
            this.#welcomed = resolve;
            this.#refuse = (why) => reject(new Error(why));
        });
        this.ready.catch(() => {});     // avoid an unhandled rejection if nobody awaits it

        this.online = false;        // whether the signaling socket is up
        this.#connect();
    }

    #welcomed;
    #refuse;
    #hang = null;               // this socket's wait for its welcome
    #retries = new Map();       // peer id -> { delay, timer }: re-offering a failed link
    #assigned = null;
    #watchdog = null;
    #backoff = RECONNECT_MIN_MS;
    #closed = false;            // stop() was called; do not reconnect
    #rejoining = false;         // this socket is a reconnect, not a first join

    // Open the signaling socket. Called again after every drop.
    //
    // The roster is kept across a drop. The datachannels are peer to peer and
    // keep working, so a peer mid-generation continues while this reconnects.
    // While the socket is down no new peers are learned; `online` reports it.
    //
    // Each socket is dropped at most once, and only while it is the current
    // one. The watchdog closing a socket and that socket's later close event
    // must not open two replacements: the extra one would join the mesh as a
    // peer nothing controls.
    #connect() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${proto}://${location.host}/ws`);
        this.ws = ws;
        let gone = false;
        const drop = (why) => {
            if (gone || ws !== this.ws) return;
            gone = true;
            try { ws.close(); } catch { /* already gone */ }
            this.#dropped(why);
        };
        this.#drop = drop;

        // A socket must be welcomed within READY_TIMEOUT_MS.
        clearTimeout(this.#hang);
        this.#hang = setTimeout(() => drop(`no response from the signaling server after ` +
            `${this.readyTimeout / 1000}s`), this.readyTimeout);

        ws.addEventListener('open', () => {
            if (gone || ws !== this.ws) return;
            this.#kick();
            // `join` is the first message on every socket: it names the mesh.
            this.#ws({ type: 'join', v: PROTOCOL, room: this.room.joinToken,
                       ...(this.chats && { chats: this.chats }),
                       ...(this.askSessions && { sessions: this.askSessions }),
                       ...(this.askCtx && { ctx: this.askCtx }) });
        });
        ws.addEventListener('message', (e) => {
            if (gone || ws !== this.ws) return;
            this.#kick();
            const m = JSON.parse(e.data);
            // The pong satisfies the server's liveness check, and traffic in
            // either direction stops a proxy closing the connection as idle.
            if (m.type === 'ping') { this.#ws({ type: 'pong' }); return; }
            this.#signal(m);
        });
        ws.addEventListener('close', (e) => {
            if (gone || ws !== this.ws) return;
            if (REFUSED[e.code]) this.#stopFor(REFUSED[e.code]);
            drop(`the connection to the signaling server closed (code ${e.code})`);
        });
        ws.addEventListener('error', () => drop('could not open a connection to the signaling server'));
    }

    #drop = null;               // the current socket's drop(), for the watchdog

    // Restart the silence timer. A tunnel can drop a socket without a close
    // event, so missing pings are the only signal.
    #kick() {
        this.online = true;
        clearTimeout(this.#watchdog);
        this.#watchdog = setTimeout(() => this.#drop?.('the signaling server went silent'),
            PING_TIMEOUT_MS);
    }

    #dropped(why) {
        clearTimeout(this.#watchdog);
        clearTimeout(this.#hang);
        this.online = false;
        if (!this.#closed) {
            this.#rejoining = true;
            this.trouble = `${why}, retrying in ${Math.round(this.#backoff / 1000)}s`;
            console.warn(`signaling: ${this.trouble}`);
            setTimeout(() => this.#connect(), this.#backoff);
            this.#backoff = Math.min(this.#backoff * 2, RECONNECT_MAX_MS);
        }
        this.onChange?.();
    }

    // Stop reconnecting. The page is going away.
    stop() {
        this.#closed = true;
        clearTimeout(this.#watchdog);
        clearTimeout(this.#hang);
        clearTimeout(this.#held?.timer);
        for (const r of this.#retries.values()) clearTimeout(r.timer);
        this.#retries.clear();
        try { this.ws.close(); } catch { /* gone */ }
    }

    // Stop permanently and record why: the server refused this tab, or put it
    // in a mesh it did not ask for. Every link is closed too, since in the
    // second case they may be to peers of the wrong mesh.
    #stopFor(why) {
        if (this.refused) return;
        this.refused = why;
        console.warn(`signaling: ${why}`);
        this.#refuse?.(why);
        this.stop();
        for (const link of this.links.values()) { clearTimeout(link.openTimer); link.pc.close(); }
        this.links.clear();
        this.peers = [];
        this.onChange?.();
    }

    // My own roster entry, as the server and every other peer see me. `range`
    // holds the same range and arrives first, in the `assign` reply.
    get me() { return this.peers.find((p) => p.id === this.id); }

    // Every ready peer holding position `x`, lowest id first: the candidates
    // for a cursor at `x`. Unlike `wants`, this excludes peers that are not
    // ready, so a client never appears here.
    next(x) {
        return holders(this.readyPeers, x);
    }

    // Every position a peer holding `[lo, hi)` could hand a job over at, and
    // who is ready to take it there, e.g. `{ 11: [c], 12: [d], 13: [q, f],
    // 15: [] }`.
    //
    // An empty entry is a position nobody holds, where this peer must not
    // stop. plan() uses the map to walk back to the last position with a
    // taker.
    exits(lo, hi) {
        const out = new Map();
        for (let x = lo + 1; x <= Math.min(hi, this.positions); x++) {
            out.set(x, holders(this.readyPeers, x));
        }
        return out;
    }

    // How many positions peer `id` runs starting from `pos`. Used to choose
    // between two peers that can both take a job, since hop count dominates
    // decode latency.
    reach(id, pos) {
        const p = this.peers.find((q) => q.id === id);
        return holding(p) ? (p.have ?? p.hi) - pos : 0;
    }

    // Where a job of mine enters the network, and where an answer for me is
    // handed over. Ready holders only.
    get entry() { return this.next(0); }

    // The entry point as a relay: the first holder of position 0 I have an
    // open channel to, ready or not. Forwarding an answer needs no GPU, and
    // every peer links to it, so it reaches any addressee in one hop.
    get hub() { return holders(this.peers, 0).find((id) => this.#live(id)) ?? null; }

    // Whether the ready peers' ranges cover every position.
    get staffed() { return this.positions > 0 && covers(this.readyPeers, this.positions); }

    // The number of ready peers on the least-covered position. 0: not
    // runnable. 1: any departure there breaks the mesh. 2: any single peer can
    // leave. For display, not routing.
    get redundancy() {
        return this.positions > 0 ? minDepth(this.readyPeers, this.positions) : 0;
    }

    // Every peer that can run a job, with its range as the holding
    // `[lo, have)`, not the claim. `ready` means the peer's self-checks and
    // warm-up passed. `have` says which positions have loaded. All roster
    // queries above go through this.
    get readyPeers() { return this.peers.filter((p) => p.ready).map(served); }

    get coverage() { return coverage(this.readyPeers, this.positions); }

    // Whether an answer to a prompt of mine can reach me. For most peers that
    // means an open link to the entry point. For a peer that holds position 0,
    // the answer arrives from a holder of the last position, which may be me.
    // False for the second or so a new peer's ICE takes.
    get reachable() {
        const hub = holders(this.peers, 0);
        if (!hub.length) return false;
        if (!hub.includes(this.id)) return hub.some((id) => this.#live(id));
        const last = this.next(this.positions - 1);
        return last.includes(this.id) || last.some((id) => this.#live(id));
    }

    // Whether this peer can send a prompt: the model is covered, a job of mine
    // can enter, an answer can come back, and, if I am a ready holder short of
    // the end of the model, my own hand-off is open. A peer sees only its own
    // links. Another peer's broken link shows up as a failed route.
    get connected() {
        if (!this.staffed || !this.reachable) return false;

        const entry = this.entry;
        if (!entry.includes(this.id) && !entry.some((id) => this.#live(id))) return false;

        // My own hand-off: a job leaves this peer at whichever position in
        // `(lo, have]` plan() picks, so one reachable taker at any of them is
        // enough.
        const lo = this.range?.[0], have = this.me?.have ?? this.range?.[1];
        if (this.me?.ready && lo != null && have < this.positions) {
            for (const ids of this.exits(lo, have).values()) {
                if (ids.some((id) => id !== this.id && this.#live(id))) return true;
            }
            return false;
        }
        return true;
    }

    has(id) { return this.peers.some((p) => p.id === id); }

    // Whether an answer (`tok`) may arrive over the link to `id`: only from a
    // holder of the last position, which sampled it, or from a hub relaying
    // it. Uses the roster's claims, not `ready` peers: a claim only grows, so
    // a peer that just sampled a token passes.
    mayAnswer(id) {
        return holders(this.peers, this.positions - 1).includes(id)
            || holders(this.peers, 0).includes(id);
    }

    // Whether a message can be sent to this peer now: the channel is open, not
    // only an RTCPeerConnection gathering ICE.
    linked(id) { return this.#live(id); }

    // State how many positions this tab can hold. Resolves to the range the
    // server assigns, or null if it assigns none (the capacity is below the
    // minimum). `claim(0)` gives a range back after a failed load.
    //
    // `at` requests an exact range: used on reconnect to keep weights already
    // in VRAM, and by `?at=` in peer.js. The server falls back to placement if
    // the range is invalid, so the caller must validate it.
    claim(cap, at = null) {
        this.#ws(at ? { type: 'capacity', cap, lo: at[0], hi: at[1] }
                    : { type: 'capacity', cap });
        return new Promise((resolve, reject) => {
            const timer = setTimeout(
                () => reject(new Error(`the signaling server did not answer a capacity of ${cap} blocks`)),
                ASSIGN_TIMEOUT_MS);
            this.#assigned = (range) => { clearTimeout(timer); this.#assigned = null; resolve(range); };
        });
    }

    // Announce that this peer's blocks are loaded and warmed up. Until then
    // other peers do not route through it. Called once; #flush() sends it when
    // the links this peer needs are open.
    announceReady() { this.#wantReady = true; this.#flush(); }

    // Announce that everything below position `have` is downloaded and warmed
    // up. Called each time the holding grows; #flush() sends it. `have` only
    // rises: the server clamps it to the claim and rejects a lower value,
    // since sessions may be routed through the positions already announced.
    announceHave(have) {
        if (have <= (this.#wantHave ?? -1)) return;
        this.#wantHave = have;
        this.#flush();
    }

    // The ready gate. Other peers route through a ready peer, and a route
    // needs an open link, but a WebRTC link opens about a second after the
    // claim that asks for it. A peer that announced immediately could be sent
    // a job it cannot receive or pass on. So an announcement waits for the
    // links it needs (#missing()). After `linkWait` it is sent anyway, with a
    // warning, so a peer that cannot reach one neighbour still serves the
    // rest.
    //
    // `have` is announced in steps: the highest requested holding whose links
    // are open. `ready` is not sent before a requested `have`. When no `have`
    // was requested, the server counts the whole claim as held, so the links
    // are checked for the whole claim.
    //
    // Runs on every request, roster change, link open or close, and on the
    // `linkWait` timer.
    #flush() {
        if (!this.range) return this.#settle();
        const [lo, hi] = this.range;
        const late = this.#held && performance.now() - this.#held.since >= this.linkWait;
        if (this.#wantHave == null) {
            if (this.#wantReady && !this.#announced && (late || !this.#missing(lo, hi).length)) {
                this.#announced = true;
                this.#ws({ type: 'ready' });
            }
            return this.#settle();
        }
        const want = this.#wantHave;

        let h = want;
        if (!late) while (h > (this.#have ?? lo) && this.#missing(lo, h).length) h--;
        if (late && want > (this.#have ?? -1) && this.#missing(lo, want).length) {
            console.warn(`announcing positions up to ${want} with no open link to ` +
                `peer${this.#missing(lo, want).length === 1 ? '' : 's'} ${this.#missing(lo, want).join(', ')} ` +
                `after ${this.linkWait / 1000}s. Jobs may not get through them yet.`);
        }
        if (h > (this.#have ?? -1) && h > lo) {
            this.#have = h;
            this.#ws({ type: 'have', have: h });
        }
        if (this.#wantReady && !this.#announced && this.#have != null
            && (late || !this.#missing(lo, this.#have).length)) {
            this.#announced = true;
            this.#ws({ type: 'ready' });
        }
        this.#settle();
    }

    // Run a timer while an announcement is held, so #flush() runs again at
    // `linkWait`. Clear it when nothing is held.
    #settle() {
        const holding = this.#wantHave != null && this.#wantHave > (this.#have ?? -1)
            || (this.#wantReady && !this.#announced);
        if (!holding) {
            clearTimeout(this.#held?.timer);
            this.#held = null;
            return;
        }
        if (this.#held) return;
        this.#held = { since: performance.now(), timer: setTimeout(() => this.#flush(), this.linkWait + 1) };
    }

    // The peers an announcement of the holding `[lo, h)` is waiting to link
    // to. For each side below that has ready holders, none of them linked,
    // all of that side's holders are returned. One open link per side is
    // enough, since routing prefers linked peers. Empty when nothing is
    // missing.
    //
    //   the hub     holders of position 0: where jobs enter and answers return
    //               (skipped if this peer holds position 0)
    //   below       holders of `lo - 1`: who hands jobs to this peer at `lo`
    //   above       holders of `h`: who this peer hands jobs to at its top
    #missing(lo, h) {
        const others = this.readyPeers.filter((p) => p.id !== this.id);
        const need = new Set();
        const side = (x) => {
            const who = holders(others, x);
            if (who.length && !who.some((id) => this.#live(id))) for (const id of who) need.add(id);
        };
        if (lo > 0) { side(0); side(lo - 1); }
        if (h < this.positions) side(h);
        return [...need];
    }

    // The peers a held announcement is waiting to link to, or null when
    // nothing is held. For a status line.
    get linking() {
        if (!this.#held || !this.range) return null;
        const missing = this.#missing(this.range[0], this.#wantHave ?? this.range[1]);
        return missing.length ? missing : null;
    }

    // Give back the unfilled tail `[have, hi)` of the claim. Nothing routes to
    // it, but placement counts claims, so keeping it would keep other peers
    // out of positions nobody serves.
    //
    // claim(0) gives up the whole range, and the server refuses that once a
    // peer is ready.
    releaseTail(have) {
        this.range = [this.range?.[0] ?? 0, have];
        this.#ws({ type: 'shrink' });
    }

    // Give up the whole holding, ready or not, and become a client. Used
    // when the GPU is lost: other peers stop routing here at once. A later
    // claim() starts the announcements from scratch.
    release() {
        this.range = null;
        this.#wantReady = false;
        this.#announced = false;
        this.#wantHave = null;
        this.#have = null;
        clearTimeout(this.#held?.timer);
        this.#held = null;
        this.#ws({ type: 'release' });
    }

    // Announce a range widened by one position on either edge. Unlike claim(),
    // this widens a ready peer's range in place, and the server accepts both
    // numbers as given. No reply.
    //
    // Sent only once the new position is servable: served() trusts `lo`, so a
    // lower edge must not be announced before it is held.
    //
    // `#wantHave` is raised too. A reconnect re-states the holding from it,
    // and would otherwise announce only the top of the initial claim.
    grow(lo, hi) {
        this.range = [lo, hi];
        if (hi > (this.#wantHave ?? -1)) this.#wantHave = hi;
        if (this.#ws({ type: 'grow', lo, hi }) && hi > (this.#have ?? -1)) this.#have = hi;
    }

    #wantReady = false;   // announceReady() was called
    #announced = false;   // `ready` was sent
    #wantHave = null;     // the highest holding requested
    #have = null;         // the highest holding sent
    #held = null;         // { since, timer } while an announcement waits for links

    // Send a message to a peer I link to. Returns false if the link is not
    // open or is backed up; the caller treats the route as broken.
    send(id, header, payload = null) {
        if (!this.#live(id)) return false;
        const link = this.links.get(id);

        const buf = header instanceof ArrayBuffer ? header : pack(header, payload);

        // A backed-up channel takes no more jobs. Sending more would overrun
        // the buffer, and Chrome would close the channel and break every
        // session on this link. The router uses backedUp() to tell this
        // refusal from a link that is down.
        const mark = buf.byteLength < SEND_SMALL ? SEND_CEILING : SEND_HIGH_WATER;
        if (link.ch.bufferedAmount > mark) {
            console.warn(`link to peer ${id} is backed up (` +
                `${(link.ch.bufferedAmount / 1048576).toFixed(1)} MiB queued)`);
            return false;
        }

        // Every fragment is sent before this returns, so the far end can
        // reassemble without reordering.
        link.frag = (link.frag ?? 0) + 1;
        for (const part of fragment(buf, wireLimit(link.pc), link.frag)) link.ch.send(part);
        return true;
    }

    // Bytes queued to go out on the link to `id`. route.js watches this while
    // waiting for an ack, to tell a slow link from a dead one.
    buffered(id) { return this.links.get(id)?.ch?.bufferedAmount ?? 0; }

    // Whether the link to `id` is open but has too much queued for send() to
    // add a job.
    backedUp(id) { return this.#live(id) && this.buffered(id) > SEND_HIGH_WATER; }

    #live(id) { return this.links.get(id)?.ch?.readyState === 'open'; }

    // Ask the signaling server for a seat before a reply generates. Resolves
    // { ok, max }: ok false means the mesh already has `max` replies
    // generating. With no answer within SEAT_WAIT_MS, or no open socket, it
    // resolves ok, so a signaling hiccup never blocks chatting. One request
    // at a time: the page sends one prompt at a time.
    seat() {
        return new Promise((resolve) => {
            if (!this.#ws({ type: 'chat-start' })) return resolve({ ok: true, max: this.maxChats });
            const timer = setTimeout(() => {
                this.#seating = null;
                resolve({ ok: true, max: this.maxChats });
            }, SEAT_WAIT_MS);
            this.#seating = { resolve, timer };
        });
    }

    // The reply has ended: give the seat back.
    unseat() { this.#ws({ type: 'chat-end' }); }

    #seating = null;

    // Messages are dropped, not queued, when the socket is down. Each one
    // describes the roster at the time it was sent and would be stale later.
    // The reconnect re-sends the claim and readiness.
    #ws(msg) {
        if (this.ws?.readyState !== 1) return false;
        this.ws.send(JSON.stringify(msg));
        return true;
    }

    #signal(m) {
        if (m.type === 'welcome') {
            // A tab that asked for a private mesh and was put elsewhere, or
            // the reverse, stops here.
            if (!!m.private !== this.room.private) {
                this.#stopFor(this.room.private
                    ? 'the signaling server did not put this tab in its private cluster'
                    : 'the signaling server put this tab in a private cluster it did not ask for');
                return;
            }
            // Joined: clear the welcome timeout and reset the backoff. Reset
            // here and not on `open`, so a server that accepts sockets and
            // never welcomes them is still backed off.
            clearTimeout(this.#hang);
            this.#backoff = RECONNECT_MIN_MS;
            this.trouble = null;
            const rejoin = this.#rejoining;
            const held = this.range;
            const sessions = Number.isInteger(m.sessions) && m.sessions >= 1 ? m.sessions : null;
            // A runner is sized once, so a different number after a reconnect
            // (a restarted server) does not apply to it.
            if (this.sessions != null && sessions !== this.sessions) {
                console.warn(`the cluster now keeps ${sessions ?? 'the default number of'} chats in `
                    + `memory per peer; this tab keeps ${this.sessions} until it is reloaded`);
            }
            const ctx = Number.isInteger(m.ctx) && m.ctx >= 1 ? m.ctx : null;
            if (this.ctx != null && ctx !== this.ctx) {
                console.warn(`the cluster now has a ${ctx ?? 'default'}-token context; this tab `
                    + `keeps ${this.ctx} tokens until it is reloaded`);
            }
            Object.assign(this, {
                id: m.id, blocks: m.blocks, positions: m.positions ?? 0,
                minBlocks: m.minBlocks, ice: m.ice ?? ICE,
                // Null if the server does not send one. The loader then uses
                // /checkpoint.json.
                weightsUrl: m.weightsUrl ?? null,
                maxChats: m.maxChats ?? null,
                sessions: this.sessions ?? sessions,
                ctx: this.ctx ?? ctx,
            });
            this.#welcomed();

            // After a reconnect the server sees a new peer: new id, no range,
            // not ready. The weights are still in VRAM, so re-send the claim
            // and readiness.
            if (rejoin) {
                this.#rejoining = false;
                if (held) {
                    this.#ws({ type: 'capacity', cap: held[1] - held[0],
                               lo: held[0], hi: held[1] });
                    this.range = held;
                    // Placement resets `have` to `lo`, so the holding and
                    // readiness are announced again through #flush(), since
                    // every link is new under the new id.
                    this.#have = null;
                    this.#announced = false;
                    clearTimeout(this.#held?.timer);
                    this.#held = null;
                    this.#flush();
                }
            }
        }
        if (m.type === 'assign') {
            this.range = m.lo == null ? null : [m.lo, m.hi];
            this.#assigned?.(this.range);
        }
        if (m.type === 'welcome' || m.type === 'roster' || m.type === 'assign') {
            if (m.peers) this.peers = m.peers;
            this.#sync();
            this.#flush();
            this.onChange?.();
        }
        if (m.type === 'chat') {
            const wait = this.#seating;
            this.#seating = null;
            if (wait) { clearTimeout(wait.timer); wait.resolve({ ok: m.ok !== false, max: m.max ?? null }); }
        }
        if (m.type === 'offer') this.#answer(m).catch((e) => console.warn(e.message));
        if (m.type === 'answer') this.#answered(m).catch((e) => console.warn(e.message));
    }

    // Who peer `p` links to: the holders of position 0 (the hub), plus the
    // holders of every position it could hand a job over at. That is any of
    // `(lo, hi]`, since plan() in route.js may stop anywhere in its holding.
    //
    // Not filtered on readiness: a client must be on the entry point's list,
    // or nobody opens a channel to it.
    //
    // Uses the whole claim, not `have`, so #sync() never prunes a link a live
    // session is pinned across, and the link set depends only on the roster.
    //
    // The initial claim is broadcast before it is downloaded, so links to a
    // new peer open while it fetches. A grow() is announced only once the
    // position is servable, so its links can open about a second late.
    // Routing tolerates an unlinked holder.
    #targets(p) {
        const t = new Set(holders(this.peers, 0));
        if (holding(p)) {
            const top = Math.min(p.hi, this.positions - 1);
            for (let x = p.lo + 1; x <= top; x++) for (const id of holders(this.peers, x)) t.add(id);
        }
        t.delete(p.id);
        return t;
    }

    // Who I link to: the symmetric closure of #targets. Both ends compute the
    // same set from the same roster, and the lower id offers. Uses
    // `this.range`, not `this.me`, because the `assign` reply arrives before
    // the roster broadcast with the same range.
    get wants() {
        const me = { id: this.id, lo: this.range?.[0] ?? null, hi: this.range?.[1] ?? null };
        const set = this.#targets(me);
        for (const p of this.peers) {
            if (p.id !== this.id && this.#targets(p).has(this.id)) set.add(p.id);
        }
        return [...set].sort((a, b) => a - b);
    }

    // Bring the links in line with `wants`. Links are keyed by peer, so a join
    // only adds links and in-flight work is unaffected. A link closes only
    // when its peer leaves, stops being a neighbour, or the link is dead.
    #sync() {
        if (this.id == null) return;
        const want = new Set(this.wants);

        // Prune first, so a failed connection is replaced in the same pass.
        // Runs on every roster change and on a failed link's retry timer
        // (#linkFailed).
        for (const [id, link] of this.links) {
            const dead = link.dead || link.pc.connectionState === 'failed'
                || link.pc.connectionState === 'closed';
            if (!want.has(id) || !this.has(id) || dead) {
                this.links.delete(id);        // before close(), so #linkFailed ignores it
                clearTimeout(link.openTimer);
                link.pc.close();
            }
        }
        for (const [id, r] of this.#retries) {
            if (!want.has(id)) { clearTimeout(r.timer); this.#retries.delete(id); }
        }

        // The lower id offers. The higher id waits for the offer.
        for (const id of want) if (id > this.id && !this.links.has(id)) this.#offer(id);
    }

    #attach(link, ch) {
        ch.binaryType = 'arraybuffer';
        link.ch = ch;
        // Reassembly state is per link, because fragment ids are per link. One
        // message at a time: the channel is ordered and send() sends a
        // message's fragments together.
        const frag = {};
        // Buckets are per channel, so a new link starts with a full budget.
        // Bytes are charged per frame and messages once reassembled. If a
        // fragment is dropped, reassembly of that message fails on its next
        // fragment.
        const bytes = bucket(...this.rate.bytes);
        const msgs = bucket(...this.rate.msgs);
        link.dropped = 0;
        let warned = 0;
        const drop = (why) => {
            link.dropped++;
            const now = performance.now();
            if (now - warned < 1000) return;
            warned = now;
            console.warn(`peer ${link.id ?? '?'}: ${why} (${link.dropped} dropped so far)`);
        };
        ch.addEventListener('message', (ev) => {
            if (!(ev.data instanceof ArrayBuffer)) return drop('a message that is not binary');
            if (!bytes(ev.data.byteLength)) return drop('over the byte rate');
            let raw, header, payload;
            try {
                raw = reassemble(frag, ev.data);
                if (!raw) return;             // more fragments to come
                if (!msgs()) return drop('over the message rate');
                // `raw` is the reassembled message. Forwarding fragments it
                // again for the next link's limit.
                ({ header, payload } = unpack(raw));
                if (header === null || typeof header !== 'object') throw new Error('header is not an object');
            } catch (err) {
                console.warn(`peer ${link.id ?? '?'}: ${err.message}`);
                return;
            }
            this.onMessage?.(header, payload, raw, link.id);
        });
        // `connected` changes when a channel opens or closes, and no roster
        // message arrives then, so notify the UI here. A held announcement may
        // also be ready to send (#flush()).
        ch.addEventListener('open', () => {
            // The link is up: reset this peer's retry backoff.
            clearTimeout(link.openTimer);
            clearTimeout(this.#retries.get(link.id)?.timer);
            this.#retries.delete(link.id);
            this.#flush();
            this.onChange?.();
        });
        ch.addEventListener('close', () => {
            this.#linkFailed(link, 'closed');
            this.#flush();
            this.onChange?.();
        });
    }

    // Detect a link that will not open or has failed: its ICE failed, its
    // channel closed, or it has not opened within `linkOpen`. Runs on both
    // ends. The lower id offers again. The higher id drops the dead link and
    // waits for that offer.
    #watch(link) {
        const { pc } = link;
        pc.addEventListener?.('connectionstatechange', () => {
            if (pc.connectionState === 'failed') this.#linkFailed(link, 'failed');
        });
        // Some browsers lack connectionState and report failure only here.
        pc.addEventListener?.('iceconnectionstatechange', () => {
            if (pc.iceConnectionState === 'failed') this.#linkFailed(link, 'failed');
        });
        link.openTimer = setTimeout(() => {
            if (link.ch?.readyState !== 'open') {
                this.#linkFailed(link, `did not open within ${this.linkOpen / 1000}s`);
            }
        }, this.linkOpen);
    }

    // Mark a link dead and schedule a #sync() to replace it, after this peer's
    // backoff (LINK_RETRY_MS). One timer per peer. The backoff doubles each
    // time it fires and resets when a link to that peer opens.
    #linkFailed(link, why) {
        if (link.dead || this.#closed || this.links.get(link.id) !== link) return;
        link.dead = true;
        clearTimeout(link.openTimer);
        const [first, max] = this.linkRetry;
        const r = this.#retries.get(link.id) ?? { delay: first, timer: null };
        this.#retries.set(link.id, r);
        if (r.timer == null) {
            console.warn(`link to peer ${link.id} ${why}. Trying again in ${r.delay / 1000}s.`);
            r.timer = setTimeout(() => {
                r.timer = null;
                r.delay = Math.min(r.delay * 2, max);
                this.#sync();
                this.onChange?.();
            }, r.delay);
        }
        this.onChange?.();
    }

    async #offer(to) {
        const pc = new RTCPeerConnection(this.#rtcConfig());
        const link = { pc, ch: null, offerer: true, id: to };
        this.links.set(to, link);                 // before any await, so #sync cannot offer twice
        this.#watch(link);
        this.#attach(link, pc.createDataChannel('mesh', { ordered: true }));

        await pc.setLocalDescription(await pc.createOffer());
        await iceDone(pc);
        this.#ws(await this.#seal(to, 'offer', pc.localDescription));
    }

    // Seal an outgoing SDP to this room. `from` and `to` are in the MAC, so
    // the server cannot relabel or redirect it. A plain object is sent, not
    // the RTCSessionDescription, so the sealed fields are the sent fields.
    async #seal(to, kind, desc) {
        const sdp = { type: desc.type, sdp: this.relayOnly ? relayOnlySdp(desc.sdp) : desc.sdp };
        const { mac } = await this.room.seal({ kind, from: this.id, to, sdp });
        return { type: 'signal', to, kind, sdp, ...(mac != null && { mac }) };
    }

    #rtcConfig() {
        return { iceServers: this.ice, iceTransportPolicy: this.relayOnly ? 'relay' : 'all' };
    }

    // Whether the welcome brought a TURN relay. In relay-only mode no link can
    // open without one.
    get relay() { return hasTurn(this.ice); }

    // Whether an arriving offer or answer carries this room's seal. A failure
    // is counted and the message is dropped.
    async #verified(m) {
        if (await this.room.verify({ kind: m.type, from: m.from, to: this.id, sdp: m.sdp, mac: m.mac })) {
            return true;
        }
        this.authFailures++;
        console.warn(`dropped an ${m.type} from peer ${m.from}: it does not carry this cluster's seal`);
        this.onChange?.();
        return false;
    }

    async #answered(m) {
        const link = this.links.get(m.from);
        if (!link?.offerer) return;
        if (!(await this.#verified(m))) return;
        if (this.links.get(m.from) !== link) return;     // replaced meanwhile
        await link.pc.setRemoteDescription(m.sdp);
    }

    // Answer an offer. The offerer is always the lower id, so a peer never
    // both offers to and answers the same peer. A repeat offer replaces the
    // link.
    async #answer(m) {
        const { from, sdp } = m;
        if (this.id != null && from > this.id) return;   // the lower id offers
        // Verified before the existing link is closed, so an unsealed offer
        // cannot close a working link.
        if (!(await this.#verified(m))) return;
        const existing = this.links.get(from);
        if (existing) {
            this.links.delete(from);
            clearTimeout(existing.openTimer);
            existing.pc.close();
        }

        const pc = new RTCPeerConnection(this.#rtcConfig());
        const link = { pc, ch: null, offerer: false, id: from };
        this.links.set(from, link);
        this.#watch(link);
        pc.addEventListener('datachannel', (e) => {
            this.#attach(link, e.channel);
            this.onChange?.();
        });

        await pc.setRemoteDescription(sdp);
        await pc.setLocalDescription(await pc.createAnswer());
        await iceDone(pc);
        this.#ws(await this.#seal(from, 'answer', pc.localDescription));
    }
}
