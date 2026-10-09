// Node tests for the parts of the mesh that need no browser: the Mesh getters
// that decide where a message goes and who links to whom, and the signaling
// server's roster and placement. Everything below the datachannel is stubbed.
// The tests cover membership rules, not WebRTC.
//
// The assignment rule is tested in tools/split.test.mjs.
//
//   node tools/net.test.mjs

import { WebSocket as WS } from 'ws';
import { webcrypto, createHmac } from 'node:crypto';
// In a Node 18 module, WebCrypto is not a global as it is in a browser.
globalThis.crypto ??= webcrypto;
const { openRoom, newSecret } = await import('../web/room.js');

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);

// --- Mesh getters ---
//
// net.js uses `location` and `WebSocket` in the constructor and nothing else
// until a message arrives, so stubs are enough to reach the getters.

globalThis.location = { protocol: 'http:', host: 'localhost:8000' };
globalThis.WebSocket = class { addEventListener() {} send() {} close() {} readyState = 1; };

const { Mesh, hasTurn, pack, unpack, fragment, reassemble, wireLimit,
        FRAG_HEAD, WIRE_DEFAULT, MAX_MESSAGE, bucket,
        SEND_HIGH_WATER, SEND_CEILING } = await import('../web/net.js');

// --- Framing and fragmentation ---
//
// A T_MAX prefill chunk at hidden 2048 is 512 KiB, more than the 256 KiB a
// datachannel negotiates, so it must be fragmented and reassembled.

{
    const H = 2048, T_MAX = 64;
    const hidden = (t) => Float32Array.from({ length: t * H }, (_, i) => Math.sin(i) * 1e3);
    const job = { k: 'job', s: 'sess-1', from: 3, pos: 7, past: 0, n: T_MAX };

    // A full prefill chunk is larger than one datachannel message.
    const whole = pack(job, hidden(T_MAX));
    ok('a T_MAX chunk at hidden 2048 exceeds the datachannel limit',
        whole.byteLength > WIRE_DEFAULT, `${whole.byteLength} B`);

    // A decode token fits, and is sent unfragmented.
    const one = pack({ ...job, n: 1 }, hidden(1));
    eq('one decode token is a single fragment', fragment(one, WIRE_DEFAULT).length, 1);
    ok('and is passed through by identity, not copied',
        fragment(one, WIRE_DEFAULT)[0] === one);
    ok('an unfragmented buffer reassembles to itself',
        reassemble({}, one) === one);

    const parts = fragment(whole, WIRE_DEFAULT, 1);
    ok('a T_MAX chunk splits', parts.length > 1, `${parts.length} parts`);
    ok('and every part fits the wire',
        parts.every((b) => b.byteLength <= WIRE_DEFAULT),
        `max ${Math.max(...parts.map((b) => b.byteLength))} B`);
    ok('and each part carries the fragment prefix',
        parts.every((b) => b.byteLength > FRAG_HEAD));

    // Reassembly returns nothing until the last fragment arrives.
    const st = {};
    const got = parts.map((b) => reassemble(st, b));
    eq('reassembly is null until the final fragment',
        got.slice(0, -1).map((x) => x === null), got.slice(0, -1).map(() => true));
    ok('the final fragment completes the message', got.at(-1) instanceof ArrayBuffer);

    eq('and the reassembled message is byte-identical in length',
        got.at(-1).byteLength, whole.byteLength);
    const a = new Uint8Array(whole), b = new Uint8Array(got.at(-1));
    ok('and byte-identical in content', a.every((v, i) => v === b[i]));

    // The floats the next peer reads must equal the ones sent, not only the
    // bytes.
    const src = hidden(T_MAX);
    const stRt = {};
    const rt = unpack(fragment(pack(job, src), WIRE_DEFAULT, 2)
        .map((f) => reassemble(stRt, f)).at(-1));
    eq('the header survives the round trip', rt.header, job);
    eq('the payload survives it too, in full', rt.payload.length, src.length);
    ok('with every float intact',
        src.every((v, i) => Object.is(v, rt.payload[i])));

    // A tiny limit: many fragments, with a short last one.
    const tiny = fragment(whole, FRAG_HEAD + 4096, 3);
    ok('a small limit splits further', tiny.length > parts.length);
    const st2 = {};
    const back = tiny.map((f) => reassemble(st2, f)).at(-1);
    ok('and still reassembles exactly',
        new Uint8Array(back).every((v, i) => v === a[i]) && back.byteLength === whole.byteLength);

    // Cannot happen with ordered delivery, and must throw if it does.
    let threw = false;
    try { reassemble({}, tiny[1]); } catch { threw = true; }
    ok('a fragment arriving out of order throws rather than corrupting', threw);

    // A channel that enforces its negotiated limit as Chrome does, by throwing
    // synchronously, with a full T_MAX prefill chunk sent through it.
    {
        const chrome = { sent: [], send(b) {
            if (b.byteLength > WIRE_DEFAULT) {
                throw new Error("Failed to execute 'send' on 'RTCDataChannel': " +
                    'Trying to send message larger than max-message-size');
            }
            this.sent.push(b);
        } };

        let threw = false;
        try { chrome.send(whole); } catch { threw = true; }
        ok('sending a T_MAX chunk unsplit is what threw in the browser', threw);

        threw = false;
        try { for (const f of fragment(whole, WIRE_DEFAULT, 7)) chrome.send(f); }
        catch { threw = true; }
        ok('sending it fragmented does not', !threw);

        const st3 = {};
        const back = chrome.sent.map((f) => reassemble(st3, f)).filter(Boolean);
        eq('and the far end sees exactly one message', back.length, 1);
        const { header, payload } = unpack(back[0]);
        eq('carrying the job it started as', header.n, T_MAX);
        eq('with the whole residual stream', payload.length, T_MAX * H);
    }

    eq('wireLimit falls back when sctp says nothing', wireLimit(undefined), WIRE_DEFAULT);
    eq('wireLimit falls back on a nonsense value',
        wireLimit({ sctp: { maxMessageSize: 0 } }), WIRE_DEFAULT);
    eq('wireLimit takes a smaller negotiated limit',
        wireLimit({ sctp: { maxMessageSize: 65536 } }), 65536);
    eq('wireLimit never believes a larger one',
        wireLimit({ sctp: { maxMessageSize: 2 ** 30 } }), WIRE_DEFAULT);
}


// 30 blocks is 32 positions: the embedding at 0, blocks 0–29 at 1–30, the head
// at 31. Peers are placed on that line, so a covering chain must reach 32. A
// peer stopping at 30 leaves the head unheld, and nothing can sample.
const L = 30;
const N = 32;
const P = (id, lo, hi, ready = true) => ({ id, lo, hi, ready });
const client = (id, ready = false) => ({ id, lo: null, hi: null, ready });

// `me` is this tab's own roster entry. `links` are the ids it has an open
// channel to, and `gathering` the ones still doing ICE.
function mesh(me, peers, { links = [], gathering = [] } = {}) {
    const m = new Mesh({});
    // This Mesh is never welcomed, so stop it. Otherwise it would keep
    // reconnecting (READY_TIMEOUT_MS) through any WebSocket stub a later
    // section installs.
    m.stop();
    Object.assign(m, {
        id: me.id, blocks: L, positions: N, minBlocks: 10,
        range: me.lo == null ? null : [me.lo, me.hi],
        peers: [me, ...peers.filter((p) => p.id !== me.id)],
    });
    for (const id of links) m.links.set(id, { pc: {}, ch: { readyState: 'open' } });
    for (const id of gathering) m.links.set(id, { pc: {}, ch: { readyState: 'connecting' } });
    return m;
}

// A chain that covers the line end to end, used throughout. The last peer
// holds two more positions than the others, because it includes the head.
const thirds = [P(0, 0, 10), P(1, 10, 20), P(2, 20, 32)];

console.log('\nMesh.next(x): ready holders of a position, lowest id first');
{
    eq('a cursor at 0 goes to the holder of the embedding',
        mesh(P(2, 20, 32), thirds).next(0), [0]);
    eq('a cursor at 10 goes to the holder of position 10',
        mesh(P(0, 0, 10), thirds).next(10), [1]);
    eq('skips a holder that is still loading',
        mesh(P(0, 0, 10), [P(1, 10, 20, false), P(4, 10, 20)]).next(10), [4]);
    // The list includes me if I hold the position. route() checks whether it
    // can run a job before asking who else could, so this list is only used by
    // a peer that has declined the work.
    eq('overlapping holders are all candidates, sorted',
        mesh(P(9, 0, 10), [P(3, 5, 20), P(1, 8, 25)]).next(9), [1, 3, 9]);
    eq('nobody holds a position past the line', mesh(P(0, 0, 10), thirds).next(32), []);
    eq('the head is a position like any other', mesh(P(0, 0, 10), thirds).next(31), [2]);
    eq('a peer holding nothing is never a candidate',
        mesh(P(0, 0, 32), [client(7, true)]).next(0), [0]);
}

console.log('\nMesh: a peer serves what it has, not what it claimed');
{
    // `have` is how far a download has got. A peer claiming [10,20) that has
    // loaded four positions is a four-position peer to everyone else. No extra
    // message is needed: every browser computes it from the same roster.
    const partial = (id, lo, hi, have) => ({ id, lo, hi, have, ready: true });
    const chain = [P(0, 0, 10), partial(1, 10, 20, 14), P(2, 20, 32)];

    eq('a job goes to the holder of a position it has actually filled',
        mesh(P(0, 0, 10), chain).next(13), [1]);
    eq('and not to one whose claim covers it but whose download has not',
        mesh(P(0, 0, 10), chain).next(14), []);
    ok('a half-filled chain does not cover the model',
        mesh(P(0, 0, 10), chain).staffed === false);

    const full = [P(0, 0, 10), partial(1, 10, 20, 20), P(2, 20, 32)];
    ok('and covers it once the range is filled', mesh(P(0, 0, 10), full).staffed);

    const cov = mesh(P(0, 0, 10), chain).coverage;
    eq('coverage counts positions, not peers', [cov.covered, cov.total], [26, 32]);
    eq('and names the holes', cov.holes, [[14, 20]]);
    eq('r is 0 while one exists', cov.r, 0);

    // An entry with no `have` holds its whole claim (a peer that loaded its
    // range in one step).
    eq('an entry without a have is its claim', mesh(P(0, 0, 10), thirds).coverage.covered, 32);
}

console.log('\nMesh.exits / reach: every position a peer can hand off at');
{
    // A peer holding [10,20) can hand off at any position in (10,20]: at the
    // end of its holding, at the end of its claim once full, or earlier when
    // nobody covers the next position. The map is keyed by position, and an
    // empty entry is a position it must not stop at.
    const m = mesh(P(0, 10, 20), [P(0, 10, 20), P(1, 14, 24), P(2, 18, 32)]);
    const e = m.exits(10, 20);
    eq('a position only its own holder covers has no taker', e.get(12), [0]);
    eq('one an overlapping peer covers names them', e.get(14), [0, 1]);
    eq('and both, where two overlap', e.get(18), [0, 1, 2]);
    eq('the map runs from lo+1 to hi', [...e.keys()].at(0), 11);
    eq('and stops at hi', [...e.keys()].at(-1), 20);
    eq('a position nobody holds is empty, not missing: that is the hole plan() walks back from',
        mesh(P(0, 0, 10), [P(0, 0, 10), P(2, 12, 32)]).exits(0, 10).get(10), []);

    // Which of two holders runs furthest. hops() prefers it, since it can save
    // several hops.
    eq('reach is how far a candidate carries the cursor', m.reach(1, 14), 10);
    eq('a peer that reaches further wins', m.reach(2, 14) > m.reach(1, 14), true);
    eq('and a partial holding reaches only as far as it has filled',
        mesh(P(0, 0, 10), [P(1, 10, 20, true)].map((p) => ({ ...p, have: 14 }))).reach(1, 10), 4);
}

console.log('\nMesh.wants: the link set spans the claim, and only grows');
{
    // The forward links span the whole claim. Peer 2 below holds 16–19 and
    // ends where peer 1 ends, so neither `holders(1.hi)` nor `holders(2.hi)`
    // includes the other. Peer 1 still hands off at 16 while it is loading,
    // and whenever nobody covers 20.
    const roster = [P(0, 0, 10), P(1, 10, 20), P(2, 16, 20), P(3, 20, 32)];
    const w = mesh(P(1, 10, 20), roster).wants;
    eq('a peer links to the holders of every position it could exit at, not just hi',
        w, [0, 2, 3]);

    // Peer 1 is halfway through its claim here. The set is the same: links
    // follow the claim, which does not change, so a hand-off boundary moving
    // from 16 to 20 as blocks load never prunes the link under a session
    // pinned at 16.
    const half = mesh({ ...P(1, 10, 20), have: 16 }, roster).wants;
    eq('and a peer half way through its claim already holds them all', half, [0, 2, 3]);

    // The hub rule also applies: everyone links to the entry point.
    eq('a client links to the entry point and nothing else',
        mesh(client(9), roster).wants, [0]);
}

console.log('\nMesh.staffed / redundancy: coverage is runnability');
{
    ok('a chain reaching the head covers the model', mesh(P(0, 0, 10), thirds).staffed);
    ok('the same chain stopping one short of the head does not',
        mesh(P(0, 0, 10), [P(0, 0, 10), P(1, 10, 20), P(2, 20, 31)]).staffed === false);
    eq('at r = 1', mesh(P(0, 0, 10), thirds).redundancy, 1);

    ok('a gap is not staffed',
        mesh(P(0, 0, 10), [P(0, 0, 10), P(2, 20, 32)]).staffed === false);
    eq('and r = 0', mesh(P(0, 0, 10), [P(0, 0, 10), P(2, 20, 32)]).redundancy, 0);

    ok('a holder still downloading does not staff its positions',
        mesh(P(0, 0, 10), [P(0, 0, 10), P(1, 10, 20, false), P(2, 20, 32)]).staffed === false);

    ok('one peer holding the whole model is staffed on its own',
        mesh(P(5, 0, 32), []).staffed);

    eq('doubled-up peers raise r rather than coverage',
        mesh(P(0, 0, 10), [...thirds, P(3, 0, 10), P(4, 10, 20), P(5, 20, 32)]).redundancy, 2);

    ok('clients must not disguise a missing stretch',
        mesh(P(0, 0, 10), [P(0, 0, 10), P(1, 10, 20), client(9, true)]).staffed === false);
}

console.log('\nMesh.wants: forward edges plus the entry hub, symmetrically closed');
{
    // Rule 1: a holder links to whoever can pick up its cursor. Rule 2:
    // everyone links to the holders of position 0.
    eq('a middle peer wants its successor and the entry point',
        mesh(P(1, 10, 20), thirds).wants, [0, 2]);
    eq('the entry point wants its own successor, and everyone who wants it',
        mesh(P(0, 0, 10), thirds).wants, [1, 2]);
    // A terminal peer has no forward links, only rule 2, which is why an
    // answer takes at most two hops. Peer 1 appears through the symmetric
    // closure: 1's cursor is handed to 2.
    eq('a terminal peer wants the entry point, and is wanted by its predecessor',
        mesh(P(2, 20, 32), thirds).wants, [0, 1]);

    // A tab holding nothing must still be linked, or the answer to its own
    // prompt could not reach it.
    eq('a client wants the entry point', mesh(client(9), thirds).wants, [0]);
    ok('and the entry point wants it back: the closure, not a special case',
        mesh(P(0, 0, 10), [...thirds, client(9)]).wants.includes(9));
    ok('while a peer in the middle does not owe a client anything',
        mesh(P(1, 10, 20), [...thirds, client(9)]).wants.includes(9) === false);

    // Two peers splitting the model each want a link to the other, so the
    // offerer is chosen by id.
    const pair = [P(0, 0, 16), P(1, 16, 32)];
    eq('two peers splitting the model each want the other, exactly once',
        [mesh(pair[0], pair).wants, mesh(pair[1], pair).wants], [[1], [0]]);

    eq('a peer holding the whole model wants nobody', mesh(P(5, 0, 32), []).wants, []);

    // A peer can hold the embedding and no blocks, and is linked like any
    // other peer.
    const embedOnly = [P(0, 0, 1), P(1, 1, 32)];
    eq('an embedding-only peer wants its successor, and is the hub for everyone',
        [mesh(embedOnly[0], embedOnly).wants, mesh(embedOnly[1], embedOnly).wants],
        [[1], [0]]);
    ok('and it covers the line with one other peer', mesh(embedOnly[0], embedOnly).staffed);

    // Readiness does not filter links: a peer needs an open channel before it
    // can announce anything.
    eq('a holder still downloading is still owed a link',
        mesh(P(0, 0, 10), [P(0, 0, 10), P(1, 10, 20, false)]).wants, [1]);
}

console.log('\nMesh.entry / hub: where jobs go in and answers come back');
{
    eq('entry is the ready holders of the embedding', mesh(P(2, 20, 32), thirds).entry, [0]);
    eq('hub is the first holder of the embedding I can actually reach',
        mesh(P(2, 20, 32), thirds, { links: [0] }).hub, 0);
    eq('a hub still gathering ICE is not a hub',
        mesh(P(2, 20, 32), thirds, { gathering: [0] }).hub, null);
    eq('an unready holder of the embedding can still relay an answer',
        mesh(P(2, 20, 32), [P(0, 0, 10, false), P(1, 10, 20), P(2, 20, 32)],
            { links: [0] }).hub, 0);
}

console.log('\nMesh.reachable: an answer can find its way home');
{
    ok('a middle peer is reached through the entry point it links to',
        mesh(P(1, 10, 20), thirds, { links: [0] }).reachable);
    ok('with no link to the entry point, nothing can answer me',
        mesh(P(1, 10, 20), thirds).reachable === false);
    ok('a client is reached the same way',
        mesh(client(9), thirds, { links: [0] }).reachable);

    // For the entry point, the answer arrives from the peer that sampled it,
    // the holder of the head.
    ok('the entry point is reached from the holder of the head',
        mesh(P(0, 0, 10), thirds, { links: [2] }).reachable);
    ok('the entry point with no link from the head is not reachable',
        mesh(P(0, 0, 10), thirds, { links: [1] }).reachable === false);
    ok('a peer holding the whole model answers itself', mesh(P(5, 0, 32), []).reachable);
    ok('nobody holds the embedding at all',
        mesh(client(9), [P(1, 10, 32)]).reachable === false);
}

console.log('\nMesh.connected: covered, a way in, a way home, and my own hand-off open');
{
    ok('a middle peer with both links up', mesh(P(1, 10, 20), thirds, { links: [0, 2] }).connected);
    ok('my hand-off is not open yet',
        mesh(P(1, 10, 20), thirds, { links: [0] }).connected === false);
    ok('nothing can answer me',
        mesh(P(1, 10, 20), thirds, { links: [2] }).connected === false);
    ok('a stretch of the model is still loading',
        mesh(P(1, 10, 20), [P(0, 0, 10, false), P(1, 10, 20), P(2, 20, 32)],
            { links: [0, 2] }).connected === false);
    ok('a client only needs the entry point',
        mesh(client(9), thirds, { links: [0] }).connected);
    ok('a client with no link to the entry point cannot send',
        mesh(client(9), thirds).connected === false);
    ok('one peer holding the whole model needs no links at all',
        mesh(P(5, 0, 32), []).connected);

    // The hand-off can be at any position in `(lo, have]`, not only `hi`. Peer
    // 1 below has no link to the holder of 20 but has one to a peer holding
    // 16–19. A job it runs is handed over at 19, so `connected` is true.
    ok('a hand-off short of my own end still counts',
        mesh(P(1, 10, 20), [P(0, 0, 10), P(1, 10, 20), P(2, 16, 20), P(3, 20, 32)],
            { links: [0, 2] }).connected);
    ok('but no reachable exit at all does not',
        mesh(P(1, 10, 20), [P(0, 0, 10), P(1, 10, 20), P(2, 16, 20), P(3, 20, 32)],
            { links: [0] }).connected === false);

    // A peer that holds blocks but is not ready runs no jobs, so its hand-off
    // is not checked. It can still prompt the mesh like any client.
    ok('an unready holder can still prompt through the entry point',
        mesh(P(3, 10, 20, false), [...thirds, P(3, 10, 20, false)], { links: [0] }).connected);
}

console.log('\nMesh.linked: an open channel, not merely a pending connection');
{
    const m = mesh(P(0, 0, 10), thirds, { links: [1], gathering: [2] });
    ok('open link', m.linked(1));
    ok('a gathering link is not linked', m.linked(2) === false);
    ok('unknown peer', m.linked(99) === false);
}

console.log('\nMesh.send: a backed-up link takes no more work, and still carries the news');
{
    // What one hop sends the next: a job with its residual stream, and
    // header-only messages.
    const job = [{ k: 'job', s: 'a', pos: 10, past: 0, n: 64 }, new Float32Array(64 * 64)];
    const err = [{ k: 'err', s: 'a', to: 0, from: 1, msg: 'refused', busy: true }];
    const at = (queued) => {
        const m = mesh(P(0, 0, 10), thirds, { links: [1] });
        const sent = [];
        m.links.set(1, { pc: {}, ch: { readyState: 'open', bufferedAmount: queued,
                                         send: (part) => sent.push(part) } });
        return { m, sent };
    };

    let t = at(0);
    eq('an idle link takes a job', [t.m.send(1, ...job), t.sent.length > 0], [true, true]);
    eq('and is not backed up', t.m.backedUp(1), false);

    t = at(SEND_HIGH_WATER + 1);
    eq('past the mark a job is refused, and nothing is queued', [t.m.send(1, ...job), t.sent.length], [false, 0]);
    eq('the link says it is backed up, not down', [t.m.backedUp(1), t.m.linked(1)], [true, true]);
    eq('a header on its own still goes', [t.m.send(1, ...err), t.sent.length], [true, 1]);
    eq('and so does one being forwarded as bytes', t.m.send(1, pack(...err)), true);
    eq('a job being forwarded as bytes does not', t.m.send(1, pack(...job)), false);

    t = at(SEND_CEILING + 1);
    eq('past the ceiling nothing goes', [t.m.send(1, ...err), t.sent.length], [false, 0]);

    eq('a link that is not there is not backed up', mesh(P(0, 0, 10), thirds).backedUp(1), false);
    eq('nor one still opening', mesh(P(0, 0, 10), thirds, { gathering: [1] }).backedUp(1), false);
}

console.log('\nMesh.ready: retries a failed or hung first connect, never gives up on it');
{
    // A WebSocket stub that records listeners, so a test can fire
    // 'error'/'close'/'message' directly. Installed for this section only.
    // Mesh reads the global `WebSocket` at construction time, so reassigning
    // it here does not affect earlier sections.
    class FakeWS {
        static made = 0;
        // readyState 1 = OPEN. Mesh does not send in any other state: a real
        // WebSocket throws once it is CLOSING or CLOSED.
        constructor() { this.listeners = {}; this.sent = []; this.readyState = 1; FakeWS.made++; }
        addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
        send(m) { this.sent.push(JSON.parse(m)); }
        close() { this.readyState = 3; }
        emit(type, ev = {}) { for (const fn of this.listeners[type] ?? []) fn(ev); }
    }
    globalThis.WebSocket = FakeWS;

    const welcome = { type: 'welcome', id: 1, blocks: 30, minBlocks: 10, peers: [] };

    {
        const m = new Mesh({});
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        ok('resolves on welcome', await m.ready.then(() => true, () => false));
        eq('and carries the model depth the server is splitting', m.blocks, 30);
        eq('a peer holds nothing until it is placed', m.range, null);
    }
    // Whether a promise has settled, checked without waiting for it.
    const settled = async (p) => {
        let s = 'pending';
        p.then(() => { s = 'resolved'; }, () => { s = 'rejected'; });
        await new Promise((r) => setTimeout(r, 0));
        return s;
    };
    const later = (ms) => new Promise((r) => setTimeout(r, ms));

    // A first socket that fails must be retried and `ready` must stay pending.
    // If `ready` rejected, the page would report "cannot reach the mesh" while
    // the Mesh reconnected and joined as a peer nothing could prompt from.
    for (const [what, fail, why] of [
        ['a WebSocket error', (ws) => ws.emit('error'), /could not open a connection/],
        ['a close before the welcome', (ws) => ws.emit('close', { code: 1006 }), /closed \(code 1006\)/],
    ]) {
        const m = new Mesh({});
        const first = m.ws;
        fail(first);
        eq(`${what} does not reject ready`, await settled(m.ready), 'pending');
        ok('it says why, and that it is retrying', why.test(m.trouble ?? '') && /retrying/.test(m.trouble),
            `${m.trouble}`);
        ok('and is offline meanwhile', m.online === false);
        await later(1200);
        ok('a new socket is opened on its own', m.ws !== first);
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        eq('whose welcome resolves ready', await settled(m.ready), 'resolved');
        eq('and clears the trouble', m.trouble, null);
        m.stop();
    }
    {
        // A socket that neither opens nor fails. Only the timer moves it on.
        const before = FakeWS.made;
        const m = new Mesh({ readyTimeout: 300 });
        const first = m.ws;
        await later(350);
        ok('a socket with no welcome by readyTimeout is given up on',
            /no response from the signaling server/.test(m.trouble ?? ''), `${m.trouble}`);
        eq('without rejecting ready', await settled(m.ready), 'pending');
        // Its close event arrives after it was dropped and must not open a
        // second replacement, which would join as an extra peer.
        first.emit('close', { code: 1006 });
        await later(1100);
        eq('and exactly one replacement is opened', FakeWS.made - before, 2);
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        eq('which can still join', await settled(m.ready), 'resolved');
        m.stop();
    }
    {
        // Resolve, then close: a later disconnect must leave `ready` resolved.
        const m = new Mesh({});
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        m.ws.emit('close', { code: 1006 });
        ok('a later close does not retroactively reject',
            await m.ready.then(() => true, () => false));
        m.stop();
    }

    console.log('\nMesh: the signalling socket dropping, and coming back');
    {
        const m = new Mesh({});
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        await m.ready;
        ok('online once welcomed', m.online === true);

        // A server ping is answered. The traffic stops a proxy closing the
        // connection as idle, which Cloudflare does after about 100s.
        m.ws.emit('message', { data: JSON.stringify({ type: 'ping' }) });
        eq('a ping is answered with a pong', m.ws.sent.at(-1), { type: 'pong' });

        // Claim a range, so there is something to reclaim after the reconnect.
        const p = m.claim(9);
        m.ws.emit('message', { data: JSON.stringify({ type: 'assign', lo: 0, hi: 9 }) });
        await p;
        m.announceReady();

        const first = m.ws;
        first.readyState = 3;
        m.ws.emit('close', { code: 1006 });
        ok('a drop is visible rather than silent', m.online === false);
        ok('but the roster is kept: datachannels are unaffected by it',
            Array.isArray(m.peers));
        eq('and nothing is sent down the dead socket',
            first.sent.filter((x) => x.type === 'capacity').length, 1);

        // After the reconnect the server sees a new id with no range, but this
        // tab still has its blocks in VRAM.
        await new Promise((r) => setTimeout(r, 1200));
        ok('it reconnects on its own', m.ws !== first);
        m.ws.emit('message', { data: JSON.stringify({ ...welcome, id: 7 }) });
        const reclaim = m.ws.sent.find((x) => x.type === 'capacity');
        eq('and reclaims the blocks it already holds', reclaim,
            { type: 'capacity', cap: 9, lo: 0, hi: 9 });
        ok('re-announcing readiness too',
            m.ws.sent.some((x) => x.type === 'ready'));
        ok('online again', m.online === true);
        m.stop();
    }

    console.log('\nMesh: a message carries the id of the link it arrived on');
    {
        // Covers #attach. The other tests here call the framing functions
        // directly, and mesh.test.mjs passes the router its own `from`. The
        // error path back one hop depends on the peer id being passed from the
        // link to onMessage.
        //
        // `from` is not a header field. The link knows its far end, so the
        // reverse path adds nothing to the message, works with the zero-copy
        // `raw` forward path, and is authenticated by the connection.
        const chans = [];
        const channel = () => {
            const l = {};
            const ch = {
                readyState: 'open',
                addEventListener(t, fn) { (l[t] ??= []).push(fn); },
                emit(t, ev) { for (const fn of l[t] ?? []) fn(ev); },
            };
            chans.push(ch);
            return ch;
        };
        globalThis.RTCPeerConnection = class {
            constructor() { this.iceGatheringState = 'complete'; this.connectionState = 'new'; }
            createDataChannel() { return channel(); }
            addEventListener() {}
            async createOffer() { return { type: 'offer', sdp: '' }; }
            async setLocalDescription(d) { this.localDescription = d; }
            async setRemoteDescription() {}
            close() {}
        };

        const seen = [];
        const m = new Mesh({ onMessage: (...a) => seen.push(a) });
        // Peer 2 holds position 0, so this tab links to it (the entry point).
        // This tab's id is lower, so it offers.
        m.ws.emit('message', { data: JSON.stringify({
            ...welcome,
            peers: [{ id: 1, lo: null, hi: null, have: null, ready: false },
                    { id: 2, lo: 0, hi: 10, have: 10, ready: true }],
        }) });
        await m.ready;
        eq('the offer opened one link', [...m.links.keys()], [2]);

        const job = { k: 'job', s: 'x-1', from: 9, pos: 0, past: 0, n: 1, ids: [7] };
        const whole = pack(job, new Float32Array([1.5, 2.5]));
        for (const f of fragment(whole, FRAG_HEAD + 32, 4)) {
            chans[0].emit('message', { data: f });
        }

        eq('one whole message came out of the fragments', seen.length, 1);
        eq('with its header', seen[0][0], job);
        eq('and its payload', [...seen[0][1]], [1.5, 2.5]);
        eq('and the id of the peer that sent it, which is on no wire', seen[0][3], 2);
        m.stop();

        // The same link on a tab with a tiny budget: a flooding peer's
        // messages are dropped above the rate, and the link keeps working.
        console.log('\nMesh: a flooding link is dropped at the rate, not routed');
        chans.length = 0;
        const got = [];
        const slow = new Mesh({ onMessage: (...a) => got.push(a),
                                rate: { bytes: [1e9, 1e9], msgs: [10, 5] } });
        slow.ws.emit('message', { data: JSON.stringify({
            ...welcome,
            peers: [{ id: 1, lo: null, hi: null, have: null, ready: false },
                    { id: 2, lo: 0, hi: 10, have: 10, ready: true }],
        }) });
        await slow.ready;
        const tok = pack({ k: 'tok', s: 'x-1', to: 1, token: 3 });
        for (let i = 0; i < 1000; i++) chans[0].emit('message', { data: tok.slice(0) });
        ok(`1000 messages at once deliver about the burst (${got.length})`, got.length >= 5 && got.length <= 6);
        eq('and the rest are counted as dropped', slow.links.get(2).dropped, 1000 - got.length);
        await new Promise((res) => setTimeout(res, 250));
        chans[0].emit('message', { data: tok.slice(0) });
        eq('once the budget refills, the link delivers again', got.length >= 6 && got.at(-1)[0].token, 3);

        // A string, or a header that is not an object, is not routed.
        const before = got.length;
        chans[0].emit('message', { data: 'hello' });
        chans[0].emit('message', { data: pack(7) });
        eq('garbage on the channel is dropped, not thrown', got.length, before);
        slow.stop();

        // Bytes are charged per frame, before reassembly.
        chans.length = 0;
        const got2 = [];
        const thin = new Mesh({ onMessage: (...a) => got2.push(a),
                                rate: { bytes: [1, 4096], msgs: [1e9, 1e9] } });
        thin.ws.emit('message', { data: JSON.stringify({
            ...welcome,
            peers: [{ id: 1, lo: null, hi: null, have: null, ready: false },
                    { id: 2, lo: 0, hi: 10, have: 10, ready: true }],
        }) });
        await thin.ready;
        chans[0].emit('message', { data: pack(job, new Float32Array(2048)) });
        eq('a message over the byte budget is dropped', got2.length, 0);
        chans[0].emit('message', { data: pack(job, new Float32Array(2)) });
        eq('while a small one still fits', got2.length, 1);
        thin.stop();
        delete globalThis.RTCPeerConnection;
    }

    console.log('\nreassemble: a claimed fragment count buys nothing');
    {
        // count = 2^32 - 1 with endless 256 KiB fragments: refused at
        // MAX_MESSAGE, not buffered.
        const st = {};
        const frame = (i) => {
            const b = new ArrayBuffer(WIRE_DEFAULT);
            const dv = new DataView(b);
            dv.setUint32(0, 0xFFFFFFFF, true);
            dv.setUint32(4, 1, true);
            dv.setUint32(8, i, true);
            dv.setUint32(12, 0xFFFFFFFF, true);
            return b;
        };
        let threw = null, i = 0;
        try { for (; i < 1e6; i++) reassemble(st, frame(i)); } catch (e) { threw = e; }
        ok('an endless message throws', threw !== null);
        ok(`at MAX_MESSAGE, not later (${i} fragments)`,
            i * (WIRE_DEFAULT - FRAG_HEAD) <= MAX_MESSAGE + WIRE_DEFAULT);
        eq('and holds nothing afterwards', st.parts, null);
    }

    console.log('\nbucket: a token bucket');
    {
        const take = bucket(10, 3, 0);
        eq('starts full', [take(1, 0), take(1, 0), take(1, 0), take(1, 0)], [true, true, true, false]);
        eq('refills at the rate', [take(1, 100), take(1, 100)], [true, false]);
        eq('never past the burst', [take(3, 10_000), take(1, 10_000)], [true, false]);
        eq('a charge bigger than the burst never passes', take(4, 1e9), false);
    }

    console.log('\nMesh.claim: states a capacity, resolves with the range');
    {
        const m = new Mesh({});
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        await m.ready;

        const p = m.claim(12);
        eq('the capacity goes out on the wire', m.ws.sent.at(-1), { type: 'capacity', cap: 12 });
        m.ws.emit('message', { data: JSON.stringify({ type: 'assign', lo: 10, hi: 22 }) });
        eq('and the range comes back', await p, [10, 22]);
        eq('and is remembered', m.range, [10, 22]);
    }
    {
        const m = new Mesh({});
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        await m.ready;

        const p = m.claim(3);
        m.ws.emit('message', { data: JSON.stringify({ type: 'assign', lo: null, hi: null }) });
        eq('a capacity below the floor resolves to no range at all', await p, null);
    }

    console.log('\nMesh.grow: widens an already-held range, fire-and-forget');
    {
        const m = new Mesh({});
        m.ws.emit('message', { data: JSON.stringify(welcome) });
        await m.ready;

        const p = m.claim(10);
        m.ws.emit('message', { data: JSON.stringify({ type: 'assign', lo: 5, hi: 15 }) });
        await p;

        // Unlike claim(), grow() awaits nothing. The server accepts a widening
        // request from a ready peer and sends no reply.
        m.grow(4, 16);
        eq('the wider range goes out on the wire', m.ws.sent.at(-1),
            { type: 'grow', lo: 4, hi: 16 });
        eq('and is remembered locally, the same way claim() remembers assign()',
            m.range, [4, 16]);
        m.stop();
    }

    console.log('\nMesh: the ready gate: a peer counts once it can pass a job on');
    {
        // A roster over a ten-position line. The peer under test has the
        // highest id, so it never offers a link (the lower id offers, and
        // offering needs an RTCPeerConnection). Its links are set directly to
        // 'connecting', then 'open', and a roster message makes it re-check,
        // as a link opening does in a browser.
        const TEN = { type: 'welcome', blocks: 8, positions: 10, minBlocks: 1 };
        const R = (id, lo, hi, ready = true) => ({ id, lo, hi, have: hi, ready });
        async function gated(range, peers, { open = [], linkWait = 60_000 } = {}) {
            const m = new Mesh({ linkWait });
            m.ws.emit('message', { data: JSON.stringify({ ...TEN, id: 99, peers }) });
            await m.ready;
            const placed = m.claim(range[1] - range[0]);
            m.ws.emit('message', { data: JSON.stringify({ type: 'assign', lo: range[0], hi: range[1] }) });
            await placed;
            for (const p of peers) {
                m.links.set(p.id, { pc: {}, ch: { readyState: open.includes(p.id) ? 'open' : 'connecting' } });
            }
            return m;
        }
        const told = (m) => m.ws.sent.filter((x) => x.type === 'have' || x.type === 'ready')
            .map((x) => (x.type === 'have' ? `have ${x.have}` : 'ready'));
        const linkOpens = (m, id, peers) => {
            m.links.get(id).ch.readyState = 'open';
            m.ws.emit('message', { data: JSON.stringify({ type: 'roster', peers: [...peers, R(99, 0, 0, false)] }) });
        };

        // The hub and the peer below are the same peer here.
        {
            const peers = [R(1, 0, 3)];
            const m = await gated([3, 6], peers);
            m.announceHave(4);
            m.announceReady();
            eq('nothing is announced while the only way in is still linking', told(m), []);
            eq('and the peer it waits on is named', m.linking, [1]);
            linkOpens(m, 1, peers);
            eq('the link opening lets it out, holding first and then readiness',
                told(m), ['have 4', 'ready']);
            eq('and it is no longer waiting on anyone', m.linking, null);
            m.stop();
        }

        // In the middle, with two holders on each side: one open link per side
        // is enough.
        {
            const peers = [R(1, 0, 3), R(2, 0, 3), R(3, 6, 10), R(4, 6, 10)];
            const m = await gated([3, 6], peers, { open: [2, 4] });
            m.announceHave(6);
            m.announceReady();
            eq('one linked holder below and one above is enough, whoever they are',
                told(m), ['have 6', 'ready']);
            m.stop();
        }
        {
            const peers = [R(1, 0, 3), R(3, 6, 10)];
            const m = await gated([3, 6], peers, { open: [1] });
            m.announceHave(6);
            m.announceReady();
            eq('but a side with no open link holds back what needs it: and only that',
                told(m), ['have 5', 'ready']);
            eq('naming that side', m.linking, [3]);
            m.stop();
        }

        // No ready peer to link to: nothing to wait for.
        {
            const m = await gated([3, 6], [R(1, 0, 3, false)]);
            m.announceHave(4);
            m.announceReady();
            eq('with no ready neighbours it announces at once', told(m), ['have 4', 'ready']);
            m.stop();
        }
        {
            const m = await gated([0, 4], []);
            m.announceHave(1);
            m.announceReady();
            eq('as does the first peer of a mesh', told(m), ['have 1', 'ready']);
            m.stop();
        }

        // The holding is announced in steps: the reachable part is sent and
        // the rest waits.
        {
            const peers = [R(1, 0, 3), R(3, 6, 10)];
            const m = await gated([3, 6], peers, { open: [1] });
            m.announceHave(4);
            m.announceHave(5);
            m.announceHave(6);
            m.announceReady();
            eq('the steps with nobody above go out; the one handing to an unlinked peer waits',
                told(m), ['have 4', 'have 5', 'ready']);
            linkOpens(m, 3, peers);
            eq('and goes out when its link opens', told(m), ['have 4', 'have 5', 'ready', 'have 6']);

            m.announceHave(5);
            eq('a lower holding asked for later is ignored, as ever', told(m).length, 4);
            m.stop();
        }

        // `ready` is not sent before a requested holding.
        {
            const peers = [R(1, 0, 3)];
            const m = await gated([3, 6], peers);
            m.announceHave(4);
            m.announceReady();
            eq('readiness waits with the holding it would otherwise credit wrongly', told(m), []);
            m.stop();
        }
        {
            // A peer that requests no holding is checked for its whole claim.
            // Here that means the peer above the claim.
            const peers = [R(1, 0, 3), R(3, 6, 10)];
            const m = await gated([3, 6], peers, { open: [1] });
            m.announceReady();
            eq('readiness with no holding is checked against the whole claim', told(m), []);
            linkOpens(m, 3, peers);
            eq('and goes out when that is linked', told(m), ['ready']);
            m.stop();
        }

        // A link that never opens: the announcement is sent anyway, with a
        // warning.
        {
            const warned = [];
            const warn = console.warn;
            console.warn = (...a) => warned.push(a.join(' '));
            const m = await gated([3, 6], [R(1, 0, 3)], { linkWait: 40 });
            m.announceHave(4);
            m.announceReady();
            eq('held at first', told(m), []);
            await new Promise((r) => setTimeout(r, 90));
            console.warn = warn;
            eq('announced anyway once the wait is up', told(m), ['have 4', 'ready']);
            ok('with a warning naming who it could not reach',
                warned.some((w) => /no open link to peer 1/.test(w)), JSON.stringify(warned));
            m.stop();
        }

        // A reconnect: every link is new, so the announcements wait for links
        // again.
        {
            const peers = [R(1, 0, 3)];
            const m = await gated([3, 6], peers, { open: [1] });
            m.announceHave(4);
            m.announceReady();
            eq('announced while linked', told(m), ['have 4', 'ready']);
            const first = m.ws;
            first.readyState = 3;
            m.ws.emit('close', { code: 1006 });
            await new Promise((r) => setTimeout(r, 1200));
            ok('it reconnects', m.ws !== first);
            m.links.get(1).ch.readyState = 'connecting';
            m.ws.emit('message', { data: JSON.stringify({ ...TEN, id: 99, peers }) });
            eq('it claims its range back at once', m.ws.sent.filter((x) => x.type === 'capacity').length, 1);
            eq('but re-announces nothing until the link to the hub is open again', told(m), []);
            linkOpens(m, 1, peers);
            eq('and then the holding and readiness, as before', told(m), ['have 4', 'ready']);
            m.stop();
        }

        // A reconnect after growth. The server resets `have` to `lo` on a
        // reclaim, so the peer must re-announce its grown holding. Announcing
        // only the top of its initial claim would leave the grown positions
        // unserved.
        {
            const m = await gated([4, 5], []);
            m.announceHave(5);
            m.announceReady();
            m.grow(3, 5);
            m.grow(3, 6);
            m.grow(0, 10);
            eq('growth is announced as growth, not as a holding', told(m), ['have 5', 'ready']);
            const first = m.ws;
            first.readyState = 3;
            m.ws.emit('close', { code: 1006 });
            await new Promise((r) => setTimeout(r, 1200));
            ok('it reconnects', m.ws !== first);
            m.ws.emit('message', { data: JSON.stringify({ ...TEN, id: 99, peers: [] }) });
            eq('it claims the grown range back',
                m.ws.sent.find((x) => x.type === 'capacity'), { type: 'capacity', cap: 10, lo: 0, hi: 10 });
            eq('and re-states all of it as held', told(m), ['have 10', 'ready']);
            m.stop();
        }
    }

    globalThis.WebSocket = class { addEventListener() {} send() {} close() {} readyState = 1; };  // restore
}

console.log('\nfetchRetry: survives a transient failure, gives up on a real one');
{
    const { fetchRetry } = await import('../web/model/loader.js');
    const realFetch = globalThis.fetch;
    const opts = { retries: 3, delayMs: 1 };   // 1ms, to keep the test fast

    // A thrown fetch (DNS failure, connection reset, "NetworkError when
    // attempting to fetch resource") succeeds once the network recovers, and
    // the caller does not see the earlier failures.
    {
        let calls = 0;
        globalThis.fetch = async () => {
            calls++;
            if (calls < 3) throw new Error('NetworkError when attempting to fetch resource');
            return { ok: true, status: 200 };
        };
        const r = await fetchRetry('https://example.test/x', opts);
        ok('recovers from two thrown failures then succeeds', r.ok && calls === 3);
    }

    // A 5xx is treated like a thrown error.
    {
        let calls = 0;
        globalThis.fetch = async () => {
            calls++;
            return calls < 2 ? { ok: false, status: 503 } : { ok: true, status: 200 };
        };
        const r = await fetchRetry('https://example.test/x', opts);
        ok('retries a 5xx', r.ok && calls === 2);
    }

    // A 4xx means the request is wrong. It is returned immediately and not
    // retried.
    {
        let calls = 0;
        globalThis.fetch = async () => { calls++; return { ok: false, status: 404 }; };
        const r = await fetchRetry('https://example.test/x', opts);
        ok('does not retry a 4xx', !r.ok && r.status === 404 && calls === 1,
            `called ${calls} times`);
    }

    // A persistent failure must throw after the last retry.
    {
        let calls = 0;
        globalThis.fetch = async () => { calls++; throw new Error('still down'); };
        let threw = null;
        try { await fetchRetry('https://example.test/x', opts); } catch (e) { threw = e; }
        ok('gives up after retries+1 attempts and throws the last error',
            threw?.message === 'still down' && calls === opts.retries + 1,
            `called ${calls} times, threw ${threw}`);
    }

    globalThis.fetch = realFetch;
}

// --- Signaling server, over real WebSockets ---

console.log('\nsignal-server: placement, roster and readiness');

process.env.PORT = '8099';
process.env.N_LAYER = '30';
// Set the floor as well as the depth. Otherwise the server derives it from the
// checkpoint `checkpoint.json` names, and this suite's result would depend on
// the deployed checkpoint (smaller blocks give a lower floor).
process.env.MIN_BLOCKS = '10';
// Small enough for the rooms section below to reach both caps and the join
// timeout quickly. The global mesh here never holds more than a dozen peers.
process.env.MAX_ROOMS = '3';
process.env.MAX_PEERS_PER_ROOM = '20';
process.env.JOIN_TIMEOUT_MS = '400';
// Our own TURN server, so every welcome carries a per-peer credential.
process.env.TURN_SECRET = 'testsecret';
process.env.TURN_URLS = 'turn:turn.example:3478?transport=udp, turn:turn.example:3478?transport=tcp';
// Every server log line, so the rooms section can check that none contains a
// token.
const logged = [];
{
    const log = console.log;
    console.log = (...a) => { logged.push(a.join(' ')); log(...a); };
}
const { coturnCredential } = await import('../signal-server.js');
await new Promise((r) => setTimeout(r, 200));

// `room` is the join token to send first (null: the global mesh), or `false`
// to send nothing. The server sends nothing to a socket until it has joined.
function connect(room = null) {
    const ws = new WS('ws://localhost:8099');
    const msgs = [];
    const waiters = [];
    // The close code, or 'open' if the server has not closed the socket
    // within 2 s. A server that fails to refuse then fails an assertion and
    // does not hang the suite.
    const closed = new Promise((r) => {
        const t = setTimeout(() => r('open'), 2000);
        ws.on('close', (code) => { clearTimeout(t); r(code); });
    });
    if (room !== false) ws.on('open', () => ws.send(JSON.stringify({ type: 'join', v: 1, room })));
    ws.on('message', (d) => {
        const m = JSON.parse(d);
        msgs.push(m);
        for (let i = waiters.length - 1; i >= 0; i--) {
            if (waiters[i].match(m)) waiters.splice(i, 1)[0].resolve(m);
        }
    });
    return {
        msgs,
        next: (match) => new Promise((resolve, reject) => {
            const hit = msgs.find(match);
            if (hit) return resolve(hit);
            const t = setTimeout(() => reject(new Error('timed out waiting for a message')), 3000);
            waiters.push({ match, resolve: (m) => { clearTimeout(t); resolve(m); } });
        }),
        send: (m) => ws.send(JSON.stringify(m)),
        close: () => ws.close(),
        closed,
        ws,
    };
}

const isAssign = (m) => m.type === 'assign';
const range = (m) => [m.lo, m.hi];

// A 30-block model has 32 positions: the embedding, blocks 0–29, and the head.
// Peers are placed on that line, so a claim of ten at the start is the
// embedding plus blocks 0–8, one block fewer than its width. `wte` is
// budgeted as a position.
const a = connect();
const wa = await a.next((m) => m.type === 'welcome');
eq('the welcome carries the model depth, not a slot', wa.blocks, 30);
eq('and the line peers are actually placed on', wa.positions, 32);
eq('and the floor a peer has to clear', wa.minBlocks, 10);
eq('a peer holds nothing until it says what it can hold',
    [wa.peers.find((p) => p.id === wa.id).lo, wa.peers.find((p) => p.id === wa.id).ready],
    [null, false]);

a.send({ type: 'capacity', cap: 10 });
eq('the first peer takes the embedding and the blocks behind it',
    range(await a.next(isAssign)), [0, 10]);

const b = connect();
const wb = await b.next((m) => m.type === 'welcome');
b.send({ type: 'capacity', cap: 10 });
eq('the second is placed after it, even though the first is still unready',
    range(await b.next(isAssign)), [10, 20]);

// Twelve, not ten: the line is two longer than the model, and the last peer
// must reach the head or nothing can sample.
const c = connect();
const wc = await c.next((m) => m.type === 'welcome');
c.send({ type: 'capacity', cap: 12 });
eq('and the third covers the rest, head included',
    range(await c.next(isAssign)), [20, 32]);

// A peer under the floor is a client. It is in the roster and is linked, but
// holds nothing.
const spectator = connect();
await spectator.next((m) => m.type === 'welcome');
spectator.send({ type: 'capacity', cap: 4 });
eq('a peer below the floor holds nothing', range(await spectator.next(isAssign)), [null, null]);

// The fourth capable peer has no uncovered position to take, so it overlaps an
// existing range and raises r.
const d = connect();
await d.next((m) => m.type === 'welcome');
d.send({ type: 'capacity', cap: 10 });
{
    const r = range(await d.next(isAssign));
    ok('a fourth peer on a covered network doubles up to raise r',
        r[1] - r[0] === 10 && r[0] >= 0 && r[1] <= 32, `got ${JSON.stringify(r)}`);
}

// A bigger peer takes more positions. Ranges need not be equal.
const big = connect();
await big.next((m) => m.type === 'welcome');
big.send({ type: 'capacity', cap: 25 });
{
    const r = range(await big.next(isAssign));
    eq('a 25-position peer gets 25 positions', r[1] - r[0], 25);
}

// Readiness is broadcast to the other peers.
a.send({ type: 'ready' });
const seen = await b.next((m) => m.type === 'roster' && m.peers.some((p) => p.id === wa.id && p.ready));
ok('readiness reaches the other peers', true);
eq('and only that peer is ready', seen.peers.filter((p) => p.ready).map((p) => p.id), [wa.id]);
eq('the roster carries ranges, not slots',
    seen.peers.find((p) => p.id === wa.id).lo, 0);

// `ready` is never unset: a repeat, or a message trying to clear it, does
// nothing.
b.msgs.length = 0;
a.send({ type: 'ready' });
a.send({ type: 'ready', ready: false });
await new Promise((r) => setTimeout(r, 300));
eq('a repeat announcement does not re-broadcast',
    b.msgs.filter((m) => m.type === 'roster').length, 0);

// Other peers route on a ready peer's range, so it cannot be re-placed.
a.msgs.length = 0;
a.send({ type: 'capacity', cap: 30 });
await new Promise((r) => setTimeout(r, 300));
eq('a ready peer cannot be re-placed', a.msgs.filter(isAssign).length, 0);

// An unready peer can be re-placed. A tab that failed to load uses this to
// give its blocks back.
d.msgs.length = 0;                       // otherwise `next` would return its first assign
d.send({ type: 'capacity', cap: 0 });
eq('an unready peer can hand its range back', range(await d.next(isAssign)), [null, null]);

// Restores a[0,10) b[10,20) c[20,32), so the hole below is the only one.
big.msgs.length = 0;
big.send({ type: 'capacity', cap: 0 });
await big.next(isAssign);

// A departure leaves a hole and moves nobody. The next peer fills it.
b.close();
await new Promise((r) => setTimeout(r, 300));
const e = connect();
const we = await e.next((m) => m.type === 'welcome');
e.msgs.length = 0;
e.send({ type: 'capacity', cap: 10 });
eq("a departed peer's positions are the first place the next peer goes",
    range(await e.next(isAssign)), [10, 20]);
{
    const seen = (await e.next((m) => m.type === 'roster' && m.peers.some((p) => p.id === we.id && p.lo === 10)))
        .peers.find((p) => p.id === we.id);
    eq('the roster shows it holding them, unready', [seen.lo, seen.hi, seen.ready], [10, 20, false]);
}

// --- `have`: the holding, relayed ---
//
// Placement counts the claim, and routing counts the holding. The server does
// not infer the holding, since only the peer knows what has loaded. It relays
// the peer's value, clamped to the claim and never decreasing. A `have` that
// fell would break every session routed through those positions.
console.log('\nsignal-server: have, the holding inside the claim');
{
    const g = connect();
    const wg = await g.next((m) => m.type === 'welcome');
    g.send({ type: 'capacity', cap: 10 });
    // `have` is a position on the line, not an offset into the claim, so the
    // numbers here are relative to where this peer was placed. That depends on
    // the peers earlier sections left on the server.
    const gc = await g.next(isAssign);
    const mine = (m) => m.peers.find((p) => p.id === wg.id);

    g.send({ type: 'have', have: gc.lo + 3 });
    eq('a partial holding reaches the roster',
        mine(await g.next((m) => m.type === 'roster' && mine(m)?.have === gc.lo + 3)).have,
        gc.lo + 3);

    g.msgs.length = 0;
    g.send({ type: 'have', have: gc.lo + 2 });
    await new Promise((r) => setTimeout(r, 200));
    eq('a holding never falls', g.msgs.filter((m) => m.type === 'roster').length, 0);

    g.send({ type: 'have', have: gc.hi + 99 });
    eq('and never exceeds the claim',
        mine(await g.next((m) => m.type === 'roster' && mine(m)?.have === gc.hi)).have, gc.hi);

    // A peer that never sends `have` is counted as holding its whole claim
    // when it sends `ready`.
    const h = connect();
    const wh = await h.next((m) => m.type === 'welcome');
    h.send({ type: 'capacity', cap: 10 });
    const asg = await h.next(isAssign);
    h.send({ type: 'ready' });
    const his = (m) => m.peers.find((p) => p.id === wh.id);
    eq('a peer that never states one is credited with the whole claim at ready',
        his(await h.next((m) => m.type === 'roster' && his(m)?.ready)).have, asg.hi);

    // Giving back the unfilled tail is the only way a ready peer's range
    // shrinks, and only to a boundary it has announced.
    const k = connect();
    const wk = await k.next((m) => m.type === 'welcome');
    k.send({ type: 'capacity', cap: 12 });
    const claim = await k.next(isAssign);
    k.send({ type: 'have', have: claim.lo + 4 });
    k.send({ type: 'ready' });
    const kis = (m) => m.peers.find((p) => p.id === wk.id);
    await k.next((m) => m.type === 'roster' && kis(m)?.ready);

    k.send({ type: 'shrink' });
    {
        const me = kis(await k.next((m) => m.type === 'roster' && kis(m)?.hi === claim.lo + 4));
        eq('a peer that stopped growing gives back what it never downloaded',
            [me.lo, me.hi, me.have], [claim.lo, claim.lo + 4, claim.lo + 4]);
    }

    // With nothing left to give, a second shrink does nothing.
    k.msgs.length = 0;
    k.send({ type: 'shrink' });
    await new Promise((r) => setTimeout(r, 200));
    eq('and shrinking a full range changes nothing',
        k.msgs.filter((m) => m.type === 'roster').length, 0);

    g.close(); h.close(); k.close();
}

console.log('\nsignal-server: grow, a claim widening in either direction');
{
    // An explicit range (holdClaim(), tested below), not a placed one, so it
    // has room to grow on both edges wherever placement would have put it on
    // the shared server.
    const g = connect();
    const wg = await g.next((m) => m.type === 'welcome');
    g.send({ type: 'capacity', cap: 10, lo: 10, hi: 20 });
    const claim = await g.next(isAssign);
    eq('lands exactly where asked, with room on both sides', [claim.lo, claim.hi], [10, 20]);
    g.send({ type: 'ready' });
    const mine = (m) => m.peers.find((p) => p.id === wg.id);
    await g.next((m) => m.type === 'roster' && mine(m)?.ready);

    // Growing right: `have` rises with `hi`, since the new position is already
    // served when the message is sent. The reclaim path of `capacity` resets
    // `have`; this does not.
    g.msgs.length = 0;
    g.send({ type: 'grow', lo: claim.lo, hi: claim.hi + 1 });
    {
        const me = mine(await g.next((m) => m.type === 'roster' && mine(m)?.hi === claim.hi + 1));
        eq('growing right widens hi and have together',
            [me.lo, me.hi, me.have], [claim.lo, claim.hi + 1, claim.hi + 1]);
    }

    // Growing left: `have` is unchanged. It marks the right edge, which a new
    // low position does not move.
    g.msgs.length = 0;
    g.send({ type: 'grow', lo: claim.lo - 1, hi: claim.hi + 1 });
    {
        const me = mine(await g.next((m) => m.type === 'roster' && mine(m)?.lo === claim.lo - 1));
        eq('growing left widens lo and leaves have alone',
            [me.lo, me.hi, me.have], [claim.lo - 1, claim.hi + 1, claim.hi + 1]);
    }

    // Shrinking either edge through `grow` is refused. `shrink` lowers `hi`,
    // and `lo` never moves inward.
    g.msgs.length = 0;
    g.send({ type: 'grow', lo: claim.lo, hi: claim.hi });
    await new Promise((r) => setTimeout(r, 200));
    eq('a grow that would shrink either edge changes nothing',
        g.msgs.filter((m) => m.type === 'roster').length, 0);

    // A range past the model's bounds is refused too.
    g.msgs.length = 0;
    g.send({ type: 'grow', lo: claim.lo - 1, hi: wa.positions + 1 });
    await new Promise((r) => setTimeout(r, 200));
    eq('growing past the model bounds changes nothing',
        g.msgs.filter((m) => m.type === 'roster').length, 0);

    g.close();
}

console.log('\nsignal-server: grow is refused before a peer is placed or ready');
{
    // Not ready: `grow` is ignored, as `have` must not be credited for a
    // position the peer has not announced it can serve.
    const p = connect();
    await p.next((m) => m.type === 'welcome');
    p.send({ type: 'capacity', cap: 10 });
    const claim = await p.next(isAssign);
    p.msgs.length = 0;
    p.send({ type: 'grow', lo: claim.lo, hi: claim.hi + 1 });
    await new Promise((r) => setTimeout(r, 200));
    eq('a grow before ready is ignored', p.msgs.filter((m) => m.type === 'roster').length, 0);
    p.close();

    // No range: a spectator has nothing to grow.
    const s = connect();
    await s.next((m) => m.type === 'welcome');
    s.msgs.length = 0;
    s.send({ type: 'grow', lo: 0, hi: 1 });
    await new Promise((r) => setTimeout(r, 200));
    eq('a grow with no claim at all is ignored', s.msgs.filter((m) => m.type === 'roster').length, 0);
    s.close();
}

// --- An explicit claim ---
//
// `holdClaim` is used by a reconnecting peer to keep the range in its VRAM,
// and by `?at=lo-hi` in peer.js on a first join, to build arrangements
// placement would not produce.
console.log('\nsignal-server: a claim that names its own range');
{
    const m1 = connect();
    await m1.next((w) => w.type === 'welcome');
    m1.send({ type: 'capacity', cap: 6, lo: 4, hi: 10 });
    eq('a coherent explicit range is honoured exactly',
        range(await m1.next(isAssign)), [4, 10]);

    // Granted below the floor: `holdClaim` runs before `place()` and does not
    // check MIN_BLOCKS. Six is under the ten this suite sets, so a claim that
    // placement would refuse is granted as an explicit one. The floor exists
    // to stop ranges too small to be worth their hops, which does not apply to
    // a peer reclaiming weights already in VRAM. It also lets `?at=` build a
    // mesh the server would not place.
    ok(`an explicit range below the floor of ${wa.minBlocks} is still granted`,
        10 - 4 < wa.minBlocks, `floor is ${wa.minBlocks}, so 6 was not below it`);

    // Every refused range falls back to placing the peer, with no error. So
    // peer.js validates before asking and checks the reply, to catch a typo
    // that would otherwise be placed elsewhere. The cap is above the floor
    // here, so the fallback is a real placement. Below it, each case would
    // become a client and the assertion would pass without testing the
    // refusal.
    const bad = [
        ['a width that disagrees with cap', { cap: 12, lo: 4, hi: 9 }],
        ['a range past the end of the line', { cap: 12, lo: 24, hi: 36 }],
        ['a negative bound', { cap: 12, lo: -2, hi: 10 }],
        ['an empty range', { cap: 12, lo: 5, hi: 5 }],
        ['a non-integer bound', { cap: 12, lo: 'x', hi: 12 }],
    ];
    for (const [what, msg] of bad) {
        const c2 = connect();
        await c2.next((w) => w.type === 'welcome');
        c2.send({ type: 'capacity', ...msg });
        const got = range(await c2.next(isAssign));
        ok(`${what} is placed instead, never honoured`,
            got[0] != null && got[1] - got[0] === msg.cap
                && JSON.stringify(got) !== JSON.stringify([msg.lo, msg.hi]),
            `got ${JSON.stringify(got)}`);
        c2.close();
    }
    m1.close();
    await new Promise((r) => setTimeout(r, 200));
}

for (const p of [a, c, d, e, big, spectator]) p.close();
await new Promise((r) => setTimeout(r, 200));

// --- Private meshes: one Room per invite, isolated from each other ---

console.log('\nsignal-server: joining a mesh');
{
    const silent = connect(false);
    await new Promise((r) => setTimeout(r, 100));
    eq('nothing is sent to a socket before it joins', silent.msgs, []);
    eq('and one that never joins is closed', await silent.closed, 4001);

    const rude = connect(false);
    rude.ws.on('open', () => rude.send({ type: 'capacity', cap: 10 }));
    eq('a first message that is not a join is refused', await rude.closed, 4001);

    for (const [what, token] of [['a short token', 'abc'], ['a token with a bad character', '!'.repeat(43)],
                                 ['a token that is not a string', 42]]) {
        const bad = connect(token);
        eq(`${what} is refused`, await bad.closed, 4001);
    }

    const g = connect(null);
    eq('the global mesh says it is not private', (await g.next((m) => m.type === 'welcome')).private, false);
    g.close();
}

console.log('\nsignal-server: TURN credentials');
{
    // Checked against coturn's scheme, computed separately:
    //   echo -n 1000086400:7 | openssl dgst -binary -sha1 -hmac testsecret | base64
    eq('a coturn credential is <expiry>:<name> and base64 HMAC-SHA1 of it',
        coturnCredential('testsecret', ['turn:t'], 7, 86400, 1_000_000_000_000),
        { urls: ['turn:t'], username: '1000086400:7', credential: 'Oq7gSfU8RRpcOjBnarhF65iMHwo=' });

    const a = connect(null), b = connect(null);
    const [wa, wb] = await Promise.all([a, b].map((p) => p.next((m) => m.type === 'welcome')));
    const turnOf = (w) => w.ice.find((s) => [].concat(s.urls).some((u) => u.startsWith('turn:')));
    const ta = turnOf(wa), tb = turnOf(wb);
    eq('the welcome keeps STUN first', wa.ice[0].urls, 'stun:stun.l.google.com:19302');
    eq('and brings the TURN_URLS', ta?.urls,
        ['turn:turn.example:3478?transport=udp', 'turn:turn.example:3478?transport=tcp']);
    ok('with a credential named for this peer', ta?.username.endsWith(`:${wa.id}`), ta?.username);
    ok('so no two peers share one', ta && tb && ta.username !== tb.username && ta.credential !== tb.credential);
    const expiry = Number(ta?.username.split(':')[0]);
    ok('which expires a day from now', Math.abs(expiry - (Date.now() / 1000 + 86400)) < 60, expiry);
    eq('and verifies against the secret', ta?.credential,
        createHmac('sha1', 'testsecret').update(ta?.username ?? '').digest('base64'));
    a.close();
    b.close();
}

console.log('\nsignal-server: rooms are disjoint');
{
    const tA = (await openRoom(newSecret())).joinToken;
    const tB = (await openRoom(newSecret())).joinToken;
    logged.length = 0;

    const g1 = connect(null), a1 = connect(tA), a2 = connect(tA), b1 = connect(tB);
    const [wg1, wa1, wa2, wb1] = await Promise.all(
        [g1, a1, a2, b1].map((p) => p.next((m) => m.type === 'welcome')));
    eq('a private room says so in the welcome', [wa1.private, wb1.private], [true, true]);

    const ids = [wg1.id, wa1.id, wa2.id, wb1.id];
    eq('ids are unique across rooms', new Set(ids).size, 4);

    // After everyone has joined, read each room's latest roster.
    await new Promise((r) => setTimeout(r, 100));
    const last = (p) => p.msgs.filter((m) => m.type === 'roster' || m.type === 'welcome').at(-1).peers
        .map((q) => q.id).sort((x, y) => x - y);
    eq('room A sees exactly its two peers', last(a1), [wa1.id, wa2.id].sort((x, y) => x - y));
    eq('and both of them see the same', last(a2), last(a1));
    eq('room B sees only its one', last(b1), [wb1.id]);
    ok('the global mesh sees none of them',
        !last(g1).some((id) => [wa1.id, wa2.id, wb1.id].includes(id)));

    // Placement is per room: each room starts as an empty model.
    a1.send({ type: 'capacity', cap: 10 });
    b1.send({ type: 'capacity', cap: 10 });
    eq('the first peer in room A is placed at the start of the line',
        range(await a1.next(isAssign)), [0, 10]);
    eq('and so is the first in room B, whatever A holds', range(await b1.next(isAssign)), [0, 10]);

    // A signal names its target by id. It is not delivered across rooms.
    const sdp = { type: 'offer', sdp: 'v=0' };
    b1.send({ type: 'signal', to: wa1.id, kind: 'offer', sdp, mac: 'x' });
    g1.send({ type: 'signal', to: wa2.id, kind: 'offer', sdp });
    a2.send({ type: 'signal', to: wa1.id, kind: 'offer', sdp, mac: 'y' });
    const relayed = await a1.next((m) => m.type === 'offer');
    eq('within a room an offer is relayed, MAC and all, from the right id',
        [relayed.from, relayed.mac], [wa2.id, 'y']);
    await new Promise((r) => setTimeout(r, 100));
    eq('from another room, nothing arrives', a1.msgs.filter((m) => m.type === 'offer').length, 1);
    eq('nor from the global mesh', a2.msgs.filter((m) => m.type === 'offer').length, 0);

    // `kind` becomes the relayed `type`, so it must not be a server message type.
    a2.send({ type: 'signal', to: wa1.id, kind: 'welcome', sdp });
    a2.send({ type: 'signal', to: wa1.id, kind: 'roster', peers: [] });
    await new Promise((r) => setTimeout(r, 100));
    eq('a peer cannot forge a welcome or roster through the relay',
        a1.msgs.filter((m) => m.type === 'welcome').length, 1);

    // A departure is broadcast to its own room only.
    const gBefore = g1.msgs.length, aBefore = a1.msgs.length;
    b1.close();
    await new Promise((r) => setTimeout(r, 150));
    eq('a leave in room B reaches no one in A', a1.msgs.length, aBefore);
    eq('or in the global mesh', g1.msgs.length, gBefore);

    // An empty room is deleted: a new join finds nobody.
    const b2 = connect(tB);
    const wb2 = await b2.next((m) => m.type === 'welcome');
    eq('a room emptied and joined again starts empty', wb2.peers.map((q) => q.id), [wb2.id]);
    b2.close();

    const text = logged.join('\n');
    ok('no join token is ever written to the log', !text.includes(tA) && !text.includes(tB));
    ok('rooms are logged by tag', /room [0-9a-f]{6} /.test(text));

    for (const p of [g1, a1, a2]) p.close();
    await new Promise((r) => setTimeout(r, 150));
}

console.log('\nsignal-server: limits');
{
    // MAX_PEERS_PER_ROOM is 20 here.
    const t = (await openRoom(newSecret())).joinToken;
    const crowd = Array.from({ length: 20 }, () => connect(t));
    await Promise.all(crowd.map((p) => p.next((m) => m.type === 'welcome')));
    const late = connect(t);
    eq('the peer past a room\'s cap is refused', await late.closed, 4002);
    const other = connect(null);
    ok('and the global mesh is untouched by it',
        (await other.next((m) => m.type === 'welcome')).private === false);
    for (const p of crowd) p.close();
    other.close();
    await new Promise((r) => setTimeout(r, 150));

    // MAX_ROOMS is 3 private rooms.
    const held = [];
    for (let i = 0; i < 3; i++) {
        const p = connect((await openRoom(newSecret())).joinToken);
        await p.next((m) => m.type === 'welcome');
        held.push(p);
    }
    const fourth = connect((await openRoom(newSecret())).joinToken);
    eq('a room past the server\'s cap is refused', await fourth.closed, 4002);
    const existing = connect(null);
    ok('while the global mesh still takes joins',
        (await existing.next((m) => m.type === 'welcome')).private === false);
    for (const p of [...held, existing]) p.close();
    await new Promise((r) => setTimeout(r, 150));
}

// --- Mesh: which mesh it joins, and what it passes to WebRTC ---
//
// Uses a WebSocket the test controls and an RTCPeerConnection that records
// what it is given. The tests check what net.js does with incoming messages,
// not whether WebRTC works.

{
    class FakeWS extends EventTarget {
        static all = [];
        constructor(url) { super(); this.url = url; this.readyState = 0; this.sent = []; FakeWS.all.push(this); }
        send(d) { this.sent.push(JSON.parse(d)); }
        close(code = 1000) {
            if (this.readyState === 3) return;
            this.readyState = 3;
            this.dispatchEvent(Object.assign(new Event('close'), { code }));
        }
        open() { this.readyState = 1; this.dispatchEvent(new Event('open')); }
        deliver(m) { this.dispatchEvent(Object.assign(new Event('message'), { data: JSON.stringify(m) })); }
    }
    class FakePC extends EventTarget {
        static all = [];
        constructor(config) { super(); this.config = config; this.remote = []; this.iceGatheringState = 'complete';
                        this.connectionState = 'new'; FakePC.all.push(this); }
        createDataChannel() { return Object.assign(new EventTarget(), { readyState: 'connecting' }); }
        // Every kind of candidate, as a browser that ignored the policy would
        // gather them. Only the relay line may be sent.
        async createOffer() {
            return { type: 'offer', sdp: ['v=0 offer',
                'a=candidate:1 1 udp 2122260223 192.168.1.20 51000 typ host generation 0',
                'a=candidate:2 1 udp 1686052607 81.2.3.4 51000 typ srflx raddr 192.168.1.20 rport 51000',
                'a=candidate:3 1 udp 41885439 104.30.1.1 3478 typ relay raddr 81.2.3.4 rport 51000',
                'a=end-of-candidates', ''].join('\r\n') };
        }
        async createAnswer() { return { type: 'answer', sdp: 'v=0 answer' }; }
        async setLocalDescription(d) { this.localDescription = d; }
        async setRemoteDescription(d) { this.remote.push(d); }
        close() { this.connectionState = 'closed'; }
    }
    const was = { ws: globalThis.WebSocket, pc: globalThis.RTCPeerConnection };
    globalThis.WebSocket = FakeWS;
    globalThis.RTCPeerConnection = FakePC;
    const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));
    const welcome = (id, priv, peers = []) =>
        ({ type: 'welcome', id, blocks: 30, positions: 32, minBlocks: 10, private: priv, peers });

    console.log('\nMesh: join');
    const secret = newSecret();
    const room = await openRoom(secret);
    const twin = await openRoom(secret);
    const stranger = await openRoom(newSecret());

    const m = new Mesh({ room });
    let ws = FakeWS.all.at(-1);
    ws.open();
    eq('the first frame on the socket is the join, with the room\'s token',
        ws.sent[0], { type: 'join', v: 1, room: room.joinToken });
    ok('and the secret itself is never sent', !JSON.stringify(ws.sent).includes(secret));
    ws.deliver(welcome(5, true));
    await m.ready;
    ok('a private welcome for a private room is accepted', m.id === 5 && !m.refused);

    console.log('\nMesh: offers and answers carry the room\'s seal');
    const sdp = { type: 'offer', sdp: 'v=0 from 2' };
    ws.deliver({ type: 'offer', from: 2, sdp });
    await tick();
    eq('an unsealed offer never reaches the RTCPeerConnection',
        FakePC.all.flatMap((pc) => pc.remote).length, 0);
    eq('and is counted', m.authFailures, 1);

    ws.deliver({ type: 'offer', from: 2, ...(await stranger.seal({ kind: 'offer', from: 2, to: 5, sdp })) });
    await tick();
    eq('nor does one sealed by another mesh', FakePC.all.flatMap((pc) => pc.remote).length, 0);

    ws.deliver({ type: 'offer', from: 2, ...(await twin.seal({ kind: 'offer', from: 2, to: 9, sdp })) });
    await tick();
    eq('nor one sealed for a different recipient', FakePC.all.flatMap((pc) => pc.remote).length, 0);
    eq('three refused so far', m.authFailures, 3);

    ws.deliver({ type: 'offer', from: 2, ...(await twin.seal({ kind: 'offer', from: 2, to: 5, sdp })) });
    await tick();
    eq('an offer sealed by a peer with the same invite is taken',
        FakePC.all.flatMap((pc) => pc.remote), [sdp]);
    const reply = ws.sent.find((x) => x.type === 'signal' && x.kind === 'answer');
    ok('and the answer goes back sealed, verifiable by that peer',
        reply && await twin.verify({ kind: 'answer', from: 5, to: 2, sdp: reply.sdp, mac: reply.mac }));
    ok('and not by anyone else',
        reply && !(await stranger.verify({ kind: 'answer', from: 5, to: 2, sdp: reply.sdp, mac: reply.mac })));

    // This tab offers to a higher id: peer 9 holds the entry point.
    ws.deliver({ type: 'roster', peers: [{ id: 5, lo: null, hi: null, ready: false },
                                          { id: 9, lo: 0, hi: 32, have: 32, ready: true }] });
    await tick();
    const offer = ws.sent.find((x) => x.type === 'signal' && x.kind === 'offer' && x.to === 9);
    ok('an offer this tab makes is sealed',
        offer && await twin.verify({ kind: 'offer', from: 5, to: 9, sdp: offer.sdp, mac: offer.mac }));
    const pc9 = m.links.get(9)?.pc;
    const answer = { type: 'answer', sdp: 'v=0 from 9' };
    ws.deliver({ type: 'answer', from: 9, sdp: answer });
    await tick();
    eq('an unsealed answer is not applied', pc9.remote.length, 0);
    ws.deliver({ type: 'answer', from: 9, ...(await twin.seal({ kind: 'answer', from: 9, to: 5, sdp: answer })) });
    await tick();
    eq('a sealed one is', pc9.remote, [answer]);

    console.log('\nMesh: relay only');
    ok('every link asks the browser for relay candidates only',
        FakePC.all.length > 0 && FakePC.all.every((pc) => pc.config.iceTransportPolicy === 'relay'));
    ok('an offer leaves with no host or server-reflexive candidate',
        !/typ host|typ srflx|192\.168\.1\.20 51000 typ|81\.2\.3\.4 51000 typ/.test(offer.sdp.sdp), offer?.sdp?.sdp);
    ok('but keeps the relay candidate', / typ relay /.test(offer.sdp.sdp));
    ok('and the seal covers the SDP actually sent', await twin.verify(
        { kind: 'offer', from: 5, to: 9, sdp: offer.sdp, mac: offer.mac }));
    ok('the relay is noticed when the welcome brings TURN',
        hasTurn([{ urls: 'stun:x' }, { urls: ['turn:turn.cloudflare.com:3478?transport=udp', 'turns:t:443'] }]));
    ok('and its absence when it only brings STUN', !hasTurn([{ urls: 'stun:stun.l.google.com:19302' }]));
    ok('which the Mesh reports', m.relay === false && m.relayOnly === true);
    {
        const direct = new Mesh({ relayOnly: false });
        const s1 = FakeWS.all.at(-1);
        s1.open();
        s1.deliver({ ...welcome(3, false), peers: [{ id: 3, lo: null, hi: null, ready: false },
                                                   { id: 4, lo: 0, hi: 32, have: 32, ready: true }] });
        await tick();
        const o = s1.sent.find((x) => x.type === 'signal' && x.kind === 'offer');
        ok('?direct=1 lets the browser use every route', FakePC.all.at(-1).config.iceTransportPolicy === 'all');
        ok('and sends the SDP untouched', / typ host /.test(o?.sdp?.sdp ?? ''));
        direct.stop();
    }

    console.log('\nMesh: a reconnect joins the same mesh');
    const before = FakeWS.all.length;
    ws.close(1006);
    await tick(1100);                    // RECONNECT_MIN_MS
    ok('a dropped socket is reopened', FakeWS.all.length === before + 1);
    ws = FakeWS.all.at(-1);
    ws.open();
    eq('and its first frame is the same join', ws.sent[0], { type: 'join', v: 1, room: room.joinToken });
    m.stop();

    console.log('\nMesh: fails closed');
    {
        const p = new Mesh({ room });
        const s1 = FakeWS.all.at(-1);
        s1.open();
        s1.deliver(welcome(1, false));
        let why = null;
        await p.ready.catch((e) => { why = e.message; });
        ok('a private tab put in the global mesh does not join', /did not put this tab/.test(why ?? ''), `${why}`);
        ok('and says so', !!p.refused);
        const n = FakeWS.all.length;
        await tick(1200);
        eq('and does not reconnect', FakeWS.all.length, n);
    }
    {
        const p = new Mesh({});
        const s1 = FakeWS.all.at(-1);
        s1.open();
        eq('a global tab sends a join with no room', s1.sent[0], { type: 'join', v: 1, room: null });
        s1.deliver(welcome(1, true));
        let why = null;
        await p.ready.catch((e) => { why = e.message; });
        ok('and refuses a private welcome it did not ask for', /did not ask for/.test(why ?? ''), `${why}`);
    }
    {
        const p = new Mesh({ room });
        const s1 = FakeWS.all.at(-1);
        s1.open();
        s1.close(4002);
        let why = null;
        await p.ready.catch((e) => { why = e.message; });
        eq('a full mesh is reported', why, 'this mesh is full');
        const n = FakeWS.all.length;
        await tick(1200);
        eq('and not retried', FakeWS.all.length, n);
    }

    // A spectator's only link, to the one holder of position 0, fails ICE.
    // Only #sync re-offers, and in a quiet mesh no roster arrives to trigger
    // it, so the retry timer must.
    console.log('\nMesh: a failed link heals without a roster change');
    const hub = { id: 9, lo: 0, hi: 32, have: 32, ready: true };
    const fail = (pc) => { pc.connectionState = 'failed'; pc.dispatchEvent(new Event('connectionstatechange')); };
    {
        const p = new Mesh({ linkRetry: [50, 400], linkOpen: 60_000 });
        const s = FakeWS.all.at(-1);
        s.open();
        s.deliver(welcome(5, false, [{ id: 5, lo: null, hi: null, ready: false }, hub]));
        await p.ready;
        await tick();
        const offers = () => s.sent.filter((x) => x.type === 'signal' && x.kind === 'offer' && x.to === 9).length;
        eq('a spectator offers the hub a link', offers(), 1);

        const pc1 = p.links.get(9).pc;
        fail(pc1);
        await tick(20);
        eq('a failed link is not offered again at once', offers(), 1);
        await tick(60);
        eq('but after its backoff, with no roster change', offers(), 2);
        ok('on a new connection, the dead one closed',
            p.links.get(9).pc !== pc1 && pc1.connectionState === 'closed');

        fail(p.links.get(9).pc);
        await tick(70);
        eq('a second failure waits longer', offers(), 2);
        await tick(60);
        eq('twice as long', offers(), 3);

        const up = p.links.get(9);
        up.ch.readyState = 'open';
        up.ch.dispatchEvent(new Event('open'));
        fail(up.pc);
        await tick(80);
        eq('a link that opened starts the backoff over', offers(), 4);
        p.stop();
    }
    {
        // An offer or answer lost while the socket was down: no failure event
        // fires, and the link never opens.
        const p = new Mesh({ linkRetry: [30, 400], linkOpen: 50 });
        const s = FakeWS.all.at(-1);
        s.open();
        s.deliver(welcome(6, false, [{ id: 6, lo: null, hi: null, ready: false }, hub]));
        await p.ready;
        await tick();
        const offers = () => s.sent.filter((x) => x.type === 'signal' && x.kind === 'offer' && x.to === 9).length;
        await tick(110);
        eq('a link that never opens is offered again', offers(), 2);
        p.stop();
    }
    {
        // The higher id never offers. It drops the dead link and waits for the
        // lower id's new offer, which replaces it.
        const p = new Mesh({ linkRetry: [30, 400], linkOpen: 60_000 });
        const s = FakeWS.all.at(-1);
        s.open();
        s.deliver(welcome(12, false, [{ id: 12, lo: null, hi: null, ready: false }, hub]));
        await p.ready;
        s.deliver({ type: 'offer', from: 9, sdp: { type: 'offer', sdp: 'v=0 from 9' } });
        await tick();
        const pc1 = p.links.get(9)?.pc;
        ok('the higher id answers the hub\'s offer', !!pc1);
        fail(pc1);
        await tick(60);
        ok('and drops the failed link', !p.links.has(9) && pc1.connectionState === 'closed');
        eq('but does not offer one itself', s.sent.filter((x) => x.kind === 'offer').length, 0);
        s.deliver({ type: 'offer', from: 9, sdp: { type: 'offer', sdp: 'v=0 again' } });
        await tick();
        ok('the next offer from the hub is taken', !!p.links.get(9) && p.links.get(9).pc !== pc1);
        p.stop();
    }
    globalThis.WebSocket = was.ws;
    globalThis.RTCPeerConnection = was.pc;
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
