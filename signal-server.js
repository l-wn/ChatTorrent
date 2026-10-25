// Signaling for the peer mesh:
//
//   1. Give every peer an id and, once it states its capacity, a range of
//      positions placed by web/split.js.
//   2. Relay each peer's holding and readiness, and broadcast the roster.
//   3. Relay SDP between two peers of the same mesh.
//   4. Ping every peer, and drop one that stops answering.
//
// Each mesh (the global one, and one per private invite) is a Room
// (signal-room.js). The messages are listed in docs/architecture.md (Joining).
//
//   node signal-server.js        # ws://127.0.0.1:8080, proxied at /ws by Caddy

import { readFileSync } from 'node:fs';
import { createHash, createHmac } from 'node:crypto';
import { WebSocketServer } from 'ws';
import { costDims, floorBlocks, blockBytes, positions, headPos, SESSIONS, PROTOCOL,
         CONTEXTS, runnerCtx } from './web/split.js';
import { Room } from './signal-room.js';

const PORT = Number(process.env.PORT ?? 8080);

// Loopback only by default: in the deploy every socket arrives through Caddy.
// HOST=0.0.0.0 opens it to a LAN for a local mesh.
const HOST = process.env.HOST || '127.0.0.1';

// Pages allowed to open a socket, by Origin header, so another site cannot
// connect its visitors' browsers to the mesh. Any localhost port is allowed,
// for local runs and the test harness. A client with no Origin is not a
// browser (the Node tests) and could send any Origin, so it is allowed.
const ALLOWED_ORIGINS = new Set((process.env.ALLOWED_ORIGINS
    ?? 'https://clusterchat.io')
    .split(',').map((o) => o.trim()).filter(Boolean));
const LOCAL_ORIGIN = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/;
const originAllowed = (origin) => !origin || ALLOWED_ORIGINS.has(origin) || LOCAL_ORIGIN.test(origin);

// Maximum sockets per client address. The address is Cloudflare's
// CF-Connecting-IP, which exists only behind the tunnel. A local run has none
// and is not limited.
const MAX_CONN_PER_IP = Number(process.env.MAX_CONN_PER_IP) || 16;

// The largest message a peer sends is an SDP offer or answer, a few KiB.
const MAX_PAYLOAD = 64 * 1024;

// The server reads the same manifest the peers download, for the model's
// depth and the dimensions used to convert MIN_BYTES (split.js) into blocks.
//
// The checkpoint comes from ./checkpoint.json, the file loader.js also
// fetches, so the server and every page agree. WEIGHTS_URL overrides it.
const CHECKPOINT = './checkpoint.json';

function readCheckpoint() {
    try {
        const cfg = JSON.parse(readFileSync(CHECKPOINT, 'utf8'));
        if (typeof cfg.url !== 'string' || !cfg.url) {
            console.warn(`${CHECKPOINT} has no "url". Expected { "url": "/weights/<dir>" }`);
            return null;
        }
        return checkUrl(cfg.url.replace(/\/$/, ''), CHECKPOINT);
    } catch (e) {
        console.warn(`could not read ${CHECKPOINT}: ${e.message}`);
    }
    return null;
}

// The URL must be same-origin (`/weights/<dir>`) or absolute with a scheme. A
// bare host is rejected: the browser would read it as a relative path.
function checkUrl(url, where) {
    if (url.startsWith('/') || /^https?:\/\//.test(url)) return url;
    console.warn(`${where}: "${url}" has no scheme and is not a rooted path. Use ` +
        `"/weights/<dir>" for a checkpoint in this repo, or "https://<host>/<prefix>" ` +
        `for a remote one. The browser reads a bare host as a relative path.`);
    return null;
}

// `||`, not `??`: an empty `WEIGHTS_URL=` in .env must not override the file.
const WEIGHTS_URL = process.env.WEIGHTS_URL
    || readCheckpoint()
    || 'https://weights.clusterchat.io/smollm2';

// The manifest for that checkpoint: read from disk for a same-origin
// `/weights/<dir>` (Caddy serves the repo root), fetched otherwise. MANIFEST
// overrides both.
const MANIFEST = process.env.MANIFEST
    || (WEIGHTS_URL.startsWith('/') ? `.${WEIGHTS_URL}/manifest.json` : null);

async function readManifest() {
    if (MANIFEST) {
        try {
            return JSON.parse(readFileSync(MANIFEST, 'utf8'));
        } catch { /* not on disk; try the weight host */ }
    }
    try {
        const r = await fetch(`${WEIGHTS_URL}/manifest.json`);
        if (r.ok) return await r.json();
        console.warn(`${WEIGHTS_URL}/manifest.json: ${r.status}`);
    } catch (e) {
        console.warn(`could not fetch the manifest: ${e.message}`);
    }
    return null;
}

const manifest = await readManifest();
const dims = manifest && costDims(manifest);

// N_LAYER and MIN_BLOCKS override the manifest, for tests and for a server
// that cannot reach its manifest. A peer refuses to boot if its range does not
// fit its own manifest, so a wrong N_LAYER is reported immediately.
const BLOCKS = Number(process.env.N_LAYER ?? dims?.n_layer ?? 30);

// `BLOCKS` is the model's depth, which a peer checks against its manifest.
// `POSITIONS` is the length of the line that placement and coverage use.
const POSITIONS = positions(BLOCKS);
const HEAD = headPos(BLOCKS);
const MIN_BLOCKS = Number(process.env.MIN_BLOCKS)
    || (dims ? floorBlocks(dims, POSITIONS) : 10);

// STUN gives a peer its public address. TURN relays when two NATs cannot
// connect directly. Two kinds of TURN are supported, and the first one
// configured is used:
//
//   - Our own TURN server (coturn with `use-auth-secret`): TURN_SECRET is its
//     static-auth-secret, TURN_URLS its comma-separated turn: URLs. Each peer
//     gets its own credential, minted locally.
//   - Cloudflare Realtime TURN: TURN_KEY_ID and TURN_API_TOKEN. Credentials
//     are minted through Cloudflare's API and shared by every peer until
//     re-minted.
//
// With neither, peers get STUN only, which works on one LAN.
const TURN_SECRET = process.env.TURN_SECRET;
const TURN_URLS = (process.env.TURN_URLS ?? '').split(',').map((u) => u.trim()).filter(Boolean);
const OWN_TURN = Boolean(TURN_SECRET && TURN_URLS.length);
const TURN_KEY = process.env.TURN_KEY_ID;
const TURN_TOKEN = process.env.TURN_API_TOKEN;
const TURN_TTL = 86400;

const STUN = [{ urls: 'stun:stun.l.google.com:19302' }];
let cached = { ice: STUN, until: 0 };

// A coturn credential in the TURN REST API scheme (`use-auth-secret`): the
// username is `<expiry, unix seconds>:<name>` and the password is base64
// HMAC-SHA1(secret, username). coturn verifies both, so it needs no user list,
// and a credential stops working at its expiry. One per peer, named by its id,
// so coturn's log ties every relay allocation to a peer.
export function coturnCredential(secret, urls, name, ttl = TURN_TTL, now = Date.now()) {
    const username = `${Math.floor(now / 1000) + ttl}:${name}`;
    const credential = createHmac('sha1', secret).update(username).digest('base64');
    return { urls, username, credential };
}

async function iceServers(id) {
    if (OWN_TURN) return STUN.concat(coturnCredential(TURN_SECRET, TURN_URLS, id));
    if (!TURN_KEY || !TURN_TOKEN || Date.now() < cached.until) return cached.ice;
    try {
        const r = await fetch(
            `https://rtc.live.cloudflare.com/v1/turn/keys/${TURN_KEY}/credentials/generate-ice-servers`,
            {
                method: 'POST',
                headers: { authorization: `Bearer ${TURN_TOKEN}`, 'content-type': 'application/json' },
                body: JSON.stringify({ ttl: TURN_TTL }),
            });
        if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
        const { iceServers } = await r.json();
        // Re-minted at half the TTL, so a credential outlives any link opened
        // with it.
        cached = { ice: STUN.concat(iceServers), until: Date.now() + TURN_TTL * 500 };
        console.log('minted TURN credentials');
    } catch (e) {
        console.warn(`TURN unavailable, peers get STUN only: ${e.message}`);
        cached = { ice: STUN, until: Date.now() + 60000 };
    }
    return cached.ice;
}

// --- Rooms ---
//
// One Room (signal-room.js) per mesh: the global one, which always exists, and
// one per private invite, created on its first join and deleted when its last
// peer leaves. A private room is keyed by SHA-256 of its join token and logged
// under the first six hex characters of that hash. The token is not stored as
// a key or logged. web/room.js explains the token and why it cannot open a
// link by itself.

const MAX_ROOMS = Number(process.env.MAX_ROOMS) || 1000;
const MAX_PEERS_PER_ROOM = Number(process.env.MAX_PEERS_PER_ROOM) || 64;

// Replies that may generate at once: MAX_CHATS in the global mesh, and in a
// private mesh the limit its creator chose, capped by MAX_PRIVATE_CHATS (also
// the limit of a private mesh whose creator chose none). Unset or 0: no limit.
const limit = (v) => Number(v) > 0 ? Math.floor(Number(v)) : Infinity;
const MAX_CHATS = limit(process.env.MAX_CHATS);
const MAX_PRIVATE_CHATS = limit(process.env.MAX_PRIVATE_CHATS);
// A seat whose `chat-end` never came is freed after this long. Longer than a
// page waits for any one job (route.js's JOB_TIMEOUT, 300 s).
const CHAT_LEASE_MS = Number(process.env.CHAT_LEASE_MS) || 600000;

// Chats each peer keeps in memory: MESH_SESSIONS in the global mesh and in a
// private mesh whose creator chose none, otherwise the creator's number capped
// by MAX_PRIVATE_SESSIONS. Peers size their KV pool to match (split.js's
// poolSlots()), so more costs VRAM and leaves fewer blocks per peer.
const count = (v, d) => Number.isInteger(Number(v)) && Number(v) >= 1 ? Number(v) : d;
const MESH_SESSIONS = count(process.env.MESH_SESSIONS, SESSIONS);
const MAX_PRIVATE_SESSIONS = count(process.env.MAX_PRIVATE_SESSIONS, 32);

// The context window of every chat, in tokens: MESH_CTX in the global mesh
// and in a private mesh whose creator chose none, otherwise the creator's
// choice capped by MAX_PRIVATE_CTX. Always a length in split.js's CONTEXTS
// that the model's n_pos covers. Peers size their KV pool for it, so a longer
// window costs VRAM and leaves fewer blocks per peer.
const ctxFor = (n) => runnerCtx(n, null, manifest?.n_pos);
const MAX_PRIVATE_CTX = ctxFor(count(process.env.MAX_PRIVATE_CTX, CONTEXTS.at(-1)));
const MESH_CTX = Math.min(ctxFor(count(process.env.MESH_CTX, undefined)), MAX_PRIVATE_CTX);

// How long a new socket has to send its `join`.
const JOIN_TIMEOUT_MS = Number(process.env.JOIN_TIMEOUT_MS) || 10000;

// Close codes a page reports and does not retry. 4001: a malformed or missing
// join, or one from a page with another PROTOCOL (a reload fixes it). 4002: a
// full room or server.
export const CLOSE_BAD_JOIN = 4001;
export const CLOSE_FULL = 4002;

// A join token is HKDF output: 32 bytes, base64url.
const TOKEN_RE = /^[A-Za-z0-9_-]{43}$/;

const makeRoom = (label, maxChats, sessions, ctx) => new Room({
    label, positions: POSITIONS, head: HEAD, minBlocks: MIN_BLOCKS, maxPeers: MAX_PEERS_PER_ROOM,
    maxChats, chatLeaseMs: CHAT_LEASE_MS, sessions, ctx,
});

const GLOBAL = 'global';
const rooms = new Map([[GLOBAL, makeRoom('', MAX_CHATS, MESH_SESSIONS, MESH_CTX)]]);

// One id counter for all rooms: monotonic and never reused. A browser keys
// links by peer id, so an id must never refer to a later peer, even after a
// room empties and refills.
let nextId = 0;

const send = (ws, msg) => ws.readyState === 1 && ws.send(JSON.stringify(msg));

// Cloudflare closes an idle WebSocket after about 100 seconds, and a socket
// can die without a close frame, leaving the tab with a stale roster. So the
// server pings every 30s and drops a peer that misses two. The ping is an
// application message, so the page's watchdog can see it.
const PING_MS = 30000;

setInterval(() => {
    for (const room of rooms.values()) {
        for (const [id, p] of room.peers) {
            if (p.alive === false) {
                room.log(`  peer ${id} missed two pings, dropping`);
                p.ws.terminate();
                continue;
            }
            p.alive = false;
            send(p.ws, { type: 'ping' });
        }
    }
}, PING_MS).unref?.();

const conns = new Map();          // CF-Connecting-IP -> open socket count

const wss = new WebSocketServer({
    host: HOST, port: PORT, maxPayload: MAX_PAYLOAD,
    verifyClient: ({ origin, req }, done) => {
        if (!originAllowed(origin)) return done(false, 403, 'origin not allowed');
        const ip = req.headers['cf-connecting-ip'];
        if (ip && (conns.get(ip) ?? 0) >= MAX_CONN_PER_IP) return done(false, 429, 'too many connections');
        done(true);
    },
});
const per = dims ? ` (${(blockBytes(dims, { nMax: MESH_CTX }) / 2 ** 20).toFixed(0)} MiB each on average, `
    + `with KV at ${MESH_CTX} tokens)` : '';
// Logged at boot to show which checkpoint the mesh serves.
console.log(`signaling on ws://${HOST}:${PORT}: ${manifest?.model ?? 'unknown model'}, `
    + `${BLOCKS} blocks${per} over ${POSITIONS} positions (embedding, blocks, head), `
    + `min ${MIN_BLOCKS} per peer`);
console.log(`  weights: ${WEIGHTS_URL}${process.env.WEIGHTS_URL ? ' (WEIGHTS_URL override)'
    : ` (from ${CHECKPOINT})`}`);
console.log(`  chats at once: ${Number.isFinite(MAX_CHATS) ? MAX_CHATS : 'no limit'} in the global mesh, `
    + `${Number.isFinite(MAX_PRIVATE_CHATS) ? `at most ${MAX_PRIVATE_CHATS}` : 'no cap'} in a private one`);
console.log(`  chats in memory per peer: ${MESH_SESSIONS} in the global mesh, `
    + `at most ${MAX_PRIVATE_SESSIONS} in a private one`);
console.log(`  context: ${MESH_CTX} tokens in the global mesh, at most ${MAX_PRIVATE_CTX} in a private one`);
if (OWN_TURN) console.log(`  TURN: ${TURN_URLS.join(', ')} (a credential per peer)`);
else if (TURN_SECRET || TURN_URLS.length) console.warn('TURN_SECRET and TURN_URLS go together. '
    + 'With only one set, peers are not given this TURN server.');
if (!OWN_TURN && !(TURN_KEY && TURN_TOKEN)) console.warn('no TURN (TURN_SECRET/TURN_URLS or '
    + 'TURN_KEY_ID/TURN_API_TOKEN). Peers get STUN only, and peer.html links through a TURN '
    + 'relay by default, so no link will open (except for tabs opened with ?direct=1).');
if (!dims) console.warn('no manifest, falling back to a flat 10-block floor. '
    + 'Set MIN_BLOCKS if that is wrong for this model.');

// The room a `join` names, created if new, or a close code if it is refused.
// `chats`, `sessions` and `ctx` are the creator's choices, read only when
// this join creates the room.
function roomFor(token, chats, sessions, ctx) {
    if (token == null) return { key: GLOBAL, room: rooms.get(GLOBAL) };
    if (typeof token !== 'string' || !TOKEN_RE.test(token)) return { refuse: CLOSE_BAD_JOIN };
    const key = createHash('sha256').update(token).digest('hex');
    let room = rooms.get(key);
    if (!room) {
        if (rooms.size - 1 >= MAX_ROOMS) return { refuse: CLOSE_FULL, why: 'too many private meshes' };
        const asked = Number.isInteger(chats) && chats >= 1 ? chats : Infinity;
        const kept = Number.isInteger(sessions) && sessions >= 1
            ? Math.min(sessions, MAX_PRIVATE_SESSIONS) : Math.min(MESH_SESSIONS, MAX_PRIVATE_SESSIONS);
        const windowCtx = Number.isInteger(ctx) && ctx >= 1
            ? Math.min(ctxFor(ctx), MAX_PRIVATE_CTX) : MESH_CTX;
        room = makeRoom(`room ${key.slice(0, 6)} `, Math.min(asked, MAX_PRIVATE_CHATS), kept, windowCtx);
        rooms.set(key, room);
        room.log(`opened, ${room.chatLimit === 1 ? '1 chat' : `${room.chatLimit ?? 'no limit on'} chats`} at once, `
            + `${kept} in memory per peer, ${windowCtx}-token context`);
    }
    return { key, room };
}

// A socket is in no room until its first message, which must be
//
//   peer -> join { v, room }        v: PROTOCOL; room: a join token, or null for the global mesh
//
// Nothing is sent to it before then, including the global roster, which a tab
// joining a private mesh should not see.
wss.on('connection', (ws, req) => {
    let joined = null;            // { id, key, room } once joined
    const timer = setTimeout(() => ws.close(CLOSE_BAD_JOIN, 'no join'), JOIN_TIMEOUT_MS);
    const ip = req.headers['cf-connecting-ip'];
    if (ip) conns.set(ip, (conns.get(ip) ?? 0) + 1);

    ws.on('message', async (data) => {
        let m;
        try { m = JSON.parse(data); } catch { return; }
        if (m === null || typeof m !== 'object') return;
        if (joined) return joined.room.handle(joined.id, m);

        if (m.type !== 'join') return ws.close(CLOSE_BAD_JOIN, 'join first');
        clearTimeout(timer);
        if (m.v !== PROTOCOL) return ws.close(CLOSE_BAD_JOIN, `protocol ${m.v}, not ${PROTOCOL}`);
        const { key, room, refuse, why } = roomFor(m.room, m.chats, m.sessions, m.ctx);
        if (refuse) return ws.close(refuse, why ?? 'bad join');
        if (room.full) {
            if (!room.size) rooms.delete(key);
            return ws.close(CLOSE_FULL, 'this mesh is full');
        }

        const id = nextId++;
        joined = { id, key, room };
        room.add(id, ws);

        // Handlers are attached before this await: minting can take a round
        // trip to Cloudflare, and an offer arriving meanwhile must be handled.
        const ice = await iceServers(id);
        if (!room.peers.has(id)) return;          // left during the mint
        send(ws, {
            type: 'welcome', id, blocks: BLOCKS, positions: POSITIONS, minBlocks: MIN_BLOCKS,
            weightsUrl: WEIGHTS_URL,          // the checkpoint this mesh runs
            // The kind of mesh this socket is in. The page compares it with
            // the kind it asked for and stops if they differ.
            private: key !== GLOBAL,
            // Replies that may generate at once here, or null for no limit.
            maxChats: room.chatLimit,
            // Chats each peer keeps in memory here.
            sessions: room.sessions,
            // The context window of every chat here, in tokens.
            ctx: room.ctx,
            ice, peers: room.roster(),
        });
        room.broadcast();
    });

    ws.on('close', () => {
        clearTimeout(timer);
        if (ip) {
            const n = conns.get(ip) - 1;
            if (n > 0) conns.set(ip, n); else conns.delete(ip);
        }
        if (!joined) return;
        const { id, key, room } = joined;
        room.remove(id);
        if (key !== GLOBAL && !room.size) {
            rooms.delete(key);
            room.log('closed');
        }
    });
});
