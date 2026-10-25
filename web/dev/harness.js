// One peer of the GPU harness: an Engine with an API in place of a chat
// window.
//
// tools/mesh.gpu.test.mjs opens one tab of this per peer and drives them all
// from node. Under the API is the production code: web/engine.js,
// web/route.js, web/net.js, web/model/stage.js, web/turn.js, the real kernels
// on a real GPU, the real signaling server, and (unless --wire=bc) real
// datachannels. This file adds a way to set what the peer holds, a way to send
// prompts, and a way to read back what happened.
//
// Query parameters
// ----------------
//   role=worker|spectator   serve blocks, or only prompt
//   at=lo-hi                hold exactly this range (engine.serve's `at`, which
//                           peer.html exposes as ?at=). Fixed placement makes
//                           a topology reproducible.
//   blocks=N                offer this capacity and let the server place it
//   vramlimit=bytes         set the VRAM limit before serving, as the welcome
//                           popup does on peer.html. 0 pauses background growth
//                           at its first step whatever the model's size. Raise
//                           it again with harness.setVramLimit.
//   wire=bc                 use a BroadcastChannel in place of the datachannel
//   label=name              this peer's name in the report
//   mesh=<secret>           join this private mesh (web/room.js). Passed in
//                           the query, not the fragment, since the harness URL
//                           is not a real invite.
//   relay=1                 link through TURN only, as peer.html does by
//                           default. Off by default here, since a local run
//                           has no TURN.
//   meshToken=<token>       test only: join with this join token but the link
//                           key of `mesh`. Simulates an intruder that knows
//                           the join token (as the server does) but not the
//                           secret.
//
// A spectator does not call serve(). serve() opens a GPUDevice and runs every
// kernel check before downloading anything, which a hundred spectators would
// each repeat, and a spectator never runs a block. A spectator here is a tab
// that only prompts, as in the mesh.

import { Engine, ContextFullError } from '../engine.js';
import { chatFormat, stopRe } from '../prompt.js';
import { runTurn } from '../turn.js';
import { installBroadcastWire } from './bcwire.js';
import { openRoom } from '../room.js';

// Assigned in boot(). Nothing in this module may use `location`, the DOM or
// the GPU at import time: tools/parse.test.mjs imports every browser module
// under Node to check that it evaluates.
let q, ROLE, LABEL, WIRE, wire, log;

const lines = [];
const say = (s, err = false) => {
    lines.push(s);
    if (lines.length > 200) lines.shift();
    if (log) {
        log.textContent = lines.join('\n');
        log.className = err ? 'err' : '';
    }
    console.log(`[${LABEL}] ${s}`);
};

// Every background event the engine reports, recorded so the harness can count
// them. 'repair' is the enqueuer rebuilding its context. 'fix' is a local
// replay another peer did for this chat.
const events = [];
let engine = null;

let CHAT = 'transcript';
let TURN_RE = stopRe(CHAT);
let booted = null;
const progress = { done: 0, total: 0, errors: 0, running: false };

// --- Boot ---

// `?at=lo-hi`, in the format peer.js accepts, places a peer at an exact range.
function placement() {
    const raw = q.get('at');
    if (!raw) return null;
    const m = /^(\d+)[-:](\d+)$/.exec(raw.trim());
    if (!m) throw new Error(`?at=${raw} is not a range. Write it as ?at=lo-hi.`);
    const [lo, hi] = [Number(m[1]), Number(m[2])];
    if (lo >= hi) throw new Error(`?at=${raw} is empty`);
    return [lo, hi];
}

export async function boot() {
    q = new URLSearchParams(location.search);
    ROLE = q.get('role') === 'spectator' ? 'spectator' : 'worker';
    LABEL = q.get('label') ?? ROLE;
    WIRE = q.get('wire') === 'bc' ? 'bc' : 'rtc';
    log = document.getElementById('log');

    // Installed before the Engine is created, so no link is negotiated over
    // the real WebRTC API first.
    wire = WIRE === 'bc' ? installBroadcastWire() : { wire: 'rtc' };

    engine = new Engine({
        onEvent: (type, detail) => {
            if (type === 'change' || type === 'progress') return;   // fire per token
            events.push({ at: performance.now(), type, ...detail });
            if (type === 'status') say(`status: ${detail.text}`);
            if (type === 'repair') say(`repair (${detail.reason}): ${detail.message ?? ''}`);
        },
    });

    booted = (async () => {
        const room = await openRoom(q.get('mesh') || null,
            { overrideToken: q.get('meshToken') || null });
        const { manifest, canPrompt } = await engine.connect({ room,
            relayOnly: q.get('relay') === '1' });
        CHAT = chatFormat(manifest);
        TURN_RE = stopRe(CHAT);
        say(`joined as peer ${engine.net.id}: ${manifest.model}, chat ${CHAT}, ` +
            `tokenizer ${canPrompt ? 'ok' : 'MISSING'}`);

        const out = { id: engine.net.id, label: LABEL, role: ROLE, wire: wire.wire,
                      model: manifest.model, canPrompt, eos: manifest.eos_token_id,
                      held: null, note: null, error: false, gpu: null };

        if (ROLE === 'spectator') {
            say('spectator: no GPU, no blocks, prompts only');
            return out;
        }

        const blocks = q.has('blocks') ? Number(q.get('blocks')) : null;
        // `?sessions=N`: how many concurrent chats this peer keeps frames for.
        // KV is a shared page pool (model.js's makePagePool), so raising this
        // does not multiply KV. On a hybrid checkpoint it does multiply the
        // recurrent state.
        const sessions = q.has('sessions') ? Number(q.get('sessions')) : undefined;
        // `?pages=N`: tokens per KV page. The output must not depend on it,
        // which checks the paging arithmetic: the same prompts at 16, 64 and a
        // page as wide as the whole window must give the same token ids.
        const pageTokens = q.has('pages') ? Number(q.get('pages')) : undefined;
        // `?floor=N`: claim this many positions initially, then grow as usual.
        // Unlike `?blocks=`, growth stays on (see engine.serve).
        const floor = q.has('floor') ? Number(q.get('floor')) : null;
        // `?wait=ms`: how long a new chat waits here for a frame before it is
        // refused (split.js's WAIT_MS). Set to 0 by the scenarios that test
        // the refusal.
        const waitMs = q.has('wait') ? Number(q.get('wait')) : undefined;
        // Set before serve(), as peer.js does with the welcome popup's answer.
        // It has to be in place before #growClaim starts: a Runner that has run
        // no job counts as idle, so the first growth step is taken with no
        // GROWTH_QUIET_MS wait, and a limit set after boot() resolves races it.
        if (q.has('vramlimit')) engine.setVramLimit(Number(q.get('vramlimit')));
        const r = await engine.serve({ at: placement(), blocks, sessions, pageTokens, floor, waitMs });
        Object.assign(out, { role: r.role, held: r.held, note: r.note, error: r.error });
        // The adapter this tab got. A headless Chrome without the Vulkan flags
        // falls back to SwiftShader and everything still passes, on the CPU.
        // The driver asserts on this.
        out.gpu = await adapterName();
        say(`${r.role}: ${r.note}`, r.error);
        return out;
    })();
    try {
        await booted;
    } catch (err) {
        say(`boot failed: ${err.message}`, true);
        booted = Promise.resolve({ label: LABEL, role: ROLE, error: true, note: err.message });
    }
    return booted;
}

async function adapterName() {
    try {
        const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        const i = a?.info ?? {};
        return `${i.vendor ?? '?'}/${i.architecture ?? '?'}${i.description ? ` ${i.description}` : ''}`;
    } catch {
        return null;
    }
}

// --- Prompting ---

// Greedy, with every penalty off. `temp 0` makes the sampler an argmax, so the
// same context must give the same token however the work was split.
// `rep`/`freq`/`pres`/`ngram` change the answer if two sides disagree about
// them, so all four are set explicitly.
const GREEDY = { temp: 0, topK: 0, rep: 1, freq: 0, pres: 0, ngram: 0,
                 chat: true, thinking: false, prof: 0 };

// Classifies a failure message into one of the error kinds the mesh produces.
// They are counted separately because the kind identifies the cause.
//
// `busy` is a peer refusing a new chat because every frame or page is held by
// an active one. `evicted` is a session whose idle cache was reclaimed: its
// next job arrives with past > 0 and no cache (stage.js's #cacheFor).
function classify(message) {
    if (/try again in a moment/.test(message)) return 'busy';
    if (/no longer cached here/.test(message)) return 'evicted';
    if (/no answer from the cluster/.test(message)) return 'timeout';
    if (/is unreachable|is reachable/.test(message)) return 'unreachable';
    if (/rebuild cancelled/.test(message)) return 'cancelled';
    if (/context/.test(message) && /full/.test(message)) return 'context-full';
    return 'other';
}

// One turn through the mesh, using web/turn.js, the function peer.js calls, so
// the production code decides where the reply ends.
// `retryBusy` retries after a capacity refusal, which is the correct response:
// the context is still cached, so asking again after a short wait works. Each
// retry resets first, because a prompt longer than one T_MAX chunk may have
// had some chunks accepted before the refusal. Those are in the history, and
// feeding them again would change the prompt.
async function attempt(text, opts, { retryBusy = 0, backoffMs = 200, fresh = true } = {}) {
    let waited = 0;
    for (let n = 0; ; n++) {
        const r = await one(text, opts);
        if (r.ok || r.why !== 'busy' || n >= retryBusy) {
            return { ...r, busyWaits: waited };
        }
        waited++;
        if (fresh) engine.reset();
        await new Promise((res) => setTimeout(res, backoffMs));
    }
}

// `opts.stopAfter` calls stop() once that many tokens have arrived. With the
// closed loop, tokens already fed still arrive afterwards.
async function one(text, { stopAfter = 0, ...opts } = {}) {
    const t0 = performance.now();
    const before = events.length;
    try {
        const turn = await runTurn(engine, text, { ...GREEDY, ...opts }, {
            chat: CHAT, turnRe: TURN_RE,
            onToken: stopAfter ? ({ n }) => { if (n >= stopAfter) engine.stop(); } : null,
        });
        return {
            ok: true, prompt: text, ids: turn.ids, text: turn.text,
            stopped: turn.stopped, rewound: turn.rewound, lastFed: engine.lastFed,
            // Of the tokens this reply fed back in, how many the head fed (the
            // closed loop) and not this tab.
            closed: (engine.lastProfile?.samples ?? []).filter((x) => x.loop && x.closed).length,
            promptTokens: turn.promptTokens, ctxLen: engine.ctxLen,
            ms: Math.round(performance.now() - t0),
            // The peers this token crossed, from the profile the router
            // returns. Reported so a wrong answer can be traced to its path.
            path: hops(),
            events: events.slice(before).map((e) => e.type),
        };
    } catch (err) {
        return {
            ok: false, prompt: text, ids: [], text: '',
            why: err instanceof ContextFullError ? 'context-full'
                : err.busy ? 'busy' : classify(err.message),
            error: err.message,
            ms: Math.round(performance.now() - t0),
            events: events.slice(before).map((e) => e.type),
        };
    }
}

// The peer ids on the last decode job's path, in order.
function hops() {
    const s = engine.lastProfile?.samples ?? [];
    const last = s.filter((x) => x.kind === 'decode').at(-1) ?? s.at(-1);
    return (last?.trace ?? []).map((h) => h.id);
}

export const api = {
    // Everything the driver needs to know about this peer once it is up.
    // Starts the boot if it has not started, so the driver's first call does
    // not depend on module evaluation order.
    boot: () => booted ?? boot(),

    // `prompts` is a list of strings, run sequentially because one Engine is
    // one chat. Concurrency in this harness is across tabs.
    //
    // `fresh` starts a new chat before each prompt, giving independent turns
    // that spread sessions across the peers' KV slots. Without it the turns
    // form one conversation and the context grows.
    //
    // `perTurn`, if given, is merged into `opts` for each turn. A reference
    // run uses it to replay a conversation whose turns were cut at different
    // lengths.
    async run({ prompts, opts = {}, fresh = true, retryBusy = 0, backoffMs = 200,
                gapMs = 0, perTurn = null } = {}) {
        // One call is one conversation. With `fresh` each prompt resets below.
        // Without it the whole list is one chat, reset here so it does not
        // continue the previous call's context.
        if (!fresh) engine.reset();
        progress.running = true;
        progress.done = 0;
        progress.total = prompts.length;
        progress.errors = 0;
        const out = [];
        for (const [i, text] of prompts.entries()) {
            // A pause between turns. Background growth backs off while a peer
            // is serving (engine.js's GROWTH_QUIET_MS), so a run with no gaps
            // never grows a claim.
            if (gapMs && i) await new Promise((res) => setTimeout(res, gapMs));
            if (fresh) engine.reset();
            const r = await attempt(text, { ...opts, ...perTurn?.[i] }, { retryBusy, backoffMs, fresh });
            if (!r.ok) progress.errors++;
            out.push(r);
            progress.done++;
        }
        progress.running = false;
        say(`ran ${out.length} turns, ${progress.errors} failed`);
        return out;
    },

    // Polled by the driver during a long run, so a stalled soak is noticed
    // before the timeout.
    progress: () => ({ ...progress, label: LABEL, id: engine.net?.id ?? null }),

    // The VRAM limit, as peer.html's field sets it. `null` is the blank field:
    // no limit. JSON has no Infinity, so the driver sends null. Returns what
    // this peer holds, in MiB, so a caller can compare it against a later read.
    setVramLimit(bytes = null) {
        engine.setVramLimit(bytes ?? Infinity);
        return Math.round(engine.vramBytes / 2 ** 20);
    },

    // This peer's state at the end. The runner numbers show whether there was
    // KV pressure: `live` is how many sessions hold frames and `slots` is how
    // many frames exist.
    stats() {
        const net = engine.net;
        const r = engine.runner;
        // Uses net.coverage, the getter that gates a prompt, so the harness
        // reports the same number.
        const cov = net?.positions ? net.coverage : null;
        return {
            label: LABEL, id: net?.id ?? null, role: engine.role,
            ctxLen: engine.ctxLen, ran: engine.ran,
            peers: net?.peers?.length ?? 0,
            covered: cov ? `${cov.covered}/${cov.total}` : null,
            r: cov?.r ?? null,
            // The stretches no ready peer holds, as half-open position ranges.
            holes: cov?.holes ?? [],
            staffed: net?.staffed ?? false,
            online: net?.online ?? false,
            connected: net?.connected ?? false,
            links: net?.links ? [...net.links.keys()] : [],
            // `open`: the peers with an open channel, the only ones a message
            // can reach. `roster`: every peer id this tab knows of.
            open: net?.links ? [...net.links.keys()].filter((id) => net.linked(id)) : [],
            roster: net?.peers?.map((p) => p.id) ?? [],
            private: net?.room?.private ?? false,
            authFailures: net?.authFailures ?? 0,
            refused: net?.refused ?? null,
            // Links a held ready/have announcement is waiting on (net.js).
            linking: net?.linking ?? null,
            runner: r ? {
                lo: r.lo, hi: r.hi, have: r.have,
                live: r.live.size, slots: r.live.size + r.free.length,
                cuts: r.cuts.size, logs: r.logs.size,
                // The pool: how many pages are free, and how many tokens one
                // page holds. `pagesFree` at 0 with live sessions means page
                // pressure, not frame pressure.
                pagesFree: r.pool.free.length, pagesTotal: r.pool.nPages,
                pageTokens: r.pool.pageTokens,
                vramMiB: Math.round(engine.vramBytes / 2 ** 20),
                ...r.counts,
            } : null,
            events: events.map((e) => ({ type: e.type, at: Math.round(e.at) })),
            nMax: engine.ctxMax,
        };
    },

    // This tab's log lines, for reading a failure in detail.
    lines: () => lines.slice(),
};
