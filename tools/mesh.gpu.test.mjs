// Tests whether the mesh answers correctly when several people prompt at
// once: real peers in real tabs on a real GPU, with the real signaling server
// and datachannels, running a 1.3 MB model (tools/tinymodel.mjs).
//
//   node tools/mesh.gpu.test.mjs                     the default matrix
//   node tools/mesh.gpu.test.mjs --only 8x3          one scenario
//   node tools/mesh.gpu.test.mjs --wire bc           BroadcastChannel, no ICE
//   node tools/mesh.gpu.test.mjs --prompts 200       a soak
//   node tools/mesh.gpu.test.mjs --show              stream every peer's console
//   node tools/mesh.gpu.test.mjs --round-trip        every row without the closed loop
//   node tools/mesh.gpu.test.mjs --long              prompts of several T_MAX chunks
//   node tools/mesh.gpu.test.mjs --fuzz 12 --seed 3  random scenarios, each replayed
//                                                    against a one-peer reference
//
// Decoding is greedy with every penalty off, so the output depends only on the
// context. Every row is compared, token for token, with one peer holding the
// whole model.

import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { serve } from './harness/serve.mjs';
import { launch } from './harness/cdp.mjs';
import { positions } from '../web/split.js';
import { webcrypto } from 'node:crypto';
// In a Node 18 module, WebCrypto is not a global. web/room.js needs it.
globalThis.crypto ??= webcrypto;
const { openRoom, newSecret } = await import('../web/room.js');

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// --- Arguments ---

const argv = process.argv.slice(2);
const has = (f) => argv.includes(`--${f}`);
const opt = (f, d) => { const i = argv.indexOf(`--${f}`); return i < 0 ? d : argv[i + 1]; };

const WIRE     = opt('wire', 'rtc');
const ONLY     = opt('only', null);
const PROMPTS  = Number(opt('prompts', 8));
const TOKENS   = Number(opt('tokens', 24));
const SEED     = Number(opt('seed', 7));
const SHOW     = has('show');
const HEAVY    = has('heavy');
const KEEP     = has('keep');
const WEIGHTS  = opt('weights', 'weights/tiny-test');
const FUZZ     = Number(opt('fuzz', 0));
// Prompts long enough to be sent as several pipelined prefill chunks.
const LONG     = has('long');
// Run every row on the round trip, to compare with the closed loop.
const ROUND_TRIP = has('round-trip');

if (!['rtc', 'bc'].includes(WIRE)) {
    console.error(`--wire ${WIRE}: expected rtc or bc`);
    process.exit(2);
}

// --- The model, and the line peers are placed on ---

if (!existsSync(`${ROOT}/${WEIGHTS}/manifest.json`)) {
    console.error(`no checkpoint at ${WEIGHTS}: run:  node tools/tinymodel.mjs`);
    process.exit(2);
}
const manifest = JSON.parse(
    await import('node:fs').then((fs) => fs.readFileSync(`${ROOT}/${WEIGHTS}/manifest.json`, 'utf8')));
const L = manifest.n_layer;
const N = positions(L);          // the embedding, every block, the head

// --- Topologies ---

// `n` contiguous ranges covering the whole line, as equal as possible. One
// chain: every position is held once, r = 1, and every job crosses all n
// peers.
function chain(n) {
    const out = [];
    for (let i = 0; i < n; i++) {
        out.push([Math.floor((i * N) / n), Math.floor(((i + 1) * N) / n)]);
    }
    return out.filter(([lo, hi]) => hi > lo);
}

// Two independent chains over the same line, so every position has two holders
// and hops() has a choice. `n` is the total number of peers.
function deep(n) {
    const a = Math.ceil(n / 2);
    return [...chain(a), ...chain(n - a)];
}

// The matrix, as data. `--only 8x3` runs one row with the same harness code.
const SCENARIOS = [
    // The reference. One peer holds the whole model and one spectator asks it,
    // one turn at a time, with no cuts and no concurrency, using the round
    // trip.
    { name: 'reference', workers: [[0, N]], askers: 1, reference: true, loop: false },

    // Cuts with one asker: checks that splitting the line does not change the
    // answer.
    { name: '3x1',  workers: chain(3), askers: 1 },
    { name: '8x1',  workers: chain(8), askers: 1 },

    // Eight peers and three askers. Admission control refuses the turns there
    // is no room for and does not evict a live chat, so these are `refusals`
    // rows: no eviction, no cold cache, no repair.
    //
    // Two frames (set explicitly; the default is eight) are too few for three
    // askers. `8x3paged` below has the same shape with enough frames, and
    // `busywait` has the same shape with a client that waits.
    //
    // Every `refusals` row sets `waitMs: 0`. By default a chat with no frame
    // waits in line (split.js's WAIT_MS), and these rows would only be slow.
    { name: '3x3',  workers: chain(3), askers: 3, sessions: 2, waitMs: 0, expect: 'refusals' },
    { name: '8x3',  workers: chain(8), askers: 3, sessions: 2, waitMs: 0, expect: 'refusals' },

    // Depth: two holders per position, so a job has a choice of route, and two
    // sessions can be pinned to different peers for the same position.
    { name: '10x2deep', workers: deep(10), askers: 2 },

    // peer.html's normal case: a tab that holds blocks and also prompts. It
    // runs the first stretch of its own jobs locally (route()'s iCanRun, no
    // hop), so its own runner holds a cache for its own chat. No `free`
    // message reaches that cache, so it must be released locally. The other
    // scenarios prompt from spectators and do not test this.
    // Here one tab holds everything and prompts itself. Every job runs locally,
    // so only route.js's `runner()?.release()` returns a finished chat's KV
    // slot. Six sequential chats use two slots. Without the local release this
    // row reports four evictions.
    { name: '1x1self', workers: [[0, N]], askers: 1, selfAsk: true },
    { name: '3x3self', workers: chain(3), askers: 3, selfAsk: true, sessions: 2, waitMs: 0, expect: 'refusals' },

    // The shape of `8x3`, with frames for four concurrent chats per peer. The
    // KV pool is the same size: paging allocates by tokens held, and these
    // chats hold ~40 tokens in a pool sized for 2 x 4096. So this row uses no
    // more VRAM than `8x3`.
    { name: '8x3paged', workers: chain(8), askers: 3, sessions: 4 },
    { name: '4x12paged', workers: chain(4), askers: 12, sessions: 16 },

    // Capacity. One frame per peer and three askers: two must be refused as
    // busy, not admitted by evicting a live chat. The row asserts that nothing
    // was repaired or rebuilt and no cache went cold.
    { name: 'busy', workers: chain(3), askers: 3, sessions: 1, waitMs: 0, expect: 'refusals' },

    // The same mesh with default settings: a chat with no frame waits in line
    // at the peer, first come first served, so nobody is starved. Every turn
    // must complete with the reference tokens. `retryBusy` covers a wait
    // longer than WAIT_MS.
    { name: 'busywait', workers: chain(3), askers: 3, sessions: 1, retryBusy: 40 },

    // A peer leaves mid-generation in a mesh with spare coverage: twelve
    // workers in two independent chains, so every position still has a holder.
    // Every reply must arrive and match.
    { name: 'drop', workers: deep(12), askers: 3, sessions: 4, kill: { after: 0.35 } },

    // Paging must not change the output. The same setup with pages a quarter
    // the size, which changes how many pages a context spans and where page
    // boundaries fall in attn_av_gqa's loop.
    { name: 'pages16', workers: chain(3), askers: 2, sessions: 4, pageTokens: 16 },

    // A conversation: `fresh: false`, so each turn is wrapped with
    // `first: false`, the context accumulates, and `past` grows across turns.
    // The other rows reset between prompts, so only this one tests the
    // multi-turn path: the growing context, wrapTurn's later-turn form, and
    // the rewind after a cut turn.
    { name: 'multiturn', workers: chain(3), askers: 2, sessions: 4, multiTurn: true },

    // The same conversation until the context is full. The last turn that fits
    // must be truncated and report 'context full', the next must be refused
    // before any work, and a new chat must work immediately afterwards.
    { name: 'ctxfull', workers: chain(3), askers: 1, sessions: 4, multiTurn: true,
      promptCount: 40, tokens: 200, expect: 'context-full', followUp: true },

    // Growth during a conversation: peers start at a small floor and widen in
    // the background while a chat is live. The gap between turns allows it,
    // since growth backs off while a peer is serving.
    //
    // The assertion is token identity. A session's cut is fixed when it first
    // runs on a peer. If a widening claim moved the cut, the next peer would
    // receive a different stretch of the residual stream and the reply would
    // differ, or come from a cache that was never written.
    //
    // Four peers at floor 3 cover the line at boot (3+3+3+1). With three
    // peers, position 9 would be unheld, meshReady would wait for growth to
    // cover it, and growth would finish before the conversation started. With
    // the line covered from the start, growth happens between turns, while the
    // chat's KV is live on peers whose claims are widening.
    { name: 'growth', workers: 'capacity4', askers: 1, sessions: 4, multiTurn: true,
      floor: 3, gapMs: 2600, promptCount: 5 },

    // A hole in the line: nobody holds positions 3-5. The mesh must refuse to
    // answer and name the missing stretch, without hanging or answering from a
    // cache nobody wrote. It must work as soon as a peer fills the hole.
    { name: 'hole', workers: [[0, 3], [6, 10]], askers: 1, sessions: 4,
      expect: 'hole', fill: [3, 6] },

    // Many more askers than peers. Each is a separate session competing for
    // the same KV slots.
    { name: '4x12', workers: chain(4), askers: 12, sessions: 2, waitMs: 0, expect: 'refusals' },

    // Private meshes (web/room.js). A private mesh and the global one run on
    // one signaling server, both prompted at once, and both must match the
    // one-peer reference. Also: neither sees the other's peers, a tab with a
    // different invite gets a room of its own, and two intruders that hold
    // room A's join token but not its secret (as the server does) appear in
    // A's roster and never get an open link. One joins before A's workers and
    // one after, so both directions are tested: A refusing an intruder's
    // offer, and an intruder refusing A's.
    { name: 'rooms', rooms: true, workers: chain(2), askers: 1 },

    // A peer joins to cover a stretch nobody else holds, and a prompt is sent
    // as soon as the mesh reports it can answer. The ready gate (net.js) makes
    // the new peer count only once its links are open, so the prompt does not
    // fail with "unreachable" on real WebRTC.
    { name: 'join', workers: chain(3), askers: 1, sessions: 4, joinRow: true },

    // Raising the VRAM limit after growth has paused on it. One worker starts
    // at floor 1 with a limit of 0, so its first growth step is refused and the
    // line stays uncovered: the asker cannot prompt. Clearing the limit must
    // cover the line without a reload, growing from the holding already in
    // VRAM.
    { name: 'vramgrow', workers: 'capacity1', askers: 1, sessions: 4, floor: 1,
      vramLimit: true },

    // Rows specific to the closed loop. Every row above already decodes with
    // it (the engine's default). These test what only the loop does, and keep
    // the round trip tested on real hardware.
    //
    // Stop is pressed after a few tokens. Tokens already fed still arrive and
    // stay in the context, so a stopped reply is checked by replaying the same
    // conversation on a one-peer reference.
    { name: 'loopstop', workers: chain(3), askers: 2, sessions: 4, stopAfter: 3, replay: true },
    { name: 'loopstopmulti', workers: chain(3), askers: 2, sessions: 4, stopAfter: 2,
      multiTurn: true, replay: true },
    // Half the askers on the loop and half on the round trip, through the same peers.
    { name: 'loopmixed', workers: deep(6), askers: 4, sessions: 4, mixed: true },
    // The round trip only.
    { name: '3x1rt', workers: chain(3), askers: 1, loop: false },
    { name: '8x3pagedrt', workers: chain(8), askers: 3, sessions: 4, loop: false },
    { name: 'droprt', workers: deep(12), askers: 3, sessions: 4, kill: { after: 0.35 }, loop: false },
    { name: 'multiturnrt', workers: chain(3), askers: 2, sessions: 4, multiTurn: true, loop: false },

    // The large case: a hundred spectators and four workers. On the rtc wire
    // that is a hundred RTCPeerConnections to the holder of position 0
    // (web/net.js's #targets links every spectator to the entry point). It
    // needs a hundred renderer processes, so it runs only with --heavy or
    // --only 4x100.
    { name: '4x100', workers: chain(4), askers: 100, heavy: true },
];

// --fuzz: scenarios drawn at random from the same fields as the matrix: cuts,
// depth, how many askers and where, conversations or fresh chats, the loop or
// the round trip or a mix, Stop or not, a peer killed or not. Each is checked
// by replay against a one-peer reference (`replay`), so no expected output is
// needed.
function randomScenario(rand, i) {
    const cuts = () => {
        const c = [0];
        for (let x = 1; x < N; x++) if (rand() < 0.35) c.push(x);
        c.push(N);
        return c.slice(1).map((hi, j) => [c[j], hi]);
    };
    const deep = rand() < 0.5;
    const workers = deep ? [...cuts(), ...cuts()] : cuts();
    const selfAsk = rand() < 0.25;
    const askers = 1 + Math.floor(rand() * Math.min(3, selfAsk ? workers.length : 3));
    const loop = rand() < 0.2 ? false : undefined;
    return {
        name: `fuzz${i}`, workers, askers, selfAsk, sessions: 4, replay: true,
        multiTurn: rand() < 0.5, loop, mixed: loop === undefined && rand() < 0.25,
        stopAfter: rand() < 0.35 ? 1 + Math.floor(rand() * 6) : 0,
        kill: deep && rand() < 0.3 ? { after: 0.2 + rand() * 0.4 } : null,
        promptCount: 3 + Math.floor(rand() * 4), tokens: 6 + Math.floor(rand() * 18),
    };
}

const chosen = FUZZ ? (() => {
    const rand = rng(SEED);
    return Array.from({ length: FUZZ }, (_, i) => randomScenario(rand, i));
})() : (ONLY ? SCENARIOS.filter((s) => s.name === ONLY || s.reference)
             : SCENARIOS.filter((s) => HEAVY || !s.heavy));
if (ONLY && !SCENARIOS.some((s) => s.name === ONLY)) {
    console.error(`--only ${ONLY}: no such scenario. Have: ` +
        `${SCENARIOS.map((s) => s.name).join(', ')}`);
    process.exit(2);
}

// --- Prompts ---

// xorshift32. One integer on the command line reproduces the whole run.
function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => {
        s ^= s << 13; s >>>= 0;
        s ^= s >>> 17;
        s ^= s << 5; s >>>= 0;
        return s / 0x100000000;
    };
}

// The words have no meaning to a model with random weights. The prompts must
// differ in length, because prompt length sets how many T_MAX chunks a prefill
// takes, and a rebuild re-chunks the context.
const WORDS = ('the a of and to in is it that was for on are with as his they be at one have '
    + 'this from or had by hot word but what some we can out other were all there when up use '
    + 'your how said an each she which do their time if will way about many then them write')
    .split(' ');

function prompts(n, rand) {
    const out = [];
    for (let i = 0; i < n; i++) {
        const len = LONG ? 120 + Math.floor(rand() * 180) : 2 + Math.floor(rand() * 18);
        const words = [];
        for (let j = 0; j < len; j++) words.push(WORDS[Math.floor(rand() * WORDS.length)]);
        out.push(`${words.join(' ')}?`);
    }
    return out;
}

const PROMPT_LIST = prompts(PROMPTS, rng(SEED));

// A scenario that needs a different number of turns draws them from the same
// seed, so its first N prompts are the matrix's first N and a reference
// recorded for one is valid for the other.
const promptsFor = (scenario) => scenario.promptCount
    ? prompts(scenario.promptCount, rng(SEED)) : PROMPT_LIST;

// Two prompts that end in the same words and differ before them. If the
// model's next token depended only on its last token, both would give the same
// reply, and so would a peer answering from another chat's KV cache, which
// this harness must detect. They run in the reference scenario to check that
// the model's output depends on the earlier context.
const PROBE = ['alpha beta gamma delta the word is', 'zeta eta theta iota the word is'];

// --- Reporting ---

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name}${detail ? `\n        ${detail}` : ''}`); }
};

// The longest run of one repeated id. A model stuck in a cycle can still
// report many distinct ids overall while its last thirty tokens are identical,
// and those tokens add nothing to a comparison.
function longestRun(ids) {
    let best = 0, run = 0;
    for (let i = 0; i < ids.length; i++) {
        run = i && ids[i] === ids[i - 1] ? run + 1 : 1;
        best = Math.max(best, run);
    }
    return best;
}

const pct = (xs, p) => {
    if (!xs.length) return 0;
    const s = [...xs].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};

// --- The world ---

// The real signaling server runs as a child process. Placement, the roster,
// readiness and the reconnect are its code, and `assign()` placing eight peers
// is part of what is tested.
// freePort() asks the OS for an unused port and releases it. signal-server.js
// takes its port from the environment and prints that value, so with `PORT=0`
// it would bind a random port and print ":0", and the real port could not be
// read back.
async function freePort() {
    const { createServer } = await import('node:net');
    return new Promise((res) => {
        const s = createServer();
        s.listen(0, '127.0.0.1', () => {
            const { port } = s.address();
            s.close(() => res(port));
        });
    });
}

async function signalServer() {
    const port = await freePort();
    const proc = spawn('node', ['signal-server.js'], {
        cwd: ROOT,
        env: {
            ...process.env,
            PORT: String(port),
            WEIGHTS_URL: `/${WEIGHTS}`,
            // The server derives this from the manifest, but it is set
            // explicitly: a peer refuses to boot on a wrong N_LAYER.
            N_LAYER: String(L),
            // The floor is a byte budget sized for SmolLM2-shaped blocks
            // (135 MiB), and a block of this model is 148 KiB. With the
            // default, floorBlocks() would make the smallest range the whole
            // model, and no cut topology could be placed.
            MIN_BLOCKS: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    let out = '';
    await new Promise((res, rej) => {
        const timer = setTimeout(() => rej(new Error(`signal server silent for 15s:\n${out}`)), 15000);
        const look = (d) => {
            out += d;
            if (!/signaling on ws:/.test(out)) return;
            clearTimeout(timer);
            res();
        };
        proc.stdout.on('data', look);
        proc.stderr.on('data', look);
        proc.on('exit', (c) => { clearTimeout(timer); rej(new Error(`signal server exited ${c}:\n${out}`)); });
    });
    if (SHOW) { proc.stdout.on('data', (d) => process.stdout.write(`  [signal] ${d}`)); }

    // Kills the server if the run never reaches its `finally`, as cdp.mjs does
    // for the browser. Otherwise the server would keep its port.
    const stop = () => { try { proc.kill('SIGKILL'); } catch { /* gone */ } };
    process.once('exit', stop);
    for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.once(sig, stop);

    return { port, stop, log: () => out };
}

async function main() {
    const signal = await signalServer();
    const http = await serve({ root: ROOT, signalPort: signal.port });
    const browser = await launch({ verbose: false });
    const base = `http://localhost:${http.port}/web/dev/harness.html`;

    console.log(`\n${manifest.model}: ${L} blocks over ${N} positions, wire ${WIRE}, ` +
        `${PROMPTS} prompts x ${TOKENS} tokens, seed ${SEED}`);
    console.log(`signaling on :${signal.port}, files on :${http.port}\n`);

    const rows = [];
    let reference = null;

    try {
        if (chosen.some((s) => s.replay)) REPLAY = await replayMesh(browser, base);
        for (const scenario of chosen) {
            const r = await runScenario(browser, base, scenario, reference);
            if (scenario.reference) reference = r;
            rows.push(r);
        }
        verdict(rows, reference);
    } finally {
        await REPLAY?.close();
        if (!KEEP) await browser.close();
        await http.close();
        signal.stop();
    }
}

// The replay oracle's mesh: one peer holding the whole model and one
// spectator, in a private room, so no scenario's jobs are routed to them. Kept
// for the whole run.
let REPLAY = null;
async function replayMesh(browser, base) {
    const room = `&mesh=${newSecret()}`;
    const open = async (query, label) => {
        const tab = await browser.tab(`${base}?${query}${room}&wire=${WIRE}&label=${label}`);
        await tab.waitFor('window.harness');
        const info = await tab.eval('window.harness.boot()');
        if (info.error) throw new Error(`the replay reference failed to boot: ${info.note}`);
        return { tab, info };
    };
    const worker = await open(`role=worker&at=0-${N}&sessions=4`, 'ref-w');
    const asker = await open('role=spectator', 'ref-a');
    await meshReady([asker], 2);
    console.log(`replay reference: peer ${worker.info.id} holding everything, asked by ` +
        `${asker.info.id}, in a private room`);
    return {
        // Replay one asker's conversation with the round trip, cutting each
        // turn at the length the scenario's turn had. The ids must match
        // exactly. Greedy decoding makes the reply depend only on the context,
        // so this holds whatever the scenario did.
        async replay(spec, turns) {
            // Replays the conversation only up to its first failed turn. What
            // a failed turn left in the context is unknown.
            const failedAt = turns.findIndex((t) => !t.ok);
            const kept = spec.fresh || failedAt < 0 ? turns : turns.slice(0, failedAt);
            const idx = kept.map((t, i) => i).filter((i) => kept[i].ok);
            if (!idx.length) return [];
            const run = await asker.tab.eval(`window.harness.run(${JSON.stringify({
                prompts: idx.map((i) => spec.prompts[i]), fresh: spec.fresh,
                opts: { ...spec.opts, loop: false, stopAfter: 0 },
                perTurn: idx.map((i) => ({ maxTokens: kept[i].ids.length })),
            })})`, { timeout: 900000 });
            return idx.map((i, j) => ({ i, got: kept[i].ids, want: run[j]?.ids ?? null,
                                        error: run[j]?.error }))
                .filter((x) => x.error || x.got.join(',') !== x.want.join(','));
        },
        close: () => Promise.all([worker.tab.close(), asker.tab.close()]),
    };
}

// One row of the matrix: start the peers, wait until the mesh can answer,
// prompt it from every asker at once, then close everything.
async function runScenario(browser, base, scenario, reference) {
    const { name, askers } = scenario;
    // `'capacityN'` means N workers placed by the server from a floor, not
    // given explicit ranges. Only then does a peer have room to grow.
    const byCapacity = typeof scenario.workers === 'string';
    const workers = byCapacity
        ? Array.from({ length: Number(scenario.workers.replace(/\D/g, '')) }, () => null)
        : scenario.workers;
    console.log(`\n${name}: ${workers.length} worker${workers.length === 1 ? '' : 's'} ` +
        `${byCapacity ? `placed by the server from floor ${scenario.floor}`
                      : JSON.stringify(workers)}, ${askers} asker${askers === 1 ? '' : 's'}`);

    const tabs = [];
    const open = async (query, label) => {
        const tab = await browser.tab(`${base}?${query}&wire=${WIRE}&label=${label}`, {
            onConsole: (kind, text) => {
                if (SHOW) console.log(`    [${label}] ${text}`);
                else if (kind === 'error') console.log(`    [${label}] ${text}`);
            },
        });
        tabs.push(tab);
        // The page sets its API during module evaluation, after the document
        // has loaded, so wait for it before calling anything.
        await tab.waitFor('window.harness');
        return tab;
    };

    try {
        if (scenario.rooms) return await rooms(open, scenario, reference);
        if (scenario.joinRow) return await joinRow(open, scenario, reference);
        if (scenario.vramLimit) return await vramLimitRow(open, scenario, reference);

        // Workers first, each awaited: serve() resolves once the range is
        // downloaded, warmed up and announced, so the line is covered when the
        // last one returns. They boot one at a time because they share one
        // GPU.
        const opts = (i) => (workers[i] ? `role=worker&at=${workers[i][0]}-${workers[i][1]}`
                                        : `role=worker&floor=${scenario.floor ?? 1}`)
            + (scenario.sessions ? `&sessions=${scenario.sessions}` : '')
            + (scenario.waitMs !== undefined ? `&wait=${scenario.waitMs}` : '')
            + (scenario.pageTokens ? `&pages=${scenario.pageTokens}` : '');

        const peers = [];
        for (let i = 0; i < workers.length; i++) {
            const t0 = Date.now();
            const tab = await open(opts(i), `w${i}`);
            const info = await tab.eval('window.harness.boot()');
            peers.push({ tab, info });
            if (info.error) throw new Error(`worker ${i} failed to serve: ${info.note}`);
            console.log(`    w${i} peer ${info.id} holds ${info.held?.[0]}-${info.held?.[1] - 1} ` +
                `(${((Date.now() - t0) / 1000).toFixed(1)}s, ${info.gpu})`);
        }

        // A SwiftShader adapter means Chrome fell back to software rendering.
        // Every check would pass on the CPU, which is not what this harness
        // tests, so the run stops.
        const soft = peers.find((p) => /swiftshader|llvmpipe|software/i.test(p.info.gpu ?? ''));
        if (soft) {
            throw new Error(`this browser gave out a software adapter (${soft.info.gpu}): ` +
                `the GPU flags in tools/harness/cdp.mjs did not take effect`);
        }

        const spectators = [];
        if (!scenario.selfAsk) {
            for (let i = 0; i < askers; i++) {
                const tab = await open('role=spectator', `a${i}`);
                spectators.push({ tab, info: await tab.eval('window.harness.boot()') });
            }
        }

        // Who prompts: dedicated spectators, or the workers themselves. The
        // code below uses `prompters`, not `spectators`.
        const prompters = scenario.selfAsk ? peers.slice(0, askers) : spectators;
        // A mesh with a hole never becomes answerable, so that row waits for
        // the roster to settle.
        if (scenario.expect === 'hole') await rosterSettled(prompters, workers.length + askers);
        else await meshReady(prompters, workers.length + spectators.length);

        // With a hole, the composer stays disabled and names the missing
        // stretch. This is checked before anything is sent: refusing to send
        // is correct, and sending a prompt that then fails is not.
        if (scenario.expect === 'hole') {
            const before = await prompters[0].tab.eval('window.harness.stats()');
            ok(`${name}: a mesh with a hole is not answerable`,
                before.connected === false, `connected=${before.connected}`);
            ok(`${name}: and it names the stretch nobody holds`,
                before.holes.length > 0 && before.staffed === false,
                `holes=${JSON.stringify(before.holes)} staffed=${before.staffed}`);
            console.log(`    ${before.covered} held, holes ${JSON.stringify(before.holes)}`);
        }

        // Every asker runs the same prompts at the same time. One Engine is
        // one chat, so each tab runs its list sequentially and the concurrency
        // is across tabs.
        const t0 = Date.now();
        // The runs are not awaited here: a scenario that kills a peer must do
        // it while they are in flight, so the promises are kept and progress
        // is polled.
        // `atStart` is each worker's range when the conversation starts.
        // Growth begins at boot, since a Runner that has run no job counts as
        // idle. Comparing with the placement would count growth from before
        // the first prompt.
        const atStart = await Promise.all(peers.map(({ tab }) => tab.eval(
            'window.harness.stats()')));
        if (scenario.gapMs) {
            console.log(`    at the first prompt: ${atStart.map((x) => x.runner
                ? `${x.label} ${x.runner.lo}-${x.runner.hi - 1}` : `${x.label} -`).join(', ')}`);
        }

        const list = promptsFor(scenario);
        const spec = {
            prompts: list,
            opts: { maxTokens: scenario.tokens ?? TOKENS,
                    ...(ROUND_TRIP ? { loop: false } : scenario.loop !== undefined && { loop: scenario.loop }),
                    ...(scenario.stopAfter && { stopAfter: scenario.stopAfter }) },
            fresh: !scenario.multiTurn,
            retryBusy: scenario.retryBusy ?? 0,
            gapMs: scenario.gapMs ?? 0,
        };
        // `mixed`: every other asker uses the round trip.
        const specFor = (i) => (scenario.mixed && i % 2 ? { ...spec, opts: { ...spec.opts, loop: false } } : spec);
        const inFlight = prompters.map(({ tab }, i) => tab.eval(
            `window.harness.run(${JSON.stringify(specFor(i))})`, { timeout: 1800000 }));

        let killed = null;
        if (scenario.kill) killed = await killOne(peers, prompters, scenario.kill, list.length);

        const runs = await Promise.all(inFlight);
        const wall = Date.now() - t0;

        // After every reply has ended, nothing may be running: no peer's count
        // of positions run changes for a second. This detects a closed loop
        // that kept running after its reply. A turn refused partway through a
        // long prompt leaves its prefill chunks already sent (at most
        // route.js's PREFILL_WINDOW) to finish, so the counts are first
        // given up to 5 s to settle. A loop never settles.
        const live = peers.filter(({ tab }) => !tab.closed);
        const ranAt = async () => (await Promise.all(live.map(({ tab }) =>
            tab.eval('window.harness.stats()')))).map((x) => x.ran);
        let ran0 = await ranAt();
        for (const until = Date.now() + 5000; Date.now() < until;) {
            await new Promise((r) => setTimeout(r, 250));
            const now = await ranAt();
            if (JSON.stringify(now) === JSON.stringify(ran0)) break;
            ran0 = now;
        }
        await new Promise((r) => setTimeout(r, 1000));
        const ran1 = await ranAt();
        ok(`${name}: nothing still running once every reply had ended`,
            ran0.every((x, i) => x === ran1[i]), `ran ${JSON.stringify(ran0)} → ${JSON.stringify(ran1)}`);

        // The replay oracle, for rows that can only be checked by replay.
        let replayed = null;
        if (scenario.replay) {
            replayed = [];
            for (const [i, run] of runs.entries()) replayed.push(...await REPLAY.replay(specFor(i), run));
        }

        // Fill the hole and ask again. The asker is unchanged, and the mesh
        // can now answer.
        let filled = null;
        if (scenario.fill) {
            const [lo, hi] = scenario.fill;
            console.log(`    filling ${lo}-${hi - 1}`);
            const tab = await open(`role=worker&at=${lo}-${hi}`
                + (scenario.sessions ? `&sessions=${scenario.sessions}` : ''), 'wfill');
            const info = await tab.eval('window.harness.boot()');
            if (info.error) throw new Error(`the filling worker failed: ${info.note}`);
            peers.push({ tab, info });
            await meshReady(prompters, workers.length + askers + 1);
            const linked = await Promise.all(peers.map(({ tab }) => tab.eval('window.harness.stats()')));
            console.log(`    open links once the asker is ready: ${linked.map((x) =>
                `peer ${x.id} → ${JSON.stringify(x.open)}`).join('; ')}`);
            // Uses `retryBusy`. Each failed turn made a rebuild, and each
            // rebuild minted a new session id, which the position-0 holder
            // admits and keeps until it has been idle for IDLE_MS (recover()
            // does not free the old session's caches). So when the hole is
            // filled, the frames are held by dead sessions and the first
            // prompt gets a capacity refusal. The client waits and retries,
            // and this row tests that interaction.
            filled = await prompters[0].tab.eval(
                `window.harness.run(${JSON.stringify({ ...spec, prompts: list.slice(0, 2),
                                                       fresh: true, retryBusy: 40 })})`,
                { timeout: 300000 });
        }

        // A new chat after the context ran out. A full context belongs to one
        // conversation and does not affect the mesh.
        let after = null;
        if (scenario.followUp) {
            after = await prompters[0].tab.eval(
                `window.harness.run(${JSON.stringify({ ...spec, prompts: list.slice(0, 1),
                                                       fresh: true })})`,
                { timeout: 300000 });
        }

        // The killed peer's tab is closed, and evaluating in a closed tab
        // throws, so its stats are skipped.
        const stats = await Promise.all([...peers, ...spectators]
            .filter(({ tab }) => !tab.closed)
            .map(({ tab }) => tab.eval('window.harness.stats()')));

        // The reference for multi-turn rows: the same prompts as one
        // accumulating conversation on a single peer. A fresh-chat reference
        // does not apply, since turn 3 of a conversation depends on turns 1
        // and 2.
        const chain = scenario.reference ? await prompters[0].tab.eval(
            `window.harness.run(${JSON.stringify({
                prompts: PROMPT_LIST, opts: { maxTokens: TOKENS, loop: false }, fresh: false })})`,
            { timeout: 900000 }) : null;

        // Reference row only: a check of the oracle, not of the mesh.
        const probe = scenario.reference ? await prompters[0].tab.eval(
            `window.harness.run(${JSON.stringify({
                prompts: PROBE, opts: { maxTokens: TOKENS, loop: false }, fresh: true })})`) : null;

        const row = { name, workers: workers.length, askers, runs, stats, wall, probe,
                       killed, scenario, filled, after, chain, replayed,
                       grew: grewDuring(atStart, stats) };
        check(row, reference);
        return row;
    } finally {
        await Promise.all(tabs.map((t) => t.close()));
    }
}

// The `rooms` row: room A (private) and the global mesh, each with two workers
// and an asker, prompted at once. Also a stranger with its own invite, and two
// intruders that hold A's join token without its secret.
async function rooms(open, scenario, reference) {
    const secret = newSecret();
    const token = (await openRoom(secret)).joinToken;
    const inA = `&mesh=${secret}`;
    // A new secret with A's token: the server puts the tab in room A, but its
    // link key differs from A's.
    const intruder = `&mesh=${newSecret()}&meshToken=${token}`;

    const boot = async (query, label) => {
        const tab = await open(query, label);
        const info = await tab.eval('window.harness.boot()');
        if (info.error) throw new Error(`${label} failed to boot: ${info.note}`);
        return { tab, info, label };
    };

    // Joins before A's workers, so its id is lower and it makes the offer.
    const early = await boot(`role=spectator${intruder}`, 'x-early');
    const a = [], g = [];
    for (const [i, [lo, hi]] of scenario.workers.entries()) {
        a.push(await boot(`role=worker&at=${lo}-${hi}${inA}`, `a-w${i}`));
        g.push(await boot(`role=worker&at=${lo}-${hi}`, `g-w${i}`));
    }
    const askA = await boot(`role=spectator${inA}`, 'a-ask');
    const askG = await boot('role=spectator', 'g-ask');
    const late = await boot(`role=spectator${intruder}`, 'x-late');
    const stranger = await boot(`role=spectator&mesh=${newSecret()}`, 'stranger');
    console.log(`    room A: ${a.map((p) => p.info.id)} + asker ${askA.info.id}; global: ` +
        `${g.map((p) => p.info.id)} + asker ${askG.info.id}; intruders ${early.info.id}, ` +
        `${late.info.id}; stranger ${stranger.info.id}`);

    await meshReady([askA], a.length + 1);
    await meshReady([askG], g.length + 1);

    const spec = { prompts: PROMPT_LIST, opts: { maxTokens: TOKENS }, fresh: true };
    const t0 = Date.now();
    const runs = await Promise.all([askA, askG].map(({ tab }) => tab.eval(
        `window.harness.run(${JSON.stringify(spec)})`, { timeout: 1800000 })));
    const wall = Date.now() - t0;

    const everyone = [...a, askA, ...g, askG, early, late, stranger];
    const stats = await Promise.all(everyone.map(({ tab }) => tab.eval('window.harness.stats()')));
    const by = Object.fromEntries(everyone.map((p, i) => [p.label, stats[i]]));
    const idsA = new Set([...a, askA].map((p) => p.info.id));
    const idsG = new Set([...g, askG].map((p) => p.info.id));
    const intruders = new Set([early.info.id, late.info.id]);
    const sub = (xs, set) => xs.every((id) => set.has(id));
    const inRoomA = [...a, askA].map((p) => by[p.label]);
    const inGlobal = [...g, askG].map((p) => by[p.label]);

    ok('rooms: room A says it is private, the global mesh says it is not',
        inRoomA.every((x) => x.private) && inGlobal.every((x) => !x.private));
    ok('rooms: room A\'s roster holds only room A and the token-holding intruders',
        inRoomA.every((x) => sub(x.roster, new Set([...idsA, ...intruders]))),
        JSON.stringify(inRoomA.map((x) => x.roster)));
    ok('rooms: the global roster holds only the global mesh',
        inGlobal.every((x) => sub(x.roster, idsG)), JSON.stringify(inGlobal.map((x) => x.roster)));
    ok('rooms: every open link in room A is to room A',
        inRoomA.every((x) => sub(x.open, idsA)), JSON.stringify(inRoomA.map((x) => x.open)));
    ok('rooms: every open link in the global mesh is to the global mesh',
        inGlobal.every((x) => sub(x.open, idsG)), JSON.stringify(inGlobal.map((x) => x.open)));
    ok('rooms: a tab with another invite is alone in a room of its own',
        JSON.stringify(by.stranger.roster) === JSON.stringify([stranger.info.id])
            && by.stranger.open.length === 0,
        `roster ${JSON.stringify(by.stranger.roster)}, open ${JSON.stringify(by.stranger.open)}`);
    for (const x of [early, late]) {
        const st = by[x.label];
        ok(`rooms: ${x.label}, holding A's join token, is in A's roster`,
            [...idsA].every((id) => st.roster.includes(id)), JSON.stringify(st.roster));
        ok(`rooms: and has no open link to anyone`, st.open.length === 0, JSON.stringify(st.open));
    }
    const refusedByA = inRoomA.reduce((n, x) => n + x.authFailures, 0);
    ok('rooms: room A refused the early intruder\'s offer', refusedByA > 0, `${refusedByA} refused`);
    ok('rooms: the late intruder refused room A\'s offer', by['x-late'].authFailures > 0,
        `${by['x-late'].authFailures} refused`);
    ok('rooms: nobody in the global mesh refused anything',
        inGlobal.every((x) => x.authFailures === 0));

    const row = { name: 'rooms', workers: a.length + g.length, askers: 2, runs, stats, wall,
                  probe: null, killed: null, scenario, filled: null, after: null, chain: null,
                  grew: [] };
    check(row, reference);
    return row;
}

// The `join` row: a chain is answering, its middle peer leaves, a replacement
// joins to cover that stretch, and the asker prompts as soon as the mesh
// reports it can answer. The prompt must succeed with no rebuild and no
// "unreachable" error. On real WebRTC that requires the replacement to count
// only once its links are open.
async function joinRow(open, scenario, reference) {
    const boot = async (query, label) => {
        const tab = await open(query, label);
        const info = await tab.eval('window.harness.boot()');
        if (info.error) throw new Error(`${label} failed to boot: ${info.note}`);
        return { tab, info };
    };
    const at = ([lo, hi]) => `role=worker&at=${lo}-${hi}&sessions=${scenario.sessions ?? 4}`;
    const [a, mid, c] = scenario.workers;
    const w0 = await boot(at(a), 'w0');
    const w1 = await boot(at(mid), 'w1');
    const w2 = await boot(at(c), 'w2');
    const asker = await boot('role=spectator', 'a0');
    await meshReady([asker], 4);

    const list = PROMPT_LIST.slice(0, 2);
    const spec = (prompts) => ({ prompts, opts: { maxTokens: TOKENS }, fresh: true });
    const t0 = Date.now();
    const before = await asker.tab.eval(`window.harness.run(${JSON.stringify(spec(list.slice(0, 1)))})`);

    console.log(`    peer ${w1.info.id} (${mid[0]}-${mid[1] - 1}) leaves; a replacement joins`);
    await w1.tab.close();
    const w3tab = await open(at(mid), 'w3');
    // Poll the replacement until the mesh can answer, recording whether it was
    // ever held back waiting for links.
    let sawLinking = false, joined = null;
    const watching = (async () => {
        for (;;) {
            const [mine, theirs] = await Promise.all([
                w3tab.eval('window.harness.stats()').catch(() => null),
                asker.tab.eval('window.harness.stats()'),
            ]);
            if (mine?.linking) sawLinking = true;
            joined ??= mine?.id;
            if (theirs.connected && theirs.covered?.split('/')[0] === theirs.covered?.split('/')[1]
                && theirs.peers >= 4) return;
            await new Promise((r) => setTimeout(r, 20));
        }
    })();
    const booting = w3tab.eval('window.harness.boot()');
    await watching;
    const after = await asker.tab.eval(`window.harness.run(${JSON.stringify(spec(list.slice(1, 2)))})`);
    await booting;
    const wall = Date.now() - t0;
    console.log(`    replacement peer ${joined} ${sawLinking ? 'waited on its links before it counted'
        : 'was never seen waiting on its links'}; the next prompt ${after[0].ok ? 'went through'
        : `failed: ${after[0].error}`}`);
    if (WIRE === 'rtc') {
        ok('join: the replacement counted only once its links were open', sawLinking);
    }

    const tabs = [w0, w2, asker, { tab: w3tab }];
    const stats = await Promise.all(tabs.map(({ tab }) => tab.eval('window.harness.stats()')));
    const row = { name: 'join', workers: 3, askers: 1, runs: [[...before, ...after]], stats, wall,
                  probe: null, killed: null, scenario, filled: null, after: null, chain: null, grew: [] };
    check(row, reference);
    return row;
}

// Growth paused on the VRAM limit, then resumed by raising it. `?vramlimit=0`
// refuses the first growth step whatever the model's size, so the worker sits
// on its floor and the line stays uncovered. Clearing the limit must cover it
// from the weights already in VRAM: the peer keeps its id and its Runner
// throughout, so it cannot have re-downloaded.
async function vramLimitRow(open, scenario, reference) {
    const boot = async (query, label) => {
        const tab = await open(query, label);
        const info = await tab.eval('window.harness.boot()');
        if (info.error) throw new Error(`${label} failed to boot: ${info.note}`);
        return { tab, info };
    };
    const sessions = scenario.sessions ?? 4;
    const t0 = Date.now();
    const w0 = await boot(`role=worker&floor=${scenario.floor ?? 1}&vramlimit=0&sessions=${sessions}`,
                          'w0');
    const asker = await boot('role=spectator', 'a0');
    await rosterSettled([w0, asker], 2);

    const notes = async () => (await w0.tab.eval('window.harness.lines()'))
        .filter((l) => l.startsWith('status: '));
    const paused = (ls) => ls.filter((l) => /Paused growing at /.test(l));

    const held = await w0.tab.eval('window.harness.stats()');
    const seen = await asker.tab.eval('window.harness.stats()');
    console.log(`    peer ${w0.info.id} holds ${held.runner?.lo}-${held.runner?.hi - 1} ` +
        `(${held.runner?.vramMiB} MiB), covered ${seen.covered}, holes ` +
        `${JSON.stringify(seen.holes)}`);
    ok('vramgrow: the limit stopped growth short of covering the line',
        seen.holes.length > 0 && !seen.connected,
        `covered ${seen.covered}, connected ${seen.connected}`);
    // Long enough for several quiet periods to pass with the holding not
    // moving, which separates a parked loop from a slow one. It also covers the
    // first step's GROWTH_QUIET_MS, during which the pause note is emitted:
    // serve() can resolve before a freshly loaded page has been idle that long.
    await new Promise((r) => setTimeout(r, 3000));
    const still = await w0.tab.eval('window.harness.stats()');
    ok('vramgrow: the holding does not creep while the limit stands',
        still.runner?.lo === held.runner?.lo && still.runner?.hi === held.runner?.hi,
        `${held.runner?.lo}-${held.runner?.hi} became ${still.runner?.lo}-${still.runner?.hi}`);
    ok('vramgrow: and it said so once, as a pause rather than a stop',
        paused(await notes()).length === 1, JSON.stringify(await notes()));

    // Clear it, as a visitor blanking the field does.
    console.log(`    clearing the VRAM limit on peer ${w0.info.id}`);
    await w0.tab.eval('window.harness.setVramLimit(null)');
    await meshReady([asker], 2);

    const grown = await w0.tab.eval('window.harness.stats()');
    const ls = await notes();
    ok('vramgrow: growth resumed and covered the line', grown.runner?.lo === 0
        && grown.runner?.hi === N && grown.runner?.have === grown.runner?.hi,
        `holds ${grown.runner?.lo}-${grown.runner?.hi}, have ${grown.runner?.have}`);
    ok('vramgrow: it is the same peer, so nothing was re-downloaded',
        grown.id === w0.info.id, `peer ${w0.info.id} became ${grown.id}`);
    ok('vramgrow: and it is holding more VRAM than when it paused',
        grown.runner?.vramMiB > held.runner?.vramMiB,
        `${held.runner?.vramMiB} MiB became ${grown.runner?.vramMiB} MiB`);
    ok('vramgrow: the resume was announced once', ls.filter((l) =>
        /The VRAM limit has room again/.test(l)).length === 1, JSON.stringify(ls));
    ok('vramgrow: and the pause was never repeated', paused(ls).length === 1, JSON.stringify(ls));

    // One real turn, so the row compares tokens like every other row and the
    // latency summary has something to report.
    const spec = { prompts: PROMPT_LIST.slice(0, 1), opts: { maxTokens: TOKENS }, fresh: true };
    const runs = [await asker.tab.eval(`window.harness.run(${JSON.stringify(spec)})`)];
    const wall = Date.now() - t0;

    const stats = await Promise.all([w0, asker].map(({ tab }) => tab.eval('window.harness.stats()')));
    const row = { name: 'vramgrow', workers: 1, askers: 1, runs, stats, wall, probe: null,
                  killed: null, scenario, filled: null, after: null, chain: null,
                  grew: grewDuring([held], stats) };
    check(row, reference);
    return row;
}

// A reduced meshReady for a mesh that never becomes answerable: wait until
// every peer has seen every other, so the hole is in the settled roster and
// not the result of a message in flight.
async function rosterSettled(prompters, expected, timeout = 60000) {
    const until = Date.now() + timeout;
    for (;;) {
        const s = await Promise.all(prompters.map(({ tab }) => tab.eval('window.harness.stats()')));
        if (s.every((x) => x.peers >= expected)) return;
        if (Date.now() > until) {
            throw new Error(`the roster never reached ${expected} peers: ` +
                JSON.stringify(s.map((x) => x.peers)));
        }
        await new Promise((r) => setTimeout(r, 200));
    }
}

// Which workers hold more of the line at the end than at the start. Growth
// backs off while a peer is serving, so this shows whether it ran.
function grewDuring(atStart, stats) {
    const out = [];
    for (const start of atStart) {
        const end = stats.find((x) => x.label === start.label);
        if (!start.runner || !end?.runner) continue;
        if (end.runner.lo < start.runner.lo || end.runner.hi > start.runner.hi) {
            out.push({ label: start.label,
                       from: [start.runner.lo, start.runner.hi],
                       to: [end.runner.lo, end.runner.hi] });
        }
    }
    return out;
}

// Close a worker's tab partway through the run. This is the ordinary departure
// that repair and rebuild handle: the datachannel fires close, `net.send`
// returns false, and the signaling server removes the peer from the roster
// shortly after.
//
// Waits for progress first, so the departure happens mid-generation, after
// something has been cached. Picks a peer that is carrying sessions, since a
// peer holding no session leaves nothing to repair. Never picks the
// position-0 holder: the enqueuer keeps no boundary log for its own first hop,
// so that departure always costs a full rebuild. mesh.test.mjs's soak tests it
// in murderous mode.
async function killOne(peers, prompters, { after = 0.35 }, prompts = PROMPTS, timeout = 120000) {
    const target = Math.max(1, Math.round(prompts * prompters.length * after));
    const until = Date.now() + timeout;
    for (;;) {
        const done = (await Promise.all(prompters.map(({ tab }) =>
            tab.eval('window.harness.progress()')))).reduce((n, p) => n + p.done, 0);
        if (done >= target) break;
        if (Date.now() > until) throw new Error(`only ${done} of ${target} turns ran in ` +
            `${timeout / 1000}s, so there was never a mid-generation moment to kill in`);
        await new Promise((r) => setTimeout(r, 100));
    }

    const live = await Promise.all(peers.map(({ tab }) => tab.eval('window.harness.stats()')));
    const carrying = peers
        .map((p, i) => ({ ...p, stats: live[i] }))
        .filter((p) => p.stats.runner && p.stats.runner.live > 0 && p.stats.runner.lo > 0);
    const victim = carrying[0] ?? peers.at(-1);
    console.log(`    killing ${victim.stats.label} (peer ${victim.stats.id}, holds ` +
        `${victim.stats.runner.lo}-${victim.stats.runner.have - 1}, ` +
        `${victim.stats.runner.live} chats cached)`);
    await victim.tab.close();
    return { label: victim.stats.label, id: victim.stats.id,
             held: [victim.stats.runner.lo, victim.stats.runner.have] };
}

// A peer can prompt once the ready peers' ranges cover the model and it has a
// link to the entry and a path for the answer: `net.connected`, which peer.js
// also uses to enable its composer. Polling that, not sleeping, keeps a slow
// ICE gather from causing a failure.
async function meshReady(spectators, expected, timeout = 120000) {
    const until = Date.now() + timeout;
    for (;;) {
        const s = await Promise.all(spectators.map(({ tab }) => tab.eval('window.harness.stats()')));
        const ready = s.every((x) => x.connected && x.peers >= expected
            && x.covered && x.covered.split('/')[0] === x.covered.split('/')[1]);
        if (ready) return;
        if (Date.now() > until) {
            throw new Error(`the mesh never became answerable in ${timeout / 1000}s: ` +
                JSON.stringify(s.map((x) => ({ id: x.id, peers: x.peers, covered: x.covered,
                                               r: x.r, connected: x.connected }))));
        }
        await new Promise((r) => setTimeout(r, 250));
    }
}

// --- Assertions ---

function check(row, reference) {
    const all = row.runs.flat();
    const failed = all.filter((r) => !r.ok);
    const why = {};
    for (const f of failed) why[f.why] = (why[f.why] ?? 0) + 1;
    // What the row expects. A capacity row should refuse turns and a departure
    // row should repair them, so "every turn completed" and "nothing needed
    // repairing" do not apply there. Those rows assert that the refusals and
    // repairs happened as expected.
    const expectRefusals = row.scenario?.expect === 'refusals';
    const expectRepairs = !!row.scenario?.kill;
    const expectFull = row.scenario?.expect === 'context-full';
    const expectHole = row.scenario?.expect === 'hole';
    const waits = all.reduce((n, r) => n + (r.busyWaits ?? 0), 0);
    if (waits) {
        console.log(`    ${waits} turns waited out a refusal and asked again`);
        // Per asker and per turn: who waited, and for how long.
        for (const [i, run] of row.runs.entries()) {
            console.log(`      asker ${i}: ${run.map((r) => `${r.ok ? '' : '✗'}${r.busyWaits ?? 0}`).join(' ')}`);
        }
    }

    const ms = all.filter((r) => r.ok).map((r) => r.ms);
    const tokens = all.reduce((n, r) => n + r.ids.length, 0);
    console.log(`    ${all.length} turns, ${tokens} tokens, ${(row.wall / 1000).toFixed(1)}s ` +
        `p50 ${pct(ms, 50)}ms p95 ${pct(ms, 95)}ms, ` +
        `${(tokens / (row.wall / 1000)).toFixed(0)} tok/s aggregate`);

    if (expectFull) {
        // Running out of context is not a mesh failure. It can end two ways:
        // the last turn that fits is truncated and says so, or a turn is
        // refused before any work. The context must reach the limit for this
        // row to test anything. The check is on how close it got, not on a
        // stop reason, because the ending is not deterministic. A turn is
        // truncated only when the remaining room is in (0, maxTokens), and a
        // conversation whose steps are prompt+maxTokens long can skip that
        // range and go straight to the refusal. Both endings are correct, and
        // only the refusal always happens, so only the refusal is asserted.
        const reached = Math.max(...all.filter((r) => r.ok).map((r) => r.ctxLen ?? 0));
        const room = row.stats[0]?.nMax - reached;
        ok(`${row.name}: the conversation filled the context`,
            room >= 0 && room <= (row.scenario.tokens ?? TOKENS),
            `got within ${room} tokens of ${row.stats[0]?.nMax}`);
        const truncated = all.filter((r) => r.ok && r.stopped === 'context full');
        console.log(`    the wall was reached by ${truncated.length
            ? 'a truncated turn' : 'a refusal'} (both are correct)`);
        ok(`${row.name}: and the next one is refused outright`,
            (why['context-full'] ?? 0) > 0, JSON.stringify(why));
        ok(`${row.name}: every failure was the context filling, nothing else`,
            failed.length === (why['context-full'] ?? 0), JSON.stringify(why));
        // The mesh still works. Only that conversation is over.
        const after = row.after?.[0];
        ok(`${row.name}: a new chat works immediately afterwards`, after?.ok === true,
            `${after?.error ?? 'no follow-up ran'}`);
        console.log(`    ran ${all.length} turns to ${all.filter((r) => r.ok).at(-1)?.ctxLen} ` +
            `tokens of context before it filled`);
    } else if (expectHole) {
        // Nothing could be sent, so nothing should have been sent. The
        // composer must refuse before a job exists.
        ok(`${row.name}: no turn was attempted into a mesh that cannot answer`,
            all.length === 0 || failed.length === all.length,
            `${all.length - failed.length} turns somehow completed`);
        const back = row.filled ?? [];
        ok(`${row.name}: and once the hole is filled the same asker is answered`,
            back.length > 0 && back.every((r) => r.ok),
            `${back.find((r) => !r.ok)?.error ?? 'nothing ran after filling'}`);
    } else if (expectRefusals) {
        // A refusal must take nothing from the chats the peer is serving.
        ok(`${row.name}: chats were refused rather than admitted`, (why.busy ?? 0) > 0,
            `no turn was refused: ${JSON.stringify(why)}`);
        ok(`${row.name}: and every failure was a refusal, nothing else broke`,
            failed.length === (why.busy ?? 0), JSON.stringify(why));
        ok(`${row.name}: no live chat was robbed to make room`,
            (why.evicted ?? 0) === 0, `${why.evicted} chats lost their KV`);
    } else {
        // Every turn must complete. A reply that never started or stopped
        // early counts as a failure, not as a slow answer.
        ok(`${row.name}: every turn completed`, failed.length === 0,
            failed.length ? `${failed.length}/${all.length} failed: ${JSON.stringify(why)}\n` +
                `        first: ${failed[0].error}` : '');
    }

    // Checks whether the mesh slows down over time within one scenario. A
    // drift here comes from the run itself, not from a difference between
    // topologies. It is reported and not asserted: a timing threshold on a
    // shared GPU would be flaky, and the completion and token assertions catch
    // the failures a slowdown would cause.
    if (row.runs[0].length >= 8) {
        const quarter = Math.floor(row.runs[0].length / 4);
        const slice = (from, to) => row.runs.flatMap((r) => r.slice(from, to))
            .filter((r) => r.ok).map((r) => r.ms);
        // 'n/a', not 0, when every turn in a bucket failed. A p50 over no
        // samples would print as 0ms and look like a large speed change.
        const p50 = (xs) => (xs.length ? `${pct(xs, 50)}ms` : 'n/a');
        const early = slice(0, quarter), late = slice(-quarter);
        const first = p50(early), last = p50(late);
        const ratio = early.length && late.length
            ? `${(pct(late, 50) / (pct(early, 50) || 1)).toFixed(2)}x` : 'n/a';
        const lateFails = row.runs.flatMap((r) => r.slice(-quarter)).filter((r) => !r.ok).length;
        const earlyFails = row.runs.flatMap((r) => r.slice(0, quarter)).filter((r) => !r.ok).length;
        console.log(`    over time: first quarter p50 ${first}, last quarter p50 ${last} ` +
            `(${ratio}), failures ${earlyFails} then ${lateFails}`);
    }

    // The closed loop must close on real peers. A head that always declined
    // would give the same answers through the enqueuer. Every token of a reply
    // except the last is fed, and the head should have fed all of them for the
    // askers on the loop.
    const looping = ROUND_TRIP ? [] : row.runs.filter((_, i) => row.scenario?.loop !== false
        && !(row.scenario?.mixed && i % 2) && !row.scenario?.reference);
    const feedable = looping.flat().filter((r) => r.ok).reduce((n, r) => n + Math.max(0, r.lastFed ?? 0), 0);
    const closed = looping.flat().filter((r) => r.ok).reduce((n, r) => n + (r.closed ?? 0), 0);
    if (looping.length && feedable) {
        ok(`${row.name}: the loop closed at the head for nearly every token`,
            closed >= 0.9 * feedable - looping.flat().length,
            `the head fed ${closed} of ${feedable} tokens`);
        console.log(`    closed loop: the head fed ${closed} of ${feedable} tokens`);
    }

    const ladder = all.flatMap((r) => r.events ?? []);
    const repairs = ladder.filter((e) => e === 'repair').length;
    const fixes = ladder.filter((e) => e === 'fix').length;
    if (repairs || fixes) {
        console.log(`    repairs: ${fixes} local repairs, ` +
            `${repairs} context rebuilds`);
    }
    if (expectHole) {
        // The enqueuer tries to rebuild around a hole, twice per turn
        // (maxRepairs), and fails, because no peer holds the missing position
        // and a new session does not change that. This is asserted to show the
        // failure path ran and the prompt was not dropped without an error. An
        // unreachable position cannot be repaired, like a capacity refusal,
        // but it is not marked as such, so the rebuilds are spent.
        ok(`${row.name}: the enqueuer tried to rebuild and could not`, repairs > 0,
            `${repairs} rebuilds`);
    } else if (expectRepairs) {
        // A peer left, so a repair or rebuild must have run. Without this
        // check the row would pass if the kill had not happened.
        ok(`${row.name}: the departure was actually absorbed by repair or rebuild`,
            repairs + fixes > 0,
            `peer ${row.killed?.id} left and no repair or rebuild happened: the kill ` +
            `may not have landed mid-generation`);
    } else {
        // No peer left, was killed or was throttled. A repair or rebuild here
        // means the mesh broke a hand-off by itself, which is reported even
        // when the answer is correct.
        ok(`${row.name}: nothing needed repairing`, repairs === 0 && fixes === 0,
            repairs || fixes ? `${fixes} local repairs, ${repairs} rebuilds with ` +
                `no peer having left` : '');
    }

    if (row.replayed) {
        const bad = row.replayed;
        const turns = all.filter((r) => r.ok).length;
        ok(`${row.name}: every reply is what one peer says, replaying the exact conversation`,
            bad.length === 0,
            bad.length ? `${bad.length} of ${turns} turns differ. first, turn ${bad[0].i}:\n` +
                `        got  ${JSON.stringify(bad[0].got)}\n        want ${JSON.stringify(bad[0].want)}` +
                (bad[0].error ? `\n        (replay failed: ${bad[0].error})` : '') : '');
        if (row.scenario.stopAfter && row.scenario.loop !== false) {
            const past = all.filter((r) => r.ok && r.ids.length > row.scenario.stopAfter).length;
            console.log(`    ${past} of ${turns} stopped turns had tokens arrive after Stop (closed loop)`);
        }
        return;
    }

    if (!reference) {
        // The reference row must be non-degenerate. If the model's argmax were
        // always the same token, every comparison below would pass trivially.
        const ids = row.runs[0].map((r) => r.ids);
        const distinct = new Set(ids.flat()).size;
        const runs = ids.map((x) => longestRun(x));
        console.log(`    reference: ${distinct} distinct ids over ${ids.flat().length} tokens, ` +
            `longest repeat ${Math.max(...runs)}`);
        console.log(`    ${ids.slice(0, 2).map((x) => JSON.stringify(x)).join('\n    ')}`);
        ok('reference: the model produces varied tokens', distinct > 3,
            `only ${distinct} distinct ids across ${ids.flat().length}: the random ` +
            `weights have collapsed to a near-constant argmax, so every diff below ` +
            `would pass vacuously. Retune tools/tinymodel.mjs.`);
        ok('reference: different prompts produce different replies',
            new Set(ids.map((x) => x.join(','))).size > 1);
        // The probe: the same last words with different context. If the
        // replies matched, the model could not tell the right prefix from a
        // wrong one, and the comparisons below would not detect an evicted
        // cache.
        const [a, b] = row.probe ?? [];
        ok('reference: a different prefix changes the reply (the oracle can see context)',
            !!a?.ok && !!b?.ok && a.ids.join(',') !== b.ids.join(','),
            a?.ok && b?.ok ? `both replies were ${JSON.stringify(a.ids)}: the model is ` +
                `conditioned on its last token alone, so a cold KV cache would be ` +
                `undetectable. Retune tools/tinymodel.mjs's attn-scale.`
                : `probe turns failed: ${a?.error ?? ''} ${b?.error ?? ''}`);
        return;
    }

    // The comparison: every asker and every prompt against the one-peer
    // answer. Independent turns use the fresh-chat baseline, and a multi-turn
    // row uses the conversation baseline.
    const want = row.scenario?.multiTurn ? reference.chain : reference.runs[0];
    if (!want) return;                       // no baseline to compare with

    if (row.scenario?.gapMs) {
        ok(`${row.name}: a peer actually grew its claim during the conversation`,
            row.grew.length > 0,
            `no worker widened: growth backs off while a peer is serving ` +
            `(engine.js's GROWTH_QUIET_MS), so the ${row.scenario.gapMs}ms gap may ` +
            `be too short`);
        if (row.grew.length) {
            console.log(`    grew: ${row.grew.map((g) => `${g.label} ` +
                `${g.from[0]}-${g.from[1] - 1} -> ${g.to[0]}-${g.to[1] - 1}`).join(', ')}`);
        }
        // A cut exists only if the chat was handed from one peer to another.
        // Without this check the row could pass with the whole conversation on
        // one peer, where a widening claim has no boundary to move.
        // `runner.admitted`: harness.js's stats() puts the counters in the
        // runner block.
        const admits = row.stats.filter((x) => x.runner)
            .reduce((n, x) => n + (x.runner.admitted ?? 0), 0);
        ok(`${row.name}: and the live chat was cut across peers while they grew`,
            admits > 1, `only ${admits} peer admitted the session, so no cut existed`);
    }
    if (expectHole || expectFull) return;    // these rows compare no tokens

    const wrong = [];
    for (const run of row.runs) {
        for (const [i, got] of run.entries()) {
            if (!got.ok) continue;
            const w = want[i];
            if (got.ids.join(',') === w.ids.join(',')) continue;
            wrong.push({ i, prompt: got.prompt, got: got.ids, want: w.ids, path: got.path });
        }
    }
    ok(`${row.name}: every reply matches the single-peer reference`, wrong.length === 0,
        wrong.length ? `${wrong.length} replies differ. first, prompt ${wrong[0].i}:\n` +
            `        got  ${JSON.stringify(wrong[0].got)}\n` +
            `        want ${JSON.stringify(wrong[0].want)}\n` +
            `        over peers ${JSON.stringify(wrong[0].path)}` : '');

    // An early ending is already caught by the comparison above. It is also
    // counted separately here. It is indexed per asker: flattening first would
    // compare a reply with the wrong prompt.
    const short = [];
    for (const run of row.runs) {
        for (const [i, got] of run.entries()) {
            if (got.ok && got.ids.length < want[i].ids.length) {
                short.push({ prompt: i, got: got.ids.length, want: want[i].ids.length,
                             stopped: got.stopped });
            }
        }
    }
    ok(`${row.name}: no reply ended earlier than the reference`, short.length === 0,
        short.length ? `${short.length} replies were shorter: ` +
            `${JSON.stringify(short.slice(0, 3))}` : '');
}

// --- The verdict ---

function verdict(rows, reference) {
    console.log(`\n${'scenario'.padEnd(12)}${'peers'.padStart(6)}${'askers'.padStart(7)}` +
        `${'turns'.padStart(7)}${'failed'.padStart(8)}${'p50'.padStart(8)}${'p95'.padStart(8)}` +
        `${'tok/s'.padStart(8)}  failures`);
    for (const row of rows) {
        const all = row.runs.flat();
        const failed = all.filter((r) => !r.ok);
        const why = {};
        for (const f of failed) why[f.why] = (why[f.why] ?? 0) + 1;
        const ms = all.filter((r) => r.ok).map((r) => r.ms);
        const tokens = all.reduce((n, r) => n + r.ids.length, 0);
        console.log(row.name.padEnd(12) + String(row.workers).padStart(6) +
            String(row.askers).padStart(7) + String(all.length).padStart(7) +
            String(failed.length).padStart(8) + `${pct(ms, 50)}ms`.padStart(8) +
            `${pct(ms, 95)}ms`.padStart(8) +
            (tokens / (row.wall / 1000)).toFixed(0).padStart(8) +
            `  ${Object.entries(why).map(([k, v]) => `${v} ${k}`).join(', ')}`);
    }

    // What the KV pool did, from the counters Runner keeps (`this.counts` in
    // web/model/stage.js). `live` at the end of a run, compared with the
    // number of slots, shows the pressure the scenario put on the pool.
    // `evicted` alone is not a problem, since a finished chat losing its slot
    // costs nothing. `cold` is the number to read: a job that found its own
    // cache gone, so its turn needs a rebuild or fails.
    console.log();
    for (const row of rows) {
        const w = row.stats.filter((s) => s.runner);
        const sum = (k) => w.reduce((n, s) => n + (s.runner[k] ?? 0), 0);
        // VRAM per worker is shown next to the frame count: more concurrent
        // chats should raise the frames and leave the bytes unchanged.
        const vram = w.length ? Math.round(sum('vramMiB') / w.length) : 0;
        const pages = w.length
            ? `${w[0].runner.pagesTotal} pages x ${w[0].runner.pageTokens} tok` : '';
        console.log(`${row.name.padEnd(12)} KV pool: ${sum('admitted')} admitted, ` +
            `${sum('released')} released, ${sum('evicted')} evicted, ` +
            `${sum('cold')} cold (${w.length} workers, ${vram} MiB each, ${pages})`);
    }

    const pressure = rows.flatMap((row) => row.stats.filter((s) => s.runner)
        .map((s) => ({ row: row.name, id: s.id, live: s.runner.live, slots: s.runner.slots,
                       logs: s.runner.logs, vram: s.runner.vramMiB })));
    // Per-session state a worker still holds after every asker has stopped.
    // `logs` is pruned with the KV pool, so it is bounded by `slots`. A worker
    // with more boundary logs than slots is keeping rows for sessions whose
    // caches are gone.
    const leaky = pressure.filter((p) => p.logs > p.slots);
    if (leaky.length) {
        console.log(`\n${leaky.length} workers hold more boundary logs than KV slots: ` +
            `${JSON.stringify(leaky.slice(0, 4))}`);
    }

    const full = pressure.filter((p) => p.live >= p.slots);
    if (full.length) {
        console.log(`\n${full.length} of ${pressure.length} workers finished with every KV slot ` +
            `in use (slots=${pressure[0].slots}, web/split.js's SESSIONS)`);
    }

    console.log(`\n${pass} passed, ${fail} failed`);
    if (fail) {
        console.log(`reproduce with:  node tools/mesh.gpu.test.mjs --wire ${WIRE} ` +
            `--prompts ${PROMPTS} --tokens ${TOKENS} --seed ${SEED}`);
    }
}

await main();
process.exit(fail ? 1 : 0);
