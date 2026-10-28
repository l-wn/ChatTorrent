// A peer in the mesh: the DOM layer around one Engine (web/engine.js), which
// joins the mesh, holds and grows a range, and generates tokens. This file
// boots the page, wires events, picks the chat format, and renders the
// transcript, the chat list, the health line, the Contribute drawer and the
// piece bar. The page calls a mesh a "cluster" and a peer a "computer".

import { dimsFromManifest } from './model/loader.js';
import { Engine, ContextFullError } from './engine.js';
import { positions, coverage, holders, holding, describeRange, storedBytes } from './split.js';
import { chatFormat, stopRe } from './prompt.js';
import { runTurn } from './turn.js';
import { makeMessage } from './message.js';
import { summarize, formatLine, formatText } from './profile.js';
import { readInvite, openRoom, inviteUrl, cleanName, NAME_MAX, nameKey, creatorKey, chatsKey, sessionsKey, ctxKey } from './room.js';
import { saveChoice } from './welcome.js';
import { PROBLEMS, classify, report } from './errors.js';
import { openHistory, idbBackend, memoryBackend, newRecord, groupByDay } from './history.js';

// The turn wrapper and the regex that cuts a runaway reply, from the
// manifest's chat_format (see prompt.js). The plain transcript until the
// manifest loads.
let CHAT = 'transcript';
let TURN_RE = stopRe(CHAT);

let engine = null;
const state = { busy: false };

// This tab's invite secret, or null in the global mesh. Kept only to build the
// invite link (see web/room.js).
let secret = null;

// The room this tab joined (null before boot or after a bad link), and
// whether this tab lends its GPU (the welcome dialog's answer).
let room = null;
let lending = false;

// Stored bytes of each position of the model, from the manifest, for the
// "needs about N GB more" line. Null until the manifest is read or when it
// lists no files.
let posBytes = null;

// The checkpoint the mesh runs, set once the manifest is read. A saved chat's
// token ids are only valid for the tokenizer they were made with.
let running = null;       // { model, weights }

// --- Chats ---
//
// The chat on screen is `current`: its saved record (web/history.js; null
// until its first message is sent) and its conversation in the engine (null
// until the engine exists). The transcript is rendered from `rec.messages`.
// Notes shown around it belong to this tab and are not saved. A chat opened in
// this tab keeps its conversation in `convs`, so switching back to it reaches
// the caches the peers hold for it.
let store = null;         // this mesh's chats (see openStore())
let current = { rec: null, conv: null };
const convs = new Map();  // record id -> conversation

// --- DOM ---

const ui = {};
const cacheDom = () => {
    for (const el of document.querySelectorAll('[id]')) ui[el.id] = el;
};

// This tab's own state, at the end of the health line: "ready",
// "generating… 12 tok · 4000 left", a rebuild. tools/load.mjs reads it.
function setStatus(text, kind = '') {
    ui.status.textContent = text;
    ui.status.className = kind;
    // Shown only during a reply and on an error: otherwise the health line
    // already says it.
    ui.status.hidden = !(state.busy || kind === 'err');
}

function addMessage(who, text = '', cls = '') {
    const msg = makeMessage(who, text, cls);
    // In an empty chat, notes go above the invite card and example prompts.
    const empty = /\bnote\b/.test(cls) && ui.messages.querySelector('.share, .suggest');
    if (empty) ui.messages.insertBefore(msg.el, empty);
    else ui.messages.appendChild(msg.el);
    scrollDown(true);
    return msg;
}

const note = (text, err = false) => addMessage('', text, `note${err ? ' err' : ''}`);

// A problem in the chat: its code, one sentence and what to do (errors.js).
// The raw error goes to the console. tools/load.mjs reads data-code.
function problem(key, err = null) {
    // A calm one is about this browser, not the chat: a line above the input.
    if (PROBLEMS[key]?.calm) {
        ui.composerNote.textContent = report(key, err);
        ui.composerNote.hidden = false;
        return null;
    }
    const msg = note(report(key, err), true);
    msg.el.dataset.code = PROBLEMS[key]?.code ?? '';
    return msg;
}

// Growth that stopped: the GPU being full is the normal end, shown as a
// limit; anything else is a problem.
function grewTo(cause, text) {
    if (text) console.warn(text);
    if (classify(cause, '') === 'GPU_FULL') note('Limit reached: your GPU is full. It keeps lending what it holds.');
    else problem('GROW_FAILED', cause);
}

// Why lending could not start.
function lendProblem(cause) {
    const m = String(cause?.message ?? cause ?? '');
    return !navigator.gpu || classify(cause, '') === 'NO_WEBGPU' ? 'NO_WEBGPU'
        : /checks on .* failed on this GPU/.test(m) ? 'GPU_CHECK'
        : 'LEND_FAILED';
}

// Insert `msg` above the reply being streamed, if there is one. A rebuild
// happens before the reply it delays.
function beforeReply(msg) {
    const reply = ui.messages.querySelector('.msg.streaming');
    if (reply) ui.messages.insertBefore(msg.el, reply);
}

function scrollDown(force = false) {
    const el = ui.transcript;
    if (force || el.scrollHeight - el.scrollTop - el.clientHeight < 80) {
        el.scrollTop = el.scrollHeight;
    }
}

function setBusy(busy) {
    state.busy = busy;
    for (const el of [ui.input, ui.send]) el.disabled = busy;
    lockChats(busy);
    ui.send.hidden = busy;
    ui.stop.hidden = !busy;
    if (!busy) ui.input.focus();
}

// Engine events, translated into DOM updates.
function handleEngineEvent(type, detail) {
    switch (type) {
        case 'change':
        case 'grow':
            refresh();
            break;
        case 'progress':
            ui.bootnote.textContent = detail.text;
            if (detail.fraction != null) ui.bootbar.style.width = `${detail.fraction * 100}%`;
            break;
        // Engine notes by code: limits and problems are shown; the rest
        // (retries, the MTP head, growth resuming) only go to the console.
        case 'status':
            switch (detail.code) {
                case 'limit':
                    console.info(detail.text);
                    note(`Limit reached: you're lending your full ${Math.round(detail.limit / 2 ** 30)} GB.`);
                    break;
                case 'grow-stopped':
                    grewTo(detail.cause, detail.text);
                    break;
                case 'whole':
                    note("You're holding the whole model.");
                    break;
                case 'gpu-lost':
                    console.error(detail.text);
                    problem('GPU_LOST', detail.cause);
                    setStatus('GPU stopped', 'err');
                    break;
                // The GPU ran out of memory running everything it held. The
                // engine reloads a smaller part on its own.
                case 'gpu-recovering':
                    console.warn(detail.text);
                    note('Your GPU ran out of memory. Reloading a smaller part of the model…');
                    break;
                case 'gpu-recovered':
                    console.info(detail.text);
                    note('Your GPU is lending again, with less of the model.');
                    break;
                case 'gpu-error':
                    console.error(detail.text);
                    problem('GPU_ERROR', detail.cause);
                    break;
                default:
                    console.info(detail.text);
            }
            break;
        // A rebuild of the context from this chat's token ids. The message
        // depends on the reason (route.js's rebuild()).
        case 'repair': {
            // A repair is usually a computer that cleared this chat from its
            // memory (room for other chats, or the cluster rearranged), not
            // one that left. The error text says which.
            const left = detail.reason === 'repair' && LEFT_RE.test(detail.message ?? '');
            rebuilding = left ? REBUILDS.left : REBUILDS[detail.reason] ?? REBUILDS.repair;
            if (detail.reason === 'resume') {
                beforeReply(note('Picking this chat back up…'));
            } else if (detail.reason !== 'rewind') {
                console.info('rebuild:', detail.message);
                beforeReply(note(left
                    ? 'Someone running this chat left. Picking it back up on the computers still here…'
                    : 'A computer running this chat cleared it from its memory. Re-reading the conversation to continue…'));
            }
            setStatus(`${rebuilding}…`, 'busy');
            break;
        }
        case 'rebuild-progress':
            setStatus(`${rebuilding}… (${detail.sent}/${detail.total})`, 'busy');
            break;
        case 'repaired':
            setStatus('generating…', 'busy');
            break;
        // No status change: when a 'fix' arrives the local repair is done and
        // the job is in flight. Shown to explain the pause.
        case 'fix':
            console.info(`fix: peer ${detail.from} replayed ${detail.rows} tokens`);
            note('Part of this chat was restored from a neighbouring computer.');
            break;
    }
}

// Status pill text while a rebuild runs, by reason.
const REBUILDS = {
    left: 'someone left, rebuilding context',
    repair: 're-reading context',
    resume: 'resuming chat',
    rewind: 're-reading context',
};
let rebuilding = REBUILDS.repair;
// Errors that mean a computer left or stopped answering (route.js fail() and
// the job timeout), as opposed to one that dropped this chat's cache.
const LEFT_RE = /is unreachable|no peer that can run|no answer from the cluster/;

// A prompt can be sent when there is a tokenizer and net.connected is true:
// the ready peers' ranges cover the model, a job of mine can enter, my own
// hand-off is open, and an answer can come back. What this tab holds does not
// matter.
function refresh() {
    const net = engine.net;

    // The Contribute drawer's numbers. #covered and #role keep the formats
    // tools/load.mjs reads.
    const cov = net.coverage;
    ui.covered.textContent = net.positions
        ? `${cov.covered}/${cov.total} held` : 'model unknown';
    ui.covered.className = !net.positions ? '' : cov.covered === cov.total ? 'ok'
        : cov.covered ? 'busy' : 'err';
    ui.covered.title = net.positions ? summarise(cov, net) : '';

    ui.role.textContent = roleLine(net);

    const loading = net.peers.filter((p) => holding(p) && (p.have ?? p.hi) < p.hi).length;
    ui.peers.textContent = `${net.peers.length}${loading ? ` (${loading} downloading)` : ''}`
        + (net.online ? '' : ' · offline');

    // The fewest ready computers holding any one piece. 0 cannot run, 1 runs
    // but stops if that computer leaves, 2 survives anyone leaving.
    const r = cov.r;
    ui.depth.textContent = String(r);
    ui.depth.className = r >= 2 ? 'ok' : r === 1 ? 'busy' : 'err';

    // This tab's own GPU memory, not the cluster's.
    ui.vram.textContent = engine.runner ? `${(engine.vramBytes / 2 ** 30).toFixed(2)} GiB` : '—';
    ui.ran.textContent = `${engine.ran.toLocaleString()} tokens`;
    ui.ctx.textContent = `${engine.ctxLen.toLocaleString()} / ${engine.ctxMax.toLocaleString()} tokens`;

    renderMap(net, cov);
    renderLanes(net);
    showHealth(net, cov);
    showLending();
    if (net.positions) {
        // GB of the model held, of the whole model.
        let held = 0, total = 0;
        for (let x = 0; x < cov.total; x++) {
            const b = posBytes?.[x] ?? (27 * 2 ** 30) / cov.total;
            total += b;
            if (cov.depths[x]) held += b;
        }
        ui.modelHeld.textContent = `${Math.round((cov.covered / cov.total) * 100)}%`;
        ui.modelGb.textContent = `(${Math.round(held / 2 ** 30)}/${Math.round(total / 2 ** 30)} GB)`;
    } else {
        ui.modelHeld.textContent = '—';
        ui.modelGb.textContent = '';
    }
    ui.map.classList.toggle('full', !!net.positions && cov.covered === cov.total);
    // For the setup screen (welcome.js): the pieces, the other computers'
    // claims, and how many pieces this tab would hold at a given limit.
    document.dispatchEvent(new CustomEvent('cluster', {
        detail: {
            L: net.positions,
            peers: net.peers.filter((p) => p.id !== net.id).map(({ lo, hi, have }) => ({ lo, hi, have })),
            floor: net.minBlocks,
            // The lobby: which cluster, who else is here, and whether it runs.
            name: !room ? 'No cluster' : room.private ? `Private cluster · ${room.tag}` : 'Public demo',
            crowd: net.peers.filter((p) => p.id !== net.id)
                .map((p) => ({ name: p.name ?? `GPU ${p.id}`, lending: holding(p) })),
            running: !!net.positions && cov.covered === cov.total,
            needGb: Math.ceil(missingBytes(cov) / 2 ** 30),
            modelGb: (posBytes ? posBytes.reduce((x, y) => x + y, 0) : 27 * 2 ** 30) / 2 ** 30,
            pieceGb: posBytes ? posBytes.map((b) => b / 2 ** 30) : null,
            capacity: (gb) => engine.capacityFor(gb * 2 ** 30),
        },
    }));

    ui.authfail.hidden = !net.authFailures;
    ui.authfail.textContent = `${net.authFailures} refused link${net.authFailures === 1 ? '' : 's'}`;

    const ok = engine.canPrompt && net.connected && net.online && !net.refused && !foreign();
    if (state.busy) return;

    // Not welcomed yet and retrying (net.js): show the reason in the boot box.
    if (net.trouble && engine.canPrompt === null && ui.boot?.isConnected) {
        ui.bootnote.textContent = net.trouble;
    }

    const why = foreign() ? 'read-only: made on another model'
        : net.refused ? net.refused
        : engine.canPrompt === null && !net.online && net.trouble ? 'signalling offline, reconnecting…'
        : net.relayOnly && !net.relay ? 'no relay server: links cannot open'
        : engine.canPrompt === null ? 'starting up…'
        : !engine.canPrompt ? 'no tokenizer'
        : !net.online ? 'signalling offline, reconnecting…'
        : !net.staffed ? missing(cov, net)
        : 'linking…';
    setStatus(ok ? 'ready' : why, ok ? 'ok' : net.refused ? 'err' : 'busy');
    ui.input.disabled = !ok;
    ui.send.disabled = !ok;
    const short = Math.ceil(missingBytes(cov) / 2 ** 30);
    ui.input.placeholder = ok ? 'Ask anything'
        : net.online && net.positions && !net.staffed
            ? (short ? `Can't chat yet: needs ${short} GB more. Invite people, or lend yours.`
                : "Can't chat yet: waiting for more people to lend.")
            : 'Waiting for the cluster…';
}

// --- The health line ---
//
// One sentence on whether the cluster can run and whether it survives someone
// leaving, e.g. "Running · 3 GPUs · safe if one person leaves" or "Needs about
// 6 GB more to start".

const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function showHealth(net, cov) {
    let text, kind;
    if (net.refused) {
        text = net.refused.replace(/\bmesh\b/g, 'cluster');
        text = text[0].toUpperCase() + text.slice(1);
        kind = 'err';
    } else if (!net.online) {
        text = 'Reconnecting…';
        kind = 'busy';
    } else if (!net.positions) {
        text = 'Joining…';
        kind = '';
    } else if (cov.covered < cov.total) {
        const need = missingBytes(cov);
        text = need ? `Needs ${Math.max(1, Math.ceil(need / 2 ** 30))} GB more` : 'Waiting for more people';
        kind = 'busy';
    } else {
        text = 'Running';
        kind = 'ok';
    }
    if (ui.health.textContent !== text) ui.health.textContent = text;
    ui.health.dataset.state = kind;
}

// Stored bytes of the pieces nobody ready holds, or 0 when unknown.
function missingBytes(cov) {
    if (!posBytes) return 0;
    let n = 0;
    for (const [lo, hi] of cov.holes) for (let x = lo; x < hi; x++) n += posBytes[x] ?? 0;
    return n;
}

// Bytes per position (the embedding, each block, the head) from the manifest.
function positionBytes(manifest) {
    const b = storedBytes(manifest);
    if (!b) return null;
    return [b.wte, ...b.blocks, b.norm + b.lm_head];
}

// E.g. `peer 3 · the embedding + blocks 0–5 (+3 loading)`: id, holding, and
// how much of the claim is still loading, from this tab's own roster entry.
function roleLine(net) {
    const id = net.id ?? '—';
    if (engine.role === 'spectator') return `peer ${id} · spectator`;

    const me = net.me;
    if (!holding(me)) return `peer ${id} · joining…`;

    const have = me.have ?? me.hi;
    // Loaded but not announced: its links to neighbouring peers are still
    // opening (net.js's ready gate).
    const waiting = net.linking;
    if (waiting) {
        return `peer ${id} · linking to peer${waiting.length === 1 ? '' : 's'} ${waiting.join(', ')}…`;
    }
    if (have <= me.lo) return `peer ${id} · loading ${describeRange(me.lo, me.hi, net.blocks)}`;
    const left = me.hi - have;
    return `peer ${id} · ${describeRange(me.lo, have, net.blocks)}`
        + (left ? ` (+${left} loading)` : '');
}

// --- The block map ---
//
// One cell per position, left to right: the embedding, every block, the head.
// Shows what the network holds, what this tab holds, and what is loading.
const map = {
    n: 0,          // positions the strip is built for
    cells: [],
    seen: null,    // whether each position has been covered since this tab opened
};

const posName = (x, L) => x === 0 ? 'the embedding' : x === L + 1 ? 'the head' : `block ${x - 1}`;

function buildMap(n, L) {
    map.n = n;
    map.cells = [];
    map.seen = new Uint8Array(n);
    ui.map.replaceChildren();
    for (let x = 0; x < n; x++) {
        const cell = document.createElement('i');
        cell.className = 'cell';
        if (x === 0 || x === n - 1) cell.classList.add('end');
        cell.dataset.name = posName(x, L);
        cell.style.setProperty('--i', x);
        ui.map.appendChild(cell);
        map.cells.push(cell);
    }
}

// Redraw from the roster.
function renderMap(net, cov) {
    const n = net.positions;
    if (!n) return;
    if (map.n !== n) buildMap(n, net.blocks);

    const claimed = coverage(net.peers.filter(holding), n);
    const me = net.me;
    const mineTo = me?.lo == null ? -1 : (me.have ?? me.hi);

    for (let x = 0; x < n; x++) {
        const d = cov.depths[x];
        const cell = map.cells[x];

        if (d > 0 && !map.seen[x]) {
            map.seen[x] = 1;
            cell.classList.remove('lit');
            void cell.offsetWidth;                       // restart the animation
            cell.classList.add('lit');
        }

        const look = d >= 2 ? 'deep' : d === 1 ? 'thin'
            : claimed.depths[x] ? 'loading' : map.seen[x] ? 'lost' : 'empty';
        if (cell.dataset.state !== look) cell.dataset.state = look;
        cell.classList.toggle('mine', x >= (me?.lo ?? 0) && x < mineTo);

        const who = holders(net.readyPeers, x);
        cell.title = `${cell.dataset.name}: `
            + (who.length ? `held by ${plural(who.length, 'computer')} (${who.join(', ')})`
                : claimed.depths[x] ? 'downloading' : 'missing');
    }
}

// --- The lanes: one line per computer, under the pieces it holds ---
//
// Keyed by peer id, so a line stays put while its numbers change, slides in
// when a computer joins and fades out when it leaves. `hopMs` is each
// computer's time per word on this tab's last reply; `firstSeen` is when this
// tab first saw it.

const hopMs = new Map();
const firstSeen = new Map();
const lanes = new Map();       // id -> { el, bar, have, label, detail }
let openLane = null;

const LANES_KEY = 'chattorrent.lanes';
// Collapsed until opened; the choice is remembered.
let lanesShown = (() => { try { return localStorage.getItem(LANES_KEY) === '1'; } catch { return false; } })();

function wireLanes() {
    const sync = () => {
        ui.lanes.hidden = !lanesShown;
        ui.chatters.hidden = !lanesShown || !ui.chattersIcons.childElementCount;
        ui.lanesBtn.setAttribute('aria-expanded', String(lanesShown));
    };
    ui.lanesBtn.addEventListener('click', () => {
        lanesShown = !lanesShown;
        try { localStorage.setItem(LANES_KEY, lanesShown ? '1' : '0'); } catch { /* not kept */ }
        sync();
    });
    sync();
}

const ago = (ms) => {
    const m = Math.round(ms / 60000);
    return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
};

function laneBytes(net, lo, hi) {
    let b = 0;
    for (let x = lo; x < hi; x++) b += posBytes ? posBytes[x] ?? 0 : (27 * 2 ** 30) / (net.positions || 1);
    return b;
}

function makeLane(id) {
    const el = document.createElement('div');
    el.className = 'lane entering';
    el.setAttribute('role', 'listitem');
    el.innerHTML = '<button type="button" class="lane-track lane-grid">' +
        '<span class="lane-name"></span>' +
        '<span class="lane-rail"><span class="lane-bar"><span class="lane-have"></span></span></span>' +
        '<span class="lane-stat"></span>' +
        '</button><div class="lane-detail" hidden></div>';
    const lane = {
        el,
        bar: el.querySelector('.lane-bar'),
        have: el.querySelector('.lane-have'),
        name: el.querySelector('.lane-name'),
        stat: el.querySelector('.lane-stat'),
        detail: el.querySelector('.lane-detail'),
    };
    el.querySelector('.lane-track').addEventListener('click', () => {
        openLane = openLane === id ? null : id;
        for (const [k, l] of lanes) {
            l.detail.hidden = k !== openLane;
            l.el.classList.toggle('open', k === openLane);
        }
    });
    requestAnimationFrame(() => el.classList.remove('entering'));
    return lane;
}

function renderLanes(net) {
    const L = net.positions;
    if (!L) return;
    const now = performance.now();
    const holders = net.peers.filter(holding).sort((a, b) => (a.lo - b.lo) || (a.id - b.id));
    const ids = new Set(holders.map((p) => p.id));
    for (const p of net.peers) if (!firstSeen.has(p.id)) firstSeen.set(p.id, now);

    // Lines for computers that left fade out, then go.
    for (const [id, lane] of lanes) {
        if (ids.has(id) || lane.el.classList.contains('leaving')) continue;
        lane.el.classList.add('leaving');
        setTimeout(() => { lane.el.remove(); lanes.delete(id); }, 400);
        hopMs.delete(id);
    }

    let prev = null;
    for (const p of holders) {
        let lane = lanes.get(p.id);
        if (!lane || lane.el.classList.contains('leaving')) {
            lane?.el.remove();
            lane = makeLane(p.id);
            lanes.set(p.id, lane);
        }
        // Keep the order: sorted by where each range starts.
        const after = prev ? prev.el.nextSibling : ui.lanes.firstChild;
        if (lane.el !== after) ui.lanes.insertBefore(lane.el, after);
        prev = lane;

        const me = p.id === net.id;
        const have = Math.min(p.have ?? p.hi, p.hi);
        const name = `${p.name ?? `GPU ${p.id}`}${me ? ' (you)' : ''}`;
        const ms = hopMs.get(p.id);
        const loading = have < p.hi;
        const pct = Math.round(((have - p.lo) / (p.hi - p.lo)) * 100);
        const gb = laneBytes(net, p.lo, p.hi) / 2 ** 30;
        const stat = loading ? `${pct}%` : ms != null ? `${Math.round(ms)} ms` : `${gb.toFixed(1)} GB`;

        lane.el.classList.toggle('me', me);
        lane.el.classList.toggle('loading', loading);
        lane.bar.style.left = `${(p.lo / L) * 100}%`;
        lane.bar.style.width = `${((p.hi - p.lo) / L) * 100}%`;
        lane.have.style.width = `${((have - p.lo) / (p.hi - p.lo)) * 100}%`;

        if (lane.name.textContent !== name) lane.name.textContent = name;
        if (lane.stat.textContent !== stat) lane.stat.textContent = stat;
        lane.stat.title = loading ? 'Downloaded so far' : ms != null ? 'Time per word on your last reply' : 'Size of its pieces';

        const held = describeRange(p.lo, have, net.blocks) || 'nothing yet';
        const lines = [
            `Holds ${held}${loading ? ` (${pct}% downloaded)` : ''}`,
            `${gb.toFixed(1)} GB of the model · joined ${ago(now - firstSeen.get(p.id))}`,
        ];
        if (ms != null) lines.push(`${Math.round(ms)} ms per word on your last reply`);
        const detail = lines.join('\n');
        if (lane.detail.textContent !== detail) lane.detail.textContent = detail;
    }

    // Everyone only chatting: one person icon each, under the box.
    const chatters = net.peers.filter((p) => !holding(p));
    ui.chatters.hidden = !chatters.length || !lanesShown;
    ui.chattersLabel.textContent = `${chatters.length} just chatting`;
    ui.chattersLabel.title = `${plural(chatters.length, 'person', 'people')} chatting without lending a GPU`;
    const sig = chatters.map((p) => p.id).join(',');
    if (ui.chattersIcons.dataset.sig !== sig) {
        ui.chattersIcons.dataset.sig = sig;
        ui.chattersIcons.replaceChildren(...chatters.slice(0, 60).map((p) => {
            const i = icon(PERSON, 'person');
            if (p.id === net.id) i.classList.add('me');
            const who = `${p.name ?? `person ${p.id}`}${p.id === net.id ? ' (you)' : ''}`;
            i.setAttribute('aria-label', who);
            i.insertAdjacentHTML('afterbegin', '<title></title>');
            i.querySelector('title').textContent = who;
            return i;
        }));
        if (chatters.length > 60) ui.chattersIcons.append(`+${chatters.length - 60}`);
    }

    // The toggle: computers lending, and everyone in the cluster.
    const label = `${holders.length}|${net.peers.length}`;
    if (ui.lanesBtn.dataset.sig !== label) {
        ui.lanesBtn.dataset.sig = label;
        const everyone = document.createElement('span');
        everyone.className = 'count';
        everyone.textContent = plural(net.peers.length, 'person', 'people');
        const computers = document.createElement('span');
        computers.textContent = plural(holders.length, 'computer');
        ui.lanesBtn.replaceChildren(icon(PERSON), everyone, icon(MONITOR), computers);
        ui.lanesBtn.title = `${plural(net.peers.length, 'person', 'people')} in this cluster, ` +
            `${plural(holders.length, 'computer')} lending a GPU`;
    }
}

// Small line icons for the lanes.
const MONITOR = '<rect x="2" y="3" width="20" height="13" rx="2"/><path d="M8 21h8M12 16v5"/>';
const PERSON = '<circle cx="12" cy="7" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>';

function icon(paths, cls = '') {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '14');
    svg.setAttribute('height', '14');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    if (cls) svg.setAttribute('class', cls);
    svg.innerHTML = paths;
    return svg;
}

// The status text for why a prompt cannot be sent.
function missing(cov, net) {
    if (!cov.holes.length || !cov.covered) return 'waiting for peers…';
    const first = describeRange(...cov.holes[0], net.blocks);
    const more = cov.holes.length - 1;
    return `waiting for ${first}${more ? ` and ${more} more stretch${more > 1 ? 'es' : ''}` : ''}…`;
}

// Coverage in words, for the pill's tooltip.
function summarise(cov, net) {
    const L = net.blocks;
    const held = runs(cov.depths, (d) => d > 0).map(([a, b]) => describeRange(a, b, L));
    const unheld = cov.holes.map(([a, b]) => describeRange(a, b, L));
    return [held.length ? `held: ${held.join(', ')}` : 'nothing held',
            unheld.length ? `unheld: ${unheld.join(', ')}` : null,
            `r ${cov.r}`].filter(Boolean).join(' · ');
}

// Maximal runs of positions satisfying `pred`, as half-open ranges.
function runs(depths, pred) {
    const out = [];
    for (let x = 0; x < depths.length; x++) {
        if (!pred(depths[x])) continue;
        const lo = x;
        while (x + 1 < depths.length && pred(depths[x + 1])) x++;
        out.push([lo, x + 1]);
    }
    return out;
}

// --- Boot ---

// connect() takes a second or two, and the composer opens when it resolves.
// serve() (WebGPU, kernel checks, a range, the download) takes minutes and
// runs unawaited behind the open composer. `gpuConsent` resolves with
// { contribute, limit }, the welcome popup's answer. The default serves with
// no limit.
export async function boot(gpuConsent = Promise.resolve({ contribute: true, limit: '' })) {
    cacheDom();
    wireEvents();

    // The mesh comes from the URL fragment. A damaged invite stops here,
    // before any socket opens. Falling back to the global mesh would put
    // someone expecting a private mesh in the public one.
    try {
        secret = readInvite(location.hash);
        room = await openRoom(secret);
    } catch (err) {
        ui.title.textContent = 'ClusterChat';
        ui.boot.remove();
        showMesh(null);
        problem('BAD_LINK', err);
        setStatus('bad invite link', 'err');
        return;
    }
    showMesh(room);
    showSide();
    openStore(room);

    ui.bootnote.textContent = 'connecting to the signaling server…';
    // Holding the head, also hold the MTP draft head; ?mtp=0 turns it off.
    // ?warm=0: no keep-warm kernel between jobs (stage.js KEEP_WARM_MS).
    const params = new URLSearchParams(location.search);
    engine = new Engine({ onEvent: handleEngineEvent,
                          mtp: params.get('mtp') !== '0',
                          warm: params.get('warm') !== '0' });

    let manifest;
    try {
        ({ manifest } = await engine.connect({ room, relayOnly: !directParam(),
                                               chats: creatorChats(secret),
                                               sessions: creatorSessions(secret),
                                               ctx: creatorCtx(secret) }));
    } catch (err) {
        console.warn('connect failed:', err);
        ui.title.textContent = 'ClusterChat';
        ui.boot.remove();
        problem('NO_CONNECTION', err);
        refresh();
        return;
    }
    showChatLimit();

    CHAT = chatFormat(manifest);
    TURN_RE = stopRe(CHAT);
    running = { model: manifest.model ?? null, weights: engine.net.weightsUrl ?? null };
    posBytes = positionBytes(manifest);
    showSide();
    // A chat picked from the list during connect() is given to the engine
    // here.
    attach();
    render();
    ui.thinkingLabel.hidden = CHAT !== 'qwen';
    // The model's name: beside the brand in the header, and heading its bar
    // on the pre-join card.
    const modelName = manifest.model.split('/').pop();
    ui.model.textContent = modelName;
    ui.model.title = `model: ${manifest.model}`;
    ui.model.hidden = false;
    ui.stripModel.title = manifest.model;
    ui.modelLabel.textContent = modelName;

    if (!engine.canPrompt) {
        problem('NO_TOKENIZER');
    }

    if (engine.net.relayOnly && !engine.net.relay) {
        // For whoever runs the server: TURN_KEY_ID and TURN_API_TOKEN are
        // unset, and this tab only links through a relay. ?direct=1 links
        // directly instead, showing its address to the peers it links to.
        problem('NO_RELAY', 'the signaling server offers no TURN relay (set TURN_KEY_ID and TURN_API_TOKEN, or open with ?direct=1)');
    }


    refresh();
    if (engine.canPrompt) ui.input.focus();

    // `?at=` is validated before opening a GPU, so a typo is reported and does
    // not fall back to ordinary placement.
    let at;
    try {
        at = placement(manifest);
    } catch (err) {
        console.warn('placement:', err);
        ui.boot.remove();
        problem('LEND_FAILED', err);
        return;
    }

    // The GPU is opened only after the visitor agrees in the welcome popup,
    // and only up to the amount they chose.
    ui.bootwhat.textContent = 'Waiting for your answer…';
    ui.bootnote.textContent = 'about lending your GPU';
    const { contribute, limit } = await gpuConsent;
    ui.vramlimit.value = limit;
    if (!contribute) {
        ui.boot.remove();
        lending = false;
        refresh();
        return;
    }
    lending = true;
    ui.bootwhat.textContent = 'Lending your GPU…';

    applyVramLimit();
    // Not awaited. serve() reports its own failures; the catch is for bugs.
    engine.serve({ blocks: blocksParam(), vram: vramParam(), at, slowMs: slowMs() })
        .then(({ note: msg, code, cause }) => {
            ui.boot.remove();
            if (code === 'grow-stopped') grewTo(cause, msg);
            else if (code === 'lend-failed') { console.warn(msg); problem(lendProblem(cause), cause); }
            else console.info(msg);
            refresh();
        })
        .catch((err) => console.error('serve:', err));
}

// Parse the VRAM-limit input, freeform ("4gb", "512mb", "2 GiB"). Blank means
// no limit: growth then stops only at a GPU allocation failure or when the
// whole model is held. Anything else that does not parse returns null, which
// the caller reads as "leave the limit alone": the field fires on every
// keystroke, so "22gb" is typed through "22g".
function parseByteLimit(str) {
    const s = (str ?? '').trim();
    if (!s) return Infinity;
    const m = /^(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)?$/i.exec(s);
    if (!m) return null;
    const unit = (m[2] ?? 'gb').toLowerCase();
    return Number(m[1]) * (unit.startsWith('m') ? 2 ** 20 : 2 ** 30);
}

// How long the VRAM-limit field must be quiet before its value is applied.
const LIMIT_SETTLE_MS = 750;

// The field's value to the engine, skipping a value that does not parse. A
// limit that reaches the engine takes effect at the next growth step, and a
// raised one wakes a paused loop (engine.js's setVramLimit).
function applyVramLimit() {
    const bytes = parseByteLimit(ui.vramlimit.value);
    if (bytes != null) engine?.setVramLimit(bytes);
}

// `?blocks=N`: hold exactly this many positions, with no growth.
function blocksParam() {
    const q = new URLSearchParams(location.search);
    return q.has('blocks') ? Number(q.get('blocks')) : null;
}

// `?gb=N`: hold as many positions as fit in this many gigabytes, by the cost
// model in split.js.
function vramParam() {
    const q = new URLSearchParams(location.search);
    return q.has('gb') ? Number(q.get('gb')) * 2 ** 30 : null;
}

// `?at=lo-hi`: take exactly this range, not the server's placement. A
// development option for reproducing specific overlaps (see also `?slow=`).
function placement(manifest) {
    const raw = new URLSearchParams(location.search).get('at');
    if (!raw) return null;

    const d = dimsFromManifest(manifest);
    const N = engine.net.positions || positions(d.n_layer);
    const m = /^(\d+)\s*[-:]\s*(\d+)$/.exec(raw.trim());
    if (!m) throw new Error(`?at=${raw} is not a range. Write it as ?at=lo-hi, e.g. ?at=0-10`);
    const [lo, hi] = [Number(m[1]), Number(m[2])];
    if (lo >= hi) throw new Error(`?at=${raw} is empty: lo must be below hi`);
    if (hi > N) {
        throw new Error(`?at=${raw} runs past the end of the model, which has ${N} positions ` +
            `(0–${N - 1})`);
    }
    return [lo, hi];
}

// `?direct=1`: link to peers directly, not through the TURN relay. One hop
// faster, and it works with no TURN on the server (a LAN), but every peer this
// tab links to sees its IP address.
const directParam = () => new URLSearchParams(location.search).get('direct') === '1';

// `?slow=ms`: wait this long between positions while downloading.
const slowMs = () => Math.max(0, Number(new URLSearchParams(location.search).get('slow')) || 0);

// The header's cluster controls. `r` null means no cluster (a damaged
// invite); only the link to the public demo is shown.
function showMesh(r) {
    const priv = !!r?.private;
    ui.mesh.textContent = !r ? 'no cluster' : priv ? `private · ${r.tag}` : 'public demo';
    ui.mesh.className = `pill ${!r ? 'err' : priv ? 'private' : 'public'}`;
    ui.mesh.title = !r ? 'This tab has not joined any cluster.'
        : priv ? `A private cluster. Everyone in it sees the same tag (${r.tag}). Only people ` +
                 'with its link can join, and only their computers run your chats.'
        : 'The public demo: anyone can join, and their computers may run your chats. ' +
          "Don't send anything personal.";
    ui.newmesh.hidden = !r || priv;
    ui.invite.hidden = !priv;
    ui.leave.hidden = !(priv || !r);
    ui.leave.href = location.pathname;         // this page with no invite or query
    ui.leave.textContent = priv ? 'Leave' : 'Public demo';
    ui.chatsMesh.textContent = !r ? 'no cluster' : priv ? `private ${r.tag}` : 'public demo';
}

// Add the mesh's limit on replies at once to its pill, once the welcome has
// brought it.
function showChatLimit() {
    const max = engine?.net?.maxChats;
    if (max) ui.mesh.title += ` Up to ${max} replies can run at once.`;
    const kept = engine?.net?.sessions;
    if (kept) ui.mesh.title += ` Each GPU keeps ${plural(kept, 'chat')} in memory.`;
    if (engine?.net) ui.mesh.title += ` Each chat holds up to ${engine.ctxMax.toLocaleString()} tokens.`;
}

// The limit on replies at once chosen when this tab created its private mesh,
// kept in sessionStorage so the tab's reload and reconnects send it again.
// Null for an invitee and in the global mesh.
function creatorNumber(s, key) {
    if (!s) return null;
    try {
        const n = Number(sessionStorage.getItem(key(s)));
        return Number.isInteger(n) && n >= 1 ? n : null;
    } catch {
        return null;
    }
}
const creatorChats = (s) => creatorNumber(s, chatsKey);
// The chats each peer keeps in memory, chosen and kept the same way.
const creatorSessions = (s) => creatorNumber(s, sessionsKey);
// The context window, in tokens, chosen and kept the same way.
const creatorCtx = (s) => creatorNumber(s, ctxKey);

// The invite link, with this person's name if they gave one.
const myName = () => { try { return cleanName(sessionStorage.getItem(nameKey(secret))); } catch { return ''; } };
const shareUrl = () => inviteUrl(secret, location, myName());

async function copyText(text, button, label) {
    try {
        await navigator.clipboard.writeText(text);
        button.textContent = 'Copied';
    } catch {
        // The clipboard API needs a secure origin (not plain http on a LAN).
        window.prompt('Copy this invite link and send it only to people you trust:', text);
    }
    setTimeout(() => { button.textContent = label; }, 2000);
}

const copyInvite = () => copyText(shareUrl(), ui.invite, 'Invite');

function wireEvents() {
    // A new fragment names a new mesh, and a fragment change does not reload
    // the page. Reload so the engine always matches the mesh in the URL.
    // Covers Create, a hand-edited link, and Back/Forward.
    window.addEventListener('hashchange', () => location.reload());
    // Starting a private cluster happens on the create page.
    ui.newmesh.addEventListener('click', () => {
        if (state.busy && !confirm('A reply is still being written. Leave it and start a private cluster?')) return;
        location.href = '/create';
    });
    ui.invite.addEventListener('click', copyInvite);

    ui.send.addEventListener('click', () => submitPrompt());
    ui.stop.addEventListener('click', () => { engine.stop(); ui.stop.disabled = true; });
    ui.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitPrompt(); }
    });
    ui.input.addEventListener('input', () => {
        ui.input.style.height = 'auto';
        ui.input.style.height = `${ui.input.scrollHeight}px`;
    });
    // Applied once typing has settled. Clearing the field to retype a smaller
    // value passes through blank, which means "no limit" and would resume
    // growth, so an intermediate value must not reach the engine. A growth step
    // is gated by GROWTH_QUIET_MS (2 s), so the delay costs a lowered limit
    // nothing.
    let limitTimer = 0;
    ui.vramlimit.addEventListener('input', () => {
        clearTimeout(limitTimer);
        limitTimer = setTimeout(applyVramLimit, LIMIT_SETTLE_MS);
    });
    wireSettings();
    wireChats();
    wireLanes();
    wireSide();
}

// --- The settings and Contribute drawers ---
//
// Settings is presentation only: submitPrompt() reads the fields by id. Each
// field's HTML value is its default, so Reset and the gear's changed-dot
// compare against defaultValue / defaultSelected.

const settingFields = () => ui.settings.querySelectorAll('input, select');

const isDefault = (el) => el.tagName === 'SELECT'
    ? [...el.options].every((o) => o.selected === o.defaultSelected)
    : el.value === el.defaultValue;

// The right-hand drawers, by name: each has a button that opens it and a
// close button. At most one is open.
const DRAWERS = {
    settings: { el: 'settings', button: 'settingsBtn', close: 'settingsClose' },
    contrib: { el: 'contrib', button: 'contribBtn', close: 'contribClose' },
};

function openDrawer(name, open) {
    if (open) openChats(false);
    for (const [key, d] of Object.entries(DRAWERS)) {
        const show = open && key === name;
        const wasOpen = !ui[d.el].hidden;
        ui[d.el].hidden = !show;
        ui[d.button].setAttribute('aria-expanded', String(show));
        if (show) ui[d.close].focus();
        else if (wasOpen && (key === name || name === null)) ui[d.button].focus();
    }
    ui.scrim.hidden = !open;
}
const openSettings = (open) => openDrawer('settings', open);

function markChanged() {
    ui.settingsBtn.classList.toggle('changed', ![...settingFields()].every(isDefault));
}

function wireSettings() {
    for (const [name, d] of Object.entries(DRAWERS)) {
        ui[d.button].addEventListener('click', () => openDrawer(name, ui[d.el].hidden));
        ui[d.close].addEventListener('click', () => openDrawer(name, false));
    }
    ui.scrim.addEventListener('click', () => openDrawer(null, false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && Object.values(DRAWERS).some((d) => !ui[d.el].hidden)) openDrawer(null, false);
    });
    ui.settings.addEventListener('input', markChanged);
    ui.settings.addEventListener('change', markChanged);
    ui.settingsReset.addEventListener('click', () => {
        for (const el of settingFields()) {
            if (el.tagName === 'SELECT') for (const o of el.options) o.selected = o.defaultSelected;
            else el.value = el.defaultValue;
        }
        markChanged();
    });

    // Lending is decided at boot, so starting or stopping saves the choice
    // and reloads. Chats are saved, so the chat on screen comes back.
    const relend = (contribute) => {
        if (state.busy && !confirm('A reply is still being written. Stop it and reload?')) return;
        saveChoice({ contribute, limit: ui.vramlimit.value.trim() });
        location.reload();
    };
    ui.lendStart.addEventListener('click', () => relend(true));
    ui.lendStop.addEventListener('click', () => relend(false));
    ui.vramlimit.addEventListener('change', () => {
        if (parseByteLimit(ui.vramlimit.value) != null) saveChoice({ contribute: lending, limit: ui.vramlimit.value.trim() });
    });
}

// The Contribute drawer's lending state, and the live square on its button.
function showLending() {
    const canLend = !!navigator.gpu;
    const me = engine?.net?.me;
    const holds = lending && holding(me);
    const limit = ui.vramlimit.value.trim();
    ui.contribBtn.classList.toggle('lending', holds);
    ui.lendStart.hidden = lending || !canLend;
    ui.lendStop.hidden = !lending;
    ui.contribState.textContent = !canLend && !lending
        ? "This browser can't lend its GPU. Chatting works as normal."
        : !lending ? 'Not lending. You can still chat.'
        : !holds ? `Starting to lend${limit ? ` up to ${limit.toUpperCase()}` : ''}…`
        : `Lending ${limit ? `up to ${limit.toUpperCase()}` : 'with no limit'}. ` +
          `Thank you: this keeps the cluster running.`;
}

// --- The chat list ---
//
// Below 1100px, a modal drawer. At 1100px and up, a sidebar the header button
// docks and undocks. The docked state is saved in localStorage.

const wide = window.matchMedia('(min-width: 1100px)');
const DOCK_KEY = 'chattorrent.chats.docked';
let docked = (() => { try { return localStorage.getItem(DOCK_KEY) !== '0'; } catch { return true; } })();
const isDocked = () => wide.matches && docked;

// Chats cannot be switched during a reply. The engine refuses it too.
function lockChats(busy) {
    for (const el of [ui.chatsNew, ui.chatsForget, ...ui.chatList.querySelectorAll('button')]) {
        el.disabled = busy;
    }
}

function layoutChats() {
    const dock = isDocked();
    document.body.classList.toggle('chats-docked', dock);
    ui.chats.classList.toggle('docked', dock);
    if (dock) {
        ui.chats.hidden = false;
        ui.chats.setAttribute('role', 'complementary');
        ui.chats.removeAttribute('aria-modal');
        if (ui.settings.hidden && ui.contrib.hidden) ui.scrim.hidden = true;
    } else {
        ui.chats.setAttribute('role', 'dialog');
        ui.chats.setAttribute('aria-modal', 'true');
        if (ui.scrim.hidden) ui.chats.hidden = true;
    }
    ui.chatsBtn.setAttribute('aria-expanded', String(!ui.chats.hidden));
}

// Open or close the modal drawer on a narrow screen. No-op when docked.
function openChats(open) {
    if (isDocked()) return;
    if (open) openSettings(false);
    if (!open && ui.chats.hidden) return;
    ui.chats.hidden = ui.scrim.hidden = !open;
    ui.chatsBtn.setAttribute('aria-expanded', String(open));
    (open ? ui.chatsClose : ui.chatsBtn).focus();
}

function wireChats() {
    ui.chatsBtn.addEventListener('click', () => {
        if (!wide.matches) return openChats(ui.chats.hidden);
        docked = !docked;
        try { localStorage.setItem(DOCK_KEY, docked ? '1' : '0'); } catch { /* not kept */ }
        layoutChats();
    });
    ui.chatsClose.addEventListener('click', () => {
        if (!isDocked()) return openChats(false);
        docked = false;
        try { localStorage.setItem(DOCK_KEY, '0'); } catch { /* not kept */ }
        layoutChats();
    });
    ui.scrim.addEventListener('click', () => openChats(false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !ui.chats.hidden && !isDocked()) openChats(false);
    });
    wide.addEventListener('change', layoutChats);
    ui.chatsNew.addEventListener('click', () => showChat(null));
    ui.chatsForget.addEventListener('click', forgetChats);
    layoutChats();
}

// Open this mesh's chats from IndexedDB. Where the browser denies IndexedDB
// (some private windows), they are kept in memory for this tab only.
async function openStore(room) {
    const channel = 'BroadcastChannel' in window ? new BroadcastChannel('chattorrent-history') : null;
    const open = (backend) => openHistory({ mesh: room.historyTag, backend, channel });
    try {
        store = open(idbBackend());
        await store.list();
    } catch (err) {
        console.warn('chat storage:', err);
        store?.close();
        store = open(memoryBackend());
        note('This browser will not let the page save chats, so they last only as long as ' +
            'this tab.', true);
    }
    store.onChange(onStoreChange);
    renderList();
}

// Another tab of this mesh changed the list. If it changed the chat on screen,
// this tab's conversation differs from the saved one. Load the saved one; the
// next turn rebuilds from it.
async function onStoreChange(m) {
    renderList();
    const id = current.rec?.id;
    if (!id || (m.op !== 'clear' && m.id !== id) || state.busy) return;
    const rec = await store.get(id);
    if (!rec) {
        convs.delete(id);
        showChat(null);
        note('That chat was deleted in another tab.');
        return;
    }
    if (rec.rev === current.rec.rev) return;
    const ids = rec.ids.length !== current.rec.ids.length
        || rec.ids.some((x, i) => x !== current.rec.ids[i]);
    current.rec = rec;
    if (ids) { convs.delete(id); current.conv = null; attach(); }
    render();
    refresh();
}

async function renderList() {
    if (!store) return;
    const [recs, elsewhere] = await Promise.all([store.list(), store.countElsewhere()]);

    const priv = store.mesh !== 'global';
    ui.chatsElsewhere.hidden = !(priv && elsewhere);
    ui.chatsElsewhere.textContent = `${elsewhere} chat${elsewhere === 1 ? ' is' : 's are'} saved ` +
        'for other clusters, including the public demo. Leave this cluster to see them.';
    ui.chatsEmpty.hidden = recs.length > 0;
    ui.chatsForget.hidden = !recs.length;

    ui.chatList.replaceChildren(...groupByDay(recs).map(({ label, chats }) => {
        const group = document.createElement('section');
        group.className = 'group';
        const h = document.createElement('h3');
        h.textContent = label;
        group.append(h, ...chats.map(chatItem));
        return group;
    }));
    lockChats(state.busy);
}

function chatItem(rec) {
    const item = document.createElement('div');
    item.className = `chat-item${rec.id === current.rec?.id ? ' current' : ''}`;
    const open = document.createElement('button');
    open.className = `open${sameModel(rec) ? '' : ' foreign'}`;
    open.textContent = rec.title || 'New chat';
    open.title = sameModel(rec) ? rec.title : `${rec.title}: made on ${rec.model}; read-only here`;
    open.addEventListener('click', () => showChat(rec));
    const rename = iconButton('Rename', 'M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4');
    rename.addEventListener('click', () => renameChat(item, rec));
    const del = iconButton('Delete', 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13');
    del.addEventListener('click', () => deleteChat(rec));
    item.append(open, rename, del);
    return item;
}

function iconButton(label, path) {
    const b = document.createElement('button');
    b.className = 'icon';
    b.setAttribute('aria-label', label);
    b.title = label;
    b.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" fill="none" ' +
        `stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg>`;
    return b;
}

function renameChat(item, rec) {
    const input = document.createElement('input');
    input.value = rec.title;
    input.setAttribute('aria-label', 'Chat name');
    let done = false;
    const finish = async (keep) => {
        if (done) return;
        done = true;
        const title = input.value.trim();
        if (keep && title && title !== rec.title) {
            const saved = await store.rename(rec.id, title);
            if (saved && current.rec?.id === rec.id) current.rec = saved;
        }
        renderList();
    };
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') finish(true);
        if (e.key === 'Escape') { e.stopPropagation(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
    item.replaceChildren(input);
    input.focus();
    input.select();
}

// Deleting frees the chat's caches on the peers immediately. Switching away
// only parks them.
async function deleteChat(rec) {
    if (state.busy || !confirm(`Delete "${rec.title}"? It is removed from this browser.`)) return;
    const conv = convs.get(rec.id);
    convs.delete(rec.id);
    if (conv && engine?.conversation) engine.discard(conv);
    await store.remove(rec.id);
    if (current.rec?.id === rec.id) showChat(null);
    renderList();
}

async function forgetChats() {
    const n = (await store.list()).length;
    if (state.busy || !n) return;
    if (!confirm(`Delete all ${n} chat${n === 1 ? '' : 's'} saved for this cluster in this browser?`)) return;
    for (const conv of convs.values()) if (engine?.conversation) engine.discard(conv);
    convs.clear();
    await store.forgetAll();
    showChat(null);
    renderList();
}

// --- Chat ---

// Whether this chat was made on the checkpoint the mesh runs. Its token ids
// are not valid for another tokenizer. True until the manifest is read.
const sameModel = (rec) => !running || !rec
    || (rec.model === running.model && rec.weights === running.weights);
const foreign = () => !sameModel(current.rec);

// Put `rec` on screen and in the engine. null is a new chat. The previous chat
// is parked, not freed.
function showChat(rec) {
    if (state.busy) return;
    if (rec && rec.id === current.rec?.id) return openChats(false);
    current = { rec, conv: rec ? convs.get(rec.id) ?? null : null };
    attach();
    render();
    renderList();
    openChats(false);
    if (engine?.conversation) refresh();
    if (!ui.input.disabled) ui.input.focus();
}

// Give the engine the chat on screen. A stored chat is passed as its token
// ids, and its next turn rebuilds the peers' caches from them (route.js's
// submit()). Before connect() there is no engine; boot() calls this again.
function attach() {
    if (!engine?.conversation) return;
    if (!current.conv) {
        const ids = current.rec && sameModel(current.rec) ? current.rec.ids : [];
        current.conv = engine.newConversation(ids);
        if (current.rec) convs.set(current.rec.id, current.conv);
    }
    engine.switchTo(current.conv);
}

// Render the transcript from the chat's messages. The boot progress box is
// kept while boot is running.
function render() {
    ui.messages.replaceChildren(...(ui.boot?.isConnected ? [ui.boot] : []));
    for (const m of current.rec?.messages ?? []) {
        if (m.role === 'user') { addMessage('you', m.text, 'user'); continue; }
        const out = addMessage('ClusterChat', m.text, 'model');
        if (m.reasoning != null) out.reasoning = m.reasoning;
        out.meta = m.meta ?? '';
    }
    if (current.rec && foreign()) {
        note(`This chat was made on ${current.rec.model ?? 'another model'}, and the cluster now ` +
            `runs ${running.model}, so it is read-only here.`);
    }
    if (!current.rec?.messages?.length) {
        if (room?.private && !sideWide.matches) ui.messages.appendChild(shareCard());
        ui.messages.appendChild(suggestions());
        ui.transcript.scrollTop = 0;
        return;
    }
    scrollDown(true);
}

// --- The invite panel, down the right side of a private cluster ---

const sideWide = window.matchMedia('(min-width: 1200px)');

function showSide() {
    const on = !!room?.private && sideWide.matches;
    ui.sidePanel.hidden = !on;
    document.body.classList.toggle('side-open', on);
    if (on) ui.sidePanel.replaceChildren(shareCard());
}

function wireSide() {
    sideWide.addEventListener('change', () => { showSide(); render(); });
}

// The card shown in an empty chat in a private cluster: the invite link to
// copy, and an optional name shown to whoever opens it.
function shareCard() {
    const card = document.createElement('section');
    card.className = 'share';
    const creator = (() => { try { return sessionStorage.getItem(creatorKey(secret)) === '1'; } catch { return false; } })();
    const total = posBytes ? posBytes.reduce((a, b) => a + b, 0) : 0;
    const gb = total ? Math.round(total / 2 ** 30) : 27;
    card.innerHTML = `<h2></h2><p>Anyone with this link can join, lend their GPU and chat. Send it only to people you trust.</p>
        <div class="share-link"><input type="text" readonly aria-label="Invite link"><button type="button" class="primary">Copy link</button></div>
        <label class="share-name">Your name, so they see "… invited you" <input type="text" placeholder="e.g. Ana (optional)" autocomplete="off"></label>
        <p class="small"></p>`;
    card.querySelector('h2').textContent = creator ? 'Your private cluster is ready. Invite your group.' : 'Invite people to this cluster';
    card.querySelector('.small').textContent = `It runs once the group lends about ${gb} GB of GPU memory in total. ` +
        'A little extra lets people leave without stopping the chats.';
    const link = card.querySelector('.share-link input');
    const name = card.querySelector('.share-name input');
    const copy = card.querySelector('.share-link button');
    name.maxLength = NAME_MAX;
    name.value = myName();
    link.value = shareUrl();
    name.addEventListener('input', () => {
        try { sessionStorage.setItem(nameKey(secret), cleanName(name.value)); } catch { /* not kept */ }
        link.value = shareUrl();
    });
    link.addEventListener('focus', () => link.select());
    copy.addEventListener('click', () => copyText(shareUrl(), copy, 'Copy link'));
    return card;
}

// Example prompts for an empty chat. A click puts one in the input.
// Two of about the same length on the first row, so it centres evenly, and
// one centred under them.
const EXAMPLES = [
    'Plan a cheap 3-day trip to Lisbon',
    'Help me write a birthday message',
    'Why is my Python script so slow?',
];

function suggestions() {
    const box = document.createElement('section');
    box.className = 'suggest';
    const h = document.createElement('p');
    h.className = 'suggest-head';
    h.textContent = 'Try asking';
    box.appendChild(h);
    for (const text of EXAMPLES) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = text;
        b.addEventListener('click', () => {
            // While chat cannot start, point at the input's reason instead.
            if (ui.input.disabled) {
                ui.input.classList.remove('nudge');
                void ui.input.offsetWidth;            // restart the animation
                ui.input.classList.add('nudge');
                clearTimeout(ui.input.nudgeTimer);
                ui.input.nudgeTimer = setTimeout(() => ui.input.classList.remove('nudge'), 1400);
                return;
            }
            ui.input.value = text;
            ui.input.dispatchEvent(new Event('input'));
            ui.input.focus();
        });
        box.appendChild(b);
    }
    return box;
}

// Save one turn to the chat's record: the messages as shown, and the token ids
// as the peers hold them after any rewind. A rebuild re-feeds those ids.
async function saveTurn(user, reply) {
    if (!store) return;
    try {
        if (!current.rec) {
            current.rec = newRecord({ mesh: store.mesh, ...running });
            convs.set(current.rec.id, current.conv);
        }
        current.rec.messages.push(user, ...(reply ? [reply] : []));
        current.rec.ids = engine.history;
        await store.save(current.rec);
    } catch (err) {
        console.warn('saving the chat:', err);
        problem('SAVE_FAILED', err);
    }
    renderList();
}

const options = () => ({
    temp: Number(ui.temp.value),
    topK: Number(ui.topk.value),
    // The head peer samples but holds no token ids, so the router turns these
    // four settings into a summary before the job is sent (web/route.js).
    rep: Number(ui.rep.value),
    freq: Number(ui.freq.value),
    pres: Number(ui.pres.value),
    ngram: Number(ui.ngram.value),
    // 0 means no cap: run until the reply ends or the context is full.
    maxTokens: Number(ui.maxtok.value) || Infinity,
    chat: ui.mode.value === 'chat',
    thinking: ui.thinking.checked,
    prof: Number(ui.prof.value),
    loop: ui.loop.value === 'closed',
    spec: ui.spec.value === 'on' && ui.loop.value === 'closed',
});

async function submitPrompt() {
    if (state.busy || foreign()) return;
    const text = ui.input.value.trim();
    if (!text) return;

    const opts = options();
    // A seat under the mesh's limit on replies at once, before the prompt is
    // shown. A refused prompt stays in the input to send again.
    setBusy(true);
    ui.stop.disabled = true;
    const seat = await engine.net.seat();
    if (!seat.ok) {
        setBusy(false);
        problem('REPLY_LIMIT', `the cluster's limit of ${seat.max} replies at once`);
        setStatus('cluster busy', 'busy');
        return;
    }

    ui.input.value = '';
    ui.input.style.height = 'auto';
    addMessage('you', text, 'user');
    const how = { chat: opts.chat, thinking: opts.chat && opts.thinking };
    const user = { role: 'user', text, reasoning: null, meta: '', opts: how };
    // The reply as shown, updated while streaming, so a failed turn saves what
    // was displayed.
    const reply = { role: 'assistant', text: '', reasoning: null, meta: '', opts: how };

    ui.stop.disabled = false;
    setStatus('generating…', 'busy');
    ui.messages.querySelector('.share')?.remove();
    ui.messages.querySelector('.suggest')?.remove();
    const out = addMessage('ClusterChat', '', 'model streaming');
    out.meta = 'Writing…';

    let fed = true;
    try {
        fed = await respond(text, opts, out, reply);
    } catch (err) {
        console.error(err);
        let key = classify(err);
        if (key === 'GPU_FULL' || key === 'NO_WEBGPU') key = 'REPLY_FAILED';
        problem(key, err);
        setStatus('error', 'err');
        reply.meta = `Failed (${PROBLEMS[key].code})`;
        out.meta = reply.meta;
    } finally {
        engine.net.unseat();
        out.el.classList.remove('streaming');
        // If nothing reached the mesh (the context was full), nothing is saved.
        if (fed) await saveTurn(user, reply.text || reply.reasoning ? reply : null);
        setBusy(false);
        refresh();
    }
}

// Turn wrapping and stop detection are in web/turn.js (shared with
// tools/mesh.gpu.test.mjs). This renders the reply and fills in `reply` for
// the chat's record. Returns false when the prompt was refused before
// anything was sent.
async function respond(text, opts, out, reply) {
    const show = (field, v) => { out[field] = v; reply[field === 'body' ? 'text' : field] = v; };
    let turn;
    try {
        turn = await runTurn(engine, text, opts, {
            chat: CHAT,
            turnRe: TURN_RE,
            onToken: ({ text: shown, reasoning, answer, n }) => {
                if (reasoning !== null) show('reasoning', reasoning);
                show('body', (reasoning === null ? shown : answer).replace(/^\s+/, ''));
                // A reply can run for minutes, so show progress.
                setStatus(`generating… ${n} tok · ${
                    Math.max(0, engine.ctxMax - engine.ctxLen)} left`, 'busy');
                scrollDown();
            },
        });
    } catch (err) {
        if (err instanceof ContextFullError) {
            out.el.remove();
            problem('CHAT_FULL', err);
            setStatus('context full', 'err');
            return false;
        }
        throw err;
    }

    // Checked on the answer, not the whole reply: a reply that is all
    // reasoning would leave an empty bubble.
    if (!turn.answer.trim()) {
        show('body', turn.text.trim()
            ? '(the model was still reasoning when this reply ended. Raise max tok '
              + 'or turn thinking off)'
            : '(nothing. Try a higher temperature or a longer prompt)');
    }
    const summary = summarize(engine.lastProfile);
    const hopsLine = formatLine(summary);
    // Under the reply: how many GPUs ran it and how fast, in tokens. The token
    // timings are the hover text; tools/load.mjs reads them from there.
    const gpus = new Set((summary.decode?.hops ?? []).map((h) => h.id)).size;
    for (const h of summary.decode?.hops ?? []) if (h.ms?.med != null) hopMs.set(h.id, h.ms.med);
    const tps = turn.ids.length && turn.perTok > 0 ? 1000 / turn.perTok : 0;
    show('meta', [gpus ? `via ${plural(gpus, 'GPU')}` : null,
                  tps ? `${tps < 10 ? tps.toFixed(1) : tps.toFixed(0)} tok/s` : null]
        .filter(Boolean).join(' · '));
    out.detail = `${turn.ids.length} tokens · ${turn.perTok.toFixed(0)} ms/tok · ` +
        `prefill ${turn.prefillMs.toFixed(0)} ms for ${turn.promptTokens} · ${turn.stopped}` +
        (hopsLine ? ` · ${hopsLine}` : '');
    showProfile(out.el, summary, engine.lastProfile);
    ui.speed.textContent = turn.perTok ? `${(1000 / turn.perTok).toFixed(1)} tokens/s` : '—';
    setStatus('ready', 'ok');
    scrollDown();
    return true;
}

// The timing breakdown for one reply, collapsed under it: readable text, and
// the raw samples as JSON for comparing runs. The kernel tables are logged to
// the console.
function showProfile(el, summary, profile) {
    if (!summary.decode && !summary.prefill) return;
    const box = document.createElement('details');
    box.className = 'prof';
    box.innerHTML = '<summary>Timing</summary><pre></pre><button type="button">Copy JSON</button>';
    box.querySelector('pre').textContent = formatText(summary);
    const button = box.querySelector('button');
    button.addEventListener('click', async () => {
        const json = JSON.stringify({ at: new Date().toISOString(), summary, samples: profile.samples });
        try {
            await navigator.clipboard.writeText(json);
            button.textContent = 'Copied';
        } catch {
            console.log(json);
            button.textContent = 'Copy failed, logged to the console';
        }
    });
    el.appendChild(box);

    console.log(formatText(summary));
    for (const h of summary.decode?.hops ?? []) {
        if (h.kernels) console.table(h.kernels.map((k) => ({ peer: h.id, ...k })));
    }
    window.lastProfile = { summary, samples: profile.samples };
}
