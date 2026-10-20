// A peer in the mesh: the DOM layer around one Engine (web/engine.js), which
// joins the mesh, holds and grows a range, and generates tokens. This file
// boots the page, wires events, picks the chat format, and renders the
// transcript, the chat list, the status pills and the block map.

import { dimsFromManifest } from './model/loader.js';
import { Engine, ContextFullError } from './engine.js';
import { positions, coverage, holders, holding, describeRange, N_MAX } from './split.js';
import { chatFormat, stopRe } from './prompt.js';
import { runTurn } from './turn.js';
import { makeMessage } from './message.js';
import { summarize, formatLine, formatText } from './profile.js';
import { readInvite, openRoom, newSecret, inviteUrl } from './room.js';
import { openHistory, idbBackend, memoryBackend, newRecord, groupByDay } from './history.js';

// The turn wrapper and the regex that cuts a runaway reply, from the
// manifest's chat_format (see prompt.js). The plain transcript until the
// manifest loads.
let CHAT = 'transcript';
let TURN_RE = stopRe(CHAT);

let engine = null;
const state = { busy: false };

// This tab's invite secret, or null in the global mesh. Kept only to build the
// link for "Copy invite link" (see web/room.js).
let secret = null;

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

function setStatus(text, kind = '') {
    ui.status.textContent = text;
    ui.status.className = `pill ${kind}`;
}

function addMessage(who, text = '', cls = '') {
    const msg = makeMessage(who, text, cls);
    ui.messages.appendChild(msg.el);
    scrollDown(true);
    return msg;
}

const note = (text, err = false) => addMessage('', text, `note${err ? ' err' : ''}`);

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
        case 'status':
            note(detail.text, detail.error);
            if (detail.pill) setStatus(detail.pill, 'err');
            break;
        // A rebuild of the context from this chat's token ids. The message
        // depends on the reason (route.js's rebuild()).
        case 'repair':
            rebuilding = REBUILDS[detail.reason] ?? REBUILDS.repair;
            if (detail.reason === 'resume') {
                beforeReply(note(`Picking this chat back up: re-reading ${detail.tokens} tokens…`));
            } else if (detail.reason !== 'rewind') {
                beforeReply(note(`A peer carrying this chat left (${detail.message}). Rebuilding ` +
                    `${detail.tokens} tokens of context on the peers that are still here…`));
            }
            setStatus(`${rebuilding}…`, 'busy');
            break;
        case 'rebuild-progress':
            setStatus(`${rebuilding}… (${detail.sent}/${detail.total})`, 'busy');
            break;
        case 'repaired':
            setStatus('generating…', 'busy');
            break;
        // No status change: when a 'fix' arrives the local repair is done and
        // the job is in flight. Shown to explain the pause.
        case 'fix':
            note(`A peer carrying this chat left. Peer ${detail.from} replayed ` +
                `${detail.rows} tokens of it into another peer, so this chat did not have ` +
                'to be rebuilt.');
            break;
    }
}

// Status pill text while a rebuild runs, by reason.
const REBUILDS = {
    repair: 'a peer left, rebuilding context',
    resume: 'resuming chat',
    rewind: 're-reading context',
};
let rebuilding = REBUILDS.repair;

// A prompt can be sent when there is a tokenizer and net.connected is true:
// the ready peers' ranges cover the model, a job of mine can enter, my own
// hand-off is open, and an answer can come back. What this tab holds does not
// matter.
function refresh() {
    const net = engine.net;

    // How much of the model the mesh holds.
    const cov = net.coverage;
    ui.covered.textContent = net.positions
        ? `${cov.covered}/${cov.total} held` : 'model unknown';
    ui.covered.className = `pill ${!net.positions ? '' : cov.covered === cov.total ? 'ok'
        : cov.covered ? 'busy' : 'err'}`;
    ui.covered.title = net.positions ? summarise(cov, net) : '';

    ui.role.textContent = roleLine(net);

    ui.peers.textContent = `${net.peers.length} peer${net.peers.length === 1 ? '' : 's'}`;
    const loading = net.peers.filter((p) => holding(p) && (p.have ?? p.hi) < p.hi).length;
    if (loading) ui.peers.textContent += ` · ${loading} loading`;

    // `r`: 0 cannot run, 1 runs but any departure at the thinnest position
    // breaks it, 2 survives any single departure.
    const r = cov.r;
    ui.depth.textContent = `r ${r}`;
    ui.depth.className = `pill ${r >= 2 ? 'ok' : r === 1 ? 'busy' : 'err'}`;

    // This tab's own VRAM, not the network's.
    ui.vram.textContent = engine.runner
        ? `${(engine.vramBytes / 2 ** 30).toFixed(2)} GiB` : '— GiB';

    renderMap(net, cov);
    ui.ctx.textContent = `context ${engine.ctxLen}/${N_MAX}`;
    ui.ran.textContent = `ran ${engine.ran.toLocaleString()} tok`;

    if (!net.online) ui.peers.textContent += ' · offline';

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
            + (who.length ? `${who.length} peer${who.length === 1 ? '' : 's'} (${who.join(', ')})`
                : claimed.depths[x] ? 'downloading' : 'nobody');
    }
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
    let room;
    try {
        secret = readInvite(location.hash);
        room = await openRoom(secret);
    } catch (err) {
        ui.title.textContent = 'ChatTorrent';
        ui.boot.remove();
        showMesh(null);
        note(`This link cannot be used: ${err.message} This tab has not joined any mesh. ` +
            `Press "Global mesh" to join the public one instead.`, true);
        setStatus('bad invite link', 'err');
        return;
    }
    showMesh(room);
    openStore(room);

    ui.bootnote.textContent = 'connecting to the signaling server…';
    engine = new Engine({ onEvent: handleEngineEvent });

    let manifest;
    try {
        ({ manifest } = await engine.connect({ room, relayOnly: !directParam() }));
    } catch (err) {
        console.warn('connect failed:', err);
        ui.title.textContent = 'ChatTorrent';
        ui.boot.remove();
        note(`This tab cannot reach the mesh or the checkpoint (${err.message}), so it can ` +
            `neither prompt the mesh nor hold blocks for it.`, true);
        refresh();
        return;
    }

    CHAT = chatFormat(manifest);
    TURN_RE = stopRe(CHAT);
    running = { model: manifest.model ?? null, weights: engine.net.weightsUrl ?? null };
    // A chat picked from the list during connect() is given to the engine
    // here.
    attach();
    render();
    ui.thinkingLabel.hidden = CHAT !== 'qwen';
    ui.model.textContent = manifest.model.split('/').pop();
    ui.model.title = `model: ${manifest.model}`;
    ui.model.hidden = false;

    if (!engine.canPrompt) {
        note(`This tab could not load the tokenizer and so cannot send prompts. It can ` +
            `still hold blocks for other peers.`, true);
    }

    if (engine.net.relayOnly && !engine.net.relay) {
        note('The signaling server offers no TURN relay, and this tab only links through one ' +
            '(so no peer ever sees its IP address). No link can open until the server is given ' +
            'TURN_KEY_ID and TURN_API_TOKEN. Or open this page with ?direct=1 to link directly, ' +
            'which shows your address to the peers you link to.', true);
    }

    if (room.private) {
        note(`This is a private mesh (${room.tag}). Only people with its invite link can join, ` +
            `and only their tabs will run your chats. It works once its members' GPUs hold the ` +
            `whole model between them. Share the link with "Copy invite link".`);
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
        note(`Spectator: this tab holds no blocks and will never be routed a job ` +
            `(${err.message}). It can still prompt the mesh like any other tab.`, true);
        return;
    }

    // The GPU is opened only after the visitor agrees in the welcome popup,
    // and only up to the amount they chose.
    ui.bootwhat.textContent = 'waiting for your answer…';
    ui.bootnote.textContent = 'about contributing your GPU';
    const { contribute, limit } = await gpuConsent;
    if (!contribute) {
        ui.boot.remove();
        note(`Chat only: this tab is not using your GPU and holds no part of the model. ` +
            `Reload the page to change your mind.`);
        refresh();
        return;
    }
    ui.bootwhat.textContent = 'joining the mesh…';
    ui.vramlimit.value = limit;

    applyVramLimit();
    // Not awaited. serve() reports its own failures; the catch is for bugs.
    engine.serve({ blocks: blocksParam(), vram: vramParam(), at, slowMs: slowMs() })
        .then(({ note: msg, error }) => {
            ui.boot.remove();
            note(msg, error);
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

// The header's mesh controls. `room` null means no mesh (a damaged invite);
// only the link back to the global mesh is shown.
function showMesh(room) {
    const priv = !!room?.private;
    ui.mesh.textContent = !room ? 'no mesh' : priv ? `private mesh · ${room.tag}` : 'global mesh';
    ui.mesh.className = `pill ${!room ? 'err' : priv ? 'private' : ''}`;
    ui.mesh.title = !room ? 'this tab has not joined any mesh'
        : priv ? `A private mesh. Everyone in it sees the same tag (${room.tag}); only tabs ` +
                 'opened from its invite link can link to this one.'
        : 'The public mesh: anyone who opens this page joins it and may run your chats.';
    ui.newmesh.hidden = !room || priv;
    ui.invite.hidden = !priv;
    ui.leave.hidden = !(priv || !room);
    ui.leave.href = location.pathname;         // this page with no invite or query
    ui.leave.textContent = priv ? 'Leave' : 'Global mesh';
    ui.chatsMesh.textContent = !room ? 'no mesh' : priv ? `private mesh ${room.tag}` : 'global mesh';
}

async function copyInvite() {
    const url = inviteUrl(secret);
    try {
        await navigator.clipboard.writeText(url);
        ui.invite.textContent = 'copied';
    } catch {
        // The clipboard API needs a secure origin (not plain http on a LAN).
        window.prompt('Copy this invite link and send it only to people you trust:', url);
    }
    setTimeout(() => { ui.invite.textContent = 'Copy invite link'; }, 2000);
}

function wireEvents() {
    // A new fragment names a new mesh, and a fragment change does not reload
    // the page. Reload so the engine always matches the mesh in the URL.
    // Covers Create, a hand-edited link, and Back/Forward.
    window.addEventListener('hashchange', () => location.reload());
    ui.newmesh.addEventListener('click', () => {
        if (state.busy && !confirm('A reply is still being generated. Leave it and move to a new private mesh?')) return;
        location.hash = `mesh=${newSecret()}`;
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
}

// --- The settings drawer ---
//
// Presentation only: submitPrompt() reads the fields by id. Each field's HTML
// value is its default, so Reset and the gear's changed-dot compare against
// defaultValue / defaultSelected.

const settingFields = () => ui.settings.querySelectorAll('input, select');

const isDefault = (el) => el.tagName === 'SELECT'
    ? [...el.options].every((o) => o.selected === o.defaultSelected)
    : el.value === el.defaultValue;

function openSettings(open) {
    if (open) openChats(false);
    ui.settings.hidden = ui.scrim.hidden = !open;
    ui.settingsBtn.setAttribute('aria-expanded', String(open));
    (open ? ui.settingsClose : ui.settingsBtn).focus();
}

function markChanged() {
    ui.settingsBtn.classList.toggle('changed', ![...settingFields()].every(isDefault));
}

function wireSettings() {
    ui.settingsBtn.addEventListener('click', () => openSettings(ui.settings.hidden));
    ui.settingsClose.addEventListener('click', () => openSettings(false));
    ui.scrim.addEventListener('click', () => openSettings(false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !ui.settings.hidden) openSettings(false);
    });
    ui.settings.addEventListener('input', markChanged);
    ui.settings.addEventListener('change', markChanged);
    ui.settingsReset.addEventListener('click', () => {
        for (const el of settingFields()) {
            if (el.tagName === 'SELECT') for (const o of el.options) o.selected = o.defaultSelected;
            else el.value = el.defaultValue;
        }
        markChanged();
        // #vramlimit has no `value` attribute, so Reset blanks it, and
        // assigning `value` fires no `input` event. The engine is told
        // explicitly, so it does not stay limited while the field reads blank.
        applyVramLimit();
    });
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
        if (ui.settings.hidden) ui.scrim.hidden = true;
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
        'for other meshes. Your global-mesh chats are still saved. Leave this mesh to see them.';
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
    if (!confirm(`Delete all ${n} chat${n === 1 ? '' : 's'} saved for this mesh in this browser?`)) return;
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
        const out = addMessage('mesh', m.text, 'model');
        if (m.reasoning != null) out.reasoning = m.reasoning;
        out.meta = m.meta ?? '';
    }
    if (current.rec && foreign()) {
        note(`This chat was made on ${current.rec.model ?? 'another model'}; the mesh now runs ` +
            `${running.model}. Its token ids mean nothing to this one's tokenizer, so it is ` +
            'read-only here.');
    }
    scrollDown(true);
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
        note(`This turn could not be saved (${err.message}).`, true);
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
});

async function submitPrompt() {
    if (state.busy || foreign()) return;
    const text = ui.input.value.trim();
    if (!text) return;

    const opts = options();
    ui.input.value = '';
    ui.input.style.height = 'auto';
    addMessage('you', text, 'user');
    const how = { chat: opts.chat, thinking: opts.chat && opts.thinking };
    const user = { role: 'user', text, reasoning: null, meta: '', opts: how };
    // The reply as shown, updated while streaming, so a failed turn saves what
    // was displayed.
    const reply = { role: 'assistant', text: '', reasoning: null, meta: '', opts: how };

    setBusy(true);
    ui.stop.disabled = false;
    setStatus('generating…', 'busy');
    const out = addMessage('mesh', '', 'model streaming');

    let fed = true;
    try {
        fed = await respond(text, opts, out, reply);
    } catch (err) {
        console.error(err);
        note(`generation failed: ${err.message}`, true);
        setStatus('error', 'err');
        reply.meta = `failed: ${err.message}`;
    } finally {
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
                    Math.max(0, N_MAX - engine.ctxLen)} left`, 'busy');
                scrollDown();
            },
        });
    } catch (err) {
        if (err instanceof ContextFullError) {
            out.el.remove();
            note(`That prompt needs ${err.needed} tokens and ${err.room} are left in the ` +
                `${N_MAX}-token context. Start a new chat from the chat list.`, true);
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
    show('meta', `${turn.ids.length} tokens · ${turn.perTok.toFixed(0)} ms/tok · ` +
        `prefill ${turn.prefillMs.toFixed(0)} ms for ${turn.promptTokens} · ${turn.stopped}` +
        (hopsLine ? ` · ${hopsLine}` : ''));
    showProfile(out.el, summary, engine.lastProfile);
    ui.speed.textContent = turn.perTok ? `${(1000 / turn.perTok).toFixed(1)} tok/s` : '—';
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
    box.innerHTML = '<summary>timing</summary><pre></pre><button type="button">copy JSON</button>';
    box.querySelector('pre').textContent = formatText(summary);
    const button = box.querySelector('button');
    button.addEventListener('click', async () => {
        const json = JSON.stringify({ at: new Date().toISOString(), summary, samples: profile.samples });
        try {
            await navigator.clipboard.writeText(json);
            button.textContent = 'copied';
        } catch {
            console.log(json);
            button.textContent = 'copy failed, logged to the console';
        }
    });
    el.appendChild(box);

    console.log(formatText(summary));
    for (const h of summary.decode?.hops ?? []) {
        if (h.kernels) console.table(h.kernels.map((k) => ({ peer: h.id, ...k })));
    }
    window.lastProfile = { summary, samples: profile.samples };
}
