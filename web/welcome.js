// The pre-join screen: peer.html's <dialog id="welcome">, full screen. It asks
// how much GPU memory to lend, and pressing "Join and chat" is the visitor's
// agreement to the terms. askWelcome() resolves with the answer, and peer.js
// does not open the GPU before that.
//
// The model bar shows the cluster being joined, one cell per piece. The
// pieces this tab would hold at the amount on the slider are previewed in
// violet the way a real join gets them (lend.js's previewRange()). peer.js
// sends the cluster as a `cluster` event on document.
//
// The public demo has a limit on people chatting (signal-server.js's
// MAX_PEOPLE). At it, the card says so and Join waits until a spot frees.
// A visitor lending their GPU can still join, to lend, and chats once a spot
// frees. A tab turned away
// at the limit on tabs sees the same card while net.js retries.
//
// A browser that can't lend (no WebGPU, no adapter, or only a software one)
// gets the slider locked at zero and a line saying why.
//
// It shows each time a tab arrives at a cluster, and not when that tab
// reloads: answering marks the cluster joined for the tab (joinedKey, in
// sessionStorage). The create page marks its new cluster joined itself. The
// last answer is kept in localStorage (KEY) as the slider's starting point,
// and the Contribute drawer changes it with saveChoice().

import { readInvite, inviteFrom } from './room.js';
import { depths, holding } from './split.js';
import { lendSlider, previewRange, cellsFor, limitGb, DEFAULT_GB } from './lend.js';

const KEY = 'chattorrent.welcome.v1';

function load() {
    try {
        const v = JSON.parse(localStorage.getItem(KEY));
        return v && typeof v === 'object' && v.agreed ? v : null;
    } catch {
        return null;
    }
}

function save(v) {
    try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* the slider starts at the default */ }
}

// A short, non-reversible label for a cluster's secret (FNV-1a), so the
// clusters this tab joined can be recognised without storing secrets.
function label(secret) {
    let h = 0x811c9dc5;
    for (let i = 0; i < secret.length; i++) h = Math.imul(h ^ secret.charCodeAt(i), 0x01000193);
    return (h >>> 0).toString(16).padStart(8, '0');
}

// Set in this tab once it has joined the cluster with `secret` (null: the
// public demo).
export const joinedKey = (secret) => `chattorrent:joined:${secret ? label(secret) : 'public'}`;

// Save a GPU choice: the Contribute drawer's, or the create page's.
// `limit` is a value for the "Lend up to" field ("4gb"; "" means no limit).
export function saveChoice({ contribute, limit }) {
    const v = load() ?? {};
    save({ ...v, v: 1, agreed: true, contribute, limit: limit ?? v.limit ?? '' });
}

// Whether this browser can lend its GPU. Asking for an adapter does not open
// the GPU, so this runs before the visitor has agreed to anything.
async function canLend() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        return !!adapter && !(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter);
    } catch {
        return false;
    }
}

// Resolves with { contribute: boolean, limit: string }.
export function askWelcome() {
    let secret = null;
    try { secret = readInvite(location.hash); } catch { /* peer.js reports a bad link */ }
    const saved = load();
    const joined = (() => { try { return sessionStorage.getItem(joinedKey(secret)) === '1'; } catch { return false; } })();
    if (saved && joined) {
        return Promise.resolve({ contribute: !!saved.contribute, limit: saved.limit ?? '' });
    }

    const $ = (id) => document.getElementById(id);
    const dlg = $('welcome');
    const barEl = $('previewBar');

    // The cluster: `L` pieces, the other computers' claims ({ lo, hi, have }),
    // the server's minimum claim `floor`, and `capacity(gb)`, how many pieces
    // this tab would hold at `gb`.
    let info = null;

    // The lobby: the cluster's name, who invited this person (named in the
    // link), and how many are connected.
    $('lobbyName').textContent = secret ? 'Private cluster' : 'Public demo';
    const inviter = secret ? inviteFrom(location.hash) : '';
    if (inviter) {
        $('lobbyInvite').textContent = `${inviter} invited you`;
        $('lobbyInvite').hidden = false;
    }
    function renderLobby() {
        const crowd = info.crowd ?? [];
        const lending = crowd.filter((c) => c.lending).length;
        if (info.name) $('lobbyName').textContent = info.name;
        $('lobbyLine').textContent = crowd.length
            ? `${crowd.length} connection${crowd.length === 1 ? '' : 's'} (${lending} lending)`
            : "Nobody else is here yet. You'd be the first.";
    }

    // Whether Join can be pressed: not while the demo is full, unless the
    // visitor joins to lend.
    function renderGate() {
        const go = $('welcomeYes');
        const atLimit = info.maxPeople != null && info.people >= info.maxPeople;
        const blocked = info.full || (atLimit && !slider.value);
        go.disabled = blocked;
        go.textContent = blocked ? 'Full, waiting for a spot'
            : atLimit ? 'Join and lend' : 'Join and chat';
        if (info.full) {
            $('lobbyLine').textContent = "The public demo is full right now. You'll get in when a spot opens.";
        } else if (atLimit) {
            $('lobbyLine').textContent = `Full right now: ${info.people} of ${info.maxPeople} people are chatting. `
                + (slider.value ? 'You can lend your GPU now and chat once a spot frees up.'
                    : 'Wait until a spot frees up.');
        }
    }

    function render() {
        if (!info) return;
        if (!info.L) { renderGate(); return; }
        renderLobby();
        renderGate();
        const { L, peers } = info;
        const cells = cellsFor(barEl, L);
        const pct = (n) => Math.round((n / L) * 100);
        const gb = slider.value;

        // What runs now (downloaded pieces) and what is reserved (claims).
        const held = depths(peers.map((p) => ({ lo: p.lo, hi: p.have ?? p.hi })).filter(holding), L);
        const range = gb ? previewRange(peers.filter(holding), held, L, info.floor, info.capacity(gb)) : null;
        const [lo, hi] = range ?? [0, 0];

        let now = 0, withYou = 0;
        for (let x = 0; x < L; x++) {
            const mine = x >= lo && x < hi;
            if (held[x]) now++;
            if (held[x] || mine) withYou++;
            const state = mine ? 'you' : held[x] >= 2 ? 'deep' : held[x] === 1 ? 'thin' : 'empty';
            if (cells[x].dataset.state !== state) cells[x].dataset.state = state;
        }

        // GB of the model held with this tab, of the whole model.
        const gbAt = (x) => info.pieceGb?.[x] ?? info.modelGb / L;
        let haveGb = 0;
        for (let x = 0; x < L; x++) if (held[x] || (x >= lo && x < hi)) haveGb += gbAt(x);
        $('modelPct').textContent = `${pct(withYou)}% (${Math.round(haveGb)}/${Math.round(info.modelGb)} GB)`;
        barEl.classList.toggle('full', withYou === L);
        const n = hi - lo;
        // Missing pieces another computer has claimed and is downloading.
        const claimed = depths(peers.filter(holding), L);
        let missing = 0, coming = 0;
        for (let x = 0; x < L; x++) {
            if (held[x] || (x >= lo && x < hi)) continue;
            missing++;
            if (claimed[x]) coming++;
        }
        $('previewCaption').textContent = !n
            ? (now === L ? 'The cluster holds the whole model. You can chat without lending.'
                : `${pct(now)}% held. Chat starts once the cluster holds all of it.`)
            : withYou === L && now < L ? 'With you, the cluster holds the whole model.'
            : withYou === L ? `You'd hold ${pct(n)}% as a second copy, so people can leave safely.`
            : missing === coming
                ? `${withYou > now ? `You'd add ${pct(withYou - now)}%.` : `You'd hold a second copy of ${pct(n)}%.`} ` +
                  'The rest is downloading on another computer.'
            : `You'd add ${pct(withYou - now)}%. It still needs ${100 - pct(withYou)}% more.`;
        if (noGpu) {
            $('previewCaption').textContent = "This browser can't lend its GPU, but you can still chat. "
                + 'To lend, use desktop Chrome, Edge or Safari 26.';
        }
    }

    // A first visit starts at 0: chat only, lending is opt-in.
    const start = saved ? (saved.contribute ? limitGb(saved.limit) ?? DEFAULT_GB : 0) : 0;
    const slider = lendSlider({
        bar: $('lendBar'), fill: $('lendYou'), handle: $('lendHandle'), out: $('lendOut'),
        value: start, onChange: render,
    });
    let noGpu = false;
    canLend().then((ok) => {
        if (ok) return;
        noGpu = true;
        slider.lock();
        $('lendOut').textContent = "Can't lend";
        render();
    });
    document.addEventListener('cluster', (e) => {
        if (!e.detail?.L && !e.detail?.full) return;
        info = e.detail;
        render();
    });

    // Escape would close a modal <dialog> without an answer, so the cancel
    // event is prevented. Before the page has had a click, Chrome closes the
    // dialog without firing cancel, so a close with no answer reopens it.
    dlg.addEventListener('cancel', (e) => e.preventDefault());
    let answered = false;
    dlg.addEventListener('close', () => { if (!answered) dlg.showModal(); });

    return new Promise((resolve) => {
        $('welcomeYes').addEventListener('click', () => {
            const gb = slider.value;
            const choice = gb ? { contribute: true, limit: `${gb}gb` } : { contribute: false, limit: '' };
            answered = true;
            saveChoice(choice);
            try { sessionStorage.setItem(joinedKey(secret), '1'); } catch { /* shown again on reload */ }
            dlg.close();
            resolve(choice);
        });
        dlg.showModal();
    });
}
