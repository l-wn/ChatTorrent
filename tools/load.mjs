// A load test: how many people can use a deploy at once, and what each sees.
//
// tools/mesh.gpu.test.mjs tests whether the mesh answers correctly under
// concurrency, on a toy model, through the dev harness page. This script
// measures how the production page behaves for visitors. It opens the real
// chat page in headless tabs, answers the welcome popup, types prompts into
// the composer and reads the transcript. It uses only what a visitor can see
// and press, so it runs against production unmodified.
//
//   node tools/load.mjs --url https://chattorrent.net/chat --users 8 --ramp 60
//   node tools/load.mjs --users 16 --turns 3 --think 20 --leave 0.2 --out a.jsonl
//   node tools/load.mjs --users 70 --turns 0 --stay       who is turned away at the cap
//   node tools/load.mjs --summarize a.jsonl b.jsonl       merge runs from several machines
//
//   --url URL          the chat page (default https://chattorrent.net/chat). A
//                      local one needs ?direct=1, since it has no relay
//   --users N          visitors who only prompt (they answer "no" to the GPU
//                      question, so a tab uses no GPU and one machine can run many)
//   --ramp S           arrivals are spread over this many seconds (default 30)
//   --turns N          prompts each visitor sends (default 2; 0 = join only)
//   --think S          pause before each prompt, 0.5x to 1.5x this (default 10)
//   --tokens N         the page's "Max tokens" setting for every reply (default 64)
//   --leave P          chance, per turn, that the visitor closes the tab part
//                      way into the reply (default 0)
//   --leave-after S    when: 0.3x to 1.7x this many seconds after Send
//                      (default 6)
//   --stay             keep every tab open until the last visitor is done,
//                      not closing each as it finishes
//   --contributors N   tabs that answer "yes" and serve on this machine's GPU,
//                      opened first; visitors arrive once the model is covered
//   --ready S          how long a visitor waits to be able to prompt (default 120)
//   --timeout S        how long a visitor waits for one reply (default 600)
//   --seed N           seeds arrivals, pauses, prompts and departures
//   --label NAME       names this machine in the records (default: its hostname)
//   --out FILE         append one JSON line per join and per turn
//   --show             stream every tab's console
//
// If the site is behind basic auth, the password is read from CHAT_PASS (user
// CHAT_USER, default chatbot). It is an environment variable, not a flag, so
// it does not appear in `ps`.
//
// Each visitor gets its own browser context: separate storage, cache and
// login. They all share this machine's address, which the signaling server's
// per-address socket cap and the relay's per-address rate limit count. One
// machine can therefore run at most that cap. For more visitors, run this on
// several machines and merge the records with --summarize.

import { appendFileSync, readFileSync } from 'node:fs';
import { hostname } from 'node:os';
import { launch } from './harness/cdp.mjs';

// --- Arguments ---

const argv = process.argv.slice(2);
const has = (f) => argv.includes(`--${f}`);
const opt = (f, d) => { const i = argv.indexOf(`--${f}`); return i < 0 ? d : argv[i + 1]; };

const URL_       = opt('url', 'https://chattorrent.net/chat');
const USERS      = Number(opt('users', 4));
const RAMP       = Number(opt('ramp', 30));
const TURNS      = Number(opt('turns', 2));
const THINK      = Number(opt('think', 10));
const TOKENS     = Number(opt('tokens', 64));
const LEAVE      = Number(opt('leave', 0));
const LEAVE_S    = Number(opt('leave-after', 6));
const STAY       = has('stay');
const CONTRIB    = Number(opt('contributors', 0));
const READY_S    = Number(opt('ready', 120));
const TIMEOUT_S  = Number(opt('timeout', 600));
const SEED       = Number(opt('seed', 1));
const LABEL      = opt('label', hostname());
const OUT        = opt('out', null);
const SHOW       = has('show');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// mulberry32. A run is reproducible from its seed.
function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Short questions with short answers. The load is the number of visitors; a
// long prompt would measure prefill.
const PROMPTS = [
    'What is the capital of Australia?',
    'Explain what a hash table is in two sentences.',
    'Give me three names for a grey cat.',
    'Why is the sky blue? Keep it short.',
    'Write a haiku about rain.',
    'What is 17 times 23?',
    'Suggest a quick weeknight dinner.',
    'What does HTTP stand for?',
    'Tell me one fact about octopuses.',
    'Translate "good morning" into French and Spanish.',
];

// --- What runs in the page ---
//
// Each is a function sent as source and called with JSON arguments. They use
// only what a visitor can see or press, by the ids in web/peer.html.

const call = (fn, arg) => `(${fn})(${JSON.stringify(arg ?? null)})`;

// Tick both agreement boxes, then press one of the two answers (web/welcome.js).
const answerWelcome = (contribute) => {
    for (const id of ['agreeAge', 'agreeTerms']) {
        const box = document.getElementById(id);
        box.checked = true;
        box.dispatchEvent(new Event('change'));
    }
    document.getElementById(contribute ? 'welcomeYes' : 'welcomeNo').click();
};

// Wait until the composer is enabled, the mesh refuses this tab, or the
// timeout. Times are measured from the start of navigation, so they include
// the page load.
const waitReady = async ({ timeout }) => {
    const $ = (id) => document.getElementById(id);
    let joined = null;
    for (;;) {
        const now = performance.now();
        const status = $('status').textContent;
        if (joined === null && /^peer \d+/.test($('role').textContent)) joined = now;
        if (status === 'ready' && !$('send').disabled) {
            return { outcome: 'ok', joinMs: joined ?? now, readyMs: now, status };
        }
        if (/mesh is full/.test(status)) return { outcome: 'full', joinMs: joined, status };
        if (/did not accept this page/.test(status)) return { outcome: 'refused', joinMs: joined, status };
        if (now > timeout) return { outcome: 'never-ready', joinMs: joined, status };
        await new Promise((r) => setTimeout(r, 50));
    }
};

// For a contributor: wait until the roster shows every position held.
const waitCovered = async ({ timeout }) => {
    for (;;) {
        const text = document.getElementById('covered').textContent;
        const m = /^(\d+)\/(\d+) held/.exec(text);
        if (m && m[1] === m[2]) return true;
        if (performance.now() > timeout) return false;
        await new Promise((r) => setTimeout(r, 200));
    }
};

// Send one prompt and wait from the click on Send until the composer is
// enabled again.
const turn = async ({ text, tokens, timeout }) => {
    const $ = (id) => document.getElementById(id);
    const errors = () => [...document.querySelectorAll('#messages .msg.note.err')];
    const notes = () => [...document.querySelectorAll('#messages .msg.note:not(.err)')];
    const errsBefore = errors().length, notesBefore = notes().length;

    // The composer can be disabled between turns (a peer left and the model is
    // not covered, or the signaling socket is reconnecting). Wait for it, as a
    // person would.
    const opens = performance.now() + 30000;
    while ($('send').disabled && performance.now() < opens) {
        await new Promise((r) => setTimeout(r, 100));
    }
    if ($('send').disabled) return { outcome: 'not-ready', error: $('status').textContent };

    $('maxtok').value = String(tokens);
    $('input').value = text;
    const t0 = performance.now();
    $('send').click();
    if ($('stop').hidden) return { outcome: 'not-sent', error: $('status').textContent };

    const reply = [...document.querySelectorAll('#messages .msg.model')].pop();
    let first = null, last = null, n = 0, timedOut = false;
    while (!$('stop').hidden) {
        const now = performance.now();
        // The status pill counts tokens as they arrive (peer.js's respond()).
        const m = /generating… (\d+) tok/.exec($('status').textContent);
        if (m) {
            if (first === null) first = now;
            if (Number(m[1]) !== n) { n = Number(m[1]); last = now; }
        }
        if (now - t0 > timeout) { timedOut = true; break; }
        await new Promise((r) => setTimeout(r, 25));
    }
    const end = performance.now();

    const errs = errors().slice(errsBefore).map((e) => e.textContent.trim());
    const told = notes().slice(notesBefore).map((e) => e.textContent);
    // E.g. "64 tokens · 250 ms/tok · prefill 900 ms for 21 · max tokens" (peer.js).
    const meta = reply?.querySelector('.meta').textContent ?? '';
    const mm = /^(\d+) tokens · (\d+) ms\/tok · prefill (\d+) ms for (\d+) · ([^·]*)/.exec(meta);

    const failed = errs.find((e) => /generation failed/.test(e)) ?? errs[0] ?? null;
    const outcome = timedOut ? 'timeout'
        // A peer with no free frame, page or queue slot (stage.js's
        // #atCapacity). The page reports it and the visitor must ask again.
        : failed && /try again in a moment/.test(failed) ? 'busy'
        : failed || !mm ? 'error'
        : 'ok';
    return {
        outcome,
        error: failed ?? (outcome === 'error' ? `no reply summary: "${meta}"` : null),
        ttftMs: first === null ? null : first - t0,
        totalMs: end - t0,
        tokens: mm ? Number(mm[1]) : n,
        // Decode rate measured in this tab, first token to last.
        tokPerS: n > 1 && last > first ? ((n - 1) * 1000) / (last - first) : null,
        msPerTok: mm ? Number(mm[2]) : null,
        prefillMs: mm ? Number(mm[3]) : null,
        stopped: mm ? mm[5].trim() : null,
        // A peer on this chat's path left. Either this tab rebuilt the
        // context, or the peer before the gap replayed it (the two notes in
        // peer.js).
        rebuilds: told.filter((t) => /Rebuilding/.test(t)).length,
        fixes: told.filter((t) => /replayed/.test(t)).length,
    };
};

// --- Summary ---

const pct = (xs, p) => {
    const s = xs.filter((x) => x != null).sort((a, b) => a - b);
    return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null;
};
const secs = (ms) => ms == null ? 'n/a' : `${(ms / 1000).toFixed(1)}s`;
const rate = (x) => x == null ? 'n/a' : x.toFixed(1);
const tally = (rows) => {
    const by = {};
    for (const r of rows) by[r.outcome] = (by[r.outcome] ?? 0) + 1;
    return Object.entries(by).map(([k, v]) => `${v} ${k}`).join(', ') || 'none';
};

// How many replies this machine's visitors were waiting on when a turn was
// sent, including that turn. A rough measure of load. Grouped into buckets, so
// a ramp is a few rows.
const BUCKETS = [[1, 1], [2, 2], [3, 4], [5, 8], [9, 16], [17, 32], [33, Infinity]];
const bucketOf = (n) => BUCKETS.find(([lo, hi]) => n >= lo && n <= hi);
const bucketName = ([lo, hi]) => lo === hi ? `${lo}` : hi === Infinity ? `${lo}+` : `${lo}–${hi}`;

function summarize(records) {
    const joins = records.filter((r) => r.kind === 'join');
    const turns = records.filter((r) => r.kind === 'turn');
    const labels = [...new Set(records.map((r) => r.label))];
    const lines = [];

    lines.push(`${joins.length} visitors from ${labels.length} machine${labels.length === 1 ? '' : 's'}`
        + ` (${labels.join(', ')})`);
    lines.push(`  joins:  ${tally(joins)}`);
    const in_ = joins.filter((j) => j.outcome === 'ok');
    lines.push(`  time until the composer opened:  p50 ${secs(pct(in_.map((j) => j.readyMs), 0.5))}`
        + `  p95 ${secs(pct(in_.map((j) => j.readyMs), 0.95))}`);
    for (const j of joins.filter((x) => x.outcome !== 'ok').slice(0, 5)) {
        lines.push(`    visitor ${j.label}/${j.user}: ${j.outcome}: "${j.status ?? j.error ?? ''}"`);
    }
    if (!turns.length) return lines.join('\n');

    lines.push(`  turns:  ${tally(turns)}`);
    const rebuilt = turns.reduce((n, t) => n + (t.rebuilds ?? 0), 0);
    const fixed = turns.reduce((n, t) => n + (t.fixes ?? 0), 0);
    if (rebuilt || fixed) lines.push(`  repairs:  ${rebuilt} rebuilds, ${fixed} local replays`);

    lines.push('');
    lines.push('  replies in flight   turns   ok   first token p50 / p95   tok/s p50 / worst');
    for (const b of BUCKETS) {
        const rows = turns.filter((t) => bucketOf(t.inflight) === b);
        if (!rows.length) continue;
        const ok = rows.filter((t) => t.outcome === 'ok');
        lines.push(`  ${bucketName(b).padEnd(17)}   ${String(rows.length).padStart(5)}`
            + `   ${String(ok.length).padStart(2)}`
            + `   ${secs(pct(ok.map((t) => t.ttftMs), 0.5)).padStart(11)} / ${secs(pct(ok.map((t) => t.ttftMs), 0.95)).padEnd(7)}`
            + `   ${rate(pct(ok.map((t) => t.tokPerS), 0.5)).padStart(9)} / ${rate(pct(ok.map((t) => t.tokPerS), 0))}`);
    }

    const bad = turns.filter((t) => !['ok', 'left'].includes(t.outcome));
    if (bad.length) {
        lines.push('');
        const seen = {};
        for (const t of bad) {
            const key = `${t.outcome}: ${t.error ?? ''}`;
            seen[key] = (seen[key] ?? 0) + 1;
        }
        for (const [k, v] of Object.entries(seen).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
            lines.push(`  ${String(v).padStart(3)} x ${k}`);
        }
    }
    return lines.join('\n');
}

if (has('summarize')) {
    const files = argv.slice(argv.indexOf('--summarize') + 1).filter((a) => !a.startsWith('--'));
    if (!files.length) {
        console.error('--summarize: name one or more files written with --out');
        process.exit(2);
    }
    const records = files.flatMap((f) => readFileSync(f, 'utf8').split('\n')
        .filter(Boolean).map((l) => JSON.parse(l)));
    console.log(summarize(records));
    process.exit(0);
}

// --- The run ---

const records = [];
const started = Date.now();
function record(r) {
    const row = { label: LABEL, at: Date.now() - started, ...r };
    records.push(row);
    if (OUT) appendFileSync(OUT, `${JSON.stringify(row)}\n`);
    return row;
}

const page = new URL(URL_);
const PASS = process.env.CHAT_PASS;
const USER = process.env.CHAT_USER || 'chatbot';

// Checked once before any tab opens. A tab refused by the password gate would
// otherwise show only as a page that never loads, thirty seconds later.
try {
    const r = await fetch(page, { redirect: 'manual', headers: PASS
        ? { authorization: `Basic ${Buffer.from(`${USER}:${PASS}`).toString('base64')}` } : {} });
    if (r.status === 401) {
        console.error(PASS
            ? `${page.href} refused the login for user "${USER}": check CHAT_PASS (and CHAT_USER).`
            : `${page.href} is behind the site's password gate. Give the login in the environment:\n`
              + `  CHAT_PASS=… node tools/load.mjs ${argv.join(' ')}`);
        process.exit(2);
    }
    if (r.status >= 400) {
        console.error(`${page.href} answered ${r.status}`);
        process.exit(2);
    }
} catch (err) {
    console.error(`could not reach ${page.href}: ${err.cause?.message ?? err.message}`);
    process.exit(2);
}

const chrome = await launch();

// One visitor's browser: its own context, logged in if the site is gated. The
// login uses a temporary tab on the site's root. Chrome answers the basic-auth
// challenge from the URL's userinfo and keeps the credentials for the origin
// (setup-gpu-peer.sh does the same and explains why the page URL cannot carry
// them).
//
// A contributor is opened in the browser's default context (`own: false`).
// Headless Chrome gives a tab in any other context no WebGPU adapter, and the
// page would stay a client.
let loggedIn = false;
async function openPage(name, { own = true } = {}) {
    const context = own ? await chrome.context() : undefined;
    if (PASS && (own || !loggedIn)) {
        if (!own) loggedIn = true;
        const login = new URL('/', page);
        login.username = USER;
        login.password = PASS;
        const { targetId } = await chrome.browser.send('Target.createTarget',
            { url: login.href, ...(context && { browserContextId: context }) });
        await sleep(3000);
        await chrome.browser.send('Target.closeTarget', { targetId }).catch(() => {});
    }
    return chrome.tab(page.href, {
        context,
        onConsole: SHOW ? (type, text) => console.log(`  [${name}] ${type}: ${text}`) : undefined,
    });
}

let inflight = 0;
let peak = 0;
const open = new Set();

async function visitor(i) {
    const rand = rng(SEED * 7919 + i);
    // Spread over the ramp with some jitter, so arrivals are not evenly spaced.
    await sleep(((i + rand() * 0.5) / Math.max(1, USERS)) * RAMP * 1000);

    let tab;
    try {
        tab = await openPage(`v${i}`);
        open.add(tab);
        await tab.eval(call(answerWelcome, false));
        const j = await tab.eval(call(waitReady, { timeout: READY_S * 1000 }),
            { timeout: (READY_S + 30) * 1000 });
        record({ kind: 'join', user: i, ...j });
        console.log(`v${i}: ${j.outcome === 'ok' ? `ready in ${secs(j.readyMs)}` : `${j.outcome}: "${j.status}"`}`);
        if (j.outcome !== 'ok') return;
    } catch (err) {
        record({ kind: 'join', user: i, outcome: 'error', error: err.message });
        console.log(`v${i}: ${err.message}`);
        return;
    }

    for (let t = 0; t < TURNS; t++) {
        await sleep((0.5 + rand()) * THINK * 1000);
        const text = PROMPTS[Math.floor(rand() * PROMPTS.length)];
        const leaves = rand() < LEAVE;
        const at = ++inflight;
        peak = Math.max(peak, inflight);
        let r;
        try {
            const asked = tab.eval(call(turn, { text, tokens: TOKENS, timeout: TIMEOUT_S * 1000 }),
                { timeout: (TIMEOUT_S + 60) * 1000 });
            if (leaves) {
                // A visitor who gives up: the tab closes mid-reply.
                asked.catch(() => {});
                const gone = sleep((0.3 + rand() * 1.4) * LEAVE_S * 1000).then(() => null);
                r = await Promise.race([asked, gone]) ?? { outcome: 'left' };
                if (r.outcome === 'left') await tab.close();
            } else {
                r = await asked;
            }
        } catch (err) {
            r = { outcome: 'error', error: err.message };
        } finally {
            inflight--;
        }
        record({ kind: 'turn', user: i, turn: t, inflight: at, ...r });
        console.log(`v${i} turn ${t + 1}: ${r.outcome}`
            + (r.outcome === 'ok' ? `: first token ${secs(r.ttftMs)}, ${rate(r.tokPerS)} tok/s, ${r.tokens} tokens` : '')
            + (r.error ? `: ${r.error}` : '')
            + (r.rebuilds || r.fixes ? ` (${r.rebuilds} rebuilds, ${r.fixes} replays)` : ''));
        if (tab.closed) return;
    }
    if (!STAY) { open.delete(tab); await tab.close(); }
}

let failed = false;
try {
    console.log(`${page.origin}${page.pathname}: ${USERS} visitors over ${RAMP}s, ${TURNS} turns each`
        + `, ${TOKENS} tokens a reply${LEAVE ? `, ${LEAVE * 100}% leave mid-reply` : ''}`);

    for (let c = 0; c < CONTRIB; c++) {
        const tab = await openPage(`c${c}`, { own: false });
        open.add(tab);
        await tab.eval(call(answerWelcome, true));
        console.log(`c${c}: contributing`);
    }
    if (CONTRIB) {
        const first = [...open][0];
        const covered = await first.eval(call(waitCovered, { timeout: 600000 }), { timeout: 660000 });
        if (!covered) throw new Error('the model was not covered after 10 minutes');
        console.log('model covered');
    }

    await Promise.all(Array.from({ length: USERS }, (_, i) => visitor(i)));
    console.log(`\n${summarize(records)}`);
    console.log(`  most replies in flight at once, from here: ${peak}`);
    if (OUT) console.log(`  records appended to ${OUT}`);
} catch (err) {
    failed = true;
    console.error(err);
} finally {
    await chrome.close();
}
process.exit(failed ? 1 : 0);
