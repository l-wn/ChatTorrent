// The landing page's script: the headline's intro, the network behind the
// hero, a line when this browser cannot lend its GPU, and the scripted demo
// of a cluster (never the real model). The create page uses the network too.
// A classic script, not a module, so the page also works opened from disk.

(() => {
// --- The demo: four computers, four chats, one model ---
//
// Four people open the same link. Each browser lends a different amount and
// holds some pieces of the model (violet in its own window: they are that
// person's); together they make the whole model. They chat at their own
// times. Ben's gaming PC grows while idle, so when Ana closes her laptop
// mid-reply her pieces are already held twice and the other chats carry on.
// Scripted, never the real model. render(t) draws the whole scene for a time
// t in the loop, so pause, loop and the reduced-motion still frame share one
// path.

const N = 32;                   // pieces of the model shown
const MODEL_GB = 27;
const LOOP_MS = 21000;
const LEAVE_AT = 10200;         // Ana closes her laptop
const DOWNLOAD_MS = 2300;

// Four shares of different sizes that come together into the whole model.
// Each starts on one piece nobody holds yet (`root`, or the middle of its
// range), as the server places a newcomer on a gap.
// Ana's is small. Ben's grows while it is idle, before Ana leaves, so her
// pieces are already held twice when she goes. They ask at their own times;
// Mia asks only after Ana has left.
const PEOPLE = [
    {
        name: "Ana's laptop", at: 300, range: [13, 19], lend: 3,
        q: 'Write a short poem about LAN parties.', askAt: 5400, startAt: 6000, wordMs: 240,
        a: 'Cables snake across the floor, / four towers humming by the door. / Pizza cooling, ' +
           'screens aglow, / one more round before we go. / Dawn comes creeping through the blinds, / ' +
           'still we play, and no one minds.',
    },
    {
        name: "Ben's gaming PC", at: 1500, range: [21, 32], lend: 12,
        grow: { at: 5200, to: [12, 32], ms: 4800 },
        q: 'Plan a 3-day trip to Lisbon on a student budget.', askAt: 8200, startAt: 8800, wordMs: 190,
        a: 'Day 1: Alfama and the castle. Wander the old streets, then watch the sunset from Miradouro da Senhora do Monte. ' +
           'Day 2: Belém. Try the pastéis, see the monastery, and walk back along the river for free. ' +
           'Day 3: LX Factory in the morning, then a cheap dinner in Bairro Alto.',
    },
    {
        name: 'Lab PC 3', at: 900, range: [0, 13], lend: 8,
        q: 'Give me three causes of the First World War.', askAt: 4200, startAt: 4800, wordMs: 205,
        a: 'First, alliances: an attack on one country pulled its partners in. Second, militarism: ' +
           'the great powers had been racing to build armies and navies. Third, nationalism, which ' +
           'the assassination of Archduke Franz Ferdinand in Sarajevo set alight.',
    },
    {
        name: "Mia's desktop", at: 2200, range: [6, 21], root: 20, lend: 10,
        q: 'Explain recursion with a short example.', askAt: 11400, startAt: 12000, wordMs: 215,
        a: 'Recursion is when a function solves a problem by calling itself on a smaller version of it. ' +
           'To count down from 3: print 3, then count down from 2, then from 1. ' +
           'It stops at 0. That stopping point is called the base case.',
    },
];
for (const p of PEOPLE) {
    p.words = p.a.split(' ');
    p.endAt = p.startAt + p.words.length * p.wordMs;
}
const ANA = 0;

const CAPTIONS = [0, 4200, LEAVE_AT, LEAVE_AT + 1200];
// One soft glow per chat glides along the pieces while its reply is written:
// a word passing through the model. One pass takes PASS_MS.
const PASS_MS = 1400;
const GLOW_W = 5;               // pieces a glow covers
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const smooth = (x) => { const k = clamp01(x); return k * k * (3 - 2 * k); };   // ease in and out

// What person i holds at time t: [from, hi) and downloaded up to `have`.
// A share starts as one piece, the moment its window lights up, and grows
// outwards to both sides as it downloads. A growing share then
// widens towards its target the same way.
function holdingAt(p, i, t) {
    if (t < p.at || (i === ANA && t >= LEAVE_AT)) return null;
    const [lo, hi] = p.range;
    const root = p.root ?? Math.floor((lo + hi) / 2);
    const k = smooth((t - p.at) / DOWNLOAD_MS);
    if (!p.grow || t < p.grow.at) {
        const from = root - (root - lo) * k;
        const to = root + 1 + (hi - root - 1) * k;
        return { from, hi: to, have: to, of: hi - lo };
    }
    const g = smooth((t - p.grow.at) / p.grow.ms);
    const from = lo - (lo - p.grow.to[0]) * g;
    const to = hi + (p.grow.to[1] - hi) * g;
    return { from, hi: to, have: to, of: to - from };
}

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

// A piece bar, with one glow per chat over it (hidden until used).
function cellRow(cls) {
    const wrap = el('div', `dbar ${cls}`);
    const row = wrap.appendChild(el('div', 'dcells'));
    const cells = Array.from({ length: N }, (_, x) => {
        const c = el('i');
        c.style.setProperty('--i', x);
        return row.appendChild(c);
    });
    const glows = PEOPLE.map(() => wrap.appendChild(el('span', 'dglow')));
    return { wrap, row, cells, glows };
}

// Show glow k at piece position x (fractional), or hide it with x = null.
function glowAt(bar, k, x) {
    const g = bar.glows[k];
    g.classList.toggle('on', x != null);
    if (x != null) g.style.left = `${((x - GLOW_W / 2) / N) * 100}%`;
}

function startDemo(root) {
    const $ = (sel) => root.querySelector(sel);
    const toggle = $('#demoToggle');
    const flag = $('#demoFlag');
    const captions = [...root.querySelectorAll('#demoCaptions li')];

    const cluster = cellRow('dbar-cluster');
    $('#demoCells').replaceWith(cluster.wrap);

    // One browser window per person.
    const frames = PEOPLE.map((p) => {
        const f = el('div', 'dframe');
        const bar = f.appendChild(el('div', 'dframe-bar'));
        bar.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="3" width="20" height="13" rx="2"/><path d="M8 21h8M12 16v5"/></svg>';
        bar.append(el('span', '', p.name));
        const busy = bar.appendChild(el('span', 'dbusy'));
        const hold = f.appendChild(el('div', 'dframe-hold'));
        const line = hold.appendChild(el('div', 'dhold-line'));
        line.append(el('span', '', `Lending ${p.lend} GB`));
        const count = line.appendChild(el('span', '', ''));
        const pieces = cellRow('dbar-own');
        hold.appendChild(pieces.wrap);
        const body = f.appendChild(el('div', 'dframe-body'));
        const q = body.appendChild(el('div', 'dq', p.q));
        const a = body.appendChild(el('div', 'da'));
        const text = a.appendChild(document.createTextNode(''));
        const caret = a.appendChild(el('span', 'dcaret'));
        const meta = body.appendChild(el('div', 'dmeta', ''));
        $('#demoFrames').appendChild(f);
        return { f, busy, count, pieces, q, a, text, caret, meta, shown: -1 };
    });

    const setText = (node, s) => { if (node.textContent !== s) node.textContent = s; };

    function render(t, live = true) {
        const held = PEOPLE.map((p, i) => holdingAt(p, i, t));
        const depth = new Int8Array(N);
        held.forEach((h) => {
            if (!h) return;
            for (let x = Math.ceil(h.from - 1e-6); x < Math.floor(h.have + 1e-6); x++) depth[x]++;
        });
        const lending = held.filter(Boolean).length;

        // The cluster bar: dark held, pale missing; a slow wave once full.
        let covered = 0;
        cluster.cells.forEach((c, x) => {
            if (depth[x]) covered++;
            const s = depth[x] ? 'held' : '';
            if (c.dataset.state !== s) c.dataset.state = s;
        });
        const full = covered === N;
        cluster.wrap.classList.toggle('full', full && live);

        // The glows: each chat's word passing through the pieces, run by one
        // of the computers holding each piece (chat k takes the k-th holder,
        // so different computers carry different chats at the same moment).
        const work = new Array(PEOPLE.length).fill(0);
        PEOPLE.forEach((p, k) => {
            const closed = k === ANA && t >= LEAVE_AT;
            const writing = live && full && !closed && t >= p.startAt && t < p.endAt;
            const x = writing ? (((t - p.startAt + k * 350) % PASS_MS) / PASS_MS) * (N + GLOW_W) - GLOW_W / 2 : null;
            glowAt(cluster, k, x);
            const at = x == null ? -1 : Math.max(0, Math.min(N - 1, Math.floor(x)));
            const holders = at < 0 ? [] : held.map((h, i) => (h && at >= Math.ceil(h.from - 1e-6) && at < Math.floor(h.have + 1e-6) ? i : -1))
                .filter((i) => i >= 0);
            const runner = holders.length ? holders[k % holders.length] : -1;
            if (runner >= 0) work[runner]++;
            frames.forEach((fr, i) => glowAt(fr.pieces, k, i === runner ? x : null));
        });
        frames.forEach((fr, i) => fr.busy.classList.toggle('on', work[i] > 0));
        setText(flag, full ? `Running · ${lending} GPUs` : `Needs ${Math.ceil(((N - covered) / N) * MODEL_GB)} GB`);
        flag.dataset.state = full ? 'ok' : 'busy';

        frames.forEach((fr, i) => {
            const p = PEOPLE[i], h = held[i];
            const closed = i === ANA && t >= LEAVE_AT;
            fr.f.classList.toggle('waiting', t < p.at);
            fr.f.classList.toggle('closed', closed);

            // Its own pieces violet, the cluster's other pieces dark.
            let mine = 0;
            fr.pieces.cells.forEach((c, x) => {
                const own = h && x >= Math.ceil(h.from - 1e-6) && x < Math.floor(h.have + 1e-6);
                if (own) mine++;
                const s = closed ? '' : own ? 'you' : depth[x] ? 'held' : '';
                if (c.dataset.state !== s) c.dataset.state = s;
            });
            const loading = h && t < p.at + DOWNLOAD_MS;
            setText(fr.count, closed ? 'closed' : t < p.at ? 'opening…'
                : loading ? `downloading ${Math.round(((h.have - h.from) / h.of) * 100)}%`
                : `${mine} pieces`);

            // Its own chat.
            fr.q.hidden = t < p.askAt;
            fr.a.hidden = t < p.startAt - 200;
            const until = closed ? Math.min(t, LEAVE_AT) : t;
            const k = until < p.startAt ? 0 : Math.min(p.words.length, Math.floor((until - p.startAt) / p.wordMs) + 1);
            if (k !== fr.shown) { fr.shown = k; fr.text.data = p.words.slice(0, k).join(' '); }
            const done = t >= p.endAt && !closed;
            fr.caret.hidden = done || closed || t < p.startAt;
            setText(fr.meta, closed ? 'Laptop closed.' : done ? `via ${lending} GPUs · ${Math.round(1000 / p.wordMs)} words/s`
                : t >= p.startAt ? 'Writing…' : '');
        });

        let cap = 0;
        CAPTIONS.forEach((at, i) => { if (t >= at) cap = i; });
        captions.forEach((li, i) => li.classList.toggle('now', live && i === cap));
    }

    // A still frame for reduced motion: near the end, all captions.
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
        root.classList.add('static');
        toggle.hidden = true;
        render(LOOP_MS - 2500, false);
        return;
    }

    // The clock runs only while the demo is on screen, the tab is visible and
    // it is not paused.
    let t = 0, last = null, paused = false, visible = false, raf = 0;
    const running = () => !paused && visible && !document.hidden;
    const frame = (now) => {
        raf = 0;
        if (last != null) t = (t + Math.min(now - last, 100)) % LOOP_MS;
        last = now;
        render(t);
        if (running()) raf = requestAnimationFrame(frame);
    };
    const wake = () => { if (running() && !raf) { last = null; raf = requestAnimationFrame(frame); } };
    toggle.addEventListener('click', () => {
        paused = !paused;
        toggle.textContent = paused ? 'Play' : 'Pause';
        toggle.setAttribute('aria-label', paused ? 'Play the demo' : 'Pause the demo');
        cluster.wrap.classList.toggle('paused', paused);
        wake();
    });
    document.addEventListener('visibilitychange', wake);
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; wake(); }, { threshold: 0 }).observe(root);
    render(0);
}

// --- The hero's background: a faint network with flashes hopping along it ---
//
// Nodes on a jittered grid, each linked to its nearest neighbours. A flash
// starts at a random node and hops a few links, lighting each node it
// reaches. Drawn on a canvas behind the hero; site.css fades it out towards
// the centre so the headline stays on clean white.

function startNetwork(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const css = getComputedStyle(document.documentElement);
    const SIGNAL = css.getPropertyValue('--signal').trim() || '#7022d6';
    const CELL = 120;           // grid spacing, px
    const SPEED = 0.22;         // px per ms along a link
    const FLASH_MS = 900;       // how long a reached node stays lit
    const SPAWN_MS = 380;       // a new flash about this often

    let w = 0, h = 0, nodes = [], links = [], flashes = [], lit = new Map();

    function build() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        w = canvas.clientWidth; h = canvas.clientHeight;
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        nodes = [];
        const cols = Math.ceil(w / CELL) + 1, rows = Math.ceil(h / CELL) + 1;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                nodes.push({
                    x: (c + 0.5 + (Math.random() - 0.5) * 0.8) * CELL - CELL / 2,
                    y: (r + 0.5 + (Math.random() - 0.5) * 0.8) * CELL - CELL / 2,
                    near: [],
                });
            }
        }
        const seen = new Set();
        links = [];
        nodes.forEach((a, i) => {
            const byDist = nodes.map((b, j) => [j, Math.hypot(a.x - b.x, a.y - b.y)])
                .filter(([j, d]) => j !== i && d < CELL * 1.7)
                .sort((p, q) => p[1] - q[1]).slice(0, 3);
            for (const [j] of byDist) {
                const key = i < j ? `${i}-${j}` : `${j}-${i}`;
                if (seen.has(key)) continue;
                seen.add(key);
                links.push([i, j]);
                a.near.push(j);
                nodes[j].near.push(i);
            }
        });
        flashes = [];
        lit = new Map();
    }

    function spawn(now, start = Math.floor(Math.random() * nodes.length)) {
        flashes.push({ path: [start], hops: 2 + Math.floor(Math.random() * 4), from: start, to: null, t0: now });
        lit.set(start, now);
    }

    // A clicked node: a signal leaves along every one of its links and
    // carries on hopping from there.
    function burst(start, now) {
        lit.set(start, now);
        for (const next of nodes[start].near) {
            flashes.push({ path: [start], hops: 3 + Math.floor(Math.random() * 4), from: start, to: next, t0: now });
        }
    }

    // Pick the next node of a flash, not going straight back.
    function nextHop(f) {
        const options = nodes[f.from].near.filter((j) => j !== f.path.at(-2));
        return options.length ? options[Math.floor(Math.random() * options.length)] : null;
    }

    function draw(now, animate) {
        ctx.clearRect(0, 0, w, h);
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(11, 11, 12, 0.08)';
        ctx.beginPath();
        for (const [i, j] of links) { ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); }
        ctx.stroke();

        ctx.fillStyle = 'rgba(11, 11, 12, 0.18)';
        for (const n of nodes) { ctx.beginPath(); ctx.arc(n.x, n.y, 1.6, 0, Math.PI * 2); ctx.fill(); }
        if (!animate) return;

        // Lit nodes: a dot and a ring that grows and fades.
        for (const [i, at] of lit) {
            const k = (now - at) / FLASH_MS;
            if (k >= 1) { lit.delete(i); continue; }
            const n = nodes[i];
            ctx.globalAlpha = 1 - k;
            ctx.fillStyle = SIGNAL;
            ctx.beginPath(); ctx.arc(n.x, n.y, 2.6, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = SIGNAL;
            ctx.beginPath(); ctx.arc(n.x, n.y, 3 + k * 10, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.globalAlpha = 1;

        // Moving flashes: a bright head with a short tail along the link.
        flashes = flashes.filter((f) => {
            if (f.to == null) {
                f.to = nextHop(f);
                if (f.to == null) return false;
                f.t0 = now;
            }
            const a = nodes[f.from], b = nodes[f.to];
            const len = Math.hypot(b.x - a.x, b.y - a.y);
            const k = Math.min(1, (now - f.t0) * SPEED / len);
            const x = a.x + (b.x - a.x) * k, y = a.y + (b.y - a.y) * k;
            const tk = Math.max(0, k - 28 / len);
            const grad = ctx.createLinearGradient(a.x + (b.x - a.x) * tk, a.y + (b.y - a.y) * tk, x, y);
            grad.addColorStop(0, 'rgba(112, 34, 214, 0)');
            grad.addColorStop(1, SIGNAL);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(a.x + (b.x - a.x) * tk, a.y + (b.y - a.y) * tk);
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.lineWidth = 1;
            if (k < 1) return true;
            lit.set(f.to, now);
            f.path.push(f.to);
            f.from = f.to;
            f.to = null;
            return f.path.length <= f.hops;
        });
    }

    build();
    let resizeTimer = 0;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => { build(); if (still) draw(0, false); }, 150);
    });

    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (still) { draw(0, false); return; }

    // Clicking a node starts a burst from it, unannounced: the pointer does
    // not change over a node. Only nodes that show count: the
    // canvas fades out towards the centre (site.css's mask), and clicks on
    // links, buttons and text keep their usual meaning.
    const area = canvas.parentElement;
    const skip = 'a, button, input, select, label, h1, h2, p, .create-card';
    function nodeAt(e) {
        if (e.target.closest?.(skip)) return -1;
        const r = canvas.getBoundingClientRect();
        const x = e.clientX - r.left, y = e.clientY - r.top;
        // Outside the faded centre: the mask is clear within about half of
        // its ellipse (46% by 52% of the canvas, centred).
        const dx = (x - w / 2) / (w * 0.46), dy = (y - h / 2) / (h * 0.52);
        if (Math.hypot(dx, dy) < 0.6) return -1;
        let best = -1, bestD = 22;
        nodes.forEach((n, i) => {
            const d = Math.hypot(n.x - x, n.y - y);
            if (d < bestD) { bestD = d; best = i; }
        });
        return best;
    }
    area.addEventListener('click', (e) => {
        const i = nodeAt(e);
        if (i < 0) return;
        burst(i, performance.now());
        wake();
    });

    let visible = true, raf = 0, nextSpawn = 0;
    const loop = (now) => {
        raf = 0;
        if (now >= nextSpawn) { spawn(now); nextSpawn = now + SPAWN_MS * (0.5 + Math.random()); }
        draw(now, true);
        if (visible && !document.hidden) raf = requestAnimationFrame(loop);
    };
    const wake = () => { if (visible && !document.hidden && !raf) raf = requestAnimationFrame(loop); };
    document.addEventListener('visibilitychange', wake);
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; wake(); }).observe(canvas);
}

// --- The headline: three words fly in, snap together and light up ---
//
// "Cluster", "your" and "computers." arrive from the left, from below and
// from the right, stop just apart, then snap together with a little
// overshoot. Sparks flash at the two joins and a violet pulse runs through
// the words, as a message runs through a cluster. Then the subline and the
// buttons fade in. Reduced motion skips all of it.

function introTitle(hero) {
    const title = hero.querySelector('.hero-title');
    if (!title || !title.animate || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const words = [...title.querySelectorAll('.hw')];
    const joints = [...title.querySelectorAll('.hj')];
    if (words.length !== 3) return;

    // One continuous move per word, on one curve: fast at first, slowing as
    // it nears its neighbour, then a soft overshoot as it clings (an After
    // Effects ease-out with a little bounce). The opacity fades in on its own
    // softer curve over the first part of the move.
    const MOVE = 1600;                                 // ms
    const STAGGER = 80;                                // ms between words, left to right
    const cling = 'cubic-bezier(.2, 1.18, .32, 1)';
    const easeOutSine = 'cubic-bezier(.61, 1, .88, 1)';
    const glide = 'cubic-bezier(.16, 1, .3, 1)';

    // Each word starts about a word-width out, not off-screen.
    const from = ['translateX(-1.3em)', 'translateY(.4em)', 'translateX(1.3em)'];
    words.forEach((w, i) => {
        // fill 'both': each animation holds its last frame rather than handing
        // the element back, so the text is not redrawn (and re-snapped to the
        // pixel grid) the moment it ends.
        w.animate([{ transform: from[i] }, { transform: 'translate(0, 0)' }],
            { duration: MOVE, delay: i * STAGGER, easing: cling, fill: 'both' });
        w.animate([{ opacity: 0 }, { opacity: 1 }],
            { duration: MOVE * 0.55, delay: i * STAGGER, easing: easeOutSine, fill: 'both' });
    });

    // Sparks at the joins, the moment the words meet.
    // The curve reaches its neighbour about 45% of the way through.
    const meet = MOVE * 0.45;
    for (const j of joints) {
        const spark = document.createElement('i');
        spark.className = 'spark';
        j.appendChild(spark);
        spark.animate([
            { transform: 'translate(-50%, -50%) scale(0)', opacity: 1 },
            { transform: 'translate(-50%, -50%) scale(1)', opacity: .9, offset: .3 },
            { transform: 'translate(-50%, -50%) scale(2.4)', opacity: 0 },
        ], { duration: 1050, delay: meet, easing: 'ease-out', fill: 'both' });
    }

    // A pulse through the words, left to right.
    const ink = getComputedStyle(title).color;
    const signal = getComputedStyle(document.documentElement).getPropertyValue('--signal').trim() || '#7022d6';
    words.forEach((w, i) => w.animate([
        { color: ink }, { color: signal, offset: .35 }, { color: ink },
    ], { duration: 1100, delay: meet + 160 + i * 210, easing: 'ease-in-out' }));

    // Then the rest of the hero.
    const rest = [hero.querySelector('.lede'), hero.querySelector('.actions'), hero.querySelector('.gpucheck')].filter(Boolean);
    rest.forEach((el, i) => el.animate([
        { opacity: 0, transform: 'translate(0, 10px)' }, { opacity: 1, transform: 'translate(0, 0)' },
    ], { duration: 1100, delay: meet + 200 + i * 260, easing: glide, fill: 'both' }));
}

async function checkGpu(line) {
    let ok = false;
    try {
        ok = !!(navigator.gpu && await navigator.gpu.requestAdapter());
    } catch { /* no adapter */ }
    if (ok) return;
    line.textContent = "This browser can chat, but can't lend its graphics card. For that, use Chrome, Edge or Safari 26.";
}

const line = document.getElementById('gpucheck');
if (line) checkGpu(line);

const hero = document.querySelector('.hero');
if (hero) introTitle(hero);

const net = document.getElementById('heroNet');
if (net) startNetwork(net);

const demo = document.getElementById('demo');
if (demo) startDemo(demo);
})();
