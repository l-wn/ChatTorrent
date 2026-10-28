// Tests that background growth pauses at the VRAM limit instead of ending, and
// that raising or clearing the limit resumes it from the holding already in
// VRAM.
//
//   node tools/grow.test.mjs
//
// Runs everywhere: web/engine.js imports cleanly under Node, and nothing here
// needs a tokenizer, so this file never skips.
// The Runner is the real one from web/model/stage.js on tools/stub-gpu.mjs.

import { Runner } from '../web/model/stage.js';
import { makeGpu, D, L, N, weightsFor, pieceFor } from './stub-gpu.mjs';
import { Mesh } from '../web/net.js';
import { DEFAULT_CTX as N_MAX, SLOTS, positions } from '../web/split.js';

globalThis.location = { protocol: 'http:', host: 'localhost:8000' };
globalThis.WebSocket = class { addEventListener() {} send() {} close() {} readyState = 1; };
// A growth step that gets past the limit goes on to download its position. No
// weights are served here, and a 404 is permanent, so the attempt fails at once
// instead of retrying with backoff. That failure is the proof the loop resumed.
globalThis.fetch = async () => ({
    ok: false, status: 404, body: null, headers: { get: () => null }, json: async () => ({}),
});

const { Engine } = await import('../web/engine.js');

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};

// Long enough for a parked loop to wake, take its step and fail the download.
const settle = () => new Promise((resolve) => setTimeout(resolve, 50));

// One worker holding positions 2-3 of the toy model's 8, so growth has room on
// both sides and neither neighbour is covered.
//
// `lastJobAt` is pushed well into the past so the runner counts as idle and the
// first growth step is taken with no GROWTH_QUIET_MS wait, which makes the pause
// below land synchronously inside forTest(). A real peer is idle like this by
// the time it has downloaded a claim; in a fresh Node process performance.now()
// is only a few hundred ms, so the default 0 would not be idle enough.
function grower({ vramLimit, grow = true, load = null, growRetryMs }) {
    const net = new Mesh({});
    net.ready.catch(() => {});
    Object.assign(net, {
        id: 0, blocks: L, positions: N, minBlocks: 1, range: [2, 4],
        peers: [{ id: 0, lo: 2, hi: 4, have: 4, ready: true }],
    });
    net.send = () => false;

    const { device } = makeGpu();
    const runner = new Runner(device, D, weightsFor(2, 4),
        { lo: 2, hi: 4, have: 4, nMax: N_MAX, tMax: 8, slots: SLOTS, idleMs: 0 });
    runner.lastJobAt = -60000;

    const events = [];
    const engine = Engine.forTest({
        net, router: null, runner, dims: D, device, vramLimit, grow, load, growRetryMs,
        onEvent: (type, detail) => events.push({ type, detail }),
    });
    const notes = () => events.filter((e) => e.type === 'status').map((e) => e.detail.text);
    return { engine, runner, events, notes };
}

console.log('the toy model and the holding this file grows from');
{
    ok('positions(L) matches the stub model', N === positions(L) && N === 8, `N=${N}`);
}

console.log('\ngrowth pauses at the VRAM limit instead of ending');
{
    const { engine, runner, notes } = grower({ vramLimit: 1 });

    const paused = notes();
    ok('one note, and it says growth paused', paused.length === 1 && /^Paused growing at /.test(paused[0]),
        JSON.stringify(paused));
    ok('the note names what the next position would cost and the limit in force',
        /would take this tab to [\d.]+ GiB, past its [\d.]+ GiB VRAM limit/.test(paused[0] ?? ''),
        paused[0]);
    ok('the note promises nothing is re-downloaded', /nothing already held is/.test(paused[0] ?? ''));
    ok('and it no longer claims growth has stopped for good',
        !/Stopped growing/.test(paused[0] ?? ''), paused[0]);
    ok('nothing was grown', runner.lo === 2 && runner.hi === 4,
        `holds ${runner.lo}-${runner.hi}`);
    ok('the limit is readable back', engine.vramLimit === 1, `got ${engine.vramLimit}`);
}

console.log('\nre-setting the same limit neither grows nor repeats the note');
{
    const { engine, runner, notes } = grower({ vramLimit: 1 });
    engine.setVramLimit(1);
    engine.setVramLimit(1);
    await settle();
    ok('no second note for an unchanged limit', notes().length === 1, JSON.stringify(notes()));
    ok('and still nothing grown', runner.lo === 2 && runner.hi === 4,
        `holds ${runner.lo}-${runner.hi}`);
}

console.log('\na raise that is still too small pauses again, naming the new limit');
{
    const { engine, runner, notes } = grower({ vramLimit: 1 });
    // Above the limit set above but below what the next position needs, which
    // is this peer's own VRAM plus a block on top.
    const raised = engine.vramBytes;
    ok('the raise really is a raise, and really is still too small', raised > 1);
    engine.setVramLimit(raised);
    await settle();

    const all = notes();
    ok('exactly one further note', all.length === 2, JSON.stringify(all));
    ok('and it names the new limit, not the old one',
        all[1]?.includes(`${(raised / 2 ** 30).toFixed(2)} GiB VRAM limit`), all[1]);
    ok('still nothing grown', runner.lo === 2 && runner.hi === 4,
        `holds ${runner.lo}-${runner.hi}`);
}

console.log('\nclearing the limit wakes the parked loop, which takes a step');
{
    const { engine, notes, events } = grower({ vramLimit: 1 });
    engine.setVramLimit(Infinity);
    await settle();

    const all = notes();
    ok('the resume is announced', all.some((t) => /^The VRAM limit has room again/.test(t)),
        JSON.stringify(all));
    ok('the resume comes after the pause', all.findIndex((t) => /room again/.test(t)) === 1,
        JSON.stringify(all));
    // The step got past the limit and went on to download, which fails here
    // because no weights are served. That failure is what distinguishes a
    // resumed loop from one that only re-checked the limit.
    ok('and the loop went on to a real growth step', all.some((t) => /^Stopped growing at /.test(t)),
        JSON.stringify(all));
    ok('no grow was announced, since the download failed',
        !events.some((e) => e.type === 'grow'), JSON.stringify(events.map((e) => e.type)));

    // The download failure is terminal: a raised limit must not retry a GPU
    // OOM or a dead weights host.
    const after = all.length;
    engine.setVramLimit(Infinity);
    engine.setVramLimit(2 ** 40);
    await settle();
    ok('a later raise does not restart a loop that failed', notes().length === after,
        JSON.stringify(notes().slice(after)));
}

console.log('\na download that stalls is retried, and growth carries on');
{
    // The first two attempts at each position stall, as block_43.bin did on a
    // rented peer; the third succeeds.
    const tries = new Map();
    const load = async (p) => {
        const n = (tries.get(p) ?? 0) + 1;
        tries.set(p, n);
        if (n <= 2) throw new Error(`block_${p - 1}.bin: no data for 20000ms`);
        return pieceFor(p);
    };
    const { runner, notes, events } = grower({ vramLimit: Infinity, load, growRetryMs: [10, 20] });
    await new Promise((resolve) => setTimeout(resolve, 400));
    ok('the stall is reported as a retry, not a stop',
        notes().some((t) => /^Downloading .* failed .*Trying again in/.test(t))
        && !notes().some((t) => /^Stopped growing/.test(t)), JSON.stringify(notes()));
    ok('and growth went on past it', events.some((e) => e.type === 'grow'),
        JSON.stringify(events.map((e) => e.type)));
    ok('to positions it had to retry', runner.lo < 2 || runner.hi > 4,
        `holds ${runner.lo}-${runner.hi}`);
}

console.log('\na GPU out of memory still ends growth');
{
    const load = async () => { throw new Error('w.bin: GPU ran out of memory uploading its tensors'); };
    const { notes, events } = grower({ vramLimit: Infinity, load, growRetryMs: [10, 20] });
    await new Promise((resolve) => setTimeout(resolve, 100));
    ok('it stops, once', notes().filter((t) => /^Stopped growing/.test(t)).length === 1, JSON.stringify(notes()));
    ok('without a retry', !notes().some((t) => /Trying again/.test(t)), JSON.stringify(notes()));
    ok('and nothing grown', !events.some((e) => e.type === 'grow'));
}

// A GPU that allocates without complaint but loses its device when one pass
// runs more than `fits` positions, as D3D12 does on a shared-memory GPU. Each
// device it opens behaves the same way. `lose()` loses the current device.
function lossy({ fits, vramLimit = Infinity, minBlocks = 1 }) {
    const net = new Mesh({});
    net.ready.catch(() => {});
    Object.assign(net, {
        id: 0, blocks: L, positions: N, minBlocks, range: [2, 4],
        peers: [{ id: 0, lo: 2, hi: 4, have: 4, ready: true }],
    });
    net.send = () => false;
    const claims = [];
    net.claim = async (cap, at) => { claims.push(at); net.range = at; return at; };
    const released = [];
    const release = net.release.bind(net);
    net.release = () => { released.push(true); release(); };

    let engine = null, lose = null, opened = 0;
    const open = () => {
        opened++;
        const { device } = makeGpu();
        device.lost = new Promise((resolve) => { lose = resolve; });
        const submit = device.queue.submit;
        device.queue.submit = (cmds) => {
            const r = engine?.runner;
            if (r && r.have - r.lo > fits) lose({ reason: 'unknown', message: 'MakeResident failed' });
            return submit(cmds);
        };
        return device;
    };

    const device = open();
    const runner = new Runner(device, D, weightsFor(2, 4),
        { lo: 2, hi: 4, have: 4, nMax: N_MAX, tMax: 8, slots: SLOTS, idleMs: 0 });
    runner.lastJobAt = -60000;

    const events = [];
    engine = Engine.forTest({
        net, router: null, runner, dims: D, device, vramLimit, grow: true,
        load: async (p) => pieceFor(p), openDevice: async () => open(),
        serveArgs: { sessions: undefined, slots: SLOTS, ctx: N_MAX, autoGrow: true },
        onEvent: (type, detail) => events.push({ type, detail }),
    });
    const codes = () => events.filter((e) => e.type === 'status').map((e) => e.detail.code);
    const grown = () => events.filter((e) => e.type === 'grow').map((e) => [e.detail.lo, e.detail.hi]);
    return { engine, events, codes, grown, claims, released, lose: (i) => lose(i), opened: () => opened };
}

console.log('\na GPU that loses its device past what one pass can run rolls back and stops');
{
    const { engine, codes, grown, claims, released, opened } = lossy({ fits: 4 });
    await new Promise((resolve) => setTimeout(resolve, 300));
    const r = engine.runner;
    ok('it holds the most positions a pass ran on', r && r.have - r.lo === 4,
        r ? `holds ${r.lo}-${r.have}` : 'no runner');
    ok('the step that failed was never announced', grown().every(([lo, hi]) => hi - lo <= 4),
        JSON.stringify(grown()));
    ok('it gave the holding back, then claimed the good one on a new device',
        released.length === 1 && claims.length === 1 && claims[0][1] - claims[0][0] === 4
        && opened() === 2, `released ${released.length}, claims ${JSON.stringify(claims)}, opened ${opened()}`);
    ok('the reload is reported, then lending again, then the ceiling',
        JSON.stringify(codes().filter((c) => /gpu|ceiling/.test(c)))
        === JSON.stringify(['gpu-recovering', 'gpu-recovered', 'ceiling']), JSON.stringify(codes()));
    ok('and no growth failure or GPU-lost error', !codes().some((c) => c === 'grow-stopped' || c === 'gpu-lost'),
        JSON.stringify(codes()));
    ok('the tab is a worker again', engine.role === 'worker', engine.role);
}

console.log('\na loss outside a check drops one position');
{
    // Parked at the limit, so the loss comes from "a real job", not a check.
    const { engine, codes, claims, lose } = lossy({ fits: 99, vramLimit: 1 });
    await settle();
    lose({ reason: 'unknown', message: 'MakeResident failed' });
    await new Promise((resolve) => setTimeout(resolve, 200));
    ok('it reloads one position fewer', JSON.stringify(claims) === JSON.stringify([[2, 3]]),
        JSON.stringify(claims));
    const r = engine.runner;
    ok('and holds it', r && r.lo === 2 && r.have === 3, r ? `holds ${r.lo}-${r.have}` : 'no runner');
    ok('growth stays at that size', codes().includes('ceiling'), JSON.stringify(codes()));
}

console.log('\na loss with nothing smaller to hold stops lending');
{
    const { engine, codes, claims, lose } = lossy({ fits: 99, vramLimit: 1, minBlocks: 2 });
    await settle();
    lose({ reason: 'unknown', message: 'MakeResident failed' });
    await settle();
    ok('no reload below the minimum', claims.length === 0, JSON.stringify(claims));
    ok('the GPU-lost error is shown', codes().includes('gpu-lost'), JSON.stringify(codes()));
    ok('and the tab is a spectator', engine.role === 'spectator' && !engine.runner, engine.role);
}

console.log('\na destroyed device is not a loss');
{
    const { codes, claims, lose } = lossy({ fits: 99, vramLimit: 1 });
    await settle();
    lose({ reason: 'destroyed', message: '' });
    await settle();
    ok('nothing happens', claims.length === 0 && !codes().some((c) => /gpu/.test(c)), JSON.stringify(codes()));
}

console.log('\na tab pinned by ?at=/?blocks=/?gb= has no loop to wake');
{
    const { engine, runner, notes } = grower({ vramLimit: 1, grow: false });
    ok('no note while growth was never started', notes().length === 0, JSON.stringify(notes()));
    engine.setVramLimit(Infinity);
    await settle();
    ok('raising the limit stays silent', notes().length === 0, JSON.stringify(notes()));
    ok('and the holding is unchanged', runner.lo === 2 && runner.hi === 4,
        `holds ${runner.lo}-${runner.hi}`);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
