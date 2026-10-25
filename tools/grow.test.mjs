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
