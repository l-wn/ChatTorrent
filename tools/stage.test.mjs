// Tests the Runner's position arithmetic in Node, against this property:
//
//   A chain of peers covering the position line dispatches the same work as a
//   single tab, in the same order.
//
// web/chat.html runs `wte`, then blocks 0..L-1, then the final norm and the LM
// head. If a mesh of peers runs the same list in the same order, the split is
// correct: the kernels, the block code and the sampler are shared with the
// single-tab path and are checked against the oracle. What remains to test is
// which weights are bound at which cursor, which is index arithmetic and needs
// no GPU.
//
// The test uses a stub device that records the buffers each dispatch binds.
// Every weight tensor is a tagged marker, and the tag sequence a walk produces
// is compared with the single-tab one. No WebGPU, WebRTC or browser.
//
//   node tools/stage.test.mjs

import { Runner, MAX_QUEUE } from '../web/model/stage.js';
import { device, D, L, N, SINGLE, BLOCK_TENSORS, marker, weightsFor, pieceFor,
         resetTrace, takeTrace, makeGpu, DH, LAYER_TYPES, hybridWeightsFor } from './stub-gpu.mjs';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `\n        got  ${JSON.stringify(got)}\n        want ${JSON.stringify(want)}`);

// The stub device, the tiny model and the weight markers are in
// tools/stub-gpu.mjs, which tools/mesh.test.mjs also uses, so both suites
// share one definition of what a peer holding [lo, hi) has loaded.

// `idleMs: 0` throughout this suite. These tests cover the mechanics of
// eviction: which session's KV is taken, and what happens to that session. At
// the default IDLE_MS a session whose last job just ran cannot be evicted, and
// the newcomer is refused. Refusal has its own tests below ("a peer at
// capacity refuses").
const runner = (lo, hi, slots = 1) =>
    new Runner(device, D, weightsFor(lo, hi), { lo, hi, nMax: 16, tMax: 4, slots, idleMs: 0 });

// Walk a cursor from 0 to N across a chain of runners, as route.js's route()
// does: pass the job to the runner that holds the cursor. Only running blocks
// moves the cursor.
async function walk(cuts, { session = 's', T = 1, past = 0 } = {}) {
    const chain = [];
    for (let i = 0; i < cuts.length - 1; i++) chain.push(runner(cuts[i], cuts[i + 1]));

    resetTrace();
    let pos = 0, hidden = null, token;
    let hops = 0;
    while (pos < N) {
        const r = chain.find((x) => x.holds(pos));
        if (!r) throw new Error(`nobody holds position ${pos}`);
        const h = { k: 'job', s: session, pos, past, n: T, ids: Array(T).fill(1), opts: {} };
        const out = await r.run(h, hidden);
        hidden = out.hidden ?? null;
        token = out.token ?? token;
        // Where the Runner stopped, not where its range ends. A session has
        // its own cut, and after growth the two differ.
        pos = out.at;
        if (++hops > N + 2) throw new Error('the cursor stopped advancing');
    }
    return { trace: takeTrace(), token, hops };
}

// ---

console.log('\none peer holding everything: the reference trace');
{
    const { trace, hops } = await walk([0, N]);
    eq('a single peer dispatches wte, every block, norm, head', trace, SINGLE);
    eq('in one hop', hops, 1);
}

console.log('\nthe ends are positions: a peer can hold one and no blocks');
{
    const embed = runner(0, 1);
    eq('an embedding-only peer loads no blocks', embed.w.layers.length, 0);
    ok('and runs the gather', embed.embed === true && embed.head === false);

    const head = runner(N - 1, N);
    eq('a head-only peer loads no blocks', head.w.layers.length, 0);
    ok('and runs the norm and the LM head', head.head === true && head.embed === false);

    const { trace } = await walk([0, 1, N - 1, N]);
    eq('the two ends and one middle peer still reproduce it', trace, SINGLE);
}

console.log('\nevery way of cutting the line: the split must be invisible');
{
    // 2^(N-1) arrangements: each interior position is or is not a cut.
    let bad = 0, arrangements = 0, worst = 0;
    for (let mask = 0; mask < 1 << (N - 1); mask++) {
        const cuts = [0];
        for (let x = 1; x < N; x++) if (mask & (1 << (x - 1))) cuts.push(x);
        cuts.push(N);
        const { trace, hops } = await walk(cuts, { session: `s${mask}` });
        arrangements++;
        worst = Math.max(worst, hops);
        if (JSON.stringify(trace) !== JSON.stringify(SINGLE)) bad++;
    }
    ok(`every chain dispatches the single-tab list (${arrangements} arrangements, ` +
        `longest ${worst} hops)`, bad === 0, `${bad} disagreed`);
}

console.log('\nentering at an interior cursor: overlap is what this is for');
{
    // A peer holding [2, 7) that receives a job at 4 must run positions 4..6
    // (blocks 3..5), not everything it holds. Otherwise the job would re-run
    // blocks whose KV rows are already written.
    const r = runner(2, 7);
    resetTrace();
    await r.run({ k: 'job', s: 'interior', pos: 4, past: 0, n: 1, opts: {} },
        new Float32Array(D.hidden));
    eq('a cursor at 4 runs only the blocks from there on', takeTrace(), ['L3', 'L4', 'L5']);

    resetTrace();
    await r.run({ k: 'job', s: 'whole', pos: 2, past: 0, n: 1, opts: {} },
        new Float32Array(D.hidden));
    eq('and a cursor at its own lo runs all of them', takeTrace(),
        ['L1', 'L2', 'L3', 'L4', 'L5']);
}

console.log('\nthe checks that keep a stale route from being answered');
{
    const r = runner(2, 5);
    const threw = await r.run({ k: 'job', s: 'x', pos: 6, past: 0, n: 1, opts: {} },
        new Float32Array(D.hidden)).then(() => null, (e) => e.message);
    ok('a cursor outside the range is refused', /position 6 is not held here/.test(threw),
        `got ${threw}`);

    const short = await r.run({ k: 'job', s: 'y', pos: 2, past: 0, n: 2, opts: {} },
        new Float32Array(D.hidden)).then(() => null, (e) => e.message);
    ok('a payload of the wrong length is refused', /expected 16/.test(short), `got ${short}`);

    const cold = await r.run({ k: 'job', s: 'never-seen', pos: 2, past: 4, n: 1, opts: {} },
        new Float32Array(D.hidden)).then(() => null, (e) => e.message);
    ok('a session with history but no cache is refused, not answered cold',
        /no longer cached here/.test(cold), `got ${cold}`);
}

console.log('\nwarmup: entered at lo, so every position held is exercised');
{
    for (const [lo, hi, want] of [
        [0, N, SINGLE],
        [0, 1, ['wte']],
        [N - 1, N, ['norm', 'head']],
        [3, 6, ['L2', 'L3', 'L4']],
    ]) {
        resetTrace();
        await runner(lo, hi).warmup();
        eq(`warmup of [${lo}, ${hi}) touches what it holds`, takeTrace(), want);
    }
}

console.log("\nlastJobAt: a warm-up is not real traffic");
{
    // #growClaim() in engine.js reads lastJobAt to delay a growth step while
    // this peer is serving jobs. A warm-up pass, which growth itself runs,
    // must not count as a job.
    const r = runner(0, N);
    eq('a freshly built Runner has done no real work yet', r.lastJobAt, 0);
    await r.warmup();
    eq('and warming up does not count as real work', r.lastJobAt, 0);
    await r.run({ k: 'job', s: 'real', pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null);
    ok('but an ordinary job does', r.lastJobAt > 0);
}

// --- append: a range loaded one position at a time ---

console.log('\nappend: filling a claim one position at a time');
{
    const r = new Runner(device, D, { layers: [] },
        { lo: 0, hi: N, have: 0, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });

    ok('an empty Runner holds nothing', r.holds(0) === false);
    r.append(pieceFor(0));
    ok('and serves the embedding as soon as it lands', r.holds(0) && !r.holds(1));
    for (let p = 1; p < 4; p++) r.append(pieceFor(p));
    eq('have is the leading edge of the download', r.have, 4);
    ok('a position past it is not held yet', r.holds(4) === false);

    // A Runner loaded one position at a time must behave the same as one given
    // the same range at once.
    for (let p = 4; p < N; p++) r.append(pieceFor(p));
    resetTrace();
    await r.run({ k: 'job', s: 'g', pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null);
    eq('a fully appended Runner dispatches the single-tab list', takeTrace(), SINGLE);

    // The KV cache is sized for the whole claim from the start, so an appended
    // block indexes rows that already exist and nothing is reallocated.
    eq('and its cache was allocated for the claim, not the first block',
        r.free.length + r.live.size, 1);
}

console.log('\nappend: positions arrive in order, and only in order');
{
    const r = new Runner(device, D, { layers: [] },
        { lo: 2, hi: 5, have: 2, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });
    const threw = (f) => { try { f(); return null; } catch (e) { return e.message; } };

    ok('a block where a block belongs is fine', threw(() => r.append(pieceFor(2))) === null);
    ok('the embedding cannot be appended in the middle',
        /needs its tensors/.test(threw(() => r.append({ wte: marker('wte') }))));
    r.append(pieceFor(3));
    r.append(pieceFor(4));
    ok('and a full range has nothing left to take',
        /nothing left to append/.test(threw(() => r.append(pieceFor(5)))));
}

// --- growLeft/growRight/prepend: widening a claim that is being served ---

console.log('\ngrowLeft/growRight: a claim widening one position at a time');
{
    // Start in the middle of the line and grow outward in both directions, as
    // #growClaim() in engine.js does, until the whole model is held. At every
    // step, growth must add a KV block only when the new position is a block,
    // not at either end.
    const mid = Math.floor(N / 2);
    const r = new Runner(device, D, weightsFor(mid, mid + 1),
        { lo: mid, hi: mid + 1, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });

    while (r.lo > 0 || r.hi < N) {
        if (r.lo > 0) {
            const addsBlock = r.lo > 1, before = r.blocks, newLo = r.lo - 1;
            const reserved = r.growLeft();
            eq('growLeft() itself never touches blocks: prepend() commits it',
                r.blocks, before);
            r.prepend(pieceFor(newLo), reserved);
            eq(`prepend onto position ${newLo} ${addsBlock ? 'adds' : 'adds no'} a block`,
                r.blocks, before + (addsBlock ? 1 : 0));
            ok(`and moved lo to ${newLo}`, r.lo === newLo);
        }
        if (r.hi < N) {
            const addsBlock = r.hi < N - 1, before = r.blocks, newPos = r.hi;
            r.growRight();
            eq(`growRight onto position ${newPos} ${addsBlock ? 'adds' : 'adds no'} a block`,
                r.blocks, before + (addsBlock ? 1 : 0));
            r.append(pieceFor(newPos));
            ok(`append moved have to ${newPos + 1}`, r.have === newPos + 1);
        }
    }

    resetTrace();
    await r.run({ k: 'job', s: 'grown', pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null);
    eq('a Runner grown from the middle out dispatches the single-tab list', takeTrace(), SINGLE);
    eq('and holds every block exactly once', r.blocks, L);
}

console.log('\nthe two ends add no KV block when a claim grows into them');
{
    const r = runner(1, N - 1, 1);      // every position except the embedding and the head
    const before = r.blocks;

    r.prepend(pieceFor(0), r.growLeft());
    eq('growing into position 0 (the embedding) adds no block', r.blocks, before);
    ok('and it is now held', r.embed === true);

    r.growRight();
    r.append(pieceFor(N - 1));
    eq('growing into the head adds no block', r.blocks, before);
    ok('and it is now held', r.head === true);
}

console.log("\ngrowing must not disturb a live session's cache");
{
    // One slot with a live session, then the claim grows on both edges. Growth
    // must update the live session's frame, which is in `live`, not in `free`.
    const r = runner(2, 5, 1);
    const job = (s, past) => ({ k: 'job', s, pos: 2, past, n: 1, ids: [1], opts: {} });
    await r.run(job('live', 0), new Float32Array(D.hidden));
    ok('the session is live, not idle', r.live.has('live') && r.free.length === 0);

    const before = r.blocks;
    r.growRight();
    r.append(pieceFor(5));
    r.prepend(pieceFor(1), r.growLeft());
    eq('both edges added a block', r.blocks, before + 2);

    const after = await r.run(job('live', 1), new Float32Array(D.hidden))
        .then(() => null, (e) => e.message);
    ok('the live session survives growth on both edges', after === null, `got ${after}`);
    eq("its cache grew to match the wider claim", r.live.get('live').K.length, r.blocks);
}

// --- The cut: growth must not affect a session that is already running ---

console.log('\nwarming up must not evict a live chat');
{
    // One slot with a live session, then a block is appended. Warming the new
    // block uses its own cache, so the live session stays cached.
    const r = runner(1, 7, 1);
    r.have = 3;
    const job = (s, past) => ({ k: 'job', s, pos: 1, past, n: 1, ids: [1], opts: {} });
    await r.run(job('live', 0), new Float32Array(D.hidden));

    r.have = 4;
    await r.warmup(3);

    const after = await r.run(job('live', 1), new Float32Array(D.hidden))
        .then(() => null, (e) => e.message);
    ok('the live session survives a warm-up next to it', after === null, `got ${after}`);
    eq('and the pool still holds exactly its one session', r.live.size, 1);
}

console.log('\nthe cut: new blocks only ever serve new sessions');
{
    // A peer that claimed positions 1–6 and has downloaded up to 4. The KV
    // cache is sized for the whole claim, so growth changes only `w.layers`,
    // and setting `have` simulates a partial download, as Runner.append() does
    // when the loader fills a range step by step.
    // Four slots, so the sessions below coexist. Eviction is tested in the
    // next block.
    const r = runner(1, 7, 4);
    r.have = 4;

    const job = (s, past = 0) =>
        ({ k: 'job', s, pos: 1, past, n: 1, ids: [1], opts: {} });
    const run = async (s, past = 0, want) => {
        resetTrace();
        const out = await r.run(job(s, past), new Float32Array(D.hidden), want);
        return { at: out.at, trace: takeTrace() };
    };

    const first = await run('a');
    eq('a session runs what its peer holds when it arrives', first.trace, ['L0', 'L1', 'L2']);
    eq('and exits at the holding boundary', first.at, 4);

    // Two more blocks are loaded.
    r.have = 6;

    const again = await run('a', 1);
    eq('the same session still runs exactly those blocks', again.trace, ['L0', 'L1', 'L2']);
    eq('and still exits where it did', again.at, 4);

    const fresh = await run('b');
    eq('a new session gets the new blocks', fresh.trace, ['L0', 'L1', 'L2', 'L3', 'L4']);
    eq('and exits further along', fresh.at, 6);

    // A router asking for an earlier exit gets it, which is how a peer routes
    // around a hole at its own boundary. Asking for more than the peer holds
    // is clamped.
    const short = await run('c', 0, 3);
    eq('a shorter exit than the holding is honoured', [short.trace, short.at], [['L0', 'L1'], 3]);
    const over = await run('d', 0, 99);
    eq('one past the holding is clamped to it', over.at, 6);

    // An existing cut overrides the proposal, whether it asks for more or less.
    const later = await run('c', 1, 6);
    eq('a pinned session ignores a later proposal', [later.trace, later.at], [['L0', 'L1'], 3]);
}

console.log('\nthe cut is evicted with the KV it describes');
{
    const r = runner(1, 7);          // slots: 1, so the second session evicts the first
    r.have = 4;
    const run = async (s, past = 0) => (await r.run(
        { k: 'job', s, pos: 1, past, n: 1, ids: [1], opts: {} },
        new Float32Array(D.hidden))).at;

    eq('a session pins its cut', await run('a'), 4);
    r.have = 7;
    eq('another evicts it', await run('b'), 7);
    // 'a' is gone from both maps. A job for it with past > 0 would throw. With
    // past 0 it is a new session.
    eq('and the evicted one is a new session, not a stale cut', await run('a'), 7);
}

console.log('\nthe boundary log: the rows handed downstream, kept');
{
    const H = D.hidden;
    const job = (s, past, n, opts = {}) =>
        ({ k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {}, ...opts });

    const r = runner(0, 4, 2);       // two slots, so the sessions below coexist
    eq('a Runner that has handed nothing over holds no log', r.logs.size, 0);

    await r.run(job('a', 0, 2), null);
    // route.js looks the log up by boundary: a repair starts from a broken
    // hand-off at a boundary, and `routes` and `upstream` use the same key. A
    // log keyed by the entry cursor (0 here) could not be found.
    eq('the log is keyed by the cursor the rows leave at, not the one they arrived at',
        [...r.logs.get('a').keys()], [4]);
    eq('and its length is the rows fed so far', r.logs.get('a').get(4).len, 2);

    await r.run(job('a', 2, 3), null);
    eq('which grows with the context', r.logs.get('a').get(4).len, 5);

    // rewind() drops the tail of the enqueuer's history when a turn is cut, so
    // the next job arrives at a lower `past` and overwrites those rows in the
    // downstream KV caches. The log must shrink to match.
    await r.run(job('a', 1, 1), null);
    eq('and falls again when a job arrives at a lower past', r.logs.get('a').get(4).len, 2);

    // The stub GPU reads back zeros, so to check the offset the array is
    // filled with a marker value first, and the test checks which slice the
    // run overwrote. Writing at 0 and not at `past * hidden` would make a
    // replay feed a peer the wrong prefix.
    const log = r.logs.get('a').get(4);
    log.rows.fill(7);
    await r.run(job('a', 2, 2), null);
    ok('rows land at past * hidden',
        log.rows.subarray(0, 2 * H).every((x) => x === 7)
        && log.rows.subarray(2 * H, 4 * H).every((x) => x === 0));

    // The log grows on demand and is not allocated for all of nMax. A
    // 4096-token window would otherwise cost 84 MiB of host RAM per session
    // per boundary, even for a twenty-token chat. Growing copies into a larger
    // array, and the existing rows must be preserved, or a replay would feed a
    // peer zeros.
    const big = new Runner(device, D, weightsFor(0, 4),
        { lo: 0, hi: 4, nMax: 1024, tMax: 4, slots: 1, idleMs: 0 });
    await big.run(job('g', 0, 4), null);
    const glog = big.logs.get('g').get(4);
    const startCap = glog.rows.length / H;
    ok('a fresh log is a chunk, not the whole window',
        startCap < 1024, `got ${startCap}`);
    glog.rows.fill(7, 0, 4 * H);          // mark the rows that must be preserved

    for (let past = 4; past < startCap + 4; past += 4) await big.run(job('g', past, 4), null);
    ok('it grew once the context passed the chunk',
        glog.rows.length / H > startCap, `still ${glog.rows.length / H}`);
    ok('and never past the window', glog.rows.length / H <= 1024);
    ok('the rows written before the growth are still there',
        glog.rows.subarray(0, 4 * H).every((x) => x === 7));
    eq('and len still counts the whole context', glog.len, startCap + 4);

    // Nothing is downstream of the head, so it keeps no log.
    const tail = runner(4, N);
    await tail.run({ k: 'job', s: 'a', pos: 4, past: 0, n: 1, opts: {} },
                   new Float32Array(H));
    eq('a peer holding the head keeps no log', tail.logs.size, 0);

    const warm = runner(0, 4);
    await warm.warmup();
    eq('and a warm-up leaves none behind', warm.logs.size, 0);

    await r.run(job('b', 0, 1), null);
    eq('two sessions on one peer keep separate logs', [...r.logs.keys()], ['a', 'b']);
}

console.log('\nthe log is evicted with the KV it describes');
{
    const r = runner(0, 4);          // slots: 1, so the second session evicts the first
    const run = (s) => r.run({ k: 'job', s, pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null);

    await run('a');
    ok('a session logs what it handed over', r.logs.has('a'));
    await run('b');
    // Eviction drops the log with the frame. A replay from it would only help
    // for one hop, since the next job throws here with no cache. Dropping logs
    // with frames also bounds their memory, because `live` is capped at
    // `slots`.
    ok('and loses it with its cut when a third chat evicts it', !r.logs.has('a'));
    eq('while the session that evicted it keeps its own', [...r.logs.keys()], ['b']);
}

console.log('\nreplay: the KV is written, nothing is sampled');
{
    const H = D.hidden;
    const r = runner(0, N);
    const job = (s, past, n, opts = {}) =>
        ({ k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {}, ...opts });

    resetTrace();
    const done = await r.run(job('a', 0, 2, { replay: true }), null);
    const trace = takeTrace();

    eq('a replay reaching the end of the line advances the cursor', done.at, N);
    ok('and returns neither a token nor hidden state: the third shape route.js reads',
        done.token === undefined && done.hidden === undefined);
    // The norm and the LM head are not dispatched at all. That matmul is
    // hidden x vocab, the widest in the model, and a replay does not use its
    // output.
    eq('having dispatched no norm and no LM head', trace, ['wte', ...SINGLE.slice(1, 1 + L)]);
    eq('and it keeps no log, being the end of the line', r.logs.size, 0);

    // The replay wrote its KV rows, so an ordinary job continuing from there
    // finds a cache.
    const next = await r.run(job('a', 2, 1), null);
    eq('but the KV rows it came to write are there for the job that follows',
        next.token !== undefined, true);

    // A replay differs only at the end of the line. A middle peer must hand it
    // on, or the caches after it are not rebuilt.
    const mid = runner(0, 4);
    const on = await mid.run(job('b', 0, 2, { replay: true }), null);
    ok('a replay stopping short of the end is an ordinary job',
        on.at === 4 && on.hidden?.length === 2 * H);
    eq('and is logged like one', mid.logs.get('b').get(4).len, 2);
}

// --- Timing: model/gpuprof.js on the job path ---

console.log('\ntiming: a profiled job dispatches exactly what an unprofiled one does');
{
    // Each level uses its own stub GPU, so the pass counts below are per run.
    const job = (s, prof) => ({ k: 'job', s, pos: 0, past: 0, n: 1, ids: [1], opts: {}, prof });
    for (const [prof, timestamps] of [[0, true], [1, true], [2, true], [1, false], [2, false]]) {
        const gpu = makeGpu({ timestamps });
        const r = new Runner(gpu.device, D, weightsFor(0, N), { lo: 0, hi: N, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });
        gpu.resetTrace();
        const out = await r.run(job(`p${prof}`, prof), null);
        const what = `prof ${prof}${timestamps ? '' : ', no timestamp-query'}`;
        eq(`${what}: the single-tab list`, gpu.takeTrace(), SINGLE);
        const PH = ['queue', 'encode', 'submit', 'wait', 'copy', 'sample'];
        ok(`${what}: phases present and non-negative`,
            PH.every((p) => typeof out.phases?.[p] === 'number' && out.phases[p] >= 0)
            && out.phases.dispatches > 0, `got ${JSON.stringify(out.phases)}`);

        const dispatches = out.phases.dispatches;
        if (prof === 0) {
            ok(`${what}: no profile, one untimed pass`, out.gpu === undefined
                && gpu.passes.opened === 1 && gpu.passes.timed === 0);
        } else if (prof === 1) {
            ok(`${what}: one pass, timed only if it can be`,
                gpu.passes.opened === 1 && gpu.passes.timed === (timestamps ? 1 : 0)
                && gpu.passes.resolved === (timestamps ? 1 : 0) && out.gpu.ts === timestamps);
        } else if (timestamps) {
            ok(`${what}: one timed pass per dispatch`,
                gpu.passes.opened === dispatches && gpu.passes.timed === dispatches,
                `${JSON.stringify(gpu.passes)} for ${dispatches} dispatches`);
        } else {
            ok(`${what}: nothing to time, so the pass is not split`,
                gpu.passes.opened === 1 && gpu.passes.timed === 0);
        }

        if (prof === 2) {
            const names = Object.keys(out.gpu.kernels);
            const count = Object.values(out.gpu.kernels).reduce((a, k) => a + k.n, 0);
            eq(`${what}: every dispatch filed once`, count, dispatches);
            ok(`${what}: filed under the kind of layer it ran in`,
                names.includes('embed/gather_embed') && names.includes('full/add_inplace')
                && names.includes(`head/matmul f32 ${D.hidden}→${D.vocab}`)
                && names.includes(`head/rmsnorm ${D.hidden}`)
                && !names.some((n) => n.startsWith('head/') && n.includes('attn')),
                `got ${JSON.stringify(names)}`);
            ok(`${what}: gpu ms only when there were timestamps`,
                Object.values(out.gpu.kernels).every((k) => (k.gpuMs === null) === !timestamps
                    && k.cpuMs >= 0));
        }
    }

    // A second profiled job on the same device reuses the pool without growing
    // it, and resolves.
    const gpu = makeGpu({ timestamps: true });
    const r = new Runner(gpu.device, D, weightsFor(0, N), { lo: 0, hi: N, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });
    await r.run(job('again', 2), null);
    const second = await r.run({ ...job('again', 2), past: 1 }, null);
    ok('a second profiled job on the device reads back too', second.gpu?.kernels !== undefined);
}

console.log('\ntiming: nothing is profiled that nobody is waiting on');
{
    const gpu = makeGpu({ timestamps: true });
    const r = new Runner(gpu.device, D, weightsFor(0, N), { lo: 0, hi: N, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });
    const replay = await r.run({ k: 'job', s: 'rp', pos: 0, past: 0, n: 1, ids: [1], opts: {},
                                 replay: true, prof: 2 }, null);
    ok('a replay returns no profile and opens no timed pass',
        replay.gpu === undefined && gpu.passes.timed === 0);
    await r.warmup();
    ok('nor does a warm-up', gpu.passes.timed === 0);

    const mid = new Runner(gpu.device, D, weightsFor(0, 4), { lo: 0, hi: 4, nMax: 16, tMax: 4, slots: 1, idleMs: 0 });
    const out = await mid.run({ k: 'job', s: 'mid', pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null);
    ok('a peer that hands on logs instead of sampling',
        out.phases.log >= 0 && out.phases.sample === undefined);
}

console.log('\nthe page pool: pages are per peer, frames are per session');
{
    // nMax 16 clamps the page to 16 tokens, so one page is one session's whole
    // window and the pool holds `slots` pages. With three frames and one page,
    // pages run out before frames do.
    const r = new Runner(device, D, weightsFor(0, 4),
        { lo: 0, hi: 4, nMax: 16, tMax: 4, slots: 1, sessions: 3, idleMs: 0 });
    const job = (s, past, n = 1) => r.run(
        { k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {} }, null);

    eq('the pool is sized slots x pages-per-session', r.pool.nPages, 1);
    eq('a page is clamped to the window', r.pool.pageTokens, 16);
    eq('and there are more frames than pages', r.free.length, 3);

    await job('a', 0);
    eq('the first session takes the only page', r.pool.free.length, 0);
    eq('and one frame', r.free.length, 2);

    await job('b', 0);
    eq('a second session finds a free frame but no page, so it takes the page',
        [r.pool.free.length, r.live.size], [0, 1]);

    let err = null;
    await job('a', 1).catch((e) => { err = e; });
    ok('and the session whose pages went is cold, not silently answered',
        /no longer cached here/.test(err?.message ?? ''), `got ${err?.message}`);

    await r.release('b');
    eq('releasing gives the page back to the pool', r.pool.free.length, 1);
    eq('and the frame back to its own pool', r.free.length, 3);

    // A context wider than one page: two pages, and the table must list both.
    const wide = new Runner(device, D, weightsFor(0, 4),
        { lo: 0, hi: 4, nMax: 32, tMax: 4, slots: 2, sessions: 1, pageTokens: 16, idleMs: 0 });
    eq('a wider window is more pages per session', wide.maxPages, 2);
    eq('and the pool still holds slots x that', wide.pool.nPages, 4);
    await wide.run({ k: 'job', s: 'w', pos: 0, past: 0, n: 4,
                     ids: [1, 1, 1, 1], opts: {} }, null);
    eq('a job inside the first page takes one', wide.live.get('w').owned.length, 1);
    // Fed contiguously: a job may not start past the rows the cache holds.
    for (let past = 4; past <= 20; past += 4) {
        await wide.run({ k: 'job', s: 'w', pos: 0, past, n: 4,
                         ids: [1, 1, 1, 1], opts: {} }, null);
    }
    eq('a job past the page boundary takes the next',
        wide.live.get('w').owned.length, 2);
    ok('and the two pages are distinct physical pages',
        new Set(wide.live.get('w').owned).size === 2);
}

console.log('\na peer at capacity refuses instead of robbing a live chat');
{
    // The default policy: one frame, held by a session that has just run. The
    // new session is refused and the existing one keeps its cache. Under plain
    // LRU the new session would take the cache, and the existing one would
    // fail on its next job, mid-reply.
    // `waitMs: 0` gives an immediate refusal. Waiting is tested further down.
    const r = new Runner(device, D, weightsFor(0, 4),
        { lo: 0, hi: 4, nMax: 16, tMax: 4, slots: 1, idleMs: 60_000, waitMs: 0 });
    const job = (s, past, n = 1) => r.run(
        { k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {} }, null);

    await job('a', 0);
    let err = null;
    await job('b', 0).catch((e) => { err = e; });
    ok('the second chat is refused', /no session free/.test(err?.message ?? ''),
        `got ${err?.message}`);
    ok('and told to try again', /try again/.test(err?.message ?? ''));
    ok('marked as capacity, not failure, so nothing repairs it', err?.busy === true);
    eq('it is counted as a refusal, not an eviction',
        [r.counts.refused, r.counts.evicted], [1, 0]);

    // The existing session still has its cache.
    const again = await job('a', 1);
    ok('the chat that was already here keeps running', again.token !== undefined
        || again.hidden !== undefined);
    eq('and nothing went cold', r.counts.cold, 0);

    // After it is released, another session can take the frame.
    await r.release('a');
    err = null;
    await job('b', 0).catch((e) => { err = e; });
    ok('after the incumbent ends, the waiting chat is admitted', err === null,
        `got ${err?.message}`);
    eq('with no eviction anywhere in the story', r.counts.evicted, 0);
}

console.log('\nan idle chat is still reclaimed, so a lost `free` costs a frame briefly');
{
    // An idle session is reclaimed. Without this, a tab that closed without a
    // clean `free` (a dropped message, a truncated walk, a repaired-away peer)
    // would hold a frame forever. `idleMs` is short enough here for the
    // reclaim to happen within the test.
    const r = new Runner(device, D, weightsFor(0, 4),
        { lo: 0, hi: 4, nMax: 16, tMax: 4, slots: 1, idleMs: 20 });
    const job = (s, past, n = 1) => r.run(
        { k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {} }, null);

    await job('gone', 0);
    await new Promise((res) => setTimeout(res, 40));
    let err = null;
    await job('new', 0).catch((e) => { err = e; });
    ok('an abandoned chat is evicted once it has gone quiet', err === null,
        `got ${err?.message}`);
    eq('as an eviction, which is what it is', [r.counts.evicted, r.counts.refused], [1, 0]);
}

console.log('\na chat with no frame waits in line for one, first come first served');
{
    const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
    const make = (opts) => {
        const r = new Runner(device, D, weightsFor(0, 4),
            { lo: 0, hi: 4, nMax: 16, tMax: 4, slots: 1, ...opts });
        const order = [];
        const job = (s, past, n = 1) => r.run(
            { k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {} }, null)
            .then((x) => { order.push(`${s}@${past}`); return x; },
                  (e) => { order.push(`${s}@${past}!`); throw e; });
        return { r, job, order };
    };

    // One frame, held by a chat that goes idle. `b` and `c` wait in line and
    // are served in arrival order, each as soon as the frame is idle. If they
    // were refused and retried on timers, either could get the frame first.
    {
        const { r, job, order } = make({ idleMs: 40, waitMs: 2000 });
        await job('a', 0);
        const b = job('b', 0), c = job('c', 0);
        await sleep(10);
        eq('nobody is refused, and nobody is admitted over a chat that just ran',
            [r.counts.refused, r.counts.evicted, r.waiting.length], [0, 0, 2]);
        await Promise.all([b, c]);
        eq('they are served in the order they came', order, ['a@0', 'b@0', 'c@0']);
        eq('each taking the frame from a chat that had gone idle',
            [r.counts.evicted, r.counts.refused, r.counts.waited], [2, 0, 2]);
        await sleep(30);
        ok('and the line is empty, with its timer stopped',
            r.waiting.length === 0 && r.waitTimer === undefined);
    }

    // An active chat keeps its frame, and its tokens are not delayed by the
    // waiting session.
    {
        const { r, job, order } = make({ idleMs: 60, waitMs: 2000 });
        await job('a', 0);
        const b = job('b', 0);
        for (let i = 1; i <= 4; i++) { await sleep(20); await job('a', i); }
        eq('the incumbent ran on while the newcomer waited', order, ['a@0', 'a@1', 'a@2', 'a@3', 'a@4']);
        eq('with nothing cold and nothing evicted', [r.counts.cold, r.counts.evicted], [0, 0]);
        await b;
        eq('and the newcomer is served once it stops', order.at(-1), 'b@0');
    }

    // A chat whose next job waits in the queue longer than `idleMs` is not
    // idle. Under load the queue gets that long, and a newcomer ahead of it in
    // the queue must not take its frame, or its job fails cold at the front.
    {
        const { r, job, order } = make({ idleMs: 40, waitMs: 2000 });
        await job('a', 0);
        r.queue = r.queue.then(() => sleep(80));     // a busy GPU, longer than idleMs
        const b = job('b', 0);                        // the newcomer, first in line
        let err = null;
        await job('a', 1).catch((e) => { err = e; });
        ok('a chat queued past idleMs keeps its frame', err === null, `got ${err?.message}`);
        eq('nothing cold and nothing evicted while it waited', [r.counts.cold, r.counts.evicted], [0, 0]);
        await b;
        eq('the newcomer waits and is served once the chat goes quiet', order, ['a@0', 'a@1', 'b@0']);
    }

    // Jobs of one waiting session stay in order. A replay sends its chunks
    // without waiting for answers, and the later chunks must not be rejected
    // as uncached.
    {
        const { r, job, order } = make({ idleMs: 40, waitMs: 2000 });
        await job('a', 0);
        const all = Promise.all([job('b', 0, 2), job('b', 2, 2), job('b', 4)]);
        await sleep(10);
        eq('all three wait', r.waiting.length, 3);
        await all;
        eq('and run in the order they were sent', order.slice(1), ['b@0', 'b@2', 'b@4']);
        eq('as one admission', [r.counts.waited, r.counts.admitted, r.counts.cold], [3, 2, 0]);
    }

    // A released frame is taken immediately, without waiting for the timer.
    {
        const { r, job } = make({ idleMs: 60_000, waitMs: 2000 });
        await job('a', 0);
        const t0 = performance.now();
        const b = job('b', 0);
        await sleep(10);
        await r.release('a');
        await b;
        ok('a released frame goes straight to the head of the line', performance.now() - t0 < 200,
            `took ${(performance.now() - t0).toFixed(0)}ms`);
        eq('with no eviction', r.counts.evicted, 0);
    }

    // The wait is bounded. After `waitMs` the job is refused with `busy`.
    {
        const { r, job } = make({ idleMs: 60_000, waitMs: 40 });
        await job('a', 0);
        let err = null;
        const t0 = performance.now();
        await job('b', 0).catch((e) => { err = e; });
        ok('a chat that waits too long is refused as busy',
            err?.busy === true && /no session free/.test(err.message), `got ${err?.message}`);
        ok('after its wait, not before', performance.now() - t0 >= 35);
        eq('the incumbent untouched', [r.counts.evicted, r.counts.refused, r.waiting.length], [0, 1, 0]);
        ok('and still running', (await job('a', 1)) !== undefined);
    }

    // A chat that ends while waiting is removed from the line.
    {
        const { r, job } = make({ idleMs: 60_000, waitMs: 2000 });
        await job('a', 0);
        let err = null;
        const b = job('b', 0).catch((e) => { err = e; });
        await sleep(10);
        await r.release('b');
        await b;
        ok('its job is answered, not left waiting', /ended while it waited/.test(err?.message ?? ''),
            `got ${err?.message}`);
        eq('and the line is empty', r.waiting.length, 0);
    }
}

// Every field of a job header comes from another peer and is validated.
console.log('\na job that does not fit the buffers is refused before any dispatch');
{
    const r = runner(0, N);
    const bad = async (h) => {
        resetTrace();
        let err = null;
        await r.run({ k: 'job', s: 'x', pos: 0, past: 0, opts: {}, ...h }, null)
            .catch((e) => { err = e; });
        return err !== null && takeTrace().length === 0;
    };
    ok('n = 0', await bad({ n: 0, ids: [] }));
    ok('n past tMax', await bad({ n: 5, ids: [1, 1, 1, 1, 1] }));
    ok('n not an integer', await bad({ n: 1.5, ids: [1] }));
    ok('a negative past', await bad({ n: 1, past: -1, ids: [1] }));
    ok('fewer ids than n', await bad({ n: 2, ids: [1] }));
    ok('a negative id', await bad({ n: 1, ids: [-1] }));
    ok('ids that are not integers', await bad({ n: 1, ids: ['1'] }));
    ok('and a well-formed job still runs',
        (await r.run({ k: 'job', s: 'y', pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null))
            .token !== undefined);
}

console.log('\na flooded runner refuses with `busy` instead of queueing without end');
{
    const r = runner(0, N);
    const job = (i) => r.run({ k: 'job', s: 'f', pos: 0, past: 0, n: 1, ids: [1], opts: {} }, null)
        .then(() => null, (e) => e);
    // Queued synchronously, so none has finished when the next arrives.
    const all = await Promise.all(Array.from({ length: MAX_QUEUE + 3 }, (_, i) => job(i)));
    const refused = all.filter((e) => e?.busy);
    eq(`everything past ${MAX_QUEUE} is refused as busy`, refused.length, 3);
    ok('and a job after the backlog drains runs', (await job()) === null);
}

console.log('\nrecurrent state cannot be rewound: a job must continue exactly where the last ended');
{
    // Gated DeltaNet folds every row it is fed into one state with no position
    // axis, so a tail cannot be overwritten as with KV. A job at a lower
    // `past` would run on top of rows the enqueuer has dropped. The Runner
    // counts the rows each block holds and refuses a job that does not match,
    // with the same error as an evicted cache, so it is handled the same way.
    const gpu = makeGpu();
    const hy = (lo, hi, opts = {}) => new Runner(gpu.device, DH, hybridWeightsFor(lo, hi),
        { lo, hi, nMax: 64, tMax: 8, slots: 2, idleMs: 0, ...opts });
    const ids = (n) => Array(n).fill(1);
    const at0 = (s, past, n) => ({ k: 'job', s, pos: 0, past, n, ids: ids(n), opts: {} });
    const refusal = (p) => p.then(() => null, (e) => e?.message ?? String(e));

    const r = hy(0, N);
    gpu.resetTrace();
    await r.run(at0('a', 0, 4), null);
    ok('the hybrid stub really runs linear blocks', gpu.takeTrace().includes('R0'));

    const cold = r.counts.cold;
    const back = await refusal(r.run(at0('a', 2, 1), null));
    ok('a job below what a linear block holds is refused',
        /no longer cached here/.test(back ?? ''), `got ${back}`);
    eq('and counted as a cold cache', r.counts.cold, cold + 1);
    const ahead = await refusal(r.run(at0('a', 6, 1), null));
    ok('and so is one above it', /no longer cached here/.test(ahead ?? ''), `got ${ahead}`);
    ok('while the exact continuation runs', (await refusal(r.run(at0('a', 4, 2), null))) === null);

    // Block 1 is full_attention: position 2, alone.
    const full = hy(2, 3);
    const mid = (s, past, n) => ({ k: 'job', s, pos: 2, past, n, opts: {} });
    await full.run(mid('f', 0, 4), new Float32Array(4 * DH.hidden));
    ok('a full_attention block still takes a rewind, overwriting its tail',
        (await refusal(full.run(mid('f', 2, 1), new Float32Array(DH.hidden)))) === null);
    const gap = await refusal(full.run(mid('f', 9, 1), new Float32Array(DH.hidden)));
    ok('but never a job past the rows it holds, which were never written',
        /no longer cached here/.test(gap ?? ''), `got ${gap}`);

    // Recurrent state can only be restarted from past 0.
    const linear = LAYER_TYPES.filter((k) => k === 'linear_attention').length;
    let cleared = gpu.passes.cleared;
    ok('a job at past 0 on a live frame is taken',
        (await refusal(r.run(at0('a', 0, 3), null))) === null);
    eq('and clears the state of every linear block it crosses first',
        gpu.passes.cleared - cleared, linear * 4);
    ok('after which it continues from the new length',
        (await refusal(r.run(at0('a', 3, 1), null))) === null);

    cleared = gpu.passes.cleared;
    await r.run(at0('fresh', 0, 2), null);
    eq('a fresh frame has nothing to clear', gpu.passes.cleared - cleared, 0);

    // The row counts belong to the session and are reset with the frame.
    await r.release('a');
    ok('a released session re-admitted from 0 starts counting again',
        (await refusal(r.run(at0('a', 0, 2), null))) === null
        && (await refusal(r.run(at0('a', 2, 1), null))) === null);

    const one = hy(0, N, { slots: 1 });
    await one.run(at0('x', 0, 6), null);
    await one.run(at0('y', 0, 2), null);            // evicts x, takes its frame
    ok('an evicting session inherits none of the counts in the frame it takes',
        (await refusal(one.run(at0('y', 2, 1), null))) === null);
}

console.log('\na job the head fed itself must continue exactly, on every kind of block');
{
    // Full attention accepts a rewind from the enqueuer. For a job fed by the
    // head, a `past` below the rows held is a duplicate, and one above them is
    // a gap.
    const r = new Runner(device, D, weightsFor(0, N), { lo: 0, hi: N, nMax: 64, tMax: 8, slots: 2, idleMs: 0 });
    const job = (past, n, auto) => ({ k: 'job', s: 'a', pos: 0, past, n, ids: Array(n).fill(1), opts: {}, auto });
    const refusal = (p) => p.then(() => null, (e) => e);
    await r.run(job(0, 4, false), null);
    await r.run(job(4, 1, true), null);
    ok('the exact continuation runs', r.live.has('a'));
    const dup = await refusal(r.run(job(3, 1, true), null));
    ok('a duplicate from behind is refused as stale', dup?.stale === true, `got ${dup?.message}`);
    const hole = await refusal(r.run(job(9, 1, true), null));
    ok('a job from beyond what is held is refused as cold, not stale',
        /no longer cached/.test(hole?.message ?? '') && !hole?.stale, `got ${hole?.message}`);
    ok('while the enqueuer may still rewind, from anywhere below',
        (await refusal(r.run(job(2, 1, false), null))) === null);
    ok('and a fresh start from 0 by the head is no rewind either',
        (await refusal(r.run(job(0, 1, true), null)))?.stale === true);
}

// A VRAM limit is checked against vramBytes, so it must equal what was
// allocated on the GPU. The stub counts bytes created and not destroyed. The
// weights are markers, so they are counted by the `bytes` their upload
// reported, as in loader.js.
console.log('\nvramBytes: what a Runner counts is what it allocated');
{
    const { hybridPieceFor } = await import('./stub-gpu.mjs');
    const piece = (p) => Object.assign(hybridPieceFor(p), { bytes: 1000 + p });
    const gpu = makeGpu();
    const base = gpu.allocated();
    const r = new Runner(gpu.device, DH, { layers: [] },
        { lo: 3, hi: 5, have: 3, nMax: 16, tMax: 4, slots: 2, sessions: 3, pageTokens: 4, idleMs: 0 });
    let weights = 0;
    const agrees = (when) => eq(`${when}: counted is allocated`,
        r.vramBytes - weights, gpu.allocated() - base);

    agrees('a fresh claim, nothing downloaded');
    ok('and that is not nothing: the pool, the frames and the scratch',
        r.vramBytes > r.pool.bytes && r.pool.bytes > 0 && r.scratch.bytes > 0);
    for (const p of [3, 4]) { r.append(piece(p)); weights += 1000 + p; }
    agrees('filled');
    eq('the weights are counted as the loader measured them', r.weightBytes, weights);

    await r.warmup();
    const uniforms = gpu.allocated() - base - (r.vramBytes - weights);
    ok('a warm-up cache is counted too', r.warm?.bytes > 0);

    // Grow one position at a time on both sides, over linear and full blocks,
    // out to both ends. `uniforms` is what the kernels cached during the
    // warm-up pass. It is not the Runner's and does not change after this.
    while (r.lo > 0 || r.hi < N) {
        if (r.lo > 0) {
            const p = r.lo - 1;
            r.prepend(piece(p), r.growLeft());
            weights += 1000 + p;
            eq(`grown left onto position ${p}: counted is allocated`,
                r.vramBytes - weights, gpu.allocated() - base - uniforms);
        }
        if (r.hi < N) {
            const p = r.hi;
            r.growRight();
            r.append(piece(p));
            weights += 1000 + p;
            eq(`grown right onto position ${p}: counted is allocated`,
                r.vramBytes - weights, gpu.allocated() - base - uniforms);
        }
    }
    ok('growth destroyed the warm-up cache it replaced', r.warm === null);
    eq('the whole model: every piece counted once', r.weightBytes, weights);
}

// What route.js reports in its acks: the positions this peer runs per token
// for its active chats, and how long a position takes.
console.log('\nload: the work a peer reports for routing');
{
    const r = new Runner(device, D, weightsFor(0, 4),
        { lo: 0, hi: 4, nMax: 16, tMax: 4, slots: 2, idleMs: 0 });
    const job = (s, past, want, n = 1) => r.run(
        { k: 'job', s, pos: 0, past, n, ids: Array(n).fill(1), opts: {} }, null, want);
    eq('an idle peer has no load', r.load, 0);
    eq('and no measured speed yet', r.msPerPos, null);
    await job('a', 0, 3, 2);
    ok('a prefill does not set the speed', r.msPerPos === null);
    await job('a', 2, 3);
    eq('a chat cut at 3 adds three positions', r.load, 3);
    ok('and a decode token measures the speed', r.msPerPos > 0, `${r.msPerPos}`);
    await job('b', 0, 2);
    eq('a second chat adds its own', r.load, 5);
    await r.release('a');
    eq('a chat that ended no longer counts', r.load, 2);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
