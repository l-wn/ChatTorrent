// Tests the Engine's generate()/stop()/rewind()/reset() in Node, with one peer
// holding the whole toy model, so every job runs locally and no wire is
// needed.
//
// connect()/serve() are not tested here: they open a signaling WebSocket and a
// GPU, which tools/mesh.gpu.test.mjs tests in headless Chrome.
// Engine.forTest() (engine.js) builds an Engine around an existing
// net/router/runner, as World.join() in mesh.test.mjs does for a Mesh.
//
//   node tools/engine.test.mjs
//
// generate() needs a tokenizer to turn text into ids, so this file runs only
// when a tokenizer.json is present locally, like tools/tokenizer.test.mjs.
// weights/ is gitignored and not every environment has one.

import { readFileSync, existsSync } from 'node:fs';
import { Runner } from '../web/model/stage.js';
import { makeGpu, D, L, N, weightsFor } from './stub-gpu.mjs';
import { Mesh } from '../web/net.js';
import { makeRouter, T_MAX } from '../web/route.js';
import { N_MAX, SLOTS } from '../web/split.js';

const TOKENIZER = 'weights/llama-3.1-8b-instruct/tokenizer/tokenizer.json';
if (!existsSync(TOKENIZER)) {
    console.log(`generate()/stop()/rewind() need a real tokenizer to turn text into ids: ` +
        `skipped, no ${TOKENIZER}`);
    process.exit(0);
}

globalThis.location = { protocol: 'http:', host: 'localhost:8000' };
globalThis.WebSocket = class { addEventListener() {} send() {} close() {} readyState = 1; };
// route.js mints session keys with Web Crypto, which is a global in browsers
// and in Node 19+. Older Node versions have it in node:crypto.
globalThis.crypto ??= (await import('node:crypto')).webcrypto;
// tokenizer.js fetches its json and memoizes the result. It is the only
// network access on this path, so it is the only fetch stubbed here.
globalThis.fetch = async () => ({
    ok: true, status: 200, json: async () => JSON.parse(readFileSync(TOKENIZER, 'utf8')),
});

const { source } = await import('../web/model/tokenizer.js');
source(TOKENIZER.replace(/\/tokenizer\/tokenizer\.json$/, ''));
const { Engine, ContextFullError } = await import('../web/engine.js');

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};

// One peer holding the whole toy model (stub-gpu.mjs's `N` positions), with
// web/model/stage.js's real Runner on a stub GPU that does no arithmetic. The
// argmax of an all-zero logits row is token 0, which is enough here: the tests
// check counts, timing and error handling, not sampled values.
function soloEngine({ eos = -1 } = {}) {
    const net = new Mesh({});
    net.ready.catch(() => {});
    Object.assign(net, {
        id: 0, blocks: L, positions: N, minBlocks: 1, range: [0, N],
        peers: [{ id: 0, lo: 0, hi: N, have: N, ready: true }],
    });
    net.send = () => false;   // not called: a solo peer runs every job itself

    const { device } = makeGpu();
    const runner = new Runner(device, D, weightsFor(0, N),
        { lo: 0, hi: N, have: N, nMax: N_MAX, tMax: T_MAX, slots: SLOTS, idleMs: 0 });
    const router = makeRouter({ net, runner: () => runner });
    router.eos = eos;

    return Engine.forTest({ net, router, runner, dims: D });
}

console.log('generate(): length, and rewind()/reset() against context, on a solo peer');
{
    const engine = soloEngine();
    const chunks = [];
    for await (const chunk of engine.generate('hi', { temp: 0, topK: 0, maxTokens: 5 })) {
        chunks.push(chunk);
    }
    ok('yields exactly the requested budget', chunks.length === 5,
        `got ${chunks.length}`);
    ok('every chunk carries the sampled token id and decoded text',
        chunks.every((c) => typeof c.tokenId === 'number' && typeof c.text === 'string'));
    ok('lastStopReason is "length" once the budget is exhausted',
        engine.lastStopReason === 'length', `got ${engine.lastStopReason}`);

    const ctxAfterFirst = engine.ctxLen;
    ok('ctxLen grew by the prompt plus every generated token',
        ctxAfterFirst === engine.lastPromptTokens + 5,
        `ctxLen ${ctxAfterFirst}, prompt ${engine.lastPromptTokens}`);

    engine.rewind(2);
    ok('rewind(2) drops exactly two tokens from context',
        engine.ctxLen === ctxAfterFirst - 2, `got ${engine.ctxLen}`);

    engine.reset();
    ok('reset() clears context back to zero', engine.ctxLen === 0, `got ${engine.ctxLen}`);
}

console.log('\nstop(): round trip: ends generation after the in-flight token, not immediately');
{
    const engine = soloEngine();
    const chunks = [];
    for await (const chunk of engine.generate('hi', { temp: 0, topK: 0, maxTokens: 20, loop: false })) {
        chunks.push(chunk);
        if (chunks.length === 1) engine.stop();
    }
    ok('exactly one token was yielded after stop() was called on the first',
        chunks.length === 1, `got ${chunks.length}`);
    ok('lastStopReason is "stopped"', engine.lastStopReason === 'stopped',
        `got ${engine.lastStopReason}`);
    ok('and every token yielded is in the context', engine.lastFed === chunks.length,
        `lastFed ${engine.lastFed}, yielded ${chunks.length}`);
}

console.log('\nstop(): closed loop: the tokens already fed still arrive, and are the reply');
{
    const engine = soloEngine();
    const chunks = [];
    for await (const chunk of engine.generate('hi', { temp: 0, topK: 0, maxTokens: 20 })) {
        chunks.push(chunk);
        if (chunks.length === 1) engine.stop();
    }
    ok('it stops well short of the budget', chunks.length >= 1 && chunks.length < 20,
        `got ${chunks.length}`);
    ok('lastStopReason is "stopped"', engine.lastStopReason === 'stopped',
        `got ${engine.lastStopReason}`);
    ok('every token that reached the context was shown, and nothing else',
        engine.lastFed === chunks.length, `lastFed ${engine.lastFed}, yielded ${chunks.length}`);
    ok('the context is the prompt plus exactly those',
        engine.ctxLen === engine.lastPromptTokens + chunks.length);
}

console.log('\neos: generation ends with no chunks when the first token is already end-of-text');
{
    // The stub GPU always samples token 0, so with router.eos set to 0 the
    // first sampled token, the one the prefill returns, is end-of-text.
    const engine = soloEngine({ eos: 0 });
    const chunks = [];
    for await (const chunk of engine.generate('hi', { temp: 0, topK: 0, maxTokens: 20 })) {
        chunks.push(chunk);
    }
    ok('no chunks are yielded', chunks.length === 0, `got ${chunks.length}`);
    ok('lastStopReason is "eos"', engine.lastStopReason === 'eos',
        `got ${engine.lastStopReason}`);
}

console.log('\ngenerate() throws ContextFullError before running anything, on a fresh engine');
{
    const engine = soloEngine();
    // "a " is a single token in almost any BPE vocab. The prompt is sized
    // from N_MAX with a 20% margin for tokenizer differences.
    const bigPrompt = 'a '.repeat(Math.round(N_MAX * 1.2));
    let threw = null;
    try {
        for await (const chunk of engine.generate(bigPrompt, { temp: 0, topK: 0 })) { void chunk; }
    } catch (err) {
        threw = err;
    }
    ok('throws ContextFullError', threw instanceof ContextFullError, `got ${threw}`);
    ok('names how many tokens were needed and how much room was left',
        !!threw && threw.needed > N_MAX && threw.room === N_MAX,
        `needed ${threw?.needed}, room ${threw?.room}`);
}

// The repetition settings end to end: the enqueuer holds the ids, the head
// holds the logits, and the summary travels in the job header across every
// hop. The stub GPU returns an all-zero logits row, so an unpenalized argmax
// always picks token 0, which gives a repeating loop to test against.
console.log('\nthe repetition summary reaches the sampler, three peers away');
{
    const run = async (opts) => {
        const engine = soloEngine();
        const ids = [];
        for await (const chunk of engine.generate('hi', { temp: 0, topK: 0, maxTokens: 6, ...opts })) {
            ids.push(chunk.tokenId);
        }
        return ids;
    };

    const plain = await run({});
    ok('with everything off, an all-zero row loops on token 0 for ever',
        plain.every((id) => id === 0), `got ${plain.join(',')}`);

    // "0 0" has occurred, so a third 0 would repeat a 2-gram and is banned.
    const banned = await run({ ngram: 2 });
    ok('a no-repeat n-gram breaks the loop', banned.some((id) => id !== 0),
        `got ${banned.join(',')}`);
    ok('and only once the n-gram has actually been seen twice',
        banned[0] === 0 && banned[1] === 0, `got ${banned.join(',')}`);

    // The counts path, not the ban path: token 0 is the only id with a count,
    // so any penalty moves the argmax off it.
    const penalized = await run({ freq: 1 });
    ok('a frequency penalty moves the argmax off the repeated token',
        penalized.some((id) => id !== 0), `got ${penalized.join(',')}`);

    const presence = await run({ pres: 1 });
    ok('so does a presence penalty', presence.some((id) => id !== 0),
        `got ${presence.join(',')}`);
}

console.log('\nchats: what the page saves, and switching between them');
{
    const engine = soloEngine();
    const turn = async (text, n) => {
        for await (const chunk of engine.generate(text, { temp: 0, topK: 0, maxTokens: n })) void chunk;
    };

    await turn('hi', 4);
    engine.rewind(1);
    const saved = engine.history;
    ok('history is the ids the router holds, after the rewind',
        saved instanceof Uint32Array && saved.length === engine.ctxLen
        && saved.length > 0, `${saved.constructor.name} of ${saved.length}, ctxLen ${engine.ctxLen}`);
    const before = saved[0];
    saved[0] = before + 1;
    ok('and a copy, so saving it cannot disturb the live chat', engine.history[0] === before);

    const a = engine.conversation;
    const b = engine.newChat();
    ok('newChat() makes a fresh chat active', engine.conversation === b && engine.ctxLen === 0);
    await turn('there', 2);
    const lenB = engine.ctxLen;

    engine.switchTo(a);
    ok('switching back brings back its context', engine.ctxLen === saved.length,
        `got ${engine.ctxLen}, want ${saved.length}`);
    ok('so the next turn is wrapped as a continuation (runTurn\'s first: !ctxLen)',
        engine.ctxLen > 0);
    engine.switchTo(b);
    ok('and the other chat kept its own', engine.ctxLen === lenB);

    const stored = engine.newConversation(saved);
    ok('a stored chat arrives in this tab\'s mesh, not yet active',
        stored.mesh === engine.meshTag && engine.conversation === b);
    engine.switchTo(stored);
    await turn('again', 2);
    ok('and its first turn continues it', engine.ctxLen > saved.length);

    let threw = null;
    for await (const chunk of engine.generate('busy', { temp: 0, topK: 0, maxTokens: 3 })) {
        void chunk;
        try { engine.switchTo(a); } catch (err) { threw ??= err; }
    }
    ok('no switching while a reply is being generated',
        /still being generated/.test(threw?.message ?? ''), `got ${threw?.message}`);
    ok('and none happened', engine.conversation === stored);
    engine.switchTo(a);
    ok('once it has ended, switching works again', engine.conversation === a);
}

console.log('\nthe closed loop against the round trip, through the engine');
{
    const { runTurn } = await import('../web/turn.js');
    const ids = async (opts) => {
        const engine = soloEngine();
        const out = [];
        for await (const c of engine.generate('hello there', { temp: 0, topK: 0, ...opts })) out.push(c.tokenId);
        return { out, engine };
    };

    for (const pens of [{}, { freq: 1, pres: 0.5, ngram: 2 }]) {
        const label = Object.keys(pens).length ? 'with every penalty on' : 'with penalties off';
        const a = await ids({ maxTokens: 9, loop: false, ...pens });
        const b = await ids({ maxTokens: 9, ...pens });
        ok(`the same ids either way, ${label}`, JSON.stringify(a.out) === JSON.stringify(b.out),
            `round trip ${a.out.join(',')}, loop ${b.out.join(',')}`);
        ok(`and the same context left behind, ${label}`,
            JSON.stringify([...a.engine.history]) === JSON.stringify([...b.engine.history]));
    }

    const full = await ids({ maxTokens: 9 });
    ok('a budget is exact: nine tokens, all nine in the context',
        full.out.length === 9 && full.engine.lastFed === 9
        && full.engine.lastStopReason === 'length', `${full.out.length}, ${full.engine.lastFed}`);

    // A nearly full context: the budget is the remaining room, and the loop
    // stops there.
    const engine = soloEngine();
    for await (const c of engine.generate('a '.repeat(N_MAX - 40), { temp: 0, topK: 0 })) void c;
    ok('with room for only a few tokens, the loop fills the context exactly',
        engine.ctxLen === N_MAX && engine.lastStopReason === 'context',
        `ctxLen ${engine.ctxLen}, reason ${engine.lastStopReason}`);

    // A stop sequence cuts the reply. The tokens fed past the cut leave the
    // context, whether or not they were yielded, and the ones before it stay.
    // The stub samples token 0 every time, one character each, so "cut after
    // the third" is exact.
    for (const loop of [false, true]) {
        const e = soloEngine();
        const turnRe = { exec: (text) => (text.length >= 5 ? { index: 3 } : null) };
        const turn = await runTurn(e, 'hi', { temp: 0, topK: 0, maxTokens: 20, chat: true, loop },
            { chat: 'transcript', turnRe });
        const promptTokens = e.lastPromptTokens;
        ok(`${loop ? 'closed loop' : 'round trip'}: a cut turn keeps exactly the text before the cut`,
            e.ctxLen === promptTokens + 3 && turn.rewound === turn.inCache - 3,
            `ctxLen ${e.ctxLen}, prompt ${promptTokens}, inCache ${turn.inCache}, rewound ${turn.rewound}`);
    }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
