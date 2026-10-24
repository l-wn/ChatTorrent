// Tests web/model/sampling.js in Node: the repetition penalties, and the
// property the mesh's greedy comparisons depend on, that with the penalties
// off the sampler is exact argmax / top-k.
//
// Pure functions with no weights and no GPU, so this always runs.
//
//   node tools/sampling.test.mjs

import { penaltyState, sampleFromLogitsRow, sampleFromCandidates, candidatesFor, clampSampling, LIMITS,
         TOPK_MAX } from '../web/model/sampling.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);

const row = (o) => { const r = new Float32Array(8); r.set(o); return r; };

console.log('\npenaltyState: counts');
{
    eq('nothing repeated says nothing at all', penaltyState([1, 2, 3]), {});
    eq('repeats are counted, singletons dropped',
        penaltyState([1, 2, 1, 3, 1, 2]).pen, [[1, 3], [2, 2]]);
    // 9 is the most repeated id in the whole context but is outside the
    // window, so it is not in the summary.
    eq('the window is the tail, not the whole context',
        penaltyState([9, 9, 9, 1, 2, 1], { window: 3 }).pen, [[1, 2]]);
    eq('top caps how many ids ride along',
        penaltyState([1, 1, 2, 2, 3, 3], { top: 2 }).pen.length, 2);
    eq('top 0 asks for no counts at all',
        penaltyState([1, 1, 2, 2], { top: 0 }).pen, undefined);
    eq('skip keeps an id out of the counts entirely',
        penaltyState([1, 1, 2, 2], { skip: [1] }).pen, [[2, 2]]);
}

console.log('\npenaltyState: the no-repeat n-gram ban');
{
    eq('off by default', penaltyState([1, 2, 3, 1, 2]).ban, undefined);
    // "1 2" occurred before and was followed by 3, so 3 is banned next.
    eq('a repeated prefix bans what followed it last time',
        penaltyState([1, 2, 3, 4, 1, 2], { ngram: 3 }).ban, [3]);
    eq('a prefix seen twice with different continuations bans both',
        penaltyState([1, 2, 3, 1, 2, 4, 1, 2], { ngram: 3 }).ban, [3, 4]);
    eq('a prefix that has never recurred bans nothing',
        penaltyState([1, 2, 3, 4, 5, 6], { ngram: 3 }).ban, undefined);
    eq('the suffix does not count as its own earlier occurrence',
        penaltyState([1, 2], { ngram: 3 }).ban, undefined);
    eq('a longer n needs a longer match',
        penaltyState([1, 2, 3, 4, 1, 2], { ngram: 4 }).ban, undefined);
    // An end-of-turn id is in the history after the first turn (wrapTurn
    // feeds `<|im_end|>` back in). Banning it would stop a long reply from
    // ending.
    eq('skip keeps eos out of the ban',
        penaltyState([7, 1, 2, 9, 7, 1, 2], { ngram: 4, skip: [9] }).ban, undefined);
    eq('…and it would be banned without skip',
        penaltyState([7, 1, 2, 9, 7, 1, 2], { ngram: 4 }).ban, [9]);
}

console.log('\nsampleFromLogitsRow: off is off');
{
    const r = row([0, 5, 1, 0, 0, 0, 0, 0]);
    eq('greedy argmax with no opts', sampleFromLogitsRow(r), 1);
    eq('greedy argmax is unmoved by a summary it was not told to use',
        sampleFromLogitsRow(r, { pen: [[1, 9]] }), 1);
    eq('rep 1.0 / freq 0 / pres 0 changes nothing',
        sampleFromLogitsRow(r, { pen: [[1, 9]], rep: 1.0, freq: 0, pres: 0 }), 1);
}

console.log('\nsampleFromLogitsRow: penalties move the pick');
{
    const r = row([0, 5, 4.5, 0, 0, 0, 0, 0]);
    eq('presence alone can hand the token to the runner-up',
        sampleFromLogitsRow(r, { pen: [[1, 1]], pres: 1 }), 2);
    eq('frequency scales with the count',
        sampleFromLogitsRow(r, { pen: [[1, 2]], freq: 0.3 }), 2);
    eq('a small frequency penalty on a single occurrence is not enough',
        sampleFromLogitsRow(r, { pen: [[1, 1]], freq: 0.3 }), 1);
    eq('the multiplicative rep penalty works',
        sampleFromLogitsRow(r, { pen: [[1, 1]], rep: 1.2 }), 2);
    eq('ban is absolute, whatever the logit says',
        sampleFromLogitsRow(r, { ban: [1] }), 2);
    eq('a banned token stays banned under sampling too',
        sampleFromLogitsRow(row([0, 100, 0, 0, 0, 0, 0, 0]),
            { ban: [1], temp: 1, topK: 1 }) !== 1, true);
}

console.log('\nsampleFromLogitsRow: the frequency penalty saturates');
{
    // Without a cap, `freq * count` in a looping reply would push every real
    // candidate down and let eos (exempt from penalties) win, ending the reply
    // mid-sentence. Token 1 leads token 2 by 5 logits. A capped penalty cannot
    // close that gap, however often token 1 has repeated.
    ok('a huge count cannot bury a token 5 logits ahead',
        sampleFromLogitsRow(row([0, 5, 0, 0, 0, 0, 0, 0]),
            { pen: [[1, 1000]], freq: 0.4 }) === 1);
    // An uncapped penalty would: 0.4 * 1000 = 400.
    ok('but a 1.5-logit gap still yields, which is the point of having one at all',
        sampleFromLogitsRow(row([0, 1.4, 0, 0, 0, 0, 0, 0]),
            { pen: [[1, 1000]], freq: 0.4 }) !== 1);
    // The presence penalty is flat and bounded by its input range, so adding
    // it keeps the total bounded.
    ok('presence still stacks on top of a saturated frequency penalty',
        sampleFromLogitsRow(row([0, 1.6, 0, 0, 0, 0, 0, 0]),
            { pen: [[1, 1000]], freq: 0.4, pres: 0.2 }) !== 1);
}

// opts arrives in a job header from any peer, and the head peer runs it.
console.log('\nclampSampling: a job cannot ask for more than LIMITS');
{
    const c = clampSampling({ temp: 1e9, topK: 99999.7, rep: -3, freq: NaN, pres: '1' });
    eq('every number is held to its range, and garbage falls back to the default',
        [c.temp, c.topK, c.rep, c.freq, c.pres],
        [LIMITS.temp[1], LIMITS.topK[1], LIMITS.rep[0], 0, 0]);
    eq('topK is a whole number', clampSampling({ topK: 3.9 }).topK, 3);
    eq('ban keeps only integer ids inside the vocabulary',
        clampSampling({ ban: [1, -1, 2.5, 8, 'x', 7] }, 8).ban, [1, 7]);
    eq('ban is capped', clampSampling({ ban: new Array(1e5).fill(0) }).ban.length, LIMITS.ban);
    eq('pen drops malformed pairs', clampSampling({ pen: [[1, 2], [9, 1], [2, -1], 3, [3, 1]] }, 8).pen,
        [[1, 2], [3, 1]]);
    eq('pen is capped', clampSampling({ pen: new Array(1e4).fill([1, 1]) }).pen.length, LIMITS.pen);

    // top-k is O(vocab · k), so an unbounded k would freeze a peer.
    const V = 150_000;
    const big = new Float32Array(V);
    for (let i = 0; i < V; i++) big[i] = Math.sin(i * 12.9898) * 10;
    const t0 = performance.now();
    sampleFromLogitsRow(big, { temp: 1, topK: 99999 });
    const ms = performance.now() - t0;
    ok(`topK 99999 on a ${V}-token row is cheap (${ms.toFixed(0)} ms)`, ms < 1000);
}

console.log('\nsampleFromCandidates: the raw top-M give the full row\'s token');
{
    // The GPU hands the head only the raw top-M logits (candidatesFor), and
    // the head must pick the token the full row would. Math.random is
    // replaced by a seeded sequence, replayed for both paths.
    let seed = 12345;
    const next = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const real = Math.random;
    const V = 3001;                                 // not a multiple of 1024
    let cases = 0, same = 0, fellBack = 0, first = null, wrongShort = 0;
    for (let c = 0; c < 3000; c++) {
        const row = new Float32Array(V);
        // Peaky logits, like a real head: a few tall values over noise.
        for (let i = 0; i < V; i++) row[i] = (next() - 0.5) * 4;
        for (let j = 0; j < 30; j++) row[Math.floor(next() * V)] += next() * 12;
        const byValue = Array.from(row.keys()).sort((a, b) => row[b] - row[a] || a - b);
        const pick = (n, from) => Array.from({ length: n }, () => byValue[Math.floor(next() * from)]);
        const kind = c % 6;
        const opts = {
            temp: kind === 0 ? 0 : [0.3, 0.7, 1, 1.5][c % 4],
            topK: [1, 5, 40, 200][c % 4],
            rep: kind >= 2 ? 1 + next() : 1,
            freq: kind >= 3 ? next() * 2 : 0,
            pres: kind >= 3 ? next() * 2 : 0,
            // Penalized and banned ids from inside and around the top.
            pen: kind >= 2 ? pick(1 + (c % 30), 120).map((id) => [id, 1 + (c % 5)]) : undefined,
            ban: kind >= 4 ? pick(1 + (c % 8), 60) : undefined,
        };
        const M = candidatesFor(opts, V);
        if (M == null) { fellBack++; continue; }
        const top = byValue.slice(0, M);
        // The control: k candidates only, ignoring the penalized ones.
        const k = opts.temp <= 0 ? 1 : opts.topK;
        const short = byValue.slice(0, k);
        const s0 = next() * 4294967296 >>> 0;
        seed = s0; Math.random = next;
        const want = sampleFromLogitsRow(row, opts);
        seed = s0;
        const got = sampleFromCandidates(top, top.map((i) => row[i]), opts, V);
        seed = s0;
        const tooFew = sampleFromCandidates(short, short.map((i) => row[i]), opts, V);
        Math.random = real;
        cases++;
        if (got === want) same++;
        else first ??= { c, opts, got, want, M };
        if (tooFew !== want) wrongShort++;
    }
    ok(`control: with only k candidates, ${wrongShort} of ${cases} tokens differ`, wrongShort > 0);
    ok(`every one of ${cases} random rows and settings gives the same token`, same === cases,
        JSON.stringify(first));
    eq('none needed the whole row (each has a top-k or is greedy)', fellBack, 0);
    // Greedy with no penalties needs one candidate: the GPU's argmax.
    eq('greedy, no penalties: one candidate', candidatesFor({ temp: 0 }, V), 1);
    eq('top-k 40 with 3 penalized ids and 2 banned: 45',
        candidatesFor({ temp: 0.7, topK: 40, pres: 0.1, pen: [[1, 1], [2, 1], [3, 1]], ban: [4, 5] }, V), 45);
    eq('a banned id that is also penalized counts once',
        candidatesFor({ temp: 0.7, topK: 40, pres: 0.1, pen: [[1, 1]], ban: [1] }, V), 41);
    eq('penalties that are off add nothing',
        candidatesFor({ temp: 0.7, topK: 40, pen: [[1, 1], [2, 1]] }, V), 40);
    eq('temp > 0 with no top-k needs the whole row', candidatesFor({ temp: 0.7, topK: 0 }, V), null);
    eq('more than TOPK_MAX candidates needs the whole row',
        candidatesFor({ temp: 0.7, topK: 40, ban: Array.from({ length: TOPK_MAX }, (_, i) => i) }, 1e6), null);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
