// The sampler, and the history summary it needs. The mesh samples on the head
// peer (web/model/stage.js), which has a logits row and no token ids. The
// summary is built at the enqueuer (web/route.js), which has the ids and no
// GPU. Pure functions, no WebGPU.

// The history-dependent inputs to the sampler, computed where the token ids
// are and small enough to send to where the logits are. Sending the summary to
// the head is much cheaper than sending the logits row back.
//
//   `pen`  [[id, count], …]: the `top` most repeated ids of the last `window`
//          tokens. Lossy, which is acceptable for a heuristic penalty. The
//          dropped ids have low counts.
//   `ban`  the ids that would complete an `ngram`-gram already in `ctx`.
//          Exact. This is what stops a repeating loop.
//
// `skip` holds the end-of-turn id, which is in the history after the first
// turn. Penalizing or banning it would make a long reply less likely to stop.
//
// Returns `{}` when there is nothing to report, so the job header carries no
// extra bytes.
export function penaltyState(ctx, { window = 256, top = 32, ngram = 0, skip = [] } = {}) {
    const out = {};
    const skipped = new Set(skip);

    const tail = ctx.length > window ? ctx.slice(ctx.length - window) : ctx;
    const counts = new Map();
    for (const id of tail) {
        if (skipped.has(id)) continue;
        counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    const repeated = top > 0 ? [...counts].filter(([, n]) => n > 1) : [];
    if (repeated.length) {
        repeated.sort((a, b) => b[1] - a[1]);
        out.pen = repeated.slice(0, top);
    }

    // The suffix is the last n-1 ids. Every earlier occurrence of it is
    // followed by a token that is banned. A linear scan over a context of at
    // most N_MAX tokens, once per token.
    if (ngram > 1 && ctx.length >= ngram) {
        const k = ngram - 1;
        const suffix = ctx.slice(ctx.length - k);
        const ban = new Set();
        for (let i = 0; i + k < ctx.length; i++) {
            let same = true;
            for (let j = 0; j < k; j++) if (ctx[i + j] !== suffix[j]) { same = false; break; }
            if (same && !skipped.has(ctx[i + k])) ban.add(ctx[i + k]);
        }
        if (ban.size) out.ban = [...ban];
    }

    return out;
}

// The most a frequency penalty may subtract from one logit. Without a cap, a
// looping reply's counts would push every candidate far down, and eos, which
// is exempt from penalties, would win and end the reply mid-sentence.
const PEN_CAP = 1.5;

// Limits on what a job may ask of the sampler. `opts` arrives in a job header
// from any peer, and the head peer runs it on its own CPU. Top-k is
// O(vocab · k), so an unbounded k would freeze the tab. The ranges match the
// inputs in chat.html and peer.html. The `pen` and `ban` limits are well above
// what route.js's jobOpts() builds (PEN_TOP ids, and a ban at most as long as
// the context).
export const LIMITS = {
    temp: [0, 2], topK: [0, 200], rep: [1, 2], freq: [0, 2], pres: [0, 2],
    pen: 64, ban: 4096,
};

// Returns `opts` clamped to LIMITS, for a row of `vocab` logits. A value that
// is not a finite number is replaced by the sampler's default; the job does
// not fail.
export function clampSampling(opts = {}, vocab = Infinity) {
    const num = (v, [lo, hi], dflt) =>
        typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : dflt;
    const id = (v) => Number.isInteger(v) && v >= 0 && v < vocab;

    const out = {
        temp: num(opts.temp, LIMITS.temp, 0),
        topK: Math.trunc(num(opts.topK, LIMITS.topK, 0)),
        rep: num(opts.rep, LIMITS.rep, 1.0),
        freq: num(opts.freq, LIMITS.freq, 0),
        pres: num(opts.pres, LIMITS.pres, 0),
        pen: null,
        ban: null,
    };
    if (Array.isArray(opts.pen)) {
        out.pen = opts.pen.slice(0, LIMITS.pen).filter((p) =>
            Array.isArray(p) && id(p[0]) && Number.isFinite(p[1]) && p[1] >= 0);
    }
    if (Array.isArray(opts.ban)) out.ban = opts.ban.slice(0, LIMITS.ban).filter(id);
    return out;
}

// CPU-side sampling from one logits row [vocab]. The history-dependent inputs
// arrive in `opts`, already computed by penaltyState(), and are clamped to
// LIMITS here.
//
//   ban           ids that may not be sampled (the no-repeat n-gram block)
//   rep           1.0 is off; > 1 divides the logits of tokens already seen
//   pres / freq   additive penalties over `pen`: `pres` is flat once a token
//                 is seen, `freq` scales with its count, up to PEN_CAP
//   temp <= 0     → greedy argmax
//   topK > 0      → keep the top k logits before the softmax
//
// With every penalty off this is exact argmax / top-k.
export function sampleFromLogitsRow(row, opts = {}) {
    const { temp, topK, rep, freq, pres, pen, ban } = clampSampling(opts, row.length);
    const adjusted = Float32Array.from(row);

    if (ban) for (const id of ban) adjusted[id] = -Infinity;

    if (pen && (rep !== 1.0 || freq !== 0 || pres !== 0)) {
        for (const [id, count] of pen) {
            let v = adjusted[id];
            if (rep !== 1.0) v = v > 0 ? v / rep : v * rep;
            v -= pres + Math.min(freq * count, PEN_CAP);
            adjusted[id] = v;
        }
    }

    if (temp <= 0) {
        let bestI = 0, bestV = -Infinity;
        for (let i = 0; i < adjusted.length; i++) {
            if (adjusted[i] > bestV) { bestV = adjusted[i]; bestI = i; }
        }
        return bestI;
    }

    const scaled = adjusted.map((v) => v / temp);
    if (topK > 0 && topK < scaled.length) {
        // The cutoff is the k-th largest logit, found with an insertion-sorted
        // window of the k best: one pass, O(V·k). Most tokens are rejected
        // with one comparison.
        const best = new Float32Array(topK).fill(-Infinity);
        for (let i = 0; i < scaled.length; i++) {
            const v = scaled[i];
            if (v <= best[topK - 1]) continue;
            let j = topK - 1;
            while (j > 0 && best[j - 1] < v) { best[j] = best[j - 1]; j--; }
            best[j] = v;
        }
        const cutoff = best[topK - 1];
        for (let i = 0; i < scaled.length; i++) if (scaled[i] < cutoff) scaled[i] = -Infinity;
    }

    let maxV = -Infinity;
    for (const v of scaled) if (v > maxV) maxV = v;
    let sum = 0;
    for (let i = 0; i < scaled.length; i++) {
        const e = scaled[i] === -Infinity ? 0 : Math.exp(scaled[i] - maxV);
        scaled[i] = e;
        sum += e;
    }
    const r = Math.random() * sum;
    let cum = 0;
    for (let i = 0; i < scaled.length; i++) {
        cum += scaled[i];
        if (r < cum) return i;
    }
    return scaled.length - 1;
}
