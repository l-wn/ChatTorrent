// Speculative decoding's drafter, run by the head of the closed loop
// (route.js feed()). Pure functions, no WebGPU.
//
// The head drafts up to K tokens after the one it just sampled and sends them
// in the same job (`drafts`). Every peer runs the K + 1 rows in one pass, the
// head samples every row and keeps the drafts while they match its samples.
// A hop costs about the same for K + 1 rows as for one, so each accepted draft
// is a token that cost no lap.

// Prompt lookup: find the most recent earlier place where the context's last
// n tokens (n from `maxN` down to `minN`) occurred, and propose the tokens
// that followed it there, up to `maxK`. Code, quotes and lists repeat, and
// this catches them without a draft model. Searches the last `window` tokens.
export function lookup(ctx, maxK, { maxN = 4, minN = 2, window = 16384 } = {}) {
    const L = ctx.length;
    if (maxK < 1) return [];
    const lo = Math.max(0, L - window);
    for (let n = Math.min(maxN, L - 1); n >= minN; n--) {
        const s0 = L - n;
        for (let i = s0 - 1; i >= lo; i--) {
            let j = 0;
            while (j < n && ctx[i + j] === ctx[s0 + j]) j++;
            if (j === n) return ctx.slice(i + n, Math.min(i + n + maxK, L));
        }
    }
    return [];
}

// The fewest drafts worth a verify. With one, the extra rows rarely pay.
export const MIN_DRAFTS = 2;

// How many drafts to ask for, per session. Measured, not modelled: each step's
// tokens per second goes into an average for the depth it ran at, and the
// fastest depth wins. Depth 0 is a plain step, so a busy mesh, where extra
// rows cost other chats' time, falls back to plain decoding by itself. A step
// with no drafts (lookup found nothing) is a plain step and is measured as 0.
//
// The first steps run at the smallest depth, then each depth is tried once,
// and every `probe` steps the runner-up is tried again so an average cannot
// go stale.
export function makeDepth({ cand = [0, 3, 5, 7], warm = 3, probe = 16 } = {}) {
    return { cand, warm, probe, ema: new Map(), step: 0 };
}

export function pickDepth(st) {
    st.step++;
    const first = st.cand.find((k) => k > 0) ?? 0;
    if (st.step <= st.warm) return first;
    const untried = st.cand.find((k) => !st.ema.has(k));
    if (untried !== undefined) return untried;
    const ranked = [...st.cand].sort((a, b) => st.ema.get(b) - st.ema.get(a));
    if (st.step % st.probe === 0 && ranked.length > 1) return ranked[1];
    return ranked[0];
}

// A step at depth `k` produced `tokens` tokens in `ms`.
export function noteDepth(st, k, tokens, ms) {
    if (!(ms > 0) || !st.cand.includes(k)) return;
    const rate = (tokens * 1000) / ms;
    const prev = st.ema.get(k);
    st.ema.set(k, prev == null ? rate : 0.6 * prev + 0.4 * rate);
}
