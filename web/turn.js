// Where a chat turn ends.
//
// The engine stops on EOS, on its budget, or on stop() (web/engine.js). A turn
// can end earlier: a base checkpoint continues into an invented next turn, and
// a chat checkpoint may write its end-of-turn marker as text. The boundary
// depends on the prompt format (web/prompt.js), so this file wraps the turn,
// watches the decoded text for the stop sequence, and removes the tokens past
// it from the context. Otherwise the next turn would be conditioned on text
// the user did not see.
//
// No DOM, so tools/mesh.gpu.test.mjs runs the same cut arithmetic as the page.

import { decode } from './model/tokenizer.js';
import { wrapTurn } from './prompt.js';

// With thinking on, the text up to the first `</think>` is reasoning and the
// rest is the answer. Returns [reasoning, answer]. reasoning is null when
// thinking is off.
export function splitReasoning(text, thinking) {
    if (!thinking) return [null, text];
    const i = text.indexOf('</think>');
    return i < 0 ? [text, ''] : [text.slice(0, i), text.slice(i + 8)];
}

// Display names for the engine's `lastStopReason` values.
const REASONS = {
    eos: 'end of text',
    stopped: 'stopped',
    length: 'hit the token cap',
    context: 'context full',
};

// Run one turn. A ContextFullError from the engine is not caught.
//
// `onToken` is called once per token with the text already split into
// reasoning and answer and already cut at the stop sequence. The turn ends
// when the engine ends or the stop sequence appears, whichever is first.
export async function runTurn(engine, text, opts, { chat, turnRe, onToken } = {}) {
    const prompt = opts.chat
        ? wrapTurn(chat, text, { first: !engine.ctxLen, thinking: opts.thinking })
        : text;

    const t0 = performance.now();
    let tPrefillDone = null;
    const ids = [];
    let shown = '';
    let cutAt = -1;
    let stopped = null;

    for await (const chunk of engine.generate(prompt, opts)) {
        if (tPrefillDone == null) tPrefillDone = performance.now();
        ids.push(chunk.tokenId);
        shown = chunk.text;
        const hit = opts.chat ? turnRe.exec(shown) : null;
        if (hit) { cutAt = hit.index; shown = shown.slice(0, cutAt); stopped = 'turn ended'; }
        const [reasoning, answer] = splitReasoning(shown, opts.chat && opts.thinking);
        onToken?.({ text: shown, reasoning, answer, tokenId: chunk.tokenId, n: ids.length });
        // Breaking the `for await` finalizes the generator. The engine needs
        // no other call.
        if (hit) break;
    }
    // How many generated tokens the context holds once the generator has
    // finished, from the engine's count. In round-trip mode, a break leaves
    // the last yielded token unfed. With the closed loop, tokens fed after the
    // break are counted too.
    const inCache = engine.lastFed;

    const prefillMs = (tPrefillDone ?? performance.now()) - t0;
    stopped ??= REASONS[engine.lastStopReason] ?? engine.lastStopReason;

    // Text cut from the display must leave the context too: every token the
    // context holds past the cut, whether or not it was yielded.
    let rewound = 0;
    if (cutAt >= 0) {
        let keep = 0;
        while (keep < Math.min(inCache, ids.length)
            && (await decode(ids.slice(0, keep + 1))).length <= cutAt) keep++;
        rewound = inCache - keep;
        engine.rewind(rewound);
    }

    const perTok = ids.length ? (performance.now() - (tPrefillDone ?? t0)) / ids.length : 0;
    const [reasoning, answer] = splitReasoning(shown, opts.chat && opts.thinking);

    return { ids, text: shown, reasoning, answer, stopped, cutAt, rewound, inCache,
             prefillMs, perTok, promptTokens: engine.lastPromptTokens };
}
