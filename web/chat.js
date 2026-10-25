// Single-tab chat: the whole model on this tab's GPU, no mesh.
//
// The KV cache is the session. A turn feeds only the new tokens, at T_past =
// the number of tokens fed so far. "New chat" resets T_past to 0. Rows past
// T_past are never read, so nothing needs clearing.
//
// The turn wrapper comes from the manifest's chat_format, via web/prompt.js.
// Raw mode skips the wrapper and continues the typed text.

import { checkDims, makeCache, makeScratch } from './model/model.js';
import { DEFAULT_CTX } from './split.js';
import { loadManifest, loadWeights, dimsFromManifest, activeWeights } from './model/loader.js';
import { forwardIds } from './model/generate.js';
import { penaltyState, sampleFromLogitsRow } from './model/sampling.js';
import { encode, decode } from './model/tokenizer.js';
import { chatFormat, wrapTurn, stopRe } from './prompt.js';
import { splitReasoning } from './turn.js';
import { makeMessage } from './message.js';

// --- Session shape ---

// Context window for a session: the mesh's default, so this page and the
// mesh give the same answer on a long chat.
const N_MAX = DEFAULT_CTX;

// Longest run of tokens fed in one forward pass. A prompt is chunked to fit.
// Bounds the scratch buffers.
const T_MAX = 64;

// End-of-text id, from the manifest's eos_token_id at boot.
let EOS = 0;

// The turn wrapper and the regex that cuts a runaway reply, from the manifest
// at boot (see web/prompt.js). The plain transcript until then.
let CHAT = 'transcript';
let TURN_RE = stopRe(CHAT);

const state = {
    device: null,
    w: null,
    d: null,           // filled from the manifest at boot (dimsFromManifest)
    cache: null,
    scratch: null,
    ctxIds: [],        // every token the cache holds, in order
    busy: false,
    stopRequested: false,
};

// --- DOM ---

const $ = (id) => document.getElementById(id);
const ui = {};

// Index every element with an id, as peer.js does.
function cacheDom() {
    for (const el of document.querySelectorAll('[id]')) ui[el.id] = el;
}

function setStatus(text, kind = '') {
    ui.status.textContent = text;
    ui.status.className = `pill ${kind}`;
}

function atBottom() {
    const el = ui.transcript;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}

// Auto-scroll only when the user is already at the bottom, so they can scroll
// up to read during a generation.
function scrollDown(force = false) {
    if (force || atBottom()) ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function addMessage(who, text = '', cls = '') {
    const msg = makeMessage(who, text, cls);
    ui.messages.appendChild(msg.el);
    scrollDown(true);
    return msg;
}

const note = (text, err = false) => addMessage('', text, `note${err ? ' err' : ''}`);

function updateContextPill() {
    const used = state.ctxIds.length;
    const pct = Math.round((used / N_MAX) * 100);
    ui.ctx.textContent = `context ${used}/${N_MAX}`;
    ui.ctx.className = `pill${pct > 90 ? ' err' : ''}`;
}

function setBusy(busy) {
    state.busy = busy;
    ui.input.disabled = busy;
    ui.send.disabled = busy;
    ui.clear.disabled = busy;
    ui.send.hidden = busy;
    ui.stop.hidden = !busy;
    if (!busy) ui.input.focus();
}

// --- Boot ---

export async function boot() {
    cacheDom();

    if (!navigator.gpu) throw new Error('This browser has no WebGPU. Use Chrome or Edge.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU is present but no adapter was returned. It is likely disabled.');

    // The embedding table is larger than the default 128 MiB binding limit.
    // 2 GiB fits Qwen3.8-27B's ~1.27 GiB q8 wte and lm_head. Clamped to the
    // adapter's limit.
    const want = 2 * (1 << 30);
    // shader-f16 is needed only for f16 weights. Without it an f16 checkpoint
    // fails in loader.js with an error message.
    const device = await adapter.requestDevice({
        requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
        requiredLimits: {
            maxBufferSize: Math.min(want, adapter.limits.maxBufferSize),
            maxStorageBufferBindingSize: Math.min(want, adapter.limits.maxStorageBufferBindingSize),
        },
    });
    device.lost.then((info) => {
        if (info.reason === 'destroyed') return;
        setStatus('device lost', 'err');
        note(`WebGPU device was lost: ${info.message}. Reload the page.`, true);
        setBusy(false);
        ui.input.disabled = true;
        ui.send.disabled = true;
    });
    state.device = device;

    // Weights. Dimensions, buffer sizes and the title come from the manifest
    // of the checkpoint /checkpoint.json names. The base URL is resolved once
    // and passed down, so the URL shown below is the one used.
    setStatus('loading weights…', 'busy');
    const baseUrl = await activeWeights();
    const manifest = await loadManifest(baseUrl);
    const d = dimsFromManifest(manifest);
    state.d = d;
    EOS = manifest.eos_token_id ?? 0;
    CHAT = chatFormat(manifest);
    TURN_RE = stopRe(CHAT);
    ui.thinkingLabel.hidden = CHAT !== 'qwen';

    const problems = checkDims(d);
    if (problems.length) {
        throw new Error(`${manifest.model} has dimensions these kernels can't run: ` +
            problems.join('; '));
    }

    const bytes = Object.values(manifest.files).flat().reduce((n, e) => n + e.size, 0);
    const shortName = manifest.model.split('/').pop();
    document.title = `${shortName} chat`;
    ui.title.textContent = shortName;
    ui.boot.firstElementChild.textContent =
        `loading ${manifest.model} (${(bytes / 1e6).toFixed(0)} MB)…`;

    const t0 = performance.now();
    state.w = await loadWeights(device, {
        baseUrl,
        manifest,
        onProgress: (done, total, msg) => {
            ui.bootbar.style.width = `${(done / total) * 100}%`;
            ui.bootnote.textContent = `${done}/${total}: ${msg}`;
        },
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);

    state.cache = makeCache(device, d, N_MAX);
    state.scratch = makeScratch(device, d, T_MAX, N_MAX);

    // The tokenizer loads on first encode. Load it here so the first Send does
    // not wait on a 3.5 MB fetch.
    await encode('warm');

    ui.boot.remove();
    note(`${manifest.model} loaded from ${baseUrl}/ in ${secs}s: ` +
        `${d.n_layer} layers, hidden ${d.hidden}, ${d.n_heads}/${d.n_kv_heads} heads. ` +
        `If this is a base model it continues text rather than following instructions. ` +
        `Chat mode wraps your input in the model's chat format; raw mode just continues it.`);

    setStatus('ready', 'ok');
    updateContextPill();
    ui.input.placeholder = 'Type a prompt…';
    ui.input.disabled = false;
    ui.send.disabled = false;
    ui.clear.disabled = false;
    ui.input.focus();

    wireEvents();
}

function wireEvents() {
    ui.send.addEventListener('click', () => submit());
    ui.stop.addEventListener('click', () => { state.stopRequested = true; ui.stop.disabled = true; });
    ui.clear.addEventListener('click', newChat);

    ui.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
    });
    // Grow the textarea with its content, up to the CSS max-height.
    ui.input.addEventListener('input', () => {
        ui.input.style.height = 'auto';
        ui.input.style.height = `${ui.input.scrollHeight}px`;
    });
}

function newChat() {
    if (state.busy) return;
        // Clearing ctxIds resets the session: nothing past T_past is read.
    state.ctxIds = [];
    ui.messages.replaceChildren();
    note('New chat. The KV cache is empty.');
    updateContextPill();
    setStatus('ready', 'ok');
}

// --- Generation ---

function options() {
    return {
        temp: Number(ui.temp.value),
        topK: Number(ui.topk.value),
        rep: Number(ui.rep.value),
        freq: Number(ui.freq.value),
        pres: Number(ui.pres.value),
        ngram: Number(ui.ngram.value),
        // 0 means no cap: run until the reply ends or the context is full.
        maxTokens: Number(ui.maxtok.value) || Infinity,
        chat: ui.mode.value === 'chat',
        thinking: ui.thinking.checked,
    };
}

// Feed tokens through the model at the current cache position, in T_MAX-sized
// chunks. Returns the logits row for the last token.
async function feed(ids) {
    let row = null;
    for (let i = 0; i < ids.length; i += T_MAX) {
        const chunk = ids.slice(i, i + T_MAX);
        row = await forwardIds(state.device, state.d, state.w, state.cache, state.scratch,
            chunk, state.ctxIds.length);
        state.ctxIds.push(...chunk);
        updateContextPill();
    }
    return row;
}

async function submit() {
    if (state.busy) return;
    const text = ui.input.value.trim();
    if (!text) return;

    const opts = options();
    ui.input.value = '';
    ui.input.style.height = 'auto';

    addMessage('you', text, 'user');
    setBusy(true);
    state.stopRequested = false;
    ui.stop.disabled = false;
    setStatus('thinking…', 'busy');

    const out = addMessage('model', '', 'model streaming');

    try {
        await respond(text, opts, out);
    } catch (err) {
        console.error(err);
        out.el.classList.remove('streaming');
        note(`generation failed: ${err.message}`, true);
        setStatus('error', 'err');
    } finally {
        out.el.classList.remove('streaming');
        setBusy(false);
    }
}

async function respond(text, opts, out) {
    // Chat mode continues a transcript; raw mode continues the text. Only the
    // new text is encoded. The rest is in the cache.
    const prompt = opts.chat
        ? wrapTurn(CHAT, text, { first: !state.ctxIds.length, thinking: opts.thinking })
        : text;

    const promptIds = await encode(prompt);

    // Check the room before running anything, so a prompt that cannot fit does
    // not partly fill the cache.
    const room = N_MAX - state.ctxIds.length - promptIds.length;
    if (room <= 0) {
        out.el.remove();
        note(`That prompt needs ${promptIds.length} tokens and only ${
            N_MAX - state.ctxIds.length} are left in the ${N_MAX}-token context. ` +
            `Press "New chat" to start over.`, true);
        setStatus('context full', 'err');
        return;
    }
    const budget = Math.min(opts.maxTokens, room);
    if (budget < opts.maxTokens) {
        out.meta = `only ${budget} tokens of context left`;
    }
    // The limit this reply will hit if it does not end on its own.
    const ceiling = budget < opts.maxTokens ? 'context full' : 'hit the token cap';

    const t0 = performance.now();
    let row = await feed(promptIds);
    const prefillMs = performance.now() - t0;

    const genIds = [];
    let inCache = 0;          // generated tokens fed to the cache
    let shown = '';
    let cutAt = -1;           // character index in `shown` where the turn ended
    let stopped = ceiling;
    const tDecode = performance.now();

    for (let n = 0; n < budget; n++) {
        if (state.stopRequested) { stopped = 'stopped'; break; }

        // The same repetition summary the mesh sends with a job (web/route.js),
        // so both pages sample the same way.
        const id = sampleFromLogitsRow(row, {
            ...opts,
            ...penaltyState(state.ctxIds, { ngram: opts.ngram, skip: [EOS] }),
        });
        if (id === EOS) { stopped = 'end of text'; break; }

        genIds.push(id);
        // Decode the whole run each time: a UTF-8 sequence is often split
        // across two tokens.
        shown = await decode(genIds);

        const hit = opts.chat ? TURN_RE.exec(shown) : null;
        if (hit) { cutAt = hit.index; shown = shown.slice(0, cutAt); stopped = 'turn ended'; }
        const [think, answer] = splitReasoning(shown, opts.chat && opts.thinking);
        if (think !== null) out.reasoning = think;
        out.body = (think === null ? shown : answer).replace(/^\s+/, '');
        setStatus(`generating… ${genIds.length} tok · ${
            Math.max(0, N_MAX - state.ctxIds.length)} left`, 'busy');
        scrollDown();
        if (hit) break;

        state.ctxIds.push(id);
        inCache++;
        row = await forwardIds(state.device, state.d, state.w, state.cache, state.scratch,
            [id], state.ctxIds.length - 1);
        updateContextPill();
    }

    // Text cut from the display must leave the context too, or the next turn
    // is conditioned on text the user cannot see. A rewind shortens ctxIds:
    // T_past comes from its length, nothing past T_past is read, and later
    // tokens overwrite the stale KV rows.
    //
    // The cut can fall inside a token run: "\nUser:" is three tokens and only
    // the last completes the match. So the number of tokens to keep is found
    // by decoding prefixes until one extends past the cut. On ChatML the
    // marker is one token, but text decoded before it must still be kept.
    if (cutAt >= 0) {
        let keep = 0;
        while (keep < inCache && (await decode(genIds.slice(0, keep + 1))).length <= cutAt) keep++;
        state.ctxIds.length -= inCache - keep;
        inCache = keep;
        updateContextPill();
    }

    const decodeMs = performance.now() - tDecode;
    const perTok = genIds.length ? decodeMs / genIds.length : 0;

    // Checked on the answer, not the whole reply: a reply that is all
    // reasoning would leave an empty bubble.
    const [, finalAnswer] = splitReasoning(shown, opts.chat && opts.thinking);
    if (!finalAnswer.trim()) {
        out.body = shown.trim()
            ? '(the model was still reasoning when this reply ended. Raise max tok '
              + 'or turn thinking off)'
            : '(nothing. Try a higher temperature or a longer prompt)';
    }
    out.meta = `${genIds.length} tokens · ${perTok.toFixed(0)} ms/tok · ` +
        `prefill ${prefillMs.toFixed(0)} ms for ${promptIds.length} · ${stopped}`;
    ui.speed.textContent = perTok ? `${(1000 / perTok).toFixed(1)} tok/s` : '—';
    setStatus('ready', 'ok');
    scrollDown();
}
