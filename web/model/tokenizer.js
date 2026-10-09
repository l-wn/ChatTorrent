// Pure-JS byte-level BPE for every checkpoint tools/convert_hf.py produces.
//
// Reads <weights>/tokenizer/tokenizer.json from the same host as the model
// weights. Checkpoints differ in these ways, and ignoring any of them gives
// wrong ids with no error:
//
//   1. Vocab size and merge table are per checkpoint (ids are not GPT-2 ids).
//   2. merges are ["a","b"] pairs in older exports and "a b" strings in newer.
//   3. Added tokens (<|endoftext|>, <|im_start|>, …) are outside model.vocab
//      and must be split out of the text before BPE. Otherwise
//      "<|endoftext|>" is encoded as ordinary tokens, not its special id.
//   4. SmolLM2 exports above 135M pre-tokenize with
//      Sequence[Digits{individual_digits}, ByteLevel], and the 135M snapshot
//      with a bare ByteLevel. readPreTokenizer() accepts both.
//   5. Llama 3 and Qwen pre-tokenize with Sequence[Split{Isolated, a
//      tiktoken-style regex}, ByteLevel{use_regex: false}]. The split pattern
//      is in tokenizer.json, and readPreTokenizer() compiles it
//      (translateHfRegex()).
//
// An unsupported pre_tokenizer shape throws, so the model fails to boot
// instead of producing wrong ids.
//
// API: encode(text) -> Promise<number[]>,  decode(ids) -> Promise<string>.
// Both load the tokenizer on first call. Every tab tokenizes locally.

import { activeWeights, fetchRetry } from './loader.js';

// Null until source() sets it (the mesh does, from the signaling server's
// weightsUrl). When null, the checkpoint /checkpoint.json names is used,
// resolved on first use.
let tokenizerPath = null;

let cache = null;

// Point the tokenizer at a different checkpoint. Must be called before the
// first encode/decode: load() memoizes, and changing the vocab mid-session
// would change the meaning of ids in use.
export function source(baseUrl) {
    if (cache) throw new Error('tokenizer already loaded; set the source before first use');
    tokenizerPath = `${baseUrl}/tokenizer/tokenizer.json`;
}

// tokenizerPath if source() set one, otherwise the checkpoint
// /checkpoint.json names. Shared by load() and selfTest().
async function resolveTokenizerPath() {
    return tokenizerPath ?? `${await activeWeights()}/tokenizer/tokenizer.json`;
}

export async function load(path) {
    // Checked before resolving the path, so a loaded tokenizer does not fetch
    // the config.
    if (cache) return cache;
    path ??= await resolveTokenizerPath();

    // Fetched with retries: it is several MB and every tab needs it.
    const res = await fetchRetry(path);
    if (!res.ok) {
        throw new Error(`tokenizer.json not found at ${path} (status ${res.status}). ` +
            `tools/convert_hf.py writes it next to the weights.`);
    }
    const tk = await res.json();
    if (!tk?.model?.vocab || !tk?.model?.merges) {
        throw new Error('tokenizer.json missing model.vocab or model.merges');
    }
    const { splitDigits, splitRe } = readPreTokenizer(tk.pre_tokenizer);

    const vocab = tk.model.vocab;
    const idToToken = [];
    for (const [tokStr, id] of Object.entries(vocab)) idToToken[id] = tokStr;

    // Merge rank is the index in the list. Lower ranks merge first.
    const mergeRanks = new Map();
    tk.model.merges.forEach((m, i) => {
        mergeRanks.set(Array.isArray(m) ? m.join(' ') : m, i);
    });

    // Added/special tokens are exact-match literals, matched longest first, so
    // a token that is a prefix of another does not match in its place.
    const added = (tk.added_tokens ?? []).slice().sort((a, b) => b.content.length - a.content.length);
    for (const a of added) idToToken[a.id] = a.content;
    const addedById = new Map(added.map((a) => [a.id, a]));
    const addedByContent = new Map(added.map((a) => [a.content, a.id]));
    const specialRe = added.length
        ? new RegExp(added.map((a) => escapeRe(a.content)).join('|'), 'g')
        : null;

    const byteEncoder = makeByteEncoder();
    const byteDecoder = new Map();
    for (let b = 0; b < 256; b++) byteDecoder.set(byteEncoder[b], b);

    cache = { vocab, idToToken, mergeRanks, addedById, addedByContent, specialRe,
              byteEncoder, byteDecoder, splitDigits, splitRe, bpeCache: new Map() };
    return cache;
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// tokenizers' Rust regex crate supports an inline case-insensitive group,
// "(?i:...)", which ECMAScript regex syntax lacks. The patterns in use (GPT-4's
// o200k/cl100k split, which Llama 3 reuses) have it in one place, the literal
// ASCII contractions ('s, 't, …). Nothing else in them is case-sensitive:
// \p{L}, \p{N} and \s match either case. So removing the inline group and
// compiling the whole pattern case-insensitive gives the same matches.
export function translateHfRegex(pattern) {
    return new RegExp(pattern.replace(/\(\?i:/g, '(?:'), 'giu');
}

// Validate the pre_tokenizer and return { splitDigits, splitRe }: whether a
// digit split comes before the ByteLevel step, and the regex that chunks text
// before BPE. Accepts a bare ByteLevel, Sequence[Digits?, ByteLevel] (every
// SmolLM2 export), or Sequence[Split{Isolated regex},
// ByteLevel{use_regex:false}] (Llama 3's tokenizer, and others using
// tiktoken's split-then-BPE scheme). Anything else throws, because an
// unsupported pre-tokenizer would produce wrong ids with no error.
export function readPreTokenizer(pre) {
    const steps = pre?.type === 'Sequence' ? (pre.pretokenizers ?? []) : [pre];
    const kinds = steps.map((s) => s?.type);
    const bad = (why) => {
        throw new Error(`unsupported pre_tokenizer ${JSON.stringify(pre)}: ${why}`);
    };

    if (kinds.filter((k) => k === 'ByteLevel').length !== 1)
        bad('this encoder needs exactly one ByteLevel step');
    if (kinds.some((k) => k !== 'ByteLevel' && k !== 'Digits' && k !== 'Split'))
        bad(`only Digits, Split and ByteLevel are modelled, got ${kinds.join('+')}`);

    const byteLevel = steps.find((s) => s?.type === 'ByteLevel');
    const split = steps.find((s) => s?.type === 'Split');
    const digits = steps.find((s) => s?.type === 'Digits');
    if (split && digits) bad('Split and Digits together is not a shape any export uses');

    if (split) {
        if (kinds.indexOf('Split') > kinds.indexOf('ByteLevel'))
            bad('Split must run before ByteLevel');
        if (split.behavior !== 'Isolated')
            bad(`Split{behavior:${JSON.stringify(split.behavior)}} is not supported. Only ` +
                `Isolated (each match becomes its own pre-token) is`);
        if (split.invert) bad('Split{invert:true} is not modelled');
        const regex = split.pattern?.Regex;
        if (typeof regex !== 'string') bad('Split needs a string pattern.Regex');
        // The Split step has done the splitting. A ByteLevel after it with
        // use_regex:true would apply the GPT-2 pattern again to text already
        // cut on different boundaries.
        if (byteLevel.use_regex !== false)
            bad('ByteLevel after a Split step must have use_regex:false. The Split step ' +
                'already split the text');
        if (byteLevel.add_prefix_space) bad('add_prefix_space is not implemented');
        return { splitDigits: false, splitRe: translateHfRegex(regex) };
    }

    if (kinds.indexOf('Digits') > kinds.indexOf('ByteLevel'))
        bad('Digits must run before ByteLevel');
    // individual_digits:false groups runs of digits, which is a different
    // split and is not used by any SmolLM2 export.
    if (digits && digits.individual_digits === false)
        bad('Digits{individual_digits:false} is a different split');
    if (byteLevel.use_regex === false) bad('ByteLevel{use_regex:false} is not the GPT-2 split');
    if (byteLevel.add_prefix_space) bad('add_prefix_space is not implemented');

    return { splitDigits: Boolean(digits), splitRe: PRETOK_RE };
}

// tokenizers' Digits{individual_digits:true}: every digit is its own pre-token
// and the runs between them stay whole. Uses \p{N}, not [0-9], to match Rust's
// char::is_numeric(), which is Nd|Nl|No. JS's \p{N} is the same class, and the
// ByteLevel regex below uses it too.
export function digitSplit(text) {
    const out = [];
    let run = '';
    for (const ch of text) {
        if (/\p{N}/u.test(ch)) {
            if (run) { out.push(run); run = ''; }
            out.push(ch);
        } else {
            run += ch;
        }
    }
    if (run) out.push(run);
    return out;
}

// OpenAI's bytes_to_unicode: maps every byte 0..255 to a printable codepoint,
// so BPE operates on a string with one codepoint per byte.
function makeByteEncoder() {
    const bs = [];
    for (let b = 0x21; b <= 0x7e; b++) bs.push(b); // '!'..'~'
    for (let b = 0xa1; b <= 0xac; b++) bs.push(b); // '¡'..'¬'
    for (let b = 0xae; b <= 0xff; b++) bs.push(b); // '®'..'ÿ'
    const cs = bs.slice();
    const present = new Set(bs);
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!present.has(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const enc = new Array(256);
    for (let i = 0; i < bs.length; i++) enc[bs[i]] = String.fromCodePoint(cs[i]);
    return enc;
}

// The GPT-2 pretokenizer regex, which tokenizers' ByteLevel{use_regex:true}
// applies. The default splitRe for a checkpoint with no explicit Split step.
export const PRETOK_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bpe(word, c) {
    const hit = c.bpeCache.get(word);
    if (hit) return hit;

    let chars = Array.from(word);
    if (chars.length < 2) return chars;

    while (true) {
        let bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < chars.length - 1; i++) {
            const r = c.mergeRanks.get(chars[i] + ' ' + chars[i + 1]);
            if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
        }
        if (bestIdx === -1) break;

        const a = chars[bestIdx], b = chars[bestIdx + 1];
        const next = [];
        let i = 0;
        while (i < chars.length) {
            if (i < chars.length - 1 && chars[i] === a && chars[i + 1] === b) { next.push(a + b); i += 2; }
            else { next.push(chars[i]); i++; }
        }
        chars = next;
        if (chars.length === 1) break;
    }
    c.bpeCache.set(word, chars);
    return chars;
}

function encodeChunk(text, c, out) {
    const utf8 = new TextEncoder();
    // The pre-tokenizers apply in order: the digit split (when the export has
    // one) cuts the text first, and the GPT-2 regex then runs inside each
    // piece.
    const segments = c.splitDigits ? digitSplit(text) : [text];
    for (const chunk of segments.flatMap((seg) => seg.match(c.splitRe) ?? [])) {
        let mapped = '';
        for (const b of utf8.encode(chunk)) mapped += c.byteEncoder[b];
        for (const t of bpe(mapped, c)) {
            const id = c.vocab[t];
            if (id === undefined) throw new Error(`unknown BPE token ${JSON.stringify(t)}`);
            out.push(id);
        }
    }
}

function encodeSync(text, c) {
    const out = [];
    if (!c.specialRe) { encodeChunk(text, c, out); return out; }

    // Walk the added-token matches and BPE the gaps between them.
    c.specialRe.lastIndex = 0;
    let last = 0;
    for (let m = c.specialRe.exec(text); m; m = c.specialRe.exec(text)) {
        if (m.index > last) encodeChunk(text.slice(last, m.index), c, out);
        out.push(c.addedByContent.get(m[0]));
        last = m.index + m[0].length;
    }
    if (last < text.length) encodeChunk(text.slice(last), c, out);
    return out;
}

function decodeSync(ids, c, { skipSpecial = false } = {}) {
    let out = '';
    let mapped = '';                      // pending byte-level run
    const flush = () => {
        if (!mapped) return;
        const bytes = new Uint8Array(Array.from(mapped, (ch) => c.byteDecoder.get(ch) ?? 0));
        out += new TextDecoder('utf-8', { fatal: false }).decode(bytes);
        mapped = '';
    };

    for (const id of ids) {
        const special = c.addedById.get(id);
        if (special) {
            // Added tokens are literal text, not byte-encoded. Emit or drop
            // them without byteDecoder.
            flush();
            if (!skipSpecial) out += special.content;
            continue;
        }
        const t = c.idToToken[id];
        if (t === undefined) throw new Error(`unknown id ${id}`);
        mapped += t;
    }
    flush();
    return out;
}

export async function encode(text) {
    return encodeSync(text, await load());
}

export async function decode(ids, opts) {
    return decodeSync(ids, await load(), opts);
}

// Round-trips the fixtures in oracle/<slug>/tokenizer_cases.json (ids from the
// HF tokenizer). The slug is the last path segment of the loaded checkpoint's
// URL, unless `casesUrl` is given. Returns { rows, ok }.
export async function selfTest(casesUrl) {
    const c = await load();
    if (!casesUrl) {
        const path = await resolveTokenizerPath();
        const slug = path.replace(/\/tokenizer\/tokenizer\.json$/, '').replace(/\/$/, '').split('/').pop();
        casesUrl = `/oracle/${slug}/tokenizer_cases.json`;
    }
    const res = await fetch(casesUrl);
    if (!res.ok) {
        throw new Error(`${casesUrl}: ${res.status}. Either this checkpoint has no oracle dump ` +
            `yet (run whichever tools/dump_*_oracle.py produced it) or pass an explicit fixture ` +
            `URL (dev.html's ?oracle=).`);
    }

    const rows = [];
    for (const { text, ids } of await res.json()) {
        const got = encodeSync(text, c);
        const same = got.length === ids.length && got.every((v, i) => v === ids[i]);
        const round = decodeSync(got, c) === text;
        rows.push({
            name: `encode ${JSON.stringify(text)}`,
            ok: same && round,
            detail: same
                ? (round ? `${got.length} ids, decode round-trips` : 'ids match but decode differs')
                : `got [${got}]  want [${ids}]`,
        });
    }
    return { rows, ok: rows.every((r) => r.ok) };
}
