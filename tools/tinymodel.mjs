// Generates a checkpoint small enough to run eight copies on one GPU.
//
// tools/mesh.test.mjs uses a stub GPU and a toy digest, because it tests which
// blocks are dispatched over which positions and needs no numbers.
// tools/mesh.gpu.test.mjs tests whether a mesh of real peers, on a real GPU
// and a real wire, produces the same tokens under load as without load. That
// needs real kernels and real weights, and several copies of the 27B
// checkpoint do not fit on one GPU.
//
// This script writes a model with the same architecture, about a thousand
// times smaller. The weights are random and seeded, in the on-disk format
// tools/convert_hf.py writes, so web/model/loader.js loads it like any
// checkpoint. The tokenizer is byte-level with no merges, which is enough for
// a harness: text encodes to ids and decodes back to the same text.
//
//   node tools/tinymodel.mjs [--out weights/tiny-test] [--layers 8] [--hidden 64]
//
// The output is ~1.3 MB and deterministic: the same --seed writes
// byte-identical files, so a recorded reference token sequence stays valid.

import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';

// --- Arguments ---

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
    const i = argv.indexOf(`--${name}`);
    return i < 0 ? fallback : argv[i + 1];
};
const num = (name, fallback) => Number(flag(name, fallback));

// The defaults give a whole model in ~1.3 MB, with a shape that uses every
// path in model.js: GQA (n_head > n_kv_head), an even head_dim for RoPE's
// split-half pairing, an untied head so the LM head's weight is
// distinguishable from the embedding's, and n_pos covering split.js's N_MAX so
// a context can fill up.
const OUT      = resolve(flag('out', 'weights/tiny-test'));
const SEED     = num('seed', 1);
const HIDDEN   = num('hidden', 64);
const N_LAYER  = num('layers', 8);
const N_HEAD   = num('heads', 4);
const N_KV     = num('kv-heads', 2);
const HEAD_DIM = num('head-dim', HIDDEN / N_HEAD);
const FFN      = num('ffn', 2 * HIDDEN);
const N_POS    = num('n-pos', 4096);
// How strongly each block writes into the residual stream. Without these
// scales the model's argmax is the same token for every input (see attnOut
// below). Attention and the MLP have separate scales because they fail in
// different ways. Attention is the only term that carries the prefix, so it is
// kept strong. SwiGLU's output has a nonzero mean direction that accumulates
// over blocks into a constant, and a constant in the residual stream makes one
// LM-head row win for every input.
const ATTN_SCALE = num('attn-scale', 0.6);
const MLP_SCALE  = num('mlp-scale', 0.25);
// How peaked attention is. Random q/k projections give scores of order 1, so
// the softmax is nearly flat and the attention output is close to an average
// over the prefix, which is almost the same vector for any prefix. The reply
// would then depend only on the last token, and a peer answering from another
// chat's KV cache would produce the same tokens. Scaling q and k up peaks the
// softmax, so attention picks out one earlier position, and which one depends
// on the context. The harness asserts this (see its PROBE).
const QK_SCALE = num('qk-scale', 4);

// --- Numbers ---

// xorshift32, the generator tools/mesh.test.mjs seeds its soak from. One
// integer seed reproduces the whole file.
function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => {
        s ^= s << 13; s >>>= 0;
        s ^= s >>> 17;
        s ^= s << 5; s >>>= 0;
        return s / 0x100000000;
    };
}

const rand = rng(SEED);

// Box-Muller. Gaussian, not uniform, because a uniform [-a, a] weight matrix
// has a sample mean far enough from zero to bias every projection the same
// way. That pushes the model toward emitting one token, and a reference
// sequence of one repeated id would make the harness's assertions pass
// trivially.
function normal(std) {
    const u = Math.max(rand(), 1e-12), v = rand();
    return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const gaussian = (n, std) => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = normal(std);
    return a;
};

const ones = (n) => new Float32Array(n).fill(1);

// `n` rows of `dim`, each scaled to the same norm. Used for the embedding and
// the LM head. With unequal norms, the longest head row wins the argmax for
// almost every input, and a model that emits one token cannot distinguish a
// working mesh from a broken one. With equal norms no token is favoured.
function unitRows(n, dim, scale = 1) {
    const a = new Float32Array(n * dim);
    for (let r = 0; r < n; r++) {
        let sum = 0;
        for (let i = 0; i < dim; i++) { const v = normal(1); a[r * dim + i] = v; sum += v * v; }
        const k = scale / Math.sqrt(sum || 1);
        for (let i = 0; i < dim; i++) a[r * dim + i] *= k;
    }
    return a;
}

// --- The .bin format ---

// A file is its tensors concatenated, and the manifest gives each tensor's
// offset. Every tensor is a whole number of f32s, so offsets are f32-aligned,
// which loader.js's alignment check requires.
function binFile(tensors) {
    const entries = [];
    let offset = 0;
    for (const { name, shape, data } of tensors) {
        entries.push({ name, shape, offset, size: data.byteLength, dtype: 'f32' });
        offset += data.byteLength;
    }
    const buf = Buffer.allocUnsafe(offset);
    let at = 0;
    for (const { data } of tensors) {
        Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(buf, at);
        at += data.byteLength;
    }
    return { buf, entries };
}

// --- The tokenizer ---

// OpenAI's bytes_to_unicode, the table web/model/tokenizer.js also builds.
// Duplicated, not imported, because that module is a browser module that
// fetches its vocabulary, and this script writes one.
function byteEncoder() {
    const bs = [];
    for (let b = 0x21; b <= 0x7e; b++) bs.push(b);
    for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
    for (let b = 0xae; b <= 0xff; b++) bs.push(b);
    const cs = bs.slice();
    const present = new Set(bs);
    let n = 0;
    for (let b = 0; b < 256; b++) if (!present.has(b)) { bs.push(b); cs.push(256 + n++); }
    const enc = new Array(256);
    for (let i = 0; i < bs.length; i++) enc[bs[i]] = String.fromCodePoint(cs[i]);
    return enc;
}

// Byte-level with an empty merge list: every byte is its own token. The
// harness needs `decode(encode(s)) === s` and every id to be inside the
// vocabulary the LM head is sized for. It does not need merges.
//
// The four ChatML specials are real added_tokens, so web/prompt.js's 'chatml'
// wrapper and its stop-sequence regex match and the turn-boundary code is
// tested.
function tokenizer() {
    const enc = byteEncoder();
    const vocab = {};
    enc.forEach((ch, b) => { vocab[ch] = b; });

    const specials = ['<|im_start|>', '<|im_end|>', '<|endoftext|>'];
    const added_tokens = specials.map((content, i) => ({
        id: 256 + i, content, special: true,
        single_word: false, lstrip: false, rstrip: false, normalized: false,
    }));
    for (const a of added_tokens) vocab[a.content] = a.id;

    return {
        json: {
            version: '1.0',
            added_tokens,
            normalizer: null,
            // A bare ByteLevel, the simplest shape readPreTokenizer accepts.
            // The GPT-2 split regex applies, as in the production encoder.
            pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false,
                             trim_offsets: true, use_regex: true },
            post_processor: null,
            decoder: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true },
            model: { type: 'BPE', dropout: null, unk_token: null,
                     continuing_subword_prefix: null, end_of_word_suffix: null,
                     fuse_unk: false, byte_fallback: false,
                     vocab, merges: [] },
        },
        // Exactly the number of entries, not rounded up. With a head wider
        // than the tokenizer, the model can sample an id that has no token,
        // and web/model/tokenizer.js's decode() throws `unknown id N`, which
        // looks like a mesh failure. At vocab 320 with 259 entries, two of
        // three weight seeds sampled an unmapped id within a dozen tokens.
        size: Object.keys(vocab).length,
        eos: 257,          // <|im_end|>
        bos: 256,          // <|im_start|>
    };
}

// --- Writing it out ---

const tk = tokenizer();
const VOCAB = tk.size;
const H = HIDDEN, qDim = N_HEAD * HEAD_DIM, kvDim = N_KV * HEAD_DIM;

if (qDim !== H) {
    console.error(`n_head*head_dim (${qDim}) != hidden (${H}): web/model/model.js's ` +
        `checkDims() rejects that for a non-hybrid checkpoint`);
    process.exit(1);
}
if (HEAD_DIM % 2) {
    console.error(`head_dim ${HEAD_DIM} must be even for RoPE's split-half pairing`);
    process.exit(1);
}
if (N_HEAD % N_KV) {
    console.error(`n_head ${N_HEAD} is not divisible by n_kv_head ${N_KV}`);
    process.exit(1);
}

mkdirSync(`${OUT}/tokenizer`, { recursive: true });

const files = {};
const write = (name, tensors) => {
    const { buf, entries } = binFile(tensors);
    writeFileSync(`${OUT}/${name}`, buf);
    files[name] = entries;
    return buf.length;
};

// 1/sqrt(fan_in) on every projection, which keeps the residual stream from
// vanishing or saturating through eight blocks.
const proj = (out, inn, scale = 1) => gaussian(out * inn, scale / Math.sqrt(inn));

// The two projections that write into the residual stream. Each has its own
// scale, and both are damped by 1/sqrt(2L), as in GPT-2's initialization.
//
// A greedy decode is an argmax, so the reply is a deterministic walk of the
// map `next = f(context)`. The harness checks that every arrangement of peers
// walks the same path, which is only meaningful if the path is long and
// depends on the context. Undamped, eight blocks of random weights add noise
// with a large constant component, one head row wins every argmax, and the
// model emits the same id repeatedly. Damped too much, the residual is only
// the embedding, f depends on the last token alone, and a corrupted KV cache
// would produce the same tokens, so the harness could not detect it.
//
// Attention is therefore kept strong enough to carry the prefix, and the MLP
// is scaled down to keep its mean out of the residual.
// tools/mesh.gpu.test.mjs asserts both properties (varied tokens, and
// different prompts diverging).
const attnOut = (out, inn) => proj(out, inn, ATTN_SCALE / Math.sqrt(2 * N_LAYER));
const mlpOut  = (out, inn) => proj(out, inn, MLP_SCALE / Math.sqrt(2 * N_LAYER));

const wte = unitRows(VOCAB, H);

let bytes = 0;
bytes += write('client.bin', [
    { name: 'wte',  shape: [VOCAB, H], data: wte },
    { name: 'norm', shape: [H],        data: ones(H) },
]);
// The LM head is the embedding shifted by one row. This part of the model is
// not random.
//
// A greedy decode walks `next = f(context)`. With a random head, f has an
// attractor: one id wins the argmax from almost every context, the walk
// reaches it within two or three steps, and the reply is one token repeated.
// This happens at every scale: the residual's constant component decides the
// argmax, and damping the blocks enough to remove the constant leaves f
// depending only on the last token.
//
// With the shifted head, f is close to a permutation, which has cycles and no
// attractor. Here it is one cycle over the whole vocabulary. When the residual
// is dominated by the current token's embedding, logit j is
// <wte[(j+1) % V], h> ~= <wte[j+1], wte[t]>, which is 1 at j = t-1 and
// ~1/sqrt(hidden) elsewhere, so the model counts down through the vocabulary
// and emits V distinct tokens before repeating.
//
// The blocks add the context dependence: their output carries the whole prefix
// through attention, and where it is large enough to overcome that margin it
// changes the next token. The reply is long and varied because of the shift,
// and depends on the whole context because of the blocks.
// tools/mesh.gpu.test.mjs's oracle needs both properties and asserts them.
const shifted = (rows, n, dim) => {
    const out = new Float32Array(n * dim);
    for (let r = 0; r < n; r++) {
        out.set(rows.subarray(((r + 1) % n) * dim, ((r + 1) % n) * dim + dim), r * dim);
    }
    return out;
};

bytes += write('lm_head.bin', [
    { name: 'lm_head', shape: [VOCAB, H], data: shifted(wte, VOCAB, H) },
]);

for (let i = 0; i < N_LAYER; i++) {
    const fn = `block_${String(i).padStart(2, '0')}.bin`;
    bytes += write(fn, [
        { name: 'input_layernorm',          shape: [H],         data: ones(H) },
        { name: 'q_proj',                   shape: [qDim, H],   data: proj(qDim, H, QK_SCALE) },
        { name: 'k_proj',                   shape: [kvDim, H],  data: proj(kvDim, H, QK_SCALE) },
        { name: 'v_proj',                   shape: [kvDim, H],  data: proj(kvDim, H) },
        { name: 'o_proj',                   shape: [H, qDim],   data: attnOut(H, qDim) },
        { name: 'post_attention_layernorm', shape: [H],         data: ones(H) },
        { name: 'gate_proj',                shape: [FFN, H],    data: proj(FFN, H) },
        { name: 'up_proj',                  shape: [FFN, H],    data: proj(FFN, H) },
        { name: 'down_proj',                shape: [H, FFN],    data: mlpOut(H, FFN) },
    ]);
}

const manifest = {
    model: `tiny-test-${N_LAYER}L-${H}h`,
    arch: 'llama',
    hidden: H,
    n_layer: N_LAYER,
    n_head: N_HEAD,
    n_kv_head: N_KV,
    head_dim: HEAD_DIM,
    group: N_HEAD / N_KV,
    ffn: FFN,
    vocab: VOCAB,
    n_pos: N_POS,
    rms_eps: 1e-5,
    rope_theta: 10000.0,
    norm: 'rmsnorm',
    act: 'silu',
    layout: 'row_major',
    weight_layout: 'out_in',
    dtype: 'f32',
    tied_lm_head: false,
    chat_format: 'chatml',
    bos_token_id: tk.bos,
    eos_token_id: tk.eos,
    // Not read by any code. Records how this checkpoint was generated.
    generated_by: `tools/tinymodel.mjs --seed ${SEED}`,
    files,
};

writeFileSync(`${OUT}/manifest.json`, `${JSON.stringify(manifest, null, 1)}\n`);
writeFileSync(`${OUT}/tokenizer/tokenizer.json`, `${JSON.stringify(tk.json)}\n`);

console.log(`${OUT}: ${manifest.model}, ${N_LAYER} blocks over ${N_LAYER + 2} positions, ` +
    `vocab ${VOCAB}, ${(bytes / 1024).toFixed(0)} KiB of weights (seed ${SEED})`);
