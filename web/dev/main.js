// Entry point for dev.html: boot WebGPU, test the kernels, then load and check
// a real checkpoint (/checkpoint.json's, or ?weights=).
//
// The stages run cheapest first, so a broken kernel is found before any
// weights are downloaded:
//
//   1. kernel unit tests            (kernels.test.js, random inputs vs CPU refs)
//   2. architecture smoke on TINY   (random weights; checks wiring, not numerics)
//   3. tokenizer self-test          (vs ids from the HF tokenizer)
//   4. load the checkpoint          (.bin files → GPU buffers)
//   5. oracle verification          (verify.js, against oracle/<model>/)
//   6. generate
//
// dev.html imports start() from here and provides #log. Everything is also
// logged to the console.

import { DIMS, adapterLabel } from '../model/kernels.js';
import { runAllTests } from './kernels.test.js';
import { runSpecTests } from './spec.test.js';
import { runMtpTests } from './mtp.test.js';
import { benchKernels, benchForward, formatRows, formatForward } from './bench.js';
import { smokeForward } from './smoke.js';
import { checkDims, makeCache, makeScratch, weightBudget } from '../model/model.js';
import { loadWeights, loadManifest, activeWeights, dimsFromManifest, checkManifest } from '../model/loader.js';
import { loadOracle, verifyAgainstOracle, verifyAgainstOracleStreaming } from './verify.js';
import { generate } from '../model/generate.js';
import * as tokenizer from '../model/tokenizer.js';

// The prompt to generate from once every check passes.
const PROMPT = 'The capital of France is';
const MAX_NEW_TOKENS = 40;

// --- Output: to the console and the page, so headless and browser runs match ---

const el = () => document.getElementById('log');

function log(msg = '') {
    console.log(msg);
    const node = el();
    if (node) { node.textContent += msg + '\n'; node.scrollTop = node.scrollHeight; }
}

// Overwrites the last line, for per-file progress.
function logSameLine(msg) {
    const node = el();
    if (!node) return;
    const lines = node.textContent.split('\n');
    lines[lines.length - 2] = msg;
    node.textContent = lines.join('\n');
}

function logRows(title, rows) {
    const failed = rows.filter((x) => !x.ok);
    log(`\n--- ${title}: ${rows.length - failed.length}/${rows.length} passed ---`);
    for (const row of rows) log(`  ${row.ok ? 'PASS' : 'FAIL'}  ${row.name.padEnd(38)} ${row.detail ?? ''}`);
    return failed.length === 0;
}

const MB = (b) => `${(b / 1024 / 1024).toFixed(1)} MB`;

// --- Boot ---

export async function start() {
    if (!navigator.gpu) {
        log('this browser does not support webgpu.');
        return;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        log('this browser supports webgpu but it appears disabled.');
        return;
    }

    log(`  GPU: ${await adapterLabel(adapter)}`);
    log(`  adapter caps: maxBufferSize ${(adapter.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB, `
        + `shader-f16 ${adapter.features.has('shader-f16') ? 'yes' : 'NO'}`);

    // The embedding table is larger than the default 128 MiB buffer limit.
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
        console.error(`WebGPU device was lost: ${info.message}`);
        if (info.reason !== 'destroyed') {
            log('device lost, restarting.');
            start();
        }
    });

    // Awaited so dev.html's .catch() shows kernel errors on the page.
    await main(device);
}

// --- The real model ---

// ?weights=<url> runs the whole page (load, oracle check, generation) on
// another checkpoint: a locally converted one, an f16 build, a different
// model. Relative URLs work, so `?weights=/weights/smollm2-f16` is the local
// converter output as Caddy serves it.
//
// Dev only. A peer in the mesh passes activations to every peer downstream of
// its range, so a URL that chose its weights could corrupt other people's
// tokens. peer.html has no such option.
const WEIGHTS_OVERRIDE = new URLSearchParams(location.search).get('weights');

// Without an override this is the checkpoint /checkpoint.json names. The
// signaling server reads the same file and sends the value to peer.html, so a
// dev run and the mesh test the same checkpoint.
const weightsUrl = async () => WEIGHTS_OVERRIDE ?? await activeWeights();

// ?stream=1 runs oracle verification one block at a time (load, run, diff,
// free), for a checkpoint whose weights exceed this device's VRAM. It is
// slower, since every decode step re-fetches every block, and it stops after
// verification: generation needs the whole model in VRAM.
const STREAM = new URLSearchParams(location.search).get('stream');

// The oracle dump to diff this checkpoint against. Required with ?stream=1:
// the whole-model path only diffs a SmolLM2-135M-shaped checkpoint, against
// oracle/smollm2. Also selects the tokenizer fixture. E.g.
// `/oracle/llama-3.1-8b-instruct`.
const ORACLE_OVERRIDE = new URLSearchParams(location.search).get('oracle');

// The greedy decode check runs 20 steps and re-streams every block once per
// step. ?decodeSteps=3 gives a quicker first result.
const DECODE_STEPS_OVERRIDE = new URLSearchParams(location.search).get('decodeSteps');

// How many blocks verifyAgainstOracleStreaming fetches concurrently and holds
// in VRAM at once. The default is one, the minimum. VRAM use is roughly
// `window * one block's bytes + client.bin`. A larger window shortens the
// decode check's wall-clock time roughly in proportion.
const WINDOW_OVERRIDE = new URLSearchParams(location.search).get('window');

// ?bench runs the microbenchmarks. `?bench=1` uses the checkpoint's dims.
// `?bench=2048x24x32x8x128x8192x128256` benchmarks a shape without downloading
// weights (hidden × n_layer × n_heads × n_kv × head_dim × ffn × vocab). The
// fixture is one layer, so a large model's kernels can be measured on a GPU
// too small to hold the model.
const BENCH = new URLSearchParams(location.search).get('bench');

function benchDims(spec, fallback) {
    if (!spec || spec === '1') return fallback;
    const n = spec.split('x').map(Number);
    if (n.length !== 7 || n.some((v) => !Number.isFinite(v) || v <= 0)) {
        log(`  ignoring ?bench=${spec}: want hidden x n_layer x n_heads x n_kv x head_dim x ffn x vocab`);
        return fallback;
    }
    const [hidden, n_layer, n_heads, n_kv_heads, head_dim, ffn, vocab] = n;
    return { ...fallback, hidden, n_layer, n_heads, n_kv_heads, head_dim, ffn, vocab,
             group: n_heads / n_kv_heads };
}

// N_max and the prefill width the mesh uses (web/split.js), so the numbers
// describe the deployed configuration.
const BENCH_N_MAX = 1024;
const BENCH_T_MAX = 64;

async function runBenchmarks(device, d) {
    // Use the weight dtype this device can run, not the checkpoint's. An f16
    // pipeline on a device without shader-f16 does not compile, its dispatches
    // are dropped, and it would measure as taking no time.
    const f16 = device.features.has('shader-f16');
    log(`\n=== benchmarks: ${d.hidden}x${d.n_layer}, ${d.n_heads}/${d.n_kv_heads} heads,`
        + ` ffn ${d.ffn}, vocab ${d.vocab} ===`);
    log(`  ${f16 ? 'f16' : 'f32 (no shader-f16 on this device)'} weights,`
        + ` N_max ${BENCH_N_MAX}, best-of-3`);

    // Decode at a full context is the worst case. Every token after the prompt
    // pays it, and the attention kernels are slowest at T_act = N_max.
    log(formatRows(`decode  T_new=1, T_past=${BENCH_N_MAX - 1}`,
        await benchKernels(device, d, { T_new: 1, N_max: BENCH_N_MAX, T_past: BENCH_N_MAX - 1 })));
    log(formatRows(`prefill T_new=${BENCH_T_MAX}, T_past=0`,
        await benchKernels(device, d, { T_new: BENCH_T_MAX, N_max: BENCH_N_MAX, T_past: 0 })));
}

async function loadRealWeights(device) {
    const WEIGHTS = await weightsUrl();
    const manifest = await loadManifest(WEIGHTS);
    log(`\nloading ${manifest.model} (${manifest.dtype ?? 'f32'}) from ${WEIGHTS}/ …`);

    const t0 = performance.now();
    log('');   // the line logSameLine overwrites
    const w = await loadWeights(device, {
        manifest,
        baseUrl: WEIGHTS,
        onProgress: (done, total, msg) => logSameLine(`  [${done}/${total}] ${msg}`),
    });
    // Per-file progress overwrites one line in the DOM and is not logged to
    // the console. The summary line goes to both.
    const { params, bytes } = weightBudget(w.d);
    const summary = `  ${(params / 1e6).toFixed(1)} M params, ${MB(bytes)} uploaded in` +
        ` ${((performance.now() - t0) / 1000).toFixed(1)}s`;
    logSameLine(summary);
    console.log(summary);
    return w;
}

async function runGeneration(device, w) {
    const d = w.d ?? DIMS;   // the loaded checkpoint's dims
    const promptIds = await tokenizer.encode(PROMPT);
    const N_max = promptIds.length + MAX_NEW_TOKENS;

    const cache = makeCache(device, d, N_max);
    const s = makeScratch(device, d, promptIds.length, N_max);

    log(`\n--- generating ---`);
    log(`  prompt   ${JSON.stringify(PROMPT)}  →  [${promptIds}]`);
    log(`  kv cache ${MB(cache.bytes)} at N_max=${N_max}, scratch ${MB(s.bytes)}`);
    log('');

    const t0 = performance.now();
    let text = PROMPT;
    const { ids, tokenMs } = await generate(device, d, w, cache, s, promptIds, {
        maxNewTokens: MAX_NEW_TOKENS,
        eosId: w.manifest?.eos_token_id ?? 0,   // <|endoftext|> on SmolLM2
        onToken: () => {},
    });
    // Decoding token by token would split multi-byte characters, so the whole
    // run is decoded once at the end.
    text += await tokenizer.decode(ids);

    const total = (performance.now() - t0) / 1000;
    const prefillMs = tokenMs[0];
    const decodeMs = tokenMs.slice(1);
    const avg = decodeMs.length ? decodeMs.reduce((a, b) => a + b, 0) / decodeMs.length : 0;

    log(text);
    log('');
    log(`  ${ids.length} tokens in ${total.toFixed(1)}s,` +
        ` prefill ${prefillMs.toFixed(0)} ms (${promptIds.length} tok),` +
        ` decode ${avg.toFixed(0)} ms/tok (${(1000 / avg).toFixed(1)} tok/s)`);
    return text;
}

// --- Main ---

async function main(device) {
    const problems = checkDims(DIMS);
    if (problems.length) {
        console.error('DIMS is inconsistent:', problems);
        log(`DIMS is inconsistent: ${problems.join('; ')}`);
        return;
    }

    const kernels = await runAllTests(device);
    const kernelsOk = logRows('kernel tests', kernels.rows);
    logRows('speculative decoding (bitwise against plain decoding)', (await runSpecTests(device)).rows);
    logRows('MTP draft head (plumbing, random weights)', (await runMtpTests(device)).rows);

    // Runs before the smoke test and the weights: the bench builds its own
    // fixture.
    if (BENCH) await runBenchmarks(device, benchDims(BENCH, DIMS));
    const archOk = logRows('architecture smoke (TINY config, random weights)',
        (await smokeForward(device)).rows);

    // The tokenizer must come from the same checkpoint as the weights. With no
    // override both resolve /checkpoint.json.
    if (WEIGHTS_OVERRIDE) tokenizer.source(WEIGHTS_OVERRIDE);

    const tok = await tokenizer.selfTest(
        ORACLE_OVERRIDE ? `${ORACLE_OVERRIDE}/tokenizer_cases.json` : undefined);
    const tokOk = logRows('tokenizer', tok.rows);

    // Streaming path: verify block by block without loading the whole
    // checkpoint onto this GPU. No generation follows (see STREAM).
    if (STREAM) {
        if (!ORACLE_OVERRIDE) {
            log('\n?stream=1 needs ?oracle=<url> naming the oracle dump for this checkpoint '
                + '(e.g. ?oracle=/oracle/llama-3.1-8b-instruct).');
            return;
        }
        const WEIGHTS = await weightsUrl();
        const manifest = await loadManifest(WEIGHTS);
        const d = dimsFromManifest(manifest);
        const problems2 = checkManifest(manifest, d);
        if (problems2.length) {
            log(`${manifest.model ?? WEIGHTS} cannot run on these kernels:\n  ${problems2.join('\n  ')}`);
            return;
        }
        const window = WINDOW_OVERRIDE ? Number(WINDOW_OVERRIDE) : 8;
        log(`\nstreaming verification of ${manifest.model} (${manifest.dtype ?? 'f32'}) `
            + `against ${ORACLE_OVERRIDE}: ${window} of ${d.n_layer} blocks resident at once `
            + `(?window= to change it). This re-fetches every block once for the prefill and once `
            + `per decode step, so it is still slow. Expect minutes, not seconds. The line below `
            + `moves as each window finishes.`);
        const oracle = await loadOracle(d, ORACLE_OVERRIDE);
        log('');   // the line logSameLine overwrites
        const v = await verifyAgainstOracleStreaming(device, { baseUrl: WEIGHTS, manifest, d }, oracle,
            { dtype: d.dtype ?? 'f32',
              maxDecodeSteps: DECODE_STEPS_OVERRIDE ? Number(DECODE_STEPS_OVERRIDE) : 20,
              blockWindow: WINDOW_OVERRIDE ? Number(WINDOW_OVERRIDE) : 8,
              onProgress: (msg) => logSameLine(`  ${msg}`) });
        const verifyOk = logRows(
            `streaming oracle verification, prompt ${JSON.stringify(oracle.tokens.prompt)}`, v.rows);
        log(`\n${kernelsOk && archOk && tokOk && verifyOk
            ? 'all green: kernels verified, weights stream-verified against HuggingFace'
            : 'FAILURES above, see the per-test detail'}`);
        log('\n(?stream=1 stops here. Generation for a checkpoint this size runs on the '
            + 'mesh, not one tab)');
        return;
    }

    const w = await loadRealWeights(device);

    // The oracle is a dump of SmolLM2-135M's intermediates, so it applies only
    // to that shape. Another checkpoint skips it and only loads and generates.
    // ?stream=1&oracle= diffs a checkpoint against its own dump.
    const oracular = w.d.n_layer === DIMS.n_layer && w.d.hidden === DIMS.hidden;
    let verifyOk = true;
    if (oracular) {
        log('\nfetching oracle…');
        const oracle = await loadOracle(DIMS);
        const v = await verifyAgainstOracle(device, DIMS, w, oracle,
                                            { dtype: w.d?.dtype ?? 'f32' });
        verifyOk = logRows(`oracle verification, prompt ${JSON.stringify(oracle.tokens.prompt)}`,
            v.rows);
    } else {
        log(`\nskipping oracle: it is a SmolLM2-135M dump (${DIMS.n_layer}x${DIMS.hidden}) and `
            + `this is ${w.manifest?.model ?? 'another model'} (${w.d.n_layer}x${w.d.hidden})`);
    }

    if (BENCH) {
        log(formatForward(await benchForward(device, w.d, w, { T_new: 1, T_past: 0, N_max: 256 })));
        log(formatForward(await benchForward(device, w.d, w, { T_new: 64, T_past: 0, N_max: 256, iters: 4 })));
    }

    await runGeneration(device, w);

    const all = kernelsOk && archOk && tokOk && verifyOk;
    log(`\n${all
        ? 'all green: kernels verified, weights loaded, output matches HuggingFace'
        : 'FAILURES above, see the per-test detail'}`);
}
