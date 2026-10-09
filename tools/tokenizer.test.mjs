// Tests the tokenizer in Node. A tokenizer bug produces no error: a wrong
// pre-tokenizer or a missed merge gives plausible ids, the model runs on them,
// and the output is fluent but wrong.
//
//   node tools/tokenizer.test.mjs
//
// Two parts:
//
//   1. readPreTokenizer / digitSplit as unit tests, on the pre_tokenizer
//      shapes in use and the ones that must be rejected. Always runs.
//   2. The full encoder against ids from HuggingFace's tokenizer (the
//      REFERENCE table below, generated with transformers). Runs only when a
//      tokenizer.json is present, since weights/ is gitignored.

import { readFileSync, existsSync } from 'node:fs';
import { readPreTokenizer, digitSplit, translateHfRegex, PRETOK_RE } from '../web/model/tokenizer.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
const throws = (name, fn, match) => {
    try { fn(); ok(name, false, 'did not throw'); }
    catch (e) { ok(name, match ? e.message.includes(match) : true, `${e.message}`); }
};

// The three shapes in use. BYTELEVEL is the 135M snapshot's. SEQUENCE is used
// by every SmolLM2 export above 135M. SPLIT_SEQUENCE is Llama 3's, with the
// pattern transformers ships for meta-llama/Llama-3.1-8B-Instruct.
const BYTELEVEL = { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: true };
const BYTELEVEL_NOREGEX = { ...BYTELEVEL, use_regex: false };
const SEQUENCE = { type: 'Sequence', pretokenizers: [
    { type: 'Digits', individual_digits: true }, BYTELEVEL] };
const LLAMA3_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const SPLIT_SEQUENCE = { type: 'Sequence', pretokenizers: [
    { type: 'Split', pattern: { Regex: LLAMA3_REGEX }, behavior: 'Isolated', invert: false },
    BYTELEVEL_NOREGEX] };

console.log('\nreadPreTokenizer: all three shapes load, everything else is refused');
{
    eq('a bare ByteLevel has no digit split', readPreTokenizer(BYTELEVEL).splitDigits, false);
    eq('...and uses the default GPT-2 split',
        readPreTokenizer(BYTELEVEL).splitRe.source, PRETOK_RE.source);
    eq('Sequence[Digits, ByteLevel] has one', readPreTokenizer(SEQUENCE).splitDigits, true);
    eq('a Sequence with only ByteLevel has none',
        readPreTokenizer({ type: 'Sequence', pretokenizers: [BYTELEVEL] }).splitDigits, false);

    const split = readPreTokenizer(SPLIT_SEQUENCE);
    eq('Sequence[Split, ByteLevel{use_regex:false}] has no digit split', split.splitDigits, false);
    ok('...and its splitRe is the translated Split pattern, not the GPT-2 default',
        split.splitRe.source !== PRETOK_RE.source);
    eq('the translated regex applies the case-insensitive contraction',
        "It'S".match(split.splitRe), ['It', "'S"]);
    eq('...and caps digit runs at 3, unlike SmolLM2\'s Digits step',
        '1234'.match(split.splitRe), ['123', '4']);

    throws('a sentencepiece tokenizer (no pre_tokenizer) is refused',
        () => readPreTokenizer(null), 'exactly one ByteLevel');
    throws('a wholly unmodelled step is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'WhitespaceSplit' }, BYTELEVEL] }), 'only Digits, Split and ByteLevel');
    throws('Digits after ByteLevel is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            BYTELEVEL, { type: 'Digits', individual_digits: true }] }), 'must run before');
    throws('grouped digits are refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'Digits', individual_digits: false }, BYTELEVEL] }), 'different split');
    throws('ByteLevel without the GPT-2 regex, with no Split step, is refused',
        () => readPreTokenizer({ ...BYTELEVEL, use_regex: false }), 'use_regex');
    throws('add_prefix_space is refused',
        () => readPreTokenizer({ ...BYTELEVEL, add_prefix_space: true }), 'add_prefix_space');
    throws('two ByteLevel steps are refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [BYTELEVEL, BYTELEVEL] }),
        'exactly one');
    throws('Split after ByteLevel is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            BYTELEVEL_NOREGEX, { type: 'Split', pattern: { Regex: 'x' }, behavior: 'Isolated' }] }),
        'must run before');
    throws('Split with a non-Isolated behavior is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'Split', pattern: { Regex: 'x' }, behavior: 'Removed', invert: false },
            BYTELEVEL_NOREGEX] }), 'behavior');
    throws('Split{invert:true} is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'Split', pattern: { Regex: 'x' }, behavior: 'Isolated', invert: true },
            BYTELEVEL_NOREGEX] }), 'invert');
    throws('Split without a string pattern.Regex is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'Split', pattern: { String: 'x' }, behavior: 'Isolated', invert: false },
            BYTELEVEL_NOREGEX] }), 'pattern.Regex');
    throws('Split paired with a still-splitting ByteLevel is refused (double split)',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'Split', pattern: { Regex: 'x' }, behavior: 'Isolated', invert: false },
            BYTELEVEL] }), 'use_regex:false');
    throws('Split and Digits together is refused',
        () => readPreTokenizer({ type: 'Sequence', pretokenizers: [
            { type: 'Digits', individual_digits: true },
            { type: 'Split', pattern: { Regex: 'x' }, behavior: 'Isolated', invert: false },
            BYTELEVEL_NOREGEX] }), 'Split and Digits together');
}

console.log('\ntranslateHfRegex: the one syntax gap between Rust regex and ECMAScript');
{
    ok('strips the inline case-insensitive group',
        !translateHfRegex("(?i:'s)").source.includes('?i:'));
    eq('and matches case-insensitively where the group was',
        "'S".match(translateHfRegex("(?i:'s|'t)")), ["'S"]);
    eq('leaves \\p{L}/\\p{N} classes working under the u flag',
        'a1'.match(translateHfRegex('\\p{L}|\\p{N}')), ['a', '1']);
}

console.log('\ndigitSplit: every digit its own pre-token, runs between them intact');
{
    eq('a bare number splits', digitSplit('2024'), ['2', '0', '2', '4']);
    eq('the leading space stays with nothing', digitSplit(' 2024'), [' ', '2', '0', '2', '4']);
    eq('letters around digits stay whole', digitSplit('abc123def'), ['abc', '1', '2', '3', 'def']);
    eq('no digits is one segment', digitSplit('no digits here'), ['no digits here']);
    eq('empty stays empty', digitSplit(''), []);
    // \p{N}, matching Rust's char::is_numeric(), not [0-9].
    eq('arabic-indic digits count', digitSplit('١٢'), ['١', '٢']);
    eq('roman numerals count (Nl)', digitSplit('aⅫb'), ['a', 'Ⅻ', 'b']);
}

// --- Against HuggingFace ---
//
// Ids from transformers' AutoTokenizer. SmolLM2-135M and SmolLM2-1.7B have the
// same vocab and agree on every case, so one table covers both. Regenerate
// with:
//
//   tok.encode(case, add_special_tokens=False)  for each case below
const REFERENCE = [
    ["The capital of France is", [504, 3575, 282, 4649, 314]],
    ["2024", [34, 32, 34, 36]],
    [" 2024", [216, 34, 32, 34, 36]],
    ["In 1999, 42 people arrived.", [788, 216, 33, 41, 41, 41, 28, 216, 36, 34, 701, 6612, 30]],
    ["3.14159", [35, 30, 33, 36, 33, 37, 41]],
    ["abc123def", [25276, 33, 34, 35, 1604]],
    ["v2.0.1", [102, 34, 30, 32, 30, 33]],
    ["no digits here", [4607, 18638, 1535]],
    ["0", [32]],
    ["100000000", [33, 32, 32, 32, 32, 32, 32, 32, 32]],
    ["tabs\tand\nnewlines", [100, 7366, 197, 397, 198, 2241, 5110]],
    ["unicode: caf\u00e9 \u2014 na\u00efve \u2728", [35852, 42, 37366, 1841, 15486, 46494, 4636, 246, 118]],
    ["\u0661\u0662\u0663 arabic digits", [164, 111, 164, 112, 164, 113, 46179, 286, 18638]],
    ["\u216b roman numeral", [173, 223, 121, 41800, 861, 1555]],
    ["<|endoftext|>", [0]],
    ["<|im_start|>user\nhi 42<|im_end|>", [1, 4093, 198, 6004, 216, 36, 34, 2]],
    ["  leading spaces", [216, 2899, 5600]],
    ["trailing  ", [4861, 4966, 256]],
    ["9lives", [41, 92, 898]],
    ["x9", [104, 41]],
    ["1,234,567.89", [33, 28, 34, 35, 36, 28, 37, 38, 39, 30, 40, 41]]
];

// The same for Llama 3.1's tokenizer, which has a different vocab and merge
// table and uses the Sequence[Split, ByteLevel] pre-tokenizer. Generated with:
//
//   HF_HUB_OFFLINE=1 tok.encode(case, add_special_tokens=False)
//   AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
const REFERENCE_LLAMA = [
    ["The capital of France is", [791, 6864, 315, 9822, 374]],
    ["2024", [2366, 19]],
    [" 2024", [220, 2366, 19]],
    ["In 1999, 42 people arrived.", [644, 220, 2550, 24, 11, 220, 2983, 1274, 11721, 13]],
    ["3.14159", [18, 13, 9335, 2946]],
    ["abc123def", [13997, 4513, 755]],
    ["v2.0.1", [85, 17, 13, 15, 13, 16]],
    ["no digits here", [2201, 19016, 1618]],
    ["0", [15]],
    ["100000000", [1041, 931, 931]],
    ["tabs\tand\nnewlines", [32093, 53577, 198, 943, 8128]],
    ["unicode: café — naïve ✨", [25535, 25, 53050, 2001, 95980, 588, 26602, 101]],
    ["١٢٣ arabic digits", [109610, 110546, 117413, 52412, 292, 19016]],
    ["Ⅻ roman numeral", [71567, 104, 48876, 84392]],
    ["<|begin_of_text|>", [128000]],
    ["  leading spaces", [220, 6522, 12908]],
    ["trailing  ", [376, 14612, 256]],
    ["9lives", [24, 75, 1924]],
    ["x9", [87, 24]],
    ["1,234,567.89", [16, 11, 11727, 11, 19282, 13, 4578]],
    ["It'S a test", [2181, 13575, 264, 1296]],   // the case-insensitive contraction
    ["CAFÉ", [82418, 27887]],
];

const CANDIDATES = [
    ['weights/smollm2/tokenizer/tokenizer.json',           'bare ByteLevel (135M snapshot)',
     REFERENCE],
    ['weights/smollm2-f16/tokenizer/tokenizer.json',       'f16 build',
     REFERENCE],
    ['weights/smollm2-1.7b/tokenizer/tokenizer.json',      'Sequence[Digits, ByteLevel] (1.7B)',
     REFERENCE],
    ['weights/llama-3.1-8b-instruct/tokenizer/tokenizer.json',
     'Sequence[Split, ByteLevel{use_regex:false}] (Llama 3.1)', REFERENCE_LLAMA],
];

for (const [path, label, reference] of CANDIDATES) {
    if (!existsSync(path)) {
        console.log(`\nencode vs HuggingFace, ${label}: skipped, no ${path}`);
        continue;
    }
    console.log(`\nencode vs HuggingFace, ${label}`);
    // tokenizer.js fetches its json and memoizes the result, so each candidate
    // gets a new module instance with fetch returning that file.
    globalThis.fetch = async () => ({
        ok: true, status: 200, json: async () => JSON.parse(readFileSync(path, 'utf8')),
    });
    const tk = await import(`../web/model/tokenizer.js?${encodeURIComponent(path)}`);
    // Set the checkpoint explicitly, as engine.js does. Otherwise the module
    // would fetch /checkpoint.json, and this stub answers every request with
    // the tokenizer under test, which has no "url" field.
    tk.source(path.replace(/\/tokenizer\/tokenizer\.json$/, ''));
    let bad = 0;
    for (const [text, want] of reference) {
        let got;
        try { got = await tk.encode(text); } catch (e) { got = `threw: ${e.message}`; }
        if (JSON.stringify(got) !== JSON.stringify(want)) {
            bad++;
            if (bad <= 5) console.log(`  FAIL  ${JSON.stringify(text)}\n          got  ${JSON.stringify(got)}\n          want ${JSON.stringify(want)}`);
        }
        // A round trip catches a decode table that disagrees with encode.
        if (Array.isArray(got) && await tk.decode(got) !== text) {
            bad++;
            console.log(`  FAIL  decode round trip ${JSON.stringify(text)}`);
        }
    }
    ok(`all ${reference.length} cases match HuggingFace, and decode round-trips`, bad === 0,
        `${bad} mismatches`);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
