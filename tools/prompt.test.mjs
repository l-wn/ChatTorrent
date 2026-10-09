// Tests web/prompt.js in Node: the turn wrapping and stop detection. With the
// wrong wrapper a checkpoint is not prompted the way it was trained, and it
// echoes or rambles. Pure functions with no weights, so this always runs.
//
//   node tools/prompt.test.mjs

import {
    chatFormat, wrapTurn, stopRe, SYSTEM, LLAMA3_SYSTEM, QWEN_SYSTEM,
} from '../web/prompt.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, got === want,
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);

const QWEN_REASONING =
    'Reasoning effort is set to xhigh. Please think carefully through the task, '
    + 'validate key assumptions, consider plausible alternatives, and prioritize '
    + 'correctness, consistency, and clarity in the final answer.';

console.log('\nchatFormat: manifest.chat_format maps to the right format');
{
    eq('chatml', chatFormat({ chat_format: 'chatml' }), 'chatml');
    eq('qwen', chatFormat({ chat_format: 'qwen' }), 'qwen');
    eq('llama3', chatFormat({ chat_format: 'llama3' }), 'llama3');
    eq('unknown string falls back to transcript', chatFormat({ chat_format: 'bogus' }), 'transcript');
    eq('missing field falls back to transcript', chatFormat({}), 'transcript');
    eq('no manifest at all falls back to transcript', chatFormat(undefined), 'transcript');
}

console.log('\nwrapTurn(\'chatml\'): no thinking block');
{
    eq('first turn opens with the SmolLM2 default system prompt',
        wrapTurn('chatml', 'hi', { first: true }),
        `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n`);
    eq('later turn has no system block',
        wrapTurn('chatml', 'again', { first: false }),
        `<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n`);
}

console.log('\nwrapTurn(\'qwen\'), thinking on (opt-in): reasoning instructions + open <think>');
{
    // Thinking is off by default: an xhigh reasoning trace is hundreds to
    // thousands of tokens before the answer, which is too slow on this mesh.
    eq('first turn, thinking defaulted (no option passed): reasoning suppressed',
        wrapTurn('qwen', 'hi', { first: true }),
        `<|im_start|>system\n${QWEN_SYSTEM}<|im_end|>\n`
        + `<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`);
    eq('later turn, thinking defaulted: suppressed, no system block',
        wrapTurn('qwen', 'again', { first: false }),
        `<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`);
    eq('first turn, thinking explicitly true',
        wrapTurn('qwen', 'hi', { first: true, thinking: true }),
        `<|im_start|>system\n${QWEN_REASONING}\n\n${QWEN_SYSTEM}<|im_end|>\n`
        + `<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n`);
    eq('later turn keeps the open <think> and no system block',
        wrapTurn('qwen', 'again', { first: false, thinking: true }),
        `<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n<think>\n`);
    eq('a custom system message still gets the reasoning instructions prepended',
        wrapTurn('qwen', 'hi', { first: true, thinking: true, system: 'Be terse.' }),
        `<|im_start|>system\n${QWEN_REASONING}\n\nBe terse.<|im_end|>\n`
        + `<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n`);
}

console.log('\nwrapTurn(\'qwen\'), thinking off: no reasoning instructions, pre-closed <think>');
{
    eq('first turn, thinking false',
        wrapTurn('qwen', 'hi', { first: true, thinking: false }),
        `<|im_start|>system\n${QWEN_SYSTEM}<|im_end|>\n`
        + `<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`);
    eq('later turn, thinking false',
        wrapTurn('qwen', 'again', { first: false, thinking: false }),
        `<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`);
    eq('custom system message, thinking false: no reasoning text',
        wrapTurn('qwen', 'hi', { first: true, thinking: false, system: 'Be terse.' }),
        `<|im_start|>system\nBe terse.<|im_end|>\n`
        + `<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`);
}

console.log('\nwrapTurn(\'llama3\'): no thinking block');
{
    eq('first turn opens with the llama3 default system prompt',
        wrapTurn('llama3', 'hi', { first: true }),
        `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${LLAMA3_SYSTEM}<|eot_id|>`
        + `<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>`
        + `<|start_header_id|>assistant<|end_header_id|>\n\n`);
}

console.log('\nstopRe: qwen shares chatml\'s markers, think tags are not a stop condition');
{
    ok('chatml matches <|im_start|>', stopRe('chatml').test('<|im_start|>'));
    ok('chatml matches <|im_end|>', stopRe('chatml').test('<|im_end|>'));
    ok('qwen matches <|im_start|>', stopRe('qwen').test('<|im_start|>'));
    ok('qwen matches <|im_end|>', stopRe('qwen').test('<|im_end|>'));
    ok('qwen does NOT stop on <think>', !stopRe('qwen').test('<think>'));
    ok('qwen does NOT stop on </think>', !stopRe('qwen').test('</think>'));
    ok('llama3 matches <|eot_id|>', stopRe('llama3').test('<|eot_id|>'));
    ok('llama3 does not match chatml\'s markers', !stopRe('llama3').test('<|im_end|>'));
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
