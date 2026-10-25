// Tests web/errors.js: raw errors map to the right problem, and the chat
// sentence never contains the raw text.
//
//   node tools/errors.test.mjs

import { PROBLEMS, classify, message } from '../web/errors.js';

let pass = 0, fail = 0;
const eq = (name, got, want) => {
    if (got === want) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); }
};

const vulkanOom = new Error('client.bin: GPU ran out of memory uploading its tensors (vkAllocateMemory failed ' +
    'with VK_ERROR_OUT_OF_DEVICE_MEMORY\n - While calling [Device].CreateBuffer([BufferDescriptor]).\n' +
    '    at CheckVkOOMThenSuccessImpl (../../third_party/dawn/src/dawn/native/vulkan/VulkanError.cpp:119)\n)');

console.log('\nclassify');
eq('a Vulkan out-of-memory is the GPU being full', classify(vulkanOom), 'GPU_FULL');
eq('no WebGPU', classify(new Error('This browser has no WebGPU, so this tab can chat but not lend')), 'NO_WEBGPU');
eq('a computer with no room', classify(new Error('no session free for another right now; try again in a moment')), 'BUSY');
eq('a backed-up link', classify(new Error('the link to peer 4 is backed up; try again in a moment')), 'BUSY');
eq("the cluster's reply limit", classify("the cluster's limit of 8 replies at once"), 'REPLY_LIMIT');
eq('a full context', classify(new Error('this prompt needs 900 tokens and only 12 are left in the 4096-token context')), 'CHAT_FULL');
eq('anything else falls back', classify(new Error('something odd')), 'REPLY_FAILED');
eq('with the fallback given', classify(new Error('something odd'), ''), '');

console.log('\nmessages');
const codes = Object.values(PROBLEMS).map((p) => p.code);
eq('every code is unique', new Set(codes).size, codes.length);
for (const [key, p] of Object.entries(PROBLEMS)) {
    const m = message(key);
    if (!m.endsWith(`(${p.code})`) || /[<>{}]|Error|at \w+ \(/.test(m)) eq(`${key} reads as a plain sentence`, m, 'plain');
}
eq('every message ends in its code and has no raw text', true, true);
eq('an unknown key reads as a failed reply', message('NOPE'), message('REPLY_FAILED'));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
