// How a turn is wrapped before it reaches the model, and how the end of a
// reply is detected. Four formats:
//
//   'chatml'      <|im_start|>role\n…<|im_end|>. <|im_end|> is an end-of-turn
//                 token the checkpoint was trained to emit.
//   'qwen'        The same tokens as 'chatml', plus a thinking mode: a
//                 reasoning-effort instruction prepended to the system turn,
//                 and a <think> block opening the assistant turn. The block is
//                 left open (the model reasons, then closes it) or closed
//                 empty (no reasoning). See wrapTurn's `thinking` option.
//   'llama3'      <|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>, the
//                 format of Llama 3/3.1's chat template.
//   'transcript'  "User:"/"Assistant:" lines, for a base model. It continues
//                 them because its training data contains dialogue. It was not
//                 trained to stop, so the reply is cut by stopRe().
//
// The checkpoint chooses the format: convert_hf.py writes it into the manifest
// as `chat_format`. A manifest without one gets 'transcript'.

// SmolLM2-Instruct's default system message, from its chat template. Not used
// by 'transcript', which has no system turn.
export const SYSTEM =
    'You are a helpful AI assistant named SmolLM, trained by Hugging Face';

// Llama 3's default system message, without the tool-calling and date preamble
// its full template adds.
export const LLAMA3_SYSTEM = 'You are a helpful assistant.';

// Qwen3's chat template has no default system message, so this one is our
// choice: helpful, and refusing illegal and seriously harmful requests, which
// the site's terms forbid. Kept short: a long list of prohibitions makes a
// model refuse ordinary requests too.
export const QWEN_SYSTEM =
    'You are a helpful assistant. Do not help with anything illegal or seriously harmful, '
    + 'such as sexual content involving minors, weapons capable of mass casualties, malware, '
    + 'or planning violence, fraud or other crimes. If asked, briefly and politely decline. '
    + 'Otherwise, be as helpful as you can.';

// The template's text for its default reasoning_effort ('xhigh'), prepended to
// the system turn when thinking is on. Copied verbatim, so the model sees the
// instruction it was tuned with.
const QWEN_REASONING_INSTRUCTIONS =
    'Reasoning effort is set to xhigh. Please think carefully through the task, '
    + 'validate key assumptions, consider plausible alternatives, and prioritize '
    + 'correctness, consistency, and clarity in the final answer.';

// An unknown format falls back to 'transcript'; it does not throw.
export function chatFormat(manifest) {
    return manifest?.chat_format === 'chatml' ? 'chatml'
        : manifest?.chat_format === 'qwen' ? 'qwen'
        : manifest?.chat_format === 'llama3' ? 'llama3'
        : 'transcript';
}

// One user turn, as the text to encode and feed. Only the new text is encoded,
// since everything before it is in the KV cache. `first` says whether this is
// the first turn.
//
// A later turn opens with `<|im_end|>\n` (ChatML/Qwen) or `<|eot_id|>`
// (llama3). The generator stops before feeding the end-of-turn token back (or
// stops on its budget), so the cache holds an unclosed assistant turn. Without
// the closing token the model sees nested turns.
//
// `thinking` applies only to 'qwen' and reproduces the template's two
// `enable_thinking` behaviours. On: prepend a reasoning-effort instruction and
// leave the assistant's <think> block open. Off: no instruction, and an empty
// closed <think></think>.
//
// Off by default: at the mesh's token rate, a reasoning trace of hundreds to
// thousands of tokens would delay the answer.
export function wrapTurn(format, text, { first, system, thinking = false } = {}) {
    if (format === 'chatml') {
        const sys = system ?? SYSTEM;
        const open = first ? `<|im_start|>system\n${sys}<|im_end|>\n` : '<|im_end|>\n';
        return `${open}<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
    }
    if (format === 'qwen') {
        const sys = system ?? QWEN_SYSTEM;
        const sysBody = thinking ? `${QWEN_REASONING_INSTRUCTIONS}\n\n${sys}` : sys;
        const open = first ? `<|im_start|>system\n${sysBody}<|im_end|>\n` : '<|im_end|>\n';
        const think = thinking ? '<think>\n' : '<think>\n\n</think>\n\n';
        return `${open}<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n${think}`;
    }
    if (format === 'llama3') {
        const sys = system ?? LLAMA3_SYSTEM;
        const open = first
            ? `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${sys}<|eot_id|>`
            : '<|eot_id|>';
        return `${open}<|start_header_id|>user<|end_header_id|>\n\n${text}<|eot_id|>`
            + `<|start_header_id|>assistant<|end_header_id|>\n\n`;
    }
    return `${first ? '' : '\n'}User: ${text}\nAssistant:`;
}

// Where to cut a reply that did not stop on its own, matched against the
// decoded text so far.
//
//   transcript  a base model does not reliably reuse the labels it was given.
//               Prompted with "User:/Assistant:" it produced "The Student:",
//               "Assistant:" and "Toyotomi:" in one reply. So the pattern is
//               any speaker label at the start of a line. A false positive
//               ("\nNote:") truncates the reply.
//   chatml      EOS normally ends the reply first. This catches a reply that
//               writes one of the two markers wrapTurn uses as literal text.
//   qwen        The same two markers as chatml. <think>/</think> are not turn
//               boundaries; the pages separate the reasoning with turn.js's
//               splitReasoning().
//   llama3      The same, for its own two markers.
export function stopRe(format) {
    if (format === 'chatml' || format === 'qwen') return /<\|im_(?:start|end)\|>/;
    if (format === 'llama3') return /<\|(?:start_header_id|eot_id)\|>/;
    return /\n[ \t]*[A-Z][A-Za-z0-9 ._'-]{0,24}:/;
}
