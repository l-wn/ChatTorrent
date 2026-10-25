// What the chat page tells people when something goes wrong: a short code,
// one plain sentence and what to do about it. `calm` ones are shown as an
// ordinary note, not an error. The raw error goes to the
// console under the same code, so a code seen in the chat leads to the
// details there.

export const PROBLEMS = {
    NO_WEBGPU:     { code: 'E01', text: "This browser can't lend its graphics card.", hint: 'You can still chat. To lend, use Chrome, Edge or Safari 26.', calm: true },
    GPU_CHECK:     { code: 'E02', text: 'Your GPU failed its start-up check, so it is not lending.', hint: 'You can still chat.' },
    GPU_LOST:      { code: 'E03', text: 'Your GPU stopped responding.', hint: 'Reload the page.' },
    GPU_ERROR:     { code: 'E04', text: 'Your GPU reported an error.', hint: 'If replies start looking wrong, reload the page.' },
    LEND_FAILED:   { code: 'E05', text: "Couldn't start lending your GPU.", hint: 'You can still chat. Reload to try again.' },
    GROW_FAILED:   { code: 'E06', text: 'Stopped taking on more of the model.', hint: 'Your GPU keeps lending what it already holds.' },

    BAD_LINK:      { code: 'E10', text: "This invite link doesn't work.", hint: 'Ask for the link again, or join the public demo.' },
    NO_CONNECTION: { code: 'E11', text: "Can't reach the cluster.", hint: 'Check your connection and reload the page.' },
    NO_RELAY:      { code: 'E12', text: 'The server has no relay, so computers cannot connect.', hint: 'Whoever runs this server needs to set one up.' },
    NO_TOKENIZER:  { code: 'E13', text: "Couldn't load what this tab needs to chat.", hint: 'Reload the page. Your GPU can still lend meanwhile.' },

    REPLY_FAILED:  { code: 'E20', text: 'The reply failed.', hint: 'Send your message again.' },
    BUSY:          { code: 'E21', text: 'The cluster is busy right now.', hint: 'Try again in a moment.' },
    REPLY_LIMIT:   { code: 'E22', text: 'The cluster is answering as many people as it can.', hint: 'Try again in a moment.' },
    CHAT_FULL:     { code: 'E23', text: 'This chat is too long for the model to read.', hint: 'Start a new chat from the chat list.' },

    SAVE_FAILED:   { code: 'E30', text: "This message couldn't be saved in your browser.", hint: 'The chat still works, but may be gone after a reload.' },

    START_FAILED:  { code: 'E99', text: 'Something went wrong starting the page.', hint: 'Reload it. If it keeps happening, send the code to whoever runs the site.' },
};

// The problem a raw error most likely is, by its message.
export function classify(err, fallback = 'REPLY_FAILED') {
    const m = String(err?.message ?? err ?? '');
    if (/out of memory|OUT_OF_DEVICE_MEMORY|OutOfMemory/i.test(m)) return 'GPU_FULL';
    if (/no WebGPU|navigator\.gpu|requestAdapter|no adapter/i.test(m)) return 'NO_WEBGPU';
    if (/limit of \d+ replies/.test(m)) return 'REPLY_LIMIT';
    if (/try again in a moment|backed up|free for another/i.test(m)) return 'BUSY';
    if (/tokens and only \d+ are left|context is full/i.test(m)) return 'CHAT_FULL';
    return fallback;
}

// The sentence shown in the chat, e.g. "The reply failed. Send your message
// again. (E20)".
export function message(key) {
    const p = PROBLEMS[key] ?? PROBLEMS.REPLY_FAILED;
    return `${p.text} ${p.hint} (${p.code})`;
}

// Log the raw error under the problem's code and return the chat sentence.
export function report(key, err) {
    const p = PROBLEMS[key] ?? PROBLEMS.REPLY_FAILED;
    if (err != null) console.error(`[${p.code}]`, err);
    return message(key);
}
