// One transcript message, shared by peer.js and chat.js. Returns the element
// and setters for its parts. The caller appends it.

export function makeMessage(who, text = '', cls = '') {
    const el = document.createElement('div');
    el.className = `msg ${cls}`;
    el.innerHTML = `<div class="who"></div><div class="body"></div><div class="meta"></div>`;
    el.querySelector('.who').textContent = who;
    el.querySelector('.body').textContent = text;
    return {
        el,
        set body(v) { el.querySelector('.body').textContent = v; },
        set meta(v) { el.querySelector('.meta').textContent = v; },
        // The reasoning block, collapsed. With thinking on, the opening
        // <think> is part of the prompt, so the reply starts inside it.
        set reasoning(v) {
            let box = el.querySelector('.think');
            if (!box) {
                box = document.createElement('details');
                box.className = 'think';
                box.innerHTML = '<summary></summary><div></div>';
                el.insertBefore(box, el.querySelector('.body'));
            }
            box.querySelector('summary').textContent = `reasoning (${v.length} chars)`;
            box.querySelector('div').textContent = v;
        },
    };
}
