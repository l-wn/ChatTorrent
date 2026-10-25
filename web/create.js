// The create page (create.html): start a private cluster. The model bar
// previews how much of the model this tab would hold at the amount on the
// slider; nobody else is in the cluster yet, so the preview grows outwards
// from the middle. "Create cluster" saves the choices, makes the invite and
// opens the chat in it, past the pre-join screen.

import { newSecret, cleanName, nameKey, creatorKey, chatsKey } from './room.js';
import { storedBytes } from './split.js';
import { saveChoice, joinedKey } from './welcome.js';
import { lendSlider, previewRange, cellsFor } from './lend.js';

// What one tab spends on the GPU before it holds any piece (its working
// memory and the larger end table), roughly as engine.js's #capacity()
// charges it.
const FIXED_GB = 1.2;

const $ = (id) => document.getElementById(id);
let model = { L: 66, gb: 27 };      // replaced by the checkpoint's numbers

function render(gb) {
    const { L } = model;
    const cells = cellsFor($('previewBar'), L);
    const n = Math.max(0, Math.min(L, Math.floor((gb - FIXED_GB) / (model.gb / L))));
    const [lo, hi] = previewRange([], new Int32Array(L), L, 1, n) ?? [0, 0];
    cells.forEach((c, x) => { c.dataset.state = x >= lo && x < hi ? 'you' : 'empty'; });
    const pct = Math.round(((hi - lo) / L) * 100);
    $('modelPct').textContent = `${pct}% (${Math.round(((hi - lo) / L) * model.gb)}/${Math.round(model.gb)} GB)`;
    $('previewBar').classList.toggle('full', hi - lo === L);
    $('previewCaption').textContent = !gb ? ''
        : pct >= 100 ? 'You would hold the whole model on your own.'
        : `You'd hold about ${pct}%. Invite people to lend the rest, about ${Math.ceil(model.gb * (1 - pct / 100))} GB.`;
}

// Lending starts at 0: the visitor chooses to give some.
const slider = lendSlider({
    bar: $('lendBar'), fill: $('lendYou'), handle: $('lendHandle'), out: $('lendOut'),
    value: 0, onChange: render,
});
render(slider.value);


// The model's size and piece count, from the checkpoint the cluster runs.
(async () => {
    try {
        const { url } = await (await fetch('/checkpoint.json')).json();
        const m = await (await fetch(`${url.replace(/\/$/, '')}/manifest.json`)).json();
        const b = storedBytes(m);
        if (!b) return;
        const bytes = b.wte + b.blocks.reduce((a, x) => a + x, 0) + b.norm + b.lm_head;
        model = { L: m.n_layer + 2, gb: bytes / 2 ** 30 };
        $('modelNeed').textContent = `${Math.round(model.gb)} GB`;
        render(slider.value);
    } catch (err) {
        console.info('create: using the default model size', err);
    }
})();

$('create').addEventListener('click', () => {
    const secret = newSecret();
    const gb = slider.value;
    const store = (fn) => { try { fn(); } catch { /* not kept */ } };
    store(() => sessionStorage.setItem(creatorKey(secret), '1'));
    store(() => sessionStorage.setItem(joinedKey(secret), '1'));
    // Room capacity: the cluster's limit on replies at once (blank: none).
    const cap = Number($('capacity').value);
    if (Number.isInteger(cap) && cap >= 1) store(() => sessionStorage.setItem(chatsKey(secret), String(cap)));
    store(() => sessionStorage.setItem(nameKey(secret), cleanName($('name').value)));
    saveChoice(gb ? { contribute: true, limit: `${gb}gb` } : { contribute: false, limit: '' });
    location.href = `/chat#mesh=${secret}`;
});
