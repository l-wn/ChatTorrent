// The lending slider and the model bar's preview, shared by the pre-join
// screen (welcome.js) and the create page (create.js).

import { place } from './split.js';

export const MAX_GB = 32;          // the most the slider offers
export const DEFAULT_GB = 4;

// "4gb" -> 4. Blank (no limit) and anything unreadable -> null.
export function limitGb(limit) {
    const m = /^(\d+(?:\.\d+)?)\s*(gb|gib)?$/i.exec(String(limit ?? '').trim());
    return m ? Math.round(Number(m[1])) : null;
}

// The range a new tab ends up holding with `n` pieces: the server's place()
// for the minimum claim, then growth toward the side with fewer copies,
// alternating when both sides have the same (as engine.js's #growClaim()).
// `held` is the copies of each piece that can run now.
export function previewRange(claims, held, L, floor, n) {
    const f = Math.min(Math.max(1, floor || 1), L);
    if (n < f) return null;
    let lo = place(claims, f, L), hi = lo + f, side = 'left';
    while (hi - lo < n) {
        const canLeft = lo > 0, canRight = hi < L;
        if (!canLeft && !canRight) break;
        const left = canLeft ? held[lo - 1] : Infinity;
        const right = canRight ? held[hi] : Infinity;
        const dir = !canLeft ? 'right' : !canRight ? 'left'
            : left < right ? 'left' : right < left ? 'right' : side;
        if (dir === 'left') lo--; else hi++;
        side = dir === 'left' ? 'right' : 'left';
    }
    return [lo, hi];
}

// One cell per piece in `bar`, rebuilt when the count changes.
export function cellsFor(bar, L) {
    if (bar.childElementCount !== L) {
        bar.replaceChildren(...Array.from({ length: L }, (_, x) => {
            const cell = document.createElement('i');
            cell.style.setProperty('--i', x);        // its place in the full-bar wave
            return cell;
        }));
        bar.style.setProperty('--n', L);
    }
    return [...bar.children];
}

// Wire a slider: `bar` the track, `fill` the part up to the value, `handle`
// the knob (role="slider"), `out` the number shown. Drag or click the track,
// or use the arrow keys on the handle. Calls onChange(gb) on every change.
// lock() sets it to zero for good, for a browser that can't lend.
export function lendSlider({ bar, fill, handle, out, value = DEFAULT_GB, onChange = () => {} }) {
    let gb = Math.max(0, Math.min(MAX_GB, value));
    let locked = false;
    const show = () => {
        out.textContent = gb ? `${gb} GB` : 'None';
        fill.style.width = `${(gb / MAX_GB) * 100}%`;
        handle.setAttribute('aria-valuemax', String(MAX_GB));
        handle.setAttribute('aria-valuenow', String(gb));
        handle.setAttribute('aria-valuetext', gb ? `${gb} GB` : 'none');
    };
    const set = (v) => {
        if (locked) return;
        const next = Math.max(0, Math.min(MAX_GB, Math.round(v)));
        if (next === gb) return;
        gb = next;
        show();
        onChange(gb);
    };
    const fromPointer = (e) => {
        const r = bar.getBoundingClientRect();
        set(((e.clientX - r.left) / r.width) * MAX_GB);
    };
    bar.addEventListener('pointerdown', (e) => {
        bar.setPointerCapture(e.pointerId);
        bar.classList.add('dragging');
        fromPointer(e);
        handle.focus({ preventScroll: true });
    });
    bar.addEventListener('pointermove', (e) => { if (bar.hasPointerCapture(e.pointerId)) fromPointer(e); });
    const stop = () => bar.classList.remove('dragging');
    bar.addEventListener('pointerup', stop);
    bar.addEventListener('pointercancel', stop);
    handle.addEventListener('keydown', (e) => {
        const step = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 4, PageDown: -4 }[e.key];
        if (step) { e.preventDefault(); set(gb + step); }
        else if (e.key === 'Home') { e.preventDefault(); set(0); }
        else if (e.key === 'End') { e.preventDefault(); set(MAX_GB); }
    });
    show();
    return {
        get value() { return gb; },
        lock() {
            set(0);
            locked = true;
            bar.classList.add('locked');
            handle.setAttribute('aria-disabled', 'true');
            handle.tabIndex = -1;
        },
    };
}
