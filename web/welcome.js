// The welcome popup: peer.html's <dialog id="welcome">, shown on every visit.
// It collects the visitor's agreement and their GPU consent. askWelcome()
// resolves with their answer, and peer.js does not open the GPU before that.
// It has two pages: what ChatTorrent is, then the agreement and the GPU
// question. The answer buttons are disabled until both agreement boxes are
// ticked, and the popup cannot be dismissed any other way. The text is in the
// HTML.

// Resolves with { contribute: boolean, limit: string }. `limit` is a value for
// the VRAM limit field ("4gb"; "" means no limit).
export function askWelcome() {
    const dlg = document.getElementById('welcome');
    const boxes = [document.getElementById('agreeAge'), document.getElementById('agreeTerms')];
    const yes = document.getElementById('welcomeYes');
    const no = document.getElementById('welcomeNo');
    const amount = document.getElementById('welcomeAmount');

    const sync = () => {
        const agreed = boxes.every((b) => b.checked);
        for (const b of [yes, no]) {
            b.disabled = !agreed;
            b.title = agreed ? '' : 'Tick both boxes above first';
        }
    };
    for (const b of boxes) b.addEventListener('change', sync);
    sync();
    // The browser restores form state late in the load and on a back/forward
    // return, without a `change` event. The boxes can then be ticked while the
    // buttons are still disabled, so sync again on these events.
    window.addEventListener('load', sync);
    window.addEventListener('pageshow', sync);
    // Also sync whenever the visitor interacts with the popup.
    dlg.addEventListener('pointerover', sync);
    dlg.addEventListener('focusin', sync);

    // The two pages. Back and Next switch between them and move focus to the
    // new page's heading, so a screen reader starts from the top.
    const pages = [document.getElementById('welcomePage1'), document.getElementById('welcomePage2')];
    const headings = ['welcomeTitle', 'welcomeAgreeTitle'];
    const back = document.getElementById('welcomeBack');
    const next = document.getElementById('welcomeNext');
    const pageNo = document.getElementById('welcomePageNo');
    let page = 0;
    const show = (i, focus = true) => {
        page = i;
        pages.forEach((p, j) => { p.hidden = j !== i; });
        pages[i].scrollTop = 0;
        back.disabled = i === 0;
        next.disabled = i === pages.length - 1;
        pageNo.textContent = `${i + 1}/${pages.length}`;
        if (focus) document.getElementById(headings[i]).focus();
    };
    back.addEventListener('click', () => show(page - 1));
    next.addEventListener('click', () => show(page + 1));
    show(0, false);

    // Escape would close a modal <dialog> without an answer, so the cancel
    // event is prevented. Before the page has had a click, Chrome closes the
    // dialog without firing cancel, so a close with no answer reopens it.
    dlg.addEventListener('cancel', (e) => e.preventDefault());
    let answered = false;
    dlg.addEventListener('close', () => { if (!answered) dlg.showModal(); });

    return new Promise((resolve) => {
        const answer = (choice) => { answered = true; dlg.close(); resolve(choice); };
        yes.addEventListener('click', () => answer({ contribute: true, limit: amount.value }));
        no.addEventListener('click', () => answer({ contribute: false, limit: '' }));
        dlg.showModal();
    });
}
