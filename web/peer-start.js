// peer.html's entry point: boot, and show a failure in the page as well as in
// the console. A separate file, so the deploy's Content-Security-Policy can
// block inline scripts.
//
// `/chat#new` (the landing page's "Start a private cluster") becomes a new
// invite here, before the welcome dialog or the engine reads the URL. This
// tab is then marked as the cluster's creator for the session.

import { boot } from './peer.js';
import { askWelcome } from './welcome.js';
import { newSecret, creatorKey } from './room.js';
import { PROBLEMS, report } from './errors.js';

if (location.hash === '#new') {
    const secret = newSecret();
    try { sessionStorage.setItem(creatorKey(secret), '1'); } catch { /* shown as an invitee then */ }
    history.replaceState(null, '', `${location.pathname}${location.search}#mesh=${secret}`);
}

boot(askWelcome()).catch((err) => {
    // The raw error goes to the console only (errors.js).
    const status = document.getElementById('status');
    status.textContent = 'failed';
    status.className = 'err';
    status.hidden = false;
    const msg = document.createElement('div');
    msg.className = 'msg note err';
    msg.dataset.code = PROBLEMS.START_FAILED.code;
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = report('START_FAILED', err);
    msg.append(body);
    document.getElementById('messages').append(msg);
});
