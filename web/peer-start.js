// peer.html's entry point: boot, and show a failure in the page as well as in
// the console. A separate file, so the deploy's Content-Security-Policy can
// block inline scripts.

import { boot } from './peer.js';
import { askWelcome } from './welcome.js';

boot(askWelcome()).catch((err) => {
    console.error(err);
    const status = document.getElementById('status');
    status.textContent = 'failed';
    status.className = 'pill err';
    const msg = document.createElement('div');
    msg.className = 'msg note err';
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = String(err.stack || err);
    msg.append(body);
    document.getElementById('messages').append(msg);
});
