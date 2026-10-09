// A small Chrome DevTools Protocol client, enough to run one tab per peer.
//
// The mesh needs several tabs. One tab is one peer, with its own Engine,
// GPUDevice, WebSocket to the signaling server and RTCPeerConnections. Several
// Engines in one document would share a tokenizer, a device and an event loop,
// all of which are under test.
//
// The harness therefore opens N tabs and talks to each. The repo's only
// dependency is `ws`, and the calls needed are few (open a tab, evaluate an
// expression in it, read its console, close it), so this file implements them
// directly.
//
// Not implemented: waiting for loads (the page signals readiness through a
// promise the harness awaits), screenshots, input and network interception.
// Use Playwright if those are needed.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import WebSocket from 'ws';

// Headless WebGPU on Linux needs Vulkan through ANGLE and the GPU sandbox
// disabled. `--headless=new` is the headless mode with WebGPU.
const FLAGS = [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-angle=vulkan',
    '--ignore-gpu-blocklist',
    // A hundred spectator tabs are a hundred renderers, and Chrome throttles
    // every tab that is not visible: timers are clamped to once a second,
    // which makes a peer waiting on a job look hung. All three flags are
    // needed. The first two cover occlusion and backgrounding, and the third
    // stops the renderer being frozen.
    '--disable-backgrounding-occluded-windows',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    // mDNS host candidates (.local) do not resolve reliably in a headless
    // container, and a peer whose ICE never completes never links. This
    // makes Chrome use real addresses.
    '--disable-features=WebRtcHideLocalIpsWithMdns',
    '--no-first-run',
    '--no-default-browser-check',
];

export async function launch({ port = 0, verbose = false } = {}) {
    const profile = mkdtempSync(join(tmpdir(), 'chattorrent-harness-'));
    const chrome = spawn('google-chrome', [
        ...FLAGS,
        `--remote-debugging-port=${port}`,
        `--user-data-dir=${profile}`,
        'about:blank',
    ], {
        stdio: ['ignore', 'pipe', 'pipe'],
        // Its own process group, so one signal reaches every renderer. A
        // hundred-spectator scenario is a hundred renderer processes plus a
        // GPU process and two zygotes. Killing only the parent would leave
        // them running, holding VRAM and a profile directory.
        detached: true,
    });

    // A run that never reaches its `finally` (Ctrl-C, an outer `timeout`, a
    // throw) must still kill that group. It is registered here because the
    // caller's cleanup may not run.
    const reap = () => {
        try { process.kill(-chrome.pid, 'SIGKILL'); } catch { /* already gone */ }
        try { rmSync(profile, { recursive: true, force: true }); } catch { /* ditto */ }
    };
    process.once('exit', reap);
    for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
        process.once(sig, () => { reap(); process.exit(130); });
    }

    let stderr = '';
    chrome.stderr.on('data', (d) => {
        stderr += d;
        if (verbose) process.stderr.write(d);
    });
    chrome.stdout.resume();

    // The port Chrome bound. With `--remote-debugging-port=0` Chrome picks
    // one, so several harness runs can share a machine.
    const url = await new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`chrome did not announce a debugging port in 20s:\n${stderr}`)),
            20000);
        const seen = () => {
            const m = /ws:\/\/[^\s]+/.exec(stderr);
            if (!m) return false;
            clearTimeout(timer);
            resolve(m[0]);
            return true;
        };
        chrome.stderr.on('data', seen);
        chrome.on('exit', (code) => {
            clearTimeout(timer);
            reject(new Error(`chrome exited with ${code} before listening:\n${stderr}`));
        });
        seen();
    });

    const browser = await connect(url);

    return {
        browser,
        // `context` is a browser context from context() below. Without one the
        // tab shares the profile's storage and cache with every other tab.
        async tab(pageUrl, { onConsole, context } = {}) {
            const { targetId } = await browser.send('Target.createTarget',
                { url: pageUrl, ...(context && { browserContextId: context }) });
            const tab = await attach(browser, targetId, onConsole);
            await tab.loaded(pageUrl);
            return tab;
        },
        // A browser context has its own cookies, storage, HTTP cache and
        // cached logins, like a separate visitor's browser. Tabs opened in it
        // share IndexedDB and BroadcastChannel with each other only.
        async context() {
            const { browserContextId } = await browser.send('Target.createBrowserContext',
                { disposeOnDetach: true });
            return browserContextId;
        },
        async close() {
            try { await browser.send('Browser.close'); } catch { /* already going */ }
            browser.ws.close();
            // Ask Chrome to close, wait briefly, then kill the whole group.
            // Chrome does not reliably honour Browser.close with a hundred
            // tabs open.
            await new Promise((r) => { chrome.once('exit', r); setTimeout(r, 3000); });
            reap();
        },
        get stderr() { return stderr; },
    };
}

// One WebSocket carries every session (`flatten: true`): each session's
// messages carry a sessionId, so no socket per session is needed.
async function connect(url) {
    const ws = new WebSocket(url, { maxPayload: 256 * 1024 * 1024 });
    await new Promise((res, rej) => {
        ws.once('open', res);
        ws.once('error', rej);
    });

    let next = 1;
    const waiting = new Map();          // id -> { resolve, reject }
    const listeners = new Set();        // (method, params, sessionId) -> void

    ws.on('message', (data) => {
        const m = JSON.parse(data);
        if (m.id !== undefined) {
            const w = waiting.get(m.id);
            if (!w) return;
            waiting.delete(m.id);
            if (m.error) w.reject(new Error(`${m.error.message} (${m.error.code})`));
            else w.resolve(m.result);
            return;
        }
        for (const fn of listeners) fn(m.method, m.params, m.sessionId);
    });
    ws.on('close', () => {
        for (const w of waiting.values()) w.reject(new Error('devtools socket closed'));
        waiting.clear();
    });

    return {
        ws,
        send(method, params = {}, sessionId) {
            const id = next++;
            return new Promise((resolve, reject) => {
                waiting.set(id, { resolve, reject });
                ws.send(JSON.stringify({ id, method, params, ...(sessionId && { sessionId }) }));
            });
        },
        on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    };
}

async function attach(browser, targetId, onConsole) {
    const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
    // The browser forgets a closed tab's session, and any call to it answers
    // `Session with given id not found`. Closing a tab is a test action
    // (killing a peer mid-generation), so callers can check `closed`.
    let closed = false;
    await browser.send('Runtime.enable', {}, sessionId);
    await browser.send('Log.enable', {}, sessionId);

    const off = browser.on((method, params, from) => {
        if (from !== sessionId || !onConsole) return;
        // Both channels are read. console.log arrives as
        // Runtime.consoleAPICalled. A failed fetch, a WebGPU validation error
        // and an uncaught exception arrive as Log.entryAdded, and are needed
        // to debug a peer that fails to start.
        if (method === 'Runtime.consoleAPICalled') {
            onConsole(params.type, params.args.map(preview).join(' '));
        } else if (method === 'Log.entryAdded') {
            onConsole(params.entry.level, params.entry.text);
        } else if (method === 'Runtime.exceptionThrown') {
            const d = params.exceptionDetails;
            onConsole('error', d.exception?.description ?? d.text);
        }
    });

    return {
        sessionId,
        targetId,
        get closed() { return closed; },

        // A new target answers Runtime.evaluate before it has navigated, from
        // the initial empty document. `navigator.gpu` is absent there, because
        // an opaque-origin about:blank is not a secure context. An evaluation
        // made too early does not fail: it reports that the browser has no
        // WebGPU. So every tab waits until it is the page that was requested.
        async loaded(pageUrl, { timeout = 30000 } = {}) {
            const until = Date.now() + timeout;
            const want = new URL(pageUrl, 'http://localhost').pathname;
            for (;;) {
                const at = await this.eval(
                    'JSON.stringify([location.pathname, document.readyState])');
                const [path, state] = JSON.parse(at);
                // 'complete', not only 'not loading'. readyState becomes
                // 'interactive' when parsing ends, before deferred and module
                // scripts run. A tab at that point has a DOM but none of its
                // code.
                if (path === want && state === 'complete') return;
                if (Date.now() > until) {
                    throw new Error(`${pageUrl} did not load in ${timeout / 1000}s ` +
                        `(at ${path}, ${state})`);
                }
                await new Promise((r) => setTimeout(r, 50));
            }
        },
        // The expression is awaited in the page: a returned promise is
        // resolved before this returns. Every page entry point is async.
        async eval(expression, { timeout = 600000 } = {}) {
            if (closed) throw new Error(`tab ${targetId} is closed; nothing to evaluate in`);
            const r = await Promise.race([
                browser.send('Runtime.evaluate', {
                    expression, awaitPromise: true, returnByValue: true,
                    // A page that throws fails the harness. It does not
                    // return undefined.
                    userGesture: true,
                }, sessionId),
                new Promise((_, rej) => setTimeout(
                    () => rej(new Error(`page eval timed out after ${timeout / 1000}s: ` +
                        `${expression.slice(0, 80)}`)), timeout)),
            ]);
            if (r.exceptionDetails) {
                const d = r.exceptionDetails;
                throw new Error(d.exception?.description ?? d.text);
            }
            return r.result.value;
        },
        // Poll an expression until it is true. The driver uses it to wait for
        // a page to report readiness, without a fixed sleep.
        async waitFor(expression, { timeout = 60000, every = 50 } = {}) {
            const until = Date.now() + timeout;
            for (;;) {
                if (await this.eval(`!!(${expression})`)) return;
                if (Date.now() > until) {
                    throw new Error(`waiting for \`${expression}\` timed out after ` +
                        `${timeout / 1000}s`);
                }
                await new Promise((r) => setTimeout(r, every));
            }
        },

        async close() {
            if (closed) return;
            closed = true;
            off();
            try { await browser.send('Target.closeTarget', { targetId }); } catch { /* gone */ }
        },
    };
}

// Console arguments arrive as RemoteObjects. Values are shown inline and
// everything else by its description, which is enough to read a stack trace or
// a class name from a log line without another round trip per argument.
const preview = (a) => a.value !== undefined ? String(a.value)
    : a.description ?? a.unserializableValue ?? `[${a.type}]`;
