// One origin for the harness: the repo's files, and the signaling server's /ws
// on the same host and port.
//
// web/net.js opens `ws://${location.host}/ws`, so the page and the signaling
// server share an origin. In the real deploy Caddy provides that (see the
// Caddyfile's reverse_proxy). The harness needs the same without Caddy, basic
// auth or a fixed port, so this is a static file server over the repo root
// plus a raw socket proxy for the /ws upgrade.
//
// The proxy does not parse WebSocket frames. After the 101 a WebSocket is two
// byte streams, so the proxy forwards the client's handshake to the signaling
// server unchanged, returns its answer, and pipes the two sockets.

import { createServer } from 'node:http';
import { connect as tcp } from 'node:net';
import { createReadStream, statSync } from 'node:fs';
import { resolve, normalize, extname, join } from 'node:path';

const TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'text/javascript; charset=utf-8',
    '.mjs':  'text/javascript; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.bin':  'application/octet-stream',
    '.npy':  'application/octet-stream',
    '.wasm': 'application/wasm',
};

// `signalPort` is the port signal-server.js listens on. Every /ws upgrade is
// proxied to it. `onReport` receives whatever a page POSTs to /report, so a
// hundred tabs can return results without a hundred CDP round trips.
export function serve({ root, signalPort, onReport } = {}) {
    const base = resolve(root);

    const server = createServer((req, res) => {
        const url = new URL(req.url, 'http://localhost');

        if (req.method === 'POST' && url.pathname === '/report') {
            const chunks = [];
            req.on('data', (c) => chunks.push(c));
            req.on('end', () => {
                let body = null;
                try { body = JSON.parse(Buffer.concat(chunks).toString()); } catch { /* ignore */ }
                onReport?.(body);
                res.writeHead(204).end();
            });
            return;
        }

        // Every tab requests this and the repo has none. Answering 204 keeps
        // a 404 per peer out of the report.
        if (url.pathname === '/favicon.ico') return res.writeHead(204).end();

        // Serves only files under the repo root: a path that normalizes
        // outside `base` gets a 403. The working tree also contains .env and
        // .git.
        const path = normalize(join(base, decodeURIComponent(url.pathname)));
        if (!path.startsWith(base)) return res.writeHead(403).end('outside the root');
        if (/(^|\/)\.(git|env|venv)/.test(path.slice(base.length))) {
            return res.writeHead(403).end('not served');
        }

        let stat;
        try {
            stat = statSync(path);
        } catch {
            return res.writeHead(404).end(`${url.pathname} not found`);
        }
        if (stat.isDirectory()) return res.writeHead(403).end('no directory listing');

        // Range requests are supported because loader.js's loadFile() falls
        // back to them, and answering 200 would leave that path untested.
        // Single ranges only, which is all it requests.
        const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
        const type = TYPES[extname(path)] ?? 'application/octet-stream';
        const head = { 'content-type': type, 'accept-ranges': 'bytes',
                       'cache-control': 'no-store' };

        if (range) {
            const from = range[1] ? Number(range[1]) : 0;
            const to = Math.min(range[2] ? Number(range[2]) : stat.size - 1, stat.size - 1);
            if (from > to) {
                return res.writeHead(416, { 'content-range': `bytes */${stat.size}` }).end();
            }
            res.writeHead(206, { ...head, 'content-length': to - from + 1,
                                 'content-range': `bytes ${from}-${to}/${stat.size}` });
            return createReadStream(path, { start: from, end: to }).pipe(res);
        }

        res.writeHead(200, { ...head, 'content-length': stat.size });
        if (req.method === 'HEAD') return res.end();
        return createReadStream(path).pipe(res);
    });

    // The proxy. `head` is whatever arrived in the same TCP segment as the
    // upgrade request. It must be forwarded too, or a fast client's first
    // frame would be lost.
    server.on('upgrade', (req, socket, head) => {
        if (!req.url.startsWith('/ws')) return socket.destroy();

        const up = tcp({ host: '127.0.0.1', port: signalPort }, () => {
            const lines = [`GET ${req.url} HTTP/1.1`];
            for (let i = 0; i < req.rawHeaders.length; i += 2) {
                lines.push(`${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}`);
            }
            up.write(`${lines.join('\r\n')}\r\n\r\n`);
            if (head?.length) up.write(head);
            up.pipe(socket);
            socket.pipe(up);
        });
        const drop = () => { socket.destroy(); up.destroy(); };
        up.on('error', drop);
        socket.on('error', drop);
    });

    return new Promise((res) => {
        server.listen(0, '127.0.0.1', () => res({
            port: server.address().port,
            close: () => new Promise((done) => { server.closeAllConnections?.(); server.close(done); }),
        }));
    });
}
