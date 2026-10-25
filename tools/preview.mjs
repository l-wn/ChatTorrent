// Serves dist/ the way production's Caddy does (clean URLs, the CSP), for
// looking at the site pages locally. No signaling server, so the chat page
// loads but cannot join a cluster; use ./start.sh dev for that.
//
//   npm run preview            # builds dist/ first, then http://localhost:8090

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIST = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const PORT = Number(process.env.PORT) || 8090;

const TYPES = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
    '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8',
    '.webmanifest': 'application/manifest+json', '.png': 'image/png', '.jpg': 'image/jpeg',
};
const CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; " +
    "connect-src 'self' https://weights.chattorrent.net; manifest-src 'self'; object-src 'none'; " +
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

const isFile = async (p) => { try { return (await stat(p)).isFile(); } catch { return false; } };

createServer(async (req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (path.includes('..')) { res.writeHead(400).end(); return; }
    const tries = path === '/' ? ['/index.html'] : [path, `${path}.html`];
    let file = null;
    for (const t of tries) if (await isFile(join(DIST, t))) { file = join(DIST, t); break; }
    const status = file ? 200 : 404;
    file ??= join(DIST, '404.html');
    res.writeHead(status, {
        'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
        'Content-Security-Policy': CSP,
        'Cache-Control': 'no-store',
    });
    res.end(await readFile(file));
}).listen(PORT, '127.0.0.1', () => console.log(`preview: http://localhost:${PORT}`));
