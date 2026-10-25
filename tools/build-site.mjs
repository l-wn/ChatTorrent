// Builds dist/, the only directory the public deploy serves.
//
// It is an allowlist: a file reaches dist/ only if a page references it. For
// each page, the <script src> and <link href> files it references are copied,
// then every module they import, recursively. This includes
// web/dev/kernels.test.js and web/dev/smoke.js (engine.js runs them as its
// GPU self-checks) and excludes dev.html, the benchmarks, tools/, oracle/ and
// the repo's dotfiles.
//
//   node tools/build-site.mjs        # or: npm run build
//
// dist/ has the same layout as web/, so the pages' relative imports resolve
// unchanged:
//
//   chat.html            web/peer.html, served at /chat
//   create.html          web/create.html, served at /create
//   *.js, model/, dev/   the modules chat.html imports
//   checkpoint.json      which checkpoint to load
//   everything in site/  404.html, favicon, robots.txt, site.css, and any
//                        other page added there

import { readFileSync, mkdirSync, rmSync, readdirSync, statSync, copyFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WEB = join(ROOT, 'web');
const SITE = join(ROOT, 'site');
const DIST = join(ROOT, 'dist');

// Published path (in dist/) -> source file. Pages from web/ are listed here. A
// page added to site/ needs no entry.
const PAGES = { 'chat.html': join(WEB, 'peer.html'), 'create.html': join(WEB, 'create.html') };

// Never copied to dist/, even if imported. Guards against a stray import of a
// file that must stay private.
const FORBIDDEN = [/(^|\/)\./, /(^|\/)dev\.html$/, /(^|\/)harness\./, /\.(bin|npy|py|mjs)$/];

const out = new Map();     // dist-relative path -> source file

function add(published, source) {
    if (FORBIDDEN.some((re) => re.test(published))) {
        throw new Error(`refusing to publish ${published} (from ${relative(ROOT, source)})`);
    }
    if (out.has(published) && out.get(published) !== source) {
        throw new Error(`${published} would come from both ${relative(ROOT, out.get(published))} ` +
            `and ${relative(ROOT, source)}`);
    }
    out.set(published, source);
}

// A web/ file's path inside dist/.
function underWeb(file) {
    const rel = relative(WEB, file);
    if (rel.startsWith('..')) throw new Error(`${relative(ROOT, file)} is outside web/`);
    return rel.split(sep).join('/');
}

// Relative specifiers only: the app has no bare imports and no bundler.
const IMPORT_RE = /\b(?:import|export)\s[^'"`]*?from\s*['"](\.{1,2}\/[^'"]+)['"]|\bimport\s*\(?\s*['"](\.{1,2}\/[^'"]+)['"]/g;

function crawlModule(file) {
    const published = underWeb(file);
    if (out.has(published)) return;
    add(published, file);
    const src = readFileSync(file, 'utf8');
    for (const m of src.matchAll(IMPORT_RE)) crawlModule(resolve(dirname(file), m[1] ?? m[2]));
}

// A page's local scripts and stylesheets. Absolute paths (/favicon.svg) are
// served from site/, and URLs with a scheme are external.
const REF_RE = /<(?:script|link)\b[^>]*?\b(?:src|href)="(\.{1,2}\/[^"]+)"/g;

for (const [published, source] of Object.entries(PAGES)) {
    add(published, source);
    for (const m of readFileSync(source, 'utf8').matchAll(REF_RE)) {
        const file = resolve(dirname(source), m[1]);
        if (file.endsWith('.js')) crawlModule(file);
        else add(underWeb(file), file);
    }
}

add('checkpoint.json', join(ROOT, 'checkpoint.json'));

(function walk(dir) {
    for (const name of readdirSync(dir)) {
        const file = join(dir, name);
        if (statSync(file).isDirectory()) walk(file);
        else add(relative(SITE, file).split(sep).join('/'), file);
    }
})(SITE);

rmSync(DIST, { recursive: true, force: true });
for (const [published, source] of [...out].sort()) {
    const dest = join(DIST, published);
    mkdirSync(dirname(dest), { recursive: true });
    copyFileSync(source, dest);
}

console.log(`dist/: ${out.size} files`);
for (const p of [...out.keys()].sort()) console.log(`  ${p}`);
