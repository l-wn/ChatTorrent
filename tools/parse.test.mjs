// Every browser module must parse and evaluate under Node.
//
// The kernels are WGSL inside JS template literals. A backtick in a WGSL
// comment ends the literal, the rest of the shader is parsed as JavaScript,
// and the error is reported far from its cause. The other suites stub the
// device and do not import kernels.js, so they do not catch this, and in a
// browser it shows as a blank page.
//
// Modules that use `document` or `navigator.gpu` at import time are excluded.
// Everything else is imported, so a bad export or a circular import fails
// here too.
import { readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = new URL('../web/', import.meta.url).pathname;

// These use the DOM as soon as they are evaluated.
const BROWSER_ONLY = new Set(['dev/main.js', 'chat.js', 'peer.js', 'peer-start.js']);

function walk(dir) {
    return readdirSync(dir).flatMap((name) => {
        const full = join(dir, name);
        return statSync(full).isDirectory() ? walk(full)
             : name.endsWith('.js') ? [full] : [];
    });
}

let pass = 0, fail = 0;
for (const file of walk(ROOT).sort()) {
    const rel = relative(ROOT, file);
    if (BROWSER_ONLY.has(rel)) continue;
    try {
        await import(file);
        console.log(`  ok    ${rel}`);
        pass++;
    } catch (e) {
        console.log(`  FAIL  ${rel}: ${e.message}`);
        fail++;
    }
}

// An import catches a syntax error, but a stray backtick can end a WGSL
// template early and leave valid JavaScript. The shader is then truncated, or
// the next kernel's source is appended to it, and nothing fails until a GPU
// rejects it.
//
// So the shaders are checked directly. Every `/* wgsl */` marker opens a
// template, and the text up to the next unescaped backtick is one shader. A
// shader without `fn main` was cut short by a misplaced backtick.
const src = await (await import('node:fs')).promises
    .readFile(new URL('../web/model/kernels.js', import.meta.url), 'utf8');

let shaders = 0, truncated = [];
for (let i = src.indexOf('/* wgsl */'); i !== -1; i = src.indexOf('/* wgsl */', i + 1)) {
    const open = src.indexOf('`', i);
    if (open === -1) continue;

    // Walk to the closing backtick, skipping ${...} interpolations. Some
    // kernels select a whole binding block with a nested template, and those
    // inner backticks do not end the shader.
    let end = open + 1, depth = 0;
    while (end < src.length) {
        const c = src[end];
        if (c === '\\') { end += 2; continue; }
        if (c === '$' && src[end + 1] === '{') { depth++; end += 2; continue; }
        if (c === '}' && depth > 0) { depth--; end++; continue; }
        if (c === '`' && depth === 0) break;
        end++;
    }
    const body = src.slice(open + 1, end);
    shaders++;
    if (!body.includes('fn main')) {
        truncated.push(`  ...near: ${src.slice(i - 60, i + 40).split('\n').pop()}`);
    }
}
if (shaders && !truncated.length) {
    console.log(`  ok    ${shaders} WGSL templates are intact`); pass++;
} else {
    console.log(`  FAIL  ${truncated.length}/${shaders} WGSL templates end early `
        + `a backtick inside a shader comment?\n${truncated.join('\n')}`);
    fail++;
}

const ticks = (src.match(/`/g) ?? []).length;
if (ticks % 2 === 0) { console.log(`  ok    kernels.js backticks balanced (${ticks})`); pass++; }
else { console.log(`  FAIL  kernels.js has ${ticks} backticks`); fail++; }

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
