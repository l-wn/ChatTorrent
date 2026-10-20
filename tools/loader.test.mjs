// Tests weight downloading against a real HTTP server in Node.
//
//   node tools/loader.test.mjs
//
// A download usually fails in the streamed body, not the request, so the retry
// must cover the body. The cases: a server that cuts a response short, one
// that ignores Range, one that fails intermittently, and one that goes silent,
// which needs a deadline because `fetch` then neither settles nor rejects.

import { createServer } from 'node:http';
import { loadFile } from '../web/model/loader.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};

// A deterministic body, large enough to span several 8 MiB chunks.
const SIZE = 20 * 1024 * 1024;
const BODY = Buffer.alloc(SIZE);
for (let i = 0; i < SIZE; i++) BODY[i] = (i * 2654435761) >>> 24;

// `behave` decides how each request is answered, so a test can make the
// server unreliable.
function serve(behave) {
    const server = createServer((req, res) => {
        const verdict = behave(req);
        if (verdict === 'silent') {
            // Correct headers, a few bytes, then nothing: no more data, no
            // FIN, no RST. The status is always 200 because nothing reads past
            // the stall. The socket is left open; closeAllConnections() ends it.
            res.writeHead(200, { 'content-length': String(SIZE) });
            res.write(BODY.subarray(0, 1024));
            return;
        }
        if (verdict === 'deaf') {
            // No headers either: the request is accepted and never answered.
            return;
        }
        if (verdict === 'cut') {
            // Correct headers, a short body, then the connection is dropped.
            res.writeHead(206, { 'content-range': `bytes 0-${SIZE - 1}/${SIZE}`,
                                 'content-length': String(SIZE) });
            res.write(BODY.subarray(0, 1024));
            res.socket.destroy();
            return;
        }
        const range = verdict === 'ignore-range' ? null : req.headers.range;
        if (!range) {
            res.writeHead(200, { 'content-length': String(SIZE) });
            res.end(BODY);
            return;
        }
        const [from, to] = range.replace('bytes=', '').split('-').map(Number);
        const end = Math.min(to, SIZE - 1);
        const slice = BODY.subarray(from, end + 1);
        res.writeHead(206, {
            'content-range': `bytes ${from}-${end}/${SIZE}`,
            'content-length': String(slice.length),
        });
        res.end(slice);
    });
    return new Promise((r) => server.listen(0, '127.0.0.1',
        () => r({ server, url: `http://127.0.0.1:${server.address().port}` })));
}

const identical = (buf) => Buffer.from(buf).equals(BODY);

console.log('\nloadFile: the happy path is one cacheable request');
{
    let requests = 0, ranged = 0;
    const { server, url } = await serve((req) => {
        requests++;
        if (req.headers.range) ranged++;
        return 'ok';
    });
    const buf = await loadFile(url, 'w.bin');
    ok('the whole file arrives byte-for-byte', identical(buf));
    ok('in a single request', requests === 1, `made ${requests}`);
    // The browser and Cloudflare's edge cache a 200 but do not cache a 206 the
    // same way, so the normal path must not send Range.
    ok('with no Range header, so the response is cacheable', ranged === 0,
        `${ranged} ranged requests`);
    server.close();
}

console.log('\nloadFile: a server that ignores Range');
{
    const { server, url } = await serve(() => 'ignore-range');
    const buf = await loadFile(url, 'w.bin');
    ok('falls back to the whole body it was sent', identical(buf));
    server.close();
}

console.log('\nloadFile: a body that dies mid-stream (the QUIC failure)');
{
    // Fails the first two attempts of every range, then succeeds. This only
    // completes if the retry covers the body.
    const seen = new Map();
    const { server, url } = await serve((req) => {
        const k = req.headers.range ?? 'all';
        const n = (seen.get(k) ?? 0) + 1;
        seen.set(k, n);
        return n <= 2 ? 'cut' : 'ok';
    });
    const buf = await loadFile(url, 'w.bin');
    ok('falls back to ranges and still reassembles correctly', identical(buf));
    ok('and it really did fail first', [...seen.values()].some((n) => n > 2));
    server.close();
}

console.log('\nloadFile: a failure that never clears');
{
    const { server, url } = await serve(() => 'cut');
    let threw = null;
    try { await loadFile(url, 'w.bin'); } catch (e) { threw = e; }
    ok('gives up rather than looping forever', threw !== null,
        `got ${threw}`);
    server.close();
}

console.log('\nloadFile: a 404 is not retried');
{
    const server = createServer((_, res) => { res.writeHead(404); res.end(); });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    const url = `http://127.0.0.1:${server.address().port}`;
    const t0 = Date.now();
    let threw = null;
    try { await loadFile(url, 'missing.bin'); } catch (e) { threw = e; }
    ok('a 404 throws', threw !== null);
    ok('and does so immediately, without backing off', Date.now() - t0 < 700,
        `took ${Date.now() - t0}ms`);
    server.close();
}

// A connection that goes silent mid-body leaves `fetch` neither settling nor
// rejecting, so without a deadline the whole-file GET never throws and the
// resumable ranged path in its `catch` is never reached.
console.log('\nloadFile: a connection that goes silent');
{
    // Silent on the whole-file GET, answering ranges normally.
    let silenced = 0;
    const { server, url } = await serve((req) => {
        if (req.headers.range) return 'ok';
        silenced++;
        return 'silent';
    });
    const t0 = Date.now();
    const buf = await loadFile(url, 'w.bin', undefined, { stallMs: 150 });
    ok('the stalled whole-file GET is abandoned rather than awaited forever',
        Date.now() - t0 < 5000, `took ${Date.now() - t0}ms`);
    ok('and it really did go silent first', silenced > 0);
    ok('the ranged path then reassembles the file byte-for-byte', identical(buf));
    server.closeAllConnections();
    server.close();
}
{
    // Every range goes silent on its first attempt and answers on its second,
    // so this finishes only if the deadline covers the ranged path too: an
    // 8 MiB chunk stalls the same way a whole file does.
    const seen = new Map();
    const { server, url } = await serve((req) => {
        const n = (seen.get(req.headers.range) ?? 0) + 1;
        seen.set(req.headers.range, n);
        return n === 1 ? 'silent' : 'ok';
    });
    const buf = await loadFile(url, 'w.bin', undefined,
        { bytes: 100 * 1024 * 1024, stallMs: 150 });
    ok('a stalled range aborts, retries and resumes', identical(buf));
    ok('and every range really did stall once',
        seen.size === 3 && [...seen.values()].every((n) => n === 2), `${[...seen]}`);
    server.closeAllConnections();
    server.close();
}
{
    // A timeout must not be marked permanent: loadFile rethrows a permanent
    // error without falling back, so the download would be abandoned instead of
    // retried. The retries happening at all, and the error being the timeout
    // rather than a 4xx, is what shows it is not marked permanent.
    const { server, url } = await serve(() => 'silent');
    let threw = null;
    try { await loadFile(url, 'w.bin', undefined, { stallMs: 100, retries: 1 }); }
    catch (e) { threw = e; }
    ok('a server silent on everything gives up rather than hanging', threw !== null);
    ok('and says why', /no data for 100ms/.test(threw?.message ?? ''), `${threw?.message}`);
    server.closeAllConnections();
    server.close();
}
{
    // The deadline covers the headers too: a static file whose response has
    // not started is as dead as one that stops mid-body.
    const { server, url } = await serve(() => 'deaf');
    let threw = null;
    try { await loadFile(url, 'w.bin', undefined, { stallMs: 100, retries: 0 }); }
    catch (e) { threw = e; }
    ok('a request that is never answered at all also times out',
        /no data for 100ms/.test(threw?.message ?? ''), `${threw?.message}`);
    server.closeAllConnections();
    server.close();
}

// A whole-file GET cannot resume, so a stall on one discards every byte it had
// downloaded. Above loader.js's WHOLE_MAX the ranged path is taken from the
// start, and the browser cache the whole-file path exists for does not reliably
// store a body that big.
console.log('\nloadFile: a file too big to risk a non-resumable attempt on');
{
    let bare = 0, ranged = 0;
    const { server, url } = await serve((req) => {
        if (req.headers.range) ranged++; else bare++;
        return 'ok';
    });
    const buf = await loadFile(url, 'w.bin', undefined, { bytes: 100 * 1024 * 1024 });
    ok('is fetched correctly', identical(buf));
    ok('without one request that could not have resumed', bare === 0, `${bare} bare GETs`);
    ok('in 8 MiB ranges', ranged === 3, `${ranged} ranged requests`);
    server.close();
}
{
    // Below that threshold, and when the manifest gives no size, the file is
    // taken in one cacheable 200.
    for (const opts of [{ bytes: SIZE }, {}, undefined]) {
        let requests = 0, ranged = 0;
        const { server, url } = await serve((req) => {
            requests++;
            if (req.headers.range) ranged++;
            return 'ok';
        });
        const buf = await loadFile(url, 'w.bin', undefined, opts);
        ok(`bytes=${opts?.bytes ?? 'unknown'} still takes one cacheable request`,
            identical(buf) && requests === 1 && ranged === 0,
            `${requests} requests, ${ranged} ranged`);
        server.close();
    }
}

// The whole-file path reads its body chunk by chunk, so it drives the progress
// callback as the ranged path does.
console.log('\nloadFile: progress while the whole file streams in');
{
    const { server, url } = await serve(() => 'ok');
    const seen = [];
    const buf = await loadFile(url, 'w.bin', (b, t) => seen.push([b, t]));
    ok('the whole-file GET reports progress too', seen.length > 0, `${seen.length} callbacks`);
    ok('against the real total, ending at the file length',
        seen.at(-1)?.[0] === SIZE && seen.every(([, t]) => t === SIZE), `${seen.at(-1)}`);
    ok('never past it, and the buffer is still exact',
        seen.every(([b]) => b <= SIZE) && identical(buf));
    server.close();
}

// The two ends of the model. An untied checkpoint has two vocabulary-sized
// tables, and each end reads one: position 0 the embedding, the last position
// the LM head. Each end must upload only its own table, and a peer that grows
// into its second end must not upload a table again.
console.log('\nloadEmbed / loadHead: each end uploads only the table it reads');
{
    const { loadEmbed, loadHead, loadClient } = await import('../web/model/loader.js');
    const V = 8, H = 4;
    const d = { vocab: V, hidden: H, tied_lm_head: false };
    const f32 = (n, at) => Buffer.from(Float32Array.from({ length: n }, (_, i) => at + i).buffer);
    const bins = { 'client.bin': Buffer.concat([f32(V * H, 100), f32(H, 900)]),
                   'lm_head.bin': f32(V * H, 500) };
    const manifest = { model: 'ends', dtype: 'f32', files: {
        'client.bin': [{ name: 'wte', shape: [V, H], offset: 0, size: V * H * 4 },
                       { name: 'norm', shape: [H], offset: V * H * 4, size: H * 4 }],
        'lm_head.bin': [{ name: 'lm_head', shape: [V, H], offset: 0, size: V * H * 4 }] } };

    // A stub device that keeps the buffers it is given and counts live ones.
    const gpu = () => {
        let live = 0;
        const device = {
            features: new Set(),
            pushErrorScope() {}, popErrorScope: async () => null,
            createBuffer: ({ size }) => {
                live += size;
                return { size, destroy() { live -= size; } };
            },
            queue: { writeBuffer(buf, _, data) { buf.data = Array.from(data); } },
        };
        return { device, live: () => live };
    };
    const serveBins = (ranges = true) => {
        const got = [];
        const server = createServer((req, res) => {
            const body = bins[req.url.slice(1)];
            const range = ranges ? req.headers.range : null;
            got.push(req.url.slice(1) + (req.headers.range ? ' ' + req.headers.range : ''));
            if (!range) { res.writeHead(200); res.end(body); return; }
            const [from, to] = range.replace('bytes=', '').split('-').map(Number);
            res.writeHead(206, { 'content-range': `bytes ${from}-${to}/${body.length}` });
            res.end(body.subarray(from, to + 1));
        });
        return new Promise((r) => server.listen(0, '127.0.0.1',
            () => r({ server, got, baseUrl: `http://127.0.0.1:${server.address().port}` })));
    };
    const TABLE = V * H * 4, NORM = Math.max(H * 4, 16);

    {
        const { server, got, baseUrl } = await serveBins();
        const g = gpu();
        const e = await loadEmbed(g.device, { baseUrl, manifest, d });
        ok('the embedding is wte and the norm that shares its file', e.wte && e.norm && !e.lm_head);
        ok('fetched from client.bin alone', got.join() === 'client.bin', `${got}`);
        ok('and it reports the bytes it put on the GPU',
            e.bytes === TABLE + H * 4 && g.live() === TABLE + NORM, `${e.bytes}, live ${g.live()}`);

        // The same peer growing into the head: the norm is already loaded.
        got.length = 0;
        const h = await loadHead(g.device, { baseUrl, manifest, d, have: e });
        ok('a peer holding the embedding fetches only lm_head.bin for the head',
            got.join() === 'lm_head.bin', `${got}`);
        ok('and shares the norm it has', h.norm === e.norm && h.bytes === TABLE);
        ok('so both ends are two tables, not four', g.live() === 2 * TABLE + NORM, `${g.live()}`);
        server.close();
    }
    {
        const { server, got, baseUrl } = await serveBins();
        const g = gpu();
        const h = await loadHead(g.device, { baseUrl, manifest, d });
        ok('a head-only peer has the norm and the LM head, and no embedding',
            h.norm && h.lm_head && !h.wte);
        ok('the norm comes as a range, not as the table beside it',
            got[0] === `client.bin bytes=${TABLE}-${TABLE + H * 4 - 1}`, `${got}`);
        ok('and is the right four floats', h.norm.data.join() === '900,901,902,903', `${h.norm.data}`);
        ok('one table on the GPU', g.live() === TABLE + NORM && h.bytes === TABLE + H * 4);

        // Growing down into the embedding later adds wte and no second norm.
        const e = await loadEmbed(g.device, { baseUrl, manifest, d, have: h });
        ok('growing into the embedding adds only wte', e.wte && !e.norm && e.bytes === TABLE);
        server.close();
    }
    {
        const { server, baseUrl } = await serveBins(false);
        const g = gpu();
        const h = await loadHead(g.device, { baseUrl, manifest, d });
        ok('a server that ignores Range still yields the right norm',
            h.norm.data.join() === '900,901,902,903' && g.live() === TABLE + NORM, `${h.norm.data}`);
        server.close();
    }
    {
        const { server, got, baseUrl } = await serveBins();
        const g = gpu();
        const tied = { ...d, tied_lm_head: true };
        const e = await loadEmbed(g.device, { baseUrl, manifest, d: tied });
        got.length = 0;
        const h = await loadHead(g.device, { baseUrl, manifest, d: tied, have: e });
        ok('a tied head is the embedding it already holds: nothing fetched, nothing uploaded',
            got.length === 0 && h.bytes === 0 && h.wte === e.wte && g.live() === TABLE + NORM);
        const c = await loadClient(gpu().device, { baseUrl, manifest, d });
        ok('loadClient still loads everything, and says how much',
            c.wte && c.norm && c.lm_head && c.bytes === 2 * TABLE + H * 4, `${c.bytes}`);
        server.close();
    }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
