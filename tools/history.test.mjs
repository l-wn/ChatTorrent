// Tests web/history.js: saved chats, scoped by mesh, synced across tabs,
// exported.
//
// Uses memoryBackend(). The IndexedDB backend has the same interface over a
// browser API Node lacks; web/dev/history.html runs this file's scoping checks
// on it in a browser.
//
//   node tools/history.test.mjs

import { webcrypto } from 'node:crypto';
globalThis.crypto ??= webcrypto;

import { openHistory, memoryBackend, newRecord, titleFrom, groupByDay, TITLE_MAX,
         exportChats, importChats, idsToBase64, idsFromBase64 } from '../web/history.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
const rejects = async (p) => p.then(() => null, (e) => e);

const PRIV = 'a1b2c3d4e5f60718293a4b5c6d7e8f90';
const chat = (mesh, text = 'hello', ids = [1, 2, 3]) => ({
    ...newRecord({ mesh, model: 'Qwen/Qwen3.8-27B', weights: 'https://w.example/q8' }),
    messages: [{ role: 'user', text, reasoning: null, meta: '', opts: { chat: true, thinking: false } },
               { role: 'assistant', text: 'hi', reasoning: 'hmm', meta: '3 tokens', opts: {} }],
    ids: Uint32Array.from(ids),
});
const quiet = { persist: () => {} };

console.log('\na store sees only its own mesh');
{
    const backend = memoryBackend();
    const global = openHistory({ mesh: 'global', backend, ...quiet });
    const priv = openHistory({ mesh: PRIV, backend, ...quiet });

    const g1 = await global.save(chat('global', 'one'));
    await global.save(chat('global', 'two'));
    const p1 = await priv.save(chat(PRIV, 'secret plans'));

    eq('the global list never shows a private chat',
        (await global.list()).map((r) => r.title).sort(), ['one', 'two']);
    eq('nor the private list a global one', (await priv.list()).map((r) => r.title), ['secret plans']);
    eq('reading another mesh\'s chat by id finds nothing', await global.get(p1.id), null);
    ok('the same as reading one that does not exist', (await global.get('nope')) === null);
    ok('saving into another mesh is refused',
        /another mesh/.test((await rejects(global.save(chat(PRIV))))?.message ?? ''));
    eq('as is deleting across meshes', await priv.remove(g1.id), false);
    ok('which left it there', (await global.get(g1.id)) !== null);
    eq('each side counts the other\'s chats, and only counts them',
        [await priv.countElsewhere(), await global.countElsewhere()], [2, 1]);

    eq('forgetting a mesh\'s chats forgets exactly those', await priv.forgetAll(), 1);
    eq('and leaves the other mesh alone', (await global.list()).length, 2);
    ok('a store with no mesh cannot be opened', (() => {
        try { openHistory({ backend }); return false; } catch { return true; }
    })());
}

console.log('\nrecords round-trip');
{
    const store = openHistory({ mesh: 'global', backend: memoryBackend(), ...quiet });
    const rec = chat('global', 'round trip', [151644, 872, 198, 4294967295]);
    await store.save(rec);
    const back = await store.get(rec.id);
    ok('ids come back as a Uint32Array', back.ids instanceof Uint32Array);
    eq('with every id intact, the largest included', [...back.ids], [151644, 872, 198, 4294967295]);
    eq('and the messages unchanged', back.messages, rec.messages);
    eq('the model and checkpoint are kept', [back.model, back.weights], [rec.model, rec.weights]);

    back.ids[0] = 0;
    ok('what get() returns is a copy', (await store.get(rec.id)).ids[0] === 151644);

    const rev = rec.rev;
    await store.save(rec);
    eq('every save bumps rev', rec.rev, rev + 1);
    await store.rename(rec.id, '  a   new\nname ');
    eq('a rename collapses whitespace', (await store.get(rec.id)).title, 'a new name');
}

console.log('\ntitles and days');
{
    eq('a title is the first user message', titleFrom('What is a KV cache?'), 'What is a KV cache?');
    eq('on one line', titleFrom('two\n\nlines\tand tabs'), 'two lines and tabs');
    const long = titleFrom('x'.repeat(200));
    ok(`cut to ${TITLE_MAX} characters, with an ellipsis`, long.length === TITLE_MAX && long.endsWith('…'));
    eq('and something for an empty one', titleFrom('   '), 'New chat');

    const store = openHistory({ mesh: 'global', backend: memoryBackend(), ...quiet });
    const r = chat('global', 'first message\nof the chat');
    await store.save(r);
    eq('saving an untitled chat titles it from its first user message', r.title, 'first message of the chat');

    // Local time, since the groups are by local day.
    const now = new Date(2026, 8, 28, 0, 30).getTime();     // just past midnight
    const at = (d, h, m = 0) => ({ id: `${d}-${h}`, updated: new Date(2026, 8, d, h, m).getTime() });
    const groups = groupByDay([at(27, 23, 59), at(28, 0, 0), at(26, 23, 59), at(27, 0, 0), at(1, 12)], now);
    eq('grouped at local midnight, newest first, empty groups left out',
        groups.map((g) => [g.label, g.chats.map((c) => c.id)]),
        [['Today', ['28-0']], ['Yesterday', ['27-23', '27-0']], ['Earlier', ['26-23', '1-12']]]);
}

console.log('\nother tabs hear about changes in their own mesh');
{
    // Two tabs' channels, connected as BroadcastChannel connects them: a post
    // reaches every other instance and not the sender.
    const bus = new Set();
    const channel = () => {
        const fns = new Set();
        const ch = {
            postMessage(data) { for (const o of bus) if (o !== ch) for (const fn of o.fns) fn({ data }); },
            addEventListener(_, fn) { fns.add(fn); },
            removeEventListener(_, fn) { fns.delete(fn); },
            fns,
        };
        bus.add(ch);
        return ch;
    };
    const backend = memoryBackend();
    const tabA = openHistory({ mesh: 'global', backend, channel: channel(), ...quiet });
    const tabB = openHistory({ mesh: 'global', backend, channel: channel(), ...quiet });
    const tabP = openHistory({ mesh: PRIV, backend, channel: channel(), ...quiet });
    const heard = { A: [], B: [], P: [] };
    tabA.onChange((m) => heard.A.push(m.op));
    tabB.onChange((m) => heard.B.push(m.op));
    tabP.onChange((m) => heard.P.push(m.op));

    const r = await tabA.save(chat('global'));
    await tabA.rename(r.id, 'renamed');
    await tabA.remove(r.id);
    eq('the other tab in the mesh hears a save, a rename and a delete', heard.B, ['put', 'put', 'delete']);
    eq('the tab that made them does not hear its own', heard.A, []);
    eq('and a tab in another mesh hears nothing', heard.P, []);

    let persisted = 0;
    const once = openHistory({ mesh: 'global', backend, persist: () => { persisted++; } });
    await once.save(chat('global'));
    await once.save(chat('global'));
    eq('durable storage is asked for once, on the first save', persisted, 1);
}

console.log('\nexport and import');
{
    const ids = Uint32Array.from([0, 1, 255, 256, 65536, 4294967295]);
    eq('ids survive base64 as little-endian u32s', [...idsFromBase64(idsToBase64(ids))], [...ids]);
    eq('as little-endian on any machine', idsToBase64(Uint32Array.from([1])), 'AQAAAA==');

    const store = openHistory({ mesh: 'global', backend: memoryBackend(), ...quiet });
    const a = await store.save(chat('global', 'exported', [7, 8, 9]));
    const json = JSON.parse(JSON.stringify(exportChats([a])));
    eq('an export names its format and version', [json.format, json.v], ['chattorrent-chats', 1]);

    const imported = importChats(json, PRIV);
    const back = imported[0];
    eq('importing puts the chat in the mesh it was imported into', back.mesh, PRIV);
    ok('under a new id, so importing twice cannot overwrite', back.id !== a.id);
    eq('with its ids, messages and title intact',
        [[...back.ids], back.messages, back.title], [[7, 8, 9], a.messages, a.title]);
    ok('and a non-export is refused', (() => {
        try { importChats({ chats: [] }, 'global'); return false; } catch { return true; }
    })());
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
