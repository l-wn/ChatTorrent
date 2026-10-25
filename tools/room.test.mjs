// Tests web/room.js: invite links, key derivation, and the seal on every offer
// and answer, which limits a private mesh's links to holders of its invite.
//
//   node tools/room.test.mjs

// In a Node 18 module, WebCrypto is not a global as it is in a browser.
import { webcrypto } from 'node:crypto';
globalThis.crypto ??= webcrypto;

import { newSecret, inviteUrl, readInvite, inviteFrom, cleanName, NAME_MAX, openRoom, roomKey,
         GLOBAL, SECRET_BYTES, unb64url } from '../web/room.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
    if (cond) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name} ${detail}`); }
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want),
    `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
const throws = (name, fn) => {
    try { fn(); ok(name, false, 'did not throw'); } catch { ok(name, true); }
};

console.log('\nsecrets and invite links');
{
    const s = newSecret();
    ok('a secret is 22 base64url characters', /^[A-Za-z0-9_-]{22}$/.test(s), `${s}`);
    eq('and 16 bytes of entropy', unb64url(s).length, SECRET_BYTES);
    const many = new Set(Array.from({ length: 1000 }, newSecret));
    eq('1000 secrets never repeat', many.size, 1000);

    const url = inviteUrl(s, { href: 'https://chat.example/web/peer.html?blocks=4#old' });
    eq('the invite is the page with only the fragment set', url,
        `https://chat.example/web/peer.html#mesh=${s}`);
    ok('and the secret is only in the fragment, which is never sent',
        new URL(url).pathname + new URL(url).search === '/web/peer.html');
    eq('and reads back as the same secret', readInvite(new URL(url).hash), s);
}

console.log('\nreadInvite: fails closed');
{
    eq('no fragment is the global mesh', readInvite(''), null);
    eq('a bare "#" is the global mesh', readInvite('#'), null);
    eq('undefined is the global mesh', readInvite(undefined), null);
    const s = newSecret();
    eq('a valid invite, with the #', readInvite(`#mesh=${s}`), s);
    eq('a valid invite, without it', readInvite(`mesh=${s}`), s);
    throws('an empty code throws', () => readInvite('#mesh='));
    throws('a truncated code throws', () => readInvite(`#mesh=${s.slice(0, 21)}`));
    throws('an over-long code throws', () => readInvite(`#mesh=${s}x`));
    throws('a character outside base64url throws', () => readInvite(`#mesh=${s.slice(0, 21)}+`));
    throws('trailing junk throws', () => readInvite(`#mesh=${s}&x=1`));
    throws('a mistyped key throws rather than landing in global', () => readInvite(`#Mesh=${s}`));
    throws('any other fragment throws rather than landing in global', () => readInvite('#hello'));
}

console.log('\nthe inviter\'s name');
{
    const s = newSecret();
    const base = { href: 'https://chat.example/chat' };
    const url = inviteUrl(s, base, 'Ana Ruiz');
    eq('the name follows the code in the fragment', url, `https://chat.example/chat#mesh=${s}&from=Ana%20Ruiz`);
    eq('the link still reads back as the same secret', readInvite(new URL(url).hash), s);
    eq('and the name reads back', inviteFrom(new URL(url).hash), 'Ana Ruiz');
    eq('no name, no from', inviteUrl(s, base, '  '), `https://chat.example/chat#mesh=${s}`);
    eq('a link without a name has no inviter', inviteFrom(`#mesh=${s}`), '');
    eq('the global mesh has no inviter', inviteFrom(''), '');
    eq('a broken percent-escape is ignored', inviteFrom(`#mesh=${s}&from=%E0%A4%A`), '');
    eq('control characters are removed and spaces collapsed', cleanName(' Ana\u0000\n  Ben '), 'Ana Ben');
    eq('a long name is cut to NAME_MAX characters', [...cleanName('é'.repeat(80))].length, NAME_MAX);
    eq('an empty from still reads as the invite', readInvite(`#mesh=${s}&from=`), s);
    throws('a from with a bad code still throws', () => readInvite(`#mesh=${s.slice(0, 21)}&from=Ana`));
    throws('a second parameter after from throws', () => readInvite(`#mesh=${s}&from=Ana&x=1`));
}

console.log('\nthe key schedule');
{
    // Fixed expected values: an invite that was already shared must lead to
    // the same room after a deploy. The values were computed separately with
    // Python's hmac and hashlib.
    const zero = await openRoom('AAAAAAAAAAAAAAAAAAAAAA');
    eq('the join token for an all-zero secret is pinned', zero.joinToken,
        'HtNrfd6FlO-dR6weiX6alidQIT07Vm-KqR71uLhVuZQ');
    eq('and so is its tag', zero.tag, '90bb47');
    eq('the tag is the head of the key the server files the room under',
        zero.tag, (await roomKey(zero.joinToken)).slice(0, 6));
    const sig = await zero.seal({ kind: 'offer', from: 1, to: 2, sdp: { type: 'offer', sdp: 'v=0' } });
    eq('and so is a seal', sig.mac, 'swbmwU1NuKyRsTHaYwfv3uqERm8n6OdbKruRyWiZTZY');

    const s = newSecret();
    const a = await openRoom(s), b = await openRoom(s), c = await openRoom(newSecret());
    eq('the same secret gives the same join token', a.joinToken, b.joinToken);
    ok('a different secret gives a different one', a.joinToken !== c.joinToken);
    ok('the join token does not contain the secret', !a.joinToken.includes(s));
    ok('a private room says so', a.private && !GLOBAL.private);
    eq('the global room has no token', GLOBAL.joinToken, null);
}

console.log('\nseal and verify');
{
    const s = newSecret();
    const room = await openRoom(s);
    const twin = await openRoom(s);
    const other = await openRoom(newSecret());
    const sdp = {
        type: 'offer',
        sdp: 'v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\na=fingerprint:sha-256 AA:BB:CC:DD\r\n',
    };
    const msg = { kind: 'offer', from: 3, to: 7, sdp };
    const sealed = await room.seal(msg);

    ok('a seal adds a MAC', typeof sealed.mac === 'string' && sealed.mac.length === 43);
    ok('and leaves the rest as it was', sealed.sdp === sdp && sealed.from === 3 && sealed.to === 7);
    ok('the same room verifies it', await room.verify(sealed));
    ok('so does another tab with the same invite', await twin.verify(sealed));
    ok('a room with a different invite does not', !(await other.verify(sealed)));
    ok('an intruder with the join token but not the secret does not',
        !(await (await openRoom(newSecret(), { overrideToken: room.joinToken })).verify(sealed)));

    const tamper = [
        ['a swapped DTLS fingerprint (a man in the middle)',
            { ...sealed, sdp: { ...sdp, sdp: sdp.sdp.replace('AA:BB', 'EE:FF') } }],
        ['a relabelled sender', { ...sealed, from: 4 }],
        ['a redirected recipient', { ...sealed, to: 8 }],
        ['an offer passed off as an answer', { ...sealed, kind: 'answer' }],
        ['a changed SDP type', { ...sealed, sdp: { ...sdp, type: 'answer' } }],
        ['a flipped MAC bit', { ...sealed, mac: (sealed.mac[0] === 'A' ? 'B' : 'A') + sealed.mac.slice(1) }],
        ['no MAC', { ...msg }],
        ['a MAC that is not base64url', { ...sealed, mac: '!'.repeat(43) }],
        ['a MAC of the wrong length', { ...sealed, mac: sealed.mac.slice(1) }],
        ['a MAC that is not a string', { ...sealed, mac: 42 }],
    ];
    for (const [what, m] of tamper) ok(`${what} is rejected`, !(await room.verify(m)));

    ok('an unsealed message is accepted by the global mesh', await GLOBAL.verify(msg));
    eq('and the global mesh seals nothing', await GLOBAL.seal(msg), msg);
    let refused = false;
    try { await openRoom('short'); } catch { refused = true; }
    ok('openRoom rejects a secret that is not an invite', refused);
}

console.log('\nhistoryTag: the scope saved chats are filed under');
{
    const s = newSecret();
    const [a, again, other] = await Promise.all([openRoom(s), openRoom(s), openRoom(newSecret())]);
    ok('is 32 hex characters', /^[0-9a-f]{32}$/.test(a.historyTag), `${a.historyTag}`);
    eq('the same for everyone with the invite', again.historyTag, a.historyTag);
    ok('and different for another mesh', other.historyTag !== a.historyTag);

    // What is stored in IndexedDB must reveal nothing about the secret, or
    // about the values the server and the page show.
    const hexSecret = Array.from(unb64url(s), (b) => b.toString(16).padStart(2, '0')).join('');
    const leaks = [s, hexSecret, a.joinToken, a.tag, await roomKey(a.joinToken)]
        .filter((v) => v.includes(a.historyTag) || a.historyTag.includes(v));
    eq('and shares nothing with the secret, the join token, the tag or the room key', leaks, []);
    eq('the global mesh is "global"', GLOBAL.historyTag, 'global');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
