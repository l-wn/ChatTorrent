// Which mesh a tab belongs to, and the keys that protect a private one.
//
// A tab joins the global mesh unless its URL carries an invite,
// `#mesh=<secret>` (16 random bytes, base64url). Three values are derived from
// the secret with HKDF-SHA-256:
//
//   joinToken   sent to the signaling server, which keys the room by its hash
//   historyTag  the scope saved chats are stored under (web/history.js)
//   linkKey     HMAC key that seals every WebRTC offer and answer
//
// The secret, historyTag and linkKey never leave the browser. The global mesh
// is a room with no key: `seal` returns its input and `verify` accepts
// everything. The trust model is in docs/architecture.md (Private meshes).

const SALT = 'chattorrent/mesh/v1';
export const SECRET_BYTES = 16;
const SECRET_RE = /^[A-Za-z0-9_-]{22}$/;

// A version number included in every MAC, so MACs over a different set of
// fields never verify.
const SEAL_V = 1;

const enc = new TextEncoder();

export function b64url(bytes) {
    let s = '';
    for (const b of bytes) s += String.fromCharCode(b);
    return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function unb64url(str) {
    const s = atob(str.replace(/-/g, '+').replace(/_/g, '/'));
    return Uint8Array.from(s, (c) => c.charCodeAt(0));
}

const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');

// Generate a new invite secret.
export function newSecret() {
    return b64url(crypto.getRandomValues(new Uint8Array(SECRET_BYTES)));
}

// The link to share: this page's URL with the fragment set and the query
// removed. The query (`?blocks=`, `?gb=`) holds this tab's own settings.
export function inviteUrl(secret, base = globalThis.location) {
    const u = new URL(base.href);
    u.search = '';
    u.hash = `mesh=${secret}`;
    return u.href;
}

// The secret in a `location.hash`, or null for the global mesh.
//
// An empty hash is the global mesh. Anything else must be exactly
// `#mesh=<22 base64url characters>`. A damaged or mistyped invite throws; it
// does not fall back to the public mesh, where someone expecting a private
// chat would be exposed.
export function readInvite(hash) {
    const h = (hash ?? '').replace(/^#/, '');
    if (!h) return null;
    const m = /^mesh=(.*)$/.exec(h);
    if (!m) throw new Error(`the link's "#${h.slice(0, 40)}" is not an invite. Expected #mesh=<code>.`);
    if (!SECRET_RE.test(m[1])) {
        throw new Error('the invite code in this link is damaged. It should be 22 letters, ' +
            'digits, "-" or "_". Ask for the link again.');
    }
    return m[1];
}

// SHA-256 of a join token, hex. The server keys a room by it. Its first six
// characters are the tag shown in the server's log and on the page, which
// members can compare to confirm they are in the same room.
export async function roomKey(joinToken) {
    return hex(await crypto.subtle.digest('SHA-256', enc.encode(joinToken)));
}

// The fields a MAC covers, serialized as one unambiguous string.
const sealed = ({ kind, from, to, sdp }) =>
    JSON.stringify([SEAL_V, kind, from, to, sdp?.type ?? null, sdp?.sdp ?? null]);

// The global mesh: no token, nothing sealed, everything accepted.
export const GLOBAL = Object.freeze({
    private: false,
    joinToken: null,
    tag: null,
    historyTag: 'global',
    seal: async (signal) => signal,
    verify: async () => true,
});

// A room for this secret, or the global mesh for null.
//
// `overrideToken` is for the test harness only: it simulates an intruder that
// knows a room's join token (as the server does) but not its secret.
export async function openRoom(secret, { overrideToken = null } = {}) {
    if (secret == null) return GLOBAL;
    if (!SECRET_RE.test(secret)) throw new Error('not an invite secret');

    const ikm = await crypto.subtle.importKey('raw', unb64url(secret), 'HKDF', false,
        ['deriveBits', 'deriveKey']);
    const hkdf = (info) => ({ name: 'HKDF', hash: 'SHA-256', salt: enc.encode(SALT),
                              info: enc.encode(info) });

    const joinToken = overrideToken
        ?? b64url(new Uint8Array(await crypto.subtle.deriveBits(hkdf('join'), ikm, 256)));
    const historyTag = hex(await crypto.subtle.deriveBits(hkdf('history'), ikm, 128));
    const linkKey = await crypto.subtle.deriveKey(hkdf('link'), ikm,
        { name: 'HMAC', hash: 'SHA-256', length: 256 }, false, ['sign', 'verify']);

    return Object.freeze({
        private: true,
        joinToken,
        tag: (await roomKey(joinToken)).slice(0, 6),
        historyTag,

        // `signal` is { kind, from, to, sdp }. Returns it with `mac` added.
        async seal(signal) {
            const mac = await crypto.subtle.sign('HMAC', linkKey, enc.encode(sealed(signal)));
            return { ...signal, mac: b64url(new Uint8Array(mac)) };
        },

        // True only if `mac` is this room's MAC over these fields.
        // subtle.verify compares in constant time.
        async verify(signal) {
            if (typeof signal?.mac !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(signal.mac)) {
                return false;
            }
            try {
                return await crypto.subtle.verify('HMAC', linkKey, unb64url(signal.mac),
                    enc.encode(sealed(signal)));
            } catch {
                return false;
            }
        },
    });
}
