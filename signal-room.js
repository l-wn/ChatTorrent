// One mesh on the signaling server: its peers, their claims, its roster, and
// the messages that change them. signal-server.js owns the sockets, the
// checkpoint and TURN, and holds one Room per mesh: the global one and one per
// private invite (see web/room.js). A Room does not know which kind it is,
// only the label it logs under. Rooms are isolated: a roster goes only to its
// own peers, placement uses only its own claims, and an SDP is relayed only to
// a peer in the same Room.

import { assign, covers, minDepth, gaps, served } from './web/split.js';

const send = (ws, msg) => ws.readyState === 1 && ws.send(JSON.stringify(msg));

export class Room {
    // `label` prefixes this room's log lines: '' for the global mesh and
    // `room <tag> ` for a private one. The tag is a hash prefix, not the token.
    // `maxChats` caps the replies generating at once in this room (Infinity:
    // no cap). `chatLeaseMs` frees a seat whose `chat-end` never came.
    constructor({ label = '', positions, head, minBlocks, maxPeers = Infinity,
                  maxChats = Infinity, chatLeaseMs = 600000, log = console.log }) {
        this.label = label;
        this.positions = positions;
        this.head = head;
        this.minBlocks = minBlocks;
        this.maxPeers = maxPeers;
        this.maxChats = maxChats;
        this.chatLeaseMs = chatLeaseMs;
        // Peers with a reply generating: id -> the lease timer.
        this.chats = new Map();
        this.log = (line) => log(`${this.label}${line}`);
        // id -> { ws, cap, lo, hi, have, ready, alive }. `lo`/`hi` are null
        // until the peer states a capacity and is placed. A peer with null
        // `lo` holds nothing.
        this.peers = new Map();
    }

    get size() { return this.peers.size; }
    get full() { return this.peers.size >= this.maxPeers; }

    add(id, ws) {
        this.peers.set(id, { ws, cap: 0, lo: null, hi: null, have: null, ready: false, alive: true });
        this.log(`+ peer ${id} (${this.peers.size} online): ${this.health()}`);
    }

    // A departure leaves a hole and moves nobody. The remaining peers keep
    // what they downloaded, and the next peer to join or grow fills the hole.
    // At r >= 2 another peer already covers it.
    remove(id) {
        const me = this.peers.get(id);
        if (!this.peers.delete(id)) return;
        this.#unseat(id);
        this.log(`- peer ${id} ${me ? this.where(me) : ''} (${this.peers.size} online): ${this.health()}`);
        this.broadcast();
    }

    // `[lo, hi)` is what a peer claimed and `[lo, have)` what it can run. The
    // server does not infer `have`: it relays the peer's value, clamped to the
    // claim and never decreasing. See served() in web/split.js.
    roster() {
        return [...this.peers].map(([id, p]) =>
            ({ id, lo: p.lo, hi: p.hi, have: p.have, cap: p.cap, ready: p.ready }));
    }

    broadcast() {
        const msg = { type: 'roster', peers: this.roster() };
        for (const p of this.peers.values()) send(p.ws, msg);
    }

    // Every peer but `id`. A peer being re-placed must not count its own range.
    #others(id) { return [...this.peers].filter(([pid]) => pid !== id).map(([, p]) => p); }

    // Counts every peer with a range, ready or not, so two peers downloading
    // at once are not both given the same hole.
    #place(id) {
        const me = this.peers.get(id);
        const r = assign(this.#others(id), me.cap, this.positions, this.minBlocks);
        me.lo = r?.lo ?? null;
        me.hi = r?.hi ?? null;
        // A new range holds nothing. `have` rises only when the peer reports it.
        me.have = r?.lo ?? null;
        return r;
    }

    // One line describing the network: `r` over what is served, and over the
    // claims too when they differ. A higher claimed `r` means positions are
    // still loading.
    health() {
        const held = [...this.peers.values()];
        const now = held.map(served);
        const r = minDepth(now, this.positions);
        const claimed = minDepth(held, this.positions);
        const tail = (claimed === r ? '' : `, claimed r=${claimed}`)
            + (Number.isFinite(this.maxChats) ? `, chats ${this.chats.size}/${this.maxChats}` : '');
        if (covers(now, this.positions)) return `covered, r=${r}${tail}`;
        return `NOT covered, unheld ${gaps(now, this.positions)
            .map(([a, b]) => this.where({ lo: a, hi: b })).join(', ')}${tail}`;
    }

    // A range of positions as readable text. The two ends are named and the
    // middle is block indices.
    where(p) {
        if (p.lo == null) return 'client';
        const parts = [];
        if (p.lo === 0) parts.push('embedding');
        const bLo = Math.max(p.lo, 1), bHi = Math.min(p.hi, this.head);
        if (bHi > bLo) parts.push(bHi - bLo === 1 ? `block ${bLo - 1}` : `blocks ${bLo - 1}–${bHi - 2}`);
        if (p.hi === this.positions) parts.push('head');
        return parts.join(' + ');
    }

    // The room's limit as sent to pages: a number, or null for none.
    get chatLimit() { return Number.isFinite(this.maxChats) ? this.maxChats : null; }

    // Free a peer's seat, if it holds one.
    #unseat(id) {
        clearTimeout(this.chats.get(id));
        this.chats.delete(id);
    }

    // A requested range (reconnect or `?at=`), or null if it is invalid for
    // this model. Strict, because a bad range would put positions in the
    // roster that nobody holds.
    #holdClaim(m, cap) {
        const lo = Number(m.lo), hi = Number(m.hi);
        if (!Number.isInteger(lo) || !Number.isInteger(hi)) return null;
        if (lo < 0 || hi > this.positions || hi - lo < 1) return null;
        if (hi - lo !== cap) return null;      // must match the stated capacity
        return { lo, hi };
    }

    // Handle a message from peer `id` after its `join`.
    handle(id, m) {
        const me = this.peers.get(id);
        if (!me) return;
        const ws = me.ws;

        if (m.type === 'pong') { me.alive = true; return; }

        // capacity: how many positions the peer can hold. Answered with a
        // range, or nulls for a peer below the minimum. A peer may be
        // re-placed until it is ready; after that other peers route on its
        // range.
        if (m.type === 'capacity') {
            if (me.ready) return;
            me.cap = Number(m.cap) || 0;
            // A requested range (a reconnecting peer keeping the weights in
            // its VRAM, or `?at=`) is granted if valid. Otherwise the peer is
            // placed normally.
            const held = this.#holdClaim(m, me.cap);
            if (held) {
                me.lo = held.lo;
                me.hi = held.hi;
                // The peer sends `have` again in its next message.
                me.have = held.lo;
                send(ws, { type: 'assign', lo: me.lo, hi: me.hi });
                this.log(`  peer ${id} reclaims ${this.where(me)}: ${this.health()}`);
                this.broadcast();
                return;
            }
            this.#place(id);
            send(ws, { type: 'assign', lo: me.lo, hi: me.hi });
            this.log(`  peer ${id} cap ${me.cap} -> ${this.where(me)}: ${this.health()}`);
            this.broadcast();
            return;
        }

        // have: everything below this position is downloaded and warmed up.
        // Clamped to the claim and never decreasing, so a stale roster can
        // only understate a peer.
        if (m.type === 'have') {
            if (me.lo == null) return;
            const h = Math.max(me.have ?? me.lo, Math.min(Number(m.have) || 0, me.hi));
            if (h === me.have) return;
            me.have = h;
            this.log(`  peer ${id} holds ${this.where({ lo: me.lo, hi: me.have })}`
                + ` of ${this.where(me)}: ${this.health()}`);
            this.broadcast();
            return;
        }

        // shrink: give back the part of the claim that was never downloaded.
        // Nothing routes into `[have, hi)`, and keeping it claimed keeps other
        // peers out. Only lowers `hi` to the announced `have`.
        if (m.type === 'shrink') {
            if (me.lo == null || me.have == null || me.have >= me.hi) return;
            if (me.have <= me.lo) return;              // holds nothing; use claim(0)
            me.hi = me.have;
            me.cap = me.hi - me.lo;
            this.log(`  peer ${id} gave back its unfilled tail, keeps ${this.where(me)}`
                + `: ${this.health()}`);
            this.broadcast();
            return;
        }

        // grow: the range widened by one downloaded and warmed-up position on
        // either edge. Only for a ready peer and only outward. The new
        // position is servable, so a raised `hi` raises `have` too.
        if (m.type === 'grow') {
            if (me.lo == null || !me.ready) return;
            const lo = Number(m.lo), hi = Number(m.hi);
            if (!Number.isInteger(lo) || !Number.isInteger(hi)) return;
            if (lo < 0 || hi > this.positions) return;
            if (lo > me.lo || hi < me.hi) return;
            if (lo === me.lo && hi === me.hi) return;
            me.lo = lo;
            if (hi > me.hi) me.have = hi;
            me.hi = hi;
            me.cap = me.hi - me.lo;
            this.log(`  peer ${id} grew to ${this.where(me)}: ${this.health()}`);
            this.broadcast();
            return;
        }

        // ready: a pass has run through the peer's first positions. Never
        // unset: a peer that cannot serve closes its socket, which is a leave.
        if (m.type === 'ready') {
            if (!me.ready && me.lo != null) {
                me.ready = true;
                // A peer that never sent `have` is counted as holding its
                // whole claim.
                if (me.have == null || me.have === me.lo) me.have = me.hi;
                this.log(`  peer ${id} ${this.where(me)} ready: ${this.health()}`);
                this.broadcast();
            }
            return;
        }

        // chat-start: a reply is about to generate. Answered with `chat`,
        // refused once `maxChats` replies are generating. A peer holds at most
        // one seat, so asking again while holding one only renews its lease.
        // chat-end frees it, as does leaving.
        if (m.type === 'chat-start') {
            const max = this.chatLimit;
            if (!this.chats.has(id) && this.chats.size >= this.maxChats) {
                this.log(`  peer ${id} refused a chat: ${this.chats.size}/${max} running`);
                send(ws, { type: 'chat', ok: false, max });
                return;
            }
            clearTimeout(this.chats.get(id));
            const timer = setTimeout(() => this.chats.delete(id), this.chatLeaseMs);
            timer.unref?.();
            this.chats.set(id, timer);
            send(ws, { type: 'chat', ok: true, max });
            return;
        }
        if (m.type === 'chat-end') { this.#unseat(id); return; }

        // signal: deliver this SDP to a peer in this room. `mac` is relayed
        // unchanged. It is the sender's seal over the SDP and both ids
        // (web/room.js); the receiver checks it and this server cannot
        // produce it.
        //
        // `kind` becomes the relayed message's `type`, so only 'offer' and
        // 'answer' are allowed. Anything else would let a peer send another a
        // forged `welcome` or `roster`.
        if (m.type !== 'signal') return;
        if (m.kind !== 'offer' && m.kind !== 'answer') return;
        const to = this.peers.get(m.to);
        if (to) send(to.ws, { type: m.kind, from: id, sdp: m.sdp, ...(m.mac != null && { mac: m.mac }) });
    }
}
