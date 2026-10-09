// A stand-in for the WebRTC datachannel, installed under web/net.js without
// changing net.js.
//
// The mesh's transport is an RTCDataChannel, and the integration test uses the
// real one: ICE and SCTP between real tabs. The same harness also checks that
// N peers compute the same tokens as one peer, and the transport is irrelevant
// to that. Real ICE adds a second per link, a full mesh of workers has O(n^2)
// links, and ICE can fail for reasons unrelated to the model.
//
// This file installs a replacement for `window.RTCPeerConnection` that
// implements the methods net.js uses, over a BroadcastChannel, which is
// ordered, reliable, cross-tab and carries ArrayBuffers. net.js is unchanged:
// the offer and answer go through the real signaling server,
// `pack`/`fragment`/`reassemble` run, and `maxMessageSize` is set small so a
// residual stream is fragmented and reassembled.
//
// Not modelled, and covered only by the rtc wire: negotiation failure, ICE,
// `bufferedAmount` backpressure (always 0 here, so net.js's SEND_HIGH_WATER
// check never triggers), and packet loss. See tools/mesh.gpu.test.mjs's --wire
// flag.

// 4 KiB is 16 token positions at hidden 64, so a 64-token prefill chunk is
// sent as five fragments and every prompt exercises reassembly.
const MAX_MESSAGE = 4096;

// A BroadcastChannel delivers to every context except the one that posted, so
// between two tabs it is a two-party link. One channel per link, named by the
// offerer.
let nonce = 0;
const channelName = () =>
    `ctwire-${Math.random().toString(36).slice(2, 10)}-${Date.now().toString(36)}-${nonce++}`;

// The channel half. `readyState` drives net.js's `#live()`, so a link counts
// as up only once both ends are attached, as with a real datachannel.
class Channel extends EventTarget {
    constructor(name, label) {
        super();
        this.label = label;
        this.ordered = true;
        this.binaryType = 'arraybuffer';
        this.readyState = 'connecting';
        this.bufferedAmount = 0;
        this.bc = new BroadcastChannel(name);
        this.bc.onmessage = (ev) => {
            if (this.readyState !== 'open') return;
            // `data` is an ArrayBuffer, which net.js's #attach expects from a
            // channel with binaryType 'arraybuffer'. structuredClone has
            // already made a copy.
            this.dispatchEvent(Object.assign(new Event('message'), { data: ev.data }));
        };
    }

    open() {
        if (this.readyState !== 'connecting') return;
        this.readyState = 'open';
        this.dispatchEvent(new Event('open'));
    }

    send(buf) {
        if (this.readyState !== 'open') throw new Error('channel is not open');
        // A real SCTP association throws on a send over its size limit, and
        // net.js fragments to stay under it. The limit is enforced here too,
        // so the fragmentation code is tested.
        if (buf.byteLength > MAX_MESSAGE) {
            throw new Error(`message of ${buf.byteLength} bytes exceeds the ` +
                `${MAX_MESSAGE}-byte limit`);
        }
        this.bc.postMessage(buf);
    }

    close() {
        if (this.readyState === 'closed') return;
        this.readyState = 'closed';
        try { this.bc.close(); } catch { /* already gone */ }
        this.dispatchEvent(new Event('close'));
    }
}

// The connection half. Implements only what net.js uses: createDataChannel,
// createOffer/createAnswer, set{Local,Remote}Description, localDescription,
// iceGatheringState (read by iceDone, which returns immediately on
// 'complete'), connectionState (read by #sync to prune dead links), sctp, and
// close.
class BroadcastPeerConnection extends EventTarget {
    constructor() {
        super();
        // Nothing to gather: no candidates and no STUN.
        this.iceGatheringState = 'complete';
        this.iceConnectionState = 'connected';
        this.connectionState = 'connected';
        this.signalingState = 'stable';
        this.localDescription = null;
        this.remoteDescription = null;
        this.sctp = { maxMessageSize: MAX_MESSAGE };
        this.channel = null;
        this.name = null;
    }

    // The offerer names the channel. The name is sent in the `sdp` field,
    // which the signaling server relays without reading.
    createDataChannel(label, _opts) {
        this.name ??= channelName();
        this.channel = new Channel(this.name, label);
        return this.channel;
    }

    async createOffer() {
        this.name ??= channelName();
        return { type: 'offer', sdp: this.name };
    }

    async createAnswer() {
        return { type: 'answer', sdp: this.name };
    }

    async setLocalDescription(desc) { this.localDescription = desc; }

    async setRemoteDescription(desc) {
        this.remoteDescription = desc;
        if (desc.type === 'offer') {
            // The answering side reads the channel's name here, opens its end,
            // and emits a `datachannel` event like the real API, which net.js
            // listens for.
            this.name = desc.sdp;
            const ch = new Channel(this.name, 'mesh');
            this.channel = ch;
            this.dispatchEvent(Object.assign(new Event('datachannel'), { channel: ch }));
            ch.open();
            return;
        }
        // The answer has reached the offerer, so the far end is listening on
        // the channel and this end can open. Opening earlier would post the
        // first hand-off to a channel with no subscriber, and it would be
        // lost.
        this.channel?.open();
    }

    addIceCandidate() { return Promise.resolve(); }

    close() {
        this.connectionState = 'closed';
        this.iceConnectionState = 'closed';
        this.channel?.close();
    }
}

// Installed on the window, not injected through net.js, so net.js needs no
// test hook.
export function installBroadcastWire() {
    window.RTCPeerConnection = BroadcastPeerConnection;
    return { wire: 'bc', maxMessage: MAX_MESSAGE };
}
