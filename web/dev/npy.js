// Minimal .npy v1/v2 parser. Supports only the dtype the oracle dumps use (<f4).
// Format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

export function parseNpy(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const magic = String.fromCharCode(...bytes.slice(0, 6));
  if (magic !== '\x93NUMPY') throw new Error('Not a .npy file');

  const major = bytes[6];
  let headerLen, headerStart;
  if (major === 1) {
    headerLen   = bytes[8] | (bytes[9] << 8);
    headerStart = 10;
  } else {
    headerLen   = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    headerStart = 12;
  }

  const header = new TextDecoder('latin1')
    .decode(bytes.subarray(headerStart, headerStart + headerLen));

  const descr   = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const shapeS  = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  if (!descr || shapeS === undefined) throw new Error('Bad .npy header: ' + header);
  if (descr !== '<f4') throw new Error(`Unsupported dtype ${descr}; expected <f4`);
  if (fortran === 'True') throw new Error('Fortran-order .npy not supported');

  const shape = shapeS
    .split(',').map(s => s.trim()).filter(s => s.length).map(Number);

  const dataOffset = headerStart + headerLen;
  // A Float32Array view needs 4-byte alignment, so the data is copied to a new buffer.
  const data = new Float32Array(arrayBuffer.slice(dataOffset));
  const expected = shape.reduce((a, b) => a * b, 1);
  if (data.length !== expected) {
    throw new Error(`npy size mismatch: shape ${shape} expects ${expected}, got ${data.length}`);
  }
  return { shape, data };
}

export async function loadNpy(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Fetch ${url}: ${r.status}`);
  return parseNpy(await r.arrayBuffer());
}

export async function loadJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Fetch ${url}: ${r.status}`);
  return r.json();
}
