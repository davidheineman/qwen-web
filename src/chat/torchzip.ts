// Reads a PyTorch `torch.save` checkpoint (zip archive + pickle) without PyTorch.
// Only handles what state dicts need: dicts/lists/tuples/strings/numbers, OrderedDict and
// torch._utils._rebuild_tensor_v2 over Float/BFloat16/Half storages. Tensors come back as f32.

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

interface ZipEntry {
  offset: number; // start of the (uncompressed) file data
  size: number;
}

function readZip(buf: Uint8Array): Map<string, ZipEntry> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // End of central directory: scan back for its signature (comment is at most 64 KiB).
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 65535); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("not a zip file (is this a torch.save checkpoint?)");
  let count = dv.getUint16(eocd + 10, true);
  let cdOffset = dv.getUint32(eocd + 16, true);
  if (count === 0xffff || cdOffset === 0xffffffff) {
    // Zip64: the locator sits right before the EOCD record.
    const loc = eocd - 20;
    if (dv.getUint32(loc, true) !== 0x07064b50) throw new Error("bad zip64 locator");
    const z64 = Number(dv.getBigUint64(loc + 8, true));
    count = Number(dv.getBigUint64(z64 + 32, true));
    cdOffset = Number(dv.getBigUint64(z64 + 48, true));
  }

  const out = new Map<string, ZipEntry>();
  const dec = new TextDecoder();
  let p = cdOffset;
  for (let n = 0; n < count; n++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("bad zip central directory");
    const method = dv.getUint16(p + 10, true);
    let size = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    let local = dv.getUint32(p + 42, true);
    const name = dec.decode(buf.subarray(p + 46, p + 46 + nameLen));
    // Zip64 extended info (0x0001) carries the real values for fields set to 0xFFFFFFFF.
    let e = p + 46 + nameLen;
    const eEnd = e + extraLen;
    while (e + 4 <= eEnd) {
      const id = dv.getUint16(e, true);
      const len = dv.getUint16(e + 2, true);
      if (id === 0x0001) {
        let q = e + 4;
        if (dv.getUint32(p + 24, true) === 0xffffffff) { size = Number(dv.getBigUint64(q, true)); q += 8; }
        if (dv.getUint32(p + 20, true) === 0xffffffff) q += 8; // compressed size
        if (dv.getUint32(p + 42, true) === 0xffffffff) local = Number(dv.getBigUint64(q, true));
      }
      e += 4 + len;
    }
    if (method !== 0) throw new Error(`zip entry ${name} is compressed (method ${method}); expected stored`);
    const lName = dv.getUint16(local + 26, true);
    const lExtra = dv.getUint16(local + 28, true);
    out.set(name, { offset: local + 30 + lName + lExtra, size });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

// ---------------------------------------------------------------- pickle

class Global {
  constructor(readonly name: string) {}
}
class StorageRef {
  constructor(readonly dtype: string, readonly key: string) {}
}
class TensorRef {
  constructor(
    readonly storage: StorageRef,
    readonly offset: number,
    readonly shape: number[],
    readonly stride: number[],
  ) {}
}

function unpickle(bytes: Uint8Array): unknown {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dec = new TextDecoder();
  const stack: unknown[] = [];
  const marks: number[] = [];
  const memo = new Map<number, unknown>();
  let p = 0;
  const popMark = (): unknown[] => {
    const m = marks.pop()!;
    return stack.splice(m);
  };
  const str = (n: number) => {
    const s = dec.decode(bytes.subarray(p, p + n));
    p += n;
    return s;
  };
  const line = () => {
    const e = bytes.indexOf(10, p);
    const s = dec.decode(bytes.subarray(p, e));
    p = e + 1;
    return s;
  };
  const call = (fn: unknown, args: unknown[]): unknown => {
    if (!(fn instanceof Global)) throw new Error("pickle: REDUCE on non-global");
    switch (fn.name) {
      case "collections.OrderedDict":
        return new Map();
      case "torch._utils._rebuild_tensor_v2": {
        const [storage, offset, shape, stride] = args as [StorageRef, number, number[], number[]];
        return new TensorRef(storage, offset, shape, stride);
      }
      case "torch._utils._rebuild_parameter":
        return args[0];
      default:
        throw new Error(`pickle: unsupported callable ${fn.name}`);
    }
  };
  const setItems = (target: unknown, kv: unknown[]) => {
    for (let i = 0; i < kv.length; i += 2) {
      if (target instanceof Map) target.set(kv[i], kv[i + 1]);
      else (target as Record<string, unknown>)[String(kv[i])] = kv[i + 1];
    }
  };

  for (;;) {
    const op = bytes[p++];
    switch (op) {
      case 0x80: p++; break; // PROTO
      case 0x95: p += 8; break; // FRAME
      case 0x2e: return stack.pop(); // STOP
      case 0x28: marks.push(stack.length); break; // MARK
      case 0x7d: stack.push({}); break; // EMPTY_DICT
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x29: stack.push([]); break; // EMPTY_TUPLE
      case 0x74: stack.push(popMark()); break; // TUPLE
      case 0x6c: stack.push(popMark()); break; // LIST
      case 0x64: { const kv = popMark(); const d = {}; setItems(d, kv); stack.push(d); break; } // DICT
      case 0x85: stack.push([stack.pop()]); break; // TUPLE1
      case 0x86: { const b = stack.pop(); const a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
      case 0x87: { const c = stack.pop(); const b = stack.pop(); const a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
      case 0x4e: stack.push(null); break; // NONE
      case 0x88: stack.push(true); break; // NEWTRUE
      case 0x89: stack.push(false); break; // NEWFALSE
      case 0x4b: stack.push(bytes[p++]); break; // BININT1
      case 0x4d: stack.push(dv.getUint16(p, true)); p += 2; break; // BININT2
      case 0x4a: stack.push(dv.getInt32(p, true)); p += 4; break; // BININT
      case 0x8a: { // LONG1
        const n = bytes[p++];
        let v = 0n;
        for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(bytes[p + i]);
        if (n > 0 && bytes[p + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
        p += n;
        stack.push(Number(v));
        break;
      }
      case 0x47: stack.push(dv.getFloat64(p, false)); p += 8; break; // BINFLOAT
      case 0x58: { const n = dv.getUint32(p, true); p += 4; stack.push(str(n)); break; } // BINUNICODE
      case 0x8c: { const n = bytes[p++]; stack.push(str(n)); break; } // SHORT_BINUNICODE
      case 0x8d: { const n = Number(dv.getBigUint64(p, true)); p += 8; stack.push(str(n)); break; } // BINUNICODE8
      case 0x55: { const n = bytes[p++]; stack.push(str(n)); break; } // SHORT_BINSTRING
      case 0x54: { const n = dv.getInt32(p, true); p += 4; stack.push(str(n)); break; } // BINSTRING
      case 0x43: { const n = bytes[p++]; stack.push(bytes.slice(p, p + n)); p += n; break; } // SHORT_BINBYTES
      case 0x42: { const n = dv.getUint32(p, true); p += 4; stack.push(bytes.slice(p, p + n)); p += n; break; } // BINBYTES
      case 0x71: memo.set(bytes[p++], stack[stack.length - 1]); break; // BINPUT
      case 0x72: memo.set(dv.getUint32(p, true), stack[stack.length - 1]); p += 4; break; // LONG_BINPUT
      case 0x94: memo.set(memo.size, stack[stack.length - 1]); break; // MEMOIZE
      case 0x68: stack.push(memo.get(bytes[p++])); break; // BINGET
      case 0x6a: stack.push(memo.get(dv.getUint32(p, true))); p += 4; break; // LONG_BINGET
      case 0x63: { const mod = line(); const name = line(); stack.push(new Global(`${mod}.${name}`)); break; } // GLOBAL
      case 0x93: { const name = stack.pop() as string; const mod = stack.pop() as string; stack.push(new Global(`${mod}.${name}`)); break; } // STACK_GLOBAL
      case 0x52: { const args = stack.pop() as unknown[]; const fn = stack.pop(); stack.push(call(fn, args)); break; } // REDUCE
      case 0x81: { const args = stack.pop() as unknown[]; const cls = stack.pop(); stack.push(call(cls, args)); break; } // NEWOBJ
      case 0x62: stack.pop(); break; // BUILD (state is irrelevant for what we load)
      case 0x51: { // BINPERSID: ('storage', StorageType, key, location, numel)
        const pid = stack.pop() as unknown[];
        const type = pid[1] as Global;
        stack.push(new StorageRef(type.name.replace(/^torch\./, ""), String(pid[2])));
        break;
      }
      case 0x73: { const v = stack.pop(); const k = stack.pop(); setItems(stack[stack.length - 1], [k, v]); break; } // SETITEM
      case 0x75: { const kv = popMark(); setItems(stack[stack.length - 1], kv); break; } // SETITEMS
      case 0x61: { const v = stack.pop(); (stack[stack.length - 1] as unknown[]).push(v); break; } // APPEND
      case 0x65: { const vs = popMark(); (stack[stack.length - 1] as unknown[]).push(...vs); break; } // APPENDS
      default:
        throw new Error(`pickle: unsupported opcode 0x${op.toString(16)} at ${p - 1}`);
    }
  }
}

// ---------------------------------------------------------------- tensors

const BF16 = (u16: Uint16Array) => {
  const out = new Uint32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = u16[i] << 16;
  return new Float32Array(out.buffer);
};

function f16ToF32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

export interface Checkpoint {
  /** the whole unpickled object, with tensors materialized as {shape, data} */
  root: Record<string, unknown>;
}

/** Parse a torch.save() file. Tensors sharing a storage (tied weights) share the same Float32Array. */
export function loadTorchCheckpoint(buf: Uint8Array): Checkpoint {
  const zip = readZip(buf);
  const pklName = [...zip.keys()].find((n) => n.endsWith("/data.pkl") || n === "data.pkl");
  if (!pklName) throw new Error("checkpoint has no data.pkl");
  const prefix = pklName.slice(0, -"data.pkl".length);
  const bo = zip.get(prefix + "byteorder");
  if (bo && new TextDecoder().decode(buf.subarray(bo.offset, bo.offset + bo.size)) !== "little") {
    throw new Error("big-endian checkpoints are not supported");
  }
  const pkl = zip.get(pklName)!;
  const root = unpickle(buf.subarray(pkl.offset, pkl.offset + pkl.size));

  const storages = new Map<string, Float32Array>();
  const storage = (ref: StorageRef): Float32Array => {
    const hit = storages.get(ref.key);
    if (hit) return hit;
    const ent = zip.get(`${prefix}data/${ref.key}`);
    if (!ent) throw new Error(`missing storage data/${ref.key}`);
    // Copy out so the arrays are aligned and the download buffer can be freed.
    const raw = buf.slice(ent.offset, ent.offset + ent.size).buffer;
    let f: Float32Array;
    if (ref.dtype === "FloatStorage") f = new Float32Array(raw);
    else if (ref.dtype === "BFloat16Storage") f = BF16(new Uint16Array(raw));
    else if (ref.dtype === "HalfStorage") f = f16ToF32(new Uint16Array(raw));
    else throw new Error(`unsupported storage type ${ref.dtype}`);
    storages.set(ref.key, f);
    return f;
  };

  const materialize = (v: unknown): unknown => {
    if (v instanceof TensorRef) {
      const numel = v.shape.reduce((a, b) => a * b, 1);
      let expect = 1;
      for (let i = v.shape.length - 1; i >= 0; i--) {
        if (v.shape[i] !== 1 && v.stride[i] !== expect) throw new Error("non-contiguous tensors are not supported");
        expect *= v.shape[i];
      }
      const s = storage(v.storage);
      const data = v.offset === 0 && numel === s.length ? s : s.subarray(v.offset, v.offset + numel);
      return { shape: v.shape, data } satisfies Tensor;
    }
    if (v instanceof Map) return Object.fromEntries([...v].map(([k, x]) => [k, materialize(x)]));
    if (Array.isArray(v)) return v.map(materialize);
    if (v && typeof v === "object" && !(v instanceof Uint8Array)) {
      return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, materialize(x)]));
    }
    return v;
  };
  return { root: materialize(root) as Record<string, unknown> };
}
