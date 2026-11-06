// Fetches Qwen3-0.6B safetensors straight from the Hugging Face Hub (or a local copy),
// caches the bytes in Cache Storage, and returns per-tensor views into the bf16 data.

import { hfUrl } from "./config";

export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  /** byte offset/length within the downloaded region */
  start: number;
  length: number;
}

export interface RawWeights {
  bytes: Uint8Array<ArrayBuffer>;
  tensors: Map<string, TensorInfo>;
  source: string;
}

export type Progress = (phase: string, done: number, total: number) => void;

const LOCAL_BASE = import.meta.env.BASE_URL + "models/Qwen3-0.6B/";
const CACHE_NAME = "local-lm-viz-weights";
const CACHE_KEY = "https://local-lm-viz.invalid/Qwen3-0.6B/bf16-region-v1";

/** Prefer a local copy under public/models (dev), otherwise the HF Hub. */
export async function resolveBase(): Promise<{ weights: string; file: (f: string) => string; local: boolean }> {
  // Only the dev server can have a local copy (public/ isn't shipped in builds).
  const forceRemote = new URLSearchParams(location.search).has("remote");
  if (import.meta.env.DEV && !forceRemote) {
    try {
      const r = await fetch(LOCAL_BASE + "tokenizer_config.json", { method: "HEAD" });
      const ct = r.headers.get("content-type") ?? "";
      if (r.ok && !ct.includes("text/html")) {
        return { weights: LOCAL_BASE + "model.safetensors", file: (f) => LOCAL_BASE + f, local: true };
      }
    } catch {
      /* no local copy */
    }
  }
  return { weights: hfUrl("model.safetensors"), file: hfUrl, local: false };
}

async function fetchRange(url: string, start: number, endExclusive: number): Promise<Response> {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${endExclusive - 1}` } });
  if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${url}`);
  return r;
}

async function readAll(r: Response, expected: number, onProgress?: (n: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  const out = new Uint8Array(expected);
  const reader = r.body!.getReader();
  let off = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.length > expected) throw new Error("server sent more bytes than expected (range ignored?)");
    out.set(value, off);
    off += value.length;
    if (onProgress && (off - lastReport > 4 << 20 || off === expected)) {
      lastReport = off;
      onProgress(off);
    }
  }
  if (off !== expected) throw new Error(`short read: ${off} of ${expected} bytes`);
  return out;
}

export async function loadRawWeights(url: string, progress: Progress, useCache: boolean): Promise<RawWeights> {
  progress("Reading header", 0, 1);
  const lenBuf = new Uint8Array(await (await fetchRange(url, 0, 8)).arrayBuffer());
  const headerLen = Number(new DataView(lenBuf.buffer).getBigUint64(0, true));
  const headerBytes = new Uint8Array(await (await fetchRange(url, 8, 8 + headerLen)).arrayBuffer());
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  const dataStart = 8 + headerLen;

  // lm_head.weight is an exact duplicate of embed_tokens (tied) — skip downloading it.
  const entries = Object.entries(header).filter(([k]) => k !== "__metadata__" && k !== "lm_head.weight");
  const regionStart = Math.min(...entries.map(([, v]) => v.data_offsets[0]));
  const regionEnd = Math.max(...entries.map(([, v]) => v.data_offsets[1]));
  const total = regionEnd - regionStart;

  const tensors = new Map<string, TensorInfo>();
  for (const [name, v] of entries) {
    tensors.set(name, {
      name,
      dtype: v.dtype,
      shape: v.shape,
      start: v.data_offsets[0] - regionStart,
      length: v.data_offsets[1] - v.data_offsets[0],
    });
  }

  let bytes: Uint8Array<ArrayBuffer> | null = null;
  let source = url;
  const cache = useCache && "caches" in self ? await caches.open(CACHE_NAME) : null;
  if (cache) {
    const hit = await cache.match(CACHE_KEY);
    if (hit && Number(hit.headers.get("content-length")) === total) {
      progress("Reading from browser cache", 0, total);
      bytes = await readAll(hit, total, (n) => progress("Reading from browser cache", n, total));
      source = "browser cache";
    }
  }
  if (!bytes) {
    const r = await fetchRange(url, dataStart + regionStart, dataStart + regionEnd);
    progress("Downloading weights", 0, total);
    if (r.status === 200) {
      // Range ignored: we got the whole file. Slice out the region.
      const all = new Uint8Array(await r.arrayBuffer());
      bytes = all.slice(dataStart + regionStart, dataStart + regionEnd);
    } else {
      bytes = await readAll(r, total, (n) => progress("Downloading weights", n, total));
    }
    if (cache) {
      progress("Saving to browser cache", 0, 1);
      try {
        await cache.put(
          CACHE_KEY,
          new Response(bytes, { headers: { "content-length": String(total), "content-type": "application/octet-stream" } }),
        );
      } catch (e) {
        console.warn("Could not cache weights", e);
      }
    }
  }
  return { bytes, tensors, source };
}

export async function clearWeightCache() {
  if ("caches" in self) await caches.delete(CACHE_NAME);
}

// ---------- bf16 helpers ----------

/** f32 value of every possible bf16 bit pattern */
export const BF16_LUT: Float32Array = (() => {
  const u = new Uint32Array(65536);
  for (let i = 0; i < 65536; i++) u[i] = i << 16;
  return new Float32Array(u.buffer);
})();

export function bf16ToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = BF16_LUT[src[i]];
  return out;
}

export interface WeightThumb {
  /** RMS of each block, row-major [th, tw] */
  rms: Float32Array;
  th: number;
  tw: number;
  rows: number;
  cols: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  maxRms: number;
}

/** Downsample a [rows, cols] bf16 matrix into a block-RMS thumbnail + global stats. */
export function makeThumb(u16: Uint16Array, rows: number, cols: number, maxT = 256): WeightThumb {
  const th = Math.min(rows, maxT);
  const tw = Math.min(cols, maxT);
  const sumSq = new Float64Array(th * tw);
  const counts = new Uint32Array(th * tw);
  const colBlock = new Uint32Array(cols);
  for (let c = 0; c < cols; c++) colBlock[c] = Math.floor((c * tw) / cols);
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sum2 = 0;
  for (let r = 0; r < rows; r++) {
    const rowBase = Math.floor((r * th) / rows) * tw;
    const off = r * cols;
    for (let c = 0; c < cols; c++) {
      const v = BF16_LUT[u16[off + c]];
      const v2 = v * v;
      const b = rowBase + colBlock[c];
      sumSq[b] += v2;
      counts[b]++;
      sum += v;
      sum2 += v2;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const rms = new Float32Array(th * tw);
  let maxRms = 0;
  for (let i = 0; i < rms.length; i++) {
    rms[i] = Math.sqrt(sumSq[i] / Math.max(1, counts[i]));
    if (rms[i] > maxRms) maxRms = rms[i];
  }
  const n = rows * cols;
  const mean = sum / n;
  return { rms, th, tw, rows, cols, min, max, mean, std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), maxRms };
}
