// WebGPU forward pass for Qwen3-0.6B that captures every intermediate activation.
//
// There is no KV cache: each call recomputes the full sequence. That keeps the code simple and
// gives us attention patterns for every position, and a 0.6B model on a GPU is fast enough for
// the short contexts a visualizer deals with.

import { CFG, KV_DIM, Q_DIM } from "./config";
import { makeThumb, type RawWeights, type WeightThumb, BF16_LUT, type Progress } from "./loader";
import * as S from "./shaders";

const H = CFG.hidden;
const I = CFG.intermediate;
const L = CFG.layers;
const NH = CFG.heads;
const NKV = CFG.kvHeads;
const VOCAB = CFG.vocab;
const EMBED_CHUNK_ROWS = 32768;

/** Per-layer activations captured at the focus position (in forward order). */
export const DETAIL_FIELDS = [
  ["attnNorm", H],
  ["q", Q_DIM],
  ["k", KV_DIM],
  ["v", KV_DIM],
  ["qr", Q_DIM],
  ["kr", KV_DIM],
  ["attn", Q_DIM],
  ["o", H],
  ["mid", H],
  ["mlpNorm", H],
  ["gate", I],
  ["up", I],
  ["act", I],
  ["down", H],
] as const;
export type DetailName = (typeof DETAIL_FIELDS)[number][0];
export type LayerDetail = Record<DetailName, Float32Array>;

const DETAIL_OFF: Record<string, number> = {};
let DETAIL_REC = 0;
for (const [n, s] of DETAIL_FIELDS) {
  DETAIL_OFF[n] = DETAIL_REC;
  DETAIL_REC += s;
}

export interface TopToken {
  id: number;
  logit: number;
  prob: number;
}

export interface Capture {
  tokens: number[];
  T: number;
  focus: number;
  /** residual stream [L+1, T, H]: index 0 = embeddings, i = output of layer i-1 */
  resid: Float32Array;
  /** attention probabilities [L, NH, T, T] */
  probs: Float32Array;
  layers: LayerDetail[];
  /** logit lens: top tokens from each residual checkpoint at the focus position (index L = real output) */
  lens: TopToken[][];
  /** final logits at focus position */
  logits: Float32Array;
  ms: number;
}

export interface WeightEntry {
  name: string;
  shape: number[];
  thumb?: WeightThumb;
  /** 1-D tensors (norm gains) are kept on the CPU too */
  values?: Float32Array;
}

type Uniform = (number | { f: number })[];

export class Engine {
  readonly weights = new Map<string, WeightEntry>();
  private gpu = new Map<string, GPUBuffer>();
  private embedChunks: { buf: GPUBuffer; r0: number; r1: number }[] = [];
  private pipes: Record<string, GPUComputePipeline> = {};
  private work: Record<string, GPUBuffer> = {};
  private cap = 0;
  private busy = false;
  adapterInfo = "";

  private constructor(readonly device: GPUDevice) {}

  static async create(raw: RawWeights, progress: Progress): Promise<Engine> {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser. Try Chrome or Edge 113+.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const need = 160 * 1024 * 1024;
    if (adapter.limits.maxStorageBufferBindingSize < need)
      console.warn("maxStorageBufferBindingSize is small:", adapter.limits.maxStorageBufferBindingSize);
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    device.lost.then((i) => console.error("WebGPU device lost:", i.message));
    device.addEventListener("uncapturederror", (e) => console.error("WebGPU error:", (e as GPUUncapturedErrorEvent).error.message));
    const eng = new Engine(device);
    const info = adapter.info;
    eng.adapterInfo = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") : "";
    eng.buildPipelines();
    await eng.upload(raw, progress);
    return eng;
  }

  private buildPipelines() {
    const mk = (code: string) =>
      this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      });
    this.pipes = {
      matmul: mk(S.MATMUL),
      rmsnorm: mk(S.RMSNORM),
      rope: mk(S.ROPE),
      scores: mk(S.SCORES),
      softmax: mk(S.SOFTMAX),
      pv: mk(S.PV),
      swiglu: mk(S.SWIGLU),
      add: mk(S.ADD),
      embed: mk(S.EMBED),
    };
  }

  private async upload(raw: RawWeights, progress: Progress) {
    const all = [...raw.tensors.values()];
    let done = 0;
    const totalBytes = all.reduce((a, t) => a + t.length, 0);
    for (const t of all) {
      const u16 = new Uint16Array(raw.bytes.buffer, raw.bytes.byteOffset + t.start, t.length / 2);
      const entry: WeightEntry = { name: t.name, shape: t.shape };
      if (t.shape.length === 1) {
        const f = new Float32Array(u16.length);
        for (let i = 0; i < f.length; i++) f[i] = BF16_LUT[u16[i]];
        entry.values = f;
        this.gpu.set(t.name, this.storage(f));
      } else {
        const [rows, cols] = t.shape;
        entry.thumb = makeThumb(u16, rows, cols);
        if (t.name === "model.embed_tokens.weight") {
          for (let r0 = 0; r0 < rows; r0 += EMBED_CHUNK_ROWS) {
            const r1 = Math.min(rows, r0 + EMBED_CHUNK_ROWS);
            const buf = this.device.createBuffer({
              size: (r1 - r0) * cols * 2,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            this.device.queue.writeBuffer(buf, 0, raw.bytes, t.start + r0 * cols * 2, (r1 - r0) * cols * 2);
            this.embedChunks.push({ buf, r0, r1 });
          }
        } else {
          const buf = this.device.createBuffer({
            size: t.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          });
          this.device.queue.writeBuffer(buf, 0, raw.bytes, t.start, t.length);
          this.gpu.set(t.name, buf);
        }
      }
      this.weights.set(t.name, entry);
      done += t.length;
      progress("Uploading to GPU", done, totalBytes);
      await new Promise((r) => setTimeout(r, 0));
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  private storage(data: Float32Array<ArrayBuffer>): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(16, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  private ensureWork(T: number) {
    if (T <= this.cap) return;
    for (const b of Object.values(this.work)) b.destroy();
    const cap = Math.min(CFG.maxContext, Math.max(32, 1 << Math.ceil(Math.log2(T))));
    const f = 4;
    const mk = (n: number, extra = 0) =>
      this.device.createBuffer({
        size: Math.max(16, n * f),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra,
      });
    this.work = {
      tok: mk(cap),
      cs: mk(cap * 128),
      x: mk(cap * H),
      h: mk(cap * H),
      q: mk(cap * Q_DIM),
      k: mk(cap * KV_DIM),
      v: mk(cap * KV_DIM),
      qn: mk(cap * Q_DIM),
      kn: mk(cap * KV_DIM),
      attn: mk(cap * Q_DIM),
      tmp: mk(cap * H),
      g: mk(cap * I),
      u: mk(cap * I),
      a: mk(cap * I),
      resid: mk((L + 1) * cap * H),
      probs: mk(L * NH * cap * cap),
      detail: mk(L * DETAIL_REC),
      lensIn: mk((L + 1) * H),
      lensNorm: mk((L + 1) * H),
      lensLogits: mk((L + 1) * VOCAB),
    };
    this.cap = cap;
  }

  private uniform(vals: Uniform, trash: GPUBuffer[]): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(16, Math.ceil((vals.length * 4) / 16) * 16),
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    const dv = new DataView(buf.getMappedRange());
    vals.forEach((v, i) => (typeof v === "number" ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v.f, true)));
    buf.unmap();
    trash.push(buf);
    return buf;
  }

  private dispatch(
    enc: GPUCommandEncoder,
    trash: GPUBuffer[],
    pipe: string,
    u: Uniform,
    bufs: GPUBuffer[],
    wx: number,
    wy = 1,
  ) {
    const p = this.pipes[pipe];
    const ub = this.uniform(u, trash);
    const bg = this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [ub, ...bufs].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wx, wy);
    pass.end();
  }

  private matmul(enc: GPUCommandEncoder, trash: GPUBuffer[], X: GPUBuffer, W: GPUBuffer, Y: GPUBuffer, M: number, N: number, K: number, ldy = N, yoff = 0) {
    this.dispatch(enc, trash, "matmul", [M, N, K, ldy, yoff], [X, W, Y], Math.ceil(N / 64), Math.ceil(M / 16));
  }

  private rmsnorm(enc: GPUCommandEncoder, trash: GPUBuffer[], X: GPUBuffer, G: GPUBuffer, Y: GPUBuffer, M: number, D: number) {
    this.dispatch(enc, trash, "rmsnorm", [M, D, { f: CFG.eps }], [X, G, Y], Math.min(M, 65535), Math.ceil(M / 65535));
  }

  private ropeTable(T: number): Float32Array<ArrayBuffer> {
    const cs = new Float32Array(T * 128);
    for (let i = 0; i < 64; i++) {
      const inv = Math.fround(1 / Math.fround(Math.pow(CFG.ropeTheta, Math.fround((2 * i) / 128))));
      for (let t = 0; t < T; t++) {
        const a = Math.fround(t * inv);
        cs[t * 64 + i] = Math.cos(a);
        cs[T * 64 + t * 64 + i] = Math.sin(a);
      }
    }
    return cs;
  }

  /** Run the model over `tokens`, capturing detailed activations at position `focus`. */
  async forward(tokens: number[], focus = tokens.length - 1): Promise<Capture> {
    if (this.busy) throw new Error("forward() already running");
    this.busy = true;
    try {
      return await this.forwardInner(tokens, focus);
    } finally {
      this.busy = false;
    }
  }

  private async forwardInner(tokens: number[], focus: number): Promise<Capture> {
    const T = tokens.length;
    if (T < 1 || T > CFG.maxContext) throw new Error(`sequence length must be 1..${CFG.maxContext}`);
    focus = Math.max(0, Math.min(T - 1, focus));
    const t0 = performance.now();
    this.ensureWork(T);
    const w = this.work;
    const W = (n: string) => {
      const b = this.gpu.get(n);
      if (!b) throw new Error("missing weight " + n);
      return b;
    };
    const q = this.device.queue;
    q.writeBuffer(w.tok, 0, new Uint32Array(tokens));
    q.writeBuffer(w.cs, 0, this.ropeTable(T));

    const trash: GPUBuffer[] = [];
    const enc = this.device.createCommandEncoder();
    const D = (buf: GPUBuffer, dim: number, l: number, field: DetailName) =>
      enc.copyBufferToBuffer(buf, focus * dim * 4, w.detail, (l * DETAIL_REC + DETAIL_OFF[field]) * 4, dim * 4);
    const R = (l: number) => enc.copyBufferToBuffer(w.x, 0, w.resid, l * T * H * 4, T * H * 4);

    for (const c of this.embedChunks)
      this.dispatch(enc, trash, "embed", [T, H, c.r0, c.r1], [w.tok, c.buf, w.x], Math.ceil((T * H) / 2 / 256));
    R(0);

    const scale = 1 / Math.sqrt(CFG.headDim);
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      const probOff = l * NH * T * T;
      this.rmsnorm(enc, trash, w.x, W(p + "input_layernorm.weight"), w.h, T, H);
      D(w.h, H, l, "attnNorm");
      this.matmul(enc, trash, w.h, W(p + "self_attn.q_proj.weight"), w.q, T, Q_DIM, H);
      this.matmul(enc, trash, w.h, W(p + "self_attn.k_proj.weight"), w.k, T, KV_DIM, H);
      this.matmul(enc, trash, w.h, W(p + "self_attn.v_proj.weight"), w.v, T, KV_DIM, H);
      D(w.q, Q_DIM, l, "q");
      D(w.k, KV_DIM, l, "k");
      D(w.v, KV_DIM, l, "v");
      this.rmsnorm(enc, trash, w.q, W(p + "self_attn.q_norm.weight"), w.qn, T * NH, CFG.headDim);
      this.rmsnorm(enc, trash, w.k, W(p + "self_attn.k_norm.weight"), w.kn, T * NKV, CFG.headDim);
      this.dispatch(enc, trash, "rope", [T, NH, T * 64], [w.qn, w.cs], Math.ceil((T * NH * 64) / 64));
      this.dispatch(enc, trash, "rope", [T, NKV, T * 64], [w.kn, w.cs], Math.ceil((T * NKV * 64) / 64));
      D(w.qn, Q_DIM, l, "qr");
      D(w.kn, KV_DIM, l, "kr");
      this.dispatch(enc, trash, "scores", [T, NH, probOff, { f: scale }], [w.qn, w.kn, w.probs], Math.ceil((NH * T * T) / 64));
      this.dispatch(enc, trash, "softmax", [T, NH, probOff], [w.probs], Math.ceil((NH * T) / 64));
      this.dispatch(enc, trash, "pv", [T, NH, probOff], [w.probs, w.v, w.attn], NH, T);
      D(w.attn, Q_DIM, l, "attn");
      this.matmul(enc, trash, w.attn, W(p + "self_attn.o_proj.weight"), w.tmp, T, H, Q_DIM);
      D(w.tmp, H, l, "o");
      this.dispatch(enc, trash, "add", [T * H], [w.x, w.tmp], Math.ceil((T * H) / 256));
      D(w.x, H, l, "mid");
      this.rmsnorm(enc, trash, w.x, W(p + "post_attention_layernorm.weight"), w.h, T, H);
      D(w.h, H, l, "mlpNorm");
      this.matmul(enc, trash, w.h, W(p + "mlp.gate_proj.weight"), w.g, T, I, H);
      this.matmul(enc, trash, w.h, W(p + "mlp.up_proj.weight"), w.u, T, I, H);
      this.dispatch(enc, trash, "swiglu", [T * I], [w.g, w.u, w.a], Math.ceil((T * I) / 256));
      D(w.g, I, l, "gate");
      D(w.u, I, l, "up");
      D(w.a, I, l, "act");
      this.matmul(enc, trash, w.a, W(p + "mlp.down_proj.weight"), w.tmp, T, H, I);
      D(w.tmp, H, l, "down");
      this.dispatch(enc, trash, "add", [T * H], [w.x, w.tmp], Math.ceil((T * H) / 256));
      R(l + 1);
    }

    // Logit lens: final norm + unembedding applied to every residual checkpoint at the focus position.
    for (let l = 0; l <= L; l++)
      enc.copyBufferToBuffer(w.resid, (l * T + focus) * H * 4, w.lensIn, l * H * 4, H * 4);
    this.rmsnorm(enc, trash, w.lensIn, W("model.norm.weight"), w.lensNorm, L + 1, H);
    for (const c of this.embedChunks)
      this.matmul(enc, trash, w.lensNorm, c.buf, w.lensLogits, L + 1, c.r1 - c.r0, H, VOCAB, c.r0);

    // Read everything back in one staging buffer.
    const parts = {
      resid: (L + 1) * T * H,
      probs: L * NH * T * T,
      detail: L * DETAIL_REC,
      lensLogits: (L + 1) * VOCAB,
    };
    const total = Object.values(parts).reduce((a, b) => a + b, 0);
    const staging = this.device.createBuffer({ size: total * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    let off = 0;
    const offs: Record<string, number> = {};
    for (const [k, n] of Object.entries(parts)) {
      offs[k] = off;
      enc.copyBufferToBuffer(w[k], 0, staging, off * 4, n * 4);
      off += n;
    }
    q.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    for (const b of trash) b.destroy();

    const view = (k: keyof typeof parts) => all.subarray(offs[k], offs[k] + parts[k]);
    const detail = view("detail");
    const layers: LayerDetail[] = [];
    for (let l = 0; l < L; l++) {
      const rec = {} as LayerDetail;
      for (const [n, s] of DETAIL_FIELDS) rec[n] = detail.subarray(l * DETAIL_REC + DETAIL_OFF[n], l * DETAIL_REC + DETAIL_OFF[n] + s);
      layers.push(rec);
    }
    const lensLogits = view("lensLogits");
    const lens: TopToken[][] = [];
    for (let l = 0; l <= L; l++) lens.push(topK(lensLogits.subarray(l * VOCAB, (l + 1) * VOCAB), 10));

    return {
      tokens: [...tokens],
      T,
      focus,
      resid: view("resid"),
      probs: view("probs"),
      layers,
      lens,
      logits: lensLogits.subarray(L * VOCAB, (L + 1) * VOCAB),
      ms: performance.now() - t0,
    };
  }

  /** Read actual weight values (rows r0..r1, all columns) back from the GPU. */
  async readWeightRows(name: string, r0: number, r1: number): Promise<Float32Array> {
    const e = this.weights.get(name);
    if (!e || e.shape.length !== 2) throw new Error("not a matrix: " + name);
    const cols = e.shape[1];
    r0 = Math.max(0, r0);
    r1 = Math.min(e.shape[0], r1);
    // (buffer, first row within buffer, row count) pieces; the embedding is split across chunks
    const pieces: [GPUBuffer, number, number][] = [];
    if (name === "model.embed_tokens.weight") {
      for (const c of this.embedChunks) {
        const a = Math.max(r0, c.r0);
        const b = Math.min(r1, c.r1);
        if (a < b) pieces.push([c.buf, a - c.r0, b - a]);
      }
    } else pieces.push([this.gpu.get(name)!, r0, r1 - r0]);
    const bytes = (r1 - r0) * cols * 2;
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    let off = 0;
    for (const [buf, row, n] of pieces) {
      enc.copyBufferToBuffer(buf, row * cols * 2, staging, off, n * cols * 2);
      off += n * cols * 2;
    }
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = BF16_LUT[u16[i]];
    return out;
  }
}

export function topK(logits: Float32Array, k: number): TopToken[] {
  const ids: number[] = [];
  const vals: number[] = [];
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (v > max) max = v;
    if (ids.length < k || v > vals[k - 1]) {
      let j = Math.min(ids.length, k - 1);
      if (ids.length < k) {
        ids.push(0);
        vals.push(0);
        j = ids.length - 1;
      }
      while (j > 0 && vals[j - 1] < v) {
        ids[j] = ids[j - 1];
        vals[j] = vals[j - 1];
        j--;
      }
      ids[j] = i;
      vals[j] = v;
    }
  }
  let z = 0;
  for (let i = 0; i < logits.length; i++) z += Math.exp(logits[i] - max);
  return ids.map((id, i) => ({ id, logit: vals[i], prob: Math.exp(vals[i] - max) / z }));
}

/** Sample from logits with temperature / top-k. temperature 0 = greedy. */
export function sample(logits: Float32Array, temperature: number, k = 40): number {
  const top = topK(logits, temperature <= 0 ? 1 : k);
  if (temperature <= 0) return top[0].id;
  const m = top[0].logit;
  const ws = top.map((t) => Math.exp((t.logit - m) / temperature));
  let r = Math.random() * ws.reduce((a, b) => a + b, 0);
  for (let i = 0; i < ws.length; i++) {
    r -= ws[i];
    if (r <= 0) return top[i].id;
  }
  return top[0].id;
}
