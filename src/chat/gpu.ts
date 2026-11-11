// WebGPU forward pass (with KV cache) for the nanoGPT ladder models. Same math as gpt.ts, all f32.
//
// Decoding is memory-bound (every weight is read once per token), so the matmul kernel is a plain
// coalesced GEMV: 32 lanes per output row, vec4 loads. Prompts are processed in chunks of up to
// T_MAX tokens with the same kernel (one workgroup row per token). Per token there are ~8 dispatches
// per layer, recorded into a single command buffer, plus one logits readback.

import type { Tensor } from "./torchzip";
import { CONTEXT, type GPTConfig } from "./gpt";

const T_MAX = 128; // prompt chunk size
const EPS = 1e-6;

const MATMUL = /* wgsl */ `
struct P { nOut: u32, nIn: u32, mode: u32, alpha: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

fn erf(v: f32) -> f32 {
  let z = abs(v);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, v >= 0.0);
}

// y[t, row] = x[t, :] . w[row, :]; mode 0: store, 1: store gelu(.), 2: y += alpha * (.)
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x * 8u + lid.x / 32u;
  let lane = lid.x % 32u;
  let t = wg.y;
  let n4 = p.nIn / 4u;
  var s = 0.0;
  if (row < p.nOut) {
    let wo = row * n4;
    let xo = t * n4;
    for (var i = lane; i < n4; i += 32u) { s += dot(x[xo + i], w[wo + i]); }
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 16u; off > 0u; off >>= 1u) {
    if (lane < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < p.nOut) {
    var v = red[lid.x];
    let idx = t * p.nOut + row;
    if (p.mode == 1u) { v = 0.5 * v * (1.0 + erf(v * 0.70710678118654752)); }
    else if (p.mode == 2u) { v = y[idx] + p.alpha * v; }
    y[idx] = v;
  }
}`;

const RMSNORM = /* wgsl */ `
struct P { d: u32, inRow: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let xo = (p.inRow + wg.x) * p.d;
  let yo = wg.x * p.d;
  var s = 0.0;
  for (var i = lid.x; i < p.d; i += 256u) { let v = x[xo + i]; s += v * v; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.d) + ${EPS});
  for (var i = lid.x; i < p.d; i += 256u) { y[yo + i] = x[xo + i] * r; }
}`;

const EMBED = /* wgsl */ `
struct P { d: u32, T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tok: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.y;
  for (var i = g.x; i < p.d; i += 256u) { x[t * p.d + i] = wte[tok[t] * p.d + i]; }
}`;

// Rotary on q (in place) and k, then write k/v into the cache at positions start..start+T-1.
const ROPE = /* wgsl */ `
struct P { start: u32, d: u32, nHead: u32, half: u32, T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@group(0) @binding(4) var<storage, read> cs: array<vec2<f32>>; // [CONTEXT, half] (cos, sin)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = p.T * p.nHead * p.half;
  if (g.x >= n) { return; }
  let i = g.x % p.half;
  let h = (g.x / p.half) % p.nHead;
  let t = g.x / (p.half * p.nHead);
  let pos = p.start + t;
  let c = cs[pos * p.half + i];
  let hd = 2u * p.half;
  let qo = t * 3u * p.d + h * hd;
  let ko = qo + p.d;
  let vo = qo + 2u * p.d;
  let co = pos * p.d + h * hd;
  let q1 = qkv[qo + i]; let q2 = qkv[qo + p.half + i];
  qkv[qo + i] = q1 * c.x + q2 * c.y;
  qkv[qo + p.half + i] = -q1 * c.y + q2 * c.x;
  let k1 = qkv[ko + i]; let k2 = qkv[ko + p.half + i];
  kc[co + i] = k1 * c.x + k2 * c.y;
  kc[co + p.half + i] = -k1 * c.y + k2 * c.x;
  vc[co + i] = qkv[vo + i];
  vc[co + p.half + i] = qkv[vo + p.half + i];
}`;

// One workgroup per (head, query token): causal softmax attention over the KV cache.
const ATTN = /* wgsl */ `
struct P { start: u32, d: u32, headDim: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> sc: array<f32, ${CONTEXT}>;
var<workgroup> q: array<f32, 256>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let t = wg.y;
  let n = p.start + t + 1u;
  let hoff = h * p.headDim;
  for (var i = lid.x; i < p.headDim; i += 64u) { q[i] = qkv[t * 3u * p.d + hoff + i]; }
  workgroupBarrier();
  var m = -3.0e38;
  for (var s = lid.x; s < n; s += 64u) {
    let ko = s * p.d + hoff;
    var dot = 0.0;
    for (var i = 0u; i < p.headDim; i++) { dot += q[i] * kc[ko + i]; }
    dot *= p.scale;
    sc[s] = dot;
    m = max(m, dot);
  }
  red[lid.x] = m;
  workgroupBarrier();
  for (var off = 32u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] = max(red[lid.x], red[lid.x + off]); }
    workgroupBarrier();
  }
  m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var s = lid.x; s < n; s += 64u) { let e = exp(sc[s] - m); sc[s] = e; sum += e; }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var off = 32u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var i = lid.x; i < p.headDim; i += 64u) {
    var acc = 0.0;
    for (var s = 0u; s < n; s++) { acc += sc[s] * vc[s * p.d + hoff + i]; }
    y[t * p.d + hoff + i] = acc * inv;
  }
}`;

interface Op {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  uniform: GPUBuffer;
  /** uniform contents and workgroup counts for a chunk of T tokens starting at `start` */
  plan: (T: number, start: number) => { u: ArrayBuffer; wg: [number, number] };
}

export class GpuGPT {
  pos = 0;
  readonly backend = "WebGPU";
  private ops: Op[] = [];
  private headOps: Op[] = []; // final norm + lm_head (last chunk only)
  private tokBuf!: GPUBuffer;
  private logitsBuf!: GPUBuffer;
  private readBuf!: GPUBuffer;

  private constructor(
    readonly cfg: GPTConfig,
    readonly nVocab: number,
    private device: GPUDevice,
  ) {}

  get params(): number {
    const d = this.cfg.n_embd;
    return this.cfg.vocab_size * d + this.cfg.n_layer * 12 * d * d;
  }

  static async create(cfg: GPTConfig, sd: Record<string, Tensor>, nVocab: number): Promise<GpuGPT> {
    if (!("gpu" in navigator)) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const g = new GpuGPT(cfg, nVocab, device);
    g.build(sd);
    return g;
  }

  private build(sd: Record<string, Tensor>) {
    const dev = this.device;
    const { n_embd: d, n_head: nHead, n_layer: L, vocab_size: V } = this.cfg;
    const hd = d / nHead;
    const half = hd / 2;
    if (d % 128 || hd > 256) throw new Error(`unsupported shape d=${d}, head_dim=${hd}`);
    const S = GPUBufferUsage.STORAGE;

    const upload = (data: Float32Array) => {
      const b = dev.createBuffer({ size: data.byteLength, usage: S, mappedAtCreation: true });
      new Float32Array(b.getMappedRange()).set(data);
      b.unmap();
      return b;
    };
    const get = (name: string, shape: number[]) => {
      const t = sd[name];
      if (!t) throw new Error(`checkpoint is missing ${name}`);
      if (t.shape.join() !== shape.join()) throw new Error(`${name}: expected [${shape}], got [${t.shape}]`);
      return upload(t.data);
    };
    const buf = (floats: number, usage = S) => dev.createBuffer({ size: floats * 4, usage });

    const wte = get("wte.weight", [V, d]);
    const x = buf(T_MAX * d);
    const h = buf(T_MAX * d);
    const qkv = buf(T_MAX * 3 * d);
    const y = buf(T_MAX * d);
    const f = buf(T_MAX * 4 * d);
    const last = buf(d);
    this.tokBuf = dev.createBuffer({ size: T_MAX * 4, usage: S | GPUBufferUsage.COPY_DST });
    this.logitsBuf = buf(this.nVocab, S | GPUBufferUsage.COPY_SRC);
    this.readBuf = dev.createBuffer({ size: this.nVocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // rotary table, angle computed in f32 like torch
    const invFreq = sd["h.0.attn.rotary.inv_freq"]?.data ?? Float32Array.from({ length: half }, (_, i) => 1 / 10000 ** ((2 * i) / hd));
    const csTab = new Float32Array(CONTEXT * half * 2);
    for (let t = 0; t < CONTEXT; t++) {
      for (let i = 0; i < half; i++) {
        const a = Math.fround(t * invFreq[i]);
        csTab[(t * half + i) * 2] = Math.cos(a);
        csTab[(t * half + i) * 2 + 1] = Math.sin(a);
      }
    }
    const cs = upload(csTab);

    const pipe = (code: string) =>
      dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    const P = { matmul: pipe(MATMUL), norm: pipe(RMSNORM), embed: pipe(EMBED), rope: pipe(ROPE), attn: pipe(ATTN) };

    const op = (pipeline: GPUComputePipeline, buffers: GPUBuffer[], plan: Op["plan"]): Op => {
      const uniform = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bind = dev.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      return { pipeline, bind, uniform, plan };
    };
    const u32s = (...v: number[]) => new Uint32Array(v).buffer;
    const matmul = (inp: GPUBuffer, w: GPUBuffer, out: GPUBuffer, nOut: number, nIn: number, mode: number, alpha = 0) =>
      op(P.matmul, [inp, w, out], (T) => {
        const u = new ArrayBuffer(16);
        new Uint32Array(u, 0, 3).set([nOut, nIn, mode]);
        new Float32Array(u, 12, 1)[0] = alpha;
        return { u, wg: [Math.ceil(nOut / 8), T] };
      });
    const norm = (inp: GPUBuffer, out: GPUBuffer, lastOnly = false) =>
      op(P.norm, [inp, out], (T) => ({ u: u32s(d, lastOnly ? T - 1 : 0), wg: [lastOnly ? 1 : T, 1] }));

    const attnScale = 1 / Math.sqrt(2 * L);
    this.ops.push(op(P.embed, [this.tokBuf, wte, x], (T) => ({ u: u32s(d, T), wg: [1, T] })));
    for (let l = 0; l < L; l++) {
      const p = `h.${l}.`;
      const wQkv = get(p + "attn.c_attn.weight", [3 * d, d]);
      const wProj = get(p + "attn.c_proj.weight", [d, d]);
      const wFc = get(p + "mlp.c_fc.weight", [4 * d, d]);
      const wFcProj = get(p + "mlp.c_proj.weight", [d, 4 * d]);
      const kc = buf(CONTEXT * d);
      const vc = buf(CONTEXT * d);
      this.ops.push(
        norm(x, h),
        matmul(h, wQkv, qkv, 3 * d, d, 0),
        op(P.rope, [qkv, kc, vc, cs], (T, start) => ({ u: u32s(start, d, nHead, half, T), wg: [Math.ceil((T * nHead * half) / 64), 1] })),
        op(P.attn, [qkv, kc, vc, y], (_T, start) => {
          const u = new ArrayBuffer(16);
          new Uint32Array(u, 0, 3).set([start, d, hd]);
          new Float32Array(u, 12, 1)[0] = 1 / Math.sqrt(hd);
          return { u, wg: [nHead, _T] };
        }),
        matmul(y, wProj, x, d, d, 2, attnScale),
        norm(x, h),
        matmul(h, wFc, f, 4 * d, d, 1),
        matmul(f, wFcProj, x, d, 4 * d, 2, 1),
      );
    }
    this.headOps.push(norm(x, last, true), matmul(last, wte, this.logitsBuf, this.nVocab, d, 0));
  }

  /** Free all GPU memory (the engine is unusable afterwards). */
  destroy() {
    this.device.destroy();
  }

  /** Append `tokens` to the KV cache; returns next-token logits after the last one. */
  async forward(tokens: number[]): Promise<Float32Array> {
    if (tokens.length === 0) throw new Error("forward() needs at least one token");
    if (this.pos + tokens.length > CONTEXT) throw new Error(`context overflow (${this.pos + tokens.length} > ${CONTEXT})`);
    const dev = this.device;
    for (let c = 0; c < tokens.length; c += T_MAX) {
      const chunk = tokens.slice(c, c + T_MAX);
      const T = chunk.length;
      const isLast = c + T_MAX >= tokens.length;
      dev.queue.writeBuffer(this.tokBuf, 0, new Uint32Array(chunk));
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const o of isLast ? [...this.ops, ...this.headOps] : this.ops) {
        const { u, wg } = o.plan(T, this.pos);
        dev.queue.writeBuffer(o.uniform, 0, u);
        pass.setPipeline(o.pipeline);
        pass.setBindGroup(0, o.bind);
        pass.dispatchWorkgroups(wg[0], wg[1]);
      }
      pass.end();
      if (isLast) enc.copyBufferToBuffer(this.logitsBuf, 0, this.readBuf, 0, this.nVocab * 4);
      dev.queue.submit([enc.finish()]);
      this.pos += T;
    }
    await this.readBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readBuf.getMappedRange().slice(0));
    this.readBuf.unmap();
    return out;
  }
}
