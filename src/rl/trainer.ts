// In-browser RL fine-tuning of a nanoGPT ladder model with WebGPU.
//
// - generate(): batched sampling of B completions (one prompt per sequence, all the same token length, so
//   every sequence sits at the same positions; KV cache).
// - trainStep(): REINFORCE with group-normalized advantages (GRPO without clipping / KL; one on-policy
//   step per batch): loss = -sum_tokens A_b * log p(token) / n_tokens, then global-norm clip + AdamW.
//
// The same forward graph serves both: training runs it over [B, T] rows from position 0 and keeps every
// layer's activations for the backward pass; generation runs it over [B, 1] rows at the current position.

import type { Tensor } from "../chat/torchzip";
import type { GPTConfig } from "../chat/gpt";
import * as K from "./kernels";

export const VOCAB = 50257; // real GPT-2 vocabulary: sampling and the RL loss are restricted to it
export const EOT = 50256;
const RC = 128; // lm_head rows per chunk in the loss (logits chunk = RC x VOCAB floats)

const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

function pack(spec: string, ...vals: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(Math.ceil((spec.length * 4) / 16) * 16);
  const dv = new DataView(buf);
  for (let i = 0; i < spec.length; i++) {
    if (spec[i] === "f") dv.setFloat32(i * 4, vals[i], true);
    else dv.setUint32(i * 4, vals[i], true);
  }
  return buf;
}

type Plan = { u: ArrayBuffer; wg: [number, number?, number?] } | null;
interface Op {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  uniform: GPUBuffer;
  plan: () => Plan;
}
type Step = Op | { clear: GPUBuffer };

interface Param {
  name: string;
  n: number;
  w: GPUBuffer;
  g: GPUBuffer;
  m: GPUBuffer;
  v: GPUBuffer;
  init: Float32Array;
}

interface LayerActs {
  xmid: GPUBuffer; h1: GPUBuffer; qkv: GPUBuffer; y: GPUBuffer; lse: GPUBuffer;
  h2: GPUBuffer; f: GPUBuffer; g: GPUBuffer; kc: GPUBuffer; vc: GPUBuffer;
}

export interface TrainStats {
  gradNorm: number;
  meanLogp: number;
  entropy: number;
  tokens: number;
}

export class Trainer {
  readonly d: number;
  readonly nHead: number;
  readonly hd: number;
  readonly L: number;
  private params: Param[] = [];
  private wte!: Param;
  private layers!: { qkv: Param; proj: Param; fc: Param; fcp: Param }[];
  private adamStep = 0;
  private pipes = new Map<string, GPUComputePipeline>();
  // workspace (rebuilt when B * T capacity grows)
  private cap = { B: 0, T: 0 };
  private ws: GPUBuffer[] = [];
  private fwd: Step[] = [];
  private genHead: Step[] = [];
  private loss: Step[] = [];
  private bwd: Step[] = [];
  private optim: Step[] = [];
  private bufs!: Record<string, GPUBuffer>;
  private acts!: LayerActs[];
  private X!: GPUBuffer[]; // residual stream entering each layer (X[L] = final)
  /** current shape for plans */
  private s = { B: 1, T: 1, start: 0, nAct: 0, nGroups: 0, ent: 0, lr: 1e-4, maxNorm: 1, b1: 0.9, b2: 0.999, wd: 0 };

  private constructor(readonly cfg: GPTConfig, private dev: GPUDevice) {
    this.d = cfg.n_embd;
    this.nHead = cfg.n_head;
    this.hd = cfg.n_embd / cfg.n_head;
    this.L = cfg.n_layer;
    if (this.hd > 64 || this.d % 4) throw new Error(`unsupported head_dim ${this.hd}`);
  }

  static async create(cfg: GPTConfig, sd: Record<string, Tensor>): Promise<Trainer> {
    if (!("gpu" in navigator)) throw new Error("This page needs WebGPU (Chrome, Edge or Safari 26+).");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const dev = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
      },
    });
    const t = new Trainer(cfg, dev);
    t.initParams(sd);
    return t;
  }

  destroy() {
    this.dev.destroy();
  }

  // ------------------------------------------------------------------ setup

  private buffer(floats: number, usage = S) {
    return this.dev.createBuffer({ size: Math.max(16, Math.ceil(floats) * 4), usage });
  }

  private initParams(sd: Record<string, Tensor>) {
    const { d, L } = this;
    const get = (name: string, shape: number[]): Param => {
      const t = sd[name];
      if (!t) throw new Error(`checkpoint is missing ${name}`);
      if (t.shape.join() !== shape.join()) throw new Error(`${name}: expected [${shape}], got [${t.shape}]`);
      const n = t.data.length;
      const p: Param = { name, n, w: this.buffer(n), g: this.buffer(n), m: this.buffer(n), v: this.buffer(n), init: t.data };
      this.params.push(p);
      return p;
    };
    this.wte = get("wte.weight", [this.cfg.vocab_size, d]);
    this.layers = [];
    for (let l = 0; l < L; l++) {
      const p = `h.${l}.`;
      this.layers.push({
        qkv: get(p + "attn.c_attn.weight", [3 * d, d]),
        proj: get(p + "attn.c_proj.weight", [d, d]),
        fc: get(p + "mlp.c_fc.weight", [4 * d, d]),
        fcp: get(p + "mlp.c_proj.weight", [d, 4 * d]),
      });
    }
    this.reset();
  }

  /** Restore the original checkpoint weights and clear the optimizer state. */
  reset() {
    for (const p of this.params) {
      this.dev.queue.writeBuffer(p.w, 0, p.init as Float32Array<ArrayBuffer>);
      const enc = this.dev.createCommandEncoder();
      enc.clearBuffer(p.m);
      enc.clearBuffer(p.v);
      this.dev.queue.submit([enc.finish()]);
    }
    this.adamStep = 0;
  }

  private pipe(name: string, code: string) {
    let p = this.pipes.get(name);
    if (!p) {
      p = this.dev.createComputePipeline({
        layout: "auto",
        compute: { module: this.dev.createShaderModule({ code }), entryPoint: "main" },
      });
      this.pipes.set(name, p);
    }
    return p;
  }

  private op(pipeline: GPUComputePipeline, buffers: GPUBuffer[], plan: () => Plan): Op {
    const uniform = this.dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bind = this.dev.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.ws.push(uniform);
    return { pipeline, bind, uniform, plan };
  }

  /** Make sure activations for B sequences of up to T tokens fit; rebuilds buffers and graphs if needed. */
  ensure(B: number, T: number) {
    if (B <= this.cap.B && T <= this.cap.T) return;
    for (const b of this.ws) b.destroy();
    this.ws = [];
    this.pipes.clear(); // attention kernels bake the context size in
    const cap = { B: Math.max(B, this.cap.B), T: Math.max(T, this.cap.T) };
    this.cap = cap;
    const { d, L, nHead } = this;
    const R = cap.B * cap.T;
    const nActMax = R;
    const ws = (n: number) => {
      const b = this.buffer(n);
      this.ws.push(b);
      return b;
    };
    this.X = Array.from({ length: L + 1 }, () => ws(R * d));
    this.acts = Array.from({ length: L }, () => ({
      xmid: ws(R * d), h1: ws(R * d), qkv: ws(R * 3 * d), y: ws(R * d), lse: ws(R * nHead),
      h2: ws(R * d), f: ws(R * 4 * d), g: ws(R * 4 * d), kc: ws(R * d), vc: ws(R * d),
    }));
    const half = this.hd / 2;
    const cs = new Float32Array(cap.T * half * 2);
    const invFreq = Array.from({ length: half }, (_, i) => 1 / 10000 ** ((2 * i) / this.hd));
    for (let t = 0; t < cap.T; t++) {
      for (let i = 0; i < half; i++) {
        const a = Math.fround(t * Math.fround(invFreq[i]));
        cs[(t * half + i) * 2] = Math.cos(a);
        cs[(t * half + i) * 2 + 1] = Math.sin(a);
      }
    }
    this.bufs = {
      tok: ws(R), cs: ws(cs.length), o: ws(R * d), hf: ws(R * d),
      idx: ws(nActMax), tgt: ws(nActMax), coef: ws(nActMax), stats: ws(nActMax * 2),
      act: ws(nActMax * d), dact: ws(nActMax * d), logits: ws(RC * VOCAB),
      readLogits: (() => {
        const b = this.dev.createBuffer({ size: cap.B * VOCAB * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.ws.push(b);
        return b;
      })(),
      dX: ws(R * d), dH: ws(R * d), dG: ws(R * 4 * d), dY: ws(R * d), dqkv: ws(R * 3 * d), D: ws(R * nHead),
      groups: ws(R * 3), rows: ws(R), partial: ws(256 * this.params.length), norm: ws(4),
      readStats: (() => {
        const b = this.dev.createBuffer({ size: (nActMax * 2 + 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.ws.push(b);
        return b;
      })(),
    };
    this.dev.queue.writeBuffer(this.bufs.cs, 0, cs);
    this.build();
  }

  // ------------------------------------------------------------------ graphs

  private build() {
    const { d, L, nHead, hd } = this;
    const half = hd / 2;
    const ctx = this.cap.T;
    const s = this.s;
    const B_ = this.bufs;
    const P = {
      gemm: this.pipe("gemm", K.GEMM),
      embed: this.pipe("embed", K.EMBED),
      norm: this.pipe("norm", K.RMSNORM),
      normBwd: this.pipe("normBwd", K.RMSNORM_BWD),
      rope: this.pipe("rope", K.ROPE),
      ropeBwd: this.pipe("ropeBwd", K.ROPE_BWD),
      attn: this.pipe("attn", K.ATTN(ctx)),
      attnD: this.pipe("attnD", K.ATTN_D),
      attnDq: this.pipe("attnDq", K.ATTN_DQ(ctx)),
      attnDkv: this.pipe("attnDkv", K.ATTN_DKV(ctx)),
      axpy: this.pipe("axpy", K.AXPY),
      gelu: this.pipe("gelu", K.GELU),
      geluBwd: this.pipe("geluBwd", K.GELU_BWD),
      gather: this.pipe("gather", K.GATHER),
      scatter: this.pipe("scatter", K.SCATTER),
      xent: this.pipe("xent", K.XENT),
      embedBwd: this.pipe("embedBwd", K.EMBED_BWD),
      sumsq: this.pipe("sumsq", K.SUMSQ),
      normFinal: this.pipe("normFinal", K.NORM_FINAL),
      adam: this.pipe("adam", K.ADAMW),
    };
    const R = () => s.B * s.T;
    const grid = (n: number): [number] => [Math.max(1, Math.min(4096, Math.ceil(n / 256)))];
    const gemmU = (M: number, N: number, Kd: number, sam: number, sak: number, sbk: number, sbn: number,
      aOff = 0, bOff = 0, cOff = 0, alpha = 1, beta = 0): Plan => ({
      u: pack("uuuuuuuuuuff", M, N, Kd, sam, sak, sbk, sbn, aOff, bOff, cOff, alpha, beta),
      wg: [Math.ceil(N / 64), Math.ceil(M / 64)],
    });
    // Y[R, N] (+)= alpha * X[R, Kd] @ W[N, Kd]^T
    const linear = (x: GPUBuffer, w: GPUBuffer, y: GPUBuffer, N: number, Kd: number, alpha = 1, beta = 0) =>
      this.op(P.gemm, [x, w, y], () => gemmU(R(), N, Kd, Kd, 1, 1, Kd, 0, 0, 0, alpha, beta));
    // dX[R, Kd] (+)= alpha * dY[R, N] @ W[N, Kd]
    const linearDx = (dy: GPUBuffer, w: GPUBuffer, dx: GPUBuffer, N: number, Kd: number, alpha = 1) =>
      this.op(P.gemm, [dy, w, dx], () => gemmU(R(), Kd, N, N, 1, Kd, 1, 0, 0, 0, alpha, 0));
    // dW[N, Kd] += alpha * dY[R, N]^T @ X[R, Kd]
    const linearDw = (dy: GPUBuffer, x: GPUBuffer, dw: GPUBuffer, N: number, Kd: number, alpha = 1) =>
      this.op(P.gemm, [dy, x, dw], () => gemmU(N, Kd, R(), 1, N, Kd, 1, 0, 0, 0, alpha, 1));
    const rmsnorm = (x: GPUBuffer, y: GPUBuffer) => this.op(P.norm, [x, y], () => ({ u: pack("u", d), wg: [R()] }));
    const rmsnormBwd = (x: GPUBuffer, dy: GPUBuffer, dx: GPUBuffer, acc: boolean) =>
      this.op(P.normBwd, [x, dy, dx], () => ({ u: pack("uu", d, acc ? 1 : 0), wg: [R()] }));
    const axpy = (x: GPUBuffer, y: GPUBuffer, o: GPUBuffer, a: number) =>
      this.op(P.axpy, [x, y, o], () => ({ u: pack("uf", R() * d, a), wg: grid(R() * d) }));
    const attnScale = 1 / Math.sqrt(2 * L);
    const scale = 1 / Math.sqrt(hd);

    // ---- forward over s.B x s.T rows starting at position s.start
    this.fwd = [this.op(P.embed, [B_.tok, this.wte.w, this.X[0]], () => ({ u: pack("u", d), wg: [R()] }))];
    for (let l = 0; l < L; l++) {
      const A = this.acts[l];
      const W = this.layers[l];
      this.fwd.push(
        rmsnorm(this.X[l], A.h1),
        linear(A.h1, W.qkv.w, A.qkv, 3 * d, d),
        this.op(P.rope, [A.qkv, A.kc, A.vc, B_.cs], () => ({
          u: pack("uuuuuuu", R(), s.T, s.start, d, nHead, half, ctx), wg: grid(R() * nHead * half),
        })),
        this.op(P.attn, [A.qkv, A.kc, A.vc, A.y, A.lse], () => ({
          u: pack("uuuuuuf", s.T, s.start, d, hd, nHead, ctx, scale), wg: [nHead, s.T, s.B],
        })),
        linear(A.y, W.proj.w, B_.o, d, d),
        axpy(this.X[l], B_.o, A.xmid, attnScale),
        rmsnorm(A.xmid, A.h2),
        linear(A.h2, W.fc.w, A.f, 4 * d, d),
        this.op(P.gelu, [A.f, A.g], () => ({ u: pack("u", R() * 4 * d), wg: grid(R() * 4 * d) })),
        linear(A.g, W.fcp.w, B_.o, d, 4 * d),
        axpy(A.xmid, B_.o, this.X[l + 1], 1),
      );
    }
    this.fwd.push(rmsnorm(this.X[L], B_.hf));

    // ---- generation head: logits for rows idx[0..B) -> logits[B, VOCAB]
    this.genHead = [
      this.op(P.gather, [B_.idx, B_.hf, B_.act], () => ({ u: pack("uu", d, s.B), wg: [s.B] })),
      this.op(P.gemm, [B_.act, this.wte.w, B_.logits], () => gemmU(s.B, VOCAB, d, d, 1, 1, d)),
    ];

    // ---- loss head over the active rows, in chunks of RC rows
    this.loss = [
      { clear: B_.dact },
      this.op(P.gather, [B_.idx, B_.hf, B_.act], () => ({ u: pack("uu", d, s.nAct), wg: [s.nAct] })),
    ];
    const nChunks = Math.ceil((this.cap.B * this.cap.T) / RC);
    for (let c = 0; c < nChunks; c++) {
      const rows = () => Math.min(RC, s.nAct - c * RC);
      const live = (f: () => Plan) => () => (rows() > 0 ? f() : null);
      this.loss.push(
        // logits = act[chunk] @ wte^T
        this.op(P.gemm, [B_.act, this.wte.w, B_.logits], live(() => gemmU(rows(), VOCAB, d, d, 1, 1, d, c * RC * d))),
        this.op(P.xent, [B_.logits, B_.tgt, B_.coef, B_.stats], live(() => ({ u: pack("uuf", VOCAB, c * RC, s.ent), wg: [rows()] }))),
        // dact[chunk] = dlogits @ wte
        this.op(P.gemm, [B_.logits, this.wte.w, B_.dact], live(() => gemmU(rows(), d, VOCAB, VOCAB, 1, d, 1, 0, 0, c * RC * d))),
        // dwte += dlogits^T @ act[chunk]
        this.op(P.gemm, [B_.logits, B_.act, this.wte.g], live(() => gemmU(VOCAB, d, rows(), 1, VOCAB, d, 1, 0, c * RC * d, 0, 1, 1))),
      );
    }

    // ---- backward through the trunk
    this.bwd = [
      { clear: B_.dH },
      this.op(P.scatter, [B_.idx, B_.dact, B_.dH], () => ({ u: pack("uu", d, s.nAct), wg: [s.nAct] })),
      rmsnormBwd(this.X[L], B_.dH, B_.dX, false),
    ];
    for (let l = L - 1; l >= 0; l--) {
      const A = this.acts[l];
      const W = this.layers[l];
      this.bwd.push(
        // MLP
        linearDx(B_.dX, W.fcp.w, B_.dG, d, 4 * d),
        linearDw(B_.dX, A.g, W.fcp.g, d, 4 * d),
        this.op(P.geluBwd, [A.f, B_.dG], () => ({ u: pack("u", R() * 4 * d), wg: grid(R() * 4 * d) })),
        linearDx(B_.dG, W.fc.w, B_.dH, 4 * d, d),
        linearDw(B_.dG, A.h2, W.fc.g, 4 * d, d),
        rmsnormBwd(A.xmid, B_.dH, B_.dX, true),
        // attention (output branch is scaled by attnScale)
        linearDx(B_.dX, W.proj.w, B_.dY, d, d, attnScale),
        linearDw(B_.dX, A.y, W.proj.g, d, d, attnScale),
        this.op(P.attnD, [B_.dY, A.y, B_.D], () => ({ u: pack("uuuu", R(), d, hd, nHead), wg: grid(R() * nHead) })),
        this.op(P.attnDq, [A.qkv, A.kc, A.vc, B_.dY, A.lse, B_.D, B_.dqkv], () => ({
          u: pack("uuuuuf", s.T, d, hd, nHead, ctx, scale), wg: [nHead, s.T, s.B],
        })),
        this.op(P.attnDkv, [A.qkv, A.kc, A.vc, B_.dY, A.lse, B_.D, B_.dqkv], () => ({
          u: pack("uuuuuf", s.T, d, hd, nHead, ctx, scale), wg: [nHead, s.T, s.B],
        })),
        this.op(P.ropeBwd, [B_.dqkv, B_.cs], () => ({ u: pack("uuuuu", R(), s.T, d, nHead, half), wg: grid(R() * nHead * hd) })),
        linearDx(B_.dqkv, W.qkv.w, B_.dH, 3 * d, d),
        linearDw(B_.dqkv, A.h1, W.qkv.g, 3 * d, d),
        rmsnormBwd(this.X[l], B_.dH, B_.dX, true),
      );
    }
    this.bwd.push(
      this.op(P.embedBwd, [B_.groups, B_.rows, B_.dX, this.wte.g], () => ({ u: pack("uu", d, s.nGroups), wg: [s.nGroups] })),
    );

    // ---- optimizer: global grad norm -> clip scale -> AdamW
    this.optim = this.params.map((p, i) =>
      this.op(P.sumsq, [p.g, B_.partial], () => ({ u: pack("uu", p.n, i * 256), wg: [256] })),
    );
    this.optim.push(
      this.op(P.normFinal, [B_.partial, B_.norm], () => ({ u: pack("uf", 256 * this.params.length, s.maxNorm), wg: [1] })),
    );
    for (const p of this.params) {
      this.optim.push(
        this.op(P.adam, [p.w, p.g, p.m, p.v, B_.norm], () => ({
          u: pack("ufffffff", p.n, s.lr, s.b1, s.b2, 1e-8, s.wd, 1 - s.b1 ** this.adamStep, 1 - s.b2 ** this.adamStep),
          wg: grid(p.n),
        })),
      );
    }
  }

  private encode(enc: GPUCommandEncoder, steps: Step[]) {
    let pass: GPUComputePassEncoder | null = null;
    for (const st of steps) {
      if ("clear" in st) {
        pass?.end();
        pass = null;
        enc.clearBuffer(st.clear);
        continue;
      }
      const plan = st.plan();
      if (!plan) continue;
      this.dev.queue.writeBuffer(st.uniform, 0, plan.u);
      pass ??= enc.beginComputePass();
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.bind);
      pass.dispatchWorkgroups(plan.wg[0], plan.wg[1] ?? 1, plan.wg[2] ?? 1);
    }
    pass?.end();
  }

  // ------------------------------------------------------------------ generation

  /**
   * Sample one completion per prompt (all prompts must have the same token length; up to maxNew tokens each). A completion ends with EOT, or with the token
   * for which isDone() first returned true (e.g. one containing a newline); that last token is kept.
   * onTokens is called after every step with the tokens so far.
   */
  async generate(
    prompts: number[][],
    maxNew: number,
    temperature: number,
    topK: number,
    isDone?: (seq: number[]) => boolean,
    onTokens?: (seqs: number[][], done: boolean[]) => void,
    shouldStop?: () => boolean,
  ): Promise<number[][]> {
    const B = prompts.length;
    const P0 = prompts[0].length;
    if (prompts.some((p) => p.length !== P0)) throw new Error("generate(): prompts must have equal token length");
    this.ensure(B, P0 + maxNew);
    const out: number[][] = Array.from({ length: B }, () => []);
    const done = new Array(B).fill(false);
    let feed = prompts.flat();
    let T = P0;
    let start = 0;
    for (let step = 0; step < maxNew; step++) {
      Object.assign(this.s, { B, T, start });
      this.dev.queue.writeBuffer(this.bufs.tok, 0, new Uint32Array(feed));
      this.dev.queue.writeBuffer(this.bufs.idx, 0, new Uint32Array(Array.from({ length: B }, (_, b) => b * T + T - 1)));
      const enc = this.dev.createCommandEncoder();
      this.encode(enc, [...this.fwd, ...this.genHead]);
      enc.copyBufferToBuffer(this.bufs.logits, 0, this.bufs.readLogits, 0, B * VOCAB * 4);
      this.dev.queue.submit([enc.finish()]);
      await this.bufs.readLogits.mapAsync(GPUMapMode.READ);
      const logits = new Float32Array(this.bufs.readLogits.getMappedRange(0, B * VOCAB * 4).slice(0));
      this.bufs.readLogits.unmap();
      const next: number[] = [];
      for (let b = 0; b < B; b++) {
        const tok = done[b] ? EOT : sample(logits.subarray(b * VOCAB, (b + 1) * VOCAB), temperature, topK);
        if (!done[b]) {
          out[b].push(tok);
          if (tok === EOT || isDone?.(out[b])) done[b] = true;
        }
        next.push(tok);
      }
      onTokens?.(out, done);
      if (done.every(Boolean) || shouldStop?.()) break;
      start += T;
      T = 1;
      feed = next;
    }
    return out;
  }

  // ------------------------------------------------------------------ training

  /**
   * Accumulate gradients of  sum_i coef[i] * -log p(target[i] | row rows[i])  for a [B, T] token batch into the
   * gradient buffers (zeroed first). Returns per-row [log p, entropy] stats.
   */
  async computeGrads(tokens: number[], B: number, T: number, rows: number[], targets: number[], coefs: number[], ent = 0) {
    this.ensure(B, T);
    const nAct = rows.length;
    Object.assign(this.s, { B, T, start: 0, nAct, ent });
    const q = this.dev.queue;
    q.writeBuffer(this.bufs.tok, 0, new Uint32Array(tokens));
    q.writeBuffer(this.bufs.idx, 0, new Uint32Array(rows));
    q.writeBuffer(this.bufs.tgt, 0, new Uint32Array(targets));
    q.writeBuffer(this.bufs.coef, 0, new Float32Array(coefs));
    // embedding-gradient groups: rows sorted by token
    const byTok = new Map<number, number[]>();
    tokens.forEach((t, r) => (byTok.get(t) ?? byTok.set(t, []).get(t)!).push(r));
    const groups: number[] = [];
    const rowList: number[] = [];
    for (const [t, rs] of byTok) {
      groups.push(t, rowList.length, rs.length);
      rowList.push(...rs);
    }
    this.s.nGroups = byTok.size;
    q.writeBuffer(this.bufs.groups, 0, new Uint32Array(groups));
    q.writeBuffer(this.bufs.rows, 0, new Uint32Array(rowList));

    const enc = this.dev.createCommandEncoder();
    for (const p of this.params) enc.clearBuffer(p.g);
    this.encode(enc, [...this.fwd, ...this.loss, ...this.bwd]);
    enc.copyBufferToBuffer(this.bufs.stats, 0, this.bufs.readStats, 0, nAct * 2 * 4);
    q.submit([enc.finish()]);
    await this.bufs.readStats.mapAsync(GPUMapMode.READ);
    const stats = new Float32Array(this.bufs.readStats.getMappedRange(0, nAct * 2 * 4).slice(0));
    this.bufs.readStats.unmap();
    return stats;
  }

  /** Clip by global norm and take one AdamW step; returns the pre-clip gradient norm. */
  async applyGrads(lr: number, maxNorm: number): Promise<number> {
    this.adamStep++;
    Object.assign(this.s, { lr, maxNorm });
    const enc = this.dev.createCommandEncoder();
    this.encode(enc, this.optim);
    enc.copyBufferToBuffer(this.bufs.norm, 0, this.bufs.readStats, 0, 16);
    this.dev.queue.submit([enc.finish()]);
    await this.bufs.readStats.mapAsync(GPUMapMode.READ, 0, 16);
    const norm = new Float32Array(this.bufs.readStats.getMappedRange(0, 16).slice(0))[0];
    this.bufs.readStats.unmap();
    return norm;
  }

  /**
   * One policy-gradient step on completions of `prompts` (equal token lengths) with per-completion advantages.
   * Every completion token (including a final EOT, so the model can learn when to stop) gets weight A_b / n_tokens,
   * plus an optional entropy bonus (mean entropy per token times entropyBonus) that keeps the policy exploring.
   */
  async trainStep(prompts: number[][], completions: number[][], adv: number[], lr: number, entropyBonus = 0, maxNorm = 1): Promise<TrainStats> {
    const B = completions.length;
    const P0 = prompts[0].length;
    const T = P0 + Math.max(1, ...completions.map((c) => c.length));
    const tokens: number[] = [];
    const rows: number[] = [];
    const targets: number[] = [];
    let n = 0;
    completions.forEach((c) => (n += c.length));
    const coefs: number[] = [];
    completions.forEach((c, b) => {
      const seq = [...prompts[b], ...c];
      while (seq.length < T) seq.push(EOT);
      tokens.push(...seq);
      for (let j = 0; j < c.length; j++) {
        rows.push(b * T + P0 - 1 + j); // this row predicts completion token j
        targets.push(c[j]);
        coefs.push(adv[b] / Math.max(1, n));
      }
    });
    const stats = await this.computeGrads(tokens, B, T, rows, targets, coefs, entropyBonus / Math.max(1, n));
    const gradNorm = await this.applyGrads(lr, maxNorm);
    let lp = 0;
    let ent = 0;
    for (let i = 0; i < rows.length; i++) {
      lp += stats[2 * i];
      ent += stats[2 * i + 1];
    }
    return { gradNorm, meanLogp: lp / Math.max(1, rows.length), entropy: ent / Math.max(1, rows.length), tokens: rows.length };
  }

  // ------------------------------------------------------------------ testing helpers

  /** Read back a parameter's weights or gradient (for gradient checks). */
  async read(name: string, which: "w" | "g"): Promise<Float32Array> {
    const p = this.params.find((x) => x.name === name);
    if (!p) throw new Error(`no param ${name}`);
    const rb = this.dev.createBuffer({ size: p.n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(p[which], 0, rb, 0, p.n * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return out;
  }
}

/** Sample from softmax(logits / temperature) restricted to the topK largest logits (0 = all); temperature 0 = argmax. */
export function sample(logits: Float32Array, temperature: number, topK = 0): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (temperature <= 0) return best;
  const k = topK > 0 ? Math.min(topK, logits.length) : logits.length;
  let ids: number[];
  if (k < logits.length) {
    // threshold = k-th largest logit (partial selection via a small sorted buffer)
    const top: number[] = [];
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (top.length < k || v > logits[top[top.length - 1]]) {
        let j = top.length < k ? top.length : k - 1;
        if (top.length < k) top.push(i);
        while (j > 0 && logits[top[j - 1]] < v) { top[j] = top[j - 1]; j--; }
        top[j] = i;
      }
    }
    ids = top;
  } else ids = Array.from({ length: logits.length }, (_, i) => i);
  const m = logits[best];
  let sum = 0;
  const p = new Float64Array(ids.length);
  for (let j = 0; j < ids.length; j++) sum += p[j] = Math.exp((logits[ids[j]] - m) / temperature);
  let r = Math.random() * sum;
  for (let j = 0; j < ids.length; j++) if ((r -= p[j]) <= 0) return ids[j];
  return best;
}
