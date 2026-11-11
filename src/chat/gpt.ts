// CPU forward pass (with KV cache) for the nanoGPT ladder models: a modded-nanogpt style GPT
// (no-weight RMSNorm, rotary attention, GELU MLP, tied embeddings, attention-branch scale).
// Mirrors `model.py` shipped in each Hugging Face repo.

import type { Tensor } from "./torchzip";

export interface GPTConfig {
  vocab_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
}

export const CONTEXT = 1024; // training sequence length
const EPS = 1e-6;

interface Layer {
  qkv: Float32Array; // [3d, d]
  proj: Float32Array; // [d, d]
  fc: Float32Array; // [4d, d]
  fcProj: Float32Array; // [d, 4d]
  k: Float32Array; // cache [CONTEXT, d], rotary already applied
  v: Float32Array; // cache [CONTEXT, d]
}

/** out[t, o] = sum_i x[t, i] * w[o, i]  (x: [T, nIn], w: [nOut, nIn], i.e. nn.Linear without bias) */
function linear(out: Float32Array, x: Float32Array, T: number, w: Float32Array, nOut: number, nIn: number) {
  for (let o = 0; o < nOut; o++) {
    const wo = o * nIn;
    let t = 0;
    // 4 positions at a time: each weight load is reused 4x (matters for prompt prefill)
    for (; t + 3 < T; t += 4) {
      const x0 = t * nIn, x1 = x0 + nIn, x2 = x1 + nIn, x3 = x2 + nIn;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < nIn; i++) {
        const wv = w[wo + i];
        s0 += x[x0 + i] * wv;
        s1 += x[x1 + i] * wv;
        s2 += x[x2 + i] * wv;
        s3 += x[x3 + i] * wv;
      }
      out[t * nOut + o] = s0;
      out[(t + 1) * nOut + o] = s1;
      out[(t + 2) * nOut + o] = s2;
      out[(t + 3) * nOut + o] = s3;
    }
    for (; t < T; t++) {
      const xo = t * nIn;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      let i = 0;
      for (; i + 3 < nIn; i += 4) {
        s0 += x[xo + i] * w[wo + i];
        s1 += x[xo + i + 1] * w[wo + i + 1];
        s2 += x[xo + i + 2] * w[wo + i + 2];
        s3 += x[xo + i + 3] * w[wo + i + 3];
      }
      for (; i < nIn; i++) s0 += x[xo + i] * w[wo + i];
      out[t * nOut + o] = s0 + s1 + s2 + s3;
    }
  }
}

function rmsnorm(out: Float32Array, x: Float32Array, T: number, d: number) {
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let ss = 0;
    for (let i = 0; i < d; i++) ss += x[o + i] * x[o + i];
    const r = 1 / Math.sqrt(ss / d + EPS);
    for (let i = 0; i < d; i++) out[o + i] = x[o + i] * r;
  }
}

/** erf with |error| < 1.2e-7 (Numerical Recipes erfc Chebyshev fit) */
function erf(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? 1 - r : r - 1;
}

export class GPT {
  readonly d: number;
  readonly nHead: number;
  readonly headDim: number;
  /** number of real tokenizer ids (the embedding is padded beyond this) */
  readonly nVocab: number;
  /** tokens currently in the KV cache */
  pos = 0;
  private wte: Float32Array;
  private layers: Layer[];
  private attnScale: number;
  private cos: Float32Array; // [CONTEXT, headDim/2]
  private sin: Float32Array;

  constructor(readonly cfg: GPTConfig, sd: Record<string, Tensor>, nVocab: number) {
    const d = (this.d = cfg.n_embd);
    this.nHead = cfg.n_head;
    this.headDim = d / cfg.n_head;
    this.nVocab = nVocab;
    this.attnScale = 1 / Math.sqrt(2 * cfg.n_layer);
    const get = (name: string, shape: number[]) => {
      const t = sd[name];
      if (!t) throw new Error(`checkpoint is missing ${name}`);
      if (t.shape.join() !== shape.join()) throw new Error(`${name}: expected [${shape}], got [${t.shape}]`);
      return t.data;
    };
    this.wte = get("wte.weight", [cfg.vocab_size, d]);
    this.layers = [];
    const half = this.headDim / 2;
    let invFreq: Float32Array | null = null;
    for (let l = 0; l < cfg.n_layer; l++) {
      const p = `h.${l}.`;
      invFreq ??= sd[p + "attn.rotary.inv_freq"]?.data ?? null;
      this.layers.push({
        qkv: get(p + "attn.c_attn.weight", [3 * d, d]),
        proj: get(p + "attn.c_proj.weight", [d, d]),
        fc: get(p + "mlp.c_fc.weight", [4 * d, d]),
        fcProj: get(p + "mlp.c_proj.weight", [d, 4 * d]),
        k: new Float32Array(CONTEXT * d),
        v: new Float32Array(CONTEXT * d),
      });
    }
    if (!invFreq) {
      invFreq = new Float32Array(half);
      for (let i = 0; i < half; i++) invFreq[i] = 1 / 10000 ** ((2 * i) / this.headDim);
    }
    this.cos = new Float32Array(CONTEXT * half);
    this.sin = new Float32Array(CONTEXT * half);
    for (let t = 0; t < CONTEXT; t++) {
      for (let i = 0; i < half; i++) {
        const f = Math.fround(t * invFreq[i]); // torch computes the angle in f32
        this.cos[t * half + i] = Math.cos(f);
        this.sin[t * half + i] = Math.sin(f);
      }
    }
  }

  get params(): number {
    const d = this.d;
    return this.cfg.vocab_size * d + this.cfg.n_layer * 12 * d * d;
  }

  /**
   * Append `tokens` to the KV cache and run them through the model.
   * Returns next-token logits (over the real vocabulary) after the last one.
   */
  forward(tokens: number[]): Float32Array {
    const T = tokens.length;
    const { d, nHead, headDim } = this;
    const start = this.pos;
    if (T === 0) throw new Error("forward() needs at least one token");
    if (start + T > CONTEXT) throw new Error(`context overflow (${start + T} > ${CONTEXT})`);
    const half = headDim / 2;
    const scale = 1 / Math.sqrt(headDim);

    const x = new Float32Array(T * d);
    for (let t = 0; t < T; t++) x.set(this.wte.subarray(tokens[t] * d, tokens[t] * d + d), t * d);
    const h = new Float32Array(T * d);
    const qkv = new Float32Array(T * 3 * d);
    const y = new Float32Array(T * d);
    const o = new Float32Array(T * d);
    const f = new Float32Array(T * 4 * d);
    const att = new Float32Array(CONTEXT);

    for (const L of this.layers) {
      // --- attention
      rmsnorm(h, x, T, d);
      linear(qkv, h, T, L.qkv, 3 * d, d);
      for (let t = 0; t < T; t++) {
        const p = start + t;
        const row = t * 3 * d;
        const c = p * half;
        for (let hh = 0; hh < nHead; hh++) {
          const qo = row + hh * headDim;
          const ko = row + d + hh * headDim;
          const vo = row + 2 * d + hh * headDim;
          const co = p * d + hh * headDim;
          for (let i = 0; i < half; i++) {
            const cs = this.cos[c + i], sn = this.sin[c + i];
            const q1 = qkv[qo + i], q2 = qkv[qo + half + i];
            qkv[qo + i] = q1 * cs + q2 * sn;
            qkv[qo + half + i] = -q1 * sn + q2 * cs;
            const k1 = qkv[ko + i], k2 = qkv[ko + half + i];
            L.k[co + i] = k1 * cs + k2 * sn;
            L.k[co + half + i] = -k1 * sn + k2 * cs;
          }
          for (let i = 0; i < headDim; i++) L.v[co + i] = qkv[vo + i];
        }
      }
      for (let t = 0; t < T; t++) {
        const p = start + t;
        for (let hh = 0; hh < nHead; hh++) {
          const qo = t * 3 * d + hh * headDim;
          let max = -Infinity;
          for (let s = 0; s <= p; s++) {
            const ko = s * d + hh * headDim;
            let dot = 0;
            for (let i = 0; i < headDim; i++) dot += qkv[qo + i] * L.k[ko + i];
            dot *= scale;
            att[s] = dot;
            if (dot > max) max = dot;
          }
          let sum = 0;
          for (let s = 0; s <= p; s++) {
            const e = Math.exp(att[s] - max);
            att[s] = e;
            sum += e;
          }
          const yo = t * d + hh * headDim;
          for (let i = 0; i < headDim; i++) y[yo + i] = 0;
          for (let s = 0; s <= p; s++) {
            const a = att[s] / sum;
            const vo = s * d + hh * headDim;
            for (let i = 0; i < headDim; i++) y[yo + i] += a * L.v[vo + i];
          }
        }
      }
      linear(o, y, T, L.proj, d, d);
      for (let i = 0; i < T * d; i++) x[i] += this.attnScale * o[i];

      // --- MLP
      rmsnorm(h, x, T, d);
      linear(f, h, T, L.fc, 4 * d, d);
      for (let i = 0; i < f.length; i++) {
        const v = f[i];
        f[i] = 0.5 * v * (1 + erf(v * Math.SQRT1_2));
      }
      linear(o, f, T, L.fcProj, d, 4 * d);
      for (let i = 0; i < T * d; i++) x[i] += o[i];
    }
    this.pos += T;

    // final norm + tied lm_head, last position only
    const last = new Float32Array(d);
    rmsnorm(last, x.subarray((T - 1) * d), 1, d);
    const logits = new Float32Array(this.nVocab);
    linear(logits, last, 1, this.wte, this.nVocab, d);
    return logits;
  }
}
