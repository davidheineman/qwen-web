import type { Capture, Engine } from "../model/engine";
import type { Tok } from "../model/tokenizer";
import { CFG } from "../model/config";

/** Everything the views need: the latest capture plus callbacks into the app. */
export interface Ctx {
  cap: Capture;
  eng: Engine;
  tok: Tok;
  promptLen: number;
  /** -1 = embedding, 0..27 = transformer layer, 28 = final norm + unembedding */
  layer: number;
  head: number;
  scaleMode: "row" | "shared";
  label(i: number): string;
  piece(id: number): string;
  selectLayer(l: number): void;
  selectHead(h: number): void;
  setFocus(pos: number): void;
  setScaleMode(m: "row" | "shared"): void;
  inspect(name: string, highlightRow?: number): void;
}

export const FINAL = CFG.layers;
export const EMBED = -1;

/** Residual stream at checkpoint l (0 = embeddings, i = after layer i-1), position t */
export function resid(cap: Capture, l: number, t = cap.focus): Float32Array {
  const H = CFG.hidden;
  return cap.resid.subarray((l * cap.T + t) * H, (l * cap.T + t + 1) * H);
}

/** Attention probabilities for layer l, head h: [T, T] */
export function probs(cap: Capture, l: number, h: number): Float32Array {
  const TT = cap.T * cap.T;
  const off = (l * CFG.heads + h) * TT;
  return cap.probs.subarray(off, off + TT);
}

export function rmsnorm(x: Float32Array, g: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  const inv = 1 / Math.sqrt(s / x.length + CFG.eps);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * g[i];
  return out;
}

export function finalNormed(ctx: Ctx): Float32Array {
  return rmsnorm(resid(ctx.cap, CFG.layers), ctx.eng.weights.get("model.norm.weight")!.values!);
}

export function rms(x: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
}
