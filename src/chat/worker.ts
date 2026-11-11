// Web Worker: downloads a ladder checkpoint, keeps the model + GPT-2 tokenizer, and streams generations.

import type { Tokenizer } from "@huggingface/tokenizers";
import { downloadCheckpoint, fetchValidation, getTokenizer, type HttpError } from "./hub";
import { loadTorchCheckpoint, type Tensor } from "./torchzip";
import { CONTEXT, GPT, type GPTConfig } from "./gpt";
import { GpuGPT } from "./gpu";

export type ToWorker =
  | { type: "load"; repo: string; token: string | null; cpu?: boolean }
  | {
      type: "generate";
      id: number;
      prompt: string;
      maxTokens: number;
      temperature: number;
      topK: number;
      stop: string[];
    }
  | { type: "abort" };

export type FromWorker =
  | { type: "progress"; phase: string; done: number; total: number }
  | { type: "loaded"; repo: string; config: GPTConfig; params: number; source: string; backend: string }
  | { type: "error"; message: string; status?: number }
  | { type: "text"; id: number; text: string }
  | { type: "done"; id: number; text: string; reason: "stop" | "eos" | "length" | "abort"; promptTokens: number; newTokens: number; tokPerSec: number };

const EOT = 50256;
const N_VOCAB = 50257;

const post = (m: FromWorker) => self.postMessage(m);

let tokenizer: Tokenizer | null = null;
/** CPU (gpt.ts) or WebGPU (gpu.ts) engine; both keep a KV cache of `pos` tokens */
type Engine = (GPT | GpuGPT) & { backend: string };
let model: Engine | null = null;
let modelBackend = "";
let modelRepo = "";
/** tokens whose K/V currently sit in the model's cache (positions 0..model.pos-1) */
let cached: number[] = [];
let aborted = false;
let loadSeq = 0;

async function load(repo: string, token: string | null, cpu: boolean) {
  const seq = ++loadSeq;
  aborted = true; // stop any running generation
  try {
    const [tk, { bytes, source }] = await Promise.all([
      getTokenizer(),
      downloadCheckpoint(repo, token, (phase, done, total) => post({ type: "progress", phase, done, total }), () => seq !== loadSeq),
    ]);
    tokenizer = tk;
    if (seq !== loadSeq) return;
    post({ type: "progress", phase: "Parsing checkpoint", done: 0, total: 1 });
    const ck = loadTorchCheckpoint(bytes);
    const cfg = ck.root.config as GPTConfig;
    const sd = ck.root.model as Record<string, Tensor>;
    let engine: Engine | null = null;
    if (!cpu) {
      try {
        const gpu = await GpuGPT.create(cfg, sd, N_VOCAB);
        if (await validate(gpu, repo, token)) engine = gpu;
        else {
          console.warn("WebGPU output failed the checkpoint's reference check; using the CPU");
          gpu.destroy();
        }
      } catch (e) {
        console.warn("WebGPU unavailable, using the CPU:", e);
      }
    }
    engine ??= Object.assign(new GPT(cfg, sd, N_VOCAB), { backend: "CPU" });
    if (seq !== loadSeq) {
      if (engine instanceof GpuGPT) engine.destroy();
      return;
    }
    const old = model;
    model = engine;
    if (old instanceof GpuGPT) old.destroy(); // a running generation on it sees `aborted` and bails
    modelRepo = repo;
    modelBackend = `${engine.backend}`;
    cached = [];
    post({ type: "loaded", repo, config: cfg, params: model.params, source, backend: modelBackend });
  } catch (e) {
    if (seq !== loadSeq) return;
    post({ type: "error", message: (e as Error).message, status: (e as HttpError).status });
  }
}

const argmax = (a: Float32Array) => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
};

/**
 * Each repo's manifest.json records a PyTorch check: the argmax next token for "Hello world!".
 * Run it on the GPU (retrying once) so a broken driver falls back to the CPU instead of producing garbage.
 */
async function validate(gpu: GpuGPT, repo: string, token: string | null): Promise<boolean> {
  const check = await fetchValidation(repo, token);
  if (!check) return true; // nothing to check against
  for (let attempt = 0; attempt < 2; attempt++) {
    gpu.pos = 0;
    const got = argmax(await gpu.forward(check.input_ids));
    gpu.pos = 0;
    if (got === check.next_token_argmax) return true;
    console.warn(`WebGPU check: expected token ${check.next_token_argmax}, got ${got} (attempt ${attempt + 1})`);
  }
  return false;
}

function sample(logits: Float32Array, temperature: number, topK: number): number {
  if (temperature <= 0) return argmax(logits);
  const k = Math.max(1, Math.min(topK > 0 ? topK : logits.length, logits.length));
  // keep the k largest logits (insertion into a small sorted list)
  const ids = new Int32Array(k).fill(-1);
  const vals = new Float32Array(k).fill(-Infinity);
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (v <= vals[k - 1]) continue;
    let j = k - 1;
    while (j > 0 && vals[j - 1] < v) {
      vals[j] = vals[j - 1];
      ids[j] = ids[j - 1];
      j--;
    }
    vals[j] = v;
    ids[j] = i;
  }
  let sum = 0;
  const p = new Float64Array(k);
  for (let j = 0; j < k; j++) sum += p[j] = Math.exp((vals[j] - vals[0]) / temperature);
  let r = Math.random() * sum;
  for (let j = 0; j < k; j++) if ((r -= p[j]) <= 0) return ids[j];
  return ids[0];
}

/** yield to the event loop so an "abort" message can be handled */
const tick = () => new Promise<void>((r) => {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => r();
  ch.port2.postMessage(0);
});

async function generate(m: Extract<ToWorker, { type: "generate" }>) {
  if (!model || !tokenizer) return post({ type: "error", message: "model not loaded" });
  aborted = false;
  const tk = tokenizer;
  const gpt = model;
  let ids = tk.encode(m.prompt, { add_special_tokens: false }).ids;
  // keep room for the reply; drop the oldest context if needed
  const maxNew = Math.min(m.maxTokens, CONTEXT - 1);
  const budget = CONTEXT - maxNew;
  if (ids.length > budget) ids = ids.slice(ids.length - budget);
  if (ids.length === 0) ids = [EOT];

  // reuse the KV cache for the longest shared prefix (always recompute at least the last token)
  let common = 0;
  while (common < cached.length && common < ids.length - 1 && cached[common] === ids[common]) common++;
  gpt.pos = common;
  cached = ids.slice(0, common);

  let logits = await gpt.forward(ids.slice(common));
  cached = ids.slice();
  const tPrompt = performance.now();
  await tick();

  const out: number[] = [];
  let text = "";
  let reason: "stop" | "eos" | "length" | "abort" = "length";
  while (out.length < maxNew) {
    if (aborted) { reason = "abort"; break; }
    const next = sample(logits, m.temperature, m.topK);
    if (next === EOT) { reason = "eos"; break; }
    out.push(next);
    let full = tk.decode(out, { skip_special_tokens: false });
    // stop strings only count once the reply has some non-whitespace content
    const lead = full.length - full.trimStart().length;
    const hitStop = full.length > lead ? m.stop.map((s) => full.indexOf(s, lead)).filter((i) => i >= 0) : [];
    if (hitStop.length) {
      text = full.slice(0, Math.min(...hitStop));
      reason = "stop";
      break;
    }
    // hold back a trailing partial UTF-8 character and anything that could start a stop string
    if (full.endsWith("�")) full = full.replace(/�+$/, "");
    let hold = 0;
    for (const s of m.stop) for (let n = Math.min(s.length - 1, full.length); n > hold; n--) if (full.endsWith(s.slice(0, n))) hold = n;
    const visible = full.slice(0, full.length - hold);
    if (visible !== text) {
      text = visible;
      post({ type: "text", id: m.id, text });
    }
    if (gpt.pos >= CONTEXT) break;
    logits = await gpt.forward([next]);
    cached.push(next);
    await tick();
  }
  if (reason !== "stop") text = out.length ? tk.decode(out, { skip_special_tokens: false }).replace(/�+$/, "") : "";
  const secs = (performance.now() - tPrompt) / 1000;
  post({
    type: "done",
    id: m.id,
    text,
    reason,
    promptTokens: ids.length,
    newTokens: out.length,
    tokPerSec: out.length / Math.max(secs, 1e-3),
  });
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === "load") {
    if (m.repo === modelRepo && model && (modelBackend === "CPU") === !!m.cpu) {
      post({ type: "loaded", repo: m.repo, config: model.cfg, params: model.params, source: "memory", backend: modelBackend });
    } else {
      void load(m.repo, m.token, !!m.cpu);
    }
  } else if (m.type === "generate") {
    const engine = model;
    void generate(m).catch((err) => {
      // errors from an engine that was swapped out mid-generation are expected; ignore them
      if (engine === model) post({ type: "error", message: (err as Error).message });
    });
  } else if (m.type === "abort") {
    aborted = true;
  }
};
