// RL worker: loads a ladder model into the WebGPU trainer and runs the loop
//   sample a group of completions -> score them -> group-normalized advantages -> one policy-gradient step.

import type { Tokenizer } from "@huggingface/tokenizers";
import { loadTorchCheckpoint, type Tensor } from "../chat/torchzip";
import type { GPTConfig } from "../chat/gpt";
import { downloadCheckpoint, getTokenizer, type HttpError } from "../chat/hub";
import { EOT, Trainer } from "./trainer";
import { loadPython, PythonReward } from "./python";

export interface Settings {
  /** may contain {n}, replaced each step by a value from `targets` */
  prompt: string;
  targets: number[];
  /** end a completion at the first token containing this string (e.g. "\n"); null = only at EOT */
  stop: string | null;
  reward: string; // Python source defining reward(text, info) -> float
  /** completions per prompt (advantages are normalized within each group) */
  group: number;
  /** distinct prompts per step; with {n} each gets its own n, so every step contrasts several targets */
  promptsPerStep: number;
  maxNew: number;
  lr: number;
  /** weight of the entropy bonus (0 = plain policy gradient) */
  entropy: number;
  temperature: number;
  topK: number;
}

export type ToWorker =
  | { type: "load"; repo: string; token: string | null }
  | { type: "settings"; settings: Settings }
  | { type: "run"; on: boolean }
  | { type: "reset" };

export interface StepResult {
  step: number;
  /** per completion */
  prompts: string[];
  ns: number[];
  lengths: number[];
  ended: boolean[];
  texts: string[];
  rewards: number[];
  adv: number[];
  mean: number;
  std: number;
  trained: boolean;
  gradNorm: number;
  entropy: number;
  meanLogp: number;
  tokens: number;
  genMs: number;
  trainMs: number;
}

export type FromWorker =
  | { type: "progress"; phase: string; done: number; total: number }
  | { type: "loaded"; repo: string; config: GPTConfig }
  | { type: "error"; message: string; status?: number }
  | { type: "rollout"; prompts: string[]; texts: string[]; done: boolean[] }
  | { type: "step"; result: StepResult }
  | { type: "state"; running: boolean; step: number }
  | { type: "reward"; ok: boolean; message: string };

const post = (m: FromWorker) => self.postMessage(m);
void loadPython().catch((e) => post({ type: "error", message: `could not start Python (Pyodide): ${(e as Error).message}` }));

let tk: Tokenizer | null = null;
let trainer: Trainer | null = null;
let settings: Settings | null = null;
let running = false;
let looping = false;
let step = 0;
let loadSeq = 0;
let compiled: { src: string; fn: Promise<PythonReward> } | null = null;

/** Compile the reward source (cached by text); reports success or the Python error to the page. */
function rewardFn(src: string): Promise<PythonReward> {
  if (compiled?.src !== src) {
    const fn = PythonReward.compile(src);
    compiled = { src, fn };
    fn.then(
      () => compiled?.fn === fn && post({ type: "reward", ok: true, message: "reward() compiled" }),
      (e) => compiled?.fn === fn && post({ type: "reward", ok: false, message: (e as Error).message }),
    );
  }
  return compiled.fn;
}

const decode = (ids: number[]) => {
  const body = ids[ids.length - 1] === EOT ? ids.slice(0, -1) : ids;
  return body.length ? tk!.decode(body, { skip_special_tokens: false }) : "";
};
/** text of a completion up to (not including) the stop string */
const cut = (text: string, stop: string | null) => (stop && text.includes(stop) ? text.slice(0, text.indexOf(stop)) : text);

async function load(repo: string, token: string | null) {
  const seq = ++loadSeq;
  running = false;
  try {
    const [tok, { bytes }] = await Promise.all([
      getTokenizer(),
      downloadCheckpoint(repo, token, (phase, done, total) => post({ type: "progress", phase, done, total }), () => seq !== loadSeq),
    ]);
    tk = tok;
    if (seq !== loadSeq) return;
    post({ type: "progress", phase: "Starting Python", done: 0, total: 0 });
    await loadPython();
    post({ type: "progress", phase: "Uploading to the GPU", done: 0, total: 0 });
    const ck = loadTorchCheckpoint(bytes);
    const cfg = ck.root.config as GPTConfig;
    const t = await Trainer.create(cfg, ck.root.model as Record<string, Tensor>);
    if (seq !== loadSeq) return t.destroy();
    trainer?.destroy();
    trainer = t;
    step = 0;
    post({ type: "loaded", repo, config: cfg });
    post({ type: "state", running, step });
  } catch (e) {
    if (seq !== loadSeq) return;
    post({ type: "error", message: (e as Error).message, status: (e as HttpError).status });
  }
}

async function loop() {
  if (looping) return;
  looping = true;
  try {
    while (running && trainer && settings && tk) {
      const t = trainer;
      const s = settings;
      const reward = await rewardFn(s.reward);
      if (!running || t !== trainer) break;
      const templated = s.prompt.includes("{n}") && s.targets.length > 0;
      const k = templated ? Math.max(1, s.promptsPerStep) : 1;
      // one n per prompt: distinct values, cycling through them if there are fewer values than prompts
      const pool = [...s.targets].sort(() => Math.random() - 0.5);
      const groupNs = Array.from({ length: k }, (_, g) => (templated ? pool[g % pool.length] : NaN)); // cycle if fewer values than prompts
      const groupTexts = groupNs.map((n) => s.prompt.replaceAll("{n}", String(n)));
      let groupIds = groupTexts.map((p) => (p ? tk!.encode(p, { add_special_tokens: false }).ids.slice(-64) : [EOT]));
      // the batch shares positions, so left-pad shorter prompts with end-of-text (a document boundary)
      const P0 = Math.max(...groupIds.map((x) => x.length));
      groupIds = groupIds.map((x) => [...new Array(P0 - x.length).fill(EOT), ...x]);
      const of = (b: number) => Math.floor(b / s.group); // group of sequence b
      const B = k * s.group;
      const prompts = Array.from({ length: B }, (_, b) => groupIds[of(b)]);
      const promptTexts = Array.from({ length: B }, (_, b) => groupTexts[of(b)]);
      const ns = Array.from({ length: B }, (_, b) => groupNs[of(b)]);
      const stop = s.stop;
      const tok = tk;
      const isDone = stop ? (seq: number[]) => tok.decode([seq[seq.length - 1]], { skip_special_tokens: false }).includes(stop) : undefined;

      const t0 = performance.now();
      let lastPost = 0;
      const completions = await t.generate(prompts, s.maxNew, s.temperature, s.topK, isDone, (seqs, done) => {
        const now = performance.now();
        if (now - lastPost > 60) {
          lastPost = now;
          post({ type: "rollout", prompts: promptTexts, texts: seqs.map((x) => cut(decode(x), stop)), done });
        }
      }, () => !running);
      if (!running || t !== trainer) break;
      const genMs = performance.now() - t0;

      const texts = completions.map((c) => cut(decode(c), stop));
      const ended = completions.map((c) => c.length > 0 && (c[c.length - 1] === EOT || !!isDone?.(c)));
      const lengths = completions.map((c, i) => c.length - (ended[i] ? 1 : 0));
      const rewards = reward.score(
        texts.map((text, i) => ({
          text,
          info: { n: Number.isFinite(ns[i]) ? ns[i] : null, tokens: lengths[i], ended: ended[i], prompt: promptTexts[i] },
        })),
      );
      // GRPO: normalize within each group (same prompt)
      const adv = new Array(B).fill(0);
      for (let g = 0; g < k; g++) {
        const r = rewards.slice(g * s.group, (g + 1) * s.group);
        const m = r.reduce((a, b) => a + b, 0) / r.length;
        const sd = Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length);
        r.forEach((x, j) => (adv[g * s.group + j] = sd > 1e-6 ? (x - m) / sd : 0));
      }
      const mean = rewards.reduce((a, b) => a + b, 0) / B;
      const std = Math.sqrt(rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / B);

      const t1 = performance.now();
      let stats = { gradNorm: 0, entropy: NaN, meanLogp: NaN, tokens: 0 };
      const trained = adv.some((x) => x !== 0); // every group tied: no signal, skip the update
      if (trained) stats = await t.trainStep(prompts, completions, adv, s.lr, s.entropy);
      if (t !== trainer) break;
      step++;
      post({
        type: "step",
        result: { step, prompts: promptTexts, ns, lengths, ended, texts, rewards, adv, mean, std, trained, ...stats, genMs, trainMs: performance.now() - t1 },
      });
    }
  } catch (e) {
    running = false;
    post({ type: "error", message: (e as Error).message });
  } finally {
    looping = false;
    post({ type: "state", running, step });
  }
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === "load") void load(m.repo, m.token);
  else if (m.type === "settings") {
    settings = m.settings;
    void rewardFn(settings.reward).catch(() => {}); // compile now so errors show up before Start
  }
  else if (m.type === "run") {
    running = m.on;
    post({ type: "state", running, step });
    if (running) void loop();
  } else if (m.type === "reset") {
    running = false;
    const t = trainer;
    // wait for the loop to notice, then restore the weights
    const doReset = () => {
      if (looping) return setTimeout(doReset, 20);
      t?.reset();
      step = 0;
      post({ type: "state", running, step });
    };
    doReset();
  }
};
