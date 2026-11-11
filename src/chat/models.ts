// Models in the pickers: the completed NanoGPT Ladder v3 pretrained models
// (https://huggingface.co/collections/davidheineman/nanogpt-ladder-v3-completed-pretrained-models, public)
// and the 28 completed models of the NanoGPT Ladder v2
// (https://huggingface.co/collections/davidheineman/nanogpt-ladder-v2-28-completed-models, private).
// Both ladders share one architecture (see gpt.ts); only the shapes differ.

import type { GPTConfig } from "./gpt";

type Arch = Omit<GPTConfig, "vocab_size">;

export interface LadderModel {
  /** short id used in the URL, e.g. "base-64m-30tpp" or "web-large-4b" */
  id: string;
  /** Hugging Face repo id, optionally followed by the folder holding model.pt ("owner/name/sub/dir") */
  repo: string;
  /** option group in the picker, e.g. "v3 · base" */
  group: string;
  tokens: number;
  params: number;
  /** approximate download size of model.pt in bytes (fp32) */
  bytes: number;
  /** option text */
  label: string;
}

const fmt = (n: number) => (n >= 1e9 ? `${+(n / 1e9).toFixed(1)}B` : `${Math.round(n / 1e6)}M`);
const countParams = (a: Arch) => 50304 * a.n_embd + a.n_layer * 12 * a.n_embd * a.n_embd;
const mb = (params: number) => `${Math.round((params * 4) / 1e6)} MB`;

// ------------------------------------------------------------------ v3

const V3_ARCH: Record<string, Arch> = {
  "8m": { n_layer: 8, n_head: 4, n_embd: 128 },
  "16m": { n_layer: 14, n_head: 6, n_embd: 192 },
  "32m": { n_layer: 13, n_head: 5, n_embd: 320 },
  "64m": { n_layer: 12, n_head: 8, n_embd: 512 },
  "128m": { n_layer: 13, n_head: 12, n_embd: 768 },
};

/** Uploaded so far, per data mix: "size:tokens-per-param" (more are still training). */
const V3_RUNS: Record<string, string> = {
  base: "8m:10 8m:30 8m:100 16m:10 16m:30 16m:100 32m:10 32m:30 32m:100 64m:10 64m:30 128m:10",
  "chat-heavy": "8m:10 8m:30 8m:100 16m:10 16m:30 16m:100 32m:10 32m:30",
  "code-heavy": "8m:10 8m:30 8m:100 16m:10 16m:30 16m:100 32m:10 32m:30 32m:100 64m:10 64m:30 128m:10",
  "math-heavy": "8m:10 8m:30 8m:100 16m:10 16m:30 16m:100 32m:10 32m:30 32m:100 64m:10 64m:30",
};

const V3: LadderModel[] = Object.entries(V3_RUNS).flatMap(([mix, runs]) =>
  runs.split(" ").map((run) => {
    const [size, tpp] = run.split(":");
    const params = countParams(V3_ARCH[size]);
    const tokens = params * Number(tpp);
    return {
      id: `${mix}-${size}-${tpp}tpp`,
      repo: `davidheineman/nanogpt-ladder-v3-${mix}-${size}-${tpp}tpp-seed-1337`,
      group: `v3 · ${mix}`,
      tokens,
      params,
      bytes: params * 4,
      label: `${mix} · ${size} · ${tpp} TPP · ${fmt(tokens)} tokens (${mb(params)})`,
    };
  }),
);

// ------------------------------------------------------------------ v2

const V2_ARCH: Record<string, Arch> = {
  small: { n_layer: 6, n_head: 4, n_embd: 256 },
  bridge: { n_layer: 8, n_head: 6, n_embd: 384 },
  large: { n_layer: 8, n_head: 8, n_embd: 512 },
};

/** [model size, token budget label, requested training tokens] */
const V2_POINTS = [
  ["small", "250m", 250e6],
  ["small", "1b", 1e9],
  ["bridge", "500m", 500e6],
  ["bridge", "2b", 2e9],
  ["large", "500m", 500e6],
  ["large", "2b", 2e9],
  ["large", "4b", 4e9],
] as const;

const V2: LadderModel[] = ["general", "code", "math", "web"].flatMap((mix) =>
  V2_POINTS.map(([size, point, tokens]) => {
    const params = countParams(V2_ARCH[size]);
    return {
      id: `${mix}-${size}-${point}`,
      repo: `davidheineman/nanogpt-ladder-v2-${mix}-${size}-${point}-seed-1337`,
      group: `v2 · ${mix} (private)`,
      tokens,
      params,
      bytes: params * 4,
      label: `${mix} · ${size} · ${fmt(params)} params · ${fmt(tokens)} tokens (${mb(params)})`,
    };
  }),
);

export const MODELS: LadderModel[] = [...V3, ...V2];

export const DEFAULT_MODEL = "base-64m-30tpp";

/** Fill a <select> with every model, grouped. */
export function fillModelPicker(sel: HTMLSelectElement) {
  for (const group of new Set(MODELS.map((m) => m.group))) {
    const g = document.createElement("optgroup");
    g.label = group;
    for (const m of MODELS.filter((x) => x.group === group)) g.append(new Option(m.label, m.id));
    sel.append(g);
  }
}
