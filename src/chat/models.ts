// The 28 completed models of the NanoGPT Ladder v2
// (https://huggingface.co/collections/davidheineman/nanogpt-ladder-v2-28-completed-models),
// plus the NanoGPT Ladder v3 base 64M used by tiny-rl (same architecture, 12 layers).

const LADDER_MIXES = ["general", "code", "math", "web"] as const;
/** option groups in the model pickers, in display order */
export const MIXES = ["base", ...LADDER_MIXES] as const;

const ARCH = {
  small: { n_layer: 6, n_head: 4, n_embd: 256 },
  bridge: { n_layer: 8, n_head: 6, n_embd: 384 },
  large: { n_layer: 8, n_head: 8, n_embd: 512 },
  "64m": { n_layer: 12, n_head: 8, n_embd: 512 },
} as const;

/** [model size, token budget label, requested training tokens] */
const POINTS = [
  ["small", "250m", 250e6],
  ["small", "1b", 1e9],
  ["bridge", "500m", 500e6],
  ["bridge", "2b", 2e9],
  ["large", "500m", 500e6],
  ["large", "2b", 2e9],
  ["large", "4b", 4e9],
] as const;

export interface LadderModel {
  /** short id used in the URL, e.g. "web-large-4b" */
  id: string;
  /** Hugging Face repo id, optionally followed by the folder holding model.pt ("owner/name/sub/dir") */
  repo: string;
  mix: (typeof MIXES)[number];
  size: keyof typeof ARCH;
  tokens: number;
  params: number;
  /** approximate download size of model.pt in bytes (fp32) */
  bytes: number;
  label: string;
}

const fmt = (n: number) => (n >= 1e9 ? `${+(n / 1e9).toFixed(1)}B` : `${Math.round(n / 1e6)}M`);

function model(id: string, repo: string, mix: LadderModel["mix"], size: LadderModel["size"], tokens: number): LadderModel {
  const a = ARCH[size];
  const params = 50304 * a.n_embd + a.n_layer * 12 * a.n_embd * a.n_embd;
  return {
    id,
    repo,
    mix,
    size,
    tokens,
    params,
    bytes: params * 4,
    label: `${size} · ${fmt(params)} params · ${fmt(tokens)} tokens (${Math.round((params * 4) / 1e6)} MB)`,
  };
}

export const MODELS: LadderModel[] = [
  // https://huggingface.co/davidheineman/tiny-rl-refined-v3-model-sweep (pretrained base, before any tiny-rl SFT/RL)
  model(
    "base-64m-30tpp",
    "davidheineman/tiny-rl-refined-v3-model-sweep/pretrained/base-64m-30tpp-seed-1337",
    "base",
    "64m",
    1905131520,
  ),
  ...LADDER_MIXES.flatMap((mix) =>
    POINTS.map(([size, point, tokens]) =>
      model(`${mix}-${size}-${point}`, `davidheineman/nanogpt-ladder-v2-${mix}-${size}-${point}-seed-1337`, mix, size, tokens),
    ),
  ),
];

export const DEFAULT_MODEL = "base-64m-30tpp";
