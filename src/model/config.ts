// Qwen3-0.6B architecture (from Qwen/Qwen3-0.6B config.json)
export const CFG = {
  repo: "Qwen/Qwen3-0.6B",
  hidden: 1024,
  intermediate: 3072,
  layers: 28,
  heads: 16,
  kvHeads: 8,
  headDim: 128,
  vocab: 151936,
  eps: 1e-6,
  ropeTheta: 1_000_000,
  eosIds: [151645, 151643],
  maxContext: 256,
} as const;

export const Q_DIM = CFG.heads * CFG.headDim; // 2048
export const KV_DIM = CFG.kvHeads * CFG.headDim; // 1024

export const hfUrl = (file: string) => `https://huggingface.co/${CFG.repo}/resolve/main/${file}`;
