# Qwen3-0.6B, inside out

Runs [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) entirely in the browser with WebGPU and shows
every weight matrix and every intermediate activation, layer by layer, for each predicted token.
Inspired by [bbycroft.net/llm](https://bbycroft.net/llm), but running a real 0.6B model on your own text.

**Live demo: https://davidheineman.github.io/qwen-web/interp.html** (needs a WebGPU browser; downloads ~1.19 GB of weights once).
The site root, https://davidheineman.github.io/qwen-web/, is the Ladder playground below.

```bash
npm install
npm run dev        # http://localhost:5173 (Ladder), /interp.html (visualizer), /rl.html (needs WebGPU: Chrome / Edge / Safari 26+)
```

On first load the page downloads the original bf16 weights straight from the Hugging Face Hub (about 1.19 GB;
`lm_head` is skipped because it is tied to the embedding) and caches them in the browser's Cache Storage.

To skip the download during development, put a copy in `public/models/Qwen3-0.6B/`
(`model.safetensors`, `tokenizer.json`, `tokenizer_config.json`), e.g. from `~/.cache/huggingface/hub`.
The app uses it automatically; add `?remote=1` to force the Hub path.

## Ladder playground (`index.html`)

A second, much simpler page: pick a model from the
[NanoGPT Ladder v3](https://huggingface.co/collections/davidheineman/nanogpt-ladder-v3-completed-pretrained-models)
(public; base / chat-heavy / code-heavy / math-heavy × 8M–128M params × 10/30/100 tokens per parameter; default
`base-64m-30tpp`) or the 28 models of the
[NanoGPT Ladder v2](https://huggingface.co/collections/davidheineman/nanogpt-ladder-v2-28-completed-models)
(private, needs a Hugging Face token; 4 data mixtures × small/bridge/large × 250M–4B training tokens), type raw text and press
**Continue** (⌘/Ctrl+Enter) to have the model append to it. No chat template: what you see is exactly what the model sees.

- `src/chat/torchzip.ts` reads the original `model.pt` (`torch.save` zip + pickle) directly in the browser.
- `src/chat/gpu.ts` runs the model with WebGPU (f32, KV cache) in a web worker (`src/chat/worker.ts`): about
  300–400 tok/s for the largest model on an M-series Mac, and a 700-token prompt in under 0.2 s. After loading, it
  checks itself against the PyTorch reference token recorded in each repo's `manifest.json`.
- `src/chat/gpt.ts` is the plain-JS CPU fallback (about 18 tok/s for the large models); add `?cpu=1` to force it.
  Both match the repo's PyTorch `model.py` (greedy generations identical token for token).
- The repos are private. `npm run dev` passes `HF_TOKEN` from your shell to the page (dev server only, never in
  builds); otherwise the page asks for a token and keeps it in `localStorage`. Weights are cached in Cache Storage.

## Live RL on the NanoGPT ladder (`rl.html`)

Fine-tunes a ladder model with reinforcement learning in real time, entirely on your GPU: sample a group of
completions of one prompt, score each with a JavaScript reward function, and take one policy-gradient step towards
the above-average ones (REINFORCE with group-normalized advantages, i.e. GRPO without clipping or a KL penalty;
AdamW, global-norm clipping). The page streams the rollouts and plots the mean reward per step.

Rewards are Python, edited in a Monaco editor and run in the worker with [Pyodide](https://pyodide.org)
(CPython in WebAssembly, loaded from the jsDelivr CDN on first use): define `reward(text, info)`, where `info` has
`n`, `tokens`, `ended` and `prompt`; `re`, `math` and `is_word(w)` are available. Errors are shown on the
offending line.

- `src/rl/kernels.ts` + `trainer.ts`: hand-written WGSL forward (storing activations), backward (tiled GEMM,
  RMSNorm, GELU, RoPE, attention with recomputed softmax, chunked lm_head/loss, embedding scatter) and AdamW.
  Gradients match PyTorch autograd to float32 precision (within 2x of PyTorch-f32's own error against float64).
- Presets (`src/rl/tasks.ts`) are chosen so the base model already earns some reward: RL only sharpens what
  the model samples. "be positive" on general-large-4b goes from ~0.8 to ~23 in about 80 s (lr 3e-4) and ends up
  reward-hacking ("love you love you love you…"); "talk about cats" shows the opposite: the model never says
  "cat", so there is nothing to learn from.
- "length control" puts a target `{n}` in the prompt and rewards ending the line after n tokens. Each step mixes
  4 prompts with different n (8 samples each, advantages normalized per prompt), so every update contrasts the
  targets. With n ∈ {2, 5, 10, 15}: no effect for ~250 steps, then lengths fan out and match n by ~800 steps
  (corr(n, length) 0.99, ~5 min). It mostly memorizes one sentence per n (unseen n = 7 interpolates, n = 12 doesn't).
  One target per step only ever learned the average length. "is English" rewards words from a 19k-word list (`scripts/make_english_words.py`, from
  GPT-2's vocabulary and Webster's 1934 dictionary), times the share of distinct words (0.74 → 0.92 in 100 steps).
- About 0.8 s per step for the large models (16 × 48 tokens) and 0.5 s for the small ones on an M-series Mac.

## Dataset viewer (`data.html`)

Browses the parquet files of [tiny-rl-refined-tasks-v5.5](https://huggingface.co/datasets/davidheineman/tiny-rl-refined-tasks-v5.5)
straight from the Hub with [hyparquet](https://github.com/hyparam/hyparquet) (`src/data/main.js`): configs come from the
dataset card (per-task configs already covered by a broader one are skipped), tasks are the path segment the
config's `*` matched (folder or file name), rows stream in row group by row group, and the search box
takes a regex. Add `?repo=owner/name` to browse another dataset.

## What you see

- **Token bar**: click any token to make it the focus position. Chips are tinted by where the selected
  attention head looks from the focus token.
- **Overview (left)**: one row per layer, each showing what attention added to the residual stream, what the MLP added,
  the resulting residual stream (1024 dims), and the **logit lens** (the token the model would predict if it stopped
  at that layer).
- **Detail (right)**: the full computation of one layer: RMSNorm → Q/K/V → QK-norm + RoPE → 16 attention maps →
  o_proj → residual add → RMSNorm → gate/up → SwiGLU → down → residual add. Each weight matrix is a block-RMS
  thumbnail; click one to open the inspector and read actual bf16 values back from the GPU.
- **Walkthrough (bottom)**: steps through all 311 computation steps (or one stop per layer), narrating with the real
  numbers from the current forward pass. Keys: ←/→ step, space play/pause, ↑/↓ change layer.

## How it works

- `src/model/engine.ts` + `shaders.ts`: a hand-written Qwen3 forward pass in WGSL. Weights stay bf16 on the GPU
  (unpacked to f32 in-shader with a bit shift); activations are f32. There is no KV cache: every step recomputes the
  full sequence (max 256 tokens), which keeps attention maps for all positions and is fast enough here
  (about 100 ms per forward pass on an M-series Mac). Every intermediate at the focus position is copied into a capture
  buffer and read back in one go.
- `src/model/loader.ts`: safetensors header parsing, HTTP range download with progress, Cache Storage, and weight
  thumbnails (block RMS) computed once while uploading.
- `src/viz/*`: plain TypeScript + canvas, no framework.

## Correctness check

`scripts/reference.py` runs the same prompt through Hugging Face `transformers` and dumps logits, residual streams and
attention maps. `test.html` compares the WebGPU pass against that dump:

```bash
uv run --isolated --no-project --with torch --with "transformers>=4.51" python -I scripts/reference.py reference/reference.json
npm run test:e2e   # headless Chrome; prints SELFTEST PASS/FAIL
```

Current result: residual stream relative error ~3e-6, attention ~1e-6, logits ~3e-6, identical top-5 and identical
greedy continuation.
