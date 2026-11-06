# Qwen3-0.6B, inside out

Runs [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) entirely in the browser with WebGPU and shows
every weight matrix and every intermediate activation, layer by layer, for each predicted token.
Inspired by [bbycroft.net/llm](https://bbycroft.net/llm), but running a real 0.6B model on your own text.

**Live demo: https://davidheineman.github.io/qwen-web/** (needs a WebGPU browser; downloads ~1.19 GB of weights once)

```bash
npm install
npm run dev        # http://localhost:5173 (needs WebGPU: Chrome / Edge / Safari 26+)
```

On first load the page downloads the original bf16 weights straight from the Hugging Face Hub (about 1.19 GB;
`lm_head` is skipped because it is tied to the embedding) and caches them in the browser's Cache Storage.

To skip the download during development, put a copy in `public/models/Qwen3-0.6B/`
(`model.safetensors`, `tokenizer.json`, `tokenizer_config.json`), e.g. from `~/.cache/huggingface/hub`.
The app uses it automatically; add `?remote=1` to force the Hub path.

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
