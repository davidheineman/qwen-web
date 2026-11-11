// Shared by the chat and RL workers: GPT-2 tokenizer and ladder checkpoint download (cached in Cache Storage).

import { Tokenizer } from "@huggingface/tokenizers";

const GPT2 = "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/";
const CACHE_NAME = "nanogpt-ladder-weights";

export class HttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

let tokenizer: Promise<Tokenizer> | null = null;

export function getTokenizer(): Promise<Tokenizer> {
  tokenizer ??= Promise.all([
    fetch(GPT2 + "tokenizer.json").then((r) => r.json()),
    fetch(GPT2 + "tokenizer_config.json").then((r) => r.json()),
  ]).then(([json, config]) => new Tokenizer(json, config));
  return tokenizer;
}

/** `repo` is "owner/name" or "owner/name/sub/dir" (a model stored in a folder of a larger repo). */
export function fileUrl(repo: string, file: string): string {
  const [owner, name, ...dir] = repo.split("/");
  return `https://huggingface.co/${owner}/${name}/resolve/main/${[...dir, file].join("/")}`;
}

export type OnProgress = (phase: string, done: number, total: number) => void;

/** Download `model.pt` of a repo (or read it from Cache Storage). `cancelled()` aborts mid-download. */
export async function downloadCheckpoint(
  repo: string,
  token: string | null,
  onProgress: OnProgress,
  cancelled: () => boolean = () => false,
): Promise<{ bytes: Uint8Array; source: string }> {
  const url = fileUrl(repo, "model.pt");
  const cache = "caches" in self ? await caches.open(CACHE_NAME) : null;
  const hit = await cache?.match(url);
  if (hit) {
    onProgress("Reading from browser cache", 0, 1);
    return { bytes: new Uint8Array(await hit.arrayBuffer()), source: "browser cache" };
  }
  const r = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!r.ok) {
    const hint =
      r.status === 401 || r.status === 403 || r.status === 404
        ? token
          ? "this Hugging Face token can't read the repo"
          : "private repo, needs a Hugging Face token"
        : r.statusText;
    throw new HttpError(`HTTP ${r.status}: ${hint}`, r.status);
  }
  const total = Number(r.headers.get("content-length")) || 0;
  const chunks: Uint8Array[] = [];
  let done = 0;
  let last = 0;
  const reader = r.body!.getReader();
  for (;;) {
    const { done: end, value } = await reader.read();
    if (end) break;
    if (cancelled()) {
      await reader.cancel();
      throw new Error("superseded");
    }
    chunks.push(value);
    done += value.length;
    if (done - last > 1 << 20) {
      last = done;
      onProgress("Downloading weights", done, total);
    }
  }
  const bytes = new Uint8Array(done);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
    } catch (e) {
      console.warn("could not cache weights", e);
    }
  }
  return { bytes, source: "huggingface.co" };
}

/** Each repo's manifest.json records a PyTorch check: the argmax next token for some input ids. */
export async function fetchValidation(repo: string, token: string | null): Promise<{ input_ids: number[]; next_token_argmax: number } | null> {
  try {
    const r = await fetch(fileUrl(repo, "manifest.json"), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    const v = (await r.json()).validation;
    return v?.input_ids?.length ? v : null;
  } catch {
    return null;
  }
}
