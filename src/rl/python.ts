// Python reward functions, run with Pyodide (CPython compiled to WebAssembly) inside the RL worker.
//
// The user's code must define `reward(text, info) -> float`. Each step scores the whole batch with one
// call into Python (JSON in, JSON out). `info` is a SimpleNamespace with n, tokens, ended, prompt;
// `is_word(w)` and `re`, `math` are available as globals.

import WORDS from "./english-words.txt?raw";

// Pinned; Pyodide's version tracks CPython (314.x = Python 3.14). Loaded from the CDN, cached by the browser.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

interface Pyodide {
  runPython(code: string, opts?: { globals?: unknown }): unknown;
  globals: { get(name: string): unknown; set(name: string, v: unknown): void };
  toPy(v: unknown): { destroy(): void };
}

const PRELUDE = `
import json, math, re
from types import SimpleNamespace

_WORDS = set(_WORDS_TXT.split("\\n"))
_SUFFIXES = ("s", "es", "ed", "d", "ing", "ly", "er", "est")

def is_word(word):
    """Is this a (reasonably common) English word? Handles case, contractions and simple inflections."""
    w = re.sub(r"['’](s|t|re|ll|ve|d|m)$", "", word.lower())
    return w in _WORDS or any(w.endswith(s) and len(w) - len(s) >= 3 and w[: -len(s)] in _WORDS for s in _SUFFIXES)

def _compile(src):
    ns = {"re": re, "math": math, "is_word": is_word, "__name__": "reward"}
    exec(compile(src, "reward.py", "exec"), ns)
    fn = ns.get("reward")
    if not callable(fn):
        raise NameError("define a function: def reward(text, info): ...")
    return fn

def _score_all(fn, payload):
    rows = json.loads(payload)
    out = []
    for r in rows:
        v = fn(r["text"], SimpleNamespace(**r["info"]))
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"reward returned {v} for {r['text'][:40]!r}")
        out.append(v)
    return json.dumps(out)
`;

let py: Promise<Pyodide> | null = null;

export function loadPython(): Promise<Pyodide> {
  py ??= (async () => {
    const { loadPyodide } = (await import(/* @vite-ignore */ PYODIDE_URL + "pyodide.mjs")) as {
      loadPyodide: (o: { indexURL: string }) => Promise<Pyodide>;
    };
    const p = await loadPyodide({ indexURL: PYODIDE_URL });
    p.globals.set("_WORDS_TXT", WORDS);
    p.runPython(PRELUDE);
    return p;
  })();
  return py;
}

export interface RewardInput {
  text: string;
  info: { n: number | null; tokens: number; ended: boolean; prompt: string };
}

/** Last line of a Python traceback (plus the line number in reward.py, if any). */
function pyError(e: unknown): Error {
  const msg = String((e as Error).message ?? e);
  const lines = msg.trim().split("\n");
  const where = [...msg.matchAll(/File "reward\.py", line (\d+)/g)].pop();
  return new Error(`${lines[lines.length - 1]}${where ? ` (reward.py line ${where[1]})` : ""}`);
}

export class PythonReward {
  private constructor(private p: Pyodide, private fn: unknown) {}

  /** Compile the user's source; throws a readable error if it doesn't define reward(). */
  static async compile(src: string): Promise<PythonReward> {
    const p = await loadPython();
    const compileFn = p.globals.get("_compile") as (s: string) => unknown;
    try {
      return new PythonReward(p, compileFn(src));
    } catch (e) {
      throw pyError(e);
    }
  }

  score(rows: RewardInput[]): number[] {
    const scoreAll = this.p.globals.get("_score_all") as (fn: unknown, payload: string) => string;
    try {
      return JSON.parse(scoreAll(this.fn, JSON.stringify(rows)));
    } catch (e) {
      throw pyError(e);
    }
  }
}
