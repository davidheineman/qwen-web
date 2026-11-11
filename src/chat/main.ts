import "./chat.css";
import { DEFAULT_MODEL, MODELS, fillModelPicker, type LadderModel } from "./models";
import { EXAMPLES } from "./examples";
import type { FromWorker, ToWorker } from "./worker";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modelSel = $<HTMLSelectElement>("model");
const text = $<HTMLTextAreaElement>("text");
const goBtn = $<HTMLButtonElement>("go");
const stopBtn = $<HTMLButtonElement>("stop");
const tempIn = $<HTMLInputElement>("temp");
const maxIn = $<HTMLInputElement>("max");
const statusEl = $("status");
const progress = $("progress");
const bar = $("bar");
const tokenBar = $("tokenbar");
const tokenInput = $<HTMLInputElement>("token");

const TOKEN_KEY = "nanogpt-ladder:hf-token";
const MODEL_KEY = "nanogpt-ladder:model";
const TOP_K = 40;

declare const __DEV_HF_TOKEN__: string;
/** token saved in this browser, else (dev server only) HF_TOKEN from the shell that ran `npm run dev` */
const hfToken = () => localStorage.getItem(TOKEN_KEY) || (import.meta.env.DEV && __DEV_HF_TOKEN__) || null;

let loaded: LadderModel | null = null;
/** the running continuation: the text it started from */
let running: { id: number; base: string } | null = null;
let genId = 0;

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (m: ToWorker) => worker.postMessage(m);
const byId = (id: string) => MODELS.find((m) => m.id === id);

fillModelPicker(modelSel);

function setStatus(s: string) {
  statusEl.textContent = s;
}

function update() {
  goBtn.disabled = !loaded;
  goBtn.hidden = !!running;
  stopBtn.hidden = !running;
  text.readOnly = !!running;
}

function loadSelected() {
  const m = byId(modelSel.value)!;
  localStorage.setItem(MODEL_KEY, m.id);
  const url = new URL(location.href);
  url.searchParams.set("model", m.id);
  history.replaceState(null, "", url);
  if (running) send({ type: "abort" });
  loaded = null;
  tokenBar.hidden = true;
  progress.hidden = false;
  bar.style.width = "0";
  setStatus(`Loading ${m.id}…`);
  update();
  send({ type: "load", repo: m.repo, token: hfToken(), cpu: new URLSearchParams(location.search).has("cpu") });
}

function go() {
  if (!loaded || running) return;
  const base = text.value;
  running = { id: ++genId, base };
  setStatus(`${loaded.id}: generating…`);
  update();
  send({
    type: "generate",
    id: running.id,
    prompt: base,
    stop: [],
    temperature: Math.max(0, Number(tempIn.value) || 0),
    topK: TOP_K,
    maxTokens: Math.max(1, Math.min(1000, Math.round(Number(maxIn.value) || 128))),
  });
}

function show(continuation: string) {
  if (!running) return;
  const atBottom = text.scrollHeight - text.scrollTop - text.clientHeight < 40;
  text.value = running.base + continuation;
  if (atBottom) text.scrollTop = text.scrollHeight;
}

for (const [name, prompt] of EXAMPLES) {
  const b = document.createElement("button");
  b.textContent = name;
  b.addEventListener("click", () => {
    if (running) return;
    text.value = prompt;
    text.focus();
    text.setSelectionRange(prompt.length, prompt.length);
    update();
  });
  $("examples").append(b);
}

modelSel.addEventListener("change", loadSelected);
goBtn.addEventListener("click", go);
stopBtn.addEventListener("click", () => send({ type: "abort" }));
text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    go();
  }
});
const saveToken = () => {
  const t = tokenInput.value.trim();
  if (!t) return;
  localStorage.setItem(TOKEN_KEY, t);
  tokenInput.value = "";
  loadSelected();
};
$("token-save").addEventListener("click", saveToken);
tokenInput.addEventListener("keydown", (e) => e.key === "Enter" && saveToken());

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "progress": {
      bar.style.width = `${m.total ? (100 * m.done) / m.total : 100}%`;
      const mb = (n: number) => (n / 1e6).toFixed(0);
      setStatus(m.total ? `${m.phase}: ${mb(m.done)} / ${mb(m.total)} MB` : `${m.phase}…`);
      break;
    }
    case "loaded": {
      const model = MODELS.find((x) => x.repo === m.repo)!;
      if (model.id !== modelSel.value) return; // superseded by another pick
      loaded = model;
      progress.hidden = true;
      setStatus(`${model.id} ready (${(m.params / 1e6).toFixed(1)}M params, d=${m.config.n_embd}, ${m.config.n_layer} layers, ${m.backend})`);
      update();
      text.focus();
      (window as unknown as { __ready: boolean }).__ready = true;
      break;
    }
    case "error":
      running = null;
      progress.hidden = true;
      setStatus(`Error: ${m.message}`);
      if (m.status === 401 || m.status === 403 || m.status === 404) {
        tokenBar.hidden = false;
        tokenInput.focus();
      }
      update();
      break;
    case "text":
      if (running?.id === m.id) show(m.text);
      break;
    case "done":
      if (running?.id === m.id) {
        show(m.text);
        const why = { stop: "", eos: ", hit end-of-text", length: "", abort: ", stopped" }[m.reason];
        setStatus(`${loaded?.id}: ${m.newTokens} tokens at ${m.tokPerSec.toFixed(1)} tok/s (${m.promptTokens} prompt tokens${why})`);
        running = null;
        update();
        text.focus();
      }
      break;
  }
};

const fromUrl = new URLSearchParams(location.search).get("model");
modelSel.value = byId(fromUrl ?? "")?.id ?? byId(localStorage.getItem(MODEL_KEY) ?? "")?.id ?? DEFAULT_MODEL;
loadSelected();
