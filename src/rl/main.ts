import "./rl.css";
import { DEFAULT_MODEL, MODELS, fillModelPicker } from "../chat/models";
import { DEFAULT_MAX_NEW, TASKS } from "./tasks";
import { createEditor } from "./editor";
import type { FromWorker, Settings, StepResult, ToWorker } from "./worker";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modelSel = $<HTMLSelectElement>("model");
const taskSel = $<HTMLSelectElement>("task");
const promptIn = $<HTMLInputElement>("prompt");
const rewardStatus = $("reward-status");
const targetsIn = $<HTMLInputElement>("targets");
const stopIn = $<HTMLInputElement>("stop");
const ppsSel = $<HTMLSelectElement>("pps");
const entIn = $<HTMLInputElement>("ent");
const note = $("note");
const runBtn = $<HTMLButtonElement>("run");
const resetBtn = $<HTMLButtonElement>("reset");
const groupSel = $<HTMLSelectElement>("group");
const maxIn = $<HTMLInputElement>("maxNew");
const lrIn = $<HTMLInputElement>("lr");
const tempIn = $<HTMLInputElement>("temp");
const topkIn = $<HTMLInputElement>("topk");
const statusEl = $("status");
const statsEl = $("stats");
const progress = $("progress");
const bar = $("bar");
const list = $<HTMLOListElement>("rollouts");
const tokenBar = $("tokenbar");
const tokenInput = $<HTMLInputElement>("token");

const TOKEN_KEY = "nanogpt-ladder:hf-token";
declare const __DEV_HF_TOKEN__: string;
const hfToken = () => localStorage.getItem(TOKEN_KEY) || (import.meta.env.DEV && __DEV_HF_TOKEN__) || null;

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (m: ToWorker) => worker.postMessage(m);

let loaded = false;
let running = false;
let highlight: RegExp | null = null;
let markNonEnglish = false;
let englishWords: Set<string> | null = null;
import("./english-words.txt?raw").then((m) => (englishWords = new Set(m.default.split("\n"))));
const showTargets = () => {
  const on = promptIn.value.includes("{n}");
  $("targets-row").hidden = !on;
  $("lenchart").hidden = !on;
};

// ---------------------------------------------------------------- settings

fillModelPicker(modelSel);
TASKS.forEach((t, i) => taskSel.append(new Option(t.name, String(i))));
taskSel.append(new Option("custom", "custom"));

function applyTask() {
  const t = TASKS[Number(taskSel.value)];
  if (!t) return;
  promptIn.value = t.prompt;
  targetsIn.value = t.targets ?? "5";
  stopIn.checked = !!t.stopAtNewline;
  lrIn.value = String(t.lr ?? 3e-4);
  entIn.value = String(t.entropy ?? 0);
  groupSel.value = String(t.group ?? 16);
  ppsSel.value = String(t.promptsPerStep ?? 1);
  maxIn.value = String(t.maxNew ?? DEFAULT_MAX_NEW);
  rewardEditor.setValue(t.reward);
  note.textContent = t.note;
  highlight = t.highlight && t.highlight !== "NOT_ENGLISH" ? new RegExp(t.highlight, "gi") : null;
  markNonEnglish = t.highlight === "NOT_ENGLISH";
  showTargets();
  pushSettings();
}

/** "2-16" or "3, 5, 8" (or a mix) -> list of positive integers */
function parseTargets(s: string): number[] {
  const out: number[] = [];
  for (const part of s.split(/[\s,]+/)) {
    const m = /^(\d+)-(\d+)$/.exec(part);
    if (m) for (let x = +m[1]; x <= +m[2] && out.length < 1000; x++) out.push(x);
    else if (/^\d+$/.test(part) && +part > 0) out.push(+part);
  }
  return out;
}

function settings(): Settings {
  return {
    prompt: promptIn.value,
    targets: parseTargets(targetsIn.value),
    promptsPerStep: Number(ppsSel.value),
    stop: stopIn.checked ? "\n" : null,
    reward: rewardEditor.getValue(),
    group: Number(groupSel.value),
    maxNew: Math.max(8, Math.min(128, Math.round(Number(maxIn.value) || DEFAULT_MAX_NEW))),
    lr: Math.max(0, Number(lrIn.value) || 0),
    entropy: Math.max(0, Number(entIn.value) || 0),
    temperature: Math.max(0.1, Number(tempIn.value) || 1),
    topK: Math.max(0, Math.round(Number(topkIn.value) || 0)),
  };
}

/** Send settings to the worker (it compiles the Python reward and reports back with a "reward" message). */
function pushSettings() {
  send({ type: "settings", settings: settings() });
}

let editTimer = 0;
const rewardEditor = createEditor(
  $("reward"),
  () => {
    if (taskSel.value !== "custom") {
      taskSel.value = "custom";
      note.textContent = "Custom reward.";
      highlight = null;
      markNonEnglish = false;
    }
    clearTimeout(editTimer);
    editTimer = window.setTimeout(pushSettings, 600); // recompile shortly after typing stops
  },
  pushSettings,
);

taskSel.addEventListener("change", applyTask);
for (const el of [promptIn, targetsIn, ppsSel, entIn, stopIn, groupSel, maxIn, lrIn, tempIn, topkIn]) el.addEventListener("change", pushSettings);
promptIn.addEventListener("input", showTargets);

function loadModel() {
  const m = MODELS.find((x) => x.id === modelSel.value)!;
  const url = new URL(location.href);
  url.searchParams.set("model", m.id);
  history.replaceState(null, "", url);
  loaded = false;
  running = false;
  tokenBar.hidden = true;
  progress.hidden = false;
  bar.style.width = "0";
  setStatus(`Loading ${m.id}…`);
  update();
  chart.clear();
  lenChart.draw([]);
  lastStep = null;
  list.replaceChildren();
  send({ type: "load", repo: m.repo, token: hfToken() });
}
modelSel.addEventListener("change", loadModel);

let runError = ""; // why the last run stopped, kept on screen until the next Start
runBtn.addEventListener("click", () => {
  if (!running) {
    runError = "";
    pushSettings();
  }
  send({ type: "run", on: !running });
});
resetBtn.addEventListener("click", () => {
  send({ type: "reset" });
  lastStep = null;
  lenChart.draw([]);
  chart.clear();
  list.replaceChildren();
  statsEl.textContent = "";
});
const saveToken = () => {
  const t = tokenInput.value.trim();
  if (!t) return;
  localStorage.setItem(TOKEN_KEY, t);
  tokenInput.value = "";
  loadModel();
};
$("token-save").addEventListener("click", saveToken);
tokenInput.addEventListener("keydown", (e) => e.key === "Enter" && saveToken());

function setStatus(s: string) {
  statusEl.textContent = s;
}
function update() {
  runBtn.disabled = !loaded;
  resetBtn.disabled = !loaded;
  runBtn.textContent = running ? "Pause" : "Start";
}

// ---------------------------------------------------------------- rollouts

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const SUFFIXES = ["s", "es", "ed", "d", "ing", "ly", "er", "est"];
function looksEnglish(word: string) {
  const w = word.toLowerCase().replace(/['’](s|t|re|ll|ve|d|m)$/, "");
  return !englishWords || englishWords.has(w) ||
    SUFFIXES.some((x) => w.endsWith(x) && w.length - x.length >= 3 && englishWords!.has(w.slice(0, -x.length)));
}
function marked(s: string) {
  if (markNonEnglish) {
    return s.split(/([a-z]+(?:['’][a-z]+)?)/gi).map((part, i) => (i % 2 && !looksEnglish(part) ? `<mark>${esc(part)}</mark>` : esc(part))).join("");
  }
  if (!highlight) return esc(s);
  let out = "";
  let last = 0;
  for (const m of s.matchAll(highlight)) {
    if (!m[0]) continue;
    out += esc(s.slice(last, m.index)) + `<mark>${esc(m[0])}</mark>`;
    last = m.index! + m[0].length;
  }
  return out + esc(s.slice(last));
}

function renderRollouts(texts: string[], opts: { prompts: string[]; ns?: number[]; rewards?: number[]; adv?: number[]; done?: boolean[]; lengths?: number[] }) {
  const order = texts.map((_, i) => i);
  const nOf = (i: number) => (opts.ns && Number.isFinite(opts.ns[i]) ? opts.ns[i] : 0);
  if (opts.rewards) order.sort((a, b) => nOf(a) - nOf(b) || opts.rewards![b] - opts.rewards![a]);
  list.replaceChildren(
    ...order.map((i) => {
      const li = document.createElement("li");
      const a = opts.adv?.[i] ?? 0;
      if (opts.adv) {
        const pole = a >= 0 ? "var(--series-1)" : "var(--neg)";
        li.style.background = `color-mix(in srgb, ${pole} ${Math.min(30, Math.abs(a) * 15).toFixed(0)}%, var(--surface))`;
      } else li.className = "live";
      const r = opts.rewards ? opts.rewards[i].toFixed(2).replace(/\.00$/, "") : "…";
      const len = opts.lengths ? ` · ${opts.lengths[i]} tok` : "";
      const end = opts.done?.[i] || len ? ` <span class="eot">${opts.done?.[i] ? "⏎end" : ""}${len}</span>` : "";
      li.innerHTML = `<span class="r">${r}</span><span class="t"><span class="p">${esc(opts.prompts[i])}</span>${marked(texts[i])}${end}</span>`;
      return li;
    }),
  );
}

// ---------------------------------------------------------------- chart

class RewardChart {
  private pts: { step: number; mean: number; avg: number }[] = [];
  private hover: number | null = null;
  private ctx: CanvasRenderingContext2D;
  private tip = $("tip");
  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener("mousemove", (e) => {
      if (!this.pts.length) return;
      const { x0, x1 } = this.frame();
      const rect = canvas.getBoundingClientRect();
      const f = (e.clientX - rect.left - x0) / Math.max(1, x1 - x0);
      this.hover = Math.max(0, Math.min(this.pts.length - 1, Math.round(f * (this.pts.length - 1))));
      this.draw();
    });
    canvas.addEventListener("mouseleave", () => {
      this.hover = null;
      this.draw();
    });
  }
  clear() {
    this.pts = [];
    this.draw();
  }
  push(step: number, mean: number) {
    const w = this.pts.slice(-9).map((p) => p.mean);
    w.push(mean);
    this.pts.push({ step, mean, avg: w.reduce((a, b) => a + b, 0) / w.length });
    this.draw();
  }
  private frame() {
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    return { W, H, x0: 40, x1: W - 52, y0: 8, y1: H - 22 };
  }
  draw() {
    const { canvas, ctx } = this;
    const dpr = devicePixelRatio || 1;
    const { W, H, x0, x1, y0, y1 } = this.frame();
    if (canvas.width !== Math.round(W * dpr)) canvas.width = Math.round(W * dpr);
    if (canvas.height !== Math.round(H * dpr)) canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const C = {
      grid: css.getPropertyValue("--line").trim(),
      text: css.getPropertyValue("--muted").trim(),
      ink: css.getPropertyValue("--text").trim(),
      series: css.getPropertyValue("--series-1").trim(),
    };
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    const pts = this.pts;
    let lo = Math.min(0, ...pts.map((p) => p.mean));
    let hi = Math.max(1, ...pts.map((p) => p.mean));
    const span = hi - lo;
    lo -= span * 0.05;
    hi += span * 0.08;
    // nice ticks
    const raw = (hi - lo) / 4;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const stepT = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
    const X = (i: number) => x0 + (pts.length > 1 ? (i / (pts.length - 1)) * (x1 - x0) : 0);
    const Y = (v: number) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);
    ctx.lineWidth = 1;
    ctx.textBaseline = "middle";
    for (let v = Math.ceil(lo / stepT) * stepT; v <= hi; v += stepT) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.strokeStyle = C.grid;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.textAlign = "right";
      ctx.fillText(String(+v.toFixed(3)), x0 - 6, y);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    if (!pts.length) {
      ctx.fillStyle = C.text;
      ctx.fillText("press Start: one point per training step", x0 + 8, (y0 + y1) / 2);
      this.tip.hidden = true;
      return;
    }
    ctx.fillStyle = C.text;
    ctx.fillText(`step ${pts[0].step}`, x0, H - 6);
    ctx.textAlign = "right";
    ctx.fillText(`step ${pts[pts.length - 1].step}`, x1, H - 6);
    ctx.textAlign = "left";
    // per-step means: light dots
    ctx.fillStyle = C.series;
    ctx.globalAlpha = 0.3;
    const r = pts.length > 150 ? 2.5 : 4;
    for (let i = 0; i < pts.length; i++) {
      ctx.beginPath();
      ctx.arc(X(i), Y(pts[i].mean), r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    // moving average: 2px line
    ctx.strokeStyle = C.series;
    ctx.lineWidth = 2;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p.avg)) : ctx.moveTo(X(i), Y(p.avg))));
    ctx.stroke();
    // end dot + value label
    const last = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(X(pts.length - 1), Y(last.avg), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = C.ink;
    ctx.textBaseline = "middle";
    ctx.fillText(last.avg.toFixed(2), x1 + 8, Y(last.avg));
    // hover crosshair + tooltip
    if (this.hover !== null && pts[this.hover]) {
      const p = pts[this.hover];
      const x = Math.round(X(this.hover)) + 0.5;
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
      ctx.fillStyle = C.series;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, Y(p.avg), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      this.tip.hidden = false;
      this.tip.innerHTML = `step <b>${p.step}</b> · group mean ${p.mean.toFixed(2)} · 10-step avg ${p.avg.toFixed(2)}`;
      const tw = this.tip.offsetWidth;
      this.tip.style.left = `${Math.min(W - tw, Math.max(0, x + 10))}px`;
    } else this.tip.hidden = true;
  }
}
const chart = new RewardChart($<HTMLCanvasElement>("canvas"));

/** Mean generated length per target n (ended completions of recent steps), against the y = x line. */
class LengthChart {
  private pts: { n: number; mean: number; count: number }[] = [];
  private hover: number | null = null;
  private ctx: CanvasRenderingContext2D;
  private tip = $("lentip");
  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    new ResizeObserver(() => this.render()).observe(canvas);
    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const { X } = this.scales();
      let best: number | null = null;
      let bd = 24;
      this.pts.forEach((p, i) => {
        const dx = Math.abs(X(p.n) - (e.clientX - rect.left));
        if (dx < bd) { bd = dx; best = i; }
      });
      this.hover = best;
      this.render();
    });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.render(); });
  }
  draw(steps: StepResult[]) {
    const by = new Map<number, number[]>();
    for (const s of steps) {
      s.ns.forEach((n, i) => {
        if (!Number.isFinite(n) || !s.ended[i]) return;
        (by.get(n) ?? by.set(n, []).get(n)!).push(s.lengths[i]);
      });
    }
    this.pts = [...by].map(([n, L]) => ({ n, mean: L.reduce((a, b) => a + b, 0) / L.length, count: L.length })).sort((a, b) => a.n - b.n);
    this.render();
  }
  private scales() {
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    const x0 = 40, x1 = W - 16, y0 = 8, y1 = H - 24;
    const targets = parseTargets(targetsIn.value);
    const hi = Math.max(4, ...targets, ...this.pts.map((p) => p.n), ...this.pts.map((p) => p.mean)) * 1.05;
    const X = (v: number) => x0 + (v / hi) * (x1 - x0);
    const Y = (v: number) => y1 - (v / hi) * (y1 - y0);
    return { W, H, x0, x1, y0, y1, hi, X, Y };
  }
  private render() {
    const { canvas, ctx } = this;
    const dpr = devicePixelRatio || 1;
    const { W, H, x0, x1, y0, y1, hi, X, Y } = this.scales();
    if (!W) return;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const grid = css.getPropertyValue("--line").trim(), muted = css.getPropertyValue("--muted").trim();
    const series = css.getPropertyValue("--series-1").trim();
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    const step = hi > 40 ? 10 : hi > 16 ? 5 : 2;
    ctx.lineWidth = 1;
    for (let v = 0; v <= hi; v += step) {
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(x0, Math.round(Y(v)) + 0.5); ctx.lineTo(x1, Math.round(Y(v)) + 0.5); ctx.stroke();
      ctx.fillStyle = muted;
      ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(v), x0 - 6, Y(v));
      ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; ctx.fillText(String(v), X(v), H - 8);
    }
    // y = x reference
    ctx.strokeStyle = muted;
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(hi), Y(hi)); ctx.stroke();
    ctx.fillStyle = muted; ctx.textAlign = "left";
    ctx.fillText("target n →", x1 - 60, y1 - 6);
    if (!this.pts.length) {
      ctx.fillText("no finished replies yet", x0 + 8, y0 + 14);
      this.tip.hidden = true;
      return;
    }
    // per-n means: 2px-ringed dots
    this.pts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(X(p.n), Y(p.mean), i === this.hover ? 6 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = series; ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
    });
    if (this.hover !== null) {
      const p = this.pts[this.hover];
      this.tip.hidden = false;
      this.tip.innerHTML = `n = <b>${p.n}</b> · mean length ${p.mean.toFixed(1)} tokens · ${p.count} replies`;
      this.tip.style.left = `${Math.min(W - this.tip.offsetWidth, Math.max(0, X(p.n) + 10))}px`;
    } else this.tip.hidden = true;
  }
}
const lenChart = new LengthChart($<HTMLCanvasElement>("lencanvas"));

// ---------------------------------------------------------------- worker

const fmt = (n: number, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : "–");
const stepLog: StepResult[] = [];
(window as unknown as { __rl: StepResult[] }).__rl = stepLog; // for scripted tests
let lastStep: StepResult | null = null;
function showStep(r: StepResult) {
  lastStep = r;
  stepLog.push({ ...r, texts: [], prompts: [] });
  chart.push(r.step, r.mean);
  renderRollouts(r.texts, { prompts: r.prompts, ns: r.ns, rewards: r.rewards, adv: r.adv, done: r.ended, lengths: r.lengths });
  const nset = [...new Set(r.ns.filter(Number.isFinite))].sort((a, b) => a - b);
  $("rollouts-title").textContent = `Rollouts · step ${r.step}` + (nset.length ? ` · n = ${nset.join(", ")}` : "");
  if (nset.length) lenChart.draw(stepLog.slice(-10));
  const toks = r.texts.length * settings().maxNew;
  statsEl.textContent =
    `step ${r.step} · mean reward ${fmt(r.mean)} (std ${fmt(r.std)}) · ` +
    (r.trained ? `grad norm ${fmt(r.gradNorm)} · entropy ${fmt(r.entropy)} nats/token · ` : "all rewards equal, no update · ") +
    `sampling ${fmt(r.genMs / 1000, 1)}s (≤${toks} tokens) + training ${fmt(r.trainMs / 1000, 1)}s`;
}

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "progress":
      bar.style.width = `${m.total ? (100 * m.done) / m.total : 100}%`;
      setStatus(m.total ? `${m.phase}: ${(m.done / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB` : `${m.phase}…`);
      break;
    case "loaded":
      loaded = true;
      progress.hidden = true;
      setStatus(`${modelSel.value} ready (d=${m.config.n_embd}, ${m.config.n_layer} layers). Press Start.`);
      update();
      (window as unknown as { __ready: boolean }).__ready = true;
      break;
    case "reward": {
      rewardStatus.textContent = m.ok ? "✓ reward() compiled (Python)" : `✗ ${m.message}`;
      rewardStatus.classList.toggle("bad", !m.ok);
      const line = /reward\.py line (\d+)/.exec(m.message);
      rewardEditor.setError(m.ok ? null : line ? Number(line[1]) : 1, m.message);
      break;
    }
    case "error": {
      progress.hidden = true;
      runError = m.message;
      setStatus(`Error: ${m.message}`);
      // an exception inside reward(): show it under the editor and mark the line
      const line = /reward\.py line (\d+)/.exec(m.message);
      if (line) {
        rewardStatus.textContent = `✗ ${m.message}`;
        rewardStatus.classList.add("bad");
        rewardEditor.setError(Number(line[1]), m.message);
      }
      if (m.status === 401 || m.status === 403 || m.status === 404) tokenBar.hidden = false;
      update();
      break;
    }
    case "rollout":
      renderRollouts(m.texts, { prompts: m.prompts, done: m.done });
      break;
    case "step":
      showStep(m.result);
      break;
    case "state":
      running = m.running;
      // a generation cut short by Pause leaves half-finished rollouts: show the last scored group again
      if (!running && lastStep) {
        const r = lastStep;
        renderRollouts(r.texts, { prompts: r.prompts, ns: r.ns, rewards: r.rewards, adv: r.adv, done: r.ended, lengths: r.lengths });
      }
      if (loaded) {
        if (running) setStatus("training…");
        else if (runError) setStatus(`Stopped at step ${m.step}: ${runError}`);
        else setStatus(m.step ? `paused at step ${m.step}` : `${modelSel.value} ready. Press Start.`);
      }
      update();
      break;
  }
};

// ---------------------------------------------------------------- start

const fromUrl = new URLSearchParams(location.search).get("model");
modelSel.value = MODELS.some((x) => x.id === fromUrl) ? fromUrl! : DEFAULT_MODEL;
taskSel.value = String(Math.max(0, TASKS.findIndex((t) => t.name === "be positive")));
applyTask();
loadModel();
