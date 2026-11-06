import "./style.css";
import { CFG } from "./model/config";
import { resolveBase, loadRawWeights } from "./model/loader";
import { Engine, sample, type Capture } from "./model/engine";
import { Tok, visible } from "./model/tokenizer";
import { $, h, esc } from "./viz/dom";
import { type Ctx, EMBED, FINAL, probs } from "./viz/ctx";
import { renderOverview } from "./viz/overview";
import { renderDetail, tokenBars } from "./viz/detail";
import { openInspector } from "./viz/inspector";
import { buildStages, narrate, type Stage } from "./viz/stages";
import { showTip } from "./viz/draw";

const status = (s: string) => ($("#status").textContent = s);

const state = {
  eng: null as unknown as Engine,
  tok: null as unknown as Tok,
  tokens: [] as number[],
  promptLen: 0,
  cap: null as Capture | null,
  layer: 0,
  head: 0,
  scaleMode: "row" as "row" | "shared",
  generating: false,
  stopReq: false,
  walk: { stages: buildStages(false), i: -1, playing: false, perLayer: false, timer: 0, speed: 1600 },
};

// ---------------------------------------------------------------- forward runs (serialized)
let chain: Promise<unknown> = Promise.resolve();
function run(tokens: number[], focus = tokens.length - 1): Promise<Capture> {
  const p = chain.then(() => state.eng.forward(tokens, focus));
  chain = p.catch(() => {});
  return p;
}

async function refresh(focus?: number) {
  if (!state.tokens.length) return;
  const cap = await run(state.tokens, focus ?? state.tokens.length - 1);
  state.cap = cap;
  renderAll();
}

// ---------------------------------------------------------------- context for views
function ctx(): Ctx {
  const cap = state.cap!;
  const pieceCache = new Map<number, string>();
  const piece = (id: number) => {
    let s = pieceCache.get(id);
    if (s === undefined) pieceCache.set(id, (s = state.tok.piece(id)));
    return s;
  };
  return {
    cap,
    eng: state.eng,
    tok: state.tok,
    promptLen: state.promptLen,
    layer: state.layer,
    head: state.head,
    scaleMode: state.scaleMode,
    label: (i) => piece(cap.tokens[i]),
    piece,
    selectLayer: (l) => {
      state.layer = l;
      renderOverviewOnly();
      renderDetailOnly();
      syncWalkToLayer();
    },
    selectHead: (hh) => {
      state.head = hh;
      renderDetailOnly(true);
      renderTokens();
    },
    setFocus: (pos) => {
      if (state.generating) return;
      void refresh(pos);
    },
    setScaleMode: (m) => {
      state.scaleMode = m;
      renderOverviewOnly();
    },
    inspect: (name, row) => openInspector(ctx(), name, row),
  };
}

// ---------------------------------------------------------------- rendering
function renderAll() {
  renderTokens();
  renderPredict();
  renderOverviewOnly();
  renderDetailOnly(true);
  renderWalk();
}

function keepScroll(el: HTMLElement, fn: () => void) {
  const top = el.scrollTop;
  fn();
  el.scrollTop = top;
}

function renderOverviewOnly() {
  if (!state.cap) return;
  const el = $("#overview");
  keepScroll(el, () => el.replaceChildren(renderOverview(ctx())));
  highlightWalk();
}

function renderDetailOnly(keep = false) {
  if (!state.cap) return;
  const el = $("#detail");
  if (keep) keepScroll(el, () => el.replaceChildren(renderDetail(ctx())));
  else {
    el.replaceChildren(renderDetail(ctx()));
    el.scrollTop = 0;
  }
  highlightWalk();
}

function renderTokens() {
  const cap = state.cap;
  const el = $("#tokens");
  el.replaceChildren();
  if (!cap) return;
  const c = ctx();
  // Color chips by where the selected head (at the selected layer) attends from the focus token.
  const showAtt = state.layer >= 0 && state.layer < FINAL;
  const P = showAtt ? probs(cap, state.layer, state.head) : null;
  cap.tokens.forEach((id, i) => {
    const chip = h("span", { class: `chip${i >= state.promptLen ? " gen" : ""}${i === cap.focus ? " focus" : ""}` });
    if (P && i <= cap.focus) {
      const a = P[cap.focus * cap.T + i];
      chip.append(h("span", { class: "att", style: { opacity: String(Math.min(1, Math.sqrt(a)) * 0.85) } }));
    }
    chip.append(h("span", null, visible(c.piece(id))));
    chip.addEventListener("click", () => c.setFocus(i));
    chip.addEventListener("mousemove", (ev) => {
      const att = P && i <= cap.focus ? `\nattention from focus (L${state.layer} h${state.head}): ${(P[cap.focus * cap.T + i] * 100).toFixed(1)}%` : "";
      showTip(`<b>${esc(JSON.stringify(c.piece(id)))}</b>  id ${id}\nposition ${i}${att}\nclick to focus this position`, ev);
    });
    chip.addEventListener("mouseleave", () => showTip(null));
    el.append(chip);
  });
}

function renderPredict() {
  const cap = state.cap;
  const el = $("#predict");
  if (!cap) return;
  const c = ctx();
  const actual = cap.focus + 1 < cap.T ? cap.tokens[cap.focus + 1] : undefined;
  el.replaceChildren(
    h("div", { class: "head", html: `next token after “${esc(visible(c.label(cap.focus)))}”` }),
    tokenBars(cap.lens[CFG.layers].slice(0, 5), c, actual),
  );
}

// ---------------------------------------------------------------- walkthrough
function renderWalk() {
  const w = state.walk;
  const el = $("#walk");
  const st = w.stages[w.i];
  const where = !st ? "" : st.layer === EMBED ? "embedding" : st.layer === FINAL ? "output" : `layer ${st.layer}`;
  const sel = h(
    "select",
    { title: "Walkthrough granularity" },
    h("option", { value: "all", selected: !w.perLayer }, "every step"),
    h("option", { value: "layer", selected: w.perLayer }, "one stop per layer"),
  );
  sel.addEventListener("change", () => setGranularity(sel.value === "layer"));
  const speed = h(
    "select",
    { title: "Playback speed" },
    ...[["2600", "slow"], ["1600", "normal"], ["700", "fast"], ["250", "very fast"]].map(([v, t]) =>
      h("option", { value: v, selected: Number(v) === w.speed }, t),
    ),
  );
  speed.addEventListener("change", () => (w.speed = Number(speed.value)));
  el.replaceChildren(
    h(
      "div",
      { class: "btns" },
      h("button", { title: "Restart (Home)", onclick: () => goStage(0) }, "⏮"),
      h("button", { title: "Previous step (←)", onclick: () => goStage(w.i - 1) }, "◀"),
      h("button", { class: "primary", title: "Play / pause (space)", onclick: togglePlay }, w.playing ? "❚❚" : "▶ Walk through"),
      h("button", { title: "Next step (→)", onclick: () => goStage(w.i + 1) }, "▶"),
    ),
    h("div", { class: "where" }, st ? `${where}\n${w.i + 1}/${w.stages.length}` : `${w.stages.length} steps`),
    h(
      "div",
      { class: "txt", html: st && state.cap ? narrate(st, ctx()) : "Press <b>Walk through</b> to follow one token through the whole network, step by step. Or click any layer on the left." },
    ),
    sel,
    speed,
  );
}

function setGranularity(perLayer: boolean) {
  const w = state.walk;
  const cur = w.stages[w.i];
  w.perLayer = perLayer;
  w.stages = buildStages(perLayer);
  w.i = cur ? Math.max(0, w.stages.findIndex((s) => s.layer >= cur.layer)) : -1;
  renderWalk();
}

function goStage(i: number) {
  const w = state.walk;
  if (!state.cap) return;
  i = Math.max(0, Math.min(w.stages.length - 1, i));
  w.i = i;
  const st = w.stages[i];
  if (st.layer !== state.layer) {
    state.layer = st.layer;
    renderOverviewOnly();
    renderDetailOnly();
    renderTokens();
  }
  highlightWalk(true);
  renderWalk();
  if (i === w.stages.length - 1 && w.playing) togglePlay();
}

function togglePlay() {
  const w = state.walk;
  w.playing = !w.playing;
  clearTimeout(w.timer);
  if (w.playing) {
    if (w.i >= w.stages.length - 1 || w.i < 0) goStage(0);
    const tick = () => {
      if (!w.playing) return;
      goStage(w.i + 1);
      if (w.playing) w.timer = window.setTimeout(tick, w.speed);
    };
    w.timer = window.setTimeout(tick, w.speed);
  }
  renderWalk();
}

function syncWalkToLayer() {
  const w = state.walk;
  const idx = w.stages.findIndex((s) => s.layer === state.layer);
  if (w.i >= 0 && w.stages[w.i]?.layer !== state.layer) w.i = idx;
  renderWalk();
  renderTokens();
}

function highlightWalk(scroll = false) {
  const st: Stage | undefined = state.walk.stages[state.walk.i];
  document.querySelectorAll(".card.active").forEach((e) => e.classList.remove("active"));
  document.querySelectorAll(".ov .walk").forEach((e) => e.classList.remove("walk"));
  if (!st || st.layer !== state.layer) return;
  const card = document.querySelector<HTMLElement>(`.card[data-card="${st.card}"]`);
  if (card) {
    card.classList.add("active");
    if (scroll) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  const lname = document.querySelector<HTMLElement>(`.ov .lname[data-layer="${st.layer}"]`);
  if (lname) {
    lname.classList.add("walk");
    if (scroll) lname.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ---------------------------------------------------------------- controls
// The text box always holds exactly what gets tokenized; the checkbox wraps/unwraps it in place.
const DEFAULT_PROMPT =
  "<|im_start|>user\nWhat is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nThe capital of Germany is";

function promptText(): string {
  return $<HTMLTextAreaElement>("#prompt").value;
}

function initPrompt() {
  const box = $<HTMLTextAreaElement>("#prompt");
  const chk = $<HTMLInputElement>("#chat");
  box.value = DEFAULT_PROMPT;
  // Remember the assistant part when unwrapping so re-wrapping restores it.
  let savedSuffix = Tok.CHAT_SUFFIX;
  chk.addEventListener("change", () => {
    const parts = Tok.splitChat(box.value);
    if (!chk.checked && parts) {
      savedSuffix = parts.suffix;
      box.value = parts.user;
    } else if (chk.checked && !parts) {
      box.value = Tok.CHAT_PREFIX + box.value + savedSuffix;
    }
  });
  box.addEventListener("input", () => (chk.checked = Tok.splitChat(box.value) !== null));
}

async function runPrompt() {
  const ids = state.tok.encode(promptText());
  if (!ids.length) return;
  if (ids.length > CFG.maxContext) {
    status(`prompt is ${ids.length} tokens; the limit here is ${CFG.maxContext}`);
    return;
  }
  state.tokens = ids;
  state.promptLen = ids.length;
  await refresh();
  status(`${state.tokens.length} tokens · forward pass ${state.cap!.ms.toFixed(0)} ms · ${state.eng.adapterInfo}`);
}

async function stepToken(): Promise<boolean> {
  if (!state.cap) await runPrompt();
  const cap = state.cap!;
  if (state.tokens.length >= CFG.maxContext) return false;
  // Predict from the last position even if the user focused an earlier token.
  const logits = cap.focus === cap.T - 1 ? cap.logits : (await run(state.tokens)).logits;
  const temp = Number($<HTMLInputElement>("#temp").value) || 0;
  const next = sample(logits, temp);
  state.tokens = [...state.tokens, next];
  await refresh();
  status(`${state.tokens.length} tokens · forward pass ${state.cap!.ms.toFixed(0)} ms`);
  return !(CFG.eosIds as readonly number[]).includes(next);
}

async function generate(n: number) {
  state.generating = true;
  state.stopReq = false;
  setBusy(true);
  try {
    for (let i = 0; i < n && !state.stopReq; i++) if (!(await stepToken())) break;
  } finally {
    state.generating = false;
    setBusy(false);
  }
}

function setBusy(b: boolean) {
  for (const id of ["#run", "#step", "#gen"]) $<HTMLButtonElement>(id).disabled = b;
  $<HTMLButtonElement>("#stop").disabled = !b;
}

function wire() {
  initPrompt();
  $("#run").addEventListener("click", () => void runPrompt().catch(fail));
  $("#step").addEventListener("click", async () => {
    setBusy(true);
    await stepToken().catch(fail);
    setBusy(false);
  });
  $("#gen").addEventListener("click", () => void generate(16).catch(fail));
  $("#stop").addEventListener("click", () => (state.stopReq = true));
  $("#prompt").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) void runPrompt();
  });
  addEventListener("keydown", (ev) => {
    const tag = (ev.target as HTMLElement).tagName;
    if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT" || !$("#modal").hidden) return;
    if (ev.key === "ArrowRight") goStage(state.walk.i + 1);
    else if (ev.key === "ArrowLeft") goStage(state.walk.i - 1);
    else if (ev.key === "Home") goStage(0);
    else if (ev.key === " ") togglePlay();
    else if (ev.key === "ArrowDown" || ev.key === "j") ctx().selectLayer(Math.min(FINAL, state.layer + 1));
    else if (ev.key === "ArrowUp" || ev.key === "k") ctx().selectLayer(Math.max(EMBED, state.layer - 1));
    else return;
    ev.preventDefault();
  });
}

function fail(e: unknown) {
  console.error(e);
  status("Error: " + (e instanceof Error ? e.message : String(e)));
}

// ---------------------------------------------------------------- boot
async function boot() {
  const note = $("#load-note");
  if (!navigator.gpu) {
    note.innerHTML = "This page needs <b>WebGPU</b>. Use a recent Chrome, Edge, or Safari 26+.";
    return;
  }
  const base = await resolveBase();
  const cached = "caches" in self && (await (await caches.open("local-lm-viz-weights")).keys()).length > 0;
  const go = async () => {
    $("#load-btn").hidden = true;
    $("#progress").hidden = false;
    try {
      state.tok = await Tok.load(base.file);
      const raw = await loadRawWeights(
        base.weights,
        (phase, done, total) => {
          $("#bar").style.width = `${(100 * done) / Math.max(1, total)}%`;
          $("#load-msg").textContent = total > 1 ? `${phase}… ${(done / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB` : `${phase}…`;
        },
        !base.local,
      );
      state.eng = await Engine.create(raw, (phase, done, total) => {
        $("#bar").style.width = `${(100 * done) / total}%`;
        $("#load-msg").textContent = `${phase} and building weight maps… ${Math.round((100 * done) / total)}%`;
      });
      $("#loader").hidden = true;
      for (const id of ["#controls", "#tokbar", "#main", "#walk"]) $(id).hidden = false;
      wire();
      await runPrompt();
      (window as unknown as { __ready: boolean }).__ready = true;
    } catch (e) {
      console.error(e);
      $("#load-msg").textContent = "Failed: " + (e instanceof Error ? e.message : String(e));
    }
  };
  if (base.local || cached) {
    note.textContent = base.local ? "Using the local copy in public/models." : "Loading from the browser cache.";
    void go();
  } else {
    note.innerHTML = `Weights are downloaded straight from Hugging Face (<span class="mono">${CFG.repo}</span>, bf16, about 1.19 GB)
      and cached in your browser for next time.`;
    const btn = $<HTMLButtonElement>("#load-btn");
    btn.hidden = false;
    btn.onclick = () => void go();
    if (new URLSearchParams(location.search).has("autoload")) void go();
  }
}

void boot();
