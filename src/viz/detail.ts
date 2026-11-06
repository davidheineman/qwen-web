// Right column: the full computation of one layer at the focus token, step by step.
// Every weight matrix is shown as a thumbnail (click to inspect real values),
// every intermediate vector as a heatmap grid (hover for exact values).

import { CFG } from "../model/config";
import type { TopToken } from "../model/engine";
import { absQuantile, fmt, l2 } from "./color";
import { h, esc } from "./dom";
import { vecGrid, matrixThumb, thumbSize, attnMap, legend } from "./draw";
import { type Ctx, EMBED, FINAL, resid, probs, finalNormed } from "./ctx";
import { visible } from "../model/tokenizer";

const HD = CFG.headDim;

function card(key: string, title: string, op: string, caption: string, ...body: (Node | null)[]) {
  return h(
    "div",
    { class: "card", "data-card": key },
    h("h3", null, title, h("span", { class: "op" }, op)),
    caption ? h("div", { class: "cap", html: caption }) : null,
    ...body,
  );
}

const flow = (...items: (Node | string | null)[]) =>
  h("div", { class: "flow" }, ...items.map((i) => (typeof i === "string" ? h("div", { class: "arrow" }, i) : i)));

function fig(label: string, el: Node) {
  return h("div", { class: "fig" }, h("div", { class: "lbl", html: label }), el);
}

/** A 1024-d (or any) vector as a 64-wide grid */
function vec(label: string, v: Float32Array, scale?: number, cols = 64, cell: [number, number] = [3, 3]) {
  return fig(`${label} <span>‖·‖=${fmt(l2(v))}</span>`, vecGrid(v, { cols, cell, scale }));
}

/** Per-head vectors: one row per head, 128 dims wide */
function heads(label: string, v: Float32Array, nh: number, scale?: number) {
  const f = fig(label, vecGrid(v, { cols: HD, cell: [2, 7], scale, label: (i) => `head ${Math.floor(i / HD)}, dim ${i % HD}` }));
  f.append(h("div", { class: "lbl" }, `${nh} heads × 128`));
  return f;
}

function weight(ctx: Ctx, name: string, box = 92, highlightRow?: number) {
  const e = ctx.eng.weights.get(name)!;
  const [rows, cols] = e.shape;
  const [w, hgt] = thumbSize(rows, cols, box);
  const c = matrixThumb(e.thumb!, w, hgt, highlightRow);
  c.classList.add("weight");
  c.title = `${name}\n[${rows} × ${cols}] — click to inspect actual weights`;
  c.addEventListener("click", () => ctx.inspect(name, highlightRow));
  const short = name.replace(/^model\.(layers\.\d+\.)?/, "").replace(/\.weight$/, "").replace(/^(self_attn|mlp)\./, "");
  return fig(`<b>${short}</b> ${rows}×${cols}`, c);
}

function gain(ctx: Ctx, name: string, cols = 64, cell: [number, number] = [3, 3]) {
  const g = ctx.eng.weights.get(name)!.values!;
  const short = name.replace(/^model\.(layers\.\d+\.)?/, "").replace(/\.weight$/, "").replace(/^(self_attn)\./, "");
  return fig(`<b>γ</b> ${short}`, vecGrid(g, { cols, cell }));
}

function bars(items: TopToken[], ctx: Ctx, actual?: number) {
  const max = items[0]?.prob || 1;
  const el = h("div", { class: "bars" });
  for (const t of items) {
    const cls = t.id === actual ? "row-actual" : "";
    el.append(
      h("div", { class: `tk ${cls}`, title: `id ${t.id}, logit ${fmt(t.logit)}` }, visible(ctx.piece(t.id))),
      h("div", { class: cls }, h("div", { class: "bar", style: { width: `${(t.prob / max) * 100}%` } })),
      h("div", { class: "pv" }, `${(t.prob * 100).toFixed(t.prob < 0.1 ? 2 : 1)}%`),
    );
  }
  return el;
}

export { bars as tokenBars };

export function renderDetail(ctx: Ctx): HTMLElement {
  const { cap, layer } = ctx;
  const t = cap.focus;
  const tokLabel = esc(visible(ctx.label(t)));
  const nav = h(
    "span",
    { class: "seg" },
    h("button", { onclick: () => ctx.selectLayer(Math.max(EMBED, layer - 1)), disabled: layer <= EMBED }, "◀ prev"),
    h("button", { onclick: () => ctx.selectLayer(Math.min(FINAL, layer + 1)), disabled: layer >= FINAL }, "next ▶"),
  );
  const title = layer === EMBED ? "Embedding" : layer === FINAL ? "Output head" : `Layer ${layer}`;
  const head = h(
    "div",
    { class: "dhead" },
    h("span", { class: "title" }, title),
    h("span", { class: "dim", html: `token “${tokLabel}” · position ${t} · ${cap.ms.toFixed(0)} ms forward pass` }),
    nav,
  );
  const legends = h(
    "div",
    { class: "ov-legend" },
    legend("div", "activation −", "0", "+"),
    legend("blue", "weight RMS low", "", "high"),
    legend("orange", "attention 0", "", "1 (√p)"),
  );
  const body =
    layer === EMBED ? renderEmbed(ctx) : layer === FINAL ? renderFinal(ctx) : renderLayer(ctx, layer);
  return h("div", null, head, legends, ...body);
}

function renderEmbed(ctx: Ctx): Node[] {
  const { cap } = ctx;
  const id = cap.tokens[cap.focus];
  const x = resid(cap, 0);
  return [
    card(
      "embed",
      "Token embedding",
      "x = E[token_id]",
      `The tokenizer turned the text into ids; ${esc(JSON.stringify(ctx.label(cap.focus)))} is id <b>${id}</b>.
       Its row in the embedding matrix E (highlighted) is copied out as the 1024-d starting vector.
       Rows of this matrix are also used at the very end to score every possible next token.`,
      flow(weight(ctx, "model.embed_tokens.weight", 200, id), "→", vec(`x = E[${id}]`, x)),
    ),
  ];
}

function renderFinal(ctx: Ctx): Node[] {
  const { cap } = ctx;
  const x = resid(cap, CFG.layers);
  const xn = finalNormed(ctx);
  const actual = cap.focus + 1 < cap.T ? cap.tokens[cap.focus + 1] : undefined;
  const top = cap.lens[CFG.layers];
  return [
    card(
      "finalNorm",
      "Final RMSNorm",
      "x̂ = x / rms(x) · γ",
      "The residual stream after the last layer is normalized one final time.",
      flow(vec("x (after layer 27)", x), "÷rms ×", gain(ctx, "model.norm.weight"), "→", vec("x̂", xn)),
    ),
    h("div", { class: "down-arrow" }, "↓"),
    card(
      "unembed",
      "Unembedding → next-token probabilities",
      "logits = E · x̂ ;  p = softmax(logits)",
      `Qwen3-0.6B ties input and output embeddings, so the same 151,936 × 1024 matrix scores every token: each logit is the dot product
       of x̂ with that token's embedding row.${actual !== undefined ? " The token that actually came next is highlighted." : ""}`,
      flow(vec("x̂", xn), "→", weight(ctx, "model.embed_tokens.weight", 200), "→", fig("top 20 of 151,936", bars(topList(cap.logits, 20, top), ctx, actual))),
    ),
  ];
}

/** Use the precomputed top-10 when enough, else recompute. */
function topList(logits: Float32Array, k: number, pre: TopToken[]): TopToken[] {
  if (k <= pre.length) return pre.slice(0, k);
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  let z = 0;
  for (const v of logits) z += Math.exp(v - max);
  const idx = Array.from(logits.keys())
    .sort((a, b) => logits[b] - logits[a])
    .slice(0, k);
  return idx.map((id) => ({ id, logit: logits[id], prob: Math.exp(logits[id] - max) / z }));
}

function renderLayer(ctx: Ctx, l: number): Node[] {
  const { cap } = ctx;
  const d = cap.layers[l];
  const p = `model.layers.${l}.`;
  const x = resid(cap, l);
  const xOut = resid(cap, l + 1);
  const T = cap.T;
  const t = cap.focus;
  const arrow = () => h("div", { class: "down-arrow" }, "↓");

  // attention heads grid
  const css = Math.max(44, Math.min(84, T * 6));
  const headGrid = h("div", { class: "heads" });
  for (let hh = 0; hh < CFG.heads; hh++) {
    const P = probs(cap, l, hh);
    const cnv = attnMap(P, T, css, t, (i) => esc(visible(ctx.label(i))), () => ctx.selectHead(hh));
    // marker for the focus query row
    const tick = h("div", { class: "qtick", style: { top: `${(t / T) * css}px`, height: `${Math.max(2, css / T)}px` } });
    headGrid.append(
      h(
        "div",
        { class: `h${hh === ctx.head ? " on" : ""}`, onclick: () => ctx.selectHead(hh) },
        h("div", { class: "hm" }, cnv, tick),
        h("span", null, `h${hh}·kv${hh >> 1}`),
      ),
    );
  }
  const P = probs(cap, l, ctx.head);
  const row = Array.from({ length: t + 1 }, (_, j) => ({ j, p: P[t * T + j] })).sort((a, b) => b.p - a.p).slice(0, 8);
  const headBars = h("div", { class: "bars" });
  for (const r of row)
    headBars.append(
      h("div", { class: "tk" }, `[${r.j}] ${visible(ctx.label(r.j))}`),
      h("div", null, h("div", { class: "bar", style: { width: `${r.p * 100}%`, background: "var(--orange)" } })),
      h("div", { class: "pv" }, `${(r.p * 100).toFixed(1)}%`),
    );

  // top neurons
  const order = Array.from(d.act.keys()).sort((a, b) => Math.abs(d.act[b]) - Math.abs(d.act[a])).slice(0, 12);
  const neurons = h("div", { class: "neurons" }, ...order.map((i) => h("span", null, `n${i} ${d.act[i] >= 0 ? "+" : ""}${fmt(d.act[i])}`)));

  const sMid = absQuantile(d.mid);
  const sOut = absQuantile(xOut);

  return [
    card(
      "attnNorm",
      "RMSNorm",
      "input_layernorm: x̂ = x / rms(x) · γ",
      "Normalize the incoming residual vector, then scale each dimension by a learned gain.",
      flow(vec("x (residual in)", x), "÷rms ×", gain(ctx, p + "input_layernorm.weight"), "→", vec("x̂", d.attnNorm)),
    ),
    arrow(),
    card(
      "qkv",
      "Query, key, value projections",
      "q = W_q x̂ ;  k = W_k x̂ ;  v = W_v x̂",
      "Three weight matrices turn x̂ into a query (what this token is looking for), a key (what it offers to later tokens) and a value (what it passes on).",
      flow(
        h("div", { class: "fig" }, weight(ctx, p + "self_attn.q_proj.weight", 80), weight(ctx, p + "self_attn.k_proj.weight", 56), weight(ctx, p + "self_attn.v_proj.weight", 56)),
        "→",
        heads("<b>q</b>", d.q, 16),
        heads("<b>k</b>", d.k, 8),
        heads("<b>v</b>", d.v, 8),
      ),
    ),
    arrow(),
    card(
      "qknorm",
      "QK-norm + rotary position embedding",
      "q,k ← RoPE(RMSNorm_head(q,k), pos)",
      `Each 128-d head is RMS-normalized with its own small gain vector, then dimension pairs (i, i+64) are rotated by an angle proportional to the position (${t}).`,
      flow(
        h("div", { class: "fig" }, gain(ctx, p + "self_attn.q_norm.weight", 128, [2, 6]), gain(ctx, p + "self_attn.k_norm.weight", 128, [2, 6])),
        "→",
        heads("<b>q</b> normed + rotated", d.qr, 16),
        heads("<b>k</b> normed + rotated", d.kr, 8),
      ),
    ),
    arrow(),
    card(
      "attn",
      "Attention",
      "A = softmax(q kᵀ / √128 + causal mask)",
      `Each square is one head: rows are queries, columns are keys, and every token can only see itself and earlier tokens. The highlighted row in the
       token bar shows where head <b>${ctx.head}</b> looks from “${esc(visible(ctx.label(t)))}”. Click a head to select it.`,
      flow(headGrid, h("div", { class: "fig", style: { minWidth: "260px" } }, h("div", { class: "lbl", html: `head <b>${ctx.head}</b>: query [${t}] attends to` }), headBars)),
    ),
    arrow(),
    card(
      "oproj",
      "Combine heads",
      "o = W_o · concat(A·v)",
      "Each head's output is a weighted average of value vectors. The 16 outputs are concatenated and projected back into the residual space.",
      flow(heads("A·v (per head)", d.attn, 16), "→", weight(ctx, p + "self_attn.o_proj.weight", 80), "→", vec("o", d.o)),
    ),
    arrow(),
    card(
      "resid1",
      "Residual add",
      "x ← x + o",
      "The attention output is added on top of the residual stream (all three use the same color scale).",
      flow(vec("x", x, sMid), "+", vec("o", d.o, sMid), "=", vec("x′", d.mid, sMid)),
    ),
    arrow(),
    card(
      "mlpNorm",
      "RMSNorm",
      "post_attention_layernorm",
      "",
      flow(vec("x′", d.mid), "÷rms ×", gain(ctx, p + "post_attention_layernorm.weight"), "→", vec("x̂′", d.mlpNorm)),
    ),
    arrow(),
    card(
      "gateup",
      "MLP up-projection",
      "g = W_gate x̂′ ;  u = W_up x̂′",
      "The MLP widens the vector to 3072 “neurons” with two separate projections.",
      flow(
        h("div", { class: "fig" }, weight(ctx, p + "mlp.gate_proj.weight", 96), weight(ctx, p + "mlp.up_proj.weight", 96)),
        "→",
        vec("g (gate)", d.gate, undefined, 64, [3, 3]),
        vec("u (up)", d.up, undefined, 64, [3, 3]),
      ),
    ),
    arrow(),
    card(
      "swiglu",
      "SwiGLU",
      "a = silu(g) ⊙ u",
      "silu(g) = g·σ(g) is close to 0 for negative g, so the gate decides which neurons pass their “up” value through.",
      flow(vec("a", d.act, undefined, 64, [3, 3]), h("div", { class: "fig" }, h("div", { class: "lbl" }, "strongest neurons"), neurons)),
    ),
    arrow(),
    card(
      "down",
      "MLP down-projection",
      "m = W_down a",
      "",
      flow(vec("a", d.act, undefined, 64, [3, 3]), "→", weight(ctx, p + "mlp.down_proj.weight", 96), "→", vec("m", d.down)),
    ),
    arrow(),
    card(
      "resid2",
      "Residual add → layer output",
      "x ← x′ + m",
      `Logit lens at this point: ${cap.lens[l + 1]
        .slice(0, 3)
        .map((z) => `“${esc(visible(ctx.piece(z.id)))}” ${(z.prob * 100).toFixed(1)}%`)
        .join(", ")}`,
      flow(vec("x′", d.mid, sOut), "+", vec("m", d.down, sOut), "=", vec(`x (to layer ${l + 1 < CFG.layers ? l + 1 : "final norm"})`, xOut, sOut)),
    ),
  ];
}
