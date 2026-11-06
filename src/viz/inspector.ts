// Modal weight inspector: the whole matrix as a block-RMS map; click anywhere to read the actual
// bf16 values of a 64×64 window back from the GPU.

import { absQuantile, fmt } from "./color";
import { h, $ } from "./dom";
import { matrixThumb, thumbSize, vecGrid, hover, legend } from "./draw";
import type { Ctx } from "./ctx";
import { visible } from "../model/tokenizer";

const WIN = 64;

export function openInspector(ctx: Ctx, name: string, highlightRow?: number) {
  const e = ctx.eng.weights.get(name);
  if (!e || !e.thumb) return;
  const t = e.thumb;
  const [rows, cols] = e.shape;
  const isEmbed = name === "model.embed_tokens.weight";
  const modal = $("#modal");
  const close = () => {
    modal.hidden = true;
    modal.innerHTML = "";
    removeEventListener("keydown", onKey);
  };
  const onKey = (ev: KeyboardEvent) => ev.key === "Escape" && close();
  addEventListener("keydown", onKey);

  const [w, hgt] = thumbSize(rows, cols, 460, 6);
  const thumb = matrixThumb(t, w, hgt, highlightRow);
  thumb.style.cursor = "crosshair";
  const box = h("div", { class: "zoombox" });
  const wrap = h("div", { class: "zoomwrap", style: { width: `${w}px`, height: `${hgt}px` } }, thumb, box);
  const zoom = h("div", { class: "fig" }, h("div", { class: "dim" }, "Click the map to load the actual weight values of a 64×64 window."));

  const blockOf = (x: number, y: number) => {
    const r = Math.floor((y / hgt) * rows);
    const c = Math.floor((x / w) * cols);
    return [Math.max(0, Math.min(rows - 1, r)), Math.max(0, Math.min(cols - 1, c))];
  };
  const rowsLabel = isEmbed ? "token id" : "output feature";
  hover(
    thumb,
    (x, y) => {
      const bi = Math.min(t.th - 1, Math.floor((y / hgt) * t.th));
      const bj = Math.min(t.tw - 1, Math.floor((x / w) * t.tw));
      const r0 = Math.floor((bi * rows) / t.th);
      const r1 = Math.floor(((bi + 1) * rows) / t.th) - 1;
      const c0 = Math.floor((bj * cols) / t.tw);
      const c1 = Math.floor(((bj + 1) * cols) / t.tw) - 1;
      return `${rowsLabel} ${r0}${r1 > r0 ? "–" + r1 : ""}\ninput dim ${c0}${c1 > c0 ? "–" + c1 : ""}\nblock RMS ${fmt(t.rms[bi * t.tw + bj])}`;
    },
    (x, y) => {
      const [r, c] = blockOf(x, y);
      void loadWindow(r, c);
    },
  );

  async function loadWindow(r: number, c: number) {
    const r0 = Math.max(0, Math.min(rows - WIN, r - WIN / 2));
    const c0 = Math.max(0, Math.min(cols - WIN, c - WIN / 2));
    box.style.left = `${(c0 / cols) * w}px`;
    box.style.top = `${(r0 / rows) * hgt}px`;
    box.style.width = `${Math.max(3, (WIN / cols) * w)}px`;
    box.style.height = `${Math.max(3, (WIN / rows) * hgt)}px`;
    zoom.replaceChildren(h("div", { class: "dim" }, "reading from GPU…"));
    const data = await ctx.eng.readWeightRows(name, r0, r0 + WIN);
    const block = new Float32Array(WIN * WIN);
    for (let i = 0; i < WIN; i++) for (let j = 0; j < WIN; j++) block[i * WIN + j] = data[i * cols + c0 + j];
    const scale = absQuantile(block, 0.995);
    const grid = vecGrid(block, {
      cols: WIN,
      cell: [7, 7],
      scale,
      label: (k) => {
        const rr = r0 + Math.floor(k / WIN);
        const cc = c0 + (k % WIN);
        const tok = isEmbed ? ` (“${visible(ctx.piece(rr))}”)` : "";
        return `W[${rr}, ${cc}]${tok}`;
      },
    });
    zoom.replaceChildren(
      h("div", { class: "lbl mono" }, `rows ${r0}–${r0 + WIN - 1} × cols ${c0}–${c0 + WIN - 1}  (actual bf16 values)`),
      grid,
      h("div", { class: "ov-legend" }, legend("div", fmt(-scale), "0", fmt(scale))),
    );
  }

  const shape = isEmbed
    ? `[${rows} tokens × ${cols} dims]: each row is one token's embedding`
    : `[${rows} out × ${cols} in]: y = W·x, so row i produces output dim i`;
  const card = h(
    "div",
    { class: "modal-card" },
    h("button", { class: "x small", onclick: close }, "✕"),
    h("h2", null, name),
    h("div", { class: "dim" }, shape),
    h(
      "div",
      { class: "stats" },
      h("span", null, `params ${(rows * cols).toLocaleString()}`),
      h("span", null, `min ${fmt(t.min)}`),
      h("span", null, `max ${fmt(t.max)}`),
      h("span", null, `mean ${fmt(t.mean)}`),
      h("span", null, `std ${fmt(t.std)}`),
      h("span", null, `map: ${t.th}×${t.tw} blocks`),
      h("span", null, `of ${Math.ceil(rows / t.th)}×${Math.ceil(cols / t.tw)}`),
      h("span", null, "color = block RMS"),
    ),
    h("div", { class: "flow" }, wrap, zoom),
  );
  modal.replaceChildren(card);
  modal.hidden = false;
  modal.onclick = (ev) => ev.target === modal && close();
  if (highlightRow !== undefined) void loadWindow(highlightRow, 0);
}
