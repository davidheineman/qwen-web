// Left column: one row per layer. Three strips per row show, at the focus token, what attention
// wrote into the residual stream, what the MLP wrote, and the resulting residual stream.
// The right-most column is the logit lens: the token the model would predict if it stopped there.

import { CFG } from "../model/config";
import { absQuantile, fmt, l2 } from "./color";
import { h, esc } from "./dom";
import { strip, legend } from "./draw";
import { type Ctx, EMBED, FINAL, resid, finalNormed } from "./ctx";
import { visible } from "../model/tokenizer";

export function renderOverview(ctx: Ctx): HTMLElement {
  const { cap } = ctx;
  const L = CFG.layers;
  const rows: { l: number; name: string; vecs: (Float32Array | null)[]; lens: number }[] = [];
  rows.push({ l: EMBED, name: "embed", vecs: [null, null, resid(cap, 0)], lens: 0 });
  for (let l = 0; l < L; l++)
    rows.push({ l, name: `L${l}`, vecs: [cap.layers[l].o, cap.layers[l].down, resid(cap, l + 1)], lens: l + 1 });
  rows.push({ l: FINAL, name: "final", vecs: [null, null, finalNormed(ctx)], lens: L });

  // Per-row scale (each strip normalized by its own 99th-percentile |x|) or one shared scale per column.
  const shared = [0, 1, 2].map((c) => Math.max(...rows.map((r) => (r.vecs[c] ? absQuantile(r.vecs[c]!) : 0))));

  const grid = h("div", { class: "ov" });
  for (const t of ["", "attention → residual", "MLP → residual", "residual stream", "logit lens"])
    grid.append(h("div", { class: "hd" }, t));

  const names = ["attention output (o_proj)", "MLP output (down_proj)", "residual stream"];
  for (const r of rows) {
    const sel = r.l === ctx.layer;
    const pick = () => ctx.selectLayer(r.l);
    grid.append(h("div", { class: `lname${sel ? " sel" : ""}`, "data-layer": r.l, onclick: pick }, r.name));
    r.vecs.forEach((v, c) => {
      const cell = h("div", { class: `cell${sel ? " sel" : ""}`, "data-layer": r.l, onclick: pick });
      if (v) {
        const scale = ctx.scaleMode === "shared" ? shared[c] : absQuantile(v);
        const name = r.l === FINAL && c === 2 ? "final RMSNorm output" : names[c];
        cell.append(
          strip(v, 240, 13, scale, (lo, hi, val) =>
            `<b>${r.name}</b> · ${name}<br>dims ${lo}${hi > lo ? "–" + hi : ""}: ${fmt(val)}<br>‖x‖ = ${fmt(l2(v))}`),
        );
      }
      grid.append(cell);
    });
    const lens = cap.lens[r.lens];
    const top = lens[0];
    const lensEl = h(
      "div",
      { class: `lens${sel ? " sel" : ""}`, "data-layer": r.l, onclick: pick },
      h("span", null, visible(ctx.piece(top.id))),
      h("span", { class: "p" }, `${(top.prob * 100).toFixed(top.prob < 0.1 ? 1 : 0)}%`),
    );
    lensEl.title = lens
      .slice(0, 5)
      .map((t) => `${JSON.stringify(ctx.piece(t.id))}  ${(t.prob * 100).toFixed(1)}%`)
      .join("\n");
    grid.append(lensEl);
  }

  const seg = h("span", { class: "seg" });
  for (const [m, label] of [["row", "per-row scale"], ["shared", "shared scale"]] as const)
    seg.append(h("button", { class: ctx.scaleMode === m ? "on" : "", onclick: () => ctx.setScaleMode(m) }, label));

  const focusTok = esc(visible(ctx.label(cap.focus)));
  return h(
    "div",
    null,
    h("h2", { html: `Residual stream through all 28 layers <span class="dim">at token “${focusTok}” (position ${cap.focus})</span>` }),
    h(
      "div",
      { class: "ov-legend" },
      legend("div", "−", "0", "+"),
      h("span", null, "each strip = 1024 dims (each pixel shows its largest-magnitude dim)"),
      seg,
    ),
    grid,
  );
}
