// Small canvas helpers: heatmap grids, strips, matrix thumbnails, attention maps — all with hover tooltips.

import { DIVERGING, SEQ_BLUE, SEQ_ORANGE, absQuantile, fmt } from "./color";
import type { WeightThumb } from "../model/loader";

const tip = (() => {
  const el = document.createElement("div");
  el.id = "tip";
  document.body.appendChild(el);
  return el;
})();

export function showTip(html: string | null, ev?: MouseEvent) {
  if (!html || !ev) {
    tip.style.display = "none";
    return;
  }
  tip.innerHTML = html;
  tip.style.display = "block";
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + r.width > innerWidth - 4) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 4) y = ev.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

export function hover(
  el: HTMLElement,
  fn: (x: number, y: number) => string | null,
  onClick?: (x: number, y: number) => void,
) {
  el.addEventListener("mousemove", (ev) => {
    const r = el.getBoundingClientRect();
    showTip(fn(ev.clientX - r.left, ev.clientY - r.top), ev);
  });
  el.addEventListener("mouseleave", () => showTip(null));
  if (onClick)
    el.addEventListener("click", (ev) => {
      const r = el.getBoundingClientRect();
      onClick(ev.clientX - r.left, ev.clientY - r.top);
    });
}

/** Canvas at native data resolution, scaled up with crisp pixels by CSS. */
function pixelCanvas(w: number, h: number, cssW: number, cssH: number): [HTMLCanvasElement, ImageData] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.style.width = `${cssW}px`;
  c.style.height = `${cssH}px`;
  c.className = "px";
  return [c, c.getContext("2d")!.createImageData(w, h)];
}

function put(img: ImageData, i: number, lut: Uint8ClampedArray, t: number) {
  const k = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
  img.data[i * 4] = lut[k];
  img.data[i * 4 + 1] = lut[k + 1];
  img.data[i * 4 + 2] = lut[k + 2];
  img.data[i * 4 + 3] = 255;
}

const div = (v: number, s: number) => 0.5 + 0.5 * Math.max(-1, Math.min(1, v / s));

export interface GridOpts {
  cols: number;
  cell?: [number, number];
  scale?: number;
  label?: (i: number) => string;
}

/** Vector laid out row-major as a grid of cells, diverging colors. */
export function vecGrid(data: Float32Array, o: GridOpts): HTMLCanvasElement {
  const cols = o.cols;
  const rows = Math.ceil(data.length / cols);
  const [cw, ch] = o.cell ?? [4, 4];
  const scale = o.scale ?? absQuantile(data);
  const [c, img] = pixelCanvas(cols, rows, cols * cw, rows * ch);
  for (let i = 0; i < data.length; i++) put(img, i, DIVERGING, div(data[i], scale));
  c.getContext("2d")!.putImageData(img, 0, 0);
  hover(c, (x, y) => {
    const i = Math.floor(y / ch) * cols + Math.floor(x / cw);
    if (i < 0 || i >= data.length) return null;
    return `<b>${o.label ? o.label(i) : `dim ${i}`}</b><br>${fmt(data[i])}`;
  });
  return c;
}

/** 1-D vector squeezed into a strip; each pixel shows the largest-magnitude value it covers. */
export function strip(data: Float32Array, cssW: number, cssH: number, scale: number, label?: (lo: number, hi: number, v: number) => string) {
  const dpr = Math.min(2, devicePixelRatio || 1);
  const w = Math.min(data.length, Math.round(cssW * dpr));
  const [c, img] = pixelCanvas(w, 1, cssW, cssH);
  c.className = "px strip";
  const vals = new Float32Array(w);
  for (let p = 0; p < w; p++) {
    const lo = Math.floor((p * data.length) / w);
    const hi = Math.max(lo + 1, Math.floor(((p + 1) * data.length) / w));
    let best = 0;
    for (let i = lo; i < hi; i++) if (Math.abs(data[i]) > Math.abs(best)) best = data[i];
    vals[p] = best;
    put(img, p, DIVERGING, div(best, scale));
  }
  c.getContext("2d")!.putImageData(img, 0, 0);
  hover(c, (x) => {
    const p = Math.max(0, Math.min(w - 1, Math.floor((x / cssW) * w)));
    const lo = Math.floor((p * data.length) / w);
    const hi = Math.max(lo + 1, Math.floor(((p + 1) * data.length) / w));
    return label ? label(lo, hi - 1, vals[p]) : `dims ${lo}–${hi - 1}<br>max |x|: ${fmt(vals[p])}`;
  });
  return c;
}

/** Weight-matrix thumbnail (block RMS), sequential blue. */
export function matrixThumb(t: WeightThumb, cssW: number, cssH: number, highlightRow?: number): HTMLCanvasElement {
  const [c, img] = pixelCanvas(t.tw, t.th, cssW, cssH);
  const s = t.maxRms;
  for (let i = 0; i < t.rms.length; i++) put(img, i, SEQ_BLUE, Math.sqrt(t.rms[i] / s));
  const ctx = c.getContext("2d")!;
  ctx.putImageData(img, 0, 0);
  if (highlightRow !== undefined) {
    const r = Math.floor((highlightRow * t.th) / t.rows);
    ctx.fillStyle = "rgba(255, 216, 77, 0.95)";
    ctx.fillRect(0, r, t.tw, Math.max(1, Math.ceil(t.th / 80)));
  }
  return c;
}

/** Display size for a [rows, cols] matrix inside a box, keeping aspect up to a limit. */
export function thumbSize(rows: number, cols: number, box: number, maxAspect = 4): [number, number] {
  let ar = rows / cols;
  ar = Math.max(1 / maxAspect, Math.min(maxAspect, ar));
  return ar >= 1 ? [Math.round(box / ar), box] : [box, Math.round(box * ar)];
}

/** T×T causal attention map; color ∝ √p so weaker links stay visible. */
export function attnMap(
  p: Float32Array,
  T: number,
  css: number,
  focus: number,
  tokenLabel: (i: number) => string,
  onClick?: () => void,
): HTMLCanvasElement {
  const [c, img] = pixelCanvas(T, T, css, css);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < T; j++) {
      const k = i * T + j;
      if (j > i) {
        img.data.set([26, 26, 25, 255], k * 4);
      } else put(img, k, SEQ_ORANGE, Math.sqrt(p[k]));
    }
  const ctx = c.getContext("2d")!;
  ctx.putImageData(img, 0, 0);
  c.dataset.focus = String(focus);
  hover(
    c,
    (x, y) => {
      const i = Math.floor((y / css) * T);
      const j = Math.floor((x / css) * T);
      if (i < 0 || j < 0 || i >= T || j >= T || j > i) return null;
      return `query <b>${tokenLabel(i)}</b> [${i}]<br>→ key <b>${tokenLabel(j)}</b> [${j}]<br>p = ${fmt(p[i * T + j])}`;
    },
    onClick ? () => onClick() : undefined,
  );
  return c;
}

/** Horizontal legend bar for a LUT. */
export function legend(kind: "div" | "blue" | "orange", left: string, mid: string, right: string): HTMLElement {
  const lut = kind === "div" ? DIVERGING : kind === "blue" ? SEQ_BLUE : SEQ_ORANGE;
  const [c, img] = pixelCanvas(128, 1, 96, 8);
  for (let i = 0; i < 128; i++) put(img, i, lut, i / 127);
  c.getContext("2d")!.putImageData(img, 0, 0);
  const el = document.createElement("span");
  el.className = "legend";
  el.innerHTML = `<span>${left}</span>`;
  el.appendChild(c);
  el.insertAdjacentHTML("beforeend", `<span>${right}</span>`);
  if (mid) el.title = mid;
  return el;
}
