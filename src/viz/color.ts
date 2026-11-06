// Color ramps for the dark surface (#1a1a19).
//  - diverging blue <-> red, neutral gray midpoint: signed activations / weight values
//  - sequential blue (dark -> light): weight RMS magnitude
//  - sequential orange (dark -> light): attention probabilities

type RGB = [number, number, number];
const hex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as RGB;

function buildLut(stops: [number, string][]): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  const s = stops.map(([t, h]) => [t, hex(h)] as [number, RGB]);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < s.length - 2 && t > s[k + 1][0]) k++;
    const [t0, c0] = s[k];
    const [t1, c1] = s[k + 1];
    const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return lut;
}

/** index 0 = most negative, 128 = zero, 255 = most positive */
export const DIVERGING = buildLut([
  [0, "#cde2fb"],
  [0.18, "#5598e7"],
  [0.38, "#1c5cab"],
  [0.5, "#383835"],
  [0.62, "#a83a3a"],
  [0.82, "#e66767"],
  [1, "#fbd3d0"],
]);

export const SEQ_BLUE = buildLut([
  [0, "#1d2330"],
  [0.25, "#104281"],
  [0.5, "#256abf"],
  [0.75, "#6da7ec"],
  [1, "#cde2fb"],
]);

export const SEQ_ORANGE = buildLut([
  [0, "#241d19"],
  [0.25, "#6b2f14"],
  [0.5, "#c24f1f"],
  [0.75, "#f0915f"],
  [1, "#fde3d3"],
]);

export function lutCss(lut: Uint8ClampedArray, t: number): string {
  const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
  return `rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]})`;
}

/** Robust symmetric scale: the q-quantile of |x| (so a few outlier dims don't wash everything out). */
export function absQuantile(x: ArrayLike<number>, q = 0.99): number {
  const n = x.length;
  if (!n) return 1;
  const step = Math.max(1, Math.floor(n / 4096));
  const a: number[] = [];
  for (let i = 0; i < n; i += step) a.push(Math.abs(x[i]));
  a.sort((p, r) => p - r);
  return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))] || 1e-6;
}

export function absMax(x: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  return m || 1e-6;
}

export function l2(x: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

export function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000 || a < 0.001) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(4);
}
