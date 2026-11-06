// Compares the WebGPU forward pass against reference activations dumped by scripts/reference.py.
import { resolveBase, loadRawWeights } from "./model/loader";
import { Engine, sample } from "./model/engine";
import { Tok } from "./model/tokenizer";
import { CFG } from "./model/config";

const logEl = document.getElementById("log")!;
const lines: string[] = [];
const log = (s: string) => {
  lines.push(s);
  logEl.textContent = lines.join("\n");
  console.log(s);
};

function maxRelErr(a: ArrayLike<number>, b: ArrayLike<number>) {
  let num = 0;
  let den = 0;
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    maxAbs = Math.max(maxAbs, d);
    num += d * d;
    den += b[i] * b[i];
  }
  return { rel: Math.sqrt(num / den), maxAbs };
}

async function main() {
  const ref = await fetch("/reference/reference.json").then((r) => r.json());
  const base = await resolveBase();
  log(`weights: ${base.weights}`);
  const tok = await Tok.load(base.file);
  const ids = tok.encode(ref.prompt);
  const idsOk = JSON.stringify(ids) === JSON.stringify(ref.ids);
  log(`tokenizer ids match: ${idsOk} ${JSON.stringify(ids)}`);

  let t = performance.now();
  const raw = await loadRawWeights(base.weights, () => {}, false);
  log(`loaded ${raw.tensors.size} tensors (${(raw.bytes.length / 1e6).toFixed(0)} MB) in ${(performance.now() - t).toFixed(0)} ms`);
  t = performance.now();
  const eng = await Engine.create(raw, () => {});
  log(`GPU: ${eng.adapterInfo}; upload+thumbs ${(performance.now() - t).toFixed(0)} ms`);

  const cap = await eng.forward(ref.ids);
  log(`forward T=${cap.T}: ${cap.ms.toFixed(1)} ms`);
  const cap2 = await eng.forward(ref.ids);
  log(`forward (warm): ${cap2.ms.toFixed(1)} ms`);

  const results: Record<string, unknown> = { idsOk };
  const H = CFG.hidden;
  const T = cap.T;
  const worst = { rel: 0 };
  for (const l of [0, 1, 2, 14, 27]) {
    const mine = cap.resid.subarray((l * T + T - 1) * H, (l * T + T) * H);
    const e = maxRelErr(mine, ref.residual_last[l]);
    worst.rel = Math.max(worst.rel, e.rel);
    log(`resid[${l}] rel err ${e.rel.toExponential(2)} maxAbs ${e.maxAbs.toExponential(2)}`);
  }
  const qn = maxRelErr(cap.layers[0].qr, ref.l0_qnorm_last);
  log(`(info) layer0 q after norm+rope vs HF q_norm output (pre-rope): rel ${qn.rel.toExponential(2)}`);
  // attention layer 0, all heads
  const a0 = cap.probs.subarray(0, CFG.heads * T * T);
  const ea = maxRelErr(a0, (ref.attn_l0 as number[][][]).flat(2));
  log(`attn L0 rel err ${ea.rel.toExponential(2)} maxAbs ${ea.maxAbs.toExponential(2)}`);
  const a27 = cap.probs.subarray(27 * CFG.heads * T * T, 27 * CFG.heads * T * T + T * T);
  const ea27 = maxRelErr(a27, (ref.attn_l27_h0 as number[][]).flat());
  log(`attn L27 h0 rel err ${ea27.rel.toExponential(2)} maxAbs ${ea27.maxAbs.toExponential(2)}`);
  const el = maxRelErr(cap.logits.subarray(0, 32), ref.logits_head);
  log(`logits[0:32] rel err ${el.rel.toExponential(2)} maxAbs ${el.maxAbs.toExponential(2)}`);
  const top = cap.lens[CFG.layers];
  log(`top-5 mine: ${top.slice(0, 5).map((x) => `${JSON.stringify(tok.piece(x.id))}:${x.logit.toFixed(3)}`).join(" ")}`);
  log(`top-5 ref : ${ref.top_tokens.slice(0, 5).map((s: string, i: number) => `${JSON.stringify(s)}:${ref.top_logits[i].toFixed(3)}`).join(" ")}`);
  const topMatch = top.slice(0, 5).every((x, i) => x.id === ref.top_ids[i]);
  log(`top-5 ids match: ${topMatch}`);

  // greedy generation
  const seq = [...ref.ids];
  const gen: number[] = [];
  t = performance.now();
  for (let i = 0; i < ref.greedy_ids.length; i++) {
    const c = await eng.forward(seq);
    const nxt = sample(c.logits, 0);
    seq.push(nxt);
    gen.push(nxt);
  }
  const genMs = performance.now() - t;
  const genOk = JSON.stringify(gen) === JSON.stringify(ref.greedy_ids);
  log(`greedy: ${JSON.stringify(tok.decode(gen))} match=${genOk} (${(genMs / gen.length).toFixed(0)} ms/token)`);

  log(`logit lens L14 top: ${cap.lens[14].slice(0, 3).map((x) => JSON.stringify(tok.piece(x.id))).join(" ")}`);
  const pass = idsOk && topMatch && genOk && worst.rel < 1e-3 && ea.rel < 1e-3 && el.rel < 1e-3;
  Object.assign(results, { pass, topMatch, genOk, residRel: worst.rel, attnRel: ea.rel, logitRel: el.rel, ms: cap2.ms });
  log(pass ? "SELFTEST PASS" : "SELFTEST FAIL");
  (window as unknown as { __selftest: unknown }).__selftest = results;
}

main().catch((e) => {
  log("ERROR: " + (e?.stack ?? e));
  (window as unknown as { __selftest: unknown }).__selftest = { pass: false, error: String(e) };
});
