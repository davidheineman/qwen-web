// The walkthrough: an ordered list of computation steps, each tied to a card in the detail panel,
// with narration that quotes the actual numbers from the current forward pass.

import { CFG } from "../model/config";
import { esc } from "./dom";
import { type Ctx, EMBED, FINAL, resid, probs, rms, finalNormed } from "./ctx";
import { visible } from "../model/tokenizer";
import { fmt, l2 } from "./color";

export interface Stage {
  layer: number;
  card: string;
}

export const LAYER_CARDS = [
  "attnNorm", "qkv", "qknorm", "attn", "oproj", "resid1",
  "mlpNorm", "gateup", "swiglu", "down", "resid2",
] as const;

export function buildStages(perLayer: boolean): Stage[] {
  const s: Stage[] = [{ layer: EMBED, card: "embed" }];
  for (let l = 0; l < CFG.layers; l++) {
    if (perLayer) s.push({ layer: l, card: "resid2" });
    else for (const c of LAYER_CARDS) s.push({ layer: l, card: c });
  }
  s.push({ layer: FINAL, card: "finalNorm" }, { layer: FINAL, card: "unembed" });
  return s;
}

const q = (s: string) => `<b>“${esc(visible(s))}”</b>`;
const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

export function narrate(st: Stage, ctx: Ctx): string {
  const { cap } = ctx;
  const t = cap.focus;
  const tokId = cap.tokens[t];
  const tok = ctx.label(t);
  const l = st.layer;
  const d = l >= 0 && l < FINAL ? cap.layers[l] : null;
  switch (st.card) {
    case "embed":
      return `Token ${q(tok)} (id ${tokId}) is looked up in the 151,936 × 1024 embedding table. That row becomes the
        starting vector of the <b>residual stream</b> (‖x‖ = ${fmt(l2(resid(cap, 0)))}), which every layer reads from and adds to.`;
    case "attnNorm":
      return `<b>Layer ${l}.</b> RMSNorm divides the residual vector by its root-mean-square (${fmt(rms(resid(cap, l)))}) and multiplies
        each dimension by a learned gain γ, so attention always sees a well-scaled input.`;
    case "qkv":
      return `Three matrix-vector products: <b>q</b> = W<sub>q</sub>x (16 heads × 128), <b>k</b> = W<sub>k</sub>x and
        <b>v</b> = W<sub>v</sub>x (8 heads × 128 each). Qwen3 uses grouped-query attention: every 2 query heads share one key/value head.`;
    case "qknorm":
      return `Qwen3's <b>QK-norm</b> RMS-normalizes every 128-d head of q and k. Then <b>RoPE</b> rotates dimension pairs
        (i, i+64) by angle position·θ<sub>i</sub>, so the dot product q·k depends on how far apart two tokens are (here: position ${t}).`;
    case "attn": {
      const T = cap.T;
      let best = { h: 0, j: 0, p: 0 };
      let sink = 0;
      for (let hh = 0; hh < CFG.heads; hh++) {
        const P = probs(cap, l, hh);
        if (P[t * T] > 0.5) sink++;
        for (let j = 1; j <= t; j++) if (P[t * T + j] > best.p) best = { h: hh, j, p: P[t * T + j] };
      }
      const target = best.j === t ? "itself" : `${q(ctx.label(best.j))} (position ${best.j})`;
      const strong = best.p > 0 ? ` Strongest link to a token other than the first: head ${best.h} puts ${pct(best.p)} on ${target}.` : "";
      return `Each head compares the query for ${q(tok)} with the keys of every earlier token: softmax(q·k/√128). ${sink} of 16 heads put most of
        their weight on the very first token, a common “attention sink” that works as a no-op.${strong}`;
    }
    case "oproj":
      return `Each head returns a weighted average of the value vectors it attended to. The 16 head outputs are concatenated
        (2048) and <b>o_proj</b> maps them back into the 1024-d residual space (‖Δ‖ = ${fmt(l2(d!.o))}).`;
    case "resid1":
      return `The attention output is <b>added</b> to the residual stream: ‖x‖ goes ${fmt(l2(resid(cap, l)))} → ${fmt(l2(d!.mid))}.
        Layers never overwrite the stream, they only add to it.`;
    case "mlpNorm":
      return `A second RMSNorm (post_attention_layernorm) prepares the updated residual for the MLP.`;
    case "gateup":
      return `The MLP widens to 3072 dims with two parallel projections, <b>gate</b> and <b>up</b>.`;
    case "swiglu": {
      let bi = 0;
      let n = 0;
      const th = 0.1 * Math.max(...Array.from(d!.act, Math.abs));
      for (let i = 0; i < d!.act.length; i++) {
        if (Math.abs(d!.act[i]) > Math.abs(d!.act[bi])) bi = i;
        if (Math.abs(d!.act[i]) > th) n++;
      }
      return `<b>SwiGLU</b>: silu(gate) ⊙ up. silu is close to 0 for negative inputs, so the gate acts as a soft on/off switch per neuron.
        Only ${n} of 3072 neurons exceed 10% of the peak; the strongest is neuron ${bi} (${fmt(d!.act[bi])}).`;
    }
    case "down":
      return `<b>down_proj</b> maps the 3072 neuron activations back to 1024 dims (‖Δ‖ = ${fmt(l2(d!.down))}).`;
    case "resid2": {
      const lens = cap.lens[l + 1][0];
      return `<b>Layer ${l} done.</b> The MLP output is added: ‖x‖ = ${fmt(l2(resid(cap, l + 1)))}. If we decoded the residual stream now
        (“logit lens”), the top guess would be ${q(ctx.piece(lens.id))} (${pct(lens.prob)}).`;
    }
    case "finalNorm":
      return `After 28 layers, a final RMSNorm scales the residual vector (‖x‖ = ${fmt(l2(resid(cap, CFG.layers)))} →
        ${fmt(l2(finalNormed(ctx)))}).`;
    case "unembed": {
      const top = cap.lens[CFG.layers];
      return `The normalized vector is dot-multiplied with all 151,936 token embeddings (the same matrix as the input embedding:
        Qwen3-0.6B ties them) to give logits, and softmax turns those into probabilities.
        Prediction after ${q(tok)}: ${q(ctx.piece(top[0].id))} ${pct(top[0].prob)}, then ${q(ctx.piece(top[1].id))} ${pct(top[1].prob)}.`;
    }
  }
  return "";
}
