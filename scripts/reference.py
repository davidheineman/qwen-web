"""Dump reference activations from HF transformers for Qwen3-0.6B.

Usage: uv run --with torch --with transformers python -I scripts/reference.py [out.json] [prompt]
The browser test page compares its WebGPU forward pass against this file.
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
out_path = sys.argv[1] if len(sys.argv) > 1 else "reference/reference.json"
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is Paris. The capital of Germany is"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()

ids = tok(prompt, return_tensors="pt").input_ids
captured = {}


def hook(name):
    def fn(_mod, _inp, out):
        captured[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return fn


layer0 = model.model.layers[0]
layer0.self_attn.q_norm.register_forward_hook(hook("l0_qnorm"))
layer0.mlp.act_fn.register_forward_hook(hook("l0_gate_act"))

with torch.no_grad():
    out = model(ids, output_hidden_states=True, output_attentions=True)

logits = out.logits[0, -1]
top = torch.topk(logits, 10)
hs = out.hidden_states  # 29 entries; [0]=embeddings, [i]=after layer i-1, [28] has final norm applied
last = ids.shape[1] - 1

ref = {
    "prompt": prompt,
    "ids": ids[0].tolist(),
    "top_ids": top.indices.tolist(),
    "top_logits": top.values.tolist(),
    "top_tokens": [tok.decode([i]) for i in top.indices.tolist()],
    "logits_head": logits[:32].tolist(),
    # residual stream at last position, entries 0..27 are pre-final-norm
    "residual_last": [h[0, last].tolist() for h in hs[:28]],
    "residual_norms_all": [[float(h[0, t].norm()) for t in range(ids.shape[1])] for h in hs[:28]],
    "final_normed_last": hs[28][0, last].tolist(),
    "attn_l0": out.attentions[0][0].tolist(),  # [heads, T, T]
    "attn_l27_h0": out.attentions[27][0, 0].tolist(),
    "l0_qnorm_last": captured["l0_qnorm"][0, last].flatten().tolist(),
}

# Sanity: tied embeddings really are identical
emb = model.model.embed_tokens.weight
lm = model.lm_head.weight
ref["lm_head_tied"] = bool(torch.equal(emb, lm))

# Greedy continuation for end-to-end comparison
with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=8, do_sample=False)
ref["greedy_ids"] = gen[0, ids.shape[1]:].tolist()
ref["greedy_text"] = tok.decode(gen[0, ids.shape[1]:])

with open(out_path, "w") as f:
    json.dump(ref, f)
print("ids", ref["ids"])
print("top", list(zip(ref["top_tokens"], [round(x, 3) for x in ref["top_logits"]])))
print("tied", ref["lm_head_tied"], "greedy:", repr(ref["greedy_text"]))
