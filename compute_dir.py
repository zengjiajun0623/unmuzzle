# -*- coding: utf-8 -*-
"""Compute the 'refusal direction' at every decoder layer.
direction[L] = normalize( mean_L(harmful last-token resid) - mean_L(harmless last-token resid) )
Saves a dict {layer: tensor[hidden]} to refusal_dirs.pt
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEV)
model.eval()

def last_token_hiddens(prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    # hidden_states: tuple len (n_layers+1); [0]=embeddings, [i]=output of layer i
    hs = out.hidden_states  # each [1, seq, hidden]
    return [h[0, -1, :].float().cpu() for h in hs]  # per-layer last-token vector

d = json.load(open("direction_prompts.json", encoding="utf-8"))
n_layers = model.config.num_hidden_layers
print("layers:", n_layers, "hidden:", model.config.hidden_size)

def mean_over(prompts):
    acc = None
    for i, p in enumerate(prompts):
        hs = last_token_hiddens(p)
        hs = torch.stack(hs)  # [n_layers+1, hidden]
        acc = hs if acc is None else acc + hs
        print(f"  {i+1}/{len(prompts)}", end="\r")
    print()
    return acc / len(prompts)

print("harmful..."); mh = mean_over(d["harmful"])
print("harmless..."); mb = mean_over(d["harmless"])

dirs = {}
for L in range(1, n_layers + 1):          # skip embedding layer 0
    v = mh[L] - mb[L]
    v = v / v.norm()
    dirs[L - 1] = v                        # store 0-indexed by decoder layer
torch.save(dirs, "refusal_dirs.pt")
print("saved refusal_dirs.pt with layers", list(dirs.keys())[0], "..", list(dirs.keys())[-1])
