# -*- coding: utf-8 -*-
"""M1 rollout diagnosis: for each item, sample K=8 rollouts from R1-Distill-32B-SFT
(temp 0.6/0.95, never greedy). Saves rollouts for judging -> pass@8 tells us if a
failure is ELICITABLE (0<pass@8<8, RL can help) or knowledge-ABSENT (pass@8=0, route
to SFT/RAG). Strips reasoning; grades the answer after </think>."""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
set_seed(42)
SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter", required=True)
ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--maxnew", type=int, default=2048)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=qc)
from peft import PeftModel
model = PeftModel.from_pretrained(model, args.adapter); model.eval()

def final(t): return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()
items = json.load(open(args.items, encoding="utf-8"))
res = []
for i, it in enumerate(items):
    prompt = tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":it["q"]}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=args.maxnew, do_sample=True, temperature=0.6,
                           top_p=0.95, num_return_sequences=args.k, pad_token_id=tok.eos_token_id)
    rolls = [final(tok.decode(g[j, ids.input_ids.shape[1]:], skip_special_tokens=True)) for j in range(args.k)]
    res.append({"id":it["id"],"category":it["category"],"q":it["q"],"truth":it["truth"],"rollouts":rolls})
    print(f"[{i+1}/{len(items)}] {it['id']} ({len(rolls)} rollouts)", flush=True)
json.dump({"results":res}, open(args.out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {args.out}")
