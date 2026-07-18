# -*- coding: utf-8 -*-
"""Eval DeepSeek-R1-Distill-Qwen-32B (4-bit) on the 265-item benchmark. It's a
reasoning model, so it emits <think>...reasoning...</think>{answer}: we give it
room to think (large max_new), then STRIP everything up to and including </think>
so the judge grades the final answer, not the reasoning."""
import json, argparse, torch, sys, re
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
set_seed(42)   # reproducibility under sampling (R1 must NOT be decoded greedily)

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--bench", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--maxnew", type=int, default=2048)   # reasoning + answer needs room
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=qc)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

def final_answer(text):
    # strip the reasoning: everything up to and including the last </think>
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()

bench = json.load(open(args.bench, encoding="utf-8"))
res = []
for i, it in enumerate(bench["items"]):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": it["q"]}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=args.maxnew, do_sample=True,
                           temperature=0.6, top_p=0.95, pad_token_id=tok.eos_token_id)
    raw = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
    closed = "</think>" in raw          # did the model finish reasoning + reach an answer?
    ans = final_answer(raw)
    res.append({"id": it["id"], "category": it["category"], "topic": it["topic"], "q": it["q"],
                "gold": it["gold"], "truth": it["truth"], "answer": ans, "closed": closed,
                "full": raw[:2000]})
    print(f"[{i+1}/{len(bench['items'])}] {it['id']}", flush=True)

json.dump({"model": args.model, "adapter": args.adapter, "results": res},
          open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {args.out}")
