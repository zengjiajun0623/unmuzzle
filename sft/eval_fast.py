# -*- coding: utf-8 -*-
"""Batched eval — same benchmark/system-prompt/decoding as eval_model.py, but
generates BATCH items at once with left-padding instead of one-at-a-time. On a
loafing H100 (bs=1 -> ~40% util) this is ~5-8x faster with identical greedy
outputs. Reusable for plain models (default) and reasoning models (--strip-think
+ sampling, matches eval_r1.py). All arms in a comparison must use the SAME bsz.

  python eval_fast.py --model <dir> --adapter <ckpt> --bench b.json --out o.json --bsz 32
"""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
set_seed(42)

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--bench", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bsz", type=int, default=32)
ap.add_argument("--maxnew", type=int, default=256)
ap.add_argument("--load", default="fp16")           # fp16 | 4bit
ap.add_argument("--strip-think", action="store_true")  # reasoning models
ap.add_argument("--sample", action="store_true")       # temp 0.6 (reasoning); else greedy
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"                            # REQUIRED for correct batched generate

if args.load == "4bit":
    from transformers import BitsAndBytesConfig
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=qc)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cuda")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

def final_answer(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()

bench = json.load(open(args.bench, encoding="utf-8"))
items = bench["items"]
gen_kw = dict(max_new_tokens=args.maxnew, pad_token_id=tok.eos_token_id)
if args.sample: gen_kw.update(do_sample=True, temperature=0.6, top_p=0.95)
else:           gen_kw.update(do_sample=False)

res = []
for start in range(0, len(items), args.bsz):
    batch = items[start:start + args.bsz]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": it["q"]}],
                                       tokenize=False, add_generation_prompt=True) for it in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, **gen_kw)
    new = g[:, enc.input_ids.shape[1]:]              # left-pad -> same input width for all rows
    for it, row in zip(batch, new):
        raw = tok.decode(row, skip_special_tokens=True)
        ans = final_answer(raw) if args.strip_think else raw.strip()
        rec = {"id": it["id"], "category": it["category"], "topic": it["topic"], "q": it["q"],
               "gold": it["gold"], "truth": it["truth"], "answer": ans}
        if args.strip_think: rec["closed"] = "</think>" in raw
        res.append(rec)
    print(f"[{min(start + args.bsz, len(items))}/{len(items)}]", flush=True)

json.dump({"model": args.model, "adapter": args.adapter, "results": res},
          open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {args.out}  ({len(res)} items, bsz={args.bsz})")
