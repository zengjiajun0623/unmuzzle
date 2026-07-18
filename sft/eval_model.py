# -*- coding: utf-8 -*-
"""Run the 70-item benchmark on a model (base, or base+LoRA adapter), recording
raw answers with the SAME system prompt used in training. Grading is a separate
LLM-judge pass (keyword grading credits propaganda).

  py eval_model.py --model Qwen/Qwen2.5-7B-Instruct --out base.json
  py eval_model.py --model Qwen/Qwen2.5-7B-Instruct --adapter ./ckpt --out sft.json
"""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer

# MUST be byte-identical to SYSTEM in train_sft.py and the Ollama Modelfile
SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--bench", default="benchmark_china_v1.json")
ap.add_argument("--out", required=True)
ap.add_argument("--maxnew", type=int, default=256)
ap.add_argument("--load", default="fp16")   # fp16 | 4bit (for big models on one GPU)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
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

bench = json.load(open(args.bench, encoding="utf-8"))
res = []
for i, it in enumerate(bench["items"]):
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": it["q"]}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=args.maxnew, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    ans = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    res.append({"id": it["id"], "category": it["category"], "topic": it["topic"], "q": it["q"],
                "gold": it["gold"], "truth": it["truth"], "answer": ans})
    print(f"[{i+1}/{len(bench['items'])}] {it['id']}", flush=True)

json.dump({"model": args.model, "adapter": args.adapter, "results": res},
          open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {args.out}")
