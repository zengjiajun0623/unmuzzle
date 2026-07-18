# -*- coding: utf-8 -*-
"""Eval DeepSeek-V4-Flash (BF16 weights loaded 4-bit, multi-GPU) on the 265-item
benchmark, using V4's own encoder (chat mode). base or base+LoRA adapter."""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, "/workspace/e/enc")
from encoding_dsv4 import encode_messages
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--bench", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--maxnew", type=int, default=256)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
print("loading (4-bit, device_map=auto)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qc, device_map="auto")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    print("adapter:", args.adapter, flush=True)
model.eval()
dev = model.get_input_embeddings().weight.device

bench = json.load(open(args.bench, encoding="utf-8"))
res = []
for i, it in enumerate(bench["items"]):
    prompt = encode_messages([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": it["q"]}], thinking_mode="chat")
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=args.maxnew, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    ans = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    res.append({"id": it["id"], "category": it["category"], "topic": it["topic"], "q": it["q"],
                "gold": it["gold"], "truth": it["truth"], "answer": ans})
    print(f"[{i+1}/{len(bench['items'])}] {it['id']}", flush=True)

json.dump({"model": args.model, "adapter": args.adapter, "results": res},
          open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {args.out}", flush=True)
