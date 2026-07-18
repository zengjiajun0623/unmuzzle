# -*- coding: utf-8 -*-
"""QLoRA honesty-SFT: bake factual, non-evasive, calibrated answers into a small
local model (R1-1776 recipe, done locally). Loss on ANSWER tokens only. LoRA
targets attention + MLP projections (facts live in MLP per ROME/MEMIT).

  py -3.10 train_sft.py --model Qwen/Qwen2.5-7B-Instruct --data train.jsonl --out ./adapter
"""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--load", default="4bit")          # 4bit (QLoRA) | bf16
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bsz", type=int, default=1)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--maxlen", type=int, default=1024)
ap.add_argument("--r", type=int, default=16)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
assert tok.pad_token_id != tok.eos_token_id, "pad==eos would mask every stop token"

# fixed system prompt -- MUST be identical in the Ollama Modelfile + eval harness (review A3)
SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

if args.load == "4bit":
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda",
                quantization_config=qc)
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

lora = LoraConfig(r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.config.use_cache = False

rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
print(f"training examples: {len(rows)}")

def encode(ex):
    # TOKEN-level concat (review A1): never re-tokenize the joined string, or the
    # BPE boundary between the template's trailing \n and the answer can shift and
    # silently mis-mask the first answer token.
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": ex["q"]}],
                                     tokenize=False, add_generation_prompt=True)
    ans = ex["a"].rstrip()
    if ans.endswith("<|im_end|>"): ans = ans[:-len("<|im_end|>")].rstrip()   # A7 double-eos guard
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ans, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}

ds = []
dropped = 0
for r in rows:
    e = encode(r)
    if len(e["input_ids"]) > args.maxlen:        # A6: fail loudly, don't truncate off the eos
        dropped += 1; continue
    ds.append(e)
if dropped: print(f"WARNING dropped {dropped} examples exceeding maxlen={args.maxlen}")
# sanity-check masking on the first example (review checklist #1)
_ex = ds[0]; _ans = [t for t, l in zip(_ex["input_ids"], _ex["labels"]) if l != -100]
print("MASK CHECK — answer region decodes to:", repr(tok.decode(_ans)[:80]))
print(f"training examples: {len(ds)}")

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        pad = m - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        att.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}

targs = TrainingArguments(
    output_dir=args.out, per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=15,
    max_grad_norm=1.0, weight_decay=0.0, seed=42,
    logging_steps=5, save_strategy="epoch", bf16=True, optim="paged_adamw_8bit",
    report_to=[], gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate).train()
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}")
