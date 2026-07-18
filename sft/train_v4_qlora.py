# -*- coding: utf-8 -*-
"""Curated honesty-SFT on DeepSeek-V4-Flash via QLoRA on the BF16 weights
(bitsandbytes 4-bit, the PROVEN training path — avoids the fp8-native training
wall). LoRA on the dense path; V4's own encoder; answer-tokens-only loss.

  python train_v4_qlora.py --model RedHatAI/DeepSeek-V4-Flash-BF16 --data train_v2.jsonl --out adapter_v4
"""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, "/workspace/e/enc")
from encoding_dsv4 import encode_messages
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")
EOS = "<｜end▁of▁sentence｜>"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)          # a BF16 safetensors repo, e.g. RedHatAI/DeepSeek-V4-Flash-BF16
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--maxlen", type=int, default=1024)
ap.add_argument("--r", type=int, default=16)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
print("loading BF16 weights as 4-bit QLoRA (device_map=auto across GPUs)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qc, device_map="auto")
model = prepare_model_for_kbit_training(model)

# V4 dense-path targets (attention + shared-expert MLP); routed experts stay frozen
lora = LoraConfig(r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["kv_proj", "q_a_proj", "q_b_proj", "o_a_proj", "o_b_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.config.use_cache = False
model.is_parallelizable = True; model.model_parallel = True

rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]

def encode(ex):
    prompt = encode_messages([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": ex["q"]}], thinking_mode="chat")
    ans = ex["a"].rstrip() + EOS
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ans, add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}

ds, dropped = [], 0
for r in rows:
    e = encode(r)
    if len(e["input_ids"]) > args.maxlen: dropped += 1; continue
    ds.append(e)
if dropped: print(f"WARNING dropped {dropped} over maxlen={args.maxlen}", flush=True)
_ex = ds[0]; _ans = [t for t, l in zip(_ex["input_ids"], _ex["labels"]) if l != -100]
print("MASK CHECK — answer region:", repr(tok.decode(_ans)[:90]), flush=True)
print("training examples:", len(ds), flush=True)

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        pad = m - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        att.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}

targs = TrainingArguments(
    output_dir=args.out, per_device_train_batch_size=1,
    gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=15,
    max_grad_norm=1.0, weight_decay=0.0, seed=42,
    logging_steps=1, save_strategy="epoch", bf16=True, optim="paged_adamw_8bit",
    report_to=[], gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate).train()
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}", flush=True)
