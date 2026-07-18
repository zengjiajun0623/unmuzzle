# -*- coding: utf-8 -*-
"""Curated honesty-SFT on DeepSeek-R1-Distill-Qwen-32B (a reasoning model, Qwen2.5
arch). Reasoning-augmented data: each example is <think>{honest reasoning}</think>
{answer}, so we uncensor the model's REASONING, not just its output (R1-1776 style).
QLoRA 4-bit; loss on the think+answer tokens only.

  python train_r1.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --data train_v2_reasoning.jsonl --out ./adapter_r1
"""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--maxlen", type=int, default=2048)   # reasoning traces are longer
ap.add_argument("--r", type=int, default=16)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=qc)
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora); model.print_trainable_parameters()
model.config.use_cache = False

# figure out how R1's template renders the assistant opening (does it force <think>?)
_probe = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": "x"}],
                                 tokenize=False, add_generation_prompt=True)
FORCES_THINK = _probe.rstrip().endswith("<think>")
print("template forces <think>:", FORCES_THINK, "| gen-prompt tail:", repr(_probe[-40:]))

rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]

def encode(ex):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": ex["q"]}],
                                     tokenize=False, add_generation_prompt=True)
    think = ex.get("think", "").strip()
    ans = ex["a"].rstrip()
    if FORCES_THINK:                      # template already opened <think>
        target = f"\n{think}\n</think>\n\n{ans}" if think else f"\n\n</think>\n\n{ans}"
    else:
        target = f"<think>\n{think}\n</think>\n\n{ans}" if think else ans
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}

ds, dropped = [], 0
for r in rows:
    e = encode(r)
    if len(e["input_ids"]) > args.maxlen: dropped += 1; continue
    ds.append(e)
if dropped: print(f"WARNING dropped {dropped} over maxlen={args.maxlen}")
_ex = ds[0]; _ans = [t for t, l in zip(_ex["input_ids"], _ex["labels"]) if l != -100]
print("MASK CHECK — target region:", repr(tok.decode(_ans)[:120]))
print("training examples:", len(ds))

def collate(b):
    m = max(len(x["input_ids"]) for x in b); ids, lab, att = [], [], []
    for x in b:
        pad = m - len(x["input_ids"])
        ids.append(x["input_ids"] + [tok.pad_token_id] * pad)
        lab.append(x["labels"] + [-100] * pad); att.append([1] * len(x["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

targs = TrainingArguments(output_dir=args.out, per_device_train_batch_size=1,
    gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, learning_rate=args.lr,
    lr_scheduler_type="cosine", warmup_steps=15, max_grad_norm=1.0, seed=42, logging_steps=5,
    save_strategy="epoch", bf16=True, optim="paged_adamw_8bit", report_to=[],
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate).train()
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}")
