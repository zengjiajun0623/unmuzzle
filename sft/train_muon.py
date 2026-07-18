# -*- coding: utf-8 -*-
"""Optimizer A/B for honesty-SFT: byte-for-byte the train_sft.py recipe (same data,
LoRA config, lr, schedule, seed, answer-only masking) with ONE knob — --optimizer
{adamw|muon}. adamw = the shipped paged_adamw_8bit baseline; muon = Moonshot's Muon
on the 2D LoRA matrices. Logs per-step loss to <out>/loss.jsonl so we can compare
CONVERGENCE (Muon's real claimed edge), not just final graded accuracy.

  python train_muon.py --model Qwen/Qwen2.5-7B-Instruct --data train_v2.jsonl --out ./a7_adamw --optimizer adamw
  python train_muon.py --model Qwen/Qwen2.5-7B-Instruct --data train_v2.jsonl --out ./a7_muon  --optimizer muon
"""
import json, argparse, os, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from muon import Muon

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--optimizer", choices=["adamw", "muon"], required=True)
ap.add_argument("--load", default="4bit")
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

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

if args.load == "4bit":
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", quantization_config=qc)
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

def encode(ex):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": ex["q"]}],
                                     tokenize=False, add_generation_prompt=True)
    ans = ex["a"].rstrip()
    if ans.endswith("<|im_end|>"): ans = ans[:-len("<|im_end|>")].rstrip()
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ans, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}

ds, dropped = [], 0
for r in rows:
    e = encode(r)
    if len(e["input_ids"]) > args.maxlen: dropped += 1; continue
    ds.append(e)
if dropped: print(f"WARNING dropped {dropped} examples exceeding maxlen={args.maxlen}")
_ex = ds[0]; _ans = [t for t, l in zip(_ex["input_ids"], _ex["labels"]) if l != -100]
print("MASK CHECK — answer region decodes to:", repr(tok.decode(_ans)[:80]))
print(f"training examples: {len(ds)} | optimizer={args.optimizer}")

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

os.makedirs(args.out, exist_ok=True)
class LossLogger(TrainerCallback):
    def on_log(self, a, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            with open(os.path.join(args.out, "loss.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, "loss": logs["loss"],
                                    "epoch": logs.get("epoch")}) + "\n")

# baseline uses the shipped 8-bit AdamW; muon arm swaps optimizer only
optim_str = "paged_adamw_8bit" if args.optimizer == "adamw" else "adamw_torch"
targs = TrainingArguments(
    output_dir=args.out, per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=15,
    max_grad_norm=1.0, weight_decay=0.0, seed=42, data_seed=42,
    logging_steps=2, save_strategy="epoch", bf16=True, optim=optim_str,
    report_to=[], gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})

class MuonTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            muon_p = [p for _, p in self.model.named_parameters() if p.requires_grad and p.ndim >= 2]
            other_p = [p for _, p in self.model.named_parameters() if p.requires_grad and p.ndim < 2]
            print(f"MUON groups: 2D(muon)={len(muon_p)}  1D(adamw)={len(other_p)}")
            # design assumes a PURE-Muon arm (LoRA bias="none" -> all trainables 2D);
            # a future config change adding 1D trainables would silently mix optimizers.
            assert len(other_p) == 0, f"unexpected 1D trainables: {len(other_p)} — comparison assumes pure-Muon arm"
            self.optimizer = Muon(muon_params=muon_p, adamw_params=other_p,
                                  lr=self.args.learning_rate, momentum=0.95,
                                  weight_decay=self.args.weight_decay)
        return self.optimizer

TrainerCls = MuonTrainer if args.optimizer == "muon" else Trainer
TrainerCls(model=model, args=targs, train_dataset=ds, data_collator=collate,
           callbacks=[LossLogger()]).train()
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}  (optimizer={args.optimizer})")
