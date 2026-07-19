# -*- coding: utf-8 -*-
"""M2: QLoRA DPO on R1-Distill-32B, from the SFT policy.

Reference-model correctness (the DPO gotcha): DPO's reference must be the SFT policy,
NOT the base. So the pipeline MERGES the SFT adapter into the base first (--model must
already be the SFT-MERGED model), then trains a FRESH DPO LoRA on top. TRL's DPOTrainer
disables the trainable (DPO) adapter for the reference forward pass -> reference = the
merged SFT model. Correct by construction.

Pairs (dpo_pairs.jsonl): {prompt, chosen, rejected} where chosen/rejected are FULL
assistant completions (<think> + answer), NOT stripped -- we optimize the whole
reasoning->answer sequence. Built by reward.make_pairs from on-policy rollouts.

Two-sided reward already baked into pairing (chosen=high reward, rejected=low, margin>=1.5).
rpo_alpha=0.2 adds the NLL-on-chosen auxiliary (guards DPO chosen-likelihood decay +
acts as SFT replay). beta=0.1, lr=1e-5, 1 epoch, KL-anchored to the SFT reference.

  python dpo_train.py --model /runpod/r1_sftmerged --pairs dpo_pairs.jsonl --out ./dpo_adapter
"""
import json, argparse, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)     # MUST be the SFT-merged model (ref = SFT)
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--rpo_alpha", type=float, default=0.2)
ap.add_argument("--maxlen", type=int, default=2048)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def to_prompt(q):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":q}],
                                   tokenize=False, add_generation_prompt=True)
rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
ds = Dataset.from_list([{"prompt": to_prompt(r["q"]), "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
print(f"DPO pairs: {len(ds)}")

qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", quantization_config=qc)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

import inspect
_want = dict(output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
             num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
             warmup_steps=10, beta=args.beta, max_length=args.maxlen,
             bf16=True, optim="paged_adamw_8bit", logging_steps=2,
             save_strategy="epoch", report_to=[], gradient_checkpointing=True,
             gradient_checkpointing_kwargs={"use_reentrant": False}, seed=42)
_valid = set(inspect.signature(DPOConfig.__init__).parameters)
_dropped = [k for k in _want if k not in _valid]
if _dropped: print("DPOConfig dropped unsupported:", _dropped, flush=True)
cfg = DPOConfig(**{k: v for k, v in _want.items() if k in _valid})
# ref_model=None + peft_config -> TRL uses adapter-disabled (=SFT-merged) as the reference
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(args.out)
print(f"saved DPO adapter to {args.out}")
