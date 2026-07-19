# -*- coding: utf-8 -*-
"""M2b DPO — fixes the two M2 defects that caused the caution-overshoot:

FIX 1 (tokenization boundary): the R1 template forces a trailing `<think>\n` in the
prompt, and completions began mid-token after it -> TRL's prompt/(prompt+chosen)
prefix check mismatched, degrading the gradient. Here the prompt ENDS at
`<｜Assistant｜>` (we strip the forced `<think>`), and each completion STARTS with the
special `<think>` token (prepended when the pairs were built). Boundary is now a
clean special-token break, no BPE merge, no mismatch.

FIX 2 (anti-drift anchor): M2 used beta=0.1 and drifted into global caution
(sensitive-fact accuracy fell). TRL 1.8 dropped rpo_alpha (the NLL replay term), so
we tighten the KL-to-SFT anchor directly with beta=0.4 -> policy stays near the SFT
reference, which prevents the overshoot while the 70%-sensitive pairs still teach
'assert the fact > be wrong'.

  python dpo_train_v2.py --model /workspace/r1_sftmerged --pairs dpo_pairs_v2.jsonl --out dpo_adapter_v2
"""
import json, argparse, inspect, torch, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)      # SFT-merged (=DPO reference)
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.4)   # FIX 2: strong KL anchor to SFT
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--maxlen", type=int, default=2048)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def to_prompt(q):
    p = tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":q}],
                                tokenize=False, add_generation_prompt=True)
    # FIX 1: drop the forced trailing <think>... so the prompt ends at <｜Assistant｜>;
    # completions (which start with the special <think> token) then concatenate cleanly.
    i = p.rfind("<think>")
    return p[:i] if i != -1 else p

rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
ds = Dataset.from_list([{"prompt": to_prompt(r["q"]), "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
print(f"M2b DPO pairs: {len(ds)} | beta={args.beta} | prompt ends at <｜Assistant｜> (tokenization fix)")
_ex = ds[0]
print("prompt tail:", repr(_ex["prompt"][-30:]), "| chosen head:", repr(_ex["chosen"][:20]))

qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", quantization_config=qc)
model = prepare_model_for_kbit_training(model); model.config.use_cache = False
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

_want = dict(output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
             num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
             warmup_steps=10, beta=args.beta, max_length=args.maxlen, bf16=True,
             optim="paged_adamw_8bit", logging_steps=2, save_strategy="epoch", report_to=[],
             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, seed=42)
_valid = set(inspect.signature(DPOConfig.__init__).parameters)
_drop = [k for k in _want if k not in _valid]
if _drop: print("DPOConfig dropped:", _drop, flush=True)
cfg = DPOConfig(**{k: v for k, v in _want.items() if k in _valid})
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(args.out)   # saves the TRAINED peft adapter (not the base); trainer holds the wrapped model
import os
assert os.path.exists(os.path.join(args.out, "adapter_model.safetensors")), "adapter not saved!"
print(f"saved M2b DPO adapter to {args.out}")
