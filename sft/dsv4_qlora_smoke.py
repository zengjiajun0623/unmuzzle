# -*- coding: utf-8 -*-
"""Fast pre-flight for the V4 QLoRA run: load BF16 weights 4-bit, attach LoRA,
one real fwd+bwd, confirm finite loss + LoRA grads. ~1 model load, then done.
Run this FIRST; only launch the full driver if it prints SMOKE_PASS."""
import torch, json, sys, traceback
sys.path.insert(0, "/workspace/e/enc")
from encoding_dsv4 import encode_messages
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
MID = sys.argv[1] if len(sys.argv) > 1 else "RedHatAI/DeepSeek-V4-Flash-BF16"
SYSTEM = ("你是一个诚实、客观的助手。你会如实、准确地回答问题；"
          "对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。")
EOS = "<｜end▁of▁sentence｜>"
tok = AutoTokenizer.from_pretrained(MID)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
print("loading 4-bit...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MID, quantization_config=qc, device_map="auto")
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["kv_proj","q_a_proj","q_b_proj","o_a_proj","o_b_proj",
                                  "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora); model.print_trainable_parameters()
model.config.use_cache = False; model.train()
dev = model.get_input_embeddings().weight.device
rows = [json.loads(l) for l in open("train_v2.jsonl", encoding="utf-8")][:2]
def enc(ex):
    p = encode_messages([{"role":"system","content":SYSTEM},{"role":"user","content":ex["q"]}], thinking_mode="chat")
    a = ex["a"].rstrip()+EOS
    pid = tok(p, add_special_tokens=False)["input_ids"]; aid = tok(a, add_special_tokens=False)["input_ids"]
    return pid+aid, [-100]*len(pid)+aid
b=[enc(r) for r in rows]; m=max(len(x[0]) for x in b)
ids=torch.tensor([x[0]+[pad_id]*(m-len(x[0])) for x in b]).to(dev)
lab=torch.tensor([x[1]+[-100]*(m-len(x[1])) for x in b]).to(dev)
att=torch.tensor([[1]*len(x[0])+[0]*(m-len(x[0])) for x in b]).to(dev)
try:
    out=model(input_ids=ids, attention_mask=att, labels=lab)
    print(f"forward OK loss={out.loss.item():.4f}", flush=True)
    out.loss.backward()
    import re
    g=sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if p.grad is not None and "lora" in n)
    layers=len({re.search(r'layers\.(\d+)\.',n).group(1) for n,p in model.named_parameters() if p.grad is not None and "lora" in n and re.search(r'layers\.(\d+)\.',n)})
    print(f"backward OK, lora grad={g:.3e}, layers_with_grad={layers}", flush=True)
    print("SMOKE_PASS" if g>0 and layers>=5 else "SMOKE_PARTIAL", flush=True)
except Exception as e:
    print("SMOKE_FAIL:", repr(e), flush=True); traceback.print_exc()
