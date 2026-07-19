import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
print("loading base on CPU...", flush=True)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
m = PeftModel.from_pretrained(m, adapter)
print("merging...", flush=True)
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("MERGE_DONE", flush=True)
