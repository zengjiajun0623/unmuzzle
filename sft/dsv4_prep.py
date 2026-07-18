# -*- coding: utf-8 -*-
"""Prep step for the 8xH100 V4 run: load the per-module bf16 repo on CPU (where
the expert MergeModulelist conversion has ~1TB RAM instead of packed 80GB GPUs),
then re-save in the BATCHED runtime format with save_original_format=False so the
GPU reload skips the merge entirely (no OOM). Library-based = correct by
construction. Falls back to the streaming merge if the flag isn't supported.

  python dsv4_prep.py /runpod/dsv4bf16 /runpod/dsv4_batched
"""
import sys, os, json, inspect, torch
from transformers import AutoModelForCausalLM
SRC, OUT = sys.argv[1], sys.argv[2]

# fail fast to fallback if the flag isn't in this transformers version
if "save_original_format" not in inspect.signature(AutoModelForCausalLM.save_pretrained).parameters:
    print("PREP_FALLBACK: save_original_format unsupported -> use streaming merge", flush=True)
    sys.exit(3)

print("loading on CPU (merge runs in ~1TB RAM, not on the GPUs)...", flush=True)
m = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
print("loaded + merged in RAM. checking save signature...", flush=True)
params = inspect.signature(m.save_pretrained).parameters
if "save_original_format" not in params:
    print("PREP_FALLBACK: save_original_format unsupported -> use streaming merge", flush=True)
    sys.exit(3)
print("saving BATCHED checkpoint (save_original_format=False)...", flush=True)
os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, save_original_format=False, safe_serialization=True, max_shard_size="40GB")
# sanity: index must now have batched expert keys, no per-module
idx = json.load(open(os.path.join(OUT, "model.safetensors.index.json")))
ks = list(idx["weight_map"].keys())
batched = any("experts.gate_up_proj" in k or "experts.down_proj" in k for k in ks)
permod = any(".experts.0.w1" in k or ".experts.0.gate_proj" in k for k in ks)
print(f"batched_keys={batched} per_module_keys={permod}", flush=True)
if batched and not permod:
    print("PREP_DONE", flush=True)
else:
    print("PREP_FAIL: save did not produce batched format", flush=True); sys.exit(1)
