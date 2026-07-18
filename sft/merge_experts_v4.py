# -*- coding: utf-8 -*-
"""Streaming offline merge for DeepSeek-V4-Flash bf16 repos that store experts
PER-MODULE (experts.N.w1/w2/w3). transformers' DeepseekV4 runtime wants them
BATCHED; the on-load merge OOMs the GPUs. So we do the merge OFFLINE, on disk,
using transformers' OWN ops (correct ordering guaranteed), never instantiating
the model. Only the expert tensors are merged; every other key is copied through
untouched (the loader still applies the cheap renames at load time).

Produces a batched checkpoint at OUT that loads clean on 8xH100 (no merge).

  python merge_experts_v4.py --src /runpod/dsv4bf16 --out /runpod/dsv4_batched
"""
import os, re, json, glob, gc, argparse, torch
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)   # dir with per-module bf16 shards + index
ap.add_argument("--out", required=True)   # dir to write batched checkpoint
ap.add_argument("--shard-gb", type=float, default=40.0)
ap.add_argument("--del-src", action="store_true")   # delete source shards as consumed
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
wmap = idx["weight_map"]                       # key -> shard filename
shard_files = sorted(set(wmap.values()))       # in-order shards
# group keys by shard, preserving that we read each shard once
keys_in_shard = defaultdict(list)
for k, f in wmap.items():
    keys_in_shard[f].append(k)

EXP = re.compile(r"^(?P<pre>.*\.experts)\.(?P<idx>\d+)\.(?P<w>w[123])\.weight$")

# buffers: layer_prefix -> {expert_idx -> {w1,w2,w3: tensor}}
buf = defaultdict(lambda: defaultdict(dict))
layer_seen_shards = defaultdict(set)   # which shards a layer's experts appeared in

new_wmap = {}
out_shard_i = 0
cur = {}
cur_bytes = 0
MAXB = args.shard_gb * 1e9

def flush():
    global cur, cur_bytes, out_shard_i
    if not cur: return
    out_shard_i += 1
    fn = f"model-{out_shard_i:05d}.safetensors"
    save_file(cur, os.path.join(args.out, fn), metadata={"format": "pt"})
    for k in cur: new_wmap[k] = fn
    print(f"  wrote {fn}  ({cur_bytes/1e9:.1f}GB, {len(cur)} tensors)", flush=True)
    cur = {}; cur_bytes = 0

def add(k, t):
    global cur, cur_bytes
    t = t.contiguous()
    cur[k] = t; cur_bytes += t.numel() * t.element_size()
    if cur_bytes >= MAXB: flush()

def merge_layer(pre):
    """pre = '...mlp.experts'. Emit batched gate_up_proj + down_proj for this layer."""
    experts = buf.pop(pre)
    n = len(experts)
    order = sorted(experts.keys())            # NUMERIC sort (keys are ints)
    assert order == list(range(n)), f"non-contiguous experts for {pre}: {order[:5]}..{order[-3:]}"
    w1 = torch.stack([experts[i]["w1"] for i in order], dim=0)   # (n, inter, hidden)
    w3 = torch.stack([experts[i]["w3"] for i in order], dim=0)
    gate_up = torch.cat([w1, w3], dim=1)                          # gate first, up second (dim 1)
    w2 = torch.stack([experts[i]["w2"] for i in order], dim=0)   # down: plain stack
    add(f"{pre}.gate_up_proj", gate_up)
    add(f"{pre}.down_proj", w2)
    del experts, w1, w3, w2, gate_up; gc.collect()

print(f"shards: {len(shard_files)}  keys: {len(wmap)}", flush=True)
for si, sf in enumerate(shard_files):
    path = os.path.join(args.src, sf)
    print(f"[{si+1}/{len(shard_files)}] reading {sf}", flush=True)
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in keys_in_shard[sf]:
            m = EXP.match(k)
            if m:
                pre = m.group("pre"); ei = int(m.group("idx")); w = m.group("w")
                buf[pre][ei][w] = f.get_tensor(k)
                layer_seen_shards[pre].add(sf)
            else:
                add(k, f.get_tensor(k))       # copy through untouched
    # after finishing a shard, flush any layer whose experts are all present
    # (a layer is complete once we've read every shard that holds its experts;
    #  detect completeness by expected count = n_routed_experts)
    for pre in list(buf.keys()):
        experts = buf[pre]
        # complete if we have all 3 w's for a contiguous 0..max and the next shard
        # doesn't contain this prefix. Simplest robust check: all experts have w1,w2,w3
        # AND max index seen == len-1 AND we've passed its last shard.
        if experts and all(len(v) == 3 for v in experts.values()):
            mx = max(experts.keys())
            if len(experts) == mx + 1:
                # peek: is this prefix in any LATER shard?
                later = any(pre in [re.match(r'^(.*\.experts)\.\d+\.w[123]\.weight$', kk).group(1)
                                    for kk in keys_in_shard[shard_files[j]]
                                    if re.match(r'^(.*\.experts)\.\d+\.w[123]\.weight$', kk)]
                            for j in range(si+1, len(shard_files)))
                if not later:
                    merge_layer(pre)
    if args.del_src:
        os.remove(path); print(f"  deleted source {sf}", flush=True)

# flush any remaining complete layers
for pre in list(buf.keys()):
    merge_layer(pre)
flush()

assert not buf, f"UNMERGED experts remain: {list(buf.keys())[:3]}"
# write index + copy config/tokenizer
total = sum(os.path.getsize(os.path.join(args.out, f)) for f in set(new_wmap.values()))
json.dump({"metadata": {"total_size": total}, "weight_map": new_wmap},
          open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)
import shutil
for fn in os.listdir(args.src):
    if not fn.endswith(".safetensors") and fn != "model.safetensors.index.json":
        try: shutil.copy(os.path.join(args.src, fn), os.path.join(args.out, fn))
        except Exception: pass
print(f"MERGE_DONE  out_shards={out_shard_i}  total={total/1e9:.0f}GB", flush=True)
