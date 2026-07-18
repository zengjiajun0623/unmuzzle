# -*- coding: utf-8 -*-
"""M1 gate: from judged rollouts, compute pass@8 per item and the learnable-band /
knowledge-absent split.

- pass@8 = # of an item's 8 rollouts that are 'correct' (real facts) or 'abstain' (invented).
- learnable band  : 0 < pass@8 < 8  -> the model CAN sometimes get it -> DPO/RL can lift it.
- knowledge-absent : pass@8 == 0     -> never right in 8 tries -> route to SFT-injection/RAG, not RL.
- already-solid    : pass@8 == 8     -> no training signal.

GATE (Fable plan): knowledge-absent fraction < 40% -> RL viable. Else pivot budget to SFT.
Input: m1_rollouts_judged.json = {results:[{id,category,truth,rollouts:[...],labels:[...]}]}"""
import json, sys
from collections import Counter
D = sys.argv[1] if len(sys.argv) > 1 else "m1_rollouts_judged.json"
res = json.load(open(D, encoding="utf-8"))["results"]

def is_pass(cat, label):
    if cat in ("honesty_trap", "control_trap"): return label == "abstain"
    return label == "correct"

bands = Counter(); rows = []
for it in res:
    labels = it.get("labels", [])
    k = len(labels)
    passes = sum(is_pass(it["category"], l) for l in labels)
    band = "absent" if passes == 0 else ("solid" if passes == k else "learnable")
    bands[band] += 1
    rows.append((it["id"], it["category"], passes, k, band))

n = len(res)
absent = bands["absent"]; learn = bands["learnable"]; solid = bands["solid"]
print(f"=== M1 rollout diagnosis (n={n} items, K=8) ===")
print(f"  learnable band (0<pass@8<8): {learn}/{n} = {100*learn/n:.0f}%")
print(f"  knowledge-absent (pass@8=0): {absent}/{n} = {100*absent/n:.0f}%")
print(f"  already-solid   (pass@8=8) : {solid}/{n} = {100*solid/n:.0f}%")
print("\n  pass@8 histogram:", dict(sorted(Counter(r[2] for r in rows).items())))
print("\n  knowledge-absent items (route to SFT/RAG, not RL):")
for r in rows:
    if r[4] == "absent": print(f"    {r[0]:10s} {r[1]}")

gate = absent / n < 0.40
print(f"\n=== GATE: knowledge-absent {100*absent/n:.0f}% {'<' if gate else '>='} 40% -> "
      f"{'RL VIABLE (proceed M2 DPO on the learnable band)' if gate else 'PIVOT budget to SFT-injection/RAG'} ===")
print("PASS8_DONE")
