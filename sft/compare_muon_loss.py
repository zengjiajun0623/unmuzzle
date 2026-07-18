# -*- coding: utf-8 -*-
"""Primary Muon-vs-AdamW metric: paired per-step training loss (same data_seed=42
-> identical batch order per arm, so losses are paired). Reports EMA-smoothed
curve summary + final-epoch mean loss + area-under-curve, for the 3 arms."""
import json, os, glob
D = os.path.expanduser("~/llm-lab/unmuzzle-repo/sft/muon_out")
ARMS = [("a_adamw", "AdamW  lr1e-4 (baseline)"), ("b_muon1", "Muon   lr1e-4"), ("c_muon2", "Muon   lr2e-4")]

def load(a):
    fn = os.path.join(D, f"loss_{a}.jsonl")
    if not os.path.exists(fn): return None
    return [json.loads(l) for l in open(fn) if l.strip()]

def ema(xs, alpha=0.2):
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out

print("=== Muon-vs-AdamW 7B: paired training-loss comparison ===\n")
curves = {}
for a, name in ARMS:
    d = load(a)
    if not d:
        print(f"  {name:26s}: (loss.jsonl missing)"); continue
    losses = [r["loss"] for r in d]
    steps = [r["step"] for r in d]
    n = len(losses)
    final_third = losses[int(n*2/3):]
    curves[a] = losses
    print(f"  {name:26s}: steps={n}  first={losses[0]:.3f}  min={min(losses):.3f}  "
          f"final-third mean={sum(final_third)/len(final_third):.3f}  AUC={sum(losses)/n:.3f}")

# head-to-head: at matched steps, how often does each Muon arm beat AdamW?
if "a_adamw" in curves:
    base = curves["a_adamw"]
    for a, name in ARMS[1:]:
        if a not in curves: continue
        c = curves[a]; m = min(len(base), len(c))
        ba, ca = ema(base[:m]), ema(c[:m])
        wins = sum(ca[i] < ba[i] for i in range(m))
        gap = (sum(ca)/m) - (sum(ba)/m)
        print(f"\n  {name} vs AdamW: lower-loss at {wins}/{m} steps ({100*wins//m}%); "
              f"mean EMA gap {gap:+.4f} ({'Muon better' if gap<0 else 'AdamW better'})")
print("\nLOSS_DONE")
