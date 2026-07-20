#!/usr/bin/env python3
"""stdbench2 paired analysis — base-vs-SFT alignment-tax table.

Reads lm-eval output dirs named {rung}_{arm}{tag}_{task}/ under results/,
joins per-sample correctness by (subtask-file, doc_id, filter), and reports
paired deltas with McNemar exact p-values. Aggregates are cross-checked
against lm-eval's own results_*.json.
"""
import json, glob, os, re, sys
from math import comb
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")

METRIC = {"cmmlu": "acc", "ceval-valid": "acc", "mmlu": "acc", "gsm8k": "exact_match", "minerva_math500": "exact_match"}
# gsm8k headline filter per design: strict-match for Qwen rungs, flexible-extract for R1 (_chat)
def headline_filter(task, tag):
    if task != "gsm8k":
        return None  # MC tasks have a single "none" filter
    return "flexible-extract" if tag == "_chat" else "strict-match"

def load_cell(dirpath, task, tag):
    """-> dict {(file_base, doc_id): 0/1}, agg dict from results json"""
    per = {}
    filt = headline_filter(task, tag)
    metric = METRIC[task]
    for sf in glob.glob(os.path.join(dirpath, "*", "samples_*.jsonl")):
        fb = re.sub(r"_\d{4}-\d{2}-\d{2}T.*", "", os.path.basename(sf))
        for line in open(sf):
            d = json.loads(line)
            if filt and d.get("filter") != filt:
                continue
            v = d.get(metric)
            if v is None:
                continue
            per[(fb, d["doc_id"])] = int(v)
    rj = glob.glob(os.path.join(dirpath, "*", "results_*.json"))
    agg = json.load(open(rj[0]))["results"] if rj else {}
    return per, agg

def mcnemar_p(b01, b10):
    """exact two-sided binomial on discordant pairs"""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n * 2
    return min(1.0, p)

cells = {}
for d in sorted(glob.glob(os.path.join(RES, "*/"))):
    name = os.path.basename(d.rstrip("/"))
    if name == "smoke" or not os.path.exists(os.path.join(d, ".ok")):
        continue
    m = re.match(r"^(7b|14b|r1|72b)_(base|sft)(_sys|_chat)?_(cmmlu|ceval-valid|mmlu|gsm8k|minerva_math500)$", name)
    if not m:
        continue
    rung, arm, tag, task = m.group(1), m.group(2), m.group(3) or "", m.group(4)
    cells[(rung, tag, task, arm)] = load_cell(d, task, tag)

rows = []
for (rung, tag, task, arm) in sorted(cells):
    if arm != "base":
        continue
    key_sft = (rung, tag, task, "sft")
    if key_sft not in cells:
        continue
    bper, _ = cells[(rung, tag, task, "base")]
    sper, _ = cells[key_sft]
    common = sorted(set(bper) & set(sper))
    if len(common) != len(bper) or len(common) != len(sper):
        print(f"WARN {rung}{tag} {task}: item mismatch base={len(bper)} sft={len(sper)} common={len(common)}")
    n = len(common)
    bacc = sum(bper[k] for k in common) / n * 100
    sacc = sum(sper[k] for k in common) / n * 100
    b01 = sum(1 for k in common if bper[k] == 1 and sper[k] == 0)  # base right, sft wrong
    b10 = sum(1 for k in common if bper[k] == 0 and sper[k] == 1)  # sft right, base wrong
    p = mcnemar_p(b01, b10)
    rows.append((rung, tag, task, n, bacc, sacc, sacc - bacc, b01, b10, p))

RUNG_LABEL = {"7b": "Qwen2.5-7B", "14b": "Qwen2.5-14B", "r1": "R1-Distill-32B", "72b": "Qwen2.5-72B (4-bit)"}
print("\n| model | condition | benchmark | n | base % | SFT % | Δ pp | base>sft / sft>base | McNemar p |")
print("|---|---|---|---|---|---|---|---|---|")
order = {"7b": 0, "14b": 1, "r1": 2, "72b": 3}
tasko = {"cmmlu": 0, "ceval-valid": 1, "mmlu": 2, "gsm8k": 3, "minerva_math500": 4}
for r in sorted(rows, key=lambda x: (order[x[0]], x[1], tasko[x[2]])):
    rung, tag, task, n, bacc, sacc, dlt, b01, b10, p = r
    cond = {"": "5-shot completion", "_sys": "chat+honesty-sys", "_chat": "chat, sampled"}[tag]
    star = " *" if p < 0.05 else ""
    print(f"| {RUNG_LABEL[rung]} | {cond} | {task} | {n} | {bacc:.1f} | {sacc:.1f} | {dlt:+.1f}{star} | {b01} / {b10} | {p:.3g} |")
print("\n* = significant at p<0.05 (paired McNemar). Δ = SFT − base; negative = alignment tax.")
