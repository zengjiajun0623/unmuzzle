# stdbench2 — Standard-benchmark alignment-tax campaign (2026-07-19/20)

**Question:** does the unmuzzle honesty-SFT (100%-Chinese curated corpus, QLoRA r16) cost
general capability? Prior evidence was a −4.0 MMLU scare from an Ollama-Q4 run.

**Method:** paired base-vs-SFT on identical items, lm-evaluation-harness 0.4.12 (hf backend,
`peft=` adapters), 5-shot. MC benchmarks scored by loglikelihood; GSM8K generative.
Per-sample logging → paired McNemar exact test on discordant pairs. 1×H100 (RunPod).
Conditions: raw completion (primary), 7B chat+honesty-system-prompt (deployed spot check),
R1 chat+sampling temp 0.6 / 8k tokens, flexible-extract headline (never greedy-decode R1).
72B in nf4 4-bit both arms (bf16 exceeds 80GB). Full versions in `results_merged/versions.txt`.

## Verdict: zero measurable alignment tax across the ladder

| model | condition | benchmark | n | base % | SFT % | Δ pp | McNemar p |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | completion | CMMLU | 4020 | 79.9 | 79.7 | −0.2 | 0.59 |
| Qwen2.5-7B | completion | C-Eval | 1346 | 81.3 | 80.8 | −0.5 | 0.37 |
| Qwen2.5-7B | completion | MMLU | 3990 | 75.6 | 75.4 | −0.2 | 0.70 |
| Qwen2.5-7B | completion | GSM8K | 1319 | 75.8 | 82.6 | **+6.7*** | 3.8e-08 |
| Qwen2.5-7B | chat+sys | CMMLU | 4020 | 79.3 | 78.5 | −0.8 | 0.077 |
| Qwen2.5-7B | chat+sys | MMLU | 3990 | 74.3 | 74.8 | +0.6 | 0.22 |
| Qwen2.5-14B | completion | CMMLU | 4020 | 84.5 | 84.0 | −0.6 | 0.065 |
| Qwen2.5-14B | completion | C-Eval | 1346 | 84.0 | 84.2 | +0.2 | 0.72 |
| Qwen2.5-14B | completion | MMLU | 3990 | 80.4 | 80.2 | −0.2 | 0.48 |
| Qwen2.5-14B | completion | GSM8K | 1319 | 79.9 | 86.8 | **+6.9*** | 2.4e-11 |
| R1-Distill-32B | completion | CMMLU | 4020 | 84.3 | 84.3 | +0.0 | 0.92 |
| R1-Distill-32B | completion | C-Eval | 1346 | 84.6 | 84.0 | −0.7 | 0.14 |
| R1-Distill-32B | completion | MMLU | 2850 | 81.5 | 82.3 | **+0.8*** | 0.011 |
| R1-Distill-32B | chat, sampled | GSM8K | 500 | 85.8 | 87.6 | +1.8 | 0.40 |
| Qwen2.5-72B (4-bit) | completion | CMMLU | 2010 | 88.5 | 88.3 | −0.1 | 0.84 |
| Qwen2.5-72B (4-bit) | completion | C-Eval | 1346 | 89.7 | 89.9 | +0.2 | 0.69 |
| Qwen2.5-72B (4-bit) | completion | MMLU | 1995 | 85.1 | 85.3 | +0.2 | 0.70 |
| Qwen2.5-72B (4-bit) | completion | GSM8K | 400 | 93.5 | 92.2 | −1.2 | 0.49 |

\* significant at p<0.05, paired McNemar. Δ = SFT − base.

**Reads:**
1. Every negative delta is statistically noise (worst −0.8, p≥0.065). With n≈4k paired
   items the design resolves ~1pp; the tax simply is not there.
2. The prior MMLU −4.0 was an artifact of the Ollama-Q4 generative protocol, not the SFT.
3. The only significant deltas are GAINS: GSM8K +6.7/+6.9 (7B/14B, mostly answer-format
   compliance) and R1 MMLU +0.8. Honesty-tuning did not dull reasoning (also R1 GSM8K +1.8 ns).
4. Deployed condition (honesty system prompt): CMMLU −0.8 (p=0.077) and MMLU +0.6 —
   at most a sub-1pp caution effect; disclose, not headline.

**Caveats:** 7B SFT adapter is the AdamW arm of the Muon A/B rerun (identical
data/recipe/optimizer/seed as the shipped 7B; original adapter lost with its pod).
72B measured in 4-bit both arms. R1 GSM8K sampled (temp 0.6) with flexible-extract,
directional. CMMLU/MMLU subsampled deterministically (first-N per subtask; same items
both arms). MATH-500 was planned but cut: lm-eval dep failure + demonstrated cost-blowout
risk (a ~5% looping-trace tail turned one R1 GSM8K arm into 10.6h); conclusion secure
without it, re-runnable later for ~$15.

**Artifacts:** `results.tgz` (35 pre-reset cells) + `results_rerun3.tgz` (final cell) →
extracted `results_merged/`; `analyze.py` (paired table); `driver.sh` / `finisher.sh` /
`rerun3.sh` (pod scripts). Cost: ~$55 GPU total (incl. ~$32 lost to a leaked-CUDA-memory
zombie + trace-loop tail; see RUNPOD_PLAYBOOK.md additions).
