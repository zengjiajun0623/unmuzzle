# 265-item benchmark (benchmark_china_v2.json) — base vs. hardened Qwen2.5-7B

Cross-family LLM judge, ground-truth-anchored. Held-out, deduped vs. training.

| Metric | Base | Hardened (SFT + disambiguation) |
|---|---|---|
| Sensitive-real factual | 48% (76/157) | 68% (107/157) |
| Declines a real fact | 10/157 | 0/157 |
| Honest-abstain on invented | 66% (46/70) | 81% (57/70) |
| Fabricates on invented | 34% (24/70) | 19% (13/70) |
| General knowledge | 38/38 | 37/38 |

## Sensitive-real correct rate by topic (base -> hardened)
taiwan 88->100 · hongkong 61->89 · dissidents 50->88 · history_ccp 78->83 ·
tiananmen 24->65 · tibet 53->65 · falungong 33->50 · censorship 28->44 · xinjiang 22->33

## Honest read
- SFT improves factual accuracy +20pp across a mostly-untrained set, with zero over-abstention: the behavior generalized, not just the drilled facts.
- The larger eval is more sober than the 70-item (68% vs 80%): the ceiling is real knowledge of obscure specifics (exact dates, foreign names), where the base model is simply ignorant. Weakest = xinjiang, censorship (fact-dense with obscure specifics).
- Honesty generalizes only partially: 81% abstain on brand-new invented terms (vs 100% on the trained-distribution terms in the 70-item set), still ~19% fabrication.

## Three-way: SFT vs. abliteration vs. base (265 items, same judge)

| Model | Sensitive-factual | Honest-abstain (invented) | Fabricates | General |
|---|---|---|---|---|
| Base (unmodified) | 48% | 66% | 34% | 38/38 |
| huihui (abliterated-v2) | 42% | 73% | 27% | 38/38 |
| **Our curated SFT** | **68%** | **81%** | **19%** | 37/38 |

Verdict: abliteration removes refusals but adds no knowledge, so it is roughly base-level on factual accuracy (42% vs 48%, within the abliteration tax) and only marginally more honest. Curated SFT injects facts AND teaches calibrated abstention, winning decisively on every truthfulness axis. Training beats weight-editing for uncensored-AND-truthful.
