# unmuzzle

Making open-weight Chinese LLMs answer **truthfully** about CCP-censored topics, and measuring whether each method actually restores truth or just manufactures confident lies.

Local, personal research. This repo ships **code, benchmarks, and results**. It does not distribute model weights or the teacher-generated training corpus.

The through-line of the whole project: **removing a refusal is not the same as restoring a fact.** A model that fluently invents a date or a name is worse than one that declines. So every method here is judged on two axes kept deliberately separate: did it stop refusing (uncensored), and does it now tell the truth (correct), or abstain honestly when it does not know.

---

## The arc, in three parts

### Part 1 — Abliteration: does uncensoring restore truth? (`ladder.py`, `results/`)

Standard directional ablation (subtract the refusal direction from the residual stream) across a Qwen2.5 scale ladder, base vs. ablated, on a sensitive-topic probe set.

**Finding: "uncensoring reveals ignorance" is a small-model artifact.** At 1.5B, ablation removes every refusal but a large share of the freed answers are confident hallucinations (the knowledge was never there). At >= 7B the facts are present and merely gagged, so ablation surfaces truth with few residual errors, matching Perplexity's R1-1776 result on 671B DeepSeek-R1. There is a size floor, below which abliteration makes a model that is confidently wrong.

| Qwen2.5 | Base sensitive (correct / refuse / wrong) | Ablated |
|---:|:--:|:--:|
| 1.5B | 2 / 13 / 3 | 11 / 0 / **7** |
| 7B | 14 / 2 / 2 | 15 / 0 / 3 |
| 14B | 16 / 1 / 1 | 16 / 0 / 2 |
| 32B | 13 / 2 / 3 | 15 / 0 / 3 |

### Part 2 — Can a decode-time trick make it truthful? (`bench/multi_condition.py`, `bench/cond_3b.json`)

Since abliteration leaves residual falsehoods, we tested "verify-then-answer": the model answers, then self-checks whether the subject is real and its facts reliable, abstaining if not. With proper controls (a placebo self-check, and a one-pass "just permit I-don't-know" baseline) and a 2x2 benchmark (sensitive/neutral x real/invented).

**Finding: the decode trick fails.** It does not selectively catch lies; it induces global timidity. On Qwen2.5-3B it drove sensitive-real abstention from 1/40 to 36/40 (refusing facts it should answer). Selectivity (abstain on invented minus abstain on real): the trivial "permit I-don't-know" control scored **+31pp**, verify-then-answer scored **+20pp, the worst of all conditions**. Converges with R1-1776: you cannot cheat your way to truthful-uncensored with a prompt trick. The proven path is training.

### Part 3 — Curated honesty-SFT: the real fix (`sft/`)

**Full reproducible recipe: [METHODOLOGY.md](METHODOLOGY.md)** — data, QLoRA config, the self-verification loop, evaluation, and deployment, end to end.

R1-1776's recipe, done locally: QLoRA fine-tune Qwen2.5-7B-Instruct on ~1,100 curated Chinese Q&A that (a) answer censored topics factually with no propaganda framing, and (b) abstain honestly on invented/unknowable things. LoRA targets the MLP projections too (facts live in the MLP, per ROME/MEMIT). Answer-tokens-only loss.

**Result (70-item held-out benchmark, cross-family LLM judge, base vs. fine-tuned):**

| Metric | Base | Fine-tuned |
|---|---|---|
| Sensitive-real **factual** | 50% (20/40) | **80% (32/40)** |
| Invented traps **honest-abstain** | 87% (21/24) | **100% (24/24)** |
| Invented **fabrication** | 12.5% | **0%** |
| Over-abstention on real facts | 5% | 2.5% |
| False-balance "steering" on sensitive | 6/40 | **0/40** |

The base model, even prompted to be honest, does not refuse; it **answers wrongly** (45% of sensitive answers), split between denial, euphemism ("政治风波"), false-balance steering ("views differ, consult sources"), and confabulated names. Curated SFT removed all three propaganda techniques and cut fabrication to zero. Full per-item before/after in `sft/before_after.json` (rendered by `sft/build_artifact.py`).

**Hardening.** The self-verification loop (below) flagged ~7 residual failures. Diagnosis: not censorship but **knowledge gaps** (the model confabulated wrong names/cities: Hu Yaobang for Zhao Ziyang, Guilin for Urumqi). A targeted round of diverse, contrastive disambiguation examples fixed **all 7**, confirming SFT injects and disambiguates facts the base never learned.

---

## The self-verification loop

The SFT run gates itself against the goal instead of training once and hoping. `verify_spec.json` makes the goal measurable (G1-G7: uncensored-factual, honest-abstain, no over-abstention, general ability preserved, selective abstention, no fabrication, generalizes). `grade_workflow.js` grades eval answers with a **cross-family** judge against provided ground truth (a keyword wrapped in propaganda counts as wrong, and abstain is kept distinct from refuse). `check_criteria.py` computes the gate and, on failure, names the failure mode and the one adjustment to make (over-abstention -> cut abstain-data share; forgetting -> more replay; memorization -> more paraphrase diversity), so each iteration is directed, not a guess.

---

## Benchmarks

- `bench/benchmark_china_v1.json` — 70 items, 2x2 (sensitive/neutral x real/invented), verifiable ground truth.
- `bench/benchmark_china_v2.json` — 265 items, the same structure at 5x scale for statistical reliability and per-topic granularity, deduped against training (shared facts are the generalization test; shared phrasings would be contamination).

---

## Repo layout

```
ladder.py, ladder_multiarch.py, abliterate.py, compute_dir.py   # Part 1: abliteration study
direction_prompts.json, probe_set.json, results/                # Part 1: probes + results
bench/                                                            # benchmarks + Part 2 (decode-trick) code & data
sft/  train_sft.py  eval_model.py                                # Part 3: curated honesty-SFT
      verify_spec.json  check_criteria.py  grade_workflow.js     # the self-verification loop
      *_eval.json  *_graded.json  before_after.json              # results
      build_artifact.py                                          # before/after viewer generator
```

## Caveats and scope

- Judge-graded (cross-family, ground-truth-anchored), not human-graded; trends are robust, exact cells are indicative.
- Local, personal-use only. **Model weights and the Claude-generated training corpus are not distributed** (they are gitignored). If this were ever published, the teacher would need to be swapped for an openly-licensed model.
- The honest failure modes are kept in view: residual wrong answers on thin-coverage facts, and general-knowledge accuracy measured on a small cell (fixed in the v2 benchmark).
