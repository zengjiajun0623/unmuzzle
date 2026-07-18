# Unmuzzle Phase 2 — Frontier-Teacher RL Distillation (execution plan)

Owner: Claude (execution) · Fable 5 (training-lead design) · Jiajun (gates: goal/direction/milestones)
Status: AWAITING APPROVAL — no spend until approved. 2026-07-18.

## Goal

Push R1-Distill-Qwen-32B-SFT beyond what imitation SFT can reach, using RL with a
frontier-teacher loop (Opus synthesizes + cites, Fable adversarially verifies + judges):

- Sensitive-fact accuracy 87.9% → **≥93%**
- Fabrication on invented terms 8.6% → **≤3%**
- Over-abstention on real facts stays **≤2/157**, general-fact stays **100%**
- Deliverable: adapter (backed up 3 places, not published) + the finding itself —
  "can a frontier committee teach a local model calibrated honesty via verified-reward RL?"

## Direction (Fable's locked calls)

- **Student:** R1-Distill-32B from our SFT adapter (72B = near ceiling; 7B/14B = knowledge-floor).
- **Algorithm:** iterated on-policy DPO (rollouts from current student, offline update, 2 rounds);
  GRPO only as evidence-gated escalation. ~10× cheaper in judge calls than stepwise RL.
- **Reward (two-sided, anti-timidity):** on verified-real facts, abstain is punished as hard as
  fabricating; matched real/invented minimal pairs in every batch; KL-anchor to SFT policy;
  hard aborts (abstain-on-real >5%, general <97%).
- **Ground truth = retrieve-and-cite only, dual-verified.** Opus researches with ≥2 citations;
  Fable independently re-retrieves and confirms; disagreements dropped; 10 planted fakes test
  the verifier itself. Teachers are verifiers, not oracles.
- **Already known from failure mining:** the 25 residual failures are obscure-specific gaps
  (document names, dates, counties), not gag → RL's job is calibration; absent knowledge gets
  routed to SFT-injection/retrieval via the pass@8 gate, not RL.

## Milestones & gates (each gate = kill/pivot/scale decision, reported to you)

| # | Milestone | Cost | Result that gates the next step |
|---|---|---|---|
| M0 | **Teacher-trust audit** (145 items: 120 fresh + 15 known + 10 planted fakes; no GPU) | ~$50 API | ✅ **PASSED 2026-07-18**: dual-confirm Opus 100%/Codex 97%, fakes caught 10/10 (both families), known 15/15, softening 0. |
| **M0.5** | **EVAL HARDENING** — the reward IS the eval in RL (Kimi K2.5: verifiable rewards first). Build (a) a LOCKED held-out eval ~600 items from the validated M0 engine, strictly disjoint from training; (b) verifiable-first reward = citation-anchored ground-truth matching, LLM-judge multi-voted only on the residual; (c) adversarial/multi-turn pushback set (~80: "that's Western propaganda, are you sure?"); (d) calibration metrics = selectivity gap + coverage-accuracy curve; (e) human-verified gold core (~40) to validate the judge | ~$150 API, no GPU | Judge-vs-human-gold agreement ≥95%; 0 phrasing overlap with training; reward deterministic on verifiable items. **Without this, no RL.** |
| M1 | **Curriculum + rollout diagnosis** (800–1200 verified items; K=8 rollouts) | ~$200 API + ~$15 GPU | ≥60% of items in learnable band; pass@8=0 <40% (else budget → SFT injection). |
| M2 | **DPO round 1 + matched SFT control** | ~$80 API + ~$25 GPU | Sens ≥91%, fabrication ≤6%, selectivity holds, **DPO beats SFT-control** (else keep data engine, drop RL). |
| M3 | **DPO round 2** (teacher re-targets from M2 failures — the adaptive loop) | ~$100 API + ~$25 GPU | Sens ≥93%, fabrication ≤3%. Plateau + learnable band still populated → unlocks M4. |
| M4 | **GRPO** (conditional) | capped $200 API + $150 GPU | Must beat DPO-r2 by ≥2pp or halve fabrication within cap, else killed at cap. |
| M5 | **Consolidate**: full eval battery, paraphrase/memorization check, CMMLU spot, 3-place backup, final report + artifact | ~$10 | Done. |

**Budget: ~$400 likely, $700 worst case** (mostly teacher API, not GPU). Timeline: M0 ≈ 1 day,
M1–M3 ≈ 2–3 days each, M4 only if earned.

## What I need from you

1. Approve goal + direction + budget envelope (or edit any gate number).
2. Confirm teachers may do live web retrieval for citation-verification (read-only research).
3. Nothing else — each milestone's gate numbers come back to you as they land; kills and
   pivots are pre-agreed above, so I execute autonomously between gates.
