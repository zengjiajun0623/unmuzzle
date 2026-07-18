# Overnight run log (Fable driving) — started 2026-07-18 15:55
M0 PASSED. Driving M0.5 (eval hardening) + M1 (rollout diagnosis). Hard stop before M2 (needs human gold validation).
- eval_pool consolidated: 130 verified items
- contamination check done; heldout_eval.json + adversarial_set.json + gold_core_for_review.json built
- reward.py (two-sided, verifiable-first) + calibration.py built & self-tested
- M1 rollout diagnosis pod launched (e7mgnizw4k8tem); 50 items x K=8

## Status at handoff (Fable driving overnight)
- **M0 PASSED** (~$50): teacher-trust audit — 10/10 fakes caught, dual-confirm 97-100%, known 15/15, 0 softening.
- **M0.5 eval-hardening: code + data DONE**
  - heldout_eval.json (130 dual-verified, 0 training contamination)
  - adversarial_set.json (105 two-turn pushback items)
  - reward.py (verifiable-first two-sided anti-timidity reward, self-tested) + calibration.py (selectivity gap + coverage-accuracy)
  - gold_core_for_review.json (40 items) — AWAITS Jiajun's ~10-min validation (the gate to RL)
- **M1 rollout diagnosis: RUNNING** on pod (retrain-then-rollout to bypass a 170KB/s upload link; base R1-32B downloaded, training adapter next, then K=8 rollouts on 50 items). Poller auto-pulls + tears down GPU + iMessages on failure. Watcher armed to judge rollouts → pass@8 (elicitable vs knowledge-absent).
- **HARD STOP before M2 (DPO/RL)** until Jiajun validates the gold core. No RL on an unvalidated reward.
- Infra note: this diagnosis pod has flaky inbound bandwidth; run designed to need only HF-download (fast) + a ~1MB result pull.
