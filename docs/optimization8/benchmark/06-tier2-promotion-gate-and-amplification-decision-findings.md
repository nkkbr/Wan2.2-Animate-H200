# Step 06 Findings: Tier-2 Promotion Gate And Amplification Decision

## Summary

Step 06 completed successfully.

This step did **not** attempt to invent another Tier2 variant. Its job was to:

- normalize the evidence from Steps 03, 04, and 05
- compare those routes against the frozen non-train reference
- decide whether any Tier2 route is worth keeping as a real amplifier

The final answer is:

> No Tier2 route should be promoted as an amplifier.

What remains useful is only:

- the frozen non-train reference baseline
- the Step 03 trainable alpha route as research scaffolding

What should **not** be amplified further:

- semantic experts
- compositing-aware loss variants
- any nonexistent combined Tier2 full-stack route


## Deliverables

New files introduced in this step:

- `docs/optimization8/benchmark/tier2_manifest.json`
- `scripts/eval/run_optimization8_tier2_benchmark.py`
- `scripts/eval/evaluate_optimization8_tier2_benchmark.py`
- `scripts/eval/check_optimization8_tier2_gate.py`

Primary output directory:

- `runs/optimization8_step06_tier2_gate/`

Key outputs:

- `runs/optimization8_step06_tier2_gate/summary.json`
- `runs/optimization8_step06_tier2_gate/gate_result.json`


## Candidate Set

Tier2 decision used the following representative set:

- `non_trainable_best_from_optimization7`
- `trainable_alpha_best`
- `semantic_experts_best`
- `best_loss_stack_variant`
- `tier2_full_best`

Candidate semantics:

- `non_trainable_best_from_optimization7`
  - practical reference baseline
  - no external-alpha winner existed in `optimization7`
  - therefore the frozen reviewed-v3 non-train baseline remains the real reference
- `trainable_alpha_best`
  - Step 03 best-of-three = Round 1
- `semantic_experts_best`
  - Step 04 best-of-three = Round 1
- `best_loss_stack_variant`
  - Step 05 best-of-three = Round 2
- `tier2_full_best`
  - unavailable
  - no combined Tier2 route was strong enough to freeze


## Evidence Used

### Reference reviewed baseline

- `runs/optimization8_step01_round3/baseline_metrics_v3.json`

Reference holdout metrics:

- `boundary_f1_mean = 0.36117`
- `alpha_mae_mean = 0.01727`
- `trimap_error_mean = 0.12053`

### Tier2 baseline

- `runs/optimization8_step03_round1/candidate_metrics_v3.json`
- `runs/optimization8_step03_round1_smoke/metrics/candidate_replacement_metrics.json`

Tier2 baseline holdout metrics:

- `boundary_f1_mean = 0.65435`
- `alpha_mae_mean = 0.01725`
- `trimap_error_mean = 0.12041`

Relative to the non-train reference:

- `boundary_f1_gain_pct = +81.17%`
- `alpha_mae_reduction_pct = +0.09%`
- `trimap_error_reduction_pct = +0.10%`

Interpretation:

- strong threshold-level boundary improvement
- almost no continuous alpha / trimap improvement
- worth keeping as a research baseline, but not promotable

### Semantic experts best

- `runs/optimization8_step04_round1/candidate_metrics_v3.json`
- `runs/optimization8_step04_round1/gate_result_with_smoke.json`

Relative to the non-train reference:

- `boundary_f1_gain_pct = +81.19%`
- `alpha_mae_reduction_pct = -0.12%`
- `trimap_error_reduction_pct = -0.16%`

Relative to the Tier2 baseline:

- `boundary_f1_gain_pct = +0.009%`
- `alpha_mae_reduction_pct = -0.214%`
- `trimap_error_reduction_pct = -0.255%`
- `seam_improvement_pct = -7.61%`
- `background_fluctuation_improvement_pct = -10.13%`

Interpretation:

- no meaningful amplification
- smoke regression is already visible even for the least invasive expert route

### Best loss-stack variant

- `runs/optimization8_step05_round2/candidate_metrics_v3.json`
- `runs/optimization8_step05_round2/gate_result_with_smoke.json`

Relative to the non-train reference:

- `boundary_f1_gain_pct = +81.19%`
- `alpha_mae_reduction_pct = +0.34%`
- `trimap_error_reduction_pct = +0.36%`

Relative to the Tier2 baseline:

- `boundary_f1_gain_pct = +0.006%`
- `alpha_mae_reduction_pct = +0.249%`
- `trimap_error_reduction_pct = +0.262%`
- `seam_improvement_pct = -50.97%`
- `background_fluctuation_improvement_pct = -51.83%`

Interpretation:

- reviewed gain versus Step 03 exists but is tiny
- real-video smoke degrades sharply
- not amplifiable


## Gate Logic

Promotion gate for a Tier2 amplifier required:

- `boundary_f1` gain vs non-train reference `>= 8%`
- `alpha_mae` reduction vs non-train reference `>= 10%`
- `trimap_error` reduction vs non-train reference `>= 10%`
- at least one positive smoke signal relative to `trainable_alpha_best`

This gate is intentionally strict because the point of Tier2 is not “maybe slightly better,” but:

> worth the added data, training, maintenance, and inference complexity


## Final Decision

### non_trainable_best_from_optimization7

- decision: `reference_baseline`

Reason:

- frozen reviewed-v3 non-train reference
- remains the only reliable non-train anchor

### trainable_alpha_best

- decision: `keep_experimental`

Reason:

- it is the only Tier2 route that produced a large reviewed `boundary_f1` jump
- but it still failed the stronger continuous-alpha gate
- it is useful only as scaffolding for future training research, not as an amplifier

### semantic_experts_best

- decision: `reject_for_now`

Reason:

- no reviewed amplification
- negative smoke deltas
- more maintenance complexity than the gain justifies

### best_loss_stack_variant

- decision: `reject_for_now`

Reason:

- reviewed gains are tiny
- smoke regression is severe
- extra training complexity is not justified

### tier2_full_best

- decision: `unavailable`

Reason:

- no combined Tier2 route was strong enough to freeze


## Final Amplification Decision

Top-level result:

- `promote_tier2_as_amplifier = false`
- `amplification_decision = stop_amplification_keep_research_only`

This is the correct Tier2 stop rule.

Step 06 therefore succeeded by making the stop decision explicit.


## What We Learned

1. Tier2 can produce research signals, but not a promotable amplifier.
2. The Step 03 trainable alpha route is still the only Tier2 branch worth keeping alive, and only as a research scaffold.
3. Semantic experts and compositing-aware loss variants did not survive end-to-end evaluation.
4. The practical implication is clear:
   - do not expand Tier2 further
   - do not combine more variants
   - redirect effort toward stronger data routes or higher-risk breakthrough directions


## Recommended Next Move

After Step 06, the correct next decision is:

- freeze Tier2
- keep only `trainable_alpha_best` as experimental scaffolding
- stop amplifying Tier2 variants
- shift main effort to Tier3-style breakthrough work or a new route redesign
