# Step 03 Findings: Trainable Alpha And Matte Completion Baseline

## Summary

Step 03 completed its three-round implementation and evaluation loop.

The trainable alpha / matte completion baseline is now real and reproducible:

- a patch-level trainable alpha refiner exists
- a train -> infer -> reviewed-benchmark -> smoke-eval path exists
- the route can be compared against the frozen non-train baseline on `reviewed_edge_v3_candidate`

However, the route did **not** pass the Step 03 promotion gate.

It consistently achieved only one of the three required reviewed improvements:

- `boundary_f1` improved strongly on `holdout_eval`
- `alpha_mae` did not improve enough
- `trimap_error` did not improve enough

Therefore the correct conclusion is:

> The trainable alpha baseline is a valid research branch, but it is not yet strong enough to replace the non-train baseline.


## Deliverables

New files introduced in this step:

- `wan/utils/trainable_alpha_model.py`
- `scripts/eval/check_trainable_alpha_model.py`
- `scripts/eval/train_trainable_alpha_model.py`
- `scripts/eval/infer_trainable_alpha_model.py`
- `scripts/eval/evaluate_trainable_alpha_model.py`

Primary datasets and run directories:

- ROI dataset:
  - `runs/optimization8_step02_round3/roi_dataset_v2`
- Reviewed benchmark:
  - `runs/optimization8_step01_round3/reviewed_edge_benchmark_v3_candidate`
- Stable baseline preprocess bundle:
  - `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`

Round outputs:

- Round 1:
  - `runs/optimization8_step03_round1/`
- Round 2:
  - `runs/optimization8_step03_round2/`
- Round 3:
  - `runs/optimization8_step03_round3/`

Round 1 smoke:

- `runs/optimization8_step03_round1_smoke/`

Round 3 smoke:

- `runs/optimization8_step03_round3_smoke/`


## Round 1

### Configuration

- `width = 24`
- `residual_scale = 0.20`
- `epochs = 30`
- `lr = 1e-3`
- weighted alpha loss + BCE

### Result

Best validation epoch:

- `best_epoch = 1`

Validation metrics:

- `boundary_f1_mean = 0.7770`
- `alpha_mae_mean = 0.01145`
- `trimap_error_mean = 0.03668`

Reviewed `holdout_eval` after full-bundle inference:

- `boundary_f1_mean = 0.65435`
- `alpha_mae_mean = 0.0172527`
- `trimap_error_mean = 0.1204088`

Relative to the frozen non-train baseline:

- `holdout_boundary_f1_gain_pct = +81.17%`
- `holdout_alpha_mae_reduction_pct = +0.09%`
- `holdout_trimap_error_reduction_pct = +0.10%`

Smoke result:

- `seam_degradation_pct = +6.46%` improvement in the “lower is better” metric
- `background_fluctuation_improvement_pct = +2.81%`

Interpretation:

- Round 1 clearly learned to push more edge pixels across the best threshold.
- But it almost did not improve continuous alpha quality.
- It was stable enough to run through generate, but it did not pass the reviewed gate.


## Round 2

### Change made

Round 2 changed the training loss to be more explicit about:

- stronger supervision on unknown / semi-transparent / boundary regions
- stronger preservation outside uncertain regions

Key new terms:

- `unknown_weight`
- `preserve_weight`

### Result

Best validation epoch:

- `best_epoch = 1`

Reviewed `holdout_eval`:

- `boundary_f1_mean = 0.65435`
- `alpha_mae_mean = 0.0172450`
- `trimap_error_mean = 0.1203462`

Relative improvements:

- `holdout_boundary_f1_gain_pct = +81.17%`
- `holdout_alpha_mae_reduction_pct = +0.14%`
- `holdout_trimap_error_reduction_pct = +0.15%`

Interpretation:

- Round 2 slightly improved `alpha_mae` and `trimap_error`.
- The gain was real but tiny.
- It still passed only one of the three reviewed criteria.


## Round 3

### Change made

Round 3 intentionally tested a more aggressive route:

- wider model
- larger `residual_scale`
- stronger unknown-region emphasis
- weaker preserve term
- lower BCE pressure

### Result

Best validation epoch:

- `best_epoch = 1`

Reviewed `holdout_eval`:

- `boundary_f1_mean = 0.65432`
- `alpha_mae_mean = 0.0172242`
- `trimap_error_mean = 0.1202072`

Relative improvements:

- `holdout_boundary_f1_gain_pct = +81.17%`
- `holdout_alpha_mae_reduction_pct = +0.26%`
- `holdout_trimap_error_reduction_pct = +0.27%`

Smoke result:

- `seam_degradation_pct = +1.91%`
- `background_fluctuation_improvement_pct = -50.26%`

Interpretation:

- Round 3 achieved the best continuous alpha metrics of the three rounds.
- But the gain was still far below the `10%` target.
- Worse, the smoke run materially regressed background stability.


## Best-Of-Three Decision

The best-of-three freeze is **Round 1**.

Reason:

- Round 2 and Round 3 only produced tiny additional `alpha_mae / trimap_error` gains.
- Those gains were far below the promotion threshold and therefore not strategically meaningful.
- Round 3 introduced a large real-video background regression.
- Round 1 remained the cleanest “research baseline”:
  - strongest threshold-level boundary gain
  - stable minimal smoke behavior
  - simplest training configuration


## Final Gate Decision

Step 03 gate rule:

- `boundary_f1` gain `>= 8%`
- `alpha_mae` reduction `>= 10%`
- `trimap_error` reduction `>= 10%`
- need at least `2 / 3`

Final result:

- `boundary_f1`: PASS
- `alpha_mae`: FAIL
- `trimap_error`: FAIL

Therefore:

- `gate_passed = false`
- `pass_count = 1`


## What We Learned

1. The trainable patch refiner can learn a boundary-sensitive correction that improves the chosen hard-mask threshold behavior on `holdout_eval`.
2. That correction does **not** translate into a meaningful continuous alpha improvement.
3. The route is therefore at risk of learning “better threshold crossing” instead of “better matte quality.”
4. More aggressive residual settings do not fix the core problem; they only make smoke stability worse.


## Production Decision

Do **not** promote this trainable alpha baseline into the main preprocess path.

Keep it only as:

- a research baseline
- a dataset / trainer / infer scaffold
- a reference for future stronger trainable routes built on better GT and more specific tasks


## Recommended Next Move

Proceed to `optimization8 / Step 04`, but only after acknowledging the exact limitation revealed here:

> future trainable routes should not optimize only for generic alpha refinement;
> they should target more specific tasks such as semantic boundary experts or compositing-aware edge correction.
