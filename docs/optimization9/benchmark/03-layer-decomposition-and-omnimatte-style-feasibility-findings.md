# Step 03 Findings: Layer Decomposition And Omnimatte-Style Feasibility

## Summary

Step 03 completed successfully as a Tier3 feasibility experiment.

The route did **not** produce a promotable result, but unlike Step 02 it did show a genuinely different positive mode:

- better occlusion temporal consistency
- better occlusion-boundary gradient behavior
- improved background stability in the more balanced rounds

That is important because Tier3 is not only about finding immediate winners. It is also about distinguishing:

- routes that simply restate old failure patterns
- routes that still fail promotion, but expose a structurally different signal worth remembering

The frozen Step 03 conclusion is:

- `best-of-three = Round 3`
- bucket:
  - `interesting_but_unproven`


## Deliverables

New files introduced in this step:

- `wan/utils/layer_decomposition_proto.py`
- `scripts/eval/check_layer_decomposition_proto.py`
- `scripts/eval/infer_layer_decomposition_proto.py`
- `scripts/eval/compute_layer_decomposition_metrics.py`
- `scripts/eval/run_optimization9_layer_benchmark.py`
- `scripts/eval/evaluate_optimization9_layer_benchmark.py`

Primary output directories:

- `runs/optimization9_step03_round1_v2/`
- `runs/optimization9_step03_round2/`
- `runs/optimization9_step03_round3/`

Key outputs:

- `runs/optimization9_step03_round1_v2/gate_result_v2.json`
- `runs/optimization9_step03_round2/gate_result_v2.json`
- `runs/optimization9_step03_round3/gate_result_v2.json`


## Why This Step Was Needed

Step 02 already falsified the lightest Tier3 bridge route.

That meant the next worthwhile Tier3 experiment had to change the structure of the problem more aggressively. A layer-decomposition prototype fits that requirement because it explicitly separates:

- foreground
- occlusion or unresolved content
- background

The key question was not:

- can this route win overall reviewed metrics immediately

The key question was:

- does this route expose a new positive mode on occlusion boundaries and background recovery that the older routes never showed


## Round 1

Round 1 implemented the minimum viable three-layer prototype:

- `foreground_alpha`
- `occlusion_alpha`
- `background`
- signed residual occlusion RGB
- explicit layer ROI mask

This round also exposed two integration issues that had to be fixed before the route could be judged fairly:

1. the inferred bundle was still pointing `occlusion_band` at the old artifact instead of the layer-derived one
2. the benchmark runner was using a fallback composite path and not the explicit `layer_composite_preview`

After those fixes, the first valid run became:

- `runs/optimization9_step03_round1_v2/`

### Round 1 result

- reviewed:
  - `occluded_boundary_f1_gain_pct = -5.25%`
  - `boundary_f1_gain_pct = -21.73%`
  - `alpha_mae_reduction_pct = -56.60%`
  - `trimap_error_reduction_pct = -16.41%`
- smoke:
  - `seam_improvement_pct = -14.81%`
  - `background_improvement_pct = +7.70%`
- layer:
  - `occlusion_temporal_improvement_pct = +9.42%`
  - `occlusion_gradient_gain_pct = +14.64%`
  - `occlusion_contrast_gain_pct = -14.70%`

Interpretation:

- the route clearly showed a new occlusion-oriented positive mode
- but overall reviewed alpha quality was still much worse than baseline
- and seam deterioration was already close to the practical tolerance boundary

Bucket after reevaluation:

- `interesting_but_unproven`


## Round 2

Round 2 pushed the occlusion layer harder by increasing:

- occlusion strength
- unresolved-layer contribution
- the importance of layer residual color

This did increase occlusion-specific gains further, but it also made downstream tradeoffs worse.

### Round 2 result

- reviewed:
  - `occluded_boundary_f1_gain_pct = -5.25%`
  - `boundary_f1_gain_pct = -9.86%`
  - `alpha_mae_reduction_pct = -25.51%`
  - `trimap_error_reduction_pct = -6.48%`
- smoke:
  - `seam_improvement_pct = -25.90%`
  - `background_improvement_pct = -4.98%`
- layer:
  - `occlusion_temporal_improvement_pct = +11.08%`
  - `occlusion_gradient_gain_pct = +28.30%`
  - `occlusion_contrast_gain_pct = -12.66%`

Interpretation:

- this was the strongest occlusion-specific round
- but the smoke regression crossed the line
- the route became too destructive to keep as the frozen version

Bucket:

- `reject`

Failure patterns:

- `smoke_regression`
- `occlusion_only_tradeoff`


## Round 3

Round 3 pulled the route back from Round 2's over-commitment and tried to preserve the useful occlusion signal while reducing downstream damage.

The key changes were:

- lower alpha mix
- higher residual mix
- more balanced occlusion strength

### Round 3 result

- reviewed:
  - `occluded_boundary_f1_gain_pct = -5.25%`
  - `boundary_f1_gain_pct = -9.40%`
  - `alpha_mae_reduction_pct = -7.76%`
  - `trimap_error_reduction_pct = -1.83%`
- smoke:
  - `seam_improvement_pct = -14.11%`
  - `background_improvement_pct = +5.78%`
- layer:
  - `occlusion_temporal_improvement_pct = +9.23%`
  - `occlusion_gradient_gain_pct = +15.47%`
  - `occlusion_contrast_gain_pct = -14.12%`

Interpretation:

- Round 3 preserved most of the new occlusion-specific value
- while avoiding the severe smoke breakage of Round 2
- reviewed alpha quality was still negative relative to baseline
- but the regression was much smaller than Round 1

Bucket after reevaluation:

- `interesting_but_unproven`


## Why The Route Did Not Promote

This route still failed promotion for straightforward reasons:

1. reviewed alpha quality never turned positive
   - `alpha_mae`
   - `trimap_error`
   - global `boundary_f1`
   all stayed negative
2. occluded-boundary reviewed quality also stayed negative
   - the route improved occlusion-specific temporal and gradient behavior
   - but that did not translate into a reviewed occluded-boundary win
3. smoke never became clearly positive overall
   - Round 2 was outright unsafe
   - Round 1 and Round 3 were survivable, but still not good enough for promotion

So this is not a hidden winner. It remains a non-promotable route.


## Why The Route Is Still Interesting

Step 03 is still materially different from Step 02.

Step 02 behaved like:

- a threshold corrector
- with no new semantic or occlusion behavior

Step 03 behaves like:

- an occlusion-aware structure changer
- with positive temporal and gradient movement in occlusion-sensitive regions

That is exactly why the frozen bucket is:

- `interesting_but_unproven`

This does **not** mean “continue scaling it immediately”.

It means:

- this route exposed a new error mode
- the new mode is not yet strong enough to promote
- but it is qualitatively different enough to remember


## Frozen Best-Of-Three

`Round 3` is frozen as `best-of-three`.

Reason:

- it keeps the new occlusion-specific positive mode
- it avoids the catastrophic smoke regression of Round 2
- it greatly reduces the reviewed alpha regression seen in Round 1

Frozen bucket:

- `interesting_but_unproven`


## What Step 03 Proved

Step 03 proved four useful things:

1. a three-layer prototype can be integrated into the current repo and preprocess-bundle format
2. layer decomposition does expose a genuinely different positive mode from the bridge route
3. that positive mode is currently limited to occlusion temporal and gradient behavior
4. the route still does not justify escalation into a mainline replacement path


## Recommended Next Move

Proceed to `optimization9 / Step 04` only if we still want to test a route that changes the representation even more aggressively.

That makes sense because:

- Step 03 is not promotable
- but it is still more structurally interesting than Step 02
- so the next Tier3 route should be more radical, not a minor variation of the current three-layer prototype
