# Step 05 Findings: Renderable Foreground And 3D-Like Feasibility

## Summary

Step 05 completed successfully as a Tier3 feasibility experiment.

This route is still **not promotable**, but it is the strongest Tier3 signal so far.

Unlike Step 02, Step 03, and Step 04, the renderable-foreground prototype improves:

- silhouette temporal consistency
- occlusion temporal consistency
- real-video seam and background stability
- continuous alpha metrics on the reviewed holdout split

At the same time, it still fails to promote because:

- global reviewed `boundary_f1` remains strongly negative
- occluded-boundary reviewed quality does not turn positive

The frozen Step 05 conclusion is:

- `best-of-three = Round 2`
- bucket:
  - `interesting_but_unproven`


## Deliverables

New files introduced in this step:

- `wan/utils/renderable_foreground_proto.py`
- `scripts/eval/check_renderable_foreground_proto.py`
- `scripts/eval/infer_renderable_foreground_proto.py`
- `scripts/eval/compute_renderable_foreground_metrics.py`
- `scripts/eval/evaluate_optimization9_renderable_benchmark.py`
- `scripts/eval/run_optimization9_renderable_benchmark.py`

Primary output directories:

- `runs/optimization9_step05_round1/`
- `runs/optimization9_step05_round2/`
- `runs/optimization9_step05_round3/`

Key outputs:

- `runs/optimization9_step05_round1/gate_result.json`
- `runs/optimization9_step05_round2/gate_result.json`
- `runs/optimization9_step05_round3/gate_result.json`


## Why This Step Was Needed

By the time Step 05 started, Tier3 had already established:

- bridge-style ROI routes can change threshold behavior, but not matte quality
- layer decomposition can create a new occlusion-oriented mode
- paired RGBA foreground can create a semantic reconstruction mode

The remaining question was whether a more explicit renderable-foreground state could produce a different class of gain:

- not just a different edge patch
- not just a better paired foreground solve
- but a more stable temporal foreground representation

That is the specific value Step 05 was designed to test.


## Round 1

Round 1 implemented the minimum renderable-foreground state:

- renderable foreground RGB
- renderable alpha
- depth-like pseudo-field from foreground occupancy
- temporal reprojection using mask-centroid motion
- explicit renderable silhouette band

Round 1 also required normal integration cleanup:

- `write_rgb_artifact` and `write_person_mask_artifact` are keyword-only
- `boundary_band` had to preserve contract semantics when replaced by the renderable silhouette artifact

Once those issues were fixed, the route ran cleanly end to end.

### Round 1 result

- reviewed:
  - `boundary_f1_gain_pct = -20.13%`
  - `alpha_mae_reduction_pct = +3.01%`
  - `trimap_error_reduction_pct = +3.87%`
  - `occluded_boundary_f1_gain_pct = 0.0%`
  - `hand_boundary_f1_gain_pct = 0.0%`
- smoke:
  - `seam_improvement_pct = +25.69%`
  - `background_improvement_pct = +34.05%`
- renderable:
  - `silhouette_temporal_improvement_pct = +32.70%`
  - `motion_temporal_improvement_pct = +6.75%`
  - `occlusion_temporal_improvement_pct = +12.70%`

Interpretation:

- this was the first Tier3 route that improved continuous alpha metrics and smoke behavior together
- but global reviewed boundary quality remained clearly worse than baseline

Bucket:

- `interesting_but_unproven`


## Round 2

Round 2 increased temporal blending, silhouette gain, and depth contribution.

The goal was to see whether a stronger renderable state would:

- preserve the positive smoke behavior
- and further improve temporal consistency

### Round 2 result

- reviewed:
  - `boundary_f1_gain_pct = -20.87%`
  - `alpha_mae_reduction_pct = +4.00%`
  - `trimap_error_reduction_pct = +5.68%`
  - `occluded_boundary_f1_gain_pct = 0.0%`
  - `hand_boundary_f1_gain_pct = 0.0%`
- smoke:
  - `seam_improvement_pct = +30.94%`
  - `background_improvement_pct = +39.51%`
- renderable:
  - `silhouette_temporal_improvement_pct = +33.24%`
  - `motion_temporal_improvement_pct = +12.50%`
  - `occlusion_temporal_improvement_pct = +20.76%`

Interpretation:

- Round 2 is the strongest version of the route
- it improves every temporal/renderable metric relative to Round 1
- it also improves reviewed continuous alpha metrics more than Round 1
- but global reviewed boundary quality is still too negative for promotion

Bucket:

- `interesting_but_unproven`


## Round 3

Round 3 reduced the temporal strength and tried to rebalance the route toward safety.

The purpose was to check whether some of Round 2's representation gain could be preserved with a less aggressive temporal state.

### Round 3 result

- reviewed:
  - `boundary_f1_gain_pct = -20.39%`
  - `alpha_mae_reduction_pct = +3.49%`
  - `trimap_error_reduction_pct = +4.66%`
  - `occluded_boundary_f1_gain_pct = 0.0%`
  - `hand_boundary_f1_gain_pct = 0.0%`
- smoke:
  - `seam_improvement_pct = +27.22%`
  - `background_improvement_pct = +35.84%`
- renderable:
  - `silhouette_temporal_improvement_pct = +33.02%`
  - `motion_temporal_improvement_pct = +8.46%`
  - `occlusion_temporal_improvement_pct = +15.10%`

Interpretation:

- Round 3 remains clearly positive in the renderable/temporal sense
- but it is weaker than Round 2 on almost every meaningful new metric
- so it does not deserve to be frozen

Bucket:

- `interesting_but_unproven`


## Why The Route Did Not Promote

The route still fails promotion because the reviewed benchmark says the same hard thing in every round:

- global `boundary_f1` remains strongly negative
- `occluded_boundary_f1` stays flat

This means the route is not yet a mainline replacement candidate.

Even though:

- `alpha_mae`
- `trimap_error`
- seam
- background fluctuation
- silhouette temporal consistency
- occlusion temporal consistency

all move in the right direction, the reviewed foreground boundary itself is still not strong enough.


## Why This Route Is The Strongest Tier3 Signal So Far

This route is more interesting than earlier Tier3 routes for one reason:

it is the first route that improves **continuous alpha metrics and real-video stability metrics together**.

Step 02:

- threshold gain only

Step 03:

- occlusion-specific temporal or gradient mode

Step 04:

- semantic reconstruction mode

Step 05:

- temporal renderable-state mode
- plus continuous alpha improvement
- plus strong smoke improvement

That makes it the most structurally promising Tier3 route so far, even though it still fails promotion.


## Frozen Best-Of-Three

`Round 2` is frozen as `best-of-three`.

Reason:

- it has the strongest:
  - `silhouette_temporal_improvement_pct`
  - `motion_temporal_improvement_pct`
  - `occlusion_temporal_improvement_pct`
- it also has the strongest:
  - `alpha_mae_reduction_pct`
  - `trimap_error_reduction_pct`
  - seam and background improvement

Frozen bucket:

- `interesting_but_unproven`


## What Step 05 Proved

Step 05 proved five useful things:

1. a minimal renderable-foreground state can be prototyped inside the current repo
2. explicit temporal reprojection of the foreground state creates a genuinely different positive mode
3. that positive mode is stronger than the positive modes from Step 02 through Step 04
4. continuous alpha metrics can improve together with real-video stability under this route
5. the reviewed global boundary bottleneck still prevents promotion


## Recommended Next Move

Proceed to `optimization9 / Step 06`.

Reason:

- Step 05 is the strongest Tier3 branch so far
- but it is still not promotable
- Tier3 now needs a portfolio-level decision:
  - whether Step 05 alone is strong enough to justify further Tier3 investment
  - or whether the whole Tier3 branch should stop here with a clear ranking of the routes
