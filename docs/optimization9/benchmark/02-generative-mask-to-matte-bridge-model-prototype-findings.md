# Step 02 Findings: Generative Mask-To-Matte Bridge Model Prototype

## Summary

Step 02 completed successfully from an engineering perspective, but failed its promotion objective.

The `mask-to-matte / matte bridge` route is now a real prototype, not just a design note:

- a trainable bridge model exists
- ROI training and inference are wired into the current preprocess bundle format
- reviewed benchmark evaluation works end to end
- proxy smoke and ROI quality evaluation work end to end

However, after three rounds, the route still fell into the same failure family that earlier Tier2 routes already exposed:

- strong threshold-level `boundary_f1` gain
- no continuous-alpha improvement
- no hair or semi-transparent breakthrough
- no positive ROI quality shift

So the frozen Step 02 conclusion is:

- `best-of-three = Round 1`
- bucket:
  - `reject`


## Deliverables

New files introduced in this step:

- `wan/utils/matte_bridge_model.py`
- `scripts/eval/check_matte_bridge_model.py`
- `scripts/eval/train_matte_bridge_model.py`
- `scripts/eval/infer_matte_bridge_model.py`
- `scripts/eval/run_optimization9_bridge_benchmark.py`
- `scripts/eval/evaluate_optimization9_bridge_benchmark.py`

Primary output directories:

- `runs/optimization9_step02_round1_v7/`
- `runs/optimization9_step02_round2/`
- `runs/optimization9_step02_round3/`

Key outputs:

- `runs/optimization9_step02_round1_v7/gate_result_v2.json`
- `runs/optimization9_step02_round2/gate_result.json`
- `runs/optimization9_step02_round3/gate_result.json`


## Why This Step Was Needed

Tier3 needed a route that was:

- materially different from deterministic edge refinement
- lighter weight than full layer-decomposition or RGBA video generation
- still close enough to the current pipeline to prototype quickly

The bridge-model idea fits that slot well:

- take coarse alpha / trimap / uncertainty
- add foreground/background evidence
- learn a better matte only where the current system is weak

If this route showed a genuinely different positive mode, it would justify more investment.


## Round 1

Round 1 implemented the minimum viable bridge prototype:

- ROI-only U-Net style bridge model
- inputs:
  - foreground patch
  - background patch
  - soft alpha
  - trimap unknown
  - boundary ROI mask
  - person mask
  - boundary band
  - uncertainty
- outputs:
  - predicted matte
  - bridge gate
  - final blended alpha

Round 1 also exposed an evaluator bug:

- ROI quality JSON contained raw `before/after` statistics
- the evaluator expected precomputed percent deltas
- so ROI gains were incorrectly read as `0.0`

That bug was fixed before comparing later rounds.

### Round 1 result

The route already showed the central failure mode:

- `boundary_f1_gain_pct = +81.33%`
- `alpha_mae_reduction_pct = -0.87%`
- `trimap_error_reduction_pct = -1.25%`
- `hair_edge_quality_gain_pct = 0.0%`
- `semi_transparent_quality_gain_pct = 0.0%`
- ROI:
  - `gradient = -8.64%`
  - `contrast = +6.46%`
  - `halo reduction = -12.60%`

Interpretation:

- the model learned to cross the binary boundary threshold better
- but it did not learn a better continuous matte
- and it did not create any new semantic edge strength


## Round 2

Round 2 tried to push the route toward a more useful mode by increasing:

- compositing-aware supervision
- gradient preservation
- contrast preservation
- hair-specific weighting
- gate strength

This did not help.

### Round 2 result

- `boundary_f1_gain_pct = +81.14%`
- `alpha_mae_reduction_pct = -3.00%`
- `trimap_error_reduction_pct = -3.06%`
- `hair_edge_quality_gain_pct = 0.0%`
- `semi_transparent_quality_gain_pct = 0.0%`
- ROI:
  - `gradient = -7.94%`
  - `contrast = -1.11%`
  - `halo reduction = -4.87%`

Interpretation:

- stronger losses made the bridge route even more committed to its old pattern
- reviewed threshold behavior stayed high
- continuous-alpha quality got worse
- ROI quality remained negative


## Round 3

Round 3 made a more structural correction:

- gate target changed from `binary threshold` to `continuous error`
- `boundary_weight` and `trimap_weight` were reduced
- compositing/gradient/contrast losses were increased
- gate strength was reduced slightly

This was the best available attempt to force the route away from the old binary-mask behavior.

### Round 3 result

- `boundary_f1_gain_pct = +81.14%`
- `alpha_mae_reduction_pct = -3.04%`
- `trimap_error_reduction_pct = -3.14%`
- `hair_edge_quality_gain_pct = 0.0%`
- `semi_transparent_quality_gain_pct = 0.0%`
- ROI:
  - `gradient = -9.01%`
  - `contrast = -1.07%`
  - `halo reduction = -5.07%`

Interpretation:

- even with a softer gate target, the route still converged to the same family of behavior
- there was no semantic breakthrough
- there was no continuous-alpha recovery


## Why The Route Failed

The key issue is not that the bridge model learned nothing.

It clearly learned something:

- validation `boundary_f1` stayed high
- proxy smoke did not catastrophically regress
- background fluctuation even improved modestly

But it learned the wrong thing for this problem.

The bridge route consistently behaves like:

- a strong ROI threshold corrector
- not a true matte improver

That is exactly why the frozen failure patterns are:

- `threshold_only_gain`
- `no_semantic_breakthrough`

This matters because Tier3 is not supposed to fund routes that merely restate the old Tier2 tradeoff in a more expensive form.


## Frozen Best-Of-Three

`Round 1` is frozen as `best-of-three`.

Reason:

- it is the least-bad version of the route
- it showed the smallest continuous-alpha regression
- it was the only round with positive ROI contrast movement
- later rounds did not produce a genuinely better mode

But despite being `best-of-three`, it still remains:

- `reject`


## What Step 02 Proved

Step 02 proved something useful even though it failed promotion:

1. the bridge-model route is feasible to prototype inside the current repo
2. the reviewed benchmark is strong enough to expose “binary-only improvement”
3. a bridge model can improve thresholded boundaries without improving matte quality
4. that behavior is not enough to justify escalation

That is exactly the kind of falsification Tier3 is meant to do early.


## Recommended Next Move

Proceed to `optimization9 / Step 03` only if we still want to explore a route that changes the structure of the problem more aggressively.

Reason:

- Step 02 already falsified the lightest Tier3 candidate
- continuing to tune this bridge route would almost certainly repeat the same error mode
- if Tier3 is to remain justified, the next route must be more structurally different, not just a stronger version of the same bridge idea
