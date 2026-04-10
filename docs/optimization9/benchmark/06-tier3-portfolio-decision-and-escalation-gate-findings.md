# Step 06 Findings: Tier-3 Portfolio Decision And Escalation Gate

## Scope

Step 06 converts Optimization9 from a set of isolated high-risk prototypes into a portfolio with an explicit promotion and stop decision.

This step does not introduce a new generation model. It aggregates the frozen best-of-three results from:

- Step 02 bridge route
- Step 03 layer decomposition route
- Step 04 RGBA foreground route
- Step 05 renderable foreground route

and assigns each route one of:

- `reject`
- `interesting_but_unproven`
- `upgrade_candidate`


## Implemented Files

- `docs/optimization9/benchmark/portfolio_manifest.step06.json`
- `scripts/eval/run_optimization9_portfolio_summary.py`
- `scripts/eval/evaluate_optimization9_portfolio.py`
- `scripts/eval/check_optimization9_portfolio_gate.py`


## Round Structure

### Round 1

Goal:

- create a portfolio summary layer
- score the Tier3 routes with a novelty-first view

Result:

- all routes were classified
- no `upgrade_candidate` was found
- `renderable_round2` emerged as the strongest seed
- `bridge_round1` still scored highly because its threshold-level reviewed gain was numerically large, even though it remained structurally weak

Formal output:

- `runs/optimization9_step06_round1/summary.json`
- `runs/optimization9_step06_round1/gate_result.json`

Decision:

- `portfolio_decision = continue_screening_keep_top2`


### Round 2

Goal:

- increase the weight of smoke quality and reviewed continuous-alpha quality
- reduce the chance that a route wins purely on threshold-style gains

Result:

- `renderable_round2` remained the top seed
- `bridge_round1` still had a high numerical score, but remained locked to `reject`
- `rgba_round2` and `layer_round3` remained `interesting_but_unproven`
- still no `upgrade_candidate`

Formal output:

- `runs/optimization9_step06_round2/summary.json`
- `runs/optimization9_step06_round2/gate_result.json`

Decision:

- `portfolio_decision = stop_promotion_keep_top1_seed_only`


### Round 3

Goal:

- freeze the final portfolio rule
- add a stronger reviewed-quality penalty
- make the final escalation outcome explicit

Result:

- no route crossed the `upgrade_candidate` threshold
- `renderable_round2` remained the strongest portfolio seed
- the final Tier3 portfolio decision was frozen

Formal output:

- `runs/optimization9_step06_round3/summary.json`
- `runs/optimization9_step06_round3/gate_result.json`

Frozen decision:

- `portfolio_decision = stop_tier3_promotion_keep_renderable_seed_only`
- `top_seed_route = renderable_round2`
- `upgrade_candidates = []`


## Final Route Classification

### `bridge_round1`

- final decision:
  - `reject`
- reason:
  - very large threshold-level `boundary_f1` gain
  - but continuous alpha metrics remained negative
  - no hair or semi-transparent breakthrough
  - no credible structural gain relative to its cost

### `layer_round3`

- final decision:
  - `interesting_but_unproven`
- reason:
  - clearly different positive mode around occlusion temporal consistency
  - but reviewed alpha and occluded-boundary quality remain negative
  - not strong enough to justify promotion

### `rgba_round2`

- final decision:
  - `interesting_but_unproven`
- reason:
  - showed semantic reconstruction gains and positive hand-boundary behavior
  - but continuous alpha quality remained negative
  - still behaved like a tradeoff route instead of a clear upgrade route

### `renderable_round2`

- final decision:
  - `interesting_but_unproven`
- reason:
  - strongest Tier3 evidence so far
  - continuous alpha metrics turned positive
  - seam, background, silhouette, motion, and occlusion temporal behavior all improved
  - but global reviewed `boundary_f1` remained too negative for promotion


## Most Important Portfolio Metrics

### `renderable_round2`

- reviewed:
  - `boundary_f1_gain_pct = -20.8657`
  - `alpha_mae_reduction_pct = +4.0000`
  - `trimap_error_reduction_pct = +5.6773`
- smoke:
  - `seam_improvement_pct = +30.9399`
  - `background_improvement_pct = +39.5147`
- route-specific signal:
  - `silhouette_temporal_improvement_pct = +33.2357`
  - `motion_temporal_improvement_pct = +12.4964`
  - `occlusion_temporal_improvement_pct = +20.7641`

### `rgba_round2`

- reviewed:
  - `boundary_f1_gain_pct = -13.3205`
  - `alpha_mae_reduction_pct = -8.7585`
  - `trimap_error_reduction_pct = -4.2298`
- smoke:
  - `seam_improvement_pct = +18.3226`
  - `background_improvement_pct = -5.8696`
- route-specific signal:
  - `rgba_boundary_reconstruction_improvement_pct = +14.8954`

### `layer_round3`

- reviewed:
  - `boundary_f1_gain_pct = -9.3989`
  - `alpha_mae_reduction_pct = -7.7646`
  - `trimap_error_reduction_pct = -1.8323`
- smoke:
  - `seam_improvement_pct = -14.1121`
  - `background_improvement_pct = +5.7802`
- route-specific signal:
  - `occlusion_temporal_improvement_pct = +9.2326`
  - `occlusion_gradient_gain_pct = +15.4723`

### `bridge_round1`

- reviewed:
  - `boundary_f1_gain_pct = +81.3250`
  - `alpha_mae_reduction_pct = -0.8688`
  - `trimap_error_reduction_pct = -1.2534`
- smoke:
  - `seam_improvement_pct = -0.5493`
  - `background_improvement_pct = +11.6589`
- route-specific signal:
  - `roi_gradient_gain_pct = -8.6352`


## Final Interpretation

Step 06 proves that Tier3 now has a real portfolio decision layer instead of a list of disconnected experiments.

The portfolio result is:

- no Tier3 route is promotable today
- `renderable_round2` is the strongest remembered seed
- `rgba_round2` and `layer_round3` are worth remembering as structural alternatives
- `bridge_round1` should not be escalated further

The most important outcome is not that Tier3 found a winner. It did not.

The important outcome is that Tier3 now has a disciplined stop rule:

- do not promote any current Tier3 route
- keep only the strongest structural seed in memory
- do not continue escalation unless a future route exceeds the current `renderable_round2` evidence


## Frozen Conclusion

- `best portfolio round = Round 3`
- `upgrade_candidates = []`
- `top_seed_route = renderable_round2`
- final Tier3 decision:
  - `stop_tier3_promotion_keep_renderable_seed_only`
