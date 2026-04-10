# Optimization9 Benchmark

## Step 01

Step 01 establishes the Tier3 feasibility protocol and stop rules.

Core files:

- `docs/optimization9/benchmark/feasibility_manifest.step01.json`
- `docs/optimization9/benchmark/stop_rules.step01.json`
- `scripts/eval/run_optimization9_feasibility_suite.py`
- `scripts/eval/evaluate_optimization9_feasibility_suite.py`
- `scripts/eval/check_optimization9_feasibility.py`

Primary output directory:

- `runs/optimization9_step01_round3/`

Representative playback routes:

- `external_alpha_matanyone_round2`
- `trainable_alpha_round1`
- `semantic_experts_round1`
- `loss_stack_round2`

Frozen conclusion:

- `external_alpha_matanyone_round2 -> reject`
- `trainable_alpha_round1 -> interesting_but_unproven`
- `semantic_experts_round1 -> reject`
- `loss_stack_round2 -> reject`
- no route qualifies as `upgrade_candidate`

Tier3 stop-rule result:

- the feasibility protocol is now frozen
- future Tier3 prototypes must show a genuinely different positive error mode, not just another reviewed-vs-smoke tradeoff

## Step 02

Step 02 tests the lightest-weight Tier3 prototype:

- a `mask-to-matte / matte bridge` model that operates on boundary ROI patches
- uses:
  - foreground/background evidence
  - soft alpha
  - trimap unknown
  - boundary ROI and uncertainty
- writes refined alpha artifacts back into a cloned preprocess bundle

Core files:

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

Frozen conclusion:

- `best-of-three = Round 1`
- bucket:
  - `reject`

Frozen failure patterns:

- `threshold_only_gain`
- `no_semantic_breakthrough`

Interpretation:

- the bridge route can strongly improve threshold-level `boundary_f1`
- but it does not improve continuous alpha quality
- and it does not create a meaningful hair / semi-transparent breakthrough

## Step 03

Step 03 tests a more structural Tier3 route:

- foreground layer
- occlusion / unresolved layer
- background layer

The prototype is intentionally minimal, but unlike Step 02 it changes the representation of the problem instead of only refining an existing matte.

Core files:

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

Frozen conclusion:

- `best-of-three = Round 3`
- bucket:
  - `interesting_but_unproven`

Frozen interpretation:

- the route is still not promotable
- reviewed alpha and occluded-boundary metrics remain negative
- but it does show a genuinely different positive mode:
  - better occlusion temporal consistency
  - better occlusion-boundary gradient behavior
- this is the first Tier3 route so far that is worth remembering for its structural difference, even though it still fails promotion

## Step 04

Step 04 tests a different Tier3 representation:

- paired `foreground RGB + alpha`
- explicit RGBA-style foreground output semantics
- direct composite from the paired foreground output and background

This route is still minimal, but unlike Step 02 and Step 03 it asks whether “foreground as an output object” is itself a better problem structure.

Core files:

- `wan/utils/rgba_foreground_proto.py`
- `scripts/eval/check_rgba_foreground_proto.py`
- `scripts/eval/infer_rgba_foreground_proto.py`
- `scripts/eval/compute_rgba_foreground_metrics.py`
- `scripts/eval/run_optimization9_rgba_benchmark.py`
- `scripts/eval/evaluate_optimization9_rgba_benchmark.py`

Primary output directories:

- `runs/optimization9_step04_round1/`
- `runs/optimization9_step04_round2/`
- `runs/optimization9_step04_round3/`

Frozen conclusion:

- `best-of-three = Round 2`
- bucket:
  - `interesting_but_unproven`

Frozen interpretation:

- the route is still not promotable
- reviewed continuous alpha metrics remain negative
- but it shows a clearer semantic positive mode than Step 02:
  - hand-boundary reviewed quality turns positive
  - semantic reconstruction against the source video improves by roughly `10%` to `15%`
  - seam behavior stays strongly positive
- this makes the RGBA foreground route worth remembering as a more promising structural Tier3 branch, even though it still fails promotion

## Step 05

Step 05 tests the most aggressive Tier3 route so far:

- a minimal renderable foreground state
- explicit temporal reprojection of foreground RGB and alpha
- depth-like pseudo-field
- silhouette-aware temporal stabilization

This route is not trying to win by a better patch or a better local matte. It is testing whether a more stable foreground state itself creates a different class of gain.

Core files:

- `wan/utils/renderable_foreground_proto.py`
- `scripts/eval/check_renderable_foreground_proto.py`
- `scripts/eval/infer_renderable_foreground_proto.py`
- `scripts/eval/compute_renderable_foreground_metrics.py`
- `scripts/eval/run_optimization9_renderable_benchmark.py`
- `scripts/eval/evaluate_optimization9_renderable_benchmark.py`

Primary output directories:

- `runs/optimization9_step05_round1/`
- `runs/optimization9_step05_round2/`
- `runs/optimization9_step05_round3/`

Frozen conclusion:

- `best-of-three = Round 2`
- bucket:
  - `interesting_but_unproven`

Frozen interpretation:

- the route is still not promotable
- global reviewed boundary quality remains too negative
- but it is the strongest Tier3 signal so far:
  - `alpha_mae` and `trimap_error` both improve
  - seam and background stability improve strongly
  - silhouette, motion, and occlusion temporal consistency all improve strongly
- this makes the renderable foreground route the most structurally promising Tier3 branch so far, even though it still fails promotion

## Step 06

Step 06 converts Tier3 from a collection of independent high-risk routes into a portfolio with an explicit escalation decision.

It does not introduce a new prototype. Instead, it aggregates the frozen best-of-three outcomes from:

- Step 02 bridge route
- Step 03 layer decomposition route
- Step 04 RGBA foreground route
- Step 05 renderable foreground route

Core files:

- `docs/optimization9/benchmark/portfolio_manifest.step06.json`
- `scripts/eval/run_optimization9_portfolio_summary.py`
- `scripts/eval/evaluate_optimization9_portfolio.py`
- `scripts/eval/check_optimization9_portfolio_gate.py`

Primary output directories:

- `runs/optimization9_step06_round1/`
- `runs/optimization9_step06_round2/`
- `runs/optimization9_step06_round3/`

Frozen conclusion:

- `best portfolio round = Round 3`
- `upgrade_candidates = []`
- `top_seed_route = renderable_round2`
- final portfolio decision:
  - `stop_tier3_promotion_keep_renderable_seed_only`

Frozen interpretation:

- Tier3 still does not have a promotable route
- `renderable_round2` is the strongest remembered structural seed
- `rgba_round2` and `layer_round3` remain worth remembering as alternative structural ideas
- `bridge_round1` should not be escalated further
- future Tier3 work should only continue if it can beat the current `renderable_round2` evidence, not merely reproduce another tradeoff pattern
