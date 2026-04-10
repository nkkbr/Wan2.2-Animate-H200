# Step 04 Findings: RGBA / Transparent Foreground Feasibility

## Summary

Step 04 completed successfully as a Tier3 feasibility experiment.

The route is still **not promotable**, but unlike the bridge route it does show a clearer paired-output positive mode:

- semantic reconstruction against the source video improves
- hand-boundary reviewed quality turns clearly positive
- seam behavior remains strongly positive in all three rounds

The frozen Step 04 conclusion is:

- `best-of-three = Round 2`
- bucket:
  - `interesting_but_unproven`


## Deliverables

New files introduced in this step:

- `wan/utils/rgba_foreground_proto.py`
- `scripts/eval/check_rgba_foreground_proto.py`
- `scripts/eval/infer_rgba_foreground_proto.py`
- `scripts/eval/compute_rgba_foreground_metrics.py`
- `scripts/eval/evaluate_optimization9_rgba_benchmark.py`
- `scripts/eval/run_optimization9_rgba_benchmark.py`

Primary output directories:

- `runs/optimization9_step04_round1/`
- `runs/optimization9_step04_round2/`
- `runs/optimization9_step04_round3/`

Key outputs:

- `runs/optimization9_step04_round1/gate_result.json`
- `runs/optimization9_step04_round2/gate_result.json`
- `runs/optimization9_step04_round3/gate_result.json`


## Why This Step Was Needed

Step 03 showed that changing the representation can expose a genuinely new positive mode, but layer decomposition still remained too indirect for a practical paired foreground route.

The RGBA / transparent foreground question is more specific:

- if foreground RGB and alpha become a paired output object
- and the composite is derived from that pair directly
- does the system gain a more natural edge definition than the older replacement-style paths

That is a more direct test of whether “foreground as output” is structurally better than “alpha as helper signal”.


## Round 1

Round 1 built the minimum viable paired-foreground proto:

- keep the current decoupled preprocess bundle
- recompute a foreground alpha inside a semantic ROI
- solve premultiplied foreground RGB from the source video and clean-plate background
- rebuild:
  - `foreground_rgb`
  - `foreground_alpha`
  - `soft_alpha`
  - `boundary_band`
  - `trimap_unknown`
  - proxy composite preview

Round 1 also exposed one contract issue:

- `foreground_alpha` and `soft_alpha` cannot point to the same artifact if their metadata semantics differ

That was fixed by writing separate RGBA artifacts for:

- `foreground_alpha`
- `soft_alpha`

### Round 1 result

- reviewed:
  - `hair_boundary_f1_gain_pct = +1.46%`
  - `hand_boundary_f1_gain_pct = +10.74%`
  - `cloth_boundary_f1_gain_pct = +3.85%`
  - `boundary_f1_gain_pct = -13.27%`
  - `alpha_mae_reduction_pct = -8.74%`
  - `trimap_error_reduction_pct = -4.22%`
- smoke:
  - `seam_improvement_pct = +18.27%`
  - `background_improvement_pct = +0.06%`
- RGBA:
  - `rgba_boundary_reconstruction_improvement_pct = +9.73%`
  - `rgba_hair_reconstruction_improvement_pct = +10.18%`
  - `rgba_hand_reconstruction_improvement_pct = +9.79%`
  - `rgba_cloth_reconstruction_improvement_pct = +10.19%`

Interpretation:

- the route immediately showed a new semantic reconstruction gain
- hand-boundary reviewed quality also turned clearly positive
- but global boundary and continuous alpha metrics remained negative

Bucket:

- `interesting_but_unproven`


## Round 2

Round 2 pushed the semantic RGBA solve harder:

- stronger paired-output solve inside the semantic ROI
- stronger hair / hand / cloth emphasis
- more aggressive semantic ROI use

This round became the strongest demonstration of the route's new positive mode.

### Round 2 result

- reviewed:
  - `hair_boundary_f1_gain_pct = +1.46%`
  - `hand_boundary_f1_gain_pct = +10.74%`
  - `cloth_boundary_f1_gain_pct = +3.85%`
  - `boundary_f1_gain_pct = -13.32%`
  - `alpha_mae_reduction_pct = -8.76%`
  - `trimap_error_reduction_pct = -4.23%`
- smoke:
  - `seam_improvement_pct = +18.32%`
  - `background_improvement_pct = -5.87%`
- RGBA:
  - `rgba_boundary_reconstruction_improvement_pct = +14.90%`
  - `rgba_hair_reconstruction_improvement_pct = +15.56%`
  - `rgba_hand_reconstruction_improvement_pct = +14.98%`
  - `rgba_cloth_reconstruction_improvement_pct = +15.56%`

Interpretation:

- this is the strongest semantic reconstruction round
- it keeps seam behavior strongly positive
- background consistency regresses modestly, but stays inside the `interesting_but_unproven` tolerance band
- reviewed continuous alpha still fails to turn positive

Bucket:

- `interesting_but_unproven`


## Round 3

Round 3 rebalanced the route toward lower semantic aggression and more confidence gating.

The goal was to preserve the new paired-output value while reducing the negative movement on the reviewed alpha metrics and background behavior.

### Round 3 result

- reviewed:
  - `hair_boundary_f1_gain_pct = +1.46%`
  - `hand_boundary_f1_gain_pct = +10.74%`
  - `cloth_boundary_f1_gain_pct = +3.85%`
  - `boundary_f1_gain_pct = -13.28%`
  - `alpha_mae_reduction_pct = -8.74%`
  - `trimap_error_reduction_pct = -4.22%`
- smoke:
  - `seam_improvement_pct = +18.00%`
  - `background_improvement_pct = -0.13%`
- RGBA:
  - `rgba_boundary_reconstruction_improvement_pct = +9.75%`
  - `rgba_hair_reconstruction_improvement_pct = +10.22%`
  - `rgba_hand_reconstruction_improvement_pct = +9.81%`
  - `rgba_cloth_reconstruction_improvement_pct = +10.22%`

Interpretation:

- Round 3 is the safer version of the route
- but it does not preserve enough extra semantic reconstruction value to beat Round 2

Bucket:

- `interesting_but_unproven`


## Why The Route Did Not Promote

The route still fails promotion for the same hard reason:

- it never turns reviewed continuous alpha metrics positive

Across all rounds:

- `alpha_mae_reduction_pct` stays around `-8.7%`
- `trimap_error_reduction_pct` stays around `-4.2%`
- global `boundary_f1` stays negative

So this is not a hidden production winner.


## Why The Route Is Still More Interesting Than Step 02

Step 02 behaved like:

- a threshold corrector
- with almost no semantic breakthrough

Step 04 behaves like:

- a paired foreground-output route
- with repeatable semantic reconstruction gains
- and a stable hand-boundary reviewed gain

This matters because the new positive mode is not just:

- “threshold got better”

It is:

- “foreground RGB plus alpha as a paired output reconstructs semantic boundary regions more naturally”

That is a more meaningful structural difference than the bridge route exposed.


## Frozen Best-Of-Three

`Round 2` is frozen as `best-of-three`.

Reason:

- it preserves the same reviewed hand-boundary win seen in all rounds
- it produces the strongest semantic reconstruction gains
- seam behavior remains strongly positive
- the background regression is still within the feasibility tolerance for `interesting_but_unproven`

Frozen bucket:

- `interesting_but_unproven`


## What Step 04 Proved

Step 04 proved four useful things:

1. paired `foreground RGB + alpha` output can be integrated into the current bundle format
2. this route exposes a more meaningful semantic positive mode than the bridge route
3. semantic reconstruction and reviewed hand-boundary quality can improve together
4. continuous alpha quality still remains the blocking issue for promotion


## Recommended Next Move

Proceed to `optimization9 / Step 05` only if we still want to push Tier3 into a more explicit renderable-foreground direction.

That makes sense because:

- Step 04 is not promotable
- but it is more structurally promising than Step 02
- and it suggests that “foreground as an explicit object” may still be the right Tier3 direction, even though the current prototype is not good enough
