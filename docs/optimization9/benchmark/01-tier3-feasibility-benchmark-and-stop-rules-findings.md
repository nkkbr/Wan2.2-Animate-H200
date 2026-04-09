# Step 01 Findings: Tier3 Feasibility Benchmark And Stop Rules

## Summary

Step 01 completed successfully.

The main result is not a quality win. The main result is that Tier3 no longer starts from vague intuition.

This step now provides:

- a frozen feasibility manifest
- frozen stop rules
- a normalizer that can replay old route evidence into a single shape
- an evaluator that buckets routes into:
  - `reject`
  - `interesting_but_unproven`
  - `upgrade_candidate`

Most importantly, the protocol already distinguishes between:

- routes that are simply bad
- routes that still fail promotion, but exhibit a genuinely different positive error mode

That distinction is exactly what Tier3 needed before any heavy prototype work.


## Deliverables

New files introduced in this step:

- `docs/optimization9/benchmark/feasibility_manifest.step01.json`
- `docs/optimization9/benchmark/stop_rules.step01.json`
- `scripts/eval/run_optimization9_feasibility_suite.py`
- `scripts/eval/evaluate_optimization9_feasibility_suite.py`
- `scripts/eval/check_optimization9_feasibility.py`

Primary output directory:

- `runs/optimization9_step01_round3/`

Key outputs:

- `runs/optimization9_step01_round3/summary.json`
- `runs/optimization9_step01_round3/gate_result.json`


## Why This Step Was Needed

Before Step 01, Tier3 was still vulnerable to the worst possible failure mode:

- many expensive high-risk prototypes
- no unified criterion for “interesting enough to continue”
- no clear stop rule

That would have produced exactly the kind of low-signal, high-cost exploration we wanted to avoid.

Step 01 fixed that by making Tier3 routes pass through the same feasibility protocol before they earn more investment.


## Round 1

Round 1 established the first draft of:

- feasibility manifest
- bucket taxonomy
- stop-rule thresholds

The first draft was intentionally simple:

- strong reviewed gains
- smoke safety
- semantic-difficulty gains where available

This was enough to normalize old evidence, but still too permissive conceptually because it risked over-crediting routes that only improved `boundary_f1`.


## Round 2

Round 2 replayed old failure routes and used them to refine the bucket logic.

Representative playback set:

- `external_alpha_matanyone_round2`
- `trainable_alpha_round1`
- `semantic_experts_round1`
- `loss_stack_round2`

Two important corrections were made in this round:

1. `threshold_only_gain` became an explicit failure pattern.
   - this catches routes that show strong `boundary_f1` movement while continuous alpha metrics barely move
2. strong smoke regression became a direct demotion signal.
   - this prevents routes from being called “interesting” if they only look promising on reviewed metrics but break real-video composite behavior


## Round 3

Round 3 froze the protocol and reran the full playback set.

Frozen outputs:

- `runs/optimization9_step01_round3/summary.json`
- `runs/optimization9_step01_round3/gate_result.json`

The final buckets are:

- `optimization8_non_train_reviewed_v3`
  - `reference`
- `external_alpha_matanyone_round2`
  - `reject`
- `trainable_alpha_round1`
  - `interesting_but_unproven`
- `semantic_experts_round1`
  - `reject`
- `loss_stack_round2`
  - `reject`

There are:

- no `upgrade_candidate` routes


## What The Bucket Decisions Mean

### `external_alpha_matanyone_round2 -> reject`

Reason:

- strong negative movement on every reviewed metric
- no positive smoke evidence
- no meaningful new positive error mode

Interpretation:

- this route is not “interesting but just undertrained”
- it is simply not strong enough on our benchmark

### `trainable_alpha_round1 -> interesting_but_unproven`

Reason:

- very large `boundary_f1` gain relative to the frozen non-train reference
- no corresponding continuous alpha / trimap gain
- not enough evidence to upgrade

Interpretation:

- this route does **not** pass promotion
- but it does show a genuinely different positive mode:
  - threshold-level boundary behavior improved dramatically
- that makes it worth keeping alive as research scaffolding

### `semantic_experts_round1 -> reject`

Reason:

- almost no reviewed amplification beyond `trainable_alpha_round1`
- explicit smoke regression

Interpretation:

- this route does not create a new useful mode
- it just adds complexity and makes the downstream result worse

### `loss_stack_round2 -> reject`

Reason:

- tiny reviewed gains beyond `trainable_alpha_round1`
- severe smoke regression

Interpretation:

- this route is exactly the kind of thing Tier3 must reject quickly:
  - expensive to continue
  - no structural new behavior


## Frozen Stop Rules

The key frozen principles are:

1. `upgrade_candidate`
   - must show multi-metric reviewed strength
   - and at least one real downstream positive signal
2. `interesting_but_unproven`
   - may survive only if it shows a genuinely different positive mode
   - without strong smoke regression
3. `reject`
   - includes:
     - uniformly negative routes
     - flat reviewed routes
     - threshold-only or patch-only behavior that collapses under smoke


## Why Step 01 Succeeded

Step 01 succeeded because it now makes Tier3 executable in a disciplined way.

It does **not** promise that any Tier3 route will work.

What it guarantees is:

- Tier3 routes will no longer drift without classification
- each future prototype can be compared against frozen failure patterns
- the portfolio can stop quickly if no real upgrade candidate appears


## Recommended Next Move

Proceed to `optimization9 / Step 02`.

Reason:

- the feasibility protocol is now strong enough to evaluate whether a `mask-to-matte / bridge-model` prototype produces a truly different error mode
- that route is the lightest-weight Tier3 candidate and should be tested first
