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
