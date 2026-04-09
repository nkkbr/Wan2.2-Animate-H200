# Optimization7 Benchmark

## Step 01

Step 01 benchmarks external alpha / matting candidates against the reviewed edge benchmark:

- reviewed dataset:
  - `runs/optimization6_step01_round1/reviewed_edge_benchmark_v2`
- current stable baseline summary:
  - `runs/optimization6_step01_round1/summary.json`
- stable clean-plate-aware preprocess bundle:
  - `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`

Registry:

- `docs/optimization7/benchmark/external_model_registry.step01.json`
- `docs/optimization7/benchmark/external_model_registry.step01b.json`

Core scripts:

- `scripts/eval/check_external_alpha_candidate.py`
- `scripts/eval/run_optimization7_external_alpha_benchmark.py`
- `scripts/eval/evaluate_optimization7_external_alpha_benchmark.py`

Reviewed gate:

- `alpha_mae` reduction >= `15%`
- `trimap_error` reduction >= `15%`
- `hair_edge_quality` gain >= `10%`
- At least `2/3` gates must pass for a winner to advance.

## Step 01b

Step 01b refreshes the candidate pool with a new model family instead of retuning the original
`BackgroundMattingV2 / RVM` family.

- refreshed registry:
  - `docs/optimization7/benchmark/external_model_registry.step01b.json`
- benchmark runs:
  - `runs/optimization7_step01b_round1_ab`
  - `runs/optimization7_step01b_round2_ab`
  - `runs/optimization7_step01b_round3_ab`
- findings:
  - `docs/optimization7/benchmark/01b-external-alpha-candidate-refresh-findings.md`
