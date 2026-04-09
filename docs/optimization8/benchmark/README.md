# Optimization8 Benchmark

## Step 01

Step 01 upgrades the reviewed edge benchmark from `v2` to `v3_candidate`.

Core files:

- `docs/optimization8/benchmark/benchmark_manifest.step01.v3.json`
- `docs/optimization8/benchmark/label_schema.edge_reviewed_v3.json`
- `docs/optimization8/benchmark/data_governance.step01.md`

Core scripts:

- `scripts/eval/extract_reviewed_edge_keyframes_v3.py`
- `scripts/eval/build_reviewed_edge_benchmark_v3.py`
- `scripts/eval/check_reviewed_edge_dataset_v3.py`
- `scripts/eval/compute_reviewed_edge_metrics_v3.py`

Primary seed dataset:

- `runs/optimization6_step01_round1/reviewed_edge_benchmark_v2`

Primary stable baseline bundle:

- `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`

Frozen v3 candidate:

- `runs/optimization8_step01_round3/reviewed_edge_benchmark_v3_candidate`

Important note:

- For `optimization8 Step 02+`, do not use only the overall mean on `v3_candidate`.
- Prefer `split_metrics.expansion_eval` and especially `split_metrics.holdout_eval` as the main promotion gates.

## Step 02

Step 02 defines the trainable task taxonomy and freezes the ROI dataset pipeline.

Core files:

- `docs/optimization8/benchmark/task_taxonomy.step02.json`
- `wan/utils/roi_dataset_schema.py`
- `scripts/eval/build_roi_training_dataset_v2.py`
- `scripts/eval/check_roi_training_dataset_v2.py`

Frozen ROI dataset:

- `runs/optimization8_step02_round3/roi_dataset_v2`

## Step 03

Step 03 establishes a controlled trainable alpha / matte completion baseline.

Core files:

- `wan/utils/trainable_alpha_model.py`
- `scripts/eval/check_trainable_alpha_model.py`
- `scripts/eval/train_trainable_alpha_model.py`
- `scripts/eval/infer_trainable_alpha_model.py`
- `scripts/eval/evaluate_trainable_alpha_model.py`

Primary inputs:

- `runs/optimization8_step02_round3/roi_dataset_v2`
- `runs/optimization8_step01_round3/reviewed_edge_benchmark_v3_candidate`
- `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`

Round directories:

- `runs/optimization8_step03_round1/`
- `runs/optimization8_step03_round2/`
- `runs/optimization8_step03_round3/`

Smoke directories:

- `runs/optimization8_step03_round1_smoke/`
- `runs/optimization8_step03_round3_smoke/`

Frozen conclusion:

- `best-of-three = Round 1`
- reviewed gate failed with `pass_count = 1`
- the route improved `holdout boundary_f1`, but did not meaningfully reduce `alpha_mae` or `trimap_error`

## Step 04

Step 04 extends the Step 03 trainable route with semantic boundary experts.

Core files:

- `wan/utils/semantic_edge_experts.py`
- `scripts/eval/check_semantic_edge_experts.py`
- `scripts/eval/train_semantic_edge_experts.py`
- `scripts/eval/infer_semantic_edge_experts.py`
- `scripts/eval/evaluate_semantic_edge_experts.py`

Primary inputs:

- `runs/optimization8_step02_round3/roi_dataset_v2`
- `runs/optimization8_step03_round1/preprocess_trainable_alpha`
- `runs/optimization8_step01_round3/reviewed_edge_benchmark_v3_candidate`

Round directories:

- `runs/optimization8_step04_round1/`
- `runs/optimization8_step04_round2/`
- `runs/optimization8_step04_round3/`

Smoke directories:

- `runs/optimization8_step04_round1_smoke/`
- `runs/optimization8_step04_round2_smoke/`
- `runs/optimization8_step04_round3_smoke/`

Frozen conclusion:

- `best-of-three = Round 1`
- all three rounds failed the semantic promotion gate with `semantic_pass_count = 0`
- training learned patch-local semantic structure, but that signal did not survive end-to-end promotion
- stronger routing coverage caused larger smoke regressions in `seam` and `background_fluctuation`

## Step 05

Step 05 introduces compositing-aware losses on top of the Step 03 trainable alpha path.

Core files:

- `wan/utils/edge_losses.py`
- `scripts/eval/check_compositing_aware_losses.py`
- `scripts/eval/evaluate_compositing_aware_training.py`
- `scripts/eval/train_trainable_alpha_model.py` (loss registry extension)

Round directories:

- `runs/optimization8_step05_round1/`
- `runs/optimization8_step05_round2/`
- `runs/optimization8_step05_round3/`

Smoke directories:

- `runs/optimization8_step05_round1_smoke/`
- `runs/optimization8_step05_round2_smoke/`

Frozen conclusion:

- `best-of-three = Round 2`
- `composite_v1`, `composite_grad_v1`, `composite_grad_contrast_v1` all failed the reviewed promotion gate
- Round 1 and Round 2 also failed smoke safety
- compositing-aware losses are now available as reusable infrastructure, but they did not rescue the current trainable alpha route

## Step 06

Step 06 performs the Tier2 promotion and amplification decision.

Core files:

- `docs/optimization8/benchmark/tier2_manifest.json`
- `scripts/eval/run_optimization8_tier2_benchmark.py`
- `scripts/eval/evaluate_optimization8_tier2_benchmark.py`
- `scripts/eval/check_optimization8_tier2_gate.py`

Primary output directory:

- `runs/optimization8_step06_tier2_gate/`

Frozen conclusion:

- `promote_tier2_as_amplifier = false`
- `trainable_alpha_best` is kept only as experimental scaffolding
- `semantic_experts_best` and `best_loss_stack_variant` are rejected for now
- no `tier2_full_best` candidate exists
