# Optimization4 Benchmark

## 1. 目标

本目录用于支撑 `optimization4` 路线中的“极致边缘锐度 benchmark 与 gate”。

相对 `optimization3`，这里新增的重点是：

- 小型 edge-labeled mini-set
- 真实边缘指标：
  - boundary F-score
  - trimap error
  - alpha MAE / SAD
- 可用于后续 alpha / ROI refine / generate-side edge search 的强 gate


## 2. 目录内容

- `benchmark_manifest.step01.json`
  - Step 01 使用的 edge mini-set bootstrap manifest
- `label_schema.edge_v1.json`
  - edge-labeled mini-set 的标签结构定义
- `gate_policy.step01.json`
  - Step 01 的 gate 判定规则
- `01-edge-labeled-mini-benchmark-and-gates-findings.md`
  - Step 01 的正式结论与 baseline 指标
- `02-high-quality-alpha-matting-upgrade-findings.md`
  - Step 02 的正式结论与三轮 alpha / matting AB 结果
- `03-rich-boundary-signal-core-conditioning-findings.md`
  - Step 03 的正式结论与三轮 generate-side rich conditioning AB 结果
- `04-boundary-roi-two-stage-highres-refine-findings.md`
  - Step 04 的正式结论与三轮 boundary ROI two-stage refine AB 结果
- `06-semantic-boundary-specialization-findings.md`
  - Step 06 的正式结论与三轮 semantic boundary specialization AB 结果
- `07-local-edge-super-resolution-detail-restoration-findings.md`
  - Step 07 的正式结论与三轮 local edge restoration AB 结果


## 3. 关键脚本

### 3.1 构建 edge mini-set

```bash
python scripts/eval/extract_edge_benchmark_keyframes.py \
  --manifest docs/optimization4/benchmark/benchmark_manifest.step01.json \
  --output_dir runs/optimization4_step01_edge_mini_set
```

### 3.2 计算真实边缘指标

```bash
python scripts/eval/compute_edge_groundtruth_metrics.py \
  --dataset_dir runs/optimization4_step01_edge_mini_set \
  --prediction_preprocess_dir runs/<suite>/preprocess_edge_baseline/preprocess \
  --output_json runs/<suite>/metrics/edge_groundtruth_metrics.json
```

### 3.3 跑 Step 01 suite

```bash
python scripts/eval/run_optimization4_validation_suite.py \
  --suite_name optimization4_step01_core \
  --manifest docs/optimization4/benchmark/benchmark_manifest.step01.json \
  --gate_policy docs/optimization4/benchmark/gate_policy.step01.json
```

说明：

- 默认模式会复用 manifest 中声明的冻结 baseline preprocess bundle
- 这样可以让 Step 01 的 gate 更稳定、成本更可控
- 如果后续步骤明确需要验证“当前代码重新跑出的 preprocess”是否仍然过关，再显式使用 `--rerun_baseline_preprocess`

### 3.4 计算 alpha / matting 真值指标

```bash
python scripts/eval/compute_alpha_precision_metrics.py \
  --dataset_dir runs/optimization4_step01_core_v2/edge_mini_set \
  --prediction_preprocess_dir runs/<step02_run>/preprocess \
  --output_json runs/<step02_run>/metrics/alpha_precision_metrics.json
```

### 3.5 synthetic alpha contract check

```bash
python scripts/eval/check_alpha_matting_upgrade.py
```

### 3.6 运行 Step 03 rich boundary conditioning AB

```bash
python scripts/eval/run_optimization4_rich_conditioning_benchmark.py \
  --suite_name optimization4_step03_round1_ab
```

### 3.7 评估 Step 03 gate

```bash
python scripts/eval/evaluate_optimization4_rich_conditioning_benchmark.py \
  --summary_json runs/optimization4_step03_round1_ab/summary.json \
  --output_json runs/optimization4_step03_round1_ab/gate_result.json
```

### 3.8 生成 Step 01 gate

```bash
python scripts/eval/summarize_optimization4_validation.py \
  --summary_json runs/optimization4_step01_core/summary.json \
  --gate_policy docs/optimization4/benchmark/gate_policy.step01.json \
  --output_json runs/optimization4_step01_core/gate_result.json
```

### 3.9 运行 Step 04 ROI refine AB

```bash
python scripts/eval/run_optimization4_boundary_roi_benchmark.py \
  --suite_name optimization4_step04_round1_ab
```

### 3.10 评估 Step 04 ROI refine gate

```bash
python scripts/eval/evaluate_optimization4_boundary_roi_benchmark.py \
  --summary_json runs/optimization4_step04_round1_ab/summary.json \
  --output_json runs/optimization4_step04_round1_ab/gate_result.json
```

### 3.11 synthetic ROI refine 检查

```bash
python scripts/eval/check_boundary_roi_refine.py
```

### 3.12 synthetic semantic boundary specialization 检查

```bash
python scripts/eval/check_semantic_boundary_specialization.py
```

### 3.13 运行 Step 06 semantic boundary AB

```bash
python scripts/eval/run_optimization4_semantic_boundary_benchmark.py \
  --suite_name optimization4_step06_round1_ab \
  --src_root_path runs/<step06_preprocess>/preprocess
```

### 3.14 计算 Step 06 semantic boundary 分类指标

```bash
python scripts/eval/compute_semantic_boundary_metrics.py \
  --before runs/<suite>/v2/outputs/v2_rich_v1.mkv \
  --after runs/<suite>/semantic_v1/outputs/semantic_v1_semantic_v1.mkv \
  --src_root_path runs/<step06_preprocess>/preprocess \
  --output_json runs/<suite>/semantic_v1_vs_v2_semantic_metrics.json
```

### 3.15 评估 Step 06 semantic boundary gate

```bash
python scripts/eval/evaluate_optimization4_semantic_boundary_benchmark.py \
  --summary_json runs/optimization4_step06_round1_ab/summary.json \
  --output_json runs/optimization4_step06_round1_ab/gate_result.json
```

### 3.16 synthetic local edge restoration 检查

```bash
python scripts/eval/check_local_edge_restoration.py
```

### 3.17 运行 Step 07 local edge restoration AB

```bash
python scripts/eval/run_optimization4_local_edge_benchmark.py \
  --suite_name optimization4_step07_round1_ab \
  --src_root_path runs/optimization4_step06_round1_preprocess/preprocess
```

### 3.18 计算 Step 07 local edge restoration 指标

```bash
python scripts/eval/compute_local_edge_restoration_metrics.py \
  --before runs/optimization4_step07_round1_ab/roi_v1/outputs/roi_v1.mkv \
  --after runs/optimization4_step07_round1_ab/local_edge_v1/outputs/local_edge_v1.mkv \
  --src_root_path runs/optimization4_step06_round1_preprocess/preprocess \
  --debug_dir runs/optimization4_step07_round1_ab/local_edge_v1/debug/generate/boundary_refinement \
  --output_json runs/optimization4_step07_round1_ab/local_edge_v1_local_metrics.json
```

### 3.19 评估 Step 07 local edge restoration gate

```bash
python scripts/eval/evaluate_optimization4_local_edge_benchmark.py \
  --summary_json runs/optimization4_step07_round1_ab/summary.json \
  --output_json runs/optimization4_step07_round1_ab/gate_result.json
```


## 4. 重要说明

Step 01 的 edge mini-set v1 允许使用 `bootstrap_preprocess_label` 作为第一版标签来源，但必须在 label metadata 中显式记录：

- 标签来源 bundle
- 是否人工审核
- 标注版本

也就是说，v1 可以是“可运行、可比较、可复核”的 bootstrap 标注集，但不能把它伪装成最终人工真值。


## 5. Step 01 验收要求

Step 01 结束时，至少应满足：

1. edge mini-set 已生成
2. 关键帧数不少于 20
3. `boundary F-score / trimap error / alpha MAE` 至少 3 个真值指标可跑
4. 当前 baseline preprocess bundle 能稳定出分
5. suite 能产出：
   - `summary.json`
   - `summary.md`
   - `gate_result.json`


## 6. 与后续步骤的关系

从 Step 02 开始，所有和“边缘是否真的提升”相关的步骤，都优先使用这里的真实边缘指标与 gate，而不是只依赖 `optimization3` 的 proxy 指标。
