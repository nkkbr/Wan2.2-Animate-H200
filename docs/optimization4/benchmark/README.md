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

### 3.4 生成 gate

```bash
python scripts/eval/summarize_optimization4_validation.py \
  --summary_json runs/optimization4_step01_core/summary.json \
  --gate_policy docs/optimization4/benchmark/gate_policy.step01.json \
  --output_json runs/optimization4_step01_core/gate_result.json
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
