# Optimization6 Benchmark

## Step 01

本目录记录 `optimization6` 的 benchmark 基础设施，尤其是 Step 01 升级后的 reviewed edge benchmark v2。

当前目标不是证明某条新边缘路线已经成功，而是先建立一套：

- 更大
- 更细
- 相对 `optimization5` 更独立
- 更适合后续 alpha / ROI reconstruction / trainable edge model 的

边缘基准。

### 当前文件

- `benchmark_manifest.step01.v2.json`
- `label_schema.edge_reviewed_v2.json`
- `gate_policy.step01.v2.json`
- `01-human-reviewed-edge-benchmark-upgrade-findings.md`

### 脚本

- `scripts/eval/build_reviewed_edge_benchmark_v2.py`
- `scripts/eval/validate_reviewed_edge_labels.py`
- `scripts/eval/compute_reviewed_edge_metrics_v2.py`
- `scripts/eval/run_optimization6_validation_suite.py`

### 当前限制

尽管本轮命名为 `human-reviewed edge benchmark upgrade`，当前 v2 仍然是：

- 多 preprocess bundle 共识生成
- 带人工审核字段与 spotcheck 组织方式
- 由 Codex 进行结构化审阅与一致性校验

而不是逐像素全人工绘制真值。

这比 `optimization5` 的 reviewed set 更独立、更细，但仍然不应被误解为最终极人工 GT。

## Step 02

Step 02 开始引入正式的外部 alpha / matting 模型接入规范。

当前新增文件：

- `external_model_registry.step02.json`
- `02-external-alpha-matting-model-vetting-and-integration-findings.md`

当前新增脚本：

- `scripts/eval/check_external_alpha_model.py`
- `scripts/eval/run_external_alpha_benchmark.py`
- `scripts/eval/evaluate_external_alpha_benchmark.py`

当前 Step 02 的工程目标是：

- 让外部模型的 repo / release / 权重 / hash / license 都被显式记录；
- 让外部模型通过统一 adapter 接口接入；
- 在 reviewed benchmark 上做客观 AB；
- 通过 gate 的候选才允许升主线；
- 即使候选失败，也要保留 registry + adapter + benchmark 作为后续更强候选的复用基础。

## Step 03

Step 03 正式把 replacement 的主问题重写为：

- foreground RGB
- foreground alpha / confidence
- background RGB
- visible support / unresolved
- composite ROI

之间的 decoupled contract 与 core conditioning 协作问题。

当前新增文件：

- `03-foreground-background-decoupled-replacement-contract-and-core-path-findings.md`

当前新增脚本：

- `scripts/eval/run_optimization6_decoupled_benchmark.py`
- `scripts/eval/evaluate_optimization6_decoupled_benchmark.py`
- `scripts/eval/check_decoupled_core_path.py`

当前 Step 03 的工程目标是：

- 让 `decoupled_v2` 成为正式可选的 core conditioning path；
- 保证 decoupled bundle 的 artifact / contract / runtime 语义完整；
- 用真实 decoupled preprocess bundle 验证 `foreground/background/alpha/roi` 信号已经真正进入主路径；
- 如果真实 AB 仍然无法过 gate，则保留 contract 与代码路径，但不升默认核心路径。

## Step 04

Step 04 开始尝试真正的边界 ROI 生成式重建第二版。

当前新增文件：

- `04-boundary-roi-generative-reconstruction-v2-findings.md`

当前新增脚本：

- `scripts/eval/check_boundary_roi_generative_v2.py`
- `scripts/eval/run_optimization6_boundary_roi_benchmark.py`
- `scripts/eval/evaluate_optimization6_boundary_roi_benchmark.py`

当前 Step 04 的工程目标是：

- 让 `boundary_refine_mode=roi_gen_v2` 成为正式实验路径；
- 在 ROI 内不再只做 deterministic sharpen / blend，而是尝试更强的局部重建候选；
- 通过 reviewed edge ROI 指标和真实 smoke，判断这条路线是否真的比旧版 `roi_gen_v1` 更接近“肉眼边缘更锐”的目标；
- 若 3 轮后仍无法通过强 gate，则冻结 best-of-three，只保留 ROI 生成式框架，不升默认。


## Step 05

Step 05 开始尝试真正的 trainable edge route。

当前新增文件：

- `05-trainable-edge-models-on-reviewed-data-findings.md`

当前新增脚本：

- `scripts/eval/build_trainable_edge_dataset.py`
- `scripts/eval/check_trainable_edge_model.py`
- `scripts/eval/train_trainable_edge_model.py`
- `scripts/eval/infer_trainable_edge_model.py`
- `scripts/eval/evaluate_trainable_edge_checkpoint_against_reviewed.py`
- `scripts/eval/evaluate_optimization6_trainable_edge.py`

当前 Step 05 的结论是：

- trainable route 的工程基础设施已经完成；
- 但当前 reviewed 数据规模与任务定义下，这条路线既没有客观打赢 baseline，也没有表现出足够健康的吞吐；
- 因此只保留为研究支线，不升主线。


## Step 06

Step 06 关注的是：只有在已经选出有效路线之后，才值得把 H200 算力编排成正式 tier。

当前新增文件：

- `tier_manifest.step06.json`
- `06-h200-high-value-orchestration-after-route-selection-findings.md`

当前新增脚本：

- `scripts/eval/check_optimization6_orchestration.py`
- `scripts/eval/run_optimization6_orchestration_benchmark.py`
- `scripts/eval/evaluate_optimization6_orchestration_benchmark.py`

当前 Step 06 的结论是：

- orchestration 基础设施已经完成；
- 但在没有更强路线被证明有效的前提下，`high_quality / extreme` 都不值得保留；
- 当前生产仍应只保留单一稳定档。
