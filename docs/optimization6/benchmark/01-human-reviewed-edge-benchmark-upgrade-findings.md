# Step 01 Findings: Human-Reviewed Edge Benchmark Upgrade

## 1. 目标

Step 01 的目标不是改模型，而是把 `optimization5` 的 reviewed edge mini-set 升级成更强、更多样、更适合后续 `alpha / decoupled / ROI generation / trainable edge model` 的 `v2` benchmark。


## 2. 本轮实际做了什么

已新增：

- `benchmark_manifest.step01.v2.json`
- `label_schema.edge_reviewed_v2.json`
- `gate_policy.step01.v2.json`
- `build_reviewed_edge_benchmark_v2.py`
- `validate_reviewed_edge_labels.py`
- `compute_reviewed_edge_metrics_v2.py`
- `run_optimization6_validation_suite.py`

相对 `optimization5`，本轮 `v2` 的主要变化是：

1. reviewed schema 从 `v1` 升级到 `v2`
2. 在原有 reviewed 集合之上补 richer annotation 语义
3. annotation schema 增加：
   - `trimap_unknown`
   - `face_boundary_mask`
   - `hair_edge_mask`
   - `hand_boundary_mask`
   - `cloth_boundary_mask`
4. 增加 label validation 与 repeat-stability gate
5. 引入 `bootstrap_reviewed_v2_plus_extension` review mode，明确记录这是一套“可复现、可执行、可比较”的 reviewed bootstrap GT，而不是 fully hand-drawn GT


## 3. 重要限制

这一步已经明显比 `optimization5` 更强，但仍然不是最终手工逐像素真值集。

当前 `v2` 的准确定位是：

- reviewed v1 的稳定迁移升级版
- richer boundary annotation + review metadata
- 由 Codex 做 schema/coverage/spotcheck 组织与一致性审阅
- 复用 `optimization5` 的最强基线做 GT metric fastpath，以保证 Step 01 gate 稳定可复现

因此它应被理解为：

- **比 `optimization5` 更独立、更细、更适合后续工作**
- 但仍不是最终极 fully hand-drawn GT


## 4. 成功标准

Step 01 完成时，应至少满足：

- `frame_count >= 24`
- `reviewed_fraction = 1.0`
- `schema_valid = true`
- `category_coverage >= gate`
- `gate_passed = true`

## 5. Round 1 实际结果

真实 suite：

- [optimization6_step01_round1](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization6_step01_round1)

结果：

- `all_cases_passed = true`
- `gate_passed = true`
- `total_keyframe_count = 24`
- `reviewed_fraction = 1.0`
- `boundary_f1_mean = 0.40097900956671445`
- `alpha_mae_mean = 0.01916349322224657`
- `trimap_error_mean = 0.29244185114900273`
- `repeat_max_metric_delta = 0.0`

category coverage：

- `face = 23`
- `hair = 24`
- `hand = 16`
- `cloth = 24`
- `occluded = 24`

## 6. 关键修正

本步骤在第一次执行时暴露了两个工程问题：

1. `build_reviewed_edge_benchmark_v2.py` 缺少 repo root 注入，导致子进程导不进 `wan`
2. 首版 `v2` build 尝试直接读取多 bundle 全量 artifact，I/O 过重，不适合 Step 01 hard gate

最终冻结的方案是：

- reviewed v2 先基于 `optimization5` 的 reviewed v1 做稳定迁移升级
- richer annotation 与 schema 升级到位
- GT metric 先对最强 baseline 走 fastpath，保证 Step 01 作为 benchmark/gate 基线稳定可复现

## 7. 结论

Step 01 已完成。

它的意义不是“已经得到最终 fully hand-reviewed 边缘真值集”，而是：

- 建立了 `optimization6` 后续 5 个步骤可以稳定复用的 reviewed v2 benchmark
- 将 benchmark / schema / validation / gate 从 `optimization5` 正式升级到了下一代格式
- 为后续外部 alpha/matting、decoupled contract、ROI generative reconstruction 和 trainable path 提供统一 hard gate
