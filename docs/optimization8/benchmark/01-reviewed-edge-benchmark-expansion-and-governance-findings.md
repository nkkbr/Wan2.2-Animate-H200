# Step 01 Findings: Reviewed Edge Benchmark Expansion And Governance

## 1. 结论

Step 01 已完成，工程目标达成，并且最终冻结了 `reviewed_edge_v3_candidate`。

这一步的真实结果是：

- reviewed edge benchmark 从 `24` 帧扩到 `48` 帧，达到 `2x`
- semantic coverage 全部超过最低阈值
- `strong_revision_fraction = 0.5`，明显高于 `0.30` 的 gate
- `split` 被正式拆成：
  - `seed_eval = 24`
  - `expansion_eval = 16`
  - `holdout_eval = 8`

这一步的最终冻结目录是：

- [reviewed_edge_benchmark_v3_candidate](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization8_step01_round3/reviewed_edge_benchmark_v3_candidate)

但也要如实说明：

- `v3_candidate` 仍然不是 fully hand-redrawn GT
- 整体指标会被 `seed_eval` 明显拉高
- 真正更能代表新增难度的，是 `expansion_eval` 与 `holdout_eval`

因此，这一步的准确意义是：

> 我们已经有了一个更大、治理更完整、split 更清楚的 `v3_candidate`，可以支撑 `optimization8 Step 02+` 的训练与评测；但它还不是最终的“高度独立 reviewed GT”。

## 2. 代码与文档交付

新增：

- [benchmark_manifest.step01.v3.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/benchmark/benchmark_manifest.step01.v3.json)
- [label_schema.edge_reviewed_v3.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/benchmark/label_schema.edge_reviewed_v3.json)
- [data_governance.step01.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/benchmark/data_governance.step01.md)
- [README.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/benchmark/README.md)
- [extract_reviewed_edge_keyframes_v3.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/extract_reviewed_edge_keyframes_v3.py)
- [build_reviewed_edge_benchmark_v3.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/build_reviewed_edge_benchmark_v3.py)
- [check_reviewed_edge_dataset_v3.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_reviewed_edge_dataset_v3.py)
- [compute_reviewed_edge_metrics_v3.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/compute_reviewed_edge_metrics_v3.py)

## 3. 三轮闭环

### Round 1

目标：

- 打通 `v3` 的 selection / build / check 最小链路
- 先验证能否达到 `48` 帧与治理阈值

实际问题与修正：

- `extract_reviewed_edge_keyframes_v3.py` 一开始把所有 artifact 都按 `npz` 读，真实 bundle 中有 `mp4` / `png_seq`
- 多来源 preprocess 之间存在分辨率不一致，直接 `stack` 会失败
- `build_reviewed_edge_benchmark_v3.py` 同样遇到跨来源 shape 不一致问题

修正后结果：

- `selection_v3.json` 成功生成
- `reviewed_edge_benchmark_v3_candidate` 成功 build
- `check_v3.json` 通过

Round 1 summary：

- `total_keyframe_count = 48`
- `strong_revision_fraction = 0.5`
- `category_coverage = {face: 45, hair: 48, hand: 37, cloth: 48, occluded: 48, semi_transparent: 48}`

### Round 2

目标：

- 不再只看“是否 build 成功”，而是验证 `v3` 是否比 `v2` 更有治理价值
- 用同口径 baseline 对比 `v2` 与 `v3`

结果：

- `v2` baseline 指标：
  - `boundary_f1_mean = 0.40098`
  - `alpha_mae_mean = 0.01916`
  - `trimap_error_mean = 0.29244`
  - `hair_boundary_f1_mean = 0.57119`
- `v3` overall baseline 指标：
  - `boundary_f1_mean = 0.74807`
  - `alpha_mae_mean = 0.00572`
  - `trimap_error_mean = 0.04532`
  - `hair_boundary_f1_mean = 0.82732`

这说明：

- `v3` **整体上并不比 `v2` 更难**
- 原因是 `seed_eval` 24 帧来自 `reviewed_v2` 迁移，几乎与当前稳定 baseline 同源
- 所以必须做 split-aware 统计，不能只看 overall mean

### Round 3

目标：

- 冻结正式 `v3_candidate`
- 引入 split-aware 指标，区分 `seed / expansion / holdout`

修正：

- `compute_reviewed_edge_metrics_v3.py` 增加了 `split_metrics`
- 最终冻结目录移动到：
  - [optimization8_step01_round3](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization8_step01_round3)

## 4. 冻结结果

最终 summary：

- `total_keyframe_count = 48`
- `strong_revision_count = 24`
- `strong_revision_fraction = 0.5`

类别覆盖：

- `face = 45`
- `hair = 48`
- `hand = 37`
- `cloth = 48`
- `occluded = 48`
- `semi_transparent = 48`

split 覆盖：

- `seed_eval = 24`
- `expansion_eval = 16`
- `holdout_eval = 8`

这些都满足 Step 01 的硬治理要求。

## 5. baseline 在 v3_candidate 上的结果

使用当前稳定 baseline：

- [preprocess_video_v2/preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess)

overall 指标：

- `boundary_f1_mean = 0.74807`
- `alpha_mae_mean = 0.00572`
- `trimap_error_mean = 0.04532`
- `hair_boundary_f1_mean = 0.82732`
- `semi_transparent_boundary_f1_mean = 0.80100`

split-aware 指标：

- `seed_eval`
  - `boundary_f1_mean = 1.00000`
  - `alpha_mae_mean = 0.00004`
  - `trimap_error_mean = 0.00023`
- `expansion_eval`
  - `boundary_f1_mean = 0.56364`
  - `alpha_mae_mean = 0.00847`
  - `trimap_error_mean = 0.07535`
- `holdout_eval`
  - `boundary_f1_mean = 0.36117`
  - `alpha_mae_mean = 0.01727`
  - `trimap_error_mean = 0.12053`
  - `hair_boundary_f1_mean = 0.47533`
  - `semi_transparent_boundary_f1_mean = 0.57780`

这说明：

- `v3` 的新增部分确实比 seed 难得多
- `holdout_eval` 已经接近甚至局部严于 `v2`
- 真正值得后续训练与模型选择使用的，是：
  - `expansion_eval`
  - 尤其是 `holdout_eval`

## 6. 最终判断

Step 01 的目标已经达成：

- `reviewed_edge_v3_candidate` 正式形成
- 样本量达到 `2x`
- semantic coverage 明显增强
- governance 与 split 正式化
- baseline metrics 与 split-aware metrics 都能稳定跑通

但下一步应带着正确认知进入 `optimization8 Step 02+`：

- 不要再把 `v3 overall mean` 当成主要 gate
- 后续训练型路线应优先盯：
  - `holdout_eval`
  - `expansion_eval`
- 如果未来模型在 `overall` 上看起来很好，但在 `holdout_eval` 上没赢，那仍然不应该升主线
