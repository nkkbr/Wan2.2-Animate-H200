# Step 01: Human-Reviewed Edge Benchmark Upgrade

## 1. 为什么要做

`optimization5` 已经证明一个关键问题：

- 当前 reviewed edge mini-set 虽然比 `optimization4` 更强，但仍然与历史高质量 baseline 有较强同源性；
- 这会让很多“新方案没有超过 baseline”的结论虽然可信，但不够彻底；
- 尤其是训练型路径和更强 alpha 路径，可能会被当前 benchmark 的偏置压制。

因此，`optimization6` 的第一步必须先把边缘 benchmark 升级成更强、更独立、更适合后续训练与评测的人工审核数据集。

如果这一步不先做，后面所有“边缘真的变锐了吗”的结论都仍然带偏置。


## 2. 目标

建立一套新的 `human-reviewed edge benchmark v2`，满足：

- 至少 `30` 到 `50` 帧人工审核关键帧；
- 标签相对 `optimization5` baseline 更独立；
- 明确覆盖：
  - `soft_alpha`
  - `trimap_unknown`
  - `hair_edge`
  - `face_contour`
  - `hand_edge`
  - `cloth_boundary`
  - `occluded_boundary`
- 可直接被：
  - alpha/matting 路线
  - ROI generative reconstruction 路线
  - trainable edge model 路线
 复用。


## 3. 具体要做什么

### 3.1 数据集升级

基于现有 `optimization5_step01_round3/reviewed_edge_benchmark`，扩成 `reviewed_edge_benchmark_v2`：

- 扩充到 `30-50` 帧
- 必须覆盖以下场景：
  - 头发散开
  - 侧脸
  - 手遮挡脸
  - 衣摆与背景高对比
  - 快速动作
  - 边缘模糊 / 运动模糊
  - 背景纹理复杂

### 3.2 标签体系升级

标签 schema 至少应包含：

- `alpha`
- `trimap_unknown`
- `hair_edge_mask`
- `face_boundary_mask`
- `hand_boundary_mask`
- `cloth_boundary_mask`
- `occluded_boundary_mask`
- `review_status`
- `review_notes`

### 3.3 benchmark 工具升级

需要新增或升级：

- 数据集构建脚本
- reviewed 标签校验脚本
- GT 指标脚本
- validation suite 汇总脚本
- gate policy


## 4. 具体如何做

### 4.1 文件结构建议

建议落地在：

- `docs/optimization6/benchmark/`
- `scripts/eval/`

建议新增：

- `benchmark_manifest.step01.v2.json`
- `label_schema.edge_reviewed_v2.json`
- `gate_policy.step01.v2.json`
- `build_reviewed_edge_benchmark_v2.py`
- `validate_reviewed_edge_labels.py`
- `compute_reviewed_edge_metrics_v2.py`

### 4.2 数据生成流程

1. 从真实 10 秒 benchmark 视频中抽关键帧
2. 结合现有最优 preprocess bundle 生成初始 bootstrap 标签
3. 人工审核和修订
4. 写回 reviewed 标签
5. 运行 label validation
6. 运行 GT metrics smoke

### 4.3 指标

至少统计：

- `boundary_f1_mean`
- `alpha_mae_mean`
- `alpha_sad_mean`
- `trimap_error_mean`
- `hair_edge_quality_mean`
- `face_boundary_quality_mean`
- `hand_boundary_quality_mean`
- `cloth_boundary_quality_mean`


## 5. 如何检验

### 5.1 合法性检查

- 标签文件齐全
- schema 校验通过
- 每一帧都有 reviewed 状态
- 至少 `30` 帧
- 至少 `5` 类高价值边界场景都被覆盖

### 5.2 运行检查

至少跑：

- benchmark build smoke
- label validation
- GT metrics 汇总
- validation suite 汇总

### 5.3 质量 gate

这一阶段不是“必须提升某个旧模型结果”，而是 benchmark 自身要过 gate：

- `frame_count >= 30`
- `reviewed_fraction = 1.0`
- `schema_valid = true`
- `category_coverage >= 5`
- `gate_passed = true`


## 6. 三轮闭环规则

### Round 1

- 搭完整 v2 benchmark 结构
- 至少做 30 帧 reviewed 标签
- 跑 build / validate / metrics smoke

### Round 2

- 修掉 schema、coverage、path、脚本问题
- 增补缺失场景
- 跑完整 gate

### Round 3

- 最后清理 benchmark 偏置和标签不一致问题
- 冻结 `reviewed_edge_benchmark_v2`
- 写 findings


## 7. 成功标准

Step 01 算完成，至少要满足：

- `reviewed_edge_benchmark_v2` 建成；
- `gate_passed = true`；
- 能被后续 Step 02 / 04 / 05 共用；
- findings 文档明确写清：
  - 相比 `optimization5` benchmark 增强了什么；
  - 仍有哪些限制。


## 8. 风险与回退

### 风险

- 人工审核量不足，仍然被 bootstrap 标签污染；
- 新标签 schema 太复杂，反而降低可维护性；
- 只扩帧数，不扩场景多样性。

### 回退

- 若 v2 在第 3 轮后仍不稳定，则冻结 `best-of-three` 版本；
- 不能因为 benchmark 不完美就阻塞整轮，但必须明确写出其限制。
