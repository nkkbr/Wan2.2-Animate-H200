# Optimization8 Step 01 Data Governance

## 1. 目标

`reviewed_edge_v3` 不是简单扩容旧 `v2`，而是建立一套更强的数据治理约束，使后续训练与评测不再过度依赖旧 baseline 的同源标签。

## 2. 数据版本规则

- `v2`
  - `optimization6` 阶段的 reviewed benchmark
  - 允许旧 reviewed 标签迁移
- `v3_candidate`
  - `optimization8 Step 01` 生成的数据版本
  - 允许：
    - `v2` 种子帧迁移
    - 新增关键帧的 multi-source consensus bootstrap
  - 不等于 fully human redrawn GT

## 3. revision_strength 定义

- `seed_migrated`
  - 直接来自 `v2` reviewed 帧迁移
- `strong_bootstrap_extension`
  - 非 `v2` 种子帧
  - 由多 source preprocess 共识、差异图和 semantic mask 一起生成
- `manual_review_required`
  - 标记为后续需要人工强修的高难度样本

`Step 01` 的最低要求：

- `strong_bootstrap_extension` 样本比例至少 `30%`

## 4. split 定义

- `seed_eval`
  - 来自 `v2` 种子集
- `expansion_eval`
  - 本轮新增帧
- `holdout_eval`
  - 新增帧中 difficulty 更高的一部分，保留给后续更强路线做对照

## 5. 初始自动标签来源要求

每帧必须记录：

- `primary_preprocess`
- `consensus_preprocess_dirs`
- `seed_reviewed_dataset_dir`

## 6. 独立性要求

`v3_candidate` 允许使用旧 baseline 辅助，但不允许仅复制旧 reviewed 标签：

- 新增帧必须来自 `v2` 之外的新 keyframe
- 新增帧标签必须至少融合 `2` 个以上 source preprocess
- semantic boundary 必须重新生成，不允许只继承旧 boundary type map

## 7. Stop Rule

如果本轮 `v3` 扩容后仍然：

- 样本量 < `2x v2`
- `strong_bootstrap_extension` 比例 < `30%`
- semantic coverage 明显不足

则不得进入 `optimization8 Step 02`。
