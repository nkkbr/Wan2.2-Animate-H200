# Step 02 Findings: Trainable Task Definition And ROI Dataset Pipeline

## 1. 结论

Step 02 已完成，最终冻结了 `roi_dataset_v2`。

这一步的核心结果不是训练出模型，而是把第二档路线真正需要的训练输入定义清楚了：

- task taxonomy 已正式落地
- ROI sample schema 已正式落地
- reviewed `v3_candidate` -> ROI patch dataset 的构建链路已打通
- `train / val / test` split 已从“直接沿用 reviewed split”修正为“temporal quarantine”
- hard negative 采样已提升到可接受比例

最终冻结目录：

- [roi_dataset_v2](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization8_step02_round3/roi_dataset_v2)

最终判断：

> Step 02 已达成，后续 `optimization8 Step 03 / 04` 不再需要围绕“训练集到底是什么”反复猜测，而可以直接使用 `roi_dataset_v2` 作为正式训练输入。

## 2. 代码与文档交付

新增：

- [task_taxonomy.step02.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/benchmark/task_taxonomy.step02.json)
- [roi_dataset_schema.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/roi_dataset_schema.py)
- [build_roi_training_dataset_v2.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/build_roi_training_dataset_v2.py)
- [check_roi_training_dataset_v2.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_roi_training_dataset_v2.py)

## 3. 三轮闭环

### Round 1

目标：

- 先验证最简单的 ROI builder 是否能跑通
- 直接用 `reviewed_split_v1`

结果：

- builder 成功生成 `514` 个 ROI samples
- 但 checker 明确失败：
  - `Train hard_negative_ratio too low: 0.0941`
  - `Temporal leakage pairs found: 45`

这证明：

- 直接把 `seed_eval / expansion_eval / holdout_eval` 映射成 `train / val / test` 不够
- 也证明 hard negative 不能只做 `1 / frame`

### Round 2

修正：

- split policy 切到 `temporal_quarantine_v1`
- `hard_negative_per_frame` 从 `1` 提到 `2`

结果：

- ROI dataset 成功重建
- checker 通过
- `error_count = 0`
- `leakage_pair_count = 0`
- `train hard_negative_ratio = 0.1698`

这轮已经达到工程 gate。

### Round 3

目标：

- 验证 `Round 2` 不是偶然结果
- 冻结 `best-of-three`

结果：

- 用完全相同的构建参数在新目录重建
- 得到与 Round 2 一致的：
  - `sample_count = 490`
  - `split_counts = {train: 212, val: 119, test: 159}`
  - `error_count = 0`

因此：

- `best-of-three = Round 2/3`
- 正式冻结为 `roi_dataset_v2`

## 4. 冻结结果

最终 `roi_dataset_v2` 的关键统计：

- `sample_count = 490`
- `split_counts`
  - `train = 212`
  - `val = 119`
  - `test = 159`
- `hard_negative_ratio`
  - `train = 0.1698`
  - `val = 0.1681`
  - `test = 0.1761`

task 覆盖：

- `alpha_refinement = 126`
- `matte_completion = 42`
- `boundary_uncertainty_refinement = 42`
- `compositing_aware_edge_correction = 42`
- `semantic_boundary_expert = 238`

semantic 覆盖：

- `mixed_boundary = 168`
- `face = 39`
- `hair = 42`
- `hand = 31`
- `cloth = 42`
- `occluded = 42`
- `semi_transparent = 42`
- `hard_negative = 84`

checker 结果：

- `schema_valid = true`
- `error_count = 0`
- `leakage_pair_count = 0`

## 5. 这一步真正解决了什么

Step 02 解决的不是“边缘质量”，而是三个更根本的问题：

1. 训练任务不再模糊
2. ROI 数据不再是未定义的 patch 混合物
3. 训练 split 不再明显泄漏

这一点对后续 `Step 03` 很关键。过去 trainable 路线很难判断“失败到底是模型不行，还是数据/任务不清”。现在这个歧义已经显著缩小。

## 6. 最终判断

Step 02 的目标已经达成：

- task taxonomy 已正式化
- ROI dataset pipeline 可重复构建
- coverage 与 hard negative 比例达标
- temporal leakage 已控制到 `0`

因此，下一步可以正式进入：

- `optimization8 / Step 03`

也就是在这个更清晰的数据和任务定义之上，重新验证 trainable alpha / matte baseline 是否终于能客观打赢非训练基线。
