# Step 02: Trainable Task Definition And ROI Dataset Pipeline

## 1. 为什么要做

前几轮 trainable route 的失败，不是单纯“模型太弱”，而是：

- 训练任务定义模糊
- ROI 样本构造不清晰
- 输入输出语义不稳定
- loss 与最终边缘质量脱节

如果不先把这些问题拆清楚，继续训练只会扩大混乱。


## 2. 目标

建立一套明确的 trainable task taxonomy 和 ROI dataset pipeline，使后续训练不再是“训练一个模糊边缘增强器”，而是训练定义清晰的子任务。


## 3. 推荐任务拆分

建议优先定义以下任务：

1. `alpha_refinement`
2. `matte_completion`
3. `boundary_uncertainty_refinement`
4. `semantic_boundary_expert`
5. `compositing_aware_edge_correction`

每个任务都必须有清晰输入输出。


## 4. 要改哪些模块

建议至少新增：

- `docs/optimization8/benchmark/task_taxonomy.step02.json`
- `scripts/eval/build_roi_training_dataset_v2.py`
- `scripts/eval/check_roi_training_dataset_v2.py`
- `wan/utils/roi_dataset_schema.py`


## 5. 具体如何做

### 5.1 定义统一 ROI sample schema

每个训练样本至少包含：

- `foreground_patch`
- `background_patch`
- `soft_alpha`
- `trimap_unknown`
- `boundary_roi_mask`
- `semantic_boundary_tag`
- `source_case_id`
- `gt_target`

### 5.2 ROI 采样策略

必须明确区分：

- easy samples
- hard negative samples
- high-motion samples
- hair-dominant samples
- hand-dominant samples
- cloth-dominant samples

### 5.3 train/val/test split

split 不应只按帧随机，而应尽量按 case / clip 隔离，避免信息泄漏。


## 6. 如何检验

### 6.1 dataset correctness

- schema 正确
- patch 尺寸正确
- semantic tag 正确
- GT 对齐正确

### 6.2 coverage

每类任务、每类 semantic boundary 都应有基本覆盖。

### 6.3 leakage check

train/val/test 不应大量共享同一 clip 的近邻帧。


## 7. 强指标要求

建议 gate：

- 每个核心任务至少有最小样本量
- 每个 semantic 类别都有 train/val 覆盖
- hard negative 样本占比达到预定阈值


## 8. 实现-测评-再改闭环

### Round 1

- 定义 taxonomy
- 打通 dataset builder

### Round 2

- 根据覆盖问题补样本与 tag
- 跑 correctness + coverage 检查

### Round 3

- 冻结 `roi_dataset_v2`
- 作为 Step 03 / 04 的正式训练输入


## 9. 成功标准

- 训练任务定义清晰
- ROI dataset pipeline 可重复构建
- 训练数据不再是模糊混合物，而是面向具体边缘任务
