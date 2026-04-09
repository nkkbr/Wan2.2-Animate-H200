# Step 04: Semantic Boundary Experts Training Path

## 1. 为什么要做

我们已经反复确认：

- `face`
- `hair`
- `hand`
- `cloth`

边界不是同一个问题。

因此，如果 Step 03 的 trainable baseline 有正向信号，下一步值得做的就是：

> 不再用统一模型处理所有边界，而是引入 semantic experts。


## 2. 目标

验证 semantic experts 是否能把 Step 03 的基础训练路线进一步放大，特别是在：

- hair edge
- face contour
- hand edge
- cloth boundary

这些难点上形成更显著提升。


## 3. 要改哪些模块

建议至少新增：

- `wan/utils/semantic_edge_experts.py`
- `scripts/eval/train_semantic_edge_experts.py`
- `scripts/eval/infer_semantic_edge_experts.py`
- `scripts/eval/evaluate_semantic_edge_experts.py`
- `scripts/eval/check_semantic_edge_experts.py`


## 4. 具体如何做

### 4.1 专家形式

可选形式：

- shared backbone + semantic heads
- ROI routing + lightweight experts
- shared trunk + adapters

### 4.2 优先顺序

不建议一开始四类全做。优先：

1. hair expert
2. face contour expert
3. hand expert
4. cloth expert

### 4.3 集成策略

专家输出必须仍然能回到统一 composite contract，而不是各做各的孤立实验。


## 5. 如何检验

### 5.1 semantic metrics

分别统计：

- hair boundary quality
- face contour quality
- hand boundary quality
- cloth boundary quality

### 5.2 overall metrics

不能只看局部类别，仍需看总体：

- `boundary_f1`
- `alpha_mae`
- `trimap_error`
- `roi_gradient`
- `roi_edge_contrast`
- `halo`


## 6. 强指标要求

建议 gate：

- 至少两个 semantic 类别相对 Step 03 baseline 有显著提升
- 总体指标不能明显回退


## 7. 实现-测评-再改闭环

### Round 1

- 先接 1 个最关键 expert（建议 hair）

### Round 2

- 增加第 2 个 expert
- 做 semantic AB

### Round 3

- 冻结 best-of-three expert 配置
- 明确专家路线是否真的有放大价值


## 8. 成功标准

- semantic experts 至少在一类关键边界上带来明确收益
- 且这种收益能部分转化到总体 composite 指标
