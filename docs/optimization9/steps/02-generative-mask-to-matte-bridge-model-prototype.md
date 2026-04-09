# Step 02: Generative Mask-To-Matte Bridge Model Prototype

## 1. 为什么要做

第三档里最值得最先验证的路线，是：

> 在主链和最终 composite 之间插入一个更强的生成式 `mask-to-matte / matte bridge model`。

原因：

- 它最贴近现有系统
- 不需要立刻推翻整个主 backbone
- 最容易在当前 benchmark 上验证


## 2. 目标

验证一个 bridge model 原型是否能把：

- coarse mask / soft alpha / trimap
- foreground/background patch

转成更高质量的 matte 或边界 patch，并表现出不同于旧 heuristic 路线的错误模式。


## 3. 要改哪些模块

建议新增：

- `wan/utils/matte_bridge_model.py`
- `scripts/eval/check_matte_bridge_model.py`
- `scripts/eval/run_optimization9_bridge_benchmark.py`
- `scripts/eval/evaluate_optimization9_bridge_benchmark.py`


## 4. 具体如何做

### 4.1 定义输入输出

输入建议至少包含：

- mask / soft_alpha / trimap
- foreground patch
- background patch
- uncertainty / unresolved band

输出：

- refined matte
- 或 reconstructed boundary patch
- 以及 confidence / provenance

### 4.2 原型要求

先做最小 bridge prototype：

- 不求整图可用
- 只求 boundary ROI 小样本可用

### 4.3 与现有系统对齐

bridge model 输出必须能回到：

- decoupled contract
- alpha-aware compositor
- ROI metrics stack


## 5. 如何检验

### 5.1 reviewed benchmark

重点看：

- `boundary_f1`
- `alpha_mae`
- `trimap_error`
- `roi_gradient`
- `roi_edge_contrast`
- `roi_halo`

### 5.2 error mode

必须重点判断：

- 是否第一次避免了旧的 tradeoff 模式
- 是否至少在某类困难边界上出现质变


## 6. 强指标要求

本步骤的目标不一定是全面超过主线，但应至少满足其一：

- 在某一类困难边界上显著提升
- 或出现明显不同的正向错误模式

如果 3 轮后仍只是重现旧模式，应停止。


## 7. 实现-测评-再改闭环

### Round 1

- 打通最小 bridge prototype

### Round 2

- 调整输入组合和输出语义
- 跑 reviewed + smoke AB

### Round 3

- 冻结 best-of-three
- 判断是 `reject`、`interesting_but_unproven` 还是 `upgrade_candidate`


## 8. 成功标准

- 至少出现一个明确不同于旧 heuristic 路线的正向信号
- 或在某类困难边界上第一次表现出质变
