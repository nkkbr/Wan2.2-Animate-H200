# Step 05: Trainable Edge Models on Reviewed Data

## 1. 为什么要做

如果前面几步已经建立了：

- 更强的 reviewed benchmark
- 更强的外部 alpha / matting
- decoupled contract
- ROI generative reconstruction 原型

那么下一步才值得引入训练型边缘模型。

这一步不能再重复 `optimization5` 的问题：

- 不能只拟合旧 baseline
- 不能在弱 benchmark 上自我感觉良好


## 2. 目标

引入第一条真正有意义的训练型边缘模型路径，例如：

- edge alpha refinement model
- hair-edge refinement model
- ROI boundary patch expert

目标不是“训练出一个模型”，而是：

> 在 reviewed benchmark 上客观优于 heuristic / deterministic 路线。


## 3. 具体要做什么

### 3.1 数据准备

利用 Step 01 的 reviewed benchmark 和 Step 02 的高质量 alpha artifact，准备：

- train / val / holdout split
- edge patch dataset
- ROI patch dataset

### 3.2 模型定义

优先选择：

- 小而强
- 易部署
- 可在 H200 上快速训练 / 迭代

不要一开始就做过大的模型。

### 3.3 训练与推理接口

需要新增：

- dataset builder
- trainer
- inference adapter
- artifact writer


## 4. 如何做

### 4.1 训练目标

可考虑联合目标：

- alpha regression loss
- trimap / boundary classification loss
- ROI contrast-aware loss
- hair-edge emphasis loss

### 4.2 评测方式

必须同时看：

- reviewed benchmark GT 指标
- 真实 10 秒 smoke

不能只看训练 loss。


## 5. 如何检验

### 5.1 GT gate

相对当前最佳 deterministic / heuristic baseline：

- `boundary_f1` 提升至少 `5%`
- `alpha_mae` 降低至少 `10%`
- `trimap_error` 降低至少 `10%`
- `hair_edge_quality` 提升至少 `8%`

### 5.2 real smoke

- `roi_gradient` 提升至少 `8%`
- `roi_edge_contrast` 提升至少 `8%`
- `halo_ratio` 降低至少 `8%`


## 6. 三轮闭环规则

### Round 1

- 建 dataset + baseline trainer
- 跑最小可训练闭环

### Round 2

- 改 loss / architecture / target
- reviewed benchmark AB

### Round 3

- 对最优方案做最后一轮收敛
- 冻结 best-of-three


## 7. 成功标准

- 至少一个训练型模型在 GT 指标上客观超过所有非训练路径；
- 真实 smoke 不明显恶化 seam / background / face identity；
- 可稳定推理、可回退。


## 8. 风险与回退

### 风险

- 过拟合 reviewed mini-set
- 只学会拟合当前 preprocess baseline
- 训练型模型太重，不适合生产

### 回退

- 若 3 轮后不达标，则训练路径保留为研究支线，不进入主线。
