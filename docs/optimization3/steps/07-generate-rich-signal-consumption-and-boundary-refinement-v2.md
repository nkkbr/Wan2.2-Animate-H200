# Step 07: Generate Rich Signal Consumption 与 Boundary Refinement V2

## 1. 步骤目标

本步骤的目标是让 generate 不再只消费简化的 replacement 条件，而是正式消费：

- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`
- `background_keep_prior`
- `face pose / expression / confidence`
- 结构化 reference normalization 结果

同时，把当前 deterministic boundary refinement 升级成更强的 `Boundary Refinement V2`。


## 2. 为什么要做这一步

### 2.1 preprocess 再精细，如果 generate 不吃这些信号，收益会被浪费

本轮前半部分的升级重点都在 preprocess。  
如果 generate 还是主要依赖简化 mask，那么很多新增信息都传不到最终视频里。

### 2.2 当前 boundary refinement 只能说“有帮助”，还不能说“解决了边缘锐度问题”

当前指标已经证明：

- halo 有下降
- 但 `band_gradient` 和 `edge_contrast` 还没有真正提升

所以需要 V2 版本。

### 2.3 generate 需要知道哪里可以更激进，哪里必须更保守

这正是 `uncertainty_map`、`background_confidence`、`face confidence` 的价值。


## 3. 本步骤的交付结果

1. richer replacement conditioning
2. face geometry-aware generate conditioning
3. uncertainty-aware boundary refinement v2
4. generate 侧新的 benchmark 与 gate


## 4. 设计原则

### 4.1 让 generate 精确消费 richer signals，而不是重新推断这些语义

generate 的职责是消费和融合，而不是在采样阶段重新猜测边界、背景、脸部可信度。

### 4.2 refinement v2 必须在无损输出下客观变好

仅仅“主观上不差”不够。  
必须要求：

- halo 更低
- gradient 更高
- edge contrast 更高

### 4.3 不确定区域处理要更保守

在 uncertainty 高的区域，不应再盲目强化背景 keep 或边界锐化。


## 5. 代码改动范围

- `wan/animate.py`
- `generate.py`
- `wan/utils/replacement_masks.py`
- `wan/utils/boundary_refinement.py`
- 新增 richer conditioning helper
- 新增 generate 侧评测脚本


## 6. 实施方案

### 6.1 子步骤 A：扩展 replacement conditioning

让 `y_reft` 相关逻辑正式引入：

- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`

### 6.2 子步骤 B：扩展 face condition

将 face 分支输入从单纯 face motion 扩展到：

- landmarks
- head pose
- expression
- face confidence

### 6.3 子步骤 C：boundary refinement v2

V2 应增加：

- uncertainty-aware alpha blending
- confidence-aware sharpening
- background confidence aware compositing
- 面向高难边界的局部 adaptive 策略

### 6.4 子步骤 D：新旧路径并行 AB

必须支持：

- `conditioning_mode=legacy|rich`
- `boundary_refine_mode=deterministic|v2`


## 7. 目标指标

相对 `optimization2` 边界精修基线，至少达到：

- halo ratio 再下降 `25%`
- `band_gradient_after >= band_gradient_before * 1.05`
- `band_edge_contrast_after >= band_edge_contrast_before * 1.05`
- seam score 不恶化超过 `5%`

相对当前 generate smoke：

- total runtime 增长不超过 `50%`


## 8. 迭代规则

### Round 1

- 打通 rich conditioning
- V2 refinement 能跑通

### Round 2

- 跑无损 AB
- 优化 uncertainty-aware 规则

### Round 3

- 专攻高难边界和 seam case
- 冻结 refinement v2

若 Round 3 后仍不能同时改善 halo、gradient、contrast，则本步骤失败，不得默认启用 V2。


## 9. 验证方案

必须执行：

- legacy vs rich conditioning AB
- refinement deterministic vs v2 AB
- 无损输出下的边界指标
- seam 对照


## 10. 风险与回退

### 风险

- richer signals 过多导致 conditioning 不稳定
- refinement v2 可能产生过锐化或伪影

### 回退

- 保留当前 deterministic refinement
- 保留 legacy conditioning

