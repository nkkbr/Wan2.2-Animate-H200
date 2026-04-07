# Step 03: 边界融合、Soft Alpha 与 Uncertainty 系统

## 1. 步骤目标

本步骤的目标是把当前边界系统从：

- `person_mask`
- `soft_band`
- heuristic parsing/matting fusion

升级成真正面向高精度轮廓的多信号边界系统。

最终目标是输出：

- `hard_foreground`
- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`
- `background_keep_prior`


## 2. 为什么要做这一步

### 2.1 当前边缘模糊问题的真正主因还在这里

你最在意的是人物边缘模糊。  
目前已经做过的 Step 03/04 只能说是“改善了边缘脏圈和语义带”，还没有真正进入像素级 alpha 控制。

### 2.2 继续只调 mask 和 soft band 已经边际递减

现在的边界带虽然可用，但高难区域仍然不足：

- 发丝
- 指缝
- 袖口
- 透明/半透明边缘
- 强运动模糊边界

### 2.3 必须引入 uncertainty

如果系统不能告诉 generate“这里边界其实不确定”，generate 就会把低置信边界当成高置信先验，最终产生更明显的错误边缘。


## 3. 本步骤的交付结果

1. 新的边界 artifact 契约
2. 多模型融合的 boundary pipeline
3. uncertainty 输出与 debug
4. 对 alpha 质量和 boundary F-score 的客观评估


## 4. 设计原则

### 4.1 硬主体与软边界严格分离

- `hard_foreground` 只表达高置信主体
- `soft_alpha` 用于边界过渡
- `boundary_band` 用于 refinement focus

### 4.2 uncertainty 是第一等信号

任何 disagreement、时序不稳、局部遮挡，都应反映到 `uncertainty_map`。

### 4.3 occlusion 必须单独建模

手臂遮脸、头发遮肩、袖口遮身体时，如果不显式输出 `occlusion_band`，边界会长期不稳定。


## 5. 代码改动范围

- `wan/modules/animate/preprocess/boundary_fusion.py`
- `wan/modules/animate/preprocess/parsing_adapter.py`
- `wan/modules/animate/preprocess/matting_adapter.py`
- `wan/utils/animate_contract.py`
- `wan/utils/replacement_masks.py`
- 新增 alpha/uncertainty 评测脚本


## 6. 实施方案

### 6.1 子步骤 A：重构 boundary artifact schema

metadata 中要正式引入：

- `soft_alpha`
- `occlusion_band`
- `uncertainty_map`

并为每个 artifact 记录：

- 来源模型
- 融合版本
- 分辨率
- 置信度摘要

### 6.2 子步骤 B：边界多信号融合

输入信号：

- SAM2 时序 mask
- parsing logits / region prior
- matting alpha
- face alpha
- ROI 层局部 refine 结果

输出逻辑：

- 交集高置信区域 -> `hard_foreground`
- 边界一致区域 -> `soft_alpha`
- disagreement 区域 -> `uncertainty_map`
- 明显遮挡区域 -> `occlusion_band`

### 6.3 子步骤 C：双向时序一致性

建议：

- 前向与反向传播都计算 boundary hypothesis
- 对不一致区域提升 uncertainty

### 6.4 子步骤 D：边界 QA 与 crop 评测

要求：

- 导出边界局部 crop
- 导出 uncertainty heatmap
- 导出 alpha vs hard mask 对照


## 7. 目标指标

相对 `optimization2` 基线，本步骤结束时至少达到：

- boundary F-score：提升 `15%`
- trimap error：降低 `20%`
- halo ratio：降低 `30%`
- 不确定性校准：
  - 最高 `20%` uncertainty 区域应覆盖至少 `50%` 的边界误差像素


## 8. 迭代规则

### Round 1

- 打通新 artifact schema
- 生成 `soft_alpha / uncertainty_map / occlusion_band`

### Round 2

- 跑边界指标对照
- 调整融合权重和时序一致性逻辑

### Round 3

- 重点修头发、手部、衣角等高难边界
- 冻结 boundary fusion v2

若 Round 3 后仍达不到 F-score / trimap / halo 目标，则不得进入 generate 消费升级。


## 9. 验证方案

必须执行：

- 边界关键帧 benchmark
- 无损 alpha 输出对照
- uncertainty calibration 统计
- 高难边界 crop review


## 10. 风险与回退

### 风险

- 融合过度复杂导致 debug 困难
- uncertainty 定义不稳定

### 回退

- 保留 `optimization2` heuristic boundary fusion 路径
- 支持 `--boundary_fusion_mode legacy|v2`

