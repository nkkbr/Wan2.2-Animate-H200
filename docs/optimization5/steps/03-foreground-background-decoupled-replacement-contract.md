# Step 03: Foreground-Background Decoupled Replacement Contract

## 1. 为什么要做

当前 replacement 主体仍然是整图生成主导。  
这会导致：

- 边缘问题被埋在整图生成里
- 背景质量和人物边缘纠缠
- 后续 ROI 生成与 alpha-aware composite 缺少正式契约

若不先把前景与背景显式解耦，后续更强的 ROI 生成与局部重建也会缺少稳定落点。


## 2. 目标

建立一个正式的 foreground-background decoupled contract，使系统显式输出并消费：

- foreground render / foreground confidence
- clean plate background / visible support / unresolved region
- alpha-aware composite inputs


## 3. 交付目标

1. 新的 preprocess / generate contract
2. foreground / background / alpha artifact schema
3. generate 对 decoupled contract 的读取和 debug
4. metrics 与 findings


## 4. 范围

### In Scope

- 定义 foreground artifact
- 定义 background artifact 及其 confidence
- 定义 composite contract
- decoupled runtime stats / debug

### Out of Scope

- 最终局部生成式 ROI 模型
- 小规模训练


## 5. 实施方案

### 5.1 新 contract 设计

最小正式 contract 至少包括：

- `foreground_rgb`
- `foreground_alpha`
- `foreground_confidence`
- `background_rgb`
- `background_visible_support`
- `background_unresolved`
- `composite_roi_mask`

### 5.2 preprocess 侧生成

preprocess 应输出：

- 用于 generate 的 decoupled bundle
- 用于 debug 的 overlay / confidence / unresolved 可视化

### 5.3 generate 侧消费

generate 应支持两条路径：

- `legacy_integrated`
- `decoupled_v1`

并保证：

- 若 decoupled 路线失败，能安全回退
- seam / runtime / memory 有正式统计


## 6. 如何验证

### 6.1 contract 正确性

- preprocess bundle contract check
- generate 读取与 debug 可用

### 6.2 真实 smoke

- 10 秒真实 benchmark preprocess
- 10 秒真实 benchmark generate

### 6.3 对照指标

相对 integrated baseline 测：

- `seam_score`
- `background_fluctuation`
- `halo_ratio`
- `band_gradient`
- `edge_contrast`


## 7. 强指标要求

目标至少包括：

- seam 不恶化超过 `3%`
- background fluctuation 不恶化超过 `3%`
- halo_ratio 改善 `>= 5%`
- band_gradient / contrast 至少有一项正向改善


## 8. 实现-测评-再改闭环

### Round 1

- 把 decoupled contract 与读取路径做出来
- 跑 preprocess / generate smoke

### Round 2

若出现：

- bundle 不稳定
- generate 无法正常消费
- seam 明显变坏

则调整 contract、composite mask、background confidence 逻辑。

### Round 3

若仍未达 gate，则冻结 best-of-three，明确：

- decoupled contract 是否保留
- 是仅作为 future ROI stage 的基础设施，还是可以直接用于主链


## 9. 风险

- foreground/background 过度解耦可能带来 compositing 感
- unresolved 区域处理不当会让背景显得破碎


## 10. 完成标准

1. foreground-background decoupled contract 已正式落地
2. preprocess / generate / debug / metrics 全贯通
3. 至少一套 decoupled 路线在真实 smoke 上达到不恶化 seam/background 的基本门槛
4. findings 文档明确记录是否可作为后续 ROI 重建基础
