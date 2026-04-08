# Step 04: Boundary ROI Generative Reconstruction V2

## 1. 为什么要做

`optimization4/5` 已经证明：

- deterministic ROI refine 只能“修边”
- 不能稳定把边缘细节重建出来
- gradient、contrast、halo 很难同时变好

因此，下一步必须把边界问题升级成真正的 **局部生成式重建问题**。


## 2. 目标

建立 `roi_gen_v2` 路线，使边界 ROI 不再只是做 sharpen / blend，而是：

- 取出高价值 boundary ROI
- 构造更强条件
- 进行局部生成式恢复 / 重建
- 再高质量 paste-back

这是本轮最有希望直接改善“肉眼边缘锐度”的核心步骤。


## 3. 具体要做什么

### 3.1 ROI 提取

优先覆盖：

- hair / head-shoulder ROI
- face contour ROI
- hand ROI
- cloth motion ROI

### 3.2 ROI 条件

至少使用：

- foreground RGB patch
- background RGB patch
- soft alpha patch
- trimap unknown patch
- uncertainty patch
- visible support patch

### 3.3 ROI 生成器

建议先做一个轻量但真正具有重建能力的 patch 生成/恢复器，而不是再次做 deterministic filtering。

### 3.4 Paste-back

必须是：

- alpha-aware
- confidence-aware
- seam-safe


## 4. 如何做

### 4.1 路径设计

建议：

- `boundary_refine_mode=none`
- `boundary_refine_mode=legacy`
- `boundary_refine_mode=roi_gen_v1`
- `boundary_refine_mode=roi_gen_v2`

让 `roi_gen_v2` 明确成为新主线实验路径。

### 4.2 运行方式

先做：

- 单 ROI patch 重建
- 再做多 ROI 编排
- 再做 paste-back 汇总

### 4.3 debug

必须导出：

- ROI box
- ROI before/after
- alpha before/after
- paste-back preview


## 5. 如何检验

### 5.1 correctness

- ROI 提取正确
- patch 维度正确
- paste-back 不越界

### 5.2 reviewed benchmark

相对 baseline，硬 gate：

- `roi_gradient` 提升至少 `12%`
- `roi_edge_contrast` 提升至少 `10%`
- `halo_ratio` 降低至少 `10%`

### 5.3 real smoke

- `seam_score` 恶化不超过 `3%`
- `background_fluctuation` 恶化不超过 `3%`
- face identity 不明显恶化


## 6. 三轮闭环规则

### Round 1

- 单 ROI reconstruction 路线跑通
- synthetic correctness + 真实 smoke

### Round 2

- 多 ROI 聚合 + reviewed benchmark AB

### Round 3

- 对最优实现做最后一轮收敛
- 冻结 best-of-three


## 7. 成功标准

- 至少一轮通过 `roi_gradient/contrast/halo` 强 gate；
- seam/background 不明显恶化；
- visual inspection 上出现明确等级差异。


## 8. 风险与回退

### 风险

- ROI 重建过强，导致贴图感或伪影；
- paste-back seam 明显；
- ROI 太窄，生成器没有足够上下文。

### 回退

- 若 3 轮后仍未过 gate，则冻结 best-of-three，并转入训练型路线时复用 ROI 框架。
