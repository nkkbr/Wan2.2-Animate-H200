# Step 04: Boundary ROI Two-Stage High-Resolution Refine

## 1. 目的

本步骤的目标是建立只针对 boundary ROI 的二阶段高分辨率 refine 流程。


## 2. 为什么要做

当前最大的问题之一是：

- 全图 refine 太保守，容易一起把边缘变软
- 真正关键的是那一圈人物轮廓，不是整图

所以最有希望直接拉升“视觉锐度”的路线是：

1. 第一阶段先正常生成整图
2. 第二阶段只对边缘 ROI 做高分辨率修复


## 3. 交付目标

本步骤结束时，应具备：

1. `boundary_roi_refine` 路径
2. ROI 提取与回贴逻辑
3. ROI 级 metrics
4. 与现有 generate 主链兼容


## 4. 范围

优先修改：

- `wan/utils/boundary_refinement.py`
- `wan/animate.py`
- 可能新增：
  - `wan/utils/boundary_roi_refine.py`
  - `scripts/eval/compute_boundary_roi_metrics.py`


## 5. 实施方案

### 5.1 ROI 定义

边界 ROI 不应只由 `boundary_band` 单独决定。  
建议联合：

- `boundary_band`
- `soft_alpha`
- `occlusion_band`
- `uncertainty_map`
- `hair_boundary`
- `hand_boundary`
- `face_boundary`

### 5.2 ROI refine 策略

第一版建议从 deterministic / semi-deterministic 开始：

- ROI crop
- 高分辨率 local sharpen / local recompose / local restoration
- alpha-aware merge back

第二版再考虑：

- ROI 小模型生成
- 局部 patch diffusion

### 5.3 回贴规则

回贴时必须：

- alpha-aware
- structure-aware
- face-preserve-aware
- background-consistency-aware


## 6. 强指标要求

相对当前最优全图路径：

- ROI 内 `band_gradient` 提升 **8% 以上**
- ROI 内 `edge_contrast` 提升 **8% 以上**
- ROI 内 `halo_ratio` 降低 **10% 以上**
- 全图 `seam_score` 恶化不超过 **3%**


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 建立 ROI 提取与 debug
- 先验证 ROI 指标是否能正向

### Round 2

- 优化 ROI 合成
- 处理 hair / hand 难边界

### Round 3

- 固定最优 ROI 策略
- 决定是否升默认


## 8. 如何验证

1. synthetic ROI edge case
2. real 10s smoke
3. ROI metrics
4. full-frame seam/background regression


## 9. 风险

### 9.1 ROI 回贴边缘断层

应对：

- alpha-aware blend
- ROI 扩边与 feathering

### 9.2 hair ROI 与 face ROI 冲突

应对：

- semantic priority map


## 10. 完成标准

只有当 ROI 指标有显著正提升，并且全图没有明显副作用，本步骤才可升为正式路径。
