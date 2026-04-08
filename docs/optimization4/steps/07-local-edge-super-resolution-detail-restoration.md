# Step 07: Local Edge Super-Resolution / Detail Restoration

## 1. 目的

本步骤的目标是解决这样一种剩余问题：

即使边界位置和 alpha 已经更准了，人物轮廓仍然可能缺少足够的高频细节，看起来仍然不够锐。


## 2. 为什么要做

边缘模糊并不总是因为 mask 错。  
很多时候是：

- 高频细节丢失
- 局部纹理不够
- 头发丝和衣角缺少清晰小结构

这时只做 compositing 和 alpha refine 还不够，需要局部 detail restoration。


## 3. 交付目标

本步骤结束时，应具备：

1. boundary ROI local detail restoration
2. 可选 local edge super-resolution
3. alpha-aware / uncertainty-aware 回贴


## 4. 范围

可新增：

- `wan/utils/local_edge_restoration.py`
- `scripts/eval/compute_local_edge_restoration_metrics.py`

并接入：

- `boundary_refinement.py`
- `animate.py`


## 5. 实施方案

### 5.1 先做 deterministic / non-learning 版

第一版可先尝试：

- local contrast enhancement
- local detail boost
- edge-aware filtering

### 5.2 再探索更强版本

若 deterministic 版有效果但不够强，可再考虑：

- local SR
- ROI patch restoration model

### 5.3 严禁全图无差别锐化

必须只在边界 ROI 内启用，并受：

- alpha
- uncertainty
- occlusion
- face preserve

约束。


## 6. 强指标要求

相对 Step 04/05 最优边界路径：

- ROI `edge_contrast` 提升 **10% 以上**
- ROI `band_gradient` 提升 **10% 以上**
- `halo_ratio` 不恶化
- `face_boundary` 不出现明显伪影


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 先做 deterministic local edge restore

### Round 2

- 局部调优 hair / cloth ROI

### Round 3

- 决定是否需要更强 SR / learned model


## 8. 风险

### 8.1 产生假边或 ringing

应对：

- alpha / uncertainty 约束
- face 区域单独限幅

### 8.2 局部增强与全图生成风格不一致

应对：

- 只做 narrow-band ROI
- 使用 feather merge


## 9. 完成标准

只有在局部边缘 detail 明显增强且没有显著副作用时，本步骤才应升为默认。
