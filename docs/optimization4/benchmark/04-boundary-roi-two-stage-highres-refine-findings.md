# Step 04 Findings: Boundary ROI Two-Stage High-Resolution Refine

## 1. 目标

本步骤要回答的问题是：

- 只在 boundary ROI 上做二阶段高分辨率 refine，
- 是否能比全图 refine 更直接地提升最终边缘锐度，
- 且不明显破坏 seam/background 稳定性。


## 2. 实现内容

本步骤已完成以下工程交付：

- `roi_v1` boundary refine 路径已正式接入 generate
- ROI 提取逻辑已接入：
  - `boundary_band`
  - `soft_alpha`
  - `occlusion_band`
  - `uncertainty_map`
  - `detail_release_map`
  - `trimap_unknown_map`
  - `edge_detail_map`
- ROI 局部高分辨率 refine 与 alpha-aware paste-back 已实现
- ROI 专属 debug artifact 已实现：
  - `roi_mask`
  - `roi_blend_alpha`
  - `roi_boxes.json`
- ROI 专属指标脚本已实现：
  - `scripts/eval/compute_boundary_roi_metrics.py`
  - `scripts/eval/check_boundary_roi_refine.py`
  - `scripts/eval/run_optimization4_boundary_roi_benchmark.py`
  - `scripts/eval/evaluate_optimization4_boundary_roi_benchmark.py`


## 3. 轮次与修改

### Round 1

目标：

- 建立 ROI 提取、局部 refine、回贴和指标链路

结果：

- seam/background 没有明显恶化
- ROI halo 有极小幅改善
- ROI gradient / contrast 仍为负增益

核心结果：

- `roi_halo_reduction_pct = 0.0458%`
- `roi_gradient_gain_pct = -0.9645%`
- `roi_edge_contrast_gain_pct = -1.4007%`
- `seam_degradation_pct = -1.6382%`
- `background_degradation_pct = -0.5816%`


### Round 2

修改：

- 增强 ROI core 保留
- 减少 paste-back 对局部 refined 结果的稀释

结果：

- gradient 从 Round 1 稍有改善
- contrast 更差
- halo 仍几乎不变

核心结果：

- `roi_halo_reduction_pct = 0.0556%`
- `roi_gradient_gain_pct = -0.4881%`
- `roi_edge_contrast_gain_pct = -1.4990%`
- `seam_degradation_pct = -0.8260%`
- `background_degradation_pct = 0.1090%`


### Round 3

修改：

- 不再继续调整 ROI 范围或 paste-back
- 改为在 ROI 内显式回注原始生成结果的高频细节
- 让 `detail_release_map / edge_detail_map / trimap_unknown_map` 真正参与局部增强

结果：

- gradient 首次变成正增益
- 但幅度仍远低于强 gate
- contrast 仍为负增益
- halo 仍基本没有显著变化

核心结果：

- `roi_halo_reduction_pct = 0.0467%`
- `roi_gradient_gain_pct = 1.8744%`
- `roi_edge_contrast_gain_pct = -1.4990%`
- `seam_degradation_pct = 0.6316%`
- `background_degradation_pct = 0.9190%`


## 4. 最终判断

### 4.1 是否达到强指标要求

目标要求：

- ROI 内 `band_gradient` 提升 `8%+`
- ROI 内 `edge_contrast` 提升 `8%+`
- ROI 内 `halo_ratio` 降低 `10%+`
- 全图 `seam_score` 恶化不超过 `3%`

最终结果：

- `band_gradient`：未达标
- `edge_contrast`：未达标
- `halo_ratio`：未达标
- `seam_score`：达标

结论：

- 本步骤 **未通过强 gate**


### 4.2 best-of-three 选择

三轮里最接近目标的是 **Round 3**：

- 是唯一把 ROI `band_gradient` 拉成正增益的一轮
- seam/background 仍在可接受范围内

因此：

- `roi_v1` 当前冻结为 **Round 3 行为**
- 但它只能作为 **实验路径**
- **不能升默认**


## 5. 为什么没有达到预期

当前观察到的主要问题：

1. ROI refine 可以稍微改善局部梯度，但幅度很有限
2. 局部 sharpen 或 detail reinjection 很容易带来：
   - contrast 下降
   - 或对 background pull-back 的抵消不足
3. 当前 `roi_v1` 仍然是 deterministic / semi-deterministic 路线
4. 它更像：
   - 更局部的边界抛光
   - 而不是高质量边缘重建

因此，当前 `roi_v1` 的上限仍然明显受限。


## 6. 工程结论

这一步不是失败的无效工作，它完成了两个重要目标：

1. 证明了 ROI two-stage refine 的链路可以完整接入现有系统
2. 更明确地说明：  
   仅靠当前 deterministic ROI refine，还不足以把边缘锐度显著拉升到目标水平

换句话说，这一步最大的价值是：

- 把 ROI 局部 refine 的工程通路和验证体系建立起来
- 同时排除了“只要局部 refine 一下就能显著变锐”这个过于乐观的假设


## 7. 对下一步的指导

Step 04 之后，最合理的下一步不是继续微调 `roi_v1` 小参数，而是转向：

1. 语义化边界分治
   - `face_boundary`
   - `hair_boundary`
   - `hand_boundary`
   - `body_cloth_boundary`
2. 局部边界超分 / detail restoration
3. 更强的 alpha / matting 主导的局部重建

也就是：

- 从“更聪明的 deterministic ROI 抛光”
- 进入“更强的局部边界重建”


## 8. 关键运行目录

- Round 1:
  - `runs/optimization4_step04_round1_ab`
- Round 2:
  - `runs/optimization4_step04_round2_ab`
- Round 3:
  - `runs/optimization4_step04_round3_ab`

