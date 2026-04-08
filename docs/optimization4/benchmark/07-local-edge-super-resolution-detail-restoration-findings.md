# Optimization4 Step 07 Findings: Local Edge Super-Resolution / Detail Restoration

## 1. 目标

Step 07 要验证的问题是：

- 在 Step 04 的 `roi_v1` 基础上，
- 只对 boundary ROI 做更强的局部 detail restoration，
- 是否能把最终人物边缘的高频细节真正拉起来，
- 同时不明显带坏 halo / face boundary / seam / background。

强 gate 参考 [07-local-edge-super-resolution-detail-restoration.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/07-local-edge-super-resolution-detail-restoration.md)：

- ROI `edge_contrast` 提升 `10%+`
- ROI `band_gradient` 提升 `10%+`
- `halo_ratio` 不恶化
- `face_boundary` 不出现明显伪影


## 2. 本步骤实际交付

### 2.1 新增 local edge restoration 模块

新增：

- [local_edge_restoration.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/local_edge_restoration.py)

当前实现是 deterministic / non-learning 第一版，核心包含：

- ROI 内局部 edge focus map 构建
- 基于 `outer_band / soft_alpha / detail_release / trimap_unknown / edge_detail` 的 focus/gain
- 语义边界感知的局部 gain：
  - `face_boundary`
  - `hair_boundary`
  - `hand_boundary`
  - `cloth_boundary`
  - `occluded_boundary`
- 上采样后局部 detail reinjection：
  - `bilateral base`
  - `original/refined high-frequency residual`
  - `CLAHE luminance delta`
  - `Laplacian boost`
- alpha-aware feather merge 回贴

### 2.2 boundary refine 接线

`boundary_refinement.py` 已新增：

- `boundary_refine_mode=local_edge_v1`

当前路径是：

1. 先沿用 `roi_v1` 的 ROI 提取与 crop 流程
2. ROI 内先跑一层 `semantic_v1` refine 基线
3. 再调用 `restore_local_edge_roi(...)`
4. 最后做 alpha-aware / feather-aware paste-back

接线文件：

- [boundary_refinement.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/boundary_refinement.py)
- [animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)

### 2.3 新增 benchmark / gate

新增脚本：

- [check_local_edge_restoration.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_local_edge_restoration.py)
- [compute_local_edge_restoration_metrics.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/compute_local_edge_restoration_metrics.py)
- [run_optimization4_local_edge_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization4_local_edge_benchmark.py)
- [evaluate_optimization4_local_edge_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization4_local_edge_benchmark.py)


## 3. 测试设置

### 3.1 preprocess bundle

真实 preprocess bundle 使用：

- [optimization4_step06_round1_preprocess/preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step06_round1_preprocess/preprocess)

也就是当前最稳定、artifact 最完整的一套 optimization4 preprocess bundle：

- `boundary_fusion_mode = v2`
- `matting_mode = high_precision_v2`
- `bg_inpaint_mode = video_v2`
- `reference_normalization_mode = structure_match`
- 含 semantic boundary artifact

### 3.2 generate AB 设计

每轮都跑三组：

- `none`
  - `boundary_refine_mode=none`
- `roi_v1`
  - Step 04 的 best boundary path baseline
- `local_edge_v1`
  - 本步骤新增路径

统一参数：

- `frame_num = 45`
- `refert_num = 9`
- `sample_steps = 4`
- `sample_shift = 5.0`
- `boundary_refine_strength = 0.42`
- `boundary_refine_sharpen = 0.24`
- `replacement_conditioning_mode = legacy`
- `output_format = ffv1`

### 3.3 synthetic

- [check_local_edge_restoration.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_local_edge_restoration.py)：PASS


## 4. 三轮闭环

### Round 1

真实 AB：

- [optimization4_step07_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step07_round1_ab)

实现策略：

- deterministic local edge restore 第一版
- 保守 face guard
- 基于 high-frequency residual + CLAHE + Laplacian 的局部细节回注

结果：

- `roi_gradient_gain_pct = +2.0143%`
- `roi_edge_contrast_gain_pct = +0.0630%`
- `roi_halo_reduction_pct = -0.1949%`
- `face_boundary_gradient_gain_pct = +2.1320%`
- `face_boundary_contrast_gain_pct = +0.3676%`
- `face_boundary_delta_ratio = 1.7623`
- `seam_degradation_pct = +0.1526%`
- `background_degradation_pct = +0.1801%`

判断：

- 是三轮里最接近目标的一轮
- ROI gradient 为正
- ROI contrast 为正
- seam/background 安全
- 但幅度极小，远低于 `10%+` gate
- halo 反而轻微恶化
- face delta ratio 过高

### Round 2

真实 AB：

- [optimization4_step07_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step07_round2_ab)

改动：

- 降低 `face` 参与度
- 提高 `hair / cloth / hand` 的 semantic gain
- 减少对 face 的局部细节回注

结果：

- `roi_gradient_gain_pct = +1.4139%`
- `roi_edge_contrast_gain_pct = +0.0403%`
- `roi_halo_reduction_pct = -0.1298%`
- `face_boundary_gradient_gain_pct = +1.3653%`
- `face_boundary_contrast_gain_pct = +0.2866%`
- `face_boundary_delta_ratio = 1.7797`
- `seam_degradation_pct = +0.1041%`
- `background_degradation_pct = +0.1154%`

判断：

- face 改动没有显著被压下去
- ROI gradient / contrast 反而比 Round 1 更弱
- halo 仍然没有变成正向改善

### Round 3

真实 AB：

- [optimization4_step07_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step07_round3_ab)

改动：

- 更强约束在人物内侧边缘
- 更保守 face feather
- 更偏向 `hair / cloth / hand` 的非 face semantic 区域

结果：

- `roi_gradient_gain_pct = +0.7561%`
- `roi_edge_contrast_gain_pct = +0.0166%`
- `roi_halo_reduction_pct = -0.0606%`
- `face_boundary_gradient_gain_pct = +0.8376%`
- `face_boundary_contrast_gain_pct = +0.2311%`
- `face_boundary_delta_ratio = 2.1970`
- `seam_degradation_pct = +0.0602%`
- `background_degradation_pct = +0.0498%`

判断：

- seam/background 是三轮里最安全的
- halo 恶化幅度最小
- 但 ROI gradient / contrast 继续下降
- face delta ratio 反而变得更差


## 5. 最终判断

### 5.1 是否达成成功标准

没有。

强 gate 要求：

- ROI contrast `+10%`
- ROI gradient `+10%`
- halo 不恶化
- face boundary 不明显伪影

best-of-three 结果（Round 1）仍然只有：

- ROI gradient `+2.0143%`
- ROI contrast `+0.0630%`
- halo 轻微恶化
- face delta ratio 明显超出安全目标

所以：

- `local_edge_v1` **没有通过强 gate**

### 5.2 best-of-three 冻结

冻结为 **Round 1 行为**。

原因：

- 是唯一一轮同时让 ROI gradient / ROI contrast 取得三轮内最高正增益的版本
- seam/background 仍然安全
- 虽然仍失败，但比 Round 2 / Round 3 更接近“局部细节增强”的初衷


## 6. 为什么失败

当前观察非常明确：

1. local edge restoration 可以把 ROI gradient 拉成正增益，但幅度很小
2. local detail reinjection 容易：
   - 轻微增大 halo
   - 或把 face boundary 改动拉得过大
3. 当试图更保守地压制 face/background 副作用时：
   - ROI gradient / ROI contrast 又会进一步回落
4. 这说明 deterministic local detail restoration 的上限依然有限：
   - 它更像“局部高频抛光”
   - 不是“真正的边缘高频重建”


## 7. 工程结论

这一步不是无效工作。它至少完成了三件重要的事：

1. 把 `local_edge_v1` 作为正式实验路径接入现有系统
2. 建立了 local edge restoration 的真实 AB / gate 体系
3. 更明确地证明：
   - 仅靠 deterministic local edge restoration
   - 仍不足以把最终边缘锐度显著拉高到目标水平

换句话说：

- Step 07 进一步验证了前面 Step 03 / Step 04 的结论
- 当前真正缺的不是“再来一点 heuristic sharpen”
- 而是更强的、可能带学习能力的边缘重建路径，或更强的 generate-side edge candidate search


## 8. 对后续的指导

Step 07 之后，最合理的方向是：

1. **generate-side edge candidate search**
   - 在已有 `roi_v1 / semantic_v1 / local_edge_v1` 上做短片段多候选
   - 不再执着于手工找单组参数
2. **真正更强的 local SR / learned ROI restore**
   - deterministic v1 的上限已经较清楚
3. **H200 高价值算力分配**
   - 把更多预算花在“候选并行 + 选优”，而不是继续单一路径 heuristic 微调


## 9. 关键运行目录

- Round 1:
  - `runs/optimization4_step07_round1_ab`
- Round 2:
  - `runs/optimization4_step07_round2_ab`
- Round 3:
  - `runs/optimization4_step07_round3_ab`
