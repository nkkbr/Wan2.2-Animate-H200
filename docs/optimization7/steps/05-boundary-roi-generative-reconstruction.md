# Step 05: Boundary ROI Generative Reconstruction

## 1. 为什么要做

前几轮已经比较清楚地证明：

- deterministic ROI refine 不够
- local sharpen / detail boost 不够
- semantic rule-based experts 也不够

因此，如果 `optimization7` 要真正尝试打穿边缘问题，必须在这一步把问题升级成：

> 边界 ROI 的局部生成式重建问题。


## 2. 目标

建立一条真正的 ROI generative reconstruction 路径，并验证它是否第一次能让：

- `roi_gradient`
- `roi_edge_contrast`
- `roi_halo`

同时向正确方向移动。


## 3. 要改哪些模块

建议涉及：

- 新增 `wan/utils/roi_generative_reconstruction.py`
- `wan/utils/boundary_refinement.py`
- `wan/animate.py`
- `generate.py`
- 新增 `scripts/eval/check_roi_generative_reconstruction.py`
- 新增 `scripts/eval/run_optimization7_roi_gen_benchmark.py`
- 新增 `scripts/eval/evaluate_optimization7_roi_gen_benchmark.py`


## 4. 具体如何做

### 4.1 定义 ROI reconstruction inputs

建议至少包含：

- ROI foreground patch
- ROI alpha / trimap
- ROI background patch
- ROI unresolved / uncertainty
- semantic ROI tag

### 4.2 生成式路径必须与 compositor 对齐

输出不应直接替换整图，而应通过：

- alpha-aware paste-back
- confidence-aware blending
- semantic-aware fallback

回到主图。

### 4.3 必须保留 deterministic baseline 对照

因为如果没有与 Step 03/04 的 deterministic baseline 对照，无法证明“生成式”这件事本身是否有价值。


## 5. 如何检验

### 5.1 synthetic correctness

- ROI 提取正确
- reconstruction 输出 shape/range 正确
- paste-back 不破坏整图结构

### 5.2 reviewed benchmark

重点看：

- `boundary_f1`
- `trimap_error`
- `alpha_mae`
- `roi_gradient`
- `roi_edge_contrast`
- `roi_halo`

### 5.3 真实 10 秒 AB

必须和：

- legacy stable path
- decoupled baseline
- deterministic ROI baseline

做对照。


## 6. 强指标要求

建议目标：

- `roi_gradient_gain_pct >= 8%`
- `roi_edge_contrast_gain_pct >= 8%`
- `roi_halo_reduction_pct >= 10%`
- `seam_degradation_pct <= 3%`

如果三轮后仍不能同时接近这些目标，应停止继续打磨该版本。


## 7. 实现-测评-再改闭环

### Round 1

- 打通最小 ROI generation path
- 跑 correctness + reviewed 小样本

### Round 2

- 调整 ROI inputs / paste-back / conditioning
- 跑真实 10 秒 AB

### Round 3

- 做最后一次收敛
- 冻结 best-of-three


## 8. 成功标准

- ROI generative reconstruction 正式成形
- 至少一轮出现“gradient/contrast/halo”同时正向的信号
- 真实视频里边缘有肉眼可见提升
