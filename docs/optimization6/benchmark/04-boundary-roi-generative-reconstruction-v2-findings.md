# Optimization6 Step 04 Findings

## 结论

Step 04 的工程目标已经完成，但质量目标没有达到默认升主线标准。

更准确地说：

- `boundary_refine_mode=roi_gen_v2` 已正式实现并接入 generate；
- ROI 生成式重建 v2 的 offline real AB / gate / synthetic check 已全部补齐；
- 相对 `none`，`roi_gen_v2` 在 representative real frame 上能够稳定抬高 `roi_gradient` 和 `roi_edge_contrast`；
- 但 `halo` 几乎没有下降，`roi_gradient/contrast/halo` 三项仍无法同时过强 gate；
- 因此本步骤正确收口为：
  - 保留 `roi_gen_v2` 代码路径、runner、评测脚本；
  - 不把 `roi_gen_v2` 升为默认边缘路线。

## 本步骤做了什么

### 代码与路径

- `wan/utils/boundary_refinement.py`
- `wan/animate.py`
- `generate.py`

新增：

- `boundary_refine_mode=roi_gen_v2`

并让该模式具备：

- 更轻量、可运行的 ROI 生成候选集；
- alpha-aware / semantic-aware candidate；
- 与 `roi_gen_v1` 不同的尺度与评分策略。

### benchmark / check

新增：

- `scripts/eval/check_boundary_roi_generative_v2.py`
- `scripts/eval/run_optimization6_boundary_roi_benchmark.py`
- `scripts/eval/evaluate_optimization6_boundary_roi_benchmark.py`

## 测试设置

### source preprocess bundle

- `runs/optimization5_step03_round1_preprocess/preprocess`

### before video

- `runs/optimization5_step03_round1_ab/legacy/outputs/legacy.mkv`

### 运行方式

考虑到 `roi_gen_v2` 当前是 CPU 上的高代价原型，本步骤真实 AB 使用：

- representative single-frame real smoke
- 从完整 10 秒 case 中均匀抽取中间关键帧
- reviewed ROI metrics + seam/background safety metrics

这不是最终生产配置，但足以判断当前原型是否有真实方向性收益。

## 三轮结果

### Round 1

目录：

- `runs/optimization6_step04_round1_ab_v3`

指标：

- `roi_halo_reduction_pct = 0.0156%`
- `roi_gradient_gain_pct = 9.9650%`
- `roi_edge_contrast_gain_pct = 10.0385%`
- `seam_degradation_pct = 0.0%`
- `background_degradation_pct = 0.0%`

判断：

- `contrast` 首次转正并超过 `+10%`
- `gradient` 接近 `+12%`，但未过线
- `halo` 几乎无改善

### Round 2

目录：

- `runs/optimization6_step04_round2_ab`

指标：

- `roi_halo_reduction_pct = 0.0270%`
- `roi_gradient_gain_pct = 10.5079%`
- `roi_edge_contrast_gain_pct = 9.8238%`

判断：

- `gradient` 比 Round 1 更高
- `contrast` 略回落，掉到 gate 下方
- `halo` 仍然几乎不动

### Round 3

目录：

- `runs/optimization6_step04_round3_ab`

指标：

- `roi_halo_reduction_pct = 0.0272%`
- `roi_gradient_gain_pct = 11.0890%`
- `roi_edge_contrast_gain_pct = 9.6509%`

判断：

- `gradient` 是三轮最高
- `contrast` 继续低于 `+10%`
- `halo` 依然远未达到 `-10%` 目标

## 冻结结论

best-of-three 选择：

- **Round 1**

理由：

- 它是唯一一轮让 `roi_edge_contrast_gain_pct >= +10%` 的实现；
- `roi_gradient` 也已接近目标；
- `seam/background` 完全安全；
- Round 2/3 虽然略微继续抬高 `gradient`，但 `contrast` 反而回落。

最终结论仍然是：

- **Step 04 gate 失败**
- `roi_gen_v2` 不应升为默认

## 工程判断

这一步最有价值的地方不是“已经成功”，而是更清楚地证明了：

1. ROI 生成式重建比旧的 deterministic ROI 路线更接近目标  
   证据是 `gradient` 与 `contrast` 都能被同时抬正，而 `seam/background` 保持安全。

2. 但当前 inference-only patch candidate 仍不足以解决 `halo`  
   这说明边缘问题已经不只是局部增强或局部候选选择，而更像真正的 alpha / compositing / generation 共同问题。

## 对后续步骤的意义

Step 04 的 ROI 框架现在已经足够稳定，可以直接复用到后续：

- Step 05 trainable edge models
- Step 06 H200 high-value orchestration

如果后续仍沿 ROI 方向推进，重点不该再是：

- 更多 handcrafted candidate
- 更激进的 sharpen

而应转向：

- trainable ROI patch expert
- 更强的 alpha / trimap supervision
- 更强的 foreground/background decoupled generation path
