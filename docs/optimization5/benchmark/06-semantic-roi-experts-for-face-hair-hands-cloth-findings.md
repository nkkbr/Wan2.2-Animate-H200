# Step 06 Findings: Semantic ROI Experts for Face / Hair / Hands / Cloth

## 1. 目标

Step 06 的目标是验证：

> 把不同边界类型路由到不同的 ROI expert，是否比统一 ROI / 统一规则更容易真正提升边缘质量。

强 gate 约束：

- 至少满足下面之一：
  - `face` gradient + contrast 均 `>= 8%`
  - `hair` halo 下降 `>= 10%`
  - `hand` edge separation proxy `>= 10%`
  - `cloth` contrast `>= 8%`
- 同时 seam / background 不明显恶化


## 2. 实现内容

本步骤新增：

- `boundary_refine_mode=semantic_experts_v1`
- semantic ROI routing
- expert-specific ROI 参数：
  - `face`
  - `hair`
  - `hand`
  - `cloth`
  - `occluded`

关键实现：

- [boundary_refinement.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/boundary_refinement.py)
- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)
- [animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [check_semantic_roi_experts.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_semantic_roi_experts.py)
- [run_optimization5_semantic_roi_experts_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_semantic_roi_experts_benchmark.py)
- [evaluate_optimization5_semantic_roi_experts_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization5_semantic_roi_experts_benchmark.py)


## 3. 测试设置

- preprocess bundle:
  - `runs/optimization5_step03_round1_preprocess/preprocess`
- generate setting:
  - `frame_num=45`
  - `refert_num=5`
  - `sample_steps=4`
  - `boundary_refine_strength=0.35`
  - `boundary_refine_sharpen=0.15`

真实 AB：

- Round 1:
  - [optimization5_step06_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step06_round1_ab)
- Round 2:
  - [optimization5_step06_round2_v2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step06_round2_v2_ab)
- Round 3:
  - [optimization5_step06_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step06_round3_ab)


## 4. 三轮结果

### Round 1

结果：

- `face`:
  - gradient `+0.0065%`
  - contrast `-0.00035%`
- `hair`:
  - halo reduction `+0.0044%`
- `hand`:
  - gradient `+0.2403%`
- `cloth`:
  - contrast `-0.0180%`
- seam/background:
  - `+0.13% / +0.17%`

判断：

- 路由成功，但专家作用范围过小，几乎没有真正动到语义边界


### Round 2

Round 2 先暴露了实现 bug：

- 3D semantic mask 直接喂给 2D dilation，导致 OpenCV 报错

修复后重跑有效轮次为：

- [optimization5_step06_round2_v2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step06_round2_v2_ab)

有效结果：

- `face`:
  - gradient `+2.4503%`
  - contrast `+0.0811%`
- `hair`:
  - halo reduction `+0.0529%`
- `hand`:
  - gradient `+5.4789%`
- `cloth`:
  - contrast `-0.0536%`
- seam/background:
  - `+0.92% / +1.27%`

判断：

- 扩大 expert ROI 和提高 hair/hand/cloth 强度后，`hand` 和 `face` 的 gradient / contrast 开始出现正向变化
- 但距离强 gate 仍然很远


### Round 3

Round 3 只进一步加强 `hand expert`，不再扩大其它专家：

- `face`:
  - gradient `+2.7012%`
  - contrast `+0.0902%`
- `hair`:
  - halo reduction `+0.0530%`
- `hand`:
  - gradient `+6.5664%`
- `cloth`:
  - contrast `-0.0514%`
- seam/background:
  - `+1.00% / +1.30%`

判断：

- `hand` 是最接近有收益的类别
- 但依然没有任何类别通过强 gate


## 5. 冻结结论

best-of-three：

- **Round 3**

原因：

- 在不破坏全图 seam/background 的前提下，`hand` expert 给出了最高的类别级增益
- 其它类别没有比 Round 2 更糟

但最终结论仍然是：

- **Step 06 gate 失败**
- `semantic_experts_v1` 不应升默认


## 6. 工程判断

这一步最重要的收获不是“已经成功提高边缘锐度”，而是更清楚地证明了：

1. semantic routing 本身是可行的  
   路由没有把全图副作用带坏，说明系统层面成立。

2. 但 inference-only expert 参数化仍然不够强  
   即使把语义 ROI 拆开，当前这版 expert 也不足以把任何一个类别推过强 gate。

3. 最值得继续的类别是 `hand`  
   它是唯一在三轮中持续往正向走的类别。


## 7. 后续建议

Step 06 的结论支持后续优先级：

- 若继续走 inference-only 语义专家路线，性价比已经偏低
- 更有希望的方向是：
  - Step 07 trainable edge refinement / adapter
  - 尤其可以优先从 `hand / hair` 两类难边界开始

