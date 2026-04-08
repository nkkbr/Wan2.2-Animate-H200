# Step 04 Findings: Rich Boundary Signal Core Generation Conditioning

## 1. 目标

本步骤的目标不是继续做弱约束 mask shaping，而是验证：

- richer boundary signal 是否能以更强的位置进入 generate 主条件
- 是否能在真实 benchmark 上同时改善：
  - `halo`
  - `gradient`
  - `contrast`
- 并保持 seam / background 基本稳定


## 2. 实现内容

已完成：

- 新增 `replacement_conditioning_mode=core_rich_v1`
- richer boundary signal 由单纯 mask shaping 升级为 core conditioning RGB 路线
- `bg_pixel_values` 在 `core_rich_v1` 下不再只是 clean plate background，而是：
  - background
  - foreground edge hint
  - trimap / uncertainty / hair / face preserve
  重编码后的核心条件帧

关键代码：

- [rich_conditioning.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/rich_conditioning.py)
- [replacement_masks.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/replacement_masks.py)
- [animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)
- [run_optimization5_core_conditioning_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_core_conditioning_benchmark.py)
- [evaluate_optimization5_core_conditioning_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization5_core_conditioning_benchmark.py)


## 3. 真实 benchmark 配置

- preprocess bundle:
  [optimization5_step03_round1_preprocess/preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step03_round1_preprocess/preprocess)
- generate 对照：
  - `legacy`
  - `rich_v1`
  - `core_rich_v1`
- `frame_num=45`
- `refert_num=5`
- `sample_steps=4`


## 4. 三轮结果

### Round 1

目录：

- [optimization5_step04_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step04_round1_ab)

结果：

- `halo_reduction_pct = +18.7015%`
- `gradient_gain_pct = +10.6848%`
- `contrast_gain_pct = -52.6698%`
- `seam_degradation_pct = +187.2624%`
- `background_degradation_pct = +46.8696%`

判断：

- `core_rich_v1` 终于不再是“完全没反应”
- 但它显著带坏了 seam 和 background


### Round 2

目录：

- [optimization5_step04_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step04_round2_ab)

调整：

- 降低 foreground 注入权重
- 缩窄 core conditioning 的 ROI 影响范围

结果：

- `halo_reduction_pct = +9.2690%`
- `gradient_gain_pct = +27.9969%`
- `contrast_gain_pct = -39.1208%`
- `seam_degradation_pct = +140.1121%`
- `background_degradation_pct = +38.6105%`

判断：

- 质量主项里 halo / gradient 仍为正
- 但副作用仍然远高于允许范围


### Round 3

目录：

- [optimization5_step04_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step04_round3_ab)

调整：

- 进一步收窄 core conditioning 混合权重
- 让 `core_rich_v1` 向 legacy 邻域收敛

结果：

- `halo_reduction_pct = -0.0468%`
- `gradient_gain_pct = +24.5899%`
- `contrast_gain_pct = -7.0861%`
- `seam_degradation_pct = +33.9054%`
- `background_degradation_pct = +21.0656%`

判断：

- 副作用显著收敛
- 但仍然没有压到 gate 内
- contrast 仍未转正


## 5. 强 gate 结果

目标：

- `halo >= +8%`
- `gradient >= +8%`
- `contrast >= +8%`
- seam/background 不恶化超过 `3%`

结论：

- Round 1：失败
- Round 2：失败
- Round 3：失败


## 6. 冻结判断

best-of-three：

- **Round 3**

原因：

- 它最接近“可生产化的保守版本”
- 对比 Round 1 / 2，副作用最小
- 虽然仍未过 gate，但最适合作为后续更强路线的参考实现

是否升默认：

- **不升默认**


## 7. 工程结论

Step 04 给出的关键信息是：

1. richer boundary signal 进入更强的位置后，确实能对 halo / gradient 产生影响
2. 但当前 inference-only core conditioning 路线仍然很难兼顾：
   - edge gain
   - seam stability
   - background stability
3. 这说明：
   - richer signal 不是没有价值
   - 但单靠条件重编码，已经很难继续突破

因此，后续更值得投入的方向应是：

- ROI 生成式重建
- semantic ROI experts
- trainable edge refinement / adapter
