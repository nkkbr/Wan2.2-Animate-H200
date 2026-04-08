# Step 05 Findings: Boundary ROI Generative Reconstruction

## 1. 目标

Step 05 的目标是验证一条真正区别于 `optimization4` deterministic ROI refine 的路径：

- 先抽取狭窄、高价值的 boundary ROI
- 再在 ROI 内做候选式局部“生成式/重建式”细节恢复
- 最后做 alpha-aware paste-back

强 gate 要求相对 `optimization4` 最优 `legacy` 至少达到：

- ROI `gradient >= +10%`
- ROI `contrast >= +10%`
- ROI `halo <= -10%`
- seam / background 不恶化超过 `3%`


## 2. 实现内容

本步骤新增并接通了：

- `boundary_refine_mode=roi_gen_v1`
- ROI candidate generation / scoring / selection
- ROI benchmark / gate

关键实现：

- [boundary_refinement.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/boundary_refinement.py)
- [run_optimization5_boundary_roi_reconstruction_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_boundary_roi_reconstruction_benchmark.py)
- [evaluate_optimization5_boundary_roi_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization5_boundary_roi_benchmark.py)
- [check_boundary_roi_generative.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_boundary_roi_generative.py)


## 3. 测试设置

- source preprocess bundle:
  - `runs/optimization5_step03_round1_preprocess/preprocess`
- generate setting:
  - `frame_num=45`
  - `refert_num=5`
  - `sample_steps=4`
  - `boundary_refine_strength=0.35`
  - `boundary_refine_sharpen=0.15`

真实 AB 目录：

- Round 1:
  - [optimization5_step05_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step05_round1_ab)
- Round 2:
  - [optimization5_step05_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step05_round2_ab)
- Round 3:
  - [optimization5_step05_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step05_round3_ab)


## 4. 三轮结果

### Round 1

- `roi_halo_reduction_pct = 0.1434%`
- `roi_gradient_gain_pct = 6.2857%`
- `roi_edge_contrast_gain_pct = -1.5404%`
- `seam_degradation_pct = 1.3336%`
- `background_degradation_pct = 1.7644%`

判断：

- 副作用控制较好
- 但锐度提升远未达标


### Round 2

- `roi_halo_reduction_pct = 0.2901%`
- `roi_gradient_gain_pct = 15.9526%`
- `roi_edge_contrast_gain_pct = -1.4942%`
- `seam_degradation_pct = 9.0066%`
- `background_degradation_pct = 9.3867%`

判断：

- ROI gradient 第一次超过 `+10%`
- 但 contrast 仍为负，且 seam/background 明显恶化
- 这说明更激进的 ROI candidate 确实能“抬梯度”，但代价过大


### Round 3

- `roi_halo_reduction_pct = 0.2365%`
- `roi_gradient_gain_pct = 12.9865%`
- `roi_edge_contrast_gain_pct = -1.2457%`
- `seam_degradation_pct = 6.9251%`
- `background_degradation_pct = 7.1324%`

判断：

- 相比 Round 2，副作用有所回收
- 但仍然没有过 seam/background gate
- contrast 仍然为负


## 5. 冻结结论

best-of-three 选择：

- **Round 3**

理由：

- 仍保留了 `gradient > +10%` 的正向收益
- 比 Round 2 的副作用更小
- 比 Round 1 更接近强 gate

但最终结论仍然是：

- **Step 05 gate 失败**
- `roi_gen_v1` 不应升为默认
- 当前 inference-only ROI 生成式重建，在这一版实现下仍不足以稳定解决最终边缘锐度问题


## 6. 工程判断

这一步不是没有价值，而是明确给出了两点：

1. ROI generative reconstruction 方向并非完全无效  
   证据是 Round 2/3 都能把 ROI gradient 拉到 `+10%` 以上。

2. 但当前这版 `roi_gen_v1` 仍然没有解决两个关键问题  
   - edge contrast 仍然为负
   - paste-back 副作用仍然过大

因此，这一步的最准确定位是：

- 完成了 ROI 生成式重建的**系统级原型**
- 但还没有形成可上线的高质量路径


## 7. 后续建议

如果继续沿此方向推进，重点不应再是：

- 更强的 deterministic sharpen
- 更多 handcrafted candidate

而应转向：

- 更强的局部生成式边缘恢复
- 训练式 ROI adapter / refiner
- 更高质量的 alpha / trimap 监督

这也是为什么 `optimization5` 后续更高优先级应放在：

- Step 06 semantic ROI experts
- Step 07 trainable edge refinement

