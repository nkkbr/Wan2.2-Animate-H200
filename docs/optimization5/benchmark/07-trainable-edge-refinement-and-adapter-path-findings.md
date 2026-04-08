# Step 07 Findings: Trainable Edge Refinement and Adapter Path

## 1. 结论

Step 07 已执行完成，并按文档要求做了 3 轮“实现 -> 测评 -> 再改”。

最终结论：

- `trainable edge adapter` 路线已经正式跑通
- reviewed edge mini-set 上的训练 / 应用 / GT 指标 / generate smoke 全部贯通
- 但在当前 Step 01 的 reviewed benchmark 上，**该路线没有带来任何正向 edge gain**
- best-of-three 为 **Round 2**
- 该路线不能升默认，只能保留为实验路径


## 2. 本步骤交付

### 2.1 新增模块

- [edge_alpha_adapter.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/edge_alpha_adapter.py)

### 2.2 新增脚本

- [build_edge_adapter_dataset.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/build_edge_adapter_dataset.py)
- [train_edge_alpha_adapter.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/train_edge_alpha_adapter.py)
- [apply_edge_alpha_adapter.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/apply_edge_alpha_adapter.py)
- [check_trainable_edge_adapter.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_trainable_edge_adapter.py)
- [run_optimization5_trainable_edge_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_trainable_edge_benchmark.py)
- [evaluate_optimization5_trainable_edge_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization5_trainable_edge_benchmark.py)

### 2.3 训练/推理能力

该路线现在支持：

- 从 reviewed edge benchmark 构建训练样本
- 训练小型 residual alpha adapter
- 将 adapter 应用到现有 preprocess bundle
- 输出更新后的：
  - `soft_alpha`
  - `person_mask`
  - `trimap_unknown`
  - `alpha_confidence`
  - `hair_alpha`
  - `alpha_source_provenance`
- 跑 GT 指标与真实 generate AB


## 3. 基线观察

最关键的事实来自 reviewed GT benchmark：

- baseline bundle 本身已经几乎与 reviewed labels 完全一致
- baseline 指标：
  - `boundary_f1_mean = 1.0`
  - `alpha_mae_mean = 4.290600039288014e-05`
  - `trimap_error_mean = 9.842863267598052e-04`

这意味着：

- Step 01 的 reviewed mini-set 与现有高精度 baseline 高度同源
- 对 trainable adapter 来说，当前 benchmark 更适合发现“退化”，不适合证明“显著超过 baseline”


## 4. 三轮结果

### 4.1 Round 1

目录：

- [optimization5_step07_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step07_round1_ab)

训练结果：

- `full_dataset boundary_f1_mean = 1.0`
- `full_dataset alpha_mae_mean = 0.0026365320663899183`
- `full_dataset trimap_error_mean = 0.011459303152060726`

gate：

- `boundary_f1_gain_pct = 0.0`
- `alpha_mae_reduction_pct = -6675.0384%`
- `trimap_error_reduction_pct = -1048.7351%`
- `roi_gradient_gain_pct = 0.0`
- `roi_edge_contrast_gain_pct = 0.0`
- `roi_halo_reduction_pct = 0.0`

判断：

- 模型成功学习了一个可运行的 alpha 路径
- 但对当前 reviewed GT 明显退化


### 4.2 Round 2

目录：

- [optimization5_step07_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step07_round2_ab)

训练结果：

- `full_dataset boundary_f1_mean = 1.0`
- `full_dataset alpha_mae_mean = 7.885103696025908e-05`
- `full_dataset trimap_error_mean = 9.997381971134085e-04`

gate：

- `boundary_f1_gain_pct = 0.0`
- `alpha_mae_reduction_pct = -725.0987%`
- `trimap_error_reduction_pct = -2.5807%`
- `roi_gradient_gain_pct = 0.0`
- `roi_edge_contrast_gain_pct = 0.0`
- `roi_halo_reduction_pct = 0.0`

判断：

- Round 2 明显收敛到接近 baseline
- generate 侧 seam/background 保持稳定
- 但没有任何真正正向 edge gain


### 4.3 Round 3

目录：

- [optimization5_step07_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step07_round3_ab)

训练结果：

- `full_dataset boundary_f1_mean = 1.0`
- `full_dataset alpha_mae_mean = 1.425832451786846e-04`
- `full_dataset trimap_error_mean = 0.0011339542581678861`

gate：

- `boundary_f1_gain_pct = 0.0`
- `alpha_mae_reduction_pct = -872.4793%`
- `trimap_error_reduction_pct = -16.0095%`
- `roi_gradient_gain_pct = 0.0`
- `roi_edge_contrast_gain_pct = 0.0`
- `roi_halo_reduction_pct = 0.0`

判断：

- 比 Round 1 稳
- 但弱于 Round 2


## 5. best-of-three 冻结

冻结结果：

- **Round 2**

原因：

- 三轮里最接近 baseline
- 没有引入 seam/background 退化
- 但仍然没有形成正向 edge gain


## 6. 为什么没成功

根因不是训练代码坏了，而是 benchmark 约束决定了这个结果：

1. reviewed mini-set 与 baseline preprocess 高度同源
2. baseline 的 GT 指标已经几乎是满分
3. trainable adapter 在这个 benchmark 上更容易学成“近似 identity”
4. 即使模型学出一点差异，也更容易被记为退化，而不是提升


## 7. 工程判断

这一步的价值在于：

- 首次把 edge 问题真正变成了“可训练路径”
- 训练 / 应用 / GT / generate benchmark 全部贯通
- 后续如果有更强的 reviewed GT 或更难的 failure cases，可以直接复用这条训练路线

但当前不能说它已经解决了边缘问题。更准确的说法是：

- `trainable path` 已建立
- 但在当前 benchmark 上，**没有证明它优于 baseline**


## 8. 最终结论

Step 07 完成，但不通过强 gate。

- 不升默认
- 保留为实验路径
- 如后续继续走训练路线，应优先升级 benchmark 数据，而不是继续在同一 reviewed mini-set 上反复微调
