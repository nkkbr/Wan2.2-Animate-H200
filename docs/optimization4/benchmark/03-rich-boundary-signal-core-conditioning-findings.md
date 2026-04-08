# Step 03 Findings: Rich Boundary Signal Core Conditioning

## 1. 结论

Step 03 已完成，但结果不是“rich boundary signal 已经显著提升最终边缘锐度”，而是：

- richer boundary signal 已正式接入 generate 主 conditioning 路径
- `legacy` 与 `rich_v1` 的真实 AB benchmark 已建立并跑通
- 3 轮闭环已完成
- **没有任何一轮满足 Step 03 的强 gate**
- 因此本步骤的正确工程收口是：
  - 保留 `rich_v1` 作为实验路径
  - **不升默认**
  - 代码冻结为 best-of-three，对应 Round 1 的行为


## 2. 本步骤实际交付

本步骤完成了以下工作：

1. generate 端已能显式读取并组合 richer boundary signal：
   - `soft_alpha`
   - `alpha_v2`
   - `trimap_v2`
   - `boundary_band`
   - `fine_boundary_mask`
   - `hair_edge_mask`
   - `alpha_uncertainty_v2`
   - `uncertainty_map`
   - `occlusion_band`
   - `background_keep_prior`
   - `face_preserve`
   - `structure_guard`
2. `legacy` / `rich_v1` 双路径并存，可做真实 generate AB
3. 新增 generate benchmark：
   - `run_optimization4_rich_conditioning_benchmark.py`
   - `evaluate_optimization4_rich_conditioning_benchmark.py`
4. runtime stats 现在会记录 `boundary_conditioning_summary`


## 3. 关键代码

- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)
- [wan/animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [wan/utils/replacement_masks.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/replacement_masks.py)
- [wan/utils/rich_conditioning.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/rich_conditioning.py)
- [wan/utils/animate_contract.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/animate_contract.py)
- [scripts/eval/run_optimization4_rich_conditioning_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization4_rich_conditioning_benchmark.py)
- [scripts/eval/evaluate_optimization4_rich_conditioning_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization4_rich_conditioning_benchmark.py)


## 4. 轮次说明

### Round 1

目的：
- 先把 richer boundary signal 接到现有 replacement conditioning 主链

结果：
- 初始尝试暴露了两个真实接口问题：
  - `validate_loaded_preprocess_bundle()` 缺少 `alpha_uncertainty_v2_images`
  - `replacement_masks.py` 的模式路由缺少 `rich_v1`
- 修完入口问题后，第一次有效 AB 是：
  - [optimization4_step03_round1_v3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step03_round1_v3_ab)

Round 1 有效结果：
- `halo_reduction_pct = -0.2569%`
- `gradient_gain_pct = -1.3024%`
- `contrast_gain_pct = -0.1527%`
- `seam_degradation_pct = 1.4831%`
- `background_degradation_pct = 2.0293%`

判断：
- seam / background 仍在阈值内
- halo / gradient / contrast 全部没有改善


### Round 2

目的：
- 确认 `rich_v1` 在更深层的 `soft_band` 主分支上也能走通

结果：
- [optimization4_step03_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step03_round2_ab)
- 实际 boundary 指标与 Round 1 完全一致

解释：
- 这说明前一轮补的那处 mode alias 并不是当前真实 replacement 路径的主瓶颈
- 真正的主路径仍然被 `background_keep_prior` 分支主导


### Round 3

目的：
- 把 richer boundary signal 真正接入 `background_keep_prior` 主分支
- 尝试更窄、更聚焦地释放细边缘区域，而不是继续只扩边界带

结果：
- [optimization4_step03_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step03_round3_ab)

Round 3 指标：
- `halo_reduction_pct = -0.4824%`
- `gradient_gain_pct = -0.5592%`
- `contrast_gain_pct = -0.7226%`
- `seam_degradation_pct = 0.6247%`
- `background_degradation_pct = 2.2578%`

判断：
- 相对 Round 1：
  - seam 更稳
  - gradient 退化更少
  - 但 halo 和 contrast 更差
- 仍未满足任一核心质量 gate


## 5. Best-of-Three 冻结判断

三轮都失败，因此不能升默认。

best-of-three 的冻结依据：
- 以核心边缘质量指标优先：
  - `halo_reduction_pct`
  - `gradient_gain_pct`
  - `contrast_gain_pct`
- seam/background 作为 non-regression 约束

冻结结果：
- **Round 1 为 best-of-three**
- 当前代码已回退到 Round 1 对应的 rich conditioning 行为
- Round 3 的更激进 `background_keep_prior` 改动未被保留


## 6. 为什么这一步没有成功

最重要的结论是：

1. richer boundary signal 的接入本身是成功的
2. 但当前 generate 主链并不能把这些 richer signal 自动转化成更锐利的边缘

更具体地说：

- Step 02 提供的 `alpha_v2 / trimap_v2 / fine_boundary / hair_edge` 确实进入了 generate
- 但现有 replacement conditioning 主要仍然通过：
  - `background_keep`
  - `transition band`
  - `soft alpha`
  这种“约束式”路径起作用
- 这种路径更擅长：
  - 少犯错
  - 控制背景泄漏
  - 维持 seam 稳定
- 不擅长：
  - 提高边缘梯度
  - 提高边缘对比
  - 直接重建更锐的发丝/衣角/手边缘

因此，这一步最有价值的结论不是“rich conditioning 成功”，而是：

**单靠 richer signal 下沉到现有 replacement conditioning 结构，还不足以显著提升最终边缘锐度。**


## 7. 与 Step 04 的关系

Step 03 的结果直接支持后续策略：

- 不应继续在当前这条 `background_keep / transition band` 路径上无限打磨
- 应转入更高价值的下一阶段：
  - boundary ROI 二阶段高分辨率 refine
  - 或更强的局部 edge reconstruction 路径


## 8. 最终结论

Step 03 已完成，但性质是：

- rich boundary signal **接通成功**
- generate 主 conditioning **实验完成**
- 真实 AB 与 gate **已完成**
- **未达到默认启用标准**

正确结论：

1. `rich_v1` 保留为实验路径
2. 默认路径继续保持 `legacy`
3. 继续提升最终边缘锐度，应优先转向 Step 04，而不是继续在当前 Step 03 路线上追加细调
