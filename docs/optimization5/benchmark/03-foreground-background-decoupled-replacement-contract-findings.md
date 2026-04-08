# Step 03 Findings: Foreground-Background Decoupled Replacement Contract

## 1. 目标

本步骤的目标不是直接把边缘锐度做出来，而是把后续 ROI 生成式重建需要的显式契约正式落地：

- `foreground_rgb`
- `foreground_alpha`
- `foreground_confidence`
- `background_rgb`
- `background_visible_support`
- `background_unresolved`
- `composite_roi_mask`

并在 generate 侧验证：

- 是否能稳定读取和消费这些 artifact
- 是否能在不明显带坏 seam / background 的前提下得到至少一点边界正收益


## 2. 交付内容

已完成：

- preprocess metadata / contract 扩展，支持 foreground-background decoupled bundle
- preprocess 主流程新增 decoupled artifact 写出
- generate 新增 `replacement_conditioning_mode=decoupled_v1`
- debug / runtime stats 支持 decoupled contract 可用性记录
- Step 03 真实 AB benchmark、selection 与 gate 脚本

关键文件：

- [process_pipepline.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/process_pipepline.py)
- [animate_contract.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/animate_contract.py)
- [replacement_masks.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/replacement_masks.py)
- [animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)
- [run_optimization5_decoupled_contract_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_decoupled_contract_benchmark.py)
- [evaluate_optimization5_decoupled_contract_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization5_decoupled_contract_benchmark.py)


## 3. 真实运行配置

- source video:
  `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- reference image:
  `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- preprocess bundle:
  [optimization5_step03_round1_preprocess/preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step03_round1_preprocess/preprocess)
- generate benchmark:
  `45` frames
  `refert_num=5`
  `sample_steps=4`
  `solver=dpm++`


## 4. 三轮结果

### Round 1

目录：

- [optimization5_step03_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step03_round1_ab)

结果：

- `halo_reduction_pct = +99.2570%`
- `gradient_gain_pct = -80.7952%`
- `contrast_gain_pct = -99.1665%`
- `seam_degradation_pct = -52.8976%`
- `background_degradation_pct = -67.6706%`

判断：

- 边界梯度和对比几乎被压平
- 说明 decoupled contract 的初版消费逻辑过强


### Round 2

目录：

- [optimization5_step03_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step03_round2_ab)

调整：

- 仅下调 `decoupled_boundary_strength`

结果：

- `halo_reduction_pct = +99.2467%`
- `gradient_gain_pct = -80.7358%`
- `contrast_gain_pct = -99.1498%`

判断：

- 与 Round 1 基本一致
- 说明问题不是单纯参数太大，而是 decoupled 消费逻辑本身过强


### Round 3

目录：

- [optimization5_step03_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step03_round3_ab)

调整：

- 不再用 foreground alpha 覆盖 legacy soft alpha
- `composite_roi_mask` 只作为 very-light release hint
- decoupled release 强度大幅收窄到 legacy 邻域

结果：

- `halo_reduction_pct = -0.0683%`
- `gradient_gain_pct = -0.0069%`
- `contrast_gain_pct = -0.1311%`
- `seam_degradation_pct = +0.1609%`
- `background_degradation_pct = +0.4985%`

判断：

- seam / background 已经稳定在可接受范围内
- 但边缘核心三项仍然没有形成正收益


## 5. Gate 结果

Step 03 强 gate：

- seam 不恶化超过 `3%`
- background fluctuation 不恶化超过 `3%`
- halo 改善 `>= 5%`
- gradient / contrast 至少一项正向

最终：

- Round 1: 失败
- Round 2: 失败
- Round 3: 失败

失败原因：

- Round 1 / 2: decoupled 路径过强，导致边界对比被抹平
- Round 3: 已足够保守，但边界仍未形成正向改善


## 6. 最终冻结判断

best-of-three 选择：

- **Round 3**

原因：

- 是唯一一个在 seam / background 基本不恶化的前提下运行稳定的版本
- 质量上接近 legacy，不会明显破坏主链
- 最适合作为 future ROI generative reconstruction 的基础契约版本

最终结论：

- `decoupled_v1` **保留**
- 但定位为：
  - **基础设施 / contract 路线**
  - **不是主链质量升级路线**
- **不升默认**


## 7. 工程判断

Step 03 的价值不是直接提升边缘锐度，而是证明了：

1. foreground-background decoupled bundle 可以稳定落地
2. generate 可以安全读取并消费它
3. 当前这种 low-risk decoupled consumption 本身，不足以直接提升最终边缘质量

这为后续步骤提供了重要结论：

- 若要真正提升边缘锐度，必须进入：
  - ROI 生成式重建
  - 语义专家
  - 训练式 edge refinement

而不是继续在当前 `decoupled_v1` 上做小参数打磨。
