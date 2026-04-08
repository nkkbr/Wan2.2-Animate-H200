# Step 07 Findings: Generate Rich Signal Consumption 与 Boundary Refinement V2

## 1. 目标回顾

Step 07 的目标有两部分：

1. 让 generate 正式消费 richer preprocess signals，而不是继续只依赖简化 mask。
2. 让新的 `Boundary Refinement V2` 在无损输出下同时改善：
   - halo
   - band gradient
   - edge contrast
   同时不显著恶化 seam 和 runtime。

本步骤的 gate 来自 [07-generate-rich-signal-consumption-and-boundary-refinement-v2.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization3/steps/07-generate-rich-signal-consumption-and-boundary-refinement-v2.md)：

- halo ratio 再下降 `25%`
- `band_gradient_after >= band_gradient_before * 1.05`
- `band_edge_contrast_after >= band_edge_contrast_before * 1.05`
- seam score 不恶化超过 `5%`
- total runtime 增长不超过 `50%`


## 2. 实施内容

### 2.1 richer generate conditioning 已正式接通

generate 现在已经可以正式读取并消费：

- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`
- `background_keep_prior`
- `face_alpha / face_uncertainty`
- `face bbox / pose / expression` 推导出的 face confidence / preserve map
- 结构化 reference normalization guard

关键代码：

- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)
- [wan/animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [wan/utils/replacement_masks.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/replacement_masks.py)
- [wan/utils/rich_conditioning.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/rich_conditioning.py)

新增 generate 入口：

- `--replacement_conditioning_mode {legacy,rich}`
- `--boundary_refine_mode {none,deterministic,v2}`

其中：

- `legacy` 保留原有 mask-based conditioning 行为
- `rich` 会在 replacement mask 组合阶段引入 richer signals


### 2.2 Boundary Refinement V2 已实现

V2 当前已经是正式可运行实现，主要新增：

- uncertainty-aware blending
- background-confidence-aware compositing
- face-preserve-aware boundary handling
- structure-guard-aware edge refinement
- richer debug artifact 导出

关键代码：

- [wan/utils/boundary_refinement.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/boundary_refinement.py)


### 2.3 generate AB benchmark 与 gate 脚本已补齐

新增脚本：

- [run_generate_rich_signal_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_generate_rich_signal_benchmark.py)
- [evaluate_generate_rich_signal_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_generate_rich_signal_benchmark.py)

说明：

- benchmark 会跑 `legacy/rich × none/deterministic/v2`
- 当前 benchmark 已固定 `--offload_model False`，更符合 H200 路径
- gate 脚本会对 `v2 vs none` 做程序化判定


## 3. 测试设置

### 3.1 输入

- checkpoint:
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B`
- preprocess bundle:
  - [preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess)
- video case:
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`

### 3.2 benchmark 配置

- `frame_num = 45`
- `refert_num = 9`
- `sample_steps = 4`
- `sample_shift = 5.0`
- `output_format = ffv1`
- `offload_model = False`


## 4. 三轮结果

### 4.1 Round 1

目录：

- [optimization3_step07_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step07_round1_ab)

结论：

- richer conditioning 主链跑通
- v2 真实 smoke 跑通
- 但 v2 对 halo / gradient / contrast 的改善非常弱

关键结果：

- `rich_v2`
  - halo reduction: `0.0529%`
  - gradient gain: `-0.9721%`
  - contrast gain: `-1.5757%`


### 4.2 Round 2

目录：

- [optimization3_step07_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step07_round2_ab)

这轮主要改动：

- V2 更强调 `outer_focus` / `edge_focus` 分离
- 增加局部 edge boost
- benchmark 改成 `offload_model=False`

关键结果：

- `rich_v2`
  - halo reduction: `0.0317%`
  - gradient gain: `-0.6966%`
  - contrast gain: `-1.2295%`
  - seam degradation: `0.0149%`
  - runtime increase: `40.93%`

判断：

- runtime/seam 在 gate 范围内
- 但核心质量 gate 仍未达成


### 4.3 Round 3

目录：

- [optimization3_step07_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step07_round3_ab)
- gate:
  - [gate_result.json](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step07_round3_ab/gate_result.json)

这轮主要改动：

- V2 更靠近 deterministic 的 inner-edge sharpen
- edge detail 恢复更强
- 仍保持 outer-focus compositing

关键结果：

- `rich_v2`
  - halo reduction: `0.0293%`
  - gradient gain: `-0.6240%`
  - contrast gain: `-1.1609%`
  - seam degradation: `0.0345%`
  - runtime increase: `48.65%`

- `legacy_v2`
  - halo reduction: `0.0184%`
  - gradient gain: `-0.6187%`
  - contrast gain: `-1.1604%`
  - seam degradation: `-0.0310%`
  - runtime increase: `41.20%`

Gate 结果：

- `overall_v2_passed = false`


## 5. 冻结判断

### 5.1 richer conditioning 路径：保留

理由：

- 主链已稳定跑通
- generate runtime / debug / metrics / manifest 都已接好
- `rich_none` 与 `legacy_none` 相比：
  - seam 没有明显恶化
  - runtime 在 H200 非 offload 路径下也没有不可接受的额外成本

因此：

- `replacement_conditioning_mode=rich` 保留为正式可用选项


### 5.2 Boundary Refinement V2：保留为实验选项，但不得默认启用

这是本步骤最重要的结论。

原因：

- 经过 3 轮真实 AB，`v2` 始终没有同时改善 halo、gradient、contrast
- `v2` 虽然没有明显带坏 seam，也没有超过 runtime gate
- 但其核心目标没有达成

因此冻结判断为：

1. `boundary_refine_mode=v2` 可以保留在代码中继续实验。
2. 当前版本 **不得** 升为默认值。
3. 当前默认策略仍应保持：
   - `boundary_refine_mode=none`
   或在明确人工确认的情况下继续使用 `deterministic`


## 6. 额外观察

### 6.1 deterministic 仍然更像“梯度增强器”

无论 `legacy` 还是 `rich`：

- deterministic 都能明显提高 `band_gradient`
- 但会显著降低 `band_edge_contrast`
- 同时 seam 也略超 5% gate

因此 deterministic 也不能作为 Step 07 的最终成功答案。


### 6.2 当前 proxy 的冲突并不是偶然

当前 boundary proxy 的结果表明：

- 想降低 halo，很容易通过向背景回拉边界来做到
- 但这样往往会牺牲 edge contrast

这说明：

- 仅靠当前像素域 heuristic V2，还不足以把 richer preprocess signals 彻底转化为“边缘更干净且更锐利”的双赢结果

后续更可能成功的方向是：

- 把 `uncertainty / occlusion / face-preserve / background-confidence` 进一步下沉到 generate 主 conditioning，而不只是 boundary postprocess
- 用更显式的 alpha-aware / uncertainty-aware local operator，而不是当前这种单阶段 heuristic compositing


## 7. 最终结论

Step 07 的准确结论是：

- richer signal consumption：**完成**
- generate-side AB benchmark/gate：**完成**
- Boundary Refinement V2 达成默认启用标准：**未完成**

所以本步骤不是“完全成功”，而是：

- 成功建立了 richer-signal generate 主链和正式 benchmark/gate
- 同时客观证明了当前 V2 版本还不够好，不能默认启用

这一步的价值在于：

- 没有把一个“看起来很先进但实际未达标”的 V2 误升为正式默认
- 为 Step 08 之后的进一步 generate-side 精修留下了明确的失败边界和下一步方向
