# Step 02 Findings: High-Quality Alpha / Matting Upgrade

## 1. 目标

Step 02 的目标是把当前 replacement preprocess 里的 heuristic `soft_alpha` 升级成更高质量、可输出 richer artifact、并最终有机会驱动更锐利边缘的 alpha / matting 系统。

本步骤的最低工程目标包括：

- `alpha_v2`
- `trimap_v2`
- `alpha_uncertainty_v2`
- `fine_boundary_mask`
- `hair_edge_mask`
- `alpha_confidence`
- `alpha_source_provenance`
- metadata / contract / debug / generate 兼容

除此之外，步骤文档还要求在真实 10 秒样本上，尽量实现相对当前 `soft_alpha` 的客观提升，并按最多 3 轮的“实现-测评-再改”闭环执行。


## 2. 本步骤实际交付了什么

本步骤最终交付如下：

1. 新增高精度 alpha refinement 模块：
   - `wan/modules/animate/preprocess/alpha_refinement.py`
2. `matting_adapter.py` 升级为双路径：
   - `heuristic`
   - `high_precision_v2`
3. `boundary_fusion.py` 接入 `refined_hard_foreground`
4. `process_pipepline.py` 接入新 artifact 写出、QA 预览和 metadata 记录
5. `preprocess_data.py` 接入新 CLI 参数：
   - `--matting_mode high_precision_v2`
   - `--alpha_v2_detail_boost`
   - `--alpha_v2_shrink_strength`
   - `--alpha_v2_hair_boost`
   - `--alpha_v2_hard_threshold`
   - `--alpha_v2_bilateral_sigma_color`
   - `--alpha_v2_bilateral_sigma_space`
6. `animate_contract.py` 接入新 artifact semantics
7. 新增评测脚本：
   - `scripts/eval/compute_alpha_precision_metrics.py`
   - `scripts/eval/check_alpha_matting_upgrade.py`


## 3. 关键设计变化

### 3.1 artifact 结构升级

`high_precision_v2` 模式下，现在会额外输出：

- `alpha_v2`
- `trimap_v2`
- `alpha_uncertainty_v2`
- `fine_boundary_mask`
- `hair_edge_mask`
- `alpha_confidence_v2`
- `alpha_source_provenance_v2`

与此同时，旧的 `soft_alpha` artifact 仍然保留，保证旧 bundle 与下游 generate 路径兼容。

### 3.2 从“直接替换 alpha”改为“兼容式 active alpha”

Round 1 证明了一个重要事实：

- 让 `alpha_v2` 直接成为 active `soft_alpha`
- 会在当前真实 benchmark 上明显退化

因此，本步骤最终冻结的策略不是“激进替换”，而是：

- `alpha_v2` 继续完整输出，作为 richer artifact
- active `soft_alpha` 只在细边界区域做小幅、受控、带 gate 的偏移
- 使主链尽量保持与当前 heuristic 路径接近

这也是本步骤最重要的工程收敛结论。


## 4. 测试设置

### 4.1 真实输入

- `ckpt_path`:
  `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- 视频：
  `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- 参考图：
  `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

### 4.2 benchmark

本步骤复用 Step 01 的 edge mini benchmark：

- `runs/optimization4_step01_core_v2/edge_mini_set`

需要明确的一点是：

- 这套 mini-set 是 `bootstrap_unreviewed`
- 标签来源与当前 heuristic 高精度 preprocess 主链非常接近
- 因此这套 benchmark 对“保持与 heuristic 一致”非常敏感，对“超越 heuristic”则不具备充分区分能力

这会直接影响本步骤的 gate 解读。


## 5. 三轮闭环记录

### Round 1

真实 preprocess 目录：

- `runs/optimization4_step02_round1_alpha_v2/preprocess`

结果：

- `boundary_f1_mean = 1.0`
- `trimap_error_mean = 0.04626147278274099`
- `alpha_mae_mean = 0.003308329362577448`
- `alpha_sad_mean = 745.3004404703776`

对比 heuristic baseline：

- `trimap_error` 明显退化
- `alpha_mae / SAD` 明显退化

结论：

- 新 artifact 路径打通了
- 但 `alpha_v2` 太激进，不能作为当前主链策略

### Round 2

真实 preprocess 目录：

- `runs/optimization4_step02_round2_alpha_v2/preprocess`

主要改动：

- 减小 `alpha_v2` 相对 `base_soft_alpha` 的偏移
- 收紧 shrink / detail / hair 增益
- 增加 delta clamp 和边界 gate

结果：

- `boundary_f1_mean = 1.0`
- `trimap_error_mean = 0.02152134881665309`
- `alpha_mae_mean = 0.001091452000158218`
- `alpha_sad_mean = 245.88230895996094`

结论：

- Round 2 大幅收回了 Round 1 的退化
- 但仍明显不如 heuristic baseline

### Round 3

真实 preprocess 目录：

- `runs/optimization4_step02_round3_alpha_v2/preprocess`

主要改动：

- `alpha_v2` 继续完整输出
- active `soft_alpha` 改为：
  - 以 `legacy_soft_alpha` 为主
  - 只在 `fine_boundary / hair_edge / unknown` 区域做小幅、受控偏移

结果：

- `boundary_f1_mean = 1.0`
- `trimap_error_mean = 0.0036338007485028356`
- `alpha_mae_mean = 0.00016447573276915742`
- `alpha_sad_mean = 37.05309303601583`

结论：

- Round 3 明显优于 Round 2
- 并且已经非常接近 heuristic baseline
- 是本步骤的 best-of-three


## 6. 关键客观结果

### 6.1 heuristic baseline

目录：

- `runs/optimization4_step02_round1_heuristic/preprocess`

指标：

- `boundary_f1_mean = 1.0`
- `trimap_error_mean = 0.0009842863267598052`
- `alpha_mae_mean = 4.290600039288014e-05`
- `alpha_sad_mean = 9.665863792101542`

### 6.2 Round 1 vs Round 2 vs Round 3

| 版本 | boundary_f1 | trimap_error | alpha_mae | alpha_sad |
|---|---:|---:|---:|---:|
| heuristic | `1.0` | `0.0009843` | `0.0000429` | `9.6659` |
| alpha_v2 Round 1 | `1.0` | `0.0462615` | `0.0033083` | `745.3004` |
| alpha_v2 Round 2 | `1.0` | `0.0215213` | `0.0010915` | `245.8823` |
| alpha_v2 Round 3 | `1.0` | `0.0036338` | `0.0001645` | `37.0531` |

### 6.3 Round 3 相对 Round 1 / Round 2

Round 3 相对 Round 1：

- `trimap_error` 降低约 `92.1%`
- `alpha_mae` 降低约 `95.0%`
- `alpha_sad` 降低约 `95.0%`

Round 3 相对 Round 2：

- `trimap_error` 进一步降低约 `83.1%`
- `alpha_mae` 进一步降低约 `84.9%`
- `alpha_sad` 进一步降低约 `84.9%`

### 6.4 richer artifact 指标

Round 3 的补充指标：

- `fine_boundary_iou_mean = 0.1500942567675423`
- `trimap_unknown_iou_mean = 0.318450219625581`
- `hair_boundary_overlap_mean = 0.04492100505038585`
- `uncertainty_error_focus_ratio_mean = 1.0`

解释：

- `alpha_v2` richer artifact 已稳定输出并具备可测性
- 但在当前 benchmark 上，这些新 artifact 的价值主要体现在“提供更丰富边界语义”，而不是直接把 active alpha 变得更优


## 7. generate 兼容性

本步骤最终使用 Round 3 bundle 做了真实 generate smoke。

目录：

- preprocess:
  - `runs/optimization4_step02_round3_alpha_v2/preprocess`
- generate smoke:
  - `runs/optimization4_step02_generate_smoke`

输出：

- `runs/optimization4_step02_generate_smoke/outputs/replacement_smoke_none.mkv`

运行结果：

- generate 成功完成
- runtime stats 已落盘：
  - `background_mode = clean_plate_video_v2`
  - `reference_normalization_mode = structure_match`
  - `total_generate_sec = 51.76415343955159`
  - `peak_memory_gb = 39.854`
  - `clip_count = 2`

这说明：

- Step 02 新增的 alpha / trimap / fine boundary / confidence / provenance artifact
- 没有破坏现有 generate 主链


## 8. 为什么本步骤没有“显著打赢 heuristic”

这一步必须明确说明：

1. 当前 Step 01 mini-set 是 `bootstrap_unreviewed`
2. 标签来源与 heuristic 高精度 preprocess 主链高度一致
3. 因此，在这套 benchmark 上：
   - “接近 heuristic” 会得到很高分
   - “偏离 heuristic，即便语义上更丰富” 也会被判成退化

也就是说：

- 这个 benchmark 能很好地发现“退化”
- 但不能可靠地证明“alpha_v2 已经明显优于 heuristic”

因此，本步骤当前最正确的工程结论不是“默认切到 alpha_v2”，而是：

- richer artifact 已经建成
- active `soft_alpha` 已被收敛到与 heuristic 高兼容
- 之后只有在更强人工边缘真值集或 generate-side 真正显示收益时，才有资格考虑升默认


## 9. 是否达成步骤目标

### 已达成

- `alpha_v2` 路径：达成
- `trimap_v2`：达成
- `alpha_uncertainty_v2`：达成
- `fine_boundary_mask / hair_edge_mask`：达成
- metadata / contract / debug 接通：达成
- 与 boundary fusion 主链兼容：达成
- 真实 preprocess smoke：达成
- labeled edge benchmark：达成
- generate smoke 兼容：达成

### 未达成

相对当前 heuristic baseline 的“明显客观提升”：

- 在当前 bootstrap benchmark 上：**未达成**

因此，本步骤不能宣称“高质量 alpha/matting 已经显著优于旧 `soft_alpha`”。


## 10. 最终冻结结论

本步骤最终冻结为：

1. `high_precision_v2` 作为**实验路径**保留
2. `alpha_v2 / trimap_v2 / alpha_uncertainty_v2 / fine_boundary / hair_edge / confidence / provenance` 正式纳入 preprocess artifact 体系
3. active `soft_alpha` 采用 Round 3 的兼容式收敛策略
4. **不升为默认主路径**

最准确的总结是：

- 这一步完成了“高质量 alpha / matting richer artifact 系统”的正式建设
- 但还没有在当前可用 benchmark 上证明“显著优于 heuristic 主链”
- 所以它是重要的基础设施完成，不是最终画质胜利

