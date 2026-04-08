# Optimization4 Step 06 Findings: Semantic Boundary Specialization

## 1. 目标

Step 06 的目标是验证一件更具体的事：

- 不再把所有边界都当成同一种问题处理
- 为 `face / hair / hand / cloth / occluded` 引入独立语义边界 mask
- 在 generate 的 replacement conditioning 和 pixel-domain boundary refinement 里，按类别采用不同策略

强 gate 参考 [06-semantic-boundary-specialization.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/06-semantic-boundary-specialization.md)：

- `face_boundary`：gradient / contrast 至少提升 `8%`
- `hair_boundary`：halo 至少下降 `10%`
- 若 hair 数据不足，可由 `hand_boundary` 替代其中一类


## 2. 本步骤实际交付

### 2.1 preprocess 侧

新增正式 semantic boundary artifact：

- `src_face_boundary`
- `src_hair_boundary`
- `src_hand_boundary`
- `src_cloth_boundary`
- `src_occluded_boundary`

构造逻辑在：

- [boundary_fusion.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/boundary_fusion.py)

主流程与 metadata 接线在：

- [process_pipepline.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/process_pipepline.py)
- [animate_contract.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/animate_contract.py)

### 2.2 generate 侧

新增：

- `replacement_conditioning_mode=semantic_v1`
- `boundary_refine_mode=semantic_v1`

接线位置：

- [generate.py](/home/user1/wan2.2_20260407/Wan2.2/generate.py)
- [animate.py](/home/user1/wan2.2_20260407/Wan2.2/wan/animate.py)
- [replacement_masks.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/replacement_masks.py)
- [boundary_refinement.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/boundary_refinement.py)

当前 specialization 策略概括如下：

- `face_boundary`
  - 更强 preserve
  - 更积极 sharpen / edge boost
- `hair_boundary`
  - 更保守 sharpen
  - 尝试减轻 background keep
- `hand_boundary`
  - 更积极 detail release / sharpen
- `cloth_boundary`
  - 允许更强 edge restore
- `occluded_boundary`
  - 更保守，避免错误拉锐

### 2.3 benchmark 与 gate

新增脚本：

- [check_semantic_boundary_specialization.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_semantic_boundary_specialization.py)
- [compute_semantic_boundary_metrics.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/compute_semantic_boundary_metrics.py)
- [run_optimization4_semantic_boundary_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization4_semantic_boundary_benchmark.py)
- [evaluate_optimization4_semantic_boundary_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization4_semantic_boundary_benchmark.py)


## 3. 测试设置

### 3.1 真实 preprocess bundle

真实 bundle 路径：

- [optimization4_step06_round1_preprocess/preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step06_round1_preprocess/preprocess)

输入：

- `video`: `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- `reference`: `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- `ckpt_path`: `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`

主要 preprocess 配置：

- `analysis_resolution_area = 1280x720`
- `resolution_area = 832x480`
- `fps = 5`
- `sam_runtime_profile = h200_safe`
- `boundary_fusion_mode = v2`
- `matting_mode = high_precision_v2`
- `bg_inpaint_mode = video_v2`
- `reference_normalization_mode = structure_match`
- `lossless_intermediate = true`

contract 验证：

- `check_animate_contract.py`：PASS

### 3.2 synthetic

- `check_semantic_boundary_specialization.py`：PASS

### 3.3 真实 generate AB

每轮都跑三组：

- `none`
  - `replacement_conditioning_mode=legacy`
  - `boundary_refine_mode=none`
- `v2`
  - `replacement_conditioning_mode=rich_v1`
  - `boundary_refine_mode=v2`
- `semantic_v1`
  - `replacement_conditioning_mode=semantic_v1`
  - `boundary_refine_mode=semantic_v1`

统一参数：

- `frame_num = 45`
- `refert_num = 9`
- `sample_steps = 4`
- `sample_shift = 5.0`
- `boundary_refine_strength = 0.40`
- `boundary_refine_sharpen = 0.24`
- `output_format = ffv1`


## 4. 三轮闭环

### Round 1

真实 AB：

- [optimization4_step06_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step06_round1_ab)

结果：

- `face_boundary`
  - coverage: `0.0002415`
  - gradient: `-0.5910%`
  - contrast: `+0.3332%`
- `hair_boundary`
  - coverage: `0.0009680`
  - halo reduction: `-0.00093%`
- `hand_boundary`
  - coverage: `0.0003598`
  - gradient: `+0.1122%`
  - contrast: `+0.3637%`
- seam degradation: `+0.00030%`
- background degradation: `-0.0388%`

判断：

- specialization 已接通
- 但 face / hair / hand 的局部收益几乎为零
- 下一轮重点是提高 semantic specialization 的作用强度

### Round 2

真实 AB：

- [optimization4_step06_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step06_round2_ab)

改动：

- 提高 `semantic_v1` 在 `background_keep` 和 `boundary_strength` 上对 `face / hair / hand` 的作用强度
- 提高 `semantic_v1` 在 `boundary_refinement` 中的 face/hand sharpen 与 edge boost
- 把 per-class coverage gate 调到更符合真实窄边界面积分布的阈值：
  - face `0.0002`
  - hair `0.0008`
  - hand `0.0003`

结果：

- `face_boundary`
  - gradient: `-0.5712%`
  - contrast: `+0.6976%`
- `hair_boundary`
  - halo reduction: `+0.0261%`
- `hand_boundary`
  - gradient: `+1.1269%`
  - contrast: `+0.6017%`
- seam degradation: `+0.1363%`
- background degradation: `+0.0542%`

判断：

- 比 Round 1 好
- 但仍然远低于强 gate
- 下一轮不再继续堆系数，而改成“按 boundary band 归一化 semantic class 权重”

### Round 3

真实 AB：

- [optimization4_step06_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step06_round3_ab)

改动：

- 试图把 semantic class 权重按 boundary band 归一化，使 `face / hair / hand` 在局部 refine 中获得更强的 0-1 权重

结果：

- `face_boundary`
  - gradient: `+0.0411%`
  - contrast: `+0.6958%`
- `hair_boundary`
  - halo reduction: `+0.0432%`
- `hand_boundary`
  - gradient: `+0.9437%`
  - contrast: `+0.1446%`
- seam degradation: `+0.5037%`
- background degradation: `+0.0514%`

判断：

- face gradient 终于转正，但提升极小
- hand contrast 比 Round 2 更差
- hair 仍然远远没有达到需要的 halo 改善
- 这一轮没有把 Step 06 拉过强 gate


## 5. best-of-three 冻结判断

### 5.1 是否达成成功标准

没有。

原因：

- `face_boundary`
  - 目标：gradient / contrast 至少 `+8%`
  - 最好结果：gradient `+0.0411%`，contrast `+0.6976%`
- `hair_boundary`
  - 目标：halo 至少 `-10%`
  - 最好结果：`+0.0432%`
- `hand_boundary`
  - 作为 hair 替代也不够
  - 最好结果：gradient `+1.1269%`，contrast `+0.6017%`

所以：

- 至少两类重要边界显著改善：未达成
- seam/background safety：达成
- 最终结论：**Step 06 失败于强质量 gate，但成功于工程接线和实验验证**

### 5.2 冻结哪一轮

冻结为 **Round 2**。

原因：

- Round 1：作用太弱
- Round 2：face/hand 指标整体最好，且安全项稳定
- Round 3：引入归一化后并未带来实质性正收益，反而让 hand contrast 更差

当前代码应保留：

- Round 2 的 semantic specialization 系数
- 不保留 Round 3 的 semantic 权重归一化逻辑


## 6. 为什么会失败

这一步的失败不是因为 semantic boundary 没有接上，而是因为：

1. semantic boundary map 的存在，并不自动等于可观的视觉锐化收益
2. 当前 `semantic_v1` 仍然属于 deterministic / heuristic refine
3. semantic specialization 更擅长“防止某类边界被带坏”，不擅长“把该类边界真正重建得更锐”
4. face / hair / hand 这些窄边界，最终还在受 latent replacement 的结构性上限约束

换句话说：

- Step 06 证明了“把边界分类处理”在工程上可行
- 但也证明了“只靠分类 + 规则增强”，还不够把最终边缘锐度打出来


## 7. 当前结论

Step 06 现在应视为：

- **工程完成**
  - artifact / metadata / contract / generate / benchmark / gate 全接通
- **效果未达标**
  - 不升默认成功路线
- **代码冻结**
  - 保留 Round 2 语义 specialization 行为
- **对下一步的指导**
  - Semantic specialization 本身有价值，但它不是单独解决边缘锐度的主解法
  - 这更支持后续应优先继续做：
    - Step 07 `local edge super-resolution / detail restoration`
    - 或 Step 05 `generate-side edge candidate search` 但前提是先有真正更强的候选


## 8. 最终判断

Step 06 不是“白做了”，但也不能说“解决了边缘问题”。

它的价值在于：

- 证明了 face / hair / hand / cloth / occluded 这套语义边界体系可以正式进入 preprocess 和 generate 主链
- 并清楚证明：**只靠 semantic-aware deterministic specialization，不足以显著提升最终边缘锐度**

这一步的正确结论应当是：

**保留 semantic boundary artifacts 和 Round 2 的 specialization 代码，但不要把 Step 06 当成最终边缘锐度解法。**
