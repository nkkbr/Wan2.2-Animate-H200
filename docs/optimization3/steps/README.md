# Wan-Animate Replacement 极致精度实施路线

## 1. 文档目标

本目录基于上层文档：

- `docs/optimization3/wan_animate_replacement_extreme_precision_plan.md`

把“如何把人物轮廓、动作、脸部识别推向更高精度，并尽量榨干单张 H200 141GB”收敛成一套可执行、可验证、可回退的分步骤工程路线。


## 2. 这一轮路线与前两轮的区别

前两轮优化解决的重点是：

- 工程契约
- lossless pipeline
- runtime profile
- 边界增强、background、reference normalization
- 基础系统验证与 smoke

这一轮的目标不同。重点不再是“补齐工程骨架”，而是：

1. 让 `preprocess` 进入高分辨率、多阶段、多模型融合的高精度范式
2. 让边界控制从 latent 级继续逼近 pixel alpha 级
3. 让 generate 精确消费更细的信号
4. 让 H200 的预算真正转化为更高的识别精度，而不是只转化为更多采样步数


## 3. 统一执行纪律

本轮所有步骤必须遵守以下纪律。

### 3.1 一步一结算

每次只推进一个步骤。  
当前步骤未通过验收前，不进入下一步的大规模代码改动。

### 3.2 每一步都必须做完整测评

每一步完成后都要执行：

- synthetic / unit / contract 回归
- 真实 10 秒视频 smoke
- 必要的 AB 对照
- 无损输出下的客观指标统计

### 3.3 每一步最多允许 3 轮实现-测评闭环

每一步都采用统一的 3 轮上限：

1. `Round 1`
   - 打通功能
   - 保证契约、debug、smoke 路径可运行
2. `Round 2`
   - 针对核心指标优化
   - 必须达到该步骤定义的主指标门槛
3. `Round 3`
   - 如果仍未达标，只允许做最后一轮针对性修正
   - 若 Round 3 之后仍未达标，必须冻结 findings，记录失败原因，不允许“无限打磨”

### 3.4 指标必须显著优于上一轮

本轮所有步骤的目标不是“勉强更好”，而是要明显优于当前 `optimization2` 的基线。  
如无特别说明，评估基线统一参考：

- `runs/optimization2_validation_core_20260407`
- `runs/optimization2_validation_extended_20260407`


## 4. 统一基线

后续所有步骤的默认真实 smoke 资源：

- 视频：
  - `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- 参考图：
  - `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- preprocess checkpoint：
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- generate ckpt_dir：
  - `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B`

当前可参考的关键基线值：

- `clean_plate_video`：
  - `temporal_fluctuation_mean = 1.5511`
  - `band_adjacent_background_stability = 0.4464`
- `structure_match`：
  - `target_ratio = 0.3540`
  - `normalized_ratio = 0.3840`
- `boundary refinement`：
  - `halo_ratio_before = 0.0325`
  - `halo_ratio_after = 0.0289`
  - `band_gradient_before = 0.1391`
  - `band_gradient_after = 0.1366`
- `generate refined smoke`：
  - `total_generate_sec ≈ 44.86`
  - `peak_memory_gb ≈ 39.85`


## 5. 实施阶段

### 阶段 A：建立高精度评测和高分辨率 preprocess 主链

1. `01-high-precision-benchmark-and-gates.md`
2. `02-h200-multistage-multires-preprocess.md`

### 阶段 B：把轮廓、脸、动作识别推进到高精度系统

3. `03-boundary-fusion-alpha-uncertainty-system.md`
4. `04-complete-face-analysis-stack.md`
5. `05-multiscale-pose-and-motion-stack.md`

### 阶段 C：完善 supporting conditions 与 generate 消费

6. `06-visibility-aware-clean-plate-background.md`
7. `07-generate-rich-signal-consumption-and-boundary-refinement-v2.md`

### 阶段 D：把 H200 变成“多候选高精度 preprocess 机器”

8. `08-h200-multi-candidate-preprocess-and-auto-selection.md`


## 6. 为什么是这个顺序

### 6.1 先做 Step 01

如果没有更严格的 benchmark 和 gate，后面“更高精度”只能靠主观感受，无法稳定推进。

### 6.2 Step 02 紧随其后

下一阶段的核心是高分辨率、多阶段 preprocess。  
这一步不完成，后面的边界、脸部、动作增强都没有高质量主链可挂载。

### 6.3 Step 03/04/05 是真正决定识别精度的主体

轮廓、脸和动作，构成 replacement 的核心识别三角。  
如果这三条链不升级，后面的 generate 和自动选优都只是锦上添花。

### 6.4 Step 06/07 是把更高精度信号转成最终画质

背景与 generate 消费决定这些高精度信号是否真正转化成最终视频质量。

### 6.5 Step 08 最后做

多候选 preprocess 自动选优对 H200 很重要，但它必须建立在前面单候选高精度链已经可信的前提上。


## 7. 每个步骤文档必须回答的问题

每个步骤文档都必须明确：

- 为什么要做
- 具体改哪些模块
- 如何分阶段实施
- 如何做客观验证
- 指标门槛是什么
- 如果未达标，最多如何迭代 3 轮
- 有哪些风险与回退方案


## 8. 这一轮路线的交付标准

只有满足下面这些条件，才算本轮路线成功：

- preprocess 真正进入高分辨率、多阶段分析模式
- 人物轮廓从软带控制推进到软 alpha / uncertainty 级控制
- 脸系统和动作系统具备结构化高精度输出
- generate 可消费 richer signals，而不是继续只依赖简化 mask
- H200 能稳定承载多候选 preprocess，并通过客观指标自动选优

