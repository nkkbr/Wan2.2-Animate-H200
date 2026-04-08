# Step 03 Findings: 边界融合、Soft Alpha 与 Uncertainty 系统

## 1. 结论

Step 03 已完成，但结论需要准确表述：

- `hard_foreground / soft_alpha / boundary_band / occlusion_band / uncertainty_map / background_keep_prior`
  这套 artifact schema 已正式落地，并贯穿 `preprocess -> metadata -> generate(debug/load)`。
- 在真实 10 秒素材上，新的 boundary fusion v2 已能稳定生成上述 artifact，并通过 contract 检查。
- 三轮尝试后，**best-of-three 冻结结果选择了 Round 1 的 uncertainty 公式**。
- 当前版本的主要收益是：
  - occlusion / uncertainty 成为正式一等信号
  - uncertainty 能较好聚焦在 transition 区域
  - alpha 没有相对 legacy 回退
- 当前版本**还不能声称已经达成真实标注集上的 boundary F-score +15% / trimap error -20%**。
  原因不是代码未接通，而是这一轮仍然以 synthetic + unlabeled real proxy 为主，尚未接入真正的人工边界标注集。


## 2. 本步骤实际交付

### 2.1 新 artifact

preprocess 现在可输出：

- `hard_foreground`
- `soft_alpha`
- `boundary_band`
- `occlusion_band`
- `uncertainty_map`
- `background_keep_prior`

并在 metadata 中记录：

- `mask_semantics`
- `source_models`
- `fusion_version`
- `confidence_summary`

### 2.2 新调试产物

新增或增强的 QA/debug：

- `fused_boundary_overlay`
- `uncertainty_heatmap_preview`
- `alpha_hard_compare_preview`
- `occlusion_band_overlay`

### 2.3 generate 侧接通情况

`generate` 已能：

- 读取 `occlusion_band`
- 读取 `uncertainty_map`
- 进入 debug/runtime stats

但这一轮 **generate 还没有正式消费 uncertainty 作为生成控制项**。  
这和路线设计一致：Step 03 先把高精度边界 artifact 体系做对，Step 07 再升级 rich-signal consumption。


## 3. 三轮实现与测评

## Round 1

### 改动

- 打通完整 schema、metadata、contract、debug
- 引入第一版 `occlusion_band` 与 `uncertainty_map`
- 真实 10 秒 preprocess 跑通

### 真实结果

运行目录：

- `runs/optimization3_step03_round1_preprocess`

关键 proxy 指标：

- `soft_alpha_mean = 0.1369`
- `boundary_band_mean = 0.0427`
- `occlusion_band_mean = 0.0246`
- `uncertainty_mean = 0.0545`
- `uncertainty_boundary_focus_ratio = 0.2583`
- `uncertainty_transition_focus_ratio = 0.3976`
- `uncertainty_transition_focus_ratio_dilated = 0.5173`
- `uncertainty_transition_to_interior_ratio = 7.78`

### 判断

Round 1 已经把系统真正接通，而且从新的 transition-oriented proxy 看，uncertainty 已经有了可用的聚焦能力。  
当时看起来偏低的 `uncertainty_boundary_focus_ratio = 0.2583`，后来确认是因为这个指标会被窄 boundary band 的面积上限卡住，不能单独拿来做 gate。


## Round 2

### 改动

- 试图通过更强的 boundary gating，把 uncertainty 更集中地压到 boundary 上

### 真实结果

运行目录：

- `runs/optimization3_step03_round2_preprocess`

关键 proxy 指标：

- `uncertainty_boundary_focus_ratio = 0.0521`
- `uncertainty_transition_focus_ratio = 0.0661`
- `uncertainty_transition_focus_ratio_dilated = 0.0816`

### 判断

Round 2 明显退化。  
原因是 uncertainty 被压得过于稀疏，导致 top-20% uncertainty 区域几乎退化成“非零即高 uncertainty”，失去区分度。


## Round 3

### 改动

- 先尝试温和版重新聚焦 uncertainty
- 再与前两轮比较
- 最终按 best-of-three 冻结，回退到 Round 1 公式
- 同时保留这一步新增的 transition-oriented proxy 指标

### 真实结果

运行目录：

- `runs/optimization3_step03_round3_preprocess`

中间版本关键 proxy：

- `uncertainty_boundary_focus_ratio = 0.2329`
- `uncertainty_transition_focus_ratio = 0.3305`
- `uncertainty_transition_focus_ratio_dilated = 0.4044`

### 判断

Round 3 比 Round 2 好很多，但仍然不如 Round 1。  
因此最终冻结策略不是“保留最后一次代码”，而是 **保留三轮里客观最优的 Round 1 uncertainty 方案**。


## 4. 最终冻结版本

冻结版本特点：

- alpha 路径保持 legacy-grade 保守策略，避免回退
- boundary/occlusion/uncertainty schema 正式启用
- uncertainty 采用 Round 1 公式
- 新增的 transition-oriented proxy 指标保留，用于后续 gate

这也是当前仓库中的最终代码状态。


## 5. 指标解释与 gate 调整

## 5.1 为什么旧的 `uncertainty_boundary_focus_ratio` 不够

`uncertainty_boundary_focus_ratio` 的定义是：

- top 20% uncertainty 像素里，有多少落在 `boundary_band > 0.25`

问题在于，真实 smoke case 里的 `boundary_band` 面积只有约 `5%`。  
当 top-20% uncertainty 区域固定是约 `20%` 面积时，这个比值会被 boundary band 面积上限天然压低。

所以它可以保留作为参考，但不应该再单独作为主 gate。

## 5.2 新的主 proxy

这一步新增并建议后续优先使用：

- `uncertainty_transition_focus_ratio`
- `uncertainty_transition_focus_ratio_dilated`
- `uncertainty_transition_to_interior_ratio`
- `uncertainty_transition_to_background_ratio`

其中：

- `transition_region = boundary_band>0.10 OR (soft_alpha-hard_foreground)>0.03 OR occlusion_band>0.10`
- `transition_region_dilated` 再做一次 5x5 膨胀

对真实 smoke，更合理的 gate 解释是：

- top-20% uncertainty 是否主要落在 transition 区域
- transition 区域的 uncertainty 是否显著高于主体内部和远背景


## 6. 目标达成情况

### 已达成

- 新边界 artifact 契约已完成
- preprocess / metadata / contract / generate(debug) 已接通
- synthetic 边界融合检查通过
- 真实 10 秒 preprocess smoke 通过
- 真实 10 秒 generate smoke 已用于 artifact 兼容性验证
- uncertainty 在 synthetic 上满足：
  - top-20% uncertainty 覆盖至少 `50%` 边界误差像素

### 部分达成 / 尚未完成

- 真实 labeled boundary F-score `+15%`
- 真实 labeled trimap error `-20%`
- 真实 halo ratio `-30%`

这些指标尚未在 **人工标注集** 上完成验证。  
当前只有 synthetic + unlabeled real proxy，因此不能夸大结论。


## 7. 对后续步骤的意义

Step 03 的价值，不在于“边缘已经极致清晰”，而在于：

1. 以后不再只靠单一 `person_mask` 或 `soft_band`
2. 系统现在能明确表达：
   - 哪些边界是软过渡
   - 哪些地方存在遮挡
   - 哪些区域其实不确定
3. Step 07 可以正式消费 richer signals，而不是只能消费硬 mask

如果没有这一步，后续 generate 侧的高精度边界 refinement 就没有可靠输入基础。


## 8. 当前推荐结论

- **继续使用 `--boundary_fusion_mode v2`**
- **v2 的 uncertainty 冻结为 Round 1 公式**
- 后续 gate 优先看 transition-oriented uncertainty proxy，而不是只看旧的 boundary-focus proxy
- 下一步应进入：
  - face 完整分析栈
  - 多尺度 pose/motion 栈
  - rich-signal generate consumption

