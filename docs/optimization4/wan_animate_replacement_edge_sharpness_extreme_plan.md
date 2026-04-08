# Wan-Animate Replacement Extreme Edge Sharpness Plan

## 1. 背景与结论

经过 `optimization2` 与 `optimization3` 两轮大规模改造，当前系统已经在以下方面取得了实质提升：

- `preprocess` 侧的高精度能力显著增强：
  - 多阶段多分辨率 preprocess
  - 更强的 face analysis stack
  - 更强的 pose / motion stack
  - visibility-aware clean plate background
  - richer boundary artifacts: `soft_alpha / boundary_band / occlusion_band / uncertainty_map / background_keep_prior`
- `generate` 侧已经能够消费 richer signal，并具备多候选 preprocess 自动选优能力。

但当前最关键的问题仍然存在：

- 最终输出视频中，人物边缘的锐度仍未达到“明显质变”。
- 当前系统更擅长：
  - 减少 halo
  - 降低边缘脏圈
  - 提高边界语义理解能力
- 当前系统还不够擅长：
  - 重建头发丝、衣角、手部等细边缘
  - 同时提升 `edge contrast` 与 `boundary gradient`
  - 在不引入伪影的前提下，让替换人物轮廓显著变锐

换句话说，当前系统已经变成了一个“看得更准、选得更优、时序更稳”的高质量 pipeline，但仍未完成“把这些高精度信号转化成最终锐利边缘”的最后一公里。

本方案定义 `optimization4` 的目标：

1. 以“最终人物轮廓锐度”为最高优先级。
2. 继续榨干 H200 141GB 的价值，但把算力优先投入到真正影响边缘的模块。
3. 把当前已积累的高质量 preprocess 信号，真正转化成 generate 质量，而不是只停留在 metadata 或 postprocess 约束层。


## 2. 总体目标

`optimization4` 的总体目标不是再做一轮泛化优化，而是聚焦解决下面这个核心问题：

> 为什么系统已经能更好地识别人物、动作、脸和背景，但最终边缘依旧没有明显变锐？

本阶段要达成的结果应满足：

- 在真实 replacement case 上，边缘 proxy 指标显著优于 `optimization3` 最优结果。
- 至少在一套高质量配置下，边缘主观观感出现可见等级的提升，而不仅仅是“更干净”。
- 继续保持：
  - seam 不明显退化
  - 背景稳定性不明显退化
  - face identity 不明显退化
  - runtime 增长可控，且主要算力开销集中在高价值子模块


## 3. 核心设计原则

### 3.1 优先改善“最后一公里”

接下来的优化应优先打在：

- alpha / matting 质量
- generate 主 conditioning 对 richer boundary signal 的消费方式
- boundary ROI 的高分辨率二阶段修复

不应再把大量时间消耗在边缘收益有限的“泛参数微调”上。


### 3.2 以像素级边界为核心，不再只依赖 latent-space replacement

当前系统的上限受到 latent-space replacement 约束。  
因此，后续方案必须把一部分能力下沉到像素域或 ROI 级高分辨率阶段。


### 3.3 以真实 gate 为准，而不是靠主观印象

`optimization4` 中每项工作都应尽量绑定至少一类客观指标：

- boundary F-score
- trimap error
- alpha MAE / SAD
- hair-edge quality proxy
- band gradient
- edge contrast
- halo ratio
- seam score
- background fluctuation

没有真实指标支持的“看起来更清晰”，不应直接升为默认。


### 3.4 H200 的价值优先用于“高价值分支”

H200 141GB 的预算应优先用于：

- 高分辨率 matting / alpha estimation
- ROI 二阶段生成
- 多候选 generate / preprocess search
- 局部边界超分

而不是只在全图生成上盲目增加采样步数。


## 4. 下一阶段的 8 个核心工作流

下面 8 个方向，都是基于当前问题与前面几轮讨论整理出来的正式候选工作流。


### Workstream 1: High-Quality Alpha / Matting Upgrade

#### 目标

把当前 heuristic `soft_alpha` 升级为真正高质量、时序稳定、可用于边缘重建的 alpha 系统。

#### 为什么要做

- 边缘清晰度本质上高度依赖 alpha 质量。
- 目前的 `soft_alpha` 更偏“可用边界约束”，而不是“高质量边缘表达”。
- 头发丝、衣角、半透明边界、手指缝等区域都需要更强的 matting 能力。

#### 设计方向

- 引入 portrait matting / video matting 模型或更强的 hybrid 方案。
- 与当前：
  - `SAM2`
  - parsing
  - hard foreground
  - uncertainty
  进行融合。
- 输出升级版 artifact：
  - `alpha_v2`
  - `trimap_v2`
  - `alpha_uncertainty_v2`
  - `hair_edge_mask`
  - `fine_boundary_mask`

#### 验收指标

- `alpha MAE / SAD` 显著优于当前 baseline
- `boundary F-score` 有可见提升
- 真实 case 上 `band_gradient` 不下降
- 真实 case 上 `halo_ratio` 至少下降 10% 以上


### Workstream 2: Rich Boundary Signal Consumption in Core Generate Path

#### 目标

让 richer boundary signal 不再只影响 postprocess，而是直接影响生成主路径。

#### 为什么要做

- 当前 richer signals 大多只在：
  - `background_keep`
  - replacement mask
  - boundary postprocess
  中起作用。
- 这更像“避免生成错误”，而不是“主动生成更好的边缘”。

#### 设计方向

- 让以下信号进入更强的生成条件路径：
  - `soft_alpha`
  - `boundary_band`
  - `occlusion_band`
  - `uncertainty_map`
  - `background_keep_prior`
  - `face_preserve_map`
  - `structure_guard`
- 探索两类接法：
  1. 直接增强 `replacement_masks` 在 latent 条件中的表达
  2. 将边界/不确定性信号作为额外 conditioning branch 编码成 token 或 feature maps

#### 验收指标

- 相对 current `rich_none`，真实 case 上：
  - `halo_ratio` 下降
  - `band_gradient` 上升
  - `band_edge_contrast` 上升
- 且 seam / background fluctuation 不明显恶化


### Workstream 3: Boundary ROI Two-Stage High-Resolution Refinement

#### 目标

建立专门针对边界 ROI 的二阶段高分辨率生成 / 修复流程。

#### 为什么要做

- 当前全图 refine 太保守，容易把边缘一起变软。
- 你最在意的不是整图，而是人物外轮廓那一圈。
- 只对 boundary ROI 做二阶段高分辨率修复，能更高效、更精准地利用 H200。

#### 设计方向

- 第一阶段：照常完成整图 replacement。
- 第二阶段：
  - 基于 `boundary_band` 提取 ROI patches
  - 扩充 face / hair / hands / cloth 边界 ROI
  - 在更高分辨率上做局部生成或局部 edge restoration
- 最后基于 alpha-aware 规则回贴。

#### 验收指标

- ROI 内 `band_gradient` 提升 > 8%
- ROI 内 `edge_contrast` 提升 > 8%
- `halo_ratio` 不恶化
- 全图 seam 指标不明显变差


### Workstream 4: Generate-Side Candidate Search for Edge Refinement

#### 目标

把 Step 08 在 preprocess 侧验证成功的“多候选 + 自动选优”思想，复制到 generate 侧边缘精修。

#### 为什么要做

- 当前 Step 07 的结论不是“v2 不可能成功”，而是“当前参数/规则组合没有过 gate”。
- H200 允许我们用少量额外算力去搜索更好的边缘参数组合。

#### 设计方向

- 对同一 preprocess bundle，自动试验：
  - `boundary_refine_mode`
  - `boundary_refine_strength`
  - `boundary_refine_sharpen`
  - `replacement_boundary_strength`
  - `transition_low/high`
  - 将来新增的 `roi_refine_strength`
- 建立 `selected_vs_default_generate_edge_score`
- 自动选出最佳 generate candidate

#### 验收指标

- `selected better than default ratio >= 0.7`
- 至少在真实 10 秒 benchmark 上，自动选优版本 consistently 优于默认


### Workstream 5: Local Edge Super-Resolution / Detail Restoration

#### 目标

在 boundary ROI 上尝试局部超分或 detail restoration，补足细边缘丢失问题。

#### 为什么要做

- 边缘模糊很多时候不仅是 alpha 问题，也包括高频细节缺失。
- 局部而非全图的 detail enhancement 更适合 replacement 后的边缘修复。

#### 设计方向

- 对 boundary ROI 应用：
  - 局部 SR
  - 细节恢复
  - edge-aware enhancement
- 明确禁止全图无差别锐化。
- 必须使用 alpha / uncertainty / occlusion 约束，避免伪边和 ringing。

#### 验收指标

- `edge_contrast` 有明显提升
- `halo_ratio` 不恶化
- 人脸和发丝附近不引入明显伪影


### Workstream 6: Semantic Boundary Specialization

#### 目标

把边界分成不同语义区域，用不同策略处理，而不是继续“一套规则处理所有边缘”。

#### 为什么要做

- 头发边缘、脸边缘、手边缘、衣服边缘的问题本来就不同。
- 统一 refine 往往导致某些边界提升，另一些边界反而变差。

#### 设计方向

- 新增语义边界类别：
  - `face_boundary`
  - `hair_boundary`
  - `hand_boundary`
  - `cloth_boundary`
  - `occluded_boundary`
- 每类边界单独设定：
  - alpha trust
  - sharpen strength
  - blend strength
  - structure preservation

#### 验收指标

- 不同语义区域的局部指标分开统计
- 至少 2 类高价值边界（建议 face / hair）出现显著提升


### Workstream 7: Edge-Labeled Benchmark Mini-Set

#### 目标

建立一个小而强的边缘真实标注集，结束“全靠 proxy 评边缘”的状态。

#### 为什么要做

- 当前 proxy 有价值，但无法完全代表真实边缘质量。
- 如果后续还要做 2 到 3 轮 generate-side 改造，就必须有更接近真值的评估。

#### 设计方向

- 从真实素材中选 20 到 50 帧关键帧
- 标注：
  - trimap
  - alpha / boundary
  - hair-edge mask
  - hand-edge mask
- 先做小规模人工集，不追求大而全

#### 验收指标

- benchmark 具备：
  - boundary F-score
  - trimap error
  - alpha MAE / SAD
- 后续每个边缘相关步骤都能直接对这套标注集出分


### Workstream 8: H200 High-Value Compute Allocation Strategy

#### 目标

正式定义 H200 在 `optimization4` 中应该优先消耗在哪里。

#### 为什么要做

- 如果没有策略，H200 的资源很容易又被无差别消耗在生成步数上。
- 但当前阶段边缘锐度的主瓶颈并不主要在采样步数。

#### 设计方向

- 为 H200 定义三类模式：
  - `edge_preprocess_extreme`
  - `edge_generate_search`
  - `edge_roi_refine`
- 每类模式有独立预算：
  - 分辨率
  - batch/candidate 数
  - ROI refine 数量
  - 允许的运行时间
- 明确“优先算力投放顺序”：
  1. alpha/matting
  2. boundary ROI refine
  3. generate-side candidate search
  4. 再考虑整体采样步数

#### 验收指标

- GPU 显存和时间分布可被 profiling 和复盘
- 多数额外算力确实花在高价值边缘相关模块


## 5. 优先级排序

如果按收益 / 风险 / 对边缘锐度的直接性综合排序，推荐顺序如下：

1. **Workstream 1: High-Quality Alpha / Matting Upgrade**
2. **Workstream 2: Rich Boundary Signal Consumption in Core Generate Path**
3. **Workstream 3: Boundary ROI Two-Stage High-Resolution Refinement**
4. **Workstream 7: Edge-Labeled Benchmark Mini-Set**
5. **Workstream 4: Generate-Side Candidate Search**
6. **Workstream 6: Semantic Boundary Specialization**
7. **Workstream 5: Local Edge Super-Resolution / Detail Restoration**
8. **Workstream 8: H200 High-Value Compute Allocation Strategy**

注意：

- 从工程顺序上看，Workstream 7 不一定最先实现，但应尽快启动。
- 如果没有更真实的边缘标注，小心后续优化继续在 proxy 上“绕圈”。


## 6. 成功标准

`optimization4` 最终应满足下面三类成功标准。

### 6.1 Preprocess 成功标准

- alpha / boundary / uncertainty artifact 质量显著提升
- richer signal 可稳定输出并通过 contract
- face / pose / background 系统不因新边缘方案回退

### 6.2 Generate 成功标准

至少有一条 generate-side 边缘路径能够同时满足：

- `halo_ratio` 明显下降
- `band_gradient` 明显上升
- `band_edge_contrast` 明显上升
- seam 不明显退化
- background fluctuation 不明显退化

### 6.3 主观质量成功标准

在真实 replacement case 上，应该能出现肉眼可见的下列变化：

- 发丝边缘更锐
- 肩线 / 手臂轮廓更清楚
- 衣摆边缘更干净且更稳
- 不再主要表现为“边缘更干净但更软”


## 7. 风险

### 7.1 过拟合 proxy

风险：
- 优化某个 boundary proxy，却牺牲真实主观质量。

应对：
- 尽早引入 edge-labeled benchmark
- 保留人工对照 keyframes


### 7.2 过度锐化带来伪边

风险：
- halo 少了，但出现 ringing、硬边、脏边、发丝断裂。

应对：
- 强制跟踪 face/hair/hand 局部指标
- 限制全图锐化，优先 ROI 化


### 7.3 H200 预算浪费在低价值路径

风险：
- 额外算力并没有换来边缘质量，反而主要被重型全图 generate 吞掉。

应对：
- 建立明确 profiling
- 预算优先投向 alpha、ROI refine、候选搜索


## 8. 建议的后续落地顺序

最推荐的实际落地顺序是：

1. Workstream 1: High-Quality Alpha / Matting Upgrade
2. Workstream 7: Edge-Labeled Benchmark Mini-Set
3. Workstream 2: Rich Boundary Signal Consumption in Core Generate Path
4. Workstream 3: Boundary ROI Two-Stage High-Resolution Refinement
5. Workstream 4: Generate-Side Candidate Search
6. Workstream 6: Semantic Boundary Specialization
7. Workstream 5: Local Edge Super-Resolution / Detail Restoration
8. Workstream 8: H200 High-Value Compute Allocation Strategy

这样安排的原因是：

- 先把 alpha 做强，才有边缘修的基础
- 再尽快建立更真实的边缘评价闭环
- 然后把 richer signal 真正喂给 generate
- 最后再叠加 ROI refine、搜索、超分等更重的能力


## 9. 最终判断

`optimization4` 的核心不是“再做一轮更多的工程改造”，而是明确从这里开始：

- 把前面几轮已经做好的高精度识别能力
- 真正转化成
- 具有肉眼可见边缘质量提升的最终输出能力

如果说 `optimization3` 解决的是：

> 系统是否已经足够会看、会选、会稳定地生成

那么 `optimization4` 要解决的就是：

> 系统能不能把这些能力真正变成清晰、锐利、可信的人物边缘

这是后续优化中最值得投入的一条主线。
