# Wan-Animate Replacement Edge Reconstruction Plan

## 1. 背景与阶段结论

`optimization2`、`optimization3`、`optimization4` 三轮工作，已经把系统从“原始可跑版”推进到了“高精度、可诊断、可验证、可选优”的工程化版本：

- `preprocess` 侧已经显著增强：
  - 更稳定的 `SAM2` preprocess
  - 多阶段多分辨率分析
  - 完整 face analysis stack
  - 多尺度 pose / motion stack
  - visibility-aware clean plate background
  - richer boundary artifacts：`soft_alpha / boundary_band / occlusion_band / uncertainty_map / background_keep_prior`
- `generate` 侧也已经增强：
  - richer signal consumption
  - seam blending
  - temporal handoff
  - candidate search
  - 各类 boundary refinement 实验路径

但最关键的问题仍然没有被真正解决：

> 最终输出视频中的人物边缘，仍然没有出现稳定、显著、可复现的“更锐利”提升。

`optimization4` 已经比较明确地证明：

- 仅靠 heuristic / deterministic 的 generate-side 边缘增强，回报已经明显下降；
- `rich_v1`、`roi_v1`、`semantic_v1`、`local_edge_v1`、generate-side candidate search，都没有稳定通过强 gate；
- 继续在“现有整图生成 + 规则式后处理修边”这条线上打磨，已接近收益上限。

因此，`optimization5` 必须切换路线：

1. 把边缘问题从“规则修边”升级成“高质量 alpha + 局部生成式边缘重建”问题。
2. 把 H200 141GB 的价值优先投入到真正影响边缘质量的模块，而不是继续在低收益的规则参数搜索上消耗。
3. 用更强、更接近真值的边缘 benchmark 来约束所有后续工作，避免继续靠 proxy 自我感觉良好。


## 2. optimization5 的核心目标

`optimization5` 的目标不是继续做“更多 heuristic patch”，而是建立一条新的高价值主线：

> 用高质量 alpha / matting、前景背景解耦、边界 ROI 局部生成式重建，以及必要的小规模训练/适配，真正把 richer boundary signal 转化成最终锐利边缘。

本阶段的理想结果应满足：

- 在人工审核 edge mini-benchmark 上，`boundary F-score / trimap error / alpha MAE / halo / edge contrast / boundary gradient` 至少有一组方案显著优于 `optimization4` 最优结果；
- 在真实 10 秒 benchmark 上，至少一条高质量 generate 路径的边缘观感出现明显等级差异，而不是“仅仅更干净”；
- 继续保持：
  - seam 不明显恶化
  - background fluctuation 不明显恶化
  - face identity 不明显恶化
  - H200 开销主要投向高价值 ROI 或高价值模型，而不是全图低收益 brute force。


## 3. 为什么 optimization5 必须换路线

### 3.1 preprocess 的“看得更准”已经做出来了

经过前几轮，系统已经明显更会：

- 识别人脸
- 识别动作和手部
- 理解背景可见性
- 区分 hard foreground / soft boundary / occlusion / uncertainty

这些能力不是没用，而是为后续更强路线提供了必要输入。

### 3.2 当前最大的缺口在“最后一公里”

真正还没打透的是：

- 更高质量 alpha / trimap
- 把 richer boundary signal 深入生成主路径
- 局部高分辨率边缘重建

也就是说，瓶颈不再是“系统不会看”，而是：

> 系统还不会把这些高质量信号，变成真正锐利的最终边缘。

### 3.3 heuristic generate-side 路线已接近上限

`optimization4` 已经给出了比较清楚的证据：

- `rich_v1` 没有稳定提升 halo / gradient / contrast
- `roi_v1` 没有稳定提升 ROI 边缘锐度
- `semantic_v1` 没有显著拉起 face / hair / hand 边界指标
- `local_edge_v1` 只有轻微正向梯度变化，但远未达到强 gate
- candidate search 在当前候选库里，也没稳定选出明显更强方案

这说明后续继续磨这些规则方法的边际收益会越来越低。


## 4. optimization5 的总体设计原则

### 4.1 先建强真值，再做强优化

`optimization4` 的 edge mini-set 虽然已经有真实指标，但仍然是 `bootstrap_unreviewed` 起步。  
`optimization5` 必须先补一套更可信的人工审核边缘基准，否则后续再强的边缘方案也可能评价失真。

### 4.2 明确区分“前景生成”和“背景复合”

当前系统本质上仍是整图 replacement 主导。  
`optimization5` 应尽量把问题拆成：

- 前景人物生成 / 重建
- 高质量 alpha / trimap
- clean plate 背景
- 最终像素级复合

这样边缘才能被当作核心对象，而不是整图中的一个附带问题。

### 4.3 把 ROI 局部生成当作主手段，不把 deterministic postprocess 当主手段

后续策略要承认一个现实：

- deterministic sharpen / composite 最多能“修边”
- 真正的细边缘恢复，需要“局部生成式重建”

因此后续所有高价值边缘方案，都应以 ROI 局部生成 / restoration 为中心，而不是继续堆更多后处理规则。

### 4.4 H200 141GB 要优先花在高价值模块

H200 的预算优先级应是：

1. 高质量 alpha / matting / hair-edge estimation
2. 边界 ROI 二阶段局部生成
3. 少量高价值候选搜索
4. 必要的小模型训练 / 适配

而不是：

- 更多 heuristic 参数 sweep
- 更多整图 sharpening 试验
- 更多全图无差别采样步数

### 4.5 每一步都必须严格执行“实现 -> 测评 -> 再改”闭环

`optimization5` 的每一步默认最多 **3 轮**：

1. 实现
2. 做全面测评
3. 若未达 gate，则改代码或改策略
4. 再测评
5. 到第 3 轮仍未达标，则冻结 best-of-three，并把失败原因写进 findings

这不是建议，而是本轮默认执行约束。


## 5. optimization5 的 8 个核心工作流

### Workstream 1: Human-Reviewed Edge Benchmark and Hard Gates

把 `optimization4` 的 edge mini-set 升级为人工审核版，建立更可信的：

- alpha MAE / SAD
- trimap error
- boundary F-score
- hair-edge quality
- face/hand/cloth boundary quality

没有这一步，后续所有“边缘显著提升”都不够稳。


### Workstream 2: Production-Grade Alpha / Matting System

将当前 `alpha_v2` 路线升级成真正高质量的 production-grade alpha 系统：

- 更强 portrait/video matting
- 更强 hair-edge estimation
- 与 `SAM2 / parsing / uncertainty` 深度融合
- 输出可信的 `soft_alpha / trimap_unknown / hair_alpha / alpha_confidence`


### Workstream 3: Foreground-Background Decoupled Replacement Contract

显式建立：

- foreground render / foreground confidence
- clean plate background / visible support / unresolved region
- alpha-aware composite contract

目标是让最终边缘问题不再被埋在整图 replacement 里。


### Workstream 4: Rich Boundary Signal as Core Generation Conditioning

让 richer boundary signal 不再只停留在 mask shaping / postprocess，而是成为生成主条件的一部分。

重点包括：

- `soft_alpha`
- `trimap_unknown`
- `uncertainty_map`
- `occlusion_band`
- `face_preserve`
- `hair_alpha`
- `background_keep_prior`


### Workstream 5: Boundary ROI Generative Reconstruction

建立真正的 boundary ROI 二阶段局部生成式重建，而不是 deterministic refine：

- ROI 提取
- ROI 条件构造
- ROI 生成 / restoration
- alpha-aware paste-back

这是最有希望真正改善边缘视觉锐度的主干工作流。


### Workstream 6: Semantic ROI Experts for Face / Hair / Hands / Cloth

承认不同边界问题不同，用更细的 ROI 专家路径处理：

- face boundary
- hair boundary
- hand boundary
- cloth boundary
- occluded boundary

但这一次不再走纯规则方案，而是围绕 ROI 条件、alpha 和局部生成做差异化专家策略。


### Workstream 7: Trainable Edge Refinement / Adapter Path

如果前面几条高质量推理路线仍不足以显著拉高边缘，必须引入轻量训练路径：

- edge refinement adapter
- ROI diffusion adapter
- local restoration LoRA / lightweight model

目标不是训练整个大模型，而是在 H200 上用可控预算训练一个真正解决边缘重建的小模块。


### Workstream 8: H200 High-Value Compute Allocation and Candidate Orchestration

当有至少 1 到 2 条真正有效的高质量边缘方案后，再做 H200 级生产编排：

- preprocess candidate orchestration
- generate / ROI candidate orchestration
- compute budget policy
- default production preset

这一步必须后置，不能在方案尚未证明有效时先做。


## 6. 推荐实施顺序

推荐顺序如下：

1. Human-reviewed edge benchmark and hard gates
2. Production-grade alpha / matting system
3. Foreground-background decoupled replacement contract
4. Rich boundary signal as core generation conditioning
5. Boundary ROI generative reconstruction
6. Semantic ROI experts for face / hair / hands / cloth
7. Trainable edge refinement / adapter path
8. H200 high-value compute allocation and candidate orchestration

这个顺序有两个核心原则：

- 先建立可信 benchmark，再做高价值模型化改造；
- 先让局部生成式重建真正出现正向结果，再去做算力最优化。


## 7. 成功标准

`optimization5` 若要被判定为成功，至少要满足下面几条：

1. 在人工审核 edge mini-benchmark 上，至少有一条路线同时满足：
   - `boundary F-score` 明显提升
   - `trimap error` 明显下降
   - `alpha MAE / SAD` 明显下降
2. 在真实 replacement benchmark 上，至少有一条路线同时满足：
   - `halo_ratio` 下降
   - `band_gradient` 上升
   - `edge_contrast` 上升
   - seam / background 不明显恶化
3. 至少一条 generate 路径可以被升为默认高质量配置，而不是继续停留在实验状态。


## 8. 主要风险

### 8.1 benchmark 成本上升

人工审核 edge mini-set 会引入更多人工成本，但这是必要代价。

### 8.2 局部生成式重建可能引入 identity 漂移

ROI 生成一旦做得过强，容易让边缘更锐但人物一致性变差。  
因此 face/hair/hand 等 ROI 必须有明确 preserve 约束和专项指标。

### 8.3 小规模训练路线可能带来维护成本

一旦引入 adapter/LoRA/lightweight model，维护复杂度会明显增加。  
因此该路线应后置，并只在前面几条 inference 路线证明确实不足时再上。


## 9. 与 optimization4 的关系

`optimization5` 不是推翻 `optimization4`，而是基于 `optimization4` 的失败结论做出策略升级：

- `optimization4` 负责证明：当前 heuristic / deterministic generate-side 边缘增强路线已接近上限；
- `optimization5` 负责进入真正更高价值的新路线：
  - 高质量 alpha / matting
  - 前景背景解耦
  - 边界 ROI 局部生成式重建
  - 必要时的小规模训练路径

因此，`optimization4` 的 benchmark、artifact、metadata、candidate search、ROI 基础设施，默认都应视为 `optimization5` 的基础设施，而不是重复建设对象。
