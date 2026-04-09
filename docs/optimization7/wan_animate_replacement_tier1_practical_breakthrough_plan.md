# Wan-Animate Replacement Tier-1 Practical Breakthrough Plan

## 1. 文档定位

本文档定义的是下一阶段**最值得立即投入主线资源**的路线，也就是前面讨论里的“第一档”。

这不是泛泛而谈的研究清单，而是面向当前代码基线、当前失败结论、当前硬件条件的**现实突破方案**。它的目标不是把系统再做得更复杂，而是尽可能快地回答一个最重要的问题：

> 在不再继续打磨既有 heuristic / deterministic 修边路线的前提下，哪一条新路线最有希望真正改善最终替换视频的人物边缘锐度？

这份文档所覆盖的内容，主要对应三条高优先级主线：

1. 更强的外部 alpha / matting 主模型接入
2. 真正的前景 / 背景解耦式 replacement 主路径
3. 边界 ROI 的局部生成式重建


## 2. 为什么第一档必须存在

`optimization4` 到 `optimization6` 已经反复证明：

- 继续沿着现有 `Wan-Animate` 主干做 generate-side heuristic 边缘增强，收益很低
- 继续做更多 candidate search，无法从一组不够强的候选里稳定选出明显更好的方案
- 继续增加 H200 采样预算，只能得到局部指标此消彼长，无法稳定换来更好的综合边缘质量
- 继续磨当前 heuristic alpha 或小型 edge adapter，也没有打破主线天花板

因此，第一档路线的存在意义就是：

> 不再继续修补旧问题结构，而是直接改变问题结构本身。

具体来说，就是把“边缘问题”从：

- 整图 latent replacement 的副作用
- 后处理修边问题
- 规则系统调参问题

升级成：

- 高质量 alpha / matte 估计问题
- 前景 / 背景显式解耦问题
- 局部边界生成式重建问题


## 3. 第一档的总体目标

第一档路线的目标，不是“一定在一步内彻底解决所有边缘问题”，而是要尽快达到以下结果：

1. 在 reviewed edge benchmark 上，至少出现一条方案**明显优于当前稳定基线**
2. 在真实 10 秒 benchmark 上，边缘出现**肉眼可见的等级差异**
3. 这种提升主要体现在：
   - hair edge
   - shoulder / cloth contour
   - hand / finger boundary
   - fast-motion boundary
4. 提升不能主要依赖牺牲：
   - seam
   - background stability
   - face identity
   - runtime 到不可接受程度

更具体地说，第一档路线应力争做到以下趋势：

- `boundary_f1` 明显提升
- `trimap_error` 明显下降
- `alpha_mae` / `alpha_sad` 明显下降
- `roi_gradient` 和 `roi_edge_contrast` 至少同时显著改善
- `halo_ratio` 不恶化，最好同步下降


## 4. 第一档的三条主工作流

### 4.1 工作流 A：外部 alpha / matting 主模型接入

#### 为什么要做

边缘质量的大部分上限，实际上取决于 `soft_alpha / trimap_unknown / hair_alpha` 这类信号本身的质量，而不是取决于后面的 sharpen 参数。

如果前面的 alpha 质量不够高，再强的 ROI 处理往往也只能做：

- 降 halo
- 控误差扩散
- 压边缘脏圈

而做不到真正恢复边界结构。

#### 这一工作流的目标

引入**真正强于当前 heuristic alpha 路线**的外部 matte 主模型，并把它稳定接进项目：

- 不是 demo 漂亮
- 不是局部主观更好
- 而是在 reviewed benchmark 和真实 smoke 上都有正向证据

#### 落地原则

必须遵守：

1. 先做 registry，再做下载，再做 adapter
2. 先做单帧 / keyframe reviewed eval，再做 10 秒视频 smoke
3. 不允许新模型直接覆盖默认主线
4. 每个外部模型都必须有：
   - source
   - version / commit
   - license
   - weight hash
   - adapter
   - smoke script
   - rollback flag

#### 推荐执行顺序

1. 先挑 2 到 3 个强候选，而不是一次引很多
2. 优先做推理成本和输入输出最清晰的模型
3. reviewed benchmark 先跑 keyframe，再跑 10 秒
4. 只有 benchmark 客观赢了，才允许进入 decoupled 主路径


### 4.2 工作流 B：前景 / 背景解耦式 replacement 主路径

#### 为什么要做

现在整图生成主导的问题在于：

- 前景边缘
- 背景 clean plate
- seam
- identity
- temporal consistency

这些目标耦在一起，导致任何边缘增强都会牵一发动全身。

如果不把前景和背景显式分开，边缘问题就永远难以成为一等公民。

#### 这一工作流的目标

正式建立一条主路径，使系统不再只是“整图 replacement 再修边”，而是显式维护：

- foreground rgb
- foreground alpha / trimap
- background clean plate
- compositing contract
- boundary ROI patch / refinement inputs

#### 应达到的结构性变化

1. `preprocess` 不再只输出“供整图 replacement 使用的条件”
2. `generate` 不再只理解“替换区域和背景保留区域”
3. `compositor` 成为显式模块，而不是散落在脚本里的局部逻辑
4. ROI 边缘增强以 compositing-aware 的方式工作，而不是盲目 sharpen


### 4.3 工作流 C：边界 ROI 的局部生成式重建

#### 为什么要做

`optimization4` 的最强结论之一是：

> deterministic ROI refine 不等于 edge reconstruction。

也就是说：

- 做 ROI 是对的
- 但只在 ROI 里做 sharpen / blend / paste-back，不够

真正需要的是：

- ROI patch 的局部生成
- 或 ROI patch 的局部 restoration
- 让模型重新长细节，而不是再加工已有像素

#### 这一工作流的目标

建立一个真正可评测、可回退的 ROI 生成式重建路径：

1. ROI 提取
2. ROI 条件构造
3. ROI reconstruction model / pipeline
4. alpha-aware paste-back
5. ROI-only metrics

#### 关键设计思想

- 只处理窄边界区域，不处理整图
- 输入必须包含：
  - foreground alpha / trimap
  - uncertain band
  - background patch
  - semantic ROI tag
  - face/hair/hand priors（如有）
- 输出必须与 compositing contract 对齐


## 5. 第一档的推荐执行顺序

第一档的正确顺序不能反。

### Phase 1：先做外部 alpha/matting 选型与 reviewed 评估

原因：

- 这是最有可能最快拉高边缘上限的因素
- 也是后续 decoupled contract 和 ROI reconstruction 的高质量输入基础

### Phase 2：再做前景 / 背景解耦 contract 和 core path

原因：

- 没有显式解耦，ROI reconstruction 会被迫继续依赖整图生成副作用

### Phase 3：最后做 ROI 生成式重建

原因：

- 它必须建立在：
  - 更强 alpha
  - 更明确的 foreground/background contract
  上，否则很容易重蹈 `optimization4` 的覆辙


## 6. 第一档的实施方法

### 6.1 不要再做的事

为了保证投入有效，第一档实施期间，不应继续把主精力放在：

- `rich_v1 / semantic_v1 / local_edge_v1 / roi_v1` 的同类变体
- 同一类 generate-side heuristic edge candidate search
- 基于当前默认主干的更多 sampling/tier sweep
- 继续细调旧版 alpha heuristic

这些工作不是完全无价值，但主收益已经基本摸清，不该再占主线资源。

### 6.2 每一步都要有独立 gate

第一档中每条工作流都必须有自己的独立 gate：

- alpha 工作流：看 `boundary_f1 / trimap_error / alpha_mae / alpha_sad`
- decoupled contract：看 contract correctness + compositing correctness + seam/bg regressions
- ROI reconstruction：看 `roi_gradient / roi_edge_contrast / roi_halo / seam`

### 6.3 每一步最多 3 轮闭环

默认规则：

1. Round 1：打通主链
2. Round 2：针对失败原因改实现或改参数
3. Round 3：做最后一次收敛尝试

如果第 3 轮仍不过线：

- 冻结 best-of-three
- 写清楚为什么不过线
- 停止继续在该版本上加码


## 7. 第一档的成功标准

### 最低成功标准

以下至少应有一条主路径满足：

- reviewed edge benchmark 上有显著正向收益
- 真实 10 秒 benchmark 上边缘观感出现肉眼可见改善
- `seam / background / identity` 没有明显恶化

### 理想成功标准

在真实 10 秒 benchmark 上，至少有一条路径相对当前稳定基线表现出：

- `boundary_f1` 明显提升
- `trimap_error` 明显下降
- `roi_gradient` 和 `roi_edge_contrast` 同时显著提升
- `halo_ratio` 不恶化
- `seam_score` 不明显恶化
- `background_fluctuation` 不明显恶化


## 8. 第一档的主要风险

### 风险 1：外部 alpha 模型 demo 很强，但工程接不住

应对：

- 先做 adapter
- 先过 keyframe reviewed eval
- 再做 10 秒 smoke

### 风险 2：decoupled contract 引入过大主链改造，短期拖慢一切

应对：

- 先做并行实验路径
- 不直接覆盖默认主链
- 先以 contract 完整性和 compositing correctness 为里程碑

### 风险 3：ROI 生成式重建成本太高

应对：

- 先 narrow ROI
- 先单 ROI path
- 先短 clip smoke
- 确定值得后，再扩大覆盖


## 9. 第一档的最终判断规则

当出现下面任意一种情况时，应判定第一档某条子路线暂不值得继续：

- 三轮后仍然没有任何 reviewed metric 明显优于基线
- 真实视频上只改善单一指标，但整体综合 gate 失败
- runtime / orchestration 成本高到不具备生产可行性
- 引入外部模型后可维护性显著恶化而质量收益不足

当出现下面情况时，应判定第一档路线成功并进入下一阶段：

- 至少一条主路径在 reviewed benchmark 和真实 10 秒 benchmark 上都客观优于基线
- 且提升不是单一 proxy 波动，而是具有跨指标一致性


## 10. 结论

第一档路线不是保守修补，而是当前最现实的突破方向。

它之所以值得优先做，不是因为它最简单，而是因为它同时满足：

- 离当前失败根因最近
- 理论上最可能真正改善边缘
- 工程上仍然可控
- 可以在 H200 上有效落地

如果后续资源有限，应优先保障第一档路线的执行与验证，而不是继续在旧路线内反复打转。
