# Optimization4 Steps

## 1. 目标

`optimization4` 的主题不是继续泛化增强 preprocess，而是集中解决当前最关键的未完成项：

> 如何把已经显著增强的识别/约束能力，转化成最终输出里真正更锐利、更可信的人物边缘。

这套 steps 是基于 [wan_animate_replacement_edge_sharpness_extreme_plan.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/wan_animate_replacement_edge_sharpness_extreme_plan.md) 拆出来的可执行路线。它既继承 `optimization2` / `optimization3` 已经建立的 benchmark 与契约体系，也避免重新走“只改参数但没有最终观感收益”的老路。


## 2. 实施原则

### 2.1 先立强评测，再追强优化

`optimization3` 已经有较完善的 proxy 指标，但对“极细边缘是否真正更锐”仍然不够终局。  
因此本轮首先补边缘标注 mini-set 和更强 gate，然后再推进 generate-side 的重活。

### 2.2 每一步都走闭环

每一步默认最多 **3 轮**：

1. 实现
2. 全面测评
3. 若未达 gate，继续改代码
4. 再测评
5. 若到第 3 轮仍未达目标，则冻结 best-of-three，并把剩余问题写入 findings

这不是建议，而是 `optimization4` 的默认执行约束。  
除非某一步本质上只是搭建 benchmark/schema，否则不能只做实现不做测评，也不能只测一次就结束。

### 2.3 强调与上一轮的“显著”差距

每一步都应该相对 `optimization3` 最优配置，给出清晰的客观提升目标。  
如果只得到轻微改善，应明确判为“未达到本轮目标”。

### 2.4 每一步都必须有强对照

除 benchmark 建设型步骤外，每一步至少应具备下面两类对照之一：

1. 相对 `optimization3` 最优基线的 before/after 对照
2. 相对本步骤 Round 1 基线的 before/after 对照

如果一个步骤最终无法给出清晰的对照结果和量化改进，就不能宣称“该步骤已经成功提升了系统能力”。


## 3. 步骤顺序

### Step 01
[01-edge-labeled-mini-benchmark-and-gates.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/01-edge-labeled-mini-benchmark-and-gates.md)

建立边缘标注 mini-set、trimap/alpha/boundary 真实指标和新的强 gate。  
这是后续所有“边缘锐度是否真的变好”的客观判据基础。

### Step 02
[02-high-quality-alpha-matting-upgrade.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/02-high-quality-alpha-matting-upgrade.md)

升级 preprocess 侧的 alpha / matting 系统。  
目标是把当前 heuristic `soft_alpha` 提升为更可信的边缘表达。

### Step 03
[03-rich-boundary-signal-core-conditioning.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/03-rich-boundary-signal-core-conditioning.md)

让 richer boundary signal 真正进入 generate 主 conditioning，而不是只停留在 postprocess 和 mask 约束。

### Step 04
[04-boundary-roi-two-stage-highres-refine.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/04-boundary-roi-two-stage-highres-refine.md)

对 boundary ROI 做二阶段高分辨率 refine。  
这是最可能直接提升视觉边缘锐度的一步。

### Step 05
[05-generate-side-edge-candidate-search.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/05-generate-side-edge-candidate-search.md)

把 Step 08 的多候选自动选优思想复制到 generate-side edge refinement。

### Step 06
[06-semantic-boundary-specialization.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/06-semantic-boundary-specialization.md)

把边界按语义拆成 face/hair/hand/cloth/occluded 等子类，避免统一策略伤害不同边缘。

### Step 07
[07-local-edge-super-resolution-detail-restoration.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/07-local-edge-super-resolution-detail-restoration.md)

在 boundary ROI 上尝试局部 detail restoration / local SR。  
目标是补足“即使边界位置对了，细节仍然不够锐”的问题。

### Step 08
[08-h200-high-value-compute-allocation.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/steps/08-h200-high-value-compute-allocation.md)

正式定义 H200 141GB 的高价值预算分配，让算力优先花在真正影响边缘锐度的模块上。


## 4. 推荐执行策略

推荐执行顺序就是上面的 Step 01 -> Step 08。  
不要跳过 Step 01，因为如果没有更强的边缘真实指标，后续会继续陷入“proxy 有波动但主观难下判断”的状态。

每一步结束时都应该留下：

- findings 文档
- benchmark 运行目录
- gate 结果
- 最优配置快照
- 明确的“是否默认启用”结论


## 5. 与前一轮的关系

`optimization4` 不是推翻 `optimization3`，而是在它的基础上继续推进：

- `optimization3` 负责把系统变成“更会看、更会选、更稳”的高精度 pipeline
- `optimization4` 负责把这些能力真正变成“更锐利的最终边缘”

因此，`optimization3` 的已有能力默认都应被视为本轮基础设施，而不是重复开发目标。
