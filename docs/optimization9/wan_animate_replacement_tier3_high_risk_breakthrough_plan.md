# Wan-Animate Replacement Tier-3 High-Risk Breakthrough Plan

## 1. 文档定位

本文档定义的是“第三档”路线，也就是：

> 高风险、高成本、高不确定性，但一旦成功最可能带来结构性突破的方向。

第三档路线不适合在当前就大规模投入主线资源。  
它更适合作为：

- 预研方向
- 结构性替代方案
- 当第一档和第二档都证明天花板明显时的突破路线

第三档路线的目标，不是做更多“优化”，而是考虑：

- 换问题建模方式
- 换生成范式
- 换主 backbone 假设


## 2. 为什么第三档仍值得保留

到目前为止，我们已经看到一个越来越清楚的事实：

即便：

- preprocess 更强
- alpha artifact 更丰富
- candidate search 更系统
- H200 预算更充足

也不代表最终边缘锐度就能显著提升。

这说明，问题很可能已经部分来自于：

- 当前 backbone 对边界细节的表达方式
- 当前 foreground/background 耦合方式
- 当前整图编辑主路径的结构限制

这类限制，不一定能靠继续优化现有路线解决。

因此，第三档存在的意义是：

> 当现有问题结构本身成为瓶颈时，准备新的问题结构。


## 3. 第三档包含哪些方向

### 3.1 换 backbone / 换视频编辑范式

#### 核心思想

不再默认“整图 latent replacement”是唯一主路径，而是考虑：

- layer-based video editing
- foreground RGBA generation
- transparent foreground rendering
- explicit compositing architectures

#### 为什么值得考虑

如果边缘一直被困在整图生成副作用里，那么再强的后续修补都只能治标。

而换范式的价值在于：

- 边缘不再是副产物
- alpha 不再只是辅助信号
- foreground / background / occlusion 可以成为一等对象


### 3.2 层分解 / omnimatte 类路线

#### 核心思想

把视频编辑问题分成显式层：

- foreground layer
- occluder layer
- shadow / effect layer
- background layer

#### 为什么有潜力

这类路线天然更适合处理：

- 遮挡边界
- 半透明边界
- 复杂发丝
- 背景恢复

它比当前“整图替换 + mask”更贴近边缘问题本质。


### 3.3 RGBA / transparent foreground 路线

#### 核心思想

不再只生成 RGB，而是尝试：

- 生成 foreground RGB + alpha
- 或生成 explicit matte-aware foreground layer

#### 为什么值得做

如果能直接把生成目标定义成带 alpha 的前景层，那么：

- 边缘质量会自然成为主输出质量的一部分
- 不再需要先整图生成，再靠后处理拆出边缘


### 3.4 3D / avatar / renderable foreground 路线

#### 核心思想

对于高价值场景，考虑更彻底的路线：

- 可渲染 foreground representation
- human mesh / neural rendering / gaussian representation
- foreground re-render + composite

#### 为什么它值得被列入第三档

虽然工程重、风险高，但它直接针对：

- 遮挡
- 动作一致性
- 视角变化
- 轮廓稳定性

这些都是纯 2D replacement 长期困难的问题。


### 3.5 更强的生成式 mask-to-matte / boundary reconstruction bridge model

#### 核心思想

不依赖主 backbone 自己学会所有边缘，而是在主链和最终 composite 之间插一个更强的生成式 bridge：

- `mask -> matte`
- `rgb + mask -> matte`
- `boundary ROI -> reconstructed boundary patch`

#### 为什么它和第一档不同

第一档的 ROI 生成式重建，仍然偏面向当前主路线。

第三档的 bridge model 更激进：

- 可以独立训练
- 可以使用专门数据
- 可以替代当前边缘重建模块，而不是只依附它


## 4. 第三档不应该怎么做

为了避免“高风险路线变成高消耗但低产出路线”，以下做法不应采用：

- 一次同时启动多个重型高风险项目
- 在 benchmark 和 contract 还没足够稳定时就全量切换主线
- 以“看 demo 图很好”作为立项依据
- 没有明确 stop rule 就持续投入

第三档必须比第一档、第二档更强调：

- 小规模预研
- 严格止损
- 快速证伪


## 5. 第三档的正确实施方法

### 5.1 只做小规模 feasibility study

每个第三档方向，初期只应做：

- 最小可验证原型
- 小样本 benchmark
- 少量真实 smoke

不要一上来就做大规模工程化。

### 5.2 每条路线都要先回答三个问题

1. 它在理论上是否真的绕开了当前失败根因？
2. 它在工程上是否有最小可落地路径？
3. 它在 benchmark 上是否有可能体现出“明显不同的错误模式”？

如果这三个问题有两个回答不了，就不值得进入原型阶段。

### 5.3 必须设置明确止损线

第三档的每条路线都应提前写清：

- 最多投入几轮
- 哪些结果算继续
- 哪些结果算停止

建议默认仍为最多 3 轮：

1. feasibility
2. directed improvement
3. final validation

第 3 轮不过线，就停。


## 6. 第三档最值得优先预研的方向

如果必须在第三档里再排优先级，我建议：

### 优先级 1：更强生成式 bridge model

原因：

- 最贴近当前系统
- 最容易做最小可行原型
- 最容易和现有 benchmark 对齐

### 优先级 2：layer decomposition / omnimatte 类路线

原因：

- 最有机会系统性解决遮挡与边缘耦合问题
- 理论收益高

### 优先级 3：RGBA / transparent foreground 路线

原因：

- 如果能成立，边缘质量定义会最自然
- 但工程实现和可用 backbone 风险更高

### 优先级 4：3D / renderable foreground 路线

原因：

- 最可能带来长期质变
- 但当前工程跨度最大，应该放在更后


## 7. 第三档的成功标准

第三档不要求一开始就替代主线，但至少应做到：

1. 出现一种明显不同于现有主线的质量分布
2. 至少在某类困难边界上表现出质的不同，而不是量的小幅变化
3. 这种不同能在 reviewed benchmark 和真实 smoke 上都观察到

具体说：

- 如果它只是又一次出现“gradient 稍好，contrast 变坏”这种旧模式，就不算成功
- 如果它能在某一类边界上第一次显著减少错误类型，即使整体还没超线，也值得继续


## 8. 第三档的风险

### 风险 1：工程跨度过大

应对：

- 先做最小原型
- 不直接改主生产链

### 风险 2：外部依赖重、可维护性差

应对：

- 严格做 registry 和 adapter
- 早期只保留少数候选

### 风险 3：研究成本高但不产生可生产结果

应对：

- 每轮都要用现有 benchmark 验证
- 不允许长期只靠定性展示图


## 9. 第三档与第一档、第二档的关系

第三档不是“当前主线失败了就立刻全切过去”的路线。

正确关系应是：

- 第一档：最现实、最可能近期见效
- 第二档：为放大有效路线做准备
- 第三档：为结构性突破做预研

换句话说：

- 第一档负责“最有希望近期成功”
- 第二档负责“让成功可持续”
- 第三档负责“在必要时突破旧范式”


## 10. 结论

第三档路线值得保留，但必须以正确方式推进：

- 小规模
- 强止损
- 快速证伪
- 只保留真正表现出结构性不同的方向

它的价值不在于“多试几个炫的方向”，而在于：

> 当第一档和第二档都逐渐触到上限时，第三档提供的是跳出当前问题结构的机会。

因此，第三档不是主线，但必须是被认真设计、被明确约束、被持续准备的后手路线。
