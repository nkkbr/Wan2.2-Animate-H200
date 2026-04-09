# Optimization7 Steps

## 1. 总览

`optimization7` 是“第一档现实突破路线”的正式实施方案。

它不再继续沿着 `optimization4/5/6` 已经证明接近上限的 heuristic / deterministic generate-side 修边路线前进，而是集中投入三条更可能真正改善最终边缘锐度的主线：

1. 更强的外部 alpha / matting 主模型
2. 真正的前景 / 背景解耦式 replacement 主路径
3. 边界 ROI 的局部生成式重建

这套步骤的核心思想是：

> 先让边缘信号本身变强，再让系统结构真正把边缘当作一等对象处理，最后才在局部边界上做生成式重建。


## 2. 步骤顺序

1. [01-external-alpha-matting-candidate-registry-and-reviewed-benchmark.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/steps/01-external-alpha-matting-candidate-registry-and-reviewed-benchmark.md)
2. [02-winning-alpha-model-production-integration.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/steps/02-winning-alpha-model-production-integration.md)
3. [03-foreground-background-decoupled-contract-and-compositor-baseline.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/steps/03-foreground-background-decoupled-contract-and-compositor-baseline.md)
4. [04-decoupled-core-replacement-path.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/steps/04-decoupled-core-replacement-path.md)
5. [05-boundary-roi-generative-reconstruction.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/steps/05-boundary-roi-generative-reconstruction.md)
6. [06-tier1-end-to-end-promotion-gate.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/steps/06-tier1-end-to-end-promotion-gate.md)


## 3. 为什么这样排序

### 3.1 Step 01 必须先做

如果没有在 reviewed benchmark 上客观打赢当前 baseline 的外部 alpha / matting 候选，后面的 decoupled contract 和 ROI reconstruction 就会建立在不够强的输入上。

### 3.2 Step 02 不能和 Step 01 混成一步

选型和接主线是两个不同问题：

- Step 01 解决“候选是否真的值得接”
- Step 02 解决“赢家如何稳定进入 preprocess 主链”

把两者分开，才能在 Step 01 失败时干净止损，而不是把整个主链一起拖进去。

### 3.3 Step 03 和 Step 04 要先于 Step 05

如果前景 / 背景 contract 还没拆开，ROI 生成式重建就仍然会被迫依赖整图 replacement 的旧结构，很容易重蹈 `optimization4` 的覆辙。

### 3.4 Step 06 最后做

只有当：

- 更强 alpha 真正成立
- decoupled 主路径真正跑通
- ROI reconstruction 至少有一轮正向信号

我们才有资格做 Tier-1 promotion gate，决定是否让新路线进入生产默认。


## 4. 统一执行规则

### 4.1 每一步默认最多 3 轮

除纯注册/清单类工作外，每一步都必须遵守：

1. `Round 1`
   - 打通最小可运行实现
   - 跑 synthetic correctness
   - 跑最小 reviewed/keyframe benchmark
2. `Round 2`
   - 根据 Round 1 的失败点调整方法或参数
   - 跑 reviewed benchmark + 真实 10 秒 AB
3. `Round 3`
   - 做最后一次有依据的修改
   - 比较 `Round 1 / Round 2 / Round 3`
   - 冻结 `best-of-three`

### 4.2 第 3 轮不过线时必须止损

如果到第 3 轮仍未达强 gate：

- 不允许继续无上限微调
- 必须冻结 `best-of-three`
- 必须在 findings 中明确写出：
  - 哪些指标达成
  - 哪些指标没达成
  - 为什么判断该路线暂时收口

### 4.3 每一步必须有强对照

每一步至少要和以下之一做强对照：

- `optimization6` 当前最稳基线
- 本步骤 `Round 1`
- 上一步冻结的最佳实现

不能只给“当前结果”，必须给“相对谁更好或更差”。


## 5. 阶段目标

`optimization7` 的阶段目标不是“把所有复杂问题一次解决”，而是要尽快回答这三个问题：

1. 是否真的存在一个外部 alpha / matting 候选，能客观打赢当前基线？
2. decoupled foreground/background 主路径是否能在不恶化 seam/bg 的前提下提供更好的边缘控制？
3. 生成式 ROI reconstruction 是否第一次真正带来可见的边缘锐度提升？

只要其中至少一个问题得到明确正向答案，`optimization7` 就已经比前几轮更有前进意义。
