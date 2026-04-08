# Optimization6 Steps

## 1. 总览

`optimization6` 是在 `optimization5` 之后的明确路线切换。

前几轮工作已经比较清楚地证明：

- 继续在现有 backbone 上堆 heuristic / deterministic 边缘增强，回报越来越低；
- `rich conditioning`、`ROI refine`、`semantic experts`、`local edge restoration`、`candidate search`、`H200 tiers` 等路线，都没有稳定通过强质量 gate；
- 问题已经不再是“系统不会看”，而是：

> 系统还没有把高质量边界信息，真正转化为最终可感知的锐利边缘。

因此，`optimization6` 的目标不是继续磨旧路线，而是：

1. 先建立更强的人工审核边缘 benchmark；
2. 再接入真正更强的外部 alpha / matting；
3. 建立前景 / 背景解耦式 replacement contract；
4. 再做边界 ROI 局部生成式重建；
5. 在更强数据和更强任务定义下，才引入训练型边缘模型；
6. 最后再做 H200 高价值算力编排。


## 2. 步骤顺序

1. [01-human-reviewed-edge-benchmark-upgrade.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization6/steps/01-human-reviewed-edge-benchmark-upgrade.md)
2. [02-external-alpha-matting-model-vetting-and-integration.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization6/steps/02-external-alpha-matting-model-vetting-and-integration.md)
3. [03-foreground-background-decoupled-replacement-contract-and-core-path.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization6/steps/03-foreground-background-decoupled-replacement-contract-and-core-path.md)
4. [04-boundary-roi-generative-reconstruction-v2.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization6/steps/04-boundary-roi-generative-reconstruction-v2.md)
5. [05-trainable-edge-models-on-reviewed-data.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization6/steps/05-trainable-edge-models-on-reviewed-data.md)
6. [06-h200-high-value-orchestration-after-route-selection.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization6/steps/06-h200-high-value-orchestration-after-route-selection.md)


## 3. 统一执行规则

### 3.1 每一步都必须执行 3 轮闭环

除纯 benchmark 建设类步骤外，每一步都必须遵守：

1. `Round 1`
   - 做最小可运行实现
   - 跑 synthetic correctness
   - 跑真实 10 秒 smoke
2. `Round 2`
   - 根据 Round 1 的失败点调整方法或参数
   - 跑 reviewed benchmark + 真实 10 秒 AB
3. `Round 3`
   - 做最后一次有依据的修改
   - 明确比较 `Round 1 / 2 / 3`
   - 冻结 `best-of-three`

### 3.2 到第 3 轮仍未达标时的规则

如果第 3 轮后仍未过强 gate：

- 不允许继续无上限地微调；
- 必须冻结 `best-of-three`；
- 必须在 findings 中明确写出：
  - 哪些指标达成；
  - 哪些指标没达成；
  - 为什么判断该路线暂时收口；
- 然后进入下一步，不在当前步骤继续消耗。

### 3.3 每一步至少要有一种强对照

每一步都必须与以下之一做强对照：

- `optimization5` 的最优稳定基线；
- 当前步骤 `Round 1` 的实现；
- 或上一步已冻结的最佳实现。

不能只报告“现在的结果是什么”，必须报告“相对什么变好了 / 变差了”。


## 4. 设计意图

这套步骤之所以这样排序，是因为：

- Step 01 决定后面所有结论是否可信；
- Step 02 决定边缘信号本身是否足够强；
- Step 03 决定系统是否真正把边缘问题当作“前景 / 背景 / alpha / composite”问题；
- Step 04 才是直接攻击最终边缘锐度的主干步骤；
- Step 05 只有在更强数据和更强路径已经成立时才值得做；
- Step 06 最后做，避免把 H200 算力浪费在还未证明有效的方向上。

这不是“文档上的流程美化”，而是前几轮失败后总结出来的最保守、最可靠的执行顺序。
