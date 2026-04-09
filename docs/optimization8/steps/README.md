# Optimization8 Steps

## 1. 总览

`optimization8` 是“第二档数据与训练扩展路线”的正式实施方案。

它的核心任务不是直接替代 `optimization7` 的主线突破，而是为真正有效的边缘路线建立更强的数据、任务定义、训练目标和验证体系。换句话说：

> `optimization7` 解决“有没有可能打穿边缘问题”，而 `optimization8` 解决“如果真的有希望，该如何把它做稳、做强、做可持续”。

因此，`optimization8` 的重点是：

1. reviewed edge benchmark 扩容与独立化
2. 训练任务与 ROI 数据管线重新定义
3. 先做清晰、受控的 trainable baseline
4. 再做 semantic experts
5. 再做 compositing-aware loss 与综合评估
6. 最后做第二档路线的综合 promotion gate


## 2. 步骤顺序

1. [01-reviewed-edge-benchmark-expansion-and-governance.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/steps/01-reviewed-edge-benchmark-expansion-and-governance.md)
2. [02-trainable-task-definition-and-roi-dataset-pipeline.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/steps/02-trainable-task-definition-and-roi-dataset-pipeline.md)
3. [03-trainable-alpha-and-matte-completion-baseline.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/steps/03-trainable-alpha-and-matte-completion-baseline.md)
4. [04-semantic-boundary-experts-training-path.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/steps/04-semantic-boundary-experts-training-path.md)
5. [05-compositing-aware-losses-and-evaluation-stack.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/steps/05-compositing-aware-losses-and-evaluation-stack.md)
6. [06-tier2-promotion-gate-and-amplification-decision.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization8/steps/06-tier2-promotion-gate-and-amplification-decision.md)


## 3. 为什么这样排序

### 3.1 Step 01 必须先做

如果 benchmark 和当前 baseline 仍然高度同源，那么训练型路线很容易“看起来有 improvement”，但其实只是继续围着旧分布打转。

### 3.2 Step 02 先于任何训练

前几轮失败已经证明：

- 任务定义模糊
- ROI 采样策略不清
- semantic 标签不明确

时，训练基本等于浪费 H200。

### 3.3 Step 03 与 Step 04 分开

trainable baseline 和 semantic experts 不是一回事：

- Step 03 先验证“训练本身是否有价值”
- Step 04 再验证“semantic 拆分能否进一步放大收益”

### 3.4 Step 05 放在后面

loss 体系必须建立在：

- 更强 GT
- 更明确任务
- 至少一个可运行训练 baseline

之上，否则只会堆更多看似高级但无效的损失。

### 3.5 Step 06 最后做

只有当：

- benchmark 已升级
- 训练 baseline 已明确
- semantic experts 和新 loss 已有清晰结果

才有资格判断第二档路线究竟应不应该继续被当作“放大器”。


## 4. 统一执行规则

### 4.1 每一步最多 3 轮闭环

除纯数据治理类工作外，每一步都必须遵守：

1. `Round 1`
   - 打通最小实现
   - 跑 synthetic / schema correctness
   - 跑最小 reviewed benchmark
2. `Round 2`
   - 根据失败点修改方法、参数或任务定义
   - 跑 reviewed benchmark + 真实 10 秒 smoke
3. `Round 3`
   - 做最后一次有依据的修改
   - 比较 `Round 1 / 2 / 3`
   - 冻结 `best-of-three`

### 4.2 不达标必须止损

若第 3 轮后仍不过强 gate：

- 不允许继续无上限地调参或扩大 sweep
- 必须冻结 `best-of-three`
- 必须在 findings 中写清楚：
  - 哪些指标达成
  - 哪些指标没达成
  - 失败来自数据、任务、loss、训练还是集成

### 4.3 每一步必须有强对照

每一步至少要和以下之一对照：

- `optimization7` 当前最优稳定结果
- 本步骤 `Round 1`
- 上一步冻结最佳实现

不能只报告绝对值，必须报告“相对谁变好或变差”。


## 5. 阶段目标

`optimization8` 的阶段目标不是“训练更多模型”，而是尽快回答这三个问题：

1. 更强 reviewed 数据是否真的改变了我们对边缘问题的判断边界？
2. 在更清晰任务定义下，trainable 路线是否终于能客观打赢非训练基线？
3. semantic experts 和 compositing-aware loss 能否真正放大已经成立的第一档路线，而不是只增加复杂度？

只有当这三个问题里至少有两个出现明确正向答案，第二档路线才值得继续作为主力增强线存在。
