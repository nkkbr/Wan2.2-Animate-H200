# Optimization9 Steps

## 1. 总览

`optimization9` 是“第三档高风险突破路线”的正式实施方案。

它不再假设当前 `Wan-Animate` 主路径、当前 foreground/background 耦合方式、当前整图编辑范式一定正确，而是开始系统性测试更激进的替代问题结构：

1. 更强的生成式 `mask-to-matte / matte bridge model`
2. layer decomposition / omnimatte 类路线
3. RGBA / transparent foreground 路线
4. renderable foreground / 3D-like 路线

这条线的核心目标不是“马上替代生产主线”，而是：

> 尽快找到一种真正表现出“错误模式不同、边缘机制不同、上限可能更高”的新路线。


## 2. 步骤顺序

1. [01-tier3-feasibility-benchmark-and-stop-rules.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization9/steps/01-tier3-feasibility-benchmark-and-stop-rules.md)
2. [02-generative-mask-to-matte-bridge-model-prototype.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization9/steps/02-generative-mask-to-matte-bridge-model-prototype.md)
3. [03-layer-decomposition-and-omnimatte-style-feasibility.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization9/steps/03-layer-decomposition-and-omnimatte-style-feasibility.md)
4. [04-rgba-transparent-foreground-feasibility.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization9/steps/04-rgba-transparent-foreground-feasibility.md)
5. [05-renderable-foreground-and-3d-like-feasibility.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization9/steps/05-renderable-foreground-and-3d-like-feasibility.md)
6. [06-tier3-portfolio-decision-and-escalation-gate.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization9/steps/06-tier3-portfolio-decision-and-escalation-gate.md)


## 3. 为什么这样排序

### 3.1 Step 01 必须先做

第三档最大的风险不是“方案太难”，而是：

- 做很多重型工程
- 却始终没有明确 stop rule

所以 Step 01 必须先把 feasibility benchmark、portfolio 维度和止损标准定下来。

### 3.2 Step 02 先于其他激进路线

`mask-to-matte / matte bridge model` 是第三档里：

- 最贴近现有系统
- 最容易做最小原型
- 最容易与现有 benchmark 对齐

所以它必须先做。

### 3.3 Step 03 和 Step 04 再往前推进问题结构

如果 Step 02 还不够，那么才去更激进地测试：

- layer decomposition
- RGBA foreground

这两条路线都比 Step 02 更重、更远离当前主线。

### 3.4 Step 05 最后做

renderable foreground / 3D-like 路线跨度最大，应在前面几条都明确存在天花板时才做。

### 3.5 Step 06 用于统一止损与升级

第三档不是“谁都做一点就算结束”。必须用 portfolio 级结论决定：

- 哪条路线值得升级到下一阶段
- 哪条路线应停止


## 4. 统一执行规则

### 4.1 每一步默认最多 3 轮

第三档所有步骤都必须遵守：

1. `Round 1`
   - 做最小可运行原型
   - 跑 synthetic correctness
   - 跑最小 reviewed / smoke
2. `Round 2`
   - 针对失败原因调整问题定义、输入输出或原型结构
   - 跑 reviewed benchmark + 最小真实视频 AB
3. `Round 3`
   - 做最后一次有依据的收敛尝试
   - 比较 `Round 1 / Round 2 / Round 3`
   - 冻结 `best-of-three`

### 4.2 不达标必须立即止损

如果第 3 轮后仍未出现“明显不同的正向错误模式”，就停止该路线。

第三档不追求“再调一点可能会好”，而追求：

- 快速证伪
- 快速发现真正值得升级的少数路线

### 4.3 每一步必须与当前主线强对照

每一步至少要和以下之一对照：

- `optimization7` 当前最优主线
- `optimization8` 当前最优训练支线
- 本步骤 `Round 1`


## 5. 阶段目标

`optimization9` 的阶段目标不是“做很多炫的原型”，而是尽快回答这三个问题：

1. 有没有某条高风险路线，第一次表现出与现有主线明显不同的正向错误模式？
2. 有没有某条路线在某一类困难边界上出现真正的质变，而不是指标此消彼长？
3. 有没有某条路线值得升级成下一阶段的正式主线候选？

如果这三个问题的答案都是否定的，`optimization9` 也算成功，因为它完成了高风险路线的系统止损。
