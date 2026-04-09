# Step 01: Tier-3 Feasibility Benchmark And Stop Rules

## 1. 为什么要做

第三档路线的最大风险，不是技术难，而是没有统一的 feasibility 框架。

如果没有先定义：

- 什么叫“表现出新价值”
- 什么叫“只是重现旧失败模式”
- 什么叫“应该停”

那么后面的 bridge model、layer decomposition、RGBA、3D 路线都会变成高成本但低结论密度的预研。


## 2. 目标

建立第三档路线专用的 feasibility benchmark、对照协议和 stop rule，使后续每条高风险路线都能被快速证伪或升级。


## 3. 要改哪些模块

建议新增：

- `docs/optimization9/benchmark/feasibility_manifest.step01.json`
- `docs/optimization9/benchmark/stop_rules.step01.json`
- `scripts/eval/run_optimization9_feasibility_suite.py`
- `scripts/eval/evaluate_optimization9_feasibility_suite.py`
- `scripts/eval/check_optimization9_feasibility.py`


## 4. 具体如何做

### 4.1 定义第三档特有的成功信号

除常规指标外，第三档必须重点看：

- 是否出现“错误类型变化”
- 是否在某类困难边界上第一次明显优于主线
- 是否避免了旧模式里常见的：
  - gradient 涨而 contrast 跌
  - contrast 涨而 seam 崩
  - halo 下降但 identity 漂

### 4.2 定义 feasibility bucket

每条路线最终必须进入下列之一：

- `reject`
- `interesting_but_unproven`
- `upgrade_candidate`

### 4.3 定义 stop rule

建议硬 stop：

- 3 轮后仍未出现任何“明显不同的正向错误模式”
- 或 reviewed benchmark 与真实 smoke 都没有出现一类困难边界上的质变


## 5. 如何检验

### 5.1 correctness

- benchmark manifest 可解析
- stop rules 可程序化读取
- evaluator 可输出统一结论

### 5.2 practical utility

- 至少能对一个旧路线结果给出清晰 bucket 归类


## 6. 强指标要求

本步骤不要求质量提升，但要求：

- feasibility evaluator 能稳定输出统一判断
- stop rules 足够具体，不依赖模糊主观描述


## 7. 实现-测评-再改闭环

### Round 1

- 建 benchmark manifest + stop rules

### Round 2

- 用至少一个旧失败案例回放，验证 evaluator 是否有区分度

### Round 3

- 冻结 feasibility protocol，供后续所有 Step 02–05 使用


## 8. 成功标准

- 第三档路线不再是“自由发挥”，而是受统一 feasibility framework 约束
- 后续每条路线都能被快速归类，而不是无限拖延
