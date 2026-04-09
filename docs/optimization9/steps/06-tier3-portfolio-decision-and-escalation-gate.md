# Step 06: Tier-3 Portfolio Decision And Escalation Gate

## 1. 为什么要做

第三档最大的风险，是做了很多高风险原型，却没有统一决策层。

因此最后必须有一步来回答：

- 哪条路线值得升级为下一阶段正式主线候选？
- 哪条路线只是 interesting but unproven？
- 哪条路线应彻底停止？


## 2. 目标

对 Step 02–05 的结果做 portfolio 级综合判断，并给出：

- `reject`
- `interesting_but_unproven`
- `upgrade_candidate`

三类最终结论。


## 3. 要改哪些模块

建议新增：

- `docs/optimization9/benchmark/portfolio_manifest.step06.json`
- `scripts/eval/run_optimization9_portfolio_summary.py`
- `scripts/eval/evaluate_optimization9_portfolio.py`
- `scripts/eval/check_optimization9_portfolio_gate.py`


## 4. 具体如何做

### 4.1 汇总维度

必须统一比较：

- reviewed benchmark gains
- error mode differences
- 真实 smoke gains
- runtime / memory / engineering complexity
- 与现有主线的结构差异价值

### 4.2 决策规则

- 只有出现明确新价值的路线，才允许进入 `upgrade_candidate`
- 只要仍然只是重现旧模式，即使实现很酷，也应归入 `reject`


## 5. 如何检验

### 5.1 consistency

- evaluator 能稳定读取各步骤输出
- portfolio bucket 规则明确

### 5.2 decision quality

- 至少能对每条路线给出清晰结论
- 不允许出现“继续看看再说”这种无限拖延式结论


## 6. 强指标要求

本步骤更关注决策质量，要求：

- 每条路线都必须被归类
- 至少一条路线若进入 `upgrade_candidate`，必须有清晰证据说明为什么
- 若没有任何路线升级，也必须明确说明整个第三档为何止损


## 7. 实现-测评-再改闭环

### Round 1

- 汇总 Step 02–05 初始结果

### Round 2

- 调整 portfolio 评分和 bucket 规则
- 做交叉验证

### Round 3

- 冻结最终 portfolio decision


## 8. 成功标准

- 第三档路线不再停留在“高风险想法列表”
- 而是有一套明确的升级/止损决策结果
