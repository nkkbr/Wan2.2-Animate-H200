# Step 06: H200 High-Value Orchestration After Route Selection

## 1. 为什么要最后做

前几轮已经证明：

- 如果路线本身不够强，再多 H200 算力也只能抬一些 proxy，不能真正解决边缘问题。

所以 `optimization6` 的 H200 orchestration 必须放在最后：

- 只有当 alpha / decoupled / ROI generation / trainable model 中，至少有一条路线已经证明有效时，
- 才值得把 H200 的预算认真编排成可生产的质量档位。


## 2. 目标

在已经证明有效的边缘路线之上，建立：

- `default`
- `high_quality`
- `extreme`

三档高价值算力编排方案。

这里的高价值不是“更多 sample steps”，而是：

- 更高质量 alpha
- 更强 ROI reconstruction
- 必要的多候选
- 合理的 runtime budget


## 3. 具体要做什么

### 3.1 建立 tier 配置

每档明确：

- preprocess profile
- external model profile
- ROI generation profile
- candidate count
- sample steps
- runtime budget

### 3.2 建立 orchestration runner

要求：

- 可显式配置 tier
- 可记录每一层的开销
- 可输出 runtime summary

### 3.3 建立 tier benchmark

至少比较：

- quality gain
- runtime cost
- GPU memory
- stability


## 4. 如何做

### 4.1 先冻结主路线

只有在 Step 02–05 中至少有一条有效主路线后，才能开始 Step 06。

### 4.2 先做小 tier，再做大 tier

建议：

- `default`：稳定生产
- `high_quality`：主推荐
- `extreme`：只在高价值 case 使用

### 4.3 编排指标

至少记录：

- preprocess wall time
- generate wall time
- ROI generation wall time
- max GPU memory
- final edge metrics


## 5. 如何检验

### 5.1 质量 gate

`high_quality` 必须相对 `default`：

- `roi_gradient` 至少提升 `8%`
- `roi_edge_contrast` 至少提升 `6%`
- `halo_ratio` 至少降低 `6%`
- `seam_score` 恶化不超过 `3%`

`extreme` 则必须相对 `high_quality`：

- 至少再有一项关键边缘指标明显改善
- 否则不值得存在

### 5.2 成本 gate

- `high_quality` runtime 不应超过 `default` 的 `2.5x`
- `extreme` runtime 不应超过 `high_quality` 的 `2x`


## 6. 三轮闭环规则

### Round 1

- 建 tier runner
- 先跑最小 tier benchmark

### Round 2

- 调整资源分配
- AB 对比 `default/high/extreme`

### Round 3

- 冻结最有价值的 tier 设计
- 如果 `extreme` 没价值，则直接删除


## 7. 成功标准

- 至少 `high_quality` tier 证明“更高质量且成本合理”；
- `extreme` 只有在客观更优时才保留；
- orchestration 配置可复现、可记录、可回退。


## 8. 风险与回退

### 风险

- 为不存在的质量收益支付大量算力；
- orchestration 变得过于复杂，不利生产；
- extreme tier 只有成本没有价值。

### 回退

- 若最终没有一档质量 tier 真正值得保留，则只保留最稳定的一档，不强行维持多 tier。
