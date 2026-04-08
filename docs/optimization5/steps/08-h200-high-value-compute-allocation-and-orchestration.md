# Step 08: H200 High-Value Compute Allocation and Orchestration

## 1. 为什么要做

前几步的目标，是找到至少一到两条真正对边缘锐度有效的路线。  
只有在这些高价值模块已经明确之后，才值得正式做 H200 的生产编排与算力分配。

否则，算力优化只是在认真优化一个还没证明有效的方向。


## 2. 目标

在 H200 141GB 上建立正式生产策略：

- 哪些模块该高分辨率跑
- 哪些模块该多候选跑
- 哪些模块该缓存
- 哪些模块只在高质量 preset 下启用


## 3. 交付目标

1. H200 生产级 profile / preset
2. preprocess / generate / ROI orchestration
3. 候选搜索和自动选优编排
4. runtime / cost / quality tradeoff 文档
5. findings 文档


## 4. 范围

### In Scope

- compute budget policy
- module prioritization
- candidate orchestration
- production preset

### Out of Scope

- 新模型研发


## 5. 实施方案

### 5.1 先定义高价值模块

只有在前面步骤已经证明有效的模块，才进入高优先级预算：

- 高质量 alpha
- ROI generative reconstruction
- semantic ROI experts
- trainable edge refinement

### 5.2 分层预算

至少分成：

- default quality
- high quality
- extreme quality

每一层明确：

- preprocess resolution
- ROI resolution
- candidate count
- whether trainable refiner is enabled

### 5.3 自动选优

把 preprocess candidate search 与 generate candidate search 统一编排，形成：

- default path
- best-quality path


## 6. 如何验证

### 6.1 真实 benchmark

至少在 10 秒 benchmark 上测试：

- default
- high quality
- extreme quality

### 6.2 tradeoff 指标

- runtime
- peak memory
- edge quality
- seam/background stability


## 7. 强指标要求

至少要满足：

- `high quality` 相对 `default` 有明确 edge quality 提升
- `extreme quality` 相对 `high quality` 仍有正收益
- H200 peak memory 和 runtime 在可接受范围内


## 8. 实现-测评-再改闭环

### Round 1

- 建 profile / budget / orchestration
- 跑第一轮 quality tiers

### Round 2

若收益不明显，则调整：

- candidate count
- ROI resolution
- module enable set

### Round 3

若仍不划算，则冻结 best-of-three，并明确：

- 哪个 profile 值得成为默认
- 哪个 profile 只保留给特殊高质量场景


## 9. 风险

- 编排复杂度高
- 候选数量失控会拖垮 runtime
- 若前面高价值模块本身不够强，则本步骤收益有限


## 10. 完成标准

1. H200 生产级 compute allocation 方案已建立
2. default / high / extreme 三层质量路径可运行
3. 至少一条高质量路径对边缘质量有显著正收益
4. findings 文档明确给出生产建议
