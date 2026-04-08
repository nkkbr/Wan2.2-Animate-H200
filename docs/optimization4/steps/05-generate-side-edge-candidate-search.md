# Step 05: Generate-Side Edge Candidate Search

## 1. 目的

本步骤的目标是把 Step 08 在 preprocess 侧已经证明有效的“多候选 + 自动选优”机制，正式复制到 generate-side edge refinement。


## 2. 为什么要做

当前 Step 07 的结论不是“generate 端边缘增强不可能成功”，而是：

- 当前固定参数下的 `boundary_refinement v2` 没过 gate

这意味着最合理的下一步不是停止尝试，而是：

- 用 H200 的预算去自动搜索更好的 generate-side 边缘参数组合


## 3. 交付目标

本步骤结束时，应具备：

1. generate-side candidate manifest
2. edge-scoring policy
3. selected-vs-default 自动对照
4. 3/3 稳定性搜索结果


## 4. 范围

优先新增：

- `scripts/eval/run_generate_edge_candidate_search.py`
- `scripts/eval/select_best_generate_candidate.py`
- `wan/utils/generate_candidate_selection.py`

候选参数可包括：

- `boundary_refine_mode`
- `boundary_refine_strength`
- `boundary_refine_sharpen`
- `replacement_boundary_strength`
- `replacement_transition_low`
- `replacement_transition_high`
- `roi_refine_strength`（若 Step 04 完成）


## 5. 实施方案

### 5.1 先定义少量高价值候选

不要从几十组开始。  
第一版建议 4 到 6 组：

- conservative
- balanced
- edge_aggressive
- face_safe
- hair_safe
- background_safe

### 5.2 定义评分函数

建议同时考虑：

- halo
- gradient
- contrast
- seam
- background fluctuation
- runtime

### 5.3 允许不同 preprocess bundle 重复使用

推荐默认使用当前 Step 08 选出的 preprocess best candidate，减少变量干扰。


## 6. 强指标要求

至少要求：

- `selected better than default ratio >= 0.8`
- `selection stable ratio >= 0.8`
- selected 版本在至少一个真实 case 上：
  - `halo_ratio` 下降
  - `gradient` 上升
  - `contrast` 上升


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 建立候选搜索框架
- 跑 1 个真实 case

### Round 2

- 修评分权重
- 让 selected 更稳定

### Round 3

- 固定候选集合与权重
- 决定是否升默认


## 8. 风险

### 8.1 过度搜索导致 runtime 爆炸

应对：

- 候选数保持小而精
- 优先搜索高价值参数

### 8.2 选优只对单个 case 有效

应对：

- 至少对 2 到 3 个真实 case 做稳定性验证


## 9. 完成标准

只有当自动选优能稳定优于默认 generate 配置，本步骤才算成功。
