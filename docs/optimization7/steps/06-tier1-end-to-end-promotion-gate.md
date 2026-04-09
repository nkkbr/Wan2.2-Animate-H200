# Step 06: Tier-1 End-To-End Promotion Gate

## 1. 为什么要做

前面 5 个步骤即使分别做成，也不代表 Tier-1 路线就应该直接升为生产默认。

我们在 `optimization4/5/6` 已经吃过很多次亏：

- 局部看起来更强
- 但一到端到端综合评估就失败

所以 Step 06 的目的不是继续发明模块，而是：

> 用端到端强 gate 决定 Tier-1 路线是否真的值得进入生产主线。


## 2. 目标

做一次严格的端到端对照，比较：

- 当前稳定生产基线
- Tier-1 最优 alpha 路线
- Tier-1 decoupled core path
- Tier-1 ROI generative path

并给出最终结论：

- promote
- keep experimental
- reject


## 3. 要改哪些模块

建议新增：

- `docs/optimization7/benchmark/tier1_manifest.json`
- `scripts/eval/run_optimization7_tier1_e2e_benchmark.py`
- `scripts/eval/evaluate_optimization7_tier1_e2e_benchmark.py`
- `scripts/eval/check_optimization7_tier1_gate.py`

必要时扩展：

- `generate.py`
- `preprocess_data.py`
- `wan/utils/experiment.py`


## 4. 具体如何做

### 4.1 定义候选集合

候选不宜过多，建议只保留：

- `baseline_stable`
- `alpha_only_best`
- `decoupled_best`
- `roi_gen_best`
- `tier1_full_best`

### 4.2 benchmark 组织

至少包含：

- reviewed edge benchmark metrics 汇总
- 真实 10 秒 preprocess + generate AB
- runtime / memory / artifact completeness

### 4.3 最终决策规则

Tier-1 只有在以下条件满足时才允许 promote：

- reviewed benchmark 明显优于 baseline
- 真实 10 秒 benchmark 有可见质量提升
- seam/background/identity 没有明显回退
- runtime 仍在可接受范围


## 5. 如何检验

### 5.1 reviewed gate

- `boundary_f1`
- `trimap_error`
- `alpha_mae`
- `alpha_sad`
- `hair_edge_quality`

### 5.2 real-video gate

- `roi_gradient`
- `roi_edge_contrast`
- `roi_halo`
- `seam`
- `background_fluctuation`
- `face identity proxy`

### 5.3 engineering gate

- preprocess / generate 均可稳定运行
- metadata / runtime stats 完整
- fallback 明确


## 6. 强指标要求

建议 promotion gate：

- reviewed benchmark 至少两项核心指标显著优于 baseline
- `roi_gradient_gain_pct >= 8%`
- `roi_edge_contrast_gain_pct >= 8%`
- `roi_halo_reduction_pct >= 10%`
- `seam_degradation_pct <= 3%`
- `background_fluctuation` 不恶化超过 `5%`


## 7. 实现-测评-再改闭环

### Round 1

- 建 benchmark manifest
- 跑 baseline vs tier1_full 初版

### Round 2

- 根据失败点剔除弱候选，只保留最强 2-3 条路径
- 重跑端到端 AB

### Round 3

- 做最后一次最优路线确认
- 冻结 `promote / keep experimental / reject`


## 8. 成功标准

Step 06 完成的唯一真正标准是：

- 端到端证据足够强，能支持明确的 promotion 结论

如果结论是“不 promote”，这一步仍然算成功，只要它把结论做实了。
