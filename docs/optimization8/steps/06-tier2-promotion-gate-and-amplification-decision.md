# Step 06: Tier-2 Promotion Gate And Amplification Decision

## 1. 为什么要做

第二档路线的目标不是“训练出更多东西”，而是判断：

> 它是否真的能放大第一档中已经有效的路径。

如果做完前 5 步后，训练型路线仍然只是增加复杂度，而不能带来稳定质量收益，那么第二档就不该继续扩大投入。


## 2. 目标

对第二档所有候选做综合评估，并给出清晰结论：

- promote as amplifier
- keep experimental
- reject for now


## 3. 要改哪些模块

建议新增：

- `docs/optimization8/benchmark/tier2_manifest.json`
- `scripts/eval/run_optimization8_tier2_benchmark.py`
- `scripts/eval/evaluate_optimization8_tier2_benchmark.py`
- `scripts/eval/check_optimization8_tier2_gate.py`


## 4. 具体如何做

### 4.1 候选集合

建议只保留最有代表性的候选：

- `non_trainable_best_from_optimization7`
- `trainable_alpha_best`
- `semantic_experts_best`
- `best_loss_stack_variant`
- `tier2_full_best`

### 4.2 benchmark 维度

至少包含：

- reviewed benchmark
- semantic boundary metrics
- real 10 秒 smoke
- runtime / memory / maintenance cost

### 4.3 决策规则

只有当 trainable / data-driven 路线相对非训练基线出现稳定且综合的收益时，第二档才应继续扩大投入。


## 5. 如何检验

### 5.1 quality gate

- reviewed benchmark 至少两项核心指标显著优于第一档最优基线
- real-video smoke 至少有一组边缘指标出现可见收益

### 5.2 engineering gate

- 训练/推理成本可接受
- 维护复杂度没有高到不成比例


## 6. 强指标要求

建议 promotion gate：

- `boundary_f1` 相对第一档基线提升 `>= 8%`
- `alpha_mae` 下降 `>= 10%`
- `trimap_error` 下降 `>= 10%`
- `roi_gradient` 与 `roi_edge_contrast` 至少一项显著提升
- `seam/background` 不明显恶化


## 7. 实现-测评-再改闭环

### Round 1

- 建 tier2 manifest
- 跑 baseline vs trainable_alpha

### Round 2

- 增加 semantic_experts / best_loss_stack
- 跑综合 AB

### Round 3

- 冻结 promote / keep experimental / reject


## 8. 成功标准

Step 06 完成的唯一真正标准是：

- 第二档路线是否能被明确判定为“值得作为第一档的放大器继续存在”

如果答案是否定的，这一步仍然算成功，只要它把结论做实了。
