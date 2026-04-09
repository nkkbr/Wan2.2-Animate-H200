# Step 05 Findings: Compositing-Aware Losses And Evaluation Stack

## 1. 结论

Step 05 已完成。

这一步的结论是：

- compositing-aware loss registry 已正式接入 Step 03 的 trainable alpha 训练路径
- `composite`
- `composite + gradient`
- `composite + gradient + contrast`

三类 loss stack 都完成了训练、full-bundle 推理、reviewed benchmark 评估，并对最佳候选补做了真实 smoke

但是：

- 三轮都没有通过 promotion gate
- reviewed holdout 改善都不到 `1%`
- 做了真实 smoke 的轮次都明显带坏 `seam` 和 `background_fluctuation`

因此：

- Step 05 的 loss stack 研究路径是成立的
- 但当前不能证明 “问题只是 loss 没定义对”
- 这条路线仍然不能升入主线

最终冻结结果：

- `best-of-three = Round 2`

冻结原因：

- Round 2 是三轮 reviewed 指标里最好的
- Round 3 明显回退
- Round 1 与 Round 2 都 smoke 不安全，但 Round 2 至少在 reviewed 维度上最强


## 2. 本步新增内容

核心文件：

- `wan/utils/edge_losses.py`
- `scripts/eval/check_compositing_aware_losses.py`
- `scripts/eval/evaluate_compositing_aware_training.py`
- `scripts/eval/train_trainable_alpha_model.py`（扩展 loss registry）

本步做成了：

- compositing-aware loss registry
- `composite_reconstruction_loss`
- `gradient_preservation_loss`
- `contrast_preservation_loss`
- reviewed + smoke 的统一评估入口


## 3. 轮次设计

### Round 1

- `loss_stack = composite_v1`

### Round 2

- `loss_stack = composite_grad_v1`

### Round 3

- `loss_stack = composite_grad_contrast_v1`

基线采用：

- `runs/optimization8_step03_round1/candidate_metrics_v3.json`
- `runs/optimization8_step03_round1_smoke/metrics/candidate_replacement_metrics.json`


## 4. Synthetic Correctness

`scripts/eval/check_compositing_aware_losses.py` 已通过。

核心结论：

- `good_composite < bad_composite`
- `good_gradient < bad_gradient`
- `good_contrast < bad_contrast`

说明：

- 三类 compositing-aware loss 在数值上是稳定的
- 对更差预测有更高响应
- 失败不在 loss 本身无法工作，而在它没有转化成 end-to-end gain


## 5. 训练阶段结果

### Round 1

- `loss_stack = composite_v1`
- `best_epoch = 2`
- best validation:
  - `boundary_f1_mean = 0.77696`
  - `boundary_roi_f1_mean = 0.81263`
  - `alpha_mae_mean = 0.01167`
  - `trimap_error_mean = 0.03683`

### Round 2

- `loss_stack = composite_grad_v1`
- `best_epoch = 1`
- best validation:
  - `boundary_f1_mean = 0.77698`
  - `boundary_roi_f1_mean = 0.81262`
  - `alpha_mae_mean = 0.01137`
  - `trimap_error_mean = 0.03669`

### Round 3

- `loss_stack = composite_grad_contrast_v1`
- `best_epoch = 8`
- best validation:
  - `boundary_f1_mean = 0.77603`
  - `boundary_roi_f1_mean = 0.81236`
  - `alpha_mae_mean = 0.01154`
  - `trimap_error_mean = 0.03686`

解释：

- Round 2 在训练验证上最稳
- Round 3 加入 contrast 后，并没有继续带来一致性收益


## 6. Reviewed Benchmark 结果

### Round 1

- `holdout_boundary_f1_gain_pct = 0.0000%`
- `holdout_alpha_mae_reduction_pct = +0.2078%`
- `holdout_trimap_error_reduction_pct = +0.2196%`
- `pass_count = 0`
- `gate_passed = false`

### Round 2

- `holdout_boundary_f1_gain_pct = +0.0061%`
- `holdout_alpha_mae_reduction_pct = +0.2492%`
- `holdout_trimap_error_reduction_pct = +0.2615%`
- `pass_count = 0`
- `gate_passed = false`

### Round 3

- `holdout_boundary_f1_gain_pct = -0.0124%`
- `holdout_alpha_mae_reduction_pct = -0.1736%`
- `holdout_trimap_error_reduction_pct = -0.1753%`
- `pass_count = 0`
- `gate_passed = false`

解释：

- Round 1 和 Round 2 有极微小正向变化
- 但量级不到 `1%`
- 远低于 Step 05 要求的“至少两项显著改善”
- Round 3 已经开始整体回退


## 7. Real-Video Smoke 结果

为了避免浪费资源，只对最有价值的候选补做 smoke：

- Round 1
- Round 2

### Round 1 smoke

- `seam_degradation_pct = -49.8381%`
- `background_fluctuation_improvement_pct = -60.4771%`
- `smoke_safe = false`

### Round 2 smoke

- `seam_degradation_pct = -50.9722%`
- `background_fluctuation_improvement_pct = -51.8290%`
- `smoke_safe = false`

解释：

- 两轮 reviewed 指标虽有极小提升
- 但这些提升没有转化成真实生成质量改善
- 反而在 smoke 上明显恶化了 seam 和背景稳定性

这说明：

- compositing-aware loss 不是“完全没学到东西”
- 但它正在把训练推向一种对 reviewed holdout 几乎无益、对实际 composite 反而更危险的方向


## 8. 为什么失败

当前最合理的失败解释有三条：

### 8.1 compositing-aware loss 仍然建立在不够强的 trainable alpha baseline 上

如果主模型本身能力不足，loss 再更像真实目标，也只能在极小范围内调整，不足以形成大幅提升。

### 8.2 当前 composite target 仍然过于依赖现有 ROI dataset 的近似前景表示

ROI dataset 中的 `foreground_patch` 是基于现有 alpha 的预乘前景近似，不是真正独立 foreground layer。

这意味着：

- compositing loss 是“更合理”
- 但还不是真正前景/背景解耦意义上的严格 composite supervision

### 8.3 训练目标和最终生成链之间仍然存在 domain gap

Step 05 再次证明了一个反复出现的问题：

- patch-level loss 可以优化
- reviewed holdout 只能微动
- generate smoke 甚至变差

这说明当前 trainable alpha route 仍然没有和最终生成主链充分对齐。


## 9. 最终判断

Step 05 的最终判断是：

- loss stack 工程化：成功
- compositing-aware 思路本身：值得保留
- 当前 trainable alpha 主线：仍不成立

不应做的事：

- 不要把任何一轮 compositing-aware loss stack 升为默认
- 不要继续在这条 trainable alpha baseline 上做更多小范围 loss 微调

应保留的部分：

- `edge_losses.py`
- compositing-aware synthetic correctness check
- reviewed + smoke 的统一 evaluator

这些内容对后续更强模型或更强数据仍然有价值。


## 10. 下一步建议

Step 05 结束后，最合理的下一步是继续 `optimization8 / Step 06`：

- `06-tier2-promotion-gate-and-amplification-decision.md`

原因：

- Step 03、Step 04、Step 05 已经形成连续证据：
  - trainable alpha baseline 不够
  - semantic experts 不够
  - compositing-aware losses 也不够

所以 Tier2 应该进入“是否继续放大、是否需要止损”的正式决策阶段，而不是继续在同一条线上追加更多局部修补。

