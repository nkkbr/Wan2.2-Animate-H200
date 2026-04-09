# Step 05 Findings: Trainable Edge Models on Reviewed Data

## 结论

Step 05 的工程目标已经完成：

- reviewed split / trainer / checkpoint / inference adapter 路径已经打通；
- 训练型 edge model 可以真正训练出 checkpoint；
- 可以把模型接进 reviewed benchmark 的评估闭环。

但质量目标没有达成，因此这一步不能升主线，只能保留为研究支线。

## 本步骤新增内容

- `wan/utils/trainable_edge_model.py`
- `scripts/eval/build_trainable_edge_dataset.py`
- `scripts/eval/check_trainable_edge_model.py`
- `scripts/eval/train_trainable_edge_model.py`
- `scripts/eval/infer_trainable_edge_model.py`
- `scripts/eval/evaluate_trainable_edge_checkpoint_against_reviewed.py`
- `scripts/eval/evaluate_optimization6_trainable_edge.py`

## Round 1

### 训练

- split: `runs/optimization6_step05_split.json`
- config: `epochs=5`, `lr=1e-3`, `size=128x80`
- best val loss: `0.4568769037723541`

### GT + real smoke

- 预测目录：`runs/optimization6_step05_round1_infer/preprocess`
- GT 指标：`runs/optimization6_step05_round1_infer/edge_metrics.json`
- real smoke：`runs/optimization6_step05_round1_smoke`

相对 Step 01 baseline：

- `boundary_f1_gain_pct = -67.05%`
- `alpha_mae_reduction_pct = -835.53%`
- `trimap_error_reduction_pct = -148.88%`
- `hair_edge_quality_gain_pct = -62.35%`
- `roi_gradient_gain_pct = +2.31%`
- `roi_edge_contrast_gain_pct = +1.10%`
- `roi_halo_reduction_pct = +0.02%`

Round 1 已经清楚表明：当前 trainable route 不只是没有提升，而且明显差于 baseline。

## Round 2

### 训练

- config: `epochs=8`, `lr=5e-4`, `size=160x96`
- best val loss: `0.7462372779846191`

Round 2 在 validation loss 上已经比 Round 1 更差。

### 评估执行情况

为避免 full-bundle inference 成为主要瓶颈，本轮新增了：

- `evaluate_trainable_edge_checkpoint_against_reviewed.py`

用于直接在 reviewed GT 帧上评估 checkpoint，而不强依赖完整 preprocess bundle 重写。

但在最初那轮执行时，系统并不空载，因此当时的吞吐结论并不干净。

## Round 3

### 训练

- config: `epochs=10`, `lr=3e-4`, `size=160x96`
- 结果：`timeout(300s)`

Round 3 直接暴露出另一个问题：

- 这条 trainable path 当前不仅质量不过线；
- 在当时的机器状态下，训练预算内也没有形成更强证据。

## 空载复评

在清理掉异常 `root` CPU 进程、确认 GPU 空闲之后，针对 Step 05 又做了一次最小复评，目的是把“机器负载导致的工程吞吐问题”和“路线本身的质量问题”分开。

复评目录：

- `runs/optimization6_step05_cleanreeval`

### 直接 GT 评估

对已经训练好的 Round 1 / Round 2 checkpoint 直接在 reviewed GT 上复评：

- Round 1 直接 GT：
  - `boundary_f1_mean = 0.0081`
  - `alpha_mae_mean = 0.1898`
  - `trimap_error_mean = 0.7784`
  - 耗时约 `3s`
- Round 2 直接 GT：
  - `boundary_f1_mean = 0.1975`
  - `alpha_mae_mean = 0.4280`
  - `trimap_error_mean = 0.4638`
  - 耗时约 `2s`

结论很明确：

- 空载后，直接 GT 评估的工程吞吐是健康的；
- 但质量结论没有翻盘，仍然明显差于 baseline。

### 最小 infer + smoke

又对 Round 2 checkpoint 做了 reviewed-only 最小 infer 和 1 帧 smoke：

- infer：
  - 输出目录：`runs/optimization6_step05_cleanreeval/round2_infer/preprocess`
  - 耗时约 `3s`
- smoke：
  - 目录：`runs/optimization6_step05_cleanreeval_smoke`
  - 耗时约 `10s`

对应综合评估：

- `boundary_f1_gain_pct = -5.06%`
- `alpha_mae_reduction_pct = -2146.21%`
- `trimap_error_reduction_pct = -58.38%`
- `hair_edge_quality_gain_pct = +72.38%`
- `roi_gradient_gain_pct = +7.00%`
- `roi_edge_contrast_gain_pct = +3.08%`
- `roi_halo_reduction_pct = +0.07%`

这说明：

- 空载后，Step 05 的工程成本确实恢复正常；
- 但核心质量 gate 依然失败；
- 因而原结论要修正为：
  - “吞吐问题主要受过系统负载污染”
  - 但“trainable edge route 当前质量不过线”这一点没有改变。

## Best-of-three 冻结

按照文档规则，这一步在 3 轮后冻结：

- 质量上唯一完整可比的轮次是 Round 1；
- Round 2 没有形成比 Round 1 更强的证据；
- Round 3 则在训练预算内超时。

因此这一步的最终结论是：

- trainable edge route 已经具备研究基础设施；
- 在空载条件下，它的推理/最小 smoke 吞吐可以接受；
- 但当前 reviewed mini-set + 当前小模型 + 当前 loss 组合，仍然没有打赢 baseline；
- 不升主线，不进入默认生产路径。

## 对后续的意义

这一步最重要的价值不是效果，而是把下面这件事验证清楚了：

- 如果 reviewed 数据规模、独立性和任务定义还不够强，训练型 edge model 只会重复 `optimization5` 的问题：
  - 训练得出来；
  - 但无法在真正关心的边缘指标上客观打赢 baseline。

因此，后续若要继续训练型路线，前提必须是：

- 更强 reviewed 数据；
- 更清晰的任务定义；
- 更高价值的目标（例如 ROI generative reconstruction expert），而不是再训练一个弱监督 alpha/boundary 回归器。
