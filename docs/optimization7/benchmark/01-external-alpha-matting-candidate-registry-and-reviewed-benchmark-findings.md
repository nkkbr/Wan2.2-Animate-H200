# Step 01 Findings: External Alpha / Matting Candidate Registry And Reviewed Benchmark

## 1. 结论

Step 01 的工程目标已经完成，但质量 gate 没有通过。

也就是说：

- `optimization7` 的外部 alpha / matting candidate registry 已建立；
- 统一 adapter 基类与多模型 factory 已建立；
- `BackgroundMattingV2` 两个官方候选与 `Robust Video Matting` 一个官方候选已实际 benchmark；
- synthetic correctness 全部通过；
- reviewed benchmark AB 已完成；
- 但 **当前候选池没有赢家**，因此本步骤不能推进到 Step 02 主线集成。

这一步的冻结结论是：

> 保留 registry / adapter / benchmark 基础设施，明确宣布“本轮无赢家”，只有更新候选池后才允许重开下一轮外部模型 vetting。

## 2. 本轮接入的候选

本轮实际 benchmark 的候选共有 3 个：

1. `backgroundmattingv2_mobilenetv2_fp32`
2. `backgroundmattingv2_resnet50_fp32`
3. `rvm_mobilenetv3_fp32_torchscript`

registry 文件：

- [external_model_registry.step01.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/benchmark/external_model_registry.step01.json)

来源与许可证：

- `BackgroundMattingV2`
  - repo: `https://github.com/PeterL1n/BackgroundMattingV2`
  - release: `v1.0.0`
  - license: `MIT`
- `RobustVideoMatting`
  - repo: `https://github.com/PeterL1n/RobustVideoMatting`
  - release: `v1.0.0`
  - license: `GPL-3.0`

需要特别说明的是：

- `rvm_mobilenetv3_fp32_torchscript` 虽然工程上可接入，但许可证比 `BackgroundMattingV2` 更重；
- 即使质量过线，后续是否能进入主线仍必须额外做 license 审核。

## 3. 代码交付

本步骤新增或升级：

- [external_alpha_base.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/external_alpha_base.py)
- [external_alpha_backgroundmattingv2.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/external_alpha_backgroundmattingv2.py)
- [external_alpha_rvm.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/external_alpha_rvm.py)
- [check_external_alpha_candidate.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_external_alpha_candidate.py)
- [run_optimization7_external_alpha_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization7_external_alpha_benchmark.py)
- [evaluate_optimization7_external_alpha_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization7_external_alpha_benchmark.py)
- [README.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/benchmark/README.md)

## 4. 三轮闭环

### Round 1

完成：

- registry 建立
- 统一 adapter 基类建立
- `RVM` 候选下载与 hash 锁定
- synthetic correctness 脚本建立

中途发现并修正：

- `BackgroundMattingV2` 在过小分辨率下会因为 internal sampling top-k 越界而失败，因此 synthetic second-size 检查需要设置最小安全分辨率；
- `RVM` 在跨分辨率测试时必须先 `reset_sequence_state()`，否则 recurrent state 与当前输入尺寸不匹配。

修正后，3 个候选 synthetic correctness 全部通过。

### Round 2

开始正式 reviewed benchmark AB。

中途发现并修正：

- reviewed benchmark v2 标签已经不再提供旧版 `trimap`，而是 `trimap_unknown`；
- benchmark runner 需要直接适配 reviewed v2 schema。

修正后重跑正式 AB。

### Round 3

完成 3 候选正式 reviewed AB，并冻结 best-of-three。

结果目录：

- [optimization7_step01_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization7_step01_round2_ab)

最终 gate：

- [gate_result.json](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization7_step01_round2_ab/gate_result.json)

结论：

- `winner_model_id = null`
- `winner_passed = false`
- `all_models_failed = true`

## 5. Reviewed Benchmark 结果

baseline（来自 `optimization6 Step 01`）：

- `boundary_f1_mean = 0.400979`
- `trimap_error_mean = 0.292442`
- `alpha_mae_mean = 0.019163`
- `alpha_sad_mean = 4317.152`
- `hair_boundary_f1_mean = 0.571189`

### 5.1 backgroundmattingv2_mobilenetv2_fp32

metrics：

- `boundary_f1_mean = 0.025350`
- `trimap_error_mean = 0.188396`
- `alpha_mae_mean = 0.135923`
- `alpha_sad_mean = 30620.662`
- `hair_boundary_f1_mean = 0.016018`
- `runtime_sec_mean = 0.05963`

gate：

- `alpha_mae_reduction_pct = -609.28%`
- `trimap_error_reduction_pct = +35.58%`
- `hair_edge_quality_gain_pct = -97.20%`
- `boundary_f1_gain_pct = -93.68%`
- `passed_count = 1`
- `reviewed_gate_passed = false`

### 5.2 backgroundmattingv2_resnet50_fp32

metrics：

- `boundary_f1_mean = 0.029305`
 - `boundary_f1_mean = 0.029304`
- `trimap_error_mean = 0.182473`
- `alpha_mae_mean = 0.139391`
 - `alpha_sad_mean = 31402.064`
- `hair_boundary_f1_mean = 0.000000`
- `runtime_sec_mean = 0.04107`

gate：

- `alpha_mae_reduction_pct = -627.38%`
- `trimap_error_reduction_pct = +37.60%`
- `hair_edge_quality_gain_pct = -100.00%`
- `boundary_f1_gain_pct = -92.69%`
- `passed_count = 1`
- `reviewed_gate_passed = false`

### 5.3 rvm_mobilenetv3_fp32_torchscript

metrics：

 - `boundary_f1_mean = 0.047845`
 - `trimap_error_mean = 0.517785`
 - `alpha_mae_mean = 0.099394`
 - `alpha_sad_mean = 22391.405`
- `hair_boundary_f1_mean = 0.062435`
- `runtime_sec_mean = 0.08018`

gate：

- `alpha_mae_reduction_pct = -418.66%`
- `trimap_error_reduction_pct = -77.06%`
- `hair_edge_quality_gain_pct = -89.07%`
- `boundary_f1_gain_pct = -88.07%`
- `passed_count = 0`
- `reviewed_gate_passed = false`

## 6. 为什么失败

本轮失败原因非常一致：

- 外部候选在 `trimap_error` 上能够反映“unknown region 更激进”的特性；
- 但在当前 reviewed benchmark 上，它们都无法在：
  - `alpha_mae`
  - `boundary_f1`
  - `hair_boundary_f1`
  这些更直接反映边缘形状质量的指标上打赢当前稳定 baseline。

更具体地说：

- `BackgroundMattingV2` 两个候选都表现出：
  - trimap 好于 baseline
  - 但 alpha/hair/boundary 大幅差于 baseline
- `RVM` 的 alpha 绝对误差虽小于 BMV2，但：
  - trimap_error 明显差于 baseline
  - hair 和 boundary 仍显著差于 baseline
  - 还存在额外的 license 风险

## 7. best-of-three 冻结

best-of-three 冻结为：

- `backgroundmattingv2_mobilenetv2_fp32`

原因不是它过线，而是：

- 它是 3 个候选里唯一通过 `trimap` 强 gate 的；
- selection score 也是 3 者里最高；
- 它没有额外的 GPL 许可证问题。

但需要强调：

- 这只是 “best-of-three failed gate”
- **不是 winner**
- **不允许进入 Step 02**

## 8. smoke 判定

按照 Step 01 文档：

- 只有通过 reviewed gate 的候选，才允许进入真实 10 秒最小 smoke

本轮没有任何候选通过 reviewed gate，因此：

- **不进行真实 smoke**
- **不推进主线接入**

## 9. 冻结结论

Step 01 的最终状态是：

- registry：完成
- adapter：完成
- 3 候选 synthetic correctness：完成
- 3 候选 reviewed benchmark AB：完成
- winner：**无**
- 进入 Step 02：**不允许**

所以这一步的正式结论是：

> Step 01 是一次成功的 candidate registry 与 benchmark 工程步骤，但不是一次成功的外部 alpha 候选选优步骤。
