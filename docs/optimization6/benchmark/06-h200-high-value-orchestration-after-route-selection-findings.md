# Step 06 Findings: H200 High-Value Orchestration After Route Selection

## 结论

Step 06 的工程目标已经完成：

- 已建立正式 tier manifest；
- 已建立 orchestration runner；
- 已建立 Step 06 的 tier evaluation 脚本；
- 已明确验证：在当前仍然没有选出更强边缘路线的前提下，`high_quality` / `extreme` 两档都不值得升生产默认。

也就是说，这一步的最终结论是：

- orchestration 基础设施保留；
- 但当前生产仍只保留单一稳定档；
- 不强行维持多 tier。

## 本步骤新增内容

- `docs/optimization6/benchmark/tier_manifest.step06.json`
- `scripts/eval/check_optimization6_orchestration.py`
- `scripts/eval/run_optimization6_orchestration_benchmark.py`
- `scripts/eval/evaluate_optimization6_orchestration_benchmark.py`

## 为什么最初没有直接做完整 live rerun

Step 06 的设计前提是：

- 只有当 Step 02–05 中至少有一条更强路线已经被证明有效时，
- 才值得把 H200 预算认真编排成 `default / high_quality / extreme` 多档。

而本轮到 Step 05 为止，结论恰恰是：

- 没有一条新边缘路线真正通过强 gate。

因此，最初先把：

1. tier manifest / runner / evaluator 正式做出来；
2. 再用 `optimization5 Step08` 已有的三轮真实 tier AB 做统一判定；

是合理的第一步。

## 第一阶段：复用的真实 tier 证据

复用的真实三轮 tier AB：

- `runs/optimization5_step08_round1_tiers`
- `runs/optimization5_step08_round2_tiers`
- `runs/optimization5_step08_round3_tiers`

使用新的 Step 06 evaluator 重新判定后，对应结果：

- `runs/optimization6_step06_round1_eval.json`
- `runs/optimization6_step06_round2_eval.json`
- `runs/optimization6_step06_round3_eval.json`

## 第一阶段三轮结果

### Round 1

- `high_quality` 相对 `default`
  - `seam_degradation_pct = +7.64%`
  - `runtime_increase_pct = +34.42%`
- `extreme` 相对 `high_quality`
  - `seam_degradation_pct = +6.37%`
  - `runtime_increase_pct = +22.70%`
- 结论：质量 gate 明确失败。

### Round 2

- `high_quality` 相对 `default`
  - `seam_degradation_pct = +10.24%`
  - `runtime_increase_pct = +15.67%`
- `extreme` 相对 `high_quality`
  - `seam_degradation_pct = -2.37%`
  - `runtime_increase_pct = +5.07%`
- 结论：虽然 `extreme` 的 seam 比 `high_quality` 好，但 `high_quality` 本身已经不值得存在，因此整轮仍然失败。

### Round 3

- `high_quality` 相对 `default`
  - `seam_degradation_pct = +5.60%`
  - `runtime_increase_pct = +12.52%`
- `extreme` 相对 `high_quality`
  - `seam_degradation_pct = +4.40%`
  - `runtime_increase_pct = -1.18%`
- 结论：这是三轮里“最不差”的一轮，但仍然没有达到 Step 06 设定的质量 gate。

## 第二阶段：空载 live rerun

在清理掉异常 `root` CPU 进程并确认 GPU 空闲后，又做了一次真正的 live tier rerun：

- suite：`runs/optimization6_step06_cleanreeval_v2`

这次 live rerun 还顺手修掉了两个 Step 06 脚本问题：

- `run_optimization6_orchestration_benchmark.py`
  - runtime summary 之前读错了字段层级；
  - ROI metrics 命令没有兼容 `boundary_refine_mode=none` 的主路径。
- `compute_boundary_roi_metrics.py`
  - 现在在没有 refinement debug band 时，会从 preprocess 的 `person_mask / boundary_band / soft_alpha / occlusion_band / uncertainty_map` 直接推导 ROI 区域，而不再报错退出。

live rerun 的结果：

- `default`
  - `total_generate_sec = 61.70`
  - `seam = 2.0762`
- `high_quality`
  - `total_generate_sec = 76.68`
  - `seam = 2.1266`
  - 相对 `default`
    - `roi_gradient_gain_pct = -17.47%`
    - `roi_edge_contrast_gain_pct = +17.19%`
    - `roi_halo_reduction_pct = -0.99%`
    - `seam_degradation_pct = +2.43%`
    - `runtime_increase_pct = +24.27%`
- `extreme`
  - `total_generate_sec = 94.45`
  - `seam = 2.3546`
  - 相对 `high_quality`
    - `roi_gradient_gain_pct = +20.46%`
    - `roi_edge_contrast_gain_pct = -20.80%`
    - `roi_halo_reduction_pct = +9.43%`
    - `seam_degradation_pct = +10.72%`
    - `runtime_increase_pct = +23.18%`

对应 gate：

- `high_quality_roi_gradient_ge_8pct = false`
- `high_quality_roi_edge_contrast_ge_6pct = true`
- `high_quality_roi_halo_reduction_ge_6pct = false`
- `high_quality_seam_degradation_le_3pct = true`
- `extreme_has_additional_key_gain = true`
- 但 `passed = false`

这个 live rerun 的意义很重要：

- 它说明之前 Step 06 的“吞吐/成本”判断确实可能被系统负载污染；
- 但在真正空载后，`high_quality` / `extreme` 依然没有形成可接受的综合收益；
- 所以最终“不升默认”的结论没有改变，只是证据更干净了。

## Best-of-three / best-evidence 冻结

按 Step 06 文档的规则，这一步冻结为：

- 第一阶段复用证据里，`Round 3` 是最接近可生产 tier 的历史结果；
- 第二阶段空载 live rerun 进一步确认了：即使在更干净的环境里，`high_quality / extreme` 也依然不过线；
- 因此 `high_quality / extreme` 不升默认。

## 最终判断

当前在没有更强路线被证明有效的前提下：

- `default`：保留，作为单一稳定生产档；
- `high_quality`：不保留；
- `extreme`：不保留。

这一步最重要的结论不是“多档成功了”，而是：

- H200 orchestration 必须建立在已经证明有效的质量路线之上；
- 如果路线本身没有通过强 gate，那么继续为其设计多档 tier 只会增加复杂度，而不会带来真实边缘收益。
