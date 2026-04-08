# Step 01 Findings: Edge-Labeled Mini Benchmark And Gates

## 1. 目标

Step 01 的目标不是直接提升边缘质量，而是先把后续所有 edge-sharpness 优化所依赖的 benchmark 与 gate 基础设施搭起来。具体包括：

- edge-labeled mini-set
- 真实边缘指标脚本
- 可执行 gate
- 可复用 baseline 记录方式

本 findings 文档记录这一基础设施真正跑通后的结果，而不是只记录设计意图。


## 2. baseline 设计与实际执行方式

Step 01 的 bootstrap label 来源使用冻结的高精度 preprocess bundle；baseline prediction 默认也复用一套冻结的高精度 preprocess bundle。

这样做的原因是：

- Step 01 的重点是建立稳定、低摩擦的 benchmark 与 gate
- 不应把 suite 默认成本绑死在一次次重跑重型 preprocess 上
- 只有在后续步骤需要验证“当前代码重新跑出的 preprocess 是否仍然稳定出分”时，才显式开启 baseline 重跑

本步骤最终使用的默认参考为：

- label source：
  - `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`
- baseline prediction：
  - `runs/optimization3_step08_round2_generate_ab_v3/repeat_01/face_focus_highres/preprocess`

对应 manifest 见：

- `docs/optimization4/benchmark/benchmark_manifest.step01.json`


## 3. Step 01 交付项

本步骤最终交付：

1. `benchmark_manifest.step01.json`
2. `label_schema.edge_v1.json`
3. `gate_policy.step01.json`
4. `extract_edge_benchmark_keyframes.py`
5. `compute_edge_groundtruth_metrics.py`
6. `summarize_optimization4_validation.py`
7. `run_optimization4_validation_suite.py`
8. 一套真实跑通的 edge mini-set baseline suite


## 4. 三轮闭环记录

### Round 1

完成内容：

- 建立 `edge mini-set` 构建脚本
- 从冻结的高精度 preprocess bundle 中抽取 `24` 个 keyframe
- 产出 bootstrap label PNG/JSON
- 首次跑通 `compute_edge_groundtruth_metrics.py`

暴露问题：

- label JSON 到标签根目录的解析写错，导致标签路径读取不正确
- 初版 `boundary_f1` 采用过于严格的像素级边界重叠，真实结果几乎全部压到 `0.0`

处理结果：

- 修正 label 根目录解析
- 保留这轮产出的 mini-set，不重新抽帧


### Round 2

完成内容：

- 将 `boundary_f1` 改成带容差的边界匹配方式
- 对冻结 baseline preprocess bundle 重算真值指标

结果：

- `boundary_f1_mean` 回到合理区间
- mini-set 与指标脚本具备可比较性


### Round 3

完成内容：

- 将 suite 默认模式改成复用 manifest 中声明的冻结 baseline preprocess bundle
- 重跑完整 suite，确认 gate 真正可执行
- 补齐 `summary.json / summary.md / gate_result.json`

最终结果：

- `all_cases_passed = true`
- `gate_passed = true`


## 5. 真实 baseline 结果

最终通过的 suite 为：

- `runs/optimization4_step01_core_v2`

对应输出物全部存在且通过 gate：

- `summary.json`
- `summary.md`
- `gate_result.json`

最终 baseline 数值如下：

- keyframe 总数：`24`
- `boundary_f1_mean = 0.40097900956671445`
- `alpha_mae_mean = 0.01916349322224657`
- `trimap_error_mean = 0.29244185114900273`
- `alpha_sad_mean = 4317.151753743489`
- `hard_mask_iou_mean = 0.88895904597636`
- `boundary_mask_iou_mean = 0.6302592073753078`
- `occlusion_band_iou_mean = 0.5445858904947006`
- `all_passed = true`
- `gate_passed = true`

对照 `docs/optimization4/benchmark/gate_policy.step01.json`，阈值检查全部通过：

- `minimum_keyframe_count >= 20`
- `boundary_f1_mean >= 0.30`
- `alpha_mae_mean <= 0.05`
- `trimap_error_mean <= 0.35`


## 6. 结果解释

Step 01 的成功含义是：

- 现在已经有一套真实视频派生的 edge mini benchmark
- 这套 benchmark 包含 `24` 帧关键帧，超过步骤要求的 `20`
- 至少 3 个边界真值指标已经真实跑通
- 当前高精度 baseline preprocess bundle 已经能稳定被这套 benchmark 打分
- 后续所有 edge-sharpness 相关步骤都可以复用这一套 gate，而不需要重新发明验证方式

必须明确说明的一点是：

- 当前 mini-set v1 仍然是 `bootstrap_unreviewed`
- 也就是说，它是“可运行、可比较、可复核”的第一版标注集
- 它还不是人工审核后的最终权威真值集

因此，Step 01 已经完成了“工程化 benchmark 与 gate 建设”，但没有完成“人工精修边界真值集”。


## 7. 完成判据

Step 01 应被判定为成功，当且仅当：

- edge mini-set 已存在
- 至少 20 帧关键帧被成功构建
- 至少 3 个真值指标可稳定跑通
- suite 能稳定产出 `summary.json / summary.md / gate_result.json`

本步骤当前已经满足上述判据。

若后续继续迭代本 benchmark，应优先推进：

- bootstrap label 的人工审核
- 更细的 hair/hand/cloth/face boundary 子标签
- 更强的 boundary F-score 与 trimap error gate
