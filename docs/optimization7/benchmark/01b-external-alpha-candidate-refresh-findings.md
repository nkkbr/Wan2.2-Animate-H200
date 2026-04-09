# Step 01b Findings: External Alpha Candidate Refresh

## 1. 结论

Step 01b 的目标是更新外部 alpha / matting 候选池，而不是继续在 `BackgroundMattingV2 / RVM`
旧家族内微调。

这一轮的工程目标已经完成：

- 新候选 registry 已建立；
- 新家族 `MatAnyone` 已被正式接入统一 adapter / benchmark 框架；
- sequence + first-frame mask 语义已经进入 reviewed benchmark；
- synthetic correctness、reviewed benchmark 和 gate 全部跑通。

但质量结论仍然是：

> 本轮仍然没有赢家，`optimization7` 不能进入 Step 02。

Step 01b 的冻结结论是：

- 保留新的 sequence adapter 与 refreshed registry；
- 将 `MatAnyone` 记为 `benchmarked_step01b_failed_gate`；
- 如果要继续 `optimization7`，必须再换一批候选家族，而不是继续调这一个。

## 2. 为什么要做 Step 01b

Step 01 的 3 个候选：

- `backgroundmattingv2_mobilenetv2_fp32`
- `backgroundmattingv2_resnet50_fp32`
- `rvm_mobilenetv3_fp32_torchscript`

都已经明确输给当前 reviewed baseline。

因此 Step 01b 的核心目标不是增加更多同家族变体，而是验证：

> 一个真正不同家族的视频 matting 模型，能否在同一套 reviewed benchmark 上打赢当前基线。

这一轮选择的候选是：

- `matanyone_v1_official`

原因：

- 它是明确面向 video matting 的新家族；
- 官方 repo、官方权重、官方 inference 路径都存在；
- 它原生需要 `first-frame mask + sequence memory`，与当前项目的 replacement preprocess 语义兼容。

## 3. 本轮接入的候选

registry 文件：

- [external_model_registry.step01b.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/benchmark/external_model_registry.step01b.json)

本轮实际 benchmark 的新候选：

1. `matanyone_v1_official`

来源：

- repo: `https://github.com/pq-yang/MatAnyone`
- release: `v1.0.0`
- weight: `https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth`
- sha256:
  `dd26b991d020ed5eb4be50996f97354c45cfdfc0f59958e8983ac6a198f4809d`
- license: `S-Lab License 1.0`

另有 2 个候选作为后续计划记录，但没有在本轮进入 benchmark：

- `modnet_portrait_video_planned`
- `matte_anything_planned`

## 4. 代码交付

本轮新增或修改：

- [external_alpha_matanyone.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/external_alpha_matanyone.py)
- [external_alpha_base.py](/home/user1/wan2.2_20260407/Wan2.2/wan/modules/animate/preprocess/external_alpha_base.py)
- [check_external_alpha_candidate.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_external_alpha_candidate.py)
- [run_optimization7_external_alpha_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization7_external_alpha_benchmark.py)
- [README.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization7/benchmark/README.md)

关键工程变化：

- adapter factory 新增 `video_matting_with_first_frame_mask`；
- benchmark runner 不再只支持单帧 candidate；
- 对于 sequence candidate，会：
  - 从原始视频中按 preprocess stride 还原序列；
  - 使用 stable preprocess bundle 的首帧 mask 初始化；
  - 仅在 reviewed keyframe 上取指标。

## 5. 三轮闭环

### Round 1

接通 `MatAnyone` adapter 与 reviewed benchmark。

结果目录：

- [optimization7_step01b_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization7_step01b_round1_ab)

关键结果：

- `boundary_f1_mean = 0.09979`
- `trimap_error_mean = 0.47810`
- `alpha_mae_mean = 0.09341`
- `hair_boundary_f1_mean = 0.07299`

判断：

- 能跑通
- 明显不过线

本轮识别出的主要问题不是“模型完全不可用”，而是 sequence warmup 逻辑不够贴近官方路径。

### Round 2

将 warmup 逻辑改为：

- 首帧用 mask 编码；
- 再对首帧重复 warmup；
- 然后才开始正式时序推进。

结果目录：

- [optimization7_step01b_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization7_step01b_round2_ab)

关键结果：

- `boundary_f1_mean = 0.09590`
- `trimap_error_mean = 0.38367`
- `alpha_mae_mean = 0.08575`
- `hair_boundary_f1_mean = 0.17564`

相对 Round 1：

- `trimap_error` 有明显改善
- `alpha_mae` 有改善
- `hair_boundary_f1` 有明显改善

但仍然远弱于 baseline。

### Round 3

再调整首帧初始化来源：

- 优先使用 stable preprocess bundle 中更细的 `hard_foreground`；
- 若不存在，才回退到 `person_mask`。

结果目录：

- [optimization7_step01b_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization7_step01b_round3_ab)

结果与 Round 2 本质一致：

- `boundary_f1_mean = 0.09590`
- `trimap_error_mean = 0.38367`
- `alpha_mae_mean = 0.08575`
- `hair_boundary_f1_mean = 0.17564`
- `runtime_sec_mean = 0.08489`

因此 best-of-three 冻结为：

- **Round 2 / Round 3 等效，按 Round 2 行为记为最终实现**

## 6. Gate 结果

最终 gate：

- [gate_result.json](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization7_step01b_round3_ab/gate_result.json)

结论：

- `winner_model_id = null`
- `winner_passed = false`
- `all_models_failed = true`

Round 3 相对 baseline 的具体 gate：

- `alpha_mae_reduction_pct = -347.45%`
- `trimap_error_reduction_pct = -31.20%`
- `hair_edge_quality_gain_pct = -69.25%`
- `boundary_f1_gain_pct = -76.08%`

也就是说：

- 没有一项强 gate 通过；
- 它不是“差一点赢”，而是整体明显输掉。

## 7. 这次失败说明了什么

Step 01b 的价值不在于找到了赢家，而在于证明了两件事：

1. `optimization7` 的 reviewed benchmark 框架足以公平评测 sequence + first-frame-mask 类候选；
2. 即使换到 `MatAnyone` 这种不同家族的视频 matting 模型，当前 reviewed benchmark 上仍然没有形成足够强的 edge quality 优势。

这说明：

- `optimization7` 仍然值得继续的唯一方式，是再换候选家族；
- 不值得继续在 `MatAnyone` 这一个候选上做更多小参数调节。

## 8. Promotion 判断

本轮不允许进入 Step 02，原因有两层：

1. **质量层面**：reviewed gate 明确失败；
2. **许可证层面**：即使质量过线，`S-Lab License 1.0` 也仍然需要单独审查。

因此它现在的正确定位是：

- 一个被真实 benchmark 过的新家族候选；
- 不是 production winner；
- 不是 Step 02 的输入基础。

## 9. 最终冻结结论

Step 01b 完成了应该完成的事：

- 新候选池刷新：完成
- 新 adapter：完成
- sequence reviewed benchmark：完成
- synthetic correctness：完成
- 3 轮闭环：完成

但最终结论是：

> `MatAnyone` 不是当前 reviewed benchmark 上的赢家，`optimization7` 不应进入 Step 02。

因此，下一步如果继续 `optimization7`，只能做：

- **再换一批新的外部 alpha / matting 候选**

而不能做：

- 在 `MatAnyone` 上继续盲目微调
- 强推它进入主线
