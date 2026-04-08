# Step 01 Findings: Human-Reviewed Edge Benchmark and Hard Gates

## 1. 目标回顾

Step 01 的目标不是提升模型质量，而是把 `optimization4` 的 edge mini-set 升级成一套更可信、更可重复、更适合作为后续强 gate 的 reviewed benchmark。

本步骤要求：

- reviewed keyframe `>= 20`
- `face / hair / hand / cloth / occluded` 每类覆盖 `>= 3`
- GT metrics / gate 连续 3 次运行稳定一致


## 2. 本步骤实际交付

已新增：

- reviewed benchmark 文档：
  - [README.md](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/README.md)
  - [benchmark_manifest.step01.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/benchmark_manifest.step01.json)
  - [label_schema.edge_reviewed_v1.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/label_schema.edge_reviewed_v1.json)
  - [gate_policy.step01.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization5/benchmark/gate_policy.step01.json)
- 执行脚本：
  - [build_reviewed_edge_benchmark.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/build_reviewed_edge_benchmark.py)
  - [compute_reviewed_edge_metrics.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/compute_reviewed_edge_metrics.py)
  - [summarize_optimization5_validation.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/summarize_optimization5_validation.py)
  - [run_optimization5_validation_suite.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization5_validation_suite.py)

reviewed benchmark 额外提供：

- `boundary_type_map`
- `review_metadata`
- category-wise coverage 汇总
- review overlay contact sheet


## 3. 三轮执行情况

### Round 1

运行目录：

- [optimization5_step01_round1](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step01_round1)

结果：

- suite 主体跑通
- 但 gate 失败

失败原因：

1. `gate_result.json` 在生成自身时被误判成不存在
2. `hand` category coverage 为 `0`

根因：

- `summarize_optimization5_validation.py` 把当前输出文件也当成 pre-existing required output 检查
- `build_reviewed_edge_benchmark.py` 误把 `src_hand_tracks.json` 中的归一化坐标当成像素坐标

### Round 2

运行目录：

- [optimization5_step01_round2](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step01_round2)

修复后结果：

- `overall_passed = true`
- reviewed keyframe count = `24`
- category coverage：
  - `face = 23`
  - `hair = 24`
  - `hand = 16`
  - `cloth = 24`
  - `occluded = 24`

### Round 3

运行目录：

- [optimization5_step01_round3](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step01_round3)

目的：

- 复跑确认 reviewed benchmark / GT metrics / gate 稳定性

结果：

- 与 Round 2 一致
- `repeat_count = 3`
- `max_metric_delta = 0.0`


## 4. 最终冻结结果

本步骤冻结为 **Round 2 / Round 3 一致版本**。

正式 baseline suite：

- [optimization5_step01_round3](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step01_round3)

关键指标：

- `reviewed_keyframe_count = 24`
- `reviewed_ratio = 1.0`
- `boundary_f1_mean = 0.40097900956671445`
- `alpha_mae_mean = 0.01916349322224657`
- `trimap_error_mean = 0.29244185114900273`
- `face_boundary_f1_mean = 0.10942168938993709`
- `hair_boundary_f1_mean = 0.1264861173167629`
- `hand_boundary_f1_mean = 0.02276157951946537`
- `cloth_boundary_f1_mean = 0.3644010586377152`
- `occluded_boundary_f1_mean = 0.7951618943435954`


## 5. 人工抽查说明

已对 reviewed benchmark 的 review overlay contact sheet 做内部抽查：

- [review_spotcheck_contact_sheet.jpg](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization5_step01_round2/reviewed_edge_benchmark/review_spotcheck_contact_sheet.jpg)

抽查结论：

- `face`、`hair`、`cloth`、`occluded` 的分区整体合理
- `hand` 在部分帧较窄，但覆盖已达到 Step 01 的 category minimum
- 当前 reviewed benchmark 已经足以作为 `optimization5` 后续步骤的强 gate 起点

需要明确的一点：

- 这套 reviewed benchmark 仍然是 `internal_review_v1`
- 它比 `bootstrap_unreviewed` 明显更强
- 但还不是长期终局版人工精修真值


## 6. 结论

Step 01 已完成，并满足完成标准：

1. reviewed edge mini-benchmark 已正式落盘
2. GT metrics 与 gate 脚本可稳定重复运行
3. category-wise 边界指标可用
4. final gate 通过

因此：

- Step 01 `PASS`
- `optimization5` 后续步骤默认以这套 reviewed benchmark 为准
