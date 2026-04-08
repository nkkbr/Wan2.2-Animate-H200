# Step 08 Findings: H200 多候选 Preprocess 与自动选优

## 1. 目标回顾

Step 08 的目标是把 H200 从“高精度单候选 preprocess 运行平台”升级为“多候选高精度 preprocess 选优平台”。

本步骤要求：

1. 同一段视频可以生成多个高价值 preprocess candidate。
2. 候选可以被客观打分。
3. 系统可以自动选出最优 bundle。
4. 自动选优结果要可解释、可重复。
5. 相对默认单候选，自动选中的 candidate 在综合分上应明显更好，并且 candidate search 本身要稳定。


## 2. 实施内容

### 2.1 新增多候选评分与选优基础设施

新增模块与脚本：

- [preprocess_candidate_selection.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/preprocess_candidate_selection.py)
- [select_best_preprocess_candidate.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/select_best_preprocess_candidate.py)
- [run_optimization3_candidate_search.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_optimization3_candidate_search.py)
- [check_candidate_selection.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_candidate_selection.py)

其中：

- `run_optimization3_candidate_search.py`
  - 负责 orchestration
  - 串行运行多候选 preprocess
  - 自动执行 contract 与四类 proxy 指标
  - 自动输出 candidate summary、selection result、summary csv/json
  - 可选串联 default vs selected generate smoke
- `select_best_preprocess_candidate.py`
  - 负责独立评分与选择
- `preprocess_candidate_selection.py`
  - 负责统一的 metric normalization、group score、total score 与 explainable ranking


### 2.2 新增 Step 08 candidate manifest 与 score policy

正式文件：

- [candidate_manifest.step08.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization3/benchmark/candidate_manifest.step08.json)
- [candidate_score_policy.step08.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization3/benchmark/candidate_score_policy.step08.json)

第一版候选数量保持在 `4`，避免暴力穷举：

1. `default_precision`
2. `face_focus_highres`
3. `motion_focus_highres`
4. `boundary_background_highres`

评分采用 5 组加权：

- boundary: `0.28`
- face: `0.26`
- pose: `0.24`
- background: `0.18`
- runtime: `0.04`

这符合 Step 08 的设计原则：

- 质量主导
- runtime 只作为轻量惩罚
- 选优必须可解释


## 3. 候选定义

### 3.1 default_precision

含义：

- 当前 optimization3 Step 02-06 冻结后的高精度单候选基线

主要特点：

- `multistage_preprocess_mode = h200_extreme`
- `face_analysis_mode = heuristic`
- `pose_motion_stack_mode = v1`
- `boundary_fusion_mode = v2`
- `bg_inpaint_mode = video_v2`
- `reference_normalization_mode = structure_match`


### 3.2 face_focus_highres

含义：

- 在默认高精度基线之上，优先争取 face tracking / rerun / ROI 精度

额外参数：

- `analysis_resolution_area = 960x540`
- 更低的 `face_rerun_difficulty_threshold`
- 更高的 `face_difficulty_expand_ratio`
- 更紧的 `face_tracking_max_center_shift`
- 更强的 `multistage_face_bbox_extra_smooth`


### 3.3 motion_focus_highres

含义：

- 在默认基线之上，更激进地追求 pose/motion stack 的平滑与局部 refine


### 3.4 boundary_background_highres

含义：

- 在默认基线之上，更激进地追求 boundary/band/background 稳定性


## 4. 预执行修正

在正式 benchmark 之前，修掉了两个 orchestration 级问题：

1. `multistage_preprocess_mode` 不能再写旧值 `v1`，当前 preprocess 只接受 `none/h200_extreme`。
2. generate 对照中的 replacement metrics 需要优先读取 lossless bundle 的 `src_mask.npz`，而不是旧的 `src_mask.mp4`。

这两个问题都属于 runner 接线错误，不属于算法本身问题。修正后重新执行正式 benchmark。


## 5. Round 1：打通多候选搜索

目录：

- [optimization3_step08_round1_candidate_search_v2](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step08_round1_candidate_search_v2)

目的：

- 验证候选 orchestration 跑通
- 验证评分器确实会选出一个非默认候选

结果：

- `selected_candidate = face_focus_highres`
- `default_candidate = default_precision`
- `selected_better_than_default = true`
- `score_margin_vs_default = 0.04655`

排名：

1. `face_focus_highres`: `0.5838`
2. `boundary_background_highres`: `0.5470`
3. `default_precision`: `0.5373`
4. `motion_focus_highres`: `0.3412`

最关键的优势来自：

- `face_center_jitter_mean`: `1.3288 -> 1.1452`

相对改善约：

- `13.82%`

这说明第一轮已经满足 Step 08 的最基础成功条件：

- orchestration 跑通
- 评分器没有退化成“永远选默认”
- 非默认候选在客观分上确实更好


## 6. Round 2：selected vs default generate AB

目录：

- [optimization3_step08_round2_generate_ab_v3](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step08_round2_generate_ab_v3)

目的：

- 验证 auto-selected preprocess bundle 不只是 preprocess proxy 分更高
- 还要确认下游 generate smoke 不回退

结果：

- `selected_candidate = face_focus_highres`
- `score_margin_vs_default = 0.10330`
- `selected_better_than_default = true`

group score 对比：

- `default_precision`
  - boundary: `0.5598`
  - face: `0.5031`
  - pose: `0.5500`
  - background: `0.4594`
  - runtime: `0.2061`
  - total: `0.5105`
- `face_focus_highres`
  - boundary: `0.5599`
  - face: `0.7975`
  - pose: `0.5500`
  - background: `0.4594`
  - runtime: `0.8750`
  - total: `0.6138`

这说明 Round 2 的提升本质上是：

- face group 显著提升
- 其他组基本不退化
- runtime 虽然增加，但在当前轻量惩罚下仍然值得

generate 对照结果：

- `seam_delta = -0.09047`
- `background_fluctuation_delta = -0.02433`

符号含义：

- 负值表示 selected 优于 default

并且 Step 08 runner 的正式 gate 现在已经把 generate 非回退写入 summary：

- `generate_nonregression_ratio = 1.0`
- `generate_nonregression_passed = true`

因此 Round 2 的结论是：

- auto-selected bundle 不仅 preprocess 分更高
- 下游 generate smoke 也没有回退
- seam 和背景波动反而更好


## 7. Round 3：3/3 稳定性复跑

目录：

- [optimization3_step08_round3_stability](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization3_step08_round3_stability)

目的：

- 验证 candidate search 是否可重复
- 验证 selected candidate 是否稳定

三轮结果：

1. `repeat_01`
   - selected: `face_focus_highres`
   - margin: `0.07770`
2. `repeat_02`
   - selected: `face_focus_highres`
   - margin: `0.07615`
3. `repeat_03`
   - selected: `face_focus_highres`
   - margin: `0.09699`

最终 gate：

- `selected_better_than_default_ratio = 1.0`
- `selection_stable_ratio = 1.0`
- `selection_stable_3_of_3 = true`
- `overall_passed = true`

这意味着：

- 3 轮都稳定选中同一个非默认候选
- 3 轮都稳定优于默认候选


## 8. 冻结判断

Step 08 现在可以冻结为 v1。

冻结内容：

1. 多候选 orchestration：保留
2. `candidate_manifest.step08.json`：保留
3. `candidate_score_policy.step08.json`：保留
4. 当前自动选优默认 winner：
   - `face_focus_highres`

原因：

- 在真实 10 秒素材上，selected candidate 明显优于默认基线
- 这种优势不只体现在 preprocess proxy
- 也体现在 generate smoke 的 seam/background 非回退
- 搜索本身 3/3 稳定


## 9. 限制与后续空间

这一步的胜利点很明确，但也要说准边界：

1. 当前 candidate 数仍然很小，只覆盖第一版高价值轴。
2. 当前 score policy 仍然是 proxy-driven，不是人工标注 quality-driven。
3. 当前 winner 基本由 face group 拉开差距，说明 boundary/motion/background 候选还没有像 face 候选那样带来稳定的大幅收益。

因此后续真正值得继续扩展的方向是：

- 引入高难 clip 集
- 引入 label-driven 权重拟合
- 让 Step 08 的候选搜索和 Step 07 rich conditioning / boundary refinement benchmark 联动


## 10. 最终结论

Step 08 的准确结论是：

- 多候选 preprocess 与自动选优：**完成**
- 自动选优相对默认单候选的综合分优势：**成立**
- selected vs default generate 非回退：**成立**
- 3/3 稳定性：**成立**

因此：

1. `auto-selection` 可以保留为高精度模式的正式能力。
2. 默认单候选主链仍然保留，作为更低成本回退路径。
3. 在当前高精度模式下，推荐把 `face_focus_highres` 作为第一版自动选优的首个稳定优胜解。
