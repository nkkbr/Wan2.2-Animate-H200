# Optimization4 Step 05 Findings: Generate-Side Edge Candidate Search

## 1. 目标

Step 05 的目标不是再发明新的边缘增强算法，而是回答一个更直接的问题：

- 在当前已经实现的 generate-side edge 路径中，
- 是否存在一组候选配置，能稳定优于默认 `legacy + none`，
- 并且至少在一个真实 case 上同时满足：
  - `halo_ratio` 下降
  - `band_gradient` 上升
  - `band_edge_contrast` 上升


## 2. 本步骤实际交付

### 2.1 新增 generate candidate selection 基础层

新增：

- [generate_candidate_selection.py](/home/user1/wan2.2_20260407/Wan2.2/wan/utils/generate_candidate_selection.py)

提供：

- candidate manifest 读取
- score policy 读取
- case-level generate candidate 打分
- multi-case 汇总
- `selected_better_than_default_ratio`
- `selection_stable_ratio`
- `positive_edge_triplet_ratio`

### 2.2 新增 benchmark / selection / gate 脚本

新增：

- [run_generate_edge_candidate_search.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/run_generate_edge_candidate_search.py)
- [select_best_generate_candidate.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/select_best_generate_candidate.py)
- [evaluate_optimization4_generate_candidate_search.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/evaluate_optimization4_generate_candidate_search.py)
- [check_generate_candidate_selection.py](/home/user1/wan2.2_20260407/Wan2.2/scripts/eval/check_generate_candidate_selection.py)

### 2.3 正式 benchmark 配置文件

新增：

- [candidate_manifest.step05.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/benchmark/candidate_manifest.step05.json)
- [candidate_score_policy.step05.json](/home/user1/wan2.2_20260407/Wan2.2/docs/optimization4/benchmark/candidate_score_policy.step05.json)

当前冻结版本对应 **Round 3** 的最终 shortlist：

- `default_legacy`
- `legacy_loose`
- `legacy_ultra_loose`
- `semantic_cond_only`


## 3. 测试设置

### 3.1 preprocess bundle

统一使用：

- [optimization4_step06_round1_preprocess/preprocess](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step06_round1_preprocess/preprocess)

原因：

- artifact 最完整
- contract 已验证稳定
- 适合作为 generate-side edge 搜索的固定输入

### 3.2 Round 1 候选

Round 1 先从 6 个高价值候选开始：

- `default_legacy`
- `rich_safe`
- `roi_balanced`
- `semantic_balanced`
- `local_edge_balanced`
- `hybrid_face_safe`

真实目录：

- [optimization4_step05_round1_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step05_round1_ab)

### 3.3 Round 2 候选

Round 1 已证明：

- 重型 refine 候选运行更慢
- 且很难改善 contrast

因此 Round 2 缩到 conditioning-only 路线：

- `default_legacy`
- `legacy_loose`
- `rich_loose`
- `rich_mid`
- `semantic_cond_only`

真实目录：

- [optimization4_step05_round2_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step05_round2_ab)

### 3.4 Round 3 候选与 case

Round 3 固定最终 shortlist，并正式做 3 个真实 case 的稳定性验证：

候选：

- `default_legacy`
- `legacy_loose`
- `legacy_ultra_loose`
- `semantic_cond_only`

真实 case：

- `short_33f`
- `baseline_45f`
- `long_53f`

真实目录：

- [optimization4_step05_round3_ab](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step05_round3_ab)


## 4. 三轮闭环

### Round 1

结果：

- 默认 `default_legacy` 直接排第一
- `selected_better_than_default_ratio = 0.0`
- `selection_stable_ratio = 1.0`
- `positive_edge_triplet_ratio = 0.0`

关键现象：

- `rich_safe` 是最好的非默认候选
- 但仍然：
  - `halo` 下降失败
  - `contrast` 仍为负
- `roi / semantic / local_edge / hybrid` 都没有打赢默认

结论：

- 不能继续把 runtime 花在重型 refine 候选上
- Round 2 改为 conditioning-only 搜索

### Round 2

结果：

- `legacy_loose` 成为选中候选
- `selected_better_than_default_ratio = 1.0`
- `selection_stable_ratio = 1.0`
- `positive_edge_triplet_ratio = 0.0`

说明：

- 这证明 generate candidate selection **不是完全没用**
- 但 `legacy_loose` 打赢默认的原因主要是：
  - runtime 更低
  - seam / background 不更差
  - `halo` 轻微变好
- 而不是出现了真正的“边缘三项一起变好”

关键数值：

- `halo_reduction_pct = +0.00051%`
- `gradient_gain_pct = +0.00004%`
- `contrast_gain_pct = -0.00014%`

结论：

- 单 case 下可以选出一个“略优于默认”的 candidate
- 但还不能证明它在多 case 上稳定成立

### Round 3

结果：

- 总体选中候选：`legacy_ultra_loose`
- 但：
  - `selected_better_than_default_ratio = 0.3333`
  - `selection_stable_ratio = 0.3333`
  - `positive_edge_triplet_ratio = 0.0`

关键现象：

- `short_33f` 上：`legacy_ultra_loose` 略优于默认
- `baseline_45f` 与 `long_53f` 上：默认仍然排第一
- 也就是说，所选 candidate **不稳定**

Round 3 gate 结果：

- [gate_result.json](/home/user1/wan2.2_20260407/Wan2.2/runs/optimization4_step05_round3_ab/gate_result.json)

为：

- `selected_better_than_default_ratio_ge_0.8 = false`
- `selection_stable_ratio_ge_0.8 = false`
- `positive_edge_triplet_ratio_gt_0 = false`
- `overall_passed = false`


## 5. 最终判断

### 5.1 是否达成成功标准

没有。

原因非常明确：

1. 最终选中的 candidate 不能稳定优于默认
2. 没有任何 case 达成：
   - `halo_ratio` 下降
   - `gradient` 上升
   - `contrast` 上升
   的正向 triplet
3. 这说明当前 generate-side heuristic candidate 库虽然可以做“相对更稳”的选择，
   但还不能真正产出“明显更锐”的边缘版本

### 5.2 best-of-three 冻结

这一步没有“质量上成功的赢家”。

因此冻结策略是：

- **框架层**：保留 Step 05 的 candidate search / score / gate 基础设施
- **配置层**：保留 Round 3 的 shortlist manifest 与最终 policy 作为正式文档版本
- **默认 generate 配置**：**不改变**，仍保持 `default_legacy`


## 6. 这一步说明了什么

Step 05 的价值不在于“已经找到了最佳边缘方案”，而在于把一个重要判断做实了：

- 在当前 `legacy / rich / roi / semantic / local_edge` 这些 heuristic / deterministic generate-side edge 路线中，
- 即使做多候选自动选优，
- 也无法稳定选出一个在真实 case 上明显优于默认、并真正提升边缘锐度的方案

这说明：

1. candidate search 框架是有用的
2. 但 candidate pool 本身还不够强
3. 下一步不该继续死磨这组 deterministic 候选，而应转向：
   - 更强的 semantic/learning-based edge reconstruction
   - 或更强的 ROI-local generative/detail restoration 路径


## 7. 最终结论

Step 05：

- **工程交付：成功**
- **质量 gate：失败**

它成功回答了一个非常重要的问题：

> 当前 generate-side heuristic edge enhancement 路线，就算上 candidate search，也还不足以稳定打赢默认 generate 配置。
