# Step 02 Findings: Production-Grade Alpha / Matting System

## 1. 目标

Step 02 的目标不是再做一版 heuristic-compatible alpha，而是把 `optimization5` 的 alpha / matting 路线正式升级成可以被后续边界重建信任的 production-grade artifact 系统，至少稳定输出：

- `soft_alpha`
- `trimap_unknown`
- `hair_alpha`
- `alpha_confidence`
- `alpha_source_provenance`

同时，本步骤要求在 reviewed edge mini-benchmark 上，至少有一套结果能显著优于 `optimization4` 的 frozen baseline；如果做不到，则必须保守收口，保留 artifact 而不切默认 active alpha。


## 2. 本步骤实际交付了什么

本步骤最终交付如下：

1. 新增 production-grade alpha refinement 主模块：
   - `wan/modules/animate/preprocess/alpha_refinement.py`
2. `matting_adapter.py` 升级为多路径：
   - `heuristic`
   - `high_precision_v2`
   - `production_v1`
3. `production_v1` 新增并贯通的 artifact：
   - `trimap_unknown`
   - `hair_alpha`
   - `alpha_confidence`
   - `alpha_source_provenance`
4. preprocess / metadata / contract 接入：
   - `wan/modules/animate/preprocess/process_pipepline.py`
   - `wan/modules/animate/preprocess/preprocess_data.py`
   - `wan/utils/animate_contract.py`
5. 新增 Step 02 评测脚本：
   - `scripts/eval/compute_alpha_precision_metrics.py`
   - `scripts/eval/check_alpha_matting_upgrade.py`


## 3. 关键设计变化

### 3.1 从 `alpha_v2` 兼容式 refinement，升级到 `production_v1`

`production_v1` 在 `heuristic` 与 `alpha_v2` 基础上进一步构造：

- `trimap_unknown`
- `hair_alpha`
- `alpha_confidence`
- `alpha_source_provenance`

并允许用更强的：

- head / hair prior
- hand prior
- occlusion prior
- fine boundary prior
- uncertainty prior

来形成 `alpha_v3` 路径。

### 3.2 明确引入 fallback policy

三轮实验的结果已经证明：

- 直接让 `alpha_v3` 接管 active `soft_alpha`
- 会在当前 reviewed benchmark 上显著退化

因此本步骤最终冻结的策略不是“production_v1 全面接管主链”，而是：

- `production_v1` 继续完整输出 richer alpha artifact
- active `soft_alpha` 允许回落到 legacy / heuristic 路线
- 使 `production_v1` 成为“artifact-first、active-alpha 保守”的实验路径

这就是本步骤最重要的工程收敛结论。


## 4. 测试设置

### 4.1 真实输入

- `ckpt_path`:
  `/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`
- 视频：
  `/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- 参考图：
  `/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`

### 4.2 reviewed benchmark

本步骤使用 Step 01 冻结通过的 reviewed benchmark：

- `runs/optimization5_step01_round3/reviewed_edge_benchmark`

基线对照采用 `optimization4` 最优冻结 preprocess：

- `runs/optimization3_step08_round2_generate_ab_v3/repeat_01/face_focus_highres/preprocess`

必须明确一点：

- 当前 reviewed edge mini-set 仍是 `bootstrap_unreviewed`
- 标签来源与现有高精度 preprocess 主链仍存在较强耦合
- 因此它非常擅长发现“明显退化”，但不一定足够公平地证明“新方法强于旧主线”

这会直接影响本步骤的 gate 解读。


## 5. 三轮闭环记录

### Round 1

真实 preprocess 目录：

- `runs/optimization5_step02_round1_v2/preprocess`

结果：

- `boundary_f1_mean = 0.12991649198452881`
- `alpha_mae_mean = 0.039944287622347474`
- `trimap_error_mean = 0.3534276559948921`
- `alpha_sad_mean = 8998.69243367513`

补充指标：

- `fine_boundary_iou_mean = 0.09874924815357349`
- `trimap_unknown_iou_mean = 0.22109180461624778`
- `uncertainty_error_focus_ratio_mean = 4.113278317219038`

结论：

- `production_v1` artifact 路线已经打通
- 但作为 active alpha，它相对 reviewed baseline 严重退化

### Round 2

真实 preprocess 目录：

- `runs/optimization5_step02_round2/preprocess`

主要改动：

- 降低 `alpha_v3_detail_boost`
- 降低 `alpha_v3_color_mix`
- 降低 `alpha_v3_active_blend`
- 收紧 `alpha_v3_delta_clip`

结果：

- `boundary_f1_mean = 0.15387748664629752`
- `alpha_mae_mean = 0.0398254357278347`
- `trimap_error_mean = 0.35194829603036243`
- `alpha_sad_mean = 8971.874237060547`

补充指标：

- `fine_boundary_iou_mean = 0.09874924815357349`
- `trimap_unknown_iou_mean = 0.22109180461624778`
- `uncertainty_error_focus_ratio_mean = 2.107767167682734`

结论：

- Round 2 相对 Round 1 只有非常有限的收回
- reviewed 主指标仍远差于 baseline

### Round 3

真实 preprocess 目录：

- `runs/optimization5_step02_round3/preprocess`

主要改动：

- `production_v1` 继续完整输出 richer artifact
- 将 `alpha_v3_active_blend = 0.0`
- 把 `production_v1` 收口成 artifact-only fallback 路线

结果：

- `boundary_f1_mean = 0.19346110785357504`
- `alpha_mae_mean = 0.0398006890512382`
- `trimap_error_mean = 0.35155156689385575`
- `alpha_sad_mean = 8966.299153645834`

补充指标：

- `fine_boundary_iou_mean = 0.09874924815357349`
- `trimap_unknown_iou_mean = 0.22109180461624778`
- `uncertainty_error_focus_ratio_mean = 2.106775595841199`

结论：

- Round 3 是 best-of-three
- 但它赢的方式不是“显著更强的 active alpha”
- 而是“保留 richer artifact，同时不再让 active alpha 继续伤主链”


## 6. 关键客观结果

### 6.1 reviewed baseline

目录：

- `runs/optimization5_step02_baseline_metrics/reviewed_edge_metrics.json`

指标：

- `boundary_f1_mean = 0.40097900956671445`
- `alpha_mae_mean = 0.01916349322224657`
- `trimap_error_mean = 0.29244185114900273`

### 6.2 heuristic probe

目录：

- `runs/optimization5_step02_heuristic_probe/reviewed_edge_metrics.json`

指标：

- `boundary_f1_mean = 0.19346110785357504`
- `alpha_mae_mean = 0.0398006890512382`
- `trimap_error_mean = 0.35155156689385575`

### 6.3 Round 1 / Round 2 / Round 3

| 版本 | boundary_f1 | alpha_mae | trimap_error |
|---|---:|---:|---:|
| reviewed baseline | `0.40098` | `0.01916` | `0.29244` |
| heuristic probe | `0.19346` | `0.03980` | `0.35155` |
| Round 1 | `0.12992` | `0.03994` | `0.35343` |
| Round 2 | `0.15388` | `0.03983` | `0.35195` |
| Round 3 | `0.19346` | `0.03980` | `0.35155` |

### 6.4 强 gate 判断

Step 02 目标要求相对 `optimization4` baseline 至少达到：

- `alpha_mae` 改善 `>= 20%`
- `trimap_error` 改善 `>= 20%`
- `hair_boundary_f1` 提升 `>= 10%`

最终判断：

- `alpha_mae`：失败
- `trimap_error`：失败
- `hair_boundary_f1`：失败

所以本步骤**没有达到强 gate**。


## 7. generate 兼容性

本步骤最终使用 Round 3 bundle 做了真实 generate smoke。

目录：

- preprocess:
  - `runs/optimization5_step02_round3/preprocess`
- generate smoke:
  - `runs/optimization5_step02_generate_smoke`

输出：

- `runs/optimization5_step02_generate_smoke/outputs/replacement_smoke_none.mkv`

该 smoke 仅用于确认：

- preprocess 新 artifact 不会破坏 generate 链路
- metadata / contract / loader 均保持兼容


## 8. 最终结论

Step 02 的准确结论不是“production-grade alpha 已经成功替换主链”，而是：

1. production-grade alpha / trimap / hair-alpha / confidence / provenance artifact 系统已经正式落地
2. reviewed benchmark、真实 preprocess smoke、generate 兼容 smoke 都已经跑通
3. 但在当前 reviewed benchmark 上，`production_v1` active alpha **没有客观打赢 `optimization4` baseline**
4. 因此：
   - `production_v1` 只能保留为实验路径
   - 当前更合理的冻结策略是：
     - 保留 richer artifact
     - 不让 `production_v1` 接管默认 active `soft_alpha`

也就是说，本步骤真正的产出是：

> 更强的 alpha artifact 基础设施已经做成，但“仅靠更复杂的 alpha fusion 就能显著提高边缘锐度”这一假设，在当前 benchmark 上没有被证明成立。

这也正是为什么后续步骤必须继续推进：

- richer boundary signal 主生成条件化
- boundary ROI 生成式重建
- semantic ROI expert
- 以及最终可能需要的 trainable edge refinement 路线

