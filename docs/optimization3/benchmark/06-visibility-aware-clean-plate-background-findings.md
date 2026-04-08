# Step 06 Findings: 可见性驱动的 Clean Plate Background 2.0

## 1. 结论

Step 06 已完成，但结论需要说准：

- 工程交付已经完成。
- `clean_plate_video_v2` 已经正式接入 preprocess 主链，并产出：
  - `src_visible_support`
  - `src_unresolved_region`
  - `src_background_confidence`
  - `src_background_source_provenance`
- generate 已经能够稳定消费这些新 background artifact。
- 真实 10 秒 replacement case 上，`clean_plate_video_v2` 相对 `clean_plate_video` 已经显著降低：
  - `temporal_fluctuation_mean`
  - `band_adjacent_background_stability`
  - `unresolved_ratio_mean`

但本步骤设定的 4 条 gate 中，最激进的一条：

- `band_adjacent_background_stability <= 0.27`

在 5 轮内没有达到。

因此，本步骤的最终判断是：

- **工程目标达成**
- **4 项核心交付均完成**
- **4 条 gate 中 3 条达成**
- **`clean_plate_video_v2` 已冻结为当前高精度背景默认实现**
- **remaining gap 明确保留到后续 generate-rich-signal/background refinement 步骤继续处理**


## 2. 本步骤实际交付

### 2.1 新增和改动模块

主改动文件：

- `wan/modules/animate/preprocess/background_clean_plate.py`
- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/utils/animate_contract.py`
- `wan/utils/replacement_masks.py`
- `wan/animate.py`

指标与回归：

- `scripts/eval/compute_background_precision_metrics.py`
- `scripts/eval/check_video_clean_plate_background.py`
- `scripts/eval/run_visibility_aware_background_benchmark.py`
- `scripts/eval/run_optimization3_validation_suite.py`

### 2.2 新 artifact

preprocess 现在会额外输出：

- `src_visible_support`
- `src_unresolved_region`
- `src_background_confidence`
- `src_background_source_provenance`

并在 metadata 中声明对应语义：

- `background_visible_support`
- `background_unresolved_region`
- `background_confidence`
- `background_source_provenance`

### 2.3 新 background 语义

这一轮真正新增的核心能力包括：

1. `visible_support_map`
   - 聚合可见次数与可见时间跨度
2. `background_confidence`
   - 结合支持度与一致性偏差估计
3. `unresolved_region`
   - 从硬不可恢复区域改成 soft unresolved severity
4. `background_source_provenance`
   - 显式区分 `image_fallback / global_temporal / local_temporal`
5. boundary-aware temporal smoothing
   - 针对边界带和低置信区域做双向平滑

### 2.4 generate 侧接线

这一步最重要的不是“多导出几个 npz”，而是：

- `visible_support`
- `unresolved_region`
- `background_confidence`
- `background_source_provenance`

已经真正作用到 `compose_background_keep_mask(...)` 的背景约束构造。

这保证了低置信背景不会再被一视同仁地强行当成高置信 conditioning。


## 3. 基线

与本步骤对照的真实 10 秒 `clean_plate_video` 基线来自：

- `runs/optimization3_step06_round1_ab/preprocess_video`

关键基线指标：

- `temporal_fluctuation_mean = 0.9144021190550863`
- `band_adjacent_background_stability = 0.34617632950606936`
- `unresolved_ratio_mean = 0.8762372136116028`

Step 06 的目标是相对这个 `clean_plate_video` 基线继续向前推进，而不是只证明 `video > image`。


## 4. Round 1

### 4.1 目标

Round 1 的目标是：

- 打通 `visible_support / unresolved / confidence / provenance`
- 让 `clean_plate_video_v2` 真实 10 秒 case 跑通
- 确认 metadata 和 generate 兼容

### 4.2 结果

真实运行目录：

- `runs/optimization3_step06_round1_ab`

Round 1 `video_v2`：

- `temporal_fluctuation_mean = 0.9251269807620924`
- `band_adjacent_background_stability = 0.3401570602161384`
- `unresolved_ratio_mean = 0.7562512159347534`
- `background_confidence_mean = 0.15125226974487305`
- `support_confidence_corr = 0.7957742558450926`
- `support_unresolved_corr = -0.9336489103577615`
- `confidence_unresolved_corr = -0.6394438186147225`

判断：

- schema/artifact/generate 接线全部打通
- `unresolved` 已有明显下降
- 但：
  - `temporal` 仍差于 v1
  - `band` 只改善很小
  - `unresolved` 尚未达到 `20%` 下降目标

因此继续进入 Round 2。


## 5. Round 2

### 5.1 改动

Round 2 的核心改动是：

1. 调整 `global_min_visible_count / confidence_threshold / global_blend_strength`
2. 扩大 `global temporal` 对 `video_v2` 的参与范围
3. 修正 benchmark 默认值，避免上一轮脚本仍在使用旧参数

### 5.2 结果

真实运行目录：

- `runs/optimization3_step06_round2b_ab`

Round 2 `video_v2`：

- `temporal_fluctuation_mean = 0.907296750618487`
- `band_adjacent_background_stability = 0.33978221255584296`
- `unresolved_ratio_mean = 0.7475640773773193`
- `support_confidence_corr = 0.7957742558450926`
- `support_unresolved_corr = -0.9315312395744059`
- `confidence_unresolved_corr = -0.6299895556605817`

判断：

- `temporal` 首次逼近并优于 v1
- `band` 继续改善
- `unresolved` 继续下降
- 但仍未达成三条目标：
  - `temporal <= 0.9`
  - `band <= 0.27`
  - `unresolved` 相对基线下降 `20%`

因此继续进入 Round 3。


## 6. Round 3

### 6.1 改动

Round 3 做了两件更关键的事：

1. `unresolved_region`
   - 从硬二值判定改为 soft unresolved severity
2. `boundary-aware smoothing`
   - 对 `soft_band + low-confidence background` 做定向双向时序平滑

### 6.2 结果

真实运行目录：

- `runs/optimization3_step06_round3_ab`

Round 3 `video_v2`：

- `temporal_fluctuation_mean = 0.9044727500604124`
- `band_adjacent_background_stability = 0.3222938276169673`
- `unresolved_ratio_mean = 0.6934364438056946`
- `support_unresolved_corr = -0.9637669445115863`
- `confidence_unresolved_corr = -0.7716003827593162`

判断：

- `unresolved` 首次达到目标
  - 相对基线下降约 `20.86%`
- `temporal` 与 `band` 继续改善
- 但仍未跨过：
  - `temporal <= 0.9`
  - `band <= 0.27`

因此继续进入 Round 4。


## 7. Round 4

### 7.1 改动

Round 4 的策略不是再改 uncertainty，而是做“稳定性强化”：

1. 高置信区域更偏向 global temporal prefill
2. boundary smoothing 权重扩大到：
   - `soft_band`
   - `unresolved`
   - `1 - confidence`

### 7.2 结果

真实运行目录：

- `runs/optimization3_step06_round4_ab`

Round 4 `video_v2`：

- `temporal_fluctuation_mean = 0.90035316925876`
- `band_adjacent_background_stability = 0.3100167113739202`
- `unresolved_ratio_mean = 0.6934364438056946`

判断：

- `temporal` 已逼近目标，但仍略高于阈值
- `band` 继续改善，但仍明显高于 `0.27`
- `unresolved` 保持达标

因此继续进入最后一轮 Round 5。


## 8. Round 5

### 8.1 改动

Round 5 不再改算法，只做参数冻结前的最后一次高精度收敛：

- `bg_temporal_smooth_strength = 0.14`
- `bg_video_global_blend_strength = 0.95`

这一轮的目标是：

- 把 `temporal` 推过阈值
- 尽量再压 `band`

### 8.2 结果

真实运行目录：

- `runs/optimization3_step06_round5_ab`

Round 5 `video_v2`：

- `temporal_fluctuation_mean = 0.878150360924857`
- `band_adjacent_background_stability = 0.2985503441217066`
- `unresolved_ratio_mean = 0.6934364438056946`
- `support_confidence_corr = 0.7957742558450926`
- `support_unresolved_corr = -0.9637669445115863`
- `confidence_unresolved_corr = -0.7716003827593162`

相对基线的改善：

- `temporal_fluctuation_mean`
  - `0.9144 -> 0.8782`
  - 改善约 `3.96%`
- `band_adjacent_background_stability`
  - `0.3462 -> 0.2986`
  - 改善约 `13.75%`
- `unresolved_ratio_mean`
  - `0.8762 -> 0.6934`
  - 改善约 `20.86%`

判断：

- `temporal` 达标
- `unresolved` 达标
- `visible_support / confidence / unresolved` 的相关性满足“显著相关”
- 但 `band <= 0.27` 仍未达标

按步骤文档要求，这里已经达到 5 轮上限，因此停止继续调参。


## 9. 为什么最终冻结 Round 5

best-of-five 判断如下：

- Round 1：
  - 完成 schema 与 artifact 打通
  - 但质量提升不够
- Round 2：
  - `temporal` 首次优于 v1
  - 但 `unresolved` 仍未达标
- Round 3：
  - `unresolved` 首次达标
  - `band` 与 `temporal` 继续向前
- Round 4：
  - `band` 明显优于 Round 3
  - `temporal` 接近阈值但还差一点
- Round 5：
  - `temporal` 达标
  - `unresolved` 达标
  - `band` 达到 5 轮中最好结果

所以最终冻结 Round 5。


## 10. generate 兼容性验证

本步骤不是只停留在 preprocess。

我额外使用：

- `runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess`

跑过真实 generate smoke：

- `runs/optimization3_step06_generate_smoke`

输出：

- `outputs/replacement_smoke_none.mkv`

runtime stats 已确认：

- `background_mode = clean_plate_video_v2`
- `visible_support_available = true`
- `unresolved_region_available = true`
- `background_confidence_available = true`
- `background_source_provenance_available = true`
- `boundary_band_available = true`
- `background_keep_prior_available = true`

这说明 Step 06 的 richer background signal 已经被 generate 主链稳定消费，而不是只停留在 metadata 或 debug。


## 11. 最终冻结实现

当前冻结后的实现包括：

1. `clean_plate_video_v2`
2. `visible_support`
3. `background_confidence`
4. `soft unresolved_region`
5. `background_source_provenance`
6. boundary-aware bidirectional smoothing
7. 默认高精度参数：
   - `bg_temporal_smooth_strength = 0.14`
   - `bg_video_global_blend_strength = 0.95`


## 12. 对 Step 06 的最终判断

这一步现在可以认为“完成了”，但结论必须说准确：

- `clean_plate_video_v2` 已经显著优于 `clean_plate_video`
- 它已经足以成为当前高精度 replacement 路线的默认背景实现
- 但它还没有达到“边界带背景稳定性已经极致”的程度

换句话说：

- **背景 2.0 已经成立**
- **它已经明显减轻了长期遮挡区域和边界脏边问题**
- **但 `band_adjacent_background_stability <= 0.27` 这条超激进目标仍未完成**

因此，Step 06 的 remaining gap 应明确留给：

- Step 07 richer signal consumption
- 后续 boundary refinement v2 / uncertainty-aware compositing

而不是继续在这一步里做第 6 轮参数微调。
