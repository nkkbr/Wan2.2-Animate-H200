# Step 05 Findings: 多尺度 Pose 与 Motion Stack

## 1. 结论

Step 05 已完成，但结论需要说准：

- 工程交付已经完成。
- 多尺度 pose/motion stack 已经正式接入 preprocess 主链。
- 真实 10 秒素材上，`body_jitter`、`hand_jitter`、`limb continuity`、`ROI coverage` 都取得了显著改善。
- 但本步骤设定的 6 条 gate 中，最难的一条 `velocity_spike_rate` 仍未在同一轮里达到 `30%` 以上改善。

因此，本步骤的最终判断是：

- **工程目标达成**
- **5/6 条 proxy gate 达成**
- **motion stack v1 已冻结为可用版本**
- **remaining gap 明确保留到后续步骤继续处理**

不能夸大为“动作识别已经极致”，但也不能低估这一步的价值。  
到 Step 05 结束时，系统已经从“只有全身 pose 和简单平滑”升级成了真正的：

- 全身关键点轨迹
- limb-level refine
- hand-level refine
- visibility states
- pose uncertainty
- occlusion-aware motion trace

并且这些 artifact 已经真正作用到了最终 `src_pose.mp4`，不是只停留在额外诊断文件里。


## 2. 本步骤实际交付

### 2.1 新增模块与主链改动

新增模块：

- `wan/modules/animate/preprocess/pose_motion_analysis.py`

主链接入：

- `wan/modules/animate/preprocess/process_pipepline.py`
- `wan/modules/animate/preprocess/preprocess_data.py`
- `wan/utils/animate_contract.py`
- `wan/animate.py`

指标与回归：

- `scripts/eval/compute_pose_precision_metrics.py`
- `scripts/eval/check_pose_motion_stack.py`
- `scripts/eval/run_optimization3_validation_suite.py`

### 2.2 新 artifact

preprocess 现在会额外输出：

- `src_pose_tracks.json`
- `src_limb_tracks.json`
- `src_hand_tracks.json`
- `src_pose_visibility.json`
- `src_pose_uncertainty.npz`

### 2.3 新 motion stack 能力

这一轮真正新增的核心能力包括：

1. `global body` 双向时序平滑
2. `face keypoints` 双向时序平滑
3. `hand keypoints` 双向时序平滑
4. limb ROI local refine
5. hand ROI local refine
6. spike-aware velocity suppression
7. `visible / low_confidence / occluded / interpolated` 四态 visibility
8. pose uncertainty heatmap

### 2.4 关键接线位置

这一步最重要的不是“多导出几个 JSON”，而是：

- motion stack 运行在 boundary fusion 之后
- `optimized_pose_metas` 会真正回灌：
  - `src_pose.mp4`
  - `src_face.mp4` 的 face bbox 重新稳定化
  - downstream replacement preprocess bundle

这保证了 Step 05 对 generate 是有真实影响的，不是旁路分析。


## 3. Round 1

### 3.1 目标

Round 1 的目标是：

- 打通多尺度 motion artifact 输出
- 让 `v1` 真正改写 `tpl_pose_metas`
- 在真实 10 秒 replacement preprocess 上跑通

### 3.2 结果

真实运行目录：

- `runs/optimization3_step05_round1_baseline`
- `runs/optimization3_step05_round1_v1`

关键结果，相对 baseline：

- `body_jitter` 改善约 `35.96%`
- `hand_jitter` 改善约 `31.30%`
- `limb_continuity` 改善约 `31.49%`
- `velocity_spike_rate` 改善约 `71.14%`
- `limb_roi_coverage_ratio = 1.0`
- `hand_roi_coverage_ratio = 1.0`

判断：

- Round 1 已证明 motion stack 方向是正确的
- 但 `hand_jitter` 还没有跨过步骤文档要求的 `35%` 目标

因此继续进入 Round 2。


## 4. Round 2

### 4.1 改动

Round 2 做了两类关键修正：

1. **指标修正**
   - `velocity_spike_rate` 不再使用天然固定比例的分位数计数逻辑
   - `limb_continuity_score` 不再只看“是否 visible”，而是纳入：
     - visibility
     - interpolation
     - spike
     - jitter
     - ROI coverage

2. **motion stack 强化**
   - 更强的 body/hand bidirectional smoothing
   - 更强的 limb/hand local refine
   - 更保守的 velocity suppression

Round 2 使用的最终对照参数为：

- `pose_motion_body_bidirectional_strength = 0.86`
- `pose_motion_hand_bidirectional_strength = 0.82`
- `pose_motion_face_bidirectional_strength = 0.88`
- `pose_motion_local_refine_strength = 0.82`
- `pose_motion_limb_roi_expand_ratio = 1.32`
- `pose_motion_hand_roi_expand_ratio = 1.75`
- `pose_motion_velocity_spike_quantile = 0.88`

这些也已经被提升为当前默认值。

### 4.2 真实结果

真实运行目录：

- `runs/optimization3_step05_round2_baseline`
- `runs/optimization3_step05_round2_v1`

基线：

- `body_jitter_mean = 0.0002764`
- `hand_jitter_mean = 0.0745494`
- `limb_continuity_score = 0.6372785`
- `velocity_spike_rate = 0.0648387`
- `limb_roi_coverage_ratio = 0.0`
- `hand_roi_coverage_ratio = 0.0`

Round 2 v1：

- `body_jitter_mean = 0.00016995`
- `hand_jitter_mean = 0.04254943`
- `limb_continuity_score = 0.79757854`
- `velocity_spike_rate = 0.04838710`
- `limb_roi_coverage_ratio = 1.0`
- `hand_roi_coverage_ratio = 1.0`

改善幅度：

- `body_jitter` 改善约 `38.51%`
- `hand_jitter` 改善约 `42.92%`
- `limb_continuity` 改善约 `25.15%`
- `velocity_spike_rate` 改善约 `25.37%`

Round 2 gate 结果：

- `body_jitter_pass = true`
- `hand_jitter_pass = true`
- `limb_continuity_pass = true`
- `velocity_spike_pass = false`
- `limb_roi_pass = true`
- `hand_roi_pass = true`

判断：

- Round 2 是目前综合最强的一轮
- 已经实现 `5/6` 条 gate 通过


## 5. Round 3

### 5.1 目标

Round 3 的目标很明确：

- 在不明显钝化真实动作的前提下
- 尝试把 `velocity_spike_rate` 再往下压

### 5.2 改动

Round 3 做了最后一轮保守强化：

- 两次 spike suppression pass
- 略强的 hand/body smoothing
- 略大的 limb/hand ROI
- 更激进的 spike quantile

### 5.3 结果

真实运行目录：

- `runs/optimization3_step05_round3_v1`

Round 3 v1：

- `body_jitter_mean = 0.00015856`
- `hand_jitter_mean = 0.04159475`
- `limb_continuity_score = 0.83820860`
- `velocity_spike_rate = 0.06354839`
- `limb_roi_coverage_ratio = 1.0`
- `hand_roi_coverage_ratio = 1.0`

改善幅度，相对 baseline：

- `body_jitter` 改善约 `42.63%`
- `hand_jitter` 改善约 `44.21%`
- `limb_continuity` 改善约 `31.53%`
- `velocity_spike_rate` 仅改善约 `1.99%`

判断：

- Round 3 继续压低了 jitter
- 但 `velocity_spike_rate` 明显恶化
- 说明这一步再继续推进，已经开始进入“指标互相拉扯”的区域

因此按步骤文档要求，**Round 3 结束后冻结，不再继续第四轮。**


## 6. 为什么最终冻结 Round 2

best-of-three 判断如下：

- Round 1：
  - spike 最好
  - 但 `hand_jitter` 不达标
- Round 2：
  - `5/6` 条 gate 通过
  - 综合表现最好
- Round 3：
  - jitter / continuity 更好
  - 但 spike 明显回退

所以最终冻结选择：

- **Round 2**

当前默认参数也已切到 Round 2。


## 7. Generate Smoke 结果

为确认新的 `src_pose` 和 motion artifact 不会破坏 generate，我使用：

- preprocess bundle：
  - `runs/optimization3_step05_round2_v1/preprocess`
- generate smoke：
  - `runs/optimization3_step05_generate_smoke`

命令配置：

- `frame_num = 45`
- `refert_num = 9`
- `sample_steps = 4`
- `boundary_refine_mode = none`
- `output_format = ffv1`

generate smoke 已完整通过。

输出：

- `runs/optimization3_step05_generate_smoke/outputs/replacement_smoke_none.mkv`

runtime stats：

- `total_generate_sec = 41.81`
- `pose_tracks_available = true`
- `limb_tracks_available = true`
- `hand_tracks_available = true`
- `pose_visibility_available = true`
- `pose_uncertainty_available = true`

replacement metrics：

- `seam_score.mean = 1.6994`
- `background_fluctuation.mean = 1.0461`
- `mask_area.mean = 0.1260`

这说明：

- Step 05 产物没有破坏 replacement 生成链
- generate 已经能稳定看到新的 motion artifact availability


## 8. 本步骤的最终判断

准确的结论应该是：

- Step 05 **不是完全满分通过**
- 但它已经把动作系统推进到了一个新的台阶

可以确认达成的：

1. 多尺度 motion artifact 正式交付
2. limb / hand ROI refine 正式交付
3. visibility / uncertainty 正式交付
4. `src_pose.mp4` 已被 motion stack 真正改善
5. `body_jitter` 大幅下降
6. `hand_jitter` 大幅下降
7. `limb continuity` 明显提升
8. generate smoke 兼容通过

仍未完全解决的：

1. `velocity_spike_rate` 在同一轮里未达到 `30%` 改善 gate
2. 当前仍是 heuristic motion stack v1，不是更高精度的外部局部姿态模型
3. 还没有人工标注集上的：
   - hand PCK
   - limb-local NME
   - long-tail occlusion 精度

所以 Step 05 的最准确定位是：

- **motion stack v1 成功建立并冻结**
- **显著提升了动作轨迹稳定性**
- **但还没有把动作识别问题彻底做到“极致”**


## 9. 对后续步骤的意义

Step 05 的价值主要体现在三个方面：

1. 为后续 richer generate conditioning 留出了正式接口
   - 现在可以安全地考虑把部分 motion uncertainty / visibility 用进 generate

2. 为 face / boundary / background 系统提供了更可靠的动作底座
   - 肢体漂移减少后，边界系统的收益更容易显现

3. 明确了后续 motion 精度真正的剩余问题
   - 不再是“有没有 limb/hand refine”
   - 而是“如何进一步减少高速片段里的 spike 和误跟踪”
