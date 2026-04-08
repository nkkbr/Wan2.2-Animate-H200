# Step 02 Findings: H200 Multistage Multires Preprocess

## 1. 结果结论

Step 02 已完成。

结论不是“全部目标都达成”，而是：

- 多阶段多分辨率 preprocess 主链已经正式接入并稳定可运行。
- `h200_extreme` 现在是一个真实可用的 preprocess profile，而不是文档占位。
- 真实 10 秒 smoke 在 `h200_extreme` 下已经达到 `3/3` 稳定通过。
- `person_roi` 与 `face_roi` 的 proposal coverage 都达到 `100%`。
- 但本步骤设定的强目标并未全部达成：
  - `face bbox jitter` 没有比 Step 01 基线降低 `25%`
  - `pose jitter` 也没有比 Step 01 基线降低 `20%`

因此，Step 02 的准确判断是：

- **工程交付成功**
- **稳定性目标成功**
- **高精度收益方向正确，但收益幅度不足**

按照步骤文档中的规则，在 3 轮“实现 -> 测评 -> 再改”之后，当前结果应被冻结为正式 findings，并把后续更大幅度的精度提升留给 Step 03 及之后的 richer signal 路线。


## 2. 本步骤交付了什么

本步骤最终交付了以下能力：

1. 三层 preprocess 结构
   - `global_analysis`
   - `person_roi_analysis`
   - `face_roi_analysis`

2. H200 extreme preprocess profile
   - `preprocess_runtime_profile = h200_extreme`
   - 分离 `analysis_resolution` 与 `export_resolution`
   - 单独定义 `person ROI` / `face ROI` 的 stage resolution 与 target long side

3. ROI proposal / rerun / fusion 主链
   - 新增模块：
     - `wan/modules/animate/preprocess/multistage_preprocess.py`

4. 多阶段 runtime 与 metadata 记录
   - `preprocess_runtime_stats.json` 现在可记录：
     - per-stage shape
     - per-stage runtime
     - per-stage peak memory
     - ROI proposal coverage
     - fusion stats

5. benchmark 脚本升级
   - `scripts/eval/run_preprocess_profile_benchmark.py`
   - `scripts/eval/run_preprocess_stability_smoke.py`
   - `scripts/eval/run_optimization3_validation_suite.py`
   - `scripts/eval/compute_background_precision_metrics.py`


## 3. 执行过程

### 3.1 Round 1

目标：

- 打通 `global -> person ROI -> face ROI` 三层数据流
- 确保 `h200_extreme` 可以真实跑通

执行：

- 新增多阶段 ROI helper
- 在 preprocess 主流程中插入：
  - person ROI proposal
  - face ROI proposal
  - ROI rerun
  - pose meta fusion
  - runtime / metadata 输出

结果：

- 跑通
- `ROI coverage = 100%`
- 但 face/pose proxy 提升不足

关键指标：

- `h200_safe`
  - `face_center_jitter_mean = 3.7851`
  - `pose_body_conf_delta_mean = 0.0092837`
- `h200_extreme`
  - `face_center_jitter_mean = 3.2728`
  - `pose_body_conf_delta_mean = 0.0091391`
  - `person_roi_coverage_ratio = 1.0`
  - `face_roi_coverage_ratio = 1.0`

问题：

- `person_update_frames = 0`
- 说明 person ROI 路径虽然被执行，但并没有真正转化成有效的 body/hand 更新

### 3.2 Round 2

判断：

- 发现 ROI crop 只会“缩小到 target long side”，不会“放大到 target long side”
- 这意味着 face/person ROI 还没有真正吃到高分辨率重跑收益

修正：

- 将 ROI crop resize 改成真正对齐到 `target_long_side`
  - 放大时用 `INTER_CUBIC`
  - 缩小时用 `INTER_AREA`
- 修正 runtime summary 中 `preprocess_runtime_profile` 被错误记录为 `sam_runtime_profile` 的问题

结果：

- `h200_extreme` 的 ROI crop 终于真正达到高分辨率
- 但指标仍未达目标

典型指标（core smoke preprocess）：

- `face_center_jitter_mean = 3.2916`
- `pose_body_conf_delta_mean = 0.0093815`

说明：

- 结构打通了
- 但多阶段裸输出并不足以自然变成更低 jitter

### 3.3 Round 3

判断：

- 仅靠 ROI rerun 不够
- 需要把 `h200_extreme` 定义成“多阶段 + 更强稳定化约束”的完整 preset

修正：

- 收紧 `h200_extreme` 默认稳定化参数：
  - 更强 `face_bbox_smooth_strength`
  - 更小 `face_bbox_max_scale_change`
  - 更小 `face_bbox_max_center_shift`
  - 更长 `face_bbox_hold_frames`
  - 更强 body/hand/face pose smoothing
  - 更积极的 `multistage_pose_extra_smooth`
  - 更积极的 `multistage_face_bbox_extra_smooth`
  - 更积极的 `person_roi_fuse_weight`
  - 更宽松的 `person_roi_conf_margin`
  - 更保守的 `face_roi_fuse_weight`

结果：

- 稳定性通过
- 指标进一步改善
- 但仍未达到强目标


## 4. 最终验证

### 4.1 真实素材与 checkpoint

统一使用：

- `video_path = /home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- `reference_path = /home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- `ckpt_path = /home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`

### 4.2 真实 benchmark 产物

关键 runs：

- Round 1 matrix:
  - `runs/optimization3_step02_attempt1_profile_matrix`
- Round 2 core suite:
  - `runs/optimization3_step02_attempt2_core_suite`
- Round 3 matrix:
  - `runs/optimization3_step02_attempt3_profile_matrix`
- Round 3 stability:
  - `runs/optimization3_step02_attempt3_stability`

### 4.3 最终冻结指标

相对 Step 01 proxy baseline：

- Step 01 baseline
  - `face_center_jitter_mean = 2.7700`
  - `pose_body_conf_delta_mean = 0.0095136`

- Step 02 final (`h200_extreme`, Round 3 profile matrix)
  - `face_center_jitter_mean = 3.1124`
  - `pose_body_conf_delta_mean = 0.0088402`
  - `person_roi_coverage_ratio = 1.0`
  - `face_roi_coverage_ratio = 1.0`
  - `preprocess_total_seconds = 106.81`

相对 Step 01 baseline 的变化：

- `face_center_jitter_mean`
  - 目标：至少下降 `25%`
  - 实际：**上升约 12.4%**
  - 结论：**未达标**

- `pose_body_conf_delta_mean`
  - 目标：至少下降 `20%`
  - 实际：下降约 `7.1%`
  - 结论：**未达标**

- `ROI coverage`
  - 目标：`>= 95%`
  - 实际：`100%`
  - 结论：**达标**

- `稳定性`
  - 目标：真实 10 秒 smoke `3/3` 通过
  - 实际：`3/3` 通过
  - 结论：**达标**

- `总时间`
  - 目标：不超过当前结构化 baseline 的 `3x`
  - Round 3 matrix 中：
    - `h200_safe = 70.81s`
    - `h200_extreme = 106.81s`
  - 比值约 `1.51x`
  - 结论：**达标**


## 5. 质量判断

### 5.1 成功的部分

1. 多阶段 preprocess 主链已经完成工程化接入
2. `h200_extreme` profile 已真实跑通
3. ROI coverage 达到 `100%`
4. `3/3` 稳定性 smoke 已通过
5. 运行时间虽然明显上升，但仍控制在可接受范围内

### 5.2 没有达到预期的部分

1. `face bbox jitter` 没有下降到 Step 01 baseline 以下
2. `pose jitter` 只得到轻度改善
3. `person_roi` 虽然 proposal / rerun 正常，但 `fusion.stats.person_update_frames` 仍然为 `0`

这说明当前 Step 02 最大的问题不是“多阶段跑不通”，而是：

- ROI 结果还没有充分转化成真正更优的最终控制信号
- 当前 face/pose proxy 的主要收益，更多来自更强平滑，而不是更强识别


## 6. 解释与判断

为什么会这样：

1. `person_roi` 使用的仍然是同一套 `Pose2d` 主干，只是换成更高分辨率 crop 重跑  
   这会提高局部可见性，但不一定直接提高 proxy confidence 或降低 bbox jitter。

2. 当前 face 指标是 `face bbox jitter proxy`，它对稳定化参数非常敏感  
   换句话说，这一步虽然确实提升了“ROI 层的可分析性”，但还不足以自然压低最终 bbox jitter 到目标值。

3. 当前缺的不是更多分辨率，而是更强的专门 face stack 和 richer boundary / uncertainty 信号  
   这正是 Step 03 之后要解决的方向。


## 7. 是否可以进入下一步

可以。

理由：

- Step 02 的工程目标已经完成
- 强收益目标虽然未达成，但已经通过 3 轮闭环确认：仅靠本步骤的 multistage/multires 设计，收益幅度有限
- 继续在本步骤内反复微调 ROI 参数，性价比已经明显下降

因此应冻结 Step 02，并进入下一步：

- Step 03 `boundary + alpha + uncertainty` 系统


## 8. 后续建议

1. 不要再把 Step 02 当作边界精度的主攻点  
   它更适合作为高精度 preprocess 的基础设施层。

2. 对 face 精度的真正大幅提升，应该放到完整 face stack 去做  
   例如：
   - face landmarks
   - head pose
   - face parsing
   - face alpha

3. 对动作与轮廓的真正大幅提升，应该依赖：
   - richer boundary signals
   - uncertainty map
   - 多源融合而不是单一 pose rerun

4. `h200_extreme` 当前可以保留为实验 profile  
   但不应宣称它已经在 proxy 指标上“显著超过 Step 01 基线”。
