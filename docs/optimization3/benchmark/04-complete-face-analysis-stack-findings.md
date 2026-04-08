# Step 04 Findings: 完整 Face Analysis Stack

## 1. 结论

Step 04 已完成。

这一步的结论不是“脸部精度已经极致”，而是：

- face analysis stack 已经从单一 `src_face.mp4` motion 驱动，升级成了正式的结构化脸系统。
- preprocess 现在可以稳定输出：
  - tracking-aware face bbox
  - dense face landmarks
  - head pose
  - expression summary
  - face parsing
  - face alpha
  - face uncertainty
- 在真实 10 秒素材上，Round 1 就已经满足本步骤最关键的主 gate：
  - `face bbox jitter` 相比 Step 01 proxy baseline 下降超过 `30%`
- 新的 face artifact 契约、指标脚本与 generate 兼容性都已经接通。

但同样要明确：

- 当前 face stack 仍然是 **heuristic v1**，不是外部高精度 face landmark / face parsing / face matting 大模型版。
- 这一轮还没有基于人工标注关键帧跑出：
  - landmark NME
  - face boundary MAE
  - head pose 绝对误差
- 因此不能夸大结论为“脸部高精度问题已经彻底解决”。

准确判断应为：

- **工程交付成功**
- **主 gate 达标**
- **真实 10 秒 smoke 跑通**
- **为后续 richer face conditioning 和更高精度 face model 替换打下了接口基础**


## 2. 本步骤实际交付

### 2.1 新 face artifact

preprocess 现在会额外输出：

- `src_face_landmarks.json`
- `src_face_pose.json`
- `src_face_expression.json`
- `src_face_alpha.npz`
- `src_face_parsing.npz`
- `src_face_uncertainty.npz`

这些 artifact 也已经写入 metadata，并通过 contract 检查。

### 2.2 face tracking-aware crop 主链

`src_face.mp4` 不再只依赖简单 bbox 平滑，而是通过 tracking-aware face bbox 重新生成：

- detection / landmark bbox
- track association
- center/scale clamp
- hold / predict fallback
- difficulty score
- high-difficulty frame 的 crop expand

### 2.3 新的结构化 face 指标

`compute_face_precision_metrics.py` 现在可统计：

- bbox jitter
- landmark confidence / motion proxy
- head pose jitter
- expression step
- face uncertainty 与 transition 的聚焦程度

### 2.4 generate 侧兼容

当前 `generate` 还没有把这些新 face signal 接入 face adapter 主路径，但已经完成：

- artifact 读取契约兼容
- runtime stats availability 标记
- smoke 级 generate 兼容性验证

这符合步骤文档要求：

- 若 landmark / alpha 真实收益还没有被充分证明，不应贸然把新 face signal 强行塞进 generate 主路径。


## 3. Round 1 实现与结果

## 3.1 改动

Round 1 实际完成了以下内容：

1. 新增 `face_analysis.py`
   - tracking-aware bbox
   - dense landmarks export
   - head pose estimate
   - expression summary
   - face parsing
   - face alpha
   - face uncertainty

2. 接入 preprocess 主流程
   - 在 boundary fusion 之后运行 face analysis
   - 重新生成 `src_face.mp4`
   - 输出结构化 JSON/NPZ artifact
   - 记录 runtime / metadata / QA

3. 指标与回归
   - 扩展 `compute_face_precision_metrics.py`
   - 新增 `check_face_analysis_stack.py`

## 3.2 遇到的问题

Round 1 第一次真实运行时，暴露了一个清晰的接口 bug：

- `face_analysis["face_images"]` 是 `numpy` 数组
- 但主流程在接收时又被错误拆回了 Python `list`
- 写盘阶段因此触发：
  - `ValueError: src_face must be a numpy array.`

该问题已在同轮修复，并用 `Round 1 v2` 重新跑完整真实 preprocess。

## 3.3 真实 smoke 结果

真实运行目录：

- `runs/optimization3_step04_round1_preprocess_v2`

真实输入：

- `video_path = /home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4`
- `reference_path = /home/user1/wan2.2/test1/Gemini_Generated_Image_2.png`
- `ckpt_path = /home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint`

contract 检查：

- `PASS`

关键 proxy 指标：

- `center_jitter_mean = 1.7132`
- `center_jitter_max = 11.1803`
- `width_jitter_mean = 0.2449`
- `height_jitter_mean = 3.5714`
- `valid_face_points_mean = 68.82`
- `valid_face_points_min = 68`
- `difficulty_mean = 0.0136`
- `rerun_ratio = 0.0`
- `landmark_confidence_mean = 0.9742`
- `landmark_norm_step_mean = 0.00182`
- `head_pose_yaw_jitter_mean = 0.4012`
- `head_pose_pitch_jitter_mean = 2.2916`
- `head_pose_roll_jitter_mean = 1.3275`
- `expression_step_mean = 0.0266`
- `face_uncertainty_transition_to_interior_ratio = 10.18`

artifact availability：

- `face_landmarks_available = true`
- `head_pose_available = true`
- `expression_available = true`
- `face_alpha_available = true`
- `face_parsing_available = true`
- `face_uncertainty_available = true`


## 4. 与 Step 01 baseline 的对比

baseline 来源：

- `runs/optimization3_validation_core_20260407_v3/metrics/face_precision_metrics.json`

Step 01 baseline：

- `center_jitter_mean = 2.7700`
- `center_jitter_max = 9.5131`
- `width_jitter_mean = 0.6531`
- `height_jitter_mean = 1.8776`
- `valid_face_points_mean = 67.82`
- `valid_face_points_min = 59`

Step 04 Round 1 v2：

- `center_jitter_mean = 1.7132`
- `center_jitter_max = 11.1803`
- `width_jitter_mean = 0.2449`
- `height_jitter_mean = 3.5714`
- `valid_face_points_mean = 68.82`
- `valid_face_points_min = 68`

变化：

- `center_jitter_mean`
  - 从 `2.7700` 降到 `1.7132`
  - 改善约 `38.2%`
  - **达到文档中“降低 30%”的主目标**

- `width_jitter_mean`
  - 从 `0.6531` 降到 `0.2449`
  - 改善约 `62.5%`

- `valid_face_points_mean`
  - 从 `67.82` 提升到 `68.82`
  - 略有改善

- `valid_face_points_min`
  - 从 `59` 提升到 `68`
  - 明显改善

- `center_jitter_max`
  - 从 `9.5131` 升到 `11.1803`
  - 出现回退

- `height_jitter_mean`
  - 从 `1.8776` 升到 `3.5714`
  - 出现回退

这说明：

- face tracking 更稳定了，尤其中心和宽度明显更稳
- 但在纵向尺度上仍然存在保守 heuristic 带来的波动
- 当前 face stack 更偏向“整体跟踪稳定 + 结构化输出完整”，不是“所有 bbox 维度都全面最优”


## 5. 为什么没有继续做 Round 2 / Round 3

步骤文档规定最多 3 轮，但并没有要求必须凑满 3 轮。

这一轮没有继续往下做 Round 2 / Round 3，原因是：

1. **主 gate 已达标**
   - `face bbox jitter` 已经超过目标改善幅度

2. **当前缺的是标注集，而不是更多 heuristic 微调**
   - `landmark NME`
   - `face boundary MAE`
   - `head pose abs error`
   这些真正高精度指标当前都还没有人工标注集支撑

3. **继续只调 heuristic tracking，容易把 improvement 变成 overfit**
   - 例如进一步压中心 jitter，可能会换来更差的侧脸/快速运动响应

因此本步骤的合理冻结策略是：

- 在 Round 1 v2 达到主 gate 后冻结 face stack v1
- 不在没有标注集的情况下，继续为了漂亮 proxy 过度调参
- 把更进一步的 face 精度提升，留给后续完整 face model / richer signal consumption 阶段


## 6. generate 兼容性验证

为了避免 face artifact 只在 preprocess 里“能写不能用”，本步骤额外做了 generate smoke 兼容性检查：

- `runs/optimization3_step04_generate_smoke`

目的不是验证画质，而是验证：

- 新 metadata / artifact 不会破坏当前 generate 主链
- runtime stats 能识别 face artifact availability

本步骤的 generate 兼容性定位是：

- **兼容已验证**
- **主路径未消费**

这与步骤文档的风控要求一致。

此外，还额外补跑了：

- `runs/optimization3_step04_core_suite`

结果：

- `summary.json` 存在
- `all_cases_passed = true`

说明新增的 face stack 已兼容 `optimization3` 现有 benchmark / gate 体系，没有把已有 smoke 主链打坏。


## 7. 质量判断

### 7.1 成功的部分

1. 完整 face artifact schema 已落地
2. preprocess 主链已真正输出这些 artifact
3. contract 与 metric 脚本都已接通
4. 真实 10 秒 preprocess smoke 通过
5. `center_jitter_mean` 达到超过 `30%` 的改善
6. `valid_face_points_min` 明显提升，说明低质量帧更稳

### 7.2 仍然不足的部分

1. 仍然没有人工标注 landmark / alpha 指标
2. `center_jitter_max` 和 `height_jitter_mean` 出现回退
3. 当前 `head pose / expression / parsing / alpha / uncertainty` 仍然是 heuristic v1
4. 这些新信号还没有进入 generate 的主 conditioning 路径


## 8. 最终冻结结论

Step 04 应冻结为：

- `face_analysis_mode = heuristic`
- 保留当前 tracking-aware face bbox
- 保留当前结构化 face artifact 输出
- 保留 face precision proxy 指标脚本
- 暂不把新 face signal 强制接入 generate 主路径

下一步真正值得做的，不是继续在这一步里微调 tracking 参数，而是：

1. 用标注关键帧建立真实 `landmark NME / face alpha error` 基准
2. 在后续 rich-signal generate 阶段，设计如何消费：
   - `face_landmarks`
   - `face_pose`
   - `face_expression`
   - `face_alpha`
   - `face_uncertainty`
3. 如需继续提升 face 分析精度，再考虑替换 heuristic v1 为专用高精度 face model
