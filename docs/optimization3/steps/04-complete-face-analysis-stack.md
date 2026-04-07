# Step 04: 完整 Face Analysis Stack

## 1. 步骤目标

本步骤的目标是把当前“face crop + motion”体系升级成完整的 face analysis stack，使脸部控制从粗粒度驱动提升到高精度几何与边界约束。

交付形态应至少包含：

- tracking-aware face bbox
- dense face landmarks
- head pose
- expression summary
- face parsing
- face alpha
- face uncertainty


## 2. 为什么要做这一步

### 2.1 脸是 replacement 中视觉敏感度最高的区域

只要脸的几何和边界稍有漂移，用户会立刻感知为：

- 人不像
- 边缘虚
- 嘴部奇怪
- 发际线/下颌线不稳

### 2.2 当前 face 系统仍然偏“驱动”，不够“约束”

现在的 `src_face.mp4` 更偏 motion conditioning。  
下一阶段需要补齐：

- geometry
- parsing
- alpha
- uncertainty


## 3. 本步骤的交付结果

1. 结构化 face outputs
2. tracking-aware face pipeline
3. face geometry + parsing + alpha
4. face benchmark 与 gate


## 4. 设计原则

### 4.1 脸系统拆成三部分

- geometry: landmarks, head pose
- semantic: face parsing
- optical: face alpha, occlusion, uncertainty

### 4.2 跟踪优先于逐帧孤立检测

脸 bbox、landmarks、head pose 必须具备时序跟踪一致性，不能再只依赖逐帧结果。

### 4.3 高难帧允许单独 rerun

大转头、遮挡、张口、低头等片段，应允许 face ROI 单独以更高分辨率重跑。


## 5. 代码改动范围

- `wan/modules/animate/preprocess/signal_stabilization.py`
- `wan/modules/animate/preprocess/reference_normalization.py`
- 新增 `face_analysis.py` 或同类模块
- `wan/utils/animate_contract.py`
- generate face condition 编码逻辑


## 6. 实施方案

### 6.1 子步骤 A：统一 face artifact schema

新增建议 artifact：

- `src_face_landmarks.json`
- `src_face_pose.json`
- `src_face_expression.json`
- `src_face_alpha.npz`
- `src_face_parsing.npz`

### 6.2 子步骤 B：tracking-aware face bbox

将 face bbox 由简单平滑升级为：

- 检测
- track association
- 缺失恢复
- 遮挡标记

### 6.3 子步骤 C：dense landmark / pose / parsing / alpha

对 face ROI 输出：

- 多点 landmark
- yaw/pitch/roll
- face semantic regions
- soft alpha

### 6.4 子步骤 D：高难帧 rerun

触发条件：

- landmark confidence 低
- pose 变化突变
- 遮挡高
- alpha boundary 不稳定


## 7. 目标指标

相对当前基线，至少达到：

- face bbox jitter：降低 `30%`
- landmark NME：降低 `25%`
- head pose jitter：降低 `20%`
- face boundary MAE：降低 `20%`


## 8. 迭代规则

### Round 1

- 打通结构化 face outputs
- 跟踪与 metadata 可用

### Round 2

- 跑 landmark / pose / alpha 指标
- 优化 tracking 与 rerun 触发

### Round 3

- 修侧脸、遮挡、快速运动 case
- 冻结 face stack v1

若 Round 3 后 landmark NME 与 face boundary MAE 仍未明显改善，则不得把新 face 信号接入 generate 主路径。


## 9. 验证方案

必须执行：

- face benchmark 关键帧评估
- landmark NME / pose jitter 统计
- face alpha error
- 高难脸部 crop review


## 10. 风险与回退

### 风险

- face stack 过于复杂，调试成本高
- 跟踪失败时会放大错误

### 回退

- 保留当前 `src_face.mp4` 作为保底条件
- 支持新旧 face pipeline 并行对比

