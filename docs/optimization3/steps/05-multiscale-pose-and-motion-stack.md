# Step 05: 多尺度 Pose 与 Motion Stack

## 1. 步骤目标

本步骤的目标是把当前以全身 pose 为主的动作系统升级成真正的多尺度、多时序约束的 motion stack，使动作识别精度显著提升，并减少由动作误差导致的轮廓软化、肢体漂移和局部边缘错位。

交付形态应至少包括：

- 全身关键点轨迹
- limb-level refine 轨迹
- hand refine 轨迹
- 可见性与置信度
- 速度与加速度摘要
- occlusion-aware motion trace


## 2. 为什么要做这一步

### 2.1 轮廓问题有相当一部分来自动作系统

替换后边缘看起来模糊，不一定全是边界系统本身的问题。  
如果动作轨迹不够准，模型会在错误的位置长出四肢和衣物边缘，最终表现为：

- 手臂边缘漂
- 肩部边缘抖
- 腿部轮廓拖影
- 袖口和裙摆错位

### 2.2 当前系统缺少 limb 和 hand 的高精度局部分析

即使全身主骨架正确，局部细节不准仍会明显影响最终观感。  
最典型的是：

- 手部
- 手肘
- 膝盖
- 脚踝

### 2.3 需要从“点位结果”升级到“时序轨迹系统”

下一阶段动作系统的产出不应只是单帧关键点，而应是带有：

- 置信度
- 可见性
- 速度
- 缺失恢复状态

的时序轨迹。


## 3. 本步骤的交付结果

1. 多尺度 pose artifact schema
2. limb/hand ROI refine
3. 双向时序轨迹平滑
4. 运动不确定性输出
5. 动作 benchmark 与 gate


## 4. 设计原则

### 4.1 global pose 负责整体结构，local pose 负责高精度

- `global pose`：
  - 提供完整人体主结构
- `limb refine`：
  - 提高手臂、腿部等局部精度
- `hand refine`：
  - 提高手部关键点和局部运动精度

### 4.2 必须明确 occlusion 和不可见状态

当前系统容易把“不可见”误解成“检测失败”。  
下一阶段必须把：

- `occluded`
- `low_confidence`
- `interpolated`

三种状态分开表示。

### 4.3 轨迹优先于单帧最优

最终目标是让时序更稳，而不是让某一帧检测得看起来最漂亮。


## 5. 代码改动范围

- `wan/modules/animate/preprocess/pose2d.py`
- `wan/modules/animate/preprocess/signal_stabilization.py`
- 新增 limb / hand refine 模块
- `wan/utils/animate_contract.py`
- 相关 benchmark 脚本和 debug 导出


## 6. 实施方案

### 6.1 子步骤 A：扩展动作 artifact schema

建议新增：

- `src_pose_tracks.json`
- `src_limb_tracks.json`
- `src_hand_tracks.json`
- `src_pose_visibility.json`
- `src_pose_uncertainty.npz`

### 6.2 子步骤 B：局部 ROI refine

ROI 类型建议：

- upper-body
- lower-body
- left-hand
- right-hand

触发条件：

- 手部运动剧烈
- limb 关键点置信度低
- limb 速度异常突变

### 6.3 子步骤 C：双向轨迹优化

建议引入：

- 前向平滑
- 反向平滑
- 缺失点插值
- 速度/加速度约束

### 6.4 子步骤 D：运动不确定性建模

对以下场景显式增加 uncertainty：

- 高速模糊
- 遮挡
- limb 与 body 模型结果不一致


## 7. 目标指标

相对当前基线，至少达到：

- body jitter：降低 `25%`
- hand jitter：降低 `35%`
- limb continuity：提升 `20%`
- velocity spike rate：降低 `30%`

并要求高难动作片段中手部与 limb ROI refine 覆盖率 `>= 95%`。


## 8. 迭代规则

### Round 1

- 打通多尺度 artifact 输出
- limb/hand ROI 流程能运行

### Round 2

- 跑关键点、jitter、velocity 指标
- 优化轨迹平滑和插值策略

### Round 3

- 重点打磨高速手部和遮挡 limb 片段
- 冻结 motion stack v1

若 Round 3 后 hand jitter 和 body jitter 仍无明显下降，则不得继续把更复杂的动作信号接入 generate。


## 9. 验证方案

必须执行：

- pose benchmark
- hand benchmark
- 长时序 stress 测试
- limb-local crop review


## 10. 风险与回退

### 风险

- ROI refine 规则过多导致流程复杂
- 双向平滑可能过度钝化真实动作

### 回退

- 保留现有全身 pose 主链
- 局部 refine 可按模块关闭

