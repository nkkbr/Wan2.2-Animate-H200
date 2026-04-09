# Step 05: Renderable Foreground And 3D-Like Feasibility

## 1. 为什么要做

这是第三档里跨度最大的一条线。

只有当前面几条仍然无法在：

- 遮挡
- 复杂动作
- 轮廓稳定性

上给出新价值时，才值得考虑更重的 renderable foreground / 3D-like 方案。


## 2. 目标

验证最小 renderable foreground / 3D-like 原型是否在以下方面有潜在优势：

- 轮廓稳定
- 遮挡一致
- 动作一致
- foreground/background 交互更明确


## 3. 要改哪些模块

建议新增：

- `wan/utils/renderable_foreground_proto.py`
- `scripts/eval/check_renderable_foreground_proto.py`
- `scripts/eval/run_optimization9_renderable_benchmark.py`
- `scripts/eval/evaluate_optimization9_renderable_benchmark.py`


## 4. 具体如何做

### 4.1 只做最小原型

初期不要做完整 3D 系统，而应只验证：

- 是否能形成稳定 renderable foreground representation
- 是否能在少量 case 上带来新型优势

### 4.2 评测重点

更关注：

- 轮廓一致性
- occlusion handling
- motion consistency

而不是只看单帧边缘锐度。


## 5. 如何检验

### 5.1 reviewed benchmark

只作为辅助手段，重点看某些边界类型是否出现明显不同错误模式。

### 5.2 real-video smoke

重点是：

- temporal stability
- silhouette consistency
- occlusion behavior


## 6. 强指标要求

本步骤的成功标准不是全面超线，而是：

- 至少在时间一致性或遮挡边界上出现明显不同于当前主线的正向信号


## 7. 实现-测评-再改闭环

### Round 1

- 做最小 representation proto

### Round 2

- 调整 rendering / compositing 细节
- 跑难 case smoke

### Round 3

- 冻结 best-of-three
- 判断是否值得长期投入


## 8. 成功标准

- renderable / 3D-like 路线至少证明自己在某一类问题上不是“更复杂的同样失败”
