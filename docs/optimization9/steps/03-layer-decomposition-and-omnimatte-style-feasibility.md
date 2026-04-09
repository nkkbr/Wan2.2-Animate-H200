# Step 03: Layer Decomposition And Omnimatte-Style Feasibility

## 1. 为什么要做

如果 bridge model 仍然无法从根本上解决 foreground/background/occlusion 耦合问题，那么下一条最值得验证的高风险路线，就是 layer decomposition。

这条路线的核心价值在于：

- foreground
- occluder
- shadow/effect
- background

可以成为显式层，而不是被压在单一 replacement 表示里。


## 2. 目标

验证 layer decomposition / omnimatte-style 原型是否能够：

- 在遮挡边界上表现出明显不同的错误模式
- 提高复杂边界和背景恢复的一致性


## 3. 要改哪些模块

建议新增：

- `wan/utils/layer_decomposition_proto.py`
- `scripts/eval/check_layer_decomposition_proto.py`
- `scripts/eval/run_optimization9_layer_benchmark.py`
- `scripts/eval/evaluate_optimization9_layer_benchmark.py`


## 4. 具体如何做

### 4.1 定义最小层结构

初期不要做全复杂度层系统，先做：

- foreground layer
- occlusion / unresolved layer
- background layer

### 4.2 compositing 对齐

必须明确：

- 每层语义
- 每层可见性
- 最终 composite 规则

### 4.3 只做小样本 feasibility

先挑最难的：

- 遮挡严重
- 发丝复杂
- 背景恢复困难

的小样本 case。


## 5. 如何检验

### 5.1 reviewed benchmark

重点看：

- occlusion boundary quality
- hair / cloth boundary quality
- unresolved region handling

### 5.2 real-video smoke

重点看：

- background stability
- seam
- occlusion consistency


## 6. 强指标要求

本步骤更关注“质变信号”，建议 gate：

- 至少在遮挡相关指标上显著优于当前主线
- 或明显减少一种旧主线反复出现的错误类型


## 7. 实现-测评-再改闭环

### Round 1

- 打通最小三层 proto

### Round 2

- 调整层语义与 composite 规则
- 跑最难 case AB

### Round 3

- 冻结 best-of-three
- 决定是否进入 `upgrade_candidate`


## 8. 成功标准

- layer decomposition 至少在某类遮挡或背景恢复问题上表现出明确新价值
