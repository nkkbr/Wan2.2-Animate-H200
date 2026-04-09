# Step 04: RGBA / Transparent Foreground Feasibility

## 1. 为什么要做

如果问题的根源之一是：

- alpha 始终只是辅助信号
- foreground 从来不是主输出对象

那么 RGBA / transparent foreground 路线值得被单独验证。


## 2. 目标

验证最小 RGBA / transparent foreground 路线是否能让：

- alpha
- foreground appearance
- composite result

在定义上更自然地对齐，从而为边缘质量提供更好的问题结构。


## 3. 要改哪些模块

建议新增：

- `wan/utils/rgba_foreground_proto.py`
- `scripts/eval/check_rgba_foreground_proto.py`
- `scripts/eval/run_optimization9_rgba_benchmark.py`
- `scripts/eval/evaluate_optimization9_rgba_benchmark.py`


## 4. 具体如何做

### 4.1 定义最小 proto

初期不要追求完整 RGBA video backbone，先做：

- ROI-level RGBA foreground prototype
- 或 foreground RGB + alpha paired output prototype

### 4.2 输出语义

必须明确：

- RGB 与 alpha 是否同步输出
- 如何与 background composite
- 如何与 reviewed benchmark 对齐

### 4.3 feasibility 策略

只在：

- hair-heavy
- hand-heavy
- cloth contour-heavy

case 上做最小原型验证。


## 5. 如何检验

### 5.1 reviewed benchmark

重点看：

- alpha quality
- boundary_f1
- hair_edge_quality

### 5.2 real-video smoke

重点看：

- composite quality
- seam
- background consistency


## 6. 强指标要求

本步骤不要求全面超越主线，但至少要满足：

- alpha/edge 指标中出现一类明显正向变化
- 且不只是旧主线 tradeoff 的重复


## 7. 实现-测评-再改闭环

### Round 1

- 做最小 RGBA proto

### Round 2

- 调整输出语义和 composite 规则
- 跑 reviewed + smoke

### Round 3

- 冻结 best-of-three
- 判断是否值得继续


## 8. 成功标准

- RGBA foreground 路线至少在边缘定义和 composite 质量上表现出结构性优势
