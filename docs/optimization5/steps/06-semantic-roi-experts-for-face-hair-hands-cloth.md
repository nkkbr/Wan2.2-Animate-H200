# Step 06: Semantic ROI Experts for Face / Hair / Hands / Cloth

## 1. 为什么要做

`optimization4` Step 06 已经证明：

- 语义化 boundary specialization 方向是合理的
- 但仅靠 rule-based semantic refine 不足以显著提升边缘锐度

因此，本步骤不再走“统一 deterministic 规则”，而是建立：

> semantic ROI experts

让不同边界进入不同的局部生成 / alpha / preserve 路线。


## 2. 目标

针对不同 ROI 建立不同的专家策略：

- face edge expert
- hair edge expert
- hand edge expert
- cloth edge expert
- occluded edge expert


## 3. 交付目标

1. semantic ROI routing
2. 各类 ROI 的条件构造策略
3. category-wise metrics
4. findings 文档


## 4. 范围

### In Scope

- semantic ROI 分类与 routing
- expert-specific conditioning / alpha / preserve 策略
- category-wise benchmark

### Out of Scope

- 完整训练多专家大模型


## 5. 实施方案

### 5.1 face expert

重点：

- preserve identity
- preserve jaw / cheek / ear contour
- 控制 over-sharpen

### 5.2 hair expert

重点：

- high-quality alpha
- fine strand structure
- background bleed 抑制

### 5.3 hand expert

重点：

- small moving edges
- finger separation
- motion-consistency

### 5.4 cloth expert

重点：

- edge contrast
- local texture continuity
- anti-halo


## 6. 如何验证

### 6.1 category-wise GT 指标

- `face_boundary_f1`
- `hair_boundary_f1`
- `hand_boundary_f1`
- `cloth_boundary_f1`

### 6.2 category-wise real proxy

- face boundary gradient / contrast
- hair halo
- hand edge separation
- cloth edge contrast


## 7. 强指标要求

至少满足下面之一：

- `face` boundary `gradient + contrast` 均提升 `>= 8%`
- `hair` halo 降低 `>= 10%`
- `hand` edge separation proxy 提升 `>= 10%`
- `cloth` edge contrast 提升 `>= 8%`

且全图 seam/background 不明显恶化。


## 8. 实现-测评-再改闭环

### Round 1

- 跑通 semantic ROI routing
- 建立 category-wise AB

### Round 2

若 category 指标不够，则改：

- routing
- ROI expand rule
- preserve / alpha weighting

### Round 3

若仍未过 gate，则冻结 best-of-three，并明确：

- 哪类 ROI 真有收益
- 哪类 ROI 暂不值得继续投入


## 9. 风险

- semantic routing 错误会直接伤害对应 ROI
- hair/hand 这类小 ROI 对时间稳定性要求更高


## 10. 完成标准

1. semantic ROI experts 已落地
2. category-wise benchmark 可运行
3. 至少一个 ROI 类别在强指标上显著优于 baseline
4. findings 明确给出哪些专家值得保留
