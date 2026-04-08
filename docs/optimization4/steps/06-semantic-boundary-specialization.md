# Step 06: Semantic Boundary Specialization

## 1. 目的

本步骤的目标是让系统不再用单一策略处理所有边界，而是按照语义类别分别优化。


## 2. 为什么要做

当前统一边界策略的问题很明显：

- 发丝边缘和手边缘不是同一种问题
- 脸边缘和衣服边缘也不是同一种问题

统一策略容易出现：

- 某一类边界更好了
- 另一类边界却被带坏


## 3. 交付目标

新增边界语义子类：

- `face_boundary`
- `hair_boundary`
- `hand_boundary`
- `cloth_boundary`
- `occluded_boundary`

并让 refine / alpha / generate 条件对不同边界使用不同策略。


## 4. 范围

优先修改：

- `boundary_fusion.py`
- `matting_adapter.py`
- `parsing_adapter.py`
- `boundary_refinement.py`
- `animate.py`


## 5. 实施方案

### 5.1 先建立子类边界 mask

不一定一开始就完美，但至少要从：

- parsing
- face analysis
- hand tracks
- hair heuristic

生成可用的 semantic boundary maps。

### 5.2 不同语义边界用不同 refine 策略

例如：

- hair：优先 alpha，弱 sharpen
- face：强 preserve，谨慎 sharpen
- hand：优先 motion consistency
- cloth：允许较强 edge restore

### 5.3 指标也按类别统计

不要只看总平均值。  
必须分别看：

- face boundary metrics
- hair boundary metrics
- hand boundary metrics


## 6. 强指标要求

至少有两类高价值边界达到显著改善：

- `face_boundary`：gradient / contrast 提升 **8% 以上**
- `hair_boundary`：halo 降低 **10% 以上**

若 hair 数据不足，可用 `hand_boundary` 替代其中一类。


## 7. 实现-测评-再改闭环

最多 3 轮。

### Round 1

- 建 semantic masks
- 跑分类边界 metrics

### Round 2

- 修分类 refine 策略

### Round 3

- 固定最优组合


## 8. 风险

### 8.1 分类边界 mask 本身不准

应对：

- 先只对高置信区域启用 specialization

### 8.2 规则太多导致系统复杂度上升

应对：

- 先只专门化 face / hair / hand 三类


## 9. 完成标准

如果至少两类重要边界在局部指标上明显提升，而总图无明显回退，则可认为本步骤成功。
